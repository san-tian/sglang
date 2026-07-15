// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{RouteDecisionLogContext, SelectionContext};
use crate::workers::Worker;
use axum::http::HeaderMap;
use serde_json::json;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

const DEFAULT_HEADER: &str = "x-sgl-route-decision-log";
const DEFAULT_CANDIDATE_LIMIT: usize = 32;
const RATE_DENOMINATOR: u64 = 1_000_000;

#[derive(Debug)]
struct DecisionLogConfig {
    sample_ppm: u64,
    header: String,
    candidate_limit: usize,
}

impl DecisionLogConfig {
    fn from_env() -> Self {
        let sample_ppm = std::env::var("ROUTE_DECISION_LOG_SAMPLE_RATE")
            .ok()
            .and_then(|raw| raw.parse::<f64>().ok())
            .filter(|rate| rate.is_finite() && *rate > 0.0)
            .map(|rate| (rate.min(1.0) * RATE_DENOMINATOR as f64).round() as u64)
            .unwrap_or(0);
        let header = std::env::var("ROUTE_DECISION_LOG_HEADER")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_HEADER.to_string())
            .to_ascii_lowercase();
        let candidate_limit = std::env::var("ROUTE_DECISION_LOG_MAX_CANDIDATES")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_CANDIDATE_LIMIT);
        Self {
            sample_ppm,
            header,
            candidate_limit,
        }
    }
}

fn config() -> &'static DecisionLogConfig {
    static CONFIG: OnceLock<DecisionLogConfig> = OnceLock::new();
    CONFIG.get_or_init(DecisionLogConfig::from_env)
}

pub fn context_from_headers<'a>(
    headers: &HeaderMap,
    request_id: &'a str,
    endpoint: &'a str,
    request_priority: i64,
) -> Option<RouteDecisionLogContext<'a>> {
    let cfg = config();
    let forced = headers
        .get(cfg.header.as_str())
        .and_then(|value| value.to_str().ok())
        .is_some_and(is_truthy);
    if !forced && !sampled(request_id, cfg.sample_ppm) {
        return None;
    }
    Some(RouteDecisionLogContext {
        request_id,
        endpoint,
        request_priority,
        candidate_limit: cfg.candidate_limit,
    })
}

pub fn log_generic_selection(
    ctx: &SelectionContext<'_>,
    workers: &[std::sync::Arc<Worker>],
    selected: &Worker,
    policy: &dyn std::fmt::Debug,
) {
    let Some(log_ctx) = ctx.route_decision_log() else {
        return;
    };
    let mut candidates: Vec<_> = workers
        .iter()
        .take(log_ctx.candidate_limit)
        .map(|worker| generic_candidate_json(worker, worker.url == selected.url))
        .collect();
    candidates.sort_by_key(|candidate| {
        candidate
            .get("effective_load_reported")
            .and_then(|value| value.as_u64())
            .unwrap_or(u64::MAX)
    });
    let decision = json!({
        "event": "route_decision",
        "policy": format!("{policy:?}"),
        "policy_detail": "generic",
        "endpoint": log_ctx.endpoint,
        "request_id": log_ctx.request_id,
        "model": ctx.model().0,
        "request_priority": log_ctx.request_priority,
        "selected_worker": selected.url,
        "candidate_count": workers.len(),
        "candidate_limit": log_ctx.candidate_limit,
        "candidates_truncated": workers.len() > log_ctx.candidate_limit,
        "candidates": candidates,
    });
    tracing::info!(decision = %decision, "route_decision");
}

pub fn generic_candidate_json(worker: &Worker, selected: bool) -> serde_json::Value {
    let breaker = worker.breaker.snapshot();
    json!({
        "worker": worker.url,
        "selected": selected,
        "mode": format!("{:?}", worker.mode()),
        "backend": format!("{:?}", worker.backend()),
        "tier": format!("{:?}", worker.tier()),
        "reported_load": worker.reported_load(),
        "active_load": worker.active_load(),
        "pending_requests": worker.pending_load(),
        "pending_tokens": worker.pending_token_load(),
        "global_pending_requests": worker.global_pending_load(),
        "global_pending_tokens": worker.global_pending_token_load(),
        "effective_load_local": worker.effective_load(false),
        "effective_load_reported": worker.effective_load(true),
        "effective_ttft_load_reported": worker.effective_ttft_load(true, 256),
        "prefill_capacity_milli": worker.prefill_capacity_milli(),
        "min_priority": worker.min_priority(),
        "max_context_tokens": worker.max_context_tokens(),
        "breaker_admit": breaker.admit,
        "breaker_state_code": breaker.state_code,
        "introspection_probe_allows_routing": worker.introspection_probe_allows_routing(),
        "prefill_members": worker.prefill_members(),
    })
}

fn sampled(request_id: &str, sample_ppm: u64) -> bool {
    if sample_ppm == 0 {
        return false;
    }
    if sample_ppm >= RATE_DENOMINATOR {
        return true;
    }
    let mut hasher = DefaultHasher::new();
    request_id.hash(&mut hasher);
    hasher.finish() % RATE_DENOMINATOR < sample_ppm
}

fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_header_values_are_case_insensitive() {
        assert!(is_truthy("TRUE"));
        assert!(is_truthy(" yes "));
        assert!(!is_truthy("0"));
    }

    #[test]
    fn zero_sample_rate_never_samples() {
        assert!(!sampled("req-1", 0));
    }

    #[test]
    fn full_sample_rate_always_samples() {
        assert!(sampled("req-1", RATE_DENOMINATOR));
    }
}
