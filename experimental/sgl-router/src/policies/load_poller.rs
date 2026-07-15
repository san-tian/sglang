// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Background worker-load poller.
//!
//! The router-side in-flight counter (`Worker::active_load`) is a poor
//! "how busy is this worker" signal for a mixed short/long workload: one
//! 200k-token request and one 2k request both count as load 1, yet load
//! the engine completely differently. This poller periodically asks each
//! worker for its REAL request pressure via the worker's `/get_load` endpoint
//! and verifies scheduler liveness through `/health`. It stores the request
//! pressure on `Worker::reported_load`, which the `cache_aware_zmq` policy
//! consumes (via `Worker::effective_load`) for its min-load / imbalance /
//! hit-load-guard decisions.
//!
//! Auth: `/get_load` is behind the worker's `--api-key` (401 without), so
//! the poller carries the same `worker_introspect_key` bearer the KV-event
//! discovery already uses. Unlike the ZMQ KV-event feed, `/get_load` is
//! plain HTTP on the worker's normal port — reachable over NAT/Vast public
//! mappings with no special port.
//!
//! Failure handling: any `/get_load` or `/health` error (timeout, non-2xx,
//! parse) writes the `REPORTED_LOAD_FAILED` sentinel so the policy treats that
//! worker as HIGH load and PD admission removes it. `/get_load` alone is not a
//! sufficient liveness check: the HTTP process can still return cached load
//! while the scheduler health endpoint is wedged.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::policies::active_load::JanitorHandle;
use crate::workers::worker::{
    CandidatePrefillLoad, MemberPrefillLoadSnapshot, PrefillLoadRole, PrefillLoadSnapshot,
    PrefillPriorityLoad, REPORTED_LOAD_FAILED, REPORTED_LOAD_UNSET,
};
use crate::workers::WorkerRegistry;

/// Per-`dp_rank` entry from a worker's `/get_load` response, e.g.
/// `[{"dp_rank":0,"num_reqs":0,"num_waiting_reqs":0,"num_tokens":0,...}, ...]`.
/// Unknown fields are ignored; routing needs the request pressure.
#[derive(Debug, Deserialize)]
struct GetLoadEntry {
    #[serde(default)]
    worker_url: Option<String>,
    #[serde(default)]
    num_reqs: i64,
    #[serde(default)]
    num_waiting_reqs: i64,
    #[serde(default)]
    num_running_reqs: Option<i64>,
    #[serde(default)]
    num_waiting_uncached_tokens: Option<i64>,
    #[serde(default)]
    load_role: Option<String>,
    #[serde(default)]
    prefill_capacity_milli: Option<usize>,
    #[serde(default)]
    prefill_queue: Option<PrefillQueueEntry>,
}

#[derive(Debug, Deserialize)]
struct PrefillQueueEntry {
    #[serde(default)]
    detail_complete: bool,
    #[serde(default)]
    chunked_remaining_uncached_tokens: i64,
    #[serde(default)]
    work_bucket_bounds: Vec<i64>,
    #[serde(default)]
    priority_scheduling_enabled: bool,
    #[serde(default)]
    schedule_low_priority_values_first: bool,
    #[serde(default)]
    priority_values: Vec<i64>,
    #[serde(default)]
    priority_total_uncached_tokens: Vec<i64>,
    #[serde(default)]
    priority_ahead_uncached_tokens: Vec<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct V1LoadsResponse {
    #[serde(default)]
    logical_request_pressure: Option<i64>,
    loads: Vec<GetLoadEntry>,
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedWorkerLoad {
    request_pressure: i64,
    prefill: Option<PrefillLoadSnapshot>,
    prefill_members: Vec<MemberPrefillLoadSnapshot>,
}

/// Sum active plus waiting requests across all dp ranks reported by one worker.
///
/// Borrowing production B200 capacity must treat already-running production
/// requests as busy, not only queued requests. Counting both fields keeps
/// idle-borrow routing conservative.
/// Returns `None` if the body does not parse as the expected array.
#[cfg(test)]
fn parse_total_request_pressure(body: &str) -> Option<i64> {
    parse_worker_load(body).map(|load| load.request_pressure)
}

fn parse_worker_load(body: &str) -> Option<ParsedWorkerLoad> {
    let entries: Vec<GetLoadEntry> = serde_json::from_str(body).ok()?;
    let request_pressure = entries.iter().fold(0i64, |total, entry| {
        total.saturating_add(
            entry
                .num_reqs
                .max(0)
                .saturating_add(entry.num_waiting_reqs.max(0)),
        )
    });
    let prefill_role = entries
        .first()
        .and_then(|entry| normalize_prefill_load_role(entry.load_role.as_deref()))
        .filter(|role| {
            entries.iter().all(|entry| {
                entry.num_running_reqs.is_some()
                    && entry.num_waiting_uncached_tokens.is_some()
                    && normalize_prefill_load_role(entry.load_role.as_deref()) == Some(*role)
            })
        });
    let prefill = prefill_role.map(|role| PrefillLoadSnapshot {
        role,
        running_requests: entries
            .iter()
            .map(|entry| entry.num_running_reqs.unwrap_or_default().max(0) as usize)
            .fold(0usize, usize::saturating_add),
        total_waiting_uncached_tokens: entries
            .iter()
            .map(|entry| entry.num_waiting_uncached_tokens.unwrap_or_default().max(0) as usize)
            .fold(0usize, usize::saturating_add),
        candidate: aggregate_candidate_prefill(&entries),
    });
    Some(ParsedWorkerLoad {
        request_pressure,
        prefill,
        prefill_members: Vec::new(),
    })
}

fn parse_v1_worker_load(
    body: &str,
    worker: &crate::workers::worker::Worker,
) -> Option<ParsedWorkerLoad> {
    let response: V1LoadsResponse = serde_json::from_str(body).ok()?;
    if worker.backend() == crate::discovery::WorkerBackend::SglangProxy {
        return parse_proxy_v1_load(response, worker);
    }

    let request_pressure = response.loads.iter().fold(0i64, |total, entry| {
        total.saturating_add(
            entry
                .num_running_reqs
                .unwrap_or_default()
                .max(0)
                .saturating_add(entry.num_waiting_reqs.max(0)),
        )
    });
    let role = match worker.mode() {
        crate::discovery::WorkerMode::Plain => Some(PrefillLoadRole::Integrated),
        crate::discovery::WorkerMode::Prefill => Some(PrefillLoadRole::Prefill),
        crate::discovery::WorkerMode::Decode => None,
    };
    let prefill = role.and_then(|role| snapshot_from_entries(&response.loads, role));
    Some(ParsedWorkerLoad {
        request_pressure,
        prefill,
        prefill_members: Vec::new(),
    })
}

fn parse_proxy_v1_load(
    response: V1LoadsResponse,
    worker: &crate::workers::worker::Worker,
) -> Option<ParsedWorkerLoad> {
    let configured = worker.prefill_members().iter().collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut members = Vec::with_capacity(response.loads.len());
    for entry in &response.loads {
        let member_url = entry.worker_url.as_ref()?;
        if entry.load_role.as_deref() != Some("prefill")
            || !configured.contains(member_url)
            || !seen.insert(member_url)
        {
            return None;
        }
        let capacity = entry.prefill_capacity_milli.filter(|value| *value > 0)?;
        let snapshot =
            snapshot_from_entries(std::slice::from_ref(entry), PrefillLoadRole::Prefill)?;
        members.push(MemberPrefillLoadSnapshot {
            worker_url: member_url.clone(),
            prefill_capacity_milli: capacity,
            snapshot,
        });
    }
    if members.is_empty() {
        return None;
    }
    let request_pressure = response.logical_request_pressure.unwrap_or_else(|| {
        response.loads.iter().fold(0i64, |total, entry| {
            total.saturating_add(
                entry
                    .num_running_reqs
                    .unwrap_or_default()
                    .max(0)
                    .saturating_add(entry.num_waiting_reqs.max(0)),
            )
        })
    });
    Some(ParsedWorkerLoad {
        request_pressure: request_pressure.max(0),
        prefill: None,
        prefill_members: members,
    })
}

fn snapshot_from_entries(
    entries: &[GetLoadEntry],
    role: PrefillLoadRole,
) -> Option<PrefillLoadSnapshot> {
    if entries.is_empty()
        || entries.iter().any(|entry| {
            entry.num_running_reqs.is_none() || entry.num_waiting_uncached_tokens.is_none()
        })
    {
        return None;
    }
    Some(PrefillLoadSnapshot {
        role,
        running_requests: entries
            .iter()
            .map(|entry| entry.num_running_reqs.unwrap_or_default().max(0) as usize)
            .fold(0usize, usize::saturating_add),
        total_waiting_uncached_tokens: entries
            .iter()
            .map(|entry| entry.num_waiting_uncached_tokens.unwrap_or_default().max(0) as usize)
            .fold(0usize, usize::saturating_add),
        candidate: aggregate_candidate_prefill(entries),
    })
}

fn normalize_prefill_load_role(role: Option<&str>) -> Option<PrefillLoadRole> {
    match role {
        Some("null" | "plain") => Some(PrefillLoadRole::Integrated),
        Some("prefill") => Some(PrefillLoadRole::Prefill),
        _ => None,
    }
}

fn aggregate_candidate_prefill(entries: &[GetLoadEntry]) -> Option<CandidatePrefillLoad> {
    let first = entries.first()?.prefill_queue.as_ref()?;
    if !first.detail_complete
        || first.work_bucket_bounds.is_empty()
        || first.work_bucket_bounds.len() > 32
        || first.work_bucket_bounds.iter().any(|value| *value < 0)
        || first
            .work_bucket_bounds
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return None;
    }

    let bucket_bounds = first
        .work_bucket_bounds
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let mut groups: BTreeMap<i64, (usize, Vec<usize>)> = BTreeMap::new();
    let mut chunked_remaining = 0usize;
    for entry in entries {
        let queue = entry.prefill_queue.as_ref()?;
        if !queue.detail_complete
            || queue.priority_scheduling_enabled != first.priority_scheduling_enabled
            || queue.schedule_low_priority_values_first != first.schedule_low_priority_values_first
            || queue.work_bucket_bounds != first.work_bucket_bounds
            || queue.priority_values.len() != queue.priority_total_uncached_tokens.len()
            || queue.priority_values.len() != queue.priority_ahead_uncached_tokens.len()
            || queue.priority_values.len() > 32
            || queue.chunked_remaining_uncached_tokens < 0
        {
            return None;
        }
        chunked_remaining =
            chunked_remaining.saturating_add(queue.chunked_remaining_uncached_tokens as usize);
        let mut seen = HashSet::new();
        for ((priority, total), ahead) in queue
            .priority_values
            .iter()
            .zip(&queue.priority_total_uncached_tokens)
            .zip(&queue.priority_ahead_uncached_tokens)
        {
            if !seen.insert(*priority)
                || *total < 0
                || ahead.len() != bucket_bounds.len()
                || ahead.iter().any(|value| *value < 0)
                || ahead.iter().any(|value| *value > *total)
                || ahead.windows(2).any(|pair| pair[0] > pair[1])
            {
                return None;
            }
            let group = groups
                .entry(*priority)
                .or_insert_with(|| (0, vec![0; bucket_bounds.len()]));
            group.0 = group.0.saturating_add(*total as usize);
            for (aggregate, value) in group.1.iter_mut().zip(ahead) {
                *aggregate = aggregate.saturating_add(*value as usize);
            }
        }
        let detailed_total = queue
            .priority_total_uncached_tokens
            .iter()
            .fold(queue.chunked_remaining_uncached_tokens, |total, value| {
                total.saturating_add(*value)
            });
        if detailed_total != entry.num_waiting_uncached_tokens?.max(0) {
            return None;
        }
    }

    Some(CandidatePrefillLoad {
        chunked_remaining_uncached_tokens: chunked_remaining,
        work_bucket_bounds: bucket_bounds,
        priority_scheduling_enabled: first.priority_scheduling_enabled,
        schedule_low_priority_values_first: first.schedule_low_priority_values_first,
        priorities: groups
            .into_iter()
            .map(
                |(priority, (total_uncached_tokens, ahead_uncached_tokens))| PrefillPriorityLoad {
                    priority,
                    total_uncached_tokens,
                    ahead_uncached_tokens,
                },
            )
            .collect(),
    })
}

fn worker_get(
    client: &reqwest::Client,
    worker: &crate::workers::worker::Worker,
    url: &str,
) -> reqwest::RequestBuilder {
    let mut req = client.get(url);
    if let Some(token) = worker.bearer_token() {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .expect("worker bearer token must be a valid HTTP header value");
        value.set_sensitive(true);
        req = req.header(reqwest::header::AUTHORIZATION, value);
    }
    req
}

/// Poll one worker's `/get_load` and `/health` concurrently and store the load
/// only when both probes succeed. Never panics; never returns an error.
async fn poll_one(client: &reqwest::Client, worker: &Arc<crate::workers::worker::Worker>) {
    if !worker.backend().supports_sglang_load() {
        worker.set_reported_load(REPORTED_LOAD_UNSET);
        worker.set_reported_prefill_load(None);
        worker.set_reported_prefill_members(Vec::new());
        tracing::debug!(
            worker_url = %worker.url,
            backend = ?worker.backend(),
            "load-poller: skipping worker backend without SGLang /get_load"
        );
        return;
    }
    let base = worker.url.trim_end_matches('/');
    let v1_loads_url = format!("{base}/v1/loads?include=core,prefill_queue");
    let legacy_load_url = format!("{base}/get_load");
    let health_url = format!("{base}/health");
    let load_probe = async {
        if worker.prefill_members().is_empty() {
            let legacy = worker_get(client, worker, &legacy_load_url)
                .send()
                .await
                .ok()?;
            if !legacy.status().is_success() {
                return None;
            }
            return parse_worker_load(&legacy.text().await.ok()?);
        }
        let resp = worker_get(client, worker, &v1_loads_url)
            .send()
            .await
            .ok()?;
        if resp.status().is_success() {
            let body = resp.text().await.ok()?;
            return parse_v1_worker_load(&body, worker);
        }
        if !matches!(
            resp.status(),
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::METHOD_NOT_ALLOWED
        ) {
            return None;
        }
        let legacy = worker_get(client, worker, &legacy_load_url)
            .send()
            .await
            .ok()?;
        if !legacy.status().is_success() {
            return None;
        }
        parse_worker_load(&legacy.text().await.ok()?)
    };
    let health_probe = async {
        let resp = match worker_get(client, worker, &health_url).send().await {
            Ok(resp) => resp,
            Err(_) => return false,
        };
        resp.status().is_success()
    };
    let (load, health_ok) = tokio::join!(load_probe, health_probe);
    match (load, health_ok) {
        (Some(load), true) => {
            worker.set_reported_load(load.request_pressure);
            worker.set_reported_prefill_load(load.prefill);
            worker.set_reported_prefill_members(load.prefill_members);
        }
        (load, health_ok) => {
            worker.set_reported_load(REPORTED_LOAD_FAILED);
            worker.set_reported_prefill_load(None);
            worker.set_reported_prefill_members(Vec::new());
            tracing::debug!(
                worker_url = %worker.url,
                load_ok = load.is_some(),
                health_ok,
                "load-poller: worker introspection failed; marking HIGH load"
            );
        }
    }
}

/// One poll round: fan out `/get_load` + `/health` to every registered worker
/// concurrently and update each worker's `reported_load`.
async fn poll_round(client: &reqwest::Client, registry: &Arc<WorkerRegistry>) {
    let workers = registry.all();
    let futs = workers.iter().map(|w| poll_one(client, w));
    futures::future::join_all(futs).await;
}

/// Spawn the background load poller. Mirrors `spawn_sweeper`'s lifecycle:
/// the returned [`JanitorHandle`] cancels the task on drop / `shutdown()`.
/// `bearer` is the worker introspect key (same one KV-event discovery
/// uses); `None` means `/get_load` is hit unauthenticated (workers with no
/// `--api-key`).
pub fn spawn_load_poller(
    registry: Arc<WorkerRegistry>,
    interval: Duration,
    probe_timeout: Duration,
    bearer: Option<String>,
) -> JanitorHandle {
    let client = build_probe_client(probe_timeout, bearer.as_deref());

    let cancel = CancellationToken::new();
    let cancel_for_task = cancel.clone();
    let join = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                biased;
                _ = cancel_for_task.cancelled() => {
                    tracing::debug!("load-poller: shutdown requested");
                    return;
                }
                _ = ticker.tick() => {
                    poll_round(&client, &registry).await;
                }
            }
        }
    });
    JanitorHandle::from_parts(cancel, join)
}

fn build_probe_client(probe_timeout: Duration, bearer: Option<&str>) -> reqwest::Client {
    let mut builder = reqwest::Client::builder().timeout(probe_timeout);
    if let Some(token) = bearer {
        let mut headers = reqwest::header::HeaderMap::new();
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .expect("worker introspect key must be a valid HTTP header value");
        value.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, value);
        builder = builder.default_headers(headers);
    }
    builder.build().expect("load-poller http client builds")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerBackend, WorkerId, WorkerMode, WorkerSpec};
    use crate::workers::worker::REPORTED_LOAD_FAILED;
    use axum::http::StatusCode;
    use axum::{routing::get, Json, Router};
    use serde_json::json;
    use tokio::net::TcpListener;

    #[test]
    fn parse_sums_active_and_waiting_across_dp_ranks() {
        let body = r#"[
            {"dp_rank":0,"num_reqs":3,"num_waiting_reqs":2,"num_tokens":100},
            {"dp_rank":1,"num_reqs":1,"num_waiting_reqs":5,"num_tokens":50}
        ]"#;
        assert_eq!(parse_total_request_pressure(body), Some(11));
    }

    #[test]
    fn parse_single_rank() {
        let body = r#"[{"dp_rank":0,"num_reqs":0,"num_waiting_reqs":0,"num_tokens":0}]"#;
        assert_eq!(parse_total_request_pressure(body), Some(0));
    }

    #[test]
    fn parse_negative_clamped_to_zero() {
        // Defensive: a bogus negative waiting count must not underflow the sum.
        let body = r#"[{"num_reqs":2,"num_waiting_reqs":-4},{"num_reqs":-5,"num_waiting_reqs":3}]"#;
        assert_eq!(parse_total_request_pressure(body), Some(5));
    }

    #[test]
    fn parse_rejects_non_array() {
        assert_eq!(parse_total_request_pressure("not json"), None);
        assert_eq!(
            parse_total_request_pressure(r#"{"num_waiting_reqs":1}"#),
            None
        );
    }

    #[test]
    fn parse_aggregates_prefill_snapshot_across_dp_ranks() {
        let body = r#"[
          {
            "num_reqs":3,"num_waiting_reqs":2,"num_running_reqs":1,"load_role":"null",
            "num_waiting_uncached_tokens":1000,
            "prefill_queue":{
              "detail_complete":true,"chunked_remaining_uncached_tokens":50,
              "work_bucket_bounds":[256,1024],"priority_scheduling_enabled":true,
              "schedule_low_priority_values_first":false,
              "priority_values":[0,10],"priority_total_uncached_tokens":[650,300],
              "priority_ahead_uncached_tokens":[[100,650],[50,300]]
            }
          },
          {
            "num_reqs":4,"num_waiting_reqs":1,"num_running_reqs":2,"load_role":"null",
            "num_waiting_uncached_tokens":2000,
            "prefill_queue":{
              "detail_complete":true,"chunked_remaining_uncached_tokens":75,
              "work_bucket_bounds":[256,1024],"priority_scheduling_enabled":true,
              "schedule_low_priority_values_first":false,
              "priority_values":[0,10],"priority_total_uncached_tokens":[1725,200],
              "priority_ahead_uncached_tokens":[[200,1725],[50,200]]
            }
          }
        ]"#;

        let parsed = parse_worker_load(body).expect("valid load");
        let prefill = parsed.prefill.expect("token totals available");
        assert_eq!(prefill.role, PrefillLoadRole::Integrated);
        assert_eq!(prefill.running_requests, 3);
        assert_eq!(prefill.total_waiting_uncached_tokens, 3000);
        let candidate = prefill.candidate.expect("candidate detail available");
        assert_eq!(candidate.chunked_remaining_uncached_tokens, 125);
        assert_eq!(candidate.priorities[0].total_uncached_tokens, 2375);
        assert_eq!(
            candidate.priorities[0].ahead_uncached_tokens,
            vec![300, 2375]
        );
        assert_eq!(candidate.work_ahead_tokens(0, 100), Some(925));
        assert_eq!(candidate.work_ahead_tokens(10, 100), Some(225));
    }

    #[test]
    fn parse_old_worker_has_no_prefill_snapshot() {
        let parsed =
            parse_worker_load(r#"[{"dp_rank":0,"num_reqs":2,"num_waiting_reqs":1}]"#).unwrap();
        assert_eq!(parsed.request_pressure, 3);
        assert_eq!(parsed.prefill, None);
    }

    #[test]
    fn incomplete_candidate_detail_keeps_conservative_totals() {
        let parsed = parse_worker_load(
            r#"[{
              "num_reqs":2,"num_waiting_reqs":1,"num_running_reqs":1,"load_role":"null",
              "num_waiting_uncached_tokens":900,
              "prefill_queue":{"detail_complete":false}
            }]"#,
        )
        .unwrap();
        let prefill = parsed.prefill.unwrap();
        assert_eq!(prefill.running_requests, 1);
        assert_eq!(prefill.total_waiting_uncached_tokens, 900);
        assert_eq!(prefill.candidate, None);
    }

    #[test]
    fn decode_role_is_not_treated_as_prefill_work() {
        let parsed = parse_worker_load(
            r#"[{
              "num_reqs":2,"num_waiting_reqs":1,"num_running_reqs":2,
              "num_waiting_uncached_tokens":900,"load_role":"decode"
            }]"#,
        )
        .unwrap();
        assert_eq!(parsed.request_pressure, 3);
        assert_eq!(parsed.prefill, None);
    }

    #[test]
    fn native_prefill_role_is_retained_as_prefill_only_work() {
        let parsed = parse_worker_load(
            r#"[{
              "num_reqs":2,"num_waiting_reqs":1,"num_running_reqs":1,
              "num_waiting_uncached_tokens":900,"load_role":"prefill"
            }]"#,
        )
        .unwrap();
        assert_eq!(parsed.request_pressure, 3);
        let prefill = parsed.prefill.expect("native Prefill work is available");
        assert_eq!(prefill.role, PrefillLoadRole::Prefill);
        assert_eq!(prefill.running_requests, 1);
        assert_eq!(prefill.total_waiting_uncached_tokens, 900);
    }

    #[test]
    fn mixed_integrated_and_prefill_roles_reject_token_snapshot() {
        let parsed = parse_worker_load(
            r#"[
              {
                "num_reqs":1,"num_waiting_reqs":0,"num_running_reqs":1,
                "num_waiting_uncached_tokens":100,"load_role":"null"
              },
              {
                "num_reqs":1,"num_waiting_reqs":0,"num_running_reqs":1,
                "num_waiting_uncached_tokens":200,"load_role":"prefill"
              }
            ]"#,
        )
        .unwrap();
        assert_eq!(parsed.request_pressure, 2);
        assert_eq!(parsed.prefill, None);
    }

    #[test]
    fn parses_direct_engine_v1_loads_envelope() {
        let worker = test_worker(
            "direct",
            "http://direct",
            WorkerMode::Plain,
            WorkerBackend::Sglang,
            Vec::new(),
        );
        let parsed = parse_v1_worker_load(
            r#"{
              "timestamp":"2026-07-15T00:00:00Z",
              "loads":[{
                "dp_rank":0,
                "num_running_reqs":2,
                "num_waiting_reqs":3,
                "num_waiting_uncached_tokens":900,
                "prefill_queue":null
              }]
            }"#,
            &worker,
        )
        .expect("v1 loads parses");
        assert_eq!(parsed.request_pressure, 5);
        assert!(parsed.prefill_members.is_empty());
        let snapshot = parsed.prefill.expect("integrated snapshot");
        assert_eq!(snapshot.role, PrefillLoadRole::Integrated);
        assert_eq!(snapshot.running_requests, 2);
        assert_eq!(snapshot.total_waiting_uncached_tokens, 900);
    }

    #[test]
    fn parses_proxy_member_snapshots_without_summing_members() {
        let worker = test_worker(
            "proxy",
            "http://proxy",
            WorkerMode::Plain,
            WorkerBackend::SglangProxy,
            vec!["http://p0".into(), "http://p1".into()],
        );
        let parsed = parse_v1_worker_load(
            r#"{
              "logical_request_pressure":7,
              "loads":[
                {
                  "worker_url":"http://p0","dp_rank":0,"load_role":"prefill",
                  "prefill_capacity_milli":1000,"num_running_reqs":1,
                  "num_waiting_reqs":2,"num_waiting_uncached_tokens":300
                },
                {
                  "worker_url":"http://p1","dp_rank":1,"load_role":"prefill",
                  "prefill_capacity_milli":2000,"num_running_reqs":2,
                  "num_waiting_reqs":4,"num_waiting_uncached_tokens":900
                }
              ]
            }"#,
            &worker,
        )
        .expect("proxy v1 loads parses");
        assert_eq!(parsed.request_pressure, 7);
        assert_eq!(parsed.prefill, None);
        assert_eq!(parsed.prefill_members.len(), 2);
        assert_eq!(parsed.prefill_members[0].worker_url, "http://p0");
        assert_eq!(
            parsed.prefill_members[0]
                .snapshot
                .total_waiting_uncached_tokens,
            300
        );
        assert_eq!(parsed.prefill_members[1].prefill_capacity_milli, 2000);
        assert_eq!(
            parsed.prefill_members[1]
                .snapshot
                .total_waiting_uncached_tokens,
            900
        );
    }

    #[test]
    fn proxy_v1_loads_rejects_unknown_or_duplicate_member_identity() {
        let worker = test_worker(
            "proxy",
            "http://proxy",
            WorkerMode::Plain,
            WorkerBackend::SglangProxy,
            vec!["http://p0".into()],
        );
        for body in [
            r#"{"loads":[{"worker_url":"http://unknown","prefill_capacity_milli":1000,"num_running_reqs":0,"num_waiting_uncached_tokens":0}]}"#,
            r#"{"loads":[{"worker_url":"http://p0","prefill_capacity_milli":1000,"num_running_reqs":0,"num_waiting_uncached_tokens":0},{"worker_url":"http://p0","prefill_capacity_milli":1000,"num_running_reqs":0,"num_waiting_uncached_tokens":0}]}"#,
        ] {
            assert_eq!(parse_v1_worker_load(body, &worker), None);
        }
    }

    #[tokio::test]
    async fn poll_round_skips_vllm_worker_without_get_load() {
        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("vllm".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: "http://127.0.0.1:9".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::Vllm,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 1000,
                prefill_members: Vec::new(),
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        worker.set_reported_load(REPORTED_LOAD_FAILED);
        worker.set_reported_prefill_load(Some(PrefillLoadSnapshot {
            role: PrefillLoadRole::Integrated,
            running_requests: 1,
            total_waiting_uncached_tokens: 1,
            candidate: None,
        }));
        let client = reqwest::Client::new();

        poll_round(&client, &registry).await;

        assert_eq!(worker.reported_load(), REPORTED_LOAD_UNSET);
        assert_eq!(worker.reported_prefill_load(), None);
    }

    #[tokio::test]
    async fn poll_round_marks_sglang_worker_failed_on_missing_get_load() {
        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("sglang".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: "http://127.0.0.1:9".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::Sglang,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 1000,
                prefill_members: Vec::new(),
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        let client = reqwest::Client::new();

        poll_round(&client, &registry).await;

        assert_eq!(worker.reported_load(), REPORTED_LOAD_FAILED);
        assert_eq!(worker.reported_prefill_load(), None);
    }

    #[tokio::test]
    async fn poll_round_reads_sglang_proxy_running_and_queue_load() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/get_load",
                get(|| async {
                    Json(json!([
                        {
                            "dp_rank": 0,
                            "num_reqs": 3,
                            "num_waiting_reqs": 2,
                            "num_running_reqs": 1,
                            "num_waiting_uncached_tokens": 700,
                            "load_role": "null"
                        },
                        {
                            "dp_rank": 1,
                            "num_reqs": 1,
                            "num_waiting_reqs": 5,
                            "num_running_reqs": 2,
                            "num_waiting_uncached_tokens": 900,
                            "load_role": "null"
                        }
                    ]))
                }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("sglang-proxy".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: worker_url,
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::SglangProxy,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 1000,
                prefill_members: Vec::new(),
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        let client = reqwest::Client::new();

        poll_round(&client, &registry).await;

        assert_eq!(worker.reported_load(), 11);
        let snapshot = worker
            .reported_prefill_load()
            .expect("successful load and health probes store the Prefill snapshot");
        assert_eq!(snapshot.role, PrefillLoadRole::Integrated);
        assert_eq!(snapshot.running_requests, 3);
        assert_eq!(snapshot.total_waiting_uncached_tokens, 1600);
        server.abort();
    }

    #[tokio::test]
    async fn poll_round_reads_member_snapshots_from_proxy_v1_loads() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/v1/loads",
                get(|| async {
                    Json(json!({
                        "logical_request_pressure": 4,
                        "loads": [{
                            "worker_url": "http://p0",
                            "dp_rank": 0,
                            "load_role": "prefill",
                            "prefill_capacity_milli": 1500,
                            "num_running_reqs": 1,
                            "num_waiting_reqs": 2,
                            "num_waiting_uncached_tokens": 700
                        }]
                    }))
                }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("sglang-proxy-v1".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: worker_url,
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::SglangProxy,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 1000,
                prefill_members: vec!["http://p0".into()],
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();

        poll_round(&reqwest::Client::new(), &registry).await;

        assert_eq!(worker.reported_load(), 4);
        assert_eq!(worker.reported_prefill_load(), None);
        let members = worker.reported_prefill_members();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].worker_url, "http://p0");
        assert_eq!(members[0].prefill_capacity_milli, 1500);
        assert_eq!(members[0].snapshot.total_waiting_uncached_tokens, 700);
        server.abort();
    }

    #[tokio::test]
    async fn poll_round_does_not_fallback_on_v1_auth_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route("/v1/loads", get(|| async { StatusCode::UNAUTHORIZED }))
            .route(
                "/get_load",
                get(|| async { Json(json!([{"num_reqs": 0, "num_waiting_reqs": 0}])) }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("auth-failure".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: worker_url,
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::Sglang,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 1000,
                prefill_members: vec!["http://p0".into()],
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();

        poll_round(&reqwest::Client::new(), &registry).await;

        assert_eq!(worker.reported_load(), REPORTED_LOAD_FAILED);
        assert!(!worker.introspection_probe_allows_routing());
        server.abort();
    }

    #[tokio::test]
    async fn poll_round_uses_legacy_load_api_for_physical_engine() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::OK }))
            .route(
                "/v1/loads",
                get(|| async { StatusCode::INTERNAL_SERVER_ERROR }),
            )
            .route(
                "/get_load",
                get(|| async {
                    Json(json!([{
                        "num_reqs": 1,
                        "num_waiting_reqs": 2,
                        "num_running_reqs": 1,
                        "num_waiting_uncached_tokens": 700,
                        "load_role": "prefill"
                    }]))
                }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("physical-prefill".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: worker_url,
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::Sglang,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 400,
                prefill_members: Vec::new(),
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();

        poll_round(&reqwest::Client::new(), &registry).await;

        assert_eq!(worker.reported_load(), 3);
        let snapshot = worker
            .reported_prefill_load()
            .expect("physical Prefill should keep its legacy scheduler snapshot");
        assert_eq!(snapshot.role, PrefillLoadRole::Prefill);
        assert_eq!(snapshot.total_waiting_uncached_tokens, 700);
        server.abort();
    }

    fn test_worker(
        id: &str,
        url: &str,
        mode: WorkerMode,
        backend: WorkerBackend,
        prefill_members: Vec<String>,
    ) -> Arc<crate::workers::worker::Worker> {
        Arc::new(crate::workers::worker::Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: url.into(),
            mode,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend,
            tier: Default::default(),
            routes: crate::discovery::WorkerRouteSet::all(),
            prefill_capacity_milli: 1000,
            prefill_members,
        }))
    }

    #[tokio::test]
    async fn poll_round_rejects_worker_when_health_fails_but_load_is_zero() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route("/health", get(|| async { StatusCode::SERVICE_UNAVAILABLE }))
            .route(
                "/get_load",
                get(|| async {
                    Json(json!([
                        {"dp_rank": 0, "num_reqs": 0, "num_waiting_reqs": 0}
                    ]))
                }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("idle-but-unhealthy".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: worker_url,
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::Sglang,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 1000,
                prefill_members: Vec::new(),
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        worker.set_reported_prefill_load(Some(PrefillLoadSnapshot {
            role: PrefillLoadRole::Prefill,
            running_requests: 1,
            total_waiting_uncached_tokens: 512,
            candidate: None,
        }));

        poll_round(&reqwest::Client::new(), &registry).await;

        assert_eq!(worker.reported_load(), REPORTED_LOAD_FAILED);
        assert_eq!(worker.reported_prefill_load(), None);
        assert!(!worker.introspection_probe_allows_routing());
        server.abort();
    }

    #[tokio::test]
    async fn poll_round_honors_configured_probe_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let worker_url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new()
            .route(
                "/health",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    StatusCode::OK
                }),
            )
            .route(
                "/get_load",
                get(|| async {
                    Json(json!([
                        {"dp_rank": 0, "num_reqs": 0, "num_waiting_reqs": 0}
                    ]))
                }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let registry = Arc::new(WorkerRegistry::default());
        let id = WorkerId("slow-health".into());
        registry
            .add(WorkerSpec {
                id: id.clone(),
                url: worker_url,
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                min_priority: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: WorkerBackend::Sglang,
                tier: Default::default(),
                routes: crate::discovery::WorkerRouteSet::all(),
                prefill_capacity_milli: 1000,
                prefill_members: Vec::new(),
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        let client = build_probe_client(Duration::from_millis(20), None);

        tokio::time::timeout(Duration::from_secs(1), poll_round(&client, &registry))
            .await
            .expect("probe round must respect the configured client timeout");

        assert_eq!(worker.reported_load(), REPORTED_LOAD_FAILED);
        assert!(!worker.introspection_probe_allows_routing());
        server.abort();
    }
}
