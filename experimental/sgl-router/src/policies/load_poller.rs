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

use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::policies::active_load::JanitorHandle;
use crate::workers::worker::{REPORTED_LOAD_FAILED, REPORTED_LOAD_UNSET};
use crate::workers::WorkerRegistry;

/// Per-`dp_rank` entry from a worker's `/get_load` response, e.g.
/// `[{"dp_rank":0,"num_reqs":0,"num_waiting_reqs":0,"num_tokens":0,...}, ...]`.
/// Unknown fields are ignored; routing needs the request pressure.
#[derive(Debug, Deserialize)]
struct GetLoadEntry {
    #[serde(default)]
    num_reqs: i64,
    #[serde(default)]
    num_waiting_reqs: i64,
}

/// Per-request timeout for worker introspection GETs. Small: both responses
/// are tiny and a slow worker should fail fast rather than stall the round.
const WORKER_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Sum active plus waiting requests across all dp ranks reported by one worker.
///
/// Borrowing production B200 capacity must treat already-running production
/// requests as busy, not only queued requests. Counting both fields keeps
/// idle-borrow routing conservative.
/// Returns `None` if the body does not parse as the expected array.
fn parse_total_request_pressure(body: &str) -> Option<i64> {
    let entries: Vec<GetLoadEntry> = serde_json::from_str(body).ok()?;
    Some(
        entries
            .iter()
            .map(|e| e.num_reqs.max(0).saturating_add(e.num_waiting_reqs.max(0)))
            .sum(),
    )
}

fn worker_get(
    client: &reqwest::Client,
    worker: &crate::workers::worker::Worker,
    url: &str,
) -> reqwest::RequestBuilder {
    let mut req = client.get(url).timeout(WORKER_PROBE_TIMEOUT);
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
        tracing::debug!(
            worker_url = %worker.url,
            backend = ?worker.backend(),
            "load-poller: skipping worker backend without SGLang /get_load"
        );
        return;
    }
    let base = worker.url.trim_end_matches('/');
    let load_url = format!("{base}/get_load");
    let health_url = format!("{base}/health");
    let load_probe = async {
        let resp = worker_get(client, worker, &load_url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let body = resp.text().await.ok()?;
        parse_total_request_pressure(&body)
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
        (Some(waiting), true) => worker.set_reported_load(waiting),
        (load, health_ok) => {
            worker.set_reported_load(REPORTED_LOAD_FAILED);
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
    bearer: Option<String>,
) -> JanitorHandle {
    let mut builder = reqwest::Client::builder().timeout(WORKER_PROBE_TIMEOUT);
    if let Some(token) = bearer.as_deref() {
        let mut headers = reqwest::header::HeaderMap::new();
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .expect("worker introspect key must be a valid HTTP header value");
        value.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, value);
        builder = builder.default_headers(headers);
    }
    let client = builder.build().expect("load-poller http client builds");

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
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        worker.set_reported_load(REPORTED_LOAD_FAILED);
        let client = reqwest::Client::new();

        poll_round(&client, &registry).await;

        assert_eq!(worker.reported_load(), REPORTED_LOAD_UNSET);
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
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        let client = reqwest::Client::new();

        poll_round(&client, &registry).await;

        assert_eq!(worker.reported_load(), REPORTED_LOAD_FAILED);
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
                        {"dp_rank": 0, "num_reqs": 3, "num_waiting_reqs": 2},
                        {"dp_rank": 1, "num_reqs": 1, "num_waiting_reqs": 5}
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
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();
        let client = reqwest::Client::new();

        poll_round(&client, &registry).await;

        assert_eq!(worker.reported_load(), 11);
        server.abort();
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
            })
            .unwrap();
        let worker = registry.get(&id).unwrap();

        poll_round(&reqwest::Client::new(), &registry).await;

        assert_eq!(worker.reported_load(), REPORTED_LOAD_FAILED);
        assert!(!worker.introspection_probe_allows_routing());
        server.abort();
    }
}
