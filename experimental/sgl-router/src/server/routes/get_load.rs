// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang-compatible logical load endpoint for dedicated PD proxies.

use crate::discovery::{ModelId, WorkerMode};
use crate::server::app_context::AppContext;
use crate::workers::worker::REPORTED_LOAD_UNSET;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct GetLoadEntry {
    pub dp_rank: usize,
    pub num_reqs: i64,
    pub num_waiting_reqs: i64,
    pub num_tokens: i64,
    pub num_pending_tokens: i64,
}

/// Return the aggregate load of the healthy Decode side of a PD proxy.
///
/// An outer SGLang gateway registers this endpoint as `sglang_proxy` and
/// polls `/get_load`. Each entry represents one healthy Decode worker. The
/// engine-reported running/queued pressure is cached by this router's load
/// poller; router-local reservations are reported separately as waiting work
/// so requests selected between poll ticks are visible immediately.
pub async fn get_load(
    State(ctx): State<Arc<AppContext>>,
) -> Result<Json<Vec<GetLoadEntry>>, StatusCode> {
    let model = ModelId(ctx.config.model.id.clone());
    let entries: Vec<GetLoadEntry> = ctx
        .registry
        .routable_workers_for(&model)
        .into_iter()
        .filter(|worker| worker.mode() == WorkerMode::Decode)
        .map(|worker| {
            let reported = worker.reported_load();
            let num_reqs = if reported == REPORTED_LOAD_UNSET {
                saturating_i64(worker.active_load())
            } else {
                reported.max(0)
            };
            (worker, num_reqs)
        })
        .enumerate()
        .map(|(dp_rank, (worker, num_reqs))| GetLoadEntry {
            dp_rank,
            num_reqs,
            num_waiting_reqs: saturating_i64(
                worker
                    .pending_load()
                    .saturating_add(worker.global_pending_load()),
            ),
            num_tokens: 0,
            num_pending_tokens: saturating_i64(
                worker
                    .pending_token_load()
                    .saturating_add(worker.global_pending_token_load()),
            ),
        })
        .collect();

    if entries.is_empty() {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    } else {
        Ok(Json(entries))
    }
}

fn saturating_i64(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeMode;
    use crate::discovery::{WorkerId, WorkerRouteSet, WorkerSpec};
    use crate::workers::worker::REPORTED_LOAD_FAILED;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn pd_context(loads: &[i64]) -> Arc<AppContext> {
        let mut ctx = AppContext::stub();
        ctx.config.runtime_mode = RuntimeMode::PdProxy;
        for (index, reported_load) in loads.iter().copied().enumerate() {
            let id = WorkerId(format!("decode-{index}"));
            ctx.registry
                .add(WorkerSpec {
                    id: id.clone(),
                    url: format!("http://decode-{index}:30200"),
                    mode: WorkerMode::Decode,
                    model_ids: vec![ModelId("stub-model".into())],
                    bootstrap_port: None,
                    min_priority: None,
                    max_context_tokens: None,
                    bearer_token: None,
                    backend: Default::default(),
                    tier: Default::default(),
                    routes: WorkerRouteSet::all(),
                    prefill_capacity_milli: 1000,
                    prefill_members: Vec::new(),
                })
                .expect("decode worker accepted");
            ctx.registry
                .get(&id)
                .expect("decode worker registered")
                .set_reported_load(reported_load);
        }
        Arc::new(ctx)
    }

    #[tokio::test]
    async fn returns_only_decodes_with_usable_load_snapshots() {
        let app = crate::server::app::build_router(pd_context(&[
            3,
            REPORTED_LOAD_FAILED,
            REPORTED_LOAD_UNSET,
        ]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/get_load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let entries: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
        assert_eq!(entries.len(), 2);
        let mut loads: Vec<i64> = entries
            .iter()
            .map(|entry| entry["num_reqs"].as_i64().unwrap())
            .collect();
        loads.sort_unstable();
        assert_eq!(loads, vec![0, 3]);
    }

    #[tokio::test]
    async fn returns_503_when_every_decode_load_probe_failed() {
        let app = crate::server::app::build_router(pd_context(&[REPORTED_LOAD_FAILED]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/get_load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
