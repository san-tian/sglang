// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Member-level Prefill load snapshots for nested PD routers.

use crate::discovery::{ModelId, WorkerMode};
use crate::server::app_context::AppContext;
use crate::workers::worker::{CandidatePrefillLoad, PrefillLoadSnapshot};
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
pub struct LoadsResponse {
    timestamp: f64,
    version: &'static str,
    logical_request_pressure: i64,
    loads: Vec<PrefillMemberLoad>,
}

#[derive(Debug, Serialize)]
struct PrefillMemberLoad {
    worker_url: String,
    dp_rank: usize,
    load_role: &'static str,
    prefill_capacity_milli: usize,
    num_running_reqs: usize,
    num_waiting_reqs: usize,
    num_waiting_uncached_tokens: usize,
    prefill_queue: Option<PrefillQueueLoad>,
}

#[derive(Debug, Serialize)]
struct PrefillQueueLoad {
    detail_complete: bool,
    chunked_remaining_uncached_tokens: usize,
    work_bucket_bounds: Vec<usize>,
    priority_scheduling_enabled: bool,
    schedule_low_priority_values_first: bool,
    priority_values: Vec<i64>,
    priority_total_uncached_tokens: Vec<usize>,
    priority_ahead_uncached_tokens: Vec<Vec<usize>>,
}

impl From<CandidatePrefillLoad> for PrefillQueueLoad {
    fn from(value: CandidatePrefillLoad) -> Self {
        Self {
            detail_complete: true,
            chunked_remaining_uncached_tokens: value.chunked_remaining_uncached_tokens,
            work_bucket_bounds: value.work_bucket_bounds,
            priority_scheduling_enabled: value.priority_scheduling_enabled,
            schedule_low_priority_values_first: value.schedule_low_priority_values_first,
            priority_values: value
                .priorities
                .iter()
                .map(|group| group.priority)
                .collect(),
            priority_total_uncached_tokens: value
                .priorities
                .iter()
                .map(|group| group.total_uncached_tokens)
                .collect(),
            priority_ahead_uncached_tokens: value
                .priorities
                .into_iter()
                .map(|group| group.ahead_uncached_tokens)
                .collect(),
        }
    }
}

pub async fn get_loads(
    State(ctx): State<Arc<AppContext>>,
) -> Result<Json<LoadsResponse>, StatusCode> {
    let model = ModelId(ctx.config.model.id.clone());
    let max_snapshot_age_ms = ctx
        .config
        .load_poll_interval_secs
        .unwrap_or(1)
        .saturating_mul(3)
        .max(5)
        .saturating_mul(1000);
    let loads = ctx
        .registry
        .routable_workers_for(&model)
        .into_iter()
        .filter(|worker| worker.mode() == WorkerMode::Prefill)
        .filter(|worker| {
            worker
                .reported_prefill_load_age_ms()
                .is_some_and(|age| age <= max_snapshot_age_ms)
        })
        .filter_map(|worker| {
            worker
                .reported_prefill_load()
                .map(|snapshot| member_load(&worker, snapshot))
        })
        .enumerate()
        .map(|(dp_rank, mut load)| {
            load.dp_rank = dp_rank;
            load
        })
        .collect::<Vec<_>>();

    if loads.is_empty() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let logical_request_pressure = crate::server::routes::get_load::decode_load_entries(&ctx)
        .iter()
        .fold(0i64, |total, entry| {
            total
                .saturating_add(entry.num_reqs.max(0))
                .saturating_add(entry.num_waiting_reqs.max(0))
        });
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    Ok(Json(LoadsResponse {
        timestamp,
        version: env!("CARGO_PKG_VERSION"),
        logical_request_pressure,
        loads,
    }))
}

fn member_load(
    worker: &crate::workers::worker::Worker,
    snapshot: PrefillLoadSnapshot,
) -> PrefillMemberLoad {
    PrefillMemberLoad {
        worker_url: worker.url.clone(),
        dp_rank: 0,
        load_role: "prefill",
        prefill_capacity_milli: worker.prefill_capacity_milli(),
        num_running_reqs: snapshot.running_requests,
        num_waiting_reqs: 0,
        num_waiting_uncached_tokens: snapshot.total_waiting_uncached_tokens,
        prefill_queue: snapshot.candidate.map(Into::into),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeMode;
    use crate::discovery::{WorkerId, WorkerRouteSet, WorkerSpec};
    use crate::workers::worker::{PrefillLoadRole, REPORTED_LOAD_UNSET};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn exposes_member_level_prefill_snapshots_and_decode_pressure() {
        let mut ctx = AppContext::stub();
        ctx.config.runtime_mode = RuntimeMode::PdProxy;
        add_worker(&mut ctx, "p0", WorkerMode::Prefill, 1500);
        add_worker(&mut ctx, "d0", WorkerMode::Decode, 1000);
        let prefill = ctx.registry.get(&WorkerId("p0".into())).unwrap();
        prefill.set_reported_prefill_load(Some(PrefillLoadSnapshot {
            role: PrefillLoadRole::Prefill,
            running_requests: 2,
            total_waiting_uncached_tokens: 900,
            candidate: None,
        }));
        let decode = ctx.registry.get(&WorkerId("d0".into())).unwrap();
        decode.set_reported_load(3);

        let response = crate::server::app::build_router(Arc::new(ctx))
            .oneshot(
                Request::builder()
                    .uri("/v1/loads?include=core,prefill_queue")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["logical_request_pressure"], 3);
        assert_eq!(value["loads"].as_array().unwrap().len(), 1);
        assert_eq!(value["loads"][0]["worker_url"], "http://p0");
        assert_eq!(value["loads"][0]["prefill_capacity_milli"], 1500);
        assert_eq!(value["loads"][0]["num_running_reqs"], 2);
        assert_eq!(value["loads"][0]["num_waiting_uncached_tokens"], 900);
    }

    #[tokio::test]
    async fn returns_503_without_a_usable_prefill_snapshot() {
        let mut ctx = AppContext::stub();
        ctx.config.runtime_mode = RuntimeMode::PdProxy;
        add_worker(&mut ctx, "p0", WorkerMode::Prefill, 1000);

        let response = crate::server::app::build_router(Arc::new(ctx))
            .oneshot(
                Request::builder()
                    .uri("/v1/loads")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn returns_only_prefills_with_usable_snapshots() {
        let mut ctx = AppContext::stub();
        ctx.config.runtime_mode = RuntimeMode::PdProxy;
        add_worker(&mut ctx, "p0", WorkerMode::Prefill, 1000);
        add_worker(&mut ctx, "p1", WorkerMode::Prefill, 1000);
        let prefill = ctx.registry.get(&WorkerId("p0".into())).unwrap();
        prefill.set_reported_prefill_load(Some(PrefillLoadSnapshot {
            role: PrefillLoadRole::Prefill,
            running_requests: 0,
            total_waiting_uncached_tokens: 0,
            candidate: None,
        }));

        let response = crate::server::app::build_router(Arc::new(ctx))
            .oneshot(
                Request::builder()
                    .uri("/v1/loads")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["loads"].as_array().unwrap().len(), 1);
        assert_eq!(value["loads"][0]["worker_url"], "http://p0");
    }

    fn add_worker(ctx: &mut AppContext, id: &str, mode: WorkerMode, capacity: usize) {
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId(id.into()),
                url: format!("http://{id}"),
                mode,
                model_ids: vec![ModelId(ctx.config.model.id.clone())],
                bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
                min_priority: None,
                min_context_tokens: None,
                max_context_tokens: None,
                bearer_token: None,
                backend: Default::default(),
                tier: Default::default(),
                routes: WorkerRouteSet::all(),
                prefill_capacity_milli: capacity,
                prefill_members: Vec::new(),
            })
            .unwrap();
        let worker = ctx.registry.get(&WorkerId(id.into())).unwrap();
        assert_eq!(worker.reported_load(), REPORTED_LOAD_UNSET);
    }
}
