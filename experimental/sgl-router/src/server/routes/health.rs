// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::{
    config::RuntimeMode,
    discovery::{ModelId, WorkerMode},
};
use axum::extract::State;
use axum::http::StatusCode;
use std::sync::Arc;

/// Always returns 200 — liveness probe.
pub async fn healthz() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe — 200 only when the pod can actually serve traffic.
///
/// Requires BOTH:
/// 1. `AppContext::mark_ready()` was called by main (process bootstrap
///    finished — config loaded, tokenizers built, server bound), AND
/// 2. A usable worker shape is registered. A gateway needs any worker;
///    a dedicated PD proxy needs at least one healthy Prefill and Decode.
///    Without this second check,
///    `/readyz` flips green before the first `DiscoveryEvent::Added`
///    has been processed — the Service starts sending traffic to a
///    pod whose registry is empty, and every request returns 503
///    `no_healthy_workers`.
pub async fn readyz(State(ctx): State<Arc<AppContext>>) -> StatusCode {
    let workers_ready = match ctx.config.runtime_mode {
        RuntimeMode::PdProxy => {
            let model = ModelId(ctx.config.model.id.clone());
            let workers = ctx.registry.routable_workers_for(&model);
            workers
                .iter()
                .any(|worker| worker.mode() == WorkerMode::Prefill)
                && workers
                    .iter()
                    .any(|worker| worker.mode() == WorkerMode::Decode)
        }
        RuntimeMode::Gateway | RuntimeMode::CacheState | RuntimeMode::RouterState => {
            !ctx.registry.is_empty()
        }
    };
    if ctx.is_ready() && workers_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn liveness_aliases_always_200() {
        let app = crate::server::app::build_router(test_ctx(false, false));
        for path in ["/health", "/healthz"] {
            let res = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{path}");
        }
    }

    #[tokio::test]
    async fn readyz_503_when_not_ready() {
        let app = crate::server::app::build_router(test_ctx(false, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readyz_503_when_ready_but_registry_empty() {
        // Regression: `/readyz` previously returned 200 the moment
        // `mark_ready()` was called, even with an empty worker
        // registry. The Service would route traffic to a pod that
        // could only return 503 no_healthy_workers.
        let app = crate::server::app::build_router(test_ctx(true, false));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "ready=true + empty registry must still be 503"
        );
    }

    #[tokio::test]
    async fn readyz_200_when_ready_and_worker_registered() {
        let app = crate::server::app::build_router(test_ctx(true, true));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn pd_proxy_readyz_requires_both_prefill_and_decode() {
        let only_prefill = test_ctx_with_modes(true, RuntimeMode::PdProxy, &[WorkerMode::Prefill]);
        let res = crate::server::app::build_router(only_prefill)
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);

        let pd = test_ctx_with_modes(
            true,
            RuntimeMode::PdProxy,
            &[WorkerMode::Prefill, WorkerMode::Decode],
        );
        let res = crate::server::app::build_router(pd)
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn pd_proxy_readyz_rejects_load_probe_failed_decode() {
        let pd = test_ctx_with_modes(
            true,
            RuntimeMode::PdProxy,
            &[WorkerMode::Prefill, WorkerMode::Decode],
        );
        let decode = pd
            .registry
            .workers_for(&ModelId("stub-model".into()))
            .into_iter()
            .find(|worker| worker.mode() == WorkerMode::Decode)
            .unwrap();
        decode.set_reported_load(crate::workers::worker::REPORTED_LOAD_FAILED);

        let res = crate::server::app::build_router(pd)
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn pd_proxy_readyz_accepts_one_usable_decode_when_peer_probe_failed() {
        let pd = test_ctx_with_modes(
            true,
            RuntimeMode::PdProxy,
            &[WorkerMode::Prefill, WorkerMode::Decode, WorkerMode::Decode],
        );
        let failed = pd
            .registry
            .workers_for(&ModelId("stub-model".into()))
            .into_iter()
            .find(|worker| worker.mode() == WorkerMode::Decode)
            .unwrap();
        failed.set_reported_load(crate::workers::worker::REPORTED_LOAD_FAILED);

        let res = crate::server::app::build_router(pd)
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    fn test_ctx(ready: bool, with_worker: bool) -> Arc<AppContext> {
        let modes = if with_worker {
            &[WorkerMode::Plain][..]
        } else {
            &[]
        };
        test_ctx_with_modes(ready, RuntimeMode::Gateway, modes)
    }

    fn test_ctx_with_modes(
        ready: bool,
        runtime_mode: RuntimeMode,
        modes: &[WorkerMode],
    ) -> Arc<AppContext> {
        use crate::discovery::{WorkerId, WorkerSpec};
        let mut ctx = AppContext::stub();
        ctx.config.runtime_mode = runtime_mode;
        if ready {
            ctx.mark_ready();
        }
        for (index, mode) in modes.iter().copied().enumerate() {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(format!("test-w-{index}")),
                    url: format!("http://test-{index}:30000"),
                    mode,
                    model_ids: vec![ModelId("stub-model".into())],
                    bootstrap_port: (mode == WorkerMode::Prefill).then_some(8998),
                    min_priority: None,
                    max_context_tokens: None,
                    bearer_token: None,
                    backend: Default::default(),
                    tier: Default::default(),
                    routes: crate::discovery::WorkerRouteSet::all(),
                    prefill_capacity_milli: 1000,
                    prefill_members: Vec::new(),
                })
                .expect("test worker accepted");
        }
        Arc::new(ctx)
    }
}
