// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for Gateway reasoning-effort compatibility. These
//! tests inspect the bytes received by a mock SGLang worker, not only the
//! pure normalization helpers.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const GLM_MODEL: &str = "zai-org/GLM-5.2-FP8";

fn build_ctx(model_id: &str, worker_url: &str) -> Arc<AppContext> {
    let cfg = Config {
        runtime_mode: sgl_router::config::RuntimeMode::Gateway,
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: model_id.into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            tiered_spillover: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
            bearer_keys: Vec::new(),
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        trace: sgl_router::config::TraceConfig::default(),
        priority_override: sgl_router::config::PriorityOverrideConfig::default(),
        worker_introspect_key: None,
        load_poll_interval_secs: None,
        cache_tree_page_size: None,
        cache_tree_bigram: false,
        cache_tree_max_nodes: 1_000_000,
        cache_state_url: None,
        cache_state_timeout_ms: 20,
        alias_fallback: None,
        external_model: None,
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("w1".into()),
            url: worker_url.to_string(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(model_id.into())],
            bootstrap_port: None,
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend: Default::default(),
            tier: Default::default(),
            routes: Default::default(),
            prefill_capacity_milli: 1000,
            prefill_members: Vec::new(),
        })
        .unwrap();
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn send(app: &axum::Router, path: &str, body: Value) -> StatusCode {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}

fn captured_body(worker: &crate::common::mock_worker::MockWorker) -> Value {
    let captured = worker.captured.lock().unwrap();
    serde_json::from_slice(captured.last_body.as_ref().expect("worker request body")).unwrap()
}

#[tokio::test]
async fn glm_worker_receives_endpoint_specific_reasoning_encodings() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(GLM_MODEL, &worker.url);
    let app = build_router(Arc::clone(&ctx));

    assert_eq!(
        send(
            &app,
            "/v1/chat/completions",
            json!({
                "model":GLM_MODEL,
                "messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"minimal"
            }),
        )
        .await,
        StatusCode::OK
    );
    assert_eq!(captured_body(&worker)["reasoning_effort"], "none");

    assert_eq!(
        send(
            &app,
            "/v1/chat/completions",
            json!({
                "model":GLM_MODEL,
                "messages":[{"role":"user","content":"hi"}],
                "reasoning":{"effort":"xhigh","summary":"auto"}
            }),
        )
        .await,
        StatusCode::OK
    );
    let forwarded = captured_body(&worker);
    assert_eq!(forwarded["reasoning_effort"], "max");
    assert_eq!(forwarded["reasoning"], json!({"summary":"auto"}));

    assert_eq!(
        send(
            &app,
            "/v1/responses",
            json!({
                "model":GLM_MODEL,
                "input":"hi",
                "reasoning":{"effort":"max","summary":"auto"}
            }),
        )
        .await,
        StatusCode::OK
    );
    let forwarded = captured_body(&worker);
    assert_eq!(forwarded["reasoning"]["effort"], "xhigh");
    assert_eq!(forwarded["reasoning"]["summary"], "auto");

    assert_eq!(
        send(
            &app,
            "/v1/messages",
            json!({
                "model":GLM_MODEL,
                "max_tokens":16,
                "messages":[{"role":"user","content":"hi"}],
                "output_config":{"effort":"low"}
            }),
        )
        .await,
        StatusCode::OK
    );
    let forwarded = captured_body(&worker);
    assert_eq!(forwarded["thinking"]["type"], "disabled");
    assert!(forwarded.get("output_config").is_none());

    assert_eq!(
        send(
            &app,
            "/v1/messages/count_tokens",
            json!({
                "model":GLM_MODEL,
                "messages":[{"role":"user","content":"hi"}],
                "output_config":{"effort":"future-super"}
            }),
        )
        .await,
        StatusCode::OK
    );
    let forwarded = captured_body(&worker);
    assert_eq!(forwarded["thinking"]["type"], "disabled");
    assert!(forwarded.get("output_config").is_none());

    let metrics = ctx.metrics.render();
    assert!(metrics.contains(
        r#"sgl_router_reasoning_effort_normalized_total{route="/v1/chat/completions",requested_class="minimal",effective="off"} 1"#
    ));
    assert!(metrics.contains(
        r#"sgl_router_reasoning_effort_normalized_total{route="/v1/responses",requested_class="max",effective="max"} 1"#
    ));
    assert!(metrics.contains(
        r#"sgl_router_reasoning_effort_normalized_total{route="/v1/messages/count_tokens",requested_class="unknown",effective="off"} 1"#
    ));
}

#[tokio::test]
async fn explicit_thinking_overrides_effort_quantization_across_routes() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(GLM_MODEL, &worker.url);
    let app = build_router(ctx);

    assert_eq!(
        send(
            &app,
            "/v1/responses",
            json!({
                "model":GLM_MODEL,
                "input":"hi",
                "reasoning":{"effort":"low","type":"enabled"}
            }),
        )
        .await,
        StatusCode::OK
    );
    let responses_enabled = captured_body(&worker);
    assert_eq!(responses_enabled["reasoning"]["effort"], "high");
    assert_eq!(responses_enabled["reasoning"]["type"], "enabled");

    assert_eq!(
        send(
            &app,
            "/v1/messages",
            json!({
                "model":GLM_MODEL,
                "max_tokens":16,
                "messages":[{"role":"user","content":"hi"}],
                "thinking":{"type":"enabled","budget_tokens":4096},
                "output_config":{"effort":"low"}
            }),
        )
        .await,
        StatusCode::OK
    );
    let enabled = captured_body(&worker);
    assert_eq!(enabled["thinking"]["type"], "enabled");
    assert_eq!(enabled["thinking"]["budget_tokens"], 4096);
    assert_eq!(enabled["output_config"]["effort"], "high");

    assert_eq!(
        send(
            &app,
            "/v1/messages",
            json!({
                "model":GLM_MODEL,
                "max_tokens":16,
                "messages":[{"role":"user","content":"hi"}],
                "thinking":{"type":"disabled"},
                "output_config":{"effort":"max"}
            }),
        )
        .await,
        StatusCode::OK
    );
    let disabled = captured_body(&worker);
    assert_eq!(disabled["thinking"]["type"], "disabled");
    assert!(disabled.get("output_config").is_none());
}

#[tokio::test]
async fn malformed_reasoning_is_rejected_before_worker_dispatch() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(GLM_MODEL, &worker.url);
    let app = build_router(ctx);

    assert_eq!(
        send(
            &app,
            "/v1/chat/completions",
            json!({
                "model":GLM_MODEL,
                "messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":7
            }),
        )
        .await,
        StatusCode::BAD_REQUEST
    );
    assert!(worker.captured.lock().unwrap().last_body.is_none());

    assert_eq!(
        send(
            &app,
            "/v1/messages",
            json!({
                "model":GLM_MODEL,
                "messages":[{"role":"user","content":"hi"}],
                "thinking":{"type":"future"},
                "output_config":{"effort":"low"}
            }),
        )
        .await,
        StatusCode::BAD_REQUEST
    );
    assert!(worker.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn non_glm_local_model_is_not_rewritten() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx("qwen3", &worker.url);
    let app = build_router(ctx);

    assert_eq!(
        send(
            &app,
            "/v1/chat/completions",
            json!({
                "model":"qwen3",
                "messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"minimal"
            }),
        )
        .await,
        StatusCode::OK
    );
    assert_eq!(captured_body(&worker)["reasoning_effort"], "minimal");
}
