// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP-level coverage for heterogeneous worker context windows.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

fn config() -> Config {
    config_for_model("tiny")
}

fn config_for_model(model: &str) -> Config {
    Config {
        runtime_mode: sgl_router::config::RuntimeMode::Gateway,
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: model.into(),
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
        allow_raw_context_tokens: false,
    }
}

fn worker_spec(id: &str, url: &str, max_context_tokens: Option<usize>) -> WorkerSpec {
    worker_spec_for_model(id, url, "tiny", max_context_tokens)
}

fn worker_spec_for_model(
    id: &str,
    url: &str,
    model: &str,
    max_context_tokens: Option<usize>,
) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(model.into())],
        bootstrap_port: None,
        min_priority: None,
        min_context_tokens: None,
        max_context_tokens,
        bearer_token: None,
        backend: Default::default(),
        tier: Default::default(),
        routes: Default::default(),
        prefill_capacity_milli: 1000,
        prefill_members: Vec::new(),
    }
}

fn ranged_worker_spec(
    id: &str,
    url: &str,
    min_context_tokens: Option<usize>,
    max_context_tokens: Option<usize>,
) -> WorkerSpec {
    let mut spec = worker_spec(id, url, max_context_tokens);
    spec.min_context_tokens = min_context_tokens;
    spec
}

fn build_ctx(specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    build_ctx_with_config(config(), specs)
}

fn build_raw_context_ctx(specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    let mut cfg = config();
    cfg.allow_raw_context_tokens = true;
    build_ctx_with_config(cfg, specs)
}

fn build_ctx_with_config(cfg: Config, specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for spec in specs {
        registry.add(spec).unwrap();
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn request(path: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn was_hit(worker: &MockWorker) -> bool {
    worker.captured.lock().unwrap().last_body.is_some()
}

#[tokio::test]
async fn over_limit_completion_only_hits_unlimited_worker() {
    let limited = MockWorker::start(vec![]).await;
    let unlimited = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        worker_spec("limited", &limited.url, Some(1)),
        worker_spec("unlimited", &unlimited.url, None),
    ]);

    for _ in 0..4 {
        let response = build_router(Arc::clone(&ctx))
            .oneshot(request(
                "/v1/completions",
                serde_json::json!({
                    "model":"tiny",
                    "prompt":"hello",
                    "max_tokens":8,
                    "stream":false
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    assert!(!was_hit(&limited));
    assert!(was_hit(&unlimited));
    assert!(ctx
        .metrics
        .render()
        .contains(r#"sgl_router_context_filtered_total{reason="worker_excluded_over_limit"} 4"#));
}

#[tokio::test]
async fn below_minimum_completion_only_hits_unbounded_worker() {
    let long_only = MockWorker::start(vec![]).await;
    let fallback = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        ranged_worker_spec("long-only", &long_only.url, Some(500_000), None),
        ranged_worker_spec("fallback", &fallback.url, None, None),
    ]);

    let response = build_router(Arc::clone(&ctx))
        .oneshot(request(
            "/v1/completions",
            serde_json::json!({
                "model":"tiny",
                "prompt":"hello",
                "max_tokens":8,
                "stream":false
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(!was_hit(&long_only));
    assert!(was_hit(&fallback));
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_context_filtered_total{reason="worker_excluded_below_minimum"} 1"#
    ));
}

#[tokio::test]
async fn default_chat_without_chat_encoder_fails_closed_for_bounded_pool() {
    let short = MockWorker::start(vec![]).await;
    let long = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        ranged_worker_spec("short", &short.url, None, Some(65_535)),
        ranged_worker_spec("long", &long.url, Some(65_536), None),
    ]);

    let response = build_router(Arc::clone(&ctx))
        .oneshot(request(
            "/v1/chat/completions",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "max_tokens":8,
                "stream":false
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(!was_hit(&short));
    assert!(!was_hit(&long));
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_context_filtered_total{reason="empty_set_rejected_unknown_length"} 1"#
    ));
}

#[tokio::test]
async fn raw_context_opt_in_chat_routes_short_request_to_short_worker() {
    let short = MockWorker::start(vec![]).await;
    let long = MockWorker::start(vec![]).await;
    let ctx = build_raw_context_ctx(vec![
        ranged_worker_spec("short", &short.url, None, Some(65_535)),
        ranged_worker_spec("long", &long.url, Some(65_536), None),
    ]);

    let response = build_router(Arc::clone(&ctx))
        .oneshot(request(
            "/v1/chat/completions",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "max_tokens":8,
                "stream":false
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(was_hit(&short));
    assert!(!was_hit(&long));
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_context_filtered_total{reason="worker_excluded_below_minimum"} 1"#
    ));
}

#[tokio::test]
async fn raw_context_opt_in_chat_with_reasoning_effort_routes_by_existing_tokens() {
    let short = MockWorker::start(vec![]).await;
    let long = MockWorker::start(vec![]).await;
    let ctx = build_raw_context_ctx(vec![
        ranged_worker_spec("short", &short.url, None, Some(65_535)),
        ranged_worker_spec("long", &long.url, Some(65_536), None),
    ]);

    let response = build_router(Arc::clone(&ctx))
        .oneshot(request(
            "/v1/chat/completions",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "max_tokens":8,
                "reasoning_effort":"high",
                "stream":false
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(was_hit(&short));
    assert!(!was_hit(&long));
    let forwarded: serde_json::Value = serde_json::from_slice(
        short
            .captured
            .lock()
            .unwrap()
            .last_body
            .as_ref()
            .expect("short worker request body"),
    )
    .unwrap();
    assert_eq!(forwarded["reasoning_effort"], "high");
    assert!(forwarded.get("input_ids").is_none());
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_context_filtered_total{reason="worker_excluded_below_minimum"} 1"#
    ));
}

#[tokio::test]
async fn raw_context_opt_in_chat_agent_shapes_route_by_existing_tokens() {
    let cases = [
        (
            "tools",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "tools":[{"type":"function","function":{"name":"noop","parameters":{"type":"object"}}}],
                "max_tokens":8,
                "reasoning_effort":"high",
                "stream":false
            }),
        ),
        (
            "content-array",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],
                "max_tokens":8,
                "reasoning_effort":"high",
                "stream":false
            }),
        ),
        (
            "template-kwargs",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "chat_template_kwargs":{"enable_thinking":true},
                "max_tokens":8,
                "reasoning_effort":"high",
                "stream":false
            }),
        ),
        (
            "task",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "task":"chat",
                "max_tokens":8,
                "reasoning_effort":"high",
                "stream":false
            }),
        ),
        (
            "continuation",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"assistant","content":"partial"}],
                "continue_final_message":true,
                "max_tokens":8,
                "reasoning_effort":"high",
                "stream":false
            }),
        ),
    ];

    for (case_name, request_body) in cases {
        let short = MockWorker::start(vec![]).await;
        let long = MockWorker::start(vec![]).await;
        let ctx = build_raw_context_ctx(vec![
            ranged_worker_spec("short", &short.url, None, Some(65_535)),
            ranged_worker_spec("long", &long.url, Some(65_536), None),
        ]);

        let response = build_router(Arc::clone(&ctx))
            .oneshot(request("/v1/chat/completions", request_body))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK, "case={case_name}");
        assert!(was_hit(&short), "case={case_name}");
        assert!(!was_hit(&long), "case={case_name}");
        let forwarded: serde_json::Value = serde_json::from_slice(
            short
                .captured
                .lock()
                .unwrap()
                .last_body
                .as_ref()
                .expect("short worker request body"),
        )
        .unwrap();
        assert!(forwarded.get("input_ids").is_none(), "case={case_name}");
    }
}

#[tokio::test]
async fn within_limit_completion_keeps_limited_worker_in_rotation() {
    let limited = MockWorker::start(vec![]).await;
    let unlimited = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        worker_spec("limited", &limited.url, Some(500_000)),
        worker_spec("unlimited", &unlimited.url, None),
    ]);

    for _ in 0..4 {
        let response = build_router(Arc::clone(&ctx))
            .oneshot(request(
                "/v1/completions",
                serde_json::json!({
                    "model":"tiny",
                    "prompt":"hello",
                    "max_tokens":8,
                    "stream":false
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    assert!(was_hit(&limited));
    assert!(was_hit(&unlimited));
}

#[tokio::test]
async fn over_limit_request_rejects_when_only_limited_worker_is_healthy() {
    let limited = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![worker_spec("limited", &limited.url, Some(1))]);
    let response = build_router(Arc::clone(&ctx))
        .oneshot(request(
            "/v1/completions",
            serde_json::json!({
                "model":"tiny",
                "prompt":"hello",
                "max_tokens":8,
                "stream":false
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers().get("x-router-error-code").unwrap(),
        "no_context_eligible_workers"
    );
    assert!(!was_hit(&limited));
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_context_filtered_total{reason="empty_set_rejected_over_limit"} 1"#
    ));
}

#[tokio::test]
async fn unknown_length_chat_messages_and_responses_skip_limited_worker() {
    // This fixture intentionally has no chat template. The router cannot
    // prove native chat-shaped prompt length, so limited workers must be
    // excluded while the unbounded worker remains available.
    let limited = MockWorker::start(vec![]).await;
    let unlimited = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        worker_spec("limited", &limited.url, Some(500_000)),
        worker_spec("unlimited", &unlimited.url, None),
    ]);

    let cases = [
        (
            "/v1/chat/completions",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "max_tokens":8,
                "stream":false
            }),
        ),
        (
            "/v1/messages",
            serde_json::json!({
                "model":"tiny",
                "messages":[{"role":"user","content":"hello"}],
                "max_tokens":8,
                "stream":false
            }),
        ),
        (
            "/v1/responses",
            serde_json::json!({
                "model":"tiny",
                "input":"hello",
                "max_output_tokens":8,
                "stream":false
            }),
        ),
    ];
    for (path, body) in cases {
        let response = build_router(Arc::clone(&ctx))
            .oneshot(request(path, body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "path: {path}");
    }

    assert!(!was_hit(&limited));
    assert!(was_hit(&unlimited));
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_context_filtered_total{reason="worker_excluded_unknown_length"} 3"#
    ));
}

#[tokio::test]
async fn responses_without_explicit_output_limit_skip_limited_worker() {
    const MODEL: &str = "deepseek-v4-tiny";

    let limited = MockWorker::start(vec![]).await;
    let unlimited = MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_config(
        config_for_model(MODEL),
        vec![
            worker_spec_for_model("limited", &limited.url, MODEL, Some(500_000)),
            worker_spec_for_model("unlimited", &unlimited.url, MODEL, None),
        ],
    );

    for _ in 0..4 {
        let response = build_router(Arc::clone(&ctx))
            .oneshot(request(
                "/v1/responses",
                serde_json::json!({
                    "model":MODEL,
                    "input":"hello",
                    "stream":false
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    assert!(!was_hit(&limited));
    assert!(was_hit(&unlimited));

    limited.captured.lock().unwrap().last_body = None;
    unlimited.captured.lock().unwrap().last_body = None;
    for _ in 0..4 {
        let response = build_router(Arc::clone(&ctx))
            .oneshot(request(
                "/v1/responses",
                serde_json::json!({
                    "model":MODEL,
                    "input":"hello",
                    "max_output_tokens":8,
                    "stream":false
                }),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    assert!(was_hit(&limited));
    assert!(was_hit(&unlimited));
}
