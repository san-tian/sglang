// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Integration coverage for the fixed external-model route used by the
//! production gateway to keep `macaron-a2ui-tall` outside the local pool.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ExternalModelConfig, ModelConfig,
    ObservabilityConfig, PolicyKind, PriorityOverrideConfig, ProxyConfig, ServerConfig,
    StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::{build_router, build_router_with_gateway_keyring};
use sgl_router::server::app_context::AppContext;
use sgl_router::server::entry_auth::GatewayKeyring;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const EXTERNAL_MODEL: &str = "macaron-a2ui-tall";
const PROVIDER_TOKEN: &str = "provider-secret";
const LOCAL_GLM_MODEL: &str = "zai-org/GLM-5.2-FP8";

fn base_config(external_url: String) -> Config {
    Config {
        runtime_mode: sgl_router::config::RuntimeMode::Gateway,
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
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
        priority_override: PriorityOverrideConfig {
            force_request_priority: Some(100),
            trusted_priority_header: None,
            trusted_priority_secret_header: None,
            trusted_priority_secret: None,
        },
        worker_introspect_key: None,
        load_poll_interval_secs: None,
        cache_tree_page_size: None,
        cache_tree_bigram: false,
        cache_tree_max_nodes: 1_000_000,
        cache_state_url: None,
        cache_state_timeout_ms: 20,
        alias_fallback: None,
        external_model: Some(ExternalModelConfig {
            model_id: EXTERNAL_MODEL.into(),
            base_url: external_url,
            bearer_token: PROVIDER_TOKEN.into(),
        }),
    }
}

fn build_test_app(cfg: Config, local_worker_url: String) -> axum::Router {
    let ctx = build_test_context(cfg, local_worker_url);
    build_router(ctx)
}

fn build_authenticated_test_app(cfg: Config, local_worker_url: String) -> axum::Router {
    let ctx = build_test_context(cfg, local_worker_url);
    let keyring = Arc::new(
        GatewayKeyring::from_json(
            r#"{
                "version": 1,
                "keys": [
                    {"key_id":"external-test","class":"external","enabled":true},
                    {"key_id":"internal-test","class":"internal","enabled":true}
                ]
            }"#,
            r#"{
                "external-test":"external-client-secret",
                "internal-test":"internal-client-secret"
            }"#,
        )
        .unwrap(),
    );
    build_router_with_gateway_keyring(ctx, keyring)
}

fn build_test_context(cfg: Config, local_worker_url: String) -> Arc<AppContext> {
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("local".into()),
            url: local_worker_url,
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
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

fn external_request(path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("authorization", "Bearer client-secret")
        .header("x-api-key", "client-anthropic-secret")
        .header("ocp-apim-subscription-key", "legacy-apim-secret")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn authenticated_external_request(api_key: &str, priority: i64) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {api_key}"))
        .header("x-api-key", api_key)
        .header("ocp-apim-subscription-key", api_key)
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": EXTERNAL_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "priority": priority,
            }))
            .unwrap(),
        ))
        .unwrap()
}

#[tokio::test]
async fn external_model_uses_fixed_upstream_for_all_supported_paths() {
    let external = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let app = build_test_app(base_config(external.url.clone()), local.url.clone());

    let cases = [
        (
            "/v1/chat/completions",
            json!({"model": EXTERNAL_MODEL, "messages": [{"role": "user", "content": "hi"}], "priority": 0}),
        ),
        (
            "/v1/completions",
            json!({"model": EXTERNAL_MODEL, "prompt": "hi", "priority": 0}),
        ),
        (
            "/v1/messages",
            json!({"model": EXTERNAL_MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "priority": 0}),
        ),
        (
            "/v1/messages/count_tokens",
            json!({"model": EXTERNAL_MODEL, "messages": [{"role": "user", "content": "hi"}], "priority": 0}),
        ),
        (
            "/v1/responses",
            json!({"model": EXTERNAL_MODEL, "input": "hi", "max_output_tokens": 16, "priority": 0}),
        ),
    ];

    for (path, body) in cases {
        let response = app
            .clone()
            .oneshot(external_request(path, body))
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "external route failed for {path}");

        let captured = external.captured.lock().unwrap();
        assert_eq!(
            captured.headers.get("authorization").map(String::as_str),
            Some("Bearer provider-secret"),
            "gateway must replace client auth for {path}",
        );
        assert!(!captured.seen.contains("x-api-key"));
        assert!(!captured.seen.contains("ocp-apim-subscription-key"));
        let forwarded: Value =
            serde_json::from_slice(captured.last_body.as_ref().unwrap()).unwrap();
        assert_eq!(forwarded["model"], EXTERNAL_MODEL);
        assert_eq!(forwarded["priority"], 100);
    }

    assert!(
        local.captured.lock().unwrap().last_body.is_none(),
        "external requests must not enter the local worker pool",
    );
}

#[tokio::test]
async fn external_model_reasoning_dialect_is_forwarded_without_glm_normalization() {
    let external = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let mut cfg = base_config(external.url.clone());
    // Make the local side GLM-5.2 so this test proves the external-route
    // ordering, rather than passing merely because compatibility is disabled.
    cfg.model.id = LOCAL_GLM_MODEL.into();
    let app = build_test_app(cfg, local.url.clone());

    let cases = [
        (
            "/v1/chat/completions",
            json!({
                "model":EXTERNAL_MODEL,
                "messages":[{"role":"user","content":"hi"}],
                "reasoning_effort":"minimal"
            }),
            json!("minimal"),
        ),
        (
            "/v1/responses",
            json!({
                "model":EXTERNAL_MODEL,
                "input":"hi",
                "reasoning":{"effort":"max"}
            }),
            json!("max"),
        ),
        (
            "/v1/messages",
            json!({
                "model":EXTERNAL_MODEL,
                "messages":[{"role":"user","content":"hi"}],
                "output_config":{"effort":"low"}
            }),
            json!("low"),
        ),
    ];

    for (path, body, expected_effort) in cases {
        let response = app
            .clone()
            .oneshot(external_request(path, body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let forwarded: Value = serde_json::from_slice(
            external
                .captured
                .lock()
                .unwrap()
                .last_body
                .as_ref()
                .unwrap(),
        )
        .unwrap();
        let actual = match path {
            "/v1/chat/completions" => &forwarded["reasoning_effort"],
            "/v1/responses" => &forwarded["reasoning"]["effort"],
            "/v1/messages" => &forwarded["output_config"]["effort"],
            _ => unreachable!(),
        };
        assert_eq!(actual, &expected_effort, "{path}");
        assert!(forwarded.get("thinking").is_none(), "{path}");
    }

    assert!(local.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn external_chat_repairs_double_encoded_tool_arguments_before_forwarding() {
    let external = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let app = build_test_app(base_config(external.url.clone()), local.url.clone());
    let object = r#"{"city":"Beijing"}"#;

    let response = app
        .oneshot(external_request(
            "/v1/chat/completions",
            json!({
                "model": EXTERNAL_MODEL,
                "messages": [{
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": serde_json::to_string(object).unwrap(),
                        }
                    }]
                }]
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let captured = external.captured.lock().unwrap();
    let forwarded: Value =
        serde_json::from_slice(captured.last_body.as_ref().expect("external request body"))
            .unwrap();
    let arguments = forwarded["messages"][0]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(arguments).unwrap(),
        json!({"city": "Beijing"})
    );
    assert!(local.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn external_chat_rejects_unrepairable_tool_arguments_before_upstream() {
    let external = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let app = build_test_app(base_config(external.url.clone()), local.url.clone());

    let response = app
        .oneshot(external_request(
            "/v1/chat/completions",
            json!({
                "model": EXTERNAL_MODEL,
                "messages": [{
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"city\":\"SENSITIVE_VALUE\"",
                        }
                    }]
                }]
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
    assert_eq!(
        response
            .headers()
            .get("x-router-error-code")
            .and_then(|value| value.to_str().ok()),
        Some("bad_request")
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("messages[0].tool_calls[0].function.arguments"));
    assert!(!body.contains("SENSITIVE_VALUE"));
    assert!(external.captured.lock().unwrap().last_body.is_none());
    assert!(local.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn external_model_streams_without_exposing_client_credentials() {
    let chunks = vec!["data: {\"delta\":\"ok\"}\n\n", "data: [DONE]\n\n"];
    let external = crate::common::mock_worker::MockWorker::start(chunks.clone()).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let app = build_test_app(base_config(external.url.clone()), local.url.clone());

    let response = app
        .oneshot(external_request(
            "/v1/chat/completions",
            json!({
                "model": EXTERNAL_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
                "priority": 0,
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream"),
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(bytes.as_ref(), chunks.concat().as_bytes());

    let captured = external.captured.lock().unwrap();
    assert_eq!(
        captured.headers.get("authorization").map(String::as_str),
        Some("Bearer provider-secret"),
    );
    assert!(!captured.seen.contains("x-api-key"));
    assert!(!captured.seen.contains("ocp-apim-subscription-key"));
}

#[tokio::test]
async fn gateway_key_class_controls_external_model_priority_without_leaking_client_keys() {
    let external = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let app = build_authenticated_test_app(base_config(external.url.clone()), local.url.clone());

    for (api_key, requested_priority, expected_priority) in [
        ("external-client-secret", -7, 100),
        ("internal-client-secret", 999, 0),
    ] {
        let response = app
            .clone()
            .oneshot(authenticated_external_request(api_key, requested_priority))
            .await
            .unwrap();
        assert_eq!(response.status(), 200);

        let captured = external.captured.lock().unwrap();
        assert_eq!(
            captured.headers.get("authorization").map(String::as_str),
            Some("Bearer provider-secret"),
        );
        assert!(!captured.seen.contains("x-api-key"));
        assert!(!captured.seen.contains("ocp-apim-subscription-key"));
        let forwarded: Value =
            serde_json::from_slice(captured.last_body.as_ref().unwrap()).unwrap();
        assert_eq!(forwarded["priority"], expected_priority);
    }

    assert!(local.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn non_external_model_stays_in_local_pool() {
    let external = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let app = build_test_app(base_config(external.url.clone()), local.url.clone());

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), 200);
    assert!(local.captured.lock().unwrap().last_body.is_some());
    assert!(external.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn models_lists_local_and_external_model_ids() {
    let external = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let local = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let app = build_test_app(base_config(external.url.clone()), local.url.clone());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let value: Value = serde_json::from_slice(&body).unwrap();
    let ids: Vec<&str> = value["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    assert!(ids.contains(&"tiny"));
    assert!(ids.contains(&EXTERNAL_MODEL));
}
