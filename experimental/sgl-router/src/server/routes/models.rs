// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::entry_auth::GatewayKeyIdentity;
use axum::extract::{Extension, State};
use axum::Json;
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize)]
pub struct ModelsList {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

pub async fn list_models(
    State(ctx): State<Arc<AppContext>>,
    identity: Option<Extension<GatewayKeyIdentity>>,
) -> Json<ModelsList> {
    // The router serves a single configured model; OpenAI clients still
    // expect a list shape, so return a one-element `data` array.
    let m = &ctx.config.model;
    let mut data = vec![ModelEntry {
        id: m.id.clone(),
        object: "model",
        owned_by: "sglang",
    }];
    let external_allowed = match identity.as_ref() {
        Some(identity) => identity.allows_external_model(),
        None => true,
    };
    if external_allowed {
        if let Some(external) = &ctx.config.external_model {
            data.push(ModelEntry {
                id: external.model_id.clone(),
                object: "model",
                owned_by: "external",
            });
        }
    }
    Json(ModelsList {
        object: "list",
        data,
    })
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use crate::config::PolicyKind;

    #[tokio::test]
    async fn lists_configured_model() {
        let mut ctx = crate::server::app_context::AppContext::stub();
        ctx.config.model = crate::config::ModelConfig {
            id: "qwen3".into(),
            tokenizer_path: "x".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            tiered_spillover: None,
            sticky: None,
        };
        let app = crate::server::app::build_router(std::sync::Arc::new(ctx));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        let ids: Vec<&str> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["qwen3"]);
        assert_eq!(v["data"][0]["object"], "model");
        // Pin `owned_by` so a refactor that flips the hardcoded value to
        // "openai" / "" / a typo would fail loudly here. OpenAI clients
        // expect this field and some (e.g. langchain-openai) treat
        // `owned_by != "system"` as a meaningful signal.
        assert_eq!(v["data"][0]["owned_by"], "sglang");
    }

    #[tokio::test]
    async fn lists_configured_external_model() {
        let mut ctx = crate::server::app_context::AppContext::stub();
        ctx.config.external_model = Some(crate::config::ExternalModelConfig {
            model_id: "macaron-a2ui-tall".into(),
            base_url: "https://provider.example".into(),
            bearer_token: "provider-secret".into(),
        });
        let app = crate::server::app::build_router(std::sync::Arc::new(ctx));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let ids: Vec<&str> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["stub-model", "macaron-a2ui-tall"]);
        assert_eq!(v["data"][1]["owned_by"], "external");
    }
}
