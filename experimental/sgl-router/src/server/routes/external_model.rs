// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Direct routing for one fixed external OpenAI-compatible model.

use crate::server::app_context::AppContext;
use crate::server::entry_auth::GatewayKeyIdentity;
use crate::server::error::ApiError;
use crate::server::routes::chat::make_client_disconnect_hook;
use crate::server::trace::TraceContext;
use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, Response};
use bytes::Bytes;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
struct ModelProbe {
    model: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
}

/// Forward a matching configured external model and return `None` for every
/// other model. Invalid or missing model bodies are left to the normal route
/// handler so it keeps the endpoint-specific error envelope.
pub async fn maybe_forward(
    ctx: &Arc<AppContext>,
    inbound_headers: &HeaderMap,
    body: &Bytes,
    path: &'static str,
    identity: Option<&GatewayKeyIdentity>,
) -> Result<Option<Response<Body>>, ApiError> {
    let Some(cfg) = ctx.config.external_model.as_ref() else {
        return Ok(None);
    };
    let Ok(probe) = serde_json::from_slice::<ModelProbe>(body) else {
        return Ok(None);
    };
    if probe.model.as_deref() != Some(cfg.model_id.as_str()) {
        return Ok(None);
    }
    if identity.is_some_and(|identity| !identity.allows_external_model()) {
        return Err(ApiError::ModelNotFound(cfg.model_id.clone()));
    }
    let streaming = probe.stream.unwrap_or(false);

    let mut headers = inbound_headers.clone();
    let mut value =
        HeaderValue::from_str(&format!("Bearer {}", cfg.bearer_token)).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context("build external model auth header"))
        })?;
    value.set_sensitive(true);
    headers.insert(header::AUTHORIZATION, value);
    headers.remove("x-api-key");
    headers.remove("ocp-apim-subscription-key");

    let trace_ctx = TraceContext::new(
        &mut headers,
        "POST",
        path,
        Some(cfg.model_id.clone()),
        streaming,
        body.clone(),
    );
    let breaker = ctx.external_model_breaker.as_ref().ok_or_else(|| {
        ApiError::Internal(anyhow::anyhow!(
            "external model configured without circuit breaker"
        ))
    })?;
    let request_id = inbound_headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    tracing::info!(
        request_id,
        model = %cfg.model_id,
        path,
        upstream = %cfg.base_url,
        "external model route selected",
    );

    let response = if streaming {
        ctx.proxy
            .forward_streaming_to_traced(
                &cfg.base_url,
                breaker,
                path,
                &headers,
                body.clone(),
                None,
                None,
                Some(make_client_disconnect_hook(Arc::clone(&ctx.metrics))),
                ctx.trace_sink.clone(),
                trace_ctx,
            )
            .await?
    } else {
        ctx.proxy
            .forward_json_to_traced(
                &cfg.base_url,
                breaker,
                path,
                &headers,
                body.clone(),
                ctx.trace_sink.clone(),
                trace_ctx,
            )
            .await?
    };
    Ok(Some(response))
}
