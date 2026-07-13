// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::entry_auth::{authenticate_gateway_key, GatewayKeyring};
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use crate::server::routes::messages::MAX_MESSAGES_BODY_BYTES;
use crate::server::routes::passthrough::MAX_PASSTHROUGH_BODY_BYTES;
use crate::server::routes::responses::MAX_RESPONSES_BODY_BYTES;
use axum::extract::{DefaultBodyLimit, MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

/// Edge counters: `requests_total{route,method}` at entry (true intake, incl.
/// requests parked/shed/cancelled before dispatch), `responses_total{...,
/// status_code}` on exit (incl. early-exit 400/413/503). Their difference =
/// received-but-not-answered, invisible to post-dispatch `worker_requests_total`.
/// `route` is the matched template (not raw URI) to bound label cardinality.
async fn count_requests(State(ctx): State<Arc<AppContext>>, req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_owned();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".to_owned());
    ctx.metrics.record_ingress(&route, &method);
    let resp = next.run(req).await;
    ctx.metrics
        .record_response(&route, &method, resp.status().as_u16());
    resp
}

/// Middleware: log 413 PAYLOAD_TOO_LARGE responses with the request method
/// and URI so an operator investigating "client X gets 413s" has a
/// server-side breadcrumb. The 413 is produced by axum's `DefaultBodyLimit`
/// layer BEFORE the handler runs, so without this we would have no record
/// of which request was rejected.
async fn log_413(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let resp = next.run(req).await;
    if resp.status() == StatusCode::PAYLOAD_TOO_LARGE {
        tracing::warn!(
            %method,
            %uri,
            "request rejected with 413 PAYLOAD_TOO_LARGE (body exceeded route limit)",
        );
    }
    resp
}

pub fn build_router(ctx: Arc<AppContext>) -> Router {
    build_router_with_gateway_keyring(ctx, Arc::new(GatewayKeyring::disabled()))
}

pub fn build_router_with_gateway_keyring(
    ctx: Arc<AppContext>,
    keyring: Arc<GatewayKeyring>,
) -> Router {
    let public_routes = Router::new()
        .route("/healthz", get(crate::server::routes::health::healthz))
        .route("/readyz", get(crate::server::routes::health::readyz))
        .route("/metrics", get(crate::server::routes::metrics::metrics));

    let mut protected_routes = Router::new()
        .route(
            "/v1/models",
            get(crate::server::routes::models::list_models),
        )
        .route(
            "/v1/tokenize",
            post(crate::server::routes::tokenize::tokenize),
        )
        .route(
            "/v1/detokenize",
            post(crate::server::routes::tokenize::detokenize),
        )
        .route(
            "/v1/chat/completions",
            post(crate::server::routes::chat::chat_completions)
                .layer(DefaultBodyLimit::max(MAX_CHAT_BODY_BYTES))
                .layer(middleware::from_fn(log_413)),
        );

    if ctx.config.runtime_mode == crate::config::RuntimeMode::Gateway {
        protected_routes = protected_routes
            .route(
                "/v1/completions",
                post(crate::server::routes::passthrough::completions)
                    .layer(DefaultBodyLimit::max(MAX_PASSTHROUGH_BODY_BYTES))
                    .layer(middleware::from_fn(log_413)),
            )
            .route(
                "/v1/messages",
                post(crate::server::routes::messages::messages)
                    .layer(DefaultBodyLimit::max(MAX_MESSAGES_BODY_BYTES))
                    .layer(middleware::from_fn(log_413)),
            )
            .route(
                "/v1/messages/count_tokens",
                post(crate::server::routes::messages::count_tokens)
                    .layer(DefaultBodyLimit::max(MAX_MESSAGES_BODY_BYTES))
                    .layer(middleware::from_fn(log_413)),
            )
            .route(
                "/v1/responses",
                post(crate::server::routes::responses::responses)
                    .layer(DefaultBodyLimit::max(MAX_RESPONSES_BODY_BYTES))
                    .layer(middleware::from_fn(log_413)),
            )
            .route(
                "/flush_cache",
                post(crate::server::routes::cache::flush_cache_for_gateway),
            );
    }

    protected_routes = protected_routes.layer(middleware::from_fn_with_state(
        keyring,
        authenticate_gateway_key,
    ));

    public_routes
        .merge(protected_routes)
        // After routing, so MatchedPath is set for every route.
        .layer(middleware::from_fn_with_state(ctx.clone(), count_requests))
        .with_state(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn pd_proxy_does_not_expose_stateful_or_non_pd_generation_routes() {
        let mut ctx = AppContext::stub();
        ctx.config.runtime_mode = crate::config::RuntimeMode::PdProxy;
        let app = build_router(Arc::new(ctx));

        for path in [
            "/v1/completions",
            "/v1/messages",
            "/v1/responses",
            "/flush_cache",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }
    }
}
