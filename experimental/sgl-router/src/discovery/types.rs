// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Opaque worker identifier. Wraps a string so callsites can't confuse it
/// with other string types (e.g. `ModelId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub String);

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque model identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Prefill/Decode/Plain role of a worker.
///
/// Serialises as `"plain"`, `"prefill"`, `"decode"` (snake_case).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerMode {
    Plain,
    Prefill,
    Decode,
}

/// Serving backend exposed by a worker URL.
///
/// Static URL discovery defaults to SGLang. Non-SGLang backends must be
/// opted in explicitly because they may not expose SGLang-only control
/// endpoints such as `/server_info`, `/get_load`, or KV event metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WorkerBackend {
    #[default]
    Sglang,
    /// A logical plain worker implemented by an SGLang compatibility proxy.
    ///
    /// Model discovery uses `/v1/models` so an inner PD engine's
    /// `/server_info` cannot reclassify this logical endpoint as prefill or
    /// decode. Unlike vLLM, the proxy implements SGLang `/get_load`, so it
    /// still participates in worker-reported running/queue load balancing.
    /// It does not expose a logical KV-event stream for the outer router.
    SglangProxy,
    Vllm,
}

impl WorkerBackend {
    /// Whether the logical endpoint implements SGLang `/get_load`.
    pub fn supports_sglang_load(self) -> bool {
        matches!(self, Self::Sglang | Self::SglangProxy)
    }

    /// Whether the outer router should attach to this worker's KV events.
    pub fn supports_sglang_kv_events(self) -> bool {
        matches!(self, Self::Sglang)
    }
}

/// Operator-defined capacity tier for cross-pool routing.
///
/// The default tier preserves existing behavior. Static URL discovery can
/// seed `bulk` for internal H20/vLLM capacity and `shared` for production
/// B200 workers that a low-priority gateway may borrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum WorkerTier {
    #[default]
    #[value(name = "default")]
    Default,
    #[value(name = "bulk")]
    Bulk,
    #[value(name = "shared")]
    Shared,
}

/// Request routes a worker is allowed to serve.
///
/// Defaults to all routes so existing worker registries keep their current
/// behavior. Static URL discovery can narrow this with `@routes=...` for
/// heterogeneous pools where a compatibility proxy only implements part of the
/// router-facing API surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerRouteSet {
    pub chat: bool,
    pub completions: bool,
    pub messages: bool,
    pub responses: bool,
}

impl WorkerRouteSet {
    pub const fn all() -> Self {
        Self {
            chat: true,
            completions: true,
            messages: true,
            responses: true,
        }
    }

    pub const fn chat_only() -> Self {
        Self {
            chat: true,
            completions: false,
            messages: false,
            responses: false,
        }
    }

    pub fn supports(self, route: WorkerRoute) -> bool {
        match route {
            WorkerRoute::Chat => self.chat,
            WorkerRoute::Completions => self.completions,
            WorkerRoute::Messages => self.messages,
            WorkerRoute::Responses => self.responses,
        }
    }
}

impl Default for WorkerRouteSet {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRoute {
    Chat,
    Completions,
    Messages,
    Responses,
}

/// Immutable worker description emitted by a discovery backend.
///
/// Backends emit [`DiscoveryEvent::Added`] carrying a `WorkerSpec` when a
/// new worker becomes available, and [`DiscoveryEvent::Removed`] when it
/// leaves.
///
/// `bootstrap_port` is the SGLang disagg bootstrap server port for
/// prefill workers (set via `--disaggregation-bootstrap-port` at worker
/// startup). Resolved from each worker's `/server_info` response (see
/// [`crate::workers::introspect`]); discovery backends seed it as
/// `None`. `None` for decode and plain workers — they don't own a
/// bootstrap server. The router copies the selected prefill worker's
/// `bootstrap_host`/`bootstrap_port` plus a random `bootstrap_room`
/// u64 onto every PD-disagg request body so the prefill engine can
/// match incoming KV-transfer requests from the decode peer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub id: WorkerId,
    pub url: String,
    pub mode: WorkerMode,
    pub model_ids: Vec<ModelId>,
    #[serde(default)]
    pub bootstrap_port: Option<u16>,
    /// Minimum request priority this worker will accept. `Some(N)` means
    /// the worker is eligible only for requests whose effective priority
    /// is `>= N`; `None` (the default) means it accepts any request.
    ///
    /// Used to isolate heterogeneous capacity — e.g. an RTX-6000 worker
    /// registered with `min_priority = Some(100)` only serves
    /// high-priority production traffic and never internal/long requests,
    /// which carry priority `0`. Seeded by the static-urls discovery
    /// backend (`url@min_priority=N`) today; the k8s backend currently
    /// always sets `None` (pod-label seeding is a future addition). NOT
    /// overridden by `/server_info` introspection (which only resolves
    /// mode/bootstrap) nor dropped on reconcile re-introspection.
    #[serde(default)]
    pub min_priority: Option<i64>,
    /// Maximum total context (prompt plus requested output tokens) this
    /// worker can safely serve. `None` leaves context validation to the
    /// engine. Static URL discovery seeds this from
    /// `url@max_context_tokens=N`.
    #[serde(default)]
    pub max_context_tokens: Option<usize>,
    /// Optional worker-local bearer token. When set, the router uses it
    /// for its own `/server_info` and `/get_load` calls and overrides the
    /// proxied request's `Authorization` header for this worker. This keeps
    /// legacy pools with per-worker SGLang keys compatible while leaving
    /// shared-key pools on normal inbound Authorization forwarding.
    #[serde(default)]
    pub bearer_token: Option<String>,
    /// Worker serving backend. Defaults to SGLang for backwards
    /// compatibility with existing discovery payloads.
    #[serde(default)]
    pub backend: WorkerBackend,
    /// Operator-defined routing tier. Defaults to `default` for backwards
    /// compatibility with existing discovery payloads and policies.
    #[serde(default)]
    pub tier: WorkerTier,
    /// Router-facing API routes this worker can safely serve.
    #[serde(default)]
    pub routes: WorkerRouteSet,
}

/// Event produced by a discovery backend and consumed by `WorkerManager`.
///
/// Tagged with `"event"` for JSON clarity:
/// ```json
/// {"event":"added","id":"w1","url":"http://…","mode":"plain","model_ids":["m"]}
/// {"event":"removed","id":"w1"}
/// {"event":"mode_changed","id":"w1","mode":"decode"}
/// ```
///
/// The `Added` variant wraps the full [`WorkerSpec`]; the others carry only
/// what changed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum DiscoveryEvent {
    Added(WorkerSpec),
    Removed {
        id: WorkerId,
    },
    /// Used by the k8s backend when only the PD label flips (rare).
    ModeChanged {
        id: WorkerId,
        mode: WorkerMode,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_spec_serde_round_trip() {
        let w = WorkerSpec {
            id: WorkerId("w1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("qwen".into())],
            bootstrap_port: None,
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend: WorkerBackend::Sglang,
            tier: WorkerTier::Default,
            routes: WorkerRouteSet::all(),
        };
        let s = serde_json::to_string(&w).unwrap();
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_with_bootstrap_port_round_trip() {
        let w = WorkerSpec {
            id: WorkerId("p1".into()),
            url: "http://10.0.0.1:30000".into(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("qwen".into())],
            bootstrap_port: Some(8997),
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend: WorkerBackend::Sglang,
            tier: WorkerTier::Default,
            routes: WorkerRouteSet::all(),
        };
        let s = serde_json::to_string(&w).unwrap();
        assert!(s.contains("\"bootstrap_port\":8997"));
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_with_min_priority_round_trip() {
        let w = WorkerSpec {
            id: WorkerId("rtx1".into()),
            url: "http://10.0.0.9:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("glm".into())],
            bootstrap_port: None,
            min_priority: Some(100),
            max_context_tokens: None,
            bearer_token: None,
            backend: WorkerBackend::Sglang,
            tier: WorkerTier::Default,
            routes: WorkerRouteSet::all(),
        };
        let s = serde_json::to_string(&w).unwrap();
        assert!(s.contains("\"min_priority\":100"));
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_deserializes_with_missing_min_priority() {
        // Older configs / hand-written JSON without the field should still
        // parse — min_priority defaults to None (worker accepts any request).
        let json = r#"{"id":"w","url":"http://x","mode":"plain","model_ids":["m"]}"#;
        let w: WorkerSpec = serde_json::from_str(json).unwrap();
        assert_eq!(w.min_priority, None);
    }

    #[test]
    fn worker_spec_deserializes_with_missing_max_context_tokens() {
        let json = r#"{"id":"w","url":"http://x","mode":"plain","model_ids":["m"]}"#;
        let w: WorkerSpec = serde_json::from_str(json).unwrap();
        assert_eq!(w.max_context_tokens, None);
    }

    #[test]
    fn worker_spec_with_max_context_tokens_round_trip() {
        let mut w: WorkerSpec = serde_json::from_str(
            r#"{"id":"amd","url":"http://10.0.0.8:30000","mode":"plain","model_ids":["glm"]}"#,
        )
        .unwrap();
        w.max_context_tokens = Some(500_000);
        let s = serde_json::to_string(&w).unwrap();
        assert!(s.contains("\"max_context_tokens\":500000"));
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_deserializes_with_missing_backend() {
        let json = r#"{"id":"w","url":"http://x","mode":"plain","model_ids":["m"]}"#;
        let w: WorkerSpec = serde_json::from_str(json).unwrap();
        assert_eq!(w.backend, WorkerBackend::Sglang);
    }

    #[test]
    fn worker_spec_deserializes_with_missing_tier() {
        let json = r#"{"id":"w","url":"http://x","mode":"plain","model_ids":["m"]}"#;
        let w: WorkerSpec = serde_json::from_str(json).unwrap();
        assert_eq!(w.tier, WorkerTier::Default);
    }

    #[test]
    fn worker_spec_with_vllm_backend_round_trip() {
        let w = WorkerSpec {
            id: WorkerId("vllm1".into()),
            url: "http://10.0.0.7:8006".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("glm".into())],
            bootstrap_port: None,
            min_priority: Some(100),
            max_context_tokens: None,
            bearer_token: None,
            backend: WorkerBackend::Vllm,
            tier: WorkerTier::Bulk,
            routes: WorkerRouteSet::all(),
        };
        let s = serde_json::to_string(&w).unwrap();
        assert!(s.contains("\"backend\":\"vllm\""));
        assert!(s.contains("\"tier\":\"bulk\""));
        let d: WorkerSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(w, d);
    }

    #[test]
    fn worker_spec_deserializes_with_missing_bootstrap_port() {
        // Older configs / hand-written JSON without the field should
        // still parse — bootstrap_port defaults to None for non-PD
        // deployments.
        let json = r#"{"id":"w","url":"http://x","mode":"plain","model_ids":["m"]}"#;
        let w: WorkerSpec = serde_json::from_str(json).unwrap();
        assert_eq!(w.bootstrap_port, None);
    }

    #[test]
    fn worker_mode_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&WorkerMode::Plain).unwrap(),
            "\"plain\""
        );
        assert_eq!(
            serde_json::to_string(&WorkerMode::Prefill).unwrap(),
            "\"prefill\""
        );
        assert_eq!(
            serde_json::to_string(&WorkerMode::Decode).unwrap(),
            "\"decode\""
        );
    }

    #[test]
    fn discovery_event_round_trip() {
        let e = DiscoveryEvent::Added(WorkerSpec {
            id: WorkerId("w1".into()),
            url: "http://x:30000".into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m1".into())],
            bootstrap_port: None,
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend: WorkerBackend::Sglang,
            tier: Default::default(),
            routes: WorkerRouteSet::all(),
        });
        let s = serde_json::to_string(&e).unwrap();
        let d: DiscoveryEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(e, d);
    }
}
