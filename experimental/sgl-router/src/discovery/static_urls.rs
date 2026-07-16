// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static-URL discovery backend.
//!
//! Takes a fixed list of worker URLs and fans one [`DiscoveryEvent::Added`]
//! per entry. After the initial fan-out the task exits — there is no
//! hot-reload; topology changes require a restart.
//!
//! Each emitted [`WorkerSpec`] uses the URL itself as the `WorkerId` and
//! seeds `mode = Plain` with empty `model_ids` and `bootstrap_port = None`.
//! The worker manager fills those in from each worker's `/server_info`
//! response (see [`crate::workers::introspect`]) and overrides the seeded
//! mode/bootstrap when the worker self-discloses a PD role — so prefill,
//! decode, and plain workers can all appear in the same `urls` list and
//! end up classified correctly.
//!
//! Requires modern SGLang that exposes `disaggregation_mode` in
//! `/server_info`. Workers on older SGLang versions that predate that
//! field stay seeded as `Plain` because the manager has no signal to
//! override with — operators running PD with such a worker should use
//! the K8s backend (which can still classify via pod labels).

use crate::config::StaticUrlsDiscoveryConfig;
use crate::discovery::{
    default_prefill_capacity_milli, DiscoveryEvent, WorkerBackend, WorkerId, WorkerMode,
    WorkerRouteSet, WorkerSpec, WorkerTier,
};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tokio::sync::mpsc;

/// Token separating a worker URL from an optional minimum-priority
/// capability suffix in `--worker-urls` entries:
/// `http://host:port@min_priority=100`. A distinctive literal (not a bare
/// `@`) so it cannot collide with URL userinfo (`user:pass@host`).
const MIN_PRIORITY_TOKEN: &str = "@min_priority=";
const MIN_CONTEXT_TOKENS_TOKEN: &str = "@min_context_tokens=";
const MAX_CONTEXT_TOKENS_TOKEN: &str = "@max_context_tokens=";
const BACKEND_TOKEN: &str = "@backend=";
const TIER_TOKEN: &str = "@tier=";
const ROUTES_TOKEN: &str = "@routes=";
const PREFILL_CAPACITY_TOKEN: &str = "@prefill_capacity=";
const PREFILL_PROFILE_TOKEN: &str = "@prefill_profile=";
const PREFILL_MEMBERS_TOKEN: &str = "@prefill_members=";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkerCapabilities {
    pub min_priority: Option<i64>,
    pub min_context_tokens: Option<usize>,
    pub max_context_tokens: Option<usize>,
    pub backend: WorkerBackend,
    pub tier: WorkerTier,
    pub routes: WorkerRouteSet,
    pub prefill_capacity_milli: usize,
    pub prefill_members: Vec<String>,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            min_priority: None,
            min_context_tokens: None,
            max_context_tokens: None,
            backend: WorkerBackend::Sglang,
            tier: WorkerTier::Default,
            routes: WorkerRouteSet::all(),
            prefill_capacity_milli: default_prefill_capacity_milli(),
            prefill_members: Vec::new(),
        }
    }
}

/// Split a `--worker-urls` entry into its base URL and optional
/// `min_priority` capability. `http://h:p@min_priority=100` yields
/// `("http://h:p", Some(100))`; a plain URL yields `(url, None)`.
/// A deployment-only `@tier=...` suffix is stripped and ignored.
///
/// A present-but-unparseable suffix (e.g. `@min_priority=abc`) is a config
/// error the caller surfaces, rather than silently dropping the isolation
/// guarantee — a worker that should be priority-gated must never fall back
/// to "accept everything" because of a typo. `rsplit_once` so a `@` inside
/// the URL (userinfo) doesn't get mistaken for the capability token.
///
/// `pub(crate)` so config validation ([`crate::config::Config::validate`])
/// can strip the suffix BEFORE URL-parsing/deduping the base URL — otherwise
/// validation would run against the raw suffixed string and `url::Url` would
/// misparse `host:port@min_priority=N` as userinfo, letting malformed base
/// URLs and with/without-suffix duplicates slip past startup checks.
pub(crate) fn parse_worker_entry(entry: &str) -> Result<(String, WorkerCapabilities)> {
    let mut base = entry;
    let mut caps = WorkerCapabilities::default();
    let mut saw_prefill_capacity = false;
    let mut saw_prefill_profile = false;
    loop {
        let Some((pos, token)) = [
            MIN_PRIORITY_TOKEN,
            MIN_CONTEXT_TOKENS_TOKEN,
            MAX_CONTEXT_TOKENS_TOKEN,
            BACKEND_TOKEN,
            TIER_TOKEN,
            ROUTES_TOKEN,
            PREFILL_CAPACITY_TOKEN,
            PREFILL_PROFILE_TOKEN,
            PREFILL_MEMBERS_TOKEN,
        ]
        .into_iter()
        .filter_map(|token| base.rfind(token).map(|pos| (pos, token)))
        .max_by_key(|(pos, _)| *pos) else {
            break;
        };
        let value = &base[pos + token.len()..];
        base = &base[..pos];
        if token == MIN_PRIORITY_TOKEN {
            let prio = value.trim().parse::<i64>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid min_priority in worker URL entry {entry:?}: \
                     {value:?} is not an integer"
                )
            })?;
            caps.min_priority = Some(prio);
        } else if token == MIN_CONTEXT_TOKENS_TOKEN {
            let min_context_tokens = value.trim().parse::<usize>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid min_context_tokens in worker URL entry {entry:?}: \
                     {value:?} is not a positive integer"
                )
            })?;
            if min_context_tokens == 0 {
                return Err(anyhow::anyhow!(
                    "invalid min_context_tokens in worker URL entry {entry:?}: \
                     value must be greater than zero"
                ));
            }
            caps.min_context_tokens = Some(min_context_tokens);
        } else if token == MAX_CONTEXT_TOKENS_TOKEN {
            let max_context_tokens = value.trim().parse::<usize>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid max_context_tokens in worker URL entry {entry:?}: \
                     {value:?} is not a positive integer"
                )
            })?;
            if max_context_tokens == 0 {
                return Err(anyhow::anyhow!(
                    "invalid max_context_tokens in worker URL entry {entry:?}: \
                     value must be greater than zero"
                ));
            }
            caps.max_context_tokens = Some(max_context_tokens);
        } else if token == BACKEND_TOKEN {
            caps.backend = match value.trim() {
                "sglang" => WorkerBackend::Sglang,
                "sglang_proxy" => WorkerBackend::SglangProxy,
                "vllm" => WorkerBackend::Vllm,
                other => {
                    return Err(anyhow::anyhow!(
                        "invalid backend in worker URL entry {entry:?}: \
                         {other:?} is not one of: sglang, sglang_proxy, vllm"
                    ));
                }
            };
        } else if token == TIER_TOKEN {
            caps.tier = match value.trim() {
                "default" => WorkerTier::Default,
                "bulk" => WorkerTier::Bulk,
                "shared" => WorkerTier::Shared,
                "dedicated" => WorkerTier::Dedicated,
                other => {
                    return Err(anyhow::anyhow!(
                        "invalid tier in worker URL entry {entry:?}: \
                         {other:?} is not one of: default, bulk, shared, dedicated"
                    ));
                }
            };
        } else if token == ROUTES_TOKEN {
            caps.routes = parse_routes(value.trim(), entry)?;
        } else if token == PREFILL_CAPACITY_TOKEN {
            if saw_prefill_profile {
                return Err(anyhow::anyhow!(
                    "invalid worker URL entry {entry:?}: \
                     prefill_capacity and prefill_profile are mutually exclusive"
                ));
            }
            saw_prefill_capacity = true;
            caps.prefill_capacity_milli = parse_prefill_capacity_milli(value.trim(), entry)?;
        } else if token == PREFILL_PROFILE_TOKEN {
            if saw_prefill_capacity {
                return Err(anyhow::anyhow!(
                    "invalid worker URL entry {entry:?}: \
                     prefill_capacity and prefill_profile are mutually exclusive"
                ));
            }
            saw_prefill_profile = true;
            caps.prefill_capacity_milli = prefill_profile_capacity_milli(value.trim(), entry)?;
        } else {
            caps.prefill_members = parse_prefill_members(value.trim(), entry)?;
        }
    }
    if let (Some(min), Some(max)) = (caps.min_context_tokens, caps.max_context_tokens) {
        if min > max {
            return Err(anyhow::anyhow!(
                "invalid context range in worker URL entry {entry:?}: \
                 min_context_tokens ({min}) must not exceed max_context_tokens ({max})"
            ));
        }
    }
    Ok((base.to_string(), caps))
}

fn parse_prefill_members(value: &str, entry: &str) -> Result<Vec<String>> {
    if value.is_empty() {
        return Err(anyhow::anyhow!(
            "invalid prefill_members in worker URL entry {entry:?}: value must not be empty"
        ));
    }

    let mut members = Vec::new();
    for raw_member in value.split(',') {
        let member = raw_member.trim();
        if member.is_empty() {
            return Err(anyhow::anyhow!(
                "invalid prefill_members in worker URL entry {entry:?}: empty member URL"
            ));
        }
        members.push(normalize_worker_url(member).map_err(|e| {
            anyhow::anyhow!(
                "invalid prefill_members in worker URL entry {entry:?}: member {member:?} is invalid: {e}"
            )
        })?);
    }

    Ok(members)
}

fn prefill_profile_capacity_milli(value: &str, entry: &str) -> Result<usize> {
    match value {
        // Baseline profiles: user convention is to treat B200, B300, HK,
        // NVFP4, and NVR-P4 shapes as equivalent prefill capacity.
        "baseline"
        | "b200"
        | "b300"
        | "nvr-p4"
        | "nvfp4"
        | "b200-fp8"
        | "b300-fp8"
        | "b200-nvfp4"
        | "b300-nvfp4"
        | "alibaba-b300"
        | "hk-l20d"
        | "hk-l20d-standalone"
        | "hk-l20d-1p1d"
        | "fp8-mtp-dp1-tp8"
        | "fp8-mtp-dp1-tp8-alibaba-b300"
        | "fp8-mtp-dp1-tp8-l20d"
        | "nvfp4-mtp-dp1-tp4"
        | "nvfp4-nomtp-dp1-tp4-b300" => Ok(1000),

        // MI300X convention: one 2P2D logical worker counts as 0.5x
        // baseline prefill capacity.
        "mi300x" | "mi300x-2p2d" | "mi300x-rdma-2p2d" | "mi300x-rdma02-2p2d"
        | "mi300x-rdma03-2p2d" | "mi300x-rdma04-2p2d" | "mi300x-rdma05-2p2d" => Ok(500),

        // France 20P10D is estimated as ten MI300X 2P2D groups.
        "france-20p10d" | "mi300x-france-20p10d" => Ok(5000),

        other => Err(anyhow::anyhow!(
            "invalid prefill_profile in worker URL entry {entry:?}: \
             {other:?} is not one of the known profiles"
        )),
    }
}

fn parse_prefill_capacity_milli(value: &str, entry: &str) -> Result<usize> {
    let capacity = value.parse::<f64>().map_err(|_| {
        anyhow::anyhow!(
            "invalid prefill_capacity in worker URL entry {entry:?}: \
             {value:?} is not a positive number"
        )
    })?;
    if !capacity.is_finite() || capacity <= 0.0 {
        return Err(anyhow::anyhow!(
            "invalid prefill_capacity in worker URL entry {entry:?}: \
             value must be finite and greater than zero"
        ));
    }
    let milli = (capacity * 1000.0).round();
    if !(1.0..=(usize::MAX as f64)).contains(&milli) {
        return Err(anyhow::anyhow!(
            "invalid prefill_capacity in worker URL entry {entry:?}: \
             value is outside the supported range"
        ));
    }
    Ok(milli as usize)
}

fn parse_routes(value: &str, entry: &str) -> Result<WorkerRouteSet> {
    if value.is_empty() {
        return Err(anyhow::anyhow!(
            "invalid routes in worker URL entry {entry:?}: value must not be empty"
        ));
    }
    if value == "all" {
        return Ok(WorkerRouteSet::all());
    }

    let mut routes = WorkerRouteSet {
        chat: false,
        completions: false,
        messages: false,
        responses: false,
    };
    for item in value.split(',') {
        match item.trim() {
            "chat" | "chat_completions" | "chat-completions" => routes.chat = true,
            "completions" => routes.completions = true,
            "messages" => routes.messages = true,
            "responses" => routes.responses = true,
            "all" => return Ok(WorkerRouteSet::all()),
            "" => {
                return Err(anyhow::anyhow!(
                    "invalid routes in worker URL entry {entry:?}: empty route name"
                ));
            }
            other => {
                return Err(anyhow::anyhow!(
                    "invalid routes in worker URL entry {entry:?}: \
                     {other:?} is not one of: all, chat, completions, messages, responses"
                ));
            }
        }
    }

    Ok(routes)
}

/// Normalize worker URLs for config-key matching. This mirrors
/// `Config::validate`: strip capability suffixes, parse as an HTTP URL, and
/// drop a trailing slash so `http://x:30000` and `http://x:30000/` match the
/// same bearer-key entry.
pub(crate) fn normalize_worker_url(entry: &str) -> Result<String> {
    let (base, _caps) = parse_worker_entry(entry)?;
    let parsed = url::Url::parse(&base)
        .map_err(|e| anyhow!("worker URL entry {entry:?} is not a valid URL: {e}"))?;
    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

/// Parse static discovery configuration into the exact `WorkerSpec`s emitted
/// by the backend. Runtime-lease discovery reuses this helper so drain/rejoin
/// preserves URL identity, all capability suffixes, and per-worker bearer
/// credentials exactly as ordinary static discovery does.
pub(crate) fn build_worker_specs(cfg: &StaticUrlsDiscoveryConfig) -> Result<Vec<WorkerSpec>> {
    let parsed: Vec<(String, WorkerCapabilities)> = cfg
        .urls
        .iter()
        .map(|entry| parse_worker_entry(entry))
        .collect::<Result<_>>()?;
    let bearer_keys: HashMap<String, String> = cfg
        .bearer_keys
        .iter()
        .map(|entry| {
            Ok((
                normalize_worker_url(&entry.worker_url)?,
                entry.bearer_token.clone(),
            ))
        })
        .collect::<Result<_>>()?;

    parsed
        .into_iter()
        .map(|(url, caps)| {
            let bearer_token = normalize_worker_url(&url)
                .ok()
                .and_then(|normalized| bearer_keys.get(&normalized).cloned());
            Ok(WorkerSpec {
                id: WorkerId(url.clone()),
                url,
                mode: WorkerMode::Plain,
                model_ids: Vec::new(),
                bootstrap_port: None,
                min_priority: caps.min_priority,
                min_context_tokens: caps.min_context_tokens,
                max_context_tokens: caps.max_context_tokens,
                bearer_token,
                backend: caps.backend,
                tier: caps.tier,
                routes: caps.routes,
                prefill_capacity_milli: caps.prefill_capacity_milli,
                prefill_members: caps.prefill_members,
            })
        })
        .collect()
}

/// Spawn the static-URLs producer task and return its `JoinHandle`.
///
/// Returns `Result` for parity with [`crate::discovery::k8s::spawn`] (which
/// can fail to construct a `kube::Client`) AND because a malformed
/// `@min_priority=` suffix is rejected here rather than ignored.
pub async fn spawn(
    cfg: StaticUrlsDiscoveryConfig,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    // Parse + validate every entry up front so a bad suffix fails startup
    // loudly instead of after the task is detached.
    let specs = build_worker_specs(&cfg)?;
    let handle = tokio::spawn(async move {
        for spec in specs {
            if tx.send(DiscoveryEvent::Added(spec)).await.is_err() {
                tracing::info!(
                    "static_urls discovery: event channel closed during fan-out; exiting"
                );
                return;
            }
        }
        tracing::debug!(
            "static_urls discovery: initial fan-out complete; parking until channel closes"
        );
        // After fan-out the static backend has no further work — but
        // `server::supervisor::supervise_critical_tasks` treats *any*
        // discovery exit as fatal and flips `/readyz` to 503. Park here
        // until the consumer drops the receiver. `tx.closed()` resolves
        // the moment every `Receiver` has been dropped; the supervisor's
        // normal-shutdown path aborts this task before that. So
        // reaching the `info!` below means either (a) we lost the abort
        // race during a clean shutdown, or (b) the worker manager exited
        // unexpectedly — in case (b) the supervisor will catch the
        // subsequent discovery-task exit and `error!` + mark unready,
        // and this breadcrumb gives operator triage a starting point.
        tx.closed().await;
        tracing::info!(
            "static_urls discovery: event channel closed by receiver \
             (worker manager dropped its end, or shutdown abort raced); exiting"
        );
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_entry_plain_url_has_no_min_priority() {
        let (url, caps) = parse_worker_entry("http://w0:30000").unwrap();
        assert_eq!(url, "http://w0:30000");
        assert_eq!(caps.min_priority, None);
        assert_eq!(caps.min_context_tokens, None);
        assert_eq!(caps.max_context_tokens, None);
        assert_eq!(caps.backend, WorkerBackend::Sglang);
        assert_eq!(caps.tier, WorkerTier::Default);
        assert_eq!(caps.prefill_capacity_milli, 1000);
    }

    #[test]
    fn parse_entry_extracts_min_priority_suffix() {
        let (url, caps) = parse_worker_entry("http://rtx-01:30000@min_priority=100").unwrap();
        assert_eq!(url, "http://rtx-01:30000");
        assert_eq!(caps.min_priority, Some(100));
        assert_eq!(caps.backend, WorkerBackend::Sglang);
        assert_eq!(caps.tier, WorkerTier::Default);
    }

    #[test]
    fn parse_entry_extracts_max_context_tokens_suffix() {
        let (url, caps) =
            parse_worker_entry("http://amd-01:30000@max_context_tokens=500000").unwrap();
        assert_eq!(url, "http://amd-01:30000");
        assert_eq!(caps.max_context_tokens, Some(500_000));
        assert_eq!(caps.min_priority, None);
    }

    #[test]
    fn parse_entry_extracts_min_context_tokens_suffix() {
        let (url, caps) =
            parse_worker_entry("http://nvidia-01:30000@min_context_tokens=65536").unwrap();
        assert_eq!(url, "http://nvidia-01:30000");
        assert_eq!(caps.min_context_tokens, Some(65_536));
        assert_eq!(caps.max_context_tokens, None);
    }

    #[test]
    fn parse_entry_extracts_prefill_capacity_suffix() {
        let (url, caps) = parse_worker_entry("http://mi300x:30000@prefill_capacity=0.5").unwrap();
        assert_eq!(url, "http://mi300x:30000");
        assert_eq!(caps.prefill_capacity_milli, 500);

        let (url, caps) = parse_worker_entry(
            "http://france:30000@backend=sglang_proxy@routes=chat@prefill_capacity=5.0",
        )
        .unwrap();
        assert_eq!(url, "http://france:30000");
        assert_eq!(caps.backend, WorkerBackend::SglangProxy);
        assert!(caps.routes.chat);
        assert_eq!(caps.prefill_capacity_milli, 5000);
    }

    #[test]
    fn parse_entry_extracts_prefill_profile_suffix() {
        for profile in [
            "baseline",
            "b200",
            "b300",
            "nvr-p4",
            "nvfp4",
            "fp8-mtp-dp1-tp8",
            "fp8-mtp-dp1-tp8-alibaba-b300",
            "fp8-mtp-dp1-tp8-l20d",
            "nvfp4-mtp-dp1-tp4",
            "nvfp4-nomtp-dp1-tp4-b300",
            "hk-l20d-1p1d",
        ] {
            let (_url, caps) =
                parse_worker_entry(&format!("http://worker:30000@prefill_profile={profile}"))
                    .unwrap();
            assert_eq!(caps.prefill_capacity_milli, 1000, "profile={profile}");
        }

        for profile in [
            "mi300x",
            "mi300x-2p2d",
            "mi300x-rdma02-2p2d",
            "mi300x-rdma03-2p2d",
            "mi300x-rdma04-2p2d",
            "mi300x-rdma05-2p2d",
        ] {
            let (_url, caps) = parse_worker_entry(&format!(
                "http://mi300x:30000@backend=sglang_proxy@routes=chat@prefill_profile={profile}"
            ))
            .unwrap();
            assert_eq!(caps.prefill_capacity_milli, 500, "profile={profile}");
            assert_eq!(caps.backend, WorkerBackend::SglangProxy);
            assert!(caps.routes.chat);
        }

        let (_url, caps) =
            parse_worker_entry("http://france:30000@prefill_profile=mi300x-france-20p10d").unwrap();
        assert_eq!(caps.prefill_capacity_milli, 5000);
    }

    #[test]
    fn parse_entry_extracts_vllm_backend_suffix() {
        let (url, caps) = parse_worker_entry("http://h20-r0:8006@backend=vllm").unwrap();
        assert_eq!(url, "http://h20-r0:8006");
        assert_eq!(caps.backend, WorkerBackend::Vllm);
        assert_eq!(caps.min_priority, None);
        assert_eq!(caps.tier, WorkerTier::Default);
    }

    #[test]
    fn parse_entry_extracts_sglang_proxy_backend_suffix() {
        let (url, caps) =
            parse_worker_entry("http://mi300x-1p3d:30000@backend=sglang_proxy").unwrap();
        assert_eq!(url, "http://mi300x-1p3d:30000");
        assert_eq!(caps.backend, WorkerBackend::SglangProxy);
        assert_eq!(caps.min_priority, None);
        assert_eq!(caps.tier, WorkerTier::Default);
    }

    #[test]
    fn parse_entry_extracts_route_suffix() {
        let (url, caps) =
            parse_worker_entry("http://mi300x-1p3d:30000@backend=sglang_proxy@routes=chat")
                .unwrap();
        assert_eq!(url, "http://mi300x-1p3d:30000");
        assert!(caps.routes.chat);
        assert!(!caps.routes.completions);
        assert!(!caps.routes.messages);
        assert!(!caps.routes.responses);

        let (_url, caps) =
            parse_worker_entry("http://b200:30000@routes=messages,responses").unwrap();
        assert!(!caps.routes.chat);
        assert!(!caps.routes.completions);
        assert!(caps.routes.messages);
        assert!(caps.routes.responses);
    }

    #[test]
    fn parse_entry_extracts_tier_suffix() {
        let (url, caps) = parse_worker_entry("http://h20-r0:8006@tier=bulk").unwrap();
        assert_eq!(url, "http://h20-r0:8006");
        assert_eq!(caps.tier, WorkerTier::Bulk);
        assert_eq!(caps.backend, WorkerBackend::Sglang);
        assert_eq!(caps.min_priority, None);
    }

    #[test]
    fn parse_entry_extracts_dedicated_tier_suffix() {
        let (url, caps) =
            parse_worker_entry("https://rdma06-router.example@tier=dedicated").unwrap();
        assert_eq!(url, "https://rdma06-router.example");
        assert_eq!(caps.tier, WorkerTier::Dedicated);
    }

    #[test]
    fn parse_entry_extracts_combined_suffixes_in_either_order() {
        let (url, caps) = parse_worker_entry(
            "http://h20-r0:8006@backend=vllm@tier=bulk@min_priority=100@min_context_tokens=65536@max_context_tokens=500000",
        )
        .unwrap();
        assert_eq!(url, "http://h20-r0:8006");
        assert_eq!(caps.backend, WorkerBackend::Vllm);
        assert_eq!(caps.tier, WorkerTier::Bulk);
        assert_eq!(caps.min_priority, Some(100));
        assert_eq!(caps.min_context_tokens, Some(65_536));
        assert_eq!(caps.max_context_tokens, Some(500_000));

        let (url, caps) = parse_worker_entry(
            "http://h20-r0:8006@max_context_tokens=500000@min_priority=100@tier=bulk@backend=vllm",
        )
        .unwrap();
        assert_eq!(url, "http://h20-r0:8006");
        assert_eq!(caps.backend, WorkerBackend::Vllm);
        assert_eq!(caps.tier, WorkerTier::Bulk);
        assert_eq!(caps.min_priority, Some(100));
        assert_eq!(caps.max_context_tokens, Some(500_000));
    }

    #[test]
    fn parse_entry_rejects_invalid_or_empty_context_range() {
        for value in ["0", "-1", "many", ""] {
            let err = parse_worker_entry(&format!("http://w:30000@min_context_tokens={value}"))
                .unwrap_err()
                .to_string();
            assert!(err.contains("min_context_tokens"), "got: {err}");
        }
        let err =
            parse_worker_entry("http://w:30000@min_context_tokens=65536@max_context_tokens=65535")
                .unwrap_err()
                .to_string();
        assert!(err.contains("context range"), "got: {err}");
    }

    #[test]
    fn parse_entry_extracts_prefill_members_suffix() {
        let (url, caps) = parse_worker_entry(
            "http://pd-proxy:30000@backend=sglang_proxy@routes=chat@prefill_members=http://prefill-0:30000/,http://prefill-1:30000",
        )
        .unwrap();

        assert_eq!(url, "http://pd-proxy:30000");
        assert_eq!(caps.backend, WorkerBackend::SglangProxy);
        assert_eq!(
            caps.prefill_members,
            vec![
                "http://prefill-0:30000".to_string(),
                "http://prefill-1:30000".to_string(),
            ],
        );
    }

    #[test]
    fn build_worker_specs_propagates_prefill_members() {
        let specs = build_worker_specs(&StaticUrlsDiscoveryConfig {
            urls: vec![
                "http://pd-proxy:30000@prefill_members=http://prefill-0:30000,http://prefill-1:30000"
                    .into(),
            ],
            bearer_keys: Vec::new(),
        })
        .unwrap();

        assert_eq!(
            specs[0].prefill_members,
            vec![
                "http://prefill-0:30000".to_string(),
                "http://prefill-1:30000".to_string(),
            ],
        );
    }

    #[test]
    fn parse_entry_strips_tier_suffix() {
        let (url, caps) = parse_worker_entry("http://b200-01:10100@tier=shared").unwrap();
        assert_eq!(url, "http://b200-01:10100");
        assert_eq!(caps.min_priority, None);
        assert_eq!(caps.tier, WorkerTier::Shared);
        assert_eq!(caps.backend, WorkerBackend::Sglang);
    }

    #[test]
    fn parse_entry_strips_tier_after_min_priority() {
        let (url, caps) =
            parse_worker_entry("http://rtx-01:30000@min_priority=100@tier=shared").unwrap();
        assert_eq!(url, "http://rtx-01:30000");
        assert_eq!(caps.min_priority, Some(100));
        assert_eq!(caps.tier, WorkerTier::Shared);
        assert_eq!(caps.backend, WorkerBackend::Sglang);
    }

    #[test]
    fn parse_entry_rejects_empty_tier_suffix() {
        let err = parse_worker_entry("http://w:30000@tier=")
            .unwrap_err()
            .to_string();
        assert!(err.contains("tier"), "got: {err}");
    }

    #[test]
    fn parse_entry_rejects_non_integer_min_priority() {
        let err = parse_worker_entry("http://w:30000@min_priority=high")
            .unwrap_err()
            .to_string();
        assert!(err.contains("min_priority"), "got: {err}");
    }

    #[test]
    fn parse_entry_rejects_invalid_max_context_tokens() {
        for value in ["0", "-1", "many", ""] {
            let err = parse_worker_entry(&format!("http://w:30000@max_context_tokens={value}"))
                .unwrap_err()
                .to_string();
            assert!(err.contains("max_context_tokens"), "got: {err}");
        }
    }

    #[test]
    fn parse_entry_rejects_invalid_prefill_capacity() {
        for value in ["0", "-1", "nan", "inf", "many", ""] {
            let err = parse_worker_entry(&format!("http://w:30000@prefill_capacity={value}"))
                .unwrap_err()
                .to_string();
            assert!(err.contains("prefill_capacity"), "got: {err}");
        }
    }

    #[test]
    fn parse_entry_rejects_invalid_prefill_members() {
        for value in ["", "http://prefill-0:30000,", "not-a-url"] {
            let err = parse_worker_entry(&format!("http://w:30000@prefill_members={value}"))
                .unwrap_err()
                .to_string();
            assert!(err.contains("prefill_members"), "got: {err}");
        }
    }

    #[test]
    fn parse_entry_rejects_invalid_prefill_profile() {
        let err = parse_worker_entry("http://w:30000@prefill_profile=h20-unknown")
            .unwrap_err()
            .to_string();
        assert!(err.contains("prefill_profile"), "got: {err}");
    }

    #[test]
    fn parse_entry_rejects_mixed_prefill_capacity_and_profile() {
        for entry in [
            "http://w:30000@prefill_profile=mi300x@prefill_capacity=0.5",
            "http://w:30000@prefill_capacity=0.5@prefill_profile=mi300x",
        ] {
            let err = parse_worker_entry(entry).unwrap_err().to_string();
            assert!(err.contains("mutually exclusive"), "got: {err}");
        }
    }

    #[test]
    fn parse_entry_rejects_unknown_tier() {
        let err = parse_worker_entry("http://w:30000@tier=gold")
            .unwrap_err()
            .to_string();
        assert!(err.contains("tier"), "got: {err}");
    }

    #[test]
    fn parse_entry_rejects_unknown_backend() {
        let err = parse_worker_entry("http://w:30000@backend=trtllm")
            .unwrap_err()
            .to_string();
        assert!(err.contains("backend"), "got: {err}");
    }

    #[test]
    fn parse_entry_rejects_unknown_route() {
        let err = parse_worker_entry("http://w:30000@routes=chat,images")
            .unwrap_err()
            .to_string();
        assert!(err.contains("routes"), "got: {err}");
    }

    #[test]
    fn parse_entry_userinfo_at_is_not_mistaken_for_token() {
        // A bare `@` (here in a hypothetical userinfo position) must not be
        // treated as the capability token — only `@min_priority=` splits.
        let (url, caps) = parse_worker_entry("http://user@host:30000").unwrap();
        assert_eq!(url, "http://user@host:30000");
        assert_eq!(caps.min_priority, None);
        assert_eq!(caps.backend, WorkerBackend::Sglang);
    }

    /// Task exits cleanly when the consumer drops the receiver mid-fanout.
    /// Without this early exit, the producer would block forever on the
    /// closed channel and shutdown would have to abort it. Kept in-source
    /// (rather than as a component test) because it inspects the
    /// `send().is_err()` branch, which is an implementation detail of
    /// this module — fan-out and event-shape assertions live in
    /// `tests/component/discovery/static_urls.rs`.
    #[tokio::test]
    async fn exits_when_receiver_dropped() {
        let cfg = StaticUrlsDiscoveryConfig {
            urls: (0..10).map(|i| format!("http://w{i}:30000")).collect(),
            bearer_keys: Vec::new(),
        };
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let h = spawn(cfg, tx).await.unwrap();
        // No panic, no hang — task exits on the first send error.
        h.await.unwrap();
    }

    /// After fan-out the task must STAY ALIVE so the critical-task
    /// supervisor (`server::supervisor::supervise_critical_tasks`)
    /// doesn't treat the exit as a failure and flip `/readyz` to 503.
    /// The static_urls backend has no hot-reload, so the only reasons
    /// it should ever exit are (a) the consumer dropped the receiver,
    /// or (b) the supervisor aborted it on shutdown. A "natural" exit
    /// after fan-out used to be the third path, and was wrongly
    /// interpreted as a panic by the supervisor — pinned here so a
    /// regression to "exit after fan-out" can't sneak back in.
    #[tokio::test]
    async fn stays_alive_after_fanout_until_receiver_dropped() {
        use std::time::Duration;

        let cfg = StaticUrlsDiscoveryConfig {
            urls: vec!["http://w0:30000".into(), "http://w1:30000".into()],
            bearer_keys: Vec::new(),
        };
        let (tx, mut rx) = mpsc::channel(8);
        let h = spawn(cfg, tx).await.unwrap();

        // Drain the fan-out so the task is past the for-loop.
        for _ in 0..2 {
            let _ = rx.recv().await.expect("fan-out event");
        }

        // Now give the task a long-by-test-standards moment to exit
        // post-fanout. Pre-fix this would have completed in under a
        // millisecond; post-fix it must time out.
        let mut handle = h;
        let exited = tokio::time::timeout(Duration::from_millis(200), &mut handle).await;
        let still_running = exited.is_err();
        if !still_running {
            panic!(
                "static_urls task exited after fan-out (joined as {exited:?}); \
                 this trips `supervise_critical_tasks` → mark_unready and the pod \
                 becomes /readyz 503. The task must park until the receiver is dropped."
            );
        }
        // Clean shutdown: dropping the receiver closes the channel, which
        // the post-fix task uses as its "time to exit" signal. Pin both
        // halves of the contract — parks while the receiver is alive AND
        // exits cleanly once it's dropped — so a future refactor that
        // parks the task on the wrong signal (e.g., a sleep, a token that
        // never fires) is caught here rather than silently lingering.
        drop(rx);
        let joined = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("task must exit promptly after the receiver is dropped");
        joined.expect("task panicked during clean shutdown");
    }
}
