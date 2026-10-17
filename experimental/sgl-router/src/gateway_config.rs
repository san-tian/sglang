// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Administrator-owned gateway YAML compiled into the existing startup
//! contract. Raw keys remain out of logs and CLI diagnostics.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;

pub const GATEWAY_CONFIG_FILE_ENV: &str = "GATEWAY_CONFIG_FILE";

const MAX_BACKENDS: usize = 1_024;
const MAX_KEYS: usize = 1_024;
const REQUIRED_KEY_IDS: [&str; 4] = [
    "external-all-length-high",
    "external-amd-high",
    "external-nvidia-high",
    "internal-fp8-low",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GatewayFileConfig {
    version: u32,
    revision: String,
    runtime: RuntimeConfig,
    backends: Vec<BackendConfig>,
    scheduling: SchedulingConfig,
    keys: Vec<ClientKeyConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeConfig {
    cache_state: CacheStateRuntimeConfig,
    prefill_work: PrefillWorkRuntimeConfig,
    router_state: RouterStateRuntimeConfig,
    observability: ObservabilityRuntimeConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheStateRuntimeConfig {
    mode: CacheStateMode,
    timeout_ms: u64,
    page_size: usize,
    bigram: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CacheStateMode {
    RemoteOnly,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefillWorkRuntimeConfig {
    stale_grace_ms: u64,
    failure_threshold: usize,
    recovery_threshold: usize,
    profile_source: PrefillProfileSource,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PrefillProfileSource {
    EnvironmentOptional,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterStateRuntimeConfig {
    mode: RouterStateMode,
    key_prefix: String,
    timeout_ms: u64,
    snapshot_interval_ms: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RouterStateMode {
    RedisRequired,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservabilityRuntimeConfig {
    sls: SlsMode,
    route_decision: RouteDecisionRuntimeConfig,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SlsMode {
    Required,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteDecisionRuntimeConfig {
    enabled: bool,
    sample_rate: f64,
    max_candidates: usize,
    force_header: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BackendType {
    Worker,
    PdRouter,
    ExternalApi,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendConfig {
    backend_id: String,
    #[serde(rename = "type")]
    backend_type: BackendType,
    url: String,
    api_key: String,
    hardware: Option<HardwareConfig>,
    model: BackendModelConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HardwareConfig {
    vendor: String,
    product: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendModelConfig {
    id: String,
    quantization: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulingConfig {
    default_policy: String,
    #[serde(default)]
    policies: Vec<RoutingPolicyConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RoutingPolicyConfig {
    policy_id: String,
    #[serde(rename = "type")]
    policy_type: String,
    model_id: String,
    candidate_rules: Vec<CandidateRuleConfig>,
    within_candidates: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateRuleConfig {
    when: CandidateWhenConfig,
    backend_selector: BackendSelectorConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateWhenConfig {
    input_tokens_lt: Option<usize>,
    input_tokens_gte: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendSelectorConfig {
    #[serde(default)]
    hardware_vendor_in: Vec<String>,
    #[serde(default)]
    quantization_in: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClientKeyConfig {
    key_id: String,
    api_key: String,
    priority: i64,
    allowed_models: Vec<String>,
    backend_selector: Option<BackendSelectorConfig>,
    routing_policy_id: Option<String>,
}

#[derive(Serialize)]
struct CompiledPolicyDocument {
    version: u32,
    keys: Vec<CompiledPolicyEntry>,
}

#[derive(Serialize)]
struct CompiledPolicyEntry {
    key_id: String,
    class: &'static str,
    enabled: bool,
    priority: i64,
    allowed_models: Vec<String>,
    allowed_worker_urls: Vec<String>,
    input_length_routing: bool,
}

struct LengthPolicy<'a> {
    config: &'a RoutingPolicyConfig,
    threshold: usize,
    below: &'a BackendSelectorConfig,
    above: &'a BackendSelectorConfig,
}

/// Secret-bearing environment values produced from a validated YAML file.
/// This type intentionally does not implement `Debug`.
pub struct CompiledGatewayConfig {
    env: BTreeMap<&'static str, String>,
    required_environment: Vec<&'static str>,
    pub revision: String,
    pub backend_count: usize,
    pub local_backend_count: usize,
    pub external_backend_count: usize,
    pub key_count: usize,
}

impl CompiledGatewayConfig {
    /// Apply before any worker discovery or authentication task is spawned.
    pub fn apply_to_environment(&self) {
        for name in [
            "GATEWAY_NVIDIA_WORKER_URLS",
            "WORKER_BEARER_KEY",
            "WORKER_BEARER_KEYS",
        ] {
            std::env::remove_var(name);
        }
        for name in [
            "EXTERNAL_MODEL_ID",
            "EXTERNAL_MODEL_URL",
            "EXTERNAL_MODEL_BEARER_TOKEN",
            "ALLOW_RAW_CONTEXT_TOKENS",
            "CACHE_TREE_MAX_NODES",
            "ROUTER_STATE_URL",
        ] {
            std::env::remove_var(name);
        }
        for (name, value) in &self.env {
            std::env::set_var(name, value);
        }
    }

    /// Validate fixed external secret/service contracts without exposing values.
    pub fn validate_environment(&self) -> Result<()> {
        self.validate_environment_with(|name| std::env::var(name).ok())
    }

    fn validate_environment_with<F>(&self, lookup: F) -> Result<()>
    where
        F: Fn(&str) -> Option<String>,
    {
        let missing = self
            .required_environment
            .iter()
            .copied()
            .filter(|name| {
                lookup(name)
                    .as_deref()
                    .map(str::trim)
                    .is_none_or(str::is_empty)
            })
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            bail!(
                "gateway runtime requires non-empty environment variables: {}",
                missing.join(", ")
            );
        }
        Ok(())
    }
}

pub fn load(path: &Path) -> Result<CompiledGatewayConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("read gateway config file {}", path.display()))?;
    compile(&contents).context("validate gateway YAML")
}

fn compile(contents: &str) -> Result<CompiledGatewayConfig> {
    let config: GatewayFileConfig = serde_yaml::from_str(contents).context("parse gateway YAML")?;
    if config.version != 2 {
        bail!("unsupported gateway config version (expected 2)");
    }
    validate_identifier("revision", &config.revision)?;
    if config.backends.is_empty() || config.backends.len() > MAX_BACKENDS {
        bail!("backends must contain between 1 and {MAX_BACKENDS} entries");
    }
    if config.keys.is_empty() || config.keys.len() > MAX_KEYS {
        bail!("keys must contain between 1 and {MAX_KEYS} entries");
    }
    let configured_key_ids = config
        .keys
        .iter()
        .map(|key| key.key_id.as_str())
        .collect::<BTreeSet<_>>();
    if configured_key_ids.len() != REQUIRED_KEY_IDS.len()
        || !REQUIRED_KEY_IDS
            .iter()
            .all(|required| configured_key_ids.contains(required))
    {
        bail!("keys must contain exactly: {}", REQUIRED_KEY_IDS.join(", "));
    }
    if config.scheduling.default_policy != "predicted_ttft" {
        bail!("scheduling.default_policy must be predicted_ttft");
    }

    let mut backend_ids = HashSet::new();
    let mut backend_urls = HashSet::new();
    let mut local = Vec::new();
    let mut external = Vec::new();
    for backend in &config.backends {
        validate_identifier("backend_id", &backend.backend_id)?;
        validate_http_url(&backend.url)?;
        validate_api_key(&backend.api_key)?;
        if !backend_ids.insert(&backend.backend_id) {
            bail!("backend_id values must be unique");
        }
        if !backend_urls.insert(normalize_url(&backend.url)?) {
            bail!("backend URLs must be unique");
        }
        validate_identifier("model.id", &backend.model.id)?;
        match backend.backend_type {
            BackendType::Worker | BackendType::PdRouter => {
                let hardware = backend
                    .hardware
                    .as_ref()
                    .ok_or_else(|| anyhow!("worker and pd_router backends require hardware"))?;
                validate_label("hardware.vendor", &hardware.vendor)?;
                validate_label("hardware.product", &hardware.product)?;
                validate_label(
                    "model.quantization",
                    backend.model.quantization.as_deref().ok_or_else(|| {
                        anyhow!("worker and pd_router backends require model.quantization")
                    })?,
                )?;
                local.push(backend);
            }
            BackendType::ExternalApi => {
                if backend.hardware.is_some() || backend.model.quantization.is_some() {
                    bail!("external_api backends must not declare hardware or quantization");
                }
                external.push(backend);
            }
        }
    }
    if local.is_empty() {
        bail!("at least one worker or pd_router backend is required");
    }
    if external.len() > 1 {
        bail!("the current gateway supports at most one external_api backend");
    }
    let local_models = local
        .iter()
        .map(|backend| backend.model.id.as_str())
        .collect::<BTreeSet<_>>();
    if local_models.len() != 1 {
        bail!("all worker and pd_router backends must serve the same model.id");
    }
    let local_model_id = (*local_models.first().expect("local is non-empty")).to_string();
    let local_api_keys = local
        .iter()
        .map(|backend| backend.api_key.as_str())
        .collect::<HashSet<_>>();
    if local_api_keys.len() != 1 {
        bail!("all local backends must share one api_key for worker introspection");
    }

    let length_policy = compile_length_policy(&config.scheduling, &local_model_id)?;
    let mut worker_urls = Vec::with_capacity(local.len());
    for backend in &local {
        let mut entry = backend.url.clone();
        if backend.backend_type == BackendType::PdRouter {
            entry.push_str("@backend=sglang_proxy");
        }
        if let Some(policy) = &length_policy {
            let below = selector_matches(policy.below, backend);
            let above = selector_matches(policy.above, backend);
            match (below, above) {
                (true, false) => entry.push_str(&format!(
                    "@max_context_tokens={}",
                    policy.threshold - 1
                )),
                (false, true) => {
                    entry.push_str(&format!("@min_context_tokens={}", policy.threshold))
                }
                _ => bail!(
                    "input_length candidate selectors must partition every local backend exactly once"
                ),
            }
        }
        worker_urls.push(entry);
    }

    let configured_models = config
        .backends
        .iter()
        .map(|backend| backend.model.id.as_str())
        .collect::<HashSet<_>>();
    let mut key_ids = HashSet::new();
    let mut raw_keys = HashSet::new();
    let mut api_keys = BTreeMap::new();
    let mut policy_entries = Vec::with_capacity(config.keys.len());
    let policy_ids = config
        .scheduling
        .policies
        .iter()
        .map(|policy| policy.policy_id.as_str())
        .collect::<HashSet<_>>();
    for key in config.keys {
        validate_identifier("key_id", &key.key_id)?;
        validate_api_key(&key.api_key)?;
        if !key_ids.insert(key.key_id.clone()) {
            bail!("key_id values must be unique");
        }
        if !raw_keys.insert(key.api_key.clone()) {
            bail!("client api_key values must be unique");
        }
        if key.allowed_models.is_empty()
            || key
                .allowed_models
                .iter()
                .any(|model| !configured_models.contains(model.as_str()))
        {
            bail!("allowed_models must be non-empty and reference configured models");
        }
        if let Some(policy_id) = &key.routing_policy_id {
            if !policy_ids.contains(policy_id.as_str()) {
                bail!("routing_policy_id must reference a configured scheduling policy");
            }
            let policy = length_policy
                .as_ref()
                .ok_or_else(|| anyhow!("routing_policy_id requires an input_length policy"))?;
            if !key.allowed_models.contains(&policy.config.model_id) {
                bail!("a routing key must allow the policy model_id");
            }
        }
        let selector = key.backend_selector.as_ref().cloned().unwrap_or_default();
        validate_selector(&selector)?;
        let allowed_worker_urls = local
            .iter()
            .filter(|backend| selector_matches(&selector, backend))
            .map(|backend| normalize_url(&backend.url))
            .collect::<Result<Vec<_>>>()?;
        if key.allowed_models.contains(&local_model_id) && allowed_worker_urls.is_empty() {
            bail!("backend_selector must match at least one local backend");
        }
        let input_length_routing = key.routing_policy_id.is_some();
        let nvidia_only = selector.hardware_vendor_in.len() == 1
            && selector.hardware_vendor_in[0].eq_ignore_ascii_case("nvidia")
            && selector.quantization_in.is_empty();
        let class = if input_length_routing {
            "external_length"
        } else if key.priority == 0 {
            "internal"
        } else if nvidia_only {
            "external_nvidia"
        } else {
            "external"
        };
        api_keys.insert(key.key_id.clone(), key.api_key);
        policy_entries.push(CompiledPolicyEntry {
            key_id: key.key_id,
            class,
            enabled: true,
            priority: key.priority,
            allowed_models: key.allowed_models,
            allowed_worker_urls,
            input_length_routing,
        });
    }

    let policy_json = serde_json::to_string(&CompiledPolicyDocument {
        version: 1,
        keys: policy_entries,
    })?;
    let api_keys_json = serde_json::to_string(&api_keys)?;

    let mut env = BTreeMap::new();
    env.insert("GATEWAY_CONFIG_REV", config.revision.clone());
    env.insert("MODEL_ID", local_model_id);
    env.insert("POLICY", "cache_aware_zmq".to_string());
    env.insert("WORKER_URLS", worker_urls.join(" "));
    env.insert("WORKER_BEARER_KEY", local[0].api_key.clone());
    env.insert("WORKER_INTROSPECT_KEY", local[0].api_key.clone());
    env.insert("GATEWAY_API_KEYS_JSON", api_keys_json);
    env.insert("GATEWAY_KEY_POLICIES_JSON", policy_json);
    env.insert("TTFT_FIRST_ROUTING", "1".to_string());
    env.insert("TTFT_SCORE_MODE", "predicted-ttft".to_string());
    env.insert("LOAD_POLL_INTERVAL_SECS", "1".to_string());
    env.insert("CACHE_THRESHOLD", "0".to_string());
    let required_environment = compile_runtime(&config.runtime, &mut env)?;
    if length_policy.is_some() {
        env.insert("ALLOW_RAW_CONTEXT_TOKENS", "1".to_string());
    }
    if let Some(backend) = external.first() {
        env.insert("EXTERNAL_MODEL_ID", backend.model.id.clone());
        env.insert("EXTERNAL_MODEL_URL", backend.url.clone());
        env.insert("EXTERNAL_MODEL_BEARER_TOKEN", backend.api_key.clone());
    }

    Ok(CompiledGatewayConfig {
        revision: config.revision,
        backend_count: config.backends.len(),
        local_backend_count: local.len(),
        external_backend_count: external.len(),
        key_count: api_keys.len(),
        env,
        required_environment,
    })
}

fn compile_runtime(
    runtime: &RuntimeConfig,
    env: &mut BTreeMap<&'static str, String>,
) -> Result<Vec<&'static str>> {
    if runtime.cache_state.mode != CacheStateMode::RemoteOnly {
        bail!("runtime.cache_state.mode must be remote_only");
    }
    if runtime.cache_state.timeout_ms == 0 || runtime.cache_state.page_size == 0 {
        bail!("cache-state timeout_ms and page_size must be positive");
    }
    if runtime.prefill_work.stale_grace_ms == 0
        || runtime.prefill_work.failure_threshold == 0
        || runtime.prefill_work.recovery_threshold == 0
    {
        bail!("Prefill Work stale grace and thresholds must be positive");
    }
    if runtime.prefill_work.profile_source != PrefillProfileSource::EnvironmentOptional {
        bail!("runtime.prefill_work.profile_source must be environment_optional");
    }
    if runtime.router_state.mode != RouterStateMode::RedisRequired {
        bail!("runtime.router_state.mode must be redis_required");
    }
    validate_redis_key_prefix(&runtime.router_state.key_prefix)?;
    if runtime.router_state.timeout_ms == 0 || runtime.router_state.snapshot_interval_ms == 0 {
        bail!("router-state timeout_ms and snapshot_interval_ms must be positive");
    }
    if runtime.observability.sls != SlsMode::Required {
        bail!("runtime.observability.sls must be required");
    }
    let decision = &runtime.observability.route_decision;
    if !decision.enabled {
        bail!("runtime.observability.route_decision.enabled must be true");
    }
    if !decision.sample_rate.is_finite()
        || !(0.0..=1.0).contains(&decision.sample_rate)
        || decision.sample_rate == 0.0
    {
        bail!("route-decision sample_rate must be finite and in (0, 1]");
    }
    if decision.max_candidates == 0 || decision.max_candidates > MAX_BACKENDS {
        bail!("route-decision max_candidates must be between 1 and {MAX_BACKENDS}");
    }
    validate_header_name(&decision.force_header)?;

    env.insert("CACHE_TREE_SOURCE", "remote".to_string());
    env.insert(
        "CACHE_TREE_PAGE_SIZE",
        runtime.cache_state.page_size.to_string(),
    );
    env.insert(
        "CACHE_TREE_BIGRAM",
        if runtime.cache_state.bigram { "1" } else { "0" }.to_string(),
    );
    env.insert(
        "CACHE_STATE_TIMEOUT_MS",
        runtime.cache_state.timeout_ms.to_string(),
    );
    env.insert(
        "PREFILL_SCORE_STALE_GRACE_MS",
        runtime.prefill_work.stale_grace_ms.to_string(),
    );
    env.insert(
        "PREFILL_LOAD_FAILURE_THRESHOLD",
        runtime.prefill_work.failure_threshold.to_string(),
    );
    env.insert(
        "PREFILL_LOAD_RECOVERY_THRESHOLD",
        runtime.prefill_work.recovery_threshold.to_string(),
    );
    env.insert(
        "ROUTER_STATE_REDIS_KEY_PREFIX",
        runtime.router_state.key_prefix.clone(),
    );
    env.insert(
        "ROUTER_STATE_TIMEOUT_MS",
        runtime.router_state.timeout_ms.to_string(),
    );
    env.insert(
        "ROUTER_STATE_SNAPSHOT_INTERVAL_MS",
        runtime.router_state.snapshot_interval_ms.to_string(),
    );
    env.insert(
        "ROUTE_DECISION_LOG_SAMPLE_RATE",
        runtime.observability.route_decision.sample_rate.to_string(),
    );
    env.insert(
        "ROUTE_DECISION_LOG_MAX_CANDIDATES",
        runtime
            .observability
            .route_decision
            .max_candidates
            .to_string(),
    );
    env.insert(
        "ROUTE_DECISION_LOG_HEADER",
        runtime.observability.route_decision.force_header.clone(),
    );

    Ok(vec![
        "CACHE_STATE_URL",
        "CACHE_STATE_API_TOKEN",
        "ROUTER_STATE_REDIS_URL",
        "SLS_ENDPOINT",
        "SLS_ACCESS_KEY_ID",
        "SLS_ACCESS_KEY_SECRET",
    ])
}

fn compile_length_policy<'a>(
    scheduling: &'a SchedulingConfig,
    local_model_id: &str,
) -> Result<Option<LengthPolicy<'a>>> {
    if scheduling.policies.len() > 1 {
        bail!("the current gateway supports at most one scheduling policy");
    }
    let Some(policy) = scheduling.policies.first() else {
        return Ok(None);
    };
    validate_identifier("policy_id", &policy.policy_id)?;
    if policy.policy_type != "input_length"
        || policy.within_candidates != "predicted_ttft"
        || policy.model_id != local_model_id
    {
        bail!(
            "scheduling policy must be input_length for the local model with predicted_ttft candidates"
        );
    }
    if policy.candidate_rules.len() != 2 {
        bail!("input_length policy requires exactly two candidate_rules");
    }
    let mut below = None;
    let mut above = None;
    for rule in &policy.candidate_rules {
        validate_selector(&rule.backend_selector)?;
        match (rule.when.input_tokens_lt, rule.when.input_tokens_gte) {
            (Some(threshold), None) => below = Some((threshold, &rule.backend_selector)),
            (None, Some(threshold)) => above = Some((threshold, &rule.backend_selector)),
            _ => bail!("each candidate rule must set exactly one input token boundary"),
        }
    }
    let (below_threshold, below) = below.ok_or_else(|| anyhow!("missing input_tokens_lt rule"))?;
    let (above_threshold, above) = above.ok_or_else(|| anyhow!("missing input_tokens_gte rule"))?;
    if below_threshold == 0 || below_threshold != above_threshold {
        bail!("input token rules must share one positive split threshold");
    }
    Ok(Some(LengthPolicy {
        config: policy,
        threshold: below_threshold,
        below,
        above,
    }))
}

fn selector_matches(selector: &BackendSelectorConfig, backend: &BackendConfig) -> bool {
    let Some(hardware) = backend.hardware.as_ref() else {
        return false;
    };
    let vendor_matches = selector.hardware_vendor_in.is_empty()
        || selector
            .hardware_vendor_in
            .iter()
            .any(|vendor| vendor.eq_ignore_ascii_case(&hardware.vendor));
    let quantization_matches = selector.quantization_in.is_empty()
        || backend
            .model
            .quantization
            .as_ref()
            .is_some_and(|quantization| {
                selector
                    .quantization_in
                    .iter()
                    .any(|value| value.eq_ignore_ascii_case(quantization))
            });
    vendor_matches && quantization_matches
}

fn validate_selector(selector: &BackendSelectorConfig) -> Result<()> {
    for vendor in &selector.hardware_vendor_in {
        validate_label("hardware_vendor_in", vendor)?;
    }
    for quantization in &selector.quantization_in {
        validate_label("quantization_in", quantization)?;
    }
    Ok(())
}

fn validate_identifier(field: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/'))
    {
        bail!("{field} must be a non-empty safe identifier");
    }
    Ok(())
}

fn validate_label(field: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        bail!("{field} must be a non-empty safe label");
    }
    Ok(())
}

fn validate_redis_key_prefix(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 256
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        })
    {
        bail!("router-state key_prefix must be a non-empty safe Redis key prefix");
    }
    Ok(())
}

fn validate_header_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-'))
    {
        bail!("route-decision force_header must be a lowercase HTTP header name");
    }
    Ok(())
}

fn validate_api_key(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 4_096
        || !value.bytes().all(|byte| matches!(byte, 0x21..=0x7e))
    {
        bail!("api_key values must be non-empty unique visible ASCII strings");
    }
    Ok(())
}

fn validate_http_url(value: &str) -> Result<()> {
    let parsed = url::Url::parse(value).map_err(|_| anyhow!("backend URL is invalid"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        bail!("backend URL must be a plain http(s) URL without credentials, query, or fragment");
    }
    Ok(())
}

fn normalize_url(value: &str) -> Result<String> {
    validate_http_url(value)?;
    Ok(value.trim_end_matches('/').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
version: 2
revision: test-revision
runtime:
  cache_state:
    mode: remote_only
    timeout_ms: 500
    page_size: 64
    bigram: true
  prefill_work:
    stale_grace_ms: 10000
    failure_threshold: 3
    recovery_threshold: 2
    profile_source: environment_optional
  router_state:
    mode: redis_required
    key_prefix: test:router-state
    timeout_ms: 5000
    snapshot_interval_ms: 100
  observability:
    sls: required
    route_decision:
      enabled: true
      sample_rate: 1.0
      max_candidates: 64
      force_header: x-sgl-route-decision-log
backends:
  - backend_id: gpu-fp8
    type: worker
    url: http://nvidia.example:30000
    api_key: worker-secret
    hardware: { vendor: nvidia, product: b200 }
    model: { id: served-model, quantization: fp8 }
  - backend_id: gpu-nvfp4
    type: worker
    url: http://nvfp4.example:30000
    api_key: worker-secret
    hardware: { vendor: nvidia, product: b200 }
    model: { id: served-model, quantization: nvfp4 }
  - backend_id: amd-pd
    type: pd_router
    url: https://amd.example
    api_key: worker-secret
    hardware: { vendor: amd, product: mi300x }
    model: { id: served-model, quantization: fp8 }
  - backend_id: external
    type: external_api
    url: https://provider.example
    api_key: provider-secret
    model: { id: external-model }
scheduling:
  default_policy: predicted_ttft
  policies:
    - policy_id: split
      type: input_length
      model_id: served-model
      candidate_rules:
        - when: { input_tokens_lt: 65536 }
          backend_selector: { hardware_vendor_in: [amd] }
        - when: { input_tokens_gte: 65536 }
          backend_selector: { hardware_vendor_in: [nvidia] }
      within_candidates: predicted_ttft
keys:
  - key_id: internal-fp8-low
    api_key: client-internal
    priority: 0
    allowed_models: [served-model]
    backend_selector:
      hardware_vendor_in: [amd, nvidia]
      quantization_in: [fp8]
  - key_id: external-amd-high
    api_key: client-amd
    priority: 100
    allowed_models: [served-model]
    backend_selector: { hardware_vendor_in: [amd] }
  - key_id: external-all-length-high
    api_key: client-length
    priority: 100
    allowed_models: [served-model, external-model]
    routing_policy_id: split
  - key_id: external-nvidia-high
    api_key: client-nvidia
    priority: 100
    allowed_models: [served-model]
    backend_selector: { hardware_vendor_in: [nvidia] }
"#;

    #[test]
    fn compiles_admin_yaml_into_runtime_contract() {
        let compiled = compile(CONFIG).unwrap();
        assert_eq!(compiled.backend_count, 4);
        assert_eq!(compiled.local_backend_count, 3);
        assert_eq!(compiled.external_backend_count, 1);
        assert_eq!(compiled.key_count, 4);
        assert_eq!(compiled.env["MODEL_ID"], "served-model");
        assert_eq!(compiled.env["WORKER_BEARER_KEY"], "worker-secret");
        assert!(!compiled.env.contains_key("WORKER_BEARER_KEYS"));
        assert!(compiled.env["WORKER_URLS"]
            .contains("https://amd.example@backend=sglang_proxy@max_context_tokens=65535"));
        assert!(compiled.env["WORKER_URLS"]
            .contains("http://nvidia.example:30000@min_context_tokens=65536"));
        assert_eq!(compiled.env["EXTERNAL_MODEL_ID"], "external-model");
        assert_eq!(compiled.env["CACHE_TREE_SOURCE"], "remote");
        assert_eq!(compiled.env["CACHE_TREE_PAGE_SIZE"], "64");
        assert_eq!(compiled.env["PREFILL_SCORE_STALE_GRACE_MS"], "10000");
        assert_eq!(compiled.env["PREFILL_LOAD_FAILURE_THRESHOLD"], "3");
        assert_eq!(compiled.env["PREFILL_LOAD_RECOVERY_THRESHOLD"], "2");
        assert_eq!(
            compiled.env["ROUTER_STATE_REDIS_KEY_PREFIX"],
            "test:router-state"
        );
        assert_eq!(compiled.env["ROUTE_DECISION_LOG_SAMPLE_RATE"], "1");
        assert!(!compiled.env.contains_key("PREFILL_SCORE_PROFILES_JSON"));

        let policies: serde_json::Value =
            serde_json::from_str(&compiled.env["GATEWAY_KEY_POLICIES_JSON"]).unwrap();
        let internal = policies["keys"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["key_id"] == "internal-fp8-low")
            .unwrap();
        assert_eq!(internal["priority"], 0);
        assert_eq!(internal["allowed_worker_urls"].as_array().unwrap().len(), 2);
        let length = policies["keys"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["key_id"] == "external-all-length-high")
            .unwrap();
        assert_eq!(length["input_length_routing"], true);
        assert_eq!(length["allowed_models"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn rejects_selector_that_matches_no_local_backend() {
        let invalid = CONFIG.replace("hardware_vendor_in: [amd]", "hardware_vendor_in: [h20]");
        let error = compile(&invalid)
            .err()
            .expect("invalid selector should fail")
            .to_string();
        assert!(error.contains("partition every local backend"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let invalid = CONFIG.replace(
            "revision: test-revision",
            "revision: test-revision\napi: {}",
        );
        assert!(compile(&invalid).is_err());
    }

    #[test]
    fn rejects_unknown_runtime_fields() {
        let invalid = CONFIG.replace(
            "    timeout_ms: 500\n    page_size: 64",
            "    timeout_ms: 500\n    local_fallback: true\n    page_size: 64",
        );
        assert!(compile(&invalid).is_err());
    }

    #[test]
    fn requires_exact_four_key_inventory() {
        let invalid = CONFIG.replace("external-amd-high", "legacy-amd-high");
        let error = compile(&invalid)
            .err()
            .expect("legacy key inventory should fail")
            .to_string();
        assert!(error.contains("keys must contain exactly"));
        assert!(!error.contains("client-amd"));
    }

    #[test]
    fn environment_validation_reports_names_without_values() {
        let compiled = compile(CONFIG).unwrap();
        let error = compiled
            .validate_environment_with(|name| match name {
                "CACHE_STATE_URL" => Some("https://cache.example".to_string()),
                "CACHE_STATE_API_TOKEN" => Some("cache-secret".to_string()),
                "ROUTER_STATE_REDIS_URL" => Some("redis://redis-secret".to_string()),
                "SLS_ENDPOINT" => Some("example.log.aliyuncs.com".to_string()),
                "SLS_ACCESS_KEY_ID" => Some("sls-id-secret".to_string()),
                _ => None,
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("SLS_ACCESS_KEY_SECRET"));
        for secret in ["cache-secret", "redis-secret", "sls-id-secret"] {
            assert!(!error.contains(secret));
        }
    }

    #[test]
    fn complete_environment_contract_validates() {
        let compiled = compile(CONFIG).unwrap();
        compiled
            .validate_environment_with(|_| Some("configured".to_string()))
            .unwrap();
    }
}
