// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Gateway-entry API-key authentication.
//!
//! Policy metadata is non-sensitive and belongs in Git. Raw API keys are
//! supplied separately by the secret store. The two inventories must contain
//! exactly the same key IDs; startup fails instead of silently accepting a
//! partial configuration.

use crate::discovery::static_urls::normalize_worker_url;
use crate::server::error::X_ROUTER_ERROR_CODE;
use crate::workers::Worker;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::de::{Error as _, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

pub const GATEWAY_KEY_POLICIES_ENV: &str = "GATEWAY_KEY_POLICIES_JSON";
pub const GATEWAY_API_KEYS_ENV: &str = "GATEWAY_API_KEYS_JSON";
pub const GATEWAY_NVIDIA_WORKER_URLS_ENV: &str = "GATEWAY_NVIDIA_WORKER_URLS";
pub const PD_PROXY_API_KEY_ENV: &str = "PD_PROXY_API_KEY";

const MAX_KEY_COUNT: usize = 1_024;
const MAX_KEY_ID_BYTES: usize = 64;
const MAX_API_KEY_BYTES: usize = 4_096;
const MAX_CREDENTIAL_HEADERS: usize = 8;
const X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
const OCP_APIM_SUBSCRIPTION_KEY: HeaderName = HeaderName::from_static("ocp-apim-subscription-key");

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GatewayKeyClass {
    External,
    ExternalNvidia,
    Dedicated,
    Internal,
    /// Internal upstream gateway identity. This class cannot be selected by
    /// the Git-managed gateway policy document; it is built only by the
    /// dedicated PD-proxy key loader.
    #[serde(skip)]
    Proxy,
}

impl GatewayKeyClass {
    pub const fn priority_override(self) -> Option<i64> {
        match self {
            Self::External | Self::ExternalNvidia | Self::Dedicated => Some(100),
            Self::Internal => Some(0),
            Self::Proxy => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::External => "external",
            Self::ExternalNvidia => "external_nvidia",
            Self::Dedicated => "dedicated",
            Self::Internal => "internal",
            Self::Proxy => "proxy",
        }
    }
}

impl fmt::Display for GatewayKeyClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Safe request identity. Raw credentials never enter request extensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatewayKeyIdentity {
    key_id: Arc<str>,
    class: GatewayKeyClass,
    allowed_worker_urls: Option<Arc<HashSet<String>>>,
}

impl GatewayKeyIdentity {
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub const fn class(&self) -> GatewayKeyClass {
        self.class
    }

    pub const fn priority_override(&self) -> Option<i64> {
        self.class.priority_override()
    }

    pub const fn is_nvidia_only(&self) -> bool {
        matches!(self.class, GatewayKeyClass::ExternalNvidia)
    }

    pub const fn is_dedicated(&self) -> bool {
        matches!(self.class, GatewayKeyClass::Dedicated)
    }

    pub fn allows_worker_url(&self, worker_url: &str) -> bool {
        if !self.is_nvidia_only() {
            return true;
        }
        match self.allowed_worker_urls.as_ref() {
            Some(urls) => {
                normalize_worker_url(worker_url).is_ok_and(|normalized| urls.contains(&normalized))
            }
            None => false,
        }
    }

    pub const fn allows_external_model(&self) -> bool {
        !self.is_nvidia_only() && !self.is_dedicated()
    }

    pub(crate) fn new(key_id: impl Into<Arc<str>>, class: GatewayKeyClass) -> Self {
        Self {
            key_id: key_id.into(),
            class,
            allowed_worker_urls: None,
        }
    }

    fn with_allowed_worker_urls(
        key_id: impl Into<Arc<str>>,
        class: GatewayKeyClass,
        allowed_worker_urls: Option<Arc<HashSet<String>>>,
    ) -> Self {
        Self {
            key_id: key_id.into(),
            class,
            allowed_worker_urls,
        }
    }
}

#[derive(Debug)]
pub struct KeyScopeCandidates {
    pub workers: Vec<Arc<Worker>>,
    pub excluded_all: bool,
}

pub fn filter_key_scope(
    workers: &[Arc<Worker>],
    identity: Option<&GatewayKeyIdentity>,
) -> KeyScopeCandidates {
    let Some(identity) = identity.filter(|identity| identity.is_nvidia_only()) else {
        return KeyScopeCandidates {
            workers: workers.to_vec(),
            excluded_all: false,
        };
    };
    let scoped: Vec<_> = workers
        .iter()
        .filter(|worker| identity.allows_worker_url(&worker.url))
        .cloned()
        .collect();
    let excluded_all = scoped.is_empty() && !workers.is_empty();
    KeyScopeCandidates {
        excluded_all,
        workers: scoped,
    }
}

struct KeyEntry {
    digest: [u8; 32],
    identity: GatewayKeyIdentity,
    enabled: bool,
}

/// Runtime keyring. It retains only fixed-size SHA-256 digests and safe policy
/// metadata, never raw API keys.
pub struct GatewayKeyring {
    entries: Vec<KeyEntry>,
}

impl fmt::Debug for GatewayKeyring {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayKeyring")
            .field("configured_keys", &self.configured_key_count())
            .field("enabled_keys", &self.enabled_key_count())
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum GatewayKeyringError {
    #[error("{GATEWAY_KEY_POLICIES_ENV} and {GATEWAY_API_KEYS_ENV} are both required")]
    MissingEnvironment,

    #[error("{PD_PROXY_API_KEY_ENV} is required in pd_proxy mode")]
    MissingPdProxyEnvironment,

    #[error("{0} must contain valid UTF-8 JSON")]
    InvalidEnvironmentEncoding(&'static str),

    #[error("invalid {GATEWAY_KEY_POLICIES_ENV}: {0}")]
    InvalidPolicies(String),

    #[error("invalid {GATEWAY_API_KEYS_ENV} JSON")]
    InvalidSecrets,

    #[error("gateway key policy and secret ID inventories must match exactly")]
    InventoryMismatch,

    #[error("gateway API keys must be non-empty, unique, visible ASCII strings")]
    InvalidApiKeys,

    #[error("invalid {GATEWAY_NVIDIA_WORKER_URLS_ENV}: {0}")]
    InvalidNvidiaWorkerUrls(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyDocument {
    version: u32,
    keys: Vec<PolicyEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyEntry {
    key_id: String,
    class: GatewayKeyClass,
    enabled: bool,
}

/// serde_json maps otherwise accept duplicate object keys with last-write-wins
/// semantics. Secret inventory must reject duplicates instead.
struct UniqueStringMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueStringMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueStringMapVisitor;

        impl<'de> Visitor<'de> for UniqueStringMapVisitor {
            type Value = UniqueStringMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object mapping unique key IDs to API keys")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key_id, api_key)) = map.next_entry::<String, String>()? {
                    if values.insert(key_id, api_key).is_some() {
                        return Err(M::Error::custom("duplicate gateway key ID"));
                    }
                }
                Ok(UniqueStringMap(values))
            }
        }

        deserializer.deserialize_map(UniqueStringMapVisitor)
    }
}

impl GatewayKeyring {
    /// Load the required production contract directly from secret/config env.
    /// Raw keys are deliberately not translated into CLI arguments.
    pub fn from_env() -> Result<Self, GatewayKeyringError> {
        let policies = read_required_env(GATEWAY_KEY_POLICIES_ENV)?;
        let api_keys = read_required_env(GATEWAY_API_KEYS_ENV)?;
        let nvidia_worker_urls = std::env::var(GATEWAY_NVIDIA_WORKER_URLS_ENV).ok();
        Self::from_json_with_nvidia_workers(&policies, &api_keys, nvidia_worker_urls.as_deref())
    }

    /// Load the single upstream credential used by a dedicated PD proxy.
    /// The raw key is reduced to a digest immediately and never enters CLI
    /// arguments, logs, or request extensions.
    pub fn from_pd_proxy_env() -> Result<Self, GatewayKeyringError> {
        let api_key = match std::env::var(PD_PROXY_API_KEY_ENV) {
            Ok(value) if !value.is_empty() => value,
            Ok(_) | Err(std::env::VarError::NotPresent) => {
                return Err(GatewayKeyringError::MissingPdProxyEnvironment)
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(GatewayKeyringError::InvalidEnvironmentEncoding(
                    PD_PROXY_API_KEY_ENV,
                ))
            }
        };
        Self::from_pd_proxy_key(&api_key)
    }

    fn from_pd_proxy_key(api_key: &str) -> Result<Self, GatewayKeyringError> {
        if !valid_api_key(api_key) {
            return Err(GatewayKeyringError::InvalidApiKeys);
        }
        Ok(Self {
            entries: vec![KeyEntry {
                digest: digest(api_key.as_bytes()),
                identity: GatewayKeyIdentity::new("pd-proxy-upstream", GatewayKeyClass::Proxy),
                enabled: true,
            }],
        })
    }

    pub fn from_json(
        policies_json: &str,
        api_keys_json: &str,
    ) -> Result<Self, GatewayKeyringError> {
        Self::from_json_with_nvidia_workers(policies_json, api_keys_json, None)
    }

    pub fn from_json_with_nvidia_workers(
        policies_json: &str,
        api_keys_json: &str,
        nvidia_worker_urls: Option<&str>,
    ) -> Result<Self, GatewayKeyringError> {
        let document: PolicyDocument = serde_json::from_str(policies_json)
            .map_err(|error| GatewayKeyringError::InvalidPolicies(error.to_string()))?;
        if document.version != 1 {
            return Err(GatewayKeyringError::InvalidPolicies(
                "unsupported schema version (expected 1)".to_string(),
            ));
        }
        if document.keys.is_empty() || document.keys.len() > MAX_KEY_COUNT {
            return Err(GatewayKeyringError::InvalidPolicies(format!(
                "keys must contain between 1 and {MAX_KEY_COUNT} entries"
            )));
        }

        let mut policies = BTreeMap::new();
        for policy in document.keys {
            if !valid_key_id(&policy.key_id) {
                return Err(GatewayKeyringError::InvalidPolicies(
                    "key_id must be 1-64 ASCII letters, digits, '.', '_' or '-'".to_string(),
                ));
            }
            let key_id = policy.key_id.clone();
            if policies.insert(key_id, policy).is_some() {
                return Err(GatewayKeyringError::InvalidPolicies(
                    "duplicate key_id".to_string(),
                ));
            }
        }

        // Do not propagate serde's secret-side error text: it may describe an
        // invalid value. Line/column detail is less important than ensuring a
        // raw secret can never reach startup logs.
        let UniqueStringMap(mut api_keys) =
            serde_json::from_str(api_keys_json).map_err(|_| GatewayKeyringError::InvalidSecrets)?;
        if policies.len() != api_keys.len() || !policies.keys().eq(api_keys.keys()) {
            return Err(GatewayKeyringError::InventoryMismatch);
        }

        let requires_nvidia_workers = policies
            .values()
            .any(|policy| policy.class == GatewayKeyClass::ExternalNvidia);
        let nvidia_worker_urls = if requires_nvidia_workers {
            let raw = nvidia_worker_urls.ok_or_else(|| {
                GatewayKeyringError::InvalidNvidiaWorkerUrls("value is required".to_string())
            })?;
            let mut urls = HashSet::new();
            for worker_url in raw.split_ascii_whitespace() {
                let normalized = normalize_worker_url(worker_url).map_err(|e| {
                    GatewayKeyringError::InvalidNvidiaWorkerUrls(format!(
                        "worker URL is invalid: {e}"
                    ))
                })?;
                if !urls.insert(normalized) {
                    return Err(GatewayKeyringError::InvalidNvidiaWorkerUrls(
                        "worker URLs must be unique".to_string(),
                    ));
                }
            }
            if urls.is_empty() {
                return Err(GatewayKeyringError::InvalidNvidiaWorkerUrls(
                    "at least one worker URL is required".to_string(),
                ));
            }
            Some(Arc::new(urls))
        } else {
            None
        };

        let mut seen_digests = HashSet::with_capacity(policies.len());
        let mut entries = Vec::with_capacity(policies.len());
        for (key_id, policy) in policies {
            let api_key = api_keys
                .remove(&key_id)
                .expect("policy and secret inventories were compared above");
            if !valid_api_key(&api_key) {
                return Err(GatewayKeyringError::InvalidApiKeys);
            }
            let digest = digest(api_key.as_bytes());
            if !seen_digests.insert(digest) {
                return Err(GatewayKeyringError::InvalidApiKeys);
            }
            entries.push(KeyEntry {
                digest,
                identity: GatewayKeyIdentity::with_allowed_worker_urls(
                    Arc::<str>::from(key_id),
                    policy.class,
                    (policy.class == GatewayKeyClass::ExternalNvidia)
                        .then(|| Arc::clone(nvidia_worker_urls.as_ref().expect("validated above"))),
                ),
                enabled: policy.enabled,
            });
        }

        Ok(Self { entries })
    }

    pub fn configured_key_count(&self) -> usize {
        self.entries.len()
    }

    pub fn enabled_key_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.enabled).count()
    }

    pub fn disabled_key_count(&self) -> usize {
        self.configured_key_count() - self.enabled_key_count()
    }

    fn authentication_required(&self) -> bool {
        !self.entries.is_empty()
    }

    pub(crate) fn disabled() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn authenticate(&self, api_key: &str) -> Option<GatewayKeyIdentity> {
        let candidate = digest(api_key.as_bytes());
        let mut matched = None;

        // Scan the full keyring and compare fixed-size digests in constant
        // time. Duplicate API keys are rejected at startup, so at most one
        // entry can match.
        for entry in &self.entries {
            if constant_time_eq(&candidate, &entry.digest) {
                matched = Some(entry);
            }
        }

        matched
            .filter(|entry| entry.enabled)
            .map(|entry| entry.identity.clone())
    }
}

fn read_required_env(name: &'static str) -> Result<String, GatewayKeyringError> {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => Ok(value),
        Ok(_) | Err(std::env::VarError::NotPresent) => Err(GatewayKeyringError::MissingEnvironment),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(GatewayKeyringError::InvalidEnvironmentEncoding(name))
        }
    }
}

fn valid_key_id(key_id: &str) -> bool {
    !key_id.is_empty()
        && key_id.len() <= MAX_KEY_ID_BYTES
        && key_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn valid_api_key(api_key: &str) -> bool {
    !api_key.is_empty()
        && api_key.len() <= MAX_API_KEY_BYTES
        && api_key.bytes().all(|byte| matches!(byte, 0x21..=0x7e))
}

fn digest(value: &[u8]) -> [u8; 32] {
    Sha256::digest(value).into()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

fn credential_from_headers(headers: &HeaderMap) -> Result<String, ()> {
    let mut credentials = Vec::new();

    for value in headers.get_all(header::AUTHORIZATION).iter() {
        let value = value.to_str().map_err(|_| ())?;
        let mut parts = value.split_ascii_whitespace();
        let scheme = parts.next().ok_or(())?;
        let api_key = parts.next().ok_or(())?;
        if !scheme.eq_ignore_ascii_case("bearer") || parts.next().is_some() {
            return Err(());
        }
        push_credential(&mut credentials, api_key)?;
    }

    for name in [&X_API_KEY, &OCP_APIM_SUBSCRIPTION_KEY] {
        for value in headers.get_all(name).iter() {
            let api_key = value.to_str().map_err(|_| ())?.trim();
            push_credential(&mut credentials, api_key)?;
        }
    }

    let first = credentials.first().ok_or(())?;
    let first_digest = digest(first.as_bytes());
    if credentials
        .iter()
        .skip(1)
        .map(|value| digest(value.as_bytes()))
        .any(|candidate| !constant_time_eq(&first_digest, &candidate))
    {
        return Err(());
    }
    Ok(first.clone())
}

fn push_credential(credentials: &mut Vec<String>, api_key: &str) -> Result<(), ()> {
    if credentials.len() >= MAX_CREDENTIAL_HEADERS || !valid_api_key(api_key) {
        return Err(());
    }
    credentials.push(api_key.to_string());
    Ok(())
}

pub async fn authenticate_gateway_key(
    State(keyring): State<Arc<GatewayKeyring>>,
    mut request: Request,
    next: Next,
) -> Response {
    // `build_router` keeps the historical unauthenticated library/test
    // contract. The production binary exclusively uses the required-env
    // keyring path, whose parser rejects an empty inventory.
    if !keyring.authentication_required() {
        return next.run(request).await;
    }

    let api_key = match credential_from_headers(request.headers()) {
        Ok(api_key) => api_key,
        Err(()) => return unauthorized_response(),
    };
    let Some(identity) = keyring.authenticate(&api_key) else {
        return unauthorized_response();
    };

    // Entry credentials authorize the caller only. Worker credentials are a
    // separate static-discovery concern and are injected later by
    // `Worker::headers_for`; never forward a client credential upstream.
    request.headers_mut().remove(header::AUTHORIZATION);
    request.headers_mut().remove(&X_API_KEY);
    request.headers_mut().remove(&OCP_APIM_SUBSCRIPTION_KEY);
    request.extensions_mut().insert(identity);
    next.run(request).await
}

#[derive(Serialize)]
struct AuthenticationErrorEnvelope {
    error: AuthenticationErrorBody,
}

#[derive(Serialize)]
struct AuthenticationErrorBody {
    #[serde(rename = "type")]
    typ: &'static str,
    code: &'static str,
    message: &'static str,
}

fn unauthorized_response() -> Response {
    const CODE: &str = "invalid_api_key";
    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(AuthenticationErrorEnvelope {
            error: AuthenticationErrorBody {
                typ: "authentication_error",
                code: CODE,
                message: "invalid API key",
            },
        }),
    )
        .into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer realm=\"gateway\""),
    );
    response
        .headers_mut()
        .insert(X_ROUTER_ERROR_CODE, HeaderValue::from_static(CODE));
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICIES: &str = r#"{
        "version": 1,
        "keys": [
            {"key_id":"external-a","class":"external","enabled":true},
            {"key_id":"internal-a","class":"internal","enabled":true},
            {"key_id":"disabled-a","class":"external","enabled":false}
        ]
    }"#;
    const SECRETS: &str = r#"{"external-a":"external-secret","internal-a":"internal-secret","disabled-a":"disabled-secret"}"#;

    #[test]
    fn parses_and_matches_enabled_keys_without_retaining_raw_values() {
        let keyring = GatewayKeyring::from_json(POLICIES, SECRETS).unwrap();
        let external = keyring.authenticate("external-secret").unwrap();
        assert_eq!(external.key_id(), "external-a");
        assert_eq!(external.class(), GatewayKeyClass::External);
        assert_eq!(external.priority_override(), Some(100));

        let internal = keyring.authenticate("internal-secret").unwrap();
        assert_eq!(internal.class(), GatewayKeyClass::Internal);
        assert_eq!(internal.priority_override(), Some(0));
        assert!(keyring.authenticate("disabled-secret").is_none());
        assert!(keyring.authenticate("unknown-secret").is_none());

        let debug = format!("{keyring:?}");
        assert!(!debug.contains("external-secret"));
        assert!(!debug.contains("internal-secret"));
        assert_eq!(keyring.configured_key_count(), 3);
        assert_eq!(keyring.enabled_key_count(), 2);
        assert_eq!(keyring.disabled_key_count(), 1);
    }

    #[test]
    fn rejects_policy_and_secret_inventory_mismatch() {
        let error = GatewayKeyring::from_json(POLICIES, r#"{"external-a":"secret"}"#).unwrap_err();
        assert!(matches!(error, GatewayKeyringError::InventoryMismatch));
    }

    #[test]
    fn rejects_duplicate_policy_ids() {
        let policies = r#"{"version":1,"keys":[
            {"key_id":"same","class":"external","enabled":true},
            {"key_id":"same","class":"internal","enabled":true}
        ]}"#;
        let error = GatewayKeyring::from_json(policies, r#"{"same":"secret"}"#).unwrap_err();
        assert!(matches!(error, GatewayKeyringError::InvalidPolicies(_)));
    }

    #[test]
    fn rejects_duplicate_secret_ids_in_json() {
        let policies = r#"{"version":1,"keys":[
            {"key_id":"same","class":"external","enabled":true}
        ]}"#;
        let error = GatewayKeyring::from_json(
            policies,
            r#"{"same":"first-secret","same":"second-secret"}"#,
        )
        .unwrap_err();
        assert!(matches!(error, GatewayKeyringError::InvalidSecrets));
    }

    #[test]
    fn rejects_duplicate_or_empty_api_keys() {
        let policies = r#"{"version":1,"keys":[
            {"key_id":"one","class":"external","enabled":true},
            {"key_id":"two","class":"internal","enabled":true}
        ]}"#;
        for secrets in [
            r#"{"one":"same-secret","two":"same-secret"}"#,
            r#"{"one":"","two":"other-secret"}"#,
        ] {
            let error = GatewayKeyring::from_json(policies, secrets).unwrap_err();
            assert!(matches!(error, GatewayKeyringError::InvalidApiKeys));
        }
    }

    #[test]
    fn nvidia_key_requires_a_nonempty_unique_worker_allowlist() {
        let policies = r#"{"version":1,"keys":[
            {"key_id":"nvidia","class":"external_nvidia","enabled":true}
        ]}"#;
        let secrets = r#"{"nvidia":"nvidia-secret"}"#;

        for urls in [
            None,
            Some(""),
            Some("http://nvidia:30000 http://nvidia:30000"),
        ] {
            let error =
                GatewayKeyring::from_json_with_nvidia_workers(policies, secrets, urls).unwrap_err();
            assert!(matches!(
                error,
                GatewayKeyringError::InvalidNvidiaWorkerUrls(_)
            ));
        }
    }

    #[test]
    fn nvidia_key_is_priority_100_and_fail_closed_to_its_allowlist() {
        let unconfigured = GatewayKeyIdentity::new("unconfigured", GatewayKeyClass::ExternalNvidia);
        assert!(!unconfigured.allows_worker_url("http://nvidia:30000"));

        let keyring = GatewayKeyring::from_json_with_nvidia_workers(
            r#"{"version":1,"keys":[
                {"key_id":"nvidia","class":"external_nvidia","enabled":true}
            ]}"#,
            r#"{"nvidia":"nvidia-secret"}"#,
            Some("http://nvidia:30000"),
        )
        .unwrap();
        let identity = keyring.authenticate("nvidia-secret").unwrap();
        assert_eq!(identity.class(), GatewayKeyClass::ExternalNvidia);
        assert_eq!(identity.priority_override(), Some(100));
        assert!(identity.allows_worker_url("http://nvidia:30000"));
        assert!(identity.allows_worker_url("http://nvidia:30000/"));
        assert!(!identity.allows_worker_url("http://amd:30000"));
        assert!(!identity.allows_worker_url("not-a-worker-url"));
        assert!(!identity.allows_external_model());
    }

    #[test]
    fn dedicated_key_is_priority_100_and_cannot_use_external_models() {
        let keyring = GatewayKeyring::from_json(
            r#"{"version":1,"keys":[
                {"key_id":"dedicated","class":"dedicated","enabled":true}
            ]}"#,
            r#"{"dedicated":"dedicated-secret"}"#,
        )
        .unwrap();
        let identity = keyring.authenticate("dedicated-secret").unwrap();
        assert_eq!(identity.class(), GatewayKeyClass::Dedicated);
        assert_eq!(identity.priority_override(), Some(100));
        assert!(identity.is_dedicated());
        assert!(!identity.allows_external_model());
    }

    #[test]
    fn rejects_unknown_class_or_missing_enabled() {
        for policies in [
            r#"{"version":1,"keys":[{"key_id":"one","class":"partner","enabled":true}]}"#,
            r#"{"version":1,"keys":[{"key_id":"one","class":"proxy","enabled":true}]}"#,
            r#"{"version":1,"keys":[{"key_id":"one","class":"external"}]}"#,
        ] {
            let error = GatewayKeyring::from_json(policies, r#"{"one":"secret"}"#).unwrap_err();
            assert!(matches!(error, GatewayKeyringError::InvalidPolicies(_)));
        }
    }

    #[test]
    fn credential_headers_accept_migration_aliases_and_reject_conflicts() {
        for (name, value) in [
            (header::AUTHORIZATION, "Bearer secret"),
            (X_API_KEY.clone(), "secret"),
            (OCP_APIM_SUBSCRIPTION_KEY.clone(), "secret"),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(name, value.parse().unwrap());
            assert_eq!(credential_from_headers(&headers).unwrap(), "secret");
        }

        let mut same = HeaderMap::new();
        same.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        same.insert(&X_API_KEY, "secret".parse().unwrap());
        assert_eq!(credential_from_headers(&same).unwrap(), "secret");

        let mut conflict = HeaderMap::new();
        conflict.insert(header::AUTHORIZATION, "Bearer secret".parse().unwrap());
        conflict.insert(&X_API_KEY, "other".parse().unwrap());
        assert!(credential_from_headers(&conflict).is_err());
    }

    #[test]
    fn constant_time_digest_comparison_is_exact() {
        let one = digest(b"one");
        let another_one = digest(b"one");
        let two = digest(b"two");
        assert!(constant_time_eq(&one, &another_one));
        assert!(!constant_time_eq(&one, &two));
    }

    #[test]
    fn pd_proxy_key_authenticates_without_priority_override_or_secret_retention() {
        let keyring = GatewayKeyring::from_pd_proxy_key("proxy-secret").unwrap();
        let identity = keyring.authenticate("proxy-secret").unwrap();
        assert_eq!(identity.key_id(), "pd-proxy-upstream");
        assert_eq!(identity.class(), GatewayKeyClass::Proxy);
        assert_eq!(identity.priority_override(), None);
        assert!(keyring.authenticate("wrong-secret").is_none());
        assert!(!format!("{keyring:?}").contains("proxy-secret"));
    }

    #[test]
    fn pd_proxy_key_rejects_invalid_values() {
        for key in ["", "contains a space", "contains\nnewline"] {
            assert!(matches!(
                GatewayKeyring::from_pd_proxy_key(key),
                Err(GatewayKeyringError::InvalidApiKeys)
            ));
        }
    }
}
