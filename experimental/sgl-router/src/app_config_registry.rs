// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Context, Result};
use reqwest::header::{ETAG, IF_NONE_MATCH};
use reqwest::StatusCode;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use url::Url;

const APP_CONFIG_SCOPE_RESOURCE: &str = "https://azconfig.io";
const MANAGED_IDENTITY_TOKEN_URL: &str = "http://169.254.169.254/metadata/identity/oauth2/token";
const ACA_MANAGED_IDENTITY_API_VERSION: &str = "2019-08-01";
const IMDS_MANAGED_IDENTITY_API_VERSION: &str = "2018-02-01";

#[derive(Clone, Debug, Default)]
pub struct RegistrySource {
    pub inline_json: Option<String>,
    pub file: Option<String>,
    pub app_config: Option<AppConfigSource>,
}

#[derive(Clone, Debug)]
pub struct AppConfigSource {
    pub endpoint: String,
    pub key: String,
    pub label: Option<String>,
    pub managed_identity_client_id: Option<String>,
    pub timeout_secs: u64,
}

#[derive(Debug, Deserialize)]
pub struct WorkerRegistry {
    pub schema: String,
    #[serde(default)]
    pub pools: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub workers: HashMap<String, RegistryWorker>,
}

#[derive(Debug, Deserialize)]
pub struct RegistryWorker {
    pub url: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub pool_url_suffixes: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct ManagedIdentityToken {
    access_token: String,
    #[serde(default)]
    expires_on: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct AppConfigKeyValue {
    value: String,
    #[serde(default)]
    etag: Option<serde_json::Value>,
    #[serde(default, rename = "@etag")]
    alternate_etag: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedWorkerUrl {
    pub name: String,
    pub url: String,
}

#[derive(Debug)]
pub struct AppConfigValue {
    pub value: String,
    pub etag: Option<String>,
}

#[derive(Clone)]
pub struct AppConfigClient {
    source: AppConfigSource,
    client: reqwest::Client,
    cached_token: Arc<Mutex<Option<CachedToken>>>,
}

#[derive(Clone, Debug)]
struct CachedToken {
    access_token: String,
    expires_at: SystemTime,
}

#[derive(Debug)]
struct ManagedIdentityRequest {
    url: Url,
    headers: Vec<(&'static str, String)>,
}

fn default_enabled() -> bool {
    true
}

impl RegistrySource {
    pub fn is_configured(&self) -> bool {
        self.inline_json.is_some() || self.file.is_some() || self.app_config.is_some()
    }

    pub async fn load(&self) -> Result<String> {
        let configured = self.inline_json.is_some() as u8
            + self.file.is_some() as u8
            + self.app_config.is_some() as u8;
        if configured > 1 {
            return Err(anyhow!(
                "configure only one worker registry source: inline JSON, file, or App Configuration"
            ));
        }
        if let Some(json) = &self.inline_json {
            return Ok(json.clone());
        }
        if let Some(path) = &self.file {
            return std::fs::read_to_string(path)
                .with_context(|| format!("read worker registry file {path}"));
        }
        if let Some(app_config) = &self.app_config {
            return app_config.fetch_value().await;
        }
        Err(anyhow!("worker registry source is not configured"))
    }
}

impl AppConfigSource {
    pub async fn fetch_value(&self) -> Result<String> {
        let client = AppConfigClient::new(self.clone())?;
        client
            .fetch(None)
            .await
            .map_err(|_| anyhow!("fetch worker registry from Azure App Configuration failed"))?
            .map(|value| value.value)
            .context("Azure App Configuration unexpectedly returned not-modified")
    }
}

impl AppConfigClient {
    pub fn new(source: AppConfigSource) -> Result<Self> {
        if source.timeout_secs == 0 {
            return Err(anyhow!("App Configuration timeout must be greater than 0"));
        }
        // Validate the endpoint/key before the background task is detached so a
        // local configuration error fails startup rather than becoming a poll
        // loop that can never succeed.
        app_config_key_url(&source.endpoint, &source.key, source.label.as_deref())?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(source.timeout_secs))
            .build()
            .context("build App Configuration HTTP client")?;
        Ok(Self {
            source,
            client,
            cached_token: Arc::new(Mutex::new(None)),
        })
    }

    /// Fetch an App Configuration value, optionally using an accepted ETag.
    ///
    /// `None` means the service returned HTTP 304. Callers must only pass the
    /// ETag of their last *validated and applied* document; retaining an ETag
    /// from a rejected document would pin the consumer to invalid state.
    pub async fn fetch(&self, accepted_etag: Option<&str>) -> Result<Option<AppConfigValue>> {
        let token = self.access_token(false).await?;
        let response = self.send(&token, accepted_etag).await?;
        if response.status() == StatusCode::UNAUTHORIZED {
            // A token can be revoked before its advertised expiry. Invalidate
            // and retry exactly once with a newly minted token.
            let token = self.access_token(true).await?;
            return self
                .decode_response(self.send(&token, accepted_etag).await?)
                .await;
        }
        self.decode_response(response).await
    }

    async fn send(&self, token: &str, accepted_etag: Option<&str>) -> Result<reqwest::Response> {
        let url = app_config_key_url(
            &self.source.endpoint,
            &self.source.key,
            self.source.label.as_deref(),
        )?;
        let mut request = self.client.get(url).bearer_auth(token);
        if let Some(etag) = accepted_etag.filter(|etag| !etag.trim().is_empty()) {
            request = request.header(IF_NONE_MATCH, etag);
        }
        request
            .send()
            .await
            .context("fetch value from Azure App Configuration")
    }

    async fn decode_response(&self, response: reqwest::Response) -> Result<Option<AppConfigValue>> {
        if response.status() == StatusCode::NOT_MODIFIED {
            return Ok(None);
        }
        let response = response
            .error_for_status()
            .context("Azure App Configuration request failed")?;
        let header_etag = response
            .headers()
            .get(ETAG)
            .map(|value| {
                value
                    .to_str()
                    .context("Azure App Configuration returned a non-text ETag header")
                    .map(str::to_owned)
            })
            .transpose()?;
        let body = response
            .json::<AppConfigKeyValue>()
            .await
            .context("decode Azure App Configuration key/value response")?;
        let etag = match header_etag {
            Some(etag) => Some(etag),
            None => body_etag(body.etag, body.alternate_etag)?,
        };
        if etag.as_ref().is_some_and(|etag| etag.is_empty()) {
            return Err(anyhow!("Azure App Configuration returned an empty ETag"));
        }
        Ok(Some(AppConfigValue {
            value: body.value,
            etag,
        }))
    }

    async fn access_token(&self, force_refresh: bool) -> Result<String> {
        let mut guard = self.cached_token.lock().await;
        let now = SystemTime::now();
        if !force_refresh {
            if let Some(token) = guard.as_ref() {
                // Refresh one minute early so a token cannot expire between
                // acquisition and the App Configuration request.
                if token
                    .expires_at
                    .duration_since(now)
                    .is_ok_and(|remaining| remaining > Duration::from_secs(60))
                {
                    return Ok(token.access_token.clone());
                }
            }
        }
        let token = fetch_managed_identity_token(
            &self.client,
            self.source.managed_identity_client_id.as_deref(),
        )
        .await?;
        let access_token = token.access_token.clone();
        *guard = Some(token);
        Ok(access_token)
    }
}

fn body_etag(
    primary: Option<serde_json::Value>,
    alternate: Option<serde_json::Value>,
) -> Result<Option<String>> {
    let mut saw_empty = false;
    for (field, value) in [("etag", primary), ("@etag", alternate)] {
        match value {
            None | Some(serde_json::Value::Null) => continue,
            Some(serde_json::Value::String(value)) if value.is_empty() => {
                saw_empty = true;
            }
            Some(serde_json::Value::String(value)) => return Ok(Some(value)),
            Some(_) => {
                return Err(anyhow!(
                    "Azure App Configuration returned a non-text {field} field"
                ));
            }
        }
    }
    if saw_empty {
        return Err(anyhow!("Azure App Configuration returned an empty ETag"));
    }
    Ok(None)
}

pub fn parse_registry(raw_json: &str) -> Result<WorkerRegistry> {
    let registry: WorkerRegistry =
        serde_json::from_str(raw_json).context("parse worker registry JSON")?;
    if registry.schema != "macaron.worker_registry.v1" {
        return Err(anyhow!(
            "unsupported worker registry schema {:?}",
            registry.schema
        ));
    }
    Ok(registry)
}

pub fn worker_urls_for_pool(
    registry: &WorkerRegistry,
    pool: &str,
    default_url_suffix: Option<&str>,
) -> Result<Vec<String>> {
    named_worker_urls_for_pool(registry, pool, default_url_suffix)
        .map(|workers| workers.into_iter().map(|worker| worker.url).collect())
}

pub fn named_worker_urls_for_pool(
    registry: &WorkerRegistry,
    pool: &str,
    default_url_suffix: Option<&str>,
) -> Result<Vec<NamedWorkerUrl>> {
    let names = registry
        .pools
        .get(pool)
        .with_context(|| format!("worker registry pool {pool:?} is missing"))?;
    if names.is_empty() {
        return Err(anyhow!("worker registry pool {pool:?} is empty"));
    }

    names
        .iter()
        .map(|name| {
            let worker = registry.workers.get(name).with_context(|| {
                format!("worker registry pool {pool:?} references missing worker {name:?}")
            })?;
            if !worker.enabled {
                return Err(anyhow!(
                    "worker registry pool {pool:?} references disabled worker {name:?}"
                ));
            }
            let url = worker
                .url
                .as_deref()
                .map(str::trim)
                .filter(|url| !url.is_empty())
                .with_context(|| {
                    format!("worker registry worker {name:?} in pool {pool:?} has no url")
                })?;
            let suffix = worker
                .pool_url_suffixes
                .get(pool)
                .map(String::as_str)
                .or(default_url_suffix)
                .unwrap_or("");
            Ok(NamedWorkerUrl {
                name: name.clone(),
                url: format!("{url}{suffix}"),
            })
        })
        .collect()
}

fn app_config_key_url(endpoint: &str, key: &str, label: Option<&str>) -> Result<Url> {
    if endpoint.trim().is_empty() {
        return Err(anyhow!("App Configuration endpoint is empty"));
    }
    if key.trim().is_empty() {
        return Err(anyhow!("App Configuration key is empty"));
    }
    let authority = endpoint
        .strip_prefix("https://")
        .context("App Configuration endpoint must use canonical https:// form")?;
    if authority.is_empty() || authority.starts_with('/') {
        return Err(anyhow!(
            "App Configuration endpoint must include a canonical host authority"
        ));
    }
    let base = Url::parse(endpoint).context("parse App Configuration endpoint")?;
    if base.scheme() != "https" {
        return Err(anyhow!("App Configuration endpoint must use https"));
    }
    if base.host_str().is_none() {
        return Err(anyhow!("App Configuration endpoint must include a host"));
    }
    if !base.username().is_empty() || base.password().is_some() {
        return Err(anyhow!(
            "App Configuration endpoint must not include userinfo"
        ));
    }
    if base.path() != "/" || base.query().is_some() || base.fragment().is_some() {
        return Err(anyhow!(
            "App Configuration endpoint must be a service base URL without path, query, or fragment"
        ));
    }
    let endpoint = endpoint.strip_suffix('/').unwrap_or(endpoint);
    let encoded_key: String = url::form_urlencoded::byte_serialize(key.as_bytes()).collect();
    let mut url = Url::parse(&format!("{endpoint}/kv/{encoded_key}"))
        .context("build App Configuration key URL")?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("api-version", "1.0");
        if let Some(label) = label.filter(|label| !label.trim().is_empty()) {
            query.append_pair("label", label);
        }
    }
    Ok(url)
}

async fn fetch_managed_identity_token(
    client: &reqwest::Client,
    client_id: Option<&str>,
) -> Result<CachedToken> {
    let token_request = managed_identity_token_request(client_id)?;
    let mut request = client.get(token_request.url);
    for (name, value) in token_request.headers {
        request = request.header(name, value);
    }
    let token = request
        .send()
        .await
        .context("fetch managed identity token for Azure App Configuration")?
        .error_for_status()
        .context("managed identity token request failed")?
        .json::<ManagedIdentityToken>()
        .await
        .context("decode managed identity token response")?;
    if token.access_token.trim().is_empty() {
        return Err(anyhow!("managed identity returned an empty access token"));
    }
    let now = SystemTime::now();
    // ACA and IMDS currently return a Unix epoch value (often encoded as a
    // string). Keep a conservative five-minute cache when the optional field
    // is absent or changes shape; correctness then degrades to extra token
    // requests, never to using a token indefinitely.
    let expires_at = token
        .expires_on
        .as_ref()
        .and_then(parse_token_expiry)
        .filter(|expiry| *expiry > now)
        .unwrap_or_else(|| now + Duration::from_secs(300));
    Ok(CachedToken {
        access_token: token.access_token,
        expires_at,
    })
}

fn parse_token_expiry(value: &serde_json::Value) -> Option<SystemTime> {
    let seconds = match value {
        serde_json::Value::Number(value) => value.as_u64(),
        serde_json::Value::String(value) => value.parse::<u64>().ok(),
        _ => None,
    }?;
    UNIX_EPOCH.checked_add(Duration::from_secs(seconds))
}

fn managed_identity_token_request(client_id: Option<&str>) -> Result<ManagedIdentityRequest> {
    let mut headers = Vec::new();
    let mut url = if let Some(endpoint) = std::env::var("IDENTITY_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let header = std::env::var("IDENTITY_HEADER")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .context("IDENTITY_HEADER is required when IDENTITY_ENDPOINT is set")?;
        headers.push(("X-IDENTITY-HEADER", header));
        Url::parse(&endpoint).context("parse IDENTITY_ENDPOINT managed identity URL")?
    } else {
        headers.push(("Metadata", "true".to_string()));
        Url::parse(MANAGED_IDENTITY_TOKEN_URL).expect("valid IMDS token URL")
    };
    {
        let mut query = url.query_pairs_mut();
        query.append_pair(
            "api-version",
            if headers.iter().any(|(name, _)| *name == "X-IDENTITY-HEADER") {
                ACA_MANAGED_IDENTITY_API_VERSION
            } else {
                IMDS_MANAGED_IDENTITY_API_VERSION
            },
        );
        query.append_pair("resource", APP_CONFIG_SCOPE_RESOURCE);
        if let Some(client_id) = client_id.filter(|value| !value.trim().is_empty()) {
            query.append_pair("client_id", client_id);
        }
    }
    for (name, value) in &headers {
        if value.trim().is_empty() {
            return Err(anyhow!("managed identity header {name} is empty"));
        }
    }
    Ok(ManagedIdentityRequest { url, headers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    async fn one_shot_response(
        status: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> reqwest::Response {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let rendered_headers = headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>();
        let body = body.to_owned();
        let status = status.to_owned();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let _ = stream.read(&mut request).await.unwrap();
            let response = format!(
                "HTTP/1.1 {status}\r\n{rendered_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        reqwest::get(format!("http://{address}/")).await.unwrap()
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn sample_registry() -> WorkerRegistry {
        parse_registry(
            r#"{
              "schema": "macaron.worker_registry.v1",
              "model": "zai-org/GLM-5.2-FP8",
              "pools": {
                "glm52-main": ["b200-01", "b200-02"],
                "internal-low": ["b200-01"]
              },
              "workers": {
                "b200-01": {
                  "url": "http://10.0.0.1:30000",
                  "enabled": true,
                  "pool_url_suffixes": {"internal-low": "@tier=shared"}
                },
                "b200-02": {"url": "http://10.0.0.2:30000", "enabled": true}
              }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn builds_worker_urls_for_pool_in_order() {
        let registry = sample_registry();
        let urls = worker_urls_for_pool(&registry, "glm52-main", None).unwrap();
        assert_eq!(urls, vec!["http://10.0.0.1:30000", "http://10.0.0.2:30000"]);
    }

    #[test]
    fn applies_pool_specific_suffix_before_default_suffix() {
        let registry = sample_registry();
        let urls = worker_urls_for_pool(&registry, "internal-low", Some("@tier=bulk")).unwrap();
        assert_eq!(urls, vec!["http://10.0.0.1:30000@tier=shared"]);
    }

    #[test]
    fn rejects_unknown_schema() {
        let err = parse_registry(r#"{"schema":"other","pools":{},"workers":{}}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported worker registry schema"), "{err}");
    }

    #[test]
    fn app_config_url_encodes_slash_key_and_label() {
        let url = app_config_key_url(
            "https://macaron-llm-deploy-prod.azconfig.io/",
            "macaron/prod/worker-registry/glm52/current",
            Some("prod-candidate"),
        )
        .unwrap();
        let rendered = url.as_str();
        assert!(rendered.contains("/kv/macaron%2Fprod%2Fworker-registry%2Fglm52%2Fcurrent"));
        assert!(rendered.contains("api-version=1.0"));
        assert!(rendered.contains("label=prod-candidate"));
    }

    #[test]
    fn app_config_url_rejects_unsafe_or_non_base_endpoints() {
        for endpoint in [
            "http://config.example.test",
            "https://user@config.example.test",
            "https://config.example.test/prefix",
            "https://config.example.test///",
            "https://config.example.test?redirect=evil",
            "https://config.example.test#fragment",
            "https:///missing-host",
        ] {
            assert!(
                app_config_key_url(endpoint, "key", Some("runtime")).is_err(),
                "unsafe endpoint unexpectedly accepted: {endpoint}"
            );
        }
    }

    #[test]
    fn managed_identity_request_uses_aca_identity_endpoint_when_present() {
        let _guard = env_lock().lock().unwrap();
        std::env::set_var("IDENTITY_ENDPOINT", "http://localhost:42356/msi/token");
        std::env::set_var("IDENTITY_HEADER", "secret-header");
        let req = managed_identity_token_request(Some("client-1")).unwrap();
        let rendered = req.url.as_str();
        assert!(rendered.starts_with("http://localhost:42356/msi/token?"));
        assert!(rendered.contains("api-version=2019-08-01"));
        assert!(rendered.contains("resource=https%3A%2F%2Fazconfig.io"));
        assert!(rendered.contains("client_id=client-1"));
        assert_eq!(
            req.headers,
            vec![("X-IDENTITY-HEADER", "secret-header".to_string())]
        );
        std::env::remove_var("IDENTITY_ENDPOINT");
        std::env::remove_var("IDENTITY_HEADER");
    }

    #[test]
    fn managed_identity_request_falls_back_to_imds() {
        let _guard = env_lock().lock().unwrap();
        std::env::remove_var("IDENTITY_ENDPOINT");
        std::env::remove_var("IDENTITY_HEADER");
        let req = managed_identity_token_request(None).unwrap();
        assert!(req.url.as_str().starts_with(MANAGED_IDENTITY_TOKEN_URL));
        assert!(req.url.as_str().contains("api-version=2018-02-01"));
        assert_eq!(req.headers, vec![("Metadata", "true".to_string())]);
    }

    #[tokio::test]
    async fn app_config_decode_prefers_header_etag_then_body_variants() {
        let source = AppConfigSource {
            endpoint: "https://example.invalid".into(),
            key: "key".into(),
            label: Some("label".into()),
            managed_identity_client_id: None,
            timeout_secs: 1,
        };
        let client = AppConfigClient::new(source).unwrap();

        let response = one_shot_response(
            "200 OK",
            &[
                ("Content-Type", "application/json"),
                ("ETag", "header-etag"),
            ],
            r#"{"value":"{}","etag":"body-etag","@etag":"alternate-etag"}"#,
        )
        .await;
        let value = client.decode_response(response).await.unwrap().unwrap();
        assert_eq!(value.etag.as_deref(), Some("header-etag"));

        let response = one_shot_response(
            "200 OK",
            &[("Content-Type", "application/json")],
            r#"{"value":"{}","etag":"body-etag"}"#,
        )
        .await;
        let value = client.decode_response(response).await.unwrap().unwrap();
        assert_eq!(value.etag.as_deref(), Some("body-etag"));

        let response = one_shot_response(
            "200 OK",
            &[("Content-Type", "application/json")],
            r#"{"value":"{}","@etag":"alternate-etag"}"#,
        )
        .await;
        let value = client.decode_response(response).await.unwrap().unwrap();
        assert_eq!(value.etag.as_deref(), Some("alternate-etag"));

        let response = one_shot_response(
            "200 OK",
            &[("Content-Type", "application/json")],
            r#"{"value":"{}","etag":"","@etag":"alternate-etag"}"#,
        )
        .await;
        let value = client.decode_response(response).await.unwrap().unwrap();
        assert_eq!(value.etag.as_deref(), Some("alternate-etag"));

        let response = one_shot_response(
            "200 OK",
            &[
                ("Content-Type", "application/json"),
                ("ETag", "header-etag"),
            ],
            r#"{"value":"{}","etag":42}"#,
        )
        .await;
        let value = client.decode_response(response).await.unwrap().unwrap();
        assert_eq!(value.etag.as_deref(), Some("header-etag"));

        let response = one_shot_response(
            "200 OK",
            &[("Content-Type", "application/json")],
            r#"{"value":"{}","etag":""}"#,
        )
        .await;
        assert!(client.decode_response(response).await.is_err());
    }

    #[tokio::test]
    async fn app_config_decode_treats_304_as_unchanged() {
        let source = AppConfigSource {
            endpoint: "https://example.invalid".into(),
            key: "key".into(),
            label: Some("label".into()),
            managed_identity_client_id: None,
            timeout_secs: 1,
        };
        let client = AppConfigClient::new(source).unwrap();
        let response = one_shot_response("304 Not Modified", &[], "").await;
        assert!(client.decode_response(response).await.unwrap().is_none());
    }
}
