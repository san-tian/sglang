// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! App Configuration runtime-lease overlay for static worker discovery.
//!
//! The worker registry remains the complete, durable base topology. This
//! module only removes/re-adds workers already present in that base, using
//! URL-shaped [`WorkerId`] values identical to ordinary static discovery.
//! Invalid or unavailable runtime documents never replace the last accepted
//! state. Expired leases intentionally remain enforced until a controller
//! removes them with an App Configuration CAS update.

use crate::app_config_registry::{
    AppConfigClient, AppConfigSource, AppConfigValue, NamedWorkerUrl,
};
use crate::config::{StaticUrlsDiscoveryConfig, WorkerBearerKeyConfig};
use crate::discovery::static_urls::build_worker_specs;
use crate::discovery::{DiscoveryEvent, WorkerSpec};
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use uuid::Uuid;

pub const WORKER_RUNTIME_APP_CONFIG_KEY: &str = "macaron/prod/worker-runtime/glm52/current";
pub const WORKER_RUNTIME_APP_CONFIG_LABEL: &str = "runtime";
pub const WORKER_RUNTIME_SCHEMA: &str = "macaron.worker_runtime.v1";
pub const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;

#[derive(Clone, Debug)]
pub struct RuntimeLeaseDiscoveryConfig {
    pub source: AppConfigSource,
    pub pool: String,
    pub base_workers: Vec<NamedWorkerUrl>,
    pub known_workers: BTreeSet<String>,
    pub known_pools: BTreeSet<String>,
    pub poll_interval_secs: u64,
}

#[derive(Clone, Debug)]
struct NamedWorkerSpec {
    name: String,
    spec: WorkerSpec,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RuntimeLeaseDocument {
    schema: String,
    generation: u64,
    updated_at: String,
    leases: BTreeMap<String, RuntimeLease>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RuntimeLease {
    lease_id: String,
    state: RuntimeLeaseState,
    pools: Vec<String>,
    watchdog_mode: WatchdogMode,
    keep_metrics: bool,
    owner: String,
    reason: String,
    created_at: String,
    expires_at: String,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    baseline_registry_etag: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RuntimeLeaseState {
    Drained,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum WatchdogMode {
    Active,
    MonitorOnly,
    Disabled,
}

#[derive(Default)]
struct AppliedRuntimeState {
    generation: Option<u64>,
    document: Option<RuntimeLeaseDocument>,
    drained: BTreeSet<String>,
    accepted_etag: Option<String>,
}

struct RuntimeProposal {
    document: RuntimeLeaseDocument,
    drained: BTreeSet<String>,
    events: Vec<DiscoveryEvent>,
    accepted_etag: Option<String>,
    expired_lease_count: usize,
    stale_pool_reference_count: usize,
}

impl AppliedRuntimeState {
    fn propose(
        &self,
        raw_json: &str,
        etag: Option<String>,
        cfg: &RuntimeLeaseDiscoveryConfig,
        workers: &[NamedWorkerSpec],
    ) -> Result<RuntimeProposal> {
        let (document, drained, expired_lease_count, stale_pool_reference_count) =
            parse_and_validate(raw_json, cfg)?;

        if let Some(current_generation) = self.generation {
            if document.generation < current_generation {
                return Err(anyhow!(
                    "runtime lease generation regressed from {current_generation} to {}",
                    document.generation
                ));
            }
            if document.generation == current_generation
                && self.document.as_ref() != Some(&document)
            {
                return Err(anyhow!(
                    "runtime lease document changed without advancing generation {current_generation}"
                ));
            }
        }

        let effective_count = workers
            .iter()
            .filter(|worker| !drained.contains(&worker.name))
            .count();
        if effective_count == 0 {
            return Err(anyhow!(
                "runtime lease document would drain every worker in pool {:?}",
                cfg.pool
            ));
        }

        // Removing newly drained workers always precedes adding newly
        // released workers. The base registry order makes the event plan
        // deterministic across replicas.
        let mut events = Vec::new();
        for worker in workers {
            if drained.contains(&worker.name) && !self.drained.contains(&worker.name) {
                events.push(DiscoveryEvent::Removed {
                    id: worker.spec.id.clone(),
                });
            }
        }
        for worker in workers {
            if self.drained.contains(&worker.name) && !drained.contains(&worker.name) {
                events.push(DiscoveryEvent::Added(worker.spec.clone()));
            }
        }

        Ok(RuntimeProposal {
            document,
            drained,
            events,
            accepted_etag: etag,
            expired_lease_count,
            stale_pool_reference_count,
        })
    }

    fn commit(&mut self, proposal: RuntimeProposal) {
        self.generation = Some(proposal.document.generation);
        self.document = Some(proposal.document);
        self.drained = proposal.drained;
        self.accepted_etag = proposal.accepted_etag;
    }
}

fn accept_initial_document(
    value: AppConfigValue,
    cfg: &RuntimeLeaseDiscoveryConfig,
    workers: &[NamedWorkerSpec],
) -> Result<(AppliedRuntimeState, usize, usize)> {
    let mut state = AppliedRuntimeState::default();
    let proposal = state.propose(&value.value, value.etag, cfg, workers)?;
    let expired_lease_count = proposal.expired_lease_count;
    let stale_pool_reference_count = proposal.stale_pool_reference_count;
    state.commit(proposal);
    Ok((state, expired_lease_count, stale_pool_reference_count))
}

fn parse_and_validate(
    raw_json: &str,
    cfg: &RuntimeLeaseDiscoveryConfig,
) -> Result<(RuntimeLeaseDocument, BTreeSet<String>, usize, usize)> {
    let document: RuntimeLeaseDocument =
        serde_json::from_str(raw_json).context("parse worker runtime lease JSON")?;
    if document.schema != WORKER_RUNTIME_SCHEMA {
        return Err(anyhow!(
            "unsupported worker runtime lease schema {:?}",
            document.schema
        ));
    }
    parse_rfc3339("updated_at", &document.updated_at)?;

    let mut drained = BTreeSet::new();
    let mut expired_lease_count = 0;
    let mut stale_pool_reference_count = 0;
    let now = Utc::now();
    let current_pool_members: BTreeSet<&str> = cfg
        .base_workers
        .iter()
        .map(|worker| worker.name.as_str())
        .collect();
    for (worker_name, lease) in &document.leases {
        if !is_runtime_name(worker_name) {
            return Err(anyhow!(
                "worker runtime lease has an invalid worker name {worker_name:?}"
            ));
        }
        if !cfg.known_workers.contains(worker_name) {
            return Err(anyhow!(
                "worker runtime lease references unknown base-registry worker {worker_name:?}"
            ));
        }
        require_non_empty("lease_id", &lease.lease_id)?;
        let lease_uuid = Uuid::parse_str(&lease.lease_id).with_context(|| {
            format!("worker runtime lease for {worker_name:?} has a non-UUID lease_id")
        })?;
        if lease_uuid.to_string() != lease.lease_id {
            return Err(anyhow!(
                "worker runtime lease for {worker_name:?} lease_id must be canonical lowercase UUID"
            ));
        }
        if !is_v1_uuid_shape(&lease.lease_id) {
            return Err(anyhow!(
                "worker runtime lease for {worker_name:?} lease_id must use UUID version 1-5 and RFC4122 variant"
            ));
        }
        require_trimmed_bounded("owner", &lease.owner, 128)?;
        require_trimmed_bounded("reason", &lease.reason, 500)?;
        if lease.pools.is_empty() {
            return Err(anyhow!(
                "worker runtime lease for {worker_name:?} has no pools"
            ));
        }
        if lease.pools.iter().any(|pool| pool == "*") && lease.pools.len() != 1 {
            return Err(anyhow!(
                "worker runtime lease for {worker_name:?} must use [\"*\"] alone"
            ));
        }
        let mut seen_pools = BTreeSet::new();
        for pool in &lease.pools {
            require_non_empty("pools[]", pool)?;
            if pool != "*" && !is_runtime_name(pool) {
                return Err(anyhow!(
                    "worker runtime lease for {worker_name:?} has an invalid pool name {pool:?}"
                ));
            }
            if !seen_pools.insert(pool) {
                return Err(anyhow!(
                    "worker runtime lease for {worker_name:?} repeats pool {pool:?}"
                ));
            }
            if pool != "*" && !cfg.known_pools.contains(pool) {
                stale_pool_reference_count += 1;
            }
        }
        if !lease.keep_metrics {
            return Err(anyhow!(
                "worker runtime lease for {worker_name:?} must keep metrics enabled in v1"
            ));
        }
        if let Some(etag) = &lease.baseline_registry_etag {
            require_bounded_nonempty("baseline_registry_etag", etag, 512).with_context(|| {
                format!("worker runtime lease for {worker_name:?} has an invalid baseline ETag")
            })?;
        }
        let created_at = parse_rfc3339("created_at", &lease.created_at)?;
        let expires_at = parse_rfc3339("expires_at", &lease.expires_at)?;
        if expires_at <= created_at {
            return Err(anyhow!(
                "worker runtime lease for {worker_name:?} expires_at must be after created_at"
            ));
        }
        // The enum deserializer has already restricted state to `drained`.
        // Expiry is deliberately NOT part of this predicate: only removal of
        // the lease by the CAS controller rejoins the worker.
        let applies_to_current_pool = current_pool_members.contains(worker_name.as_str())
            && (lease.pools.iter().any(|pool| pool == "*")
                || lease.pools.iter().any(|pool| pool == &cfg.pool));
        if matches!(lease.state, RuntimeLeaseState::Drained) && applies_to_current_pool {
            drained.insert(worker_name.clone());
            if expires_at <= now {
                expired_lease_count += 1;
            }
        }
    }
    Ok((
        document,
        drained,
        expired_lease_count,
        stale_pool_reference_count,
    ))
}

fn require_non_empty(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(anyhow!("worker runtime lease field {field} is empty"));
    }
    Ok(())
}

fn require_trimmed_bounded(field: &str, value: &str, max_chars: usize) -> Result<()> {
    if value.is_empty() || value != value.trim() || value.contains('\r') || value.contains('\n') {
        return Err(anyhow!(
            "worker runtime lease field {field} must be a non-empty trimmed single-line string"
        ));
    }
    if value.chars().count() > max_chars {
        return Err(anyhow!(
            "worker runtime lease field {field} exceeds {max_chars} characters"
        ));
    }
    Ok(())
}

fn require_bounded_nonempty(field: &str, value: &str, max_chars: usize) -> Result<()> {
    if value.is_empty()
        || value != value.trim()
        || value.contains('\r')
        || value.contains('\n')
        || value.chars().count() > max_chars
    {
        return Err(anyhow!(
            "worker runtime lease field {field} must contain 1-{max_chars} trimmed single-line characters"
        ));
    }
    Ok(())
}

fn is_runtime_name(value: &str) -> bool {
    let bytes = value.as_bytes();
    (1..=128).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

fn is_v1_uuid_shape(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && bytes[14].is_ascii_digit()
        && matches!(bytes[14], b'1'..=b'5')
        && matches!(bytes[19], b'8' | b'9' | b'a' | b'b')
}

fn deserialize_optional_string<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
}

fn parse_rfc3339(field: &str, value: &str) -> Result<DateTime<Utc>> {
    if !is_rfc3339_utc_shape(value) {
        return Err(anyhow!(
            "worker runtime lease field {field} must use canonical RFC3339 UTC Z notation"
        ));
    }
    DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("worker runtime lease field {field} is not RFC3339"))
        .map(|value| value.with_timezone(&Utc))
}

fn is_rfc3339_utc_shape(value: &str) -> bool {
    let bytes = value.as_bytes();
    if !(20..=27).contains(&bytes.len())
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || *bytes.last().unwrap_or(&0) != b'Z'
    {
        return false;
    }
    for (index, byte) in bytes[..19].iter().enumerate() {
        if !matches!(index, 4 | 7 | 10 | 13 | 16) && !byte.is_ascii_digit() {
            return false;
        }
    }
    if bytes.len() == 20 {
        return true;
    }
    bytes[19] == b'.'
        && (1..=6).contains(&(bytes.len() - 21))
        && bytes[20..bytes.len() - 1].iter().all(u8::is_ascii_digit)
}

fn build_named_worker_specs(
    cfg: &RuntimeLeaseDiscoveryConfig,
    bearer_keys: Vec<WorkerBearerKeyConfig>,
) -> Result<Vec<NamedWorkerSpec>> {
    if cfg.base_workers.is_empty() {
        return Err(anyhow!("worker runtime lease base pool is empty"));
    }
    if cfg.poll_interval_secs == 0 {
        return Err(anyhow!(
            "worker runtime lease poll interval must be greater than zero"
        ));
    }
    if cfg.source.key != WORKER_RUNTIME_APP_CONFIG_KEY {
        return Err(anyhow!(
            "worker runtime lease App Configuration key is invalid"
        ));
    }
    if cfg.source.label.as_deref() != Some(WORKER_RUNTIME_APP_CONFIG_LABEL) {
        return Err(anyhow!(
            "worker runtime lease App Configuration label is invalid"
        ));
    }
    let static_cfg = StaticUrlsDiscoveryConfig {
        urls: cfg
            .base_workers
            .iter()
            .map(|worker| worker.url.clone())
            .collect(),
        bearer_keys,
    };
    let specs = build_worker_specs(&static_cfg)?;
    Ok(cfg
        .base_workers
        .iter()
        .zip(specs)
        .map(|(worker, spec)| NamedWorkerSpec {
            name: worker.name.clone(),
            spec,
        })
        .collect())
}

/// Spawn runtime-lease discovery. The initial fetch is fail-closed: no base
/// worker is emitted until a valid runtime document has been accepted. Once
/// running, every refresh failure retains the last accepted state.
pub async fn spawn(
    cfg: RuntimeLeaseDiscoveryConfig,
    bearer_keys: Vec<WorkerBearerKeyConfig>,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    let workers = build_named_worker_specs(&cfg, bearer_keys)?;
    let client = AppConfigClient::new(cfg.source.clone())?;
    let initial = client
        .fetch(None)
        .await
        .map_err(|_| anyhow!("worker runtime lease initial fetch failed"))?
        .context("worker runtime lease initial fetch unexpectedly returned not-modified")?;
    let (mut state, expired_lease_count, stale_pool_reference_count) =
        accept_initial_document(initial, &cfg, &workers)
            .context("worker runtime lease initial document rejected")?;
    tracing::info!(
        pool = %cfg.pool,
        phase = "initial",
        generation = state.generation,
        drained_count = state.drained.len(),
        active_count = workers.len() - state.drained.len(),
        "worker runtime lease state accepted"
    );
    if expired_lease_count > 0 {
        tracing::warn!(
            pool = %cfg.pool,
            generation = state.generation,
            expired_lease_count,
            "expired worker runtime leases remain drained until CAS removal"
        );
    }
    if stale_pool_reference_count > 0 {
        tracing::warn!(
            pool = %cfg.pool,
            generation = state.generation,
            stale_pool_reference_count,
            "worker runtime leases reference pools absent from the current base registry"
        );
    }

    let handle = tokio::spawn(async move {
        for worker in workers
            .iter()
            .filter(|worker| !state.drained.contains(&worker.name))
        {
            if tx
                .send(DiscoveryEvent::Added(worker.spec.clone()))
                .await
                .is_err()
            {
                tracing::info!(
                    pool = %cfg.pool,
                    "worker runtime lease event channel closed during initial fanout; exiting"
                );
                return;
            }
        }
        let mut interval = tokio::time::interval(Duration::from_secs(cfg.poll_interval_secs));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // Consume the immediate first tick; startup already performed the
        // initial fetch above.
        interval.tick().await;
        loop {
            interval.tick().await;
            let fetched: AppConfigValue = match client.fetch(state.accepted_etag.as_deref()).await {
                Ok(Some(value)) => value,
                Ok(None) => continue,
                Err(_) => {
                    tracing::warn!(
                        pool = %cfg.pool,
                        generation = state.generation,
                        reason = "fetch_failed",
                        "worker runtime lease refresh failed; keeping last-known-good state"
                    );
                    continue;
                }
            };

            let proposal = match state.propose(&fetched.value, fetched.etag, &cfg, &workers) {
                Ok(proposal) => proposal,
                Err(error) => {
                    tracing::warn!(
                        pool = %cfg.pool,
                        generation = state.generation,
                        rejection = %error,
                        "worker runtime lease refresh rejected; keeping last-known-good state"
                    );
                    continue;
                }
            };
            for event in &proposal.events {
                if tx.send(event.clone()).await.is_err() {
                    tracing::info!(
                        pool = %cfg.pool,
                        "worker runtime lease event channel closed; exiting poller"
                    );
                    return;
                }
            }
            log_accepted(&cfg, &proposal, "refresh");
            state.commit(proposal);
        }
    });
    Ok(handle)
}

fn log_accepted(
    cfg: &RuntimeLeaseDiscoveryConfig,
    proposal: &RuntimeProposal,
    phase: &'static str,
) {
    let removed = proposal
        .events
        .iter()
        .filter(|event| matches!(event, DiscoveryEvent::Removed { .. }))
        .count();
    let added = proposal.events.len() - removed;
    tracing::info!(
        pool = %cfg.pool,
        phase,
        generation = proposal.document.generation,
        drained_count = proposal.drained.len(),
        removed_count = removed,
        added_count = added,
        "worker runtime lease state accepted"
    );
    if proposal.expired_lease_count > 0 {
        tracing::warn!(
            pool = %cfg.pool,
            generation = proposal.document.generation,
            expired_lease_count = proposal.expired_lease_count,
            "expired worker runtime leases remain drained until CAS removal"
        );
    }
    if proposal.stale_pool_reference_count > 0 {
        tracing::warn!(
            pool = %cfg.pool,
            generation = proposal.document.generation,
            stale_pool_reference_count = proposal.stale_pool_reference_count,
            "worker runtime leases reference pools absent from the current base registry"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_config_registry::NamedWorkerUrl;
    use crate::discovery::{WorkerBackend, WorkerRouteSet, WorkerTier};

    const SAMPLE: &str = include_str!("../../tests/fixtures/worker_runtime_v1.sample.json");

    fn cfg() -> RuntimeLeaseDiscoveryConfig {
        RuntimeLeaseDiscoveryConfig {
            source: AppConfigSource {
                endpoint: "https://example.invalid".into(),
                key: WORKER_RUNTIME_APP_CONFIG_KEY.into(),
                label: Some(WORKER_RUNTIME_APP_CONFIG_LABEL.into()),
                managed_identity_client_id: None,
                timeout_secs: 1,
            },
            pool: "glm52-main".into(),
            base_workers: vec![
                NamedWorkerUrl {
                    name: "sg-b300-05".into(),
                    url: "http://10.0.0.5:30000@min_priority=100@max_context_tokens=500000@tier=shared@routes=chat,responses".into(),
                },
                NamedWorkerUrl {
                    name: "sg-b300-06".into(),
                    url: "http://10.0.0.6:30000".into(),
                },
            ],
            known_workers: [
                "sg-b300-05",
                "sg-b300-06",
                "h20-r0",
                "example-worker-01",
            ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            known_pools: ["glm52-main", "internal-low"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            poll_interval_secs: DEFAULT_POLL_INTERVAL_SECS,
        }
    }

    fn specs(cfg: &RuntimeLeaseDiscoveryConfig) -> Vec<NamedWorkerSpec> {
        build_named_worker_specs(
            cfg,
            vec![WorkerBearerKeyConfig {
                worker_url: "http://10.0.0.5:30000".into(),
                bearer_token: "test-only-token".into(),
            }],
        )
        .unwrap()
    }

    fn document(generation: u64, leases: &str) -> String {
        format!(
            r#"{{
              "schema":"macaron.worker_runtime.v1",
              "generation":{generation},
              "updated_at":"2026-07-11T12:00:00Z",
              "leases":{leases}
            }}"#
        )
    }

    fn lease(pools: &str, expires_at: &str) -> String {
        format!(
            r#"{{
              "lease_id":"11111111-1111-4111-8111-111111111111",
              "state":"drained",
              "pools":{pools},
              "watchdog_mode":"monitor_only",
              "keep_metrics":true,
              "owner":"test-owner",
              "reason":"test drain",
              "created_at":"2020-01-01T00:00:00Z",
              "expires_at":"{expires_at}"
            }}"#
        )
    }

    #[test]
    fn parses_authoritative_sample_fixture() {
        let (document, drained, _, stale) = parse_and_validate(SAMPLE, &cfg()).unwrap();
        assert_eq!(document.generation, 7);
        assert!(drained.is_empty());
        assert_eq!(stale, 0);
    }

    #[test]
    fn strict_schema_rejects_unknown_fields_and_invalid_fixed_values() {
        let unknown_top =
            document(1, "{}").replace("\"leases\":{}", "\"leases\":{},\"unexpected\":true");
        assert!(parse_and_validate(&unknown_top, &cfg()).is_err());

        let unknown_lease = lease("[\"*\"]", "2030-01-01T00:00:00Z")
            .replace("\"state\":\"drained\"", "\"state\":\"drained\",\"extra\":1");
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{unknown_lease}}}"#)),
            &cfg()
        )
        .is_err());

        let bad_metrics = lease("[\"*\"]", "2030-01-01T00:00:00Z")
            .replace("\"keep_metrics\":true", "\"keep_metrics\":false");
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{bad_metrics}}}"#)),
            &cfg()
        )
        .is_err());

        let bad_uuid = lease("[\"*\"]", "2030-01-01T00:00:00Z")
            .replace("11111111-1111-4111-8111-111111111111", "not-a-uuid");
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{bad_uuid}}}"#)),
            &cfg()
        )
        .is_err());

        let noncanonical_uuid = lease("[\"*\"]", "2030-01-01T00:00:00Z").replace(
            "11111111-1111-4111-8111-111111111111",
            "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        );
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{noncanonical_uuid}}}"#)),
            &cfg()
        )
        .is_err());

        let offset_time = lease("[\"*\"]", "2030-01-01T08:00:00+08:00");
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{offset_time}}}"#)),
            &cfg()
        )
        .is_err());

        let excessive_precision = lease("[\"*\"]", "2030-01-01T00:00:00.1234567Z");
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{excessive_precision}}}"#)),
            &cfg()
        )
        .is_err());

        let untrimmed_owner = lease("[\"*\"]", "2030-01-01T00:00:00Z")
            .replace("\"owner\":\"test-owner\"", "\"owner\":\" test-owner\"");
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{untrimmed_owner}}}"#)),
            &cfg()
        )
        .is_err());

        let null_etag = lease("[\"*\"]", "2030-01-01T00:00:00Z").replace(
            "\"expires_at\":\"2030-01-01T00:00:00Z\"",
            "\"expires_at\":\"2030-01-01T00:00:00Z\",\"baseline_registry_etag\":null",
        );
        assert!(parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{null_etag}}}"#)),
            &cfg()
        )
        .is_err());
    }

    #[test]
    fn accepts_all_watchdog_modes() {
        for mode in ["active", "monitor_only", "disabled"] {
            let value = lease("[\"internal-low\"]", "2030-01-01T00:00:00Z").replace(
                "\"watchdog_mode\":\"monitor_only\"",
                &format!("\"watchdog_mode\":\"{mode}\""),
            );
            parse_and_validate(
                &document(1, &format!(r#"{{"sg-b300-05":{value}}}"#)),
                &cfg(),
            )
            .unwrap();
        }
    }

    #[test]
    fn schema_lengths_count_unicode_codepoints() {
        assert!(require_trimmed_bounded("owner", &"测".repeat(128), 128).is_ok());
        assert!(require_trimmed_bounded("owner", &"测".repeat(129), 128).is_err());
        assert!(require_trimmed_bounded("reason", &"测".repeat(500), 500).is_ok());
        assert!(require_trimmed_bounded("reason", &"测".repeat(501), 500).is_err());
        assert!(require_trimmed_bounded("reason", "line one\nline two", 500).is_err());
        assert!(require_trimmed_bounded("owner", "owner\rname", 128).is_err());
        assert!(require_bounded_nonempty("etag", &"测".repeat(512), 512).is_ok());
        assert!(require_bounded_nonempty("etag", &"测".repeat(513), 512).is_err());
        assert!(require_bounded_nonempty("etag", "  ", 512).is_err());
    }

    #[test]
    fn generation_uses_full_u64_bounds() {
        let cfg = cfg();
        let workers = specs(&cfg);
        let state = AppliedRuntimeState::default();
        let proposal = state
            .propose(&document(u64::MAX, "{}"), None, &cfg, &workers)
            .unwrap();
        assert_eq!(proposal.document.generation, u64::MAX);

        let overflow =
            document(0, "{}").replace("\"generation\":0", "\"generation\":18446744073709551616");
        assert!(state.propose(&overflow, None, &cfg, &workers).is_err());
    }

    #[test]
    fn expired_lease_still_drains() {
        let value = lease("[\"glm52-main\"]", "2021-01-01T00:00:00Z");
        let (_, drained, expired, _) = parse_and_validate(
            &document(1, &format!(r#"{{"sg-b300-05":{value}}}"#)),
            &cfg(),
        )
        .unwrap();
        assert!(drained.contains("sg-b300-05"));
        assert_eq!(expired, 1);
    }

    #[test]
    fn stale_pool_is_ignored_while_other_valid_lease_still_applies() {
        let stale = lease("[\"retired-pool\"]", "2030-01-01T00:00:00Z");
        let current = lease("[\"glm52-main\"]", "2030-01-01T00:00:00Z").replace(
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        );
        let (_, drained, _, stale_pool_references) = parse_and_validate(
            &document(
                1,
                &format!(r#"{{"sg-b300-05":{stale},"sg-b300-06":{current}}}"#),
            ),
            &cfg(),
        )
        .unwrap();

        assert_eq!(drained, BTreeSet::from(["sg-b300-06".into()]));
        assert_eq!(stale_pool_references, 1);
    }

    #[test]
    fn malformed_unknown_and_generation_regression_keep_last_known_good() {
        let cfg = cfg();
        let workers = specs(&cfg);
        let mut state = AppliedRuntimeState::default();
        let value = lease("[\"*\"]", "2030-01-01T00:00:00Z");
        let first = state
            .propose(
                &document(10, &format!(r#"{{"sg-b300-05":{value}}}"#)),
                Some("etag-10".into()),
                &cfg,
                &workers,
            )
            .unwrap();
        state.commit(first);

        assert!(state.propose("{bad", None, &cfg, &workers).is_err());
        let unknown = document(11, &format!(r#"{{"unknown-worker":{value}}}"#));
        assert!(state.propose(&unknown, None, &cfg, &workers).is_err());
        assert!(state
            .propose(&document(9, "{}"), None, &cfg, &workers)
            .is_err());
        assert!(state
            .propose(&document(10, "{}"), None, &cfg, &workers)
            .is_err());
        assert_eq!(state.generation, Some(10));
        assert_eq!(state.accepted_etag.as_deref(), Some("etag-10"));
        assert_eq!(state.drained, BTreeSet::from(["sg-b300-05".into()]));
    }

    #[test]
    fn initial_invalid_document_is_fail_closed_before_base_fanout() {
        let cfg = cfg();
        let workers = specs(&cfg);
        assert!(accept_initial_document(
            AppConfigValue {
                value: "{malformed".into(),
                etag: Some("invalid-etag".into()),
            },
            &cfg,
            &workers,
        )
        .is_err());

        let a = lease("[\"*\"]", "2030-01-01T00:00:00Z");
        let b = lease("[\"*\"]", "2030-01-01T00:00:00Z");
        assert!(accept_initial_document(
            AppConfigValue {
                value: document(1, &format!(r#"{{"sg-b300-05":{a},"sg-b300-06":{b}}}"#),),
                etag: Some("empty-pool-etag".into()),
            },
            &cfg,
            &workers,
        )
        .is_err());
    }

    #[test]
    fn rejects_empty_effective_pool_without_mutating_state() {
        let cfg = cfg();
        let workers = specs(&cfg);
        let state = AppliedRuntimeState::default();
        let a = lease("[\"*\"]", "2030-01-01T00:00:00Z");
        let b = lease("[\"*\"]", "2030-01-01T00:00:00Z");
        let raw = document(1, &format!(r#"{{"sg-b300-05":{a},"sg-b300-06":{b}}}"#));
        assert!(state.propose(&raw, None, &cfg, &workers).is_err());
        assert_eq!(state.generation, None);
        assert!(state.drained.is_empty());
    }

    #[test]
    fn removal_precedes_addition_and_rejoin_preserves_worker_spec() {
        let cfg = cfg();
        let workers = specs(&cfg);
        let mut state = AppliedRuntimeState::default();
        let a = lease("[\"*\"]", "2030-01-01T00:00:00Z");
        let first = state
            .propose(
                &document(1, &format!(r#"{{"sg-b300-05":{a}}}"#)),
                None,
                &cfg,
                &workers,
            )
            .unwrap();
        assert!(matches!(first.events[0], DiscoveryEvent::Removed { .. }));
        state.commit(first);

        let b = lease("[\"*\"]", "2030-01-01T00:00:00Z").replace(
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
        );
        let second = state
            .propose(
                &document(2, &format!(r#"{{"sg-b300-06":{b}}}"#)),
                None,
                &cfg,
                &workers,
            )
            .unwrap();
        assert!(matches!(second.events[0], DiscoveryEvent::Removed { .. }));
        let DiscoveryEvent::Added(rejoined) = &second.events[1] else {
            panic!("addition must follow all removals")
        };
        assert_eq!(rejoined.id.0, "http://10.0.0.5:30000");
        assert_eq!(rejoined.min_priority, Some(100));
        assert_eq!(rejoined.max_context_tokens, Some(500_000));
        assert_eq!(rejoined.tier, WorkerTier::Shared);
        assert_eq!(rejoined.backend, WorkerBackend::Sglang);
        assert_eq!(
            rejoined.routes,
            WorkerRouteSet {
                chat: true,
                completions: false,
                messages: false,
                responses: true,
            }
        );
        assert_eq!(rejoined.bearer_token.as_deref(), Some("test-only-token"));
        state.commit(second);

        let release = state
            .propose(&document(3, "{}"), None, &cfg, &workers)
            .unwrap();
        assert!(matches!(
            release.events.as_slice(),
            [DiscoveryEvent::Added(_)]
        ));
    }
}
