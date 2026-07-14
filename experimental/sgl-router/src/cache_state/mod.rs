// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Distributed cache-state service and client primitives.
//!
//! This module is intentionally small and HTTP-shaped for the first
//! production trial: the same router binary can run as a standalone
//! in-memory cache-state service, while gateway mode can query it as an
//! optional optimization. Query failures are surfaced as `None`, and feed
//! failures as `false`, so routing never fails requests because cache-state is
//! unavailable.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache_event_stream::KvEventStreamRecord;
use crate::policies::kv_events::tree::{HashTree, KvWorkerId};
use crate::policies::kv_events::wire::{
    decode_event_batch, CacheStateReconciliationRecord, CacheStateSnapshotEntry,
    CacheStateSnapshotManifest, DecodeError, KvCacheEvent, KvEventBatch, MAX_SNAPSHOT_CHUNKS,
    MAX_SNAPSHOT_ENTRIES,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStateWorkerMatch {
    pub worker_url: String,
    pub dp_rank: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStateMatchRequest {
    pub model_id: String,
    pub block_hashes: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStateMatchResponse {
    pub matched_blocks: usize,
    pub workers: Vec<CacheStateWorkerMatch>,
    #[serde(default)]
    pub authoritative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStateInsertRequest {
    pub model_id: String,
    pub worker_url: String,
    #[serde(default)]
    pub dp_rank: u32,
    #[serde(default)]
    pub parent_hash: Option<i64>,
    pub block_hashes: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheStateKvEventsRequest {
    pub model_id: String,
    pub worker_url: String,
    #[serde(default)]
    pub dp_rank: u32,
    pub seq: i64,
    /// Raw SGLang msgpack `EventBatch` payload from the ZMQ frame, base64-encoded.
    pub payload_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStateKvEventsResponse {
    pub applied_events: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStateStreamApplyOutcome {
    pub response: CacheStateKvEventsResponse,
    pub record_kind: &'static str,
}

#[derive(Debug, Clone)]
pub struct CacheStateService {
    tree: Arc<HashTree>,
    reconciliation_config: CacheStateReconciliationConfig,
    reconciliation: Arc<Mutex<ReconciliationStore>>,
    reconciliation_metrics: Arc<ReconciliationMetrics>,
}

#[derive(Debug, Clone)]
pub struct CacheStateReconciliationConfig {
    pub enabled: bool,
    pub max_worker_ranks: usize,
    pub max_snapshot_entries: usize,
    pub max_in_progress_snapshot_entries: usize,
    pub max_snapshot_chunks: usize,
    pub dedupe_window: usize,
}

impl Default for CacheStateReconciliationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_worker_ranks: 4096,
            max_snapshot_entries: MAX_SNAPSHOT_ENTRIES,
            max_in_progress_snapshot_entries: MAX_SNAPSHOT_ENTRIES * 2,
            max_snapshot_chunks: MAX_SNAPSHOT_CHUNKS,
            dedupe_window: 8192,
        }
    }
}

impl CacheStateReconciliationConfig {
    pub fn from_env() -> Result<Self, String> {
        let mut config = Self::default();
        config.enabled = env_bool("CACHE_STATE_RECONCILIATION_ENABLED")?.unwrap_or(false);
        config.max_worker_ranks = env_usize("CACHE_STATE_RECONCILIATION_MAX_WORKER_RANKS")?
            .unwrap_or(config.max_worker_ranks);
        config.max_snapshot_entries = env_usize("CACHE_STATE_RECONCILIATION_MAX_SNAPSHOT_ENTRIES")?
            .unwrap_or(config.max_snapshot_entries);
        config.max_in_progress_snapshot_entries =
            env_usize("CACHE_STATE_RECONCILIATION_MAX_IN_PROGRESS_SNAPSHOT_ENTRIES")?
                .unwrap_or(config.max_in_progress_snapshot_entries);
        config.max_snapshot_chunks = env_usize("CACHE_STATE_RECONCILIATION_MAX_SNAPSHOT_CHUNKS")?
            .unwrap_or(config.max_snapshot_chunks);
        config.dedupe_window =
            env_usize("CACHE_STATE_RECONCILIATION_DEDUPE_WINDOW")?.unwrap_or(config.dedupe_window);
        if config.max_worker_ranks == 0
            || config.max_snapshot_entries == 0
            || config.max_in_progress_snapshot_entries == 0
            || config.max_snapshot_chunks == 0
            || config.dedupe_window == 0
        {
            return Err("cache-state reconciliation limits must be greater than zero".into());
        }
        if config.max_snapshot_entries > MAX_SNAPSHOT_ENTRIES
            || config.max_snapshot_chunks > MAX_SNAPSHOT_CHUNKS
        {
            return Err("cache-state reconciliation limits exceed wire protocol caps".into());
        }
        Ok(config)
    }
}

type CacheEntryKey = (Option<i64>, i64);

#[derive(Debug, Default)]
struct ReconciliationStore {
    workers: HashMap<KvWorkerId, WorkerReconciliationState>,
}

#[derive(Debug)]
struct WorkerReconciliationState {
    epoch: String,
    last_seq: Option<i64>,
    recent_payloads: HashMap<i64, String>,
    recent_order: VecDeque<i64>,
    media_by_entry: HashMap<CacheEntryKey, HashSet<String>>,
    digest: [u8; 32],
    trusted: bool,
    untrusted_since: Option<Instant>,
    snapshot: Option<SnapshotAssembly>,
}

impl WorkerReconciliationState {
    fn new(epoch: String) -> Self {
        Self {
            epoch,
            last_seq: None,
            recent_payloads: HashMap::new(),
            recent_order: VecDeque::new(),
            media_by_entry: HashMap::new(),
            digest: [0; 32],
            trusted: false,
            untrusted_since: Some(Instant::now()),
            snapshot: None,
        }
    }

    fn mark_untrusted(&mut self) {
        if self.trusted || self.untrusted_since.is_none() {
            self.untrusted_since = Some(Instant::now());
        }
        self.trusted = false;
    }

    fn mark_trusted(&mut self) {
        self.trusted = true;
        self.untrusted_since = None;
    }

    fn remember(&mut self, seq: i64, payload_hash: String, max_entries: usize) {
        self.recent_payloads.insert(seq, payload_hash);
        self.recent_order.push_back(seq);
        while self.recent_order.len() > max_entries {
            if let Some(old) = self.recent_order.pop_front() {
                self.recent_payloads.remove(&old);
            }
        }
    }
}

#[derive(Debug)]
struct SnapshotAssembly {
    manifest: CacheStateSnapshotManifest,
    next_chunk: usize,
    entries: Vec<CacheStateSnapshotEntry>,
    entry_keys: HashSet<CacheEntryKey>,
    started_at: Instant,
}

#[derive(Debug, Default)]
struct ReconciliationMetrics {
    sequence_gaps: AtomicU64,
    conflicting_duplicates: AtomicU64,
    digest_matches: AtomicU64,
    digest_mismatches: AtomicU64,
    snapshots_started: AtomicU64,
    snapshots_succeeded: AtomicU64,
    snapshots_failed: AtomicU64,
    reconciliation_duration_micros: AtomicU64,
    reconciliation_duration_count: AtomicU64,
    stream_apply_commit_success_micros: AtomicU64,
    stream_apply_commit_success_count: AtomicU64,
    stream_apply_commit_failure_micros: AtomicU64,
    stream_apply_commit_failure_count: AtomicU64,
}

#[derive(Debug, Clone)]
struct CacheStateRouterState {
    service: Arc<CacheStateService>,
    api_token: Option<Arc<str>>,
}

impl CacheStateService {
    pub fn new(tree: Arc<HashTree>) -> Self {
        Self::new_with_reconciliation(tree, CacheStateReconciliationConfig::default())
    }

    pub fn new_with_reconciliation(
        tree: Arc<HashTree>,
        reconciliation_config: CacheStateReconciliationConfig,
    ) -> Self {
        Self {
            tree,
            reconciliation_config,
            reconciliation: Arc::new(Mutex::new(ReconciliationStore::default())),
            reconciliation_metrics: Arc::new(ReconciliationMetrics::default()),
        }
    }

    pub fn with_empty_tree() -> Self {
        Self::new(Arc::new(HashTree::new()))
    }

    pub fn match_prefix(&self, req: &CacheStateMatchRequest) -> CacheStateMatchResponse {
        let matched = if self.reconciliation_config.enabled {
            let trusted: HashSet<_> = self
                .reconciliation
                .lock()
                .workers
                .iter()
                .filter(|(_, state)| state.trusted)
                .map(|(worker, _)| worker.clone())
                .collect();
            self.tree
                .match_prefix_for_workers(None, &req.block_hashes, &trusted)
        } else {
            self.tree.match_prefix(None, &req.block_hashes)
        };
        CacheStateMatchResponse {
            matched_blocks: matched.matched_blocks,
            workers: sorted_workers(matched.workers),
            authoritative: self.reconciliation_config.enabled,
        }
    }

    pub fn insert(&self, req: &CacheStateInsertRequest) {
        let worker = KvWorkerId::new(req.worker_url.clone(), req.dp_rank);
        self.tree
            .insert(&worker, req.parent_hash, req.block_hashes.as_slice());
    }

    pub fn apply_kv_events(
        &self,
        req: &CacheStateKvEventsRequest,
    ) -> Result<CacheStateKvEventsResponse, CacheStateError> {
        self.apply_kv_payload(&req.worker_url, req.dp_rank, req.seq, &req.payload_b64)
            .map(|outcome| outcome.response)
    }

    pub fn apply_stream_record(
        &self,
        record: &KvEventStreamRecord,
    ) -> Result<CacheStateStreamApplyOutcome, CacheStateError> {
        self.apply_kv_payload(
            &record.worker_url,
            record.dp_rank,
            record.seq,
            &record.payload_b64,
        )
    }

    fn apply_kv_payload(
        &self,
        worker_url: &str,
        dp_rank: u32,
        seq: i64,
        payload_b64: &str,
    ) -> Result<CacheStateStreamApplyOutcome, CacheStateError> {
        let worker = KvWorkerId::new(worker_url.to_string(), dp_rank);
        let payload = match decode_base64(payload_b64) {
            Ok(payload) => payload,
            Err(err) => {
                self.mark_worker_untrusted(&worker);
                return Err(CacheStateError::BadBase64(err));
            }
        };
        let batch = match decode_event_batch(&payload) {
            Ok(batch) => batch,
            Err(err) => {
                self.mark_worker_untrusted(&worker);
                return Err(CacheStateError::BadMsgpack(err));
            }
        };
        let record_kind = cache_state_record_kind(&batch);
        let response = if self.reconciliation_config.enabled && batch.publisher_epoch.is_some() {
            let payload_hash = sha256_hex(&payload);
            self.apply_reconciled_batch(&worker, seq, payload_hash, &batch)?
        } else {
            CacheStateKvEventsResponse {
                applied_events: self.apply_legacy_events(&worker, &batch),
            }
        };

        Ok(CacheStateStreamApplyOutcome {
            response,
            record_kind,
        })
    }

    fn apply_legacy_events(&self, worker: &KvWorkerId, batch: &KvEventBatch) -> usize {
        let mut applied_events = 0usize;
        for event in &batch.events {
            match event {
                KvCacheEvent::BlockStored(block) => {
                    self.tree
                        .insert(worker, block.parent_block_hash, &block.block_hashes);
                    applied_events += 1;
                }
                KvCacheEvent::BlockRemoved(block) => {
                    self.tree.remove(worker, &block.block_hashes);
                    applied_events += 1;
                }
                KvCacheEvent::AllBlocksCleared => {
                    self.tree.clear_worker(worker);
                    applied_events += 1;
                }
            }
        }
        applied_events
    }

    fn apply_reconciled_batch(
        &self,
        worker: &KvWorkerId,
        seq: i64,
        payload_hash: String,
        batch: &KvEventBatch,
    ) -> Result<CacheStateKvEventsResponse, CacheStateError> {
        if seq < 0 {
            self.mark_worker_untrusted(worker);
            return Err(CacheStateError::Reconciliation(
                "transport sequence must be non-negative".into(),
            ));
        }
        let epoch = batch
            .publisher_epoch
            .as_ref()
            .expect("caller checked publisher_epoch")
            .clone();
        let mut store = self.reconciliation.lock();
        if !store.workers.contains_key(worker)
            && store.workers.len() >= self.reconciliation_config.max_worker_ranks
        {
            return Err(CacheStateError::Reconciliation(format!(
                "worker-rank state limit {} reached",
                self.reconciliation_config.max_worker_ranks
            )));
        }

        let epoch_changed = store
            .workers
            .get(worker)
            .is_none_or(|state| state.epoch != epoch);
        if epoch_changed {
            self.tree.clear_worker(worker);
            store
                .workers
                .insert(worker.clone(), WorkerReconciliationState::new(epoch));
        }

        let in_progress_other = store
            .workers
            .iter()
            .filter(|(candidate, _)| *candidate != worker)
            .filter_map(|(_, state)| {
                state
                    .snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.manifest.block_count)
            })
            .fold(0usize, usize::saturating_add);
        let state = store
            .workers
            .get_mut(worker)
            .expect("worker state inserted above");

        if let Some(previous_hash) = state.recent_payloads.get(&seq) {
            if previous_hash == &payload_hash {
                return Ok(CacheStateKvEventsResponse { applied_events: 0 });
            }
            state.mark_untrusted();
            self.reconciliation_metrics
                .conflicting_duplicates
                .fetch_add(1, Ordering::Relaxed);
            return Err(CacheStateError::Reconciliation(format!(
                "conflicting duplicate at sequence {seq}"
            )));
        }
        if let Some(last_seq) = state.last_seq {
            if seq <= last_seq {
                state.mark_untrusted();
                self.reconciliation_metrics
                    .sequence_gaps
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(CacheStateKvEventsResponse { applied_events: 0 });
            }
            if seq != last_seq.saturating_add(1) {
                state.mark_untrusted();
                self.reconciliation_metrics
                    .sequence_gaps
                    .fetch_add(1, Ordering::Relaxed);
            }
        } else if seq != 0 {
            state.mark_untrusted();
            self.reconciliation_metrics
                .sequence_gaps
                .fetch_add(1, Ordering::Relaxed);
        }

        if batch.reconciliation.is_some() && !batch.events.is_empty() {
            state.snapshot = None;
            state.mark_untrusted();
            return Err(CacheStateError::Reconciliation(
                "control batches must not contain mutation events".into(),
            ));
        }

        let applied_events = match batch.reconciliation.as_ref() {
            Some(control) => {
                let context = ReconciliationApplyContext {
                    tree: &self.tree,
                    config: &self.reconciliation_config,
                    metrics: &self.reconciliation_metrics,
                    worker,
                    seq,
                    in_progress_other,
                };
                let result = apply_reconciliation_control(context, state, control);
                match result {
                    Ok(snapshot_duration) => {
                        if matches!(control, CacheStateReconciliationRecord::SnapshotStart(_)) {
                            self.reconciliation_metrics
                                .snapshots_started
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        if let Some(duration) = snapshot_duration {
                            self.reconciliation_metrics
                                .snapshots_succeeded
                                .fetch_add(1, Ordering::Relaxed);
                            self.reconciliation_metrics
                                .reconciliation_duration_micros
                                .fetch_add(
                                    duration.as_micros().min(u128::from(u64::MAX)) as u64,
                                    Ordering::Relaxed,
                                );
                            self.reconciliation_metrics
                                .reconciliation_duration_count
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(err) => {
                        if matches!(
                            control,
                            CacheStateReconciliationRecord::SnapshotStart(_)
                                | CacheStateReconciliationRecord::SnapshotChunk(_)
                                | CacheStateReconciliationRecord::SnapshotEnd(_)
                        ) {
                            self.reconciliation_metrics
                                .snapshots_failed
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        state.snapshot = None;
                        state.mark_untrusted();
                        return Err(CacheStateError::Reconciliation(err));
                    }
                }
                0
            }
            None => {
                if state.snapshot.take().is_some() {
                    state.mark_untrusted();
                }
                apply_reconciled_mutations(&self.tree, worker, state, &batch.events)
            }
        };
        state.last_seq = Some(seq);
        state.remember(seq, payload_hash, self.reconciliation_config.dedupe_window);
        Ok(CacheStateKvEventsResponse { applied_events })
    }

    fn mark_worker_untrusted(&self, worker: &KvWorkerId) {
        if !self.reconciliation_config.enabled {
            return;
        }
        if let Some(state) = self.reconciliation.lock().workers.get_mut(worker) {
            state.mark_untrusted();
        }
    }

    pub fn record_stream_apply_commit(&self, duration: Duration, success: bool) {
        let duration_micros = duration.as_micros().min(u128::from(u64::MAX)) as u64;
        let (sum, count) = if success {
            (
                &self
                    .reconciliation_metrics
                    .stream_apply_commit_success_micros,
                &self
                    .reconciliation_metrics
                    .stream_apply_commit_success_count,
            )
        } else {
            (
                &self
                    .reconciliation_metrics
                    .stream_apply_commit_failure_micros,
                &self
                    .reconciliation_metrics
                    .stream_apply_commit_failure_count,
            )
        };
        sum.fetch_add(duration_micros, Ordering::Relaxed);
        count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reconciliation_metrics_text(&self) -> String {
        let store = self.reconciliation.lock();
        let trusted = store.workers.values().filter(|state| state.trusted).count();
        let untrusted = store.workers.len().saturating_sub(trusted);
        let metrics = &self.reconciliation_metrics;
        format!(
            concat!(
                "# TYPE sgl_router_cache_state_sequence_gaps_total counter\n",
                "sgl_router_cache_state_sequence_gaps_total {}\n",
                "# TYPE sgl_router_cache_state_conflicting_duplicates_total counter\n",
                "sgl_router_cache_state_conflicting_duplicates_total {}\n",
                "# TYPE sgl_router_cache_state_digest_comparisons_total counter\n",
                "sgl_router_cache_state_digest_comparisons_total{{outcome=\"match\"}} {}\n",
                "sgl_router_cache_state_digest_comparisons_total{{outcome=\"mismatch\"}} {}\n",
                "# TYPE sgl_router_cache_state_snapshots_total counter\n",
                "sgl_router_cache_state_snapshots_total{{outcome=\"started\"}} {}\n",
                "sgl_router_cache_state_snapshots_total{{outcome=\"success\"}} {}\n",
                "sgl_router_cache_state_snapshots_total{{outcome=\"failure\"}} {}\n",
                "# TYPE sgl_router_cache_state_reconciliation_duration_seconds summary\n",
                "sgl_router_cache_state_reconciliation_duration_seconds_sum {:.6}\n",
                "sgl_router_cache_state_reconciliation_duration_seconds_count {}\n",
                "# TYPE sgl_router_cache_state_stream_apply_commit_duration_seconds summary\n",
                "sgl_router_cache_state_stream_apply_commit_duration_seconds_sum{{outcome=\"success\"}} {:.6}\n",
                "sgl_router_cache_state_stream_apply_commit_duration_seconds_count{{outcome=\"success\"}} {}\n",
                "sgl_router_cache_state_stream_apply_commit_duration_seconds_sum{{outcome=\"failure\"}} {:.6}\n",
                "sgl_router_cache_state_stream_apply_commit_duration_seconds_count{{outcome=\"failure\"}} {}\n",
                "# TYPE sgl_router_cache_state_trusted_worker_ranks gauge\n",
                "sgl_router_cache_state_trusted_worker_ranks {}\n",
                "# TYPE sgl_router_cache_state_untrusted_worker_ranks gauge\n",
                "sgl_router_cache_state_untrusted_worker_ranks {}\n",
            ),
            metrics.sequence_gaps.load(Ordering::Relaxed),
            metrics.conflicting_duplicates.load(Ordering::Relaxed),
            metrics.digest_matches.load(Ordering::Relaxed),
            metrics.digest_mismatches.load(Ordering::Relaxed),
            metrics.snapshots_started.load(Ordering::Relaxed),
            metrics.snapshots_succeeded.load(Ordering::Relaxed),
            metrics.snapshots_failed.load(Ordering::Relaxed),
            metrics
                .reconciliation_duration_micros
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            metrics
                .reconciliation_duration_count
                .load(Ordering::Relaxed),
            metrics
                .stream_apply_commit_success_micros
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            metrics
                .stream_apply_commit_success_count
                .load(Ordering::Relaxed),
            metrics
                .stream_apply_commit_failure_micros
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            metrics
                .stream_apply_commit_failure_count
                .load(Ordering::Relaxed),
            trusted,
            untrusted,
        )
    }

    pub fn apply_stream_records(
        &self,
        records: &[KvEventStreamRecord],
    ) -> Result<CacheStateKvEventsResponse, CacheStateError> {
        let mut seen = RecentDedupe::new(8192);
        let mut applied_events = 0usize;
        for record in records {
            if !seen.insert(record.dedupe_key()) {
                continue;
            }
            let outcome = self.apply_stream_record(record)?;
            applied_events += outcome.response.applied_events;
        }
        Ok(CacheStateKvEventsResponse { applied_events })
    }

    pub fn router(self: Arc<Self>) -> Router {
        self.router_with_api_token(None)
    }

    pub fn router_with_api_token(self: Arc<Self>, api_token: Option<String>) -> Router {
        let state = CacheStateRouterState {
            service: self,
            api_token: api_token.map(Arc::from),
        };
        Router::new()
            .route("/healthz", get(healthz))
            .route("/metrics", get(reconciliation_metrics))
            .route("/v1/cache_state/match_prefix", post(match_prefix))
            .route("/v1/cache_state/insert", post(insert_prefix))
            .route("/v1/cache_state/kv_events", post(kv_events))
            .with_state(state)
    }
}

fn cache_state_record_kind(batch: &KvEventBatch) -> &'static str {
    match batch.reconciliation.as_ref() {
        Some(CacheStateReconciliationRecord::Digest(_)) => "digest",
        Some(CacheStateReconciliationRecord::SnapshotStart(_)) => "snapshot_start",
        Some(CacheStateReconciliationRecord::SnapshotChunk(_)) => "snapshot_chunk",
        Some(CacheStateReconciliationRecord::SnapshotEnd(_)) => "snapshot_end",
        None => "mutations",
    }
}

fn apply_reconciled_mutations(
    tree: &HashTree,
    worker: &KvWorkerId,
    state: &mut WorkerReconciliationState,
    events: &[KvCacheEvent],
) -> usize {
    let mut applied_events = 0usize;
    for event in events {
        match event {
            KvCacheEvent::BlockStored(block) => {
                let medium = block.medium.as_deref().unwrap_or("GPU").to_string();
                let mut parent_hash = block.parent_block_hash;
                for &block_hash in &block.block_hashes {
                    let key = (parent_hash, block_hash);
                    let is_new = !state.media_by_entry.contains_key(&key);
                    state
                        .media_by_entry
                        .entry(key)
                        .or_default()
                        .insert(medium.clone());
                    if is_new {
                        xor_entry_digest(&mut state.digest, key);
                        tree.insert(worker, parent_hash, &[block_hash]);
                    }
                    parent_hash = Some(block_hash);
                }
                applied_events += 1;
            }
            KvCacheEvent::BlockRemoved(block) => {
                let medium = block.medium.as_deref().unwrap_or("GPU");
                let hashes: HashSet<_> = block.block_hashes.iter().copied().collect();
                let targets: Vec<_> = state
                    .media_by_entry
                    .keys()
                    .filter(|(_, block_hash)| hashes.contains(block_hash))
                    .copied()
                    .collect();
                for key in targets {
                    let remove_entry = state.media_by_entry.get_mut(&key).is_some_and(|media| {
                        media.remove(medium);
                        media.is_empty()
                    });
                    if remove_entry {
                        state.media_by_entry.remove(&key);
                        xor_entry_digest(&mut state.digest, key);
                        tree.remove_entry(worker, key.0, key.1);
                    }
                }
                applied_events += 1;
            }
            KvCacheEvent::AllBlocksCleared => {
                state.media_by_entry.clear();
                state.digest = [0; 32];
                tree.clear_worker(worker);
                applied_events += 1;
            }
        }
    }
    applied_events
}

struct ReconciliationApplyContext<'a> {
    tree: &'a HashTree,
    config: &'a CacheStateReconciliationConfig,
    metrics: &'a ReconciliationMetrics,
    worker: &'a KvWorkerId,
    seq: i64,
    in_progress_other: usize,
}

fn apply_reconciliation_control(
    context: ReconciliationApplyContext<'_>,
    state: &mut WorkerReconciliationState,
    control: &CacheStateReconciliationRecord,
) -> Result<Option<Duration>, String> {
    let ReconciliationApplyContext {
        tree,
        config,
        metrics,
        worker,
        seq,
        in_progress_other,
    } = context;
    match control {
        CacheStateReconciliationRecord::Digest(digest) => {
            state.snapshot = None;
            if digest.through_seq != seq {
                return Err(format!(
                    "digest watermark {} does not match transport sequence {seq}",
                    digest.through_seq
                ));
            }
            if digest.block_count == state.media_by_entry.len()
                && digest
                    .digest
                    .eq_ignore_ascii_case(&digest_hex(&state.digest))
            {
                metrics.digest_matches.fetch_add(1, Ordering::Relaxed);
                state.mark_trusted();
            } else {
                metrics.digest_mismatches.fetch_add(1, Ordering::Relaxed);
                state.mark_untrusted();
            }
            Ok(None)
        }
        CacheStateReconciliationRecord::SnapshotStart(manifest) => {
            if manifest.watermark != seq {
                return Err(format!(
                    "snapshot watermark {} does not match start sequence {seq}",
                    manifest.watermark
                ));
            }
            if manifest.block_count > config.max_snapshot_entries {
                return Err(format!(
                    "snapshot entry count {} exceeds configured cap {}",
                    manifest.block_count, config.max_snapshot_entries
                ));
            }
            if manifest.total_chunks > config.max_snapshot_chunks {
                return Err(format!(
                    "snapshot chunk count {} exceeds configured cap {}",
                    manifest.total_chunks, config.max_snapshot_chunks
                ));
            }
            if (manifest.block_count == 0) != (manifest.total_chunks == 0) {
                return Err(
                    "empty snapshots require zero chunks and non-empty snapshots require chunks"
                        .into(),
                );
            }
            if in_progress_other.saturating_add(manifest.block_count)
                > config.max_in_progress_snapshot_entries
            {
                return Err(format!(
                    "in-progress snapshot entries exceed global cap {}",
                    config.max_in_progress_snapshot_entries
                ));
            }
            state.mark_untrusted();
            state.snapshot = Some(SnapshotAssembly {
                manifest: manifest.clone(),
                next_chunk: 0,
                entries: Vec::new(),
                entry_keys: HashSet::new(),
                started_at: Instant::now(),
            });
            Ok(None)
        }
        CacheStateReconciliationRecord::SnapshotChunk(chunk) => {
            let Some(snapshot) = state.snapshot.as_mut() else {
                return Err("snapshot chunk arrived without an active snapshot".into());
            };
            if chunk.snapshot_id != snapshot.manifest.snapshot_id {
                return Err("snapshot chunk id does not match active snapshot".into());
            }
            if chunk.chunk_index != snapshot.next_chunk
                || chunk.chunk_index >= snapshot.manifest.total_chunks
            {
                return Err(format!(
                    "snapshot chunk index {} is not expected index {}",
                    chunk.chunk_index, snapshot.next_chunk
                ));
            }
            if snapshot.entries.len().saturating_add(chunk.entries.len())
                > snapshot.manifest.block_count
                || snapshot.entries.len().saturating_add(chunk.entries.len())
                    > config.max_snapshot_entries
            {
                return Err("snapshot chunk exceeds advertised or configured entry count".into());
            }
            for entry in &chunk.entries {
                let media: HashSet<_> = entry.media.iter().cloned().collect();
                if media.is_empty() || media.len() != entry.media.len() {
                    return Err("snapshot entry contains empty or duplicate media".into());
                }
                let key = (entry.parent_block_hash, entry.block_hash);
                if !snapshot.entry_keys.insert(key) {
                    return Err("snapshot contains a duplicate logical entry".into());
                }
                snapshot.entries.push(entry.clone());
            }
            snapshot.next_chunk += 1;
            Ok(None)
        }
        CacheStateReconciliationRecord::SnapshotEnd(manifest) => {
            let Some(snapshot) = state.snapshot.take() else {
                return Err("snapshot end arrived without an active snapshot".into());
            };
            if !snapshot_manifests_match(&snapshot.manifest, manifest) {
                return Err("snapshot end manifest does not match snapshot start".into());
            }
            if snapshot.next_chunk != snapshot.manifest.total_chunks
                || snapshot.entries.len() != snapshot.manifest.block_count
            {
                return Err("snapshot is incomplete".into());
            }
            let calculated = digest_for_entries(
                snapshot
                    .entries
                    .iter()
                    .map(|entry| (entry.parent_block_hash, entry.block_hash)),
            );
            if !manifest
                .digest
                .eq_ignore_ascii_case(&digest_hex(&calculated))
            {
                return Err("snapshot digest does not match reconstructed entries".into());
            }
            let tree_entries: Vec<_> = snapshot
                .entries
                .iter()
                .map(|entry| (entry.parent_block_hash, entry.block_hash))
                .collect();
            if !tree.replace_worker(worker, &tree_entries) {
                return Err("snapshot entries are cyclic or have unresolved parents".into());
            }
            state.media_by_entry = snapshot
                .entries
                .into_iter()
                .map(|entry| {
                    (
                        (entry.parent_block_hash, entry.block_hash),
                        entry.media.into_iter().collect(),
                    )
                })
                .collect();
            state.digest = calculated;
            let reconciliation_duration = snapshot.started_at.elapsed();
            state.mark_trusted();
            Ok(Some(reconciliation_duration))
        }
    }
}

fn snapshot_manifests_match(
    start: &CacheStateSnapshotManifest,
    end: &CacheStateSnapshotManifest,
) -> bool {
    start.snapshot_id == end.snapshot_id
        && start.watermark == end.watermark
        && start.total_chunks == end.total_chunks
        && start.block_count == end.block_count
        && start.algorithm == end.algorithm
        && start.digest.eq_ignore_ascii_case(&end.digest)
}

fn xor_entry_digest(digest: &mut [u8; 32], entry: CacheEntryKey) {
    let mut hasher = Sha256::new();
    hasher.update([u8::from(entry.0.is_some())]);
    hasher.update(entry.0.unwrap_or(0).to_be_bytes());
    hasher.update(entry.1.to_be_bytes());
    for (target, value) in digest.iter_mut().zip(hasher.finalize()) {
        *target ^= value;
    }
}

fn digest_for_entries(entries: impl IntoIterator<Item = CacheEntryKey>) -> [u8; 32] {
    let mut digest = [0; 32];
    for entry in entries {
        xor_entry_digest(&mut digest, entry);
    }
    digest
}

fn digest_hex(digest: &[u8]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn sha256_hex(payload: &[u8]) -> String {
    digest_hex(&Sha256::digest(payload))
}

struct RecentDedupe {
    max_entries: usize,
    order: VecDeque<String>,
    set: HashSet<String>,
}

impl RecentDedupe {
    fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            order: VecDeque::new(),
            set: HashSet::new(),
        }
    }

    fn insert(&mut self, key: String) -> bool {
        if self.set.contains(&key) {
            return false;
        }
        self.set.insert(key.clone());
        self.order.push_back(key);
        while self.order.len() > self.max_entries {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
        true
    }
}

fn sorted_workers(workers: HashSet<KvWorkerId>) -> Vec<CacheStateWorkerMatch> {
    let mut out: Vec<_> = workers
        .into_iter()
        .map(|w| CacheStateWorkerMatch {
            worker_url: w.url,
            dp_rank: w.dp_rank,
        })
        .collect();
    out.sort_by(|a, b| {
        a.worker_url
            .cmp(&b.worker_url)
            .then_with(|| a.dp_rank.cmp(&b.dp_rank))
    });
    out
}

async fn healthz() -> &'static str {
    "ok"
}

async fn reconciliation_metrics(State(state): State<CacheStateRouterState>) -> String {
    state.service.reconciliation_metrics_text()
}

async fn match_prefix(
    State(state): State<CacheStateRouterState>,
    headers: HeaderMap,
    Json(req): Json<CacheStateMatchRequest>,
) -> Result<Json<CacheStateMatchResponse>, CacheStateError> {
    require_auth(&state, &headers)?;
    Ok(Json(state.service.match_prefix(&req)))
}

async fn insert_prefix(
    State(state): State<CacheStateRouterState>,
    headers: HeaderMap,
    Json(req): Json<CacheStateInsertRequest>,
) -> Result<StatusCode, CacheStateError> {
    require_auth(&state, &headers)?;
    state.service.insert(&req);
    Ok(StatusCode::NO_CONTENT)
}

async fn kv_events(
    State(state): State<CacheStateRouterState>,
    headers: HeaderMap,
    Json(req): Json<CacheStateKvEventsRequest>,
) -> Result<Json<CacheStateKvEventsResponse>, CacheStateError> {
    require_auth(&state, &headers)?;
    tokio::task::spawn_blocking(move || state.service.apply_kv_events(&req))
        .await
        .map_err(|err| CacheStateError::Internal(format!("KV event apply task failed: {err}")))?
        .map(Json)
}

#[derive(Debug)]
pub enum CacheStateError {
    Unauthorized,
    BadBase64(String),
    BadMsgpack(DecodeError),
    Reconciliation(String),
    Internal(String),
}

impl IntoResponse for CacheStateError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized".to_string()),
            Self::BadBase64(err) => (
                StatusCode::BAD_REQUEST,
                format!("invalid payload_b64: {err}"),
            ),
            Self::BadMsgpack(err) => (
                StatusCode::BAD_REQUEST,
                format!("invalid KV event msgpack payload: {err}"),
            ),
            Self::Reconciliation(err) => (
                StatusCode::CONFLICT,
                format!("cache-state reconciliation rejected record: {err}"),
            ),
            Self::Internal(err) => (StatusCode::INTERNAL_SERVER_ERROR, err),
        };
        (status, message).into_response()
    }
}

fn require_auth(state: &CacheStateRouterState, headers: &HeaderMap) -> Result<(), CacheStateError> {
    let Some(expected) = state.api_token.as_ref() else {
        return Ok(());
    };
    let Some(actual) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
    else {
        return Err(CacheStateError::Unauthorized);
    };
    let Some(token) = actual.strip_prefix("Bearer ") else {
        return Err(CacheStateError::Unauthorized);
    };
    if token == expected.as_ref() {
        Ok(())
    } else {
        Err(CacheStateError::Unauthorized)
    }
}

fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err("length is not a multiple of 4".into());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut chunk = [0u8; 4];
    for raw in bytes.chunks_exact(4) {
        for (i, b) in raw.iter().copied().enumerate() {
            chunk[i] = match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => b - b'a' + 26,
                b'0'..=b'9' => b - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => 64,
                _ => return Err(format!("invalid byte 0x{b:02x}")),
            };
        }
        if chunk[0] == 64 || chunk[1] == 64 {
            return Err("padding in first two base64 positions".into());
        }
        out.push((chunk[0] << 2) | (chunk[1] >> 4));
        if chunk[2] != 64 {
            out.push((chunk[1] << 4) | (chunk[2] >> 2));
            if chunk[3] != 64 {
                out.push((chunk[2] << 6) | chunk[3]);
            }
        } else if chunk[3] != 64 {
            return Err("invalid single padding".into());
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct RemoteCacheStateClient {
    base_urls: Vec<String>,
    agent: ureq::Agent,
    api_token: Option<String>,
}

impl RemoteCacheStateClient {
    pub fn new(base_url: String, timeout: Duration) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(timeout).build();
        Self {
            base_urls: parse_cache_state_urls(&base_url),
            agent,
            api_token: std::env::var("CACHE_STATE_API_TOKEN")
                .ok()
                .filter(|s| !s.is_empty()),
        }
    }

    pub fn match_prefix(&self, req: &CacheStateMatchRequest) -> Option<CacheStateMatchResponse> {
        for base_url in &self.base_urls {
            let url = format!("{base_url}/v1/cache_state/match_prefix");
            if let Some(resp) = self
                .post(&url)
                .send_json(req)
                .ok()
                .and_then(|resp| resp.into_json::<CacheStateMatchResponse>().ok())
            {
                return Some(resp);
            }
        }
        None
    }

    pub fn insert(&self, req: &CacheStateInsertRequest) -> bool {
        let mut any_success = false;
        for base_url in &self.base_urls {
            let url = format!("{base_url}/v1/cache_state/insert");
            if self
                .post(&url)
                .send_json(req)
                .map(|resp| (200..300).contains(&resp.status()))
                .unwrap_or(false)
            {
                any_success = true;
            }
        }
        any_success
    }

    pub fn kv_events(&self, req: &CacheStateKvEventsRequest) -> bool {
        let mut any_success = false;
        for base_url in &self.base_urls {
            let url = format!("{base_url}/v1/cache_state/kv_events");
            if self
                .post(&url)
                .send_json(req)
                .map(|resp| (200..300).contains(&resp.status()))
                .unwrap_or(false)
            {
                any_success = true;
            }
        }
        any_success
    }

    fn post(&self, url: &str) -> ureq::Request {
        let req = self.agent.post(url);
        match self.api_token.as_ref() {
            Some(token) => req.set("Authorization", &format!("Bearer {token}")),
            None => req,
        }
    }
}

fn parse_cache_state_urls(raw: &str) -> Vec<String> {
    raw.split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
        .collect()
}

fn env_bool(name: &str) -> Result<Option<bool>, String> {
    let Some(raw) = std::env::var(name).ok().filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(format!("{name} must be a boolean, got {raw:?}")),
    }
}

fn env_usize(name: &str) -> Result<Option<usize>, String> {
    let Some(raw) = std::env::var(name).ok().filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    raw.parse::<usize>()
        .map(Some)
        .map_err(|err| format!("parse {name}={raw:?} as usize: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_match_returns_deepest_workers() {
        let service = CacheStateService::with_empty_tree();
        service.insert(&CacheStateInsertRequest {
            model_id: "m".into(),
            worker_url: "http://w0:30000".into(),
            dp_rank: 0,
            parent_hash: None,
            block_hashes: vec![1, 2, 3],
        });
        let matched = service.match_prefix(&CacheStateMatchRequest {
            model_id: "m".into(),
            block_hashes: vec![1, 2, 9],
        });
        assert_eq!(matched.matched_blocks, 2);
        assert_eq!(
            matched.workers,
            vec![CacheStateWorkerMatch {
                worker_url: "http://w0:30000".into(),
                dp_rank: 0,
            }]
        );
    }

    #[test]
    fn kv_events_ingest_updates_tree() {
        let service = CacheStateService::with_empty_tree();
        let payload = {
            let mut buf = Vec::new();
            rmp::encode::write_array_len(&mut buf, 3).unwrap();
            rmp::encode::write_f64(&mut buf, 1.0).unwrap();
            rmp::encode::write_array_len(&mut buf, 1).unwrap();
            rmp::encode::write_array_len(&mut buf, 7).unwrap();
            rmp::encode::write_str(&mut buf, "BlockStored").unwrap();
            rmp::encode::write_array_len(&mut buf, 2).unwrap();
            rmp::encode::write_sint(&mut buf, 10).unwrap();
            rmp::encode::write_sint(&mut buf, 20).unwrap();
            rmp::encode::write_nil(&mut buf).unwrap();
            rmp::encode::write_array_len(&mut buf, 0).unwrap();
            rmp::encode::write_uint(&mut buf, 64).unwrap();
            rmp::encode::write_nil(&mut buf).unwrap();
            rmp::encode::write_nil(&mut buf).unwrap();
            rmp::encode::write_uint(&mut buf, 1).unwrap();
            buf
        };
        let resp = service
            .apply_kv_events(&CacheStateKvEventsRequest {
                model_id: "m".into(),
                worker_url: "http://w0:30000".into(),
                dp_rank: 1,
                seq: 7,
                payload_b64: encode_base64_for_test(&payload),
            })
            .unwrap();
        assert_eq!(resp.applied_events, 1);

        let matched = service.match_prefix(&CacheStateMatchRequest {
            model_id: "m".into(),
            block_hashes: vec![10, 20, 30],
        });
        assert_eq!(matched.matched_blocks, 2);
        assert_eq!(
            matched.workers,
            vec![CacheStateWorkerMatch {
                worker_url: "http://w0:30000".into(),
                dp_rank: 1,
            }]
        );
    }

    #[test]
    fn remote_cache_state_client_parses_multiple_urls() {
        assert_eq!(
            parse_cache_state_urls(" https://a.example/ ,https://b.example/  https://c.example "),
            vec![
                "https://a.example".to_string(),
                "https://b.example".to_string(),
                "https://c.example".to_string(),
            ]
        );
    }

    #[test]
    fn stream_replay_converges_two_services() {
        let records = vec![KvEventStreamRecord {
            schema_version: 1,
            model_id: "m".into(),
            worker_url: "http://w0:30000".into(),
            dp_rank: 0,
            seq: 1,
            observed_at_ms: 1,
            payload_hash: "hash".into(),
            payload_b64: encode_base64_for_test(&block_stored_payload(&[10, 20])),
        }];
        let a = CacheStateService::with_empty_tree();
        let b = CacheStateService::with_empty_tree();

        a.apply_stream_records(&records).unwrap();
        b.apply_stream_records(&records).unwrap();

        let req = CacheStateMatchRequest {
            model_id: "m".into(),
            block_hashes: vec![10, 20, 30],
        };
        assert_eq!(a.match_prefix(&req), b.match_prefix(&req));
    }

    #[test]
    fn stream_replay_dedupes_duplicate_records() {
        let payload = block_stored_payload(&[10, 20]);
        let record = KvEventStreamRecord {
            schema_version: 1,
            model_id: "m".into(),
            worker_url: "http://w0:30000".into(),
            dp_rank: 0,
            seq: 1,
            observed_at_ms: 1,
            payload_hash: "hash".into(),
            payload_b64: encode_base64_for_test(&payload),
        };
        let service = CacheStateService::with_empty_tree();
        let resp = service
            .apply_stream_records(&[record.clone(), record])
            .unwrap();
        assert_eq!(resp.applied_events, 1);
    }

    #[test]
    fn digest_establishes_trust_after_continuous_events() {
        let service = reconciliation_service();
        let epoch = "epoch-a";
        apply_payload(
            &service,
            0,
            reconciled_batch(epoch, &[block_stored_event(&[10, 20], None, "GPU")], None),
        )
        .unwrap();

        let before_digest = service.match_prefix(&CacheStateMatchRequest {
            model_id: "m".into(),
            block_hashes: vec![10, 20],
        });
        assert!(before_digest.authoritative);
        assert_eq!(before_digest.matched_blocks, 0);

        let entries = vec![(None, 10), (Some(10), 20)];
        apply_payload(
            &service,
            1,
            reconciled_batch(epoch, &[], Some(digest_control(1, &entries))),
        )
        .unwrap();
        let trusted = service.match_prefix(&CacheStateMatchRequest {
            model_id: "m".into(),
            block_hashes: vec![10, 20, 30],
        });
        assert_eq!(trusted.matched_blocks, 2);
        assert_eq!(trusted.workers.len(), 1);
        assert!(service
            .reconciliation_metrics_text()
            .contains(r#"sgl_router_cache_state_digest_comparisons_total{outcome="match"} 1"#));
    }

    #[test]
    fn sequence_gap_fails_closed_then_snapshot_repairs_without_restart() {
        let service = reconciliation_service();
        let epoch = "epoch-gap";
        apply_payload(
            &service,
            0,
            reconciled_batch(epoch, &[block_stored_event(&[10], None, "GPU")], None),
        )
        .unwrap();
        apply_payload(
            &service,
            1,
            reconciled_batch(epoch, &[], Some(digest_control(1, &[(None, 10)]))),
        )
        .unwrap();

        // Sequence 2 is dropped before cache-state. Sequence 3 is still
        // applied to the candidate state, but the worker is excluded.
        apply_payload(
            &service,
            3,
            reconciled_batch(epoch, &[block_stored_event(&[20], Some(10), "GPU")], None),
        )
        .unwrap();
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![10, 20],
                })
                .matched_blocks,
            0
        );

        let snapshot_entries = vec![
            CacheStateSnapshotEntry {
                parent_block_hash: None,
                block_hash: 10,
                media: vec!["GPU".into()],
            },
            CacheStateSnapshotEntry {
                parent_block_hash: Some(10),
                block_hash: 20,
                media: vec!["CPU_PINNED".into(), "GPU".into()],
            },
        ];
        let manifest = snapshot_manifest("snapshot-gap", 4, &snapshot_entries, 1);
        apply_payload(
            &service,
            4,
            reconciled_batch(epoch, &[], Some(snapshot_manifest_control(true, &manifest))),
        )
        .unwrap();
        apply_payload(
            &service,
            5,
            reconciled_batch(
                epoch,
                &[],
                Some(snapshot_chunk_control("snapshot-gap", 0, &snapshot_entries)),
            ),
        )
        .unwrap();
        apply_payload(
            &service,
            6,
            reconciled_batch(
                epoch,
                &[],
                Some(snapshot_manifest_control(false, &manifest)),
            ),
        )
        .unwrap();
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![10, 20],
                })
                .matched_blocks,
            2
        );

        // The snapshot preserved both media. Removing GPU alone keeps the
        // logical entry; removing the final CPU copy removes it.
        apply_payload(
            &service,
            7,
            reconciled_batch(epoch, &[block_removed_event(&[20], "GPU")], None),
        )
        .unwrap();
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![10, 20],
                })
                .matched_blocks,
            2
        );
        apply_payload(
            &service,
            8,
            reconciled_batch(epoch, &[block_removed_event(&[20], "CPU_PINNED")], None),
        )
        .unwrap();
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![10, 20],
                })
                .matched_blocks,
            1
        );
        let metrics = service.reconciliation_metrics_text();
        assert!(metrics.contains("sgl_router_cache_state_sequence_gaps_total 1"));
        assert!(metrics.contains(r#"sgl_router_cache_state_snapshots_total{outcome="success"} 1"#));
        assert!(metrics.contains("sgl_router_cache_state_trusted_worker_ranks 1"));
    }

    #[test]
    fn stream_apply_commit_metrics_report_bounded_outcomes() {
        let service = CacheStateService::with_empty_tree();
        service.record_stream_apply_commit(Duration::from_millis(1500), true);
        service.record_stream_apply_commit(Duration::from_millis(250), false);

        let metrics = service.reconciliation_metrics_text();
        assert!(metrics.contains(
            r#"sgl_router_cache_state_stream_apply_commit_duration_seconds_sum{outcome="success"} 1.500000"#
        ));
        assert!(metrics.contains(
            r#"sgl_router_cache_state_stream_apply_commit_duration_seconds_count{outcome="success"} 1"#
        ));
        assert!(metrics.contains(
            r#"sgl_router_cache_state_stream_apply_commit_duration_seconds_sum{outcome="failure"} 0.250000"#
        ));
        assert!(metrics.contains(
            r#"sgl_router_cache_state_stream_apply_commit_duration_seconds_count{outcome="failure"} 1"#
        ));
    }

    #[test]
    fn corrupt_snapshot_stays_untrusted_and_does_not_install() {
        let service = reconciliation_service();
        let epoch = "epoch-corrupt";
        let entries = vec![CacheStateSnapshotEntry {
            parent_block_hash: None,
            block_hash: 99,
            media: vec!["GPU".into()],
        }];
        let manifest = snapshot_manifest("snapshot-corrupt", 0, &entries, 1);
        apply_payload(
            &service,
            0,
            reconciled_batch(epoch, &[], Some(snapshot_manifest_control(true, &manifest))),
        )
        .unwrap();
        apply_payload(
            &service,
            1,
            reconciled_batch(
                epoch,
                &[],
                Some(snapshot_chunk_control("snapshot-corrupt", 0, &entries)),
            ),
        )
        .unwrap();
        let mut corrupt = manifest.clone();
        corrupt.digest = "ff".repeat(32);
        assert!(apply_payload(
            &service,
            2,
            reconciled_batch(epoch, &[], Some(snapshot_manifest_control(false, &corrupt))),
        )
        .is_err());
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![99],
                })
                .matched_blocks,
            0
        );
    }

    #[test]
    fn exact_duplicate_is_idempotent_but_conflicting_duplicate_untrusts() {
        let service = reconciliation_service();
        let epoch = "epoch-duplicate";
        let digest = reconciled_batch(epoch, &[], Some(digest_control(0, &[])));
        apply_payload(&service, 0, digest.clone()).unwrap();
        assert_eq!(
            apply_payload(&service, 0, digest).unwrap().applied_events,
            0
        );

        let conflict = reconciled_batch(epoch, &[block_stored_event(&[1], None, "GPU")], None);
        assert!(apply_payload(&service, 0, conflict).is_err());
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![1],
                })
                .matched_blocks,
            0
        );
    }

    #[test]
    fn digest_mismatch_stays_untrusted_and_epoch_change_clears_old_state() {
        let service = reconciliation_service();
        apply_payload(
            &service,
            0,
            reconciled_batch("epoch-old", &[block_stored_event(&[1], None, "GPU")], None),
        )
        .unwrap();
        apply_payload(
            &service,
            1,
            reconciled_batch("epoch-old", &[], Some(digest_control(1, &[]))),
        )
        .unwrap();
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![1],
                })
                .matched_blocks,
            0
        );
        assert!(service
            .reconciliation_metrics_text()
            .contains(r#"sgl_router_cache_state_digest_comparisons_total{outcome="mismatch"} 1"#));

        apply_payload(
            &service,
            2,
            reconciled_batch("epoch-old", &[], Some(digest_control(2, &[(None, 1)]))),
        )
        .unwrap();
        assert_eq!(
            service
                .match_prefix(&CacheStateMatchRequest {
                    model_id: "m".into(),
                    block_hashes: vec![1],
                })
                .matched_blocks,
            1
        );

        // A process restart creates a new epoch and sequence generation. An
        // empty authoritative digest proves the old epoch's ownership is gone.
        apply_payload(
            &service,
            0,
            reconciled_batch("epoch-new", &[], Some(digest_control(0, &[]))),
        )
        .unwrap();
        let after_restart = service.match_prefix(&CacheStateMatchRequest {
            model_id: "m".into(),
            block_hashes: vec![1],
        });
        assert_eq!(after_restart.matched_blocks, 0);
        assert!(after_restart.authoritative);
    }

    fn block_stored_payload(block_hashes: &[i64]) -> Vec<u8> {
        let mut buf = Vec::new();
        rmp::encode::write_array_len(&mut buf, 3).unwrap();
        rmp::encode::write_f64(&mut buf, 1.0).unwrap();
        rmp::encode::write_array_len(&mut buf, 1).unwrap();
        rmp::encode::write_array_len(&mut buf, 7).unwrap();
        rmp::encode::write_str(&mut buf, "BlockStored").unwrap();
        rmp::encode::write_array_len(&mut buf, block_hashes.len() as u32).unwrap();
        for hash in block_hashes {
            rmp::encode::write_sint(&mut buf, *hash).unwrap();
        }
        rmp::encode::write_nil(&mut buf).unwrap();
        rmp::encode::write_array_len(&mut buf, 0).unwrap();
        rmp::encode::write_uint(&mut buf, 64).unwrap();
        rmp::encode::write_nil(&mut buf).unwrap();
        rmp::encode::write_nil(&mut buf).unwrap();
        rmp::encode::write_uint(&mut buf, 1).unwrap();
        buf
    }

    fn reconciliation_service() -> CacheStateService {
        CacheStateService::new_with_reconciliation(
            Arc::new(HashTree::new()),
            CacheStateReconciliationConfig {
                enabled: true,
                ..CacheStateReconciliationConfig::default()
            },
        )
    }

    fn apply_payload(
        service: &CacheStateService,
        seq: i64,
        payload: Vec<u8>,
    ) -> Result<CacheStateKvEventsResponse, CacheStateError> {
        service.apply_kv_events(&CacheStateKvEventsRequest {
            model_id: "m".into(),
            worker_url: "http://worker:30000".into(),
            dp_rank: 0,
            seq,
            payload_b64: encode_base64_for_test(&payload),
        })
    }

    fn reconciled_batch(epoch: &str, events: &[Vec<u8>], control: Option<Vec<u8>>) -> Vec<u8> {
        let mut buf = Vec::new();
        rmp::encode::write_array_len(&mut buf, 5).unwrap();
        rmp::encode::write_f64(&mut buf, 1.0).unwrap();
        rmp::encode::write_array_len(&mut buf, events.len() as u32).unwrap();
        for event in events {
            buf.extend_from_slice(event);
        }
        rmp::encode::write_uint(&mut buf, 0).unwrap();
        rmp::encode::write_str(&mut buf, epoch).unwrap();
        match control {
            Some(control) => buf.extend_from_slice(&control),
            None => rmp::encode::write_nil(&mut buf).unwrap(),
        }
        buf
    }

    fn block_stored_event(block_hashes: &[i64], parent_hash: Option<i64>, medium: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        rmp::encode::write_array_len(&mut buf, 7).unwrap();
        rmp::encode::write_str(&mut buf, "BlockStored").unwrap();
        write_i64_array(&mut buf, block_hashes);
        match parent_hash {
            Some(parent) => {
                rmp::encode::write_sint(&mut buf, parent).unwrap();
            }
            None => rmp::encode::write_nil(&mut buf).unwrap(),
        }
        rmp::encode::write_array_len(&mut buf, 0).unwrap();
        rmp::encode::write_uint(&mut buf, 64).unwrap();
        rmp::encode::write_nil(&mut buf).unwrap();
        rmp::encode::write_str(&mut buf, medium).unwrap();
        buf
    }

    fn block_removed_event(block_hashes: &[i64], medium: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        rmp::encode::write_array_len(&mut buf, 3).unwrap();
        rmp::encode::write_str(&mut buf, "BlockRemoved").unwrap();
        write_i64_array(&mut buf, block_hashes);
        rmp::encode::write_str(&mut buf, medium).unwrap();
        buf
    }

    fn write_i64_array(buf: &mut Vec<u8>, values: &[i64]) {
        rmp::encode::write_array_len(buf, values.len() as u32).unwrap();
        for value in values {
            rmp::encode::write_sint(buf, *value).unwrap();
        }
    }

    fn digest_control(through_seq: i64, entries: &[CacheEntryKey]) -> Vec<u8> {
        let digest = digest_for_entries(entries.iter().copied());
        let mut buf = Vec::new();
        rmp::encode::write_array_len(&mut buf, 5).unwrap();
        rmp::encode::write_str(&mut buf, "CacheStateDigest").unwrap();
        rmp::encode::write_sint(&mut buf, through_seq).unwrap();
        rmp::encode::write_uint(&mut buf, entries.len() as u64).unwrap();
        rmp::encode::write_str(&mut buf, "xor-sha256-v1").unwrap();
        rmp::encode::write_str(&mut buf, &digest_hex(&digest)).unwrap();
        buf
    }

    fn snapshot_manifest(
        snapshot_id: &str,
        watermark: i64,
        entries: &[CacheStateSnapshotEntry],
        total_chunks: usize,
    ) -> CacheStateSnapshotManifest {
        let digest = digest_for_entries(
            entries
                .iter()
                .map(|entry| (entry.parent_block_hash, entry.block_hash)),
        );
        CacheStateSnapshotManifest {
            snapshot_id: snapshot_id.into(),
            watermark,
            total_chunks,
            block_count: entries.len(),
            algorithm: "xor-sha256-v1".into(),
            digest: digest_hex(&digest),
        }
    }

    fn snapshot_manifest_control(start: bool, manifest: &CacheStateSnapshotManifest) -> Vec<u8> {
        let mut buf = Vec::new();
        rmp::encode::write_array_len(&mut buf, 7).unwrap();
        rmp::encode::write_str(
            &mut buf,
            if start {
                "CacheStateSnapshotStart"
            } else {
                "CacheStateSnapshotEnd"
            },
        )
        .unwrap();
        rmp::encode::write_str(&mut buf, &manifest.snapshot_id).unwrap();
        rmp::encode::write_sint(&mut buf, manifest.watermark).unwrap();
        rmp::encode::write_uint(&mut buf, manifest.total_chunks as u64).unwrap();
        rmp::encode::write_uint(&mut buf, manifest.block_count as u64).unwrap();
        rmp::encode::write_str(&mut buf, &manifest.algorithm).unwrap();
        rmp::encode::write_str(&mut buf, &manifest.digest).unwrap();
        buf
    }

    fn snapshot_chunk_control(
        snapshot_id: &str,
        chunk_index: usize,
        entries: &[CacheStateSnapshotEntry],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        rmp::encode::write_array_len(&mut buf, 4).unwrap();
        rmp::encode::write_str(&mut buf, "CacheStateSnapshotChunk").unwrap();
        rmp::encode::write_str(&mut buf, snapshot_id).unwrap();
        rmp::encode::write_uint(&mut buf, chunk_index as u64).unwrap();
        rmp::encode::write_array_len(&mut buf, entries.len() as u32).unwrap();
        for entry in entries {
            rmp::encode::write_array_len(&mut buf, 3).unwrap();
            match entry.parent_block_hash {
                Some(parent) => {
                    rmp::encode::write_sint(&mut buf, parent).unwrap();
                }
                None => rmp::encode::write_nil(&mut buf).unwrap(),
            }
            rmp::encode::write_sint(&mut buf, entry.block_hash).unwrap();
            rmp::encode::write_array_len(&mut buf, entry.media.len() as u32).unwrap();
            for medium in &entry.media {
                rmp::encode::write_str(&mut buf, medium).unwrap();
            }
        }
        buf
    }

    fn encode_base64_for_test(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0];
            let b1 = *chunk.get(1).unwrap_or(&0);
            let b2 = *chunk.get(2).unwrap_or(&0);
            out.push(ALPHABET[(b0 >> 2) as usize] as char);
            out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
            if chunk.len() >= 2 {
                out.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() == 3 {
                out.push(ALPHABET[(b2 & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }
}
