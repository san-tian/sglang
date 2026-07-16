// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware-ZMQ selection policy.
//!
//! Combines the KV-event-fed [`HashTree`] with active-load scoring and
//! tokenizer-driven block-hash lookup to pick the worker most likely to
//! already hold the request's prefix in its KV cache.
//!
//! # Selection algorithm
//!
//! Given `workers` (already filtered to healthy + matching pool by the
//! caller) and a `SelectionContext` carrying the JSON request body and the
//! ingress-precomputed routing tokens:
//!
//! 1. **Load-imbalance fast-path.** If `max_load - min_load >
//!    balance_abs_threshold` AND `max_load > min_load *
//!    balance_rel_threshold`, skip the cache lookup and pick the
//!    lowest-load worker. This prevents one hot worker from dominating
//!    cache-aware selection while every other worker idles.
//! 2. **Routing tokens.** Prefer the ingress-precomputed ids
//!    (`ctx.request_tokens()`); fall back to tokenizing the body here
//!    (chat-encoder-aware for chat traffic, raw `prompt`/`text` otherwise)
//!    for callers that didn't pre-tokenize. On any failure (no tokens, no
//!    tokenizer, encode error, empty), fall through to step 4 (min-load).
//! 3. **Hash + match.** Compute block hashes via
//!    [`super::kv_events::compute_block_hashes`], query the shared hash tree
//!    for the longest matching prefix. If `match_rate > cache_threshold`,
//!    pick the lowest-load worker whose `url` appears in the match result.
//!    Otherwise, fall through.
//! 4. **Min-load fallback.** Pick the lowest-load worker by
//!    `Worker::active_load()`.
//!
//! The implementation never returns `None` for a non-empty `workers` slice;
//! a misconfigured tree or tokenizer degrades to round-robin-with-load
//! tiebreak, not a routing failure.

use crate::cache_state::{
    CacheStateInsertRequest, CacheStateMatchRequest, CacheStateMatchResponse,
    RemoteCacheStateClient,
};
use crate::config::{CacheAwareConfig, CacheTreeSource, TtftScoreMode};
use crate::discovery::{WorkerBackend, WorkerMode};

use crate::policies::kv_events::tree::KvWorkerId;
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::{effective_priority, request_tokens_for, Policy, SelectionContext};
use crate::server::metrics::{
    MetricsRegistry, RemoteCacheStateFeedOutcome, RemoteCacheStateQueryOutcome,
};
use crate::tokenizer::TokenizerRegistry;
use crate::workers::worker::{merge_pending_load, PrefillLoadRole};
use crate::workers::Worker;
use serde_json::json;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone)]
struct PredictedTtftEstimate {
    matched_blocks: usize,
    candidate_uncached_tokens: usize,
    reported_work_tokens: Option<usize>,
    reserved_tokens: usize,
    work_ahead_tokens: usize,
    total_work_tokens: usize,
    normalized_score: usize,
    load_source: &'static str,
    prefill_capacity_milli: usize,
    selected_prefill_member: Option<String>,
}

/// Selection policy that scores candidates by tree-overlap with the
/// request's prefix and falls back to load-based picking when the tree
/// doesn't have useful signal.
pub struct CacheAwareZmqPolicy {
    config: CacheAwareConfig,
    /// Per-process KV-event hash tree, fed by the indexer. Cheap to
    /// clone an `Arc`; we never write to the tree from here.
    tree: Arc<HashTree>,
    /// Tokenizer registry — selection reads `model_id` from the context
    /// and looks up the per-model tokenizer.
    tokenizers: Arc<TokenizerRegistry>,
    /// Worker-sourced block size, shared with the `KvEventIndex` that
    /// seeds it on worker registration. Read once per request; if
    /// `None` (no worker has reported a `page_size` yet) the policy
    /// degrades to min-load — the router cannot hash a prompt without
    /// a block size that matches what the worker publishes.
    block_size_oracle: Arc<BlockSizeOracle>,
    /// Optional metrics sink. Set via [`Self::with_metrics`] by the policy
    /// factory for the production policy; `None` in unit tests and
    /// non-cache-aware call sites. When set, each cache-aware selection
    /// records the prefix-overlap block count into
    /// `sgl_router_overlap_blocks`. Set once via [`Self::with_metrics`]
    /// (tests) or the `Policy::attach_metrics` hook (production, called by
    /// `PolicyRegistry::attach_metrics` after the registry is built).
    metrics: OnceLock<Arc<MetricsRegistry>>,
    remote_cache_state: Option<Arc<RemoteCacheStateClient>>,
    /// Round-robin cursor for exact ties. This prevents cold/no-cache traffic
    /// from collapsing onto one stable worker while preserving cache affinity
    /// whenever a worker has a strictly better overlap or score.
    fair_tie_cursor: AtomicUsize,
}

impl std::fmt::Debug for CacheAwareZmqPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAwareZmqPolicy")
            .field("config", &self.config)
            .field("tree_nodes", &self.tree.node_count())
            .finish()
    }
}

impl CacheAwareZmqPolicy {
    pub fn new(
        config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Self {
        Self {
            config,
            tree,
            tokenizers,
            block_size_oracle,
            metrics: OnceLock::new(),
            remote_cache_state: None,
            fair_tie_cursor: AtomicUsize::new(0),
        }
    }

    pub fn with_remote_cache_state(mut self, client: Arc<RemoteCacheStateClient>) -> Self {
        self.remote_cache_state = Some(client);
        self
    }

    /// Attach a metrics sink so each cache-aware selection records the
    /// prefix-overlap block count into `sgl_router_overlap_blocks`. Builder
    /// form used by tests; production wiring goes through the
    /// `Policy::attach_metrics` hook.
    pub fn with_metrics(self, metrics: Arc<MetricsRegistry>) -> Self {
        let _ = self.metrics.set(metrics);
        self
    }

    /// Lowest-load worker — ties broken by stable iteration order (which
    /// is the order the registry returned, i.e. dashmap-undefined). For
    /// production traffic the ties are rare; tests pin the load skew.
    /// `use_reported` selects the real poller-reported load vs the
    /// router-side in-flight counter (see `Worker::effective_load`).
    fn pick_min_load(workers: &[Arc<Worker>], use_reported: bool) -> Option<Arc<Worker>> {
        workers
            .iter()
            .min_by_key(|w| w.effective_load(use_reported))
            .map(Arc::clone)
    }

    fn pick_fair_worker(&self, mut candidates: Vec<Arc<Worker>>) -> Option<Arc<Worker>> {
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_by(|left, right| left.url.cmp(&right.url));
        let index = self.fair_tie_cursor.fetch_add(1, Ordering::Relaxed) % candidates.len();
        Some(Arc::clone(&candidates[index]))
    }

    fn pick_min_load_fair(
        &self,
        workers: &[Arc<Worker>],
        use_reported: bool,
    ) -> Option<Arc<Worker>> {
        let groups = grouped_worker_scores(workers, |w| (w.effective_load(use_reported), 0));
        let min_load = groups.iter().map(|g| g.best_score).min()?;
        self.pick_fair_worker(
            groups
                .iter()
                .filter(|g| g.best_score == min_load)
                .map(|g| Arc::clone(&workers[g.root_index]))
                .collect(),
        )
    }

    fn pick_min_ttft_load(&self, workers: &[Arc<Worker>]) -> Option<Arc<Worker>> {
        let groups = grouped_worker_scores(workers, |w| {
            (
                w.effective_ttft_load(self.config.use_reported_load, self.config.ttft_token_scale),
                0,
            )
        });
        let min_load = groups.iter().map(|g| g.best_score).min()?;
        self.pick_fair_worker(
            groups
                .iter()
                .filter(|g| g.best_score == min_load)
                .map(|g| Arc::clone(&workers[g.root_index]))
                .collect(),
        )
    }

    fn hit_load_guard_diverts(&self, hot_load: usize, cool_load: usize) -> bool {
        self.config.hit_load_rel_threshold.is_finite()
            && hot_load.saturating_sub(cool_load) > self.config.hit_load_abs_threshold
            && (hot_load as f32) > (cool_load as f32) * self.config.hit_load_rel_threshold
    }

    /// Cache-hit load guard. Given the worker chosen by cache overlap
    /// (`hot`), divert to the globally least-loaded worker when `hot` is
    /// backed up past both thresholds relative to the coolest worker.
    /// Returns `hot` unchanged when the guard is OFF (rel = INFINITY), when
    /// `hot` *is* the coolest worker, or when the gap is below threshold.
    ///
    /// Two conditions, both required (AND):
    ///   ABS: `hot_load - min_load > hit_load_abs_threshold`
    ///   REL: `hot_load > min_load * hit_load_rel_threshold`
    /// The `is_finite()` arm-gate also dodges the `min_load == 0` edge: with
    /// min_load 0 the REL test would degenerate to `hot_load > 0`, so we only
    /// arm when the operator set a finite ratio.
    ///
    /// Setting `hit_load_abs_threshold = 1` and
    /// `hit_load_rel_threshold = 1.0` expresses the simple policy:
    /// keep cache affinity only while `hot_load <= min_load + 1`.
    fn apply_hit_load_guard(&self, hot: Arc<Worker>, workers: &[Arc<Worker>]) -> Arc<Worker> {
        if !self.config.hit_load_rel_threshold.is_finite() {
            return hot; // guard OFF — behaviour identical to plain cache-aware
        }
        let Some(cool) = Self::pick_min_load(workers, self.config.use_reported_load) else {
            return hot;
        };
        if cool.url == hot.url {
            return hot;
        }
        let c = hot.effective_load(self.config.use_reported_load);
        let m = cool.effective_load(self.config.use_reported_load);
        if self.hit_load_guard_diverts(c, m) {
            tracing::debug!(
                hot = %hot.url, hot_load = c,
                cool = %cool.url, cool_load = m,
                "cache-aware-zmq: hit-load guard diverted off backed-up cache worker",
            );
            cool
        } else {
            hot
        }
    }

    /// TTFT-first cache-hit guard. The TTFT score may still prefer a deeply
    /// cached worker while a cold worker has materially less first-token
    /// pressure. Reuse the cache-hit ABS/REL thresholds, but compare
    /// `effective_ttft_load` so token-weighted local and cross-replica pending
    /// reservations participate in the decision. The coolest candidate may
    /// also have a cache match: shared prefixes can otherwise exclude every
    /// idle worker and preserve starvation indefinitely.
    fn apply_ttft_hit_load_guard(
        &self,
        hot: Arc<Worker>,
        workers: &[Arc<Worker>],
        matched_blocks: usize,
        matched_urls: &HashSet<&str>,
    ) -> Arc<Worker> {
        if !self.config.hit_load_rel_threshold.is_finite()
            || matched_blocks_for_worker(&hot, matched_blocks, matched_urls) == 0
        {
            return hot;
        }

        let Some(cool) = self.pick_min_ttft_load(workers) else {
            return hot;
        };

        let hot_load =
            hot.effective_ttft_load(self.config.use_reported_load, self.config.ttft_token_scale);
        let cool_load =
            cool.effective_ttft_load(self.config.use_reported_load, self.config.ttft_token_scale);
        if self.hit_load_guard_diverts(hot_load, cool_load) {
            tracing::debug!(
                hot = %hot.url,
                hot_ttft_load = hot_load,
                cool = %cool.url,
                cool_ttft_load = cool_load,
                "cache-aware-zmq: TTFT hit-load guard diverted to cold worker",
            );
            cool
        } else {
            hot
        }
    }

    /// Detect load imbalance. Returns `true` when the spread between max
    /// and min load is large enough that cache-aware routing would dump
    /// even more on the hot worker.
    fn is_imbalanced(&self, workers: &[Arc<Worker>]) -> bool {
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(mn, mx), w| {
            let l = w.effective_load(self.config.use_reported_load);
            (mn.min(l), mx.max(l))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };
        let abs_diff = max_load.saturating_sub(min_load);
        let rel_threshold = (min_load as f32 * self.config.balance_rel_threshold) as usize;
        abs_diff > self.config.balance_abs_threshold && max_load > rel_threshold
    }

    #[allow(clippy::too_many_arguments)]
    fn select_ttft_first(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
        block_hashes: &[i64],
        matched_blocks: usize,
        matched_urls: &HashSet<&str>,
        candidate_tokens: usize,
        block_size: usize,
    ) -> Option<Arc<Worker>> {
        let idle_candidates;
        let score_workers: &[Arc<Worker>] = if self.config.ttft_idle_first_routing {
            let min_load = workers
                .iter()
                .map(|w| {
                    w.effective_ttft_load(
                        self.config.use_reported_load,
                        self.config.ttft_token_scale,
                    )
                })
                .min()?;
            idle_candidates = workers
                .iter()
                .filter(|w| {
                    w.effective_ttft_load(
                        self.config.use_reported_load,
                        self.config.ttft_token_scale,
                    ) == min_load
                })
                .map(Arc::clone)
                .collect::<Vec<_>>();
            &idle_candidates
        } else {
            workers
        };
        let score_mode = self.compatible_score_mode(score_workers);
        if block_hashes.is_empty() && score_mode == TtftScoreMode::Additive {
            return self.pick_min_ttft_load(workers);
        }
        let candidate_priority = if matches!(
            score_mode,
            TtftScoreMode::LmetricCandidateAware | TtftScoreMode::PredictedTtft
        ) {
            ctx.request_body()
                .and_then(|body| serde_json::from_slice::<serde_json::Value>(body).ok())
                .as_ref()
                .map(effective_priority)
                .unwrap_or(0)
        } else {
            0
        };

        let total_blocks = block_hashes.len();
        let groups = grouped_worker_scores(score_workers, |w| {
            let (score, worker_matched) = if score_mode == TtftScoreMode::PredictedTtft {
                let estimate = self.predicted_ttft_estimate_for_worker(
                    w,
                    matched_blocks,
                    matched_urls,
                    candidate_tokens,
                    block_size,
                    candidate_priority,
                );
                (estimate.normalized_score, estimate.matched_blocks)
            } else {
                let worker_matched = matched_blocks_for_worker(w, matched_blocks, matched_urls);
                (
                    self.ttft_score_with_mode(
                        w,
                        total_blocks,
                        worker_matched,
                        candidate_tokens,
                        block_size,
                        candidate_priority,
                        score_mode,
                    ),
                    worker_matched,
                )
            };
            (score, worker_matched)
        });
        let best_score = groups
            .iter()
            .map(|g| g.best_score)
            .min()
            .unwrap_or(usize::MAX);
        let score_limit = best_score.saturating_add(self.config.ttft_cache_score_margin);

        let eligible: Vec<&GroupScore> = groups
            .iter()
            .filter(|group| group.best_score <= score_limit)
            .collect();

        let chosen = eligible
            .iter()
            .map(|group| group.best_matched_blocks)
            .max()
            .and_then(|best_matched| {
                let best_score_for_match = eligible
                    .iter()
                    .filter(|group| group.best_matched_blocks == best_matched)
                    .map(|group| group.best_score)
                    .min()?;
                let candidates = eligible
                    .iter()
                    .filter(|group| {
                        group.best_matched_blocks == best_matched
                            && group.best_score == best_score_for_match
                    })
                    .map(|group| Arc::clone(&score_workers[group.root_index]))
                    .collect();
                self.pick_fair_worker(candidates)
                    .map(|w| (w, best_score_for_match, best_matched))
            })
            .map(|(w, score, worker_matched)| {
                tracing::debug!(
                    model = %ctx.model(),
                    worker = %w.url,
                    matched_blocks = worker_matched,
                    ttft_score = score,
                    best_ttft_score = best_score,
                    cache_score_margin = self.config.ttft_cache_score_margin,
                    ttft_score_mode = ?score_mode,
                    "cache-aware-zmq: ttft-first selected worker",
                );
                w
            });

        let hot_url = chosen.as_ref().map(|worker| worker.url.clone());
        let mut reason = if chosen.is_some() {
            "ttft_first_score"
        } else {
            "ttft_min_load_fallback"
        };
        let selected = if let Some(hot) = chosen {
            if score_mode == TtftScoreMode::PredictedTtft {
                // Load and cache work are already expressed in one token-time
                // score. A request-count guard would override that comparison
                // with incompatible units.
                Some(hot)
            } else {
                let selected =
                    self.apply_ttft_hit_load_guard(hot, workers, matched_blocks, matched_urls);
                if hot_url.as_deref() != Some(selected.url.as_str()) {
                    reason = "ttft_hit_load_guard_divert";
                }
                Some(selected)
            }
        } else {
            self.pick_min_ttft_load(workers)
        };
        self.log_ttft_decision(
            ctx,
            workers,
            score_workers,
            selected.as_deref(),
            hot_url.as_deref(),
            reason,
            total_blocks,
            matched_blocks,
            matched_urls,
            candidate_tokens,
            block_size,
            candidate_priority,
            score_mode,
            best_score,
            score_limit,
        );
        selected
    }

    #[cfg(test)]
    fn ttft_score(
        &self,
        worker: &Worker,
        total_blocks: usize,
        matched_blocks: usize,
        candidate_tokens: usize,
        block_size: usize,
        candidate_priority: i64,
    ) -> usize {
        let score_mode = self.compatible_score_mode_for_worker(worker);
        self.ttft_score_with_mode(
            worker,
            total_blocks,
            matched_blocks,
            candidate_tokens,
            block_size,
            candidate_priority,
            score_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ttft_score_with_mode(
        &self,
        worker: &Worker,
        total_blocks: usize,
        matched_blocks: usize,
        candidate_tokens: usize,
        block_size: usize,
        candidate_priority: i64,
        score_mode: TtftScoreMode,
    ) -> usize {
        if score_mode != TtftScoreMode::Additive {
            if let Some(score) = self.token_work_score(
                worker,
                matched_blocks,
                candidate_tokens,
                block_size,
                candidate_priority,
                score_mode,
            ) {
                return score;
            }
        }
        let pressure =
            worker.effective_ttft_load(self.config.use_reported_load, self.config.ttft_token_scale);
        let uncached_blocks = total_blocks.saturating_sub(matched_blocks);
        pressure.saturating_add(uncached_blocks)
    }

    fn token_work_score(
        &self,
        worker: &Worker,
        matched_blocks: usize,
        candidate_tokens: usize,
        block_size: usize,
        candidate_priority: i64,
        score_mode: TtftScoreMode,
    ) -> Option<usize> {
        if score_mode == TtftScoreMode::PredictedTtft {
            return Some(
                self.predicted_ttft_estimate(
                    worker,
                    matched_blocks,
                    candidate_tokens,
                    block_size,
                    candidate_priority,
                )
                .normalized_score,
            );
        }
        let snapshot = worker.reported_prefill_load()?;
        let candidate_uncached =
            candidate_tokens.saturating_sub(matched_blocks.saturating_mul(block_size));
        let existing_work = match score_mode {
            TtftScoreMode::Additive => return None,
            TtftScoreMode::PrefillWorkOnly | TtftScoreMode::Lmetric => {
                snapshot.total_waiting_uncached_tokens
            }
            TtftScoreMode::PrefillWorkNormalized => snapshot.total_waiting_uncached_tokens,
            TtftScoreMode::PredictedTtft => unreachable!("handled above"),
            TtftScoreMode::LmetricCandidateAware => snapshot
                .candidate
                .as_ref()
                .and_then(|candidate| {
                    candidate.work_ahead_tokens(candidate_priority, candidate_uncached)
                })
                .unwrap_or(snapshot.total_waiting_uncached_tokens),
        };
        let reserved_tokens = merge_pending_load(
            worker.pending_token_load(),
            worker.global_pending_token_load(),
        );
        let reserved_requests =
            merge_pending_load(worker.pending_load(), worker.global_pending_load());
        let prefill_factor = candidate_uncached
            .saturating_add(existing_work)
            .saturating_add(reserved_tokens);
        if score_mode == TtftScoreMode::PrefillWorkOnly {
            return Some(prefill_factor);
        }
        if score_mode == TtftScoreMode::PrefillWorkNormalized {
            return Some(normalize_prefill_work(
                prefill_factor,
                worker.prefill_capacity_milli(),
            ));
        }
        let batch_factor = 1usize
            .saturating_add(snapshot.running_requests)
            .saturating_add(reserved_requests);
        Some(prefill_factor.saturating_mul(batch_factor))
    }

    fn predicted_ttft_estimate(
        &self,
        worker: &Worker,
        matched_blocks: usize,
        candidate_tokens: usize,
        block_size: usize,
        candidate_priority: i64,
    ) -> PredictedTtftEstimate {
        let candidate_uncached_tokens =
            candidate_tokens.saturating_sub(matched_blocks.saturating_mul(block_size));
        let snapshot = worker.reported_prefill_load();
        self.predicted_ttft_estimate_from_snapshot(
            worker,
            snapshot.as_ref(),
            matched_blocks,
            candidate_uncached_tokens,
            candidate_priority,
            worker.prefill_capacity_milli(),
            None,
        )
    }

    fn predicted_ttft_estimate_for_worker(
        &self,
        worker: &Worker,
        matched_blocks: usize,
        matched_urls: &HashSet<&str>,
        candidate_tokens: usize,
        block_size: usize,
        candidate_priority: i64,
    ) -> PredictedTtftEstimate {
        let member_snapshots = worker.reported_prefill_members();
        let best_member = member_snapshots
            .iter()
            .map(|member| {
                let member_matched = if matched_urls.contains(member.worker_url.as_str()) {
                    matched_blocks
                } else {
                    0
                };
                let candidate_uncached_tokens =
                    candidate_tokens.saturating_sub(member_matched.saturating_mul(block_size));
                self.predicted_ttft_estimate_from_snapshot(
                    worker,
                    Some(&member.snapshot),
                    member_matched,
                    candidate_uncached_tokens,
                    candidate_priority,
                    member.prefill_capacity_milli,
                    Some(member.worker_url.clone()),
                )
            })
            .min_by_key(|estimate| {
                (
                    estimate.normalized_score,
                    std::cmp::Reverse(estimate.matched_blocks),
                )
            });
        best_member.unwrap_or_else(|| {
            let worker_matched = matched_blocks_for_worker(worker, matched_blocks, matched_urls);
            self.predicted_ttft_estimate(
                worker,
                worker_matched,
                candidate_tokens,
                block_size,
                candidate_priority,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn predicted_ttft_estimate_from_snapshot(
        &self,
        worker: &Worker,
        snapshot: Option<&crate::workers::worker::PrefillLoadSnapshot>,
        matched_blocks: usize,
        candidate_uncached_tokens: usize,
        candidate_priority: i64,
        prefill_capacity_milli: usize,
        selected_prefill_member: Option<String>,
    ) -> PredictedTtftEstimate {
        let candidate_aware_work = snapshot.and_then(|snapshot| {
            snapshot.candidate.as_ref().and_then(|candidate| {
                candidate.work_ahead_tokens(candidate_priority, candidate_uncached_tokens)
            })
        });
        let (reported_work_tokens, load_source) = if let Some(work) = candidate_aware_work {
            (Some(work), "candidate-aware-snapshot")
        } else if let Some(snapshot) = snapshot {
            (
                Some(snapshot.total_waiting_uncached_tokens),
                "waiting-token-snapshot",
            )
        } else {
            (None, "reservation-only")
        };
        let reserved_tokens = merge_pending_load(
            worker.pending_token_load(),
            worker.global_pending_token_load(),
        );
        // Worker snapshots and router reservations describe overlapping work.
        // Taking the maximum bridges poll lag without counting an admitted
        // request twice for its full lifetime.
        let work_ahead_tokens = reported_work_tokens.unwrap_or(0).max(reserved_tokens);
        let total_work_tokens = candidate_uncached_tokens.saturating_add(work_ahead_tokens);
        let probe_failed =
            self.config.use_reported_load && !worker.introspection_probe_allows_routing();
        let normalized_score = if probe_failed {
            usize::MAX / 2
        } else {
            normalize_prefill_work(total_work_tokens, prefill_capacity_milli)
        };

        PredictedTtftEstimate {
            matched_blocks,
            candidate_uncached_tokens,
            reported_work_tokens,
            reserved_tokens,
            work_ahead_tokens,
            total_work_tokens,
            normalized_score,
            load_source: if probe_failed {
                "probe-failed"
            } else if selected_prefill_member.is_some() {
                match load_source {
                    "candidate-aware-snapshot" => "member-candidate-aware-snapshot",
                    "waiting-token-snapshot" => "member-waiting-token-snapshot",
                    _ => load_source,
                }
            } else {
                load_source
            },
            prefill_capacity_milli: prefill_capacity_milli.max(1),
            selected_prefill_member,
        }
    }

    fn compatible_score_mode(&self, workers: &[Arc<Worker>]) -> TtftScoreMode {
        if self.config.ttft_score_mode == TtftScoreMode::PredictedTtft {
            return TtftScoreMode::PredictedTtft;
        }
        if self.config.ttft_score_mode == TtftScoreMode::Additive
            || !workers
                .iter()
                .all(|worker| self.supports_token_score(worker, self.config.ttft_score_mode))
        {
            return TtftScoreMode::Additive;
        }
        if self.config.ttft_score_mode == TtftScoreMode::LmetricCandidateAware
            && !workers.iter().all(|worker| {
                worker
                    .reported_prefill_load()
                    .and_then(|snapshot| snapshot.candidate)
                    .is_some()
            })
        {
            return TtftScoreMode::Lmetric;
        }
        self.config.ttft_score_mode
    }

    #[cfg(test)]
    fn compatible_score_mode_for_worker(&self, worker: &Worker) -> TtftScoreMode {
        if self.config.ttft_score_mode == TtftScoreMode::PredictedTtft {
            return TtftScoreMode::PredictedTtft;
        }
        if self.config.ttft_score_mode == TtftScoreMode::Additive
            || !self.supports_token_score(worker, self.config.ttft_score_mode)
        {
            return TtftScoreMode::Additive;
        }
        let snapshot = worker
            .reported_prefill_load()
            .expect("compatible token score requires a Prefill snapshot");
        if self.config.ttft_score_mode == TtftScoreMode::LmetricCandidateAware
            && snapshot.candidate.is_none()
        {
            return TtftScoreMode::Lmetric;
        }
        self.config.ttft_score_mode
    }

    fn supports_token_score(&self, worker: &Worker, score_mode: TtftScoreMode) -> bool {
        if score_mode == TtftScoreMode::PredictedTtft {
            return true;
        }
        if worker.backend() != WorkerBackend::Sglang {
            return false;
        }
        let Some(snapshot) = worker.reported_prefill_load() else {
            return false;
        };
        match score_mode {
            TtftScoreMode::Additive => true,
            TtftScoreMode::PrefillWorkOnly => matches!(
                (worker.mode(), snapshot.role),
                (WorkerMode::Plain, PrefillLoadRole::Integrated)
                    | (WorkerMode::Prefill, PrefillLoadRole::Prefill)
            ),
            TtftScoreMode::PrefillWorkNormalized => matches!(
                (worker.mode(), snapshot.role),
                (WorkerMode::Plain, PrefillLoadRole::Integrated)
                    | (WorkerMode::Prefill, PrefillLoadRole::Prefill)
            ),
            TtftScoreMode::PredictedTtft => true,
            TtftScoreMode::Lmetric | TtftScoreMode::LmetricCandidateAware => {
                worker.mode() == WorkerMode::Plain && snapshot.role == PrefillLoadRole::Integrated
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn log_ttft_decision(
        &self,
        ctx: &SelectionContext<'_>,
        workers: &[Arc<Worker>],
        score_workers: &[Arc<Worker>],
        selected: Option<&Worker>,
        hot_worker_url: Option<&str>,
        reason: &str,
        total_blocks: usize,
        matched_blocks: usize,
        matched_urls: &HashSet<&str>,
        candidate_tokens: usize,
        block_size: usize,
        candidate_priority: i64,
        score_mode: TtftScoreMode,
        best_score: usize,
        score_limit: usize,
    ) {
        let Some(log_ctx) = ctx.route_decision_log() else {
            return;
        };
        let selected_url = selected.map(|worker| worker.url.as_str()).unwrap_or("-");
        let score_urls: HashSet<&str> = score_workers
            .iter()
            .map(|worker| worker.url.as_str())
            .collect();
        let mut candidates: Vec<_> = workers
            .iter()
            .map(|worker| {
                let predicted_estimate = (score_mode == TtftScoreMode::PredictedTtft).then(|| {
                    self.predicted_ttft_estimate_for_worker(
                        worker,
                        matched_blocks,
                        matched_urls,
                        candidate_tokens,
                        block_size,
                        candidate_priority,
                    )
                });
                let worker_matched = predicted_estimate.as_ref().map_or_else(
                    || matched_blocks_for_worker(worker, matched_blocks, matched_urls),
                    |estimate| estimate.matched_blocks,
                );
                let score = predicted_estimate.as_ref().map_or_else(
                    || {
                        self.ttft_score_with_mode(
                            worker,
                            total_blocks,
                            worker_matched,
                            candidate_tokens,
                            block_size,
                            candidate_priority,
                            score_mode,
                        )
                    },
                    |estimate| estimate.normalized_score,
                );
                json!({
                    "worker": worker.url,
                    "selected": worker.url == selected_url,
                    "score_considered": score_urls.contains(worker.url.as_str()),
                    "matched_blocks": worker_matched,
                    "matched_by_cache": worker_matched > 0,
                    "ttft_score": score,
                    "score_breakdown": self.ttft_score_breakdown_json(
                        worker,
                        total_blocks,
                        matched_blocks,
                        worker_matched,
                        matched_urls,
                        candidate_tokens,
                        block_size,
                        candidate_priority,
                        score_mode,
                    ),
                    "state": crate::server::route_decision::generic_candidate_json(
                        worker,
                        worker.url == selected_url,
                    ),
                })
            })
            .collect();
        candidates.sort_by_key(|candidate| {
            candidate
                .get("ttft_score")
                .and_then(|value| value.as_u64())
                .unwrap_or(u64::MAX)
        });
        let truncated = candidates.len() > log_ctx.candidate_limit;
        candidates.truncate(log_ctx.candidate_limit);
        let decision = json!({
            "event": "route_decision",
            "policy": "cache_aware_zmq",
            "policy_detail": "ttft_first",
            "endpoint": log_ctx.endpoint,
            "request_id": log_ctx.request_id,
            "model": ctx.model().0,
            "request_priority": log_ctx.request_priority,
            "candidate_priority_for_score": candidate_priority,
            "reason": reason,
            "selected_worker": selected_url,
            "hot_worker_before_guard": hot_worker_url,
            "candidate_count": workers.len(),
            "candidate_limit": log_ctx.candidate_limit,
            "candidates_truncated": truncated,
            "total_blocks": total_blocks,
            "matched_blocks_global": matched_blocks,
            "matched_worker_url_count": matched_urls.len(),
            "candidate_tokens": candidate_tokens,
            "block_size": block_size,
            "ttft_score_mode": format!("{:?}", score_mode),
            "configured_ttft_score_mode": format!("{:?}", self.config.ttft_score_mode),
            "ttft_cache_score_margin": self.config.ttft_cache_score_margin,
            "best_score": best_score,
            "score_limit": score_limit,
            "ttft_idle_first_routing": self.config.ttft_idle_first_routing,
            "use_reported_load": self.config.use_reported_load,
            "ttft_token_scale": self.config.ttft_token_scale,
            "candidates": candidates,
        });
        tracing::info!(decision = %decision, "route_decision");
    }

    #[allow(clippy::too_many_arguments)]
    fn ttft_score_breakdown_json(
        &self,
        worker: &Worker,
        total_blocks: usize,
        global_matched_blocks: usize,
        matched_blocks: usize,
        matched_urls: &HashSet<&str>,
        candidate_tokens: usize,
        block_size: usize,
        candidate_priority: i64,
        score_mode: TtftScoreMode,
    ) -> serde_json::Value {
        let snapshot = worker.reported_prefill_load();
        let predicted_estimate = (score_mode == TtftScoreMode::PredictedTtft).then(|| {
            self.predicted_ttft_estimate_for_worker(
                worker,
                global_matched_blocks,
                matched_urls,
                candidate_tokens,
                block_size,
                candidate_priority,
            )
        });
        let effective_matched_blocks = predicted_estimate
            .as_ref()
            .map(|estimate| estimate.matched_blocks)
            .unwrap_or(matched_blocks);
        let candidate_uncached = predicted_estimate
            .as_ref()
            .map(|estimate| estimate.candidate_uncached_tokens)
            .unwrap_or_else(|| {
                candidate_tokens.saturating_sub(effective_matched_blocks.saturating_mul(block_size))
            });
        let existing_work_tokens = snapshot.as_ref().map(|snapshot| match score_mode {
            TtftScoreMode::Additive => 0,
            TtftScoreMode::PrefillWorkOnly
            | TtftScoreMode::PrefillWorkNormalized
            | TtftScoreMode::Lmetric => snapshot.total_waiting_uncached_tokens,
            TtftScoreMode::PredictedTtft => predicted_estimate
                .as_ref()
                .and_then(|estimate| estimate.reported_work_tokens)
                .unwrap_or(0),
            TtftScoreMode::LmetricCandidateAware => snapshot
                .candidate
                .as_ref()
                .and_then(|candidate| {
                    candidate.work_ahead_tokens(candidate_priority, candidate_uncached)
                })
                .unwrap_or(snapshot.total_waiting_uncached_tokens),
        });
        let reserved_tokens = merge_pending_load(
            worker.pending_token_load(),
            worker.global_pending_token_load(),
        );
        let reserved_requests =
            merge_pending_load(worker.pending_load(), worker.global_pending_load());
        let prefill_factor_tokens = if let Some(estimate) = predicted_estimate.as_ref() {
            Some(estimate.total_work_tokens)
        } else {
            existing_work_tokens.map(|existing| {
                candidate_uncached
                    .saturating_add(existing)
                    .saturating_add(reserved_tokens)
            })
        };
        let batch_factor = if score_mode == TtftScoreMode::PredictedTtft {
            None
        } else {
            snapshot.as_ref().map(|snapshot| {
                1usize
                    .saturating_add(snapshot.running_requests)
                    .saturating_add(reserved_requests)
            })
        };
        json!({
            "score_mode": format!("{:?}", score_mode),
            "total_blocks": total_blocks,
            "matched_blocks": effective_matched_blocks,
            "candidate_tokens": candidate_tokens,
            "block_size": block_size,
            "candidate_uncached_tokens": candidate_uncached,
            "existing_work_tokens": existing_work_tokens,
            "reserved_tokens": reserved_tokens,
            "reserved_requests": reserved_requests,
            "prefill_factor_tokens": prefill_factor_tokens,
            "prefill_capacity_milli": predicted_estimate
                .as_ref()
                .map(|estimate| estimate.prefill_capacity_milli)
                .unwrap_or_else(|| worker.prefill_capacity_milli()),
            "normalized_prefill_score": prefill_factor_tokens
                .map(|work| normalize_prefill_work(
                    work,
                    predicted_estimate
                        .as_ref()
                        .map(|estimate| estimate.prefill_capacity_milli)
                        .unwrap_or_else(|| worker.prefill_capacity_milli()),
                )),
            "predicted_ttft": predicted_estimate.as_ref().map(|estimate| json!({
                "load_source": estimate.load_source,
                "selected_prefill_member": estimate.selected_prefill_member,
                "reported_work_tokens": estimate.reported_work_tokens,
                "reserved_tokens": estimate.reserved_tokens,
                "work_ahead_tokens": estimate.work_ahead_tokens,
                "candidate_uncached_tokens": estimate.candidate_uncached_tokens,
                "total_work_tokens": estimate.total_work_tokens,
                "prefill_capacity_milli": estimate.prefill_capacity_milli,
                "normalized_score": estimate.normalized_score,
            })),
            "batch_factor": batch_factor,
            "additive_pressure": worker.effective_ttft_load(
                self.config.use_reported_load,
                self.config.ttft_token_scale,
            ),
            "final_score": predicted_estimate
                .as_ref()
                .map(|estimate| estimate.normalized_score)
                .unwrap_or_else(|| self.ttft_score_with_mode(
                    worker,
                    total_blocks,
                    matched_blocks,
                    candidate_tokens,
                    block_size,
                    candidate_priority,
                    score_mode,
                )),
            "reported_prefill_load": snapshot.map(|snapshot| json!({
                "role": format!("{:?}", snapshot.role),
                "running_requests": snapshot.running_requests,
                "total_waiting_uncached_tokens": snapshot.total_waiting_uncached_tokens,
                "candidate_aware": snapshot.candidate.is_some(),
            })),
            "reported_prefill_member_count": worker.reported_prefill_members().len(),
        })
    }

    fn match_prefix(&self, model: &crate::discovery::ModelId, block_hashes: &[i64]) -> CacheMatch {
        if let Some(client) = &self.remote_cache_state {
            let req = CacheStateMatchRequest {
                model_id: model.0.clone(),
                block_hashes: block_hashes.to_vec(),
            };
            match client.match_prefix(&req) {
                Some(resp) => {
                    let authoritative = resp.authoritative;
                    let remote_match = CacheMatch::from_remote(resp);
                    if remote_match.is_useful() {
                        self.record_remote_cache_state_query(RemoteCacheStateQueryOutcome::Hit);
                        return remote_match;
                    }
                    if authoritative {
                        self.record_remote_cache_state_query(RemoteCacheStateQueryOutcome::Miss);
                        tracing::debug!(
                            model = %model,
                            "cache-aware-zmq: authoritative remote cache-state returned no useful match",
                        );
                        return remote_match;
                    }
                    let local_match =
                        CacheMatch::from_local(self.tree.match_prefix(None, block_hashes));
                    if local_match.is_useful() {
                        self.record_remote_cache_state_query(
                            RemoteCacheStateQueryOutcome::FallbackLocalHit,
                        );
                        tracing::debug!(
                            model = %model,
                            "cache-aware-zmq: remote cache-state returned no useful match; using local cache tree",
                        );
                    } else {
                        self.record_remote_cache_state_query(RemoteCacheStateQueryOutcome::Miss);
                    }
                    return local_match;
                }
                None => {
                    let local_match =
                        CacheMatch::from_local(self.tree.match_prefix(None, block_hashes));
                    if local_match.is_useful() {
                        self.record_remote_cache_state_query(
                            RemoteCacheStateQueryOutcome::FallbackLocalHit,
                        );
                        tracing::debug!(
                            model = %model,
                            "cache-aware-zmq: remote cache-state query failed; using local cache tree",
                        );
                    } else {
                        self.record_remote_cache_state_query(RemoteCacheStateQueryOutcome::Failure);
                    }
                    tracing::debug!(
                        model = %model,
                        "cache-aware-zmq: remote cache-state query failed",
                    );
                    return local_match;
                }
            }
        }
        CacheMatch::from_local(self.tree.match_prefix(None, block_hashes))
    }

    fn record_remote_cache_state_query(&self, outcome: RemoteCacheStateQueryOutcome) {
        if let Some(m) = self.metrics.get() {
            m.record_remote_cache_state_query(outcome);
        }
    }

    fn record_remote_cache_state_feed(&self, outcome: RemoteCacheStateFeedOutcome) {
        if let Some(m) = self.metrics.get() {
            m.record_remote_cache_state_feed(outcome);
        }
    }

    /// Route-history tree feeding: in `RouteHistory` tree-source mode, record
    /// this request's prefix block hashes against the worker we actually
    /// chose, so a subsequent request sharing the prefix matches it. No-op in
    /// `Zmq` mode (there the worker's own KV-event stream owns the tree;
    /// double-feeding would corrupt the eviction-accurate state). `parent_hash
    /// = None` inserts the full chain from the root, mirroring how
    /// `match_prefix(None, ..)` queries it.
    fn feed_route_history(
        &self,
        model: &crate::discovery::ModelId,
        chosen: &Option<Arc<Worker>>,
        block_hashes: &[i64],
    ) {
        if self.config.tree_source != CacheTreeSource::RouteHistory {
            return;
        }
        let Some(w) = chosen else { return };
        if block_hashes.is_empty() {
            return;
        }
        let kw = KvWorkerId::new(w.url.clone(), 0);
        self.tree.insert(&kw, None, block_hashes);

        if let Some(client) = &self.remote_cache_state {
            let ok = client.insert(&CacheStateInsertRequest {
                model_id: model.0.clone(),
                worker_url: w.url.clone(),
                dp_rank: 0,
                parent_hash: None,
                block_hashes: block_hashes.to_vec(),
            });
            let outcome = if ok {
                RemoteCacheStateFeedOutcome::Success
            } else {
                RemoteCacheStateFeedOutcome::Failure
            };
            self.record_remote_cache_state_feed(outcome);
            if !ok {
                tracing::debug!(
                    model = %model,
                    worker = %w.url,
                    "cache-aware-zmq: remote cache-state feed failed",
                );
            }
        }
    }

    pub(crate) fn select_from_candidates(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }

        // 1. Load-imbalance fast-path: even the best cache hit gets
        //    dropped in favour of evening out load.
        if !self.config.ttft_first_routing && self.is_imbalanced(workers) {
            return self.pick_min_load_fair(workers, self.config.use_reported_load);
        }

        // 2. Routing tokens. Prefer the ids computed once at ingress; fall
        //    back to tokenizing the body here so the policy stays usable for
        //    callers that don't pre-tokenize (e.g. unit tests). In production
        //    the ingress always pre-tokenizes, so this is a single tokenize.
        let fallback_ids;
        let tokens: &[u32] = match ctx.request_tokens() {
            Some(t) if !t.is_empty() => t,
            _ => {
                let body = match ctx.request_body() {
                    Some(b) if !b.is_empty() => b,
                    _ => {
                        return if self.config.ttft_first_routing {
                            self.pick_min_ttft_load(workers)
                        } else {
                            self.pick_min_load_fair(workers, self.config.use_reported_load)
                        }
                    }
                };
                let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
                    return if self.config.ttft_first_routing {
                        self.pick_min_ttft_load(workers)
                    } else {
                        self.pick_min_load_fair(workers, self.config.use_reported_load)
                    };
                };
                let Some(rt) = request_tokens_for(&self.tokenizers, ctx.model(), &value) else {
                    return if self.config.ttft_first_routing {
                        self.pick_min_ttft_load(workers)
                    } else {
                        self.pick_min_load_fair(workers, self.config.use_reported_load)
                    };
                };
                fallback_ids = rt.ids;
                &fallback_ids
            }
        };

        // 3. Hash + match.
        // Source block_size from the worker — the router can only hash
        // prompts at the block size the workers publish at. If no worker
        // has registered yet (oracle empty), cache-aware routing has no
        // ground truth to score against; fall back to min-load.
        let Some(block_size) = self.block_size_oracle.get() else {
            tracing::debug!(
                model = %ctx.model(),
                "cache-aware-zmq: block size unknown (no worker page_size yet), falling back to min-load",
            );
            return if self.config.ttft_first_routing
                && self.config.ttft_score_mode != TtftScoreMode::Additive
            {
                self.select_ttft_first(workers, ctx, &[], 0, &HashSet::new(), tokens.len(), 1)
            } else if self.config.ttft_first_routing {
                self.pick_min_ttft_load(workers)
            } else {
                self.pick_min_load_fair(workers, self.config.use_reported_load)
            };
        };
        // EAGLE-family workers hash KV blocks over token bigrams; the query
        // hashes must match the worker's stored hashes or the tree lookup
        // always misses (overlap stays 0). The oracle carries the worker-
        // reported flag.
        let is_bigram = self.block_size_oracle.is_bigram();
        let block_hashes = if is_bigram {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if block_hashes.is_empty() {
            return if self.config.ttft_first_routing
                && self.config.ttft_score_mode != TtftScoreMode::Additive
            {
                self.select_ttft_first(
                    workers,
                    ctx,
                    &block_hashes,
                    0,
                    &HashSet::new(),
                    tokens.len(),
                    block_size as usize,
                )
            } else if self.config.ttft_first_routing {
                self.pick_min_ttft_load(workers)
            } else {
                self.pick_min_load_fair(workers, self.config.use_reported_load)
            };
        }
        let matched = self.match_prefix(ctx.model(), &block_hashes);
        let match_rate = matched.matched_blocks as f32 / block_hashes.len() as f32;
        tracing::debug!(
            model = %ctx.model(),
            hashing = if is_bigram { "bigram" } else { "unigram" },
            n_blocks = block_hashes.len(),
            matched_blocks = matched.matched_blocks,
            match_rate,
            cache_threshold = self.config.cache_threshold,
            "cache-aware-zmq match_prefix",
        );
        // Record the matched overlap into `sgl_router_overlap_blocks` before
        // the threshold branch, so the histogram captures the full
        // distribution — including low-overlap selections that fall back to
        // min-load. This is the quantitative signal that cache-aware routing
        // is matching prefixes at all.
        if let Some(m) = self.metrics.get() {
            m.observe_overlap_blocks(ctx.model().0.as_str(), matched.matched_blocks as u64);
        }
        if self.config.ttft_first_routing {
            let continuous_cache_score =
                self.config.ttft_score_mode == TtftScoreMode::PredictedTtft;
            let matched_urls: HashSet<&str> =
                if continuous_cache_score || match_rate > self.config.cache_threshold {
                    matched.worker_urls.iter().map(|url| url.as_str()).collect()
                } else {
                    HashSet::new()
                };
            let ttft_matched_blocks = if matched_urls.is_empty() {
                0
            } else {
                matched.matched_blocks
            };
            let chosen = self.select_ttft_first(
                workers,
                ctx,
                &block_hashes,
                ttft_matched_blocks,
                &matched_urls,
                tokens.len(),
                block_size as usize,
            );
            self.feed_route_history(ctx.model(), &chosen, &block_hashes);
            return chosen;
        }
        if match_rate <= self.config.cache_threshold || matched.worker_urls.is_empty() {
            tracing::debug!(
                model = %ctx.model(),
                match_rate,
                cache_threshold = self.config.cache_threshold,
                "cache-aware-zmq: overlap below threshold, falling back to min-load",
            );
            // Route-history feeding: even on a min-load fallback, record this
            // prefix against the worker we actually send it to, so the next
            // request sharing the prefix can match it. (No-op in zmq mode.)
            let chosen = self.pick_min_load_fair(workers, self.config.use_reported_load);
            self.feed_route_history(ctx.model(), &chosen, &block_hashes);
            return chosen;
        }
        // Among workers in the matched set, pick the lowest-load one.
        let matched_urls: HashSet<&str> =
            matched.worker_urls.iter().map(|url| url.as_str()).collect();
        let matched_workers: Vec<Arc<Worker>> = workers
            .iter()
            .filter(|w| matched_urls.contains(w.url.as_str()))
            .map(Arc::clone)
            .collect();
        let best_matched: Option<Arc<Worker>> =
            self.pick_min_load_fair(&matched_workers, self.config.use_reported_load);
        // Cache-hit load guard: even when a cache hit wins, the hit worker
        // may be individually backed up while the system as a whole still
        // looks balanced (so the imbalance fast-path above didn't fire).
        // Divert to the globally least-loaded worker when the hit worker
        // leads it past both thresholds. OFF by default (rel = INFINITY).
        let best_matched = best_matched.map(|hot| self.apply_hit_load_guard(hot, workers));
        let chosen = best_matched
            .or_else(|| self.pick_min_load_fair(workers, self.config.use_reported_load));
        if let Some(w) = &chosen {
            tracing::debug!(
                model = %ctx.model(),
                worker = %w.url,
                matched_blocks = matched.matched_blocks,
                "cache-aware-zmq: selected worker by cache overlap",
            );
        }
        // Route-history feeding: record this prefix against the chosen worker
        // so subsequent shared-prefix requests match it. No-op in zmq mode
        // (the worker's own ZMQ events own the tree there).
        self.feed_route_history(ctx.model(), &chosen, &block_hashes);
        chosen
    }
}

impl Policy for CacheAwareZmqPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        self.select_from_candidates(workers, ctx)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }

    fn logs_route_decisions(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
struct CacheMatch {
    matched_blocks: usize,
    worker_urls: HashSet<String>,
}

#[derive(Debug, Clone)]
struct GroupScore {
    root_index: usize,
    best_score: usize,
    best_matched_blocks: usize,
}

impl CacheMatch {
    fn from_local(matched: crate::policies::kv_events::tree::MatchResult) -> Self {
        Self {
            matched_blocks: matched.matched_blocks,
            worker_urls: matched.workers.into_iter().map(|w| w.url).collect(),
        }
    }

    fn from_remote(resp: CacheStateMatchResponse) -> Self {
        Self {
            matched_blocks: resp.matched_blocks,
            worker_urls: resp.workers.into_iter().map(|w| w.worker_url).collect(),
        }
    }

    fn is_useful(&self) -> bool {
        self.matched_blocks > 0 && !self.worker_urls.is_empty()
    }
}

fn matched_blocks_for_worker(
    worker: &Worker,
    matched_blocks: usize,
    matched_urls: &HashSet<&str>,
) -> usize {
    if matched_urls.contains(worker.url.as_str())
        || worker
            .prefill_members()
            .iter()
            .any(|member| matched_urls.contains(member.as_str()))
    {
        matched_blocks
    } else {
        0
    }
}

fn grouped_worker_scores<F>(workers: &[Arc<Worker>], mut score_fn: F) -> Vec<GroupScore>
where
    F: FnMut(&Worker) -> (usize, usize),
{
    let mut owner_by_member: HashMap<&str, usize> = HashMap::new();
    for (idx, worker) in workers.iter().enumerate() {
        for member in worker.prefill_members() {
            owner_by_member.entry(member.as_str()).or_insert(idx);
        }
    }

    let mut groups: HashMap<usize, GroupScore> = HashMap::new();
    for (idx, worker) in workers.iter().enumerate() {
        let root_index = owner_by_member
            .get(worker.url.as_str())
            .copied()
            .unwrap_or(idx);
        let (score, matched_blocks) = score_fn(worker);
        groups
            .entry(root_index)
            .and_modify(|group| {
                if score < group.best_score
                    || (score == group.best_score && matched_blocks > group.best_matched_blocks)
                {
                    group.best_score = score;
                    group.best_matched_blocks = matched_blocks;
                }
            })
            .or_insert(GroupScore {
                root_index,
                best_score: score,
                best_matched_blocks: matched_blocks,
            });
    }

    groups.into_values().collect()
}

fn normalize_prefill_work(prefill_work_tokens: usize, capacity_milli: usize) -> usize {
    let capacity_milli = capacity_milli.max(1);
    prefill_work_tokens
        .saturating_mul(1000)
        .saturating_add(capacity_milli - 1)
        / capacity_milli
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_state::{
        CacheStateInsertRequest, CacheStateMatchRequest, CacheStateWorkerMatch,
    };
    use crate::config::CacheAwareConfig;
    use crate::discovery::{ModelId, WorkerBackend, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::kv_events::tree::KvWorkerId;
    use crate::policies::kv_events::HashTree;
    use crate::router_state::{
        RouterStateLoadOverlay, RouterStateSnapshotResponse, RouterStateWorkerLoad,
    };
    use crate::tokenizer::adapter;
    use crate::workers::worker::{
        CandidatePrefillLoad, MemberPrefillLoadSnapshot, PrefillLoadRole, PrefillLoadSnapshot,
        PrefillPriorityLoad,
    };

    async fn start_cache_state_service(
        service: Arc<crate::cache_state::CacheStateService>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let app = service.router();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), server)
    }

    fn tiny_ids_and_hashes(registry: &TokenizerRegistry, text: &str) -> (Vec<u32>, Vec<i64>) {
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        (ids, hashes)
    }

    fn ttft_remote_policy(
        registry: Arc<TokenizerRegistry>,
        tree: Arc<HashTree>,
        client: Arc<RemoteCacheStateClient>,
        metrics: Arc<MetricsRegistry>,
    ) -> CacheAwareZmqPolicy {
        ttft_remote_policy_with_config(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: 0,
            },
            registry,
            tree,
            client,
            metrics,
        )
    }

    fn ttft_remote_policy_with_config(
        config: CacheAwareConfig,
        registry: Arc<TokenizerRegistry>,
        tree: Arc<HashTree>,
        client: Arc<RemoteCacheStateClient>,
        metrics: Arc<MetricsRegistry>,
    ) -> CacheAwareZmqPolicy {
        CacheAwareZmqPolicy::new(config, tree, registry, oracle_for_tests(4))
            .with_remote_cache_state(client)
            .with_metrics(metrics)
    }

    fn cfg_default() -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            hit_load_abs_threshold: 0,
            hit_load_rel_threshold: f32::INFINITY,
            use_reported_load: false,
            tree_source: CacheTreeSource::Zmq,
            ttft_first_routing: false,
            ttft_score_mode: Default::default(),
            ttft_idle_first_routing: false,
            ttft_token_scale: 64,
            ttft_cache_score_margin: 0,
        }
    }

    /// Helper: build a `BlockSizeOracle` already primed to the test's
    /// canonical block size (4). Mirrors what `KvEventIndex::add_worker`
    /// would do when the first real worker registers.
    fn oracle_for_tests(block_size: u32) -> Arc<BlockSizeOracle> {
        let o = BlockSizeOracle::new();
        o.try_set(block_size)
            .expect("fresh oracle accepts first set");
        o
    }

    fn worker(url: &str, model_id: &str) -> Arc<Worker> {
        worker_with_backend(url, model_id, WorkerBackend::Sglang)
    }

    fn worker_with_backend(url: &str, model_id: &str, backend: WorkerBackend) -> Arc<Worker> {
        worker_with_backend_and_capacity(url, model_id, backend, 1000)
    }

    fn worker_with_backend_and_capacity(
        url: &str,
        model_id: &str,
        backend: WorkerBackend,
        prefill_capacity_milli: usize,
    ) -> Arc<Worker> {
        worker_with_prefill_members(url, model_id, backend, prefill_capacity_milli, Vec::new())
    }

    fn worker_with_prefill_members(
        url: &str,
        model_id: &str,
        backend: WorkerBackend,
        prefill_capacity_milli: usize,
        prefill_members: Vec<String>,
    ) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(model_id.into())],
            bootstrap_port: None,
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend,
            tier: Default::default(),
            routes: crate::discovery::WorkerRouteSet::all(),
            prefill_capacity_milli,
            prefill_members,
        }))
    }

    fn worker_with_router_state_overlay(
        url: &str,
        model_id: &str,
        overlay: Arc<RouterStateLoadOverlay>,
    ) -> Arc<Worker> {
        let mut worker = Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(model_id.into())],
            bootstrap_port: None,
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend: Default::default(),
            tier: Default::default(),
            routes: crate::discovery::WorkerRouteSet::all(),
            prefill_capacity_milli: 1000,
            prefill_members: Vec::new(),
        });
        worker.attach_router_state_overlay(overlay);
        Arc::new(worker)
    }

    fn tokenizer_registry_with_tiny() -> Arc<TokenizerRegistry> {
        let cfg = crate::config::Config {
            runtime_mode: crate::config::RuntimeMode::Gateway,
            server: crate::config::ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                policy: crate::config::PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                tiered_spillover: None,
                sticky: None,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                    bearer_keys: Vec::new(),
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
            trace: crate::config::TraceConfig::default(),
            priority_override: crate::config::PriorityOverrideConfig::default(),
            worker_introspect_key: None,
            load_poll_interval_secs: None,
            cache_tree_page_size: None,
            cache_tree_bigram: false,
            cache_tree_max_nodes: 1_000_000,
            cache_state_url: None,
            cache_state_timeout_ms: 20,
            alias_fallback: None,
            external_model: None,
        };
        Arc::new(TokenizerRegistry::load_from_config(&cfg).expect("load tiny tokenizer"))
    }

    /// Empty workers list returns None (parity with other policies).
    #[test]
    fn empty_workers_returns_none() {
        let tree = Arc::new(HashTree::new());
        let policy = CacheAwareZmqPolicy::new(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(b"{\"prompt\":\"hi\"}"));
        assert!(policy.select(&[], &ctx).is_none());
    }

    /// Empty tree: no overlap signal anywhere, fall through to min-load.
    #[test]
    fn empty_tree_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = CacheAwareZmqPolicy::new(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0's load so min-load picks w1 deterministically.
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello world"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Empty tree and equal load: fallback traffic should not stick to one
    /// stable worker. This is the cold-prefix path for new/underused hosts.
    #[test]
    fn empty_tree_equal_load_rotates_min_load_fallback() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let policy = CacheAwareZmqPolicy::new(
            cfg_default(),
            tree,
            Arc::clone(&registry),
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
            worker("http://w2:30000", "tiny"),
        ];
        let (ids, _) = tiny_ids_and_hashes(&registry, "hello world hello world hello world");
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let picks: Vec<String> = (0..6)
            .map(|_| {
                policy
                    .select(&workers, &ctx)
                    .expect("must pick")
                    .url
                    .clone()
            })
            .collect();

        assert_eq!(
            picks,
            vec![
                "http://w0:30000",
                "http://w1:30000",
                "http://w2:30000",
                "http://w0:30000",
                "http://w1:30000",
                "http://w2:30000",
            ],
        );
    }

    /// Tree contains w0's prefix; cache-aware selection picks w0 even
    /// though w1 has lower load (the load skew is below the imbalance
    /// threshold, so cache wins).
    #[test]
    fn non_empty_tree_highest_overlap_wins() {
        let tree = Arc::new(HashTree::new());
        // Insert w0's tokens into the tree. The tiny tokenizer's hash
        // chain for our input is whatever `compute_block_hashes` returns;
        // we mimic the policy's hashing path so the test stays
        // deterministic against tokenizer changes.
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world"; // longer → more blocks
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(
            !hashes.is_empty(),
            "tiny tokenizer must produce at least one full block",
        );
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0, // any match counts
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// The cache-aware path records the matched prefix-overlap block count
    /// into `sgl_router_overlap_blocks`. Regression: the metric was defined
    /// but never observed in production, so the histogram stayed empty and
    /// gave no signal that cache-aware routing was matching anything.
    #[test]
    fn records_overlap_blocks_metric() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let metrics = MetricsRegistry::new();
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));

        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let _ = policy.select(&workers, &ctx).expect("must pick");

        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "overlap_blocks histogram must be observed on a cache-aware selection; got:\n{rendered}"
        );
    }

    /// Production wiring path: the policy is stored as `Arc<dyn Policy>` in a
    /// `PolicyRegistry`, then `PolicyRegistry::attach_metrics` injects the
    /// registry — exactly what `AppContext::with_active_load` does at startup.
    /// Exercises trait dispatch (the default no-op vs the `CacheAwareZmqPolicy`
    /// override) and the registry fan-out, neither of which the `with_metrics`
    /// builder test covers.
    #[test]
    fn attach_metrics_via_registry_records_overlap() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            toks,
            oracle_for_tests(4),
        );
        let model = ModelId("tiny".into());
        let registry = crate::policies::PolicyRegistry::default();
        registry.insert(model.clone(), Arc::new(policy));

        // The production injection point — not the `with_metrics` builder.
        let metrics = MetricsRegistry::new();
        registry.attach_metrics(Arc::clone(&metrics));

        let chosen_policy = registry.get(&model).unwrap();
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let _ = chosen_policy.select(&workers, &ctx).expect("must pick");

        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "PolicyRegistry::attach_metrics must wire overlap recording through the trait; got:\n{rendered}"
        );
    }

    /// The overlap observation is recorded *before* the cache-threshold branch,
    /// so low-overlap selections that fall back to min-load are still counted.
    /// `cache_threshold: 1.0` forces the fallback (match_rate is always <= 1.0)
    /// even on a full prefix match; assert the histogram is still observed AND
    /// the pick came from min-load (w1), not the cache-overlap worker (w0).
    #[test]
    fn overlap_recorded_even_when_selection_falls_back() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let metrics = MetricsRegistry::new();
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 1.0, // match_rate <= 1.0 always -> always fall back
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            toks,
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));

        // Bump w0's load so min-load picks w1 — distinguishing a min-load
        // fallback from the cache-overlap pick (which would be w0). Two guards
        // mirror `empty_tree_falls_back_to_min_load` (below the imbalance
        // threshold, so the cache-aware path is still reached).
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w1:30000",
            "cache_threshold 1.0 must force a min-load fallback (w1), not the overlap worker (w0)"
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "overlap must be recorded even on the below-threshold fallback; got:\n{rendered}"
        );
    }

    /// End-to-end bigram wiring (the fix that takes `overlap_blocks_sum` from
    /// 0 to non-zero for EAGLE models): an EAGLE worker publishes its blocks
    /// under BIGRAM hashes. Only a router whose oracle reports `is_bigram` —
    /// and thus hashes its query with the bigram hasher — matches them, so
    /// overlap is non-zero and it picks the cached worker. A unigram-hashing
    /// router against the SAME tree matches nothing (overlap recorded as 0).
    #[test]
    fn bigram_routing_matches_only_with_bigram_hashing() {
        fn overlap_sum(rendered: &str) -> f64 {
            rendered
                .lines()
                .find(|l| l.starts_with("sgl_router_overlap_blocks_sum{model_id=\"tiny\"}"))
                .and_then(|l| l.split_whitespace().last())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(-1.0)
        }

        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        // The EAGLE worker publishes BIGRAM block hashes.
        let bigram_hashes = compute_block_hashes_bigram(&ids, block_size as usize);
        assert!(!bigram_hashes.is_empty());
        assert_ne!(
            bigram_hashes,
            compute_block_hashes(&ids, block_size as usize),
            "bigram and unigram hashes must differ for this prefix"
        );
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();

        // Bigram-aware router (oracle.is_bigram == true): query hashes match
        // the bigram tree -> overlap > 0 and it picks the matched worker w0.
        {
            let tree = Arc::new(HashTree::new());
            tree.insert(
                &KvWorkerId::new("http://w0:30000".into(), 0),
                None,
                &bigram_hashes,
            );
            let oracle = BlockSizeOracle::new();
            oracle.try_set(block_size).unwrap();
            oracle.set_bigram(true);
            let metrics = MetricsRegistry::new();
            let policy = CacheAwareZmqPolicy::new(
                CacheAwareConfig {
                    cache_threshold: 0.0,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    hit_load_abs_threshold: 0,
                    hit_load_rel_threshold: f32::INFINITY,
                    use_reported_load: false,
                    tree_source: CacheTreeSource::Zmq,
                    ..CacheAwareConfig::default()
                },
                tree,
                Arc::clone(&registry),
                oracle,
            )
            .with_metrics(Arc::clone(&metrics));
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            let chosen = policy.select(&workers, &ctx).expect("must pick");
            assert_eq!(
                chosen.url, "http://w0:30000",
                "bigram-aware router must match w0's bigram-hashed prefix"
            );
            assert!(
                overlap_sum(&metrics.render()) > 0.0,
                "overlap_blocks_sum must be > 0 once the router hashes with bigram"
            );
        }

        // Unigram router (default is_bigram == false) vs the SAME bigram tree:
        // query hashes never match -> overlap recorded as 0.
        {
            let tree = Arc::new(HashTree::new());
            tree.insert(
                &KvWorkerId::new("http://w0:30000".into(), 0),
                None,
                &bigram_hashes,
            );
            let oracle = BlockSizeOracle::new();
            oracle.try_set(block_size).unwrap();
            let metrics = MetricsRegistry::new();
            let policy = CacheAwareZmqPolicy::new(
                CacheAwareConfig {
                    cache_threshold: 0.0,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    hit_load_abs_threshold: 0,
                    hit_load_rel_threshold: f32::INFINITY,
                    use_reported_load: false,
                    tree_source: CacheTreeSource::Zmq,
                    ..CacheAwareConfig::default()
                },
                tree,
                Arc::clone(&registry),
                oracle,
            )
            .with_metrics(Arc::clone(&metrics));
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            let _ = policy.select(&workers, &ctx).expect("must pick");
            assert_eq!(
                overlap_sum(&metrics.render()),
                0.0,
                "unigram hashing matches nothing in a bigram tree -> overlap_sum == 0"
            );
        }
    }

    /// A chat-completions request on a model with a chat template must route by
    /// the **chat-templated** tokens (BOS + role markers + content) — the tokens
    /// the engine actually cached — not by the raw joined content. Worker w0
    /// published its blocks under the templated tokens; only a router that
    /// renders the same template hashes a matching query. Hashing the raw
    /// content instead would match nothing, leaving live `overlap_blocks_sum`
    /// at 0 for chat traffic.
    #[test]
    fn chat_request_routes_by_templated_tokens() {
        let registry = tokenizer_registry_with_tiny();
        let template = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}<|assistant|>",
            "bos_token": "<s>",
        });
        registry.attach_chat_template_for_test("tiny", &template);

        let messages = serde_json::json!([{"role":"user","content":"hello world hello world"}]);
        // Engine-side blocks are keyed on tokenize(render(messages)).
        let templated_tokens = registry.encode_chat("tiny", &messages).unwrap();
        let block_size = 4u32;
        let templated_hashes = compute_block_hashes(&templated_tokens, block_size as usize);
        assert!(
            !templated_hashes.is_empty(),
            "templated prompt must produce at least one block"
        );

        let tree = Arc::new(HashTree::new());
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &templated_hashes,
        );

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(block_size),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": messages,
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "chat request must route by chat-templated tokens to the worker holding that prefix"
        );
    }

    /// Templated and raw-content hashings must genuinely differ, confirming
    /// the chat-template path does real work (a no-op template would make this
    /// assertion fail, and raw-content hashes would miss the engine's
    /// templated blocks).
    #[test]
    fn chat_templated_hashes_differ_from_raw_content_hashes() {
        let registry = tokenizer_registry_with_tiny();
        let template = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}<|assistant|>",
            "bos_token": "<s>",
        });
        registry.attach_chat_template_for_test("tiny", &template);
        let content = "hello world hello world";
        let messages = serde_json::json!([{"role":"user","content":content}]);

        let templated = registry.encode_chat("tiny", &messages).unwrap();
        let raw = adapter::encode(&registry.get("tiny").unwrap(), content).unwrap();
        assert_ne!(
            compute_block_hashes(&templated, 4),
            compute_block_hashes(&raw, 4),
            "templated and raw-content block hashes must differ"
        );
    }

    /// The DeepSeek-V4 built-in encoder is dispatched for chat requests when a
    /// model has it (no Jinja template). The query tokens come from the V4
    /// encoder, so a worker holding that encoded prefix is matched. (The V4
    /// markers aren't special tokens in the tiny fixture, but the dispatch +
    /// routing wiring is what's under test; byte-exact V4 token parity is pinned
    /// by `dsv4`'s string goldens and validated live.)
    #[test]
    fn chat_request_routes_via_dsv4_encoder() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_encoder_for_test("tiny", crate::tokenizer::ChatEncoder::DeepSeekV4);
        assert!(registry.has_chat_encoder("tiny"));

        let messages =
            serde_json::json!([{"role":"user","content":"hello world hello world hello world"}]);
        let encoded = registry.encode_chat("tiny", &messages).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&encoded, block_size as usize);
        assert!(!hashes.is_empty());

        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(block_size),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "messages": messages })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "dsv4 chat request must route by the V4-encoded prefix"
        );
    }

    /// Helper: a tree holding `content`'s RAW-tokenized block hashes on w0, the
    /// two workers, and a policy — the fixture the raw-fallback routing tests
    /// share. Returns (policy, workers, model).
    fn raw_prefix_fixture(
        registry: Arc<TokenizerRegistry>,
        content: &str,
    ) -> (CacheAwareZmqPolicy, Vec<Arc<Worker>>, ModelId) {
        let raw_tokens = adapter::encode(&registry.get("tiny").unwrap(), content).unwrap();
        let hashes = compute_block_hashes(&raw_tokens, 4);
        assert!(
            !hashes.is_empty(),
            "raw content must produce at least one block"
        );
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        (policy, workers, ModelId("tiny".into()))
    }

    /// Graceful degradation: a model that HAS a chat template whose render fails
    /// (here it always raises) must fall back to hashing the RAW content and
    /// still route by prefix — not error, not blindly min-load. Exercises the
    /// `request_tokens_for` fall-through that the leaf `encode_chat`-returns-None
    /// tests don't reach at the routing level.
    #[test]
    fn chat_render_failure_falls_back_to_raw_routing() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ raise_exception('boom') }}",
                "bos_token": "<s>",
            }),
        );
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": [{"role": "user", "content": content}],
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "a failed template render must degrade to raw-content routing"
        );
    }

    /// A chat request on a model WITHOUT a chat template routes by the raw
    /// joined `messages[*].content` — the common config where the model ships
    /// no `chat_template`. Covers the `request_tokens_for` path that skips the
    /// template block entirely for a `messages` body.
    #[test]
    fn chat_on_template_less_model_routes_by_raw_content() {
        let registry = tokenizer_registry_with_tiny(); // no template attached
        assert!(!registry.has_chat_encoder("tiny"));
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": [{"role": "user", "content": content}],
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// A `/v1/completions` (`prompt`) request on a model that DOES have a chat
    /// template must still use the raw path — the template applies only to
    /// `messages` traffic. Guards the `messages`-presence gate in
    /// `request_tokens_for`.
    #[test]
    fn completions_prompt_on_templated_model_uses_raw_path() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
                "bos_token": "<s>",
            }),
        );
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        // `prompt` body (no `messages`) -> raw path, so it matches the raw tree.
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": content })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// Two workers both hold the prefix; the lower-load one wins.
    #[test]
    fn tie_break_by_lowest_active_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        // Both workers hold the prefix.
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0 to load=1; w1 is at 0 — tiebreak picks w1.
        let _g = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// w0 holds the prefix but is heavily overloaded → imbalance branch
    /// skips cache-aware and picks w1.
    #[test]
    fn imbalanced_pool_skips_cache_check() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0, // would normally always match
                balance_abs_threshold: 5,
                balance_rel_threshold: 2.0,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0 well above the imbalance threshold.
        let mut guards = Vec::new();
        for _ in 0..20 {
            guards.push(w0.load_guard());
        }
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000", "imbalance must dominate");
    }

    /// Tokenizer is missing for the requested model → fall back to
    /// min-load (no panic, no error).
    #[test]
    fn missing_tokenizer_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let empty_registry = Arc::new(TokenizerRegistry::default());
        let policy =
            CacheAwareZmqPolicy::new(cfg_default(), tree, empty_registry, oracle_for_tests(4));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Missing body → fall back to min-load.
    #[test]
    fn missing_request_body_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = CacheAwareZmqPolicy::new(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Body present but no recognizable prompt field → fall back.
    #[test]
    fn body_without_prompt_field_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = CacheAwareZmqPolicy::new(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"frobnicate":42}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Body has a non-text shape that yields zero tokens → fall back.
    /// (Tokenizer always returns ≥0 ids; an empty string yields the
    /// empty vec, then `compute_block_hashes` returns empty too.)
    #[test]
    fn empty_text_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = CacheAwareZmqPolicy::new(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":""}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Match rate below the threshold → fall back. Threshold = 0.99
    /// means the tree must match every single block; we insert an
    /// UNRELATED chain so the rate is 0.
    #[test]
    fn low_match_rate_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        // Tree contains a chain unrelated to the test's request.
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &[999, 998, 997],
        );

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.99,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello world hello world hello world"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Byte-slice helper over the shared `extract_prompt_text_from_value` free
    /// function, so the extraction-shape tests below stay terse.
    fn extract_prompt_text(body: &[u8]) -> Option<String> {
        let v: serde_json::Value = serde_json::from_slice(body).ok()?;
        crate::policies::extract_prompt_text_from_value(&v)
    }

    /// Chat completions shape with `messages[*].content` string.
    #[test]
    fn extract_prompt_chat_string_content() {
        let body = br#"{"model":"x","messages":[{"role":"user","content":"hello"}]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "hello");
    }

    /// Chat completions shape with multimodal content blocks (text parts).
    #[test]
    fn extract_prompt_chat_block_content() {
        let body = br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":"x"}]}]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "hi");
    }

    /// `/v1/completions` array form is joined with newlines.
    #[test]
    fn extract_prompt_completions_array() {
        let body = br#"{"prompt":["a","b","c"]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "a\nb\nc");
    }

    /// SGLang native `text` field.
    #[test]
    fn extract_prompt_sglang_text_field() {
        let body = br#"{"text":"abc"}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "abc");
    }

    /// Unknown shape → None.
    #[test]
    fn extract_prompt_unknown_shape_returns_none() {
        let body = br#"{"frobnicate":42}"#;
        assert!(extract_prompt_text(body).is_none());
    }

    /// Lifecycle: removing a worker from the tree via `clear_worker`
    /// makes subsequent matches miss; the policy then falls back to
    /// min-load.
    #[test]
    fn lifecycle_clear_worker_removes_overlap() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        let kw0 = KvWorkerId::new("http://w0:30000".into(), 0);
        tree.insert(&kw0, None, &hashes);

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree.clone(),
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();

        // Before clear: w0 wins.
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");

        // After clear: tree no longer attributes the prefix to w0.
        tree.clear_worker(&kw0);
        // Bump w0's load so min-load fallback distinguishes from w1.
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let chosen2 = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen2.url, "http://w1:30000");
    }

    /// `request_tokens_for` flags chat-encoder output as engine-equivalent (safe
    /// to forward to the engine as `input_ids`): the ids match what the engine
    /// tokenizes from its own chat template.
    #[test]
    fn request_tokens_chat_encoder_is_engine_equivalent() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
                "bos_token": "<s>",
            }),
        );
        let messages = serde_json::json!([{"role":"user","content":"hello world"}]);
        let expected = registry.encode_chat("tiny", &messages).unwrap();

        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "model": "tiny", "messages": messages });
        let rt = request_tokens_for(&registry, &model, &value).expect("tokens");
        assert!(
            rt.engine_equivalent,
            "chat-encoder ids must be engine-equivalent"
        );
        assert_eq!(rt.ids, expected);
    }

    /// `request_tokens_for` on the raw-prompt path (no chat encoder) is NOT
    /// engine-equivalent — the engine would still apply its template, so the
    /// router's raw ids must not be forwarded as `input_ids`.
    #[test]
    fn request_tokens_raw_prompt_not_engine_equivalent() {
        let registry = tokenizer_registry_with_tiny(); // no template attached
        assert!(!registry.has_chat_encoder("tiny"));
        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "prompt": "hello world" });
        let rt = request_tokens_for(&registry, &model, &value).expect("tokens");
        assert!(!rt.engine_equivalent);
        assert!(!rt.ids.is_empty());
    }

    /// `request_tokens_for` returns `None` when there is no routable prompt
    /// field — the handler then forwards nothing and the engine tokenizes as
    /// usual.
    #[test]
    fn request_tokens_none_for_unroutable_body() {
        let registry = tokenizer_registry_with_tiny();
        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "frobnicate": 42 });
        assert!(request_tokens_for(&registry, &model, &value).is_none());
    }

    /// `select` consumes the ingress-precomputed tokens and does NOT
    /// re-tokenize the body: the body here tokenizes to an unrelated prefix
    /// (which the tree does not hold), but the ctx tokens point at w0's cached
    /// prefix, so w0 wins. If `select` re-tokenized the body it would miss and
    /// fall back to min-load (w1).
    #[test]
    fn select_prefers_ingress_tokens_over_body() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let tree_ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&tree_ids, 4);
        assert!(!hashes.is_empty());
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Load w0 so a min-load fallback would pick w1 — distinguishes "used
        // ctx tokens (w0)" from "re-tokenized the body and missed (w1)".
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        // Body tokenizes to an unrelated prefix the tree does NOT hold.
        let body = serde_json::to_vec(&serde_json::json!({"prompt":"zzz unrelated"})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body)).with_request_tokens(Some(&tree_ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "select must use ctx tokens (w0's prefix), not re-tokenize the body"
        );
    }

    // ---- cache-hit load guard ----

    /// Build a cache-aware-zmq policy whose tree is primed with `text`'s
    /// prefix on worker `hit_url`, with the global imbalance fast-path
    /// effectively disabled (huge balance_abs_threshold) so tests exercise
    /// the per-hit load guard in isolation. Returns (policy, tokens).
    fn guard_policy(
        text: &str,
        hit_url: &str,
        hit_load_abs_threshold: usize,
        hit_load_rel_threshold: f32,
    ) -> (CacheAwareZmqPolicy, Vec<u32>) {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty(), "need at least one full block");
        tree.insert(&KvWorkerId::new(hit_url.into(), 0), None, &hashes);
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,              // any overlap counts as a hit
                balance_abs_threshold: usize::MAX, // disable global fast-path
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold,
                hit_load_rel_threshold,
                use_reported_load: false,
                tree_source: CacheTreeSource::Zmq,
                ..CacheAwareConfig::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        (policy, ids)
    }

    /// Guard OFF (rel = INFINITY, the default): a backed-up cache worker is
    /// still chosen — behaviour identical to plain cache-aware.
    #[test]
    fn hit_load_guard_off_keeps_backed_up_cache_worker() {
        let text = "hello world hello world hello world";
        let (policy, ids) = guard_policy(text, "http://w0:30000", 0, f32::INFINITY);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Pile load on the cache worker; w1 stays idle.
        let _guards: Vec<_> = (0..10).map(|_| w0.load_guard()).collect();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "guard OFF keeps the cache hit"
        );
    }

    /// Guard ARMED as `min_load + 1`: a cache hit may be one request
    /// busier than the coolest worker, but not two.
    #[test]
    fn hit_load_guard_diverts_off_backed_up_worker() {
        let text = "hello world hello world hello world";
        let (policy, ids) = guard_policy(text, "http://w0:30000", 1, 1.0);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // w0 load 2, w1 load 0: gap 2 > slack 1 — divert.
        let _guards: Vec<_> = (0..2).map(|_| w0.load_guard()).collect();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "armed guard diverts to coolest"
        );
    }

    /// Guard ARMED but the hit worker is only one request above the coolest
    /// worker: keep the cache hit.
    #[test]
    fn hit_load_guard_below_abs_keeps_cache_worker() {
        let text = "hello world hello world hello world";
        let (policy, ids) = guard_policy(text, "http://w0:30000", 1, 1.0);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // w0 load 1, w1 load 0: gap 1 <= slack 1 → keep hit.
        let _guards: Vec<_> = (0..1).map(|_| w0.load_guard()).collect();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "below abs threshold keeps hit"
        );
    }

    /// Guard ARMED, abs gap exceeded, but the relative ratio is not: keep
    /// the cache hit. w0 load 10, w1 load 8 → gap 2 fails abs anyway, so use
    /// a high abs=1 with a steep rel to isolate the REL condition: gap 2 >
    /// abs 1, but 10 > 8*1.5 (=12) is false → REL fails → keep hit.
    #[test]
    fn hit_load_guard_below_rel_keeps_cache_worker() {
        let text = "hello world hello world hello world";
        let (policy, ids) = guard_policy(text, "http://w0:30000", 1, 1.5);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g0: Vec<_> = (0..10).map(|_| w0.load_guard()).collect();
        let _g1: Vec<_> = (0..8).map(|_| w1.load_guard()).collect();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000", "below rel ratio keeps hit");
    }

    /// Guard ARMED but the hit worker already IS the globally least-loaded
    /// one: nothing to divert to, keep it.
    #[test]
    fn hit_load_guard_hit_is_coolest_keeps_it() {
        let text = "hello world hello world hello world";
        let (policy, ids) = guard_policy(text, "http://w0:30000", 6, 1.2);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // w1 is the busy one; the cache hit (w0) is already coolest.
        let _g1: Vec<_> = (0..10).map(|_| w1.load_guard()).collect();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "hit already coolest, keep it"
        );
    }

    fn ttft_first_policy(
        text: &str,
        hit_url: &str,
        cache_score_margin: usize,
    ) -> (CacheAwareZmqPolicy, Vec<u32>) {
        ttft_first_policy_with_guard(text, hit_url, cache_score_margin, 0, f32::INFINITY)
    }

    fn ttft_first_policy_with_guard(
        text: &str,
        hit_url: &str,
        cache_score_margin: usize,
        hit_load_abs_threshold: usize,
        hit_load_rel_threshold: f32,
    ) -> (CacheAwareZmqPolicy, Vec<u32>) {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty(), "need at least one full block");
        tree.insert(&KvWorkerId::new(hit_url.into(), 0), None, &hashes);
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold,
                hit_load_rel_threshold,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: cache_score_margin,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        (policy, ids)
    }

    fn lmetric_policy(mode: TtftScoreMode) -> CacheAwareZmqPolicy {
        CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: mode,
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: 0,
            },
            Arc::new(HashTree::new()),
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        )
    }

    fn prefill_snapshot(
        running_requests: usize,
        total_waiting_uncached_tokens: usize,
        candidate: Option<CandidatePrefillLoad>,
    ) -> PrefillLoadSnapshot {
        PrefillLoadSnapshot {
            role: PrefillLoadRole::Integrated,
            running_requests,
            total_waiting_uncached_tokens,
            candidate,
        }
    }

    fn native_prefill_snapshot(
        running_requests: usize,
        total_waiting_uncached_tokens: usize,
    ) -> PrefillLoadSnapshot {
        PrefillLoadSnapshot {
            role: PrefillLoadRole::Prefill,
            running_requests,
            total_waiting_uncached_tokens,
            candidate: None,
        }
    }

    #[test]
    fn conservative_lmetric_multiplies_prefill_and_batch_factors() {
        let policy = lmetric_policy(TtftScoreMode::Lmetric);
        let w = worker("http://w0:30000", "tiny");
        w.set_reported_prefill_load(Some(prefill_snapshot(3, 1000, None)));
        let _reservation = w.pending_guard_with_tokens(50);

        let score = policy.ttft_score(&w, 25, 0, 100, 4, 0);

        assert_eq!(score, (100 + 1000 + 50) * (1 + 3 + 1));
    }

    #[test]
    fn candidate_aware_lmetric_counts_overdue_bucket_and_better_priority_only() {
        let policy = lmetric_policy(TtftScoreMode::LmetricCandidateAware);
        let w = worker("http://w0:30000", "tiny");
        w.set_reported_prefill_load(Some(prefill_snapshot(
            0,
            10_500,
            Some(CandidatePrefillLoad {
                chunked_remaining_uncached_tokens: 50,
                work_bucket_bounds: vec![256, 1024],
                priority_scheduling_enabled: true,
                schedule_low_priority_values_first: false,
                priorities: vec![
                    PrefillPriorityLoad {
                        priority: 0,
                        total_uncached_tokens: 10_000,
                        ahead_uncached_tokens: vec![100, 10_000],
                    },
                    PrefillPriorityLoad {
                        priority: 10,
                        total_uncached_tokens: 500,
                        ahead_uncached_tokens: vec![50, 500],
                    },
                ],
            }),
        )));

        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 0), 750);
        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 10), 200);
    }

    #[test]
    fn candidate_aware_lmetric_falls_back_to_conservative_total() {
        let policy = lmetric_policy(TtftScoreMode::LmetricCandidateAware);
        let w = worker("http://w0:30000", "tiny");
        w.set_reported_prefill_load(Some(prefill_snapshot(0, 900, None)));

        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 0), 1000);
    }

    #[test]
    fn lmetric_without_token_snapshot_falls_back_to_additive_score() {
        let policy = lmetric_policy(TtftScoreMode::Lmetric);
        let w = worker("http://w0:30000", "tiny");
        w.set_reported_load(2);

        assert_eq!(policy.ttft_score(&w, 10, 2, 100, 4, 0), 10);
    }

    #[test]
    fn mixed_snapshot_pool_falls_back_to_one_additive_unit_system() {
        let policy = lmetric_policy(TtftScoreMode::Lmetric);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_prefill_load(Some(prefill_snapshot(0, 100, None)));
        let workers = vec![w0, w1];

        assert_eq!(
            policy.compatible_score_mode(&workers),
            TtftScoreMode::Additive
        );
    }

    #[test]
    fn native_pd_prefill_still_falls_back_for_multiplicative_mode() {
        let policy = lmetric_policy(TtftScoreMode::Lmetric);
        let w = worker("http://w0:30000", "tiny");
        w.set_mode(WorkerMode::Prefill);
        w.set_reported_load(2);
        w.set_reported_prefill_load(Some(native_prefill_snapshot(3, 1000)));

        assert_eq!(
            policy.compatible_score_mode_for_worker(&w),
            TtftScoreMode::Additive
        );
        assert_eq!(policy.ttft_score(&w, 10, 2, 100, 4, 0), 10);
    }

    #[test]
    fn native_pd_prefill_work_only_ignores_running_and_request_reservations() {
        let policy = lmetric_policy(TtftScoreMode::PrefillWorkOnly);
        let w = worker("http://w0:30000", "tiny");
        w.set_mode(WorkerMode::Prefill);
        w.set_reported_prefill_load(Some(native_prefill_snapshot(99, 1000)));
        let _reservation = w.pending_guard_with_tokens(50);

        assert_eq!(
            policy.compatible_score_mode_for_worker(&w),
            TtftScoreMode::PrefillWorkOnly
        );
        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 0), 1150);
    }

    #[test]
    fn prefill_work_normalized_scales_by_worker_capacity() {
        let policy = lmetric_policy(TtftScoreMode::PrefillWorkNormalized);
        let baseline = worker_with_backend_and_capacity(
            "http://b300:30000",
            "tiny",
            WorkerBackend::Sglang,
            1000,
        );
        let mi300x = worker_with_backend_and_capacity(
            "http://mi300x:30000",
            "tiny",
            WorkerBackend::Sglang,
            500,
        );
        let france = worker_with_backend_and_capacity(
            "http://france:30000",
            "tiny",
            WorkerBackend::Sglang,
            5000,
        );
        for worker in [&baseline, &mi300x, &france] {
            worker.set_mode(WorkerMode::Prefill);
            worker.set_reported_prefill_load(Some(native_prefill_snapshot(99, 10_000)));
        }

        assert_eq!(
            policy.compatible_score_mode_for_worker(&baseline),
            TtftScoreMode::PrefillWorkNormalized
        );
        assert_eq!(policy.ttft_score(&baseline, 25, 0, 60_000, 4, 0), 70_000);
        assert_eq!(policy.ttft_score(&mi300x, 25, 0, 60_000, 4, 0), 140_000);
        assert_eq!(policy.ttft_score(&france, 25, 0, 60_000, 4, 0), 14_000);
    }

    #[test]
    fn predicted_ttft_uses_reservations_without_token_snapshot() {
        let policy = lmetric_policy(TtftScoreMode::PredictedTtft);
        let baseline = worker_with_backend_and_capacity(
            "http://baseline:30000",
            "tiny",
            WorkerBackend::Sglang,
            1000,
        );
        let fast = worker_with_backend_and_capacity(
            "http://fast-proxy:30000",
            "tiny",
            WorkerBackend::SglangProxy,
            2000,
        );
        let _baseline_reservation = baseline.pending_guard_with_tokens(600);
        let _fast_reservation = fast.pending_guard_with_tokens(600);

        assert_eq!(
            policy.compatible_score_mode(&[Arc::clone(&baseline), Arc::clone(&fast)]),
            TtftScoreMode::PredictedTtft,
        );
        assert_eq!(policy.ttft_score(&baseline, 25, 0, 100, 4, 0), 700);
        assert_eq!(policy.ttft_score(&fast, 25, 0, 100, 4, 0), 350);
    }

    #[test]
    fn predicted_ttft_does_not_double_count_snapshot_and_reservation() {
        let policy = lmetric_policy(TtftScoreMode::PredictedTtft);
        let w = worker("http://w0:30000", "tiny");
        w.set_reported_prefill_load(Some(prefill_snapshot(1, 1000, None)));
        let _reservation = w.pending_guard_with_tokens(600);

        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 0), 1100);
    }

    #[test]
    fn predicted_ttft_does_not_double_count_local_and_global_reservation() {
        let policy = lmetric_policy(TtftScoreMode::PredictedTtft);
        let overlay = RouterStateLoadOverlay::new();
        overlay.update(RouterStateSnapshotResponse {
            workers: [(
                "http://w0:30000".to_string(),
                RouterStateWorkerLoad {
                    pending_requests: 1,
                    pending_tokens: 600,
                },
            )]
            .into_iter()
            .collect(),
        });
        let w = worker_with_router_state_overlay("http://w0:30000", "tiny", overlay);
        let _reservation = w.pending_guard_with_tokens(600);

        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 0), 700);
    }

    #[test]
    fn predicted_ttft_uses_priority_aware_work_ahead() {
        let policy = lmetric_policy(TtftScoreMode::PredictedTtft);
        let w = worker("http://w0:30000", "tiny");
        w.set_reported_prefill_load(Some(prefill_snapshot(
            1,
            10_500,
            Some(CandidatePrefillLoad {
                chunked_remaining_uncached_tokens: 50,
                work_bucket_bounds: vec![256, 1024],
                priority_scheduling_enabled: true,
                schedule_low_priority_values_first: false,
                priorities: vec![
                    PrefillPriorityLoad {
                        priority: 0,
                        total_uncached_tokens: 10_000,
                        ahead_uncached_tokens: vec![100, 10_000],
                    },
                    PrefillPriorityLoad {
                        priority: 10,
                        total_uncached_tokens: 500,
                        ahead_uncached_tokens: vec![50, 500],
                    },
                ],
            }),
        )));

        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 0), 750);
        assert_eq!(policy.ttft_score(&w, 25, 0, 100, 4, 10), 200);
    }

    #[test]
    fn predicted_ttft_scores_logical_proxy_per_prefill_member() {
        let policy = lmetric_policy(TtftScoreMode::PredictedTtft);
        let logical = worker_with_prefill_members(
            "http://logical:30000",
            "tiny",
            WorkerBackend::SglangProxy,
            1000,
            vec!["http://p0:30000".into(), "http://p1:30000".into()],
        );
        logical.set_reported_prefill_members(vec![
            MemberPrefillLoadSnapshot {
                worker_url: "http://p0:30000".into(),
                prefill_capacity_milli: 1000,
                snapshot: prefill_snapshot(0, 10_000, None),
            },
            MemberPrefillLoadSnapshot {
                worker_url: "http://p1:30000".into(),
                prefill_capacity_milli: 2000,
                snapshot: prefill_snapshot(0, 100, None),
            },
        ]);
        let matched_urls = HashSet::from(["http://p0:30000"]);

        let estimate =
            policy.predicted_ttft_estimate_for_worker(&logical, 20, &matched_urls, 100, 4, 0);

        assert_eq!(
            estimate.selected_prefill_member.as_deref(),
            Some("http://p1:30000")
        );
        assert_eq!(estimate.matched_blocks, 0);
        assert_eq!(estimate.candidate_uncached_tokens, 100);
        assert_eq!(estimate.total_work_tokens, 200);
        assert_eq!(estimate.normalized_score, 100);
    }

    #[test]
    fn predicted_ttft_joins_cache_credit_to_the_matching_member_snapshot() {
        let policy = lmetric_policy(TtftScoreMode::PredictedTtft);
        let logical = worker_with_prefill_members(
            "http://logical:30000",
            "tiny",
            WorkerBackend::SglangProxy,
            1000,
            vec!["http://p0:30000".into(), "http://p1:30000".into()],
        );
        logical.set_reported_prefill_members(vec![
            MemberPrefillLoadSnapshot {
                worker_url: "http://p0:30000".into(),
                prefill_capacity_milli: 1000,
                snapshot: prefill_snapshot(0, 0, None),
            },
            MemberPrefillLoadSnapshot {
                worker_url: "http://p1:30000".into(),
                prefill_capacity_milli: 2000,
                snapshot: prefill_snapshot(0, 100, None),
            },
        ]);
        let matched_urls = HashSet::from(["http://p0:30000"]);

        let estimate =
            policy.predicted_ttft_estimate_for_worker(&logical, 20, &matched_urls, 100, 4, 0);

        assert_eq!(
            estimate.selected_prefill_member.as_deref(),
            Some("http://p0:30000")
        );
        assert_eq!(estimate.matched_blocks, 20);
        assert_eq!(estimate.candidate_uncached_tokens, 20);
        assert_eq!(estimate.normalized_score, 20);
    }

    #[test]
    fn predicted_ttft_credits_cache_below_configured_threshold() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let ids = vec![7u32; 12];
        let hashes = compute_block_hashes(&ids, 4);
        assert_eq!(hashes.len(), 3);
        tree.insert(
            &KvWorkerId::new("http://cached:30000".into(), 0),
            None,
            &hashes[..1],
        );
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.5,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: TtftScoreMode::PredictedTtft,
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: 0,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://cached:30000", "tiny"),
            worker("http://cold:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(chosen.url, "http://cached:30000");
    }

    #[test]
    fn predicted_ttft_is_not_overridden_by_request_count_hit_guard() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let ids = vec![7u32; 12];
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(
            &KvWorkerId::new("http://cached:30000".into(), 0),
            None,
            &hashes,
        );
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.5,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: 1.0,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: TtftScoreMode::PredictedTtft,
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: 0,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let cached = worker("http://cached:30000", "tiny");
        let cold = worker("http://cold:30000", "tiny");
        cached.set_reported_load(2);
        cold.set_reported_load(0);
        let workers = vec![cached, cold];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(chosen.url, "http://cached:30000");
    }

    #[test]
    fn predicted_ttft_does_not_select_probe_failed_cache_worker() {
        let policy = lmetric_policy(TtftScoreMode::PredictedTtft);
        let failed = worker("http://failed:30000", "tiny");
        let healthy = worker("http://healthy:30000", "tiny");
        failed.set_reported_load(crate::workers::worker::REPORTED_LOAD_FAILED);
        healthy.set_reported_load(0);

        assert_eq!(
            policy.ttft_score(&failed, 25, 25, 100, 4, 0),
            usize::MAX / 2,
        );
        assert_eq!(policy.ttft_score(&healthy, 25, 0, 100, 4, 0), 100);
    }

    #[test]
    fn prefill_work_only_rejects_role_mismatch() {
        let policy = lmetric_policy(TtftScoreMode::PrefillWorkOnly);
        let w = worker("http://w0:30000", "tiny");
        w.set_mode(WorkerMode::Prefill);
        w.set_reported_prefill_load(Some(prefill_snapshot(0, 1000, None)));

        assert_eq!(
            policy.compatible_score_mode_for_worker(&w),
            TtftScoreMode::Additive
        );
    }

    #[test]
    fn prefill_work_only_rejects_logical_pd_proxy_snapshot() {
        let policy = lmetric_policy(TtftScoreMode::PrefillWorkOnly);
        let w = worker_with_backend("http://w0:30000", "tiny", WorkerBackend::SglangProxy);
        w.set_reported_prefill_load(Some(prefill_snapshot(0, 1000, None)));

        assert_eq!(
            policy.compatible_score_mode_for_worker(&w),
            TtftScoreMode::Additive
        );
    }

    #[test]
    fn prefill_work_only_pool_falls_back_if_any_prefill_lacks_snapshot() {
        let policy = lmetric_policy(TtftScoreMode::PrefillWorkOnly);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        for worker in [&w0, &w1] {
            worker.set_mode(WorkerMode::Prefill);
        }
        w0.set_reported_prefill_load(Some(native_prefill_snapshot(0, 1000)));
        let workers = vec![w0, w1];

        assert_eq!(
            policy.compatible_score_mode(&workers),
            TtftScoreMode::Additive
        );
    }

    #[test]
    fn native_pd_prefill_work_only_selects_lower_token_backlog() {
        let policy = lmetric_policy(TtftScoreMode::PrefillWorkOnly);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        for worker in [&w0, &w1] {
            worker.set_mode(WorkerMode::Prefill);
        }
        w0.set_reported_prefill_load(Some(native_prefill_snapshot(10, 10_000)));
        w1.set_reported_prefill_load(Some(native_prefill_snapshot(1, 500)));
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ids = vec![7u32; 100];
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(chosen.url, w1.url);
    }

    #[test]
    fn matched_blocks_for_worker_credits_prefill_member_urls() {
        let logical = worker_with_prefill_members(
            "http://pd-proxy:30000",
            "tiny",
            WorkerBackend::SglangProxy,
            1000,
            vec!["http://prefill-0:30000".into()],
        );
        let unrelated = worker("http://other-proxy:30000", "tiny");
        let matched_urls = HashSet::from(["http://prefill-0:30000"]);

        assert_eq!(matched_blocks_for_worker(&logical, 3, &matched_urls), 3);
        assert_eq!(matched_blocks_for_worker(&unrelated, 3, &matched_urls), 0);
    }

    #[test]
    fn candidate_aware_selection_does_not_count_bypassable_long_work() {
        let policy = lmetric_policy(TtftScoreMode::LmetricCandidateAware);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let detail = |total, ahead| CandidatePrefillLoad {
            chunked_remaining_uncached_tokens: 0,
            work_bucket_bounds: vec![256, 1024],
            priority_scheduling_enabled: true,
            schedule_low_priority_values_first: false,
            priorities: vec![PrefillPriorityLoad {
                priority: 0,
                total_uncached_tokens: total,
                ahead_uncached_tokens: vec![ahead, total],
            }],
        };
        w0.set_reported_prefill_load(Some(prefill_snapshot(0, 10_000, Some(detail(10_000, 100)))));
        w1.set_reported_prefill_load(Some(prefill_snapshot(0, 500, Some(detail(500, 500)))));
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"priority": 0})).unwrap();
        let ids = vec![7u32; 100];
        let ctx = SelectionContext::new(&model, Some(&body)).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(chosen.url, w0.url);
    }

    #[test]
    fn ttft_first_no_cache_signal_rotates_equal_score_workers() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let (ids, _) = tiny_ids_and_hashes(&registry, "hello world hello world hello world");
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: 0,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
            worker("http://w2:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let picks: Vec<String> = (0..6)
            .map(|_| {
                policy
                    .select(&workers, &ctx)
                    .expect("must pick")
                    .url
                    .clone()
            })
            .collect();

        assert_eq!(
            picks,
            vec![
                "http://w0:30000",
                "http://w1:30000",
                "http://w2:30000",
                "http://w0:30000",
                "http://w1:30000",
                "http://w2:30000",
            ],
        );
    }

    #[test]
    fn ttft_first_cache_tie_rotates_without_spilling_to_cold_worker() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: usize::MAX,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
            worker("http://w2:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let picks: Vec<String> = (0..4)
            .map(|_| {
                policy
                    .select(&workers, &ctx)
                    .expect("must pick")
                    .url
                    .clone()
            })
            .collect();

        assert_eq!(
            picks,
            vec![
                "http://w0:30000",
                "http://w1:30000",
                "http://w0:30000",
                "http://w1:30000",
            ],
            "cache-equal workers should rotate, while the cold worker stays excluded",
        );
    }

    #[test]
    fn ttft_idle_first_prefers_idle_cold_worker_over_busy_cache_hit() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: true,
                ttft_token_scale: 4,
                ttft_cache_score_margin: usize::MAX,
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
            worker("http://w2:30000", "tiny"),
        ];
        workers[0].set_reported_load(4);
        workers[1].set_reported_load(0);
        workers[2].set_reported_load(0);
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let picks: Vec<String> = (0..4)
            .map(|_| {
                policy
                    .select(&workers, &ctx)
                    .expect("must pick")
                    .url
                    .clone()
            })
            .collect();

        assert_eq!(
            picks,
            vec![
                "http://w1:30000",
                "http://w2:30000",
                "http://w1:30000",
                "http://w2:30000",
            ],
        );
    }

    #[test]
    fn ttft_hit_guard_gap_two_diverts_to_idle_worker() {
        let text = "hello world hello world hello world";
        let (policy, ids) =
            ttft_first_policy_with_guard(text, "http://w0:30000", usize::MAX, 1, 1.0);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(2);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w1:30000",
            "a TTFT-load gap of two must divert a cache hit when the allowed slack is one",
        );
    }

    #[test]
    fn ttft_hit_guard_considers_idle_worker_with_shared_cache_match() {
        let text = "hello world hello world hello world";
        let (policy, _) = ttft_first_policy_with_guard(text, "http://w0:30000", usize::MAX, 1, 1.0);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(2);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let matched_urls = HashSet::from([w0.url.as_str(), w1.url.as_str()]);

        let chosen = policy.apply_ttft_hit_load_guard(Arc::clone(&w0), &workers, 1, &matched_urls);

        assert_eq!(
            chosen.url, "http://w1:30000",
            "an idle worker remains eligible when it shares a cached prefix",
        );
    }

    #[test]
    fn ttft_hit_guard_gap_one_keeps_cache_hit() {
        let text = "hello world hello world hello world";
        let (policy, ids) =
            ttft_first_policy_with_guard(text, "http://w0:30000", usize::MAX, 1, 1.0);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(1);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w0:30000",
            "a TTFT-load gap equal to the configured slack must keep the cache hit",
        );
    }

    #[test]
    fn ttft_hit_guard_off_preserves_cache_hit_behavior() {
        let text = "hello world hello world hello world";
        let (policy, ids) = ttft_first_policy(text, "http://w0:30000", usize::MAX);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(2);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w0:30000",
            "an infinite relative threshold keeps the pre-fix TTFT cache-hit behavior",
        );
    }

    #[test]
    fn ttft_first_diverts_from_cache_hit_outside_score_band() {
        let text = "hello world hello world hello world";
        let (policy, ids) = ttft_first_policy(text, "http://w0:30000", 0);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(0);
        w1.set_reported_load(0);
        let _long_prefill = w0.pending_guard_with_tokens(4096);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w1:30000",
            "TTFT-first must spill to the lower predicted score when cache hit is outside the band",
        );
    }

    #[test]
    fn ttft_first_keeps_cache_hit_inside_score_band() {
        let text = "hello world hello world hello world";
        let (policy, ids) = ttft_first_policy(text, "http://w0:30000", 2048);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(0);
        w1.set_reported_load(0);
        let _long_prefill = w0.pending_guard_with_tokens(4096);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w0:30000",
            "cache affinity can win only when its predicted TTFT score is inside the configured band",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_cache_state_match_drives_ttft_first_selection() {
        let service = Arc::new(crate::cache_state::CacheStateService::with_empty_tree());
        let (base_url, server) = start_cache_state_service(Arc::clone(&service)).await;

        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        service.insert(&CacheStateInsertRequest {
            model_id: "tiny".into(),
            worker_url: "http://w0:30000".into(),
            dp_rank: 0,
            parent_hash: None,
            block_hashes: hashes,
        });
        let client = Arc::new(RemoteCacheStateClient::new(
            base_url,
            std::time::Duration::from_millis(500),
        ));
        let metrics = MetricsRegistry::new();
        let policy = ttft_remote_policy(
            registry,
            Arc::new(HashTree::new()),
            client,
            Arc::clone(&metrics),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(0);
        w1.set_reported_load(1);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        server.abort();
        assert_eq!(chosen.url, "http://w0:30000");
        let rendered = metrics.render();
        assert!(
            rendered.contains(r#"sgl_router_remote_cache_state_query_total{outcome="hit"} 1"#),
            "remote hit must be counted; got:\n{rendered}",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_cache_state_member_match_selects_logical_pd_proxy() {
        let service = Arc::new(crate::cache_state::CacheStateService::with_empty_tree());
        let (base_url, server) = start_cache_state_service(Arc::clone(&service)).await;

        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        service.insert(&CacheStateInsertRequest {
            model_id: "tiny".into(),
            worker_url: "http://prefill-0:30000".into(),
            dp_rank: 0,
            parent_hash: None,
            block_hashes: hashes,
        });
        let client = Arc::new(RemoteCacheStateClient::new(
            base_url,
            std::time::Duration::from_millis(500),
        ));
        let metrics = MetricsRegistry::new();
        let policy = ttft_remote_policy(
            registry,
            Arc::new(HashTree::new()),
            client,
            Arc::clone(&metrics),
        );
        let logical = worker_with_prefill_members(
            "http://pd-proxy:30000",
            "tiny",
            WorkerBackend::SglangProxy,
            1000,
            vec!["http://prefill-0:30000".into()],
        );
        let member = worker("http://prefill-0:30000", "tiny");
        let cold = worker("http://cold-proxy:30000", "tiny");
        logical.set_reported_load(10);
        member.set_reported_load(0);
        cold.set_reported_load(0);
        let workers = vec![Arc::clone(&logical), Arc::clone(&member), Arc::clone(&cold)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        server.abort();
        assert_eq!(
            chosen.url, "http://pd-proxy:30000",
            "cache credit should select the logical proxy; physical Prefill stays only in cache-state"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_cache_hit_guard_honors_redis_router_state_overlay() {
        let service = Arc::new(crate::cache_state::CacheStateService::with_empty_tree());
        let (base_url, server) = start_cache_state_service(Arc::clone(&service)).await;
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        service.insert(&CacheStateInsertRequest {
            model_id: "tiny".into(),
            worker_url: "http://w0:30000".into(),
            dp_rank: 0,
            parent_hash: None,
            block_hashes: hashes,
        });
        let client = Arc::new(RemoteCacheStateClient::new(
            base_url,
            std::time::Duration::from_millis(500),
        ));
        let policy = ttft_remote_policy_with_config(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 1,
                hit_load_rel_threshold: 1.0,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: usize::MAX,
            },
            registry,
            Arc::new(HashTree::new()),
            client,
            MetricsRegistry::new(),
        );

        // This is the same overlay populated by the Redis router-state snapshot
        // poller in production. One pending request carries eight prompt tokens:
        // request-count load sees a gap of one, while TTFT load sees two units.
        let overlay = RouterStateLoadOverlay::new();
        overlay.update(RouterStateSnapshotResponse {
            workers: [(
                "http://w0:30000".to_string(),
                RouterStateWorkerLoad {
                    pending_requests: 1,
                    pending_tokens: 8,
                },
            )]
            .into_iter()
            .collect(),
        });
        let w0 = worker_with_router_state_overlay("http://w0:30000", "tiny", Arc::clone(&overlay));
        let w1 = worker_with_router_state_overlay("http://w1:30000", "tiny", overlay);
        w0.set_reported_load(0);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        server.abort();
        assert_eq!(
            chosen.url, "http://w1:30000",
            "remote cache affinity must yield to token-weighted cross-replica pressure",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ttft_hit_guard_route_history_records_final_worker() {
        let service = Arc::new(crate::cache_state::CacheStateService::with_empty_tree());
        let (base_url, server) = start_cache_state_service(Arc::clone(&service)).await;
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        service.insert(&CacheStateInsertRequest {
            model_id: "tiny".into(),
            worker_url: "http://w0:30000".into(),
            dp_rank: 0,
            parent_hash: None,
            block_hashes: hashes.clone(),
        });
        let client = Arc::new(RemoteCacheStateClient::new(
            base_url,
            std::time::Duration::from_millis(500),
        ));
        let local_tree = Arc::new(HashTree::new());
        let policy = ttft_remote_policy_with_config(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 1,
                hit_load_rel_threshold: 1.0,
                use_reported_load: true,
                tree_source: CacheTreeSource::RouteHistory,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: usize::MAX,
            },
            registry,
            Arc::clone(&local_tree),
            client,
            MetricsRegistry::new(),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(2);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");
        let local_match = local_tree.match_prefix(None, &hashes);

        server.abort();
        assert_eq!(chosen.url, "http://w1:30000");
        assert_eq!(local_match.matched_blocks, hashes.len());
        assert_eq!(local_match.workers.len(), 1);
        assert!(
            local_match
                .workers
                .iter()
                .any(|worker| worker.url == "http://w1:30000"),
            "route-history must record the post-guard worker, not the original cache hit",
        );
    }

    #[test]
    fn remote_cache_state_failure_degrades_to_load_fallback() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let client = Arc::new(RemoteCacheStateClient::new(
            "http://127.0.0.1:9".into(),
            std::time::Duration::from_millis(10),
        ));
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: true,
                tree_source: CacheTreeSource::Zmq,
                ttft_first_routing: true,
                ttft_score_mode: Default::default(),
                ttft_idle_first_routing: false,
                ttft_token_scale: 4,
                ttft_cache_score_margin: 0,
            },
            Arc::new(HashTree::new()),
            registry,
            oracle_for_tests(4),
        )
        .with_remote_cache_state(client);
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(5);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(chosen.url, "http://w1:30000");
    }

    #[test]
    fn remote_cache_state_failure_falls_back_to_local_tree() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let client = Arc::new(RemoteCacheStateClient::new(
            "http://127.0.0.1:9".into(),
            std::time::Duration::from_millis(10),
        ));
        let metrics = MetricsRegistry::new();
        let policy = ttft_remote_policy(registry, tree, client, Arc::clone(&metrics));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(0);
        w1.set_reported_load(1);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w0:30000",
            "local tree hit must survive remote cache-state failure",
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(
                r#"sgl_router_remote_cache_state_query_total{outcome="fallback_local_hit"} 1"#
            ),
            "fallback-local-hit must be counted; got:\n{rendered}",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_cache_state_empty_match_falls_back_to_local_tree() {
        let service = Arc::new(crate::cache_state::CacheStateService::with_empty_tree());
        let (base_url, server) = start_cache_state_service(service).await;
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let client = Arc::new(RemoteCacheStateClient::new(
            base_url,
            std::time::Duration::from_millis(500),
        ));
        let metrics = MetricsRegistry::new();
        let policy = ttft_remote_policy(registry, tree, client, Arc::clone(&metrics));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(0);
        w1.set_reported_load(1);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        server.abort();
        assert_eq!(
            chosen.url, "http://w0:30000",
            "empty remote cache-state response must fall back to local tree",
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(
                r#"sgl_router_remote_cache_state_query_total{outcome="fallback_local_hit"} 1"#
            ),
            "fallback-local-hit must be counted for empty remote response; got:\n{rendered}",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn authoritative_remote_miss_skips_stale_local_tree() {
        let service = Arc::new(
            crate::cache_state::CacheStateService::new_with_reconciliation(
                Arc::new(HashTree::new()),
                crate::cache_state::CacheStateReconciliationConfig {
                    enabled: true,
                    ..crate::cache_state::CacheStateReconciliationConfig::default()
                },
            ),
        );
        let (base_url, server) = start_cache_state_service(service).await;
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        let local_tree = Arc::new(HashTree::new());
        local_tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let client = Arc::new(RemoteCacheStateClient::new(
            base_url,
            std::time::Duration::from_millis(500),
        ));
        let metrics = MetricsRegistry::new();
        let policy = ttft_remote_policy(registry, local_tree, client, Arc::clone(&metrics));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        w0.set_reported_load(100);
        w1.set_reported_load(0);
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        server.abort();
        assert_eq!(
            chosen.url, "http://w1:30000",
            "an authoritative miss must use load routing, not stale local history",
        );
        assert!(metrics
            .render()
            .contains(r#"sgl_router_remote_cache_state_query_total{outcome="miss"} 1"#));
    }

    /// Route-history mode: the tree starts EMPTY (no ZMQ feed). The first
    /// request for a prefix has no match → min-load fallback, but the policy
    /// records the prefix against the chosen worker. A second, identical
    /// request must then match that same worker via the now-populated tree —
    /// proving the router-side feeding works without any ZMQ events.
    #[test]
    fn route_history_feeds_tree_and_matches_on_repeat() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        assert!(!compute_block_hashes(&ids, 4).is_empty());

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,              // any overlap counts as a hit
                balance_abs_threshold: usize::MAX, // disable imbalance fast-path
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY, // guard OFF
                use_reported_load: false,
                tree_source: CacheTreeSource::RouteHistory,
                ..CacheAwareConfig::default()
            },
            Arc::clone(&tree),
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());

        // Tree empty → first select is a min-load fallback, but it feeds the
        // tree with this prefix against whichever worker it picked.
        assert_eq!(tree.node_count(), 0, "tree starts empty (no ZMQ)");
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let first = policy.select(&workers, &ctx).expect("must pick");
        assert!(
            tree.node_count() > 0,
            "route-history must have fed the tree"
        );

        // Second identical request: now the tree has this prefix on `first`,
        // so the cache-overlap path must select the same worker.
        let ctx2 = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let second = policy.select(&workers, &ctx2).expect("must pick");
        assert_eq!(
            second.url, first.url,
            "repeat request for the same prefix must match the worker it was fed to",
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn route_history_feeds_remote_cache_state_service() {
        let service = Arc::new(crate::cache_state::CacheStateService::with_empty_tree());
        let (base_url, server) = start_cache_state_service(Arc::clone(&service)).await;
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, hashes) = tiny_ids_and_hashes(&registry, text);
        let expected_matched_blocks = hashes.len();
        let client = Arc::new(RemoteCacheStateClient::new(
            base_url,
            std::time::Duration::from_millis(500),
        ));
        let metrics = MetricsRegistry::new();
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::RouteHistory,
                ..CacheAwareConfig::default()
            },
            Arc::new(HashTree::new()),
            registry,
            oracle_for_tests(4),
        )
        .with_remote_cache_state(client)
        .with_metrics(Arc::clone(&metrics));
        let w0 = worker("http://w0:30000", "tiny");
        let workers = vec![Arc::clone(&w0)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(chosen.url, "http://w0:30000");
        let remote_match = service.match_prefix(&CacheStateMatchRequest {
            model_id: "tiny".into(),
            block_hashes: hashes,
        });
        server.abort();
        assert_eq!(remote_match.matched_blocks, expected_matched_blocks);
        assert_eq!(
            remote_match.workers,
            vec![CacheStateWorkerMatch {
                worker_url: "http://w0:30000".into(),
                dp_rank: 0,
            }]
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(r#"sgl_router_remote_cache_state_feed_total{outcome="success"} 1"#),
            "remote feed success must be counted; got:\n{rendered}",
        );
    }

    #[test]
    fn route_history_remote_feed_failure_is_non_fatal_and_counted() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let (ids, _) = tiny_ids_and_hashes(&registry, text);
        let client = Arc::new(RemoteCacheStateClient::new(
            "http://127.0.0.1:9".into(),
            std::time::Duration::from_millis(10),
        ));
        let metrics = MetricsRegistry::new();
        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: usize::MAX,
                balance_rel_threshold: f32::INFINITY,
                hit_load_abs_threshold: 0,
                hit_load_rel_threshold: f32::INFINITY,
                use_reported_load: false,
                tree_source: CacheTreeSource::RouteHistory,
                ..CacheAwareConfig::default()
            },
            Arc::clone(&tree),
            registry,
            oracle_for_tests(4),
        )
        .with_remote_cache_state(client)
        .with_metrics(Arc::clone(&metrics));
        let w0 = worker("http://w0:30000", "tiny");
        let workers = vec![Arc::clone(&w0)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));

        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(chosen.url, "http://w0:30000");
        assert!(
            tree.node_count() > 0,
            "local route-history feed must happen even when remote feed fails",
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(r#"sgl_router_remote_cache_state_feed_total{outcome="failure"} 1"#),
            "remote feed failure must be counted; got:\n{rendered}",
        );
    }

    /// Zmq mode must NOT feed the tree from routing decisions (the worker's
    /// own KV-event stream owns it). A select against an empty tree leaves it
    /// empty.
    #[test]
    fn zmq_mode_does_not_feed_tree_from_routing() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();

        let policy = CacheAwareZmqPolicy::new(
            cfg_default(), // tree_source = Zmq
            Arc::clone(&tree),
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let workers = vec![Arc::clone(&w0)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&ids));
        let _ = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            tree.node_count(),
            0,
            "zmq mode must not insert routing history into the tree",
        );
    }
}
