// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::Arc;

/// Deterministic load-based policy.
///
/// Chooses the candidate with the lowest worker-reported load plus local
/// pending reservations. Before the first successful load poll, it falls back
/// to router-local active load. Ties follow the candidate slice order, which
/// is registry-dependent.
#[derive(Debug, Default)]
pub struct LoadBasedPolicy;

impl LoadBasedPolicy {
    pub fn new() -> Self {
        Self
    }

    pub fn pick_min_load(workers: &[Arc<Worker>]) -> Option<Arc<Worker>> {
        workers
            .iter()
            .min_by_key(|w| w.effective_load(true))
            .map(Arc::clone)
    }
}

impl Policy for LoadBasedPolicy {
    fn select(&self, workers: &[Arc<Worker>], _ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        Self::pick_min_load(workers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::workers::worker::REPORTED_LOAD_FAILED;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
            min_priority: None,
            max_context_tokens: None,
            bearer_token: None,
            backend: Default::default(),
            tier: Default::default(),
            routes: crate::discovery::WorkerRouteSet::all(),
            prefill_capacity_milli: 1000,
            prefill_members: Vec::new(),
        }))
    }

    #[test]
    fn empty_returns_none() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert!(policy.select(&[], &ctx).is_none());
    }

    #[test]
    fn falls_back_to_lowest_active_load_before_first_poll() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _g0 = w0.load_guard();
        assert_eq!(
            policy.select(&[w0, Arc::clone(&w1)], &ctx).unwrap().id,
            w1.id
        );
    }

    #[test]
    fn picks_lowest_worker_reported_load() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let w0 = worker("w0");
        let w1 = worker("w1");
        w0.set_reported_load(1);
        w1.set_reported_load(0);
        assert_eq!(
            policy.select(&[w0, Arc::clone(&w1)], &ctx).unwrap().id,
            w1.id
        );
    }

    #[test]
    fn pending_reservation_breaks_reported_load_tie() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let w0 = worker("w0");
        let w1 = worker("w1");
        w0.set_reported_load(0);
        w1.set_reported_load(0);
        let _pending = w0.pending_guard();
        assert_eq!(
            policy.select(&[w0, Arc::clone(&w1)], &ctx).unwrap().id,
            w1.id
        );
    }

    #[test]
    fn avoids_failed_load_probe() {
        let policy = LoadBasedPolicy::new();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let w0 = worker("w0");
        let w1 = worker("w1");
        w0.set_reported_load(REPORTED_LOAD_FAILED);
        w1.set_reported_load(100);
        assert_eq!(
            policy.select(&[w0, Arc::clone(&w1)], &ctx).unwrap().id,
            w1.id
        );
    }
}
