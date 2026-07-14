## Why

The cache-state A/B replicas currently reconstruct worker prefix-cache state only from best-effort incremental events. A producer timeout or consumer crash window can therefore leave both replicas consistently wrong, because neither replica checks sequence continuity against a worker-authoritative digest or has a full snapshot from which to repair.

## What Changes

- Add a worker-authoritative cache epoch, sequence watermark, canonical state digest, and bounded snapshot format to the KV event publisher protocol.
- Let the local cache-event agent detect publisher gaps and transport digest/snapshot records without allowing a slow sink to block ZMQ ingestion.
- Track trust independently for every `worker_url + dp_rank`; sequence gaps or digest mismatches make only that worker rank ineligible for cache-hit routing while leaving it eligible for normal load-based inference routing.
- Reconcile an untrusted worker rank by atomically installing a verified snapshot at sequence `N`, then applying buffered incremental records newer than `N` before restoring trust.
- Commit Kafka/Event Hubs offsets only after a record has been decoded and applied successfully, and expose bounded metrics for gaps, digest comparisons, snapshot attempts, reconciliation latency, and trust state.
- Preserve the legacy event protocol and routing behavior unless reconciliation is explicitly enabled, so mixed-version rollout remains possible.

## Capabilities

### New Capabilities

- `worker-cache-state-reconciliation`: Worker-authoritative epochs, periodic digests, bounded snapshots, gap recovery, and verified snapshot transport.

### Modified Capabilities

- `distributed-cache-state`: Per-worker trust, fail-closed cache matches, atomic snapshot reconciliation, post-apply offset commits, and reconciliation observability.

## Impact

- Python SGLang scheduler/cache event publisher and its KV event wire types.
- Rust `cache-event-agent`, Kafka/Event Hubs stream envelope, cache-state service, HashTree ownership APIs, and remote match response handling.
- KV event configuration and metrics; new behavior remains opt-in for rollout compatibility.
- Unit and integration tests covering dropped events, digest mismatch, snapshot repair, replay ordering, consumer failure, mixed protocol versions, and bounded snapshot limits.
