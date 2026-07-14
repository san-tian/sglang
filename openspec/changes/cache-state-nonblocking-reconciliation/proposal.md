## Why

The reconciliation shadow cache-state replica becomes unresponsive under production Event Hubs backlog: synchronous record application and offset commit run on Tokio's serving runtime, while snapshot installation repeatedly scans unresolved entries and can approach quadratic work. ACA liveness probes then time out and restart the replica, allowing only a few offsets to advance per restart while backlog grows. The first non-blocking canary restored liveness, but production measurements then showed that a broker-confirmed commit per record caps consumption near 1.5 records/second while Event Hubs receives about 8 records/second, so the healthy replica still cannot reach a worker snapshot.

## What Changes

- Isolate Kafka record application from the asynchronous HTTP serving runtime, store an offset checkpoint only after successful application, and let librdkafka batch broker commits periodically.
- Install worker snapshots with a bounded linear-time dependency traversal instead of repeated whole-pending-list scans.
- Keep HTTP liveness responsive while cache records are decoded, reconciled, checkpointed, or committed, while preserving at-least-once post-apply offset semantics.
- Add regression tests for a single-thread Tokio runtime and reverse-ordered deep snapshots.
- Add record-level apply/checkpoint timing, periodic commit callback logs, and bounded-cardinality progress logs so slow records can be identified without exposing payloads or worker URLs as metric labels.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `distributed-cache-state`: Durable stream consumption must not starve cache-state HTTP liveness or serialize every record behind a broker round trip; offsets must still become eligible for commit only after successful application.
- `worker-cache-state-reconciliation`: Verified snapshot installation must remain bounded for deep or non-topologically ordered snapshots.

## Impact

- Rust cache-state Kafka consumer and cache-state HTTP event ingestion in `experimental/sgl-router/src/main.rs` and `cache_state/mod.rs`.
- Rust HashTree snapshot ordering in `experimental/sgl-router/src/policies/kv_events/tree.rs`.
- Cache-state runtime tests, reconciliation tests, immutable router image, and the shadow `llm-cache-state-glm52-b` rollout.
- No worker restart, worker pool, APIM, AFD, or production gateway traffic change is required.
