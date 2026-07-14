## Context

SGLang workers publish ordered msgpack KV mutation batches over ZMQ. A local Rust agent currently validates each batch, then awaits HTTP and Kafka sinks serially. Production cache-state A/B replicas consume Kafka with independent consumer groups and rebuild an in-memory `HashTree`, but the stream envelope's `seq` is not used for continuity and Kafka offset commit is initiated before tree application. Successfully accepted Event Hubs records are durable for the configured retention window, but records lost before acceptance cannot be reconstructed.

The worker event publisher already owns a background thread, a monotonically increasing transport sequence, and an optional replay buffer. The scheduler supplies authoritative cache mutations to that publisher. This change uses a worker-resident state tracker on the publisher path so digest and snapshot generation cannot be corrupted by downstream agent, HTTP, Kafka, or consumer loss.

## Goals / Non-Goals

**Goals:**

- Detect and contain downstream event loss within one worker rank.
- Periodically prove cache-state against a worker-authoritative digest.
- Repair divergence automatically from a bounded snapshot without restarting a worker, agent, gateway, or cache-state process.
- Preserve inference availability by removing only untrusted cache-hit claims, not worker load-routing eligibility.
- Keep event ingestion non-blocking across independent sinks and make Kafka consumption replay-safe.
- Support a staged, opt-in mixed-version rollout.

**Non-Goals:**

- Detect bugs where the engine itself fails to generate a cache mutation event.
- Snapshot KV tensor bytes or move HiCache/Mooncake payloads; snapshots contain routing metadata only.
- Introduce an Azure-specific Blob dependency into SGLang. The first version uses bounded chunks on the configured event stream; an object-store transport can be added behind the same manifest later.
- Change worker membership, production routing, or startup parameters as part of this source change.

## Decisions

### Extend the existing ordered event batch with opt-in reconciliation metadata

`EventBatch` gains optional trailing `publisher_epoch` and `reconciliation` fields. Reconciliation is a tagged union for `Digest`, `SnapshotStart`, `SnapshotChunk`, and `SnapshotEnd`. The legacy first three fields and mutation event types remain unchanged. Emission defaults off, because an old decoder may reject trailing fields; rollout therefore upgrades cache-state and agents before enabling worker emission.

Every published batch, including control-only heartbeats, uses the existing ZMQ transport sequence. Snapshot chunks are emitted synchronously by one publisher thread and use the same Kafka key (`worker_url + dp_rank`), preserving total order within one Event Hubs partition.

Alternative considered: a new worker HTTP snapshot endpoint. It would require scheduler RPC plumbing, externally reachable worker control ports, and authentication on every deployment shape. Keeping metadata on the existing local agent path avoids that new control plane.

### Maintain a worker-resident routing-state tracker off the scheduler hot path

The ZMQ publisher thread applies each `BlockStored`, `BlockRemoved`, and `AllBlocksCleared` mutation to a logical block map. A block remains routing-visible while any storage medium retains it. The tracker stores its parent hash and medium membership and maintains an order-independent `xor-sha256-v1` accumulator plus logical block count. Each visible `(parent_hash, block_hash)` contributes one SHA-256 item digest; insertion/removal XORs that digest, making frequent digest emission O(1) between mutations.

Digest records are emitted on an idle-capable timer. Snapshot generation sorts a copy of the tracker entries in the publisher thread, computes the same digest, and emits size-bounded chunks. Scheduler publication only enqueues mutation batches and never serializes a full snapshot.

Alternative considered: traverse each concrete Python Radix/HiCache implementation periodically. That would duplicate traversal across multiple cache classes and risks pausing the scheduler. The publisher tracker specifically targets downstream transport correctness; engine event-generation correctness remains a non-goal.

### Treat snapshot transfer as an ordered transaction

`SnapshotStart` declares epoch, snapshot id, base data watermark, chunk count, logical count, digest algorithm/value, and limits. `SnapshotChunk` carries one ordered slice of `(parent_hash, block_hash, media[])` entries. Media membership is retained so later tier-specific removals can evolve the installed snapshot without ambiguity; the digest itself remains over the routing-visible `(parent_hash, block_hash)` union. `SnapshotEnd` repeats the identity and integrity fields. Cache-state accepts only one bounded assembly per worker rank, rejects conflicts or gaps, and installs only after all metadata verifies.

The publisher does not interleave mutation batches between snapshot start and end. Therefore a successful end record is an atomic boundary: cache-state replaces the worker ownership under one HashTree write lock, marks the rank trusted, then applies later transport sequences normally.

Alternative considered: put snapshots in Blob and events in Event Hubs. This avoids stream bandwidth but introduces cloud-specific credentials and lifecycle. The protocol keeps snapshot manifests transport-neutral so Blob can be added later if measured snapshot volume requires it.

### Make trust explicit and fail closed only for cache-hit claims

Cache-state tracks `epoch`, `last_seq`, payload hash, reconciliation phase, logical state digest, and trust per `KvWorkerId`. A sequence gap, conflicting duplicate, epoch transition, malformed control record, or digest mismatch sets `trusted=false` before the next query. Incremental mutations may continue updating the candidate state, but the rank is excluded from cache matches until a digest proves equality or a snapshot installs successfully.

HashTree gains an atomic per-worker replacement operation and a filtered prefix match that can return a shallower prefix held by a trusted rank. Cache-state responses gain a backward-compatible `authoritative` flag. In reconciliation mode a successful miss is authoritative, so the gateway skips stale local route-history and uses normal load routing. Transport failure or an old response without the flag preserves today's local fallback.

Alternative considered: remove an untrusted worker from the routing pool. Cache metadata does not affect inference correctness, so removing inference capacity would be disproportionate.

### Isolate sinks with bounded ordered queues

The agent creates one ordered sender task and bounded channel per sink. The ZMQ task validates once, creates one immutable stream record, and attempts to enqueue it independently. Each sender retries delivery with bounded exponential backoff. A full queue produces an explicit drop/gap metric and log; it cannot block another sink or the subscriber indefinitely. Queue sizes, attempts, and delivery timeout are configurable with conservative defaults.

This is intentionally a bounded in-memory design for the first change. A local disk WAL would improve crash durability but creates lifecycle, disk-pressure, and replay ownership that should be a separate capability.

### Commit Kafka offsets only after apply

The consumer returns an owned record plus topic/partition/offset metadata without committing. After cache-state reports successful apply or idempotent duplicate, the loop commits `offset + 1`. Validation or apply failure leaves the offset uncommitted and uses bounded retry/backoff. This closes the current commit-before-apply crash window.

## Risks / Trade-offs

- [Publisher tracker reflects emitted mutations rather than independently walking physical KV allocations] -> Scope the guarantee to downstream delivery correctness and make that boundary explicit in metrics/docs.
- [Frequent full snapshots consume CPU and Event Hubs bandwidth] -> Generate them on the publisher thread, bound encoded chunks, make intervals configurable, and use frequent digests with less frequent snapshots.
- [A snapshot larger than configured assembly limits never repairs] -> Reject safely, expose metrics, and let operators raise reviewed limits or add object-store transport.
- [XOR digests are weaker than a sorted Merkle root against an adversarial worker] -> Workers are trusted; pair a 256-bit per-entry SHA-256 XOR with count and verify every snapshot entry. Protocol carries an algorithm field for later upgrade.
- [Mixed-version components reject new trailing wire fields] -> Default emission off and require agents/cache-state to roll first; legacy mode remains unchanged.
- [Fail-closed authoritative misses reduce cache-hit rate during recovery] -> Continue load-based routing so correctness and capacity remain intact, and expose trust/reconciliation duration.

## Migration Plan

1. Merge code with reconciliation disabled by default.
2. Build one immutable image containing the new agent, cache-state, and gateway decoder; validate an isolated event stream with synthetic dropped sequences.
3. Roll A/B cache-state and agents while workers still emit legacy batches; verify no routing behavior change.
4. Enable digest/snapshot emission on one drained or test worker rank and enable reconciliation on a canary cache-state consumer group.
5. Inject a dropped mutation, verify fail-closed cache matching and automatic snapshot repair, then compare A/B digests.
6. Roll workers one at a time under the existing startup-parameter workflow; enable authoritative responses only after all active worker ranks emit reconciliation metadata.
7. Roll back by disabling authoritative responses and worker reconciliation emission; legacy mutation ingestion remains available.

## Open Questions

- Production digest/snapshot intervals and queue/assembly limits require measurement on representative FP8 and NVFP4 workers before rollout; code defaults remain disabled until that benchmark is recorded.
