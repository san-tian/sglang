## Context

The first reconciliation-enabled GLM-5.2 cache-state shadow runs in Azure Container Apps with 1 CPU and 2 GiB memory. Its working set peaked below the limit and logs contain no panic or OOM, but both readiness and liveness repeatedly time out after five seconds. The platform has restarted the same replica at least fourteen times. Its four Event Hubs partitions are approximately 6k-12k records behind, and committed offsets advance by only a few records between crash-loop restarts.

The cache-state Kafka task currently calls synchronous record application and `CommitMode::Sync` directly inside `tokio::spawn`. Snapshot end processing holds reconciliation state while `HashTree::replace_worker` repeatedly scans every unresolved entry until parent hashes become reachable. Publisher snapshots are deterministically sorted by hashes, not topologically, so deep chains can require many full scans. The same synchronous application path is also reachable through the cache-state HTTP event endpoint.

## Goals / Non-Goals

**Goals:**

- Keep cache-state liveness responsive while Kafka/Event Hubs apply and commit operations are slow.
- Preserve at-least-once, post-apply offset commit semantics and per-partition ordering.
- Make valid snapshot dependency ordering linear in the number of logical entries.
- Preserve atomic worker ownership replacement and existing validation behavior.
- Provide enough bounded observability to identify a slow record without logging payload bytes or secrets.

**Non-Goals:**

- Change worker event emission intervals, Event Hubs retention, worker membership, or gateway traffic.
- Cancel an in-progress tree mutation on a wall-clock timeout.
- Redesign the coarse HashTree or reconciliation locks in this follow-up.
- Skip, dead-letter, or commit a failed record automatically.

## Decisions

### Run apply and synchronous commit on Tokio's blocking pool

The Kafka receive future remains on the asynchronous runtime. Once it yields an owned record, a `spawn_blocking` task performs decode/application and the synchronous offset commit in the existing apply-then-commit order. The consumer waits for that task before receiving the next record, preserving per-consumer ordering and bounded retries. The HTTP KV-event handler uses the same blocking boundary for record application.

Keeping `CommitMode::Sync` is deliberate: switching to asynchronous commit would return before the broker confirms the offset and would weaken the existing post-apply durability contract. Adding more Tokio worker threads alone would hide the immediate symptom but leave blocking work on the I/O executor and remain sensitive to quota/runtime sizing.

### Replace repeated scans with a dependency adjacency traversal

Snapshot ordering first rejects duplicate `(parent_hash, block_hash)` edges, groups non-root edges by parent hash, installs all root edges, then releases each newly resolved block hash's children from an adjacency map. Every logical edge is inserted and removed a bounded number of times. Installation succeeds only when all entries become reachable; unresolved cycles and missing parents still fail atomically before the HashTree write lock is acquired.

This preserves the existing semantics where any resolved occurrence of a block hash can satisfy a child parent reference. It does not impose a new uniqueness constraint on block hashes that legitimately appear in multiple chains.

### Do not cancel blocking state mutation

Tokio cannot safely abort a running `spawn_blocking` closure, and externally timing it out could allow the closure to mutate state or commit after the caller treats it as failed. The service therefore waits for completion while HTTP liveness remains independently schedulable. Slow-record duration is logged with partition, offset, sequence, rank, and operation outcome; metric labels remain bounded.

### Keep snapshot replacement atomic in this patch

The reconciliation mutex and HashTree replacement boundary remain unchanged to avoid exposing a partially installed snapshot. Cache match or metrics calls can still wait for one snapshot installation, but the liveness endpoint does not acquire those locks and stays responsive. A future copy-on-write tree generation can reduce match latency if production measurements justify the larger redesign.

## Risks / Trade-offs

- [A pathological record can occupy a blocking thread for a long time] -> One ordered consumer has at most one apply/commit operation in flight, and liveness remains available; logs identify the stuck partition/offset for operator action.
- [One CPU is shared by the blocking worker and HTTP runtime] -> OS scheduling still gives the I/O runtime an independent runnable thread; regression tests exercise a single-thread Tokio runtime under blocking work.
- [Linear ordering allocates an adjacency map in addition to the snapshot vector] -> Memory remains O(n) and within the existing configured snapshot-entry bounds; the old algorithm already cloned the complete entry list.
- [HTTP match requests can wait on the reconciliation lock] -> Gateway timeouts and A/B failover continue to contain this; lock-free generation swap is explicitly out of scope.

## Migration Plan

1. Validate the source change with deep reverse-ordered snapshot and single-thread runtime regression tests.
2. Build one immutable router/cache-state image from `san-tian/sglang@deploy-prod`.
3. Update only the shadow `llm-cache-state-glm52-b` revision; do not change gateway traffic, APIM, worker pools, or SGLang workers.
4. Require stable liveness, no restart growth, advancing consumer offsets, and bounded reconciliation metrics across at least two snapshot intervals before any broader rollout.
5. Roll back B to image digest `sha256:5e15b20367640491b16a59ff133d1f6893bd9fae30eb6cdd064051bb9cb48264` if health or reconciliation regresses.

## Open Questions

- Whether match-prefix latency during a large atomic replacement warrants a later copy-on-write HashTree generation will be decided from shadow metrics; it is not required for liveness recovery.
