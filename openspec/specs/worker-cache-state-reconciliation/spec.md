# worker-cache-state-reconciliation Specification

## Purpose
TBD - created by archiving change cache-state-worker-snapshot-reconciliation. Update Purpose after archive.
## Requirements
### Requirement: Worker publisher identifies authoritative state generations
When reconciliation is enabled, the worker KV event publisher SHALL assign a process-unique cache epoch and SHALL include that epoch in every published incremental, digest, and snapshot batch for a DP rank.

#### Scenario: Worker publisher starts a new generation
- **WHEN** a KV event publisher starts after worker startup or restart
- **THEN** it SHALL use a cache epoch different from the preceding publisher generation

#### Scenario: Reconciliation is disabled
- **WHEN** the worker is started without reconciliation configuration
- **THEN** it SHALL preserve the legacy KV event wire behavior and SHALL NOT require downstream reconciliation support

### Requirement: Worker emits periodic authoritative digests
The worker publisher SHALL maintain a routing-visible logical block set from cache mutations and, at the configured interval, SHALL emit an order-independent digest containing the cache epoch, transport sequence watermark, logical block count, digest algorithm, and digest value even when no new cache mutation is available.

#### Scenario: Incremental events were delivered without loss
- **WHEN** cache-state has applied every worker mutation through the advertised watermark
- **THEN** cache-state SHALL be able to calculate the same block count and digest value

#### Scenario: Worker is idle
- **WHEN** no new cache mutation occurs for one digest interval
- **THEN** the publisher SHALL still emit a digest for the current authoritative state

#### Scenario: A block exists in multiple storage media
- **WHEN** the same routing-visible block remains present in at least one storage medium
- **THEN** removing another medium SHALL NOT remove the block from the authoritative logical digest

### Requirement: Worker emits bounded recoverable snapshots
The worker publisher SHALL periodically emit a start record, bounded ordered chunks, and an end record that together describe the complete routing-visible block state and its storage-medium membership at one sequence watermark and carry an integrity digest for the logical block union.

#### Scenario: Snapshot completes successfully
- **WHEN** all chunks for one snapshot id arrive in order and their reconstructed state matches the end-record digest and count
- **THEN** the snapshot SHALL be eligible for atomic installation by cache-state

#### Scenario: Snapshot exceeds one transport record
- **WHEN** the encoded state is larger than the configured per-record limit
- **THEN** the publisher SHALL split it into independently bounded chunks without exceeding the configured record limit

#### Scenario: Snapshot is followed by a tier-specific removal
- **WHEN** a snapshot entry exists in multiple storage media and a later event removes only one medium
- **THEN** cache-state SHALL retain the logical block until the final advertised medium is removed

#### Scenario: Snapshot is generated under serving load
- **WHEN** a periodic snapshot becomes due while the scheduler is serving requests
- **THEN** state tracking, serialization, and chunk publication SHALL occur on the publisher path and SHALL NOT synchronously traverse or serialize the cache on the scheduler hot path

### Requirement: Agent preserves ordering without cross-sink head-of-line blocking
The cache-event agent SHALL ingest each worker rank in sequence order and SHALL dispatch immutable event-stream records to independently bounded sink workers so that a slow HTTP or Kafka sink does not block ZMQ reception or another sink.

#### Scenario: Kafka delivery is delayed
- **WHEN** Kafka delivery waits or retries while the HTTP sink is healthy
- **THEN** HTTP delivery and ZMQ ingestion SHALL continue until their own bounded queues reach their configured limits

#### Scenario: A sink queue reaches its bound
- **WHEN** an event cannot be enqueued without exceeding a configured sink bound
- **THEN** the agent SHALL record a bounded-cardinality drop/gap signal and SHALL NOT silently claim delivery success

#### Scenario: Snapshot records are forwarded
- **WHEN** the agent receives snapshot start, chunk, and end batches for a worker rank
- **THEN** it SHALL preserve their epoch, transport sequence, payload bytes, key, and ordering in every configured sink

### Requirement: Snapshot and digest protocol is bounded and backward compatible
The reconciliation protocol SHALL impose explicit limits on epoch length, snapshot identifiers, chunk count, entries per chunk, encoded payload size, and in-progress snapshot memory, and legacy consumers SHALL remain usable while reconciliation emission is disabled.

#### Scenario: Malformed or oversized reconciliation record arrives
- **WHEN** a record exceeds a protocol limit or contains inconsistent metadata
- **THEN** the consumer SHALL reject that record without allocating unbounded memory or terminating the cache-state service

#### Scenario: Mixed-version rollout begins
- **WHEN** agents and cache-state replicas are upgraded before worker reconciliation emission is enabled
- **THEN** legacy incremental KV events SHALL continue to be accepted with their previous routing behavior
