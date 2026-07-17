# distributed-cache-state Specification

## Purpose
Define the standalone cache-state service and gateway integration used to share prefix-cache state across gateway replicas while preserving safe fallback behavior.
## Requirements
### Requirement: Distributed cache state exposes prefix-match queries
The system SHALL provide a cache-state service mode that maintains a prefix-cache hash tree and exposes an internal HTTP API for matching request block hashes against cached worker prefixes; when reconciliation is enabled, matches SHALL contain only worker ranks whose state is currently trusted.

#### Scenario: Prefix match returns matching trusted workers
- **WHEN** the cache-state service has indexed a worker prefix, the worker rank is trusted, and a gateway queries the same model and leading block hash chain
- **THEN** the service SHALL return the number of matched leading blocks and the trusted worker URLs holding the deepest useful matched prefix

#### Scenario: Deepest holder is untrusted
- **WHEN** the deepest indexed prefix is held only by untrusted worker ranks but a shallower prefix is held by a trusted rank
- **THEN** the service SHALL return the deepest prefix whose holder set contains a trusted rank

#### Scenario: Health endpoint reports readiness
- **WHEN** an operator or ACA probe calls the cache-state service health endpoint
- **THEN** the service SHALL return a successful response without requiring worker traffic

### Requirement: Gateway can use remote cache state
The gateway SHALL support an opt-in remote cache-state client for cache-aware routing. In `remote_only` mode the remote response SHALL be the only cache-match source: useful trusted matches SHALL participate in worker scoring, while authoritative misses, transport failures, timeouts, and malformed responses SHALL continue through cache-miss Prefill/load selection without consulting or populating a local route-history tree. Other explicitly configured cache-tree modes SHALL preserve their existing behavior.

#### Scenario: Remote cache-state URL is configured
- **WHEN** cache-aware routing is enabled in `remote_only` mode with a remote cache-state URL and the remote service returns a useful trusted prefix match
- **THEN** the gateway SHALL use the remote match result for prefix-aware worker scoring

#### Scenario: Authoritative remote service returns no useful trusted match
- **WHEN** the remote query succeeds with an authoritative response but returns zero matched blocks or no trusted matching workers
- **THEN** the gateway SHALL skip local cache-history matching and continue through cache-miss load-based selection

#### Scenario: Remote-only service is unavailable or malformed
- **WHEN** the remote query fails, times out, or returns malformed data in `remote_only` mode
- **THEN** the gateway SHALL treat cache match as empty, SHALL NOT query a local prefix tree, and SHALL continue through Prefill/load fallback

#### Scenario: Another cache-tree mode is explicitly configured
- **WHEN** the administrator selects an existing non-remote-only source
- **THEN** the gateway SHALL preserve that source's existing local and remote fallback semantics

### Requirement: Remote cache-state failures degrade safely
The gateway SHALL treat remote cache-state transport failures, timeouts, and malformed responses as non-fatal. It SHALL distinguish those failures from a successful authoritative cache miss so stale local history cannot override a fail-closed reconciliation decision.

#### Scenario: Remote service is unavailable
- **WHEN** the gateway cannot reach the configured remote cache-state service during a route selection
- **THEN** the gateway SHALL not fail the user request because of that cache-state error and MAY use local cache history

#### Scenario: Remote service returns an authoritative miss
- **WHEN** the remote cache-state response is valid and authoritative but has zero matched blocks or no matching workers
- **THEN** the gateway SHALL treat it as a cache miss and SHALL NOT use local cache history for that decision

#### Scenario: Neither authoritative cache state nor load state has a preferred worker
- **WHEN** the authoritative response has no useful match
- **THEN** the gateway SHALL select through its normal cache-miss load-balancing path

### Requirement: Distributed cache-state operation is opt-in
The system SHALL require explicit configuration to run cache-state service mode or to make a gateway query a remote cache-state service.

#### Scenario: Existing gateway flags are unchanged
- **WHEN** an operator starts the gateway with the existing cache-aware flags and no distributed cache-state flags
- **THEN** startup and routing SHALL preserve the previous behavior

#### Scenario: ACA service can run separately
- **WHEN** an operator starts the binary in cache-state service mode with host and port configuration
- **THEN** it SHALL serve only cache-state endpoints and SHALL NOT require worker discovery or proxy configuration

### Requirement: Gateway can feed remote cache state
When route-history cache tree source is enabled with a remote cache-state URL, the gateway SHALL be able to submit the chosen worker and request block hashes to the remote cache-state service after worker selection.

#### Scenario: Route-history selection feeds remote cache state
- **WHEN** a cache-aware route-history gateway selects a worker for a request with non-empty block hashes and remote cache-state is configured
- **THEN** the gateway SHALL send an internal cache-state insert request containing the model id, chosen worker URL, dp rank, and block hash chain

#### Scenario: Remote feed failure is non-fatal
- **WHEN** the gateway cannot insert a selected route-history prefix into the remote cache-state service
- **THEN** the gateway SHALL still return the selected worker for the user request and keep the local route-history insertion behavior

### Requirement: Remote cache-state observability is bounded
The gateway SHALL expose bounded-cardinality metrics for remote cache-state query and feed outcomes.

#### Scenario: Remote query outcome is recorded
- **WHEN** the gateway attempts a remote cache-state match query
- **THEN** it SHALL increment a counter labeled by a bounded outcome such as hit, miss, failure, or fallback_local_hit

#### Scenario: Remote feed outcome is recorded
- **WHEN** the gateway attempts a remote cache-state insert/feed
- **THEN** it SHALL increment a counter labeled by a bounded outcome such as success or failure

### Requirement: Cache-state tracks sequence continuity and worker trust
When reconciliation is enabled, cache-state SHALL track cache epoch and last applied transport sequence independently for each `worker_url + dp_rank`, and SHALL mark only that worker rank untrusted when it observes an epoch transition, sequence gap, conflicting duplicate, malformed authoritative metadata, or digest mismatch.

#### Scenario: Incremental sequence is continuous
- **WHEN** the next record has the current epoch and sequence `last_applied + 1`
- **THEN** cache-state SHALL apply it without changing a trusted worker rank to untrusted

#### Scenario: Sequence gap is observed
- **WHEN** the next record for a worker rank skips one or more transport sequences
- **THEN** cache-state SHALL mark that rank untrusted before it can answer another cache-hit query

#### Scenario: Duplicate record is replayed
- **WHEN** a record repeats an already applied epoch, sequence, and payload hash
- **THEN** cache-state SHALL treat it idempotently without changing the tree or trust state

#### Scenario: Worker starts a new epoch
- **WHEN** a record carries an epoch different from the tracked epoch
- **THEN** cache-state SHALL invalidate the prior worker-rank state and require a matching empty digest or verified snapshot before restoring trust

### Requirement: Cache-state reconciles snapshots atomically
Cache-state SHALL assemble bounded snapshot chunks by worker rank, epoch, snapshot id, and watermark; verify ordering, completeness, count, and digest; atomically replace that worker rank's HashTree ownership; and restore trust only after verification succeeds.

#### Scenario: Complete snapshot repairs a sequence gap
- **WHEN** an untrusted worker rank receives a complete verified snapshot for its current epoch
- **THEN** cache-state SHALL atomically replace only that rank's indexed blocks and mark it trusted

#### Scenario: Snapshot is missing or corrupt
- **WHEN** a snapshot chunk is missing, duplicated with conflicting bytes, out of order, oversized, or fails its final digest
- **THEN** cache-state SHALL discard the assembly, retain the worker rank as untrusted, and keep serving other ranks

#### Scenario: Newer incremental events follow a snapshot
- **WHEN** records newer than the snapshot watermark arrive after a verified snapshot end record
- **THEN** cache-state SHALL apply them in sequence order on top of the installed snapshot

### Requirement: Kafka offsets are committed after successful application
The cache-state Kafka consumer SHALL commit a record's next offset only after that record has been decoded, validated, and successfully applied or accepted as an idempotent duplicate.

#### Scenario: Record application succeeds
- **WHEN** cache-state successfully applies a consumed record
- **THEN** the consumer SHALL commit the corresponding partition offset after the apply result

#### Scenario: Record application fails
- **WHEN** decoding, validation, snapshot assembly, or tree mutation fails
- **THEN** the consumer SHALL NOT commit that record's offset and SHALL retry or stop according to bounded failure policy

### Requirement: Reconciliation state is observable with bounded cardinality
Cache-state and the cache-event agent SHALL expose bounded metrics for sequence gaps, digest matches and mismatches, snapshot lifecycle outcomes, sink queue drops, reconciliation duration, and trusted worker-rank count without placing raw worker URLs or snapshot ids in metric labels.

#### Scenario: Digest mismatch is detected
- **WHEN** a worker digest differs from cache-state at the same epoch and watermark
- **THEN** cache-state SHALL increment a mismatch counter and expose the affected rank through logs while keeping metric labels bounded

#### Scenario: Snapshot restores trust
- **WHEN** a verified snapshot transitions a worker rank from untrusted to trusted
- **THEN** cache-state SHALL record a successful reconciliation and its duration

#### Scenario: Operator inspects current trust
- **WHEN** the cache-state metrics endpoint is scraped
- **THEN** it SHALL expose aggregate trusted and untrusted worker-rank gauges and reconciliation counters

### Requirement: Cache-state preserves physical Prefill ownership for logical groups
Cache-state SHALL continue to store and return KV prefix ownership using physical worker URLs and DP ranks, even when those physical workers are members of a logical PD proxy worker.

#### Scenario: Physical Prefill member matches
- **WHEN** a physical Prefill member has published KV events for a prefix and a gateway queries cache-state for that prefix
- **THEN** cache-state SHALL return the physical Prefill worker URL and DP rank rather than a synthetic logical proxy URL

#### Scenario: Multiple Prefill members exist in one logical group
- **WHEN** two physical Prefill members belong to the same logical proxy group and only one member holds the deepest prefix
- **THEN** cache-state SHALL report only the physical member or members that actually hold that prefix

#### Scenario: Logical group mapping is absent
- **WHEN** a gateway has no logical group mapping for a returned physical Prefill worker URL
- **THEN** cache-state SHALL NOT synthesize any logical worker match on the gateway's behalf
