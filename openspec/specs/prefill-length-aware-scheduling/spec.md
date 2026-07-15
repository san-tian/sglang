# prefill-length-aware-scheduling Specification

## Purpose
Define the opt-in worker policy that lowers short-request Prefill latency using live uncached work while preserving priority, providing deterministic 1:3 fairness for older requests, and exporting phase-aware load summaries.

## Requirements
### Requirement: Length-aware Prefill scheduling is opt-in
The worker SHALL preserve its configured existing scheduling behavior unless `prefill-length-aware` is explicitly selected, and the default schedule policy SHALL remain FCFS.

#### Scenario: Worker starts without the new policy
- **WHEN** a worker starts without selecting `prefill-length-aware`
- **THEN** it SHALL retain its configured existing waiting-prefill ordering

### Requirement: Waiting order uses current uncached work
The worker SHALL refresh prefix-cache matches before each eligible scheduling round and SHALL order each business-priority class by repeatedly selecting one earliest-arrived request followed by up to three requests with the fewest current uncached input tokens.

#### Scenario: A prefix becomes cached while a request waits
- **WHEN** an earlier request creates a reusable prefix before the next scheduling round
- **THEN** the dependent request SHALL be ordered using its newly reduced uncached work

#### Scenario: Short request and long request have equal priority and age
- **WHEN** two non-overdue requests have the same business priority and different uncached input lengths
- **THEN** the request with fewer uncached tokens SHALL precede the longer request

#### Scenario: Business priorities differ
- **WHEN** priority scheduling is enabled and requests have different business priorities
- **THEN** configured business-priority direction SHALL take precedence over prompt length

### Requirement: Same-priority fairness is bounded
The worker SHALL revisit the earliest-arrived remaining request after every group of at most three short requests, so a long request cannot be bypassed by an unbounded number of later short requests.

#### Scenario: An older long request is revisited
- **WHEN** an older long request remains after three shorter requests are selected
- **THEN** it SHALL be selected before the next group of short requests in the same business priority

#### Scenario: Fewer than three short requests remain
- **WHEN** fewer than three requests remain after selecting the oldest request
- **THEN** all remaining requests SHALL be selected by current uncached-token count and queue-entry time

### Requirement: Existing Prefill execution constraints remain authoritative
The policy SHALL only reorder requests still in the waiting queue and SHALL preserve chunked-prefill continuation and all existing admission constraints.

#### Scenario: Chunked Prefill is active
- **WHEN** a chunked request exists while a shorter request is waiting
- **THEN** the scheduler SHALL retain its existing chunked-request continuation behavior

#### Scenario: An ordered request cannot fit
- **WHEN** the next ordered request fails an existing token, KV, LoRA, or other admission check
- **THEN** `PrefillAdder` SHALL retain authority to defer or reject that admission

### Requirement: Worker exports bounded Prefill work summaries
When the policy is active, worker load reporting SHALL expose optional aggregate data for total Prefill work and candidate work-ahead without exporting individual requests, and SHALL bound represented priority groups.

#### Scenario: Candidate detail is complete
- **WHEN** the waiting queue has no more than the supported priority-group limit
- **THEN** the snapshot SHALL include chunked remainder, fixed work-bucket bounds, priority direction, and cumulative uncached work per priority and bucket

#### Scenario: Priority cardinality exceeds the limit
- **WHEN** the waiting queue exceeds the supported priority-group limit
- **THEN** the snapshot SHALL retain conservative aggregate work and SHALL mark candidate detail incomplete

#### Scenario: Legacy load client reads the endpoint
- **WHEN** a client ignores unknown `/get_load` response fields
- **THEN** all historical load fields SHALL retain their names and compatible value types

### Requirement: Prefill load reporting identifies phase ownership
Worker load reporting SHALL identify integrated, native Prefill, and Decode-only roles so token work is attributed to the phase that owns it.

#### Scenario: Native Prefill has bootstrap backlog
- **WHEN** a Prefill-only worker has requests in either its bootstrap queue or schedulable waiting queue
- **THEN** its conservative waiting uncached token total SHALL include both queues

#### Scenario: Decode-only worker reports load
- **WHEN** a Decode-only worker reports running or waiting requests
- **THEN** it SHALL label the Decode role and SHALL report zero waiting Prefill tokens

#### Scenario: Prefill policy is configured on Decode
- **WHEN** a Decode-only worker is configured with `prefill-length-aware`
- **THEN** startup validation SHALL reject the configuration

### Requirement: Legacy length-aware tunables remain parse-compatible
The worker SHALL continue accepting the historical aging-rate and maximum-wait arguments for deployment compatibility, but SHALL not use them to determine request order.
