# Retired prefill-length-aware scheduling Specification

## Purpose
Define compatibility and load-reporting behavior after worker-side length-aware Prefill scheduling is retired.

## Requirements

### Requirement: Waiting Prefill order is FCFS
The worker SHALL schedule waiting Prefill requests by business priority and queue-entry time without using request length or current uncached work to reorder requests.

#### Scenario: Same-priority requests have different lengths
- **WHEN** same-priority requests have different input lengths
- **THEN** the earlier request SHALL remain ahead of the later request

#### Scenario: Business priorities differ
- **WHEN** priority scheduling is enabled and requests have different business priorities
- **THEN** configured business-priority direction SHALL take precedence while each priority group remains FCFS

### Requirement: The retired policy name is parse-compatible
The worker SHALL accept `prefill-length-aware` as a deprecated compatibility name and SHALL execute FCFS ordering when it is selected.

#### Scenario: A stale launcher selects the retired policy
- **WHEN** a stale launcher selects `prefill-length-aware`
- **THEN** startup SHALL succeed and the scheduler SHALL use FCFS

### Requirement: Existing Prefill execution constraints remain authoritative
The policy SHALL preserve chunked-prefill continuation and all existing admission constraints.

#### Scenario: Chunked Prefill is active
- **WHEN** a chunked request exists while another request is waiting
- **THEN** the scheduler SHALL retain its existing chunked-request continuation behavior

#### Scenario: An ordered request cannot fit
- **WHEN** the next ordered request fails an existing token, KV, LoRA, or other admission check
- **THEN** `PrefillAdder` SHALL retain authority to defer or reject that admission

### Requirement: Worker exports bounded FCFS Prefill work summaries
Worker load reporting SHALL expose optional aggregate data for total Prefill work and FCFS candidate work-ahead without exporting individual requests, and SHALL bound represented priority groups.

#### Scenario: Candidate detail is complete
- **WHEN** the waiting queue has no more than the supported priority-group limit
- **THEN** the snapshot SHALL include chunked remainder, fixed work-bucket bounds, priority direction, and the full existing same-priority uncached work ahead of a new request for every bucket

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

### Requirement: Legacy length-aware tunables remain parse-compatible
The worker SHALL continue accepting the historical aging-rate and maximum-wait arguments for deployment compatibility, but SHALL not use them to determine request order.
