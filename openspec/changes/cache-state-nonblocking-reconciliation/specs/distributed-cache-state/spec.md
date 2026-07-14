## ADDED Requirements

### Requirement: Durable stream work preserves cache-state liveness
Cache-state SHALL execute potentially blocking record application and broker-confirmed offset commits outside the asynchronous HTTP serving executor while preserving ordered post-apply commit semantics.

#### Scenario: Kafka commit is slow
- **WHEN** a broker-confirmed offset commit takes longer than an HTTP liveness probe interval
- **THEN** cache-state SHALL keep its liveness endpoint schedulable and SHALL NOT receive another record on that ordered consumer until the commit completes

#### Scenario: Record application is CPU intensive
- **WHEN** decoding, snapshot verification, or HashTree mutation occupies a worker thread
- **THEN** cache-state SHALL keep the asynchronous HTTP serving executor available for liveness requests

#### Scenario: Blocking task fails
- **WHEN** the blocking apply-and-commit task panics or returns an application or commit error
- **THEN** cache-state SHALL leave the record offset uncommitted and SHALL apply the configured bounded retry or stop policy

### Requirement: Slow stream progress is diagnosable without unbounded labels
Cache-state SHALL report apply-and-commit duration and record progress using bounded metric labels and structured logs that identify partition and offset without logging payload contents or credentials.

#### Scenario: One record is slow
- **WHEN** record application or commit exceeds the configured slow-operation threshold
- **THEN** cache-state SHALL emit a structured progress event containing operation outcome, partition, offset, sequence, duration, and record kind without including payload bytes or secrets
