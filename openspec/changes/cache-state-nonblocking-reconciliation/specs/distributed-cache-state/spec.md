## ADDED Requirements

### Requirement: Durable stream work preserves cache-state liveness
Cache-state SHALL execute potentially blocking record application outside the asynchronous HTTP serving executor, store offsets only after successful application, and periodically batch broker commits while preserving ordered at-least-once semantics.

#### Scenario: Broker commit is slower than record ingestion
- **WHEN** a broker-confirmed offset commit has non-trivial network latency
- **THEN** cache-state SHALL batch stored post-apply offsets on the configured periodic commit interval and SHALL NOT wait for a broker round trip before receiving the next ordered record

#### Scenario: Record application is CPU intensive
- **WHEN** decoding, snapshot verification, or HashTree mutation occupies a worker thread
- **THEN** cache-state SHALL keep the asynchronous HTTP serving executor available for liveness requests

#### Scenario: Record application fails
- **WHEN** the blocking apply task panics or returns an application error
- **THEN** cache-state SHALL NOT store that record's offset and SHALL apply the configured bounded retry or stop policy

#### Scenario: Post-apply checkpoint succeeds
- **WHEN** an ordered record is applied successfully
- **THEN** cache-state SHALL store exactly the next offset for that partition so only applied records are eligible for the next periodic broker commit

#### Scenario: Process exits before the next periodic commit
- **WHEN** a record was applied and locally checkpointed but its offset was not yet committed to the broker
- **THEN** the restarted consumer SHALL replay the record and reconciliation SHALL accept an identical sequence/hash duplicate without skipping unprocessed records

#### Scenario: Periodic broker commit fails
- **WHEN** librdkafka reports an asynchronous auto-commit error
- **THEN** cache-state SHALL emit a bounded structured warning identifying the commit failure and affected topic-partition offsets

### Requirement: Slow stream progress is diagnosable without unbounded labels
Cache-state SHALL report apply-and-checkpoint duration and record progress using bounded metric labels and structured logs that identify partition and offset without logging payload contents or credentials.

#### Scenario: One record is slow
- **WHEN** record application or local checkpoint exceeds the configured slow-operation threshold
- **THEN** cache-state SHALL emit a structured progress event containing operation outcome, partition, offset, sequence, duration, and record kind without including payload bytes or secrets
