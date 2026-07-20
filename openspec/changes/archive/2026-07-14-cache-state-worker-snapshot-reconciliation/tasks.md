## 1. Reconciliation Protocol

- [x] 1.1 Extend Python KV event batches with opt-in epoch, digest, and bounded snapshot control records while preserving the disabled legacy wire shape
- [x] 1.2 Extend the Rust KV event decoder and validation limits for the reconciliation wire fields, with Python-compatible golden tests

## 2. Worker Authoritative State

- [x] 2.1 Implement a publisher-thread logical block tracker with storage-medium union semantics and the canonical XOR-SHA256 digest
- [x] 2.2 Emit idle-capable periodic digests and non-interleaved bounded snapshot transactions from the publisher thread
- [x] 2.3 Add Python tests for epoch compatibility, multi-medium mutations, digest stability, snapshot chunking, and idle emission

## 3. Cache Event Fanout

- [x] 3.1 Refactor the cache-event agent into independent bounded ordered sink queues with configurable delivery retry and timeout policy
- [x] 3.2 Add bounded-cardinality sink drop metrics and tests proving a delayed sink does not block another sink or ZMQ ingestion

## 4. Cache-State Reconciliation

- [x] 4.1 Add HashTree APIs for atomic per-worker replacement and trusted-worker filtered deepest-prefix matching
- [x] 4.2 Track per-worker-rank epoch, sequence, duplicate identity, logical digest, and trust with legacy behavior preserved when reconciliation is disabled
- [x] 4.3 Assemble and verify bounded snapshots, atomically install valid state, and keep incomplete or corrupt snapshots untrusted
- [x] 4.4 Return authoritative match metadata and make the gateway skip stale local cache history only after a successful authoritative miss
- [x] 4.5 Expose bounded reconciliation counters, latency, and trusted/untrusted rank gauges from cache-state

## 5. Durable Stream Semantics

- [x] 5.1 Return owned Kafka record metadata without committing during receive and commit offset plus one only after successful apply or idempotent acceptance
- [x] 5.2 Add bounded failure handling and tests showing failed application leaves the Kafka offset uncommitted

## 6. End-to-End Verification

- [x] 6.1 Add fault-injection tests for sequence gaps, digest mismatch, snapshot repair, newer post-snapshot events, conflicting duplicates, and mixed protocol versions
- [x] 6.2 Run targeted Python and Rust tests plus formatting, lint, and build checks; document configuration defaults and rollout boundaries
