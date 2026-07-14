## 1. Regression Coverage

- [x] 1.1 Add deep reverse-ordered and invalid-dependency snapshot tests that pin bounded dependency traversal and atomic rejection
- [x] 1.2 Add a single-thread Tokio regression test proving slow apply/commit work does not starve an independent liveness task

## 2. Non-Blocking Reconciliation

- [x] 2.1 Replace repeated pending-list snapshot scans with a linear adjacency traversal while preserving duplicate-edge and shared-hash semantics
- [x] 2.2 Move Kafka apply plus synchronous commit and HTTP KV-event application onto Tokio's blocking pool without changing commit ordering
- [x] 2.3 Add bounded slow-record duration/progress observability without payload or secret logging

## 3. Source Verification

- [x] 3.1 Run targeted Rust tests for HashTree, cache-state, Kafka commit semantics, and the binary runtime regression
- [x] 3.2 Run router formatting, lint, build, and strict OpenSpec validation
- [x] 3.3 Remove the temporary read-only Event Hubs diagnostic probe before commit

## 4. Shadow Rollout

- [ ] 4.1 Merge the source change into `san-tian/sglang@deploy-prod` and build an immutable router/cache-state image with recorded digest
- [ ] 4.2 Update deployment truth and roll only `llm-cache-state-glm52-b` to the fixed image with the previous digest as rollback
- [ ] 4.3 Verify B restart count is stable, health remains responsive, offsets advance, and reconciliation survives two snapshot intervals
