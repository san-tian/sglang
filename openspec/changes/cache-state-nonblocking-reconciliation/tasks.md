## 1. Regression Coverage

- [x] 1.1 Add deep reverse-ordered and invalid-dependency snapshot tests that pin bounded dependency traversal and atomic rejection
- [x] 1.2 Add a single-thread Tokio regression test proving slow apply/commit work does not starve an independent liveness task
- [x] 1.3 Add regression coverage for post-apply checkpoint ordering, offset-plus-one storage, and periodic auto-commit configuration

## 2. Non-Blocking Reconciliation

- [x] 2.1 Replace repeated pending-list snapshot scans with a linear adjacency traversal while preserving duplicate-edge and shared-hash semantics
- [x] 2.2 Move Kafka apply plus synchronous commit and HTTP KV-event application onto Tokio's blocking pool without changing commit ordering
- [x] 2.3 Add bounded slow-record duration/progress observability without payload or secret logging
- [x] 2.4 Replace per-record synchronous broker commits with post-apply local offset checkpoints and periodic batched auto-commit callbacks

## 3. Source Verification

- [x] 3.1 Run targeted Rust tests for HashTree, cache-state, Kafka commit semantics, and the binary runtime regression
- [x] 3.2 Run router formatting, lint, build, and strict OpenSpec validation
- [x] 3.3 Remove the temporary read-only Event Hubs diagnostic probe before commit
- [x] 3.4 Run the checkpoint follow-up's targeted tests, router formatting, lint, release build, and strict OpenSpec validation

## 4. Shadow Rollout

- [x] 4.1 Merge the first non-blocking source change into `san-tian/sglang@deploy-prod`, build immutable image digest `sha256:eb5c2f19d4e780d4a8b17a58dc4152e88fc9b4bceaa835114f97e66e3475e6ec`, and verify B liveness no longer restarts
- [x] 4.2 Record the first B canary's production throughput evidence and stop expansion when synchronous commits consume slower than Event Hubs produces
- [x] 4.3 Merge the checkpoint follow-up, build a new immutable image, and update deployment truth
- [x] 4.4 Roll only `llm-cache-state-glm52-b` to the checkpoint image with the first canary digest as rollback
- [x] 4.5 Verify B restart count is stable, health remains responsive, and broker offsets catch up faster than ingress and remain at the live head
- [x] 4.6 Audit live and canonical worker publisher configuration and record the legacy-wire activation prerequisite
- [x] 4.7 Roll the reviewed watchdog 1.67 cache-event-agent lifecycle image and verify stable subscriber PIDs across consecutive healthy monitor rounds
- [ ] 4.8 After separate drain/restart approval, enable reconciliation on one explicitly designated test or canary worker rank and verify two complete snapshot intervals
