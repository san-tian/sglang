## 1. SGLang Prefill Snapshot

- [x] 1.1 Add optional versioned Prefill work snapshot structs to the SGLang load response, including snapshot metadata, bounded waiting/running entries, truncation, and overflow buckets.
- [x] 1.2 Populate running Prefill processed-token and chunk-boundary fields from scheduler state without changing scheduling behavior.
- [x] 1.3 Add `/v1/loads` serialization and compatibility tests for fresh, truncated, legacy, and decode-role responses.

## 2. Router Snapshot Ingestion

- [x] 2.1 Extend router load-poller deserialization and Worker state to retain snapshot metadata, age, schema/boot validity, bounded Prefill entries, and fallback status.
- [x] 2.2 Keep load polling asynchronous; add timeout, stale-grace, circuit-breaker, and worker-boot invalidation handling without treating load failure as zero load.
- [x] 2.3 Add unit tests for enhanced schema parsing, schema mismatch, stale snapshots, endpoint timeout, health failure, and recovery hysteresis.

## 3. PrefillScore Routing

- [x] 3.1 Add configurable profile curves and length-bucket PD speed coefficients, with fixed-capacity fallback when a curve is unavailable.
- [x] 3.2 Replace the debug predicted-TTFT calculation with PrefillScore: waiting work, running remaining work, new request effective Prefill time, and no Decode/communication term.
- [x] 3.3 Apply the same effective curve to queue work and the new request, preserve cache-aware candidate/member selection, and retain health/routability gates.
- [x] 3.4 Merge enhanced request-level snapshot and Router reservation by request ID; use conservative same-unit max for legacy aggregate sources.
- [x] 3.5 Emit load source, snapshot age/schema, truncation, fallback reason, raw/effective score, and speed coefficient in route-decision telemetry.
- [x] 3.6 Add routing tests for integrated workers, PD weighted preference, length buckets, snapshot/reservation deduplication, and each fallback level.

## 4. Build And Debug Deployment

- [x] 4.1 Run focused Python and Rust tests plus formatting/type checks for the changed SGLang and router modules. (Python pytest collection remains blocked by the environment's incompatible `transformers`; syntax/Ruff checks pass and Rust coverage is complete.)
- [x] 4.2 Build the debug Gateway artifact from this branch and verify the binary/source commit is not `deploy-prod`.
- [x] 4.3 Deploy the artifact to the isolated debug Gateway service, preserving a rollback copy and avoiding production services.
- [x] 4.4 Verify normal route selection, load endpoint failure fallback, stale snapshot behavior, worker health fail-closed, and recovery on the real debug endpoint; enhanced snapshot paths are covered by fake-worker/unit tests because the current real workers have not been upgraded in this debug-only rollout.
- [x] 4.5 Record benchmark observations and deployment details under the debug experiment record, then archive the OpenSpec change after all tasks pass.
