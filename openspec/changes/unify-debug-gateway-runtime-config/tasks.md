## 1. Runtime configuration

- [x] 1.1 Add a strict YAML runtime schema for remote-only cache state, Prefill Work fallback, Redis RouterState, SLS, and route-decision logging
- [x] 1.2 Compile runtime settings to the existing Gateway environment contracts while leaving curve_ms and pd_beta in PREFILL_SCORE_PROFILES_JSON
- [x] 1.3 Fail startup and validate-only before binding when required cache-state, Redis, or SLS environment variables are missing
- [x] 1.4 Add unit tests for runtime compilation, strict parsing, and secret-safe missing-variable diagnostics

## 2. Upstream error observability

- [x] 2.1 Emit a dedicated structured worker_upstream_failure event for upstream HTTP 500/502 responses
- [x] 2.2 Emit the same event family for connection, timeout, and response-body transport failures without changing proxy response semantics
- [x] 2.3 Add bounded allowlisted JSON error summaries and tests proving prompt, authorization, and arbitrary body fields are excluded

## 3. Configuration and source reconciliation

- [x] 3.1 Update the private debug Gateway YAML to the new runtime section and exactly four key policies
- [x] 3.2 Audit the currently running debug Gateway and worker source revisions against the feature branch and merge any missing source changes

## 4. Verification and closure

- [x] 4.1 Run focused and full Rust tests, cargo check, clippy, release build, Python syntax tests, and YAML validate-only
- [ ] 4.2 Validate the OpenSpec change strictly, archive it, and validate the archived specifications
- [ ] 4.3 Scan the public diff for secrets and internal infrastructure identifiers, then commit and push the feature branch
