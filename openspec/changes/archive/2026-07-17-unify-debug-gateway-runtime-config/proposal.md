## Why

The combined debug Gateway branch now contains request-length routing and PrefillScore code, but its administrator YAML does not yet declare the live remote cache, Prefill snapshot fallback, Redis reservation, or SLS observability contract. Deploying it as-is would leave important behavior dependent on implicit shell inheritance and would not provide enough structured evidence when a worker returns HTTP 500 or 502.

## What Changes

- Extend the administrator Gateway YAML with a strict runtime section for remote-only cache state, Prefill snapshot fallback thresholds, environment-backed Prefill curves, required Redis RouterState, and required SLS/route-decision logging.
- Keep `curve_ms` and `pd_beta` in `PREFILL_SCORE_PROFILES_JSON`; do not duplicate calibrated profiles into YAML and retain fixed-throughput fallback when no profile is configured.
- Validate required cache-state, Redis, and SLS environment dependencies before the Gateway starts accepting traffic.
- Emit bounded, structured SLS-compatible events for worker HTTP 500/502 responses, including route, worker identity, status, latency, request correlation, content metadata, and a sanitized bounded error summary without credentials or request content.
- Preserve the new four-key inventory as the only administrator-configured client key set.
- Audit live debug Gateway and SGLang worker source commits against the feature branch and merge any missing source history before deployment.
- **BREAKING**: legacy debug client keys are not retained by the new administrator YAML.

## Capabilities

### New Capabilities

- `gateway-runtime-configuration`: Strict administrator configuration and environment dependency contract for cache state, Prefill fallback, profiles, Redis reservations, SLS, route-decision logging, and the four-key inventory.
- `upstream-error-observability`: Safe structured diagnostics for worker HTTP 500/502 responses delivered through the existing tracing/SLS pipeline.

### Modified Capabilities

- `distributed-cache-state`: Administrator-configured debug Gateways use remote-only cache lookup and never silently re-enable a local route-history tree.
- `prefill-score-load-snapshot`: Snapshot failure/recovery thresholds and profile source become explicit runtime configuration while preserving ordered fallback behavior.

## Impact

- Rust Gateway YAML loader, startup validation, cache/Prefill environment compilation, proxy response handling, structured tracing, and tests.
- Private administrator YAML and debug Gateway launch contract; secret values remain outside Git.
- Existing remote cache-state, Redis RouterState, SLS, route-decision, and worker load endpoints; no API shape change is required for clients.
- SGLang Python worker source is audited and only merged when the live commit is not already an ancestor of this feature branch.
