## Context

The public debug Gateway and the request-length debug Gateway currently inherit most operational settings from shared shell files. The combined feature branch contains remote-only cache state, Prefill work fallback, Redis reservation, SLS, route-decision logging, length routing, and four-key policy code, but the administrator YAML only compiles worker/key/routing data. The live services also use different cache sources, and existing INFO access logs show final status without a dedicated safe worker-error event that operators can reliably search in SLS.

The configuration contains administrator-visible client and backend credentials. Cache-state, Redis, and SLS infrastructure secrets remain process environment values and must never be serialized into public Git artifacts or error logs.

## Goals / Non-Goals

**Goals:**

- Make the target debug Gateway runtime contract explicit and startup-validated.
- Standardize on remote-only cache lookup, Prefill snapshot fallback `10000/3/2`, required Redis RouterState, required SLS, and enabled route-decision logging.
- Preserve environment-owned `PREFILL_SCORE_PROFILES_JSON` for both `curve_ms` and `pd_beta`, with the existing fixed-throughput fallback when absent.
- Produce searchable, bounded SLS events for upstream 5xx and transport failures without logging prompts or credentials.
- Prove that live debug Gateway and worker source commits are included in the combined feature branch before building the candidate artifact.

**Non-Goals:**

- Calibrating or inventing hardware curves.
- Adding Decode or network transfer time to PrefillScore.
- Moving Redis, SLS, or cache-state secrets into YAML or Git.
- Restarting workers or changing production membership as part of this code change.

## Decisions

### 1. Declare policy in YAML and keep infrastructure secrets in fixed environment contracts

Add a strict `runtime` object to the administrator YAML. It declares `cache_state.mode=remote_only`, Prefill fallback thresholds, `profile_source=environment_optional`, `router_state.mode=redis_required`, and required SLS/route-decision observability. The loader compiles non-secret values into the existing environment/CLI contract and validates that required external variables are present before argument parsing.

This avoids a generic `value_from` registry while keeping credentials out of public source. The alternative of copying Redis/cache/SLS values into the YAML would make one secret-bearing file responsible for unrelated infrastructure and would increase accidental disclosure risk.

### 2. Fail startup when required operational dependencies are absent

Remote-only requires cache-state URL/token and page size. Redis-required needs a Redis URL. SLS-required needs endpoint and access credentials. The Gateway fails before listener creation when these variables are missing. An optional Prefill profile is different: no calibrated profile is a supported state and uses fixed-throughput fallback.

The alternative of silently disabling these systems recreates configuration drift and makes the candidate differ from the debug service being replaced.

### 3. Keep one environment JSON for calibrated curves and PD coefficients

`PREFILL_SCORE_PROFILES_JSON` remains the only source for `curve_ms` and `pd_beta`. The administrator YAML selects `environment_optional` but does not duplicate the profile content. Existing parser validation and monotonicity checks continue to reject unusable curves and fall back safely.

### 4. Emit dedicated upstream failure events at the proxy boundary

The proxy has the worker URL, route path, response status, correlation headers, elapsed time, and, for buffered JSON responses, the upstream body. It emits `worker_upstream_failure` at WARN for every upstream 5xx and transport error. JSON error bodies are reduced to a bounded allowlist of scalar fields such as `message`, `type`, `code`, and `detail`; streaming errors log metadata without buffering an unbounded stream.

This event flows through the existing tracing subscriber and SLS layer. It does not depend on route-decision sampling and never includes request bodies, authorization headers, full response bodies, or arbitrary response headers.

### 5. Treat source ancestry as a release gate

Before candidate build, every currently running debug Gateway commit and every discoverable live SGLang worker source commit must be an ancestor of the feature branch or be deliberately merged. Unknown runtime versions are reported as an unresolved deployment gate instead of guessed from binary names.

## Risks / Trade-offs

- **[Risk] Required environment validation prevents startup during a partial secret/config rollout.** -> Validate candidate configuration before cutover and keep the old service running until green is ready.
- **[Risk] Worker error messages may contain sensitive or high-cardinality text.** -> Extract only bounded scalar error fields, remove control characters, cap bytes, and never record request content or headers.
- **[Risk] Streaming 5xx bodies may contain useful diagnostics that are not logged.** -> Log status/content metadata immediately; do not buffer an unbounded streaming response merely for diagnostics.
- **[Risk] Remote cache-state failure removes cache affinity.** -> Continue routing by PrefillScore/reservation fallback and record the remote query failure.
- **[Risk] A missing calibrated profile reduces hardware accuracy.** -> Preserve explicit `fixed-throughput` provenance and add profiles only after benchmark data exists.

## Migration Plan

1. Validate the combined branch contains the live Gateway/worker source commits.
2. Validate the private YAML and required environment contract with `GATEWAY_CONFIG_VALIDATE_ONLY=1`.
3. Build a release binary and start an isolated green debug Gateway on the existing private green port.
4. Verify ready, four-key model visibility, remote-only mode, Redis/SLS initialization, route-decision events, and synthetic worker-error logging.
5. After explicit operator approval, redirect new connections, gracefully drain the old process, start the formal service, remove redirect, and drain green.
6. Roll back with the previous binary/run/config through the same green-and-drain process.

## Open Questions

- Calibrated `curve_ms` and `pd_beta` values remain pending real machine profiling; absence is an intentional fallback state, not a startup error.
- Worker source versions that cannot be read from runtime metadata require host-level read-only verification before any worker rollout is considered.
