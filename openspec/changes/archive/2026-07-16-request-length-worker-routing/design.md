## Context

`origin/deploy-prod` already computes a conservative prompt-plus-requested-output token budget and filters workers by `max_context_tokens`. That implementation is sufficient for a heterogeneous context ceiling, but not for a deliberate short/long pool split: an unbounded NVIDIA worker remains eligible for short requests, and a bounded AMD worker has no way to advertise that it should not receive long requests. The existing static URL suffix parser and `WorkerSpec` discovery path are the correct configuration seam because the deployment already renders `WORKER_URLS` from environment-controlled manifests.

## Goals / Non-Goals

**Goals:**

- Carry an optional lower context bound from static URL parsing through discovery, worker state, reconciliation, and route selection.
- Combine lower and upper bounds with inclusive comparisons around a 64K boundary.
- Keep all generation routes and PD prefill/decode eligibility consistent with the same filter.
- Preserve legacy behavior and serialized payload compatibility when the new field is absent.
- Add parser, registry, and HTTP tests that prove the AMD/NVIDIA split and unknown-length fail-closed behavior.

**Non-Goals:**

- No new tokenizer or request-length estimator; use the existing `required_context_tokens` result.
- No changes to worker startup parameters, model serving, APIM/AFD, watchdog, OTel, or production truth.
- No automatic fallback from an out-of-range worker to a different range; normal healthy-worker and policy selection operates only on the filtered set.

## Decisions

1. **Use `@min_context_tokens=N` alongside `@max_context_tokens=N`.**
   This keeps registration declarative and composable with existing `WORKER_URLS`, avoids a second environment grammar, and lets operators express half-open pools as `max=65535` and `min=65536`. A separate range string was rejected because it would duplicate suffix parsing and make per-worker combinations harder to review.

2. **Store the lower bound in `WorkerSpec` and `Worker`.**
   Discovery, runtime lease reconciliation, PD resolution, and policy filtering all consume the same immutable worker metadata. The field is `Option<usize>` with serde default `None`, so older runtime lease documents and tests deserialize unchanged.

3. **Filter before policy scoring and before PD pairing.**
   Context eligibility must constrain the full candidate set before cache/load/round-robin selection. For PD, both Prefill and Decode candidate selection receive the same computed budget, preventing a long request from selecting a valid Prefill and an out-of-range Decode worker.

4. **Keep bounded metrics finite and preserve existing labels.**
   Existing over-limit and unknown-length outcomes remain unchanged. Add below-minimum and mixed-out-of-range outcomes only where needed, with fixed enum-to-label mappings; worker URLs and numeric bounds never become metric labels.

5. **Reject contradictory registration bounds at startup.**
   A `min_context_tokens` greater than `max_context_tokens` describes an empty range and almost always indicates a deployment typo. Failing before serving traffic is safer than registering a worker that can never be selected.

## Risks / Trade-offs

- **[Approximate/unknown token count]** Some valid requests cannot be tokenized at ingress and will be rejected when all workers are bounded. **Mitigation:** retain unbounded workers as an explicit compatibility fallback and expose the existing 503 reason when none exists.
- **[Broad struct change]** Adding a field to `WorkerSpec` touches many test fixtures and discovery paths. **Mitigation:** use serde defaults, update all constructors centrally, and run the full router test target.
- **[Deployment typo]** A threshold boundary can be configured inconsistently across workers. **Mitigation:** validate positive values and `min <= max`, and include exact 65535/65536 tests.

## Migration Plan

1. Build the router branch and run strict OpenSpec validation plus targeted/full router tests.
2. For the debug VM only, keep existing worker URLs as rollback and prepare a candidate registry with AMD `@max_context_tokens=65535` and NVIDIA `@min_context_tokens=65536`.
3. Start a green debug instance on the prescribed temporary port, verify readiness, member metadata, and route-decision evidence, then use the existing blue/green drain flow after explicit user approval.
4. Roll back by restoring the previous binary and `run.sh` through the same blue/green flow; no worker or production gateway restart is required.

## Open Questions

- The first debug rollout should confirm whether the operator wants the boundary based on prompt tokens alone or the existing prompt-plus-output budget. This implementation intentionally follows the existing budget contract so all routes remain consistent.
