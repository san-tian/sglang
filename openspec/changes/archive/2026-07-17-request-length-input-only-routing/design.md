## Context

The request-length debug branch registers AMD workers with an upper bound of 65535 and NVIDIA workers with a lower bound of 65536. Ingress currently combines the tokenized prompt with the largest declared output limit, or an 8192-token default reserve, before applying those bounds. That behavior models total context capacity, but this experiment is intended to model prefill work, which depends on input length rather than the caller's requested decode allowance.

The implementation is shared by Chat, Messages, Responses, and generic passthrough handlers. Only the isolated port-18084 binary from this branch currently enables the bounded mixed-hardware pool.

## Goals / Non-Goals

**Goals:**

- Make bounded-worker eligibility depend only on reliable input token count.
- Ensure output-limit values and omissions cannot change hardware selection.
- Keep unknown-input fail-closed handling and exact 65535/65536 boundary behavior.
- Keep the change confined to the request-length debug branch and service.

**Non-Goals:**

- Changing production `deploy-prod` gateways or worker membership.
- Enforcing an engine's true total context-window capacity at the Gateway.
- Changing cache-aware tokenization, request forwarding, or decode generation limits.
- Making approximate raw tokenization engine-equivalent.

## Decisions

1. Use the already computed reliable prompt token count directly as the range-filter value. Output-limit fields will no longer participate in bounded-worker eligibility. This matches prefill cost and makes a request's hardware class stable when only its decode allowance changes.
2. Retain the existing `min_context_tokens` and `max_context_tokens` registration suffixes for this experiment. Introducing parallel suffixes would expand discovery, reconciliation, telemetry, and migration work without changing the isolated experiment's outcome. Their OpenSpec contract will explicitly define the bounded value as input tokens.
3. Rename ingress variables and helper-facing terminology to `routing_input_tokens` where practical. Worker metadata remains unchanged for compatibility, while route logs and code avoid implying that output budget is included.
4. Preserve fail-closed behavior when reliable input tokenization returns `None`. Ignoring output budget does not weaken the existing raw-token reliability gates.

## Risks / Trade-offs

- [A request can ask for a large output after a near-limit input and still select AMD] -> This is intentional for the prefill-routing experiment; the engine remains authoritative for actual context validation.
- [Existing metadata names still say `context_tokens`] -> The code and specification document the input-only interpretation, and the capability remains isolated from production.
- [A route handler could accidentally keep adding an output field] -> Unit tests cover output-budget invariance and handler call sites are simplified to pass only the reliable input count.

## Migration Plan

1. Build and test the branch locally.
2. Copy the content-addressed binary to the debug VM and create a timestamped `run.sh` rollback copy.
3. Update only `sgl-router-length-split.service` to the new binary and config revision, then restart that service.
4. Verify readiness, 18-worker registration, a short-input request with a deliberately large output budget routing to AMD, and a long-input boundary unit test routing to NVIDIA.
5. Roll back by restoring the saved `run.sh` and restarting only the same service.

## Open Questions

None.
