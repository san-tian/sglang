## Context

Distributed cache-state indexes prefix ownership by physical `worker_url + dp_rank`. That is correct for standalone workers and native PD Prefill candidates, but production also registers logical PD proxy workers such as MI300X 2P2D HA Routers. Those logical workers expose one Gateway URL while the reusable KV cache lives on multiple physical Prefill engines behind the proxy.

Current cache-aware TTFT scoring compares cache-state matches directly against candidate worker URLs. A physical Prefill match such as `http://10.0.0.10:30100` cannot match the logical candidate `https://llm-mi300x-rdma02-router...`, so the logical worker gets no cache credit.

## Goals / Non-Goals

**Goals:**

- Let a logical worker receive cache credit when a configured physical Prefill member has a remote cache-state match.
- Keep cache-state identity physical and authoritative per Prefill rank.
- Make the behavior opt-in and fallback-compatible.
- Avoid request hints or any protocol coupling between the outer Gateway and inner PD Router.

**Non-Goals:**

- Do not merge physical Prefill KV events under a synthetic logical worker URL.
- Do not add `x-sglang-preferred-prefill-url` or any other Prefill-selection hint.
- Do not change the Event Hubs/cache-event-agent wire format.
- Do not make logical PD proxy workers eligible for multiplicative LMetric modes.

## Decisions

1. Add an opt-in worker URL suffix for physical Prefill members.

   The static worker entry parser already supports deployment-only suffixes such as `@backend=` and `@routes=`. Add `@prefill_members=` with a comma-separated list of normalized physical Prefill URLs. Store the parsed members in worker capabilities and propagate them to `Worker`.

   Alternative considered: a separate JSON env map. That would avoid long URLs, but it introduces another config source that must be kept in sync with `WORKER_URLS`. The suffix keeps membership co-located with the logical worker entry.

2. Preserve physical identity in cache-state.

   Workers keep publishing KV events as physical Prefill URLs. Cache-state continues to return physical URLs. The Gateway translates those physical URLs to the logical candidate only during scoring.

   Alternative considered: inserting synthetic logical ownership into cache-state. That can produce false hits because the inner proxy may route to a different Prefill; it also makes snapshot/clear semantics unsafe.

3. Score logical candidates by the best member match.

   For each candidate worker, cache-aware scoring treats a match as present when either the candidate URL itself matched or any configured member URL matched. The matched block count remains the remote deepest match count. This is conservative with the existing response shape, which returns one deepest matched prefix and all workers holding it.

   Alternative considered: extending cache-state response to return per-worker depths. That is more precise but larger and cross-cutting. The current deepest-prefix response is enough to award group-level credit safely.

4. Do not pass Prefill hints.

   The outer Gateway only chooses the logical worker. The inner PD Router independently selects Prefill from its own cache-aware/load policy. This can lose some cache-hit opportunities, but it preserves loose coupling and avoids forcing an unhealthy or overloaded Prefill.

## Risks / Trade-offs

- Group cache credit may overestimate benefit if the inner router chooses a different Prefill -> mitigate by keeping credit as a TTFT scoring input rather than a hard route guarantee.
- Misconfigured member URLs can prevent credit from being awarded -> mitigate with URL normalization at startup and validation errors for malformed member entries.
- Very long worker URL suffixes can become cumbersome -> mitigate by keeping the first implementation static and opt-in; a future config map can be added if needed.
- A logical worker without member mapping keeps current behavior -> safe fallback, but no cache benefit.

## Migration Plan

1. Land router code and tests with no production config changes.
2. Enable `@prefill_members=` only on a canary logical PD worker entry.
3. Verify cache-state match metrics, chosen worker logs, and no increase in 5xx/timeout.
4. Roll back by removing `@prefill_members=` from the worker entry and redeploying the previous Gateway config or image; no worker restart is required.

## Open Questions

None for the no-hint implementation. Future work may evaluate whether a separate config map is easier to operate than URL suffixes for large PD groups.
