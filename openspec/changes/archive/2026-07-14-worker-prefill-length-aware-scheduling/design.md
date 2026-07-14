## Context

The scheduler already refreshes prefix matches before a cache-agnostic prefill scheduling round so load snapshots can report current uncached work. FCFS does not use that information for ordering, so one long prompt can sit ahead of many short prompts. The Gateway now understands optional role-labelled token load, but old workers and default scheduling must remain compatible.

The change spans scheduler ordering, model-independent server arguments, the internal DP load snapshot, `/v1/loads`, and the legacy `/get_load` projection. Native PD deployments require phase ownership to stay explicit: Prefill backlog belongs to P workers, while D worker request pressure is not Prefill work.

## Goals / Non-Goals

**Goals:**

- Lower short-request TTFT under mixed prompt lengths by changing waiting-prefill order.
- Recompute remaining uncached work after cache state changes.
- Preserve business priority and provide a same-priority starvation bound.
- Export bounded load data for conservative and candidate-aware Gateway scoring.
- Keep source merge and default worker startup behavior inert until explicitly enabled.

**Non-Goals:**

- Preempt decode, interrupt an active chunked prefill, or replace `PrefillAdder` capacity checks.
- Automatically tune aging or max-wait values.
- Combine native PD Prefill work with a paired Decode batch factor.
- Activate or restart any existing production worker as part of the source merge.

## Decisions

### 1. Sort by live uncached work with finite aging

`--schedule-policy prefill-length-aware` uses the current prefix match and computes:

```text
uncached_tokens = max(0, sequence_length - matched_prefix_tokens)
wait_seconds = max(0, now - wait_queue_entry_time)
effective_work = max(0, uncached_tokens - aging_rate * wait_seconds)
```

The sort key is business priority, overdue class, effective work, then queue-entry time. Priority remains outermost. Overdue requests at the same priority precede non-overdue requests and use FCFS. A per-round sort is preferred over persistent size buckets because cache matches and aging change while requests wait.

### 2. Leave execution and admission ownership unchanged

The policy only reorders `waiting_queue`. Existing chunked-prefill continuation remains ahead of that queue, and `PrefillAdder` still decides whether a request fits token, KV, LoRA, and other constraints.

### 3. Export bounded aggregates instead of request details

When the policy is active, each DP-rank snapshot reports chunked remainder and cumulative uncached work for fixed effective-work buckets, grouped by at most 32 priority values. Exceeding the limit marks detail incomplete so the Gateway can fall back to conservative totals. Individual requests are not exported.

The internal shared-memory snapshot version increments because its typed payload gains fields. `/get_load` retains all historical fields and adds optional `num_running_reqs`, `num_waiting_uncached_tokens`, `load_role`, and `prefill_queue` fields.

### 4. Keep PD phase accounting conservative

Native Prefill workers include both schedulable waiting requests and not-yet-finished bootstrap requests in total waiting uncached tokens. Candidate-aware bucket detail covers only the actual schedulable waiting queue. Decode-only workers report zero Prefill token work and reject the Prefill-specific scheduling policy at startup.

### 5. Preserve defaults and fail early on invalid configuration

FCFS remains the source default. Aging must be finite and non-negative; max wait must be finite and positive. Validation runs before model loading so a bad canonical startup command fails clearly.

## Risks / Trade-offs

- [Sorting every prefill round adds queue CPU work] -> The policy is opt-in, prefix matching already occurs for load accuracy, and benchmarks must observe scheduler CPU plus throughput.
- [Aggressive aging can erase short-first benefit] -> Keep aging and hard max wait separate and use measured prompt-length TTFT buckets to tune them.
- [Bucket rounding overestimates candidate work-ahead] -> Use conservative cumulative buckets and deterministic fallback to total token work.
- [Mixed worker versions omit new fields] -> Keep fields optional and let the Gateway fall back pool-wide to its existing additive score.
- [Business priority can intentionally delay lower classes beyond max wait] -> Document that the starvation bound is within one business-priority class; priority isolation remains authoritative.

## Migration Plan

1. Merge the source capability with FCFS still default.
2. Encode explicit role-aware scheduling profiles for future worker creation; existing workers remain explicitly grandfathered as FCFS.
3. Validate a new or isolated Prefill worker with `prefill-length-aware`, then use the required drain/restart workflow for any later existing-worker activation.
4. Roll back runtime behavior by restoring `--schedule-policy fcfs`; source rollback is a normal revert because response additions are backward compatible.

## Open Questions

- The initial `256 tokens/s`, `30s`, and fixed bucket bounds are experiment-backed starting values, not universal production constants.
- A later design may add quantile sketches if real priority cardinality or prompt distributions make fixed buckets too coarse.
