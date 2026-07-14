## Why

FCFS waiting-prefill order lets a long uncached prompt delay short requests even when the short requests could complete prefill first. Prefix-cache state can also change while a request waits, so a static input-length estimate is insufficient, while strict shortest-first ordering needs explicit starvation protection.

## What Changes

- Add an opt-in `prefill-length-aware` worker scheduling policy that re-evaluates current uncached input work before each prefill scheduling round.
- Keep business priority outermost, age waiting work at a configurable finite rate, and move same-priority requests past a maximum wait into an FCFS overdue class.
- Preserve chunked-prefill continuation and existing `PrefillAdder` admission constraints.
- Export bounded, role-labelled Prefill queue summaries for Gateway token-aware routing, including native PD Prefill bootstrap work in the conservative total and reporting zero Prefill work from Decode-only workers.
- Keep FCFS as the default and reject invalid tunables or use of the policy on a Decode-only worker.

## Capabilities

### New Capabilities

- `prefill-length-aware-scheduling`: Opt-in waiting-prefill ordering, starvation protection, and bounded phase-aware load reporting.

### Modified Capabilities

None.

## Impact

The change affects the Python scheduler, server arguments, scheduler load snapshots, `/v1/loads`, and the backward-compatible `/get_load` projection. It adds optional response fields and increments the internal shared-memory snapshot version; existing scheduling remains unchanged unless the new policy is selected.
