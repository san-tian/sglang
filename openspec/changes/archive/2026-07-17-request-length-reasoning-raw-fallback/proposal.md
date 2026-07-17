# Route reasoning requests by existing raw length

## Why

The isolated request-length Gateway enables raw context counting because GLM-5.2 does not expose a usable chat template to the Router. The current raw-shape gate still rejects `reasoning` and `reasoning_effort`, causing nearly all reasoning Chat traffic to return 503 before worker dispatch even though raw message tokens are available. The operator explicitly prefers approximate routing on that existing length over rejection.

## What Changes

- When `ALLOW_RAW_CONTEXT_TOKENS=1` is enabled, accept `reasoning` and `reasoning_effort` for routing-only raw context estimation.
- Continue adding the requested output-token budget and applying the existing `65535/65536` worker boundary.
- Preserve the original reasoning body and do not inject approximate raw tokens as engine `input_ids`.
- Keep tools, multimodal content, custom template controls, task mode, and continuation shapes fail-closed.

## Scope

This changes Router code, tests, and the raw-context specification. Only the isolated `18084` debug Gateway enables the opt-in flag; production gateways remain unchanged.
