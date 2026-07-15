## Why

Logical PD proxy workers hide multiple physical Prefill engines behind one Gateway worker URL. Cache-state currently returns physical Prefill URLs, while Gateway scoring compares cache hits against logical worker URLs, so these logical workers cannot receive cache-hit credit even when one of their Prefill members holds the prefix.

## What Changes

- Add opt-in Gateway configuration for mapping a logical worker URL to its physical Prefill member URLs.
- Normalize remote cache-state matches through that mapping so a logical worker receives cache credit when any configured Prefill member matches.
- Preserve the current no-hint request flow: Gateway still dispatches only to the logical worker URL, and the inner PD router independently selects Prefill.
- Preserve safe fallback when no mapping is configured, when a mapping is malformed, or when cache-state has no useful match.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `distributed-cache-state`: remote cache-state matches remain physical-worker based, but Gateway consumers may map those physical workers to logical candidates.
- `ttft-first-routing`: TTFT-first cache-aware scoring may award cache-hit credit to a logical PD proxy candidate based on its configured physical Prefill members.

## Impact

- Affected Rust router code under `experimental/sgl-router`, primarily static worker capability parsing and cache-aware scoring.
- Adds configuration surface for logical Prefill membership without changing public inference request APIs.
- Does not change cache-event wire format, cache-state storage identity, worker KV event identity, or PD router request headers.
