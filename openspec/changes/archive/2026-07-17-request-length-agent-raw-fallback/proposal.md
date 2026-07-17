# Route agent Chat requests by existing raw length

## Why

The isolated request-length Gateway still rejects most agent Chat traffic after the reasoning-only follow-up. Requests that carry tools, text-part arrays, template controls, task controls, or continuation fields are classified as unknown even when the Router has already extracted raw message tokens. Because every worker has a context bound, that conservative classification removes the entire pool and returns 503 before dispatch.

## What Changes

- When `ALLOW_RAW_CONTEXT_TOKENS=1` is enabled, let any Chat request with available extracted text tokens use that approximate count for context-range routing.
- Preserve the requested output-token budget and the existing `65535/65536` worker boundary.
- Preserve the original request and keep approximate `input_ids` injection disabled.
- Keep the default-disabled behavior and the stricter Messages/Responses raw-shape gate unchanged.

## Scope

This changes Router code, tests, and the raw-context specification. Only the isolated `18084` debug Gateway enables this opt-in; production gateways remain unchanged.
