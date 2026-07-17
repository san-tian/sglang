## MODIFIED Requirements

### Requirement: Raw context counting is opt-in

The gateway SHALL default to fail-closed unknown-length handling for bounded workers. When `ALLOW_RAW_CONTEXT_TOKENS=1` is set, eligible generation handlers MAY use raw prompt tokenization for context-range filtering when the request shape has no unsupported template controls, tools, multimodal content, or stateful continuation fields. A Chat request carrying `reasoning` or `reasoning_effort` SHALL use the available raw message tokens for this routing-only approximation instead of becoming unknown; the gateway SHALL still forward the original request without injecting those approximate tokens as engine input.

#### Scenario: Default chat request remains fail-closed

- **WHEN** a chat model has no engine-equivalent chat encoder and the gateway has bounded workers but `ALLOW_RAW_CONTEXT_TOKENS` is unset
- **THEN** the request length SHALL remain unknown and the gateway SHALL preserve the existing bounded-worker exclusion/503 behavior

#### Scenario: Opt-in short chat request uses raw count

- **WHEN** the same gateway sets `ALLOW_RAW_CONTEXT_TOKENS=1` and a simple text chat request has a raw prompt-plus-output budget below `65536`
- **THEN** the context-range filter SHALL use that budget and allow a worker registered with `@max_context_tokens=65535`

#### Scenario: Reasoning chat uses existing raw request length

- **WHEN** the gateway sets `ALLOW_RAW_CONTEXT_TOKENS=1` and a text Chat request includes `reasoning_effort` or the compatible nested `reasoning` form
- **THEN** the context-range filter SHALL use the existing raw message token count plus requested output budget rather than rejecting the request as unknown length
- **AND** the worker SHALL receive the original reasoning request and remain responsible for engine-side prompt construction

#### Scenario: Unsupported shape remains unknown

- **WHEN** raw counting is enabled but the request contains tools, multimodal content, explicit template controls, task controls, or stateful continuation fields
- **THEN** the gateway SHALL not treat the raw count as reliable for hard context eligibility
