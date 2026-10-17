## ADDED Requirements

### Requirement: Raw context counting is opt-in

The gateway SHALL default to fail-closed unknown-length handling for bounded workers. When `ALLOW_RAW_CONTEXT_TOKENS=1` is set, eligible generation handlers MAY use raw prompt tokenization for context-range filtering when the request shape has no unsupported template controls, tools, multimodal content, or stateful continuation fields.

#### Scenario: Default chat request remains fail-closed

- **WHEN** a chat model has no engine-equivalent chat encoder and the gateway has bounded workers but `ALLOW_RAW_CONTEXT_TOKENS` is unset
- **THEN** the request length SHALL remain unknown and the gateway SHALL preserve the existing bounded-worker exclusion/503 behavior

#### Scenario: Opt-in short chat request uses raw count

- **WHEN** the same gateway sets `ALLOW_RAW_CONTEXT_TOKENS=1` and a simple text chat request has a raw prompt-plus-output budget below `65536`
- **THEN** the context-range filter SHALL use that budget and allow a worker registered with `@max_context_tokens=65535`

#### Scenario: Unsupported shape remains unknown

- **WHEN** raw counting is enabled but the request contains tools, multimodal content, template controls, or stateful continuation fields
- **THEN** the gateway SHALL not treat the raw count as reliable for hard context eligibility
