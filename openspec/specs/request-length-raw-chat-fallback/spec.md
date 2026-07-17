# request-length-raw-chat-fallback Specification

## Purpose
TBD - created by archiving change request-length-raw-chat-fallback. Update Purpose after archive.
## Requirements
### Requirement: Raw context counting is opt-in

The gateway SHALL default to fail-closed unknown-length handling for bounded workers. When `ALLOW_RAW_CONTEXT_TOKENS=1` is set, the Chat handler SHALL use available extracted text tokens for input-range filtering even when the request carries tools, text-part content, template controls, task controls, reasoning controls, or continuation fields. Requested output limits SHALL NOT be added to the extracted input length. This is a routing-only approximation: the gateway SHALL forward the original request and SHALL NOT inject approximate raw tokens as engine input. Other generation handlers SHALL retain their stricter raw-shape eligibility checks.

#### Scenario: Default chat request remains fail-closed

- **WHEN** a chat model has no engine-equivalent chat encoder and the gateway has bounded workers but `ALLOW_RAW_CONTEXT_TOKENS` is unset
- **THEN** the request length SHALL remain unknown and the gateway SHALL preserve the existing bounded-worker exclusion/503 behavior

#### Scenario: Opt-in short chat request uses raw count

- **WHEN** the same gateway sets `ALLOW_RAW_CONTEXT_TOKENS=1` and a Chat request has fewer than `65536` extractable raw message tokens, regardless of its requested output budget
- **THEN** the input-range filter SHALL allow a worker registered with `@max_context_tokens=65535`

#### Scenario: Agent chat shapes use existing raw request length

- **WHEN** the gateway sets `ALLOW_RAW_CONTEXT_TOKENS=1` and a Chat request with extractable text tokens includes tools, text-part content, template controls, task controls, reasoning controls, or continuation fields
- **THEN** the input-range filter SHALL use only the existing raw message token count rather than rejecting the request as unknown length or adding its requested output budget
- **AND** the worker SHALL receive the original request and remain responsible for engine-side prompt construction

#### Scenario: No extractable text remains unknown

- **WHEN** raw counting is enabled but no text tokens can be extracted from the Chat request
- **THEN** the request length SHALL remain unknown for hard input-range eligibility

#### Scenario: Other handlers remain strict

- **WHEN** raw counting is enabled for a Messages or Responses request with unsupported template, tool, multimodal, reasoning, task, or stateful fields
- **THEN** that handler SHALL preserve its strict raw-shape eligibility checks

