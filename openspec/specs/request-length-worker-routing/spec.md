# request-length-worker-routing Specification

## Purpose
TBD - created by archiving change request-length-worker-routing. Update Purpose after archive.
## Requirements
### Requirement: Worker registrations can declare a request-length range
The gateway SHALL accept optional positive `@min_context_tokens=N` and existing `@max_context_tokens=N` suffixes on static worker URL registrations. A worker with both bounds SHALL be eligible only when the request's computed input token count is within the inclusive range; a worker with only one bound SHALL be constrained on that side, and a worker with no bounds SHALL retain legacy eligibility. Requested output limits SHALL NOT affect this range decision.

#### Scenario: Register a lower-bounded worker
- **WHEN** `WORKER_URLS` contains `http://nvidia:30000@min_context_tokens=65536`
- **THEN** static discovery SHALL emit a worker with a lower input-length bound of `65536` and the base URL SHALL remain `http://nvidia:30000`

#### Scenario: Reject invalid or contradictory bounds
- **WHEN** a worker URL contains a zero, non-integer, or `min_context_tokens` greater than `max_context_tokens`
- **THEN** gateway startup SHALL fail with a configuration error rather than silently changing the worker's eligibility

### Requirement: Routing filters workers by registered request-length range
Before policy scoring on every generation route, the gateway SHALL remove workers whose registered range does not contain the computed input token count. A request exactly equal to a lower bound SHALL be eligible, and a request exactly equal to an upper bound SHALL be eligible. Output-limit fields, including `max_tokens`, `max_completion_tokens`, and `max_output_tokens`, SHALL NOT change the eligible range.

#### Scenario: Split a 64K pool
- **WHEN** AMD workers are registered with `@max_context_tokens=65535`, NVIDIA workers with `@min_context_tokens=65536`, and a request has `65535` input tokens
- **THEN** the router SHALL select only eligible AMD workers

#### Scenario: Route a request at or above the threshold
- **WHEN** the same pool receives a request with `65536` input tokens or more
- **THEN** the router SHALL select only eligible NVIDIA workers

#### Scenario: Ignore requested output budget
- **WHEN** two otherwise identical requests have the same input token count but different or omitted output-limit fields
- **THEN** the router SHALL produce the same bounded-worker candidate pool for both requests

#### Scenario: Preserve unbounded-worker compatibility
- **WHEN** a candidate pool includes a worker with no request-length bounds
- **THEN** that worker SHALL remain eligible for known input lengths unless another independent eligibility filter removes it

### Requirement: Unknown request length fails closed for bounded workers
When the gateway cannot reliably compute the request's input token count, it SHALL exclude every worker with either request-length bound and SHALL continue only with unbounded workers.

#### Scenario: Unknown length with an unbounded fallback
- **WHEN** a request has an unsupported or un-tokenizable prompt and the candidate pool contains one bounded worker and one unbounded worker
- **THEN** the router SHALL skip the bounded worker and route to the unbounded worker

#### Scenario: Unknown length with no eligible fallback
- **WHEN** input length is unknown and every healthy candidate has a request-length bound
- **THEN** the router SHALL reject the request with the existing no-context-eligible-workers 503 contract

