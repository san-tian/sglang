## ADDED Requirements

### Requirement: Worker upstream failures emit structured SLS-compatible events
The Gateway SHALL emit a WARN-level `worker_upstream_failure` tracing event for every worker HTTP 5xx response and every transport failure that the Gateway maps to HTTP 502.

#### Scenario: Worker returns HTTP 500 or 502
- **WHEN** a worker responds to a generation request with HTTP 500 or 502
- **THEN** the Gateway SHALL emit one event containing worker identity, route, upstream status, latency, request correlation, stream mode, content type, and error-body size

#### Scenario: Worker connection fails
- **WHEN** DNS, connect, TLS, timeout, or mid-body transport failure prevents a valid worker response
- **THEN** the Gateway SHALL emit an event with the mapped failure class and available worker, route, latency, and correlation fields

### Requirement: Upstream error summaries are bounded and safe
For buffered JSON worker errors, the Gateway SHALL extract a bounded summary from an allowlist of scalar error fields and SHALL NOT log request bodies, authorization values, arbitrary headers, or complete upstream response bodies.

#### Scenario: JSON error envelope is returned
- **WHEN** a worker error body contains scalar `message`, `type`, `code`, or `detail` fields at the top level or under `error`
- **THEN** the event SHALL contain a control-character-free bounded summary and body metadata while the client response remains unchanged

#### Scenario: Error body is non-JSON or oversized
- **WHEN** a worker returns a non-JSON or oversized error body
- **THEN** the event SHALL record content type, body size, and truncation state without logging the raw body

### Requirement: Error logging is independent of route-decision sampling
Worker failure events SHALL flow through the normal tracing subscriber and configured SLS layer regardless of route-decision sample rate or force-log header.

#### Scenario: Route decision is not sampled
- **WHEN** a request is excluded by route-decision sampling but its selected worker returns a 5xx or transport failure
- **THEN** the worker failure event SHALL still be emitted to the tracing pipeline

### Requirement: Streaming errors preserve proxy behavior
The Gateway SHALL log streaming 5xx response metadata without unbounded buffering and SHALL preserve the original upstream status, content type, and response body forwarding behavior.

#### Scenario: Streaming worker returns 5xx headers
- **WHEN** a streaming request receives an upstream 5xx response
- **THEN** the Gateway SHALL log the structured metadata immediately and continue forwarding the error response under the existing proxy contract
