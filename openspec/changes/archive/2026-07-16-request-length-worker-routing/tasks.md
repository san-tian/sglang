## 1. Worker Registration Metadata

- [x] 1.1 Parse and validate `@min_context_tokens=N`, including positive-value and `min <= max` checks.
- [x] 1.2 Carry `min_context_tokens` through `WorkerSpec`, static discovery, reconciliation, and immutable worker state while preserving serde compatibility.

## 2. Routing Semantics

- [x] 2.1 Extend context eligibility filtering to enforce inclusive lower and upper worker bounds for known request budgets and fail closed for unknown budgets.
- [x] 2.2 Extend bounded-cardinality context-filter metrics, route-decision metadata, and operator-facing CLI documentation for lower-bound outcomes.

## 3. Verification

- [x] 3.1 Add parser and registry unit tests for valid, invalid, contradictory, boundary, and unknown-length cases.
- [x] 3.2 Add HTTP routing tests proving a 65535/65536 AMD/NVIDIA split and 503 behavior when no range is eligible.
- [x] 3.3 Run formatting, strict OpenSpec validation, targeted tests, the full router test target, and a release build.
