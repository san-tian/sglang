## 1. Specification

- [x] 1.1 Define routing-only raw-length behavior for reasoning requests and document the boundary approximation.

## 2. Implementation

- [x] 2.1 Allow `reasoning` and `reasoning_effort` through the opt-in raw context gate.
- [x] 2.2 Preserve the original worker request and keep approximate `input_ids` injection disabled.

## 3. Verification

- [x] 3.1 Add a failing HTTP regression test and make it pass.
- [x] 3.2 Run focused context-window unit and proxy tests.
- [x] 3.3 Run formatting, strict OpenSpec validation, full Router tests, and release build.
