## 1. Specification

- [x] 1.1 Define opt-in approximate routing for agent Chat request shapes and its boundary risk.

## 2. Implementation

- [x] 2.1 Use available Chat request tokens for range eligibility whenever raw context routing is enabled.
- [x] 2.2 Preserve the original worker request and keep approximate `input_ids` injection disabled.

## 3. Verification

- [x] 3.1 Reproduce tools/content-array/template/task/continuation 503 behavior with an HTTP regression test and make it pass.
- [x] 3.2 Run focused context-window tests and full Router tests.
- [x] 3.3 Run formatting, strict OpenSpec validation, release build, and verify the final binary digest.
