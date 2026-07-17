## 1. Input-Only Eligibility

- [x] 1.1 Replace prompt-plus-output budget computation with reliable input token counts in all generation route handlers
- [x] 1.2 Update range-filter terminology and comments to describe input-only routing

## 2. Verification

- [x] 2.1 Add unit coverage for output-budget invariance and the 65535/65536 input boundary
- [x] 2.2 Run formatting, focused tests, the full Router test suites, and strict OpenSpec validation

## 3. Debug Gateway Rollout

- [x] 3.1 Build a content-addressed release binary and deploy it to only `sgl-router-length-split.service:18084`
- [x] 3.2 Verify readiness, worker health, and live short-input/large-output routing to AMD with route-decision evidence
- [x] 3.3 Update the experiment and work records with rollout evidence and rollback details
