## 1. Configuration

- [x] 1.1 Add parsing and validation for an opt-in `@prefill_members=` worker URL suffix.
- [x] 1.2 Propagate parsed Prefill member URLs from worker capabilities into runtime worker state.

## 2. Routing

- [x] 2.1 Normalize cache-state match URLs through candidate Prefill member mappings during cache-aware TTFT scoring.
- [x] 2.2 Preserve existing exact-worker cache matching and no-mapping fallback behavior.

## 3. Verification

- [x] 3.1 Add unit tests for parsing `@prefill_members=` and malformed member URLs.
- [x] 3.2 Add routing tests proving a logical PD proxy receives cache credit from a matched physical Prefill member without sending a Prefill hint.
- [x] 3.3 Run OpenSpec strict validation and targeted Rust router tests.
