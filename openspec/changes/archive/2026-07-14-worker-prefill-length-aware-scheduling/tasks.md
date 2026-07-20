## 1. Worker Scheduling

- [x] 1.1 Add the opt-in policy, finite aging and max-wait arguments, and early role-aware validation.
- [x] 1.2 Implement live uncached-work ordering with business priority, aging, and same-priority overdue FCFS.

## 2. Load Contract

- [x] 2.1 Add bounded Prefill queue summaries to the typed load snapshot and `/v1/loads`.
- [x] 2.2 Extend `/get_load` with optional role-labelled token fields and include native Prefill bootstrap work in conservative totals.

## 3. Verification

- [x] 3.1 Add registered CPU tests for cache refresh, ordering, starvation, priority, PD accounting, serialization, and CLI validation.
- [x] 3.2 Run formatting, pre-commit checks, focused new tests, and existing scheduling/chunk/PD priority regressions.
- [x] 3.3 Run strict OpenSpec validation and archive the completed change.
