## Why

The gateway already supports excluding workers whose declared maximum context is too small, but it cannot reserve a worker pool for long requests. A heterogeneous debug gateway therefore still sends short requests to long-context NVIDIA workers, making it impossible to measure a deliberate AMD-short/NVIDIA-long split.

## What Changes

- Add an optional lower bound for a worker's prompt-plus-requested-output token budget.
- Parse the lower bound from the existing `WORKER_URLS` registration suffixes so deployments can change routing pools through environment configuration.
- Apply lower and upper bounds together before every generation route's policy scoring, with conservative handling for requests whose token budget is unknown.
- Preserve existing behavior for workers without a bound and existing `max_context_tokens` configurations.
- Expose focused parser, registry, and HTTP routing tests for a 64K split.

## Capabilities

### New Capabilities

- `request-length-worker-routing`: Route generation requests only to workers whose registered token-range bounds contain the request's required context budget.

### Modified Capabilities

- None.

## Impact

- Affected Rust router code: static worker URL discovery, `WorkerSpec`/`Worker`, context eligibility filtering, route metrics/logging, and proxy route tests.
- New registration suffix: `@min_context_tokens=N`; it composes with `@max_context_tokens=N` and is validated as a positive integer.
- Example debug configuration: AMD entries use `@max_context_tokens=65535`; NVIDIA entries use `@min_context_tokens=65536`.
- No production gateway, worker pool, APIM/AFD, or Azure service is changed by this code branch. Deployment to the existing debug VM remains a separate approved rollout.
