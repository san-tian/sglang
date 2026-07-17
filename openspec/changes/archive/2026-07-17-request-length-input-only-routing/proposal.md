## Why

The debug Gateway currently adds the requested output budget to the prompt length before choosing AMD or NVIDIA. This sends short-input requests with large `max_tokens` values to NVIDIA even though the experiment is intended to split traffic solely by the amount of input prefill work.

## What Changes

- Change bounded-worker range eligibility to use only the computed input token count.
- Ignore output-limit fields such as `max_tokens`, `max_completion_tokens`, and `max_output_tokens` for hardware range selection.
- Preserve fail-closed behavior when the input length itself cannot be computed reliably.
- Update boundary and route-handler tests so changing only the output budget cannot change the selected hardware pool.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `request-length-worker-routing`: Worker range eligibility changes from prompt-plus-output budget to input tokens only.
- `request-length-raw-chat-fallback`: Opt-in raw Chat routing uses only the existing raw input token count.

## Impact

The change is limited to the request-length debug branch and its `experimental/sgl-router` implementation, tests, OpenSpec contracts, and the isolated public Gateway on port 18084. It does not change `deploy-prod`, production gateways, worker membership, APIM/AFD, or worker processes. Requests near the 64K input boundary may select a different hardware pool than before when they declare a large output budget.
