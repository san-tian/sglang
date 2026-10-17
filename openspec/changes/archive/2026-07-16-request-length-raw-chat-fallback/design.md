# Design: opt-in raw prompt context routing

## Configuration

Add `allow_raw_context_tokens: bool` to `Config` and `Cli`, exposed as `--allow-raw-context-tokens` and `ALLOW_RAW_CONTEXT_TOKENS`. The flag defaults to false.

## Eligibility semantics

Chat, Messages, and Responses handlers already produce `RequestTokens` for raw prompt text when no chat encoder exists. They currently count those tokens only when `engine_equivalent` is true. With the opt-in flag, raw tokens may also supply the context budget, subject to the existing request-shape reliability checks. Tool, multimodal, template-control, and stateful shapes remain unknown and continue to fail closed for bounded workers.

The existing `@min_context_tokens`/`@max_context_tokens` filtering remains unchanged. This setting only changes how a request budget is determined.

## Safety and compatibility

The default is false, preserving all existing production behavior. Operator-facing CLI help must call out that raw counts are approximate and intended only for deployments whose worker prompt format is known to be compatible with the raw text estimate.
