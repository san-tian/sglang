# Design: reasoning raw-length fallback

## Routing estimate

The Chat handler already obtains raw message tokens when context-bounded workers are present. With `ALLOW_RAW_CONTEXT_TOKENS=1`, `reasoning` and `reasoning_effort` no longer invalidate those tokens for context eligibility. The required budget remains `raw_message_tokens + max(max_tokens, max_completion_tokens)` or the existing default output reserve.

This is intentionally an approximate hardware-pool decision. Reasoning controls can add engine-side template tokens that the Router does not reproduce. The approximation therefore has a small boundary risk near 65536 tokens, accepted for this isolated debug deployment in preference to rejecting the request.

## Request fidelity

The change does not relax `input_ids_safe_to_forward` or `context_prompt_tokens_reliable`. Approximate raw tokens affect only worker filtering and cache-aware selection context. The selected worker receives the original reasoning request and performs its normal engine-side prompt construction.

## Compatibility

The behavior remains guarded by the existing opt-in flag, which defaults to false. Tools, multimodal input, explicit templates, task controls, and assistant continuation remain unknown because their substantive prompt may not be represented by raw message text.
