# Design: agent Chat raw-length fallback

## Routing estimate

The Chat handler already tokenizes extractable message text for cache routing. With `ALLOW_RAW_CONTEXT_TOKENS=1`, the presence of tools, text-part content, template controls, task controls, reasoning controls, or continuation fields no longer invalidates those available tokens for context eligibility. The required budget remains `raw_message_tokens + max(max_tokens, max_completion_tokens)` or the existing default output reserve.

This estimate intentionally omits engine-side template and tool-schema overhead. It is accepted for this isolated debug deployment in preference to rejecting the request, with a known misclassification risk near the 65536-token boundary.

## Request fidelity

The change does not relax `input_ids_safe_to_forward` or the strict engine-equivalent tokenization predicate. Approximate raw tokens affect only worker range filtering and routing. The selected worker receives the original request and performs normal engine-side prompt construction.

## Compatibility

The opt-in defaults to false. Messages and Responses retain their strict raw-shape checks. A Chat request from which the existing tokenizer cannot extract any text tokens still has unknown length and retains the bounded-pool fail-closed behavior.
