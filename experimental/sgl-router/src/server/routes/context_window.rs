// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::registry::{
    filter_context_eligible, has_context_limited_worker, ContextFilterReason,
};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::ContextFilterOutcome;
use crate::workers::Worker;
use serde_json::Value;
use std::sync::Arc;

/// Reserve used when a generation request omits every route-specific maximum
/// output-token field. The worker capability should still carry operational
/// headroom; this reserve prevents an omitted default from being treated as
/// zero output tokens.
pub const DEFAULT_OUTPUT_TOKEN_RESERVE: usize = 8_192;

/// Compute prompt plus requested output tokens. `output_fields` should list
/// the route's accepted maximum-output fields in any order. Multiple fields
/// use the largest value, which is conservative across compatibility aliases.
/// An explicitly malformed/null field makes the total unknown, while omitted
/// fields use [`DEFAULT_OUTPUT_TOKEN_RESERVE`].
pub fn required_context_tokens(
    prompt_tokens: Option<usize>,
    output_fields: &[Option<&Value>],
) -> Option<usize> {
    required_context_tokens_with_default(
        prompt_tokens,
        output_fields,
        Some(DEFAULT_OUTPUT_TOKEN_RESERVE),
    )
}

/// Compute the context budget only when the request declares an explicit
/// maximum output size. This is required for APIs whose engine-side omitted
/// default consumes the remaining context window, such as `/v1/responses`.
pub fn required_context_tokens_with_explicit_output(
    prompt_tokens: Option<usize>,
    output_fields: &[Option<&Value>],
) -> Option<usize> {
    required_context_tokens_with_default(prompt_tokens, output_fields, None)
}

/// Whether raw text tokenization is conservative enough for an explicitly
/// opted-in context-range decision. This rejects request features whose
/// engine-side prompt construction cannot be represented by raw text.
pub fn raw_context_tokens_reliable(value: &Value) -> bool {
    let nonempty = |key: &str| {
        value.get(key).is_some_and(|v| match v {
            Value::Array(items) => !items.is_empty(),
            Value::Null => false,
            _ => true,
        })
    };
    if nonempty("tools")
        || nonempty("functions")
        || value.get("input_ids").is_some_and(|v| !v.is_null())
        || value.get("chat_template").is_some_and(|v| !v.is_null())
        || value
            .get("chat_template_kwargs")
            .is_some_and(|v| !v.is_null())
        || value.get("reasoning").is_some_and(|v| !v.is_null())
        || value.get("reasoning_effort").is_some_and(|v| !v.is_null())
        || value.get("task").is_some_and(|v| !v.is_null())
        || value.get("continue_final_message").and_then(Value::as_bool) == Some(true)
    {
        return false;
    }
    if value
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages
                .iter()
                .any(|message| matches!(message.get("content"), Some(Value::Array(_))))
                || messages
                    .last()
                    .and_then(|message| message.get("role"))
                    .and_then(Value::as_str)
                    == Some("assistant")
        })
    {
        return false;
    }
    true
}

fn required_context_tokens_with_default(
    prompt_tokens: Option<usize>,
    output_fields: &[Option<&Value>],
    default_output: Option<usize>,
) -> Option<usize> {
    let prompt_tokens = prompt_tokens?;
    let mut requested_output = None;
    for value in output_fields.iter().flatten() {
        let value = json_nonnegative_integer(value)?;
        requested_output =
            Some(requested_output.map_or(value, |current: usize| current.max(value)));
    }
    prompt_tokens.checked_add(requested_output.or(default_output)?)
}

/// Apply the hard context-window eligibility gate before cache/load policy
/// scoring. Pools without a declared limit preserve the old path exactly.
pub fn enforce_context_eligibility(
    ctx: &AppContext,
    model: &str,
    workers: Vec<Arc<Worker>>,
    required_context_tokens: Option<usize>,
) -> Result<Vec<Arc<Worker>>, ApiError> {
    if !has_context_limited_worker(&workers) {
        return Ok(workers);
    }

    let input_count = workers.len();
    let filtered = filter_context_eligible(&workers, required_context_tokens);
    if filtered.reason.is_none() {
        return Ok(filtered.workers);
    }

    let reason = filtered.reason.expect("excluded workers carry a reason");
    let outcome = match (filtered.excluded_all, reason) {
        (false, ContextFilterReason::BelowMinimum) => {
            ContextFilterOutcome::WorkerExcludedBelowMinimum
        }
        (false, ContextFilterReason::OverLimit) => ContextFilterOutcome::WorkerExcludedOverLimit,
        (false, ContextFilterReason::OutsideRange) => {
            ContextFilterOutcome::WorkerExcludedOutsideRange
        }
        (false, ContextFilterReason::UnknownLength) => {
            ContextFilterOutcome::WorkerExcludedUnknownLength
        }
        (true, ContextFilterReason::BelowMinimum) => {
            ContextFilterOutcome::EmptySetRejectedBelowMinimum
        }
        (true, ContextFilterReason::OverLimit) => ContextFilterOutcome::EmptySetRejectedOverLimit,
        (true, ContextFilterReason::OutsideRange) => {
            ContextFilterOutcome::EmptySetRejectedOutsideRange
        }
        (true, ContextFilterReason::UnknownLength) => {
            ContextFilterOutcome::EmptySetRejectedUnknownLength
        }
    };
    ctx.metrics.record_context_filtered(outcome);

    if filtered.excluded_all {
        tracing::warn!(
            model,
            required_context_tokens,
            healthy_priority_eligible_workers = input_count,
            reason = ?reason,
            "context-window filter removed all candidates; rejecting request",
        );
        return Err(ApiError::NoContextEligibleWorkers {
            model: model.to_owned(),
        });
    }

    tracing::debug!(
        model,
        required_context_tokens,
        candidates_before = input_count,
        candidates_after = filtered.workers.len(),
        reason = ?reason,
        "context-window filter excluded limited workers",
    );
    Ok(filtered.workers)
}

fn json_nonnegative_integer(value: &Value) -> Option<usize> {
    let raw = value.as_u64()?;
    usize::try_from(raw).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn budget_adds_explicit_output_limit() {
        let max_tokens = json!(1_024);
        assert_eq!(
            required_context_tokens(Some(499_000), &[Some(&max_tokens)]),
            Some(500_024),
        );
    }

    #[test]
    fn budget_uses_largest_compatibility_field() {
        let legacy = json!(1_000);
        let modern = json!(2_000);
        assert_eq!(
            required_context_tokens(Some(10), &[Some(&legacy), Some(&modern)]),
            Some(2_010),
        );
    }

    #[test]
    fn budget_uses_reserve_when_output_limit_is_omitted() {
        assert_eq!(
            required_context_tokens(Some(490_000), &[]),
            Some(490_000 + DEFAULT_OUTPUT_TOKEN_RESERVE),
        );
    }

    #[test]
    fn explicit_output_budget_is_unknown_when_limit_is_omitted() {
        assert_eq!(
            required_context_tokens_with_explicit_output(Some(490_000), &[]),
            None,
        );

        let max_output_tokens = json!(1_024);
        assert_eq!(
            required_context_tokens_with_explicit_output(
                Some(490_000),
                &[Some(&max_output_tokens)],
            ),
            Some(491_024),
        );
    }

    #[test]
    fn raw_context_reliability_stays_strict_for_messages_and_responses() {
        assert!(raw_context_tokens_reliable(&json!({
            "messages": [{"role": "user", "content": "hello"}]
        })));
        assert!(!raw_context_tokens_reliable(&json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function"}]
        })));
        assert!(!raw_context_tokens_reliable(&json!({
            "messages": [{"role": "user", "content": [{"type": "image_url"}]}]
        })));
        assert!(!raw_context_tokens_reliable(&json!({
            "messages": [{"role": "assistant", "content": "partial"}]
        })));
        let reasoning_effort = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "high"
        });
        let nested_reasoning = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning": {"effort": "high"}
        });
        assert!(!raw_context_tokens_reliable(&reasoning_effort));
        assert!(!raw_context_tokens_reliable(&nested_reasoning));
    }

    #[test]
    fn budget_is_unknown_for_unknown_prompt_or_malformed_output() {
        let malformed = json!("4096");
        let null = Value::Null;
        assert_eq!(required_context_tokens(None, &[]), None);
        assert_eq!(required_context_tokens(Some(10), &[Some(&malformed)]), None);
        assert_eq!(required_context_tokens(Some(10), &[Some(&null)]), None);
    }

    #[test]
    fn budget_rejects_overflow() {
        let one = json!(1);
        assert_eq!(
            required_context_tokens(Some(usize::MAX), &[Some(&one)]),
            None
        );
    }
}
