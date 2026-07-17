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

/// Whether raw text tokenization is conservative enough for an explicitly
/// opted-in input-range decision. This rejects request features whose
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

/// Apply the hard input-length eligibility gate before cache/load policy
/// scoring. Pools without a declared limit preserve the old path exactly.
pub fn enforce_context_eligibility(
    ctx: &AppContext,
    model: &str,
    workers: Vec<Arc<Worker>>,
    routing_input_tokens: Option<usize>,
) -> Result<Vec<Arc<Worker>>, ApiError> {
    if !has_context_limited_worker(&workers) {
        return Ok(workers);
    }

    let input_count = workers.len();
    let filtered = filter_context_eligible(&workers, routing_input_tokens);
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
            routing_input_tokens,
            healthy_priority_eligible_workers = input_count,
            reason = ?reason,
            "input-length filter removed all candidates; rejecting request",
        );
        return Err(ApiError::NoContextEligibleWorkers {
            model: model.to_owned(),
        });
    }

    tracing::debug!(
        model,
        routing_input_tokens,
        candidates_before = input_count,
        candidates_after = filtered.workers.len(),
        reason = ?reason,
        "input-length filter excluded bounded workers",
    );
    Ok(filtered.workers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
}
