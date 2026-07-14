// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared reasoning-effort compatibility for Gateway protocol routes.
//!
//! GLM-5.2 exposes only three useful execution states at the worker boundary:
//! thinking disabled, high effort, and max effort. Public clients use a wider
//! vocabulary, and each protocol spells the downstream state differently.
//! This module owns that compatibility policy so Chat, Responses, and
//! Anthropic Messages cannot drift independently.

use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use bytes::Bytes;
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_EFFORT_VALUE_BYTES: usize = 64;
static UNKNOWN_EFFORT_LOG_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningEndpoint {
    Chat,
    Responses,
    Messages,
    MessagesCountTokens,
}

impl ReasoningEndpoint {
    fn route(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Responses => "/v1/responses",
            Self::Messages => "/v1/messages",
            Self::MessagesCountTokens => "/v1/messages/count_tokens",
        }
    }

    fn might_contain_effort(self, body: &Bytes) -> bool {
        let needle: &[u8] = match self {
            Self::Chat | Self::Responses => b"reasoning",
            Self::Messages | Self::MessagesCountTokens => b"effort",
        };
        body.windows(needle.len()).any(|window| window == needle)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Ultra,
    Unknown,
}

impl RequestedEffort {
    fn parse(value: &Value) -> Result<Option<Self>, ApiError> {
        let Value::String(raw) = value else {
            if value.is_null() {
                return Ok(None);
            }
            return Err(invalid_effort());
        };
        let normalized = raw.trim();
        if normalized.is_empty()
            || normalized.len() > MAX_EFFORT_VALUE_BYTES
            || !normalized
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
        {
            return Err(invalid_effort());
        }

        Ok(Some(match normalized.to_ascii_lowercase().as_str() {
            "none" => Self::None,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "max" => Self::Max,
            "ultra" => Self::Ultra,
            _ => Self::Unknown,
        }))
    }

    fn requested_class(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
            Self::Ultra => "ultra",
            Self::Unknown => "unknown",
        }
    }

    fn effective(self) -> EffectiveEffort {
        match self {
            Self::None | Self::Minimal | Self::Low | Self::Unknown => EffectiveEffort::Off,
            Self::Medium | Self::High => EffectiveEffort::High,
            Self::XHigh | Self::Max | Self::Ultra => EffectiveEffort::Max,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectiveEffort {
    Off,
    High,
    Max,
}

impl EffectiveEffort {
    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    fn chat_value(self) -> &'static str {
        match self {
            Self::Off => "none",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    fn responses_value(self) -> &'static str {
        match self {
            Self::Off => "none",
            Self::High => "high",
            // The deployed Responses worker schema accepts xhigh and maps it
            // internally to max, but rejects a literal max before inference.
            Self::Max => "xhigh",
        }
    }

    fn messages_value(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::High => Some("high"),
            Self::Max => Some("max"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NormalizationEvent {
    requested: RequestedEffort,
    effective: EffectiveEffort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingIntent {
    Unspecified,
    Disabled,
    Enabled,
}

/// Normalize one local GLM-5.2 request and record bounded compatibility
/// telemetry. Callers must invoke this only after the external-model route has
/// had a chance to forward the original provider dialect unchanged.
pub(crate) fn normalize_reasoning_request(
    ctx: &AppContext,
    endpoint: ReasoningEndpoint,
    body: Bytes,
) -> Result<Bytes, ApiError> {
    if !is_glm_52_model(&ctx.config.model.id) || !endpoint.might_contain_effort(&body) {
        return Ok(body);
    }

    let (body, event) = normalize_body(endpoint, body)?;
    if let Some(event) = event {
        ctx.metrics.record_reasoning_effort_normalized(
            endpoint.route(),
            event.requested.requested_class(),
            event.effective.label(),
        );
        if event.requested == RequestedEffort::Unknown {
            log_unknown_effort(endpoint, &ctx.config.model.id);
        }
    }
    Ok(body)
}

fn normalize_body(
    endpoint: ReasoningEndpoint,
    body: Bytes,
) -> Result<(Bytes, Option<NormalizationEvent>), ApiError> {
    let mut value: Value = serde_json::from_slice(&body)
        .map_err(|_| ApiError::BadRequest("invalid request: body must be a JSON object".into()))?;
    let object = value.as_object_mut().ok_or_else(|| {
        ApiError::BadRequest("invalid request: body must be a JSON object".into())
    })?;

    let event = match endpoint {
        ReasoningEndpoint::Chat => normalize_chat(object)?,
        ReasoningEndpoint::Responses => normalize_responses(object)?,
        ReasoningEndpoint::Messages | ReasoningEndpoint::MessagesCountTokens => {
            normalize_messages(object)?
        }
    };
    let Some(event) = event else {
        return Ok((body, None));
    };

    let encoded = serde_json::to_vec(&value).map_err(|e| {
        ApiError::Internal(anyhow::anyhow!(
            "serialize reasoning compatibility body: {e}"
        ))
    })?;
    Ok((Bytes::from(encoded), Some(event)))
}

fn normalize_chat(object: &mut Map<String, Value>) -> Result<Option<NormalizationEvent>, ApiError> {
    // The native Chat spelling wins when both forms are present. A null native
    // value is treated as omitted, allowing the Responses-style nested form
    // used by some OpenAI-compatible clients to supply the intent.
    let top_level = object
        .get("reasoning_effort")
        .map(RequestedEffort::parse)
        .transpose()?
        .flatten();
    let nested = object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("effort"))
        .map(RequestedEffort::parse)
        .transpose()?
        .flatten();
    let requested = top_level.or(nested);
    let Some(requested) = requested else {
        return Ok(None);
    };
    let effective = apply_thinking_intent(requested.effective(), openai_thinking_intent(object));
    object.insert(
        "reasoning_effort".to_string(),
        Value::String(effective.chat_value().to_string()),
    );

    let remove_empty_reasoning =
        if let Some(reasoning) = object.get_mut("reasoning").and_then(Value::as_object_mut) {
            reasoning.remove("effort");
            reasoning.is_empty()
        } else {
            false
        };
    if remove_empty_reasoning {
        object.remove("reasoning");
    }

    Ok(Some(NormalizationEvent {
        requested,
        effective,
    }))
}

fn normalize_responses(
    object: &mut Map<String, Value>,
) -> Result<Option<NormalizationEvent>, ApiError> {
    let intent = openai_thinking_intent(object);
    let Some(reasoning) = object.get_mut("reasoning").and_then(Value::as_object_mut) else {
        return Ok(None);
    };
    let Some(requested) = reasoning
        .get("effort")
        .map(RequestedEffort::parse)
        .transpose()?
        .flatten()
    else {
        return Ok(None);
    };
    let effective = apply_thinking_intent(requested.effective(), intent);
    reasoning.insert(
        "effort".to_string(),
        Value::String(effective.responses_value().to_string()),
    );
    Ok(Some(NormalizationEvent {
        requested,
        effective,
    }))
}

fn normalize_messages(
    object: &mut Map<String, Value>,
) -> Result<Option<NormalizationEvent>, ApiError> {
    let Some(requested) = object
        .get("output_config")
        .and_then(Value::as_object)
        .and_then(|output| output.get("effort"))
        .map(RequestedEffort::parse)
        .transpose()?
        .flatten()
    else {
        return Ok(None);
    };

    let intent = thinking_intent(object.get("thinking"))?;
    let effective = apply_thinking_intent(requested.effective(), intent);

    match effective.messages_value() {
        Some(value) => {
            let output = object
                .get_mut("output_config")
                .and_then(Value::as_object_mut)
                .expect("effort was read from an output_config object");
            output.insert("effort".to_string(), Value::String(value.to_string()));
        }
        None => {
            if intent == ThinkingIntent::Unspecified {
                object.insert(
                    "thinking".to_string(),
                    serde_json::json!({"type": "disabled"}),
                );
            }
            remove_output_effort(object);
        }
    }

    Ok(Some(NormalizationEvent {
        requested,
        effective,
    }))
}

fn thinking_intent(value: Option<&Value>) -> Result<ThinkingIntent, ApiError> {
    let Some(value) = value else {
        return Ok(ThinkingIntent::Unspecified);
    };
    if value.is_null() {
        return Ok(ThinkingIntent::Unspecified);
    }
    let Some(object) = value.as_object() else {
        return Err(invalid_thinking());
    };
    match object.get("type").and_then(Value::as_str) {
        Some("disabled") => Ok(ThinkingIntent::Disabled),
        Some("enabled" | "adaptive") => Ok(ThinkingIntent::Enabled),
        _ => Err(invalid_thinking()),
    }
}

fn openai_thinking_intent(object: &Map<String, Value>) -> ThinkingIntent {
    // GLM templates read `enable_thinking`. An explicitly supplied template
    // kwarg is therefore authoritative over compatibility aliases that the
    // worker later copies into chat_template_kwargs with `setdefault`.
    if let Some(intent) = object
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(boolean_intent)
    {
        return intent;
    }

    if let Some(reasoning) = object.get("reasoning").and_then(Value::as_object) {
        if let Some(intent) = reasoning
            .get("type")
            .and_then(Value::as_str)
            .and_then(type_intent)
        {
            return intent;
        }
        if let Some(intent) = reasoning
            .get("enabled")
            .or_else(|| reasoning.get("enable"))
            .and_then(boolish_intent)
        {
            return intent;
        }
    }

    if let Some(intent) = object
        .get("thinking")
        .and_then(Value::as_object)
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        .and_then(type_intent)
    {
        return intent;
    }

    object
        .get("enable_thinking")
        .and_then(boolish_intent)
        .unwrap_or(ThinkingIntent::Unspecified)
}

fn apply_thinking_intent(effort: EffectiveEffort, intent: ThinkingIntent) -> EffectiveEffort {
    match intent {
        ThinkingIntent::Disabled => EffectiveEffort::Off,
        ThinkingIntent::Enabled if effort == EffectiveEffort::Off => {
            // Thinking is an explicit, orthogonal control. Do not turn it off
            // merely because the effort vocabulary quantizes to Off; use
            // GLM's lowest enabled tier instead.
            EffectiveEffort::High
        }
        ThinkingIntent::Enabled | ThinkingIntent::Unspecified => effort,
    }
}

fn type_intent(value: &str) -> Option<ThinkingIntent> {
    match value {
        "enabled" | "adaptive" => Some(ThinkingIntent::Enabled),
        "disabled" => Some(ThinkingIntent::Disabled),
        _ => None,
    }
}

fn boolean_intent(value: &Value) -> Option<ThinkingIntent> {
    value.as_bool().map(|enabled| {
        if enabled {
            ThinkingIntent::Enabled
        } else {
            ThinkingIntent::Disabled
        }
    })
}

fn boolish_intent(value: &Value) -> Option<ThinkingIntent> {
    if let Some(intent) = boolean_intent(value) {
        return Some(intent);
    }
    value.as_str().map(|value| {
        if matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ) {
            ThinkingIntent::Enabled
        } else {
            ThinkingIntent::Disabled
        }
    })
}

fn remove_output_effort(object: &mut Map<String, Value>) {
    let remove_output_config = if let Some(output) = object
        .get_mut("output_config")
        .and_then(Value::as_object_mut)
    {
        output.remove("effort");
        output.is_empty()
    } else {
        false
    };
    if remove_output_config {
        object.remove("output_config");
    }
}

fn is_glm_52_model(model_id: &str) -> bool {
    let canonical: String = model_id
        .bytes()
        .filter(|byte| byte.is_ascii_alphanumeric())
        .map(|byte| byte.to_ascii_lowercase() as char)
        .collect();
    canonical.contains("glm52")
}

fn invalid_effort() -> ApiError {
    ApiError::BadRequest(
        "invalid reasoning effort: expected a non-empty enum string up to 64 ASCII characters"
            .into(),
    )
}

fn invalid_thinking() -> ApiError {
    ApiError::BadRequest(
        "invalid thinking configuration: expected type disabled, enabled, or adaptive".into(),
    )
}

fn log_unknown_effort(endpoint: ReasoningEndpoint, model_id: &str) {
    let occurrence = UNKNOWN_EFFORT_LOG_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if occurrence.is_power_of_two() {
        tracing::warn!(
            route = endpoint.route(),
            model = model_id,
            unknown_occurrences = occurrence,
            "unknown reasoning effort normalized to off",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn normalize(endpoint: ReasoningEndpoint, value: Value) -> (Value, NormalizationEvent) {
        let body = Bytes::from(serde_json::to_vec(&value).unwrap());
        let (body, event) = normalize_body(endpoint, body).unwrap();
        (serde_json::from_slice(&body).unwrap(), event.unwrap())
    }

    #[test]
    fn enables_only_glm_52_family_ids() {
        assert!(is_glm_52_model("zai-org/GLM-5.2-FP8"));
        assert!(is_glm_52_model("glm52"));
        assert!(!is_glm_52_model("zai-org/GLM-5.1-FP8"));
        assert!(!is_glm_52_model("Macaron-V1-Tall"));
    }

    #[test]
    fn chat_quantizes_every_known_effort() {
        let cases = [
            ("none", "none", EffectiveEffort::Off),
            ("minimal", "none", EffectiveEffort::Off),
            ("low", "none", EffectiveEffort::Off),
            ("medium", "high", EffectiveEffort::High),
            ("high", "high", EffectiveEffort::High),
            ("xhigh", "max", EffectiveEffort::Max),
            ("max", "max", EffectiveEffort::Max),
            ("ultra", "max", EffectiveEffort::Max),
            ("future-super", "none", EffectiveEffort::Off),
        ];
        for (requested, downstream, effective) in cases {
            let (value, event) = normalize(
                ReasoningEndpoint::Chat,
                json!({"model":"glm","reasoning_effort":requested}),
            );
            assert_eq!(value["reasoning_effort"], downstream, "{requested}");
            assert_eq!(event.effective, effective, "{requested}");
        }
    }

    #[test]
    fn chat_accepts_nested_effort_and_native_field_wins_conflicts() {
        let (nested, _) = normalize(
            ReasoningEndpoint::Chat,
            json!({"reasoning":{"effort":"xhigh","summary":"auto"}}),
        );
        assert_eq!(nested["reasoning_effort"], "max");
        assert_eq!(nested["reasoning"], json!({"summary":"auto"}));

        let (conflict, event) = normalize(
            ReasoningEndpoint::Chat,
            json!({"reasoning_effort":"low","reasoning":{"effort":"max"}}),
        );
        assert_eq!(conflict["reasoning_effort"], "none");
        assert!(conflict.get("reasoning").is_none());
        assert_eq!(event.requested, RequestedEffort::Low);
    }

    #[test]
    fn chat_explicit_thinking_has_precedence_over_effort() {
        let (enabled, event) = normalize(
            ReasoningEndpoint::Chat,
            json!({
                "reasoning_effort":"low",
                "reasoning":{"enabled":true}
            }),
        );
        assert_eq!(enabled["reasoning_effort"], "high");
        assert_eq!(enabled["reasoning"]["enabled"], true);
        assert_eq!(event.effective, EffectiveEffort::High);

        let (disabled, event) = normalize(
            ReasoningEndpoint::Chat,
            json!({
                "reasoning_effort":"max",
                "chat_template_kwargs":{"enable_thinking":false}
            }),
        );
        assert_eq!(disabled["reasoning_effort"], "none");
        assert_eq!(disabled["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(event.effective, EffectiveEffort::Off);
    }

    #[test]
    fn responses_encodes_max_as_xhigh_for_current_worker_schema() {
        let cases = [
            ("none", "none", EffectiveEffort::Off),
            ("minimal", "none", EffectiveEffort::Off),
            ("low", "none", EffectiveEffort::Off),
            ("medium", "high", EffectiveEffort::High),
            ("high", "high", EffectiveEffort::High),
            ("xhigh", "xhigh", EffectiveEffort::Max),
            ("max", "xhigh", EffectiveEffort::Max),
            ("ultra", "xhigh", EffectiveEffort::Max),
            ("future-super", "none", EffectiveEffort::Off),
        ];
        for (requested, downstream, effective) in cases {
            let (value, event) = normalize(
                ReasoningEndpoint::Responses,
                json!({"reasoning":{"effort":requested,"summary":"auto"}}),
            );
            assert_eq!(value["reasoning"]["effort"], downstream, "{requested}");
            assert_eq!(value["reasoning"]["summary"], "auto");
            assert_eq!(event.effective, effective, "{requested}");
        }
    }

    #[test]
    fn responses_explicit_thinking_has_precedence_over_effort() {
        let (enabled, event) = normalize(
            ReasoningEndpoint::Responses,
            json!({"reasoning":{"effort":"low","type":"enabled"}}),
        );
        assert_eq!(enabled["reasoning"]["effort"], "high");
        assert_eq!(enabled["reasoning"]["type"], "enabled");
        assert_eq!(event.effective, EffectiveEffort::High);

        let (disabled, event) = normalize(
            ReasoningEndpoint::Responses,
            json!({
                "reasoning":{"effort":"max"},
                "thinking":{"type":"disabled"}
            }),
        );
        assert_eq!(disabled["reasoning"]["effort"], "none");
        assert_eq!(disabled["thinking"]["type"], "disabled");
        assert_eq!(event.effective, EffectiveEffort::Off);
    }

    #[test]
    fn messages_turns_low_and_unknown_off_without_dropping_output_config() {
        for effort in ["minimal", "low", "future-level"] {
            let (value, event) = normalize(
                ReasoningEndpoint::Messages,
                json!({"output_config":{"effort":effort,"format":{"type":"json_schema"}}}),
            );
            assert_eq!(value["thinking"]["type"], "disabled");
            assert!(value["output_config"].get("effort").is_none());
            assert_eq!(value["output_config"]["format"]["type"], "json_schema");
            assert_eq!(event.effective, EffectiveEffort::Off);
        }
    }

    #[test]
    fn messages_explicit_thinking_has_precedence_and_preserves_budget() {
        let (enabled, event) = normalize(
            ReasoningEndpoint::Messages,
            json!({
                "thinking":{"type":"enabled","budget_tokens":4096},
                "output_config":{"effort":"low"}
            }),
        );
        assert_eq!(enabled["thinking"]["type"], "enabled");
        assert_eq!(enabled["thinking"]["budget_tokens"], 4096);
        assert_eq!(enabled["output_config"]["effort"], "high");
        assert_eq!(event.effective, EffectiveEffort::High);

        let (adaptive, event) = normalize(
            ReasoningEndpoint::Messages,
            json!({
                "thinking":{"type":"adaptive"},
                "output_config":{"effort":"future-level"}
            }),
        );
        assert_eq!(adaptive["thinking"]["type"], "adaptive");
        assert_eq!(adaptive["output_config"]["effort"], "high");
        assert_eq!(event.effective, EffectiveEffort::High);

        let (disabled, event) = normalize(
            ReasoningEndpoint::Messages,
            json!({
                "thinking":{"type":"disabled"},
                "output_config":{"effort":"max"}
            }),
        );
        assert_eq!(disabled["thinking"]["type"], "disabled");
        assert!(disabled.get("output_config").is_none());
        assert_eq!(event.effective, EffectiveEffort::Off);
    }

    #[test]
    fn messages_count_tokens_uses_the_same_policy() {
        let (value, event) = normalize(
            ReasoningEndpoint::MessagesCountTokens,
            json!({"output_config":{"effort":"xhigh"}}),
        );
        assert_eq!(value["output_config"]["effort"], "max");
        assert_eq!(event.effective, EffectiveEffort::Max);
    }

    #[test]
    fn messages_quantizes_every_known_effort_without_explicit_thinking() {
        let cases = [
            ("none", None, EffectiveEffort::Off),
            ("minimal", None, EffectiveEffort::Off),
            ("low", None, EffectiveEffort::Off),
            ("medium", Some("high"), EffectiveEffort::High),
            ("high", Some("high"), EffectiveEffort::High),
            ("xhigh", Some("max"), EffectiveEffort::Max),
            ("max", Some("max"), EffectiveEffort::Max),
            ("ultra", Some("max"), EffectiveEffort::Max),
            ("future-super", None, EffectiveEffort::Off),
        ];
        for (requested, downstream, effective) in cases {
            let (value, event) = normalize(
                ReasoningEndpoint::Messages,
                json!({"output_config":{"effort":requested}}),
            );
            match downstream {
                Some(downstream) => {
                    assert_eq!(value["output_config"]["effort"], downstream, "{requested}");
                    assert!(value.get("thinking").is_none(), "{requested}");
                }
                None => {
                    assert!(value.get("output_config").is_none(), "{requested}");
                    assert_eq!(value["thinking"]["type"], "disabled", "{requested}");
                }
            }
            assert_eq!(event.effective, effective, "{requested}");
        }
    }

    #[test]
    fn absent_and_null_effort_leave_original_bytes_unchanged() {
        for value in [
            json!({"model":"glm"}),
            json!({"reasoning_effort":null}),
            json!({"reasoning":{"effort":null}}),
        ] {
            let body = Bytes::from(serde_json::to_vec(&value).unwrap());
            let (output, event) = normalize_body(ReasoningEndpoint::Chat, body.clone()).unwrap();
            assert_eq!(output, body);
            assert!(event.is_none());
        }
    }

    #[test]
    fn malformed_effort_and_thinking_are_rejected_without_echoing_values() {
        for effort in [json!(""), json!("bad value"), json!(7), json!({})] {
            let body =
                Bytes::from(serde_json::to_vec(&json!({"reasoning_effort":effort})).unwrap());
            let error = normalize_body(ReasoningEndpoint::Chat, body).unwrap_err();
            assert!(error.to_string().contains("invalid reasoning effort"));
        }

        let body = Bytes::from(
            serde_json::to_vec(&json!({
                "reasoning_effort":"high",
                "reasoning":{"effort":7}
            }))
            .unwrap(),
        );
        let error = normalize_body(ReasoningEndpoint::Chat, body).unwrap_err();
        assert!(error.to_string().contains("invalid reasoning effort"));

        let body = Bytes::from(
            serde_json::to_vec(&json!({
                "thinking":{"type":"future"},
                "output_config":{"effort":"low"}
            }))
            .unwrap(),
        );
        let error = normalize_body(ReasoningEndpoint::Messages, body).unwrap_err();
        assert!(error.to_string().contains("invalid thinking configuration"));
    }
}
