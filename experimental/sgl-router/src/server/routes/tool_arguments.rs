// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::error::ApiError;
use axum::http::HeaderMap;
use bytes::Bytes;
use serde_json::error::Category;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

const MAX_JSON_STRING_DECODE_DEPTH: usize = 4;
const REJECT_LOG_ARGUMENT_PREVIEW_BYTES: usize = 16 * 1024;
const REJECT_LOG_TOOL_CALL_PREVIEW_BYTES: usize = 32 * 1024;
const REJECT_LOG_MESSAGE_PREVIEW_BYTES: usize = 64 * 1024;
const REJECT_LOG_BODY_PREVIEW_BYTES: usize = 128 * 1024;

#[derive(Clone, Copy)]
enum ToolArgumentsRejectReason {
    Missing,
    Null,
    Array,
    Scalar,
    TruncatedJson,
    NonJsonString,
    MultiEncodedTooDeep,
    SerializeFailed,
}

impl ToolArgumentsRejectReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Null => "null",
            Self::Array => "array",
            Self::Scalar => "scalar",
            Self::TruncatedJson => "truncated_json",
            Self::NonJsonString => "non_json_string",
            Self::MultiEncodedTooDeep => "multi_encoded_too_deep",
            Self::SerializeFailed => "serialize_failed",
        }
    }
}

enum ToolArgumentsAction {
    Unchanged,
    Replace {
        value: String,
        repair_kind: &'static str,
    },
    Reject {
        reason: ToolArgumentsRejectReason,
    },
}

/// Normalize only assistant tool-call arguments that can be repaired without
/// guessing their meaning.
///
/// Accepted compatibility shapes are native JSON objects, bounded JSON
/// string-encoding layers, complete `json` code fences, and missing/null/empty
/// arguments for a matching simple tool schema that allows `{}`. Truncated
/// JSON, arrays, scalars, mixed prose, and schema-ambiguous empty values are
/// rejected at the gateway edge. The client error deliberately reports only the
/// JSON path; the server-side warning logs bounded request excerpts for
/// incident diagnosis.
pub(crate) fn normalize_chat_tool_call_arguments(
    headers: &HeaderMap,
    body: Bytes,
) -> Result<Bytes, ApiError> {
    if !body
        .windows(b"tool_calls".len())
        .any(|window| window == b"tool_calls")
    {
        return Ok(body);
    }

    let mut value: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions tool-arguments normalize parse failed");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    let empty_argument_tools = collect_empty_argument_tools(&value);
    let Some(messages) = value.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return Ok(body);
    };

    let mut changed = false;
    for (message_index, message) in messages.iter_mut().enumerate() {
        if message.get("role").and_then(|v| v.as_str()) != Some("assistant") {
            continue;
        }
        let message_for_log = message.clone();
        let Some(tool_calls) = message.get_mut("tool_calls").and_then(|v| v.as_array_mut()) else {
            continue;
        };
        for (tool_call_index, tool_call) in tool_calls.iter_mut().enumerate() {
            let Some(function) = tool_call
                .get_mut("function")
                .and_then(|v| v.as_object_mut())
            else {
                continue;
            };
            let allow_empty = function
                .get("name")
                .and_then(|v| v.as_str())
                .and_then(|name| empty_argument_tools.get(name))
                .copied()
                .unwrap_or(false);
            if !function.contains_key("arguments") {
                if allow_empty {
                    function.insert(
                        "arguments".to_string(),
                        serde_json::Value::String("{}".to_string()),
                    );
                    changed = true;
                    tracing::debug!(
                        message_index,
                        tool_call_index,
                        repair_kind = "missing_optional",
                        "normalized assistant tool-call arguments",
                    );
                    continue;
                }
                log_unsafe_tool_arguments(
                    headers,
                    &body,
                    &message_for_log,
                    tool_call,
                    None,
                    message_index,
                    tool_call_index,
                    ToolArgumentsRejectReason::Missing,
                    allow_empty,
                );
                return Err(unsafe_tool_arguments_error(message_index, tool_call_index));
            };

            let action = {
                let arguments = function
                    .get_mut("arguments")
                    .expect("arguments presence checked above");
                normalize_tool_arguments_value(arguments, allow_empty)
            };

            match action {
                ToolArgumentsAction::Unchanged => {}
                ToolArgumentsAction::Replace { value, repair_kind } => {
                    function.insert("arguments".to_string(), serde_json::Value::String(value));
                    changed = true;
                    tracing::debug!(
                        message_index,
                        tool_call_index,
                        repair_kind,
                        "normalized assistant tool-call arguments",
                    );
                }
                ToolArgumentsAction::Reject { reason } => {
                    let arguments = tool_call.get("function").and_then(|v| v.get("arguments"));
                    log_unsafe_tool_arguments(
                        headers,
                        &body,
                        &message_for_log,
                        tool_call,
                        arguments,
                        message_index,
                        tool_call_index,
                        reason,
                        allow_empty,
                    );
                    return Err(unsafe_tool_arguments_error(message_index, tool_call_index));
                }
            }
        }
    }

    if !changed {
        return Ok(body);
    }
    serde_json::to_vec(&value).map(Bytes::from).map_err(|e| {
        ApiError::Internal(
            anyhow::Error::new(e).context("re-serialize normalized chat tool arguments"),
        )
    })
}

fn normalize_tool_arguments_value(
    arguments: &serde_json::Value,
    allow_empty: bool,
) -> ToolArgumentsAction {
    match arguments {
        serde_json::Value::Object(_) => match serde_json::to_string(arguments) {
            Ok(value) => ToolArgumentsAction::Replace {
                value,
                repair_kind: "native_object",
            },
            Err(_) => ToolArgumentsAction::Reject {
                reason: ToolArgumentsRejectReason::SerializeFailed,
            },
        },
        serde_json::Value::Null => {
            if allow_empty {
                ToolArgumentsAction::Replace {
                    value: "{}".to_string(),
                    repair_kind: "null_optional",
                }
            } else {
                ToolArgumentsAction::Reject {
                    reason: ToolArgumentsRejectReason::Null,
                }
            }
        }
        serde_json::Value::String(raw) => normalize_tool_arguments_string(raw, allow_empty),
        serde_json::Value::Array(_) => ToolArgumentsAction::Reject {
            reason: ToolArgumentsRejectReason::Array,
        },
        _ => ToolArgumentsAction::Reject {
            reason: ToolArgumentsRejectReason::Scalar,
        },
    }
}

fn normalize_tool_arguments_string(raw: &str, allow_empty: bool) -> ToolArgumentsAction {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return if allow_empty {
            ToolArgumentsAction::Replace {
                value: "{}".to_string(),
                repair_kind: "empty_optional",
            }
        } else {
            ToolArgumentsAction::Reject {
                reason: ToolArgumentsRejectReason::NonJsonString,
            }
        };
    }

    normalize_encoded_object_string(trimmed)
}

fn normalize_encoded_object_string(raw: &str) -> ToolArgumentsAction {
    let mut current = raw.trim().to_string();
    let mut saw_fence = false;

    for decode_depth in 0..=MAX_JSON_STRING_DECODE_DEPTH {
        let trimmed = current.trim();
        if let Some(fenced) = strip_complete_json_fence(trimmed) {
            saw_fence = true;
            current = fenced.trim().to_string();
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(value @ serde_json::Value::Object(_)) => {
                if decode_depth == 0 && !saw_fence {
                    return ToolArgumentsAction::Unchanged;
                }
                return match serde_json::to_string(&value) {
                    Ok(value) => ToolArgumentsAction::Replace {
                        value,
                        repair_kind: if saw_fence {
                            "encoded_json_fence"
                        } else {
                            "encoded_json_string"
                        },
                    },
                    Err(_) => ToolArgumentsAction::Reject {
                        reason: ToolArgumentsRejectReason::SerializeFailed,
                    },
                };
            }
            Ok(serde_json::Value::String(nested)) => {
                if decode_depth == MAX_JSON_STRING_DECODE_DEPTH {
                    return ToolArgumentsAction::Reject {
                        reason: ToolArgumentsRejectReason::MultiEncodedTooDeep,
                    };
                }
                current = nested.trim().to_string();
            }
            Ok(serde_json::Value::Array(_)) => {
                return ToolArgumentsAction::Reject {
                    reason: ToolArgumentsRejectReason::Array,
                };
            }
            Ok(serde_json::Value::Null) => {
                return ToolArgumentsAction::Reject {
                    reason: ToolArgumentsRejectReason::Null,
                };
            }
            Ok(_) => {
                return ToolArgumentsAction::Reject {
                    reason: ToolArgumentsRejectReason::Scalar,
                };
            }
            Err(error) => {
                return ToolArgumentsAction::Reject {
                    reason: if error.classify() == Category::Eof {
                        ToolArgumentsRejectReason::TruncatedJson
                    } else {
                        ToolArgumentsRejectReason::NonJsonString
                    },
                };
            }
        }
    }

    ToolArgumentsAction::Reject {
        reason: ToolArgumentsRejectReason::MultiEncodedTooDeep,
    }
}

fn strip_complete_json_fence(raw: &str) -> Option<&str> {
    let inner = raw.strip_prefix("```")?.strip_suffix("```")?.trim();
    if inner.is_empty() {
        return None;
    }
    if let Some((language, payload)) = inner.split_once('\n') {
        let language = language.trim_end_matches('\r').trim();
        if !language.is_empty() && !language.eq_ignore_ascii_case("json") {
            return None;
        }
        return Some(payload.trim());
    }
    if let Some(rest) = inner
        .strip_prefix("json")
        .or_else(|| inner.strip_prefix("JSON"))
        .filter(|rest| rest.starts_with(char::is_whitespace))
    {
        return Some(rest.trim());
    }
    if inner.starts_with('{') || inner.starts_with('[') {
        return Some(inner);
    }
    None
}

fn collect_empty_argument_tools(value: &serde_json::Value) -> HashMap<String, bool> {
    let mut tools = HashMap::new();
    if let Some(definitions) = value.get("tools").and_then(|v| v.as_array()) {
        for definition in definitions {
            if definition.get("type").and_then(|v| v.as_str()) != Some("function") {
                continue;
            }
            if let Some(function) = definition.get("function").and_then(|v| v.as_object()) {
                record_empty_argument_tool(&mut tools, function);
            }
        }
    }
    if let Some(definitions) = value.get("functions").and_then(|v| v.as_array()) {
        for definition in definitions {
            if let Some(function) = definition.as_object() {
                record_empty_argument_tool(&mut tools, function);
            }
        }
    }
    tools
}

fn record_empty_argument_tool(
    tools: &mut HashMap<String, bool>,
    function: &serde_json::Map<String, serde_json::Value>,
) {
    let Some(name) = function.get("name").and_then(|v| v.as_str()) else {
        return;
    };
    let allows_empty = match function.get("parameters") {
        None => true,
        Some(schema) => simple_schema_allows_empty_object(schema),
    };
    tools
        .entry(name.to_string())
        .and_modify(|existing| *existing &= allows_empty)
        .or_insert(allows_empty);
}

fn simple_schema_allows_empty_object(schema: &serde_json::Value) -> bool {
    let Some(schema) = schema.as_object() else {
        return false;
    };
    if let Some(schema_type) = schema.get("type") {
        if schema_type.as_str() != Some("object") {
            return false;
        }
    }
    if let Some(required) = schema.get("required") {
        match required {
            serde_json::Value::Null => {}
            serde_json::Value::Array(items) if items.is_empty() => {}
            _ => return false,
        }
    }
    if let Some(min_properties) = schema.get("minProperties") {
        if min_properties.as_u64() != Some(0) {
            return false;
        }
    }
    ![
        "$ref",
        "$dynamicRef",
        "$recursiveRef",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "const",
        "enum",
        "dependentRequired",
        "dependentSchemas",
        "dependencies",
    ]
    .iter()
    .any(|keyword| schema.contains_key(*keyword))
}

fn unsafe_tool_arguments_error(message_index: usize, tool_call_index: usize) -> ApiError {
    ApiError::BadRequest(format!(
        "messages[{message_index}].tool_calls[{tool_call_index}].function.arguments must be a JSON object string; automatic repair was not safe"
    ))
}

fn log_unsafe_tool_arguments(
    headers: &HeaderMap,
    body: &Bytes,
    message: &serde_json::Value,
    tool_call: &serde_json::Value,
    arguments: Option<&serde_json::Value>,
    message_index: usize,
    tool_call_index: usize,
    reason: ToolArgumentsRejectReason,
    allow_empty: bool,
) {
    let arguments_preview = arguments
        .map(|value| value_preview(value, REJECT_LOG_ARGUMENT_PREVIEW_BYTES))
        .unwrap_or_else(|| "<missing>".to_string());
    let arguments_len_bytes = arguments.map(value_len_bytes).unwrap_or(0);
    let arguments_sha256 = arguments.map(value_sha256_hex).unwrap_or_default();
    let arguments_type = arguments.map(value_type).unwrap_or("missing");
    let tool_name = tool_call
        .get("function")
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    tracing::warn!(
        trace_id = header_value(headers, "x-trace-id"),
        request_id = header_value(headers, "x-request-id"),
        traceparent = header_value(headers, "traceparent"),
        b3 = header_value(headers, "b3"),
        b3_trace_id = header_value(headers, "x-b3-traceid"),
        b3_span_id = header_value(headers, "x-b3-spanid"),
        sentry_trace = header_value(headers, "sentry-trace"),
        ms_client_request_id = header_value(headers, "x-ms-client-request-id"),
        ms_request_id = header_value(headers, "x-ms-request-id"),
        correlation_id = header_value(headers, "x-correlation-id"),
        client_trace_id = header_value(headers, "x-client-trace-id"),
        user_agent = header_value(headers, "user-agent"),
        message_index,
        tool_call_index,
        reject_reason = reason.as_str(),
        allow_empty,
        tool_name,
        arguments_type,
        arguments_len_bytes,
        arguments_sha256,
        arguments_preview,
        tool_call_len_bytes = value_len_bytes(tool_call),
        tool_call_sha256 = value_sha256_hex(tool_call),
        tool_call_preview = value_preview(tool_call, REJECT_LOG_TOOL_CALL_PREVIEW_BYTES),
        message_len_bytes = value_len_bytes(message),
        message_sha256 = value_sha256_hex(message),
        message_preview = value_preview(message, REJECT_LOG_MESSAGE_PREVIEW_BYTES),
        request_body_len_bytes = body.len(),
        request_body_sha256 = bytes_sha256_hex(body),
        request_body_preview = bytes_preview(body, REJECT_LOG_BODY_PREVIEW_BYTES),
        "assistant tool-call arguments rejected; logging invalid request excerpts"
    );
}

fn header_value(headers: &HeaderMap, name: &'static str) -> String {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| truncate_utf8(value.trim(), 2048))
        .unwrap_or_default()
}

fn value_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn value_len_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(raw) => raw.len(),
        _ => serde_json::to_vec(value).map(|v| v.len()).unwrap_or(0),
    }
}

fn value_preview(value: &serde_json::Value, max_bytes: usize) -> String {
    match value {
        serde_json::Value::String(raw) => truncate_utf8(raw, max_bytes),
        _ => serde_json::to_string(value)
            .map(|raw| truncate_utf8(&raw, max_bytes))
            .unwrap_or_else(|_| "<serialize_failed>".to_string()),
    }
}

fn value_sha256_hex(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(raw) => bytes_sha256_hex(raw.as_bytes()),
        _ => serde_json::to_vec(value)
            .map(|raw| bytes_sha256_hex(&raw))
            .unwrap_or_default(),
    }
}

fn bytes_preview(bytes: &[u8], max_bytes: usize) -> String {
    let raw = String::from_utf8_lossy(bytes);
    truncate_utf8(&raw, max_bytes)
}

fn bytes_sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn truncate_utf8(raw: &str, max_bytes: usize) -> String {
    if raw.len() <= max_bytes {
        return raw.to_string();
    }
    let mut end = max_bytes;
    while !raw.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...<truncated {} bytes>", &raw[..end], raw.len() - end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue};

    fn tool_call_body(
        arguments: serde_json::Value,
        parameters: Option<serde_json::Value>,
    ) -> Bytes {
        let mut body = serde_json::json!({
            "model": "x",
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": arguments,
                    }
                }]
            }]
        });
        if let Some(parameters) = parameters {
            body["tools"] = serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": parameters,
                }
            }]);
        }
        Bytes::from(serde_json::to_vec(&body).unwrap())
    }

    fn tool_call_body_without_arguments(parameters: Option<serde_json::Value>) -> Bytes {
        let mut body = serde_json::json!({
            "model": "x",
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup"
                    }
                }]
            }]
        });
        if let Some(parameters) = parameters {
            body["tools"] = serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": parameters,
                }
            }]);
        }
        Bytes::from(serde_json::to_vec(&body).unwrap())
    }

    fn normalized_tool_arguments(
        arguments: serde_json::Value,
        parameters: Option<serde_json::Value>,
    ) -> Result<String, ApiError> {
        let body = tool_call_body(arguments, parameters);
        let normalized = normalize_chat_tool_call_arguments(&HeaderMap::new(), body)?;
        let parsed: serde_json::Value = serde_json::from_slice(&normalized).unwrap();
        Ok(
            parsed["messages"][0]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .expect("normalized arguments must be a string")
                .to_string(),
        )
    }

    fn bad_request_message(error: ApiError) -> String {
        match error {
            ApiError::BadRequest(message) => message,
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn leaves_valid_object_string_unchanged() {
        let body = tool_call_body(
            serde_json::Value::String(r#"{"city":"Beijing"}"#.to_string()),
            None,
        );

        let out = normalize_chat_tool_call_arguments(&HeaderMap::new(), body.clone()).unwrap();

        assert_eq!(out, body);
    }

    #[test]
    fn stringifies_native_object() {
        let normalized =
            normalized_tool_arguments(serde_json::json!({"city": "Beijing", "days": 2}), None)
                .unwrap();

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&normalized).unwrap(),
            serde_json::json!({"city": "Beijing", "days": 2})
        );
    }

    #[test]
    fn repairs_double_encoded_and_fenced_objects() {
        let object = r#"{"city":"Beijing"}"#;
        let double_encoded = serde_json::to_string(object).unwrap();
        let fenced = format!("```json\n{object}\n```");

        for raw in [double_encoded, fenced] {
            let normalized = normalized_tool_arguments(serde_json::Value::String(raw), None)
                .expect("bounded repair should succeed");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&normalized).unwrap(),
                serde_json::json!({"city": "Beijing"})
            );
        }
    }

    #[test]
    fn repairs_multi_encoded_objects_up_to_bounded_depth() {
        let mut raw = r#"{"city":"Beijing"}"#.to_string();
        for _ in 0..MAX_JSON_STRING_DECODE_DEPTH {
            raw = serde_json::to_string(&raw).unwrap();
        }

        let normalized = normalized_tool_arguments(serde_json::Value::String(raw), None).unwrap();

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&normalized).unwrap(),
            serde_json::json!({"city": "Beijing"})
        );
    }

    #[test]
    fn rejects_multi_encoded_objects_past_bounded_depth() {
        let mut raw = r#"{"city":"Beijing"}"#.to_string();
        for _ in 0..=MAX_JSON_STRING_DECODE_DEPTH {
            raw = serde_json::to_string(&raw).unwrap();
        }

        let error = normalized_tool_arguments(serde_json::Value::String(raw), None).unwrap_err();
        let message = bad_request_message(error);

        assert!(message.contains("messages[0].tool_calls[0].function.arguments"));
        assert!(message.contains("automatic repair was not safe"));
    }

    #[test]
    fn repairs_one_line_json_fence() {
        let normalized = normalized_tool_arguments(
            serde_json::Value::String(r#"```json {"city":"Beijing"}```"#.to_string()),
            None,
        )
        .unwrap();

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&normalized).unwrap(),
            serde_json::json!({"city": "Beijing"})
        );
    }

    #[test]
    fn repairs_empty_for_simple_optional_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"verbose": {"type": "boolean"}}
        });
        let normalized = normalized_tool_arguments(
            serde_json::Value::String(" \n\t ".to_string()),
            Some(schema.clone()),
        )
        .unwrap();

        assert_eq!(normalized, "{}");

        let normalized =
            normalized_tool_arguments(serde_json::Value::Null, Some(schema.clone())).unwrap();

        assert_eq!(normalized, "{}");

        let body = tool_call_body_without_arguments(Some(schema));
        let normalized = normalize_chat_tool_call_arguments(&HeaderMap::new(), body).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&normalized).unwrap();

        assert_eq!(
            parsed["messages"][0]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }

    #[test]
    fn reject_path_accepts_trace_headers_for_logging() {
        let mut headers = HeaderMap::new();
        headers.insert("x-trace-id", HeaderValue::from_static("trace-test"));
        headers.insert("x-request-id", HeaderValue::from_static("request-test"));

        let body = tool_call_body(
            serde_json::Value::String(r#"{"city":"SENSITIVE_VALUE""#.to_string()),
            None,
        );
        let error = normalize_chat_tool_call_arguments(&headers, body).unwrap_err();
        let message = bad_request_message(error);

        assert!(message.contains("messages[0].tool_calls[0].function.arguments"));
        assert!(!message.contains("SENSITIVE_VALUE"));
    }

    #[test]
    fn rejects_empty_for_required_or_ambiguous_schema() {
        for schema in [
            serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
            serde_json::Value::Null,
            serde_json::json!({"$dynamicRef": "#required-arguments"}),
        ] {
            let error =
                normalized_tool_arguments(serde_json::Value::String(String::new()), Some(schema))
                    .unwrap_err();
            let message = bad_request_message(error);

            assert!(message.contains("messages[0].tool_calls[0].function.arguments"));
            assert!(message.contains("automatic repair was not safe"));
        }
    }

    #[test]
    fn rejects_truncated_or_non_object_json_without_leak() {
        for raw in [
            r#"{"city":"SENSITIVE_VALUE""#,
            "city=SENSITIVE_VALUE",
            "[1,2]",
            r#""SENSITIVE_VALUE""#,
        ] {
            let error = normalized_tool_arguments(serde_json::Value::String(raw.to_string()), None)
                .unwrap_err();
            let message = bad_request_message(error);

            assert!(message.contains("messages[0].tool_calls[0].function.arguments"));
            assert!(message.contains("automatic repair was not safe"));
            assert!(!message.contains("SENSITIVE_VALUE"));
        }
    }

    #[test]
    fn rejects_non_object_native_value() {
        let error = normalized_tool_arguments(serde_json::json!(["not", "an", "object"]), None)
            .unwrap_err();

        assert!(bad_request_message(error).contains("messages[0].tool_calls[0].function.arguments"));
    }
}
