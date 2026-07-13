// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::error::ApiError;
use bytes::Bytes;
use std::collections::HashMap;

enum ToolArgumentsAction {
    Unchanged,
    Replace {
        value: String,
        repair_kind: &'static str,
    },
    Reject,
}

/// Normalize only assistant tool-call arguments that can be repaired without
/// guessing their meaning.
///
/// Accepted compatibility shapes are native JSON objects, one extra JSON
/// string-encoding layer, complete `json` code fences, and empty arguments for
/// a matching simple tool schema that allows `{}`. Truncated JSON, arrays,
/// scalars, mixed prose, and schema-ambiguous empty values are rejected at the
/// gateway edge. The error deliberately reports only the JSON path, never the
/// argument content.
pub(crate) fn normalize_chat_tool_call_arguments(body: Bytes) -> Result<Bytes, ApiError> {
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
            let Some(arguments) = function.get_mut("arguments") else {
                return Err(unsafe_tool_arguments_error(message_index, tool_call_index));
            };

            match normalize_tool_arguments_value(arguments, allow_empty) {
                ToolArgumentsAction::Unchanged => {}
                ToolArgumentsAction::Replace { value, repair_kind } => {
                    *arguments = serde_json::Value::String(value);
                    changed = true;
                    tracing::debug!(
                        message_index,
                        tool_call_index,
                        repair_kind,
                        "normalized assistant tool-call arguments",
                    );
                }
                ToolArgumentsAction::Reject => {
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
            Err(_) => ToolArgumentsAction::Reject,
        },
        serde_json::Value::String(raw) => normalize_tool_arguments_string(raw, allow_empty),
        _ => ToolArgumentsAction::Reject,
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
            ToolArgumentsAction::Reject
        };
    }

    if let Some(fenced) = strip_complete_json_fence(trimmed) {
        return compact_object_argument(fenced, "json_fence");
    }

    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(serde_json::Value::Object(_)) => ToolArgumentsAction::Unchanged,
        Ok(serde_json::Value::String(nested)) => {
            let nested = nested.trim();
            if let Some(fenced) = strip_complete_json_fence(nested) {
                compact_object_argument(fenced, "double_encoded_json_fence")
            } else {
                compact_object_argument(nested, "double_encoded")
            }
        }
        _ => ToolArgumentsAction::Reject,
    }
}

fn compact_object_argument(raw: &str, repair_kind: &'static str) -> ToolArgumentsAction {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value @ serde_json::Value::Object(_)) => match serde_json::to_string(&value) {
            Ok(value) => ToolArgumentsAction::Replace { value, repair_kind },
            Err(_) => ToolArgumentsAction::Reject,
        },
        _ => ToolArgumentsAction::Reject,
    }
}

fn strip_complete_json_fence(raw: &str) -> Option<&str> {
    let inner = raw.strip_prefix("```")?.strip_suffix("```")?;
    let (language, payload) = inner.split_once('\n')?;
    let language = language.trim_end_matches('\r').trim();
    if !language.is_empty() && !language.eq_ignore_ascii_case("json") {
        return None;
    }
    Some(payload.trim())
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

#[cfg(test)]
mod tests {
    use super::*;

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

    fn normalized_tool_arguments(
        arguments: serde_json::Value,
        parameters: Option<serde_json::Value>,
    ) -> Result<String, ApiError> {
        let body = tool_call_body(arguments, parameters);
        let normalized = normalize_chat_tool_call_arguments(body)?;
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

        let out = normalize_chat_tool_call_arguments(body.clone()).unwrap();

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
    fn repairs_empty_for_simple_optional_schema() {
        let normalized = normalized_tool_arguments(
            serde_json::Value::String(" \n\t ".to_string()),
            Some(serde_json::json!({
                "type": "object",
                "properties": {"verbose": {"type": "boolean"}}
            })),
        )
        .unwrap();

        assert_eq!(normalized, "{}");
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
