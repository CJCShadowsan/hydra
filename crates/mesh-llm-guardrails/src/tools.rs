use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::fmt;
use url::Url;

use crate::{request_contract::GuardrailRequestContract, structured::StructuredOutputSpec};

pub const MESH_RESPOND_TOOL_NAME: &str = "_mesh_respond";
pub const MESH_EMIT_STRUCTURED_TOOL_NAME: &str = "_mesh_emit_structured";

pub fn request_uses_reserved_tool_name(
    request: &GuardrailRequestContract,
    reserved_prefix: &str,
) -> bool {
    request
        .tool_names()
        .any(|name| is_reserved_tool_name(name, reserved_prefix))
        || request
            .forced_tool_name()
            .is_some_and(|name| is_reserved_tool_name(name, reserved_prefix))
}

pub fn is_reserved_tool_name(name: &str, reserved_prefix: &str) -> bool {
    name.starts_with(reserved_prefix)
}

pub fn model_param_size_b(name: &str) -> Option<f32> {
    let bytes = name.as_bytes();
    for i in 0..bytes.len() {
        let c = bytes[i];
        if !c.is_ascii_digit() {
            continue;
        }
        if i > 0 {
            let prev = bytes[i - 1];
            if prev.is_ascii_digit() || prev == b'.' || prev.is_ascii_alphabetic() {
                continue;
            }
        }
        if c == b'0' {
            continue;
        }

        let mut end = i + 1;
        while let Some(&next) = bytes.get(end) {
            if next.is_ascii_digit() || next == b'.' {
                end += 1;
                continue;
            }
            break;
        }

        let Some(&unit) = bytes.get(end) else {
            continue;
        };
        if unit != b'b' && unit != b'B' {
            continue;
        }
        if bytes
            .get(end + 1)
            .is_some_and(|after| after.is_ascii_digit())
        {
            continue;
        }

        let number = std::str::from_utf8(&bytes[i..end])
            .ok()?
            .parse::<f32>()
            .ok()?;
        if number > 0.0 {
            return Some(number);
        }
    }
    None
}

pub fn mesh_respond_tool_definition() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": MESH_RESPOND_TOOL_NAME,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string"
                    }
                },
                "required": ["message"],
                "additionalProperties": false
            }
        }
    })
}

pub fn mesh_emit_structured_tool_definition(structured_output: &StructuredOutputSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": MESH_EMIT_STRUCTURED_TOOL_NAME,
            "parameters": structured_output.tool_parameters()
        }
    })
}

pub fn extract_tool_name_and_arguments(value: &Value) -> Option<(&str, &Value)> {
    let object = value.as_object()?;
    let nested_function = object.get("function").and_then(Value::as_object);
    let name = nested_function
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| object.get("name").and_then(Value::as_str))
        .or_else(|| object.get("function").and_then(Value::as_str))
        .or_else(|| object.get("tool").and_then(Value::as_str))?;
    let arguments = nested_function
        .and_then(|function| function.get("arguments"))
        .or_else(|| object.get("arguments"))?;
    Some((name, arguments))
}

pub fn normalize_tool_arguments(arguments: &Value) -> Option<Map<String, Value>> {
    match arguments {
        Value::Object(arguments) => Some(arguments.clone()),
        Value::String(arguments) => serde_json::from_str::<Value>(arguments)
            .ok()?
            .as_object()
            .cloned(),
        Value::Null => None,
        _ => Some(Map::new()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolArgumentSchemaError {
    MissingRequired {
        tool_name: String,
        fields: Vec<String>,
    },
}

impl fmt::Display for ToolArgumentSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequired { tool_name, fields } => {
                write!(
                    f,
                    "tool {tool_name:?} missing required argument(s): {}",
                    fields.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for ToolArgumentSchemaError {}

pub fn sanitize_tool_arguments_for_tool(
    tool_name: &str,
    arguments: &Value,
    tools: Option<&Value>,
) -> Result<Value, ToolArgumentSchemaError> {
    let mut arguments = normalize_tool_arguments(arguments)
        .map(Value::Object)
        .unwrap_or_else(|| json!({}));

    let Some(parameters) = tool_parameters(tool_name, tools) else {
        return Ok(arguments);
    };

    repair_argument_keys_for_schema(&mut arguments, parameters);
    sanitize_object_for_schema(&mut arguments, parameters);
    ensure_required_arguments(tool_name, &arguments, parameters)?;
    Ok(arguments)
}

fn tool_parameters<'a>(tool_name: &str, tools: Option<&'a Value>) -> Option<&'a Value> {
    tools?
        .as_array()?
        .iter()
        .find(|tool| {
            tool.pointer("/function/name")
                .and_then(Value::as_str)
                .is_some_and(|name| name == tool_name)
        })?
        .pointer("/function/parameters")
}

fn repair_argument_keys_for_schema(arguments: &mut Value, schema: &Value) {
    let Some(arguments) = arguments.as_object_mut() else {
        return;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };

    for repair in argument_key_repairs(arguments, properties, required_fields(schema)) {
        if arguments.contains_key(&repair.to) {
            continue;
        }
        if let Some(value) = arguments.remove(&repair.from) {
            arguments.insert(repair.to, value);
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ArgumentKeyRepair {
    from: String,
    to: String,
}

fn argument_key_repairs(
    arguments: &Map<String, Value>,
    properties: &Map<String, Value>,
    required: HashSet<&str>,
) -> Vec<ArgumentKeyRepair> {
    arguments
        .iter()
        .filter(|(key, _)| !properties.contains_key(*key))
        .filter_map(|(key, value)| {
            let destination = high_confidence_property_match(key, value, properties, &required)?;
            Some(ArgumentKeyRepair {
                from: key.clone(),
                to: destination.to_string(),
            })
        })
        .collect()
}

fn required_fields(schema: &Value) -> HashSet<&str> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

fn high_confidence_property_match<'a>(
    raw_key: &str,
    value: &Value,
    properties: &'a Map<String, Value>,
    required: &HashSet<&str>,
) -> Option<&'a str> {
    let mut matches = properties
        .iter()
        .filter(|(name, schema)| {
            argument_value_matches_schema(value, schema, Some(name))
                && (required.contains(name.as_str()))
        })
        .filter_map(|(name, _)| property_match_score(raw_key, name).map(|score| (score, name)))
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(right.1)));

    let [(score, name), rest @ ..] = matches.as_slice() else {
        return None;
    };
    let runner_up = rest.first().map(|(score, _)| *score).unwrap_or(0);
    (*score >= 90 && *score >= runner_up + 12).then_some(name.as_str())
}

fn property_match_score(raw_key: &str, candidate: &str) -> Option<u16> {
    let raw_norm = normalized_identifier(raw_key);
    let candidate_norm = normalized_identifier(candidate);
    if raw_norm.len() < 3 || candidate_norm.len() < 3 {
        return None;
    }
    if raw_norm == candidate_norm {
        return Some(120);
    }

    let raw_tokens = identifier_tokens(raw_key);
    let candidate_tokens = identifier_tokens(candidate);
    if !candidate_tokens.is_empty() && token_suffix_matches(&raw_tokens, &candidate_tokens) {
        return Some(105);
    }

    close_identifier_match(&raw_norm, &candidate_norm).then_some(95)
}

fn normalized_identifier(name: &str) -> String {
    name.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn identifier_tokens(name: &str) -> Vec<String> {
    name.split(|character: char| !character.is_ascii_alphanumeric())
        .filter_map(|token| {
            let token = token.trim().to_ascii_lowercase();
            (!token.is_empty()).then_some(token)
        })
        .collect()
}

fn token_suffix_matches(raw_tokens: &[String], candidate_tokens: &[String]) -> bool {
    raw_tokens.len() >= candidate_tokens.len()
        && raw_tokens[raw_tokens.len() - candidate_tokens.len()..] == *candidate_tokens
}

fn close_identifier_match(left: &str, right: &str) -> bool {
    if left.len().abs_diff(right.len()) > 2 || left.len().min(right.len()) < 5 {
        return false;
    }
    levenshtein_distance(left, right) <= 1
}

fn levenshtein_distance(left: &str, right: &str) -> usize {
    let mut previous: Vec<usize> = (0..=right.chars().count()).collect();
    let mut current = vec![0; previous.len()];
    for (left_index, left_char) in left.chars().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_char) in right.chars().enumerate() {
            let insert = current[right_index] + 1;
            let delete = previous[right_index + 1] + 1;
            let replace = previous[right_index] + usize::from(left_char != right_char);
            current[right_index + 1] = insert.min(delete).min(replace);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.chars().count()]
}

fn sanitize_object_for_schema(arguments: &mut Value, schema: &Value) {
    let Some(arguments) = arguments.as_object_mut() else {
        return;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    let allow_additional = matches!(schema.get("additionalProperties"), Some(Value::Bool(true)))
        || schema
            .get("additionalProperties")
            .is_some_and(Value::is_object);

    arguments.retain(|key, value| {
        let Some(property_schema) = properties.get(key) else {
            return allow_additional;
        };
        argument_value_matches_schema(value, property_schema, Some(key))
    });
}

fn argument_value_matches_schema(
    value: &Value,
    schema: &Value,
    property_name: Option<&str>,
) -> bool {
    match schema.get("enum").and_then(Value::as_array) {
        Some(enum_values) if !enum_values.iter().any(|allowed| allowed == value) => return false,
        _ => {}
    }
    if !string_value_matches_declared_format(value, schema, property_name) {
        return false;
    }

    let Some(schema_type) = schema.get("type") else {
        return true;
    };
    let types: Vec<&str> = match schema_type {
        Value::String(t) => vec![t.as_str()],
        Value::Array(types) => types.iter().filter_map(Value::as_str).collect(),
        _ => return true,
    };
    types
        .iter()
        .any(|schema_type| value_matches_type(value, schema_type))
}

fn string_value_matches_declared_format(
    value: &Value,
    schema: &Value,
    property_name: Option<&str>,
) -> bool {
    let Some(text) = value.as_str() else {
        return true;
    };
    let text = text.trim();
    match declared_string_format(schema).as_deref() {
        Some("uri") => Url::parse(text).is_ok(),
        Some("uri-reference") => uri_reference_is_valid(text),
        Some("url") => http_url_is_valid(text),
        _ if property_name_looks_url_like(property_name) => http_url_is_valid(text),
        _ => true,
    }
}

fn declared_string_format(schema: &Value) -> Option<String> {
    schema
        .get("format")
        .and_then(Value::as_str)
        .map(|format| format.to_ascii_lowercase())
}

fn property_name_looks_url_like(property_name: Option<&str>) -> bool {
    property_name.is_some_and(|name| {
        identifier_tokens(name)
            .iter()
            .any(|token| matches!(token.as_str(), "url" | "uri"))
    })
}

fn http_url_is_valid(text: &str) -> bool {
    Url::parse(text).is_ok_and(|url| matches!(url.scheme(), "http" | "https"))
}

fn uri_reference_is_valid(text: &str) -> bool {
    if Url::parse(text).is_ok() {
        return true;
    }
    Url::parse("https://example.invalid/")
        .and_then(|base| base.join(text))
        .is_ok()
}

fn value_matches_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "array" => value.is_array(),
        "boolean" => value.is_boolean(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "object" => value.is_object(),
        "string" => value.is_string(),
        _ => true,
    }
}

fn ensure_required_arguments(
    tool_name: &str,
    arguments: &Value,
    schema: &Value,
) -> Result<(), ToolArgumentSchemaError> {
    let Some(required) = schema.get("required").and_then(Value::as_array) else {
        return Ok(());
    };
    let Some(arguments) = arguments.as_object() else {
        return Ok(());
    };
    let missing: Vec<String> = required
        .iter()
        .filter_map(Value::as_str)
        .filter(|field| !arguments.contains_key(*field))
        .map(str::to_string)
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(ToolArgumentSchemaError::MissingRequired {
            tool_name: tool_name.to_string(),
            fields: missing,
        })
    }
}

pub fn tool_arguments_wire_string(arguments: &Value) -> String {
    match arguments {
        Value::String(value) => serde_json::from_str::<Value>(value)
            .ok()
            .filter(Value::is_object)
            .map_or_else(|| "{}".to_string(), |_| value.clone()),
        Value::Object(_) => serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
        Value::Null => "{}".to_string(),
        _ => "{}".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value, json};

    use super::*;

    #[test]
    fn normalize_tool_arguments_handles_null_string_and_primitive_inputs() {
        let object = json!({"path": "README.md"});
        assert_eq!(
            normalize_tool_arguments(&object).unwrap()["path"],
            "README.md"
        );

        let string = Value::String("{\"path\":\"README.md\"}".to_string());
        assert_eq!(
            normalize_tool_arguments(&string).unwrap()["path"],
            "README.md"
        );

        assert_eq!(normalize_tool_arguments(&Value::Null), None);
        assert_eq!(normalize_tool_arguments(&Value::from(42)), Some(Map::new()));
    }

    #[test]
    fn tool_arguments_wire_string_always_returns_object_json() {
        assert_eq!(tool_arguments_wire_string(&Value::Null), "{}");
        assert_eq!(tool_arguments_wire_string(&Value::from(42)), "{}");
        assert_eq!(
            tool_arguments_wire_string(&Value::String("not json".into())),
            "{}"
        );
        assert_eq!(
            tool_arguments_wire_string(&json!({"path": "README.md"})),
            "{\"path\":\"README.md\"}"
        );
    }

    #[test]
    fn schema_sanitizer_removes_unknown_and_invalid_arguments() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "exec",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "host": {"type": "string", "enum": ["gateway"]}
                    },
                    "required": ["command"],
                    "additionalProperties": false
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "exec",
            &json!({
                "command": "echo ok",
                "host": "sandbox",
                "extra": true
            }),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"command": "echo ok"}));
    }

    #[test]
    fn schema_sanitizer_rejects_missing_required_after_cleanup() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }]);

        let err = sanitize_tool_arguments_for_tool(
            "read_file",
            &json!({"path": 42, "other": "x"}),
            Some(&tools),
        )
        .unwrap_err();

        assert_eq!(
            err,
            ToolArgumentSchemaError::MissingRequired {
                tool_name: "read_file".into(),
                fields: vec!["path".into()]
            }
        );
    }

    #[test]
    fn schema_sanitizer_rejects_invalid_required_url_argument() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "web_fetch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"],
                    "additionalProperties": false
                }
            }
        }]);

        let err = sanitize_tool_arguments_for_tool(
            "web_fetch",
            &json!({"url": "s internal feedback directly."}),
            Some(&tools),
        )
        .unwrap_err();

        assert_eq!(
            err,
            ToolArgumentSchemaError::MissingRequired {
                tool_name: "web_fetch".into(),
                fields: vec!["url".into()]
            }
        );
    }

    #[test]
    fn schema_sanitizer_preserves_valid_uri_format_argument() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "fetch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "format": "uri"}
                    },
                    "required": ["target"],
                    "additionalProperties": false
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "fetch",
            &json!({"target": "https://example.com/path?q=1"}),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"target": "https://example.com/path?q=1"}));
    }

    #[test]
    fn schema_sanitizer_accepts_non_http_uri_format_argument() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "open_uri",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "format": "uri"}
                    },
                    "required": ["target"],
                    "additionalProperties": false
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "open_uri",
            &json!({"target": "mailto:user@example.com"}),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"target": "mailto:user@example.com"}));
    }

    #[test]
    fn schema_sanitizer_accepts_relative_uri_reference_argument() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "open_reference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "format": "uri-reference"}
                    },
                    "required": ["target"],
                    "additionalProperties": false
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "open_reference",
            &json!({"target": "../docs/readme.md#intro"}),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"target": "../docs/readme.md#intro"}));
    }

    #[test]
    fn schema_sanitizer_repairs_required_argument_key_from_schema() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "read_file",
            &json!({"file_path": "README.md"}),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"path": "README.md"}));
    }

    #[test]
    fn schema_sanitizer_does_not_repair_non_required_argument_key() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "additionalProperties": false
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "read_file",
            &json!({"file_path": "README.md"}),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({}));
    }

    #[test]
    fn schema_sanitizer_preserves_additional_properties_when_schema_allows_them() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "kv",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fixed": {"type": "string"}
                    },
                    "additionalProperties": true
                }
            }
        }]);

        let cleaned = sanitize_tool_arguments_for_tool(
            "kv",
            &json!({"fixed": "a", "dynamic": 1}),
            Some(&tools),
        )
        .unwrap();

        assert_eq!(cleaned, json!({"fixed": "a", "dynamic": 1}));
    }
}
