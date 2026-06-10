const MOA_DEFAULT_COMPLETION_BUDGET_TOKENS: u32 = 1024;
const MOA_CHAT_CONTEXT_RESERVE_TOKENS: u32 = 1024;
const MOA_TOOL_CONTEXT_RESERVE_TOKENS: u32 = 2048;
const MOA_TOOL_RESULT_CONTEXT_RESERVE_TOKENS: u32 = 4096;

pub(in crate::network::openai::moa_gateway) fn moa_required_tokens(
    body: &serde_json::Value,
    transport_required_tokens: Option<u32>,
) -> Option<u32> {
    let serialized_prompt_tokens = serde_json::to_vec(body)
        .ok()
        .map(|bytes| ceil_div_u32(saturating_u32(bytes.len()), 4));
    let mut required = transport_required_tokens.or(serialized_prompt_tokens)?;

    if completion_budget_missing(body) {
        required = required.saturating_add(MOA_DEFAULT_COMPLETION_BUDGET_TOKENS);
    }

    Some(required.saturating_add(moa_context_reserve_tokens(body)))
}

fn moa_context_reserve_tokens(body: &serde_json::Value) -> u32 {
    if body_has_tool_result(body) {
        MOA_TOOL_RESULT_CONTEXT_RESERVE_TOKENS
    } else if body_has_tools(body) {
        MOA_TOOL_CONTEXT_RESERVE_TOKENS
    } else {
        MOA_CHAT_CONTEXT_RESERVE_TOKENS
    }
}

fn body_has_tool_result(body: &serde_json::Value) -> bool {
    if body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .map(|messages| messages.iter().any(message_has_tool_result))
        .unwrap_or(false)
    {
        return true;
    }

    ["output", "items", "input"]
        .into_iter()
        .filter_map(|key| body.get(key))
        .any(value_has_tool_result)
}

fn message_has_tool_result(message: &serde_json::Value) -> bool {
    message.get("role").and_then(serde_json::Value::as_str) == Some("tool")
        || content_block_is_tool_result(message)
        || message
            .get("content")
            .and_then(serde_json::Value::as_array)
            .map(|parts| parts.iter().any(value_has_tool_result))
            .unwrap_or(false)
        || message.get("content").is_some_and(value_has_tool_result)
}

fn value_has_tool_result(value: &serde_json::Value) -> bool {
    if content_block_is_tool_result(value) {
        return true;
    }
    match value {
        serde_json::Value::Array(values) => values.iter().any(value_has_tool_result),
        serde_json::Value::Object(map) => {
            map.values().any(value_has_tool_result)
                || map.get("role").and_then(serde_json::Value::as_str) == Some("tool")
        }
        _ => false,
    }
}

fn content_block_is_tool_result(part: &serde_json::Value) -> bool {
    let Some(kind) = part.get("type").and_then(serde_json::Value::as_str) else {
        return false;
    };
    matches!(
        normalize_content_block_type(kind).as_str(),
        "functioncalloutput" | "tooloutput" | "toolresult" | "toolresponse"
    )
}

fn normalize_content_block_type(kind: &str) -> String {
    kind.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn body_has_tools(body: &serde_json::Value) -> bool {
    body.get("tools")
        .and_then(serde_json::Value::as_array)
        .map(|tools| !tools.is_empty())
        .unwrap_or(false)
}

fn completion_budget_missing(body: &serde_json::Value) -> bool {
    [
        "max_completion_tokens",
        "max_tokens",
        "max_output_tokens",
        "n_predict",
    ]
    .into_iter()
    .all(|key| body.get(key).and_then(serde_json::Value::as_u64).is_none())
}

fn saturating_u32(value: usize) -> u32 {
    value.try_into().unwrap_or(u32::MAX)
}

fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor - 1) / divisor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moa_required_tokens_adds_chat_completion_budget_and_reserve() {
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [{"role": "user", "content": "hello"}],
        });

        assert_eq!(
            moa_required_tokens(&body, Some(2000)),
            Some(2000 + MOA_DEFAULT_COMPLETION_BUDGET_TOKENS + MOA_CHAT_CONTEXT_RESERVE_TOKENS)
        );
    }

    #[test]
    fn moa_required_tokens_respects_explicit_max_tokens() {
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
        });

        assert_eq!(
            moa_required_tokens(&body, Some(2000)),
            Some(2000 + MOA_CHAT_CONTEXT_RESERVE_TOKENS)
        );
    }

    #[test]
    fn moa_required_tokens_keeps_extra_room_for_tools() {
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        });

        assert_eq!(
            moa_required_tokens(&body, Some(2000)),
            Some(2000 + MOA_DEFAULT_COMPLETION_BUDGET_TOKENS + MOA_TOOL_CONTEXT_RESERVE_TOKENS)
        );
    }

    #[test]
    fn moa_required_tokens_keeps_most_room_for_tool_results() {
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [
                {"role": "user", "content": "read"},
                {"role": "tool", "content": "large output"}
            ],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        });

        assert_eq!(
            moa_required_tokens(&body, Some(2000)),
            Some(
                2000 + MOA_DEFAULT_COMPLETION_BUDGET_TOKENS
                    + MOA_TOOL_RESULT_CONTEXT_RESERVE_TOKENS
            )
        );
    }

    #[test]
    fn moa_required_tokens_detects_user_wrapped_tool_result_blocks() {
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [
                {"role": "user", "content": "read"},
                {
                    "role": "user",
                    "content": [
                        {"type": "toolResult", "id": "call_1", "toolResult": {"value": "large output"}}
                    ]
                }
            ],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        });

        assert_eq!(
            moa_required_tokens(&body, Some(2000)),
            Some(
                2000 + MOA_DEFAULT_COMPLETION_BUDGET_TOKENS
                    + MOA_TOOL_RESULT_CONTEXT_RESERVE_TOKENS
            )
        );
    }

    #[test]
    fn moa_required_tokens_detects_responses_tool_outputs() {
        let body = serde_json::json!({
            "model": "mesh",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Need tool."}]},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "large output"
                }
            ],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        });

        assert_eq!(
            moa_required_tokens(&body, Some(2000)),
            Some(
                2000 + MOA_DEFAULT_COMPLETION_BUDGET_TOKENS
                    + MOA_TOOL_RESULT_CONTEXT_RESERVE_TOKENS
            )
        );
    }
}
