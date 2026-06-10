use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::tools::{extract_tool_name_and_arguments, normalize_tool_arguments};

const MAX_RESCUE_INPUT_BYTES: usize = 64 * 1024;
const MAX_JSON_CANDIDATES: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallParseError {
    Malformed,
    UnknownTool,
    InvalidArguments,
}

pub fn strip_thinking_blocks(content: &str) -> String {
    let stripped_html = strip_tag_pairs(content, "<think>", "</think>");
    let stripped_brackets = strip_tag_pairs(&stripped_html, "[THINK]", "[/THINK]");
    stripped_brackets.trim().to_string()
}

pub fn parse_tool_call_value(
    value: &Value,
    allowed_tools: &[String],
) -> Result<Vec<ParsedToolCall>, ToolCallParseError> {
    let raw_tool_calls = match raw_tool_calls_from_value(value) {
        Some(tool_calls) if !tool_calls.is_empty() => tool_calls,
        _ => return Err(ToolCallParseError::Malformed),
    };
    let allowed_tools = allowed_tools
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut parsed_calls = Vec::new();
    for tool_call in raw_tool_calls {
        parsed_calls.push(parse_one_tool_call(tool_call, &allowed_tools)?);
    }
    Ok(parsed_calls)
}

pub fn rescue_tool_call_from_text(
    content: &str,
    allowed_tools: &[String],
) -> Result<Vec<ParsedToolCall>, ToolCallParseError> {
    let content = strip_thinking_blocks(content);
    let mut last_error = ToolCallParseError::Malformed;
    for candidate in tool_call_candidates(&content) {
        match parse_tool_call_value(&candidate, allowed_tools) {
            Ok(parsed) => return Ok(parsed),
            Err(error) => last_error = more_specific_error(last_error, error),
        }
    }
    Err(last_error)
}

fn more_specific_error(
    current: ToolCallParseError,
    next: ToolCallParseError,
) -> ToolCallParseError {
    match (current, next) {
        (ToolCallParseError::InvalidArguments, _) | (_, ToolCallParseError::InvalidArguments) => {
            ToolCallParseError::InvalidArguments
        }
        (ToolCallParseError::UnknownTool, _) | (_, ToolCallParseError::UnknownTool) => {
            ToolCallParseError::UnknownTool
        }
        _ => ToolCallParseError::Malformed,
    }
}

fn strip_tag_pairs(content: &str, start_tag: &str, end_tag: &str) -> String {
    let mut remainder = content;
    let mut result = String::new();
    while let Some(start_index) = remainder.find(start_tag) {
        result.push_str(&remainder[..start_index]);
        let after_start = &remainder[start_index + start_tag.len()..];
        if let Some(end_index) = after_start.find(end_tag) {
            remainder = &after_start[end_index + end_tag.len()..];
        } else {
            remainder = &remainder[..start_index];
            break;
        }
    }
    result.push_str(remainder);
    result
}

fn tool_call_candidates(content: &str) -> Vec<Value> {
    let mut candidates = Vec::new();
    for json_candidate in json_candidates(content) {
        if let Ok(value) = serde_json::from_str::<Value>(&json_candidate) {
            candidates.push(value);
        }
    }
    if let Some(value) = parse_bracket_args_tool_syntax(content) {
        candidates.push(value);
    }
    if let Some(value) = parse_qwen_xml_syntax(content) {
        candidates.push(value);
    }
    if let Some(value) = parse_bracketed_tool_call_syntax(content) {
        candidates.push(value);
    }
    if let Some(value) = parse_sentinel_tool_call_syntax(content) {
        candidates.push(value);
    }
    if let Some(value) = parse_granite_tool_call_syntax(content) {
        candidates.push(value);
    }
    candidates
}

fn json_candidates(content: &str) -> Vec<String> {
    let content = bounded_prefix(content, MAX_RESCUE_INPUT_BYTES);
    let mut candidates = Vec::new();
    push_candidate(&mut candidates, content.trim());
    for fenced in fenced_code_blocks(content) {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        push_candidate(&mut candidates, fenced.trim());
    }
    for balanced in balanced_json_substrings(content) {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        push_candidate(&mut candidates, balanced.trim());
    }
    candidates
}

fn bounded_prefix(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }
    let mut end = max_bytes;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
}

fn push_candidate(candidates: &mut Vec<String>, candidate: &str) {
    if !candidate.is_empty() && !candidates.iter().any(|existing| existing == candidate) {
        candidates.push(candidate.to_string());
    }
}

fn fenced_code_blocks(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut remainder = content;
    while let Some(open_index) = remainder.find("```") {
        let after_open = &remainder[open_index + 3..];
        let Some(close_index) = after_open.find("```") else {
            break;
        };
        let block = &after_open[..close_index];
        let block = block
            .strip_prefix("json\n")
            .or_else(|| block.strip_prefix("JSON\n"))
            .unwrap_or(block);
        blocks.push(block.to_string());
        remainder = &after_open[close_index + 3..];
    }
    blocks
}

fn balanced_json_substrings(content: &str) -> Vec<String> {
    let bytes = content.as_bytes();
    let mut candidates = Vec::new();
    for (index, byte) in bytes.iter().enumerate() {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        let closing = match byte {
            b'{' => b'}',
            b'[' => b']',
            _ => continue,
        };
        if let Some(end) = balanced_substring_end(bytes, index, *byte, closing) {
            candidates.push(content[index..=end].to_string());
        }
    }
    candidates
}

fn balanced_substring_end(bytes: &[u8], start: usize, opening: u8, closing: u8) -> Option<usize> {
    let mut depth = 0_u32;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().copied().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match byte {
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            _ if byte == opening => depth += 1,
            _ if byte == closing => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_bracket_args_tool_syntax(content: &str) -> Option<Value> {
    let marker = "[ARGS]";
    if let Some(marker_index) = content.find(marker) {
        let name = trailing_tool_name(&content[..marker_index])?;
        let after_marker = content[marker_index + marker.len()..].trim_start();
        let json_text = first_balanced_object(after_marker)?;
        let arguments = serde_json::from_str::<Value>(&json_text).ok()?;
        return Some(json!({ "name": name, "arguments": arguments }));
    }
    parse_parenthesized_tool_call(content)
}

fn parse_qwen_xml_syntax(content: &str) -> Option<Value> {
    let function_prefix = "<function=";
    let start_index = content.find(function_prefix)?;
    let after_prefix = &content[start_index + function_prefix.len()..];
    let name_end = after_prefix.find('>')?;
    let name = after_prefix[..name_end]
        .trim()
        .trim_matches('"')
        .trim_matches('\'');
    if name.is_empty() {
        return None;
    }
    let body = &after_prefix[name_end + 1..];
    let function_end = body.find("</function>")?;
    let mut arguments = Map::new();
    let mut remainder = &body[..function_end];
    while let Some(parameter_start) = remainder.find("<parameter=") {
        let after_parameter = &remainder[parameter_start + "<parameter=".len()..];
        let name_end = after_parameter.find('>')?;
        let parameter_name = after_parameter[..name_end]
            .trim()
            .trim_matches('"')
            .trim_matches('\'');
        if parameter_name.is_empty() {
            return None;
        }
        let parameter_body = &after_parameter[name_end + 1..];
        let value_end = parameter_body.find("</parameter>")?;
        let value = parameter_body[..value_end].trim();
        let parsed_value = serde_json::from_str::<Value>(value)
            .unwrap_or_else(|_| Value::String(value.to_string()));
        arguments.insert(parameter_name.to_string(), parsed_value);
        remainder = &parameter_body[value_end + "</parameter>".len()..];
    }
    if arguments.is_empty() {
        return None;
    }
    Some(json!({ "name": name, "arguments": Value::Object(arguments) }))
}

fn parse_granite_tool_call_syntax(content: &str) -> Option<Value> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";
    let start_index = content.find(start_tag)?;
    let after_start = &content[start_index + start_tag.len()..];
    let end_index = after_start.find(end_tag)?;
    serde_json::from_str(after_start[..end_index].trim()).ok()
}

fn parse_bracketed_tool_call_syntax(content: &str) -> Option<Value> {
    let body = bracketed_tool_call_body(content)?;
    if let Ok(value) = serde_json::from_str::<Value>(body.trim()) {
        return Some(value);
    }
    let mut object = parse_pseudo_argument_object(body.trim())?;
    let name = object
        .remove("tool")
        .or_else(|| object.remove("name"))
        .or_else(|| object.remove("function"))
        .and_then(|value| value.as_str().map(ToString::to_string))?;
    let arguments = object
        .remove("args")
        .or_else(|| object.remove("arguments"))
        .unwrap_or_else(|| Value::Object(Map::new()));
    Some(json!({ "name": name, "arguments": arguments }))
}

fn bracketed_tool_call_body(content: &str) -> Option<&str> {
    let trimmed = content.trim_start();
    let after_start = trimmed.strip_prefix("[TOOL_CALL]")?;
    let end_index = after_start.find("[/TOOL_CALL]")?;
    Some(&after_start[..end_index])
}

fn parse_sentinel_tool_call_syntax(content: &str) -> Option<Value> {
    let body = sentinel_tool_call_body(content)?;
    let call_body = body.trim().strip_prefix("call:")?.trim_start();
    let name_len = call_body
        .char_indices()
        .take_while(|(_, character)| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        })
        .last()
        .map(|(index, character)| index + character.len_utf8())?;
    let name = call_body[..name_len].trim();
    if name.is_empty() {
        return None;
    }
    let arguments_text = call_body[name_len..].trim_start();
    let arguments = parse_pseudo_argument_object(arguments_text)?;
    Some(json!({ "name": name, "arguments": Value::Object(arguments) }))
}

fn sentinel_tool_call_body(content: &str) -> Option<&str> {
    let trimmed = content.trim_start();
    let after_start = trimmed.strip_prefix("<|tool_call>")?;
    let end_index = [
        "<|tool_call|>",
        "<tool_call|>",
        "<|/tool_call|>",
        "</tool_call>",
        "<|end_tool_call|>",
    ]
    .iter()
    .filter_map(|marker| after_start.find(marker))
    .min()?;
    Some(&after_start[..end_index])
}

fn parse_pseudo_argument_object(content: &str) -> Option<Map<String, Value>> {
    let object_text = first_balanced_pseudo_object(content)?;
    if let Ok(Value::Object(arguments)) = serde_json::from_str::<Value>(object_text) {
        return Some(arguments);
    }
    parse_unquoted_argument_object(object_text)
}

fn first_balanced_pseudo_object(content: &str) -> Option<&str> {
    let content = content.trim_start();
    if !content.starts_with('{') {
        return None;
    }
    let end = balanced_pseudo_object_end(content)?;
    Some(&content[..=end])
}

fn balanced_pseudo_object_end(content: &str) -> Option<usize> {
    let mut depth = 0_u32;
    let mut in_string: Option<char> = None;
    let mut escaped = false;
    let mut in_sentinel_string = false;
    let mut index = 0;
    while index < content.len() {
        if let Some(len) = sentinel_string_len_at(&content[index..]) {
            in_sentinel_string = !in_sentinel_string;
            index += len;
            continue;
        }
        let character = content[index..].chars().next()?;
        let next_index = index + character.len_utf8();
        if in_sentinel_string {
            index = next_index;
            continue;
        }
        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
                index = next_index;
                continue;
            }
            match character {
                '\\' => escaped = true,
                _ if character == quote => in_string = None,
                _ => {}
            }
            index = next_index;
            continue;
        }
        match character {
            '"' | '\'' => in_string = Some(character),
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index = next_index;
    }
    None
}

fn parse_unquoted_argument_object(object_text: &str) -> Option<Map<String, Value>> {
    let inner = object_text
        .trim()
        .strip_prefix('{')?
        .strip_suffix('}')?
        .trim();
    let mut arguments = Map::new();
    for field in split_pseudo_fields(inner) {
        let (key, value) = split_pseudo_key_value(field)?;
        arguments.insert(key.to_string(), parse_pseudo_value(value));
    }
    (!arguments.is_empty()).then_some(arguments)
}

fn split_pseudo_fields(input: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut start = 0;
    let mut depth = 0_u32;
    let mut in_string: Option<char> = None;
    let mut escaped = false;
    let mut in_sentinel_string = false;
    let mut index = 0;
    while index < input.len() {
        if let Some(len) = sentinel_string_len_at(&input[index..]) {
            in_sentinel_string = !in_sentinel_string;
            index += len;
            continue;
        }
        let Some(character) = input[index..].chars().next() else {
            break;
        };
        let next_index = index + character.len_utf8();
        if in_sentinel_string {
            index = next_index;
            continue;
        }
        if let Some(quote) = in_string {
            if escaped {
                escaped = false;
                index = next_index;
                continue;
            }
            match character {
                '\\' => escaped = true,
                _ if character == quote => in_string = None,
                _ => {}
            }
            index = next_index;
            continue;
        }
        match character {
            '"' | '\'' => in_string = Some(character),
            '{' | '[' => depth += 1,
            '}' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let field = input[start..index].trim();
                if !field.is_empty() {
                    fields.push(field);
                }
                start = index + character.len_utf8();
            }
            _ => {}
        }
        index = next_index;
    }
    let field = input[start..].trim();
    if !field.is_empty() {
        fields.push(field);
    }
    fields
}

fn sentinel_string_len_at(input: &str) -> Option<usize> {
    ["<|\"|>", "<|'|>"]
        .iter()
        .find(|marker| input.starts_with(**marker))
        .map(|marker| marker.len())
}

fn split_pseudo_key_value(field: &str) -> Option<(&str, &str)> {
    if let Some(separator) = field.find("=>") {
        return split_pseudo_key_value_at(field, separator, 2);
    }
    let separator = field.find(':').or_else(|| field.find('='))?;
    split_pseudo_key_value_at(field, separator, 1)
}

fn split_pseudo_key_value_at(
    field: &str,
    separator: usize,
    separator_len: usize,
) -> Option<(&str, &str)> {
    let key = field[..separator]
        .trim()
        .trim_matches('"')
        .trim_matches('\'');
    if key.is_empty() {
        return None;
    }
    Some((key, field[separator + separator_len..].trim()))
}

fn parse_pseudo_value(value: &str) -> Value {
    let value = value.trim();
    if let Some(unwrapped) = strip_sentinel_string(value) {
        return Value::String(unwrapped.to_string());
    }
    if let Ok(parsed) = serde_json::from_str::<Value>(value) {
        return parsed;
    }
    let pseudo_object = value
        .starts_with('{')
        .then(|| parse_pseudo_argument_object(value))
        .flatten();
    if let Some(arguments) = pseudo_object {
        return Value::Object(arguments);
    }
    let unquoted = value
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();
    Value::String(unquoted)
}

fn strip_sentinel_string(value: &str) -> Option<&str> {
    let double = "<|\"|>";
    let single = "<|'|>";
    value
        .strip_prefix(double)
        .and_then(|inner| inner.strip_suffix(double))
        .or_else(|| {
            value
                .strip_prefix(single)
                .and_then(|inner| inner.strip_suffix(single))
        })
}

fn first_balanced_object(content: &str) -> Option<String> {
    let start = content.find('{')?;
    let end = balanced_substring_end(content.as_bytes(), start, b'{', b'}')?;
    Some(content[start..=end].to_string())
}

fn parse_parenthesized_tool_call(content: &str) -> Option<Value> {
    let open_paren = content.find('(')?;
    let name = trailing_tool_name(&content[..open_paren])?;
    let after_open = content[open_paren + 1..].trim_start();
    let json_text = first_balanced_object(after_open)?;
    let after_json = after_open[json_text.len()..].trim_start();
    if !after_json.starts_with(')') {
        return None;
    }
    let arguments = serde_json::from_str::<Value>(&json_text).ok()?;
    Some(json!({ "name": name, "arguments": arguments }))
}

fn trailing_tool_name(content: &str) -> Option<&str> {
    let name = content
        .trim()
        .rsplit(|character: char| {
            !character.is_ascii_alphanumeric() && character != '_' && character != '-'
        })
        .next()?
        .trim();
    (!name.is_empty()).then_some(name)
}

fn raw_tool_calls_from_value(value: &Value) -> Option<Vec<&Value>> {
    match value {
        Value::Array(entries) => Some(entries.iter().collect()),
        Value::Object(object) => object
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|entries| entries.iter().collect())
            .or_else(|| Some(vec![value])),
        _ => None,
    }
}

fn parse_one_tool_call(
    value: &Value,
    allowed_tools: &BTreeSet<&str>,
) -> Result<ParsedToolCall, ToolCallParseError> {
    let Some((name, arguments_value)) = extract_tool_name_and_arguments(value) else {
        return Err(ToolCallParseError::Malformed);
    };
    let name = resolve_allowed_tool_name(name, allowed_tools)?;
    let Some(arguments) = normalize_tool_arguments(arguments_value) else {
        return Err(ToolCallParseError::InvalidArguments);
    };
    Ok(ParsedToolCall { name, arguments })
}

fn resolve_allowed_tool_name(
    raw_name: &str,
    allowed_tools: &BTreeSet<&str>,
) -> Result<String, ToolCallParseError> {
    if allowed_tools.is_empty() {
        return Ok(raw_name.to_string());
    }
    if allowed_tools.contains(raw_name) {
        return Ok(raw_name.to_string());
    }

    let Some(matched) = high_confidence_name_match(raw_name, allowed_tools.iter().copied()) else {
        return Err(ToolCallParseError::UnknownTool);
    };
    Ok(matched.to_string())
}

fn high_confidence_name_match<'a>(
    raw_name: &str,
    candidates: impl Iterator<Item = &'a str>,
) -> Option<&'a str> {
    let raw_norm = normalized_identifier(raw_name);
    if raw_norm.len() < 4 {
        return None;
    }

    let mut matches = candidates
        .filter_map(|candidate| {
            let candidate_norm = normalized_identifier(candidate);
            name_match_score(raw_name, &raw_norm, candidate, &candidate_norm)
                .map(|score| (score, candidate))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(right.1)));

    let [(score, candidate), rest @ ..] = matches.as_slice() else {
        return None;
    };
    let runner_up = rest.first().map(|(score, _)| *score).unwrap_or(0);
    (*score >= 90 && *score >= runner_up + 12).then_some(*candidate)
}

fn name_match_score(
    raw_name: &str,
    raw_norm: &str,
    candidate: &str,
    candidate_norm: &str,
) -> Option<u16> {
    if raw_norm == candidate_norm {
        return Some(120);
    }
    let raw_tokens = identifier_tokens(raw_name);
    let candidate_tokens = identifier_tokens(candidate);
    if !candidate_tokens.is_empty()
        && token_suffix_matches(&raw_tokens, &candidate_tokens)
        && candidate_norm.len() >= 6
    {
        return Some(105);
    }
    close_identifier_match(raw_norm, candidate_norm).then_some(95)
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
    if left.len().abs_diff(right.len()) > 2 || left.len().min(right.len()) < 6 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescues_qwen_xml_tool_call() {
        let calls = rescue_tool_call_from_text(
            r#"<function=read_file><parameter=path>README.md</parameter></function>"#,
            &[],
        )
        .unwrap();

        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn rescues_parenthesized_tool_call() {
        let calls = rescue_tool_call_from_text(r#"read_file({"path":"README.md"})"#, &[]).unwrap();

        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn rejects_unknown_tool_when_catalog_is_present() {
        let allowed_tools = vec!["read_file".to_string()];
        let error = rescue_tool_call_from_text(
            r#"{"name":"write_file","arguments":{"path":"README.md"}}"#,
            &allowed_tools,
        )
        .unwrap_err();

        assert_eq!(error, ToolCallParseError::UnknownTool);
    }

    #[test]
    fn canonicalizes_tool_name_from_allowed_catalog() {
        let allowed_tools = vec!["web_search".to_string()];
        let calls = rescue_tool_call_from_text(
            r#"{"name":"web.search","arguments":{"query":"Mesh-LLM"}}"#,
            &allowed_tools,
        )
        .unwrap();

        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "Mesh-LLM");
    }

    #[test]
    fn canonicalizes_namespaced_tool_name_only_when_unique() {
        let allowed_tools = vec!["read_file".to_string()];
        let calls = rescue_tool_call_from_text(
            r#"{"name":"filesystem.read_file","arguments":{"path":"README.md"}}"#,
            &allowed_tools,
        )
        .unwrap();

        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn ambiguous_tool_name_correction_fails_closed() {
        let allowed_tools = vec!["read_file".to_string(), "write_file".to_string()];
        let error = rescue_tool_call_from_text(
            r#"{"name":"filesystem.file","arguments":{"path":"README.md"}}"#,
            &allowed_tools,
        )
        .unwrap_err();

        assert_eq!(error, ToolCallParseError::UnknownTool);
    }

    #[test]
    fn rescues_sentinel_tool_call() {
        let calls = rescue_tool_call_from_text(
            r#"<|tool_call>call:web_search{query:<|"|>Mesh-LLM #808 Harden OpenClaw/MoA Telegram timeouts<|"|>}<tool_call|>"#,
            &[],
        )
        .unwrap();

        assert_eq!(calls[0].name, "web_search");
        assert_eq!(
            calls[0].arguments["query"],
            "Mesh-LLM #808 Harden OpenClaw/MoA Telegram timeouts"
        );
    }

    #[test]
    fn rescues_fully_delimited_sentinel_tool_call() {
        let calls = rescue_tool_call_from_text(
            r#"<|tool_call>call:web_search{query:<|"|>Mesh-LLM #808<|"|>}<|tool_call|>"#,
            &[],
        )
        .unwrap();

        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "Mesh-LLM #808");
    }

    #[test]
    fn rescues_bracketed_arrow_tool_call() {
        let calls = rescue_tool_call_from_text(
            "[TOOL_CALL]\n{tool => \"read_file\", args => {path: \"README.md\"}}\n[/TOOL_CALL]",
            &[],
        )
        .unwrap();

        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn bracketed_arrow_tool_call_rejects_unknown_tool_when_catalog_is_present() {
        let allowed_tools = vec!["read_file".to_string()];
        let error = rescue_tool_call_from_text(
            "[TOOL_CALL]\n{tool => \"filesystem_list_allowed_directories\", args => {}}\n[/TOOL_CALL]",
            &allowed_tools,
        )
        .unwrap_err();

        assert_eq!(error, ToolCallParseError::UnknownTool);
    }

    #[test]
    fn explanatory_bracketed_tool_call_text_stays_malformed() {
        let error = rescue_tool_call_from_text(
            "Use this shape: [TOOL_CALL]\n{tool => \"read_file\", args => {path: \"README.md\"}}\n[/TOOL_CALL]",
            &[],
        )
        .unwrap_err();

        assert_eq!(error, ToolCallParseError::Malformed);
    }

    #[test]
    fn sentinel_tool_call_rejects_unknown_tool_when_catalog_is_present() {
        let allowed_tools = vec!["web_fetch".to_string()];
        let error = rescue_tool_call_from_text(
            r#"<|tool_call>call:web_search{query:<|"|>Mesh-LLM<|"|>}<tool_call|>"#,
            &allowed_tools,
        )
        .unwrap_err();

        assert_eq!(error, ToolCallParseError::UnknownTool);
    }

    #[test]
    fn explanatory_sentinel_tool_call_text_stays_malformed() {
        let error = rescue_tool_call_from_text(
            r#"Use this shape when calling a tool: <|tool_call>call:web_search{query:<|"|>Mesh-LLM<|"|>}<tool_call|>"#,
            &[],
        )
        .unwrap_err();

        assert_eq!(error, ToolCallParseError::Malformed);
    }
}
