use std::{
    fs,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::cli::{DEFAULT_RUN_MAX_NEW_TOKENS, RunArgs};

#[derive(Debug, Serialize)]
pub struct OpenAiDriverReport {
    pub base_url: String,
    pub model: String,
    pub endpoint: String,
    pub prompt_count: usize,
    pub max_new_tokens: usize,
    pub request_timeout_secs: u64,
    pub corpus: Option<PathBuf>,
    pub summary: OpenAiDriverSummary,
    pub results: Vec<OpenAiDriverResult>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiDriverSummary {
    pub completion_tokens_total: u64,
    pub prompt_tokens_total: u64,
    pub total_tokens_total: u64,
    pub elapsed_ms_total: f64,
    pub elapsed_ms_mean: f64,
    pub elapsed_ms_p50: f64,
    pub elapsed_ms_p95: f64,
    pub elapsed_ms_p99: f64,
    pub generated_tokens_per_second: f64,
    pub request_count: usize,
    pub completed_request_count: usize,
    pub error_count: usize,
}

#[derive(Debug, Serialize)]
pub struct OpenAiDriverResult {
    pub sequence: usize,
    pub prompt_id: Option<String>,
    pub category: Option<String>,
    pub session_id: String,
    pub prompt_chars: usize,
    pub elapsed_ms: f64,
    pub completion_tokens: Option<u64>,
    pub prompt_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    pub output_chars: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
struct PromptCase {
    index: usize,
    prompt_id: Option<String>,
    category: Option<String>,
    session_id: Option<String>,
    prompt: String,
    messages: Option<Value>,
}

pub fn run_remote_openai_driver(
    args: &RunArgs,
    base_url: String,
    model: String,
) -> Result<OpenAiDriverReport> {
    let prompts = prompt_cases(args)?;
    if prompts.is_empty() {
        bail!("OpenAI prompt corpus is empty");
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(args.openai_request_timeout_secs))
        .build()
        .context("failed to build OpenAI benchmark HTTP client")?;
    let max_new_tokens = args.max_new_tokens.unwrap_or(DEFAULT_RUN_MAX_NEW_TOKENS);
    let mut results = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        results.push(run_openai_case(
            &client,
            &base_url,
            &model,
            max_new_tokens,
            prompt,
        ));
    }
    Ok(OpenAiDriverReport {
        base_url,
        model,
        endpoint: "/v1/chat/completions".to_string(),
        prompt_count: prompts.len(),
        max_new_tokens,
        request_timeout_secs: args.openai_request_timeout_secs,
        corpus: args.prompt_corpus.clone(),
        summary: summarize(&results),
        results,
    })
}

fn run_openai_case(
    client: &Client,
    base_url: &str,
    model: &str,
    max_new_tokens: usize,
    prompt: &PromptCase,
) -> OpenAiDriverResult {
    let session_id = prompt
        .session_id
        .clone()
        .unwrap_or_else(|| format!("skippy-openai-{}", prompt.index));
    let started = Instant::now();
    let response = client
        .post(format!(
            "{}/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .json(&request_body(model, max_new_tokens, prompt, &session_id))
        .send();
    match response {
        Ok(response) => parse_response(response, started, prompt, session_id),
        Err(error) => error_result(prompt, session_id, started, error.to_string()),
    }
}

fn request_body(
    model: &str,
    max_new_tokens: usize,
    prompt: &PromptCase,
    session_id: &str,
) -> Value {
    let messages = prompt.messages.clone().unwrap_or_else(|| {
        json!([
            {
                "role": "user",
                "content": prompt.prompt,
            }
        ])
    });
    let mut object = Map::new();
    object.insert("model".to_string(), Value::String(model.to_string()));
    object.insert("messages".to_string(), messages);
    object.insert("max_tokens".to_string(), json!(max_new_tokens));
    object.insert("stream".to_string(), json!(false));
    object.insert("user".to_string(), Value::String(session_id.to_string()));
    Value::Object(object)
}

fn parse_response(
    response: reqwest::blocking::Response,
    started: Instant,
    prompt: &PromptCase,
    session_id: String,
) -> OpenAiDriverResult {
    let status = response.status();
    let body = response.json::<Value>();
    match body {
        Ok(value) if status.is_success() => success_result(prompt, session_id, started, &value),
        Ok(value) => error_result(
            prompt,
            session_id,
            started,
            format!("chat completions request failed with status {status}: {value}"),
        ),
        Err(error) => error_result(
            prompt,
            session_id,
            started,
            format!("failed to parse chat completions JSON response: {error}"),
        ),
    }
}

fn success_result(
    prompt: &PromptCase,
    session_id: String,
    started: Instant,
    value: &Value,
) -> OpenAiDriverResult {
    let usage = value.get("usage");
    OpenAiDriverResult {
        sequence: prompt.index,
        prompt_id: prompt.prompt_id.clone(),
        category: prompt.category.clone(),
        session_id,
        prompt_chars: prompt.prompt.chars().count(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        completion_tokens: usage.and_then(|usage| usage_u64(usage, "completion_tokens")),
        prompt_tokens: usage.and_then(|usage| usage_u64(usage, "prompt_tokens")),
        total_tokens: usage.and_then(|usage| usage_u64(usage, "total_tokens")),
        finish_reason: value
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        output_chars: value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::chars)
            .map(Iterator::count)
            .unwrap_or_default(),
        error: None,
    }
}

fn error_result(
    prompt: &PromptCase,
    session_id: String,
    started: Instant,
    error: String,
) -> OpenAiDriverResult {
    OpenAiDriverResult {
        sequence: prompt.index,
        prompt_id: prompt.prompt_id.clone(),
        category: prompt.category.clone(),
        session_id,
        prompt_chars: prompt.prompt.chars().count(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        completion_tokens: None,
        prompt_tokens: None,
        total_tokens: None,
        finish_reason: None,
        output_chars: 0,
        error: Some(error),
    }
}

fn summarize(results: &[OpenAiDriverResult]) -> OpenAiDriverSummary {
    let completion_tokens_total = sum_tokens(results, |result| result.completion_tokens);
    let prompt_tokens_total = sum_tokens(results, |result| result.prompt_tokens);
    let total_tokens_total = sum_tokens(results, |result| result.total_tokens);
    let elapsed_ms_total = results.iter().map(|result| result.elapsed_ms).sum::<f64>();
    OpenAiDriverSummary {
        completion_tokens_total,
        prompt_tokens_total,
        total_tokens_total,
        elapsed_ms_total,
        elapsed_ms_mean: mean_elapsed_ms(results),
        elapsed_ms_p50: percentile_elapsed_ms(results, 0.50),
        elapsed_ms_p95: percentile_elapsed_ms(results, 0.95),
        elapsed_ms_p99: percentile_elapsed_ms(results, 0.99),
        generated_tokens_per_second: tokens_per_second(completion_tokens_total, elapsed_ms_total),
        request_count: results.len(),
        completed_request_count: results
            .iter()
            .filter(|result| result.error.is_none())
            .count(),
        error_count: results
            .iter()
            .filter(|result| result.error.is_some())
            .count(),
    }
}

fn sum_tokens(
    results: &[OpenAiDriverResult],
    field: impl Fn(&OpenAiDriverResult) -> Option<u64>,
) -> u64 {
    results.iter().filter_map(field).sum()
}

fn mean_elapsed_ms(results: &[OpenAiDriverResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    results.iter().map(|result| result.elapsed_ms).sum::<f64>() / results.len() as f64
}

fn percentile_elapsed_ms(results: &[OpenAiDriverResult], percentile: f64) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let mut values = results
        .iter()
        .map(|result| result.elapsed_ms)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let rank = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[rank.min(values.len() - 1)]
}

fn tokens_per_second(tokens: u64, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        return 0.0;
    }
    tokens as f64 / (elapsed_ms / 1000.0)
}

fn usage_u64(usage: &Value, field: &str) -> Option<u64> {
    usage.get(field).and_then(Value::as_u64)
}

fn prompt_cases(args: &RunArgs) -> Result<Vec<PromptCase>> {
    let mut prompts = match args.prompt_corpus.as_ref() {
        Some(path) => prompt_cases_from_file(path)?,
        None => vec![PromptCase {
            index: 0,
            prompt_id: None,
            category: None,
            session_id: None,
            prompt: args.prompt.clone(),
            messages: None,
        }],
    };
    if let Some(limit) = args.prompt_limit {
        prompts.truncate(limit);
    }
    for (index, prompt) in prompts.iter_mut().enumerate() {
        prompt.index = index;
    }
    Ok(prompts)
}

fn prompt_cases_from_file(path: &PathBuf) -> Result<Vec<PromptCase>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.lines()
        .enumerate()
        .filter_map(|(line_index, line)| {
            let line = line.trim();
            (!line.is_empty()).then_some((line_index, line))
        })
        .map(|(line_index, line)| {
            prompt_case_from_line(line_index, line).with_context(|| {
                format!(
                    "read prompt corpus line {} in {}",
                    line_index + 1,
                    path.display()
                )
            })
        })
        .collect()
}

fn prompt_case_from_line(index: usize, line: &str) -> Result<PromptCase> {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return Ok(PromptCase {
            index,
            prompt_id: None,
            category: None,
            session_id: None,
            prompt: line.to_string(),
            messages: None,
        });
    };
    prompt_case_from_value(index, &value)
}

fn prompt_case_from_value(index: usize, value: &Value) -> Result<PromptCase> {
    let prompt_id = value
        .get("id")
        .or_else(|| value.get("prompt_id"))
        .and_then(value_to_string);
    let category = value
        .get("category")
        .or_else(|| value.get("family"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let session_id = value
        .get("session_group")
        .or_else(|| value.get("session_id"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    if let Some(messages) = value.get("messages").and_then(Value::as_array) {
        return Ok(PromptCase {
            index,
            prompt_id,
            category,
            session_id,
            prompt: prompt_text_from_messages(messages),
            messages: Some(Value::Array(messages.clone())),
        });
    }
    Ok(PromptCase {
        index,
        prompt_id,
        category,
        session_id,
        prompt: prompt_text_from_value(value)?,
        messages: None,
    })
}

fn value_to_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .or_else(|| value.as_i64().map(|id| id.to_string()))
}

fn prompt_text_from_messages(messages: &[Value]) -> String {
    messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

fn prompt_text_from_value(value: &Value) -> Result<String> {
    if let Some(prompt) = value.get("prompt").and_then(Value::as_str) {
        return Ok(prompt.to_string());
    }
    let turns = value
        .get("turns")
        .and_then(Value::as_array)
        .context("JSONL row must include prompt, turns, or messages")?;
    turns
        .iter()
        .find_map(Value::as_str)
        .map(ToOwned::to_owned)
        .context("turns did not contain a string prompt")
}

pub fn wait_openai_ready(base_url: &str, timeout_secs: u64) -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .context("failed to build OpenAI readiness client")?;
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(timeout_secs) {
        match client
            .get(format!("{}/models", base_url.trim_end_matches('/')))
            .send()
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(_) | Err(_) => thread::sleep(Duration::from_millis(250)),
        }
    }
    bail!("OpenAI endpoint did not become ready at {base_url}");
}
