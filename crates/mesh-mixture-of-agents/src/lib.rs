//! Mixture-of-Agents (MoA) gateway.
//!
//! Fan out to N heterogeneous LLM backends in parallel, arbitrate their
//! outputs with deterministic logic, and return one coherent OpenAI-
//! compatible response.  The client thinks it talks to one model.
//!
//! Transport is abstracted behind the [`ModelBackend`] trait (see
//! [`backend`]). The default [`HttpBackend`] talks to any
//! OpenAI-compatible HTTP endpoint and is suitable for standalone/test
//! use. The mesh host-runtime provides mesh-native backends that
//! dispatch local models via direct HTTP and remote models via QUIC
//! tunnel.
//!
//! ```text
//! Agent / Goose / pi
//!     │
//!     │  POST /v1/chat/completions { "model": "mesh" }
//!     ▼
//!  MoA Gateway  (handle_turn)
//!   ├─ session / context packing (role-shaped)        — context::*
//!   ├─ parallel fan-out via ModelBackend              — fanout::gather_workers_incremental
//!   ├─ incremental gathering with early-exit          — arbiter::try_early_decision
//!   ├─ deterministic arbiter (code, not models)       — arbiter::arbitrate
//!   └─ reducer escalation only on genuine conflict    — reducer::hedged_reducer_call
//! ```
//!
//! Modules:
//! - [`backend`] — `ModelBackend` trait, `HttpBackend`, `SamplingParams`,
//!   `ModelEntry`
//! - [`reducer`] — reducer candidate ordering, hedged ladder
//! - [`fanout`] — incremental worker gathering with early-exit
//! - [`arbiter`] — deterministic arbitration + early-exit consensus
//! - [`normalize`] — 3-tier dirty-output parsing
//! - [`session`] — canonical transcript + turn classification
//! - [`context`] — role-shaped context packing
//! - [`worker`] — role assignment, think-tag stripping

pub mod arbiter;
pub mod backend;
pub mod context;
mod fanout;
pub mod normalize;
mod reducer;
pub mod session;
mod tool_guard;
pub mod worker;

pub use backend::{HttpBackend, ModelBackend, ModelEntry, SamplingParams, apply_enable_thinking};
pub(crate) use tool_guard::enforce_tool_call_contract;

use backend::call_backend;
use fanout::{GraceMode, gather_workers_incremental};
use mesh_llm_guardrails::{
    extract_tool_name_and_arguments, normalize_tool_arguments, sanitize_tool_arguments_for_tool,
    tool_arguments_wire_string,
};
use normalize::WorkerOutput;
use reducer::{hedged_reducer_call, reducer_candidates};
use serde_json::{Value, json};
use session::Session;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use worker::WorkerRole;
pub use worker::{strip_thinking, truncate_chars};

const SAME_TOOL_FORCE_ANSWER_THRESHOLD: usize = 3;
const TOOL_BUDGET_FORCE_ANSWER_THRESHOLD: usize = 6;

/// The virtual model name that triggers MoA routing.
pub const VIRTUAL_MODEL_NAME: &str = "mesh";

// ─── Configuration ───────────────────────────────────────────────────

/// Gateway configuration.
pub struct GatewayConfig {
    /// Available backends.  Models reference these by index.
    pub backends: Vec<std::sync::Arc<dyn ModelBackend>>,
    /// Available models for fan-out.
    pub models: Vec<ModelEntry>,
    /// Per-worker timeout.
    pub worker_timeout: Duration,
    /// Per-candidate wait before hedging a second reducer candidate. When the
    /// primary candidate is slow (e.g. cold KV) we don't want to wait the full
    /// reducer_timeout before kicking off candidate 2 — start the next one
    /// after hedge_delay and race them. Cost: up to 2× tokens for the rare
    /// slow case; zero cost on the happy path (candidate 1 returns first).
    pub hedge_delay: Duration,
    /// Reducer timeout.
    pub reducer_timeout: Duration,
    /// Chat-only grace: after this long since dispatch, if a single answer
    /// (conf >= 0.5) is in, accept it instead of waiting for consensus.
    /// Disabled for tool turns. Zero disables entirely.
    pub first_answer_grace: Duration,
    /// Override for whether reasoning workers should think. Propagated to
    /// every worker and the reducer as `chat_template_kwargs.enable_thinking`
    /// (and `reasoning_effort: "none"` when disabled).
    ///
    /// `None` (the default) leaves each model's default behavior alone —
    /// existing callers see no behavior change. The MoA HTTP gateway
    /// populates this from the caller's `reasoning_effort` / `enable_thinking`
    /// / `reasoning.enabled` knobs so MoA users get a single switch.
    pub enable_thinking: Option<bool>,
}

// ─── Turn result ─────────────────────────────────────────────────────

/// Which gateway path produced this turn's response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnKind {
    /// Fan-out path: arbiter decided from full worker outputs.
    Fanout,
    /// Fan-out path with early-exit consensus before all workers returned.
    EarlyExit,
    /// Explicit or safely inferred tool intent returned a native tool call directly.
    DirectTool,
    /// Tool-result turn: skipped fan-out, went straight to reducer.
    ToolResult,
    /// All workers failed and no reducer recovery happened.
    Failed,
}

impl TurnKind {
    /// Lowercase header-friendly label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fanout => "fanout",
            Self::EarlyExit => "early-exit",
            Self::DirectTool => "direct-tool",
            Self::ToolResult => "tool-result",
            Self::Failed => "failed",
        }
    }
}

/// What the gateway returns for a single turn.
#[derive(Debug)]
pub struct TurnResult {
    /// OpenAI chat.completion response body.
    pub response_body: Value,
    /// Per-worker details for observability.
    pub worker_summaries: Vec<WorkerSummary>,
    /// Whether the reducer was invoked.
    pub reducer_used: bool,
    /// How many reducer candidates were spawned (0 if reducer didn't run,
    /// 1 on the happy reducer path, ≥2 if the hedge fired or a fast-fail
    /// cascaded to the next candidate).
    pub reducer_attempts: u32,
    /// Which gateway path produced this response.
    pub turn_kind: TurnKind,
    /// Wall-clock time for this turn.
    pub elapsed_ms: u64,
}

#[derive(Debug)]
pub struct WorkerSummary {
    pub model: String,
    pub role: WorkerRole,
    pub succeeded: bool,
    pub elapsed_ms: u64,
    pub output_kind: Option<normalize::OutputKind>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone)]
struct ForcedToolChoice {
    name: String,
    fallback_arguments: Value,
}

struct DecisionResolution<'a> {
    session: &'a Session,
    decision: arbiter::Decision,
    outputs: &'a [WorkerOutput],
    has_tools: bool,
    selected_tool_names: &'a [String],
    forced_tool: Option<&'a ForcedToolChoice>,
    allowed_tools: &'a [String],
    prompt_tool_profiles: &'a [PromptToolProfile],
}

// ─── Gateway entry point ─────────────────────────────────────────────

/// Process one MoA turn.
///
/// Stateless per request.  Multi-turn state is managed by the agent client
/// which sends the full conversation on each request.
pub async fn handle_turn(config: &GatewayConfig, body: &Value) -> TurnResult {
    let start = Instant::now();

    let mut session = Session::new();
    let incoming_messages = incoming_chat_messages(body);
    let tools = body.get("tools").cloned();
    let has_native_tools = tools
        .as_ref()
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    session.ingest(&incoming_messages, &tools);
    let prompt_tool_profiles = if has_native_tools {
        Vec::new()
    } else {
        prompt_declared_tool_profiles(&session)
    };
    let allowed_tools = declared_tool_names(&session, &prompt_tool_profiles);
    let has_tools = !allowed_tools.is_empty();

    let turn_type = session.classify_turn();
    let forced_tool = forced_tool_choice(body, &session, &tools, &allowed_tools);
    tracing::info!(
        "moa: turn={:?}, {} models, tools={}, native_tools={}",
        turn_type,
        config.models.len(),
        has_tools,
        has_native_tools,
    );

    match turn_type {
        session::TurnType::ToolResult => {
            handle_tool_result(
                config,
                &session,
                has_tools,
                &allowed_tools,
                &prompt_tool_profiles,
                start,
            )
            .await
        }
        session::TurnType::Fresh => {
            handle_query(
                config,
                &session,
                has_tools,
                &allowed_tools,
                &prompt_tool_profiles,
                forced_tool.as_ref(),
                start,
            )
            .await
        }
    }
}

fn incoming_chat_messages(body: &Value) -> Vec<Value> {
    let mut messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    let completion_prompt = messages
        .is_empty()
        .then(|| completion_prompt_text(body.get("prompt")))
        .flatten();
    if let Some(prompt) = completion_prompt {
        messages.push(json!({"role": "user", "content": prompt}));
    }

    let missing_system = !messages
        .iter()
        .any(|message| message_role(message) == "system");
    let system_prompt = missing_system
        .then(|| top_level_system_prompt(body))
        .flatten();
    if let Some(system) = system_prompt {
        messages.insert(0, json!({"role": "system", "content": system}));
    }

    messages
}

fn top_level_system_prompt(body: &Value) -> Option<String> {
    ["systemPrompt", "system_prompt", "system"]
        .iter()
        .find_map(|key| body.get(*key).and_then(Value::as_str))
        .map(str::to_string)
        .filter(|text| !text.trim().is_empty())
}

fn completion_prompt_text(prompt: Option<&Value>) -> Option<String> {
    match prompt? {
        Value::String(text) => Some(text.clone()),
        Value::Array(values) => {
            let texts: Vec<&str> = values.iter().filter_map(Value::as_str).collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        _ => None,
    }
    .filter(|text| !text.trim().is_empty())
}

// ─── Query handling ──────────────────────────────────────────────────

async fn handle_query(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
    forced_tool: Option<&ForcedToolChoice>,
    start: Instant,
) -> TurnResult {
    let assignments = worker::assign_roles(&config.models);
    let grace_mode = grace_mode_for_turn(session, has_tools, prompt_tool_profiles);
    let continuation_tool = prior_shell_command_tool_choice(
        session,
        &tools_value(session),
        allowed_tools,
        prompt_tool_profiles,
    );
    let query_uses_tools = forced_tool.is_some()
        || continuation_tool.is_some()
        || matches!(grace_mode, GraceMode::Tool);
    let selected_tool_names = if let Some(tool) = forced_tool {
        vec![tool.name.clone()]
    } else if let Some(tool) = continuation_tool.as_ref() {
        vec![tool.name.clone()]
    } else if query_uses_tools {
        selected_tool_names_for_turn(session, allowed_tools, prompt_tool_profiles)
    } else {
        Vec::new()
    };
    let intent_required_tool = if forced_tool.is_none() && query_uses_tools {
        required_tool_choice_for_intent(session, &tools_value(session), &selected_tool_names)
    } else {
        None
    };
    let required_tool = forced_tool
        .or(continuation_tool.as_ref())
        .or(intent_required_tool.as_ref());
    if let Some(tool) = required_tool.filter(|_| has_tools) {
        tracing::info!(
            "moa: direct tool call for explicit {} intent",
            tool.name.as_str()
        );
        return TurnResult {
            response_body: tool_call_response(&tool.name, &tool.fallback_arguments),
            worker_summaries: Vec::new(),
            reducer_used: false,
            reducer_attempts: 0,
            turn_kind: TurnKind::DirectTool,
            elapsed_ms: start.elapsed().as_millis() as u64,
        };
    }

    tracing::info!(
        "moa: dispatching to {} workers: [{}]",
        assignments.len(),
        assignments
            .iter()
            .map(|a| format!("{}({})", a.model_name, a.role.label()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut join_set = tokio::task::JoinSet::new();
    let mut dispatched: Vec<fanout::DispatchedWorker> = Vec::with_capacity(assignments.len());

    let enable_thinking = config.enable_thinking;
    for assignment in &assignments {
        let packed = context::pack_for_worker_selected(
            session,
            assignment.role,
            query_uses_tools,
            &selected_tool_names,
        );
        let model_name = assignment.model_name.clone();
        let role = assignment.role;
        let backend = config.backends[assignment.backend_index].clone();
        let timeout = config.worker_timeout;

        dispatched.push(fanout::DispatchedWorker {
            model: model_name.clone(),
            role,
        });

        join_set.spawn(async move {
            let t0 = Instant::now();
            let result = call_backend(
                &*backend,
                &model_name,
                &packed.messages,
                packed.tools.as_ref(),
                packed.max_tokens,
                timeout,
                SamplingParams::worker().with_thinking(enable_thinking),
            )
            .await;
            let elapsed = t0.elapsed().as_millis() as u64;
            (model_name, role, result, elapsed)
        });
    }

    let (outputs, summaries, early_decision) = gather_workers_incremental(
        &mut join_set,
        &dispatched,
        query_uses_tools,
        allowed_tools,
        session.tools(),
        config.first_answer_grace,
        grace_mode,
    )
    .await;

    if outputs.is_empty() {
        return TurnResult {
            response_body: error_response("All MoA workers failed", MOA_ERR_ALL_WORKERS_FAILED),
            worker_summaries: summaries,
            reducer_used: false,
            reducer_attempts: 0,
            turn_kind: TurnKind::Failed,
            elapsed_ms: start.elapsed().as_millis() as u64,
        };
    }

    // Capture whether we took the early-exit path BEFORE we resolve the
    // decision: the arbiter never runs when early_decision is Some.
    let took_early_exit = early_decision.is_some();
    let decision = early_decision.unwrap_or_else(|| arbiter::arbitrate(&outputs, query_uses_tools));
    let (response_body, reducer_used, reducer_attempts) = resolve_decision(
        config,
        DecisionResolution {
            session,
            decision,
            outputs: &outputs,
            has_tools: query_uses_tools,
            selected_tool_names: &selected_tool_names,
            forced_tool: required_tool,
            allowed_tools,
            prompt_tool_profiles,
        },
    )
    .await;

    // turn_kind is "early-exit" only when we genuinely short-circuited via
    // consensus AND didn't need to escalate to the reducer. A reducer-
    // escalated turn is "fanout" even if early_decision was set, because
    // we still did the expensive serial call.
    let turn_kind = if took_early_exit && !reducer_used {
        TurnKind::EarlyExit
    } else {
        TurnKind::Fanout
    };

    TurnResult {
        response_body,
        worker_summaries: summaries,
        reducer_used,
        reducer_attempts,
        turn_kind,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

fn grace_mode_for_turn(
    session: &Session,
    has_tools: bool,
    prompt_tool_profiles: &[PromptToolProfile],
) -> GraceMode {
    if !has_tools {
        return GraceMode::Answer;
    }
    if looks_like_tool_intent(session, prompt_tool_profiles) {
        GraceMode::Tool
    } else {
        GraceMode::Answer
    }
}

fn looks_like_tool_intent(session: &Session, prompt_tool_profiles: &[PromptToolProfile]) -> bool {
    let text = session.last_user_text().to_ascii_lowercase();
    if contains_any(
        &text,
        &[
            "no tool",
            "without tool",
            "do not use tool",
            "don't use tool",
        ],
    ) {
        return false;
    }

    if contains_any(
        &text,
        &[
            "use a tool",
            "using a tool",
            "call a tool",
            "use the tool",
            "call the tool",
        ],
    ) {
        return true;
    }

    let available = session.tool_names();
    if !explicitly_requested_tool_names(&available, &text).is_empty() {
        return true;
    }
    if !command_intent_tool_names(session.tools(), &available, &text, prompt_tool_profiles)
        .is_empty()
    {
        return true;
    }
    if !schema_matched_tool_names(session.tools(), &available, &text, prompt_tool_profiles)
        .is_empty()
    {
        return true;
    }
    if looks_like_live_status_followup_tool_intent(session, &text) {
        return true;
    }
    extract_shell_command_block(&text)
        .and_then(|command| {
            command_tool_choice_from_command(
                &command,
                session.tools(),
                &available,
                prompt_tool_profiles,
            )
        })
        .is_some()
}

fn looks_like_live_status_followup_tool_intent(session: &Session, text: &str) -> bool {
    if recent_tool_chain_names(session).is_empty() {
        return false;
    }

    contains_any(
        text,
        &[
            "now",
            "current",
            "currently",
            "latest",
            "status",
            "still",
            "updated",
            "ci",
            "check",
            "checks",
            "green",
            "feedback",
            "comment",
            "comments",
            "review",
        ],
    )
}

fn prior_shell_command_tool_choice(
    session: &Session,
    tools: &Option<Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<ForcedToolChoice> {
    if !looks_like_prior_tool_confirmation(&session.last_user_text()) {
        return None;
    }
    let assistant = previous_assistant_text(session)?;
    let command = extract_shell_command_block(&assistant)?;
    command_tool_choice_from_command(
        &command,
        tools.as_ref(),
        allowed_tools,
        prompt_tool_profiles,
    )
}

fn looks_like_prior_tool_confirmation(text: &str) -> bool {
    let text = latest_user_request_tail(text).to_ascii_lowercase();
    if contains_any(
        &text,
        &[
            "don't do that",
            "do not do that",
            "dont do that",
            "don't run",
            "do not run",
            "dont run",
        ],
    ) || contains_word(&text, "no")
    {
        return false;
    }

    contains_any(
        &text,
        &[
            "do that",
            "do it",
            "go ahead",
            "please do",
            "run it",
            "run that",
            "execute it",
            "execute that",
        ],
    ) || ["ok", "okay", "yes", "yeah", "yep"]
        .iter()
        .any(|word| contains_word(&text, word))
}

fn contains_word(text: &str, word: &str) -> bool {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|token| token == word)
}

fn latest_user_request_tail(text: &str) -> &str {
    text.rsplit_once("\n\n")
        .map(|(_, tail)| tail.trim())
        .filter(|tail| !tail.is_empty())
        .unwrap_or_else(|| text.trim())
}

fn previous_assistant_text(session: &Session) -> Option<String> {
    let mut seen_latest_user = false;
    for message in session.messages().iter().rev() {
        match message_role(message) {
            "user" if !seen_latest_user => seen_latest_user = true,
            "assistant" if seen_latest_user => {
                return message_text_content(message).filter(|text| !text.trim().is_empty());
            }
            _ => {}
        }
    }
    None
}

fn message_text_content(message: &Value) -> Option<String> {
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    let parts = message.get("content").and_then(Value::as_array)?;
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|part| {
            if part.get("type").and_then(Value::as_str) == Some("text") {
                part.get("text").and_then(Value::as_str)
            } else {
                None
            }
        })
        .collect();
    (!texts.is_empty()).then(|| texts.join("\n"))
}

fn extract_shell_command_block(text: &str) -> Option<String> {
    let mut in_block = false;
    let mut is_shell = false;
    let mut lines = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(lang) = trimmed.strip_prefix("```") {
            if in_block {
                if is_shell {
                    let command = lines.join("\n").trim().to_string();
                    if !command.is_empty() {
                        return Some(command);
                    }
                }
                in_block = false;
                is_shell = false;
                lines.clear();
                continue;
            }
            let lang = lang.trim().to_ascii_lowercase();
            in_block = true;
            is_shell = matches!(
                lang.as_str(),
                "" | "bash" | "sh" | "shell" | "zsh" | "console"
            );
            continue;
        }
        if in_block && is_shell {
            lines.push(line);
        }
    }

    None
}

fn selected_tool_names_for_turn(
    session: &Session,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<String> {
    let available = if allowed_tools.is_empty() {
        session.tool_names()
    } else {
        allowed_tools.to_vec()
    };
    if available.is_empty() {
        return Vec::new();
    }

    let text = session.last_user_text().to_ascii_lowercase();
    let explicit = explicitly_requested_tool_names(&available, &text);
    if !explicit.is_empty() {
        return with_recent_tool_chain_names(session, &available, explicit);
    }

    let command_intent =
        command_intent_tool_names(session.tools(), &available, &text, prompt_tool_profiles);
    if !command_intent.is_empty() {
        return with_recent_tool_chain_names(session, &available, command_intent);
    }

    let selected =
        schema_matched_tool_names(session.tools(), &available, &text, prompt_tool_profiles);

    with_recent_tool_chain_names(session, &available, selected)
}

fn with_recent_tool_chain_names(
    session: &Session,
    available: &[String],
    mut selected: Vec<String>,
) -> Vec<String> {
    for tool in recent_tool_chain_names(session) {
        if available.iter().any(|available| available == &tool) && !selected.contains(&tool) {
            selected.push(tool);
        }
    }

    if selected.is_empty() && available.len() == 1 {
        available.to_vec()
    } else {
        selected
    }
}

fn explicitly_requested_tool_names(available: &[String], text: &str) -> Vec<String> {
    available
        .iter()
        .filter(|tool| tool_name_is_explicitly_requested(&tool.to_ascii_lowercase(), text))
        .cloned()
        .collect()
}

fn tool_name_is_explicitly_requested(tool: &str, text: &str) -> bool {
    if tool.len() < 3 {
        return false;
    }

    let text_tokens = lexical_tokens(text);
    let tool_tokens = lexical_tokens(&tool.replace('_', " "));
    if tool_tokens.is_empty() {
        return false;
    }

    token_sequence_matches(&text_tokens, &["use"], &tool_tokens)
        || token_sequence_matches(&text_tokens, &["use", "the"], &tool_tokens)
        || token_sequence_matches(&text_tokens, &["call"], &tool_tokens)
        || token_sequence_matches(&text_tokens, &["call", "the"], &tool_tokens)
        || token_sequence_matches(&text_tokens, &["tool"], &tool_tokens)
        || token_sequence_matches_suffix(&text_tokens, &tool_tokens, &["tool"])
        || token_sequence_matches(&text_tokens, &["or"], &tool_tokens)
        || token_sequence_matches(&text_tokens, &["and"], &tool_tokens)
}

fn token_sequence_matches(text: &[String], prefix: &[&str], target: &[String]) -> bool {
    if text.len() < prefix.len() + target.len() {
        return false;
    }
    text.windows(prefix.len() + target.len()).any(|window| {
        prefix
            .iter()
            .enumerate()
            .all(|(idx, token)| window[idx] == *token)
            && target
                .iter()
                .enumerate()
                .all(|(idx, token)| window[prefix.len() + idx] == *token)
    })
}

fn token_sequence_matches_suffix(text: &[String], target: &[String], suffix: &[&str]) -> bool {
    if text.len() < target.len() + suffix.len() {
        return false;
    }
    text.windows(target.len() + suffix.len()).any(|window| {
        target
            .iter()
            .enumerate()
            .all(|(idx, token)| window[idx] == *token)
            && suffix
                .iter()
                .enumerate()
                .all(|(idx, token)| window[target.len() + idx] == *token)
    })
}

fn tools_value(session: &Session) -> Option<Value> {
    session.tools().cloned()
}

#[derive(Debug)]
struct ToolSchemaProfile {
    name: String,
    tokens: HashSet<String>,
}

#[derive(Debug, Clone)]
struct PromptToolProfile {
    name: String,
    tokens: HashSet<String>,
}

fn declared_tool_names(
    session: &Session,
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<String> {
    let native = session.tool_names();
    if !native.is_empty() {
        native
    } else {
        prompt_tool_profiles
            .iter()
            .map(|profile| profile.name.clone())
            .collect()
    }
}

fn suppress_repeated_tool_from_selection(
    mut selected_tool_names: Vec<String>,
    declared_tool_names: &[String],
    repeated_tool: Option<&str>,
) -> Vec<String> {
    let Some(tool) = repeated_tool else {
        return selected_tool_names;
    };

    selected_tool_names.retain(|name| name != tool);
    if selected_tool_names.is_empty() {
        selected_tool_names.extend(
            declared_tool_names
                .iter()
                .filter(|name| name.as_str() != tool)
                .cloned(),
        );
    }
    selected_tool_names
}

fn prompt_declared_tool_profiles(session: &Session) -> Vec<PromptToolProfile> {
    let Some(prompt) = prompt_tool_catalog_source(session) else {
        return Vec::new();
    };
    let mut in_tooling_section = false;
    let mut profiles = Vec::new();
    let mut seen = HashSet::new();

    for line in prompt.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## ") {
            in_tooling_section = prompt_tool_heading(trimmed);
            continue;
        }
        if !in_tooling_section {
            continue;
        }
        let Some(profile) = parse_prompt_tool_line(trimmed) else {
            continue;
        };
        if seen.insert(profile.name.clone()) {
            profiles.push(profile);
        }
    }

    profiles
}

fn prompt_tool_catalog_source(session: &Session) -> Option<String> {
    if let Some(system) = session
        .system_prompt()
        .filter(|prompt| !prompt.trim().is_empty())
    {
        return Some(system);
    }

    session
        .messages()
        .first()
        .and_then(message_text_content)
        .filter(|text| !text.trim().is_empty())
}

fn prompt_tool_heading(line: &str) -> bool {
    matches!(
        line.trim(),
        "## Tooling" | "## Tools" | "## Available Tools"
    )
}

fn parse_prompt_tool_line(line: &str) -> Option<PromptToolProfile> {
    let rest = line.strip_prefix("- ")?;
    let (name, description) = rest.split_once(':')?;
    let name = name.trim();
    let description = description.trim();
    if !valid_prompt_tool_name(name) || description.is_empty() {
        return None;
    }

    let mut tokens = text_tokens(name);
    tokens.extend(text_tokens(description));
    Some(PromptToolProfile {
        name: name.to_string(),
        tokens,
    })
}

fn valid_prompt_tool_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 120 {
        return false;
    }
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

fn schema_matched_tool_names(
    tools: Option<&Value>,
    available: &[String],
    user_text: &str,
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<String> {
    let profiles = tool_text_profiles(tools, available, prompt_tool_profiles);
    if profiles.is_empty() {
        return Vec::new();
    }
    let user_tokens = text_tokens(user_text);
    if user_tokens.is_empty() {
        return Vec::new();
    }
    let explicit_tool_use = user_text_mentions_tool_use(user_text);
    let minimum_score = if explicit_tool_use { 1 } else { 2 };
    let required_margin = if explicit_tool_use { 1 } else { 2 };

    let mut scored: Vec<(String, usize)> = profiles
        .iter()
        .map(|profile| {
            let overlap = user_tokens
                .iter()
                .filter(|token| profile.tokens.contains(*token))
                .count();
            (profile.name.clone(), overlap)
        })
        .filter(|(_, score)| *score >= minimum_score)
        .collect();
    scored.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    let Some((_, top_score)) = scored.first() else {
        return Vec::new();
    };
    let top_score = *top_score;
    let runner_up = scored.get(1).map(|(_, score)| *score).unwrap_or(0);
    if top_score < minimum_score || runner_up + required_margin > top_score {
        return Vec::new();
    }

    scored
        .into_iter()
        .filter_map(|(name, score)| (score == top_score).then_some(name))
        .collect()
}

fn user_text_mentions_tool_use(user_text: &str) -> bool {
    let tokens = lexical_tokens(user_text);
    tokens
        .iter()
        .any(|token| token == "use" || token == "using" || token == "call")
        && tokens
            .iter()
            .any(|token| token == "tool" || token == "tools")
}

fn tool_text_profiles(
    tools: Option<&Value>,
    available: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<ToolSchemaProfile> {
    let schema_profiles = tool_schema_profiles(tools, available);
    if !schema_profiles.is_empty() {
        return schema_profiles;
    }

    let available: HashSet<&str> = available.iter().map(String::as_str).collect();
    prompt_tool_profiles
        .iter()
        .filter(|profile| available.is_empty() || available.contains(profile.name.as_str()))
        .map(|profile| ToolSchemaProfile {
            name: profile.name.clone(),
            tokens: profile.tokens.clone(),
        })
        .collect()
}

fn tool_schema_profiles(tools: Option<&Value>, available: &[String]) -> Vec<ToolSchemaProfile> {
    let available: HashSet<&str> = available.iter().map(String::as_str).collect();
    tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            let name = tool.pointer("/function/name")?.as_str()?;
            if !available.contains(name) {
                return None;
            }
            let mut tokens = HashSet::new();
            collect_tool_schema_tokens(tool, &mut tokens);
            Some(ToolSchemaProfile {
                name: name.to_string(),
                tokens,
            })
        })
        .collect()
}

fn collect_tool_schema_tokens(value: &Value, tokens: &mut HashSet<String>) {
    match value {
        Value::String(text) => tokens.extend(text_tokens(text)),
        Value::Array(values) => {
            for value in values {
                collect_tool_schema_tokens(value, tokens);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                tokens.extend(text_tokens(key));
                collect_tool_schema_tokens(value, tokens);
            }
        }
        _ => {}
    }
}

fn text_tokens(text: &str) -> HashSet<String> {
    lexical_tokens(text)
        .into_iter()
        .filter(|token| token.len() >= 3)
        .filter(|token| !schema_match_stopword(token))
        .collect()
}

fn lexical_tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let token = token.trim().to_ascii_lowercase();
            (!token.is_empty()).then_some(token)
        })
        .collect()
}

fn schema_match_stopword(token: &str) -> bool {
    matches!(
        token,
        "for"
            | "the"
            | "and"
            | "use"
            | "using"
            | "with"
            | "from"
            | "into"
            | "this"
            | "that"
            | "type"
            | "function"
            | "name"
            | "description"
            | "parameters"
            | "properties"
            | "required"
            | "string"
            | "object"
            | "array"
            | "boolean"
            | "integer"
            | "number"
    )
}

fn command_tool_choice_from_command(
    command: &str,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<ForcedToolChoice> {
    let candidates = command_tool_candidates(tools, allowed_tools, prompt_tool_profiles);
    let [candidate] = candidates.as_slice() else {
        tracing::info!(
            "moa: shell command block found but command-like declared tool candidates={} (need exactly one)",
            candidates.len()
        );
        return None;
    };

    let arguments = json!({ candidate.field.clone(): command });
    match sanitize_tool_arguments_for_tool(&candidate.name, &arguments, tools) {
        Ok(arguments) => Some(ForcedToolChoice {
            name: candidate.name.clone(),
            fallback_arguments: arguments,
        }),
        Err(err) => {
            tracing::info!(
                "moa: command block could not be used as {} args: {err}",
                candidate.name
            );
            None
        }
    }
}

fn command_intent_tool_names(
    tools: Option<&Value>,
    available: &[String],
    user_text: &str,
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<String> {
    let candidates = command_tool_candidates(tools, available, prompt_tool_profiles);
    let [candidate] = candidates.as_slice() else {
        return Vec::new();
    };

    if user_text_suggests_command_tool(user_text, &candidate.tokens) {
        vec![candidate.name.clone()]
    } else {
        Vec::new()
    }
}

fn user_text_suggests_command_tool(user_text: &str, schema_tokens: &HashSet<String>) -> bool {
    let user_tokens = lexical_tokens(user_text);
    if user_tokens
        .iter()
        .filter(|token| !schema_match_stopword(token))
        .any(|token| schema_tokens.contains(token))
    {
        return true;
    }

    command_execution_pattern_matches(&user_tokens)
}

fn command_execution_pattern_matches(tokens: &[String]) -> bool {
    tokens.windows(3).any(|window| {
        matches!(window[0].as_str(), "use" | "using")
            && executable_token(&window[1])
            && window[2] == "to"
    }) || tokens.windows(2).any(|window| {
        matches!(window[0].as_str(), "run" | "execute") && executable_token(&window[1])
    })
}

fn executable_token(token: &str) -> bool {
    if !(2..=40).contains(&token.len()) {
        return false;
    }
    !matches!(
        token,
        "it" | "that" | "this" | "the" | "a" | "an" | "tool" | "command"
    ) && token.chars().any(|ch| ch.is_ascii_alphabetic())
}

#[derive(Debug)]
struct CommandToolCandidate {
    name: String,
    field: String,
    tokens: HashSet<String>,
}

fn command_tool_candidates(
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<CommandToolCandidate> {
    let schema_candidates = command_schema_tool_candidates(tools, allowed_tools);
    if !schema_candidates.is_empty() {
        return clear_command_tool_candidates(schema_candidates);
    }

    clear_command_tool_candidates(prompt_command_tool_candidates(
        allowed_tools,
        prompt_tool_profiles,
    ))
}

fn clear_command_tool_candidates(
    candidates: Vec<CommandToolCandidate>,
) -> Vec<CommandToolCandidate> {
    if candidates.len() <= 1 {
        return candidates;
    }

    let mut scored: Vec<(usize, CommandToolCandidate)> = candidates
        .into_iter()
        .map(|candidate| (command_tool_candidate_score(&candidate), candidate))
        .collect();
    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.name.cmp(&right.1.name))
            .then_with(|| left.1.field.cmp(&right.1.field))
    });

    let [(score, _candidate), rest @ ..] = scored.as_slice() else {
        return Vec::new();
    };
    let runner_up = rest.first().map(|(score, _)| *score).unwrap_or(0);
    if *score >= runner_up + 3 {
        vec![scored.remove(0).1]
    } else {
        scored.into_iter().map(|(_, candidate)| candidate).collect()
    }
}

fn command_tool_candidate_score(candidate: &CommandToolCandidate) -> usize {
    command_field_score(&candidate.field, &candidate.tokens)
}

fn command_schema_tool_candidates(
    tools: Option<&Value>,
    allowed_tools: &[String],
) -> Vec<CommandToolCandidate> {
    let allowed: HashSet<&str> = allowed_tools.iter().map(String::as_str).collect();
    tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            let name = tool.pointer("/function/name")?.as_str()?;
            if !allowed.is_empty() && !allowed.contains(name) {
                return None;
            }
            let params = tool.pointer("/function/parameters")?;
            let (field, tokens) = command_like_string_field(params)?;
            Some(CommandToolCandidate {
                name: name.to_string(),
                field,
                tokens,
            })
        })
        .collect()
}

fn prompt_command_tool_candidates(
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<CommandToolCandidate> {
    let allowed: HashSet<&str> = allowed_tools.iter().map(String::as_str).collect();
    prompt_tool_profiles
        .iter()
        .filter(|profile| allowed.is_empty() || allowed.contains(profile.name.as_str()))
        .filter(|profile| prompt_profile_looks_command_executor(profile))
        .map(|profile| CommandToolCandidate {
            name: profile.name.clone(),
            field: "command".to_string(),
            tokens: profile.tokens.clone(),
        })
        .collect()
}

fn prompt_profile_looks_command_executor(profile: &PromptToolProfile) -> bool {
    let tokens = &profile.tokens;
    let has_execute = tokens.contains("run") || tokens.contains("execute");
    let has_command_target = ["command", "commands", "shell", "terminal", "cli", "clis"]
        .iter()
        .any(|token| tokens.contains(*token));
    let looks_like_existing_process = (tokens.contains("manage") || tokens.contains("background"))
        && (tokens.contains("session") || tokens.contains("sessions"))
        && (tokens.contains("running") || tokens.contains("started"));

    has_execute && has_command_target && !looks_like_existing_process
}

fn command_like_string_field(parameters: &Value) -> Option<(String, HashSet<String>)> {
    let properties = parameters.get("properties").and_then(Value::as_object)?;
    let required_matches = command_like_fields(
        parameters
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str),
        properties,
    );
    if let Some(field) = clear_command_like_field(required_matches) {
        return Some(field);
    }

    clear_command_like_field(command_like_fields(
        properties.keys().map(String::as_str),
        properties,
    ))
}

fn command_like_fields<'a>(
    fields: impl Iterator<Item = &'a str>,
    properties: &serde_json::Map<String, Value>,
) -> Vec<(String, HashSet<String>)> {
    let mut matches = Vec::new();
    for field in fields {
        let Some(schema) = properties.get(field) else {
            continue;
        };
        if !schema_allows_string(schema) {
            continue;
        }
        let mut tokens = text_tokens(field);
        collect_tool_schema_tokens(schema, &mut tokens);
        if schema_tokens_look_command_like(&tokens) {
            matches.push((field.to_string(), tokens));
        }
    }
    matches
}

fn clear_command_like_field(
    matches: Vec<(String, HashSet<String>)>,
) -> Option<(String, HashSet<String>)> {
    if matches.is_empty() {
        return None;
    }
    if matches.len() == 1 {
        let (field, tokens) = matches.into_iter().next()?;
        return Some((field, tokens));
    };

    let mut scored: Vec<(usize, String, HashSet<String>)> = matches
        .into_iter()
        .map(|(field, tokens)| (command_field_score(&field, &tokens), field, tokens))
        .filter(|(score, _, _)| *score >= 4)
        .collect();
    scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));

    let [(score, field, tokens), rest @ ..] = scored.as_slice() else {
        return None;
    };
    let runner_up = rest.first().map(|(score, _, _)| *score).unwrap_or(0);
    if *score >= runner_up + 3 {
        Some((field.clone(), tokens.clone()))
    } else {
        None
    }
}

fn command_field_score(field: &str, tokens: &HashSet<String>) -> usize {
    let field_tokens = text_tokens(field);
    let mut score = 0;
    for token in &field_tokens {
        score += match token.as_str() {
            "command" | "commands" => 8,
            "cmd" => 6,
            _ => 0,
        };
    }
    for token in tokens {
        score += match token.as_str() {
            "command" | "commands" => 2,
            "cmd" => 1,
            _ => 0,
        };
    }
    score
}

fn schema_tokens_look_command_like(tokens: &HashSet<String>) -> bool {
    tokens.contains("command") || tokens.contains("commands") || tokens.contains("cmd")
}

fn required_tool_choice_for_intent(
    session: &Session,
    tools: &Option<Value>,
    selected_tool_names: &[String],
) -> Option<ForcedToolChoice> {
    let tools = tools.as_ref()?;
    let [name] = selected_tool_names else {
        return None;
    };

    let inferred = infer_tool_arguments_from_prompt(name, Some(tools), &session.last_user_text());
    match sanitize_tool_arguments_for_tool(name, &inferred, Some(tools)) {
        Ok(arguments) => {
            if is_empty_command_tool_arguments(name, tools, &arguments) {
                tracing::info!(
                    "moa: explicit command-tool intent selected {name}, but no command arguments were inferred"
                );
                return None;
            }
            Some(ForcedToolChoice {
                name: name.clone(),
                fallback_arguments: arguments,
            })
        }
        Err(err) => {
            tracing::info!(
                "moa: explicit tool intent selected {name}, but arguments could not be inferred: {err}"
            );
            None
        }
    }
}

fn is_empty_command_tool_arguments(name: &str, tools: &Value, arguments: &Value) -> bool {
    let candidates = command_schema_tool_candidates(Some(tools), &[name.to_string()]);
    let [_candidate] = candidates.as_slice() else {
        return false;
    };
    arguments.as_object().is_some_and(serde_json::Map::is_empty)
}

fn recent_tool_chain_names(session: &Session) -> Vec<String> {
    let all = session.all_messages();
    let Some(latest_tool_idx) = all.iter().rposition(|msg| message_role(msg) == "tool") else {
        return Vec::new();
    };
    let start_idx = all[..=latest_tool_idx]
        .iter()
        .rposition(message_is_task_user)
        .unwrap_or(0);

    let mut names = Vec::new();
    for msg in &all[start_idx..=latest_tool_idx] {
        let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for tool_call in tool_calls {
            let Some(name) = tool_call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            else {
                continue;
            };
            if !names.iter().any(|existing| existing == name) {
                names.push(name.to_string());
            }
        }
    }

    names
}

fn message_role(msg: &Value) -> &str {
    msg.get("role").and_then(Value::as_str).unwrap_or("")
}

fn message_is_task_user(msg: &Value) -> bool {
    task_text_from_message(msg).is_some()
}

fn task_text_from_message(msg: &Value) -> Option<String> {
    (message_role(msg) == "user")
        .then(|| message_text(msg))
        .flatten()
        .and_then(strip_info_msg_prefix)
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn message_text(msg: &Value) -> Option<String> {
    if let Some(text) = msg.get("content").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    let text = msg
        .get("content")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(|part| {
            (part.get("type").and_then(Value::as_str) == Some("text"))
                .then(|| part.get("text").and_then(Value::as_str))
                .flatten()
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn strip_info_msg_prefix(text: String) -> Option<String> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("<info-msg>") {
        return Some(text);
    }
    let close = "</info-msg>";
    let Some(close_start) = trimmed.find(close) else {
        return Some(text);
    };
    Some(
        trimmed[close_start + close.len()..]
            .trim_start()
            .to_string(),
    )
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn forced_tool_choice(
    body: &Value,
    session: &Session,
    tools: &Option<Value>,
    allowed_tools: &[String],
) -> Option<ForcedToolChoice> {
    let name = body
        .get("tool_choice")?
        .get("function")?
        .get("name")?
        .as_str()?;
    if name.is_empty() || !allowed_tools.iter().any(|tool| tool == name) {
        return None;
    }
    tool_parameters(name, tools.as_ref())?;

    let inferred =
        infer_tool_arguments_from_prompt(name, tools.as_ref(), &session.last_user_text());
    let fallback_arguments =
        sanitize_tool_arguments_for_tool(name, &inferred, tools.as_ref()).unwrap_or(inferred);

    Some(ForcedToolChoice {
        name: name.to_string(),
        fallback_arguments,
    })
}

fn infer_tool_arguments_from_prompt(name: &str, tools: Option<&Value>, prompt: &str) -> Value {
    let Some(parameters) = tool_parameters(name, tools) else {
        return json!({});
    };
    let Some(required) = parameters.get("required").and_then(Value::as_array) else {
        return json!({});
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return json!({});
    };

    let mut args = serde_json::Map::new();
    for field in required.iter().filter_map(Value::as_str) {
        let Some(schema) = properties.get(field) else {
            continue;
        };
        if let Some(value) = infer_string_argument(field, schema, prompt) {
            args.insert(field.to_string(), Value::String(value));
        }
    }
    Value::Object(args)
}

fn infer_string_argument(field: &str, schema: &Value, prompt: &str) -> Option<String> {
    if !schema_allows_string(schema) {
        return None;
    }

    infer_enum_argument(schema, prompt)
        .or_else(|| infer_assignment_argument(field, prompt))
        .or_else(|| infer_path_like_argument(field, prompt))
        .or_else(|| infer_query_argument(field, prompt))
        .or_else(|| infer_named_argument(field, schema, prompt))
}

fn schema_allows_string(schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(value)) => value == "string",
        Some(Value::Array(values)) => values.iter().any(|value| value.as_str() == Some("string")),
        None => true,
        _ => false,
    }
}

fn infer_enum_argument(schema: &Value, prompt: &str) -> Option<String> {
    let prompt_lc = prompt.to_ascii_lowercase();
    schema
        .get("enum")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .find(|candidate| prompt_lc.contains(&candidate.to_ascii_lowercase()))
        .map(str::to_string)
}

fn infer_assignment_argument(field: &str, prompt: &str) -> Option<String> {
    let prompt_lc = prompt.to_ascii_lowercase();
    let field_lc = field.to_ascii_lowercase();
    let marker = format!("{field_lc}=");
    let start = prompt_lc.find(&marker)? + marker.len();
    let tail = prompt.get(start..)?;
    let value = tail
        .split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == ';')
        .next()
        .unwrap_or("")
        .trim_matches(|c| c == '"' || c == '\'' || c == '`');
    (!value.is_empty()).then(|| value.to_string())
}

fn infer_path_like_argument(field: &str, prompt: &str) -> Option<String> {
    let field = field.to_ascii_lowercase();
    let wants_path = contains_any(&field, &["path", "file", "url", "uri"]);
    if !wants_path {
        return None;
    }

    prompt
        .split_whitespace()
        .map(clean_path_token)
        .find(|candidate| is_path_like_candidate(candidate, &field))
}

fn clean_path_token(token: &str) -> String {
    token
        .trim_matches(|c: char| {
            matches!(
                c,
                '"' | '\'' | '`' | '<' | '>' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'
            )
        })
        .trim_end_matches(['.', ',', ';', ':', ')', ']', '}'])
        .to_string()
}

fn is_path_like_candidate(candidate: &str, field: &str) -> bool {
    if candidate.is_empty() {
        return false;
    }
    if candidate.contains("://") {
        return field.contains("url") || field.contains("uri") || field.contains("path");
    }
    candidate.starts_with('/')
        || candidate.starts_with("~/")
        || candidate.starts_with("./")
        || candidate.starts_with("../")
        || ((field.contains("path") || field.contains("file"))
            && candidate.contains('.')
            && !candidate.starts_with('.')
            && !candidate.ends_with('.'))
}

fn infer_query_argument(field: &str, prompt: &str) -> Option<String> {
    let field = field.to_ascii_lowercase();
    if !matches!(field.as_str(), "q" | "query" | "search" | "search_query") {
        return None;
    }
    let prompt = prompt.trim();
    (!prompt.is_empty()).then(|| prompt.to_string())
}

fn infer_named_argument(field: &str, schema: &Value, prompt: &str) -> Option<String> {
    let mut schema_tokens = text_tokens(field);
    collect_tool_schema_tokens(schema, &mut schema_tokens);
    if schema_tokens_look_command_like(&schema_tokens) {
        return None;
    }

    infer_quoted_argument(prompt)
        .or_else(|| infer_preposition_argument(prompt, &schema_tokens))
        .or_else(|| infer_entity_like_argument(prompt, &schema_tokens))
}

fn infer_quoted_argument(prompt: &str) -> Option<String> {
    for quote in ['"', '\'', '`'] {
        let mut parts = prompt.split(quote);
        while let Some(_before) = parts.next() {
            let Some(value) = parts.next() else {
                break;
            };
            let value = value.trim();
            if usable_named_argument(value) {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn infer_preposition_argument(prompt: &str, schema_tokens: &HashSet<String>) -> Option<String> {
    let words = prompt_words(prompt);
    for (idx, word) in words.iter().enumerate() {
        if !matches!(
            word.lower.as_str(),
            "in" | "for" | "about" | "near" | "at" | "to"
        ) {
            continue;
        }
        let mut phrase = Vec::new();
        for next in words.iter().skip(idx + 1).take(5) {
            if named_argument_boundary(&next.lower) {
                break;
            }
            phrase.push(next.clean.clone());
        }
        let value = phrase.join(" ");
        if usable_named_argument_for_schema(&value, schema_tokens) {
            return Some(value);
        }
    }
    None
}

fn infer_entity_like_argument(prompt: &str, schema_tokens: &HashSet<String>) -> Option<String> {
    let words = prompt_words(prompt);
    let mut idx = 0;
    while idx < words.len() {
        if !word_looks_entity_like(&words[idx]) {
            idx += 1;
            continue;
        }
        let mut phrase = vec![words[idx].clean.clone()];
        idx += 1;
        while idx < words.len() && word_looks_entity_like(&words[idx]) {
            phrase.push(words[idx].clean.clone());
            idx += 1;
        }
        let value = phrase.join(" ");
        if usable_named_argument_for_schema(&value, schema_tokens) {
            return Some(value);
        }
    }
    None
}

#[derive(Debug)]
struct PromptWord {
    clean: String,
    lower: String,
}

fn prompt_words(prompt: &str) -> Vec<PromptWord> {
    prompt
        .split_whitespace()
        .map(clean_named_argument_token)
        .filter(|token| !token.is_empty())
        .map(|clean| PromptWord {
            lower: clean.to_ascii_lowercase(),
            clean,
        })
        .collect()
}

fn clean_named_argument_token(token: &str) -> String {
    token
        .trim_matches(|c: char| {
            matches!(
                c,
                '"' | '\'' | '`' | '<' | '>' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'
            )
        })
        .trim_end_matches(['.', ',', ';', ':', '?', '!', ')', ']', '}'])
        .to_string()
}

fn named_argument_boundary(token: &str) -> bool {
    matches!(
        token,
        "right"
            | "now"
            | "today"
            | "currently"
            | "please"
            | "use"
            | "using"
            | "look"
            | "lookup"
            | "check"
            | "find"
            | "get"
            | "search"
            | "run"
            | "execute"
            | "call"
            | "with"
            | "the"
            | "a"
            | "an"
            | "tool"
            | "tools"
            | "available"
    )
}

fn usable_named_argument_for_schema(value: &str, schema_tokens: &HashSet<String>) -> bool {
    usable_named_argument(value)
        && !text_tokens(value)
            .iter()
            .any(|token| schema_tokens.contains(token))
}

fn usable_named_argument(value: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && value.len() <= 120
        && !value.contains('\n')
        && !matches!(
            value.to_ascii_lowercase().as_str(),
            "what" | "which" | "when" | "where" | "why" | "how" | "please"
        )
}

fn word_looks_entity_like(word: &PromptWord) -> bool {
    let Some(first) = word.clean.chars().next() else {
        return false;
    };
    first.is_ascii_uppercase()
        && !matches!(
            word.lower.as_str(),
            "what" | "which" | "when" | "where" | "why" | "how" | "use"
        )
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

// ─── Tool result handling ────────────────────────────────────────────

async fn handle_tool_result(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
    start: Instant,
) -> TurnResult {
    let candidates = reducer_candidates(config);
    let candidate_count = candidates.len();
    let repeated_tool = repeated_same_tool_results(session);
    let tool_budget_exhausted = tool_budget_exhausted(session);
    let declared_tool_names = declared_tool_names(session, prompt_tool_profiles);
    let mut selected_tool_names = suppress_repeated_tool_from_selection(
        selected_tool_names_for_turn(session, allowed_tools, prompt_tool_profiles),
        &declared_tool_names,
        repeated_tool.as_ref().map(|(tool, _)| tool.as_str()),
    );
    let latest_user_text = session.active_user_text();
    let prompt_requests_more_tool_steps =
        prompt_requests_additional_tool_step(&latest_user_text, session);
    let latest_answerable_tool = latest_completed_tool_result(session).filter(|(_, result)| {
        tool_result_has_answerable_evidence_for_prompt(result, &latest_user_text)
    });
    if let Some((tool, _)) = latest_answerable_tool.filter(|_| !prompt_requests_more_tool_steps) {
        selected_tool_names.retain(|name| name != &tool);
    }
    let latest_non_answerable_tool = latest_completed_tool_result(session).filter(|(_, result)| {
        !tool_result_has_answerable_evidence_for_prompt(result, &latest_user_text)
    });
    if let Some((tool, _)) = latest_non_answerable_tool.as_ref() {
        selected_tool_names.retain(|name| name == tool);
    }
    let tools_enabled_for_reducer =
        has_tools && !selected_tool_names.is_empty() && tool_budget_exhausted.is_none();
    let (mut messages, tools) = if tool_budget_exhausted.is_some() {
        (context::pack_for_tool_result_answer_only(session), None)
    } else {
        context::pack_for_tool_result_turn_selected(
            session,
            tools_enabled_for_reducer,
            &selected_tool_names,
        )
    };
    if let Some((tool, count)) = repeated_tool.as_ref() {
        tracing::info!(
            "moa: suppressing repeated {tool} after {count} consecutive completed tool calls"
        );
        append_tool_loop_answer_instruction(&mut messages, tool, *count);
    }
    if let Some(count) = tool_budget_exhausted {
        tracing::info!("moa: forcing answer after {count} completed tool calls");
        append_tool_budget_instruction(&mut messages, count);
    }
    let retry_tool = if tools_enabled_for_reducer {
        latest_non_answerable_tool.clone()
    } else {
        None
    };
    let retry_tool_call = retry_tool.as_ref().and_then(|(tool, _)| {
        latest_completed_tool_call(session).filter(|call| &call.name == tool)
    });
    if let Some((tool, _)) = retry_tool {
        tracing::info!(
            "moa: latest {tool} tool result looks like error/usage output; asking reducer to retry"
        );
        append_tool_error_retry_instruction(&mut messages, &tool);
    }

    // Hedged ladder: start candidate 0, hedge to candidate 1 after hedge_delay
    // (or immediately on candidate 0 error), race for the first OK. Rescues
    // tool-result turns when the first strong peer is broken (e.g. stale
    // binary that 502s on tool grammars) without paying N×timeout serially.
    tracing::info!("moa: tool result → hedged reducer over {candidate_count} candidate(s)");
    let hedge_result = hedged_reducer_call(
        &config.backends,
        candidates.clone(),
        messages.clone(),
        tools.clone(),
        config.reducer_timeout,
        config.hedge_delay,
        config.enable_thinking,
    )
    .await;

    let mut last_err: Option<String> = None;
    let (attempts, chosen): (u32, Option<(String, normalize::WorkerOutput, bool)>) =
        match hedge_result {
            Ok(reducer::HedgedReducerOk {
                winner,
                text,
                attempts: spawned,
            }) => {
                let mut reduced =
                    normalize::normalize_worker_output(&text, &winner, WorkerRole::Reducer, 0);
                enforce_tool_call_contract(&mut reduced, allowed_tools, session.tools(), &winner);
                (spawned, Some((winner, reduced, tools_enabled_for_reducer)))
            }
            Err(reducer::HedgedReducerErr {
                err,
                attempts: spawned,
            }) => {
                last_err = Some(err);
                if tools.is_some() {
                    let fallback_messages = context::pack_for_tool_result_answer_only(session);
                    tracing::warn!(
                        "moa: tool-result reducers failed with native tools; retrying answer-only reducer"
                    );
                    match hedged_reducer_call(
                        &config.backends,
                        candidates.clone(),
                        fallback_messages,
                        None,
                        config.reducer_timeout,
                        config.hedge_delay,
                        config.enable_thinking,
                    )
                    .await
                    {
                        Ok(reducer::HedgedReducerOk {
                            winner,
                            text,
                            attempts: retry_attempts,
                        }) => {
                            let reduced = normalize::normalize_worker_output(
                                &text,
                                &winner,
                                WorkerRole::Reducer,
                                0,
                            );
                            (
                                spawned.saturating_add(retry_attempts),
                                Some((winner, reduced, false)),
                            )
                        }
                        Err(reducer::HedgedReducerErr {
                            err,
                            attempts: retry_attempts,
                        }) => {
                            last_err = Some(err);
                            (spawned.saturating_add(retry_attempts), None)
                        }
                    }
                } else {
                    (spawned, None)
                }
            }
        };

    let (reducer_name, succeeded, response_body) = match chosen {
        Some((name, reduced, response_tools_enabled)) => {
            let empty_tools: &[String] = &[];
            let empty_profiles: &[PromptToolProfile] = &[];
            let response_allowed_tools = if response_tools_enabled {
                allowed_tools
            } else {
                empty_tools
            };
            let response_prompt_profiles = if response_tools_enabled {
                prompt_tool_profiles
            } else {
                empty_profiles
            };
            // When tools remain enabled, be consistent with the
            // fanout/arbiter path and emit a real `tool_calls` response for
            // valid reducer proposals. When tools are disabled, never pass
            // tool-call-shaped text through as chat content; answer from the
            // completed tool result instead.
            let body = match reduced.kind {
                normalize::OutputKind::ToolProposal if !response_tools_enabled => {
                    tool_result_answer_from_evidence_response(session)
                }
                normalize::OutputKind::ToolProposal => {
                    let body = tool_proposal_response(
                        &reduced,
                        response_tools_enabled,
                        session.tools(),
                        response_allowed_tools,
                        response_prompt_profiles,
                        Some(&session.last_user_text()),
                    );
                    retry_tool_result_response_if_plain(
                        body,
                        retry_tool_call.as_ref(),
                        latest_non_answerable_tool
                            .as_ref()
                            .map(|(_, result)| result.as_str()),
                        session.tools(),
                        response_allowed_tools,
                        response_prompt_profiles,
                        response_tools_enabled,
                    )
                }
                normalize::OutputKind::Uncertainty => {
                    if response_tools_enabled {
                        retry_tool_call
                            .as_ref()
                            .and_then(|call| {
                                retry_tool_call_response(
                                    call,
                                    latest_non_answerable_tool
                                        .as_ref()
                                        .map(|(_, result)| result.as_str()),
                                    session.tools(),
                                    response_allowed_tools,
                                    response_prompt_profiles,
                                )
                            })
                            .unwrap_or_else(|| {
                                error_response(
                                    "MoA reducer returned no usable answer",
                                    MOA_ERR_NO_USABLE_ANSWER,
                                )
                            })
                    } else {
                        error_response(
                            "MoA reducer returned no usable answer",
                            MOA_ERR_NO_USABLE_ANSWER,
                        )
                    }
                }
                _ => {
                    let repaired = repair_tool_result_answer(session, &reduced.payload);
                    match latest_non_answerable_tool.as_ref() {
                        Some((tool, result))
                            if answer_appears_to_use_non_answerable_tool_result(
                                &repaired, result,
                            ) =>
                        {
                            let answer = non_answerable_tool_result_answer(session, tool, result);
                            return TurnResult {
                                response_body: chat_response(&answer),
                                worker_summaries: vec![WorkerSummary {
                                    model: name,
                                    role: WorkerRole::Reducer,
                                    succeeded: false,
                                    elapsed_ms: start.elapsed().as_millis() as u64,
                                    output_kind: None,
                                    confidence: None,
                                }],
                                reducer_used: true,
                                reducer_attempts: attempts,
                                turn_kind: TurnKind::ToolResult,
                                elapsed_ms: start.elapsed().as_millis() as u64,
                            };
                        }
                        _ => {}
                    }
                    let body = if response_tools_enabled {
                        chat_or_schema_command_tool_response(
                            &repaired,
                            session.tools(),
                            response_allowed_tools,
                            response_prompt_profiles,
                            Some(&session.last_user_text()),
                        )
                    } else {
                        chat_response(&repaired)
                    };
                    retry_tool_result_response_if_plain(
                        body,
                        retry_tool_call.as_ref(),
                        latest_non_answerable_tool
                            .as_ref()
                            .map(|(_, result)| result.as_str()),
                        session.tools(),
                        response_allowed_tools,
                        response_prompt_profiles,
                        response_tools_enabled,
                    )
                }
            };
            (name, true, body)
        }
        None => {
            let err = last_err.unwrap_or_else(|| "no reducer candidates".into());
            tracing::warn!("moa: all {attempts} reducer candidates failed");
            if let Some(answer) = answer_from_latest_tool_result(session) {
                tracing::warn!(
                    "moa: reducers failed after answerable tool evidence; answering from tool result"
                );
                (
                    candidates.first().map(|c| c.0.clone()).unwrap_or_default(),
                    false,
                    chat_response(&answer),
                )
            } else {
                (
                    candidates.first().map(|c| c.0.clone()).unwrap_or_default(),
                    false,
                    error_response(
                        &format!("Reducer failed (tried {attempts}): {err}"),
                        MOA_ERR_ALL_REDUCERS_FAILED,
                    ),
                )
            }
        }
    };

    TurnResult {
        response_body,
        worker_summaries: vec![WorkerSummary {
            model: reducer_name,
            role: WorkerRole::Reducer,
            succeeded,
            elapsed_ms: start.elapsed().as_millis() as u64,
            output_kind: None,
            confidence: None,
        }],
        reducer_used: true,
        reducer_attempts: attempts,
        turn_kind: TurnKind::ToolResult,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

fn repair_tool_result_answer(session: &Session, answer: &str) -> String {
    if let Some(answer) = answer_from_recent_prompt_specific_structured_tool_results(session) {
        return answer;
    }

    if let Some(answer) = latest_completed_tool_result(session).and_then(|(_, result)| {
        answer_from_prompt_specific_structured_tool_result(&result, &session.active_user_text())
    }) {
        return answer;
    }

    if let Some(answer) = repair_structured_tool_result_answer(session, answer) {
        return answer;
    }

    let repaired_rows =
        repair_tabular_tool_result_answer(session, answer).unwrap_or_else(|| answer.to_string());

    if !tool_evidence_should_be_preserved(&session.last_user_text()) {
        return repaired_rows;
    }

    let missing = missing_tool_evidence_values(session, &repaired_rows);
    if missing.is_empty() {
        return repaired_rows;
    }

    let mut repaired = repaired_rows.trim().to_string();
    if !repaired.is_empty() {
        repaired.push_str("\n\n");
    }
    repaired.push_str("Tool facts: ");
    repaired.push_str(&missing.join(", "));
    repaired
}

fn answer_from_recent_prompt_specific_structured_tool_results(session: &Session) -> Option<String> {
    let prompt = session.active_user_text();
    if !(prompt_asks_for_ci_status(&prompt) && prompt_asks_for_feedback(&prompt)) {
        return None;
    }

    let mut title = None;
    let mut checks = None;
    let mut feedback = None;
    for (_, result) in session.recent_tool_results() {
        let Some(value) = parse_tool_result_json(result.trim()) else {
            continue;
        };
        title = title.or_else(|| structured_title_state_summary(&value));
        checks = checks.or_else(|| structured_check_summary_from_value(&value));
        feedback = feedback.or_else(|| structured_feedback_summary_from_value(&value, &prompt));
    }

    let mut parts = Vec::new();
    parts.extend(title);
    parts.push(checks?);
    parts.push(feedback?);
    Some(truncate_for_tool_result_answer(&parts.join("\n")))
}

fn repair_structured_tool_result_answer(session: &Session, answer: &str) -> Option<String> {
    let (_, result) = latest_completed_tool_result(session)?;
    let prompt = session.active_user_text();
    let rows = structured_tool_result_rows_from_result(&result, &prompt);
    if rows.is_empty() {
        return None;
    }

    if prompt_asks_for_work_items(&prompt)
        && !rows
            .iter()
            .any(|row| structured_tool_row_is_referenced(answer, row))
    {
        return answer_from_structured_tool_result(&result, &prompt);
    }

    if let (true, Some(trimmed)) = (
        prompt_asks_for_work_items(&prompt),
        trim_unrelated_prefix_before_structured_row(answer, &rows),
    ) {
        return Some(trimmed);
    }

    let missing_rows = rows
        .iter()
        .take(3)
        .filter_map(|row| structured_tool_row_missing_reference(answer, row, &prompt))
        .collect::<Vec<_>>();
    if missing_rows.is_empty() {
        None
    } else {
        Some(append_tool_rows(answer, &missing_rows))
    }
}

fn structured_tool_row_is_referenced(answer: &str, row: &StructuredToolRow) -> bool {
    answer_contains_text(answer, &row.title)
        || row
            .id
            .as_deref()
            .is_some_and(|id| answer_has_row_id(answer, id))
        || row
            .url
            .as_deref()
            .is_some_and(|url| answer_contains_text(answer, url))
}

fn structured_tool_row_missing_reference(
    answer: &str,
    row: &StructuredToolRow,
    prompt: &str,
) -> Option<String> {
    let has_title = answer_contains_text(answer, &row.title);
    let has_id = row
        .id
        .as_deref()
        .is_some_and(|id| answer_has_row_id(answer, id));
    ((has_id && !has_title) || (has_title && row.id.is_some() && !has_id))
        .then(|| format_structured_tool_row(row, prompt))
}

fn trim_unrelated_prefix_before_structured_row(
    answer: &str,
    rows: &[StructuredToolRow],
) -> Option<String> {
    const MIN_PREFIX_CHARS: usize = 240;
    let first_ref = rows
        .iter()
        .filter_map(|row| first_structured_row_reference_index(answer, row))
        .min()?;
    if answer[..first_ref].chars().count() < MIN_PREFIX_CHARS {
        return None;
    }
    let start = answer[..first_ref]
        .rfind('\n')
        .map_or(first_ref, |index| index + 1);
    let trimmed = answer[start..].trim_start();
    (!trimmed.is_empty() && trimmed.len() < answer.trim_start().len()).then(|| trimmed.to_string())
}

fn first_structured_row_reference_index(answer: &str, row: &StructuredToolRow) -> Option<usize> {
    let mut candidates = Vec::new();
    if let Some(id) = row.id.as_deref() {
        candidates.extend([
            answer.find(&format!("#{id}")),
            answer.find(&format!(" {id} ")),
            answer.find(&format!("PR {id}")),
            answer.find(&format!("issue {id}")),
        ]);
    }
    candidates.push(case_insensitive_find(answer, &row.title));
    if let Some(url) = row.url.as_deref() {
        candidates.push(answer.find(url));
    }
    candidates.into_iter().flatten().min()
}

fn case_insensitive_find(haystack: &str, needle: &str) -> Option<usize> {
    let haystack_lower = haystack.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();
    haystack_lower.find(&needle_lower)
}

#[derive(Debug)]
struct TabularToolRow {
    id: String,
    title: String,
}

#[derive(Debug)]
struct StructuredToolRow {
    id: Option<String>,
    title: String,
    url: Option<String>,
    details: Vec<String>,
}

fn repair_tabular_tool_result_answer(session: &Session, answer: &str) -> Option<String> {
    let (_, result) = latest_completed_tool_result(session)?;
    let rows = tabular_tool_result_rows(&result);
    if rows.is_empty() {
        return None;
    }

    for row in &rows {
        let has_id = answer_has_row_id(answer, &row.id);
        let has_title = answer_contains_text(answer, &row.title);
        if has_id && !has_title {
            let exact_id_answer = answer
                .trim()
                .trim_start_matches('#')
                .chars()
                .all(|ch| ch.is_ascii_digit());
            let row_text = format!("#{} - {}", row.id, row.title);
            if exact_id_answer {
                return Some(row_text);
            }
            return Some(append_tool_rows(answer, &[row_text]));
        }
    }

    let missing_rows: Vec<String> = rows
        .iter()
        .filter(|row| {
            answer_contains_text(answer, &row.title) && !answer_has_row_id(answer, &row.id)
        })
        .map(|row| format!("#{} - {}", row.id, row.title))
        .collect();
    if missing_rows.is_empty() {
        None
    } else {
        Some(append_tool_rows(answer, &missing_rows))
    }
}

fn tabular_tool_result_rows(result: &str) -> Vec<TabularToolRow> {
    result
        .lines()
        .filter_map(|line| {
            let mut columns = line.split('\t').map(str::trim);
            let id = columns.next()?.trim_start_matches('#');
            let title = columns.next()?;
            (!id.is_empty() && id.chars().all(|ch| ch.is_ascii_digit()) && !title.is_empty()).then(
                || TabularToolRow {
                    id: id.to_string(),
                    title: title.to_string(),
                },
            )
        })
        .collect()
}

fn answer_has_row_id(answer: &str, id: &str) -> bool {
    answer.contains(&format!("#{id}"))
        || answer
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .any(|token| token == id)
}

fn answer_contains_text(answer: &str, text: &str) -> bool {
    answer
        .to_ascii_lowercase()
        .contains(&text.to_ascii_lowercase())
}

fn append_tool_rows(answer: &str, rows: &[String]) -> String {
    let mut repaired = answer.trim().to_string();
    if !repaired.is_empty() {
        repaired.push_str("\n\n");
    }
    repaired.push_str("Tool rows: ");
    repaired.push_str(&rows.join(", "));
    repaired
}

fn tool_result_answer_from_evidence_response(session: &Session) -> Value {
    match answer_from_latest_tool_result(session) {
        Some(answer) => chat_response(&answer),
        None => error_response(
            "MoA reducer requested another tool after tools were disabled",
            MOA_ERR_NO_USABLE_ANSWER,
        ),
    }
}

fn answer_from_latest_tool_result(session: &Session) -> Option<String> {
    let (_, result) = latest_completed_tool_result(session)?;
    let prompt = session.active_user_text();
    if !tool_result_has_answerable_evidence_for_prompt(&result, &prompt) {
        return None;
    }

    if let Some(answer) = answer_from_structured_tool_result(&result, &prompt) {
        return Some(answer);
    }

    let mut lines = result
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let first = lines.next()?;
    let mut answer = format!(
        "From the tool result, one relevant entry is: {}",
        truncate_for_tool_result_answer(first)
    );
    let rest = lines.take(2).collect::<Vec<_>>();
    if !rest.is_empty() {
        answer.push_str("\n\nOther returned entries include:\n");
        for line in rest {
            answer.push_str("- ");
            answer.push_str(&truncate_for_tool_result_answer(line));
            answer.push('\n');
        }
        answer = answer.trim_end().to_string();
    }

    Some(truncate_for_tool_result_answer(&answer))
}

fn answer_appears_to_use_non_answerable_tool_result(answer: &str, result: &str) -> bool {
    let answer_lower = answer.to_ascii_lowercase();
    if answer_lower.contains("from the tool result") || answer_lower.contains("one relevant entry")
    {
        return true;
    }

    result
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .any(|line| line.len() >= 12 && answer.contains(line))
}

fn non_answerable_tool_result_answer(session: &Session, tool: &str, result: &str) -> String {
    let prompt = session.active_user_text();
    if prompt_asks_for_work_items(&prompt)
        && plain_text_tool_result_looks_like_repository_list(result)
    {
        return "The latest tool result listed repositories, not the requested work items, so I can't answer from that output. I need a corrected work-item result before I can pick an item.".to_string();
    }
    if prompt_asks_for_work_items(&prompt)
        && plain_text_tool_result_looks_like_git_repository_metadata(result)
    {
        return "The latest tool result listed git repository metadata, not the requested work items, so I can't answer from that output. I need a corrected work-item result before I can pick an item.".to_string();
    }

    let first_line = result
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("empty output");
    format!(
        "The latest `{tool}` result is not answer evidence for the request: {}. I need a corrected tool result before I can answer.",
        truncate_for_tool_result_answer(first_line)
    )
}

fn answer_from_structured_tool_result(result: &str, prompt: &str) -> Option<String> {
    let rows = match parse_tool_result_json(result.trim()) {
        Some(parsed) => {
            if let Some(answer) = answer_from_prompt_specific_structured_value(&parsed, prompt) {
                return Some(answer);
            }
            structured_tool_result_rows(&parsed, prompt)
        }
        None => structured_tool_result_rows_from_embedded_json(result, prompt),
    };
    let first = rows.first()?;
    let mut answer = format!(
        "From the tool result, one relevant entry is: {}",
        format_structured_tool_row(first, prompt)
    );
    let rest = rows
        .iter()
        .skip(1)
        .take(2)
        .map(|row| format_structured_tool_row(row, prompt))
        .collect::<Vec<_>>();
    if !rest.is_empty() {
        answer.push_str("\n\nOther returned entries include:\n");
        for row in rest {
            answer.push_str("- ");
            answer.push_str(&row);
            answer.push('\n');
        }
        answer = answer.trim_end().to_string();
    }

    Some(truncate_for_tool_result_answer(&answer))
}

fn answer_from_prompt_specific_structured_tool_result(
    result: &str,
    prompt: &str,
) -> Option<String> {
    let parsed = parse_tool_result_json(result.trim())?;
    answer_from_prompt_specific_structured_value(&parsed, prompt)
}

fn answer_from_prompt_specific_structured_value(value: &Value, prompt: &str) -> Option<String> {
    if prompt_asks_for_check_or_review_status(prompt) {
        return answer_from_structured_status_feedback_value(value, prompt);
    }
    None
}

fn parse_tool_result_json(result: &str) -> Option<Value> {
    serde_json::from_str::<Value>(result)
        .ok()
        .or_else(|| parse_concatenated_json_values(result))
        .or_else(|| parse_json_prefix_with_escaped_string_controls(result))
}

fn structured_tool_result_rows_from_result(result: &str, prompt: &str) -> Vec<StructuredToolRow> {
    parse_tool_result_json(result.trim())
        .map(|value| structured_tool_result_rows(&value, prompt))
        .unwrap_or_else(|| structured_tool_result_rows_from_embedded_json(result, prompt))
}

fn structured_tool_result_rows_from_embedded_json(
    result: &str,
    prompt: &str,
) -> Vec<StructuredToolRow> {
    embedded_json_objects(result)
        .into_iter()
        .flat_map(|value| structured_tool_result_rows(&value, prompt))
        .collect()
}

fn embedded_json_objects(input: &str) -> Vec<Value> {
    let mut values = Vec::new();
    let mut start = None;
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0usize;

    for (index, ch) in input.char_indices() {
        if depth == 0 {
            if ch == '{' {
                start = Some(index);
                depth = 1;
                in_string = false;
                escaped = false;
            }
            continue;
        }

        if in_string {
            match ch {
                '"' if !escaped => in_string = false,
                _ => {}
            }
            escaped = ch == '\\' && !escaped;
            if ch != '\\' {
                escaped = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth = depth.saturating_add(1),
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let parsed =
                        start.and_then(|start| serde_json::from_str(&input[start..=index]).ok());
                    start = None;
                    if let Some(value) = parsed {
                        values.push(value);
                    }
                }
            }
            _ => {}
        }
    }

    values
}

fn parse_concatenated_json_values(input: &str) -> Option<Value> {
    let mut values = Vec::new();
    let stream = serde_json::Deserializer::from_str(input).into_iter::<Value>();
    for value in stream {
        values.push(value.ok()?);
    }
    (values.len() > 1).then_some(Value::Array(values))
}

fn parse_json_prefix_with_escaped_string_controls(input: &str) -> Option<Value> {
    let mut out = String::new();
    let mut started = false;
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0usize;

    for ch in input.chars() {
        if !started {
            if matches!(ch, '{' | '[') {
                started = true;
                depth = 1;
                out.push(ch);
            }
            continue;
        }

        if in_string {
            match ch {
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '"' if !escaped => {
                    in_string = false;
                    out.push(ch);
                }
                _ => out.push(ch),
            }
            escaped = ch == '\\' && !escaped;
            if ch != '\\' {
                escaped = false;
            }
            continue;
        }

        match ch {
            '"' => {
                in_string = true;
                out.push(ch);
            }
            '{' | '[' => {
                depth = depth.saturating_add(1);
                out.push(ch);
            }
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                out.push(ch);
                if depth == 0 {
                    break;
                }
            }
            _ => out.push(ch),
        }
    }

    (started && depth == 0)
        .then(|| serde_json::from_str::<Value>(&out).ok())
        .flatten()
}

fn prompt_asks_for_check_or_review_status(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    prompt_asks_for_ci_status(prompt)
        || prompt_asks_for_feedback(prompt)
        || contains_any(&lower, &["status", "green"])
}

fn answer_from_structured_status_feedback_value(value: &Value, prompt: &str) -> Option<String> {
    if let Value::Array(_) = value {
        return structured_check_summary_from_value(value)
            .filter(|_| prompt_asks_for_ci_status(prompt));
    }

    let map = value.as_object()?;
    let mut parts = Vec::new();
    let prompt_lower = prompt.to_ascii_lowercase();
    if let Some(title) = first_string_field(map, &["title", "name"]) {
        let state = first_string_field(map, &["state", "status"]);
        parts.push(match state {
            Some(state) => format!("{title} ({state})"),
            None => title,
        });
    }

    if contains_any(&prompt_lower, &["ci", "check", "checks", "green", "status"]) {
        parts.extend(structured_check_summary_from_value(value));
    }

    if prompt_asks_for_feedback(prompt) {
        parts.extend(structured_feedback_summary_from_value(value, prompt));
    }

    (!parts.is_empty()).then(|| truncate_for_tool_result_answer(&parts.join("\n")))
}

fn structured_title_state_summary(value: &Value) -> Option<String> {
    let map = value.as_object()?;
    let title = first_string_field(map, &["title", "name"])?;
    let state = first_string_field(map, &["state", "status"]);
    Some(match state {
        Some(state) => format!("{title} ({state})"),
        None => title,
    })
}

fn structured_check_summary_from_value(value: &Value) -> Option<String> {
    let mut checks = Vec::new();
    collect_structured_check_items(value, &mut checks);
    structured_check_summary(&checks)
}

fn collect_structured_check_items(value: &Value, checks: &mut Vec<Value>) {
    match value {
        Value::Object(map) if structured_object_looks_like_check(map) => {
            checks.push(value.clone());
        }
        Value::Object(map) => {
            for nested in map.values() {
                collect_structured_check_items(nested, checks);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_structured_check_items(value, checks);
            }
        }
        _ => {}
    }
}

fn structured_object_looks_like_check(map: &serde_json::Map<String, Value>) -> bool {
    map.contains_key("conclusion")
        || map.contains_key("outcome")
        || map.contains_key("result")
        || (map.contains_key("status")
            && first_string_field(
                map,
                &[
                    "name",
                    "workflowName",
                    "workflow_name",
                    "jobName",
                    "job_name",
                    "detailsUrl",
                    "details_url",
                ],
            )
            .is_some())
}

fn structured_feedback_summary_from_value(value: &Value, prompt: &str) -> Option<String> {
    if !json_value_has_feedback_evidence(value, prompt) {
        return None;
    }

    let focus = feedback_focus_from_prompt(prompt);
    let mut entries = Vec::new();
    collect_structured_feedback_entries(value, focus.as_deref(), &mut entries);
    structured_feedback_summary(entries, focus.as_deref())
}

fn structured_check_summary(checks: &[Value]) -> Option<String> {
    if checks.is_empty() {
        return None;
    }

    let mut failing = Vec::new();
    let mut pending = Vec::new();
    for check in checks.iter().filter_map(Value::as_object) {
        let name = first_string_field(
            check,
            &[
                "name",
                "workflowName",
                "workflow_name",
                "jobName",
                "job_name",
                "displayName",
                "display_name",
                "title",
            ],
        )
        .unwrap_or_else(|| {
            check
                .get("__typename")
                .and_then(Value::as_str)
                .unwrap_or("check")
                .to_string()
        });
        let conclusion = first_string_field(check, &["conclusion", "outcome", "result"]);
        let status = first_string_field(check, &["status", "state"]);
        let normalized_conclusion = conclusion.as_deref().map(str::to_ascii_uppercase);
        match normalized_conclusion.as_deref() {
            Some("SUCCESS" | "PASSED" | "PASS" | "SKIPPED" | "NEUTRAL") => {}
            Some(other) => failing.push(format!("{name}: {other}")),
            None if status.as_deref().is_some_and(status_value_looks_failure) => {
                failing.push(format!("{name}: {}", status.unwrap_or_default()));
            }
            None if !status.as_deref().is_some_and(status_value_looks_completed) => {
                pending.push(format!(
                    "{name}: {}",
                    status.unwrap_or_else(|| "pending".to_string())
                ));
            }
            None => {}
        }
    }

    if failing.is_empty() && pending.is_empty() {
        Some(format!(
            "CI/checks: all {} returned checks are green.",
            checks.len()
        ))
    } else {
        let mut text = String::from("CI/checks: not all green.");
        if !failing.is_empty() {
            text.push_str(" Failing: ");
            text.push_str(&failing.into_iter().take(3).collect::<Vec<_>>().join("; "));
            text.push('.');
        }
        if !pending.is_empty() {
            text.push_str(" Pending: ");
            text.push_str(&pending.into_iter().take(3).collect::<Vec<_>>().join("; "));
            text.push('.');
        }
        Some(text)
    }
}

fn status_value_looks_completed(value: &str) -> bool {
    matches!(
        value.to_ascii_uppercase().as_str(),
        "COMPLETED" | "COMPLETE" | "DONE" | "SUCCESS" | "PASSED" | "PASS"
    )
}

fn status_value_looks_failure(value: &str) -> bool {
    matches!(
        value.to_ascii_uppercase().as_str(),
        "FAILURE" | "FAILED" | "FAIL" | "ERROR" | "CANCELLED" | "CANCELED" | "TIMED_OUT"
    )
}

fn structured_feedback_summary(entries: Vec<String>, focus: Option<&str>) -> Option<String> {
    if entries.is_empty() {
        if let Some(focus) = focus {
            return Some(format!(
                "{} feedback: no {}-authored comments or reviews were returned.",
                feedback_label(focus),
                focus
            ));
        }
        return Some("Feedback: no comments or reviews were returned.".to_string());
    }

    Some(format!(
        "{}: {}",
        focus
            .map(|focus| format!("{} feedback", feedback_label(focus)))
            .unwrap_or_else(|| "Feedback".to_string()),
        entries.into_iter().take(3).collect::<Vec<_>>().join("; ")
    ))
}

fn collect_structured_feedback_entries(
    value: &Value,
    focus: Option<&str>,
    entries: &mut Vec<String>,
) {
    match value {
        Value::Object(map) if structured_object_looks_like_feedback_entry(map) => {
            collect_structured_feedback_entry(map, "feedback", focus, entries);
        }
        Value::Object(map) => {
            for key in FEEDBACK_COLLECTION_KEYS {
                if let Some(nested) = map.get(*key) {
                    collect_feedback_collection_entries(nested, key, focus, entries);
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_structured_feedback_entries(value, focus, entries);
            }
        }
        _ => {}
    }
}

fn collect_feedback_collection_entries(
    value: &Value,
    key: &str,
    focus: Option<&str>,
    entries: &mut Vec<String>,
) {
    match value {
        Value::Array(values) => {
            for item in values.iter().filter_map(Value::as_object) {
                collect_structured_feedback_entry(item, feedback_kind_for_key(key), focus, entries);
            }
        }
        Value::Object(map) => {
            collect_structured_feedback_entry(map, feedback_kind_for_key(key), focus, entries);
        }
        _ => {}
    }
}

fn collect_structured_feedback_entry(
    item: &serde_json::Map<String, Value>,
    kind: &str,
    focus: Option<&str>,
    entries: &mut Vec<String>,
) {
    let author = item
        .get("author")
        .and_then(|author| author.get("login").or_else(|| author.get("name")))
        .or_else(|| item.get("author"))
        .or_else(|| item.get("user").and_then(|user| user.get("login")))
        .or_else(|| item.get("creator").and_then(|user| user.get("login")))
        .or_else(|| item.get("reviewer").and_then(|user| user.get("name")))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let body = first_string_field(item, &["body", "text", "comment", "message", "summary"])
        .unwrap_or_default();
    if let Some(focus) = focus {
        let author_or_body_matches = author.to_ascii_lowercase().contains(focus)
            || body.to_ascii_lowercase().contains(focus);
        if !author_or_body_matches {
            return;
        }
    }
    let state = first_string_field(item, &["state"]);
    let body_summary = first_non_empty_line(&body).unwrap_or_else(|| "(no body)".to_string());
    entries.push(match state {
        Some(state) => format!("{kind} by {author} ({state}): {body_summary}"),
        None => format!("{kind} by {author}: {body_summary}"),
    });
}

fn structured_object_looks_like_feedback_entry(map: &serde_json::Map<String, Value>) -> bool {
    first_string_field(map, &["body", "text", "comment", "message", "summary"]).is_some()
        && (map.contains_key("author")
            || map.contains_key("user")
            || map.contains_key("creator")
            || map.contains_key("reviewer"))
}

const FEEDBACK_COLLECTION_KEYS: &[&str] = &[
    "comments",
    "reviews",
    "feedback",
    "notes",
    "annotations",
    "reviewComments",
    "review_comments",
];

fn feedback_kind_for_key(key: &str) -> &str {
    match key {
        "comments" => "comment",
        "reviews" => "review",
        "annotations" => "annotation",
        "notes" => "note",
        "reviewComments" | "review_comments" => "review comment",
        _ => "feedback",
    }
}

fn feedback_focus_from_prompt(prompt: &str) -> Option<String> {
    let lower = prompt.to_ascii_lowercase();
    lower.contains("copilot").then(|| "copilot".to_string())
}

fn feedback_label(focus: &str) -> String {
    let mut chars = focus.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
        None => "Requested".to_string(),
    }
}

fn first_non_empty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(truncate_for_tool_result_answer)
}

fn structured_tool_result_rows(value: &Value, prompt: &str) -> Vec<StructuredToolRow> {
    match value {
        Value::Array(values) => values
            .iter()
            .filter_map(|value| structured_tool_row(value, prompt))
            .collect(),
        Value::Object(map) => {
            if let Some(row) = structured_tool_row(value, prompt) {
                return vec![row];
            }
            [
                "items",
                "results",
                "nodes",
                "data",
                "pullRequests",
                "mergeRequests",
                "issues",
                "tickets",
                "tasks",
                "workItems",
                "work_items",
            ]
            .iter()
            .filter_map(|key| map.get(*key))
            .flat_map(|value| structured_tool_result_rows(value, prompt))
            .collect()
        }
        _ => Vec::new(),
    }
}

fn structured_tool_row(value: &Value, prompt: &str) -> Option<StructuredToolRow> {
    let Value::Object(map) = value else {
        return None;
    };

    let title = first_string_field(map, &["title", "name", "subject", "summary"])?;
    let id = first_identifier_field(map, &["number", "id", "key"]);
    let url = first_string_field(map, &["html_url", "web_url", "permalink", "url"]);
    if id.is_none() && url.is_none() && !prompt_asks_for_work_items(prompt) {
        return None;
    }

    Some(StructuredToolRow {
        id,
        title,
        url,
        details: structured_tool_row_details(map),
    })
}

fn first_string_field(map: &serde_json::Map<String, Value>, fields: &[&str]) -> Option<String> {
    fields
        .iter()
        .find_map(|field| map.get(*field).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn first_identifier_field(map: &serde_json::Map<String, Value>, fields: &[&str]) -> Option<String> {
    fields.iter().find_map(|field| {
        let value = map.get(*field)?;
        match value {
            Value::Number(number) => Some(number.to_string()),
            Value::String(text) => {
                let trimmed = text.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            }
            _ => None,
        }
    })
}

fn structured_tool_row_details(map: &serde_json::Map<String, Value>) -> Vec<String> {
    let mut details = Vec::new();
    if let Some(state) = first_string_field(map, &["state", "status"]) {
        details.push(state);
    }
    if let Some(author) = map
        .get("author")
        .or_else(|| map.get("user"))
        .and_then(|value| value.get("login").or_else(|| value.get("name")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        details.push(format!("by {author}"));
    }
    if let Some(created) = first_string_field(map, &["createdAt", "created_at", "updatedAt"]) {
        details.push(created);
    }
    details
}

fn format_structured_tool_row(row: &StructuredToolRow, prompt: &str) -> String {
    let mut text = match row.id.as_deref() {
        Some(id)
            if prompt_asks_for_numbered_work_items(prompt)
                && id.chars().all(|ch| ch.is_ascii_digit()) =>
        {
            format!("#{id} - {}", row.title)
        }
        Some(id) => format!("{id} - {}", row.title),
        None => row.title.clone(),
    };
    if !row.details.is_empty() {
        text.push_str(" (");
        text.push_str(&row.details.join(", "));
        text.push(')');
    }
    if let Some(url) = &row.url {
        text.push_str(" - ");
        text.push_str(url);
    }
    truncate_for_tool_result_answer(&text)
}

fn truncate_for_tool_result_answer(text: &str) -> String {
    const LIMIT: usize = 900;
    if text.chars().count() <= LIMIT {
        return text.to_string();
    }

    let mut truncated = text.chars().take(LIMIT).collect::<String>();
    truncated.push_str("...");
    truncated
}

fn tool_evidence_should_be_preserved(user_text: &str) -> bool {
    let text = user_text.to_ascii_lowercase();
    contains_any(
        &text,
        &[
            "tool fact",
            "tool facts",
            "tool result",
            "tool output",
            "final recall",
            "include both",
            "include all",
            "answer with the tool",
            "return the tool",
        ],
    )
}

fn missing_tool_evidence_values(session: &Session, answer: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for (_, result) in session.recent_tool_results() {
        for value in short_tool_result_values(&result) {
            if !answer.contains(&value) && !missing.iter().any(|seen| seen == &value) {
                missing.push(value);
            }
        }
    }
    missing
}

fn short_tool_result_values(result: &str) -> Vec<String> {
    let Ok(parsed) = serde_json::from_str::<Value>(result) else {
        return Vec::new();
    };

    let mut values = Vec::new();
    collect_short_tool_result_values(&parsed, &mut values);
    values
}

fn collect_short_tool_result_values(value: &Value, values: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, nested) in map {
                if !tool_result_value_key_is_evidence(key) {
                    continue;
                }
                if let Some(scalar) = nested.as_str().filter(|s| short_exact_value(s)) {
                    values.push(scalar.to_string());
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_short_tool_result_values(item, values);
            }
        }
        _ => {}
    }
}

fn tool_result_value_key_is_evidence(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "value" | "fact" | "result" | "answer"
    )
}

fn short_exact_value(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && trimmed.len() <= 160 && !trimmed.contains('\n')
}

fn repeated_same_tool_results(session: &Session) -> Option<(String, usize)> {
    let calls = completed_tool_calls_since_latest_user_before_tool_result(session);
    let latest = calls.last()?;
    let count = calls
        .iter()
        .rev()
        .take_while(|call| call.same_signature(latest))
        .count();

    (count >= SAME_TOOL_FORCE_ANSWER_THRESHOLD).then(|| (latest.name.clone(), count))
}

fn prompt_requests_additional_tool_step(prompt: &str, session: &Session) -> bool {
    let completed = completed_tool_calls_since_latest_user_before_tool_result(session).len();
    if completed == 0 || completed >= 2 {
        return false;
    }

    let text = prompt.to_ascii_lowercase();
    contains_any(
        &text,
        &[
            "then run",
            "then execute",
            "then use",
            "then call",
            "after that run",
            "afterwards run",
            "also run",
            "and run",
        ],
    )
}

fn tool_budget_exhausted(session: &Session) -> Option<usize> {
    let count = completed_tool_names_since_latest_user_before_tool_result(session).len();
    (count >= TOOL_BUDGET_FORCE_ANSWER_THRESHOLD).then_some(count)
}

fn completed_tool_names_since_latest_user_before_tool_result(session: &Session) -> Vec<String> {
    completed_tool_calls_since_latest_user_before_tool_result(session)
        .into_iter()
        .map(|call| call.name)
        .collect()
}

#[derive(Clone, Debug)]
struct CompletedToolCall {
    name: String,
    arguments: String,
}

impl CompletedToolCall {
    fn same_signature(&self, other: &Self) -> bool {
        self.name == other.name && self.arguments == other.arguments
    }
}

fn completed_tool_calls_since_latest_user_before_tool_result(
    session: &Session,
) -> Vec<CompletedToolCall> {
    let all = session.all_messages();
    let Some(latest_tool_idx) = all.iter().rposition(|msg| message_role(msg) == "tool") else {
        return Vec::new();
    };
    let start_idx = all[..=latest_tool_idx]
        .iter()
        .rposition(message_is_task_user)
        .unwrap_or(0);

    let mut call_signatures = std::collections::HashMap::new();
    let mut completed = Vec::new();
    for msg in &all[start_idx..=latest_tool_idx] {
        match message_role(msg) {
            "assistant" => {
                let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) else {
                    continue;
                };
                for tool_call in tool_calls {
                    let Some(id) = tool_call.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    let Some(name) = tool_call.pointer("/function/name").and_then(Value::as_str)
                    else {
                        continue;
                    };
                    let arguments = tool_call
                        .pointer("/function/arguments")
                        .map(tool_call_arguments_signature)
                        .unwrap_or_default();
                    call_signatures.insert(
                        id.to_string(),
                        CompletedToolCall {
                            name: name.to_string(),
                            arguments,
                        },
                    );
                }
            }
            "tool" => {
                let Some(id) = msg.get("tool_call_id").and_then(Value::as_str) else {
                    continue;
                };
                if let Some(call) = call_signatures.get(id) {
                    completed.push(call.clone());
                }
            }
            _ => {}
        }
    }

    completed
}

fn tool_call_arguments_signature(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn latest_completed_tool_result(session: &Session) -> Option<(String, String)> {
    session.recent_tool_results().into_iter().last()
}

fn latest_completed_tool_call(session: &Session) -> Option<CompletedToolCall> {
    completed_tool_calls_since_latest_user_before_tool_result(session)
        .into_iter()
        .last()
}

fn retry_tool_result_response_if_plain(
    body: Value,
    retry_tool_call: Option<&CompletedToolCall>,
    latest_tool_result: Option<&str>,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
    tools_enabled: bool,
) -> Value {
    if !tools_enabled {
        return body;
    }
    match (first_response_tool_call(&body), retry_tool_call) {
        (Some(response_call), Some(retry_call))
            if response_call.same_signature(retry_call)
                && !same_tool_call_retry_allowed(latest_tool_result) =>
        {
            if let Some(corrected) = corrected_retry_tool_call(
                retry_call,
                latest_tool_result,
                tools,
                allowed_tools,
                prompt_tool_profiles,
            ) {
                let arguments = Value::String(corrected.arguments);
                return tool_call_response(&corrected.name, &arguments);
            }
            tracing::info!(
                "moa: suppressing repeated failing {} tool call with unchanged arguments",
                retry_call.name
            );
            return error_response(
                "MoA refused to repeat an identical failing tool call",
                MOA_ERR_NO_USABLE_ANSWER,
            );
        }
        _ => {}
    }
    if let Some(response_call) = first_response_tool_call(&body) {
        if let Some(corrected) = corrected_retry_tool_call(
            &response_call,
            latest_tool_result,
            tools,
            allowed_tools,
            prompt_tool_profiles,
        ) {
            tracing::info!(
                "moa: reducer returned malformed follow-up {}; retrying corrected tool call",
                response_call.name
            );
            let arguments = Value::String(corrected.arguments);
            return tool_call_response(&corrected.name, &arguments);
        }
        return body;
    }
    retry_tool_call
        .and_then(|call| {
            retry_tool_call_response(
                call,
                latest_tool_result,
                tools,
                allowed_tools,
                prompt_tool_profiles,
            )
        })
        .unwrap_or(body)
}

fn retry_tool_call_response(
    call: &CompletedToolCall,
    latest_tool_result: Option<&str>,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<Value> {
    if let Some(corrected) = corrected_retry_tool_call(
        call,
        latest_tool_result,
        tools,
        allowed_tools,
        prompt_tool_profiles,
    ) {
        tracing::info!(
            "moa: reducer returned plain text after non-answerable {}; retrying corrected tool call",
            call.name
        );
        let arguments = Value::String(corrected.arguments);
        return Some(tool_call_response(&corrected.name, &arguments));
    }

    if !same_tool_call_retry_allowed(latest_tool_result) {
        tracing::info!(
            "moa: reducer returned plain text after non-answerable {}; not retrying identical failing call",
            call.name
        );
        return None;
    }

    tracing::info!(
        "moa: reducer returned plain text after non-answerable {}; retrying same tool call",
        call.name
    );
    Some(tool_call_response(
        &call.name,
        &Value::String(call.arguments.clone()),
    ))
}

fn same_tool_call_retry_allowed(latest_tool_result: Option<&str>) -> bool {
    latest_tool_result.is_some_and(|result| plain_text_tool_result_looks_empty(result.trim()))
}

fn corrected_retry_tool_call(
    call: &CompletedToolCall,
    latest_tool_result: Option<&str>,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<CompletedToolCall> {
    let result = latest_tool_result?;
    let candidates = command_tool_candidates(tools, allowed_tools, prompt_tool_profiles);
    let candidate = candidates
        .iter()
        .find(|candidate| candidate.name == call.name)?;
    let mut arguments = serde_json::from_str::<Value>(&call.arguments).ok()?;
    let command = arguments
        .get(&candidate.field)
        .and_then(Value::as_str)
        .map(str::to_string)?;
    let corrected = tool_result_suggests_unquoted_shell_token_error(result)
        .then(|| quote_unquoted_shell_metachar_tokens(&command))
        .flatten()
        .or_else(|| remove_unknown_cli_flags_from_command(&command, result))
        .or_else(|| unescape_single_quoted_shell_double_quotes(&command, result))?;
    if corrected == command {
        return None;
    }
    arguments[&candidate.field] = Value::String(corrected);
    let sanitized = sanitize_tool_arguments_for_tool(&call.name, &arguments, tools).ok()?;
    let wire = tool_arguments_wire_string(&sanitized);
    if wire == call.arguments {
        return None;
    }
    Some(CompletedToolCall {
        name: call.name.clone(),
        arguments: wire,
    })
}

fn tool_result_suggests_unquoted_shell_token_error(result: &str) -> bool {
    let lower = result.to_ascii_lowercase();
    lower.contains("parse error near")
        || lower.contains("syntax error near")
        || lower.contains("no matches found")
}

fn unescape_single_quoted_shell_double_quotes(command: &str, result: &str) -> Option<String> {
    if !tool_result_suggests_shell_quoting_issue(result) {
        return None;
    }

    let mut changed = false;
    let mut out = String::with_capacity(command.len());
    let mut quote: Option<char> = None;
    let mut chars = command.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\'' | '"' if quote == Some(ch) => {
                quote = None;
                out.push(ch);
            }
            '\'' | '"' if quote.is_none() => {
                quote = Some(ch);
                out.push(ch);
            }
            '\\' if quote == Some('\'') && chars.peek() == Some(&'"') => {
                changed = true;
                out.push('"');
                chars.next();
            }
            _ => out.push(ch),
        }
    }

    changed.then_some(out)
}

fn tool_result_suggests_shell_quoting_issue(result: &str) -> bool {
    let lower = result.to_ascii_lowercase();
    lower.contains("unix shell quoting issues")
        || (lower.contains("syntax error") && lower.contains("jq:"))
}

fn remove_unknown_cli_flags_from_command(command: &str, result: &str) -> Option<String> {
    let allowed = allowed_cli_long_flags_from_usage(result)?;
    let tokens = split_shell_tokens(command)?;
    let mut changed = false;
    let mut filtered = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        match shell_long_flag_name(token) {
            Some(flag) if !allowed.contains(flag) => {
                changed = true;
                index += unknown_flag_token_span(&tokens, index);
                continue;
            }
            _ => {}
        }
        filtered.push(token.clone());
        index += 1;
    }
    changed.then(|| join_shell_tokens(&filtered))
}

fn allowed_cli_long_flags_from_usage(result: &str) -> Option<HashSet<String>> {
    let lower = result.to_ascii_lowercase();
    if !lower
        .lines()
        .take(4)
        .any(|line| line.contains("unknown flag:"))
        || !lower.contains("\nusage:")
    {
        return None;
    }

    let flags_section = result
        .lines()
        .skip_while(|line| line.trim() != "Flags:")
        .skip(1)
        .collect::<Vec<_>>()
        .join("\n");
    let allowed = flags_section
        .split(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ';' | ')' | '('))
        .filter_map(|token| token.strip_prefix("--"))
        .map(|token| {
            token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-' && ch != '_')
        })
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect::<HashSet<_>>();
    (!allowed.is_empty()).then_some(allowed)
}

fn shell_long_flag_name(token: &str) -> Option<&str> {
    token
        .strip_prefix("--")
        .map(|flag| flag.split_once('=').map_or(flag, |(name, _)| name))
        .filter(|flag| !flag.is_empty())
}

fn unknown_flag_token_span(tokens: &[String], index: usize) -> usize {
    let token = &tokens[index];
    if token.contains('=') {
        return 1;
    }
    let Some(next) = tokens.get(index + 1) else {
        return 1;
    };
    if next.starts_with('-') { 1 } else { 2 }
}

fn split_shell_tokens(command: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut quote: Option<char> = None;

    for ch in command.chars() {
        match ch {
            '\'' | '"' if quote == Some(ch) => quote = None,
            '\'' | '"' if quote.is_none() => quote = Some(ch),
            ch if ch.is_whitespace() && quote.is_none() => {
                if !token.is_empty() {
                    tokens.push(std::mem::take(&mut token));
                }
            }
            _ => token.push(ch),
        }
    }
    if quote.is_some() {
        return None;
    }
    if !token.is_empty() {
        tokens.push(token);
    }
    (!tokens.is_empty()).then_some(tokens)
}

fn join_shell_tokens(tokens: &[String]) -> String {
    tokens
        .iter()
        .map(|token| {
            if shell_token_needs_join_quotes(token) {
                format!("'{}'", token.replace('\'', r#"'"'"'"#))
            } else {
                token.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_token_needs_join_quotes(token: &str) -> bool {
    token.is_empty()
        || token
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '&' | '|' | ';' | '<' | '>' | '(' | ')'))
}

fn quote_unquoted_shell_metachar_tokens(command: &str) -> Option<String> {
    let mut changed = false;
    let mut out = String::with_capacity(command.len() + 8);
    let mut token = String::new();
    let mut quote: Option<char> = None;

    for ch in command.chars() {
        match ch {
            '\'' | '"' if quote == Some(ch) => {
                quote = None;
                token.push(ch);
            }
            '\'' | '"' if quote.is_none() => {
                quote = Some(ch);
                token.push(ch);
            }
            ch if ch.is_whitespace() && quote.is_none() => {
                flush_shell_token(&mut out, &mut token, &mut changed);
                out.push(ch);
            }
            _ => token.push(ch),
        }
    }
    flush_shell_token(&mut out, &mut token, &mut changed);

    changed.then_some(out)
}

fn flush_shell_token(out: &mut String, token: &mut String, changed: &mut bool) {
    if token.is_empty() {
        return;
    }
    if shell_token_needs_single_quotes(token) {
        *changed = true;
        out.push('\'');
        out.push_str(&token.replace('\'', r#"'"'"'"#));
        out.push('\'');
    } else {
        out.push_str(token);
    }
    token.clear();
}

fn shell_token_needs_single_quotes(token: &str) -> bool {
    token.chars().any(|ch| matches!(ch, '&' | '?' | '*' | '['))
        && !shell_token_already_quoted(token)
}

fn shell_token_already_quoted(token: &str) -> bool {
    (token.starts_with('\'') && token.ends_with('\''))
        || (token.starts_with('"') && token.ends_with('"'))
}

fn first_response_tool_call(body: &Value) -> Option<CompletedToolCall> {
    let call = body
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)?
        .first()?;
    let name = call.pointer("/function/name").and_then(Value::as_str)?;
    let arguments = call
        .pointer("/function/arguments")
        .map(tool_call_arguments_signature)
        .unwrap_or_default();
    Some(CompletedToolCall {
        name: name.to_string(),
        arguments,
    })
}

fn tool_result_has_answerable_evidence(result: &str) -> bool {
    let trimmed = result.trim();
    if trimmed.is_empty() || matches!(trimmed, "{}" | "[]") {
        return false;
    }
    if plain_text_tool_result_looks_empty(trimmed) {
        return false;
    }

    if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
        return json_value_has_answerable_evidence(&parsed);
    }

    !plain_text_tool_result_looks_like_error(trimmed)
}

fn tool_result_has_answerable_evidence_for_prompt(result: &str, prompt: &str) -> bool {
    if !tool_result_has_answerable_evidence(result) {
        return false;
    }
    if prompt_asks_for_ci_status(prompt) && !tool_result_has_ci_status_evidence(result) {
        return false;
    }
    if prompt_asks_for_feedback(prompt) && !tool_result_has_feedback_evidence(result, prompt) {
        return false;
    }
    if prompt_asks_for_work_items(prompt) && plain_text_tool_result_looks_like_auth_status(result) {
        return false;
    }
    if prompt_asks_for_work_items(prompt) && plain_text_tool_result_looks_like_git_remotes(result) {
        return false;
    }
    if prompt_asks_for_work_items(prompt)
        && plain_text_tool_result_looks_like_git_repository_metadata(result)
    {
        return false;
    }
    if plain_text_tool_result_looks_like_repository_list(result)
        && !prompt_asks_for_repositories(prompt)
    {
        return false;
    }
    true
}

fn prompt_asks_for_ci_status(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "ci",
            "checks",
            "green",
            "status check",
            "check run",
            "check suite",
            "build status",
        ],
    )
}

fn prompt_asks_for_feedback(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "feedback",
            "comment",
            "comments",
            "review comment",
            "review comments",
            "reviews",
        ],
    )
}

fn tool_result_has_ci_status_evidence(result: &str) -> bool {
    if let Some(parsed) = parse_tool_result_json(result.trim()) {
        return json_value_has_ci_status_evidence(&parsed);
    }

    let lower = result.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "statuscheckrollup",
            "checkrun",
            "check suite",
            "checks:",
            "ci/",
            "\tpass",
            "\tfail",
            "\tsuccess",
            "\tfailure",
            " pending",
            " queued",
            " in_progress",
        ],
    )
}

fn tool_result_has_feedback_evidence(result: &str, prompt: &str) -> bool {
    if let Some(parsed) = parse_tool_result_json(result.trim()) {
        return json_value_has_feedback_evidence(&parsed, prompt);
    }

    let lower = result.to_ascii_lowercase();
    let focus = feedback_focus_from_prompt(prompt);
    let focus_matches = focus.as_deref().is_some_and(|focus| lower.contains(focus));
    contains_any(
        &lower,
        &[
            "feedback",
            "comment",
            "comments",
            "review",
            "reviews",
            "no feedback",
            "no comments",
            "no reviews",
        ],
    ) && (focus.is_none() || focus_matches)
}

fn json_value_has_ci_status_evidence(value: &Value) -> bool {
    let mut checks = Vec::new();
    collect_structured_check_items(value, &mut checks);
    !checks.is_empty()
}

fn json_value_has_feedback_evidence(value: &Value, prompt: &str) -> bool {
    let focus = feedback_focus_from_prompt(prompt);
    match value {
        Value::Object(map) => {
            if FEEDBACK_COLLECTION_KEYS
                .iter()
                .any(|key| map.contains_key(*key))
                || structured_object_looks_like_feedback_entry(map)
            {
                return true;
            }
            map.values()
                .any(|value| json_value_has_feedback_evidence(value, prompt))
        }
        Value::Array(values) => values
            .iter()
            .any(|value| json_value_has_feedback_evidence(value, prompt)),
        Value::String(text) => focus
            .as_deref()
            .is_some_and(|focus| text.to_ascii_lowercase().contains(focus)),
        _ => false,
    }
}

fn plain_text_tool_result_looks_empty(result: &str) -> bool {
    let normalized = result.replace('\r', "");
    let lines: Vec<&str> = normalized
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    matches!(
        lines.as_slice(),
        ["(no output)" | "(empty output)" | "no output" | "empty output"]
    )
}

fn prompt_asks_for_work_items(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    let asks_specific_work_items = prompt_has_word(&lower, "pr")
        || prompt_has_word(&lower, "prs")
        || prompt_has_word(&lower, "pulls")
        || lower.contains("pull request")
        || lower.contains("pull requests")
        || lower.contains("merge request")
        || lower.contains("merge requests")
        || lower.contains("issue")
        || lower.contains("issues")
        || lower.contains("ticket")
        || lower.contains("tickets")
        || lower.contains("task")
        || lower.contains("tasks")
        || lower.contains("bug")
        || lower.contains("bugs")
        || lower.contains("work item")
        || lower.contains("work items");
    let asks_listing = contains_any(
        &lower,
        &[
            "list",
            "find",
            "show",
            "check",
            "get",
            "recent",
            "open",
            "important",
            "interesting",
            "status",
        ],
    );
    asks_specific_work_items && asks_listing
}

fn prompt_asks_for_numbered_work_items(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    prompt_has_word(&lower, "pr")
        || prompt_has_word(&lower, "prs")
        || prompt_has_word(&lower, "pulls")
        || lower.contains("pull request")
        || lower.contains("merge request")
        || lower.contains("issue")
        || lower.contains("ticket")
        || lower.contains("task")
        || lower.contains("work item")
}

fn prompt_asks_for_repositories(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    prompt_has_word(&lower, "repo")
        || prompt_has_word(&lower, "repos")
        || lower.contains("repository")
        || lower.contains("repositories")
}

fn prompt_has_word(prompt: &str, needle: &str) -> bool {
    prompt
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|token| token == needle)
}

fn plain_text_tool_result_looks_like_auth_status(result: &str) -> bool {
    let normalized = result.replace('\r', "");
    let lower = normalized.trim_start().to_ascii_lowercase();
    let first_line = lower.lines().next().unwrap_or("").trim();
    let first_line_looks_like_service =
        first_line.contains('.') && !first_line.contains(char::is_whitespace);
    first_line_looks_like_service
        && (lower.contains("logged in to") || lower.contains("authenticated"))
        && (lower.contains("active account") || lower.contains("account:"))
}

fn plain_text_tool_result_looks_like_repository_list(result: &str) -> bool {
    let rows = result
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let mut columns = line.split('\t').map(str::trim);
            Some((columns.next()?, columns.next()?, columns.next()?))
        });
    rows.take(3).any(|(repo, _description, visibility)| {
        repository_slug_looks_like_owner_name(repo)
            && (visibility.contains("public") || visibility.contains("private"))
    })
}

fn repository_slug_looks_like_owner_name(value: &str) -> bool {
    let mut parts = value.split('/');
    let Some(owner) = parts.next() else {
        return false;
    };
    let Some(name) = parts.next() else {
        return false;
    };
    parts.next().is_none()
        && !owner.is_empty()
        && !name.is_empty()
        && owner.chars().all(repository_slug_char)
        && name.chars().all(repository_slug_char)
}

fn repository_slug_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.')
}

fn plain_text_tool_result_looks_like_git_remotes(result: &str) -> bool {
    result.lines().map(str::trim).any(|line| {
        let mut columns = line.split('\t').map(str::trim);
        matches!(columns.next(), Some("origin" | "upstream"))
            && columns.next().is_some_and(looks_like_git_remote_location)
    })
}

fn plain_text_tool_result_looks_like_git_repository_metadata(result: &str) -> bool {
    let lines: Vec<&str> = result
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(12)
        .collect();
    if lines.is_empty() {
        return false;
    }
    let fetch_or_ref_lines = lines
        .iter()
        .filter(|line| git_metadata_status_line(line) || git_ref_listing_line(line))
        .count();
    let revision_lines = lines
        .iter()
        .filter(|line| git_revision_listing_line(line))
        .count();
    fetch_or_ref_lines > 0 || revision_lines >= 2 || revision_lines == lines.len()
}

fn git_metadata_status_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower == "already up to date."
        || lower.contains("fetch_head")
        || lower.starts_with("from https://")
        || lower.starts_with("from http://")
        || lower.starts_with("from git@")
        || lower.starts_with("* branch ")
}

fn git_ref_listing_line(line: &str) -> bool {
    let mut columns = line.split('\t').map(str::trim);
    let Some(first) = columns.next() else {
        return false;
    };
    let Some(second) = columns.next() else {
        return false;
    };
    first.len() >= 7
        && first.chars().all(|ch| ch.is_ascii_hexdigit())
        && (second.starts_with("refs/heads/")
            || second.starts_with("refs/tags/")
            || second.starts_with("refs/remotes/"))
}

fn git_revision_listing_line(line: &str) -> bool {
    let mut parts = line.splitn(2, char::is_whitespace);
    let Some(revision) = parts.next() else {
        return false;
    };
    parts.next().is_some_and(|rest| !rest.trim().is_empty())
        && (7..=40).contains(&revision.len())
        && revision.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn looks_like_git_remote_location(remote: &str) -> bool {
    remote.contains("://")
        || remote.contains('@')
        || remote.contains(".git")
        || remote.contains(" (fetch)")
        || remote.contains(" (push)")
}

fn plain_text_tool_result_looks_like_error(result: &str) -> bool {
    let normalized = result.replace('\r', "");
    let lower = normalized.trim_start().to_ascii_lowercase();
    let first_line = lower.lines().next().unwrap_or("").trim();
    first_line.starts_with("unknown flag:")
        || first_line.starts_with("unknown json field:")
        || first_line.starts_with("unknown command")
        || first_line.contains("parse error")
        || first_line.contains("syntax error")
        || first_line.contains("no matches found")
        || first_line == "no git remotes found"
        || first_line.contains("not available or not authenticated")
        || first_line.starts_with("error:")
        || first_line.starts_with("fatal:")
        || first_line.starts_with("graphql:")
        || (first_line.starts_with("unknown ") && lower.contains("\navailable fields:"))
        || (lower.contains("\nusage:") && lower.lines().take(3).any(cli_error_line))
}

fn cli_error_line(line: &str) -> bool {
    let line = line.trim();
    line.starts_with("unknown ")
        || line.starts_with("error:")
        || line.starts_with("fatal:")
        || line.contains("invalid")
}

fn json_value_has_answerable_evidence(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(_) | Value::Number(_) => true,
        Value::String(text) => !text.trim().is_empty(),
        Value::Array(values) => values.iter().any(json_value_has_answerable_evidence),
        Value::Object(map) => {
            if map
                .get("status")
                .and_then(Value::as_str)
                .is_some_and(|status| status.eq_ignore_ascii_case("error"))
            {
                return false;
            }
            map.values().any(json_value_has_answerable_evidence)
        }
    }
}

fn append_tool_loop_answer_instruction(messages: &mut [Value], tool: &str, count: usize) {
    let instruction = format!(
        "\n\nTool loop guard: the last {count} completed tool calls all used `{tool}`. \
         Do not call `{tool}` again. If a different declared tool can materially advance the \
         request, use that tool. Otherwise answer now from the gathered tool results; if the \
         evidence is incomplete, say what can be determined and what is missing."
    );
    append_system_instruction(messages, &instruction);
}

fn append_tool_budget_instruction(messages: &mut [Value], count: usize) {
    let instruction = format!(
        "\n\nTool budget guard: {count} completed tool calls are already in context. \
         Do not call another tool unless the declared tool schema and current evidence make \
         the next call clearly necessary. Prefer answering now from gathered tool results; \
         if evidence is incomplete, ambiguous, or conflicts with the user's premise, say that \
         directly instead of inventing missing facts."
    );
    append_system_instruction(messages, &instruction);
}

fn append_tool_error_retry_instruction(messages: &mut [Value], tool: &str) {
    let instruction = format!(
        "\n\nTool retry guard: the latest `{tool}` result appears to be command error or usage text, \
         not answer evidence. If the user's request still needs data, call a corrected tool once. \
         Do not answer from usage or error text."
    );
    append_system_instruction(messages, &instruction);
}

fn append_system_instruction(messages: &mut [Value], instruction: &str) {
    let system_content = messages
        .iter_mut()
        .find(|msg| msg.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|system| {
            let content = system.get("content").and_then(Value::as_str)?.to_string();
            Some((system, content))
        });
    if let Some((system, content)) = system_content {
        let mut updated = content.to_string();
        updated.push_str(instruction);
        system["content"] = Value::String(updated);
    }
}

// ─── Decision resolution ─────────────────────────────────────────────

/// Returns (response body, reducer_used, reducer_attempts).
async fn resolve_decision(
    config: &GatewayConfig,
    request: DecisionResolution<'_>,
) -> (Value, bool, u32) {
    let DecisionResolution {
        session,
        decision,
        outputs,
        has_tools,
        selected_tool_names,
        forced_tool,
        allowed_tools,
        prompt_tool_profiles,
    } = request;
    let tools_available = !allowed_tools.is_empty();
    let response_tool_names = response_tool_names(selected_tool_names, allowed_tools);

    match decision {
        arbiter::Decision::Answer(text) => {
            if let Some(tool) = forced_tool.filter(|_| tools_available) {
                (
                    tool_call_response(&tool.name, &tool.fallback_arguments),
                    false,
                    0,
                )
            } else {
                (
                    chat_or_schema_command_tool_response(
                        &text,
                        session.tools(),
                        response_tool_names,
                        prompt_tool_profiles,
                        Some(&session.last_user_text()),
                    ),
                    false,
                    0,
                )
            }
        }
        arbiter::Decision::ToolCall { name, arguments } => {
            if tools_available {
                (tool_call_response(&name, &arguments), false, 0)
            } else {
                (
                    error_response(
                        "MoA selected a tool call, but tools are disabled for this turn",
                        MOA_ERR_NO_USABLE_ANSWER,
                    ),
                    false,
                    0,
                )
            }
        }
        arbiter::Decision::NeedsReducer { reason } => {
            tracing::info!("moa: reducer — {reason}");
            let candidates = reducer_candidates(config);
            let (messages, tools) = context::pack_for_reducer_selected(
                session,
                outputs,
                &reason,
                has_tools,
                selected_tool_names,
            );

            // Hedged ladder over the ordered candidates (see hedged_reducer_call).
            let hedge_result = hedged_reducer_call(
                &config.backends,
                candidates,
                messages,
                tools,
                config.reducer_timeout,
                config.hedge_delay,
                config.enable_thinking,
            )
            .await;

            let (attempts, chosen): (u32, Option<normalize::WorkerOutput>) = match hedge_result {
                Ok(reducer::HedgedReducerOk {
                    winner,
                    text,
                    attempts: spawned,
                }) => {
                    let mut reduced =
                        normalize::normalize_worker_output(&text, &winner, WorkerRole::Reducer, 0);
                    enforce_tool_call_contract(
                        &mut reduced,
                        response_tool_names,
                        session.tools(),
                        &winner,
                    );
                    (spawned, Some(reduced))
                }
                Err(reducer::HedgedReducerErr {
                    err: _,
                    attempts: spawned,
                }) => (spawned, None),
            };

            match chosen {
                Some(reduced) => match reduced.kind {
                    normalize::OutputKind::ToolProposal => {
                        // See the matching block in `handle_tool_result`:
                        // emit `tool_calls` whenever `tool_name` is present,
                        // defaulting `arguments` to `{}` via
                        // `tool_call_response`. Agent harnesses key on
                        // `tool_calls` rather than scanning prose, so the
                        // previous "both name AND args required" gate would
                        // silently fall back to a chat_response and break
                        // the calling agent's tool loop.
                        (
                            tool_proposal_response(
                                &reduced,
                                tools_available,
                                session.tools(),
                                response_tool_names,
                                prompt_tool_profiles,
                                Some(&session.last_user_text()),
                            ),
                            true,
                            attempts,
                        )
                    }
                    normalize::OutputKind::Uncertainty => {
                        if let Some(tool) = forced_tool.filter(|_| tools_available) {
                            (
                                tool_call_response(&tool.name, &tool.fallback_arguments),
                                true,
                                attempts,
                            )
                        } else {
                            (
                                fallback_worker_response(
                                    outputs,
                                    session,
                                    session.tools(),
                                    response_tool_names,
                                    prompt_tool_profiles,
                                ),
                                true,
                                attempts,
                            )
                        }
                    }
                    _ => {
                        if let Some(tool) = forced_tool.filter(|_| tools_available) {
                            (
                                tool_call_response(&tool.name, &tool.fallback_arguments),
                                true,
                                attempts,
                            )
                        } else {
                            (
                                chat_or_schema_command_tool_response(
                                    &reduced.payload,
                                    session.tools(),
                                    response_tool_names,
                                    prompt_tool_profiles,
                                    Some(&session.last_user_text()),
                                ),
                                true,
                                attempts,
                            )
                        }
                    }
                },
                None => {
                    tracing::warn!("moa: all reducer candidates failed, using best worker");
                    // reducer_used=false here because the reducer did NOT
                    // produce the output we're returning — we fell back to
                    // a worker. attempts still reflects what was spawned so
                    // observability can see "we tried N times and all failed".
                    if let Some(tool) = forced_tool.filter(|_| tools_available) {
                        (
                            tool_call_response(&tool.name, &tool.fallback_arguments),
                            false,
                            attempts,
                        )
                    } else {
                        (
                            fallback_worker_response(
                                outputs,
                                session,
                                session.tools(),
                                response_tool_names,
                                prompt_tool_profiles,
                            ),
                            false,
                            attempts,
                        )
                    }
                }
            }
        }
    }
}

// ─── Response builders ───────────────────────────────────────────────

fn best_answer(outputs: &[WorkerOutput]) -> String {
    outputs
        .iter()
        .filter(|o| {
            matches!(o.kind, normalize::OutputKind::Answer)
                && !normalize::is_silent_reply_sentinel(&o.payload)
        })
        // `total_cmp` is total over all f32 (including NaN/Inf); `partial_cmp`
        // can return `None` on NaN, which would panic on `unwrap`.
        // `normalize_worker_output` now sanitizes non-finite confidences
        // before they reach here, but using `total_cmp` keeps this site
        // panic-free even if a future caller skips the normalizer.
        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
        .map(|o| o.payload.clone())
        .unwrap_or_default()
}

fn fallback_worker_response(
    outputs: &[WorkerOutput],
    session: &Session,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Value {
    let answer = best_answer(outputs);
    if answer.is_empty() {
        error_response(
            "MoA could not produce a usable answer",
            MOA_ERR_NO_USABLE_ANSWER,
        )
    } else {
        chat_or_schema_command_tool_response(
            &answer,
            tools,
            allowed_tools,
            prompt_tool_profiles,
            Some(&session.last_user_text()),
        )
    }
}

fn response_tool_names<'a>(
    selected_tool_names: &'a [String],
    allowed_tools: &'a [String],
) -> &'a [String] {
    if selected_tool_names.is_empty() {
        allowed_tools
    } else {
        selected_tool_names
    }
}

fn tool_proposal_response(
    output: &WorkerOutput,
    has_tools: bool,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
    user_text: Option<&str>,
) -> Value {
    if let (true, Some(name)) = (has_tools, output.tool_name.as_ref()) {
        let args = output.tool_arguments.as_ref().unwrap_or(&Value::Null);
        match validate_response_tool_arguments(
            name,
            args,
            tools,
            allowed_tools,
            prompt_tool_profiles,
        ) {
            Ok(arguments) => return tool_call_response(name, &arguments),
            Err(err) => tracing::info!("moa: tool proposal {name} rejected: {err}"),
        }
    }

    if output.payload.trim().is_empty() || normalize::is_silent_reply_sentinel(&output.payload) {
        return error_response(
            "MoA reducer returned no usable answer",
            MOA_ERR_NO_USABLE_ANSWER,
        );
    }

    if has_tools {
        return chat_or_schema_command_tool_response(
            &output.payload,
            tools,
            allowed_tools,
            prompt_tool_profiles,
            user_text,
        );
    }

    chat_response(&output.payload)
}

fn chat_or_schema_command_tool_response(
    content: &str,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
    user_text: Option<&str>,
) -> Value {
    let command_tool = if fenced_command_tool_conversion_allowed(user_text) {
        extract_shell_command_block(content).and_then(|command| {
            command_tool_choice_from_command(&command, tools, allowed_tools, prompt_tool_profiles)
        })
    } else {
        None
    };
    if let Some(tool) = command_tool {
        return tool_call_response(&tool.name, &tool.fallback_arguments);
    }
    if let Some(tool) =
        inline_json_tool_choice_from_text(content, tools, allowed_tools, prompt_tool_profiles)
    {
        return tool_call_response(&tool.name, &tool.fallback_arguments);
    }
    if let Some(tool) =
        xml_tool_choice_from_text(content, tools, allowed_tools, prompt_tool_profiles)
    {
        return tool_call_response(&tool.name, &tool.fallback_arguments);
    }
    if let Some(tool) =
        explicitly_named_tool_choice_from_text(content, tools, allowed_tools, prompt_tool_profiles)
    {
        return tool_call_response(&tool.name, &tool.fallback_arguments);
    }

    chat_response(content)
}

fn fenced_command_tool_conversion_allowed(user_text: Option<&str>) -> bool {
    let Some(user_text) = user_text else {
        return false;
    };
    let text = latest_user_request_tail(user_text).to_ascii_lowercase();
    if contains_any(
        &text,
        &[
            "don't run",
            "do not run",
            "dont run",
            "don't execute",
            "do not execute",
            "dont execute",
            "don't use",
            "do not use",
            "dont use",
        ],
    ) {
        return false;
    }

    let tokens = lexical_tokens(&text);
    command_execution_pattern_matches(&tokens)
        || contains_any(
            &text,
            &[
                "run the command",
                "run this command",
                "run that command",
                "execute the command",
                "execute this command",
                "execute that command",
                "use the command",
                "use this command",
                "use that command",
            ],
        )
}

fn inline_json_tool_choice_from_text(
    content: &str,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<ForcedToolChoice> {
    for json_text in embedded_json_object_candidates(content) {
        let Ok(value) = serde_json::from_str::<Value>(&json_text) else {
            continue;
        };
        let Some((name, raw_arguments)) = extract_tool_name_and_arguments(&value) else {
            continue;
        };
        if !declared_response_tool_names(tools, allowed_tools, prompt_tool_profiles)
            .iter()
            .any(|tool| tool == name)
        {
            continue;
        }
        let arguments = normalize_tool_arguments(raw_arguments)
            .map(Value::Object)
            .unwrap_or_else(|| json!({}));
        match validate_response_tool_arguments(
            name,
            &arguments,
            tools,
            allowed_tools,
            prompt_tool_profiles,
        ) {
            Ok(arguments) => {
                return Some(ForcedToolChoice {
                    name: name.to_string(),
                    fallback_arguments: arguments,
                });
            }
            Err(err) => {
                tracing::info!("moa: inline JSON tool {name} could not be sanitized: {err}");
            }
        }
    }
    None
}

fn xml_tool_choice_from_text(
    content: &str,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<ForcedToolChoice> {
    let invoke_start = content.find("<invoke")?;
    let invoke_tag_end = content[invoke_start..].find('>')? + invoke_start;
    let invoke_tag = &content[invoke_start..=invoke_tag_end];
    let body_start = invoke_tag_end + 1;
    let body_end = content[body_start..]
        .find("</invoke>")
        .map(|idx| body_start + idx)
        .unwrap_or(content.len());
    let body = &content[body_start..body_end];
    let mut arguments = xml_parameter_arguments(body);
    let raw_name = xml_attr_value(invoke_tag, "name").or_else(|| xml_body_tool_name(body))?;
    let name = resolve_response_tool_name(
        &raw_name,
        &mut arguments,
        tools,
        allowed_tools,
        prompt_tool_profiles,
    )?;
    match validate_response_tool_arguments(
        &name,
        &arguments,
        tools,
        allowed_tools,
        prompt_tool_profiles,
    ) {
        Ok(arguments) => Some(ForcedToolChoice {
            name,
            fallback_arguments: arguments,
        }),
        Err(err) => {
            tracing::info!("moa: XML tool choice could not be sanitized: {err}");
            None
        }
    }
}

fn resolve_response_tool_name(
    raw_name: &str,
    arguments: &mut Value,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<String> {
    let declared = declared_response_tool_names(tools, allowed_tools, prompt_tool_profiles);
    if declared.iter().any(|tool| tool == raw_name) {
        return Some(raw_name.to_string());
    }

    let command = arguments
        .get("command")
        .and_then(Value::as_str)?
        .to_string();
    if command.trim().is_empty() {
        return None;
    }
    let candidates = command_tool_candidates(tools, allowed_tools, prompt_tool_profiles);
    let [candidate] = candidates.as_slice() else {
        return None;
    };
    if let (true, Some(map)) = (candidate.field != "command", arguments.as_object_mut()) {
        map.insert(candidate.field.clone(), Value::String(command));
    }
    Some(candidate.name.clone())
}

fn xml_body_tool_name(body: &str) -> Option<String> {
    let marker = "tool:";
    let marker_start = body.find(marker)? + marker.len();
    let rest = body[marker_start..].trim_start();
    let name: String = rest
        .chars()
        .take_while(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        .collect();
    (!name.is_empty()).then_some(name)
}

fn xml_attr_value(tag: &str, attr: &str) -> Option<String> {
    xml_quoted_attr_value(tag, attr, '"').or_else(|| xml_quoted_attr_value(tag, attr, '\''))
}

fn xml_quoted_attr_value(tag: &str, attr: &str, quote: char) -> Option<String> {
    let needle = format!("{attr}={quote}");
    let value_start = tag.find(&needle)? + needle.len();
    let value_end = tag[value_start..].find(quote)? + value_start;
    Some(xml_text_unescape(&tag[value_start..value_end]))
}

fn xml_parameter_arguments(body: &str) -> Value {
    let mut map = serde_json::Map::new();
    let mut cursor = 0;
    while let Some((parameter_rel, tag_prefix)) = next_xml_parameter_tag(&body[cursor..]) {
        let parameter_start = cursor + parameter_rel;
        let Some(tag_end_rel) = body[parameter_start..].find('>') else {
            break;
        };
        let tag_end = parameter_start + tag_end_rel;
        let tag = &body[parameter_start..=tag_end];
        let Some(name) = xml_attr_value(tag, "name") else {
            cursor = tag_end + 1;
            continue;
        };

        let value_start = tag_end + 1;
        let value_end = xml_parameter_value_end(body, value_start, &name, tag_prefix);
        let value = xml_text_unescape(body[value_start..value_end].trim());
        map.insert(name, Value::String(value));
        cursor = value_end;
    }

    Value::Object(map)
}

fn next_xml_parameter_tag(body: &str) -> Option<(usize, &'static str)> {
    [("<parameter ", "parameter"), ("<param ", "param")]
        .into_iter()
        .filter_map(|(needle, tag_prefix)| body.find(needle).map(|idx| (idx, tag_prefix)))
        .min_by_key(|(idx, _)| *idx)
}

fn xml_parameter_value_end(body: &str, value_start: usize, name: &str, tag_prefix: &str) -> usize {
    let named_close = format!("</{name}>");
    let tag_close = format!("</{tag_prefix}>");
    [
        "</parameter>",
        "</param>",
        tag_close.as_str(),
        named_close.as_str(),
        "<parameter ",
        "<param ",
    ]
    .iter()
    .filter_map(|needle| {
        body[value_start..]
            .find(needle)
            .map(|idx| value_start + idx)
    })
    .min()
    .unwrap_or(body.len())
}

fn xml_text_unescape(text: &str) -> String {
    let mut decoded = text.to_string();
    for _ in 0..3 {
        let next = decoded
            .replace("&quot;", "\"")
            .replace("&#34;", "\"")
            .replace("&apos;", "'")
            .replace("&#39;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&");
        if next == decoded {
            break;
        }
        decoded = next;
    }
    decoded
}

fn embedded_json_object_candidates(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let mut stack = 0usize;
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        match ch {
            '{' => {
                if stack == 0 {
                    start = Some(idx);
                }
                stack += 1;
            }
            '}' if stack > 0 => {
                stack -= 1;
                if stack == 0 {
                    let Some(start_idx) = start.take() else {
                        continue;
                    };
                    candidates.push(text[start_idx..idx + ch.len_utf8()].to_string());
                }
            }
            _ => {}
        }
    }
    candidates
}

fn explicitly_named_tool_choice_from_text(
    content: &str,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Option<ForcedToolChoice> {
    let lower = content.to_ascii_lowercase();
    if contains_any(
        &lower,
        &[
            "cannot use",
            "can't use",
            "will not use",
            "would not use",
            "do not use",
            "don't use",
        ],
    ) {
        return None;
    }

    let available = declared_response_tool_names(tools, allowed_tools, prompt_tool_profiles);
    let selected = explicitly_requested_tool_names(&available, &lower);
    let [name] = selected.as_slice() else {
        return None;
    };

    let inferred = infer_tool_arguments_from_prompt(name, tools, content);
    match validate_response_tool_arguments(
        name,
        &inferred,
        tools,
        allowed_tools,
        prompt_tool_profiles,
    ) {
        Ok(arguments) => Some(ForcedToolChoice {
            name: name.clone(),
            fallback_arguments: arguments,
        }),
        Err(err) => {
            tracing::info!("moa: explicit prose tool {name} could not be sanitized: {err}");
            None
        }
    }
}

fn validate_response_tool_arguments(
    name: &str,
    arguments: &Value,
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Result<Value, String> {
    if tool_names_from_schemas(tools, allowed_tools)
        .iter()
        .any(|schema_name| schema_name == name)
    {
        return sanitize_tool_arguments_for_tool(name, arguments, tools)
            .map_err(|err| err.to_string());
    }

    validate_prompt_tool_arguments(name, arguments, allowed_tools, prompt_tool_profiles)
}

fn validate_prompt_tool_arguments(
    name: &str,
    arguments: &Value,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Result<Value, String> {
    let allowed: HashSet<&str> = allowed_tools.iter().map(String::as_str).collect();
    let declared = prompt_tool_profiles
        .iter()
        .any(|profile| profile.name == name && (allowed.is_empty() || allowed.contains(name)));
    if !declared {
        return Err(format!("tool {name} is not prompt-declared"));
    }

    let Some(arguments) = arguments.as_object() else {
        return Err(format!(
            "prompt-declared tool {name} requires object arguments"
        ));
    };
    if arguments.is_empty() {
        return Err(format!("prompt-declared tool {name} requires arguments"));
    }

    let candidates = prompt_command_tool_candidates(allowed_tools, prompt_tool_profiles);
    if let Some(candidate) = candidates.iter().find(|candidate| candidate.name == name) {
        let valid_command = arguments
            .get(&candidate.field)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());
        if !valid_command {
            return Err(format!(
                "prompt-declared command tool {name} requires non-empty {}",
                candidate.field
            ));
        }
    }

    Ok(Value::Object(arguments.clone()))
}

fn declared_response_tool_names(
    tools: Option<&Value>,
    allowed_tools: &[String],
    prompt_tool_profiles: &[PromptToolProfile],
) -> Vec<String> {
    let schema_names = tool_names_from_schemas(tools, allowed_tools);
    if !schema_names.is_empty() {
        return schema_names;
    }
    let allowed: HashSet<&str> = allowed_tools.iter().map(String::as_str).collect();
    prompt_tool_profiles
        .iter()
        .filter(|profile| allowed.is_empty() || allowed.contains(profile.name.as_str()))
        .map(|profile| profile.name.clone())
        .collect()
}

fn tool_names_from_schemas(tools: Option<&Value>, allowed_tools: &[String]) -> Vec<String> {
    let allowed: HashSet<&str> = allowed_tools.iter().map(String::as_str).collect();
    tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            let name = tool.pointer("/function/name")?.as_str()?;
            if allowed.is_empty() || allowed.contains(name) {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Build a response body that signals MoA-level failure to the client.
///
/// Distinguishable from a successful `chat.completion` in three ways:
///
///   * Top-level `error` object (OpenAI error-shape) so SDKs that read
///     `response.error` see the failure without parsing `choices`.
///   * `choices[0].finish_reason == "error"` (instead of `"stop"`) so
///     SDKs that branch on `finish_reason` see the failure too.
///   * The error text is still placed in `choices[0].message.content`
///     so unstructured clients still surface a useful string to the
///     human, just not as a successful assistant reply.
///
/// `code` is the machine-parseable failure mode that clients can branch
/// on. Callers pass one of the [`MOA_ERR_*`] constants so distinct
/// failure modes (all-workers-failed vs all-reducers-failed vs future
/// kinds) surface accurately to the caller rather than being collapsed
/// to a single string.
///
/// The ingress layer is responsible for choosing the HTTP status; this
/// body is the in-band signal.
fn error_response(message: &str, code: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "error": {
            "message": message,
            "type": "moa_failure",
            "code": code,
        },
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": message },
            "finish_reason": "error"
        }],
        "usage": usage_for_content(message)
    })
}

/// Estimate `completion_tokens` from output chars (OpenAI's ~chars/4 rule).
/// Returns at least 1 for non-empty so UI tok/s never divides by zero.
fn estimate_completion_tokens(content: &str) -> u64 {
    if content.is_empty() {
        return 0;
    }
    let chars = content.chars().count() as u64;
    chars.div_ceil(4).max(1)
}

fn usage_for_content(content: &str) -> Value {
    let completion = estimate_completion_tokens(content);
    json!({
        "prompt_tokens": 0,
        "completion_tokens": completion,
        "total_tokens": completion,
    })
}

/// All fanned-out workers failed before the arbiter could pick a winner.
pub const MOA_ERR_ALL_WORKERS_FAILED: &str = "all_workers_failed";
/// Every reducer candidate failed (in both the tool-result and the
/// arbiter-escalated paths).
pub const MOA_ERR_ALL_REDUCERS_FAILED: &str = "all_reducers_failed";
/// MoA only received silence directives or uncertainty after reduction.
pub const MOA_ERR_NO_USABLE_ANSWER: &str = "no_usable_answer";

fn chat_response(content: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": usage_for_content(content)
    })
}

fn tool_call_response(name: &str, arguments: &Value) -> Value {
    // OpenAI tool-call `arguments` is a JSON-object *string*. Three input
    // shapes have to collapse to a valid object string here:
    //
    //   * String form: trust the caller's JSON (worker already passed
    //     through `extract_tool_arguments` so the inner shape is sane).
    //   * Null / non-object: emit `"{}"` rather than `"null"` or
    //     `"\"foo\""`. The previous shape would serialize `Value::Null`
    //     to the literal four-char string `"null"`, which downstream
    //     OpenAI tool-call consumers reject.
    //   * Object: serialize as JSON.
    let args_str = tool_arguments_wire_string(arguments);

    // For tool-call responses, the user-visible output is the
    // arguments JSON, not free-form text. Use it as the basis of the
    // token estimate so callers still see a non-zero count.
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": format!("call_{}", short_id()),
                    "type": "function",
                    "function": { "name": name, "arguments": args_str }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": usage_for_content(&args_str)
    })
}

fn short_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", t)
}

#[cfg(test)]
mod response_builder_tests {
    use super::*;
    use crate::normalize::{OutputKind, WorkerOutput};
    use crate::worker::WorkerRole;

    fn answer(model: &str, confidence: f32, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::Answer,
            confidence,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: model.to_string(),
            role: WorkerRole::Fast,
            elapsed_ms: 1,
        }
    }

    #[test]
    fn best_answer_does_not_panic_on_nan_confidence() {
        // Regression for PR #566 review: `partial_cmp(...).unwrap()` could
        // panic if any confidence reached this site as NaN. After switching
        // to `total_cmp`, this is safe even if normalize is bypassed.
        let outputs = vec![
            answer("a", f32::NAN, "nan-answer"),
            answer("b", 0.7, "good-answer"),
            answer("c", f32::NAN, "another-nan"),
        ];
        let picked = best_answer(&outputs);
        // `total_cmp` treats NaN as greater than any finite; the assertion
        // here is *not* about which specific answer wins, only that we do
        // not panic and we return *some* answer.
        assert!(!picked.is_empty());
    }

    #[test]
    fn best_answer_ignores_silent_reply_sentinel() {
        let outputs = vec![
            answer("a", 0.99, "NO_REPLY"),
            answer("b", 0.6, "Here is a real response."),
        ];
        assert_eq!(best_answer(&outputs), "Here is a real response.");
    }

    #[test]
    fn fallback_worker_response_errors_when_only_silent_sentinel_remains() {
        let outputs = vec![answer("a", 0.99, "NO_REPLY")];
        let session = Session::new();
        let resp = fallback_worker_response(&outputs, &session, None, &[], &[]);
        assert_eq!(
            resp.pointer("/error/code").and_then(Value::as_str),
            Some(MOA_ERR_NO_USABLE_ANSWER)
        );
        assert_eq!(
            resp.pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("error")
        );
    }

    fn tool_proposal(payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: normalize::OutputKind::ToolProposal,
            confidence: 0.8,
            tool_name: Some("read_file".to_string()),
            tool_arguments: Some(json!({"path": "README.md"})),
            payload: payload.to_string(),
            model: "reducer".to_string(),
            role: WorkerRole::Reducer,
            elapsed_ms: 1,
        }
    }

    fn read_file_tool_schema() -> Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }])
    }

    #[test]
    fn tool_proposal_response_emits_tool_call_when_tools_enabled() {
        let tools = read_file_tool_schema();
        let allowed_tools = ["read_file".to_string()];
        let resp = tool_proposal_response(
            &tool_proposal("Need to read."),
            true,
            Some(&tools),
            &allowed_tools,
            &[],
            None,
        );
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("read_file")
        );
    }

    #[test]
    fn tool_proposal_response_rejects_undeclared_tool() {
        let resp =
            tool_proposal_response(&tool_proposal("Need to read."), true, None, &[], &[], None);

        assert!(
            resp.pointer("/choices/0/message/tool_calls").is_none(),
            "undeclared worker tool proposals must not become tool calls: {resp}"
        );
        assert_eq!(
            resp.pointer("/choices/0/message/content")
                .and_then(Value::as_str),
            Some("Need to read.")
        );
    }

    #[test]
    fn tool_proposal_response_rejects_prompt_tool_without_args() {
        let (session, profiles, allowed) =
            prompt_tool_session("Use gh to list recent open PRs for mesh-LLM/mesh-llm.");
        let output = WorkerOutput {
            kind: normalize::OutputKind::ToolProposal,
            confidence: 0.8,
            tool_name: Some("exec".to_string()),
            tool_arguments: Some(json!({})),
            payload: "I will use the exec tool.".to_string(),
            model: "worker".to_string(),
            role: WorkerRole::Fast,
            elapsed_ms: 1,
        };

        let resp = tool_proposal_response(
            &output,
            true,
            session.tools(),
            &allowed,
            &profiles,
            Some(&session.last_user_text()),
        );

        assert!(
            resp.pointer("/choices/0/message/tool_calls").is_none(),
            "prompt-declared command tools require a command argument: {resp}"
        );
        assert_eq!(
            resp.pointer("/choices/0/message/content")
                .and_then(Value::as_str),
            Some("I will use the exec tool.")
        );
    }

    #[test]
    fn rejected_tool_proposal_repairs_fenced_command_with_user_intent() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "exec",
                "description": "Run shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        }]);
        let allowed_tools = ["exec".to_string()];
        let output = WorkerOutput {
            kind: normalize::OutputKind::ToolProposal,
            confidence: 0.8,
            tool_name: Some("exec".to_string()),
            tool_arguments: Some(json!({})),
            payload: "I'll check it.\n\n```bash\ngh pr list --repo mesh-LLM/mesh-llm --state open --limit 10\n```".to_string(),
            model: "reducer".to_string(),
            role: WorkerRole::Reducer,
            elapsed_ms: 1,
        };

        let resp = tool_proposal_response(
            &output,
            true,
            Some(&tools),
            &allowed_tools,
            &[],
            Some(
                "Use gh to list recent open PRs for mesh-LLM/mesh-llm, then tell me one interesting one.",
            ),
        );

        assert_eq!(
            resp.pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"command\":\"gh pr list --repo mesh-LLM/mesh-llm --state open --limit 10\"}")
        );
    }

    #[test]
    fn tool_proposal_response_does_not_emit_tool_call_when_tools_disabled() {
        let resp = tool_proposal_response(
            &tool_proposal("I need to read README.md."),
            false,
            None,
            &[],
            &[],
            None,
        );
        assert!(
            resp.pointer("/choices/0/message/tool_calls").is_none(),
            "disabled tools must not leak tool_calls: {resp}"
        );
        assert_eq!(
            resp.pointer("/choices/0/message/content")
                .and_then(Value::as_str),
            Some("I need to read README.md.")
        );
    }

    #[test]
    fn tool_call_response_emits_object_args_for_null() {
        // Regression: `Value::Null` previously serialized to the literal
        // string "null", which downstream OpenAI tool-call consumers reject.
        let resp = tool_call_response("list", &Value::Null);
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    #[test]
    fn tool_call_response_emits_object_args_for_primitive() {
        let resp = tool_call_response("list", &Value::from(42));
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    #[test]
    fn tool_call_response_passes_through_string_form_when_valid() {
        let resp = tool_call_response(
            "read_file",
            &Value::String("{\"path\":\"README.md\"}".to_string()),
        );
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        let parsed: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed["path"], "README.md");
    }

    #[test]
    fn tool_call_response_rejects_invalid_string_form() {
        // If the caller hands us a bare non-JSON string, fall back to `{}`.
        let resp = tool_call_response("x", &Value::String("not json at all".to_string()));
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    // Regression for #637.

    #[test]
    fn estimate_completion_tokens_returns_zero_for_empty_content() {
        assert_eq!(estimate_completion_tokens(""), 0);
    }

    #[test]
    fn estimate_completion_tokens_returns_at_least_one_for_non_empty() {
        assert_eq!(estimate_completion_tokens("a"), 1);
    }

    #[test]
    fn estimate_completion_tokens_is_roughly_chars_over_four() {
        assert_eq!(estimate_completion_tokens("sixteen chars!!!"), 4);
        assert_eq!(estimate_completion_tokens(&"x".repeat(40)), 10);
    }

    #[test]
    fn chat_response_reports_non_zero_completion_tokens() {
        let resp = chat_response("Hi there! How can I help you today?");
        let tokens = resp
            .pointer("/usage/completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .expect("completion_tokens is u64");
        assert!(tokens > 0);
        assert_eq!(
            resp.pointer("/usage/total_tokens").and_then(|v| v.as_u64()),
            Some(tokens),
        );
    }

    #[test]
    fn tool_call_response_reports_non_zero_completion_tokens() {
        let resp = tool_call_response("read_file", &serde_json::json!({"path": "/etc/hostname"}));
        let tokens = resp
            .pointer("/usage/completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .expect("completion_tokens is u64");
        assert!(tokens > 0);
    }

    #[test]
    fn incoming_chat_messages_synthesizes_completion_prompt_and_system_prompt() {
        let messages = incoming_chat_messages(&serde_json::json!({
            "model": "mesh",
            "systemPrompt": "## Tooling\n- exec: Run shell commands",
            "prompt": "Use gh to list recent PRs."
        }));

        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].get("role").and_then(Value::as_str),
            Some("system")
        );
        assert_eq!(
            messages[0].get("content").and_then(Value::as_str),
            Some("## Tooling\n- exec: Run shell commands")
        );
        assert_eq!(
            messages[1].get("content").and_then(Value::as_str),
            Some("Use gh to list recent PRs.")
        );
    }

    #[test]
    fn incoming_chat_messages_adds_top_level_system_to_existing_messages() {
        let messages = incoming_chat_messages(&serde_json::json!({
            "systemPrompt": "system",
            "messages": [{"role": "user", "content": "hello"}],
        }));

        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].get("role").and_then(Value::as_str),
            Some("system")
        );
        assert_eq!(
            messages[1].get("role").and_then(Value::as_str),
            Some("user")
        );
    }

    #[test]
    fn forced_tool_choice_infers_enum_argument_from_prompt() {
        let body = serde_json::json!({
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup_probe_fact"}
            },
            "messages": [{
                "role": "user",
                "content": "Use lookup_probe_fact with primary and report the result."
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_probe_fact",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "enum": ["primary", "secondary"]
                            }
                        },
                        "required": ["key"]
                    }
                }
            }]
        });
        let tools = body.get("tools").cloned();
        let mut session = Session::new();
        let messages = body
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap();
        session.ingest(&messages, &tools);

        let allowed_tools = session.tool_names();
        let forced =
            forced_tool_choice(&body, &session, &tools, &allowed_tools).expect("forced tool");

        assert_eq!(forced.name, "lookup_probe_fact");
        assert_eq!(forced.fallback_arguments, json!({"key": "primary"}));
    }

    #[test]
    fn forced_tool_choice_infers_assignment_argument_from_prompt() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "lookup_probe_fact",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"]
                }
            }
        }]));

        let args = infer_tool_arguments_from_prompt(
            "lookup_probe_fact",
            tools.as_ref(),
            "Use key=Primary",
        );

        assert_eq!(args, json!({"key": "Primary"}));
    }

    #[test]
    fn forced_tool_choice_ignores_prompt_only_tool_without_native_schema() {
        let body = serde_json::json!({
            "tool_choice": {
                "type": "function",
                "function": {"name": "exec"}
            },
            "messages": [
                {"role": "system", "content": prompt_tool_catalog()},
                {"role": "user", "content": "Use gh to list PRs."}
            ]
        });
        let tools = body.get("tools").cloned();
        let mut session = Session::new();
        let messages = body
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap();
        session.ingest(&messages, &tools);
        let profiles = prompt_declared_tool_profiles(&session);
        let allowed_tools = declared_tool_names(&session, &profiles);

        assert!(forced_tool_choice(&body, &session, &tools, &allowed_tools).is_none());
    }

    #[test]
    fn tool_argument_inference_extracts_absolute_path() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }]));

        let args = infer_tool_arguments_from_prompt(
            "read",
            tools.as_ref(),
            "Use the read tool to read /tmp/moa_probe.txt.",
        );

        assert_eq!(args, json!({"path": "/tmp/moa_probe.txt"}));
    }

    #[test]
    fn explicit_single_tool_intent_becomes_required_tool_choice() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }]));
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use the read tool to read /tmp/moa_probe.txt."
            })],
            &tools,
        );

        let selected = vec!["read".to_string()];
        let required =
            required_tool_choice_for_intent(&session, &tools, &selected).expect("required tool");

        assert_eq!(required.name, "read");
        assert_eq!(
            required.fallback_arguments,
            json!({"path": "/tmp/moa_probe.txt"})
        );
    }

    #[tokio::test]
    async fn inferred_required_tool_intent_short_circuits_workers() {
        let body = serde_json::json!({
            "model": VIRTUAL_MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": "Use the read tool to read /tmp/moa_probe.txt."
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            }]
        });
        let config = GatewayConfig {
            backends: Vec::new(),
            models: Vec::new(),
            worker_timeout: Duration::from_secs(1),
            reducer_timeout: Duration::from_secs(1),
            hedge_delay: Duration::from_millis(10),
            first_answer_grace: Duration::from_millis(10),
            enable_thinking: Some(false),
        };

        let result = handle_turn(&config, &body).await;

        assert_eq!(result.turn_kind, TurnKind::DirectTool);
        assert!(result.worker_summaries.is_empty());
        assert!(!result.reducer_used);
        assert_eq!(
            result
                .response_body
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            result
                .response_body
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("read")
        );
        assert_eq!(
            result
                .response_body
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"path\":\"/tmp/moa_probe.txt\"}")
        );
    }

    #[tokio::test]
    async fn approval_of_prior_shell_proposal_short_circuits_to_exec() {
        let body = serde_json::json!({
            "model": VIRTUAL_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Any new interesting PRs? You have the gh command line I think can use"
                },
                {
                    "role": "assistant",
                    "content": "I can use the `gh` CLI. Let me fetch the list.\n\n```bash\ngh pr list --repo mesh-LLM/mesh-llm --state open --sort=updated --direction desc --limit 20\n```"
                },
                {
                    "role": "user",
                    "content": "Conversation context (untrusted): prior messages\n\nOk you want to do that?"
                }
            ],
            "tools": [{
                "name": "exec",
                "description": "Execute shell commands.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }]
        });
        let config = GatewayConfig {
            backends: Vec::new(),
            models: Vec::new(),
            worker_timeout: Duration::from_secs(1),
            reducer_timeout: Duration::from_secs(1),
            hedge_delay: Duration::from_millis(10),
            first_answer_grace: Duration::from_millis(10),
            enable_thinking: Some(false),
        };

        let result = handle_turn(&config, &body).await;

        assert_eq!(result.turn_kind, TurnKind::DirectTool);
        assert!(result.worker_summaries.is_empty());
        assert_eq!(
            result
                .response_body
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            result
                .response_body
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        assert_eq!(
            result
                .response_body
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some(
                "{\"command\":\"gh pr list --repo mesh-LLM/mesh-llm --state open --sort=updated --direction desc --limit 20\"}"
            )
        );
    }

    #[test]
    fn look_is_not_prior_shell_proposal_confirmation() {
        let tools = Some(serde_json::json!([{
            "name": "exec",
            "description": "Execute shell commands.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }]));
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({
                    "role": "user",
                    "content": "Any new interesting PRs?"
                }),
                serde_json::json!({
                    "role": "assistant",
                    "content": "I can check with:\n\n```bash\ngh pr list --repo mesh-LLM/mesh-llm --state open\n```"
                }),
                serde_json::json!({
                    "role": "user",
                    "content": "look at that first"
                }),
            ],
            &tools,
        );
        let allowed_tools = ["exec".to_string()];

        assert!(prior_shell_command_tool_choice(&session, &tools, &allowed_tools, &[]).is_none());
    }

    #[tokio::test]
    async fn forced_tool_choice_overrides_answer_decision() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use lookup_probe_fact with primary"
            })],
            &Some(serde_json::json!([{
                "type": "function",
                "function": {"name": "lookup_probe_fact"}
            }])),
        );
        let config = GatewayConfig {
            backends: Vec::new(),
            models: Vec::new(),
            worker_timeout: Duration::from_secs(1),
            reducer_timeout: Duration::from_secs(1),
            hedge_delay: Duration::from_millis(10),
            first_answer_grace: Duration::from_millis(10),
            enable_thinking: Some(false),
        };
        let forced_tool = ForcedToolChoice {
            name: "lookup_probe_fact".to_string(),
            fallback_arguments: json!({"key": "primary"}),
        };
        let selected_tool_names = ["lookup_probe_fact".to_string()];
        let allowed_tools = ["lookup_probe_fact".to_string()];
        let (resp, reducer_used, attempts) = resolve_decision(
            &config,
            DecisionResolution {
                session: &session,
                decision: arbiter::Decision::Answer("I would call the tool.".to_string()),
                outputs: &[],
                has_tools: true,
                selected_tool_names: &selected_tool_names,
                forced_tool: Some(&forced_tool),
                allowed_tools: &allowed_tools,
                prompt_tool_profiles: &[],
            },
        )
        .await;

        assert!(!reducer_used);
        assert_eq!(attempts, 0);
        assert_eq!(
            resp.pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("lookup_probe_fact")
        );
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"key\":\"primary\"}")
        );
    }

    #[test]
    fn error_response_reports_message_based_completion_tokens() {
        let resp = error_response("All MoA workers failed", MOA_ERR_ALL_WORKERS_FAILED);
        let tokens = resp
            .pointer("/usage/completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .expect("completion_tokens is u64");
        assert!(tokens > 0);
    }

    #[test]
    fn tool_enabled_chat_uses_answer_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "How are you?"})],
            &Some(serde_json::json!([{"type": "function", "function": {"name": "read"}}])),
        );
        assert_eq!(grace_mode_for_turn(&session, true, &[]), GraceMode::Answer);
    }

    #[test]
    fn tool_intent_uses_tool_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use a tool to read /tmp/moa-tool-baseline.txt",
            })],
            &Some(serde_json::json!([{"type": "function", "function": {"name": "read"}}])),
        );
        assert_eq!(grace_mode_for_turn(&session, true, &[]), GraceMode::Tool);
    }

    #[test]
    fn negated_web_prompt_uses_answer_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Plain check with no tool use: reply OK",
            })],
            &Some(serde_json::json!([{"type": "function", "function": {"name": "web_search"}}])),
        );
        assert_eq!(grace_mode_for_turn(&session, true, &[]), GraceMode::Answer);
    }

    #[test]
    fn no_tools_uses_answer_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "Reply OK"})],
            &None,
        );
        assert_eq!(grace_mode_for_turn(&session, false, &[]), GraceMode::Answer);
    }

    #[test]
    fn read_prompt_selects_only_read_tool() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "Read /tmp/file.txt"})],
            &Some(serde_json::json!([
                {"type": "function", "function": {
                    "name": "read",
                    "description": "Read file contents from a path",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "File path to read"}},
                        "required": ["path"]
                    }
                }},
                {"type": "function", "function": {"name": "web_search", "description": "Search online sources"}},
                {"type": "function", "function": {"name": "exec", "description": "Run shell commands"}}
            ])),
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["read".to_string()]
        );
    }

    #[test]
    fn live_status_followup_after_tool_chain_uses_tool_grace() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "exec",
                "description": "Run shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }
        }]));
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "Any new interesting PRs?"}),
                tool_call_msg("call_1", "exec"),
                tool_result_msg(
                    "call_1",
                    "808\tStabilize OpenClaw tool loops through MoA\tOPEN",
                ),
                serde_json::json!({
                    "role": "assistant",
                    "content": "PR 808 looks relevant."
                }),
                serde_json::json!({
                    "role": "user",
                    "content": "The stabilise one, PR 808 - is CI all green now? Any Copilot feedback?"
                }),
            ],
            &tools,
        );

        let allowed = declared_tool_names(&session, &[]);

        assert_eq!(
            grace_mode_for_turn(&session, !allowed.is_empty(), &[]),
            GraceMode::Tool
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &allowed, &[]),
            vec!["exec"]
        );
    }

    #[test]
    fn schema_description_selects_weather_tool() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Check the current Melbourne weather forecast for today",
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read", "description": "Read local files"}},
                {"type": "function", "function": {"name": "lookup", "description": "Search current weather forecast information"}},
                {"type": "function", "function": {"name": "runner", "description": "Run shell commands"}}
            ])),
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["lookup".to_string()]
        );
    }

    #[test]
    fn schema_description_selects_available_weather_tool_phrase() {
        let tools = serde_json::json!([
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read workspace file contents",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "File path"}},
                    "required": ["path"]
                }
            }},
            {"type": "function", "function": {
                "name": "weather_lookup",
                "description": "Look up current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City to check"}},
                    "required": ["city"]
                }
            }}
        ]);
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "What is the weather in Sydney right now? Use the available weather tool."
            })],
            &Some(tools.clone()),
        );

        let selected = selected_tool_names_for_turn(&session, &[], &[]);

        assert_eq!(selected, vec!["weather_lookup".to_string()]);
        let forced =
            required_tool_choice_for_intent(&session, &Some(tools), &selected).expect("tool");
        assert_eq!(forced.name, "weather_lookup");
        assert_eq!(
            forced.fallback_arguments,
            serde_json::json!({"city": "Sydney"})
        );
    }

    #[test]
    fn named_argument_inference_skips_intent_verbs_after_prepositions() {
        let tools = serde_json::json!([
            {"type": "function", "function": {
                "name": "weather_lookup",
                "description": "Look up current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City to check"}},
                    "required": ["city"]
                }
            }}
        ]);

        let args = infer_tool_arguments_from_prompt(
            "weather_lookup",
            Some(&tools),
            "Use the weather lookup tool to check Sydney.",
        );

        assert_eq!(args, serde_json::json!({"city": "Sydney"}));
    }

    #[test]
    fn executable_request_selects_command_schema_not_directory_tool() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use gh to list recent open PRs for mesh-LLM/mesh-llm."
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {
                    "name": "dir_list",
                    "description": "List directory entries for a local filesystem path",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Directory path"}},
                        "required": ["path"]
                    }
                }},
                {"type": "function", "function": {
                    "name": "exec",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                        "required": ["command"]
                    }
                }}
            ])),
        );

        assert_eq!(
            session.tool_names(),
            vec!["dir_list".to_string(), "exec".to_string()]
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["exec".to_string()]
        );
    }

    #[test]
    fn command_intent_stays_unselected_when_command_schemas_are_ambiguous() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use gh to list recent open PRs for mesh-LLM/mesh-llm."
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {
                    "name": "exec",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string", "description": "Shell command to execute"}},
                        "required": ["command"]
                    }
                }},
                {"type": "function", "function": {
                    "name": "remote_exec",
                    "description": "Run terminal commands on a remote host",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string", "description": "Terminal command to run"}},
                        "required": ["command"]
                    }
                }}
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn command_schema_can_use_single_optional_command_field() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use gh to list recent open PRs for mesh-LLM/mesh-llm."
            })],
            &Some(serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"},
                            "cwd": {"type": "string", "description": "Working directory"},
                            "timeout_ms": {"type": "integer"}
                        }
                    }
                }
            }])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["exec".to_string()]
        );
    }

    #[test]
    fn command_schema_selects_clear_command_field_with_optional_metadata_fields() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use gh to list recent open PRs for mesh-LLM/mesh-llm."
            })],
            &Some(serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Run shell commands (pty available for TTY-required CLIs)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"},
                            "workdir": {"type": "string", "description": "Working directory (defaults to cwd)"},
                            "env": {"type": "object"},
                            "yieldMs": {"type": "number", "description": "Milliseconds to wait before backgrounding"},
                            "background": {"type": "boolean", "description": "Run in background immediately"},
                            "timeout": {"type": "number", "description": "Timeout in seconds"},
                            "pty": {"type": "boolean", "description": "Run in a pseudo-terminal"},
                            "elevated": {"type": "boolean", "description": "Run on the host with elevated permissions"},
                            "host": {"type": "string", "description": "Exec host/target (auto|sandbox|gateway|node)."},
                            "security": {"type": "string", "description": "Ignored for normal calls; exec security is set by tools.exec.security and host approvals."},
                            "ask": {"type": "string", "description": "Exec ask mode (off|on-miss|always)."},
                            "node": {"type": "string", "description": "Node id/name for host=node."}
                        }
                    }
                }
            }])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["exec".to_string()]
        );
        assert_eq!(grace_mode_for_turn(&session, true, &[]), GraceMode::Tool);
    }

    #[test]
    fn optional_command_schema_intent_does_not_emit_empty_direct_tool_call() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "exec",
                "description": "Run shell commands (pty available for TTY-required CLIs)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "host": {"type": "string", "description": "Exec host/target"},
                        "ask": {"type": "string", "description": "Exec ask mode"}
                    }
                }
            }
        }]);
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use gh to list recent open PRs for mesh-LLM/mesh-llm."
            })],
            &Some(tools.clone()),
        );
        let selected = selected_tool_names_for_turn(&session, &[], &[]);

        assert_eq!(selected, vec!["exec".to_string()]);
        assert!(required_tool_choice_for_intent(&session, &Some(tools), &selected).is_none());
    }

    #[test]
    fn command_intent_selects_exec_from_agent_tool_catalog() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use gh to list recent open PRs for mesh-LLM/mesh-llm."
            })],
            &Some(serde_json::json!([
                {
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "description": "Run shell commands (pty available for TTY-required CLIs)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string", "description": "Shell command to execute"},
                                "host": {"type": "string", "description": "Exec host/target (auto|sandbox|gateway|node)."},
                                "ask": {"type": "string", "description": "Exec ask mode (off|on-miss|always)."}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "process",
                        "description": "Manage background exec sessions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "description": "Process action (list|poll|log|write|send-keys|submit|paste|kill|clear|remove)"},
                                "sessionId": {"type": "string", "description": "Session id for actions other than list"},
                                "text": {"type": "string", "description": "Text to paste for paste"}
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "gateway",
                        "description": "Restart, apply config, or run updates on the running process",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "description": "Gateway action to run"},
                                "path": {"type": "string", "description": "Config path"}
                            }
                        }
                    }
                }
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["exec".to_string()]
        );
        assert_eq!(grace_mode_for_turn(&session, true, &[]), GraceMode::Tool);
    }

    #[test]
    fn command_schema_ignores_ambiguous_optional_command_fields() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "runner",
                "description": "Run shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "fallback_command": {"type": "string", "description": "Alternative shell command"}
                    }
                }
            }
        }]);

        assert_eq!(
            command_schema_tool_candidates(Some(&tools), &["runner".to_string()]).len(),
            0
        );
    }

    #[test]
    fn command_schema_ignores_tied_optional_command_fields() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "runner",
                "description": "Run shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_command": {"type": "string", "description": "Shell command to execute"},
                        "fallback_command": {"type": "string", "description": "Shell command to execute if the primary command fails"}
                    }
                }
            }
        }]);

        assert_eq!(
            command_schema_tool_candidates(Some(&tools), &["runner".to_string()]).len(),
            0
        );
    }

    fn prompt_tool_catalog() -> String {
        [
            "You are an agent.",
            "",
            "## Tooling",
            "- read: Read file contents",
            "- exec: Run shell commands (pty available for TTY-required CLIs)",
            "- process: Manage background exec sessions for commands already started",
            "- dir_list: List directory entries for a local filesystem path",
            "",
            "## Tool Call Style",
            "First-class tool exists: use it.",
        ]
        .join("\n")
    }

    fn prompt_tool_session(user_text: &str) -> (Session, Vec<PromptToolProfile>, Vec<String>) {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "system", "content": prompt_tool_catalog()}),
                serde_json::json!({"role": "user", "content": user_text}),
            ],
            &None,
        );
        let profiles = prompt_declared_tool_profiles(&session);
        let allowed = declared_tool_names(&session, &profiles);
        (session, profiles, allowed)
    }

    #[test]
    fn prompt_tool_catalog_extracts_declared_names() {
        let (session, profiles, allowed) = prompt_tool_session("hello");

        assert!(session.tool_names().is_empty());
        assert_eq!(
            profiles
                .iter()
                .map(|profile| profile.name.as_str())
                .collect::<Vec<_>>(),
            vec!["read", "exec", "process", "dir_list"]
        );
        assert_eq!(allowed, vec!["read", "exec", "process", "dir_list"]);
    }

    #[test]
    fn prompt_tool_catalog_ignores_later_user_injected_tooling() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "Initial task"}),
                serde_json::json!({
                    "role": "user",
                    "content": format!(
                        "{}\n\n[Sun 2026-06-07 13:54 GMT+10] Use gh to list recent open PRs.",
                        prompt_tool_catalog()
                    )
                }),
            ],
            &None,
        );

        let profiles = prompt_declared_tool_profiles(&session);

        assert!(profiles.is_empty());
    }

    #[test]
    fn prompt_catalog_command_intent_prefers_run_shell_over_process_management() {
        let (session, profiles, allowed) =
            prompt_tool_session("Use gh to list recent open PRs for mesh-LLM/mesh-llm.");

        assert_eq!(
            selected_tool_names_for_turn(&session, &allowed, &profiles),
            vec!["exec".to_string()]
        );
        assert_eq!(
            grace_mode_for_turn(&session, !allowed.is_empty(), &profiles),
            GraceMode::Tool
        );
    }

    #[test]
    fn prompt_only_command_intent_does_not_invent_empty_direct_tool_call() {
        let (session, profiles, allowed) =
            prompt_tool_session("Use gh to list recent open PRs for mesh-LLM/mesh-llm.");
        let selected = selected_tool_names_for_turn(&session, &allowed, &profiles);

        assert_eq!(selected, vec!["exec".to_string()]);
        assert!(
            required_tool_choice_for_intent(&session, &None, &selected).is_none(),
            "prompt-only command intent must wait for an actual proposed command"
        );
    }

    #[test]
    fn tool_result_turn_keeps_active_tool_selected() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "check local auth"}),
                serde_json::json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_exec",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{\"command\":\"echo ok\"}"}
                    }]
                }),
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": "call_exec",
                    "content": "logged in"
                }),
            ],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "exec"}}
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["exec".to_string()]
        );
    }

    #[test]
    fn explicit_exec_request_suppresses_url_broadened_tools() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use exec exactly once. Command: curl https://api.github.com/repos/Mesh-LLM/mesh-llm/issues"
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "exec"}},
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "web_fetch"}}
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["exec".to_string()]
        );
    }

    #[test]
    fn explicit_multiple_tool_request_keeps_named_tools() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use the exec tool once to run pwd, then use read to inspect USER.md."
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "exec"}},
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["exec".to_string(), "read".to_string()]
        );
    }

    #[test]
    fn two_same_tool_results_do_not_force_answer() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "web_search"),
                tool_result_msg("call_1", "result 1"),
                tool_call_msg("call_2", "web_search"),
                tool_result_msg("call_2", "result 2"),
            ],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );

        assert_eq!(repeated_same_tool_results(&session), None);
    }

    #[test]
    fn three_same_tool_results_force_answer() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "web_search"),
                tool_result_msg("call_1", "result 1"),
                tool_call_msg("call_2", "web_search"),
                tool_result_msg("call_2", "result 2"),
                tool_call_msg("call_3", "web_search"),
                tool_result_msg("call_3", "result 3"),
            ],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );

        assert_eq!(
            repeated_same_tool_results(&session),
            Some(("web_search".to_string(), 3))
        );
    }

    #[test]
    fn repeated_tool_suppression_keeps_alternate_declared_tools_available() {
        let selected = suppress_repeated_tool_from_selection(
            vec!["web_search".to_string()],
            &[
                "web_search".to_string(),
                "read_file".to_string(),
                "exec".to_string(),
            ],
            Some("web_search"),
        );

        assert_eq!(selected, vec!["read_file".to_string(), "exec".to_string()]);
    }

    #[test]
    fn repeated_tool_suppression_keeps_non_repeated_selection() {
        let selected = suppress_repeated_tool_from_selection(
            vec!["web_search".to_string(), "read_file".to_string()],
            &[
                "web_search".to_string(),
                "read_file".to_string(),
                "exec".to_string(),
            ],
            Some("web_search"),
        );

        assert_eq!(selected, vec!["read_file".to_string()]);
    }

    #[test]
    fn repair_tool_result_answer_preserves_short_json_values_on_recall() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "lookup"),
                tool_result_msg("call_1", r#"{"key":"primary","value":"PRIMARY-FACT-123"}"#),
                tool_call_msg("call_2", "lookup"),
                tool_result_msg(
                    "call_2",
                    r#"{"key":"secondary","value":"SECONDARY-FACT-456"}"#,
                ),
                serde_json::json!({
                    "role": "user",
                    "content": "Final recall: include both tool facts."
                }),
            ],
            &None,
        );

        let repaired =
            repair_tool_result_answer(&session, "The secondary fact is SECONDARY-FACT-456.");

        assert!(repaired.contains("PRIMARY-FACT-123"));
        assert!(repaired.contains("SECONDARY-FACT-456"));
        assert!(!repaired.contains("primary"));
    }

    #[test]
    fn repair_tool_result_answer_ignores_large_or_non_evidence_tool_values() {
        let huge = "x".repeat(200);
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "lookup"),
                tool_result_msg(
                    "call_1",
                    &serde_json::json!({
                        "value": huge,
                        "debug": "SHORT-BUT-NOT-EVIDENCE",
                        "result": "multi\nline",
                    })
                    .to_string(),
                ),
                serde_json::json!({
                    "role": "user",
                    "content": "Final recall: include tool facts."
                }),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(&session, "Done.");

        assert_eq!(repaired, "Done.");
    }

    #[test]
    fn repair_tool_result_answer_adds_title_for_tabular_row_id() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "list rows"}),
                tool_call_msg("call_1", "exec"),
                tool_result_msg(
                    "call_1",
                    "808\tStabilize tool loops through MoA\tfix/moa-tool-timeouts\tOPEN\n806\tAdd meshllm.cloud website, catalog viewer, and onboarding docs\tfeature/new-public-website\tOPEN",
                ),
                serde_json::json!({
                    "role": "user",
                    "content": "Answer with exactly one interesting PR number and title from the command output."
                }),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(&session, "#808");

        assert_eq!(repaired, "#808 - Stabilize tool loops through MoA");
    }

    #[test]
    fn repair_tool_result_answer_adds_missing_row_id_for_tabular_title() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "list rows"}),
                tool_call_msg("call_1", "exec"),
                tool_result_msg(
                    "call_1",
                    "793\tLet clients mark stable prompt anchors for Skippy cache\tcodex/skippy-prompt-anchor-cache\tDRAFT",
                ),
                serde_json::json!({
                    "role": "user",
                    "content": "Tell me one interesting row from the command output."
                }),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(
            &session,
            "#805 - Let clients mark stable prompt anchors for Skippy cache",
        );

        assert!(repaired.contains("#793 - Let clients mark stable prompt anchors"));
    }

    #[test]
    fn repair_tool_result_answer_replaces_unrelated_answer_with_structured_work_item() {
        let prompt = "Use shell/developer tool to look for recent important open GitHub issues or PRs in Mesh-LLM/mesh-llm. Inspect the results with a command, then reply with one specific issue or PR number/title and why it matters. Do not answer from memory.";
        let result = serde_json::json!([{
            "number": 808,
            "title": "Stabilize OpenClaw tool loops through MoA",
            "html_url": "https://github.com/Mesh-LLM/mesh-llm/pull/808",
            "state": "open",
            "body": "OpenClaw/Gateway users can now route Telegram-style agent turns through model=mesh without small-context workers or slow peers turning ordinary tool loops into timeouts."
        }])
        .to_string();
        let session = session_with_user_and_tool_result(prompt, &result);

        let repaired =
            repair_tool_result_answer(&session, "Hey. I just came online. Who am I? Who are you?");

        assert!(repaired.contains("#808 - Stabilize OpenClaw tool loops through MoA"));
        assert!(repaired.contains("https://github.com/Mesh-LLM/mesh-llm/pull/808"));
        assert!(!repaired.contains("Who am I"));
    }

    #[test]
    fn repair_tool_result_answer_recovers_work_item_from_embedded_json_fragment() {
        let prompt = "Use shell/developer tool to look for recent important open GitHub issues or PRs in Mesh-LLM/mesh-llm, then tell me about one. Do not answer from memory.";
        let result = r###"ented Plugin-only routing setup from an earlier issue body...","closed_by":null},{"url":"https://api.github.com/repos/Mesh-LLM/mesh-llm/issues/786","html_url":"https://github.com/Mesh-LLM/mesh-llm/pull/786","number":786,"title":"[codex] make installer runtime-first","state":"open","user":{"login":"i386"},"created_at":"2026-06-03T22:46:21Z","pull_request":{"html_url":"https://github.com/Mesh-LLM/mesh-llm/pull/786"},"body":"## Summary\nMakes install.sh runtime-first."},{"url":"https://api.github.com/repos/Mesh-LLM/mesh-llm/issues/775","html_url":"https://github.com/Mesh-LLM/mesh-llm/issues/775","number":775,"title":"Plan KV cache sizing","state":"open","user":{"login":"micn"},"created_at":"2026-06-02T00:00:00Z","body":"Long design note."}]"###;
        let session = session_with_user_and_tool_result(prompt, result);

        let repaired = repair_tool_result_answer(
            &session,
            "From the tool result, one relevant entry is: ented Plugin-only routing setup...",
        );

        assert!(repaired.contains("#786 - [codex] make installer runtime-first"));
        assert!(repaired.contains("https://github.com/Mesh-LLM/mesh-llm/pull/786"));
        assert!(!repaired.contains("ented Plugin-only routing"));
    }

    #[test]
    fn repair_tool_result_answer_trims_identity_preamble_before_work_item() {
        let prompt = "Use shell/developer tool to look for recent important open GitHub issues or PRs in Mesh-LLM/mesh-llm. Inspect the results with a command, then reply with one specific issue or PR number/title and why it matters. Do not answer from memory.";
        let result = serde_json::json!([{
            "number": 808,
            "title": "Stabilize OpenClaw tool loops through MoA",
            "html_url": "https://github.com/Mesh-LLM/mesh-llm/pull/808",
            "state": "open"
        }])
        .to_string();
        let session = session_with_user_and_tool_result(prompt, &result);
        let unrelated_prefix = "Hey. I just came online. Who am I? Who are you?\n\nI'm your personal assistant, running inside OpenClaw. I'm here to help you navigate the code, track down those tricky bugs, and maybe even find a little joy in the chaos of development. My vibe is collaborative, curious, and ready to dive into the weeds.\n\nNow, about that GitHub request. I dug into the recent activity.\n\n";
        let answer = format!(
            "{unrelated_prefix}**PR #808: Stabilize OpenClaw tool loops through MoA**\n\nThis matters because it targets agent tool-loop stability."
        );

        let repaired = repair_tool_result_answer(&session, &answer);

        assert!(repaired.starts_with("**PR #808"));
        assert!(!repaired.contains("Who am I"));
        assert!(repaired.contains("tool-loop stability"));
    }

    #[test]
    fn cli_usage_error_is_not_answerable_tool_evidence() {
        let result = "unknown flag: --since\r\n\r\nUsage:  gh pr list [flags]\r\n\r\nFlags:";

        assert!(!tool_result_has_answerable_evidence(result));
        assert!(answer_from_latest_tool_result(&session_with_tool_result(result)).is_none());
    }

    #[test]
    fn cli_json_field_error_is_not_answerable_tool_evidence() {
        let result = "Unknown JSON field: \"headRepositoryName\"\nAvailable fields:\n  additions\n  author\n  title\n\n(Command exited with code 1)";

        assert!(!tool_result_has_answerable_evidence(result));
        assert!(answer_from_latest_tool_result(&session_with_tool_result(result)).is_none());
    }

    #[test]
    fn shell_parse_error_is_not_answerable_tool_evidence() {
        let result = "(eval):1: parse error near `&'\n\n(Command exited with code 1)";

        assert!(!tool_result_has_answerable_evidence(result));
        assert!(answer_from_latest_tool_result(&session_with_tool_result(result)).is_none());
    }

    #[test]
    fn cli_missing_git_remote_is_not_answerable_tool_evidence() {
        let result = "no git remotes found\n\n(Command exited with code 1)";

        assert!(!tool_result_has_answerable_evidence(result));
        assert!(answer_from_latest_tool_result(&session_with_tool_result(result)).is_none());
    }

    #[test]
    fn cli_synthetic_unavailable_message_is_not_answerable_tool_evidence() {
        let result = "gh search not available or not authenticated";

        assert!(!tool_result_has_answerable_evidence(result));
        assert!(answer_from_latest_tool_result(&session_with_tool_result(result)).is_none());
    }

    #[test]
    fn empty_tool_output_sentinel_is_not_answerable_tool_evidence() {
        let result = "(no output)";

        assert!(!tool_result_has_answerable_evidence(result));
        assert!(answer_from_latest_tool_result(&session_with_tool_result(result)).is_none());
    }

    #[test]
    fn github_auth_status_is_not_pr_request_evidence() {
        let result = "github.com\n  ✓ Logged in to github.com account user (keyring)\n  - Active account: true\n  - Git operations protocol: https";
        let prompt = "Any new interesting PRs? You have the gh command line I think can use";

        assert!(tool_result_has_answerable_evidence(result));
        assert!(!tool_result_has_answerable_evidence_for_prompt(
            result, prompt
        ));
        assert!(
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, result))
                .is_none()
        );
    }

    #[test]
    fn github_auth_status_can_answer_auth_prompt() {
        let result = "github.com\n  ✓ Logged in to github.com account user (keyring)\n  - Active account: true";
        let prompt = "Check whether gh is authenticated.";

        assert!(tool_result_has_answerable_evidence_for_prompt(
            result, prompt
        ));
    }

    #[test]
    fn github_repo_list_is_not_pr_request_evidence() {
        let result = "michaelneale/sprout\tA hive mind communication platform\tpublic, fork\t2026-06-06T05:22:29Z\nmichaelneale/ollama\tRun models\tpublic, fork\t2026-06-05T00:42:49Z";
        let prompt = "Any new interesting PRs? You have the gh command line I think can use";

        assert!(tool_result_has_answerable_evidence(result));
        assert!(!tool_result_has_answerable_evidence_for_prompt(
            result, prompt
        ));
        assert!(
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, result))
                .is_none()
        );
    }

    #[test]
    fn github_repo_list_needs_repo_prompt() {
        let result = "aaif/working-group-proposals\tPropose a new working group\tpublic\t2026-05-29T18:18:10Z";

        assert!(tool_result_has_answerable_evidence(result));
        assert!(!tool_result_has_answerable_evidence_for_prompt(
            result,
            "Any new interesting items?"
        ));
    }

    #[test]
    fn github_repo_list_can_answer_repo_prompt() {
        let result = "michaelneale/sprout\tA hive mind communication platform\tpublic, fork\t2026-06-06T05:22:29Z";
        let prompt = "List interesting GitHub repos.";

        assert!(tool_result_has_answerable_evidence_for_prompt(
            result, prompt
        ));
    }

    #[test]
    fn git_remote_listing_is_not_pr_request_evidence() {
        let result = "origin\tgit@github.com:Mesh-LLM/mesh-llm.git (fetch)\norigin\tgit@github.com:Mesh-LLM/mesh-llm.git (push)";
        let prompt = "Any new interesting PRs? You have the gh command line I think can use";

        assert!(tool_result_has_answerable_evidence(result));
        assert!(!tool_result_has_answerable_evidence_for_prompt(
            result, prompt
        ));
        assert!(
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, result))
                .is_none()
        );
    }

    #[test]
    fn git_remote_listing_can_answer_remote_prompt() {
        let result = "origin\tgit@github.com:Mesh-LLM/mesh-llm.git (fetch)";
        let prompt = "What git remote is configured?";

        assert!(tool_result_has_answerable_evidence_for_prompt(
            result, prompt
        ));
    }

    #[test]
    fn pr_prompt_rejects_bad_or_wrong_domain_tool_result_permutations() {
        let pr_prompts = [
            "Any new interesting PRs? You have the gh command line I think can use",
            "Use gh to find recent open pulls.",
            "Find important GitHub pull requests.",
        ];
        let non_evidence_results = [
            "unknown flag: --sort\n\nUsage: gh pr list [flags]",
            "UNKNOWN JSON FIELD: \"repository\"\nAvailable fields:\n  author\n  title",
            "  no git remotes found\n\n(Command exited with code 1)",
            "gh search not available or not authenticated",
            "search failed: gh search not available or not authenticated",
            "GraphQL: Could not resolve to a Repository with the name 'micn/mesh-llm'. (repository)",
            "(no output)",
            "github.com\n  ✓ Logged in to github.com account user (keyring)\n  - Active account: true",
            "michaelneale/sprout\tA hive mind communication platform\tpublic, fork\t2026-06-06T05:22:29Z",
            "origin\tgit@github.com:Mesh-LLM/mesh-llm.git (fetch)\norigin\tgit@github.com:Mesh-LLM/mesh-llm.git (push)",
            "From https://github.com/Mesh-LLM/mesh-llm\n * branch            main       -> FETCH_HEAD\nAlready up to date.\n536f30a Reuse Skippy decode wire messages\nced32cc8e662a1433d51217787eb304da9121844\trefs/heads/acp-agent-pooling",
            "536f30a Reuse Skippy decode wire messages\n70cbe09 Stabilize OpenClaw tool loops\nbfe8012 Refresh docs",
        ];

        for prompt in pr_prompts {
            for result in non_evidence_results {
                assert!(
                    !tool_result_has_answerable_evidence_for_prompt(result, prompt),
                    "prompt {prompt:?} should not accept tool output {result:?} as PR evidence"
                );
                assert!(
                    answer_from_latest_tool_result(&session_with_user_and_tool_result(
                        prompt, result
                    ))
                    .is_none(),
                    "prompt {prompt:?} should not synthesize from {result:?}"
                );
            }
        }
    }

    #[test]
    fn pr_prompt_accepts_pr_row_permutations() {
        let prompt = "Use gh to list recent open PRs for mesh-LLM/mesh-llm.";
        let pr_results = [
            "808\tStabilize OpenClaw tool loops through MoA\tOPEN\t2026-06-06T06:17:33Z",
            "#808\tStabilize OpenClaw tool loops through MoA\tOPEN",
            "808\tStabilize OpenClaw tool loops through MoA",
        ];

        for result in pr_results {
            assert!(
                tool_result_has_answerable_evidence_for_prompt(result, prompt),
                "prompt {prompt:?} should accept PR output {result:?}"
            );
            assert!(
                answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, result))
                    .is_some(),
                "prompt {prompt:?} should synthesize from {result:?}"
            );
        }
    }

    #[test]
    fn pr_prompt_summarizes_json_array_rows() {
        let prompt = "Any new interesting PRs? You have the gh command line I think can use";
        let result = serde_json::json!([
            {
                "number": 809,
                "title": "feat: distributed mesh with PagedKV and Continuous Batching",
                "url": "https://github.com/Mesh-LLM/mesh-llm/pull/809",
                "createdAt": "2026-06-06T21:40:44Z",
                "author": {"login": "Jackson57279"}
            },
            {
                "number": 808,
                "title": "Stabilize OpenClaw tool loops through MoA",
                "url": "https://github.com/Mesh-LLM/mesh-llm/pull/808",
                "author": {"login": "michaelneale"}
            }
        ])
        .to_string();

        let answer =
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, &result))
                .expect("JSON PR array should synthesize a usable answer");

        assert!(answer.contains("#809 - feat: distributed mesh"), "{answer}");
        assert!(answer.contains("https://github.com/Mesh-LLM/mesh-llm/pull/809"));
        assert!(answer.contains("#808 - Stabilize OpenClaw"), "{answer}");
        assert!(!answer.contains("one relevant entry is: ["), "{answer}");
    }

    #[test]
    fn pr_status_prompt_summarizes_checks_and_copilot_feedback_json() {
        let prompt = "The stabilise one, PR 808 - is CI all green now? Any Copilot feedback?";
        let result = serde_json::json!({
            "title": "Stabilize OpenClaw tool loops through MoA",
            "state": "OPEN",
            "statusCheckRollup": [
                {
                    "name": "Linux tests (skippy-smoke)",
                    "conclusion": "FAILURE",
                    "status": "COMPLETED"
                },
                {
                    "name": "Linux CPU",
                    "conclusion": "SUCCESS",
                    "status": "COMPLETED"
                }
            ],
            "comments": [
                {
                    "author": {"login": "github-copilot[bot]"},
                    "body": "Copilot reviewed this and requested a small error handling change."
                }
            ],
            "reviews": []
        })
        .to_string()
            + "\n[... truncated]";

        let answer =
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, &result))
                .expect("PR detail JSON should answer status prompt");

        assert!(answer.contains("Stabilize OpenClaw"), "{answer}");
        assert!(answer.contains("not all green"), "{answer}");
        assert!(
            answer.contains("Linux tests (skippy-smoke): FAILURE"),
            "{answer}"
        );
        assert!(answer.contains("Copilot feedback"), "{answer}");
        assert!(answer.contains("github-copilot[bot]"), "{answer}");
    }

    #[test]
    fn pr_status_prompt_rejects_comments_only_json_without_checks() {
        let prompt = "The stabilise one, PR 808 - is CI all green now? Any Copilot feedback?";
        let result = serde_json::json!({
            "title": "Stabilize OpenClaw tool loops through MoA",
            "state": "OPEN",
            "comments": [
                {
                    "author": {"login": "github-copilot[bot]"},
                    "body": "Copilot reviewed this PR."
                }
            ],
            "reviews": [
                {
                    "author": {"login": "copilot-pull-request-reviewer"},
                    "state": "COMMENTED",
                    "body": "Pull request overview"
                }
            ]
        })
        .to_string();

        assert!(
            !tool_result_has_answerable_evidence_for_prompt(result.as_str(), prompt),
            "CI prompt should not accept comments-only PR detail JSON"
        );
        assert!(
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, &result))
                .is_none()
        );
    }

    #[test]
    fn weather_check_prompt_does_not_trigger_ci_status_summary() {
        let prompt =
            "Use the weather lookup tool to check Sydney. Now answer in one short sentence.";
        let result = serde_json::json!({
            "city": "Sydney",
            "condition": "clear",
            "temperature_c": 21
        })
        .to_string();
        let session = session_with_user_and_tool_result(prompt, &result);

        assert_eq!(
            answer_from_prompt_specific_structured_tool_result(&result, prompt),
            None
        );
        let answer = answer_from_latest_tool_result(&session).expect("weather JSON is evidence");
        assert!(answer.contains("Sydney"), "{answer}");
        assert!(!answer.contains("CI/checks"), "{answer}");
        assert_eq!(
            repair_tool_result_answer(&session, "Sydney is clear at 21C."),
            "Sydney is clear at 21C."
        );
    }

    #[test]
    fn pr_status_prompt_summarizes_jsonl_check_output() {
        let prompt = "Please check PR 808 checks/CI status.";
        let result = r#"{"name":"changes","status":"COMPLETED","conclusion":"SUCCESS"}
{"name":"Linux tests (skippy-smoke)","status":"COMPLETED","conclusion":"FAILURE"}
{"name":"two_node_split_smoke","status":"COMPLETED","conclusion":"SUCCESS"}"#;

        let answer =
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, result))
                .expect("JSONL check output should answer CI prompt");

        assert!(answer.contains("CI/checks: not all green"), "{answer}");
        assert!(
            answer.contains("Linux tests (skippy-smoke): FAILURE"),
            "{answer}"
        );
    }

    #[test]
    fn pr_status_and_copilot_prompt_rejects_checks_only_output() {
        let prompt = "PR 808 - is CI all green now? Any Copilot feedback?";
        let result = "Linux tests (skippy-smoke)\tfail\t1m39s\thttps://example.invalid/job";

        assert!(
            !tool_result_has_answerable_evidence_for_prompt(result, prompt),
            "checks-only output should not satisfy Copilot feedback prompt"
        );
    }

    #[test]
    fn pr_status_and_copilot_prompt_combines_recent_structured_tool_results() {
        let prompt = "PR 808 - is CI all green now? Any Copilot feedback?";
        let status_result = serde_json::json!({
            "title": "Stabilize OpenClaw tool loops through MoA",
            "state": "OPEN",
            "statusCheckRollup": [
                {
                    "name": "Linux tests (skippy-smoke)",
                    "conclusion": "FAILURE",
                    "status": "COMPLETED"
                }
            ]
        })
        .to_string();
        let review_result = serde_json::json!({
            "author": "copilot-pull-request-reviewer[bot]",
            "body": "## Pull request overview\n\nCopilot reviewed this PR.",
            "state": "COMMENTED"
        })
        .to_string();
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": prompt}),
                tool_call_msg_with_args("call_1", "exec", r#"{"command":"gh pr view"}"#),
                tool_result_msg("call_1", &status_result),
                tool_call_msg_with_args("call_2", "exec", r#"{"command":"gh api reviews"}"#),
                tool_result_msg("call_2", &review_result),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(&session, "CI/checks: pending.");

        assert!(
            repaired.contains("Linux tests (skippy-smoke): FAILURE"),
            "{repaired}"
        );
        assert!(repaired.contains("Copilot feedback"), "{repaired}");
        assert!(
            repaired.contains("copilot-pull-request-reviewer[bot]"),
            "{repaired}"
        );
    }

    #[test]
    fn generic_status_prompt_summarizes_nested_job_results() {
        let prompt = "Is the deployment status green?";
        let result = serde_json::json!({
            "service": "billing-api",
            "pipeline": {
                "jobs": [
                    {"jobName": "unit", "status": "SUCCESS"},
                    {"jobName": "canary", "status": "FAILED"}
                ]
            }
        })
        .to_string();

        let answer =
            answer_from_latest_tool_result(&session_with_user_and_tool_result(prompt, &result))
                .expect("generic nested job output should answer status prompt");

        assert!(answer.contains("CI/checks: not all green"), "{answer}");
        assert!(answer.contains("canary: FAILED"), "{answer}");
    }

    #[test]
    fn generic_status_and_feedback_prompt_combines_recent_structured_tool_results() {
        let prompt = "Is the release status green, and any review feedback?";
        let status_result = serde_json::json!({
            "release": "2026.06.08",
            "checks": [
                {"name": "deploy", "result": "passed"},
                {"name": "smoke", "result": "failed"}
            ]
        })
        .to_string();
        let feedback_result = serde_json::json!({
            "reviews": [
                {
                    "author": {"name": "nick"},
                    "state": "COMMENTED",
                    "message": "Please add one more timeout guard."
                }
            ]
        })
        .to_string();
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": prompt}),
                tool_call_msg_with_args("call_1", "lookup", r#"{"id":"release"}"#),
                tool_result_msg("call_1", &status_result),
                tool_call_msg_with_args("call_2", "lookup", r#"{"id":"reviews"}"#),
                tool_result_msg("call_2", &feedback_result),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(&session, "Status is still running.");

        assert!(repaired.contains("smoke: FAILED"), "{repaired}");
        assert!(repaired.contains("Feedback"), "{repaired}");
        assert!(repaired.contains("review by nick"), "{repaired}");
    }

    #[test]
    fn normal_tabular_cli_output_is_answerable_tool_evidence() {
        let result = "808\tStabilize tool loops through MoA\tOPEN";

        assert!(tool_result_has_answerable_evidence(result));
    }

    #[test]
    fn fenced_command_answer_becomes_schema_declared_tool_call() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "runner",
                "description": "Run shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        }]));
        let allowed_tools = ["runner".to_string()];

        let response = chat_or_schema_command_tool_response(
            "I will check it.\n\n```bash\ngh pr list --repo mesh-LLM/mesh-llm --state open\n```",
            tools.as_ref(),
            &allowed_tools,
            &[],
            Some("Use gh to list recent open PRs for mesh-LLM/mesh-llm."),
        );

        assert_eq!(
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("runner")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"command\":\"gh pr list --repo mesh-LLM/mesh-llm --state open\"}")
        );
    }

    #[test]
    fn fenced_command_answer_uses_selected_tool_subset_when_catalog_is_ambiguous() {
        let tools = Some(serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "remote_exec",
                    "description": "Run terminal commands remotely",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Terminal command to run"}
                        }
                    }
                }
            }
        ]));
        let allowed_tools = ["exec".to_string(), "remote_exec".to_string()];
        let selected_tools = ["exec".to_string()];
        let content =
            "I will check it.\n\n```bash\ngh pr list --repo mesh-LLM/mesh-llm --state open\n```";

        let ambiguous_response = chat_or_schema_command_tool_response(
            content,
            tools.as_ref(),
            &allowed_tools,
            &[],
            Some("Use gh to list recent open PRs for mesh-LLM/mesh-llm."),
        );
        assert!(
            ambiguous_response
                .pointer("/choices/0/message/tool_calls")
                .is_none(),
            "full ambiguous catalog should not pick a command tool arbitrarily: {ambiguous_response}"
        );

        let selected_response = chat_or_schema_command_tool_response(
            content,
            tools.as_ref(),
            response_tool_names(&selected_tools, &allowed_tools),
            &[],
            Some("Use gh to list recent open PRs for mesh-LLM/mesh-llm."),
        );
        assert_eq!(
            selected_response
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
    }

    #[test]
    fn fenced_command_answer_becomes_prompt_declared_tool_call_without_native_tools() {
        let (session, profiles, allowed) =
            prompt_tool_session("Use gh to list recent open PRs for mesh-LLM/mesh-llm.");

        let response = chat_or_schema_command_tool_response(
            "I will check it.\n\n```bash\ngh pr list --repo mesh-LLM/mesh-llm --state open --limit 10\n```",
            session.tools(),
            &allowed,
            &profiles,
            Some(&session.last_user_text()),
        );

        assert_eq!(
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"command\":\"gh pr list --repo mesh-LLM/mesh-llm --state open --limit 10\"}")
        );
    }

    #[test]
    fn fenced_command_answer_stays_chat_when_user_did_not_request_execution() {
        let (session, profiles, allowed) = prompt_tool_session("look at that first");

        let response = chat_or_schema_command_tool_response(
            "The prior command was:\n\n```bash\ngh pr list --repo mesh-LLM/mesh-llm --state open\n```",
            session.tools(),
            &allowed,
            &profiles,
            Some(&session.last_user_text()),
        );

        assert!(
            response.pointer("/choices/0/message/tool_calls").is_none(),
            "non-execution turns must not be corrected into a tool call: {response}"
        );
        assert_eq!(
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("stop")
        );
    }

    #[test]
    fn inline_json_tool_choice_uses_declared_schema() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }]));
        let allowed_tools = ["read_file".to_string()];

        let choice = inline_json_tool_choice_from_text(
            r#"I'll read the README. {"function": "read_file", "arguments": {"path": "README.md"}}"#,
            tools.as_ref(),
            &allowed_tools,
            &[],
        )
        .expect("inline tool choice");

        assert_eq!(choice.name, "read_file");
        assert_eq!(choice.fallback_arguments, json!({"path": "README.md"}));
    }

    #[test]
    fn prompt_catalog_rejects_unknown_inline_tool() {
        let (session, profiles, allowed) = prompt_tool_session("hello");

        let response = chat_or_schema_command_tool_response(
            r#"{"function": "delete_everything", "arguments": {"path": "/"}}"#,
            session.tools(),
            &allowed,
            &profiles,
            None,
        );

        assert!(
            response.pointer("/choices/0/message/tool_calls").is_none(),
            "unknown prompt-only tools must stay as chat text: {response}"
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str),
            Some(r#"{"function": "delete_everything", "arguments": {"path": "/"}}"#)
        );
    }

    #[test]
    fn prompt_catalog_rejects_empty_explicit_tool_args() {
        let (session, profiles, allowed) =
            prompt_tool_session("Use gh to list recent open PRs for mesh-LLM/mesh-llm.");

        let response = chat_or_schema_command_tool_response(
            "I will use the `exec` tool to check.",
            session.tools(),
            &allowed,
            &profiles,
            Some(&session.last_user_text()),
        );

        assert!(
            response.pointer("/choices/0/message/tool_calls").is_none(),
            "prompt-declared tools must not emit empty argument calls: {response}"
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str),
            Some("I will use the `exec` tool to check.")
        );
    }

    #[test]
    fn prompt_catalog_accepts_malformed_xml_parameter_close() {
        let (session, profiles, allowed) =
            prompt_tool_session("Use gh to list recent open PRs for mesh-LLM/mesh-llm.");

        let response = chat_or_schema_command_tool_response(
            "<tool_call>\n<invoke name=\"exec\">\n<parameter name=\"command\">gh pr list --repo mesh-LLM/mesh-llm --state open --limit 10</command>\n</invoke>\n</tool_call>",
            session.tools(),
            &allowed,
            &profiles,
            Some(&session.last_user_text()),
        );

        assert_eq!(
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"command\":\"gh pr list --repo mesh-LLM/mesh-llm --state open --limit 10\"}")
        );
    }

    #[test]
    fn prompt_catalog_xml_accepts_single_quotes_and_double_escaped_entities() {
        let (session, profiles, allowed) = prompt_tool_session("Use printf with the exec tool.");

        let response = chat_or_schema_command_tool_response(
            "<tool_call>\n<invoke name='exec'>\n<parameter name='command'>printf &amp;quot;hello&#34;</parameter>\n</invoke>\n</tool_call>",
            session.tools(),
            &allowed,
            &profiles,
            Some(&session.last_user_text()),
        );

        assert_eq!(
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"command\":\"printf \\\"hello\\\"\"}")
        );
    }

    #[test]
    fn body_style_xml_tool_call_maps_to_single_command_tool() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute shell commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        }]));
        let response = chat_or_schema_command_tool_response(
            "I'll use the developer tool.\n<tool_call>\n<invoke>tool: developer, args: {\\n<param name=\"command\">pwd</param>}\n</tool_call>",
            tools.as_ref(),
            &["shell".to_string()],
            &[],
            Some("Run pwd."),
        );

        assert_eq!(
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("shell")
        );
        assert_eq!(
            response
                .pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"command\":\"pwd\"}")
        );
    }

    fn tool_call_msg(id: &str, name: &str) -> Value {
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": "{\"query\":\"x\"}"}
            }]
        })
    }

    fn tool_call_msg_with_args(id: &str, name: &str, arguments: &str) -> Value {
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments}
            }]
        })
    }

    fn tool_result_msg(id: &str, text: &str) -> Value {
        serde_json::json!({
            "role": "tool",
            "tool_call_id": id,
            "content": text
        })
    }

    fn command_tool_schema(name: &str) -> Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": name,
                "description": "Run shell commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        }])
    }

    fn session_with_tool_result(result: &str) -> Session {
        session_with_user_and_tool_result("run command", result)
    }

    fn session_with_user_and_tool_result(prompt: &str, result: &str) -> Session {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": prompt}),
                tool_call_msg("call_1", "exec"),
                tool_result_msg("call_1", result),
            ],
            &None,
        );
        session
    }

    #[test]
    fn empty_tool_result_plain_answer_retries_same_tool_call() {
        let command =
            r#"{"command":"gh pr list --repo Mesh-LLM/mesh-llm --state open --limit 10"}"#;
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({
                    "role": "user",
                    "content": "Use exec to list recent open PRs, then tell me one interesting one."
                }),
                tool_call_msg_with_args("call_1", "exec", command),
                tool_result_msg("call_1", "(no output)"),
            ],
            &None,
        );

        let retry = latest_completed_tool_call(&session).expect("completed tool call");
        let body = retry_tool_result_response_if_plain(
            chat_response("I could not retrieve any pull requests."),
            Some(&retry),
            Some("(no output)"),
            None,
            &[],
            &[],
            true,
        );

        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some(command)
        );
        assert!(
            body.pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .is_none()
        );
    }

    #[test]
    fn shell_parse_error_retries_schema_corrected_command_tool_call() {
        let tools = command_tool_schema("exec");
        let allowed = ["exec".to_string()];
        let command = r#"{"command":"gh api repos/Mesh-LLM/mesh-llm/issues?state=open&sort=updated&direction=desc --jq '.items[0:1]'"}"#;
        let retry = CompletedToolCall {
            name: "exec".to_string(),
            arguments: command.to_string(),
        };

        let body = retry_tool_result_response_if_plain(
            chat_response("The GitHub API command failed."),
            Some(&retry),
            Some("(eval):1: parse error near `&'\n\n(Command exited with code 1)"),
            Some(&tools),
            &allowed,
            &[],
            true,
        );

        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        let args = body
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .expect("tool args");
        assert_eq!(
            args.get("command").and_then(Value::as_str),
            Some(
                "gh api 'repos/Mesh-LLM/mesh-llm/issues?state=open&sort=updated&direction=desc' --jq '.items[0:1]'"
            )
        );
    }

    #[test]
    fn shell_glob_error_retries_schema_corrected_command_tool_call() {
        let tools = command_tool_schema("exec");
        let allowed = ["exec".to_string()];
        let command =
            r#"{"command":"gh api repos/mesh-llm/mesh-llm/issues?state=open&per_page=1"}"#;
        let retry = CompletedToolCall {
            name: "exec".to_string(),
            arguments: command.to_string(),
        };
        let result = "zsh:1: no matches found: repos/mesh-llm/mesh-llm/issues?state=open";
        assert!(!tool_result_has_answerable_evidence(result));

        let body = retry_tool_result_response_if_plain(
            chat_response("The shell tool returned no matches found."),
            Some(&retry),
            Some(result),
            Some(&tools),
            &allowed,
            &[],
            true,
        );

        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        let args = body
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .expect("tool args");
        assert_eq!(
            args.get("command").and_then(Value::as_str),
            Some("gh api 'repos/mesh-llm/mesh-llm/issues?state=open&per_page=1'")
        );
    }

    #[test]
    fn shell_glob_error_corrects_new_bad_followup_tool_call() {
        let tools = command_tool_schema("exec");
        let allowed = ["exec".to_string()];
        let previous = CompletedToolCall {
            name: "exec".to_string(),
            arguments: r#"{"command":"curl -s https://api.github.com/repos/Mesh-LLM/mesh-llm/issues?state=open\\&per_page=1"}"#.to_string(),
        };
        let followup = r#"{"command":"curl -s https://api.github.com/repos/Mesh-LLM/mesh-llm/issues?state=open&per_page=1 | jq -r '.[0]'"}"#;
        let result = "zsh:1: no matches found: https://api.github.com/repos/Mesh-LLM/mesh-llm/issues?state=open";

        let body = retry_tool_result_response_if_plain(
            tool_call_response("exec", &Value::String(followup.to_string())),
            Some(&previous),
            Some(result),
            Some(&tools),
            &allowed,
            &[],
            true,
        );

        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        let args = body
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .expect("tool args");
        assert_eq!(
            args.get("command").and_then(Value::as_str),
            Some(
                "curl -s 'https://api.github.com/repos/Mesh-LLM/mesh-llm/issues?state=open&per_page=1' | jq -r '.[0]'"
            )
        );
    }

    #[test]
    fn cli_unknown_flag_error_retries_schema_corrected_command_tool_call() {
        let tools = command_tool_schema("exec");
        let allowed = ["exec".to_string()];
        let command =
            r#"{"command":"gh api repos/Mesh-LLM/mesh-llm/issues --state open --limit 1"}"#;
        let retry = CompletedToolCall {
            name: "exec".to_string(),
            arguments: command.to_string(),
        };
        let result = "unknown flag: --state\n\nUsage:  gh api <endpoint> [flags]\n\nFlags:\n      --cache duration        Cache the response\n  -H, --header key:value      Add a HTTP request header\n      --paginate              Make additional HTTP requests\n\n(Command exited with code 1)";

        let body = retry_tool_result_response_if_plain(
            chat_response("Let me try again with a corrected command."),
            Some(&retry),
            Some(result),
            Some(&tools),
            &allowed,
            &[],
            true,
        );

        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        let args = body
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .expect("tool args");
        assert_eq!(
            args.get("command").and_then(Value::as_str),
            Some("gh api repos/Mesh-LLM/mesh-llm/issues")
        );
    }

    #[test]
    fn jq_shell_quoting_error_retries_schema_corrected_command_tool_call() {
        let tools = command_tool_schema("exec");
        let allowed = ["exec".to_string()];
        let command = r#"{"command":"curl -s \"https://api.github.com/repos/Mesh-LLM/mesh-llm/issues/808\" | jq -r '\\\"\\(.number) \\(.state) \\(.title)\\\"'"}"#;
        let retry = CompletedToolCall {
            name: "exec".to_string(),
            arguments: command.to_string(),
        };
        let result = "jq: error: syntax error, unexpected INVALID_CHARACTER, expecting end of file (Unix shell quoting issues?) at <top-level>, line 1:\n\\\"\\(.number) \\(.state) \\(.title)\\\"\njq: 1 compile error\n\n(Command exited with code 3)";

        let body = retry_tool_result_response_if_plain(
            chat_response("I encountered an error while trying to fetch the details."),
            Some(&retry),
            Some(result),
            Some(&tools),
            &allowed,
            &[],
            true,
        );

        assert_eq!(
            body.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("exec")
        );
        let args = body
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .expect("tool args");
        assert_eq!(
            args.get("command").and_then(Value::as_str),
            Some(
                "curl -s \"https://api.github.com/repos/Mesh-LLM/mesh-llm/issues/808\" | jq -r '\"\\(.number) \\(.state) \\(.title)\"'"
            )
        );
    }

    #[test]
    fn explicit_multi_step_tool_prompt_keeps_tool_available_after_first_result() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({
                    "role": "user",
                    "content": "Actually use the shell tool. Run pwd, then run ls -1 | head -5."
                }),
                tool_call_msg_with_args("call_1", "exec", r#"{"command":"pwd"}"#),
                tool_result_msg("call_1", "/tmp/workspace"),
            ],
            &None,
        );

        assert!(prompt_requests_additional_tool_step(
            &session.active_user_text(),
            &session
        ));
    }

    #[test]
    fn goose_info_message_does_not_reset_multi_step_tool_count() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({
                    "role": "user",
                    "content": "Actually use the shell/developer tool. Run pwd, then run ls -1 | head -5, then answer."
                }),
                tool_call_msg_with_args("call_1", "shell", r#"{"command":"pwd"}"#),
                tool_result_msg("call_1", "/tmp/workspace"),
                serde_json::json!({
                    "role": "user",
                    "content": "<info-msg>\nWorking directory: /tmp/workspace\n</info-msg>"
                }),
                tool_call_msg_with_args("call_2", "shell", r#"{"command":"ls -1 | head -5"}"#),
                tool_result_msg("call_2", "AGENTS.md\nCargo.toml"),
            ],
            &None,
        );

        let completed = completed_tool_calls_since_latest_user_before_tool_result(&session);

        assert_eq!(completed.len(), 2);
        assert_eq!(
            session.active_user_text(),
            "Actually use the shell/developer tool. Run pwd, then run ls -1 | head -5, then answer."
        );
        assert!(!prompt_requests_additional_tool_step(
            &session.active_user_text(),
            &session
        ));
    }

    #[test]
    fn goose_info_prefix_preserves_tool_intent_text() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "<info-msg>\nWorking directory: /tmp/project\n</info-msg>\nActually use the shell tool. Run pwd."
            })],
            &Some(serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"}
                        },
                        "required": ["command"]
                    }
                }
            }])),
        );

        assert_eq!(
            session.last_user_text(),
            "Actually use the shell tool. Run pwd."
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["shell".to_string()]
        );
    }

    #[test]
    fn malformed_goose_info_prefix_does_not_hide_tool_intent_text() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "<info-msg>\nWorking directory: /tmp/project\nActually use the shell tool. Run pwd."
            })],
            &Some(serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"}
                        },
                        "required": ["command"]
                    }
                }
            }])),
        );

        assert_eq!(
            session.last_user_text(),
            "<info-msg>\nWorking directory: /tmp/project\nActually use the shell tool. Run pwd."
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            vec!["shell".to_string()]
        );
    }

    #[test]
    fn ordinary_prompt_selects_no_tools() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "Help"})],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[], &[]),
            Vec::<String>::new()
        );
    }
}
