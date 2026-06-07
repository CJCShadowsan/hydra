//! Pin the tool-result turn classification.
//!
//! Background — PR #566 review feedback (Apr 2026):
//!
//! > The tool-result path isn't ready for agent loops:
//! > - A tool-result follow-up was treated like another fanout turn.
//! > - It wasn't handled like a controlled reducer/synthesis turn.
//! > - Tool results should be handled carefully and predictably, not
//! >   sprayed back through the whole fanout path.
//!
//! When the conversation has a recent tool result that has not yet
//! been synthesized into an assistant answer, the gateway must take
//! the reducer-only path (`TurnKind::ToolResult`), not fan-out to all
//! workers. Fanning out wastes a round-trip per worker, drowns the
//! reducer in worker outputs that ignore the tool result, and \u2014 most
//! dangerously \u2014 invites a worker to re-propose the same tool call
//! whose result we already have in-context.
//!
//! Two shapes of agent conversation must classify as ToolResult:
//!
//! 1. **OpenAI canonical shape.** The last message has role `tool`
//!    (the harness sent the tool result and expects the next assistant
//!    turn to interpret it). This is the simplest and most explicit
//!    shape. Classifying this is straightforward.
//!
//! 2. **Trailing-user shape.** The conversation ends with assistant
//!    tool_calls + tool result + a user message that just nudges
//!    ("continue", "what did you find?"). The harness has left the
//!    tool result in-context for the model to consume. There is no
//!    new tool result in *this* turn, but the previous one was never
//!    synthesized into an assistant message. Today this classifies
//!    as `Continuation` and fans out \u2014 wrong per the review.
//!
//! This file pins both shapes.

use async_trait::async_trait;
use mesh_mixture_of_agents as moa;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Backend that records every call it receives so the test can assert
/// fan-out did or did not happen.
struct RecordingBackend {
    text: String,
    delay: Duration,
    calls: AtomicUsize,
}

struct ToolSchemaFailBackend {
    calls_with_tools: AtomicUsize,
    calls_without_tools: AtomicUsize,
}

struct InspectingAnswerBackend {
    calls_with_tools: AtomicUsize,
    calls_without_tools: AtomicUsize,
    native_tool_messages: AtomicUsize,
}

impl ToolSchemaFailBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls_with_tools: AtomicUsize::new(0),
            calls_without_tools: AtomicUsize::new(0),
        })
    }

    fn calls_with_tools(&self) -> usize {
        self.calls_with_tools.load(Ordering::SeqCst)
    }

    fn calls_without_tools(&self) -> usize {
        self.calls_without_tools.load(Ordering::SeqCst)
    }
}

impl InspectingAnswerBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls_with_tools: AtomicUsize::new(0),
            calls_without_tools: AtomicUsize::new(0),
            native_tool_messages: AtomicUsize::new(0),
        })
    }

    fn calls_with_tools(&self) -> usize {
        self.calls_with_tools.load(Ordering::SeqCst)
    }

    fn calls_without_tools(&self) -> usize {
        self.calls_without_tools.load(Ordering::SeqCst)
    }

    fn native_tool_messages(&self) -> usize {
        self.native_tool_messages.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl moa::ModelBackend for ToolSchemaFailBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        if tools.is_some() {
            self.calls_with_tools.fetch_add(1, Ordering::SeqCst);
            return Err("tool grammar rejected request".to_string());
        }

        if messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                || message.get("tool_calls").is_some()
        }) {
            return Err("native tool messages leaked into answer-only retry".to_string());
        }

        self.calls_without_tools.fetch_add(1, Ordering::SeqCst);
        Ok(json!({
            "choices": [{
                "message": {"content": "The tool output lists file_a.log and file_b.log."}
            }],
        }))
    }
}

#[async_trait]
impl moa::ModelBackend for InspectingAnswerBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        if tools.is_some() {
            self.calls_with_tools.fetch_add(1, Ordering::SeqCst);
        } else {
            self.calls_without_tools.fetch_add(1, Ordering::SeqCst);
        }

        if messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                || message.get("tool_calls").is_some()
        }) {
            self.native_tool_messages.fetch_add(1, Ordering::SeqCst);
        }

        Ok(json!({
            "choices": [{
                "message": {"content": "Recent PR evidence is thin; review the latest open pull requests page and prioritize obvious bug-fix PRs."}
            }],
        }))
    }
}

impl RecordingBackend {
    fn new(text: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            text: text.into(),
            delay: Duration::from_millis(10),
            calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl moa::ModelBackend for RecordingBackend {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Value],
        _tools: Option<&Value>,
        _max_tokens: u32,
        _timeout: Duration,
        _sampling: moa::SamplingParams,
    ) -> Result<Value, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        Ok(json!({
            "choices": [{"message": {"content": self.text}}],
        }))
    }
}

fn config_with_three_recording_workers() -> (
    moa::GatewayConfig,
    Arc<RecordingBackend>,
    Arc<RecordingBackend>,
    Arc<RecordingBackend>,
) {
    let fast = RecordingBackend::new("synthesised: README says 'Hello World'");
    let mid = RecordingBackend::new("synthesised: README says 'Hello World'");
    let strong = RecordingBackend::new("synthesised: README says 'Hello World'");

    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![fast.clone(), mid.clone(), strong.clone()];
    let models = vec![
        moa::ModelEntry {
            name: "fast-3b".into(),
            backend_index: 0,
        },
        moa::ModelEntry {
            name: "mid-13b".into(),
            backend_index: 1,
        },
        moa::ModelEntry {
            name: "strong-32b".into(),
            backend_index: 2,
        },
    ];

    let config = moa::GatewayConfig {
        backends,
        models,
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
        enable_thinking: None,
    };
    (config, fast, mid, strong)
}

fn read_file_tool() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }
        }
    }])
}

fn web_tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch a URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                }
            }
        }
    ])
}

#[tokio::test]
async fn exhausted_tool_budget_forces_answer_only_context() {
    let backend = InspectingAnswerBackend::new();
    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![backend.clone()];
    let config = moa::GatewayConfig {
        backends,
        models: vec![moa::ModelEntry {
            name: "strong-32b".into(),
            backend_index: 0,
        }],
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
        enable_thinking: None,
    };

    let body = json!({
        "model": "mesh",
        "tools": web_tools(),
        "messages": [
            {"role": "user", "content": "Find recent important GitHub bug-fix PRs for Mesh-LLM/mesh-llm."},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{\"query\":\"mesh-llm recent bug fixes\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"results\":[{\"url\":\"https://github.com/Mesh-LLM/mesh-llm/pulls\"}]}"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://github.com/Mesh-LLM/mesh-llm/pulls\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_2", "content": "{\"status\":200,\"title\":\"Pull requests\"}"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_3",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{\"query\":\"site:github.com/Mesh-LLM/mesh-llm/pulls bug fix\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_3", "content": "{\"results\":[]}"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_4",
                "type": "function",
                "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://api.github.com/repos/Mesh-LLM/mesh-llm/pulls\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_4", "content": "{\"status\":200,\"items\":[]}"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_5",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{\"query\":\"Mesh-LLM mesh-llm pull request important bug\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_5", "content": "{\"status\":\"error\",\"error\":\"bot challenge\"}"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_6",
                "type": "function",
                "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://github.com/Mesh-LLM/mesh-llm/pulls?q=is%3Apr+sort%3Aupdated-desc\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_6", "content": "{\"status\":200,\"title\":\"Pull requests\"}"},
        ],
        "max_tokens": 128,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(result.turn_kind, moa::TurnKind::ToolResult);
    assert_eq!(
        backend.calls_with_tools(),
        0,
        "tool budget exhaustion must stop forwarding tool schemas"
    );
    assert_eq!(backend.calls_without_tools(), 1);
    assert_eq!(
        backend.native_tool_messages(),
        0,
        "budgeted synthesis should use compact answer-only context, not native tool messages"
    );
    assert_eq!(
        result
            .response_body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str),
        Some("stop")
    );
    assert!(
        result
            .response_body
            .pointer("/choices/0/message/tool_calls")
            .is_none(),
        "budget exhaustion must produce an answer, not another tool call"
    );
}

#[tokio::test]
async fn repeated_tool_guard_allows_different_tool_from_minimax_xml() {
    let backend = RecordingBackend::new(
        r#"<minimax:tool_call>
<invoke name="web_fetch">
<parameter name="url">https://github.com/Mesh-LLM/mesh-llm/issues</parameter>
</invoke>
</minimax:tool_call>"#,
    );
    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![backend.clone()];
    let config = moa::GatewayConfig {
        backends,
        models: vec![moa::ModelEntry {
            name: "minimax".into(),
            backend_index: 0,
        }],
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
        enable_thinking: None,
    };

    let body = json!({
        "model": "mesh",
        "tools": web_tools(),
        "messages": [
            {"role": "user", "content": "Use web_search or web_fetch as needed. Find important GitHub issues."},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{\"query\":\"Mesh-LLM issues\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"results\":[]}"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{\"query\":\"Mesh-LLM bug fixes\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_2", "content": "{\"results\":[]}"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_3",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{\"query\":\"Mesh-LLM pull requests\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_3", "content": "{\"results\":[]}"},
        ],
        "max_tokens": 64,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(result.turn_kind, moa::TurnKind::ToolResult);
    assert_eq!(backend.calls(), 1);
    assert_eq!(
        result
            .response_body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str),
        Some("tool_calls"),
        "Minimax XML tool proposal must not leak as assistant text: {}",
        result.response_body
    );
    assert_eq!(
        result
            .response_body
            .pointer("/choices/0/message/tool_calls/0/function/name")
            .and_then(Value::as_str),
        Some("web_fetch")
    );
    assert_eq!(
        result
            .response_body
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(Value::as_str),
        Some("{\"url\":\"https://github.com/Mesh-LLM/mesh-llm/issues\"}")
    );
}

#[tokio::test]
async fn non_empty_tool_result_suppresses_same_tool_followup() {
    let backend = RecordingBackend::new(
        r#"{"kind":"tool_proposal","confidence":0.9,"tool":"exec","arguments":{"command":"gh pr view 806 --repo mesh-LLM/mesh-llm"}}"#,
    );
    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![backend.clone()];
    let config = moa::GatewayConfig {
        backends,
        models: vec![moa::ModelEntry {
            name: "strong-32b".into(),
            backend_index: 0,
        }],
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
        enable_thinking: None,
    };

    let body = json!({
        "model": "mesh",
        "tools": [{
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
        }],
        "messages": [
            {"role": "user", "content": "Use gh to list recent open PRs, then tell me one interesting one."},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_exec",
                "type": "function",
                "function": {"name": "exec", "arguments": "{\"command\":\"gh pr list --repo mesh-LLM/mesh-llm\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_exec", "content": "#806 Add meshllm.cloud website and onboarding docs"},
        ],
        "max_tokens": 64,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(result.turn_kind, moa::TurnKind::ToolResult);
    assert_eq!(backend.calls(), 1);
    assert_eq!(
        result
            .response_body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str),
        Some("stop")
    );
    assert!(
        result
            .response_body
            .pointer("/choices/0/message/tool_calls")
            .is_none(),
        "same tool must not be emitted again after answerable evidence: {}",
        result.response_body
    );
    let content = result
        .response_body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        content.contains("#806 Add meshllm.cloud website and onboarding docs"),
        "answer should be grounded in the completed tool result: {content}"
    );
    assert!(
        !content.contains("tool_proposal") && !content.contains("gh pr view"),
        "suppressed tool proposals must not leak as assistant text: {content}"
    );
}

#[tokio::test]
async fn tool_result_retries_answer_only_when_schema_reducer_fails() {
    let backend = ToolSchemaFailBackend::new();
    let backends: Vec<Arc<dyn moa::ModelBackend>> = vec![backend.clone()];
    let config = moa::GatewayConfig {
        backends,
        models: vec![moa::ModelEntry {
            name: "strong-32b".into(),
            backend_index: 0,
        }],
        worker_timeout: Duration::from_secs(2),
        hedge_delay: Duration::from_millis(50),
        reducer_timeout: Duration::from_secs(2),
        first_answer_grace: Duration::ZERO,
        enable_thinking: None,
    };

    let body = json!({
        "model": "mesh",
        "tools": read_file_tool(),
        "messages": [
            {"role": "user", "content": "Summarize the tool output."},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
        "max_tokens": 64,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(result.turn_kind, moa::TurnKind::ToolResult);
    assert_eq!(backend.calls_with_tools(), 1);
    assert_eq!(
        backend.calls_without_tools(),
        1,
        "schema-backed reducer failure must retry once without native tools"
    );
    assert_eq!(result.reducer_attempts, 2);
    assert!(
        result.response_body.get("error").is_none(),
        "tool-result answer-only fallback should not surface a MoA error: {}",
        result.response_body
    );
    assert_eq!(
        result
            .response_body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str),
        Some("stop")
    );
    assert!(
        result
            .response_body
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .contains("file_a.log")
    );
}

#[tokio::test]
async fn last_message_role_tool_classifies_as_tool_result() {
    // OpenAI canonical shape. This is already supposed to work today
    // and pins the existing behavior so we don't regress when we
    // tighten the "trailing user" shape below.
    let (config, fast, mid, strong) = config_with_three_recording_workers();

    let body = json!({
        "model": "mesh",
        "tools": read_file_tool(),
        "messages": [
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "# Hello World\n"},
        ],
        "max_tokens": 64,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(
        result.turn_kind,
        moa::TurnKind::ToolResult,
        "last-msg-role=tool must classify as TurnKind::ToolResult; got {:?}",
        result.turn_kind
    );
    assert!(
        result.reducer_used,
        "tool-result turn must invoke the reducer"
    );
    // No fanout: only the reducer (a single backend in the candidate
    // ladder) should have been called.
    let total_calls = fast.calls() + mid.calls() + strong.calls();
    assert_eq!(
        total_calls,
        1,
        "tool-result turn must not fan out — expected 1 reducer call, got {total_calls} \
         (fast={}, mid={}, strong={})",
        fast.calls(),
        mid.calls(),
        strong.calls()
    );
}

#[tokio::test]
async fn trailing_user_after_unsynthesised_tool_result_classifies_as_tool_result() {
    // The bug from the PR review. The harness has appended a `user`
    // message AFTER an unsynthesised tool result. The last message is
    // now `user`, not `tool`. The gateway today classifies this as
    // Continuation and fans out to every worker — wasting a round-trip
    // per worker and risking duplicate tool calls. It must instead
    // take the reducer-only path: synthesize the tool result, address
    // the user nudge, return one coherent response.
    let (config, fast, mid, strong) = config_with_three_recording_workers();

    let body = json!({
        "model": "mesh",
        "tools": read_file_tool(),
        "messages": [
            {"role": "user", "content": "Read README.md and tell me what it says."},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "# Hello World\n"},
            // Harness leaves the tool result in-context and asks the
            // model to continue. There is NO new assistant message
            // synthesizing the tool result yet.
            {"role": "user", "content": "Go on."},
        ],
        "max_tokens": 64,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert_eq!(
        result.turn_kind,
        moa::TurnKind::ToolResult,
        "trailing-user after unsynthesised tool result must classify as \
         TurnKind::ToolResult to avoid spraying through fanout; got {:?}",
        result.turn_kind
    );
    assert!(
        result.reducer_used,
        "unsynthesised-tool-result turn must invoke the reducer"
    );
    let total_calls = fast.calls() + mid.calls() + strong.calls();
    assert_eq!(
        total_calls,
        1,
        "tool-result follow-up must not fan out — expected 1 reducer call, got \
         {total_calls} (fast={}, mid={}, strong={})",
        fast.calls(),
        mid.calls(),
        strong.calls()
    );
}

#[tokio::test]
async fn fresh_user_question_still_classifies_as_fan_out() {
    // Counterpart: a fresh conversation with just a user message must
    // continue to fan out. We must not over-trigger the tool-result
    // path and drop fan-out for normal questions.
    let (config, fast, mid, strong) = config_with_three_recording_workers();

    let body = json!({
        "model": "mesh",
        "messages": [
            {"role": "user", "content": "What is the capital of Japan? One word only."},
        ],
        "max_tokens": 32,
    });

    let result = moa::handle_turn(&config, &body).await;

    assert!(
        matches!(
            result.turn_kind,
            moa::TurnKind::Fanout | moa::TurnKind::EarlyExit
        ),
        "fresh user question must fan out (Fanout or EarlyExit); got {:?}",
        result.turn_kind
    );
    let total_calls = fast.calls() + mid.calls() + strong.calls();
    assert!(
        total_calls >= 1,
        "fresh user question must reach at least one worker; got 0 calls"
    );
}
