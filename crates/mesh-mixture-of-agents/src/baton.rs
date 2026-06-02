//! Experimental serial semantic-layer MoA orchestration.
//!
//! This is intentionally not token-level collaboration. Each selected model
//! gets one small decode budget and hands a compact baton to the next model:
//! planner -> critic/specialist -> finalizer. The goal is to test whether
//! diverse models used once in series can improve quality without paying
//! network latency at every generated token.

use crate::arbiter::{self, Decision};
use crate::backend::{SamplingParams, call_backend};
use crate::normalize::{self, OutputKind, WorkerOutput};
use crate::session::Session;
use crate::tool_guard::enforce_allowed_tools;
use crate::worker::{self, Assignment, WorkerRole, truncate_chars};
use crate::{GatewayConfig, WorkerSummary};
use serde_json::{Value, json};
use std::time::Instant;

const PLANNER_MAX_TOKENS: u32 = 192;
const CRITIC_MAX_TOKENS: u32 = 256;
const FINALIZER_MAX_TOKENS: u32 = 320;
const BATON_SNIPPET_BYTES: usize = 1_200;

pub(crate) struct BatonRun {
    pub outputs: Vec<WorkerOutput>,
    pub summaries: Vec<WorkerSummary>,
    pub decision: Option<Decision>,
    pub early_exit: bool,
}

pub(crate) async fn run_baton_query(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
) -> BatonRun {
    let assignments = worker::assign_roles(&config.models);
    let steps = select_pipeline(&assignments);
    tracing::info!(
        "moa-baton: serial semantic pipeline with {} step(s): [{}]",
        steps.len(),
        steps
            .iter()
            .map(|step| format!("{}:{}", step.kind.label(), step.assignment.model_name))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut successful = Vec::new();
    let mut summaries = Vec::new();
    for step in steps {
        match run_pipeline_step(config, session, &step, &successful, allowed_tools).await {
            PipelineStepResult::Output(output, summary) => {
                summaries.push(summary);
                successful.push(output);
            }
            PipelineStepResult::Failed(summary) => summaries.push(summary),
        }
    }

    if successful.is_empty() {
        return BatonRun {
            outputs: Vec::new(),
            summaries,
            decision: None,
            early_exit: false,
        };
    }

    clean_final_baton_outputs(&mut successful);
    let final_outputs = final_outputs(successful);
    let decision = Some(arbiter::arbitrate(&final_outputs, has_tools));
    BatonRun {
        outputs: final_outputs,
        summaries,
        decision,
        early_exit: false,
    }
}

pub(crate) fn clean_response_body(response_body: &mut Value) {
    let Some(content) = response_body
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
    else {
        return;
    };
    let Some(answer) = extract_candidate_answer(&content) else {
        return;
    };
    if let Some(slot) = response_body.pointer_mut("/choices/0/message/content") {
        *slot = Value::String(answer);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatonStepKind {
    Planner,
    Critic,
    Finalizer,
}

impl BatonStepKind {
    fn label(self) -> &'static str {
        match self {
            Self::Planner => "planner",
            Self::Critic => "critic",
            Self::Finalizer => "finalizer",
        }
    }

    fn max_tokens(self) -> u32 {
        match self {
            Self::Planner => PLANNER_MAX_TOKENS,
            Self::Critic => CRITIC_MAX_TOKENS,
            Self::Finalizer => FINALIZER_MAX_TOKENS,
        }
    }
}

#[derive(Clone)]
struct PipelineStep {
    kind: BatonStepKind,
    assignment: Assignment,
}

enum PipelineStepResult {
    Output(WorkerOutput, WorkerSummary),
    Failed(WorkerSummary),
}

async fn run_pipeline_step(
    config: &GatewayConfig,
    session: &Session,
    step: &PipelineStep,
    previous_outputs: &[WorkerOutput],
    allowed_tools: &[String],
) -> PipelineStepResult {
    let messages = step_messages(session, step, previous_outputs);
    let tools = tools_for_step(session, step);
    let model_name = step.assignment.model_name.clone();
    let role = step.assignment.role;
    let backend = config.backends[step.assignment.backend_index].clone();
    let t0 = Instant::now();
    let result = call_backend(
        &*backend,
        &model_name,
        &messages,
        tools.as_ref(),
        step.kind.max_tokens(),
        config.worker_timeout,
        SamplingParams::worker().with_thinking(config.enable_thinking),
    )
    .await;
    let elapsed_ms = t0.elapsed().as_millis() as u64;

    match result {
        Ok(text) => {
            let mut output =
                normalize::normalize_worker_output(&text, &model_name, role, elapsed_ms);
            enforce_allowed_tools(&mut output, allowed_tools, &model_name);
            tracing::info!(
                "moa-baton: {} {}({}) -> {:?} conf={:.2}",
                step.kind.label(),
                model_name,
                role.label(),
                output.kind,
                output.confidence
            );
            let summary = WorkerSummary {
                model: format!("{}#{}", model_name, step.kind.label()),
                role,
                succeeded: true,
                elapsed_ms,
                output_kind: Some(output.kind),
                confidence: Some(output.confidence),
            };
            PipelineStepResult::Output(output, summary)
        }
        Err(err) => {
            tracing::warn!(
                "moa-baton: {} {}({}) failed after {}ms: {}",
                step.kind.label(),
                model_name,
                role.label(),
                elapsed_ms,
                err
            );
            PipelineStepResult::Failed(WorkerSummary {
                model: format!("{}#{}", model_name, step.kind.label()),
                role,
                succeeded: false,
                elapsed_ms,
                output_kind: None,
                confidence: None,
            })
        }
    }
}

fn select_pipeline(assignments: &[Assignment]) -> Vec<PipelineStep> {
    let Some(first) = assignments.first() else {
        return Vec::new();
    };
    let last = assignments.last().unwrap_or(first);
    let critic = assignments
        .iter()
        .find(|assignment| matches!(assignment.role, WorkerRole::Specialist))
        .unwrap_or(last);

    let mut steps = vec![PipelineStep {
        kind: BatonStepKind::Planner,
        assignment: first.clone(),
    }];
    if critic.model_name != first.model_name || assignments.len() == 1 {
        steps.push(PipelineStep {
            kind: BatonStepKind::Critic,
            assignment: critic.clone(),
        });
    }
    if last.model_name != critic.model_name || steps.len() == 1 {
        steps.push(PipelineStep {
            kind: BatonStepKind::Finalizer,
            assignment: last.clone(),
        });
    }
    steps
}

fn final_outputs(mut outputs: Vec<WorkerOutput>) -> Vec<WorkerOutput> {
    let Some(final_answer_idx) = outputs
        .iter()
        .rposition(|output| matches!(output.kind, OutputKind::Answer | OutputKind::ToolProposal))
    else {
        return outputs.into_iter().rev().take(1).collect();
    };
    vec![outputs.swap_remove(final_answer_idx)]
}

fn step_messages(
    session: &Session,
    step: &PipelineStep,
    previous_outputs: &[WorkerOutput],
) -> Vec<Value> {
    let mut messages = vec![json!({
        "role": "system",
        "content": step_system_prompt(session, step, previous_outputs),
    })];
    messages.extend(messages_for_role(session, step.assignment.role));
    messages
}

fn step_system_prompt(
    session: &Session,
    step: &PipelineStep,
    previous_outputs: &[WorkerOutput],
) -> String {
    let mut parts = vec![
        "You are one stage in an experimental serial semantic-layer pipeline.".to_string(),
        format!(
            "Stage: {}. Decode a small bounded artifact, then hand off compact state.",
            step.kind.label()
        ),
    ];
    parts.push(match step.kind {
        BatonStepKind::Planner => {
            "Frame the task and produce a compact first candidate. Do not over-elaborate."
                .to_string()
        }
        BatonStepKind::Critic => {
            "Repair or reject the previous baton. Add missing constraints, risks, and corrections."
                .to_string()
        }
        BatonStepKind::Finalizer => {
            "Use the prior batons to produce the final user-facing answer. Prefer a direct answer over another baton unless a tool call is required."
                .to_string()
        }
    });
    parts.push(
        "Return JSON with fields: kind, confidence, payload, optional tool, optional arguments."
            .to_string(),
    );
    parts.push(
        "For kind=\"answer\", payload may be either a direct string or an object containing candidate_answer/response/final_answer."
            .to_string(),
    );

    if let Some(agent_system) = session.system_prompt() {
        parts.push(format!("\nOriginal system prompt:\n{agent_system}"));
    }
    append_tool_hint(session, &mut parts);
    append_previous_batons(previous_outputs, &mut parts);
    parts.join("\n")
}

fn append_tool_hint(session: &Session, parts: &mut Vec<String>) {
    let tool_names = session.tool_names();
    if !tool_names.is_empty() {
        parts.push(format!("\nAvailable tools: {}", tool_names.join(", ")));
    }
}

fn append_previous_batons(previous_outputs: &[WorkerOutput], parts: &mut Vec<String>) {
    if previous_outputs.is_empty() {
        return;
    }
    parts.push("\nPrior stage batons:".to_string());
    for (idx, output) in previous_outputs.iter().enumerate() {
        parts.push(format!(
            "\n[Stage {} from {} ({}) conf={:.2}]\n{}",
            idx + 1,
            output.model,
            output.role.label(),
            output.confidence,
            truncate_chars(&output.payload, BATON_SNIPPET_BYTES)
        ));
    }
}

fn tools_for_step(session: &Session, step: &PipelineStep) -> Option<Value> {
    match step.kind {
        BatonStepKind::Planner => None,
        BatonStepKind::Critic | BatonStepKind::Finalizer => session.tools().cloned(),
    }
}

fn messages_for_role(session: &Session, role: WorkerRole) -> Vec<Value> {
    match role {
        WorkerRole::Fast => vec![json!({"role": "user", "content": session.last_user_text()})],
        WorkerRole::Specialist => filtered_recent_messages(session, 4),
        WorkerRole::Strong | WorkerRole::Generalist | WorkerRole::Reducer => {
            filtered_recent_messages(session, 10)
        }
    }
}

fn filtered_recent_messages(session: &Session, limit: usize) -> Vec<Value> {
    let mut messages = Vec::new();
    for msg in session.recent_messages(limit) {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && !role.is_empty() {
            messages.push(msg);
        }
    }
    let user_text = session.last_user_text();
    if messages
        .last()
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        != Some(user_text.as_str())
    {
        messages.push(json!({"role": "user", "content": user_text}));
    }
    messages
}

fn clean_final_baton_outputs(outputs: &mut [WorkerOutput]) {
    for output in outputs {
        if !matches!(output.kind, OutputKind::Answer) {
            continue;
        }
        if let Some(answer) = extract_candidate_answer(&output.payload) {
            output.payload = answer;
        }
    }
}

fn extract_candidate_answer(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload)
        .or_else(|_| extract_json_object(payload))
        .ok()?;
    candidate_answer_from_value(&value)
        .or_else(|| value.get("payload").and_then(candidate_answer_from_value))
        .map(str::trim)
        .filter(|answer| !answer.is_empty())
        .map(ToOwned::to_owned)
}

fn extract_json_object(text: &str) -> Result<Value, serde_json::Error> {
    let start = text.find('{').unwrap_or(0);
    let end = text.rfind('}').map(|idx| idx + 1).unwrap_or(text.len());
    serde_json::from_str(&text[start..end])
}

fn candidate_answer_from_value(value: &Value) -> Option<&str> {
    value
        .get("candidate_answer")
        .or_else(|| value.get("response"))
        .or_else(|| value.get("answer"))
        .or_else(|| value.get("final_answer"))
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .get("baton")
                .and_then(|baton| baton.get("candidate_answer"))
                .and_then(Value::as_str)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ModelEntry;
    use crate::normalize::OutputKind;

    fn assignment(name: &str, role: WorkerRole) -> Assignment {
        Assignment {
            model_name: name.to_string(),
            backend_index: 0,
            role,
        }
    }

    fn output(model: &str, confidence: f32, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::Answer,
            confidence,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: model.to_string(),
            role: WorkerRole::Specialist,
            elapsed_ms: 1,
        }
    }

    #[test]
    fn select_pipeline_uses_small_middle_large_when_available() {
        let assignments = worker::assign_roles(&[
            ModelEntry {
                name: "small-3B".to_string(),
                backend_index: 0,
            },
            ModelEntry {
                name: "middle-8B".to_string(),
                backend_index: 0,
            },
            ModelEntry {
                name: "large-70B".to_string(),
                backend_index: 0,
            },
        ]);
        let steps = select_pipeline(&assignments);
        assert_eq!(
            steps
                .iter()
                .map(|step| (step.kind.label(), step.assignment.model_name.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("planner", "small-3B"),
                ("critic", "middle-8B"),
                ("finalizer", "large-70B"),
            ]
        );
    }

    #[test]
    fn select_pipeline_keeps_two_model_chain_compact() {
        let steps = select_pipeline(&[
            assignment("small", WorkerRole::Fast),
            assignment("large", WorkerRole::Strong),
        ]);
        assert_eq!(
            steps
                .iter()
                .map(|step| (step.kind.label(), step.assignment.model_name.as_str()))
                .collect::<Vec<_>>(),
            vec![("planner", "small"), ("critic", "large")]
        );
    }

    #[test]
    fn final_outputs_prefers_last_answer_like_stage() {
        let outputs = vec![
            output("planner", 0.4, "planner"),
            output("final", 0.8, "final"),
        ];
        let picked = final_outputs(outputs);
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].model, "final");
    }

    #[test]
    fn clean_final_baton_outputs_extracts_candidate_answer() {
        let mut outputs = vec![output(
            "model",
            0.9,
            r#"{"kind":"answer","confidence":0.9,"payload":{"candidate_answer":"final sentence","risks":"none"}}"#,
        )];
        clean_final_baton_outputs(&mut outputs);
        assert_eq!(outputs[0].payload, "final sentence");
    }

    #[test]
    fn clean_final_baton_outputs_extracts_nested_response() {
        let mut outputs = vec![output(
            "model",
            0.9,
            r#"{"kind":"answer","confidence":0.9,"payload":{"baton":{"candidate_answer":"baton sentence"},"response":"response sentence"}}"#,
        )];
        clean_final_baton_outputs(&mut outputs);
        assert_eq!(outputs[0].payload, "response sentence");
    }

    #[test]
    fn clean_final_baton_outputs_extracts_fenced_json_candidate() {
        let mut outputs = vec![output(
            "model",
            0.9,
            "```json\n{\"kind\":\"answer\",\"payload\":{\"candidate_answer\":\"fenced sentence\"}}\n```",
        )];
        clean_final_baton_outputs(&mut outputs);
        assert_eq!(outputs[0].payload, "fenced sentence");
    }

    #[test]
    fn clean_response_body_extracts_candidate_answer_from_chat_content() {
        let mut response = json!({
            "choices": [{
                "message": {
                    "content": r#"{"kind":"answer","payload":{"candidate_answer":"chat sentence"}}"#
                }
            }]
        });
        clean_response_body(&mut response);
        assert_eq!(
            response.pointer("/choices/0/message/content"),
            Some(&Value::String("chat sentence".to_string()))
        );
    }
}
