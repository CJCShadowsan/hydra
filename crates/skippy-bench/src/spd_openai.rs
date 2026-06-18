use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};
use skippy_runtime::spd::{
    SpdHeadManifest, SpdRollingTraceReplay, required_spd_hf_indices_for_topology,
};

use crate::{cli::SpdOpenAiSmokeArgs, support::ChildGuard};

mod attrs;
mod preflight;
mod remote;

use attrs::{
    attr_bool, attr_f64, attr_f64_array, attr_i64, attr_i64_array, attr_i64_array_map, attr_string,
    attr_u64, attr_u64_array, attrs_for, count_events, count_events_by_hf_index, read_events,
};
use remote::{
    collect_remote_case_logs, prepare_case_deployment, start_case_stages, stop_remote_case_stages,
};

const OPENAI_PATH_MODELS: &str = "/v1/models";
const OPENAI_PATH_CHAT_COMPLETIONS: &str = "/v1/chat/completions";

pub fn spd_openai_smoke(args: SpdOpenAiSmokeArgs) -> Result<()> {
    validate_args(&args)?;
    let prompts = load_prompts(&args)?;
    let stage_ranges = stage_ranges(&args.splits, args.layer_end)?;
    let manifest = SpdHeadManifest::from_path(&args.manifest)?;
    validate_manifest_activation_width(args.activation_width, &manifest)?;
    let logical_spd_stage_count = usize::try_from(manifest.topology.num_stages)
        .context("SPD manifest num_stages exceeds usize")?;
    let tap_allowlist = spd_tap_allowlist(&args, &manifest)?;
    preflight::validate_tap_coverage(&stage_ranges, &tap_allowlist)?;
    if args.preflight_only {
        return preflight::write_spd_openai_preflight(
            &args,
            &manifest,
            &stage_ranges,
            &tap_allowlist,
            prompts.len(),
        );
    }
    let work_dir = args.work_dir.clone().unwrap_or_else(default_work_dir);
    fs::create_dir_all(&work_dir)
        .with_context(|| format!("failed to create {}", work_dir.display()))?;

    let mut cases = Vec::new();
    for prompt in &prompts {
        for iteration in case_iterations(args.warmup_count, args.repeat_count) {
            if args.run_baseline {
                cases.push(run_case(
                    &args,
                    &work_dir,
                    &stage_ranges,
                    &tap_allowlist,
                    SmokeCase::Baseline,
                    prompt,
                    iteration,
                )?);
            }
            if args.run_spd {
                cases.push(run_case(
                    &args,
                    &work_dir,
                    &stage_ranges,
                    &tap_allowlist,
                    SmokeCase::Spd,
                    prompt,
                    iteration,
                )?);
            }
        }
    }

    let summary = summarize_cases(&cases, stage_ranges.len(), logical_spd_stage_count);
    let report = SpdOpenAiSmokeReport {
        mode: "spd-openai-smoke",
        model_id: args.model_id.clone(),
        prompt_count: prompts.len(),
        prompts: prompts.iter().map(PromptReport::from).collect(),
        stage_count: stage_ranges.len(),
        logical_spd_stage_count,
        splits: args.splits.clone(),
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        max_tokens: args.max_tokens,
        repeat_count: args.repeat_count,
        warmup_count: args.warmup_count,
        temperature: args.temperature,
        enable_thinking: args.enable_thinking,
        activation_wire_dtype: args.activation_wire_dtype.clone(),
        activation_width: args.activation_width,
        downstream_wire_delay_ms: args.downstream_wire_delay_ms,
        downstream_wire_mbps: args.downstream_wire_mbps,
        spd_rolling_executor: args.spd_rolling_executor,
        optimistic_min_logit_margin: args.optimistic_min_logit_margin,
        spd_tap_return_hf_indices: tap_allowlist,
        work_dir: work_dir.display().to_string(),
        summary,
        cases,
    };

    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output.as_ref() {
        fs::write(output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    ensure_content_matches(
        &report.summary.prompt_comparisons,
        args.allow_content_mismatch,
    )?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct SpdOpenAiSmokeReport {
    mode: &'static str,
    model_id: String,
    prompt_count: usize,
    prompts: Vec<PromptReport>,
    stage_count: usize,
    logical_spd_stage_count: usize,
    splits: Vec<u32>,
    layer_end: u32,
    ctx_size: u32,
    max_tokens: u32,
    repeat_count: usize,
    warmup_count: usize,
    temperature: f32,
    enable_thinking: bool,
    activation_wire_dtype: String,
    activation_width: i32,
    downstream_wire_delay_ms: f64,
    downstream_wire_mbps: Option<f64>,
    spd_rolling_executor: bool,
    optimistic_min_logit_margin: Option<f32>,
    spd_tap_return_hf_indices: Vec<u32>,
    work_dir: String,
    summary: SpdOpenAiSmokeSummary,
    cases: Vec<CaseReport>,
}

#[derive(Clone, Copy, Debug)]
struct CaseIteration {
    warmup: bool,
    repeat_index: usize,
}

fn case_iterations(warmup_count: usize, repeat_count: usize) -> Vec<CaseIteration> {
    (0..warmup_count)
        .map(|repeat_index| CaseIteration {
            warmup: true,
            repeat_index,
        })
        .chain((0..repeat_count).map(|repeat_index| CaseIteration {
            warmup: false,
            repeat_index,
        }))
        .collect()
}

#[derive(Debug)]
struct PromptInput {
    index: usize,
    label: String,
    text: String,
    messages: Vec<PromptMessage>,
}

#[derive(Clone, Debug, Serialize)]
struct PromptMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct PromptReport {
    index: usize,
    label: String,
    prompt: String,
    messages: Vec<PromptMessage>,
}

impl From<&PromptInput> for PromptReport {
    fn from(input: &PromptInput) -> Self {
        Self {
            index: input.index,
            label: input.label.clone(),
            prompt: input.text.clone(),
            messages: input.messages.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum SmokeCase {
    Baseline,
    Spd,
}

impl SmokeCase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Spd => "spd",
        }
    }

    fn uses_spd(self) -> bool {
        matches!(self, Self::Spd)
    }
}

fn load_prompts(args: &SpdOpenAiSmokeArgs) -> Result<Vec<PromptInput>> {
    let mut prompts = match args.prompt_file.as_ref() {
        Some(path) => prompts_from_file(path)?,
        None => vec![PromptInput {
            index: 0,
            label: "prompt-000".to_string(),
            text: args.prompt.clone(),
            messages: vec![PromptMessage::user(args.prompt.clone())],
        }],
    };
    if let Some(limit) = args.prompt_limit {
        prompts.truncate(limit);
    }
    if prompts.is_empty() {
        bail!("SPD OpenAI smoke has no prompts to run");
    }
    for (index, prompt) in prompts.iter_mut().enumerate() {
        prompt.index = index;
        if prompt.label.trim().is_empty() {
            prompt.label = format!("prompt-{index:03}");
        }
    }
    Ok(prompts)
}

fn prompts_from_file(path: &Path) -> Result<Vec<PromptInput>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("read prompt file {}", path.display()))?;
    content
        .lines()
        .enumerate()
        .filter_map(|(line_index, line)| {
            let line = line.trim();
            (!line.is_empty()).then_some((line_index, line))
        })
        .map(|(line_index, line)| prompt_from_line(line_index, line))
        .collect()
}

fn prompt_from_line(line_index: usize, line: &str) -> Result<PromptInput> {
    let default_label = format!("prompt-{line_index:03}");
    match serde_json::from_str::<Value>(line) {
        Ok(Value::String(text)) => prompt_input(line_index, default_label, text),
        Ok(Value::Object(object)) => {
            let label = object
                .get("label")
                .or_else(|| object.get("id"))
                .or_else(|| object.get("prompt_id"))
                .and_then(Value::as_str)
                .unwrap_or(&default_label)
                .to_string();
            if let Some(messages) = object.get("messages") {
                let messages = parse_prompt_messages(messages, line_index)?;
                return prompt_messages(line_index, label, messages);
            }
            if let Some(turns) = object.get("turns") {
                let messages = parse_prompt_turns(turns, line_index)?;
                return prompt_messages(line_index, label, messages);
            }
            let text = object
                .get("prompt")
                .or_else(|| object.get("text"))
                .or_else(|| object.get("content"))
                .and_then(Value::as_str)
                .with_context(|| {
                    format!(
                        "prompt JSON object on line {} needs prompt/text/content",
                        line_index + 1
                    )
                })?
                .to_string();
            prompt_input(line_index, label, text)
        }
        Ok(_) | Err(_) => prompt_input(line_index, default_label, line.to_string()),
    }
}

fn prompt_input(index: usize, label: String, text: String) -> Result<PromptInput> {
    prompt_messages(index, label, vec![PromptMessage::user(text)])
}

fn prompt_messages(
    index: usize,
    label: String,
    messages: Vec<PromptMessage>,
) -> Result<PromptInput> {
    if messages.is_empty() {
        bail!("prompt {label} has no messages");
    }
    if messages
        .iter()
        .any(|message| message.role.trim().is_empty())
    {
        bail!("prompt {label} contains an empty message role");
    }
    if messages
        .iter()
        .any(|message| message.content.trim().is_empty())
    {
        bail!("prompt {label} contains an empty message");
    }
    let text = prompt_summary(&messages);
    if text.trim().is_empty() {
        bail!("prompt {label} is empty");
    }
    Ok(PromptInput {
        index,
        label: safe_prompt_label(&label, index),
        text,
        messages,
    })
}

impl PromptMessage {
    fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }
}

fn parse_prompt_messages(value: &Value, line_index: usize) -> Result<Vec<PromptMessage>> {
    let messages = value
        .as_array()
        .with_context(|| format!("messages on line {} must be an array", line_index + 1))?;
    messages
        .iter()
        .enumerate()
        .map(|(message_index, message)| parse_prompt_message(message, line_index, message_index))
        .collect()
}

fn parse_prompt_message(
    value: &Value,
    line_index: usize,
    message_index: usize,
) -> Result<PromptMessage> {
    let object = value.as_object().with_context(|| {
        format!(
            "messages[{}] on line {} must be an object",
            message_index,
            line_index + 1
        )
    })?;
    let role = object
        .get("role")
        .and_then(Value::as_str)
        .with_context(|| {
            format!(
                "messages[{}].role on line {} must be a string",
                message_index,
                line_index + 1
            )
        })?;
    let content = object
        .get("content")
        .and_then(Value::as_str)
        .with_context(|| {
            format!(
                "messages[{}].content on line {} must be a string",
                message_index,
                line_index + 1
            )
        })?;
    Ok(PromptMessage {
        role: role.to_string(),
        content: content.to_string(),
    })
}

fn parse_prompt_turns(value: &Value, line_index: usize) -> Result<Vec<PromptMessage>> {
    let turns = value
        .as_array()
        .with_context(|| format!("turns on line {} must be an array", line_index + 1))?;
    let joined = turns
        .iter()
        .enumerate()
        .map(|(turn_index, turn)| {
            turn.as_str().map(ToString::to_string).with_context(|| {
                format!(
                    "turns[{}] on line {} must be a string",
                    turn_index,
                    line_index + 1
                )
            })
        })
        .collect::<Result<Vec<_>>>()?
        .join("\n\n");
    Ok(vec![PromptMessage::user(joined)])
}

fn prompt_summary(messages: &[PromptMessage]) -> String {
    if let [message] = messages
        && message.role == "user"
    {
        return message.content.clone();
    }
    messages
        .iter()
        .map(|message| format!("{}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn safe_prompt_label(label: &str, index: usize) -> String {
    let mut safe = String::with_capacity(label.len());
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            safe.push(ch);
        } else if !safe.ends_with('-') {
            safe.push('-');
        }
    }
    let safe = safe.trim_matches('-');
    if safe.is_empty() {
        format!("prompt-{index:03}")
    } else {
        safe.to_string()
    }
}

#[derive(Debug, Serialize)]
struct CaseReport {
    name: &'static str,
    prompt_index: usize,
    prompt_label: String,
    warmup: bool,
    repeat_index: usize,
    prompt: String,
    run_id: String,
    openai_base_url: String,
    logs_dir: String,
    elapsed_ms: f64,
    content: String,
    usage: Option<Value>,
    finish_reason: Option<String>,
    decode: Option<DecodeReport>,
    inline_probes: Vec<InlineProbeReport>,
    optimistic_decodes: Vec<OptimisticDecodeReport>,
    token_events: Vec<TokenEventReport>,
    tap_returns_by_hf_index: BTreeMap<String, u64>,
    tap_records_by_hf_index: BTreeMap<String, u64>,
    tap_return_failures: usize,
    tap_record_failures: usize,
    tap_ignored: usize,
}

#[derive(Debug, Serialize)]
struct DecodeReport {
    elapsed_ms: Option<f64>,
    tokens: Option<u64>,
    spec_enabled: Option<bool>,
    spec_windows: Option<u64>,
    spec_proposed: Option<u64>,
    spec_accepted: Option<u64>,
    spec_rejected: Option<u64>,
    spec_draft_propose_ms: Option<f64>,
    optimistic_requests: Option<u64>,
    optimistic_accepted: Option<u64>,
    optimistic_rejected: Option<u64>,
    optimistic_committed: Option<u64>,
    optimistic_checkpoint_ms: Option<f64>,
    optimistic_decode_elapsed_ms: Option<f64>,
    optimistic_decode_wait_ms: Option<f64>,
    optimistic_restore_ms: Option<f64>,
    chained_optimistic_requests: Option<u64>,
    chained_optimistic_accepted: Option<u64>,
    chained_optimistic_rejected: Option<u64>,
    chained_optimistic_committed: Option<u64>,
    spd_rolling_executor_launches: Option<u64>,
    spd_rolling_executor_launch_misses: Option<u64>,
    spd_rolling_executor_launch_miss_in_flight_full: Option<u64>,
    spd_rolling_executor_launch_miss_no_rows: Option<u64>,
    spd_rolling_executor_launch_miss_no_proposal: Option<u64>,
    spd_rolling_executor_launch_miss_shadow_not_seedable: Option<u64>,
    spd_rolling_executor_launch_miss_shadow_missing_view: Option<u64>,
    spd_rolling_executor_shadow_source_reseeds: Option<u64>,
    spd_rolling_executor_margin_rejects: Option<u64>,
    spd_rolling_executor_max_in_flight: Option<u64>,
    spd_rolling_executor_accepted_oldest: Option<u64>,
    spd_rolling_executor_rejected_oldest: Option<u64>,
    spd_rolling_executor_drained_younger: Option<u64>,
    stage0_compute_ms: Option<f64>,
    downstream_wait_ms: Option<f64>,
    spd_proposal_total_requested_limit: Option<u64>,
    spd_proposal_total_attempts: Option<u64>,
    spd_proposal_total_proposed: Option<u64>,
    spd_proposal_total_inline_tap_hits: Option<u64>,
    spd_proposal_total_replay_fallbacks: Option<u64>,
    spd_proposal_total_cache_hits: Option<u64>,
    spd_proposal_total_cache_misses: Option<u64>,
    spd_proposal_total_tap_collect_ms: Option<f64>,
    spd_proposal_total_cur_in_ms: Option<f64>,
    spd_proposal_total_forward_ms: Option<f64>,
    spd_proposal_total_cache_prefill_ms: Option<f64>,
    spd_proposal_total_head_fixed_stage_projection_ms: Option<f64>,
    spd_proposal_total_head_decoder_ms: Option<f64>,
    spd_proposal_total_head_final_norm_ms: Option<f64>,
    spd_proposal_total_head_lm_head_topk_ms: Option<f64>,
    spd_proposal_total_head_total_ms: Option<f64>,
    spd_proposal_total_last_cache_prefix_len: Option<u64>,
    spd_proposal_total_max_cache_prefix_len: Option<u64>,
    rolling: Option<SpdLiveRollingReport>,
}

#[derive(Debug, Serialize)]
struct InlineProbeReport {
    step: Option<u64>,
    phase: Option<String>,
    elapsed_ms: Option<f64>,
    target_wait_after_probe_ms: Option<f64>,
    current_token: Option<i64>,
    proposed_token: Option<i64>,
    proposed_logit: Option<f64>,
    proposed_logit_margin: Option<f64>,
    cache_used: Option<bool>,
    cache_prefix_len: Option<u64>,
    tap_source: Option<String>,
    tap_collect_ms: Option<f64>,
    cur_in_ms: Option<f64>,
    forward_ms: Option<f64>,
    cache_prefill_ms: Option<f64>,
    head_fixed_stage_projection_ms: Option<f64>,
    head_decoder_ms: Option<f64>,
    head_decoder_layer_ms: Vec<f64>,
    head_final_norm_ms: Option<f64>,
    head_lm_head_topk_ms: Option<f64>,
    head_total_ms: Option<f64>,
    target_token: Option<i64>,
    accepted: Option<bool>,
    trigger_hf_index: Option<u64>,
    proposal_row_positions: Vec<i64>,
    proposal_row_i_stages: Vec<i64>,
    proposal_row_evicted_prefix_position: Option<u64>,
    proposal_row_newest_position: Option<u64>,
    proposal_row_next_draft_position: Option<u64>,
    proposal_miss_reason: Option<String>,
    proposal_missing_taps: BTreeMap<String, Vec<i64>>,
    rolling: Option<SpdLiveRollingReport>,
    rolling_verified_delta: Option<SpdRollingVerifiedDeltaReport>,
}

#[derive(Debug, Serialize)]
struct SpdLiveRollingReport {
    logical_stage_count: Option<u64>,
    target_position: Option<u64>,
    next_position: Option<u64>,
    inserted_drafts: Option<u64>,
    missing_proposals: Option<u64>,
    first_missing_proposal_position: Option<u64>,
    out_of_order_proposals: Option<u64>,
    first_out_of_order_proposal_position: Option<u64>,
    verified_windows: Option<u64>,
    accepted_windows: Option<u64>,
    rejected_windows: Option<u64>,
    first_rejected_target_position: Option<u64>,
    pipeline_len: Option<u64>,
    verified_up_to: Option<u64>,
    row_evicted_prefix_position: Option<u64>,
    row_positions: Vec<u64>,
    row_i_stages: Vec<u64>,
    row_newest_position: Option<u64>,
    row_next_draft_position: Option<u64>,
}

#[derive(Debug, Serialize)]
struct SpdRollingVerifiedDeltaReport {
    start_position: Option<u64>,
    verified_up_to: Option<u64>,
    tokens: Vec<i64>,
    token_count: Option<u64>,
}

#[derive(Debug, Serialize)]
struct OptimisticDecodeReport {
    step: Option<u64>,
    chain: Option<bool>,
    chain_depth: Option<u64>,
    proposed_token: Option<i64>,
    proposed_logit: Option<f64>,
    proposed_logit_margin: Option<f64>,
    requested_tap_return: Option<bool>,
    target_token: Option<i64>,
    accepted: Option<bool>,
    next_token: Option<i64>,
    checkpoint_ms: Option<f64>,
    elapsed_ms: Option<f64>,
    start_elapsed_ms: Option<f64>,
    wait_ms: Option<f64>,
    hidden_wait_ms: Option<f64>,
    stage0_compute_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct TokenEventReport {
    step: Option<u64>,
    message_kind: Option<String>,
    predicted_token: Option<i64>,
    chain: Option<bool>,
    chain_depth: Option<u64>,
    downstream_wait_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct SpdOpenAiSmokeSummary {
    prompt_pairs: usize,
    matching_content: usize,
    baseline_wall_ms: MetricSummary,
    spd_wall_ms: MetricSummary,
    baseline_decode_ms: MetricSummary,
    spd_decode_ms: MetricSummary,
    wall_speedup_spd_vs_baseline: Option<f64>,
    decode_speedup_spd_vs_baseline: Option<f64>,
    spd_spec_windows: u64,
    spd_spec_proposed: u64,
    spd_spec_accepted: u64,
    spd_spec_rejected: u64,
    spd_accept_rate: Option<f64>,
    optimistic_requests: u64,
    optimistic_accepted: u64,
    optimistic_rejected: u64,
    optimistic_committed: u64,
    chained_optimistic_requests: u64,
    chained_optimistic_accepted: u64,
    chained_optimistic_rejected: u64,
    chained_optimistic_committed: u64,
    max_optimistic_chain_depth: u64,
    spd_rolling_executor_launches: u64,
    spd_rolling_executor_launch_misses: u64,
    spd_rolling_executor_launch_miss_in_flight_full: u64,
    spd_rolling_executor_launch_miss_no_rows: u64,
    spd_rolling_executor_launch_miss_no_proposal: u64,
    spd_rolling_executor_launch_miss_shadow_not_seedable: u64,
    spd_rolling_executor_launch_miss_shadow_missing_view: u64,
    spd_rolling_executor_shadow_source_reseeds: u64,
    spd_rolling_executor_margin_rejects: u64,
    spd_rolling_executor_max_in_flight: u64,
    spd_rolling_executor_accepted_oldest: u64,
    spd_rolling_executor_rejected_oldest: u64,
    spd_rolling_executor_drained_younger: u64,
    tap_return_failures: usize,
    tap_record_failures: usize,
    tap_ignored: usize,
    pipeline_gap: SpdPipelineGapSummary,
    paper_pipeline_estimate: SpdPaperPipelineEstimate,
    rolling_trace_replay: SpdRollingTraceReplaySummary,
    prompt_comparisons: Vec<PromptComparisonReport>,
}

#[derive(Debug, Serialize)]
struct MetricSummary {
    count: usize,
    mean_ms: Option<f64>,
    min_ms: Option<f64>,
    max_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct SpdPipelineGapSummary {
    pre_target_probes: usize,
    pre_target_proposals: usize,
    pre_target_accepted: usize,
    pre_target_accept_rate: Option<f64>,
    optimistic_commit_probes: usize,
    optimistic_commit_proposals: usize,
    optimistic_commit_accepted: usize,
    optimistic_commit_accept_rate: Option<f64>,
    post_target_probes: usize,
    post_target_empty: usize,
    post_target_empty_rate: Option<f64>,
    pre_target_probe_ms: MetricSummary,
    pre_target_wait_after_probe_ms: MetricSummary,
    optimistic_commit_probe_ms: MetricSummary,
    optimistic_commit_wait_after_probe_ms: MetricSummary,
    optimistic_decode_elapsed_ms: MetricSummary,
    optimistic_decode_start_elapsed_ms: MetricSummary,
    optimistic_decode_wait_ms: MetricSummary,
    optimistic_decode_hidden_wait_ms: MetricSummary,
    chained_optimistic_decode_hidden_wait_ms: MetricSummary,
    probe_cache_prefill_ms: MetricSummary,
    probe_head_fixed_stage_projection_ms: MetricSummary,
    probe_head_decoder_ms: MetricSummary,
    probe_head_final_norm_ms: MetricSummary,
    probe_head_lm_head_topk_ms: MetricSummary,
    probe_head_total_ms: MetricSummary,
    normal_token_downstream_wait_ms: MetricSummary,
    optimistic_token_downstream_wait_ms: MetricSummary,
    pre_target_proposals_without_tap_return: usize,
    optimistic_tap_return_requests: usize,
    optimistic_tap_return_accepted: usize,
    optimistic_tap_return_rejected: usize,
    optimistic_tap_return_accept_rate: Option<f64>,
}

#[derive(Debug, Serialize)]
struct SpdPaperPipelineEstimate {
    logical_stage_count: usize,
    physical_stage_count: usize,
    candidate_token_round_trips: u64,
    saved_token_round_trips: u64,
    unsaved_token_round_trips: u64,
    token_round_trip_save_rate: Option<f64>,
    accepted_proposal_rate: Option<f64>,
    paper_like_speedup_vs_serial_split: Option<f64>,
    estimated_decode_ms_at_baseline_stage_cost: Option<f64>,
    current_spd_decode_slowdown_vs_estimate: Option<f64>,
}

#[derive(Debug, Serialize)]
struct SpdRollingTraceReplaySummary {
    logical_stage_count: usize,
    cases_replayed: usize,
    live_cases_observed: usize,
    inserted_drafts: usize,
    missing_proposals: usize,
    first_missing_proposal_position: Option<usize>,
    out_of_order_proposals: usize,
    first_out_of_order_proposal_position: Option<usize>,
    verified_windows: usize,
    accepted_windows: usize,
    rejected_windows: usize,
    first_rejected_target_position: Option<usize>,
    final_pipeline_len: Option<usize>,
    final_verified_up_to: Option<usize>,
    final_verified_prefix_len: Option<usize>,
    final_verified_prefix_tokens: Vec<i32>,
    verified_prefix_matches_target: Option<bool>,
    first_verified_prefix_mismatch_position: Option<usize>,
}

#[derive(Debug, Serialize)]
struct PromptComparisonReport {
    prompt_index: usize,
    prompt_label: String,
    repeat_index: usize,
    content_matches: bool,
    baseline_elapsed_ms: f64,
    spd_elapsed_ms: f64,
    wall_speedup_spd_vs_baseline: Option<f64>,
    baseline_decode_ms: Option<f64>,
    spd_decode_ms: Option<f64>,
    decode_speedup_spd_vs_baseline: Option<f64>,
    spd_spec_proposed: Option<u64>,
    spd_spec_accepted: Option<u64>,
    spd_spec_rejected: Option<u64>,
    spd_accept_rate: Option<f64>,
    optimistic_committed: Option<u64>,
}

fn summarize_cases(
    cases: &[CaseReport],
    physical_stage_count: usize,
    logical_spd_stage_count: usize,
) -> SpdOpenAiSmokeSummary {
    let baseline_cases = cases_for_name(cases, SmokeCase::Baseline);
    let spd_cases = cases_for_name(cases, SmokeCase::Spd);
    let prompt_comparisons = compare_prompt_pairs(&baseline_cases, &spd_cases);
    let baseline_wall_ms = metric_summary(baseline_cases.iter().map(|case| case.elapsed_ms));
    let spd_wall_ms = metric_summary(spd_cases.iter().map(|case| case.elapsed_ms));
    let baseline_decode_ms = metric_summary(
        baseline_cases
            .iter()
            .filter_map(|case| decode_elapsed_ms(case)),
    );
    let spd_decode_ms = metric_summary(spd_cases.iter().filter_map(|case| decode_elapsed_ms(case)));
    let spd_spec_proposed = sum_decode_u64(&spd_cases, |decode| decode.spec_proposed);
    let spd_spec_accepted = sum_decode_u64(&spd_cases, |decode| decode.spec_accepted);
    let pipeline_gap = pipeline_gap_summary(&spd_cases);
    let paper_pipeline_estimate = paper_pipeline_estimate(
        logical_spd_stage_count,
        physical_stage_count,
        spd_spec_accepted,
        spd_spec_proposed,
        &baseline_decode_ms,
        &spd_decode_ms,
    );
    let rolling_trace_replay = rolling_trace_replay_summary(&spd_cases, logical_spd_stage_count);
    SpdOpenAiSmokeSummary {
        prompt_pairs: prompt_comparisons.len(),
        matching_content: prompt_comparisons
            .iter()
            .filter(|comparison| comparison.content_matches)
            .count(),
        wall_speedup_spd_vs_baseline: speedup_from_metric(&baseline_wall_ms, &spd_wall_ms),
        decode_speedup_spd_vs_baseline: speedup_from_metric(&baseline_decode_ms, &spd_decode_ms),
        spd_spec_windows: sum_decode_u64(&spd_cases, |decode| decode.spec_windows),
        spd_spec_proposed,
        spd_spec_accepted,
        spd_spec_rejected: sum_decode_u64(&spd_cases, |decode| decode.spec_rejected),
        spd_accept_rate: ratio(spd_spec_accepted, spd_spec_proposed),
        optimistic_requests: sum_decode_u64(&spd_cases, |decode| decode.optimistic_requests),
        optimistic_accepted: sum_decode_u64(&spd_cases, |decode| decode.optimistic_accepted),
        optimistic_rejected: sum_decode_u64(&spd_cases, |decode| decode.optimistic_rejected),
        optimistic_committed: sum_decode_u64(&spd_cases, |decode| decode.optimistic_committed),
        chained_optimistic_requests: sum_decode_u64(&spd_cases, |decode| {
            decode.chained_optimistic_requests
        }),
        chained_optimistic_accepted: sum_decode_u64(&spd_cases, |decode| {
            decode.chained_optimistic_accepted
        }),
        chained_optimistic_rejected: sum_decode_u64(&spd_cases, |decode| {
            decode.chained_optimistic_rejected
        }),
        chained_optimistic_committed: sum_decode_u64(&spd_cases, |decode| {
            decode.chained_optimistic_committed
        }),
        max_optimistic_chain_depth: max_optimistic_chain_depth(&spd_cases),
        spd_rolling_executor_launches: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_launches
        }),
        spd_rolling_executor_launch_misses: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_launch_misses
        }),
        spd_rolling_executor_launch_miss_in_flight_full: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_launch_miss_in_flight_full
        }),
        spd_rolling_executor_launch_miss_no_rows: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_launch_miss_no_rows
        }),
        spd_rolling_executor_launch_miss_no_proposal: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_launch_miss_no_proposal
        }),
        spd_rolling_executor_launch_miss_shadow_not_seedable: sum_decode_u64(
            &spd_cases,
            |decode| decode.spd_rolling_executor_launch_miss_shadow_not_seedable,
        ),
        spd_rolling_executor_launch_miss_shadow_missing_view: sum_decode_u64(
            &spd_cases,
            |decode| decode.spd_rolling_executor_launch_miss_shadow_missing_view,
        ),
        spd_rolling_executor_shadow_source_reseeds: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_shadow_source_reseeds
        }),
        spd_rolling_executor_margin_rejects: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_margin_rejects
        }),
        spd_rolling_executor_max_in_flight: spd_cases
            .iter()
            .filter_map(|case| {
                case.decode
                    .as_ref()
                    .and_then(|decode| decode.spd_rolling_executor_max_in_flight)
            })
            .max()
            .unwrap_or(0),
        spd_rolling_executor_accepted_oldest: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_accepted_oldest
        }),
        spd_rolling_executor_rejected_oldest: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_rejected_oldest
        }),
        spd_rolling_executor_drained_younger: sum_decode_u64(&spd_cases, |decode| {
            decode.spd_rolling_executor_drained_younger
        }),
        tap_return_failures: spd_cases.iter().map(|case| case.tap_return_failures).sum(),
        tap_record_failures: spd_cases.iter().map(|case| case.tap_record_failures).sum(),
        tap_ignored: spd_cases.iter().map(|case| case.tap_ignored).sum(),
        pipeline_gap,
        paper_pipeline_estimate,
        rolling_trace_replay,
        baseline_wall_ms,
        spd_wall_ms,
        baseline_decode_ms,
        spd_decode_ms,
        prompt_comparisons,
    }
}

fn pipeline_gap_summary(spd_cases: &[&CaseReport]) -> SpdPipelineGapSummary {
    let pre_target = inline_probes_for_phase(spd_cases, "pre_target_reply");
    let optimistic_commit = inline_probes_for_phase(spd_cases, "optimistic_commit");
    let post_target = inline_probes_for_phase(spd_cases, "post_target_reply");
    let pre_target_proposals = pre_target
        .iter()
        .filter(|probe| probe.proposed_token.is_some())
        .count();
    let pre_target_accepted = pre_target
        .iter()
        .filter(|probe| probe.accepted == Some(true))
        .count();
    let optimistic_commit_proposals = optimistic_commit
        .iter()
        .filter(|probe| probe.proposed_token.is_some())
        .count();
    let optimistic_commit_accepted = optimistic_commit
        .iter()
        .filter(|probe| probe.accepted == Some(true))
        .count();
    let post_target_empty = post_target
        .iter()
        .filter(|probe| probe.proposed_token.is_none())
        .count();
    let optimistic_decodes = spd_cases
        .iter()
        .flat_map(|case| case.optimistic_decodes.iter())
        .collect::<Vec<_>>();
    let optimistic_tap_return_requests = optimistic_decodes
        .iter()
        .filter(|decode| decode.requested_tap_return == Some(true))
        .count();
    let optimistic_tap_return_accepted = optimistic_decodes
        .iter()
        .filter(|decode| decode.requested_tap_return == Some(true) && decode.accepted == Some(true))
        .count();
    let optimistic_tap_return_rejected = optimistic_decodes
        .iter()
        .filter(|decode| {
            decode.requested_tap_return == Some(true) && decode.accepted == Some(false)
        })
        .count();
    SpdPipelineGapSummary {
        pre_target_probes: pre_target.len(),
        pre_target_proposals,
        pre_target_accepted,
        pre_target_accept_rate: ratio_usize(pre_target_accepted, pre_target_proposals),
        optimistic_commit_probes: optimistic_commit.len(),
        optimistic_commit_proposals,
        optimistic_commit_accepted,
        optimistic_commit_accept_rate: ratio_usize(
            optimistic_commit_accepted,
            optimistic_commit_proposals,
        ),
        post_target_probes: post_target.len(),
        post_target_empty,
        post_target_empty_rate: ratio_usize(post_target_empty, post_target.len()),
        pre_target_probe_ms: metric_summary(pre_target.iter().filter_map(|probe| probe.elapsed_ms)),
        pre_target_wait_after_probe_ms: metric_summary(
            pre_target
                .iter()
                .filter_map(|probe| probe.target_wait_after_probe_ms),
        ),
        optimistic_commit_probe_ms: metric_summary(
            optimistic_commit
                .iter()
                .filter_map(|probe| probe.elapsed_ms),
        ),
        optimistic_commit_wait_after_probe_ms: metric_summary(
            optimistic_commit
                .iter()
                .filter_map(|probe| probe.target_wait_after_probe_ms),
        ),
        optimistic_decode_elapsed_ms: metric_summary(
            optimistic_decodes
                .iter()
                .filter_map(|decode| decode.elapsed_ms),
        ),
        optimistic_decode_start_elapsed_ms: metric_summary(
            optimistic_decodes
                .iter()
                .filter_map(|decode| decode.start_elapsed_ms),
        ),
        optimistic_decode_wait_ms: metric_summary(
            optimistic_decodes
                .iter()
                .filter_map(|decode| decode.wait_ms),
        ),
        optimistic_decode_hidden_wait_ms: metric_summary(
            optimistic_decodes
                .iter()
                .filter_map(|decode| decode.hidden_wait_ms),
        ),
        chained_optimistic_decode_hidden_wait_ms: metric_summary(
            optimistic_decodes
                .iter()
                .filter(|decode| decode.chain_depth.is_some() || decode.chain == Some(true))
                .filter_map(|decode| decode.hidden_wait_ms),
        ),
        probe_cache_prefill_ms: metric_summary(
            pre_target
                .iter()
                .chain(optimistic_commit.iter())
                .filter_map(|probe| probe.cache_prefill_ms),
        ),
        probe_head_fixed_stage_projection_ms: metric_summary(
            pre_target
                .iter()
                .chain(optimistic_commit.iter())
                .filter_map(|probe| probe.head_fixed_stage_projection_ms),
        ),
        probe_head_decoder_ms: metric_summary(
            pre_target
                .iter()
                .chain(optimistic_commit.iter())
                .filter_map(|probe| probe.head_decoder_ms),
        ),
        probe_head_final_norm_ms: metric_summary(
            pre_target
                .iter()
                .chain(optimistic_commit.iter())
                .filter_map(|probe| probe.head_final_norm_ms),
        ),
        probe_head_lm_head_topk_ms: metric_summary(
            pre_target
                .iter()
                .chain(optimistic_commit.iter())
                .filter_map(|probe| probe.head_lm_head_topk_ms),
        ),
        probe_head_total_ms: metric_summary(
            pre_target
                .iter()
                .chain(optimistic_commit.iter())
                .filter_map(|probe| probe.head_total_ms),
        ),
        normal_token_downstream_wait_ms: token_downstream_wait_summary(spd_cases, "DecodeEmbd"),
        optimistic_token_downstream_wait_ms: token_downstream_wait_summary(
            spd_cases,
            "DecodeEmbdOptimistic",
        ),
        pre_target_proposals_without_tap_return: pre_target_proposals
            .saturating_sub(optimistic_tap_return_requests),
        optimistic_tap_return_requests,
        optimistic_tap_return_accepted,
        optimistic_tap_return_rejected,
        optimistic_tap_return_accept_rate: ratio_usize(
            optimistic_tap_return_accepted,
            optimistic_tap_return_requests,
        ),
    }
}

fn paper_pipeline_estimate(
    logical_stage_count: usize,
    physical_stage_count: usize,
    accepted: u64,
    proposed: u64,
    baseline_decode_ms: &MetricSummary,
    spd_decode_ms: &MetricSummary,
) -> SpdPaperPipelineEstimate {
    let accepted_proposal_rate = ratio(accepted, proposed);
    let unsaved_token_round_trips = proposed.saturating_sub(accepted);
    let paper_like_speedup_vs_serial_split = accepted_proposal_rate.and_then(|rate| {
        let speedup = logical_stage_count as f64 * rate;
        (speedup > 0.0).then_some(speedup)
    });
    let estimated_decode_ms_at_baseline_stage_cost = match (
        baseline_decode_ms.mean_ms,
        paper_like_speedup_vs_serial_split,
    ) {
        (Some(baseline), Some(speedup)) => Some(baseline / speedup),
        _ => None,
    };
    let current_spd_decode_slowdown_vs_estimate = match (
        spd_decode_ms.mean_ms,
        estimated_decode_ms_at_baseline_stage_cost,
    ) {
        (Some(current), Some(estimate)) if estimate > 0.0 => Some(current / estimate),
        _ => None,
    };
    SpdPaperPipelineEstimate {
        logical_stage_count,
        physical_stage_count,
        candidate_token_round_trips: proposed,
        saved_token_round_trips: accepted,
        unsaved_token_round_trips,
        token_round_trip_save_rate: accepted_proposal_rate,
        accepted_proposal_rate,
        paper_like_speedup_vs_serial_split,
        estimated_decode_ms_at_baseline_stage_cost,
        current_spd_decode_slowdown_vs_estimate,
    }
}

fn rolling_trace_replay_summary(
    spd_cases: &[&CaseReport],
    logical_stage_count: usize,
) -> SpdRollingTraceReplaySummary {
    let mut summary = SpdRollingTraceReplaySummary {
        logical_stage_count,
        cases_replayed: 0,
        live_cases_observed: 0,
        inserted_drafts: 0,
        missing_proposals: 0,
        first_missing_proposal_position: None,
        out_of_order_proposals: 0,
        first_out_of_order_proposal_position: None,
        verified_windows: 0,
        accepted_windows: 0,
        rejected_windows: 0,
        first_rejected_target_position: None,
        final_pipeline_len: None,
        final_verified_up_to: None,
        final_verified_prefix_len: None,
        final_verified_prefix_tokens: Vec::new(),
        verified_prefix_matches_target: None,
        first_verified_prefix_mismatch_position: None,
    };
    if logical_stage_count == 0 {
        return summary;
    }
    for case in spd_cases {
        if let Some(replay) = rolling_trace_replay_case(case, logical_stage_count) {
            merge_rolling_summary(&mut summary, replay);
            continue;
        }
        if let Some(live) = live_rolling_summary_case(case, logical_stage_count) {
            merge_rolling_summary(&mut summary, live);
        }
    }
    summary
}

fn merge_rolling_summary(
    summary: &mut SpdRollingTraceReplaySummary,
    case_summary: SpdRollingTraceReplaySummary,
) {
    summary.cases_replayed += case_summary.cases_replayed;
    summary.live_cases_observed += case_summary.live_cases_observed;
    summary.inserted_drafts += case_summary.inserted_drafts;
    summary.missing_proposals += case_summary.missing_proposals;
    summary.first_missing_proposal_position = summary
        .first_missing_proposal_position
        .or(case_summary.first_missing_proposal_position);
    summary.out_of_order_proposals += case_summary.out_of_order_proposals;
    summary.first_out_of_order_proposal_position = summary
        .first_out_of_order_proposal_position
        .or(case_summary.first_out_of_order_proposal_position);
    summary.verified_windows += case_summary.verified_windows;
    summary.accepted_windows += case_summary.accepted_windows;
    summary.rejected_windows += case_summary.rejected_windows;
    summary.first_rejected_target_position = summary
        .first_rejected_target_position
        .or(case_summary.first_rejected_target_position);
    summary.final_pipeline_len = case_summary.final_pipeline_len;
    summary.final_verified_up_to = case_summary.final_verified_up_to;
    summary.final_verified_prefix_len = case_summary.final_verified_prefix_len;
    summary.final_verified_prefix_tokens = case_summary.final_verified_prefix_tokens;
    summary.verified_prefix_matches_target = combine_optional_bool_and(
        summary.verified_prefix_matches_target,
        case_summary.verified_prefix_matches_target,
    );
    summary.first_verified_prefix_mismatch_position = summary
        .first_verified_prefix_mismatch_position
        .or(case_summary.first_verified_prefix_mismatch_position);
}

fn rolling_trace_replay_case(
    case: &CaseReport,
    logical_stage_count: usize,
) -> Option<SpdRollingTraceReplaySummary> {
    let target_tokens = target_tokens_by_position(case);
    let proposals = pre_target_proposals_by_position(case)
        .into_iter()
        .filter_map(|(position, token)| i32::try_from(token).ok().map(|token| (position, token)))
        .collect::<BTreeMap<_, _>>();
    let replay =
        SpdRollingTraceReplay::from_observed_trace(logical_stage_count, &target_tokens, &proposals)
            .ok()??;
    Some(SpdRollingTraceReplaySummary {
        logical_stage_count,
        cases_replayed: 1,
        live_cases_observed: 0,
        inserted_drafts: replay.inserted_drafts,
        missing_proposals: replay.missing_proposals,
        first_missing_proposal_position: replay.first_missing_proposal_position,
        out_of_order_proposals: replay.out_of_order_proposals,
        first_out_of_order_proposal_position: replay.first_out_of_order_proposal_position,
        verified_windows: replay.verified_windows,
        accepted_windows: replay.accepted_windows,
        rejected_windows: replay.rejected_windows,
        first_rejected_target_position: replay.first_rejected_target_position,
        final_pipeline_len: Some(replay.final_pipeline_len),
        final_verified_up_to: Some(replay.final_verified_up_to),
        final_verified_prefix_len: Some(replay.final_verified_prefix_tokens.len()),
        final_verified_prefix_tokens: replay.final_verified_prefix_tokens,
        verified_prefix_matches_target: Some(replay.verified_prefix_matches_target),
        first_verified_prefix_mismatch_position: replay.first_verified_prefix_mismatch_position,
    })
}

fn live_rolling_summary_case(
    case: &CaseReport,
    logical_stage_count: usize,
) -> Option<SpdRollingTraceReplaySummary> {
    let rolling = case.decode.as_ref()?.rolling.as_ref()?;
    Some(SpdRollingTraceReplaySummary {
        logical_stage_count,
        cases_replayed: 0,
        live_cases_observed: 1,
        inserted_drafts: usize_from_optional_u64(rolling.inserted_drafts)?,
        missing_proposals: usize_from_optional_u64(rolling.missing_proposals)?,
        first_missing_proposal_position: rolling
            .first_missing_proposal_position
            .and_then(usize_from_u64),
        out_of_order_proposals: usize_from_optional_u64(rolling.out_of_order_proposals)?,
        first_out_of_order_proposal_position: rolling
            .first_out_of_order_proposal_position
            .and_then(usize_from_u64),
        verified_windows: usize_from_optional_u64(rolling.verified_windows)?,
        accepted_windows: usize_from_optional_u64(rolling.accepted_windows)?,
        rejected_windows: usize_from_optional_u64(rolling.rejected_windows)?,
        first_rejected_target_position: rolling
            .first_rejected_target_position
            .and_then(usize_from_u64),
        final_pipeline_len: rolling.pipeline_len.and_then(usize_from_u64),
        final_verified_up_to: rolling.verified_up_to.and_then(usize_from_u64),
        final_verified_prefix_len: None,
        final_verified_prefix_tokens: Vec::new(),
        verified_prefix_matches_target: None,
        first_verified_prefix_mismatch_position: None,
    })
}

fn usize_from_optional_u64(value: Option<u64>) -> Option<usize> {
    usize::try_from(value?).ok()
}

fn usize_from_u64(value: u64) -> Option<usize> {
    usize::try_from(value).ok()
}

fn combine_optional_bool_and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left && right),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn target_tokens_by_position(case: &CaseReport) -> BTreeMap<usize, i32> {
    let rolling_positions = rolling_target_positions_by_step(case);
    let anchor = rolling_position_anchor(&rolling_positions);
    case.token_events
        .iter()
        .filter_map(|event| {
            let step = usize::try_from(event.step?).ok()?;
            let token = i32::try_from(event.predicted_token?).ok()?;
            let position = observed_position_for_step(&rolling_positions, anchor, step);
            Some((position, token))
        })
        .collect()
}

fn pre_target_proposals_by_position(case: &CaseReport) -> BTreeMap<usize, i64> {
    let rolling_positions = rolling_target_positions_by_step(case);
    let anchor = rolling_position_anchor(&rolling_positions);
    case.inline_probes
        .iter()
        .filter(|probe| {
            matches!(
                probe.phase.as_deref(),
                Some("pre_target_reply" | "optimistic_commit")
            )
        })
        .filter_map(|probe| {
            let step = usize::try_from(probe.step?).ok()?;
            let position = probe
                .rolling
                .as_ref()
                .and_then(|rolling| rolling.target_position)
                .and_then(|position| usize::try_from(position).ok())
                .unwrap_or_else(|| observed_position_for_step(&rolling_positions, anchor, step));
            Some((position, probe.proposed_token?))
        })
        .collect()
}

fn rolling_target_positions_by_step(case: &CaseReport) -> BTreeMap<usize, usize> {
    case.inline_probes
        .iter()
        .filter_map(|probe| {
            let step = usize::try_from(probe.step?).ok()?;
            let position = usize::try_from(probe.rolling.as_ref()?.target_position?).ok()?;
            Some((step, position))
        })
        .collect()
}

fn rolling_position_anchor(positions: &BTreeMap<usize, usize>) -> Option<(usize, usize)> {
    positions
        .iter()
        .next()
        .map(|(step, position)| (*step, *position))
}

fn observed_position_for_step(
    rolling_positions: &BTreeMap<usize, usize>,
    anchor: Option<(usize, usize)>,
    step: usize,
) -> usize {
    if let Some(position) = rolling_positions.get(&step).copied() {
        return position;
    }
    let Some((anchor_step, anchor_position)) = anchor else {
        return step;
    };
    if step >= anchor_step {
        return anchor_position
            .checked_add(step - anchor_step)
            .unwrap_or(step);
    }
    anchor_position
        .checked_sub(anchor_step - step)
        .unwrap_or(step)
}

fn inline_probes_for_phase<'a>(
    cases: &[&'a CaseReport],
    phase: &str,
) -> Vec<&'a InlineProbeReport> {
    cases
        .iter()
        .flat_map(|case| case.inline_probes.iter())
        .filter(|probe| probe.phase.as_deref() == Some(phase))
        .collect()
}

fn token_downstream_wait_summary(cases: &[&CaseReport], message_kind: &str) -> MetricSummary {
    metric_summary(
        cases
            .iter()
            .flat_map(|case| case.token_events.iter())
            .filter(|event| event.message_kind.as_deref() == Some(message_kind))
            .filter_map(|event| event.downstream_wait_ms),
    )
}

fn max_optimistic_chain_depth(cases: &[&CaseReport]) -> u64 {
    cases
        .iter()
        .flat_map(|case| {
            case.optimistic_decodes
                .iter()
                .filter_map(|decode| decode.chain_depth)
                .chain(
                    case.token_events
                        .iter()
                        .filter_map(|event| event.chain_depth),
                )
        })
        .max()
        .unwrap_or(0)
}

fn optimistic_hidden_wait_ms(
    elapsed_ms: Option<f64>,
    start_elapsed_ms: Option<f64>,
    wait_ms: Option<f64>,
) -> Option<f64> {
    let elapsed_ms = elapsed_ms?;
    let start_elapsed_ms = start_elapsed_ms?;
    let wait_ms = wait_ms?;
    Some((elapsed_ms - start_elapsed_ms - wait_ms).max(0.0))
}

fn cases_for_name(cases: &[CaseReport], case: SmokeCase) -> Vec<&CaseReport> {
    cases
        .iter()
        .filter(|report| report.name == case.as_str() && !report.warmup)
        .collect()
}

fn compare_prompt_pairs(
    baseline_cases: &[&CaseReport],
    spd_cases: &[&CaseReport],
) -> Vec<PromptComparisonReport> {
    baseline_cases
        .iter()
        .filter_map(|baseline| {
            let spd = spd_cases.iter().copied().find(|case| {
                case.prompt_index == baseline.prompt_index
                    && case.repeat_index == baseline.repeat_index
            })?;
            Some(compare_prompt_pair(baseline, spd))
        })
        .collect()
}

fn compare_prompt_pair(baseline: &CaseReport, spd: &CaseReport) -> PromptComparisonReport {
    let baseline_decode_ms = decode_elapsed_ms(baseline);
    let spd_decode_ms = decode_elapsed_ms(spd);
    let spd_proposed = decode_u64(spd, |decode| decode.spec_proposed);
    let spd_accepted = decode_u64(spd, |decode| decode.spec_accepted);
    PromptComparisonReport {
        prompt_index: baseline.prompt_index,
        prompt_label: baseline.prompt_label.clone(),
        repeat_index: baseline.repeat_index,
        content_matches: baseline.content == spd.content,
        baseline_elapsed_ms: baseline.elapsed_ms,
        spd_elapsed_ms: spd.elapsed_ms,
        wall_speedup_spd_vs_baseline: speedup(baseline.elapsed_ms, spd.elapsed_ms),
        baseline_decode_ms,
        spd_decode_ms,
        decode_speedup_spd_vs_baseline: optional_speedup(baseline_decode_ms, spd_decode_ms),
        spd_spec_proposed: spd_proposed,
        spd_spec_accepted: spd_accepted,
        spd_spec_rejected: decode_u64(spd, |decode| decode.spec_rejected),
        spd_accept_rate: optional_ratio(spd_accepted, spd_proposed),
        optimistic_committed: decode_u64(spd, |decode| decode.optimistic_committed),
    }
}

fn ensure_content_matches(
    comparisons: &[PromptComparisonReport],
    allow_content_mismatch: bool,
) -> Result<()> {
    if let Some(message) = content_mismatch_failure(comparisons, allow_content_mismatch) {
        bail!("{message}");
    }
    Ok(())
}

fn content_mismatch_failure(
    comparisons: &[PromptComparisonReport],
    allow_content_mismatch: bool,
) -> Option<String> {
    if allow_content_mismatch {
        return None;
    }
    let mismatches = comparisons
        .iter()
        .filter(|comparison| !comparison.content_matches)
        .map(|comparison| {
            format!(
                "{}#{} repeat {}",
                comparison.prompt_label, comparison.prompt_index, comparison.repeat_index
            )
        })
        .collect::<Vec<_>>();
    if mismatches.is_empty() {
        return None;
    }
    Some(format!(
        "SPD OpenAI smoke content mismatch for {} paired prompt(s): {}. Report was written; pass --allow-content-mismatch for exploratory sweeps.",
        mismatches.len(),
        mismatches.join(", ")
    ))
}

fn metric_summary(values: impl Iterator<Item = f64>) -> MetricSummary {
    let values = values.collect::<Vec<_>>();
    if values.is_empty() {
        return MetricSummary {
            count: 0,
            mean_ms: None,
            min_ms: None,
            max_ms: None,
        };
    }
    let count = values.len();
    let sum = values.iter().sum::<f64>();
    MetricSummary {
        count,
        mean_ms: Some(sum / count as f64),
        min_ms: values.iter().copied().reduce(f64::min),
        max_ms: values.iter().copied().reduce(f64::max),
    }
}

fn speedup_from_metric(baseline: &MetricSummary, spd: &MetricSummary) -> Option<f64> {
    optional_speedup(baseline.mean_ms, spd.mean_ms)
}

fn optional_speedup(baseline_ms: Option<f64>, spd_ms: Option<f64>) -> Option<f64> {
    speedup(baseline_ms?, spd_ms?)
}

fn speedup(baseline_ms: f64, spd_ms: f64) -> Option<f64> {
    (spd_ms > 0.0).then_some(baseline_ms / spd_ms)
}

fn decode_elapsed_ms(case: &CaseReport) -> Option<f64> {
    case.decode.as_ref().and_then(|decode| decode.elapsed_ms)
}

fn decode_u64(case: &CaseReport, field: impl FnOnce(&DecodeReport) -> Option<u64>) -> Option<u64> {
    case.decode.as_ref().and_then(field)
}

fn sum_decode_u64(cases: &[&CaseReport], field: impl Fn(&DecodeReport) -> Option<u64>) -> u64 {
    cases
        .iter()
        .filter_map(|case| case.decode.as_ref().and_then(&field))
        .sum()
}

fn optional_ratio(numerator: Option<u64>, denominator: Option<u64>) -> Option<f64> {
    ratio(numerator?, denominator?)
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn ratio_usize(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn validate_args(args: &SpdOpenAiSmokeArgs) -> Result<()> {
    if !args.stage_server_bin.is_file() {
        bail!(
            "stage server binary does not exist: {}",
            args.stage_server_bin.display()
        );
    }
    if !args.model_path.is_file() && !args.model_path.is_dir() {
        bail!(
            "model path must be a GGUF file or layer package directory: {}",
            args.model_path.display()
        );
    }
    if args.rsync_model_artifacts && args.model_path.is_dir() {
        bail!(
            "--rsync-model-artifacts does not support layer package directories; pre-stage packages and use --remote-model-path-map"
        );
    }
    if !args.manifest.is_file() {
        bail!("SPD manifest does not exist: {}", args.manifest.display());
    }
    if !args.fixture.is_file() {
        bail!("SPD fixture does not exist: {}", args.fixture.display());
    }
    if args.activation_width <= 0 {
        bail!("--activation-width must be greater than zero");
    }
    if args.max_tokens == 0 {
        bail!("--max-tokens must be greater than zero");
    }
    if matches!(args.prompt_limit, Some(0)) {
        bail!("--prompt-limit must be greater than zero");
    }
    if args.repeat_count == 0 {
        bail!("--repeat-count must be greater than zero");
    }
    if let Some(path) = args.prompt_file.as_ref()
        && !path.is_file()
    {
        bail!("prompt file does not exist: {}", path.display());
    }
    if args.openai_generation_concurrency == 0 {
        bail!("--openai-generation-concurrency must be greater than zero");
    }
    if args.max_inflight == 0 {
        bail!("--max-inflight must be greater than zero");
    }
    if args.downstream_wire_delay_ms < 0.0 {
        bail!("--downstream-wire-delay-ms must be non-negative");
    }
    if matches!(args.downstream_wire_mbps, Some(value) if value <= 0.0) {
        bail!("--downstream-wire-mbps must be greater than zero");
    }
    if args.speculative_window == 0 {
        bail!("--speculative-window must be greater than zero");
    }
    if args.spd_top_k == 0 {
        bail!("--spd-top-k must be greater than zero");
    }
    if matches!(args.optimistic_min_logit_margin, Some(value) if !value.is_finite() || value < 0.0)
    {
        bail!("--optimistic-min-logit-margin must be finite and non-negative");
    }
    if args.optimistic_min_logit_margin.is_some() && args.spd_top_k < 2 {
        bail!("--optimistic-min-logit-margin requires --spd-top-k >= 2");
    }
    if args.spd_rolling_executor && !args.optimistic_decode {
        bail!("--spd-rolling-executor requires --optimistic-decode true");
    }
    if !args.run_baseline && !args.run_spd {
        bail!("at least one of --run-baseline or --run-spd must be enabled");
    }
    stage_ranges(&args.splits, args.layer_end)?;
    Ok(())
}

fn validate_manifest_activation_width(
    activation_width: i32,
    manifest: &SpdHeadManifest,
) -> Result<()> {
    let activation_width =
        u32::try_from(activation_width).context("--activation-width must be greater than zero")?;
    if activation_width != manifest.topology.hidden_size {
        bail!(
            "--activation-width {} must match SPD manifest hidden_size {}",
            activation_width,
            manifest.topology.hidden_size
        );
    }
    Ok(())
}

pub(super) fn model_path_is_layer_package(path: &Path) -> bool {
    path.is_dir()
}

pub(super) fn stage_load_mode_for_model_path(path: &Path) -> &'static str {
    if model_path_is_layer_package(path) {
        "layer-package"
    } else {
        "runtime-slice"
    }
}

pub(super) fn explicit_spd_model_path(path: &Path) -> Option<&Path> {
    (!model_path_is_layer_package(path)).then_some(path)
}

fn stage_ranges(splits: &[u32], layer_end: u32) -> Result<Vec<(u32, u32)>> {
    let mut previous = 0;
    for split in splits {
        if *split <= previous || *split >= layer_end {
            bail!("--splits must partition 0..layer-end in strictly ascending order");
        }
        previous = *split;
    }
    let mut bounds = Vec::with_capacity(splits.len() + 2);
    bounds.push(0);
    bounds.extend_from_slice(splits);
    bounds.push(layer_end);
    Ok(bounds.windows(2).map(|pair| (pair[0], pair[1])).collect())
}

fn spd_tap_allowlist(args: &SpdOpenAiSmokeArgs, manifest: &SpdHeadManifest) -> Result<Vec<u32>> {
    if !args.spd_tap_return_hf_indices.is_empty() {
        return Ok(sorted_unique_nonzero(&args.spd_tap_return_hf_indices));
    }
    if !args.derive_tap_allowlist {
        return Ok(Vec::new());
    }
    Ok(sorted_unique_nonzero(
        &required_spd_hf_indices_for_topology(&manifest.topology),
    ))
}

fn sorted_unique_nonzero(indices: &[u32]) -> Vec<u32> {
    indices
        .iter()
        .copied()
        .filter(|index| *index != 0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn run_case(
    args: &SpdOpenAiSmokeArgs,
    work_dir: &Path,
    stage_ranges: &[(u32, u32)],
    tap_allowlist: &[u32],
    case: SmokeCase,
    prompt: &PromptInput,
    iteration: CaseIteration,
) -> Result<CaseReport> {
    let iteration_kind = if iteration.warmup { "warmup" } else { "repeat" };
    let case_dir = work_dir.join(format!(
        "{}-{}-{}-{:03}",
        prompt.label,
        case.as_str(),
        iteration_kind,
        iteration.repeat_index
    ));
    fs::create_dir_all(&case_dir)
        .with_context(|| format!("failed to create {}", case_dir.display()))?;
    let run_id = format!(
        "spd-openai-{}-{}-{}-{:03}-{}",
        prompt.label,
        case.as_str(),
        iteration_kind,
        iteration.repeat_index,
        timestamp_millis()
    );
    let deployment =
        prepare_case_deployment(args, &case_dir, &run_id, stage_ranges, tap_allowlist, case)?;
    let mut stage_processes = start_case_stages(args, &deployment, case)?;

    let openai_base_url = format!("http://{}", deployment.openai_addr);
    let client = Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs))
        .build()
        .context("failed to build HTTP client")?;
    wait_openai_ready(&client, &openai_base_url, args.startup_timeout_secs)
        .with_context(|| format!("{} OpenAI frontend did not become ready", case.as_str()))?;

    let request_body = json!({
        "model": args.model_id,
        "messages": &prompt.messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": false,
        "chat_template_kwargs": {
            "enable_thinking": args.enable_thinking,
        },
    });
    fs::write(
        case_dir.join("request.json"),
        serde_json::to_vec_pretty(&request_body)?,
    )
    .context("write request JSON")?;

    let request_started = Instant::now();
    let response = client
        .post(format!("{openai_base_url}{OPENAI_PATH_CHAT_COMPLETIONS}"))
        .json(&request_body)
        .send()
        .context("send OpenAI chat completion request")?;
    let elapsed_ms = elapsed_ms(request_started);
    let status = response.status();
    let response_text = response.text().context("read OpenAI response body")?;
    fs::write(case_dir.join("response.json"), &response_text).context("write response JSON")?;
    finish_case_stages(&deployment, &mut stage_processes)?;
    if !status.is_success() {
        bail!(
            "{} OpenAI request failed with status {status}: {response_text}",
            case.as_str()
        );
    }
    let response_json =
        serde_json::from_str::<Value>(&response_text).context("parse OpenAI response JSON")?;

    let summary = CaseSummaryContext {
        case,
        case_dir: &case_dir,
        stage_count: deployment.stages.len(),
        openai_base_url: &openai_base_url,
        run_id: &run_id,
        prompt,
        iteration,
        elapsed_ms,
    };
    summarize_case_logs(&summary, &response_json)
}

fn finish_case_stages(
    deployment: &remote::CaseDeployment,
    stage_processes: &mut Vec<ChildGuard>,
) -> Result<()> {
    stop_remote_case_stages(deployment)?;
    stage_processes.clear();
    collect_remote_case_logs(deployment)
}

fn wait_openai_ready(client: &Client, base_url: &str, timeout_secs: u64) -> Result<()> {
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match client.get(format!("{base_url}{OPENAI_PATH_MODELS}")).send() {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => {
                last_error = Some(anyhow::anyhow!("readiness returned {}", response.status()));
            }
            Err(error) => last_error = Some(error.into()),
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("timed out waiting for OpenAI readiness")))
}

struct CaseSummaryContext<'a> {
    case: SmokeCase,
    case_dir: &'a Path,
    stage_count: usize,
    openai_base_url: &'a str,
    run_id: &'a str,
    prompt: &'a PromptInput,
    iteration: CaseIteration,
    elapsed_ms: f64,
}

fn summarize_case_logs(context: &CaseSummaryContext<'_>, response: &Value) -> Result<CaseReport> {
    let stage_events = (0..context.stage_count)
        .map(|index| read_events(&context.case_dir.join(format!("stage{index}.log"))))
        .collect::<Result<Vec<_>>>()?;
    let stage0 = stage_events
        .first()
        .context("stage event set does not include stage 0")?;
    Ok(CaseReport {
        name: context.case.as_str(),
        prompt_index: context.prompt.index,
        prompt_label: context.prompt.label.clone(),
        warmup: context.iteration.warmup,
        repeat_index: context.iteration.repeat_index,
        prompt: context.prompt.text.clone(),
        run_id: context.run_id.to_string(),
        openai_base_url: context.openai_base_url.to_string(),
        logs_dir: context.case_dir.display().to_string(),
        elapsed_ms: context.elapsed_ms,
        content: response_content(response).unwrap_or_default(),
        usage: response.get("usage").cloned(),
        finish_reason: response_finish_reason(response),
        decode: decode_report(stage0),
        inline_probes: inline_probe_reports(stage0),
        optimistic_decodes: optimistic_decode_reports(stage0),
        token_events: token_event_reports(stage0),
        tap_returns_by_hf_index: count_events_by_hf_index(
            &stage_events,
            "stage.binary_spd_tap_return",
            "llama_stage.spd_tap_return_hf_index",
        ),
        tap_records_by_hf_index: count_events_by_hf_index(
            std::slice::from_ref(stage0),
            "stage.openai_spd_tap_record",
            "llama_stage.spd_inline_tap_hf_index",
        ),
        tap_return_failures: count_events(&stage_events, "stage.binary_spd_tap_return_failed"),
        tap_record_failures: count_events(
            std::slice::from_ref(stage0),
            "stage.openai_spd_tap_record_failed",
        ),
        tap_ignored: count_events(std::slice::from_ref(stage0), "stage.openai_spd_tap_ignored"),
    })
}

fn response_content(response: &Value) -> Option<String> {
    response
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
        .map(ToString::to_string)
}

fn response_finish_reason(response: &Value) -> Option<String> {
    response
        .get("choices")?
        .as_array()?
        .first()?
        .get("finish_reason")?
        .as_str()
        .map(ToString::to_string)
}

fn decode_report(events: &[Value]) -> Option<DecodeReport> {
    let attrs = attrs_for(events, "stage.openai_decode").pop()?;
    Some(DecodeReport {
        elapsed_ms: attr_f64(attrs, "llama_stage.elapsed_ms"),
        tokens: attr_u64(attrs, "llama_stage.decode_token_count"),
        spec_enabled: attr_bool(attrs, "llama_stage.spec.enabled"),
        spec_windows: attr_u64(attrs, "llama_stage.spec.windows"),
        spec_proposed: attr_u64(attrs, "llama_stage.spec.proposed"),
        spec_accepted: attr_u64(attrs, "llama_stage.spec.accepted"),
        spec_rejected: attr_u64(attrs, "llama_stage.spec.rejected"),
        spec_draft_propose_ms: attr_f64(attrs, "llama_stage.spec.draft_propose_ms"),
        optimistic_requests: attr_u64(attrs, "llama_stage.spec.optimistic_decode_requests"),
        optimistic_accepted: attr_u64(attrs, "llama_stage.spec.optimistic_decode_accepted"),
        optimistic_rejected: attr_u64(attrs, "llama_stage.spec.optimistic_decode_rejected"),
        optimistic_committed: attr_u64(
            attrs,
            "llama_stage.spec.optimistic_decode_committed_tokens",
        ),
        optimistic_checkpoint_ms: attr_f64(attrs, "llama_stage.spec.optimistic_checkpoint_ms"),
        optimistic_decode_elapsed_ms: attr_f64(
            attrs,
            "llama_stage.spec.optimistic_decode_elapsed_ms",
        ),
        optimistic_decode_wait_ms: attr_f64(attrs, "llama_stage.spec.optimistic_decode_wait_ms"),
        optimistic_restore_ms: attr_f64(attrs, "llama_stage.spec.optimistic_restore_ms"),
        chained_optimistic_requests: attr_u64(
            attrs,
            "llama_stage.spec.chained_optimistic_decode_requests",
        ),
        chained_optimistic_accepted: attr_u64(
            attrs,
            "llama_stage.spec.chained_optimistic_decode_accepted",
        ),
        chained_optimistic_rejected: attr_u64(
            attrs,
            "llama_stage.spec.chained_optimistic_decode_rejected",
        ),
        chained_optimistic_committed: attr_u64(
            attrs,
            "llama_stage.spec.chained_optimistic_decode_committed_tokens",
        ),
        spd_rolling_executor_launches: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_launches",
        ),
        spd_rolling_executor_launch_misses: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_launch_misses",
        ),
        spd_rolling_executor_launch_miss_in_flight_full: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_launch_miss_in_flight_full",
        ),
        spd_rolling_executor_launch_miss_no_rows: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_launch_miss_no_rows",
        ),
        spd_rolling_executor_launch_miss_no_proposal: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_launch_miss_no_proposal",
        ),
        spd_rolling_executor_launch_miss_shadow_not_seedable: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_launch_miss_shadow_not_seedable",
        ),
        spd_rolling_executor_launch_miss_shadow_missing_view: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_launch_miss_shadow_missing_view",
        ),
        spd_rolling_executor_shadow_source_reseeds: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_shadow_source_reseeds",
        ),
        spd_rolling_executor_margin_rejects: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_margin_rejects",
        ),
        spd_rolling_executor_max_in_flight: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_max_in_flight",
        ),
        spd_rolling_executor_accepted_oldest: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_accepted_oldest",
        ),
        spd_rolling_executor_rejected_oldest: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_rejected_oldest",
        ),
        spd_rolling_executor_drained_younger: attr_u64(
            attrs,
            "llama_stage.spec.spd_rolling_executor_drained_younger",
        ),
        stage0_compute_ms: attr_f64(attrs, "llama_stage.stage0_compute_ms"),
        downstream_wait_ms: attr_f64(attrs, "llama_stage.downstream_wait_ms"),
        spd_proposal_total_requested_limit: attr_u64(
            attrs,
            "llama_stage.spd_proposal.total.requested_limit",
        ),
        spd_proposal_total_attempts: attr_u64(attrs, "llama_stage.spd_proposal.total.attempts"),
        spd_proposal_total_proposed: attr_u64(attrs, "llama_stage.spd_proposal.total.proposed"),
        spd_proposal_total_inline_tap_hits: attr_u64(
            attrs,
            "llama_stage.spd_proposal.total.inline_tap_hits",
        ),
        spd_proposal_total_replay_fallbacks: attr_u64(
            attrs,
            "llama_stage.spd_proposal.total.replay_fallbacks",
        ),
        spd_proposal_total_cache_hits: attr_u64(attrs, "llama_stage.spd_proposal.total.cache_hits"),
        spd_proposal_total_cache_misses: attr_u64(
            attrs,
            "llama_stage.spd_proposal.total.cache_misses",
        ),
        spd_proposal_total_tap_collect_ms: attr_f64(
            attrs,
            "llama_stage.spd_proposal.total.tap_collect_ms",
        ),
        spd_proposal_total_cur_in_ms: attr_f64(attrs, "llama_stage.spd_proposal.total.cur_in_ms"),
        spd_proposal_total_forward_ms: attr_f64(attrs, "llama_stage.spd_proposal.total.forward_ms"),
        spd_proposal_total_cache_prefill_ms: attr_f64(
            attrs,
            "llama_stage.spd_proposal.total.cache_prefill_ms",
        ),
        spd_proposal_total_head_fixed_stage_projection_ms: attr_f64(
            attrs,
            "llama_stage.spd_proposal.total.head_fixed_stage_projection_ms",
        ),
        spd_proposal_total_head_decoder_ms: attr_f64(
            attrs,
            "llama_stage.spd_proposal.total.head_decoder_ms",
        ),
        spd_proposal_total_head_final_norm_ms: attr_f64(
            attrs,
            "llama_stage.spd_proposal.total.head_final_norm_ms",
        ),
        spd_proposal_total_head_lm_head_topk_ms: attr_f64(
            attrs,
            "llama_stage.spd_proposal.total.head_lm_head_topk_ms",
        ),
        spd_proposal_total_head_total_ms: attr_f64(
            attrs,
            "llama_stage.spd_proposal.total.head_total_ms",
        ),
        spd_proposal_total_last_cache_prefix_len: attr_u64(
            attrs,
            "llama_stage.spd_proposal.total.last_cache_prefix_len",
        ),
        spd_proposal_total_max_cache_prefix_len: attr_u64(
            attrs,
            "llama_stage.spd_proposal.total.max_cache_prefix_len",
        ),
        rolling: live_rolling_report(attrs),
    })
}

fn inline_probe_reports(events: &[Value]) -> Vec<InlineProbeReport> {
    attrs_for(events, "stage.openai_spd_inline_probe")
        .into_iter()
        .map(|attrs| InlineProbeReport {
            step: attr_u64(attrs, "llama_stage.decode_step"),
            phase: attr_string(attrs, "llama_stage.spd_inline_probe_phase"),
            elapsed_ms: attr_f64(attrs, "llama_stage.elapsed_ms"),
            target_wait_after_probe_ms: attr_f64(
                attrs,
                "llama_stage.spd_inline_probe_target_wait_after_probe_ms",
            ),
            current_token: attr_i64(attrs, "llama_stage.spd_inline_probe_current_token"),
            proposed_token: attr_i64(attrs, "llama_stage.spd_inline_probe_proposed_token"),
            proposed_logit: attr_f64(attrs, "llama_stage.spd_inline_probe_proposed_logit"),
            proposed_logit_margin: attr_f64(attrs, "llama_stage.spd_inline_probe_logit_margin"),
            cache_used: attr_bool(attrs, "llama_stage.spd_inline_probe_cache_used"),
            cache_prefix_len: attr_u64(attrs, "llama_stage.spd_inline_probe_cache_prefix_len"),
            tap_source: attr_string(attrs, "llama_stage.spd_inline_probe_tap_source"),
            tap_collect_ms: attr_f64(attrs, "llama_stage.spd_inline_probe_tap_collect_ms"),
            cur_in_ms: attr_f64(attrs, "llama_stage.spd_inline_probe_cur_in_ms"),
            forward_ms: attr_f64(attrs, "llama_stage.spd_inline_probe_forward_ms"),
            cache_prefill_ms: attr_f64(attrs, "llama_stage.spd_inline_probe_cache_prefill_ms"),
            head_fixed_stage_projection_ms: attr_f64(
                attrs,
                "llama_stage.spd_inline_probe_head_fixed_stage_projection_ms",
            ),
            head_decoder_ms: attr_f64(attrs, "llama_stage.spd_inline_probe_head_decoder_ms"),
            head_decoder_layer_ms: attr_f64_array(
                attrs,
                "llama_stage.spd_inline_probe_head_decoder_layer_ms",
            ),
            head_final_norm_ms: attr_f64(attrs, "llama_stage.spd_inline_probe_head_final_norm_ms"),
            head_lm_head_topk_ms: attr_f64(
                attrs,
                "llama_stage.spd_inline_probe_head_lm_head_topk_ms",
            ),
            head_total_ms: attr_f64(attrs, "llama_stage.spd_inline_probe_head_total_ms"),
            target_token: attr_i64(attrs, "llama_stage.spd_inline_probe_target_token"),
            accepted: attr_bool(attrs, "llama_stage.spd_inline_probe_accepted"),
            trigger_hf_index: attr_u64(attrs, "llama_stage.spd_inline_probe_trigger_hf_index"),
            proposal_row_positions: attr_i64_array(
                attrs,
                "llama_stage.spd_inline_probe_proposal_row_positions",
            ),
            proposal_row_i_stages: attr_i64_array(
                attrs,
                "llama_stage.spd_inline_probe_proposal_row_i_stages",
            ),
            proposal_row_evicted_prefix_position: attr_u64(
                attrs,
                "llama_stage.spd_inline_probe_proposal_row_evicted_prefix_position",
            ),
            proposal_row_newest_position: attr_u64(
                attrs,
                "llama_stage.spd_inline_probe_proposal_row_newest_position",
            ),
            proposal_row_next_draft_position: attr_u64(
                attrs,
                "llama_stage.spd_inline_probe_proposal_row_next_draft_position",
            ),
            proposal_miss_reason: attr_string(attrs, "llama_stage.spd_inline_probe_miss_reason"),
            proposal_missing_taps: attr_i64_array_map(
                attrs,
                "llama_stage.spd_inline_probe_missing_taps",
            ),
            rolling: live_rolling_report(attrs),
            rolling_verified_delta: rolling_verified_delta_report(attrs),
        })
        .collect()
}

fn live_rolling_report(attrs: &Value) -> Option<SpdLiveRollingReport> {
    let logical_stage_count = attr_u64(attrs, "llama_stage.spd_rolling.logical_stage_count");
    logical_stage_count.map(|logical_stage_count| SpdLiveRollingReport {
        logical_stage_count: Some(logical_stage_count),
        target_position: attr_u64(attrs, "llama_stage.spd_rolling.target_position"),
        next_position: attr_u64(attrs, "llama_stage.spd_rolling.next_position"),
        inserted_drafts: attr_u64(attrs, "llama_stage.spd_rolling.inserted_drafts"),
        missing_proposals: attr_u64(attrs, "llama_stage.spd_rolling.missing_proposals"),
        first_missing_proposal_position: attr_u64(
            attrs,
            "llama_stage.spd_rolling.first_missing_proposal_position",
        ),
        out_of_order_proposals: attr_u64(attrs, "llama_stage.spd_rolling.out_of_order_proposals"),
        first_out_of_order_proposal_position: attr_u64(
            attrs,
            "llama_stage.spd_rolling.first_out_of_order_proposal_position",
        ),
        verified_windows: attr_u64(attrs, "llama_stage.spd_rolling.verified_windows"),
        accepted_windows: attr_u64(attrs, "llama_stage.spd_rolling.accepted_windows"),
        rejected_windows: attr_u64(attrs, "llama_stage.spd_rolling.rejected_windows"),
        first_rejected_target_position: attr_u64(
            attrs,
            "llama_stage.spd_rolling.first_rejected_target_position",
        ),
        pipeline_len: attr_u64(attrs, "llama_stage.spd_rolling.pipeline_len"),
        verified_up_to: attr_u64(attrs, "llama_stage.spd_rolling.verified_up_to"),
        row_evicted_prefix_position: attr_u64(
            attrs,
            "llama_stage.spd_rolling.row_evicted_prefix_position",
        ),
        row_positions: attr_u64_array(attrs, "llama_stage.spd_rolling.row_positions"),
        row_i_stages: attr_u64_array(attrs, "llama_stage.spd_rolling.row_i_stages"),
        row_newest_position: attr_u64(attrs, "llama_stage.spd_rolling.row_newest_position"),
        row_next_draft_position: attr_u64(attrs, "llama_stage.spd_rolling.row_next_draft_position"),
    })
}

fn rolling_verified_delta_report(attrs: &Value) -> Option<SpdRollingVerifiedDeltaReport> {
    let token_count = attr_u64(attrs, "llama_stage.spd_rolling.verified_delta_token_count");
    token_count.map(|token_count| SpdRollingVerifiedDeltaReport {
        start_position: attr_u64(
            attrs,
            "llama_stage.spd_rolling.verified_delta_start_position",
        ),
        verified_up_to: attr_u64(attrs, "llama_stage.spd_rolling.verified_delta_up_to"),
        tokens: attr_i64_array(attrs, "llama_stage.spd_rolling.verified_delta_tokens"),
        token_count: Some(token_count),
    })
}

fn optimistic_decode_reports(events: &[Value]) -> Vec<OptimisticDecodeReport> {
    attrs_for(events, "stage.openai_spd_optimistic_decode")
        .into_iter()
        .map(|attrs| {
            let elapsed_ms = attr_f64(attrs, "llama_stage.spd_optimistic_decode_elapsed_ms");
            let start_elapsed_ms = attr_f64(attrs, "llama_stage.spd_optimistic_start_elapsed_ms");
            let wait_ms = attr_f64(attrs, "llama_stage.spd_optimistic_decode_wait_ms");
            OptimisticDecodeReport {
                step: attr_u64(attrs, "llama_stage.decode_step"),
                chain: attr_bool(attrs, "llama_stage.spd_optimistic_chain"),
                chain_depth: attr_u64(attrs, "llama_stage.spd_optimistic_chain_depth"),
                proposed_token: attr_i64(attrs, "llama_stage.spd_optimistic_proposed_token"),
                proposed_logit: attr_f64(attrs, "llama_stage.spd_optimistic_proposed_logit"),
                proposed_logit_margin: attr_f64(attrs, "llama_stage.spd_optimistic_logit_margin"),
                requested_tap_return: attr_bool(attrs, "llama_stage.spd_optimistic_tap_return"),
                target_token: attr_i64(attrs, "llama_stage.spd_optimistic_target_token"),
                accepted: attr_bool(attrs, "llama_stage.spd_optimistic_accepted"),
                next_token: attr_i64(attrs, "llama_stage.spd_optimistic_next_token"),
                checkpoint_ms: attr_f64(attrs, "llama_stage.spd_optimistic_checkpoint_ms"),
                elapsed_ms,
                start_elapsed_ms,
                wait_ms,
                hidden_wait_ms: optimistic_hidden_wait_ms(elapsed_ms, start_elapsed_ms, wait_ms),
                stage0_compute_ms: attr_f64(attrs, "llama_stage.stage0_compute_ms"),
            }
        })
        .collect()
}

fn token_event_reports(events: &[Value]) -> Vec<TokenEventReport> {
    attrs_for(events, "stage.openai_decode_token")
        .into_iter()
        .map(|attrs| TokenEventReport {
            step: attr_u64(attrs, "llama_stage.decode_step"),
            message_kind: attr_string(attrs, "llama_stage.message_kind"),
            predicted_token: attr_i64(attrs, "llama_stage.predicted_token"),
            chain: attr_bool(attrs, "llama_stage.spd_optimistic_chain"),
            chain_depth: attr_u64(attrs, "llama_stage.spd_optimistic_chain_depth"),
            downstream_wait_ms: attr_f64(attrs, "llama_stage.downstream_wait_ms"),
        })
        .collect()
}

fn default_work_dir() -> PathBuf {
    std::env::temp_dir().join(format!("skippy-spd-openai-smoke-{}", timestamp_millis()))
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_millis()
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests;
