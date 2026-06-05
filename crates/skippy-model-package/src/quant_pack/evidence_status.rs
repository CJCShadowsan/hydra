use std::path::{Path, PathBuf};
use std::{env, fs};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, clap::Args)]
pub(super) struct QuantPackEvidenceStatusArgs {
    plan: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    fail_on_missing: bool,
    #[arg(long)]
    fail_on_warning: bool,
    #[arg(long)]
    command_complete: Option<String>,
    #[arg(long)]
    candidate: Option<String>,
}

#[derive(Debug, Serialize)]
struct EvidenceStatusReport {
    schema_version: u32,
    kind: String,
    plan: String,
    status: EvidenceStatusKind,
    candidate_count: usize,
    total_commands: usize,
    complete_commands: usize,
    partial_commands: usize,
    missing_commands: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
    #[serde(skip_serializing_if = "EvidencePlanToolchain::is_empty")]
    toolchain: EvidencePlanToolchain,
    next_command: Option<NextEvidenceCommand>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_rank: Option<EvidenceCommandStatus>,
    candidates: Vec<CandidateEvidenceStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceStatusKind {
    Complete,
    Partial,
    Incomplete,
}

#[derive(Debug, Serialize)]
struct CandidateEvidenceStatus {
    candidate: String,
    status: EvidenceStatusKind,
    total_commands: usize,
    complete_commands: usize,
    partial_commands: usize,
    missing_commands: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
    #[serde(skip_serializing_if = "EvidencePlanToolchain::is_empty")]
    toolchain: EvidencePlanToolchain,
    next_command: Option<NextEvidenceCommand>,
    commands: Vec<EvidenceCommandStatus>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct EvidencePlanToolchain {
    #[serde(skip_serializing_if = "Option::is_none")]
    runbook_cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    skippy_bench_bin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    skippy_model_package_bin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_tool_call_script: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_tool_loop_script: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    focused_runtime: Option<serde_json::Value>,
}

impl EvidencePlanToolchain {
    fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

#[derive(Debug, Clone, Serialize)]
struct NextEvidenceCommand {
    candidate: String,
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence_type: Option<String>,
    shell: String,
    missing_outputs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_failure: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceCommandStatus {
    id: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence_type: Option<String>,
    status: EvidenceStatusKind,
    outputs: Vec<EvidenceOutputStatus>,
    missing_outputs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_failure: Option<String>,
    shell: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceOutputStatus {
    path: String,
    exists: bool,
    #[serde(skip)]
    resolved_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct EvidencePlanEnvelope {
    kind: String,
}

#[derive(Debug, Deserialize)]
struct EvidencePlanInput {
    candidate: String,
    #[serde(default)]
    warnings: Vec<String>,
    #[serde(flatten)]
    toolchain: EvidencePlanToolchain,
    commands: Vec<EvidenceCommandInput>,
}

#[derive(Debug, Deserialize)]
struct EvidencePlanAllInput {
    #[serde(default)]
    final_rank: Option<EvidenceCommandInput>,
    candidates: Vec<EvidencePlanCandidateInput>,
}

#[derive(Debug, Deserialize)]
struct EvidencePlanCandidateInput {
    candidate: String,
    #[serde(default)]
    warnings: Vec<String>,
    #[serde(flatten)]
    toolchain: EvidencePlanToolchain,
    commands: Vec<EvidenceCommandInput>,
}

#[derive(Debug, Deserialize)]
struct EvidenceCommandInput {
    id: String,
    description: String,
    #[serde(default)]
    evidence_type: Option<String>,
    shell: String,
    outputs: Vec<String>,
}

pub(super) fn run_quant_pack_evidence_status(args: QuantPackEvidenceStatusArgs) -> Result<()> {
    let report = build_evidence_status_report(&args.plan)?;
    if let Some(command_id) = args.command_complete.as_deref() {
        ensure_command_complete(&report, args.candidate.as_deref(), command_id)?;
        return Ok(());
    }
    let has_missing = report.status != EvidenceStatusKind::Complete;
    let has_warnings = !report.warnings.is_empty();
    write_status_report(args.out.as_deref(), &report)?;
    if args.fail_on_missing && has_missing {
        bail!("quant-pack evidence is incomplete");
    }
    if args.fail_on_warning && has_warnings {
        bail!("quant-pack evidence has warnings");
    }
    Ok(())
}

fn ensure_command_complete(
    report: &EvidenceStatusReport,
    candidate: Option<&str>,
    command_id: &str,
) -> Result<()> {
    let candidate_status = match candidate {
        Some(candidate) => report
            .candidates
            .iter()
            .find(|status| status.candidate == candidate)
            .with_context(|| format!("candidate {candidate:?} not found in evidence status"))?,
        None if report.candidates.len() == 1 => report
            .candidates
            .first()
            .context("evidence status has no candidates")?,
        None => bail!(
            "--candidate is required for --command-complete when evidence plan has {} candidates",
            report.candidates.len()
        ),
    };
    let command = candidate_status
        .commands
        .iter()
        .find(|command| command.id == command_id)
        .with_context(|| {
            format!(
                "command {command_id:?} not found for candidate {:?}",
                candidate_status.candidate
            )
        })?;
    if command.status == EvidenceStatusKind::Complete {
        return Ok(());
    }
    if let Some(failure) = &command.observed_failure {
        bail!(
            "evidence command {command_id:?} for candidate {:?} is {:?}: {failure}",
            candidate_status.candidate,
            command.status
        );
    }
    bail!(
        "evidence command {command_id:?} for candidate {:?} is {:?}",
        candidate_status.candidate,
        command.status
    )
}

fn build_evidence_status_report(plan: &Path) -> Result<EvidenceStatusReport> {
    let bytes = fs::read(plan).with_context(|| format!("read {}", plan.display()))?;
    let envelope: EvidencePlanEnvelope =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", plan.display()))?;
    let (candidates, final_rank) = match envelope.kind.as_str() {
        "skippy_quant_pack_evidence_plan" => {
            let input: EvidencePlanInput = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", plan.display()))?;
            (
                vec![candidate_status(
                    &input.candidate,
                    &input.warnings,
                    input.toolchain,
                    &input.commands,
                )],
                None,
            )
        }
        "skippy_quant_pack_evidence_plan_all" => {
            let input: EvidencePlanAllInput = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", plan.display()))?;
            let candidates = input
                .candidates
                .iter()
                .map(|candidate| {
                    candidate_status(
                        &candidate.candidate,
                        &candidate.warnings,
                        candidate.toolchain.clone(),
                        &candidate.commands,
                    )
                })
                .collect();
            (candidates, input.final_rank)
        }
        other => bail!("unsupported evidence plan kind {other:?}"),
    };
    Ok(status_report(plan, candidates, final_rank.as_ref()))
}

fn candidate_status(
    candidate: &str,
    warnings: &[String],
    toolchain: EvidencePlanToolchain,
    commands: &[EvidenceCommandInput],
) -> CandidateEvidenceStatus {
    let mut warnings = warnings.to_vec();
    warnings.extend(toolchain_warnings(&toolchain));
    let output_base = output_base_dir(&toolchain);
    let commands = commands
        .iter()
        .map(|command| command_status(command, output_base.as_deref()))
        .collect::<Vec<EvidenceCommandStatus>>();
    let complete_commands = commands
        .iter()
        .filter(|command| command.status == EvidenceStatusKind::Complete)
        .count();
    let partial_commands = commands
        .iter()
        .filter(|command| command.status == EvidenceStatusKind::Partial)
        .count();
    let missing_commands = commands.len() - complete_commands;
    let status = aggregate_status(missing_commands, partial_commands);
    let next_command = commands
        .iter()
        .find(|command| command.status != EvidenceStatusKind::Complete)
        .map(|command| next_command(candidate, command));
    CandidateEvidenceStatus {
        candidate: candidate.to_string(),
        status,
        total_commands: commands.len(),
        complete_commands,
        partial_commands,
        missing_commands,
        warnings,
        toolchain,
        next_command,
        commands,
    }
}

fn command_status(
    command: &EvidenceCommandInput,
    output_base: Option<&Path>,
) -> EvidenceCommandStatus {
    let outputs = command
        .outputs
        .iter()
        .map(|output| {
            let resolved_path = resolve_output_path(output, output_base);
            EvidenceOutputStatus {
                path: output.clone(),
                exists: resolved_path.exists(),
                resolved_path,
            }
        })
        .collect::<Vec<_>>();
    let missing_outputs = outputs
        .iter()
        .filter(|output| !output.exists)
        .map(|output| output.path.clone())
        .collect::<Vec<_>>();
    let mut status = command_output_status(outputs.len(), missing_outputs.len());
    let mut observed_failure = observed_failure(status, &outputs);
    let command_failure = completed_command_failure(command, &outputs);
    if status == EvidenceStatusKind::Complete && command_failure.is_some() {
        status = EvidenceStatusKind::Partial;
        observed_failure = command_failure;
    }
    EvidenceCommandStatus {
        id: command.id.clone(),
        description: command.description.clone(),
        evidence_type: command.evidence_type.clone(),
        status,
        outputs,
        missing_outputs,
        observed_failure,
        shell: command.shell.clone(),
    }
}

fn completed_command_failure(
    command: &EvidenceCommandInput,
    outputs: &[EvidenceOutputStatus],
) -> Option<String> {
    match command_semantic_kind(command) {
        "certify" | "skippy-quant-pack-certification" => certification_output_failure(outputs),
        "chat-corpus"
        | "skippy-bench-chat-corpus"
        | "long-context-chat-corpus"
        | "skippy-bench-long-context-chat-corpus" => chat_corpus_output_failure(outputs),
        "focused-runtime" | "skippy-bench-focused-runtime" => {
            focused_runtime_output_failure(outputs, FocusedRuntimeMode::Executed)
        }
        "focused-runtime-schema-smoke" | "skippy-bench-focused-runtime-schema-smoke" => {
            focused_runtime_output_failure(outputs, FocusedRuntimeMode::SchemaSmoke)
        }
        "local-split-chain" | "skippy-bench-local-split-chain" => {
            local_split_chain_output_failure(outputs)
        }
        "rank-after-evidence" | "rank-after-evidence-all" | "skippy-quant-pack-rank" => {
            rank_output_failure(outputs)
        }
        "token-lengths" | "skippy-bench-token-lengths" => token_lengths_output_failure(outputs),
        _ => None,
    }
}

fn command_semantic_kind(command: &EvidenceCommandInput) -> &str {
    command
        .evidence_type
        .as_deref()
        .filter(|evidence_type| !evidence_type.trim().is_empty())
        .unwrap_or(&command.id)
}

fn local_split_chain_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    if value.get("mode").and_then(serde_json::Value::as_str) != Some("local-split-chain-binary") {
        return Some(format!(
            "{}: local split report mode missing or invalid",
            output.path
        ));
    }
    if value.get("predicted_token").is_none() {
        return Some(format!(
            "{}: local split predicted_token missing",
            output.path
        ));
    }
    let has_payload = value
        .get("stages")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .any(|stage| {
            stage
                .get("wire_payload_bytes")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|bytes| bytes > 0)
                && stage
                    .get("payload_bytes")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|bytes| bytes > 0)
        });
    if !has_payload {
        return Some(format!(
            "{}: local split payload and wire payload bytes missing",
            output.path
        ));
    }
    None
}

fn certification_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    outputs
        .iter()
        .find(|output| output.exists)
        .and_then(certification_status_failure)
}

fn certification_status_failure(output: &EvidenceOutputStatus) -> Option<String> {
    let contents = match fs::read_to_string(&output.resolved_path) {
        Ok(contents) => contents,
        Err(error) => {
            return Some(format!(
                "{}: cannot read certification output: {error}",
                output.path
            ));
        }
    };
    let value = match serde_json::from_str::<serde_json::Value>(&contents) {
        Ok(value) => value,
        Err(error) => {
            return Some(format!(
                "{}: cannot parse certification output: {error}",
                output.path
            ));
        }
    };
    let status = value
        .get("status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("missing");
    if status == "failed" {
        return Some(format!("{}: certification status failed", output.path));
    }
    if status == "missing" {
        return Some(format!("{}: certification status missing", output.path));
    }
    None
}

fn chat_corpus_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    let errors = value
        .get("summary")
        .and_then(|summary| summary.get("errors"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    if errors > 0 {
        return Some(format!("{}: chat corpus errors {errors}", output.path));
    }
    None
}

#[derive(Clone, Copy)]
enum FocusedRuntimeMode {
    Executed,
    SchemaSmoke,
}

impl FocusedRuntimeMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Executed => "executed",
            Self::SchemaSmoke => "schema-smoke",
        }
    }
}

fn focused_runtime_output_failure(
    outputs: &[EvidenceOutputStatus],
    expected_mode: FocusedRuntimeMode,
) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    let mode = value
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("missing");
    if mode != expected_mode.as_str() {
        return Some(format!(
            "{}: focused-runtime mode {mode} != {}",
            output.path,
            expected_mode.as_str()
        ));
    }
    if focused_runtime_generated_tps(&value).is_none() {
        return Some(format!(
            "{}: focused-runtime generated throughput missing or non-positive",
            output.path
        ));
    }
    if focused_runtime_decode_p50_ms(&value).is_none() {
        return Some(format!(
            "{}: focused-runtime decode p50 missing",
            output.path
        ));
    }
    None
}

fn focused_runtime_generated_tps(value: &serde_json::Value) -> Option<f64> {
    value
        .get("throughput_tokens_per_second")
        .and_then(|throughput| throughput.get("generated"))
        .and_then(value_as_f64)
        .filter(|value| *value > 0.0)
}

fn focused_runtime_decode_p50_ms(value: &serde_json::Value) -> Option<f64> {
    value
        .get("latency_ms")
        .and_then(|latency| latency.get("decode_elapsed_ms_p50"))
        .and_then(value_as_f64)
}

fn token_lengths_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    outputs
        .iter()
        .filter(|output| output.exists)
        .find(|output| output.path.ends_with(".json"))
        .and_then(token_lengths_summary_failure)
}

fn token_lengths_summary_failure(output: &EvidenceOutputStatus) -> Option<String> {
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    let exceeds_context = value
        .get("exceeds_context")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    if exceeds_context > 0 {
        return Some(format!(
            "{}: token length rows exceed context {exceeds_context}",
            output.path
        ));
    }
    None
}

fn rank_output_failure(outputs: &[EvidenceOutputStatus]) -> Option<String> {
    let output = outputs.iter().find(|output| output.exists)?;
    let value = match read_json_output(output) {
        Ok(value) => value,
        Err(failure) => return Some(failure),
    };
    if value.get("kind").and_then(serde_json::Value::as_str) != Some("skippy_quant_pack_rank") {
        return Some(format!(
            "{}: rank report kind missing or invalid",
            output.path
        ));
    }
    let Some(candidates) = value
        .get("candidates")
        .and_then(serde_json::Value::as_array)
    else {
        return Some(format!("{}: rank report candidates missing", output.path));
    };
    let candidate_count = value
        .get("candidate_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default() as usize;
    if candidate_count == 0 {
        return Some(format!("{}: rank report candidate_count is 0", output.path));
    }
    if candidate_count != candidates.len() {
        return Some(format!(
            "{}: rank report candidate_count {candidate_count} != candidates length {}",
            output.path,
            candidates.len()
        ));
    }
    None
}

fn value_as_f64(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_u64().map(|value| value as f64))
        .or_else(|| value.as_i64().map(|value| value as f64))
}

fn read_json_output(output: &EvidenceOutputStatus) -> Result<serde_json::Value, String> {
    let contents = match fs::read_to_string(&output.resolved_path) {
        Ok(contents) => contents,
        Err(error) => {
            return Err(format!("{}: cannot read output: {error}", output.path));
        }
    };
    match serde_json::from_str::<serde_json::Value>(&contents) {
        Ok(value) => Ok(value),
        Err(error) => Err(format!("{}: cannot parse output: {error}", output.path)),
    }
}

fn output_base_dir(toolchain: &EvidencePlanToolchain) -> Option<PathBuf> {
    toolchain
        .runbook_cwd
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

fn resolve_output_path(output: &str, output_base: Option<&Path>) -> PathBuf {
    let path = Path::new(output);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    output_base
        .map(|base| base.join(path))
        .unwrap_or_else(|| path.to_path_buf())
}

fn status_report(
    plan: &Path,
    candidates: Vec<CandidateEvidenceStatus>,
    final_rank_input: Option<&EvidenceCommandInput>,
) -> EvidenceStatusReport {
    let warnings = report_warnings(&candidates);
    let toolchain = report_toolchain(&candidates);
    let output_base = output_base_dir(&toolchain);
    let final_rank =
        final_rank_input.map(|command| command_status(command, output_base.as_deref()));
    let final_rank_total = usize::from(final_rank.is_some());
    let final_rank_complete = final_rank
        .as_ref()
        .filter(|command| command.status == EvidenceStatusKind::Complete)
        .map_or(0, |_| 1);
    let final_rank_partial = final_rank
        .as_ref()
        .filter(|command| command.status == EvidenceStatusKind::Partial)
        .map_or(0, |_| 1);
    let total_commands = candidates
        .iter()
        .map(|candidate| candidate.total_commands)
        .sum::<usize>()
        + final_rank_total;
    let complete_commands = candidates
        .iter()
        .map(|candidate| candidate.complete_commands)
        .sum::<usize>()
        + final_rank_complete;
    let partial_commands = candidates
        .iter()
        .map(|candidate| candidate.partial_commands)
        .sum::<usize>()
        + final_rank_partial;
    let missing_commands = candidates
        .iter()
        .map(|candidate| candidate.missing_commands)
        .sum::<usize>()
        + (final_rank_total - final_rank_complete);
    let next_command = candidates
        .iter()
        .find_map(|candidate| candidate.next_command.clone())
        .or_else(|| {
            final_rank
                .as_ref()
                .filter(|command| command.status != EvidenceStatusKind::Complete)
                .map(|command| next_command("sweep", command))
        });
    EvidenceStatusReport {
        schema_version: 1,
        kind: "skippy_quant_pack_evidence_status".to_string(),
        plan: plan.display().to_string(),
        status: aggregate_status(missing_commands, partial_commands),
        candidate_count: candidates.len(),
        total_commands,
        complete_commands,
        partial_commands,
        missing_commands,
        warnings,
        toolchain,
        next_command,
        final_rank,
        candidates,
    }
}

fn report_toolchain(candidates: &[CandidateEvidenceStatus]) -> EvidencePlanToolchain {
    let Some(first) = candidates.first() else {
        return EvidencePlanToolchain::default();
    };
    if candidates
        .iter()
        .all(|candidate| candidate.toolchain == first.toolchain)
    {
        first.toolchain.clone()
    } else {
        EvidencePlanToolchain::default()
    }
}

fn report_warnings(candidates: &[CandidateEvidenceStatus]) -> Vec<String> {
    if candidates.len() == 1 {
        return candidates
            .first()
            .map(|candidate| candidate.warnings.clone())
            .unwrap_or_default();
    }
    candidates
        .iter()
        .flat_map(|candidate| {
            candidate
                .warnings
                .iter()
                .map(|warning| format!("{}: {warning}", candidate.candidate))
        })
        .collect()
}

fn toolchain_warnings(toolchain: &EvidencePlanToolchain) -> Vec<String> {
    let mut warnings = Vec::new();
    check_toolchain_dir(
        &mut warnings,
        "runbook_cwd",
        toolchain.runbook_cwd.as_deref(),
    );
    check_toolchain_path(
        &mut warnings,
        "skippy_bench_bin",
        toolchain.skippy_bench_bin.as_deref(),
    );
    check_toolchain_path(
        &mut warnings,
        "skippy_model_package_bin",
        toolchain.skippy_model_package_bin.as_deref(),
    );
    check_toolchain_path(
        &mut warnings,
        "agent_tool_call_script",
        toolchain.agent_tool_call_script.as_deref(),
    );
    check_toolchain_path(
        &mut warnings,
        "kv_tool_loop_script",
        toolchain.kv_tool_loop_script.as_deref(),
    );
    if let Some(focused_runtime) = toolchain.focused_runtime.as_ref() {
        for field in [
            "metrics_server_bin",
            "stage_server_bin",
            "lab_preflight_script",
        ] {
            check_toolchain_path(
                &mut warnings,
                &format!("focused_runtime.{field}"),
                focused_runtime
                    .get(field)
                    .and_then(serde_json::Value::as_str),
            );
        }
    }
    warnings
}

fn check_toolchain_dir(warnings: &mut Vec<String>, label: &str, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };
    if !Path::new(value).is_dir() {
        warnings.push(format!("toolchain {label} {value:?} is not a directory"));
    }
}

fn check_toolchain_path(warnings: &mut Vec<String>, label: &str, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };
    if is_path_like_tool(value) {
        let path = Path::new(value);
        if !path.exists() {
            warnings.push(format!("toolchain {label} {value:?} does not exist"));
        } else if !is_executable_file(path) {
            warnings.push(format!("toolchain {label} {value:?} is not executable"));
        }
    } else if find_executable_on_path(value).is_none() {
        warnings.push(format!("toolchain {label} {value:?} was not found on PATH"));
    }
}

fn is_path_like_tool(value: &str) -> bool {
    value.contains('/') || value.contains('\\') || value.starts_with('.')
}

fn find_executable_on_path(command: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|dir| dir.join(command))
        .find(|path| is_executable_file(path))
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn next_command(candidate: &str, command: &EvidenceCommandStatus) -> NextEvidenceCommand {
    NextEvidenceCommand {
        candidate: candidate.to_string(),
        id: command.id.clone(),
        evidence_type: command.evidence_type.clone(),
        shell: command.shell.clone(),
        missing_outputs: command.missing_outputs.clone(),
        observed_failure: command.observed_failure.clone(),
    }
}

fn aggregate_status(missing_commands: usize, partial_commands: usize) -> EvidenceStatusKind {
    if missing_commands == 0 {
        EvidenceStatusKind::Complete
    } else if partial_commands > 0 {
        EvidenceStatusKind::Partial
    } else {
        EvidenceStatusKind::Incomplete
    }
}

fn command_output_status(output_count: usize, missing_outputs: usize) -> EvidenceStatusKind {
    if missing_outputs == 0 {
        EvidenceStatusKind::Complete
    } else if missing_outputs < output_count {
        EvidenceStatusKind::Partial
    } else {
        EvidenceStatusKind::Incomplete
    }
}

fn observed_failure(
    status: EvidenceStatusKind,
    outputs: &[EvidenceOutputStatus],
) -> Option<String> {
    if status == EvidenceStatusKind::Complete {
        return None;
    }
    outputs
        .iter()
        .filter(|output| output.exists)
        .find_map(|output| observed_failure_in_output(&output.path, &output.resolved_path))
}

fn observed_failure_in_output(display_path: &str, read_path: &Path) -> Option<String> {
    let metadata = fs::metadata(read_path).ok()?;
    if metadata.len() > 256 * 1024 {
        return None;
    }
    let contents = fs::read_to_string(read_path).ok()?;
    let line = contents
        .lines()
        .rev()
        .find(|line| is_failure_line(line))
        .or_else(|| contents.lines().rev().find(|line| !line.trim().is_empty()))?;
    Some(format!("{display_path}: {}", line.trim()))
}

fn is_failure_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if [
        "local_network_diagnostics:",
        "local_ipv4_addresses:",
        "routes_to_failed_hosts:",
    ]
    .contains(&lower.trim())
    {
        return false;
    }
    [
        "preflight_status:",
        "remote_status: dirty",
        "qwen lab preflight:",
        "connection refused",
        "ssh_failed",
        "remote_failed",
        "failed",
        "error",
        "low_disk:",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn write_status_report(out: Option<&Path>, report: &EvidenceStatusReport) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(out) = out {
        fs::write(out, format!("{json}\n"))
            .with_context(|| format!("write evidence status {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn status_report_points_to_first_missing_command() {
        let dir = unique_test_dir("single");
        fs::create_dir_all(&dir).expect("create fixture");
        let done = dir.join("done.json");
        let missing = dir.join("missing.json");
        let skippy_bench = dir.join("tools/skippy-bench");
        let skippy_model_package = dir.join("tools/skippy-model-package");
        let agent_tool_call = dir.join("scripts/qa-agent-tool-call-reliability.py");
        let kv_tool_loop = dir.join("scripts/qa-kv-tool-loop-stability.py");
        fs::write(&done, b"{}").expect("write done output");
        write_executable_fixture(&skippy_bench);
        write_executable_fixture(&skippy_model_package);
        write_executable_fixture(&agent_tool_call);
        write_executable_fixture(&kv_tool_loop);
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "middle-compressed",
  "runbook_cwd": "{}",
  "warnings": ["runtime host alias differs from preflight host"],
  "skippy_bench_bin": "{}",
  "skippy_model_package_bin": "{}",
  "agent_tool_call_script": "{}",
  "kv_tool_loop_script": "{}",
  "focused_runtime": {{"ssh_opts": "-p 2222"}},
  "commands": [
    {{
      "id": "done",
      "description": "already complete",
      "shell": "true",
      "outputs": ["{}"]
    }},
    {{
      "id": "missing",
      "description": "not complete",
      "shell": "produce-missing",
      "outputs": ["{}"]
    }}
  ]
}}"#,
                dir.display(),
                skippy_bench.display(),
                skippy_model_package.display(),
                agent_tool_call.display(),
                kv_tool_loop.display(),
                done.display(),
                missing.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");

        assert_eq!(report.status, EvidenceStatusKind::Incomplete);
        assert_eq!(report.candidate_count, 1);
        assert_eq!(report.complete_commands, 1);
        assert_eq!(report.partial_commands, 0);
        assert_eq!(report.missing_commands, 1);
        assert_eq!(
            report.warnings,
            ["runtime host alias differs from preflight host"]
        );
        assert_eq!(
            report.candidates[0].warnings,
            ["runtime host alias differs from preflight host"]
        );
        assert_eq!(
            report.toolchain.runbook_cwd.as_deref(),
            Some(dir.to_str().expect("utf-8 path"))
        );
        assert_eq!(
            report.toolchain.skippy_bench_bin.as_deref(),
            Some(skippy_bench.to_str().expect("utf-8 path"))
        );
        assert_eq!(
            report.toolchain.skippy_model_package_bin.as_deref(),
            Some(skippy_model_package.to_str().expect("utf-8 path"))
        );
        assert_eq!(
            report.toolchain.agent_tool_call_script.as_deref(),
            Some(agent_tool_call.to_str().expect("utf-8 path"))
        );
        assert_eq!(
            report.toolchain.kv_tool_loop_script.as_deref(),
            Some(kv_tool_loop.to_str().expect("utf-8 path"))
        );
        assert_eq!(report.toolchain, report.candidates[0].toolchain);
        assert_eq!(
            report
                .toolchain
                .focused_runtime
                .as_ref()
                .and_then(|value| value.get("ssh_opts"))
                .and_then(serde_json::Value::as_str),
            Some("-p 2222")
        );
        let next = report.next_command.expect("next command");
        assert_eq!(next.candidate, "middle-compressed");
        assert_eq!(next.id, "missing");
        assert_eq!(next.shell, "produce-missing");
        assert_eq!(next.missing_outputs, [missing.display().to_string()]);
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_warns_when_toolchain_paths_are_not_runnable() {
        let dir = unique_test_dir("toolchain-warning");
        fs::create_dir_all(&dir).expect("create fixture");
        let output = dir.join("done.json");
        fs::write(&output, b"{}").expect("write output");
        let non_executable = dir.join("tools/not-executable");
        fs::create_dir_all(non_executable.parent().expect("tool parent")).expect("create tools");
        fs::write(&non_executable, b"#!/bin/sh\n").expect("write non-executable tool");
        let missing_tool = dir.join("tools/missing-skippy-bench");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "middle-compressed",
  "skippy_bench_bin": "{}",
  "agent_tool_call_script": "{}",
  "commands": [
    {{"id": "done", "description": "done", "shell": "true", "outputs": ["{}"]}}
  ]
}}"#,
                missing_tool.display(),
                non_executable.display(),
                output.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");

        assert_eq!(report.status, EvidenceStatusKind::Complete);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("toolchain skippy_bench_bin"))
        );
        #[cfg(unix)]
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("toolchain agent_tool_call_script"))
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_marks_command_partial_when_some_outputs_exist() {
        let dir = unique_test_dir("partial");
        fs::create_dir_all(&dir).expect("create fixture");
        let log = dir.join("focused-runtime-lab-preflight.txt");
        let marker = dir.join("focused-runtime-lab-preflight.ok");
        fs::write(&log, b"ssh failed").expect("write partial log");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "commands": [
    {{
      "id": "focused-runtime-lab-preflight",
      "description": "lab preflight",
      "shell": "run-preflight",
      "outputs": ["{}", "{}"]
    }}
  ]
}}"#,
                log.display(),
                marker.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert_eq!(report.partial_commands, 1);
        assert_eq!(report.missing_commands, 1);
        assert_eq!(command.status, EvidenceStatusKind::Partial);
        assert_eq!(command.missing_outputs, [marker.display().to_string()]);
        let expected_failure = format!("{}: ssh failed", log.display());
        assert_eq!(
            command.observed_failure.as_deref(),
            Some(expected_failure.as_str())
        );
        let next = report.next_command.expect("next command");
        assert_eq!(next.missing_outputs, [marker.display().to_string()]);
        assert_eq!(
            next.observed_failure.as_deref(),
            Some(expected_failure.as_str())
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_resolves_relative_outputs_from_runbook_cwd() {
        let dir = unique_test_dir("relative-output");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let log = dir.join("evidence/preflight.txt");
        fs::write(&log, b"preflight_status: ssh_failed rc=255").expect("write log");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "focused-runtime-lab-preflight",
      "description": "lab preflight",
      "shell": "run-preflight",
      "outputs": ["evidence/preflight.txt", "evidence/preflight.ok"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert!(command.outputs[0].exists);
        assert!(!command.outputs[1].exists);
        assert_eq!(command.missing_outputs, ["evidence/preflight.ok"]);
        assert_eq!(
            command.observed_failure.as_deref(),
            Some("evidence/preflight.txt: preflight_status: ssh_failed rc=255")
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_marks_failed_certification_output_partial() {
        let dir = unique_test_dir("failed-certification");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let certification = dir.join("evidence/certification.json");
        fs::write(
            &certification,
            r#"{"kind":"skippy_quant_pack_certification","status":"failed"}"#,
        )
        .expect("write failed certification");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "certify",
      "description": "certify candidate",
      "shell": "certify",
      "outputs": ["evidence/certification.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert_eq!(report.complete_commands, 0);
        assert_eq!(report.partial_commands, 1);
        assert_eq!(report.missing_commands, 1);
        assert_eq!(command.status, EvidenceStatusKind::Partial);
        assert!(command.missing_outputs.is_empty());
        assert_eq!(
            command.observed_failure.as_deref(),
            Some("evidence/certification.json: certification status failed")
        );
        let next = report.next_command.as_ref().expect("next command");
        assert_eq!(next.id, "certify");
        assert!(next.missing_outputs.is_empty());
        assert_eq!(
            next.observed_failure.as_deref(),
            Some("evidence/certification.json: certification status failed")
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_marks_chat_corpus_errors_partial() {
        let dir = unique_test_dir("chat-corpus-errors");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let chat_corpus = dir.join("evidence/chat-corpus.json");
        fs::write(&chat_corpus, r#"{"summary":{"errors":2}}"#).expect("write chat corpus");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "chat-corpus",
      "description": "chat corpus",
      "shell": "chat-corpus",
      "outputs": ["evidence/chat-corpus.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert_eq!(command.status, EvidenceStatusKind::Partial);
        assert_eq!(
            command.observed_failure.as_deref(),
            Some("evidence/chat-corpus.json: chat corpus errors 2")
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_rejects_schema_smoke_as_measured_focused_runtime() {
        let dir = unique_test_dir("focused-runtime-schema-as-measured");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let focused = dir.join("evidence/focused-runtime-report.json");
        fs::write(
            &focused,
            r#"{
  "mode": "schema-smoke",
  "latency_ms": {"decode_elapsed_ms_p50": 5},
  "throughput_tokens_per_second": {"generated": 100.0}
}"#,
        )
        .expect("write focused runtime");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "focused-runtime",
      "description": "focused runtime",
      "shell": "focused-runtime",
      "outputs": ["evidence/focused-runtime-report.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert_eq!(command.status, EvidenceStatusKind::Partial);
        assert_eq!(
            command.observed_failure.as_deref(),
            Some(
                "evidence/focused-runtime-report.json: focused-runtime mode schema-smoke != executed"
            )
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_accepts_schema_smoke_for_schema_smoke_step() {
        let dir = unique_test_dir("focused-runtime-schema-step");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let focused = dir.join("evidence/focused-runtime-schema-smoke.json");
        fs::write(
            &focused,
            r#"{
  "mode": "schema-smoke",
  "latency_ms": {"decode_elapsed_ms_p50": 5},
  "throughput_tokens_per_second": {"generated": 100.0}
}"#,
        )
        .expect("write focused runtime");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "focused-runtime-schema-smoke",
      "description": "focused runtime schema smoke",
      "shell": "focused-runtime --schema-smoke",
      "outputs": ["evidence/focused-runtime-schema-smoke.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Complete);
        assert_eq!(command.status, EvidenceStatusKind::Complete);
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_accepts_local_split_chain_payload_report() {
        let dir = unique_test_dir("local-split-chain-complete");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let local_split = dir.join("evidence/local-split-chain.json");
        fs::write(
            &local_split,
            r#"{
  "mode": "local-split-chain-binary",
  "model_identity": {"model_id": "org/qwen-coder:studio-local"},
  "predicted_token": "}",
  "stages": [
    {"index": 0, "payload_bytes": 1048576, "wire_payload_bytes": 524288},
    {"index": 1, "payload_bytes": 1048576, "wire_payload_bytes": 524288}
  ]
}"#,
        )
        .expect("write local split report");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "studio-local",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "local-split-chain",
      "description": "local split chain",
      "evidence_type": "skippy-bench-local-split-chain",
      "shell": "local-split-chain",
      "outputs": ["evidence/local-split-chain.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Complete);
        assert_eq!(command.status, EvidenceStatusKind::Complete);
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_marks_local_split_chain_without_payload_partial() {
        let dir = unique_test_dir("local-split-chain-partial");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let local_split = dir.join("evidence/local-split-chain.json");
        fs::write(
            &local_split,
            r#"{
  "mode": "local-split-chain-binary",
  "predicted_token": "}",
  "stages": [
    {"index": 0, "payload_bytes": 0, "wire_payload_bytes": 0}
  ]
}"#,
        )
        .expect("write local split report");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "studio-local",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "some-old-id",
      "description": "local split chain",
      "evidence_type": "skippy-bench-local-split-chain",
      "shell": "local-split-chain",
      "outputs": ["evidence/local-split-chain.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert_eq!(command.status, EvidenceStatusKind::Partial);
        assert_eq!(
            command.observed_failure.as_deref(),
            Some(
                "evidence/local-split-chain.json: local split payload and wire payload bytes missing"
            )
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_marks_token_lengths_context_overflow_partial() {
        let dir = unique_test_dir("token-length-overflow");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let tsv = dir.join("evidence/prompt-lengths.tsv");
        let summary = dir.join("evidence/prompt-lengths-summary.json");
        fs::write(&tsv, b"id\ttokens\n").expect("write tsv");
        fs::write(&summary, r#"{"row_count":3,"exceeds_context":1}"#).expect("write token summary");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "token-lengths",
      "description": "token lengths",
      "shell": "token-lengths",
      "outputs": ["evidence/prompt-lengths.tsv", "evidence/prompt-lengths-summary.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert_eq!(command.status, EvidenceStatusKind::Partial);
        assert_eq!(
            command.observed_failure.as_deref(),
            Some("evidence/prompt-lengths-summary.json: token length rows exceed context 1")
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn command_complete_requires_semantic_completion() {
        let dir = unique_test_dir("command-complete-semantic");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let tsv = dir.join("evidence/prompt-lengths.tsv");
        let summary = dir.join("evidence/prompt-lengths-summary.json");
        fs::write(&tsv, b"id\ttokens\n").expect("write tsv");
        fs::write(&summary, r#"{"row_count":3,"exceeds_context":1}"#).expect("write token summary");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
  "commands": [
    {{
      "id": "token-lengths",
      "description": "token lengths",
      "shell": "token-lengths",
      "outputs": ["evidence/prompt-lengths.tsv", "evidence/prompt-lengths-summary.json"]
    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");
        let report = build_evidence_status_report(&plan).expect("build status");

        let error = ensure_command_complete(&report, None, "token-lengths")
            .expect_err("overflowing token-lengths should not be complete");

        assert!(
            error
                .to_string()
                .contains("token length rows exceed context 1")
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_ignores_network_diagnostic_headings_as_failures() {
        let dir = unique_test_dir("network-diagnostics");
        fs::create_dir_all(&dir).expect("create fixture");
        let log = dir.join("focused-runtime-lab-preflight.txt");
        let marker = dir.join("focused-runtime-lab-preflight.ok");
        fs::write(
            &log,
            b"qwen lab preflight: ssh connection failed for 4 host(s)\nlocal_network_diagnostics:\nlocal_ipv4_addresses:\nroutes_to_failed_hosts:\n",
        )
        .expect("write diagnostic log");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "commands": [
    {{
      "id": "focused-runtime-lab-preflight",
      "description": "lab preflight",
      "shell": "run-preflight",
      "outputs": ["{}", "{}"]
    }}
  ]
}}"#,
                log.display(),
                marker.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        let expected_failure = format!(
            "{}: qwen lab preflight: ssh connection failed for 4 host(s)",
            log.display()
        );
        assert_eq!(
            command.observed_failure.as_deref(),
            Some(expected_failure.as_str())
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_aggregates_evidence_plan_all() {
        let dir = unique_test_dir("all");
        fs::create_dir_all(&dir).expect("create fixture");
        let candidate_a = dir.join("a.json");
        let candidate_b = dir.join("b.json");
        let final_rank = dir.join("rank-after-evidence.json");
        fs::write(&candidate_a, b"{}").expect("write candidate a");
        fs::write(&candidate_b, b"{}").expect("write candidate b");
        let plan = dir.join("evidence-plan-all.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan_all",
  "final_rank": {{
    "id": "rank-after-evidence-all",
    "description": "final rank",
    "shell": "rank-all",
    "outputs": ["{}"]
  }},
  "candidates": [
    {{
      "candidate": "a",
      "commands": [
        {{"id": "a", "description": "a", "shell": "a", "outputs": ["{}"]}}
      ]
    }},
    {{
      "candidate": "b",
      "warnings": ["uses preflight-only ssh options"],
      "commands": [
        {{"id": "b", "description": "b", "shell": "b", "outputs": ["{}"]}}
      ]
    }}
  ]
}}"#,
                final_rank.display(),
                candidate_a.display(),
                candidate_b.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");

        assert_eq!(report.status, EvidenceStatusKind::Incomplete);
        assert_eq!(report.candidate_count, 2);
        assert_eq!(report.total_commands, 3);
        assert_eq!(report.complete_commands, 2);
        assert_eq!(report.partial_commands, 0);
        assert_eq!(report.missing_commands, 1);
        assert_eq!(report.warnings, ["b: uses preflight-only ssh options"]);
        assert_eq!(
            report.candidates[1].warnings,
            ["uses preflight-only ssh options"]
        );
        let final_rank_status = report.final_rank.as_ref().expect("final rank status");
        assert_eq!(final_rank_status.id, "rank-after-evidence-all");
        assert_eq!(final_rank_status.status, EvidenceStatusKind::Incomplete);
        let next = report.next_command.as_ref().expect("next command");
        assert_eq!(next.candidate, "sweep");
        assert_eq!(next.id, "rank-after-evidence-all");
        assert_eq!(next.missing_outputs, [final_rank.display().to_string()]);
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn status_report_marks_bad_rank_output_partial() {
        let dir = unique_test_dir("bad-rank-output");
        fs::create_dir_all(dir.join("evidence")).expect("create evidence dir");
        let rank = dir.join("evidence/rank-after-evidence.json");
        fs::write(
            &rank,
            r#"{"kind":"skippy_quant_pack_rank","candidate_count":2,"candidates":[{}]}"#,
        )
        .expect("write bad rank output");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "ffn-compressed-attention-protected",
  "runbook_cwd": "{}",
	  "commands": [
	    {{
	      "id": "custom-rank-step",
	      "description": "rank after evidence",
	      "evidence_type": "skippy-quant-pack-rank",
	      "shell": "rank",
	      "outputs": ["evidence/rank-after-evidence.json"]
	    }}
  ]
}}"#,
                dir.display()
            ),
        )
        .expect("write plan");

        let report = build_evidence_status_report(&plan).expect("build status");
        let command = &report.candidates[0].commands[0];

        assert_eq!(command.id, "custom-rank-step");
        assert_eq!(
            command.evidence_type.as_deref(),
            Some("skippy-quant-pack-rank")
        );
        assert_eq!(report.status, EvidenceStatusKind::Partial);
        assert_eq!(command.status, EvidenceStatusKind::Partial);
        assert_eq!(
            command.observed_failure.as_deref(),
            Some(
                "evidence/rank-after-evidence.json: rank report candidate_count 2 != candidates length 1"
            )
        );
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    #[test]
    fn fail_on_warning_rejects_complete_plan_with_warnings() {
        let dir = unique_test_dir("warning-gate");
        fs::create_dir_all(&dir).expect("create fixture");
        let output = dir.join("done.json");
        let status = dir.join("status.json");
        fs::write(&output, b"{}").expect("write output");
        let plan = dir.join("evidence-plan.json");
        fs::write(
            &plan,
            format!(
                r#"{{
  "kind": "skippy_quant_pack_evidence_plan",
  "candidate": "middle-compressed",
  "warnings": ["runtime hosts need SSH verification"],
  "commands": [
    {{
      "id": "done",
      "description": "already complete",
      "shell": "true",
      "outputs": ["{}"]
    }}
  ]
}}"#,
                output.display()
            ),
        )
        .expect("write plan");

        let error = run_quant_pack_evidence_status(QuantPackEvidenceStatusArgs {
            plan: plan.clone(),
            out: Some(status),
            fail_on_missing: false,
            fail_on_warning: true,
            command_complete: None,
            candidate: None,
        })
        .expect_err("warning should fail");

        assert!(error.to_string().contains("has warnings"));
        fs::remove_dir_all(dir).expect("remove fixture");
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skippy-evidence-status-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn write_executable_fixture(path: &Path) {
        fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture parent");
        fs::write(path, b"#!/bin/sh\nexit 0\n").expect("write executable fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = fs::metadata(path).expect("stat fixture").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).expect("chmod fixture");
        }
    }
}
