use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const DEFAULT_TOKEN_CORPUS: &str = "target/bench-corpora/long/corpus.jsonl";
const DEFAULT_CHAT_CORPUS: &str = "target/bench-corpora/coding-loop/corpus.jsonl";
const DEFAULT_LONG_CONTEXT_CORPUS: &str = "target/bench-corpora/long-context/corpus.jsonl";
const DEFAULT_LOCAL_SPLIT_PROMPT: &str = "Write a small Rust function that parses a semver string.";
const DEFAULT_AGENT_TOOL_CALL_SCRIPT: &str = "scripts/qa-agent-tool-call-reliability.py";
const DEFAULT_KV_TOOL_LOOP_SCRIPT: &str = "scripts/qa-kv-tool-loop-stability.py";

#[derive(Debug, clap::Args)]
pub(super) struct QuantPackEvidencePlanArgs {
    run: PathBuf,
    #[arg(long)]
    hosts: String,
    #[arg(long)]
    splits: Option<String>,
    #[arg(long, default_value = "http://127.0.0.1:9337/v1")]
    base_url: String,
    #[arg(long, default_value = DEFAULT_TOKEN_CORPUS)]
    token_corpus: PathBuf,
    #[arg(long, default_value = DEFAULT_CHAT_CORPUS)]
    chat_corpus: PathBuf,
    #[arg(long, default_value = DEFAULT_LONG_CONTEXT_CORPUS)]
    long_context_corpus: PathBuf,
    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    n_gpu_layers: i32,
    #[arg(long, default_value = "f16")]
    cache_type_k: String,
    #[arg(long, default_value = "f16")]
    cache_type_v: String,
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,
    #[arg(long, default_value = "f16")]
    activation_wire_dtype: String,
    #[arg(long, default_value_t = 1)]
    attempts: u32,
    #[arg(long)]
    include_local_split_evidence: bool,
    #[arg(long, default_value = DEFAULT_LOCAL_SPLIT_PROMPT)]
    local_split_prompt: String,
    #[arg(long, default_value = "skippy-bench")]
    skippy_bench_bin: PathBuf,
    #[arg(long, default_value = "skippy-model-package")]
    skippy_model_package_bin: PathBuf,
    #[arg(long, default_value = DEFAULT_AGENT_TOOL_CALL_SCRIPT)]
    agent_tool_call_script: PathBuf,
    #[arg(long, default_value = DEFAULT_KV_TOOL_LOOP_SCRIPT)]
    kv_tool_loop_script: PathBuf,
    #[arg(long)]
    runbook_cwd: Option<PathBuf>,
    #[command(flatten)]
    focused_runtime: FocusedRuntimeEvidenceArgs,
    #[arg(long)]
    evidence_dir: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    script_out: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
pub(super) struct QuantPackEvidencePlanAllArgs {
    build_all: PathBuf,
    #[arg(long)]
    hosts: String,
    #[arg(long)]
    splits: Option<String>,
    #[arg(long = "candidate")]
    candidates: Vec<String>,
    #[arg(long)]
    top_ranked: Option<usize>,
    #[arg(long, default_value = "http://127.0.0.1:9337/v1")]
    base_url: String,
    #[arg(long, default_value = DEFAULT_TOKEN_CORPUS)]
    token_corpus: PathBuf,
    #[arg(long, default_value = DEFAULT_CHAT_CORPUS)]
    chat_corpus: PathBuf,
    #[arg(long, default_value = DEFAULT_LONG_CONTEXT_CORPUS)]
    long_context_corpus: PathBuf,
    #[arg(long)]
    ctx_size: Option<u32>,
    #[arg(long, allow_hyphen_values = true)]
    n_gpu_layers: Option<i32>,
    #[arg(long)]
    cache_type_k: Option<String>,
    #[arg(long)]
    cache_type_v: Option<String>,
    #[arg(long, default_value_t = 512)]
    max_tokens: u32,
    #[arg(long)]
    activation_wire_dtype: Option<String>,
    #[arg(long, default_value_t = 1)]
    attempts: u32,
    #[arg(long)]
    include_local_split_evidence: bool,
    #[arg(long, default_value = DEFAULT_LOCAL_SPLIT_PROMPT)]
    local_split_prompt: String,
    #[arg(long, default_value = "skippy-bench")]
    skippy_bench_bin: PathBuf,
    #[arg(long, default_value = "skippy-model-package")]
    skippy_model_package_bin: PathBuf,
    #[arg(long, default_value = DEFAULT_AGENT_TOOL_CALL_SCRIPT)]
    agent_tool_call_script: PathBuf,
    #[arg(long, default_value = DEFAULT_KV_TOOL_LOOP_SCRIPT)]
    kv_tool_loop_script: PathBuf,
    #[arg(long)]
    runbook_cwd: Option<PathBuf>,
    #[command(flatten)]
    focused_runtime: FocusedRuntimeEvidenceArgs,
    #[arg(long)]
    evidence_root: Option<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    script_out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct EvidencePlanReport {
    schema_version: u32,
    kind: String,
    runbook_cwd: String,
    run_dir: String,
    candidate: String,
    model_id: String,
    package: String,
    quantized_model: String,
    evidence_dir: String,
    hosts: Vec<String>,
    splits: String,
    split_source: String,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: String,
    cache_type_v: String,
    max_tokens: u32,
    long_context_corpus: String,
    include_local_split_evidence: bool,
    local_split_prompt: String,
    skippy_bench_bin: String,
    skippy_model_package_bin: String,
    agent_tool_call_script: String,
    kv_tool_loop_script: String,
    activation_wire_dtype: String,
    #[serde(skip_serializing_if = "FocusedRuntimeEvidenceArgs::is_default")]
    focused_runtime: FocusedRuntimeEvidenceArgs,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
    commands: Vec<EvidenceCommand>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, clap::Args)]
struct FocusedRuntimeEvidenceArgs {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    metrics_server_bin: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    stage_server_bin: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    lab_preflight_script: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    lab_preflight_hosts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    lab_preflight_min_free_gb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    lab_preflight_ports: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    lab_preflight_ssh_opts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    work_dir: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    remote_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    remote_root_map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    remote_shared_root_map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    endpoint_host_map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    ssh_opts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    metrics_otlp_grpc_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    remote_bind_host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    first_stage_port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    startup_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    stage_max_inflight: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    stage_reply_credit_limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    stage_downstream_wire_delay_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    stage_downstream_wire_mbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    stage_telemetry_queue_capacity: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[arg(long)]
    stage_telemetry_level: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    rsync_model_artifacts: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    keep_remote: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    child_logs: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    stage_async_prefill_forward: bool,
    #[serde(skip_serializing_if = "is_false")]
    #[arg(long)]
    allow_uneven_stage_ranges: bool,
}

impl FocusedRuntimeEvidenceArgs {
    fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Serialize)]
struct EvidencePlanAllReport {
    schema_version: u32,
    kind: String,
    runbook_cwd: String,
    build_all: String,
    evidence_root: String,
    selection_source: String,
    final_rank: EvidenceCommand,
    candidates: Vec<EvidencePlanReport>,
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceCommand {
    id: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence_type: Option<String>,
    argv: Vec<String>,
    shell: String,
    outputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BuildManifestInput {
    candidate: String,
    stages: usize,
    #[serde(default)]
    plan: Option<String>,
    package: String,
    quantized_model: String,
}

#[derive(Debug, Deserialize)]
struct PackageManifestInput {
    model_id: String,
    layer_count: u32,
}

#[derive(Debug, Deserialize)]
struct BuildAllManifestInput {
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: String,
    cache_type_v: String,
    activation_wire_dtype: String,
    rank: String,
    candidates: Vec<BuildAllCandidateInput>,
}

#[derive(Debug, Deserialize)]
struct BuildAllCandidateInput {
    candidate: String,
    run_dir: String,
}

#[derive(Debug, Deserialize)]
struct RankReportInput {
    candidates: Vec<RankedCandidateInput>,
}

#[derive(Debug, Deserialize)]
struct RankedCandidateInput {
    candidate: String,
    valid: bool,
}

#[derive(Debug, Deserialize)]
struct QuantPlanInput {
    candidates: Vec<QuantPlanCandidateInput>,
}

#[derive(Debug, Deserialize)]
struct QuantPlanCandidateInput {
    id: String,
    #[serde(default)]
    stage_hints: Vec<QuantPlanStageHintInput>,
}

#[derive(Debug, Clone, Deserialize)]
struct QuantPlanStageHintInput {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
}

pub(super) fn run_quant_pack_evidence_plan(args: QuantPackEvidencePlanArgs) -> Result<()> {
    let runbook_cwd = resolve_runbook_cwd(args.runbook_cwd.as_deref())?;
    let report = build_evidence_plan(EvidencePlanBuildInput {
        run: args.run,
        runbook_cwd: runbook_cwd.clone(),
        hosts: args.hosts,
        splits: args.splits,
        base_url: args.base_url,
        token_corpus: args.token_corpus,
        chat_corpus: args.chat_corpus,
        long_context_corpus: args.long_context_corpus,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        max_tokens: args.max_tokens,
        include_local_split_evidence: args.include_local_split_evidence,
        local_split_prompt: args.local_split_prompt,
        skippy_bench_bin: args.skippy_bench_bin.clone(),
        skippy_model_package_bin: args.skippy_model_package_bin.clone(),
        agent_tool_call_script: args.agent_tool_call_script,
        kv_tool_loop_script: args.kv_tool_loop_script,
        activation_wire_dtype: args.activation_wire_dtype,
        attempts: args.attempts,
        focused_runtime: args.focused_runtime,
        evidence_dir: args.evidence_dir,
    })?;
    if let Some(script_out) = args.script_out.as_deref() {
        write_single_evidence_script(
            script_out,
            args.out.as_deref(),
            &args.skippy_model_package_bin,
            &report,
        )?;
    }
    write_report(args.out.as_deref(), &report, "quant-pack evidence plan")
}

pub(super) fn run_quant_pack_evidence_plan_all(args: QuantPackEvidencePlanAllArgs) -> Result<()> {
    let runbook_cwd = resolve_runbook_cwd(args.runbook_cwd.as_deref())?;
    let manifest_path = build_all_manifest_path(&args.build_all);
    let manifest = read_json::<BuildAllManifestInput>(&manifest_path)?;
    let build_all_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let evidence_root = args
        .evidence_root
        .clone()
        .unwrap_or_else(|| build_all_dir.join("evidence"));
    let selection = select_evidence_candidates(&manifest, &build_all_dir, &args)?;
    let mut candidates = Vec::with_capacity(selection.candidates.len());
    for candidate in &selection.candidates {
        let run_dir = resolve_manifest_path(&build_all_dir, &candidate.run_dir);
        let evidence_dir = evidence_root.join(&candidate.candidate);
        candidates.push(build_evidence_plan(EvidencePlanBuildInput {
            run: run_dir,
            runbook_cwd: runbook_cwd.clone(),
            hosts: args.hosts.clone(),
            splits: args.splits.clone(),
            base_url: args.base_url.clone(),
            token_corpus: args.token_corpus.clone(),
            chat_corpus: args.chat_corpus.clone(),
            long_context_corpus: args.long_context_corpus.clone(),
            ctx_size: args.ctx_size.unwrap_or(manifest.ctx_size),
            n_gpu_layers: args.n_gpu_layers.unwrap_or(manifest.n_gpu_layers),
            cache_type_k: args
                .cache_type_k
                .clone()
                .unwrap_or_else(|| manifest.cache_type_k.clone()),
            cache_type_v: args
                .cache_type_v
                .clone()
                .unwrap_or_else(|| manifest.cache_type_v.clone()),
            max_tokens: args.max_tokens,
            include_local_split_evidence: args.include_local_split_evidence,
            local_split_prompt: args.local_split_prompt.clone(),
            skippy_bench_bin: args.skippy_bench_bin.clone(),
            skippy_model_package_bin: args.skippy_model_package_bin.clone(),
            agent_tool_call_script: args.agent_tool_call_script.clone(),
            kv_tool_loop_script: args.kv_tool_loop_script.clone(),
            activation_wire_dtype: args
                .activation_wire_dtype
                .clone()
                .unwrap_or_else(|| manifest.activation_wire_dtype.clone()),
            attempts: args.attempts,
            focused_runtime: args.focused_runtime.clone(),
            evidence_dir: Some(evidence_dir),
        })?);
    }
    let final_rank =
        final_rank_command(&candidates, &evidence_root, &args.skippy_model_package_bin)?;
    let report = EvidencePlanAllReport {
        schema_version: 1,
        kind: "skippy_quant_pack_evidence_plan_all".to_string(),
        runbook_cwd: runbook_cwd.display().to_string(),
        build_all: manifest_path.display().to_string(),
        evidence_root: evidence_root.display().to_string(),
        selection_source: selection.source,
        final_rank,
        candidates,
    };
    if let Some(script_out) = args.script_out.as_deref() {
        write_all_evidence_script(
            script_out,
            args.out.as_deref(),
            &args.skippy_model_package_bin,
            &report,
        )?;
    }
    write_report(
        args.out.as_deref(),
        &report,
        "quant-pack evidence-plan-all report",
    )
}

struct CandidateSelection<'a> {
    source: String,
    candidates: Vec<&'a BuildAllCandidateInput>,
}

struct EvidencePlanBuildInput {
    run: PathBuf,
    runbook_cwd: PathBuf,
    hosts: String,
    splits: Option<String>,
    base_url: String,
    token_corpus: PathBuf,
    chat_corpus: PathBuf,
    long_context_corpus: PathBuf,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: String,
    cache_type_v: String,
    max_tokens: u32,
    include_local_split_evidence: bool,
    local_split_prompt: String,
    skippy_bench_bin: PathBuf,
    skippy_model_package_bin: PathBuf,
    agent_tool_call_script: PathBuf,
    kv_tool_loop_script: PathBuf,
    activation_wire_dtype: String,
    attempts: u32,
    focused_runtime: FocusedRuntimeEvidenceArgs,
    evidence_dir: Option<PathBuf>,
}

fn build_evidence_plan(args: EvidencePlanBuildInput) -> Result<EvidencePlanReport> {
    let manifest_path = build_manifest_path(&args.run);
    let manifest = read_json::<BuildManifestInput>(&manifest_path)?;
    let run_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let package = resolve_manifest_path(&run_dir, &manifest.package);
    let quantized_model = resolve_manifest_path(&run_dir, &manifest.quantized_model);
    let package_manifest = read_json::<PackageManifestInput>(&package.join("model-package.json"))?;
    let hosts = parse_hosts(&args.hosts)?;
    if hosts.len() != manifest.stages {
        bail!(
            "--hosts must contain exactly {} hosts for this {}-stage quant-pack run, got {}",
            manifest.stages,
            manifest.stages,
            hosts.len()
        );
    }
    let split_plan = split_plan(
        args.splits.as_deref(),
        package_manifest.layer_count,
        manifest.stages,
        PlanSplitRequest {
            run_dir: &run_dir,
            manifest: &manifest,
        },
    )?;
    let evidence_dir = args
        .evidence_dir
        .unwrap_or_else(|| run_dir.join("evidence"));
    let allow_uneven_stage_ranges = split_plan.source.allows_uneven_stage_ranges();
    if args.include_local_split_evidence && parse_split_boundaries(&split_plan.splits)?.len() < 2 {
        bail!(
            "--include-local-split-evidence requires at least two split boundaries for a local split chain"
        );
    }
    let warnings = evidence_plan_warnings(&hosts, &args.focused_runtime);
    Ok(EvidencePlanReport {
        schema_version: 1,
        kind: "skippy_quant_pack_evidence_plan".to_string(),
        runbook_cwd: args.runbook_cwd.display().to_string(),
        run_dir: run_dir.display().to_string(),
        candidate: manifest.candidate,
        model_id: package_manifest.model_id.clone(),
        package: package.display().to_string(),
        quantized_model: quantized_model.display().to_string(),
        evidence_dir: evidence_dir.display().to_string(),
        hosts,
        splits: split_plan.splits.clone(),
        split_source: split_plan.source.as_str().to_string(),
        layer_end: package_manifest.layer_count,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        cache_type_k: args.cache_type_k.clone(),
        cache_type_v: args.cache_type_v.clone(),
        max_tokens: args.max_tokens,
        long_context_corpus: args.long_context_corpus.display().to_string(),
        include_local_split_evidence: args.include_local_split_evidence,
        local_split_prompt: args.local_split_prompt.clone(),
        skippy_bench_bin: args.skippy_bench_bin.display().to_string(),
        skippy_model_package_bin: args.skippy_model_package_bin.display().to_string(),
        agent_tool_call_script: args.agent_tool_call_script.display().to_string(),
        kv_tool_loop_script: args.kv_tool_loop_script.display().to_string(),
        activation_wire_dtype: args.activation_wire_dtype.clone(),
        focused_runtime: args.focused_runtime.clone(),
        warnings,
        commands: evidence_commands(EvidenceCommandInputs {
            run: &run_dir,
            model_id: &package_manifest.model_id,
            package: &package,
            quantized_model: &quantized_model,
            evidence_dir: &evidence_dir,
            hosts: &args.hosts,
            splits: &split_plan.splits,
            layer_end: package_manifest.layer_count,
            base_url: &args.base_url,
            token_corpus: &args.token_corpus,
            chat_corpus: &args.chat_corpus,
            long_context_corpus: &args.long_context_corpus,
            ctx_size: args.ctx_size,
            n_gpu_layers: args.n_gpu_layers,
            cache_type_k: &args.cache_type_k,
            cache_type_v: &args.cache_type_v,
            max_tokens: args.max_tokens,
            include_local_split_evidence: args.include_local_split_evidence,
            local_split_prompt: &args.local_split_prompt,
            skippy_bench_bin: &args.skippy_bench_bin,
            skippy_model_package_bin: &args.skippy_model_package_bin,
            agent_tool_call_script: &args.agent_tool_call_script,
            kv_tool_loop_script: &args.kv_tool_loop_script,
            activation_wire_dtype: &args.activation_wire_dtype,
            attempts: args.attempts,
            focused_runtime: &args.focused_runtime,
            allow_uneven_stage_ranges,
        }),
    })
}

fn evidence_plan_warnings(
    hosts: &[String],
    focused_runtime: &FocusedRuntimeEvidenceArgs,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let Some(lab_preflight_hosts) = focused_runtime.lab_preflight_hosts.as_deref() else {
        append_ssh_option_warning(&mut warnings, focused_runtime);
        return warnings;
    };
    let preflight_hosts = split_host_list(lab_preflight_hosts);
    if preflight_hosts.is_empty() || preflight_hosts == hosts {
        append_ssh_option_warning(&mut warnings, focused_runtime);
        return warnings;
    }
    warnings.push(format!(
        "lab_preflight_hosts ({}) differ from focused-runtime --hosts ({}). skippy-bench uses --hosts as SSH targets for remote stages; ensure the runtime hosts resolve over SSH, or pass SSH-able hosts to --hosts and use --endpoint-host-map for stage fabric addresses.",
        preflight_hosts.join(","),
        hosts.join(",")
    ));
    append_ssh_option_warning(&mut warnings, focused_runtime);
    warnings
}

fn append_ssh_option_warning(
    warnings: &mut Vec<String>,
    focused_runtime: &FocusedRuntimeEvidenceArgs,
) {
    if focused_runtime.lab_preflight_ssh_opts.is_some() && focused_runtime.ssh_opts.is_none() {
        warnings.push(
            "lab_preflight_ssh_opts are set but focused-runtime ssh_opts are not. Preflight SSH may succeed while skippy-bench remote launch still uses default SSH options.".to_string(),
        );
    }
}

fn split_host_list(hosts: &str) -> Vec<String> {
    hosts
        .split(',')
        .map(str::trim)
        .filter(|host| !host.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn write_report(out: Option<&Path>, report: &impl Serialize, label: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out) = out {
        fs::write(out, format!("{json}\n"))
            .with_context(|| format!("write {label} {}", out.display()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn resolve_runbook_cwd(runbook_cwd: Option<&Path>) -> Result<PathBuf> {
    let cwd = env::current_dir().context("read current directory for evidence runbook")?;
    let resolved = match runbook_cwd {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => cwd.join(path),
        None => cwd,
    };
    if !resolved.is_dir() {
        bail!(
            "--runbook-cwd {} must be an existing directory",
            resolved.display()
        );
    }
    Ok(resolved)
}

fn write_single_evidence_script(
    path: &Path,
    plan_path: Option<&Path>,
    skippy_model_package_bin: &Path,
    report: &EvidencePlanReport,
) -> Result<()> {
    let mut script = evidence_script_header(
        plan_path,
        skippy_model_package_bin,
        Path::new(&report.runbook_cwd),
    )?;
    append_candidate_script_section(
        &mut script,
        plan_path,
        skippy_model_package_bin,
        &report.candidate,
        &report.warnings,
        &report.commands,
    )?;
    write_script_file(path, &script)
}

fn write_all_evidence_script(
    path: &Path,
    plan_path: Option<&Path>,
    skippy_model_package_bin: &Path,
    report: &EvidencePlanAllReport,
) -> Result<()> {
    let mut script = evidence_script_header(
        plan_path,
        skippy_model_package_bin,
        Path::new(&report.runbook_cwd),
    )?;
    let shared_commands = shared_script_commands(&report.candidates);
    if !shared_commands.is_empty() {
        let first_candidate = report
            .candidates
            .first()
            .map(|candidate| candidate.candidate.as_str());
        append_script_section(
            &mut script,
            "shared setup",
            &shared_commands,
            semantic_skip_check(plan_path, skippy_model_package_bin, first_candidate),
        )?;
    }
    for candidate in &report.candidates {
        let commands = candidate
            .commands
            .iter()
            .filter(|command| {
                !shared_commands
                    .iter()
                    .any(|shared| shared.id == command.id && shared.shell == command.shell)
            })
            .cloned()
            .collect::<Vec<_>>();
        append_candidate_script_section(
            &mut script,
            plan_path,
            skippy_model_package_bin,
            &candidate.candidate,
            &candidate.warnings,
            &commands,
        )?;
    }
    append_script_section(
        &mut script,
        "skippy quant-pack sweep rank",
        std::slice::from_ref(&report.final_rank),
        None,
    )?;
    write_script_file(path, &script)
}

fn shared_script_commands(candidates: &[EvidencePlanReport]) -> Vec<EvidenceCommand> {
    let Some(first) = candidates.first() else {
        return Vec::new();
    };
    first
        .commands
        .iter()
        .filter(|command| is_shared_script_command(command))
        .filter(|command| {
            candidates.iter().all(|candidate| {
                candidate
                    .commands
                    .iter()
                    .any(|other| other.id == command.id && other.shell == command.shell)
            })
        })
        .cloned()
        .collect()
}

fn is_shared_script_command(command: &EvidenceCommand) -> bool {
    command.id.starts_with("prepare-corpus-")
}

fn evidence_script_header(
    plan_path: Option<&Path>,
    skippy_model_package_bin: &Path,
    runbook_cwd: &Path,
) -> Result<String> {
    let mut script = "#!/usr/bin/env bash\nset -euo pipefail\n\n".to_string();
    writeln!(
        script,
        "cd {}\n",
        shell_quote(&runbook_cwd.display().to_string())
    )?;
    if let Some(plan_path) = plan_path {
        writeln!(
            script,
            "# Refuse suspicious host/SSH plans before spending lab time."
        )?;
        writeln!(
            script,
            "{} quant-pack evidence-status {} --fail-on-warning > /dev/null\n",
            shell_quote(&skippy_model_package_bin.display().to_string()),
            shell_quote(&plan_path.display().to_string())
        )?;
    }
    Ok(script)
}

fn append_candidate_script_section(
    script: &mut String,
    plan_path: Option<&Path>,
    skippy_model_package_bin: &Path,
    candidate: &str,
    warnings: &[String],
    commands: &[EvidenceCommand],
) -> Result<()> {
    if !warnings.is_empty() {
        append_script_warnings(script, warnings)?;
    }
    append_script_section(
        script,
        &format!("skippy quant-pack evidence: {candidate}"),
        commands,
        semantic_skip_check(plan_path, skippy_model_package_bin, Some(candidate)),
    )
}

fn append_script_warnings(script: &mut String, warnings: &[String]) -> Result<()> {
    for warning in warnings {
        writeln!(
            script,
            "printf '%s\\n' {} >&2",
            shell_quote(&format!("warning: {warning}"))
        )?;
    }
    script.push('\n');
    Ok(())
}

fn append_script_section(
    script: &mut String,
    title: &str,
    commands: &[EvidenceCommand],
    skip_check: Option<SemanticSkipCheck<'_>>,
) -> Result<()> {
    writeln!(
        script,
        "printf '%s\\n' {}",
        shell_quote(&format!("== {title} =="))
    )?;
    for command in commands {
        writeln!(script, "\n# {}", command.id)?;
        writeln!(script, "# {}", command.description)?;
        append_skip_completed_guard(script, command, skip_check.as_ref())?;
        writeln!(script, "{}", command.shell)?;
        append_output_checks(script, &command.outputs)?;
        writeln!(script, "fi")?;
    }
    script.push('\n');
    Ok(())
}

#[derive(Clone, Copy)]
struct SemanticSkipCheck<'a> {
    plan_path: &'a Path,
    skippy_model_package_bin: &'a Path,
    candidate: Option<&'a str>,
}

fn semantic_skip_check<'a>(
    plan_path: Option<&'a Path>,
    skippy_model_package_bin: &'a Path,
    candidate: Option<&'a str>,
) -> Option<SemanticSkipCheck<'a>> {
    plan_path.map(|plan_path| SemanticSkipCheck {
        plan_path,
        skippy_model_package_bin,
        candidate,
    })
}

fn append_skip_completed_guard(
    script: &mut String,
    command: &EvidenceCommand,
    skip_check: Option<&SemanticSkipCheck<'_>>,
) -> Result<()> {
    let completion_test = command_completion_test(command, skip_check);
    writeln!(script, "if {}; then", completion_test)?;
    writeln!(
        script,
        "  printf '%s\\n' {}",
        shell_quote(&format!(
            "skip evidence command: {} (outputs already complete)",
            command.id
        ))
    )?;
    writeln!(script, "else")?;
    Ok(())
}

fn command_completion_test(
    command: &EvidenceCommand,
    skip_check: Option<&SemanticSkipCheck<'_>>,
) -> String {
    let outputs_test = all_outputs_exist_test(&command.outputs);
    let Some(skip_check) = skip_check else {
        return outputs_test;
    };
    if outputs_test == "false" {
        return outputs_test;
    }
    format!(
        "{outputs_test} && {} quant-pack evidence-status {} --command-complete {}{} > /dev/null 2>&1",
        shell_quote(&skip_check.skippy_model_package_bin.display().to_string()),
        shell_quote(&skip_check.plan_path.display().to_string()),
        shell_quote(&command.id),
        skip_check
            .candidate
            .map(|candidate| format!(" --candidate {}", shell_quote(candidate)))
            .unwrap_or_default()
    )
}

fn append_output_checks(script: &mut String, outputs: &[String]) -> Result<()> {
    for output in outputs {
        let quoted = shell_quote(output);
        writeln!(
            script,
            "test -e {quoted} || {{ echo {} >&2; exit 1; }}",
            shell_quote(&format!("missing expected evidence output: {output}"))
        )?;
    }
    Ok(())
}

fn all_outputs_exist_test(outputs: &[String]) -> String {
    if outputs.is_empty() {
        return "false".to_string();
    }
    outputs
        .iter()
        .map(|output| format!("test -e {}", shell_quote(output)))
        .collect::<Vec<_>>()
        .join(" && ")
}

fn write_script_file(path: &Path, script: &str) -> Result<()> {
    fs::write(path, script).with_context(|| format!("write evidence script {}", path.display()))?;
    make_executable(path)
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path)
            .with_context(|| format!("stat evidence script {}", path.display()))?
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("chmod evidence script {}", path.display()))?;
    }
    Ok(())
}

struct EvidenceCommandInputs<'a> {
    run: &'a Path,
    model_id: &'a str,
    package: &'a Path,
    quantized_model: &'a Path,
    evidence_dir: &'a Path,
    hosts: &'a str,
    splits: &'a str,
    layer_end: u32,
    base_url: &'a str,
    token_corpus: &'a Path,
    chat_corpus: &'a Path,
    long_context_corpus: &'a Path,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: &'a str,
    cache_type_v: &'a str,
    max_tokens: u32,
    include_local_split_evidence: bool,
    local_split_prompt: &'a str,
    skippy_bench_bin: &'a Path,
    skippy_model_package_bin: &'a Path,
    agent_tool_call_script: &'a Path,
    kv_tool_loop_script: &'a Path,
    activation_wire_dtype: &'a str,
    attempts: u32,
    focused_runtime: &'a FocusedRuntimeEvidenceArgs,
    allow_uneven_stage_ranges: bool,
}

fn evidence_commands(inputs: EvidenceCommandInputs<'_>) -> Vec<EvidenceCommand> {
    let prompt_lengths_tsv = inputs.evidence_dir.join("prompt-lengths.tsv");
    let prompt_lengths_summary = inputs.evidence_dir.join("prompt-lengths-summary.json");
    let focused_runtime_schema_smoke = inputs
        .evidence_dir
        .join("focused-runtime-schema-smoke.json");
    let focused_runtime_lab_preflight_log = inputs
        .evidence_dir
        .join("focused-runtime-lab-preflight.txt");
    let focused_runtime_lab_preflight_marker =
        inputs.evidence_dir.join("focused-runtime-lab-preflight.ok");
    let focused_runtime = inputs.evidence_dir.join("focused-runtime-report.json");
    let local_split_chain = inputs.evidence_dir.join("local-split-chain.json");
    let chat_corpus = inputs.evidence_dir.join("chat-corpus.json");
    let long_context_chat_corpus = inputs.evidence_dir.join("long-context-chat-corpus.json");
    let tool_results = inputs
        .evidence_dir
        .join("agent-tool-call-reliability/results.jsonl");
    let kv_dir = inputs.evidence_dir.join("kv-tool-loop-stability");
    let kv_summary = kv_dir.join("summary.json");
    let certification = inputs.evidence_dir.join("certification.json");
    let rank_after_evidence = inputs.evidence_dir.join("rank-after-evidence.json");

    let mut commands = vec![command(
        "prepare-evidence-dir",
        "Create the evidence output directory tree.",
        Some("runbook-prep"),
        vec![
            "mkdir".to_string(),
            "-p".to_string(),
            inputs.evidence_dir.display().to_string(),
        ],
        vec![inputs.evidence_dir.display().to_string()],
    )];
    commands.extend(corpus_prep_commands(&[
        inputs.token_corpus,
        inputs.chat_corpus,
        inputs.long_context_corpus,
    ]));
    commands.push(command(
        "token-lengths",
        "Audit corpus context fit with the target tokenizer and chat template.",
        Some("skippy-bench-token-lengths"),
        vec![
            inputs.skippy_bench_bin.display().to_string(),
            "token-lengths".to_string(),
            "--model-path".to_string(),
            inputs.quantized_model.display().to_string(),
            "--prompt-corpus".to_string(),
            inputs.token_corpus.display().to_string(),
            "--ctx-size".to_string(),
            inputs.ctx_size.to_string(),
            "--generation-limit".to_string(),
            inputs.max_tokens.to_string(),
            "--layer-end".to_string(),
            inputs.layer_end.to_string(),
            "--enable-thinking".to_string(),
            "false".to_string(),
            "--output-tsv".to_string(),
            prompt_lengths_tsv.display().to_string(),
            "--summary-json".to_string(),
            prompt_lengths_summary.display().to_string(),
        ],
        vec![
            prompt_lengths_tsv.display().to_string(),
            prompt_lengths_summary.display().to_string(),
        ],
    ));
    commands.push(focused_runtime_schema_smoke_command(
        &inputs,
        &focused_runtime_schema_smoke,
    ));
    if inputs.include_local_split_evidence {
        commands.push(local_split_chain_command(&inputs, &local_split_chain));
    }
    if let Some(command) = focused_runtime_lab_preflight_command(
        &inputs,
        &focused_runtime_lab_preflight_log,
        &focused_runtime_lab_preflight_marker,
    ) {
        commands.push(command);
    }
    commands.push(focused_runtime_command(&inputs, &focused_runtime));
    commands.push(command(
        "chat-corpus",
        "Drive OpenAI-compatible chat completions through the staged model.",
        Some("skippy-bench-chat-corpus"),
        vec![
            inputs.skippy_bench_bin.display().to_string(),
            "chat-corpus".to_string(),
            "--base-url".to_string(),
            inputs.base_url.to_string(),
            "--model".to_string(),
            inputs.model_id.to_string(),
            "--prompt-corpus".to_string(),
            inputs.chat_corpus.display().to_string(),
            "--max-tokens".to_string(),
            inputs.max_tokens.to_string(),
            "--concurrency-depth".to_string(),
            "1".to_string(),
            "--stream".to_string(),
            "--include-usage".to_string(),
            "true".to_string(),
            "--enable-thinking".to_string(),
            "false".to_string(),
            "--output".to_string(),
            chat_corpus.display().to_string(),
        ],
        vec![chat_corpus.display().to_string()],
    ));
    commands.push(command(
        "long-context-chat-corpus",
        "Drive long-context chat completions through the staged model.",
        Some("skippy-bench-long-context-chat-corpus"),
        vec![
            inputs.skippy_bench_bin.display().to_string(),
            "chat-corpus".to_string(),
            "--base-url".to_string(),
            inputs.base_url.to_string(),
            "--model".to_string(),
            inputs.model_id.to_string(),
            "--prompt-corpus".to_string(),
            inputs.long_context_corpus.display().to_string(),
            "--max-tokens".to_string(),
            inputs.max_tokens.to_string(),
            "--concurrency-depth".to_string(),
            "1".to_string(),
            "--stream".to_string(),
            "--include-usage".to_string(),
            "true".to_string(),
            "--enable-thinking".to_string(),
            "false".to_string(),
            "--output".to_string(),
            long_context_chat_corpus.display().to_string(),
        ],
        vec![long_context_chat_corpus.display().to_string()],
    ));
    commands.push(command(
        "agent-tool-call-reliability",
        "Probe OpenAI tool-call and streamed tool-call behavior.",
        Some("agent-tool-call-reliability"),
        vec![
            inputs.agent_tool_call_script.display().to_string(),
            "--base-url".to_string(),
            inputs.base_url.to_string(),
            "--models".to_string(),
            inputs.model_id.to_string(),
            "--attempts".to_string(),
            inputs.attempts.to_string(),
            "--output".to_string(),
            tool_results.display().to_string(),
        ],
        vec![tool_results.display().to_string()],
    ));
    commands.push(command(
        "kv-tool-loop-stability",
        "Probe repeated tool-loop cache stability and suffix prefill behavior.",
        Some("kv-tool-loop-stability"),
        vec![
            inputs.kv_tool_loop_script.display().to_string(),
            "--base-url".to_string(),
            inputs.base_url.to_string(),
            "--models".to_string(),
            inputs.model_id.to_string(),
            "--attempts".to_string(),
            inputs.attempts.to_string(),
            "--output-dir".to_string(),
            kv_dir.display().to_string(),
        ],
        vec![kv_summary.display().to_string()],
    ));
    let mut certify_argv = vec![
        inputs.skippy_model_package_bin.display().to_string(),
        "quant-pack".to_string(),
        "certify".to_string(),
        inputs.run.display().to_string(),
        "--skippy-bench-report".to_string(),
        focused_runtime.display().to_string(),
        "--skippy-bench-report".to_string(),
        chat_corpus.display().to_string(),
        "--skippy-bench-report".to_string(),
        long_context_chat_corpus.display().to_string(),
        "--skippy-bench-report".to_string(),
        prompt_lengths_summary.display().to_string(),
    ];
    if inputs.include_local_split_evidence {
        certify_argv.extend([
            "--skippy-bench-report".to_string(),
            local_split_chain.display().to_string(),
        ]);
    }
    certify_argv.extend([
        "--quality-evidence".to_string(),
        tool_results.display().to_string(),
        "--quality-evidence".to_string(),
        kv_summary.display().to_string(),
        "--require-skippy-bench".to_string(),
        "--require-quality-evidence".to_string(),
        "--activation-wire-dtype".to_string(),
        inputs.activation_wire_dtype.to_string(),
        "--ctx-size".to_string(),
        inputs.ctx_size.to_string(),
    ]);
    push_i32_option(&mut certify_argv, "--n-gpu-layers", inputs.n_gpu_layers);
    certify_argv.extend([
        "--cache-type-k".to_string(),
        inputs.cache_type_k.to_string(),
        "--cache-type-v".to_string(),
        inputs.cache_type_v.to_string(),
        "--out".to_string(),
        certification.display().to_string(),
    ]);
    commands.push(command(
        "certify",
        "Bind package, profiler, skippy-bench, and quality evidence to the quant-pack run.",
        Some("skippy-quant-pack-certification"),
        certify_argv,
        vec![certification.display().to_string()],
    ));
    let mut rank_argv = vec![
        inputs.skippy_model_package_bin.display().to_string(),
        "quant-pack".to_string(),
        "rank".to_string(),
        inputs.run.display().to_string(),
        "--activation-wire-dtype".to_string(),
        inputs.activation_wire_dtype.to_string(),
        "--ctx-size".to_string(),
        inputs.ctx_size.to_string(),
    ];
    push_i32_option(&mut rank_argv, "--n-gpu-layers", inputs.n_gpu_layers);
    rank_argv.extend([
        "--cache-type-k".to_string(),
        inputs.cache_type_k.to_string(),
        "--cache-type-v".to_string(),
        inputs.cache_type_v.to_string(),
        "--out".to_string(),
        rank_after_evidence.display().to_string(),
    ]);
    commands.push(command(
        "rank-after-evidence",
        "Rerank the candidate after skippy-bench and certification evidence are present.",
        Some("skippy-quant-pack-rank"),
        rank_argv,
        vec![rank_after_evidence.display().to_string()],
    ));
    commands
}

fn focused_runtime_schema_smoke_command(
    inputs: &EvidenceCommandInputs<'_>,
    focused_runtime_schema_smoke: &Path,
) -> EvidenceCommand {
    let mut argv = focused_runtime_base_argv(inputs);
    argv.push("--schema-smoke".to_string());
    append_focused_runtime_options(
        &mut argv,
        inputs.focused_runtime,
        inputs.allow_uneven_stage_ranges,
        FocusedRuntimeOptionMode::SchemaSmoke,
    );
    argv.extend([
        "--focused-output".to_string(),
        focused_runtime_schema_smoke.display().to_string(),
    ]);
    command(
        "focused-runtime-schema-smoke",
        "Validate focused-runtime command shape and topology locally before launching remote stages.",
        Some("skippy-bench-focused-runtime-schema-smoke"),
        argv,
        vec![focused_runtime_schema_smoke.display().to_string()],
    )
}

fn local_split_chain_command(
    inputs: &EvidenceCommandInputs<'_>,
    local_split_chain: &Path,
) -> EvidenceCommand {
    let mut argv = vec![
        inputs.skippy_bench_bin.display().to_string(),
        "local-split-chain-binary".to_string(),
        "--model-path".to_string(),
        inputs.quantized_model.display().to_string(),
        "--model-id".to_string(),
        inputs.model_id.to_string(),
        "--splits".to_string(),
        inputs.splits.to_string(),
        "--layer-end".to_string(),
        inputs.layer_end.to_string(),
        "--ctx-size".to_string(),
        inputs.ctx_size.to_string(),
    ];
    push_i32_option(&mut argv, "--n-gpu-layers", inputs.n_gpu_layers);
    argv.extend([
        "--activation-wire-dtype".to_string(),
        inputs.activation_wire_dtype.to_string(),
        "--prompt".to_string(),
        inputs.local_split_prompt.to_string(),
        "--output".to_string(),
        local_split_chain.display().to_string(),
    ]);
    push_path_option(
        &mut argv,
        "--stage-server-bin",
        inputs.focused_runtime.stage_server_bin.as_deref(),
    );
    push_display_option(
        &mut argv,
        "--startup-timeout-secs",
        inputs.focused_runtime.startup_timeout_secs,
    );
    command(
        "local-split-chain",
        "Measure a local Skippy split chain for Studio-scale staged proof.",
        Some("skippy-bench-local-split-chain"),
        argv,
        vec![local_split_chain.display().to_string()],
    )
}

fn focused_runtime_lab_preflight_command(
    inputs: &EvidenceCommandInputs<'_>,
    focused_runtime_lab_preflight_log: &Path,
    focused_runtime_lab_preflight_marker: &Path,
) -> Option<EvidenceCommand> {
    let script = inputs.focused_runtime.lab_preflight_script.as_ref()?;
    let mut preflight_argv = vec![
        script.display().to_string(),
        "--hosts".to_string(),
        inputs
            .focused_runtime
            .lab_preflight_hosts
            .clone()
            .unwrap_or_else(|| inputs.hosts.to_string()),
    ];
    push_display_option(
        &mut preflight_argv,
        "--min-free-gb",
        inputs.focused_runtime.lab_preflight_min_free_gb,
    );
    push_string_option(
        &mut preflight_argv,
        "--ports",
        inputs.focused_runtime.lab_preflight_ports.as_deref(),
    );
    push_string_option(
        &mut preflight_argv,
        "--ssh-opts",
        inputs.focused_runtime.lab_preflight_ssh_opts.as_deref(),
    );
    preflight_argv.extend([
        "--out".to_string(),
        focused_runtime_lab_preflight_log.display().to_string(),
    ]);
    let preflight_shell = argv_shell(&preflight_argv);
    let marker = focused_runtime_lab_preflight_marker.display().to_string();
    let marker_shell = format!(
        "printf '%s\\n' {} > {}",
        shell_quote("focused-runtime-lab-preflight: ok"),
        shell_quote(&marker),
    );
    Some(command(
        "focused-runtime-lab-preflight",
        "Check lab host SSH reachability, stale processes, lab ports, and free disk before remote focused-runtime.",
        Some("focused-runtime-lab-preflight"),
        vec![
            "bash".to_string(),
            "-lc".to_string(),
            format!("{preflight_shell} && {marker_shell}"),
        ],
        vec![
            focused_runtime_lab_preflight_log.display().to_string(),
            marker,
        ],
    ))
}

fn focused_runtime_command(
    inputs: &EvidenceCommandInputs<'_>,
    focused_runtime: &Path,
) -> EvidenceCommand {
    let mut argv = focused_runtime_base_argv(inputs);
    argv.push("--execute-remote".to_string());
    append_focused_runtime_options(
        &mut argv,
        inputs.focused_runtime,
        inputs.allow_uneven_stage_ranges,
        FocusedRuntimeOptionMode::ExecuteRemote,
    );
    argv.extend([
        "--focused-output".to_string(),
        focused_runtime.display().to_string(),
    ]);
    command(
        "focused-runtime",
        "Measure staged runtime latency and throughput on the selected hosts.",
        Some("skippy-bench-focused-runtime"),
        argv,
        vec![focused_runtime.display().to_string()],
    )
}

fn focused_runtime_base_argv(inputs: &EvidenceCommandInputs<'_>) -> Vec<String> {
    let mut argv = vec![
        inputs.skippy_bench_bin.display().to_string(),
        "focused-runtime".to_string(),
        "--stage-model".to_string(),
        inputs.package.display().to_string(),
        "--model-id".to_string(),
        inputs.model_id.to_string(),
        "--hosts".to_string(),
        inputs.hosts.to_string(),
        "--splits".to_string(),
        inputs.splits.to_string(),
        "--layer-end".to_string(),
        inputs.layer_end.to_string(),
        "--ctx-size".to_string(),
        inputs.ctx_size.to_string(),
    ];
    push_i32_option(&mut argv, "--n-gpu-layers", inputs.n_gpu_layers);
    argv.extend([
        "--cache-type-k".to_string(),
        inputs.cache_type_k.to_string(),
        "--cache-type-v".to_string(),
        inputs.cache_type_v.to_string(),
        "--activation-wire-dtype".to_string(),
        inputs.activation_wire_dtype.to_string(),
        "--prompt-corpus".to_string(),
        inputs.chat_corpus.display().to_string(),
        "--max-new-tokens".to_string(),
        inputs.max_tokens.to_string(),
        "--scenario".to_string(),
        "steady-decode".to_string(),
    ]);
    argv
}

#[derive(Debug, Clone, Copy)]
enum FocusedRuntimeOptionMode {
    SchemaSmoke,
    ExecuteRemote,
}

fn append_focused_runtime_options(
    argv: &mut Vec<String>,
    options: &FocusedRuntimeEvidenceArgs,
    allow_uneven_stage_ranges: bool,
    mode: FocusedRuntimeOptionMode,
) {
    push_path_option(
        argv,
        "--metrics-server-bin",
        options.metrics_server_bin.as_deref(),
    );
    push_path_option(
        argv,
        "--stage-server-bin",
        options.stage_server_bin.as_deref(),
    );
    push_path_option(argv, "--work-dir", options.work_dir.as_deref());
    push_string_option(argv, "--remote-root", options.remote_root.as_deref());
    push_string_option(
        argv,
        "--remote-root-map",
        options.remote_root_map.as_deref(),
    );
    push_string_option(
        argv,
        "--remote-shared-root-map",
        options.remote_shared_root_map.as_deref(),
    );
    push_string_option(
        argv,
        "--endpoint-host-map",
        options.endpoint_host_map.as_deref(),
    );
    push_string_option(argv, "--ssh-opts", options.ssh_opts.as_deref());
    push_string_option(
        argv,
        "--metrics-otlp-grpc-url",
        options.metrics_otlp_grpc_url.as_deref(),
    );
    push_string_option(
        argv,
        "--remote-bind-host",
        options.remote_bind_host.as_deref(),
    );
    push_display_option(argv, "--first-stage-port", options.first_stage_port);
    push_display_option(argv, "--startup-timeout-secs", options.startup_timeout_secs);
    push_display_option(argv, "--stage-max-inflight", options.stage_max_inflight);
    push_display_option(
        argv,
        "--stage-reply-credit-limit",
        options.stage_reply_credit_limit,
    );
    push_display_option(
        argv,
        "--stage-downstream-wire-delay-ms",
        options.stage_downstream_wire_delay_ms,
    );
    push_display_option(
        argv,
        "--stage-downstream-wire-mbps",
        options.stage_downstream_wire_mbps,
    );
    push_display_option(
        argv,
        "--stage-telemetry-queue-capacity",
        options.stage_telemetry_queue_capacity,
    );
    push_string_option(
        argv,
        "--stage-telemetry-level",
        options.stage_telemetry_level.as_deref(),
    );
    if matches!(mode, FocusedRuntimeOptionMode::ExecuteRemote) {
        push_bool_flag(
            argv,
            "--rsync-model-artifacts",
            options.rsync_model_artifacts,
        );
        push_bool_flag(argv, "--keep-remote", options.keep_remote);
        push_bool_flag(argv, "--child-logs", options.child_logs);
    }
    push_bool_flag(
        argv,
        "--stage-async-prefill-forward",
        options.stage_async_prefill_forward,
    );
    push_bool_flag(
        argv,
        "--allow-uneven-stage-ranges",
        options.allow_uneven_stage_ranges || allow_uneven_stage_ranges,
    );
}

fn push_path_option(argv: &mut Vec<String>, name: &str, value: Option<&Path>) {
    if let Some(value) = value {
        argv.extend([name.to_string(), value.display().to_string()]);
    }
}

fn push_string_option(argv: &mut Vec<String>, name: &str, value: Option<&str>) {
    if let Some(value) = value {
        argv.extend([name.to_string(), value.to_string()]);
    }
}

fn push_display_option<T: std::fmt::Display>(argv: &mut Vec<String>, name: &str, value: Option<T>) {
    if let Some(value) = value {
        argv.extend([name.to_string(), value.to_string()]);
    }
}

fn push_i32_option(argv: &mut Vec<String>, name: &str, value: i32) {
    if value < 0 {
        argv.push(format!("{name}={value}"));
    } else {
        argv.extend([name.to_string(), value.to_string()]);
    }
}

fn push_bool_flag(argv: &mut Vec<String>, name: &str, enabled: bool) {
    if enabled {
        argv.push(name.to_string());
    }
}

fn corpus_prep_commands(corpora: &[&Path]) -> Vec<EvidenceCommand> {
    let mut tiers = corpora
        .iter()
        .copied()
        .filter_map(default_corpus_tier)
        .collect::<Vec<_>>();
    tiers.sort_unstable();
    tiers.dedup();
    tiers
        .into_iter()
        .map(|tier| {
            command(
                &format!("prepare-corpus-{tier}"),
                &format!("Generate the default {tier} benchmark corpus if it is not present."),
                Some("skippy-bench-corpus-prep"),
                vec![
                    "just".to_string(),
                    "bench-corpus".to_string(),
                    tier.to_string(),
                ],
                vec![
                    format!("target/bench-corpora/{tier}/corpus.jsonl"),
                    format!("target/bench-corpora/{tier}/manifest.json"),
                ],
            )
        })
        .collect()
}

fn default_corpus_tier(path: &Path) -> Option<&'static str> {
    if path == Path::new(DEFAULT_TOKEN_CORPUS) {
        Some("long")
    } else if path == Path::new(DEFAULT_CHAT_CORPUS) {
        Some("coding-loop")
    } else if path == Path::new(DEFAULT_LONG_CONTEXT_CORPUS) {
        Some("long-context")
    } else {
        None
    }
}

fn final_rank_command(
    candidates: &[EvidencePlanReport],
    evidence_root: &Path,
    skippy_model_package_bin: &Path,
) -> Result<EvidenceCommand> {
    let first = candidates
        .first()
        .context("evidence-plan-all requires at least one candidate for final rank")?;
    ensure_shared_final_rank_shape(candidates, first)?;
    let final_rank = evidence_root.join("rank-after-evidence.json");
    let mut argv = vec![
        skippy_model_package_bin.display().to_string(),
        "quant-pack".to_string(),
        "rank".to_string(),
    ];
    argv.extend(candidates.iter().map(|candidate| candidate.run_dir.clone()));
    argv.extend([
        "--activation-wire-dtype".to_string(),
        first.activation_wire_dtype.clone(),
        "--ctx-size".to_string(),
        first.ctx_size.to_string(),
    ]);
    push_i32_option(&mut argv, "--n-gpu-layers", first.n_gpu_layers);
    argv.extend([
        "--cache-type-k".to_string(),
        first.cache_type_k.clone(),
        "--cache-type-v".to_string(),
        first.cache_type_v.clone(),
        "--out".to_string(),
        final_rank.display().to_string(),
    ]);
    Ok(command(
        "rank-after-evidence-all",
        "Rerank all selected candidates after skippy-bench and certification evidence are present.",
        Some("skippy-quant-pack-rank"),
        argv,
        vec![final_rank.display().to_string()],
    ))
}

fn ensure_shared_final_rank_shape(
    candidates: &[EvidencePlanReport],
    expected: &EvidencePlanReport,
) -> Result<()> {
    for candidate in candidates {
        if candidate.ctx_size != expected.ctx_size
            || candidate.n_gpu_layers != expected.n_gpu_layers
            || candidate.cache_type_k != expected.cache_type_k
            || candidate.cache_type_v != expected.cache_type_v
            || candidate.activation_wire_dtype != expected.activation_wire_dtype
        {
            bail!(
                "cannot generate final sweep rank for mixed runtime shapes: candidate {} differs from {}",
                candidate.candidate,
                expected.candidate
            );
        }
    }
    Ok(())
}

fn command(
    id: &str,
    description: &str,
    evidence_type: Option<&str>,
    argv: Vec<String>,
    outputs: Vec<String>,
) -> EvidenceCommand {
    let shell = argv_shell(&argv);
    EvidenceCommand {
        id: id.to_string(),
        description: description.to_string(),
        evidence_type: evidence_type.map(ToOwned::to_owned),
        argv,
        shell,
        outputs,
    }
}

fn argv_shell(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn infer_even_splits(layer_count: u32, stages: usize) -> Result<String> {
    if stages == 0 {
        bail!("stage count must be greater than zero");
    }
    if stages == 1 {
        return Ok(String::new());
    }
    if u32::try_from(stages).is_ok_and(|stages| stages > layer_count) {
        bail!("cannot infer {stages} stages for {layer_count} layers");
    }
    Ok((1..stages)
        .map(|stage| ((u64::from(layer_count) * stage as u64) / stages as u64).to_string())
        .collect::<Vec<_>>()
        .join(","))
}

struct SplitPlan {
    splits: String,
    source: SplitSource,
}

enum SplitSource {
    EvenInferred,
    CliOverride,
    CandidateStageHints,
}

impl SplitSource {
    fn as_str(&self) -> &'static str {
        match self {
            Self::EvenInferred => "even_inferred",
            Self::CliOverride => "cli_override",
            Self::CandidateStageHints => "candidate_stage_hints",
        }
    }

    fn allows_uneven_stage_ranges(&self) -> bool {
        matches!(self, Self::CandidateStageHints)
    }
}

struct PlanSplitRequest<'a> {
    run_dir: &'a Path,
    manifest: &'a BuildManifestInput,
}

fn split_plan(
    cli_splits: Option<&str>,
    layer_count: u32,
    stages: usize,
    plan_request: PlanSplitRequest<'_>,
) -> Result<SplitPlan> {
    if let Some(splits) = cli_splits {
        validate_splits(splits, layer_count, stages)?;
        return Ok(SplitPlan {
            splits: splits.to_string(),
            source: SplitSource::CliOverride,
        });
    }
    if let Some(splits) = candidate_stage_hint_splits(&plan_request, layer_count, stages)? {
        return Ok(SplitPlan {
            splits,
            source: SplitSource::CandidateStageHints,
        });
    }
    Ok(SplitPlan {
        splits: infer_even_splits(layer_count, stages)?,
        source: SplitSource::EvenInferred,
    })
}

fn candidate_stage_hint_splits(
    request: &PlanSplitRequest<'_>,
    layer_count: u32,
    stages: usize,
) -> Result<Option<String>> {
    let Some(plan) = request
        .manifest
        .plan
        .as_deref()
        .map(str::trim)
        .filter(|plan| !plan.is_empty())
    else {
        return Ok(None);
    };
    let plan_path = resolve_manifest_path(request.run_dir, plan);
    let quant_plan = read_json::<QuantPlanInput>(&plan_path)?;
    let candidate = quant_plan
        .candidates
        .iter()
        .find(|candidate| candidate.id == request.manifest.candidate)
        .with_context(|| {
            format!(
                "quant plan {} does not contain candidate {:?}",
                plan_path.display(),
                request.manifest.candidate
            )
        })?;
    let splits = splits_from_stage_hints(&candidate.stage_hints, layer_count, stages)
        .with_context(|| {
            format!(
                "candidate {:?} in {} has invalid stage_hints",
                candidate.id,
                plan_path.display()
            )
        })?;
    Ok(Some(splits))
}

fn splits_from_stage_hints(
    stage_hints: &[QuantPlanStageHintInput],
    layer_count: u32,
    stages: usize,
) -> Result<String> {
    if stage_hints.len() != stages {
        bail!(
            "stage_hints must contain exactly {stages} entries, got {}",
            stage_hints.len()
        );
    }
    let mut ordered = stage_hints.to_vec();
    ordered.sort_by_key(|hint| hint.stage_index);
    let mut expected_start = 0;
    for (expected_index, hint) in ordered.iter().enumerate() {
        if hint.stage_index != expected_index {
            bail!("stage_hints indexes must be contiguous from 0");
        }
        if hint.layer_start != expected_start {
            bail!(
                "stage {} starts at layer {}, expected {}",
                hint.stage_index,
                hint.layer_start,
                expected_start
            );
        }
        if hint.layer_end <= hint.layer_start || hint.layer_end > layer_count {
            bail!(
                "stage {} range {}..{} is outside 0..{}",
                hint.stage_index,
                hint.layer_start,
                hint.layer_end,
                layer_count
            );
        }
        expected_start = hint.layer_end;
    }
    if expected_start != layer_count {
        bail!("stage_hints end at layer {expected_start}, expected {layer_count}");
    }
    let splits = ordered
        .iter()
        .take(stages.saturating_sub(1))
        .map(|hint| hint.layer_end.to_string())
        .collect::<Vec<_>>()
        .join(",");
    validate_splits(&splits, layer_count, stages)?;
    Ok(splits)
}

fn validate_splits(splits: &str, layer_count: u32, stages: usize) -> Result<()> {
    if stages == 0 {
        bail!("stage count must be greater than zero");
    }
    let parsed = parse_split_boundaries(splits)?;
    let expected = stages.saturating_sub(1);
    if parsed.len() != expected {
        bail!(
            "--splits must contain exactly {expected} boundaries for {stages} stages, got {}",
            parsed.len()
        );
    }
    let mut previous = 0;
    for boundary in parsed {
        if boundary == 0 || boundary >= layer_count {
            bail!(
                "--splits boundary {boundary} must be within 1..{}",
                layer_count.saturating_sub(1)
            );
        }
        if boundary <= previous {
            bail!("--splits boundaries must be strictly increasing");
        }
        previous = boundary;
    }
    Ok(())
}

fn parse_split_boundaries(splits: &str) -> Result<Vec<u32>> {
    if splits.trim().is_empty() {
        return Ok(Vec::new());
    }
    splits
        .split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<u32>()
                .with_context(|| format!("parse --splits boundary {part:?}"))
        })
        .collect()
}

fn parse_hosts(hosts: &str) -> Result<Vec<String>> {
    let parsed = hosts
        .split(',')
        .map(str::trim)
        .filter(|host| !host.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if parsed.is_empty() {
        bail!("--hosts must contain at least one host");
    }
    Ok(parsed)
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | ':' | ',' | '='))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn build_manifest_path(run: &Path) -> PathBuf {
    if run.is_dir() {
        run.join("quant-pack-build.json")
    } else {
        run.to_path_buf()
    }
}

fn build_all_manifest_path(run: &Path) -> PathBuf {
    if run.is_dir() {
        run.join("quant-pack-build-all.json")
    } else {
        run.to_path_buf()
    }
}

fn selected_candidates<'a>(
    candidates: &'a [BuildAllCandidateInput],
    requested: &[String],
) -> Result<Vec<&'a BuildAllCandidateInput>> {
    if requested.is_empty() {
        return Ok(candidates.iter().collect());
    }
    let mut selected = Vec::with_capacity(requested.len());
    for candidate_id in requested {
        let candidate = candidates
            .iter()
            .find(|candidate| candidate.candidate == *candidate_id)
            .with_context(|| {
                format!("build-all manifest does not contain requested candidate {candidate_id:?}")
            })?;
        selected.push(candidate);
    }
    Ok(selected)
}

fn select_evidence_candidates<'a>(
    manifest: &'a BuildAllManifestInput,
    build_all_dir: &Path,
    args: &QuantPackEvidencePlanAllArgs,
) -> Result<CandidateSelection<'a>> {
    if !args.candidates.is_empty() && args.top_ranked.is_some() {
        bail!("--candidate and --top-ranked are mutually exclusive");
    }
    if !args.candidates.is_empty() {
        return Ok(CandidateSelection {
            source: "candidate_filter".to_string(),
            candidates: selected_candidates(&manifest.candidates, &args.candidates)?,
        });
    }
    if let Some(limit) = args.top_ranked {
        if limit == 0 {
            bail!("--top-ranked must be greater than zero");
        }
        let rank_path = resolve_manifest_path(build_all_dir, &manifest.rank);
        let rank = read_json::<RankReportInput>(&rank_path)?;
        let candidate_ids = rank
            .candidates
            .into_iter()
            .filter(|candidate| candidate.valid)
            .take(limit)
            .map(|candidate| candidate.candidate)
            .collect::<Vec<_>>();
        if candidate_ids.is_empty() {
            bail!(
                "rank report {} did not contain any valid candidates for --top-ranked",
                rank_path.display()
            );
        }
        return Ok(CandidateSelection {
            source: format!("top_ranked:{limit}"),
            candidates: selected_candidates(&manifest.candidates, &candidate_ids)?,
        });
    }
    Ok(CandidateSelection {
        source: "all_candidates".to_string(),
        candidates: manifest.candidates.iter().collect(),
    })
}

fn resolve_manifest_path(run_dir: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        let manifest_relative = run_dir.join(&path);
        if manifest_relative.exists() {
            return manifest_relative;
        }
        if path.exists() {
            return path;
        }
        manifest_relative
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
#[path = "evidence_plan_tests.rs"]
mod tests;
