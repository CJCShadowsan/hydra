use anyhow::{Context, Result, bail};
use mesh_llm_gpu_bench::DecodeKernelProbe;
use mesh_llm_system::hardware::HardwareSurvey;
use model_artifact::{ModelFormat, ResolvedModelArtifact, resolve_model_artifact_ref};
use model_fit::{
    AcceleratorKind, BackendKind, CpuProfile, FitStatus, GpuBenchmarkAcceleratorFacts,
    GpuBenchmarkHardwareInput, GpuBenchmarkOutput, HardwareProfile, MemoryProfile, ModelProfile,
    ModelRecommendation, SelectionConfig, TensorTypeBytes, WorkloadProfile,
    hardware_profile_from_gpu_benchmark, profile_gguf_path, score_model, throughput_sample_stats,
};
use model_hf::{HfModelRepository, ModelDownloadProgress, ModelDownloadProgressEvent};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const DEFAULT_CTX_SIZE: u32 = 8192;
const DEFAULT_WARMUP_TOKENS: usize = 16;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_REMEASURE_REPEATS: usize = 3;
const DEFAULT_REMEASURE_RAW_SPREAD: f64 = 0.25;
const DEFAULT_REMEASURE_ORDERED_DROP: f64 = 0.20;
const DEFAULT_REMEASURE_PAUSE: Duration = Duration::from_secs(3);
const DEFAULT_CONFIRM_REPEATS: usize = 3;
const DEFAULT_CONFIRM_DELTA: f64 = 0.20;
const DEFAULT_TOLERANCE: f64 = 0.10;
const DEFAULT_MAX_SPREAD: f64 = 0.10;
const DEFAULT_ABI_DECODE_REPEATS: usize = 3;
const DEFAULT_ABI_DECODE_MEASURED_TOKENS: usize = 128;
const FIRST_TOKEN_MAX_NEW_TOKENS: usize = 1;
const KV_WARM_REUSE_MAX_NEW_TOKENS: usize = 16;

#[derive(Clone, Debug)]
struct Args {
    output_json: PathBuf,
    skippy_bench_bin: PathBuf,
    skippy_server_bin: PathBuf,
    metrics_server_bin: PathBuf,
    gpu_benchmark_json: Option<PathBuf>,
    model_files: Vec<PathBuf>,
    benchmark_scenarios: Vec<String>,
    base_port: u16,
    benchmark_all: bool,
    fit_only: bool,
    dense_probe_depth: DenseProbeDepth,
    show_progress: bool,
    models: Vec<ModelInput>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseProbeDepth {
    Standard,
    Deep,
}

#[derive(Clone, Debug)]
enum ModelInput {
    Ref(String),
    Local(LocalModelInput),
}

#[derive(Clone, Debug)]
struct LocalModelInput {
    model_ref: String,
    gguf_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    schema_version: u32,
    generated_at_unix_secs: u64,
    command: Vec<String>,
    fit_input_contract: FitInputContract,
    hardware_profile: HardwareProfile,
    gpu_benchmark_outputs: Vec<GpuBenchmarkOutput>,
    gpu_benchmark_json: Value,
    selection_config: SelectionConfig,
    validation_config: ValidationConfig,
    models: Vec<ModelValidationReport>,
    summary: ValidationSummary,
}

#[derive(Debug, Serialize)]
struct FitInputContract {
    hardware_fields_consumed: Vec<&'static str>,
    model_fields_consumed: Vec<&'static str>,
    validation_backend: &'static str,
    validation_note: &'static str,
}

#[derive(Debug, Serialize)]
struct ValidationConfig {
    ctx_size: u32,
    warmup_tokens: usize,
    max_new_tokens: usize,
    repeats: usize,
    tolerance: f64,
    max_spread: f64,
    remeasure_repeats: usize,
    remeasure_raw_spread: f64,
    remeasure_ordered_drop: f64,
    confirm_repeats: usize,
    confirm_delta: f64,
    abi_decode_repeats: usize,
    abi_decode_measured_tokens: usize,
    benchmark_all: bool,
    fit_only: bool,
    dense_probe_depth: &'static str,
    show_progress: bool,
    prompt: String,
    primary_workload: String,
    scored_workloads: Vec<String>,
    benchmark_scenarios: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct ModelValidationReport {
    input_ref: String,
    resolved_ref: Option<String>,
    artifact: Option<ResolvedModelArtifact>,
    downloaded_paths: Vec<PathBuf>,
    primary_gguf_path: Option<PathBuf>,
    model_profile: Option<ModelProfile>,
    recommendation: Option<ModelRecommendation>,
    fit_interpretation: Option<FitInterpretation>,
    runtime_diagnostic: Option<RuntimeDiagnostic>,
    recommendations: Vec<WorkloadRecommendation>,
    abi_decode_probe: Option<AbiDecodeProbeSummary>,
    decode_probe_diagnostic: Option<DecodeProbeDiagnostic>,
    graph_inventory_diagnostic: Option<GraphInventoryDiagnostic>,
    operation_bucket_diagnostic: Option<OperationBucketDiagnostic>,
    model_specific_decode_kernel_probes: Vec<DecodeKernelProbe>,
    model_specific_probe_errors: Vec<String>,
    benchmarks: Vec<BenchmarkScenarioSummary>,
    benchmark: BenchmarkSummary,
    errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct FitInterpretation {
    local_accelerated_fit: bool,
    single_node_validation_allowed: bool,
    summary: String,
    details: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeDiagnostic {
    validation_shape: &'static str,
    selected_backend: String,
    selected_accelerator: Option<String>,
    layer_start: u32,
    layer_end: Option<u32>,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: &'static str,
    cache_type_v: &'static str,
    flash_attn_type: &'static str,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    load_mode: &'static str,
    filter_tensors_on_load: bool,
    include_embeddings: bool,
    include_output: bool,
    steady_decode_command: Option<Vec<String>>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct AbiDecodeProbeSummary {
    attempted: bool,
    skip_reason: Option<String>,
    tokens_per_second: Option<f64>,
    elapsed_ms: Option<f64>,
    llama_eval_tokens_per_second: Option<f64>,
    llama_eval_ms: Option<f64>,
    non_eval_overhead_ms: Option<f64>,
    non_eval_overhead_pct: Option<f64>,
    decode_call_tokens_per_second: Option<f64>,
    decode_call_ms: Option<f64>,
    sampling_tokens_per_second: Option<f64>,
    sampling_ms: Option<f64>,
    llama_eval_count: Option<u64>,
    llama_graph_reuse_count: Option<i64>,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    graph_inventory: Vec<AbiGraphInventoryBucket>,
    measured_tokens: Option<u64>,
    prompt_token_count: Option<u64>,
    command: Vec<String>,
    observations: Vec<AbiDecodeProbeObservation>,
    sample_count: usize,
    raw_sample_count: usize,
    min_tokens_per_second: Option<f64>,
    max_tokens_per_second: Option<f64>,
    spread_pct: Option<f64>,
    raw_spread_pct: Option<f64>,
    denoised_outlier_count: usize,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct AbiGraphInventoryBucket {
    family: Option<String>,
    ggml_op: Option<i64>,
    ggml_type: Option<u64>,
    node_count: Option<u64>,
    element_count: Option<u64>,
    output_bytes: Option<u64>,
    src0_bytes: Option<u64>,
    src1_bytes: Option<u64>,
    ne: Vec<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct AbiDecodeProbeObservation {
    repeat: usize,
    command: Vec<String>,
    status_code: Option<i32>,
    tokens_per_second: Option<f64>,
    elapsed_ms: Option<f64>,
    llama_eval_tokens_per_second: Option<f64>,
    llama_eval_ms: Option<f64>,
    non_eval_overhead_ms: Option<f64>,
    decode_call_tokens_per_second: Option<f64>,
    decode_call_ms: Option<f64>,
    sampling_tokens_per_second: Option<f64>,
    sampling_ms: Option<f64>,
    llama_eval_count: Option<u64>,
    llama_graph_reuse_count: Option<i64>,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    graph_inventory: Vec<AbiGraphInventoryBucket>,
    measured_tokens: Option<u64>,
    prompt_token_count: Option<u64>,
    stderr_tail: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecodeProbeDiagnostic {
    predicted_tokens_per_second: Option<f64>,
    abi_tokens_per_second: Option<f64>,
    observed_tokens_per_second: Option<f64>,
    observed_over_fit: Option<f64>,
    abi_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    observed_vs_fit: String,
    abi_vs_fit: String,
    observed_vs_abi: String,
    classification: String,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct GraphInventoryDiagnostic {
    available: bool,
    status: String,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    selected_transformer_probe: Option<String>,
    selected_transformer_probe_layers: Option<u32>,
    metadata_transformer_matmul_nodes: u64,
    graph_transformer_matmul_nodes: u64,
    metadata_transformer_weight_bytes: u64,
    graph_transformer_weight_src0_bytes: u64,
    graph_unclassified_matmul_src0_bytes: u64,
    graph_transformer_src0_over_metadata: Option<f64>,
    graph_transformer_plus_unclassified_src0_over_metadata: Option<f64>,
    estimated_transformer_block_ms: Option<f64>,
    abi_ms_per_token: Option<f64>,
    estimated_transformer_over_abi: Option<f64>,
    comparisons: Vec<GraphInventoryComparison>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct GraphInventoryComparison {
    name: &'static str,
    metadata_weight_bytes: u64,
    metadata_node_count: u64,
    graph_weight_src0_bytes: u64,
    graph_node_count: u64,
    src0_over_metadata: Option<f64>,
    node_count_delta: i64,
}

#[derive(Clone, Debug, Serialize)]
struct OperationBucketDiagnostic {
    available: bool,
    estimated_selected_ms_per_token: Option<f64>,
    abi_ms_per_token: Option<f64>,
    estimated_over_abi: Option<f64>,
    buckets: Vec<OperationBucketRow>,
    raw_graph_families: Vec<GraphOperationFamilyRow>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct OperationBucketRow {
    bucket: &'static str,
    source: String,
    graph_families: Vec<&'static str>,
    estimated_ms: Option<f64>,
    estimated_traffic_bytes: u64,
    metadata_weight_bytes: u64,
    graph_node_count: u64,
    graph_src0_bytes: u64,
    graph_src1_bytes: u64,
    graph_output_bytes: u64,
    graph_src0_over_metadata: Option<f64>,
    estimated_share_of_selected_ms: Option<f64>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct GraphOperationFamilyRow {
    family: String,
    node_count: u64,
    src0_bytes: u64,
    src1_bytes: u64,
    output_bytes: u64,
    element_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct OperationBucketSpec {
    bucket: &'static str,
    graph_families: &'static [&'static str],
    cost_group: &'static str,
    metadata_weight_bytes: u64,
    note: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct WorkloadRecommendation {
    workload: String,
    recommendation: ModelRecommendation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkScenarioKind {
    // Sustained one-token-at-a-time generation. This is the scenario model-fit
    // is currently best at predicting because it is closest to llama.cpp's
    // memory-bandwidth-bound decode loop.
    SteadyDecode,
    // Prompt ingestion only, measured as prompt tokens divided by Skippy's
    // `prefill_elapsed_ms`. This is intentionally separate from first-token
    // latency so a miss can be attributed to prefill matmul throughput rather
    // than request setup or the first decode step after prefill.
    Prefill,
    // End-to-end request latency for a long prompt and one generated token.
    // Lower is better, so verdict labels are inverted after the generic ratio
    // check. This scenario is a user-visible latency target, not a pure decode
    // or pure prefill micro-benchmark.
    FirstToken,
    // Short repeated generation with session reuse. This gives us a small
    // signal for agent/tool loops where the same prefix remains resident.
    KvWarmReuse,
}

#[derive(Clone, Debug)]
struct BenchmarkScenarioSpec {
    kind: BenchmarkScenarioKind,
    name: &'static str,
    fit_metric: &'static str,
    prompt: String,
    ctx_size: u32,
    max_new_tokens: usize,
    warmup_tokens: usize,
    request_count: usize,
    reuse_session: bool,
}

#[derive(Clone, Copy, Debug)]
struct BenchmarkExpected {
    predicted: Option<f64>,
    range: Option<(f64, f64)>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkScenarioSummary {
    scenario: String,
    fit_metric: String,
    predicted: Option<f64>,
    predicted_range: Option<(f64, f64)>,
    prediction_context_tokens: Option<u32>,
    prediction_decode_cost_breakdown: Option<model_fit::DecodeCostBreakdown>,
    observed: Option<f64>,
    observed_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    first_token_breakdown: Option<FirstTokenBreakdown>,
    verdict: String,
    benchmark: BenchmarkSummary,
}

#[derive(Clone, Debug, Serialize)]
struct FirstTokenBreakdown {
    prompt_token_count: Option<u64>,
    tokenizer_vocab_size: Option<u32>,
    chat_template_available: bool,
    predicted_prefill_ms: Option<f64>,
    predicted_decode_ms: Option<f64>,
    predicted_overhead_ms: Option<f64>,
    predicted_sampler_ms: Option<f64>,
    predicted_sampled_decode_ms: Option<f64>,
    observed_tokenize_ms: Option<f64>,
    observed_prefill_ms: Option<f64>,
    observed_decode_ms: Option<f64>,
    observed_sampled_decode_residual_ms: Option<f64>,
    observed_sampled_decode_residual_us_per_prompt_token: Option<f64>,
    observed_unattributed_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct BenchmarkSummary {
    attempted: bool,
    skip_reason: Option<String>,
    observations: Vec<BenchmarkObservation>,
    successful_repeats: usize,
    sample_count: usize,
    raw_sample_count: usize,
    // Historical field name: for throughput scenarios this is tokens/sec; for
    // first-token latency it stores milliseconds so the same denoising and
    // spread machinery can be reused. The scenario wrapper exposes the actual
    // metric name through `fit_metric`, and Markdown rendering labels the value
    // generically as predicted/observed.
    median_tokens_per_sec: Option<f64>,
    min_tokens_per_sec: Option<f64>,
    max_tokens_per_sec: Option<f64>,
    spread_pct: Option<f64>,
    raw_median_tokens_per_sec: Option<f64>,
    raw_min_tokens_per_sec: Option<f64>,
    raw_max_tokens_per_sec: Option<f64>,
    raw_spread_pct: Option<f64>,
    denoised_outlier_count: usize,
    remeasured: bool,
    remeasure_reason: Option<String>,
    initial_observations: Vec<BenchmarkObservation>,
    rejected_remeasure_observations: Vec<BenchmarkObservation>,
    observed_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    verdict: String,
    errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkObservation {
    repeat: usize,
    run_id: String,
    command: Vec<String>,
    status_code: Option<i32>,
    wall_seconds: f64,
    prompt_token_count: Option<u64>,
    generated_tokens_per_sec: Option<f64>,
    generated_token_count: Option<u64>,
    text_request_elapsed_ms: Option<f64>,
    request_count: Option<u64>,
    reuse_session: Option<bool>,
    request_results: Vec<BenchmarkRequestObservation>,
    stdout_json_path: Option<PathBuf>,
    report_json_path: PathBuf,
    stderr_tail: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkRequestObservation {
    request_id: Option<String>,
    session_id: Option<String>,
    elapsed_ms: Option<f64>,
    tokenize_elapsed_ms: Option<f64>,
    prefill_elapsed_ms: Option<f64>,
    decode_elapsed_ms: Option<f64>,
    prompt_token_count: Option<u64>,
    generated_token_count: Option<u64>,
    generated_tokens_per_sec: Option<f64>,
    decode_tokens_per_sec: Option<f64>,
}

#[derive(Debug, Default, Serialize)]
struct ValidationSummary {
    model_count: usize,
    benchmarked_count: usize,
    matched_count: usize,
    slower_than_fit_count: usize,
    faster_than_fit_count: usize,
    metadata_estimate_miss_count: usize,
    runtime_path_mismatch_count: usize,
    probe_mismatch_count: usize,
    noisy_count: usize,
    skipped_count: usize,
    error_count: usize,
    runtime_error_count: usize,
    median_observed_over_fit: Option<f64>,
    mean_observed_over_fit: Option<f64>,
    median_absolute_percent_error: Option<f64>,
    within_tolerance_count: usize,
    scenario_summaries: Vec<ScenarioValidationSummary>,
}

#[derive(Debug, Default, Serialize)]
struct ScenarioValidationSummary {
    scenario: String,
    sample_count: usize,
    matched_count: usize,
    slower_than_fit_count: usize,
    faster_than_fit_count: usize,
    metadata_estimate_miss_count: usize,
    runtime_path_mismatch_count: usize,
    probe_mismatch_count: usize,
    noisy_count: usize,
    skipped_count: usize,
    error_count: usize,
    runtime_error_count: usize,
    within_tolerance_count: usize,
    median_observed_over_fit: Option<f64>,
    mean_observed_over_fit: Option<f64>,
    median_absolute_percent_error: Option<f64>,
}

struct PreparedModel {
    input_ref: String,
    resolved_ref: Option<String>,
    artifact: Option<ResolvedModelArtifact>,
    downloaded_paths: Vec<PathBuf>,
    primary_gguf_path: PathBuf,
    profile: ModelProfile,
}

struct LoadedHardware {
    profile: HardwareProfile,
    benchmark_outputs: Vec<GpuBenchmarkOutput>,
    raw_json: Value,
}

struct LocalGpuBenchmark {
    outputs: Vec<GpuBenchmarkOutput>,
    backend: BackendKind,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    let hardware = load_hardware_profile(&args)?;
    let selection_config = selection_config(&primary_workload_profile());
    let repository = HfModelRepository::from_env().context("create Hugging Face repository")?;

    let mut models = Vec::new();
    for (index, input) in args.models.iter().enumerate() {
        let report = validate_model(&args, &repository, &hardware.profile, input, index).await;
        models.push(report);
        let partial_report = build_validation_report(&args, &hardware, &selection_config, &models);
        write_json_report(&args.output_json, &partial_report)?;
    }

    let report = build_validation_report(&args, &hardware, &selection_config, &models);
    write_json_report(&args.output_json, &report)?;
    print_markdown_table(&report.models);
    eprintln!("wrote {}", args.output_json.display());
    Ok(())
}

fn build_validation_report(
    args: &Args,
    hardware: &LoadedHardware,
    selection_config: &SelectionConfig,
    models: &[ModelValidationReport],
) -> ValidationReport {
    let summary = summarize(args, models, DEFAULT_TOLERANCE);
    ValidationReport {
        schema_version: 1,
        generated_at_unix_secs: unix_timestamp_secs(),
        command: std::env::args().collect(),
        fit_input_contract: fit_input_contract(),
        hardware_profile: hardware.profile.clone(),
        gpu_benchmark_outputs: hardware.benchmark_outputs.clone(),
        gpu_benchmark_json: hardware.raw_json.clone(),
        selection_config: selection_config.clone(),
        validation_config: ValidationConfig::from_args(args),
        models: models.to_vec(),
        summary,
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::env::args().skip(1);
        let mut parsed = Self {
            output_json: default_output_json_path(),
            skippy_bench_bin: default_binary_path("skippy-bench"),
            skippy_server_bin: default_binary_path("skippy-server"),
            metrics_server_bin: default_binary_path("metrics-server"),
            gpu_benchmark_json: None,
            model_files: Vec::new(),
            benchmark_scenarios: Vec::new(),
            base_port: 18400,
            benchmark_all: false,
            fit_only: false,
            dense_probe_depth: DenseProbeDepth::Standard,
            show_progress: true,
            models: Vec::new(),
        };

        while let Some(arg) = values.next() {
            parsed.parse_arg(arg, &mut values)?;
        }
        parsed.load_model_files()?;
        parsed.validate_benchmark_scenarios()?;

        if parsed.models.is_empty() {
            bail!("provide at least one model ref");
        }
        Ok(parsed)
    }

    fn parse_arg(&mut self, arg: String, values: &mut impl Iterator<Item = String>) -> Result<()> {
        match arg.as_str() {
            "--output-json" => {
                self.output_json = PathBuf::from(next_value(values, "--output-json")?)
            }
            "--skippy-bench-bin" => {
                self.skippy_bench_bin = PathBuf::from(next_value(values, "--skippy-bench-bin")?);
            }
            "--skippy-server-bin" => {
                self.skippy_server_bin = PathBuf::from(next_value(values, "--skippy-server-bin")?);
            }
            "--metrics-server-bin" => {
                self.metrics_server_bin =
                    PathBuf::from(next_value(values, "--metrics-server-bin")?);
            }
            "--gpu-benchmark-json" => {
                self.gpu_benchmark_json =
                    Some(PathBuf::from(next_value(values, "--gpu-benchmark-json")?));
            }
            "--models-file" => {
                self.model_files
                    .push(PathBuf::from(next_value(values, "--models-file")?));
            }
            "--scenario" => self
                .benchmark_scenarios
                .push(next_value(values, "--scenario")?),
            "--scenarios" => {
                self.benchmark_scenarios.extend(
                    next_value(values, "--scenarios")?
                        .split(',')
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string),
                );
            }
            "--base-port" => self.base_port = parse_next(values, "--base-port")?,
            "--benchmark-all" => self.benchmark_all = true,
            "--fit-only" => self.fit_only = true,
            "--dense-probe-depth" => {
                self.dense_probe_depth =
                    DenseProbeDepth::parse(&next_value(values, "--dense-probe-depth")?)?;
            }
            "--no-progress" => self.show_progress = false,
            "--model-ref" => self
                .models
                .push(ModelInput::Ref(next_value(values, "--model-ref")?)),
            "--model" => {
                self.models
                    .push(ModelInput::Local(parse_local_model(&next_value(
                        values, "--model",
                    )?)?));
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other if other.starts_with('-') => bail!("unknown argument {other}"),
            model_ref => self.models.push(ModelInput::Ref(model_ref.to_string())),
        }
        Ok(())
    }

    fn load_model_files(&mut self) -> Result<()> {
        for path in &self.model_files {
            let contents = fs::read_to_string(path)
                .with_context(|| format!("read model manifest {}", path.display()))?;
            for (line_index, line) in contents.lines().enumerate() {
                let Some(model_ref) = parse_model_manifest_line(line) else {
                    continue;
                };
                if model_ref.contains('=') {
                    bail!(
                        "invalid model manifest entry {}:{}: key/value metadata is not supported yet",
                        path.display(),
                        line_index + 1
                    );
                }
                self.models.push(ModelInput::Ref(model_ref.to_string()));
            }
        }
        Ok(())
    }

    fn validate_benchmark_scenarios(&self) -> Result<()> {
        if self.benchmark_scenarios.is_empty()
            || self
                .benchmark_scenarios
                .iter()
                .any(|scenario| scenario == "all")
        {
            return Ok(());
        }
        let valid = benchmark_scenarios()
            .into_iter()
            .map(|scenario| scenario.name)
            .collect::<Vec<_>>();
        for requested in &self.benchmark_scenarios {
            if !valid.contains(&requested.as_str()) {
                bail!(
                    "unknown benchmark scenario {requested}; valid scenarios: {}",
                    valid.join(", ")
                );
            }
        }
        Ok(())
    }
}

impl DenseProbeDepth {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "standard" => Ok(Self::Standard),
            "deep" => Ok(Self::Deep),
            other => bail!("unknown dense probe depth {other}; valid values: standard, deep"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Deep => "deep",
        }
    }
}

fn default_output_json_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("tmp").join("model-fit-validation.json"))
        .unwrap_or_else(|| PathBuf::from("model-fit-validation.json"))
}

fn parse_model_manifest_line(line: &str) -> Option<&str> {
    let without_comment = line.split_once('#').map_or(line, |(value, _)| value);
    let trimmed = without_comment.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn default_binary_path(name: &str) -> PathBuf {
    let release = PathBuf::from(format!("target/release/{name}"));
    if release.exists() {
        release
    } else {
        PathBuf::from(format!("target/debug/{name}"))
    }
}

impl ValidationConfig {
    fn from_args(args: &Args) -> Self {
        let benchmark_scenarios = selected_benchmark_scenarios(args)
            .into_iter()
            .map(|scenario| scenario.name.to_string())
            .collect();
        Self {
            ctx_size: DEFAULT_CTX_SIZE,
            warmup_tokens: DEFAULT_WARMUP_TOKENS,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            repeats: DEFAULT_REPEATS,
            tolerance: DEFAULT_TOLERANCE,
            max_spread: DEFAULT_MAX_SPREAD,
            remeasure_repeats: DEFAULT_REMEASURE_REPEATS,
            remeasure_raw_spread: DEFAULT_REMEASURE_RAW_SPREAD,
            remeasure_ordered_drop: DEFAULT_REMEASURE_ORDERED_DROP,
            confirm_repeats: DEFAULT_CONFIRM_REPEATS,
            confirm_delta: DEFAULT_CONFIRM_DELTA,
            abi_decode_repeats: DEFAULT_ABI_DECODE_REPEATS,
            abi_decode_measured_tokens: DEFAULT_ABI_DECODE_MEASURED_TOKENS,
            benchmark_all: args.benchmark_all,
            fit_only: args.fit_only,
            dense_probe_depth: args.dense_probe_depth.as_str(),
            show_progress: args.show_progress,
            prompt: validation_prompt().into(),
            primary_workload: primary_workload_label().into(),
            scored_workloads: workload_profiles()
                .iter()
                .map(|(label, _)| (*label).to_string())
                .collect(),
            benchmark_scenarios,
        }
    }
}

async fn validate_model(
    args: &Args,
    repository: &HfModelRepository,
    hardware: &HardwareProfile,
    input: &ModelInput,
    model_index: usize,
) -> ModelValidationReport {
    let label = input_label(input);
    heartbeat(
        Some(model_index),
        &label,
        "model_start",
        "starting validation",
    );
    match prepare_model(args, repository, input, model_index).await {
        Ok(prepared) => {
            heartbeat(
                Some(model_index),
                &prepared.input_ref,
                "model_prepared",
                "metadata profile is ready",
            );
            let report = validate_prepared_model(args, hardware, prepared, model_index);
            heartbeat(
                Some(model_index),
                &report.input_ref,
                "model_done",
                &format!("steady_decode_verdict={}", report.benchmark.verdict),
            );
            report
        }
        Err(err) => {
            let error = format!("{err:#}");
            heartbeat(Some(model_index), &label, "model_error", &error);
            error_report(label, error)
        }
    }
}

fn validate_prepared_model(
    args: &Args,
    hardware: &HardwareProfile,
    prepared: PreparedModel,
    model_index: usize,
) -> ModelValidationReport {
    let model_specific_probes =
        model_specific_decode_kernel_probes(args, hardware, &prepared.profile, model_index);
    let mut hardware_for_model = hardware.clone();
    for accelerator in &mut hardware_for_model.accelerators {
        accelerator
            .decode_kernel_probes
            .extend(model_specific_probes.probes.clone());
    }
    heartbeat(
        Some(model_index),
        &prepared.input_ref,
        "score_start",
        "scoring workload recommendations",
    );
    let recommendations = score_workloads(&hardware_for_model, &prepared.profile);
    let recommendation = recommendations
        .iter()
        .find(|entry| entry.workload == primary_workload_label())
        .map(|entry| entry.recommendation.clone())
        .unwrap_or_else(|| {
            recommendations
                .first()
                .expect("workload list is not empty")
                .recommendation
                .clone()
        });
    heartbeat(
        Some(model_index),
        &prepared.input_ref,
        "score_done",
        &format!(
            "fit_status={:?} selected_backend={:?} selected_accelerator={} decode_tps={}",
            recommendation.fit_status,
            recommendation.selected_backend,
            recommendation
                .selected_accelerator
                .as_deref()
                .unwrap_or("-"),
            display_opt(
                recommendation
                    .estimated_decode_tokens_per_sec
                    .map(f64::from)
            )
        ),
    );
    let mut benchmarks = if args.fit_only {
        Vec::new()
    } else {
        benchmark_model(
            args,
            &hardware_for_model,
            &prepared,
            &recommendation,
            model_index,
        )
    };
    let abi_decode_probe = if args.fit_only {
        None
    } else {
        Some(run_abi_decode_probe_for_recommendation(
            args,
            &prepared,
            &recommendation,
            model_index,
        ))
    };
    apply_observed_over_abi(&mut benchmarks, abi_decode_probe.as_ref());
    let fit_interpretation = Some(fit_interpretation(&recommendation));
    let runtime_diagnostic = Some(runtime_diagnostic(
        &prepared.profile,
        &recommendation,
        &benchmarks,
    ));
    let steady_benchmark = benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == "steady_decode");
    let benchmark = steady_benchmark
        .map(|benchmark| benchmark.benchmark.clone())
        .unwrap_or_else(|| BenchmarkSummary {
            verdict: "skipped".into(),
            skip_reason: Some("steady_decode scenario was not produced".into()),
            ..BenchmarkSummary::default()
        });
    let decode_probe_diagnostic =
        decode_probe_diagnostic(&recommendation, abi_decode_probe.as_ref(), steady_benchmark);
    let graph_inventory_diagnostic = graph_inventory_diagnostic(
        &prepared.profile,
        &recommendation,
        abi_decode_probe.as_ref(),
    );
    let operation_bucket_diagnostic = operation_bucket_diagnostic(
        &prepared.profile,
        &recommendation,
        abi_decode_probe.as_ref(),
    );
    ModelValidationReport {
        input_ref: prepared.input_ref,
        resolved_ref: prepared.resolved_ref,
        artifact: prepared.artifact,
        downloaded_paths: prepared.downloaded_paths,
        primary_gguf_path: Some(prepared.primary_gguf_path),
        model_profile: Some(prepared.profile),
        recommendation: Some(recommendation),
        fit_interpretation,
        runtime_diagnostic,
        recommendations,
        abi_decode_probe,
        decode_probe_diagnostic,
        graph_inventory_diagnostic,
        operation_bucket_diagnostic,
        model_specific_decode_kernel_probes: model_specific_probes.probes,
        model_specific_probe_errors: model_specific_probes.errors,
        benchmarks,
        benchmark,
        errors: Vec::new(),
    }
}

#[derive(Clone, Debug, Default)]
struct ModelSpecificDecodeProbes {
    probes: Vec<DecodeKernelProbe>,
    errors: Vec<String>,
}

fn model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let mut collected = if has_recurrent_attention_profile(profile) {
        linear_attention_model_specific_decode_kernel_probes(args, hardware, profile, model_index)
    } else {
        match profile.architecture_class {
            model_fit::ModelArchitectureClass::SparseMoeTransformer => {
                moe_model_specific_decode_kernel_probes(args, hardware, profile, model_index)
            }
            model_fit::ModelArchitectureClass::DenseTransformer => {
                dense_model_specific_decode_kernel_probes(args, hardware, profile, model_index)
            }
            _ => ModelSpecificDecodeProbes::default(),
        }
    };
    append_model_output_projection_probes(args, hardware, profile, model_index, &mut collected);
    collected
}

fn linear_attention_model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let plans = linear_attention_graph_probe_plans(profile);
    if plans.is_empty() {
        return ModelSpecificDecodeProbes {
            probes: Vec::new(),
            errors: vec![
                "could not derive model-shaped linear-attention graph probe dimensions from GGUF metadata"
                    .into(),
            ],
        };
    }
    let mut collected = ModelSpecificDecodeProbes::default();
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        for plan in &plans {
            heartbeat(
                Some(model_index),
                &model_label,
                "model_linear_attention_probe_start",
                &format!(
                    "backend={:?} tensor_type={} hidden={} qkv={} gate={} state={} out={} ffn={} recurrent_layers={} full_attention_layers={} kv_width={} graph_features={} norm_head_width={}",
                    accelerator.backend,
                    plan.tensor_type,
                    plan.hidden,
                    plan.qkv_width,
                    plan.gate_width,
                    plan.state_width,
                    plan.output_input_width,
                    plan.ffn,
                    plan.recurrent_layers,
                    plan.full_attention_layers,
                    plan.kv_width,
                    plan.graph_features,
                    plan.norm_head_width,
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped linear attention graph {} {} r{} f{}",
                    model_label,
                    plan.tensor_type,
                    plan.recurrent_layers,
                    plan.full_attention_layers
                ),
            );
            match mesh_llm_gpu_bench::run_model_linear_attention_graph_probe(
                backend,
                plan.tensor_type,
                mesh_llm_gpu_bench::LinearAttentionGraphProbeShape {
                    hidden: plan.hidden,
                    qkv_width: plan.qkv_width,
                    gate_width: plan.gate_width,
                    state_width: plan.state_width,
                    output_input_width: plan.output_input_width,
                    ffn: plan.ffn,
                    recurrent_layers: plan.recurrent_layers,
                    full_attention_layers: plan.full_attention_layers,
                    kv_width: plan.kv_width,
                    graph_features: plan.graph_features,
                    norm_head_width: plan.norm_head_width,
                },
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_linear_attention_probe_done",
                        &format!("tensor_type={} probes={}", plan.tensor_type, probes.len()),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!("tensor_type={}: {error:#}", plan.tensor_type);
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_linear_attention_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
        }
    }
    collected
}

fn dense_model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let plans = dense_graph_probe_plans(args, profile);
    if plans.is_empty() {
        return ModelSpecificDecodeProbes {
            probes: Vec::new(),
            errors: vec![
                "could not derive model-shaped dense graph probe dimensions from GGUF metadata"
                    .into(),
            ],
        };
    }
    let mut collected = ModelSpecificDecodeProbes::default();
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        for plan in &plans {
            for &repeat_layers in &plan.repeat_layers {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_dense_probe_start",
                    &format!(
                        "backend={:?} tensor_type={} hidden={} kv_width={} ffn={} layers={} graph_features={} norm_head_width={}",
                        accelerator.backend,
                        plan.tensor_type,
                        plan.hidden,
                        plan.kv_width,
                        plan.ffn,
                        repeat_layers,
                        plan.graph_features,
                        plan.norm_head_width,
                    ),
                );
                let _status = TerminalStatus::start(
                    args.show_progress,
                    format!(
                        "Probing model-shaped dense graph {} {} l{} {}x{} f{}",
                        model_label,
                        plan.tensor_type,
                        repeat_layers,
                        plan.ffn,
                        plan.hidden,
                        plan.graph_features
                    ),
                );
                match mesh_llm_gpu_bench::run_model_dense_graph_probe(
                    backend,
                    plan.tensor_type,
                    mesh_llm_gpu_bench::DenseGraphProbeShape {
                        hidden: plan.hidden,
                        kv_width: plan.kv_width,
                        ffn: plan.ffn,
                        repeat_layers,
                        graph_features: plan.graph_features,
                        norm_head_width: plan.norm_head_width,
                    },
                ) {
                    Ok(probes) => {
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_dense_probe_done",
                            &format!(
                                "tensor_type={} layers={} probes={}",
                                plan.tensor_type,
                                repeat_layers,
                                probes.len()
                            ),
                        );
                        collected.probes.extend(probes);
                    }
                    Err(error) => {
                        let message = format!(
                            "tensor_type={} layers={repeat_layers}: {error:#}",
                            plan.tensor_type
                        );
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_dense_probe_error",
                            &message,
                        );
                        collected.errors.push(message);
                    }
                }
            }
        }
    }
    collected
}

fn moe_model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let plans = moe_graph_probe_plans(profile);
    if plans.is_empty() {
        return ModelSpecificDecodeProbes {
            probes: Vec::new(),
            errors: vec![
                "could not derive model-shaped MoE graph probe dimensions from GGUF metadata"
                    .into(),
            ],
        };
    };
    let mut collected = ModelSpecificDecodeProbes::default();
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        for plan in &plans {
            for &repeat_layers in plan.repeat_layers {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_moe_probe_start",
                    &format!(
                        "backend={:?} tensor_type={} experts={} used={} expert_width={} hidden={} kv_width={} layers={}",
                        accelerator.backend,
                        plan.tensor_type,
                        plan.expert_count,
                        plan.experts_used,
                        plan.expert_width,
                        plan.hidden,
                        plan.kv_width,
                        repeat_layers
                    ),
                );
                let _status = TerminalStatus::start(
                    args.show_progress,
                    format!(
                        "Probing model-shaped MoE block graph {} {} l{} {}x{} kv{}",
                        model_label,
                        plan.tensor_type,
                        repeat_layers,
                        plan.expert_width,
                        plan.hidden,
                        plan.kv_width
                    ),
                );
                match mesh_llm_gpu_bench::run_model_moe_block_graph_probe(
                    backend,
                    plan.tensor_type,
                    mesh_llm_gpu_bench::MoeBlockGraphProbeShape {
                        expert_count: plan.expert_count,
                        experts_used: plan.experts_used,
                        expert_width: plan.expert_width,
                        hidden: plan.hidden,
                        kv_width: plan.kv_width,
                        repeat_layers,
                    },
                ) {
                    Ok(probes) => {
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_moe_probe_done",
                            &format!(
                                "tensor_type={} layers={} probes={}",
                                plan.tensor_type,
                                repeat_layers,
                                probes.len()
                            ),
                        );
                        collected.probes.extend(probes);
                    }
                    Err(error) => {
                        let message = format!(
                            "tensor_type={} layers={repeat_layers}: {error:#}",
                            plan.tensor_type
                        );
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_moe_probe_error",
                            &message,
                        );
                        collected.errors.push(message);
                    }
                }
            }
        }
    }
    collected
}

fn model_probe_label(profile: &ModelProfile) -> String {
    profile
        .source
        .metadata_name
        .clone()
        .unwrap_or_else(|| profile.source.id.clone())
}

fn append_model_output_projection_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
    collected: &mut ModelSpecificDecodeProbes,
) {
    let Some(plan) = output_projection_probe_plan(profile) else {
        return;
    };
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        heartbeat(
            Some(model_index),
            &model_label,
            "model_output_projection_probe_start",
            &format!(
                "backend={:?} tensor_type={} vocab={} hidden={}",
                accelerator.backend, plan.tensor_type, plan.vocab, plan.hidden
            ),
        );
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Probing model-shaped output projection {} {} {}x{}",
                model_label, plan.tensor_type, plan.vocab, plan.hidden
            ),
        );
        match mesh_llm_gpu_bench::run_model_output_projection_probe(
            backend,
            plan.tensor_type,
            mesh_llm_gpu_bench::OutputProjectionProbeShape {
                hidden: plan.hidden,
                vocab: plan.vocab,
            },
        ) {
            Ok(probes) => {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_output_projection_probe_done",
                    &format!("tensor_type={} probes={}", plan.tensor_type, probes.len()),
                );
                collected.probes.extend(probes);
            }
            Err(error) => {
                let message = format!("output tensor_type={}: {error:#}", plan.tensor_type);
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_output_projection_probe_error",
                    &message,
                );
                collected.errors.push(message);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MoeGraphProbePlan {
    tensor_type: &'static str,
    expert_count: u32,
    experts_used: u32,
    expert_width: u32,
    hidden: u32,
    kv_width: u32,
    repeat_layers: &'static [u32],
}

#[derive(Clone, Debug)]
struct DenseGraphProbePlan {
    tensor_type: &'static str,
    hidden: u32,
    kv_width: u32,
    ffn: u32,
    graph_features: u32,
    norm_head_width: u32,
    repeat_layers: Vec<u32>,
}

#[derive(Clone, Debug)]
struct LinearAttentionGraphProbePlan {
    tensor_type: &'static str,
    hidden: u32,
    qkv_width: u32,
    gate_width: u32,
    state_width: u32,
    output_input_width: u32,
    ffn: u32,
    recurrent_layers: u32,
    full_attention_layers: u32,
    kv_width: u32,
    graph_features: u32,
    norm_head_width: u32,
}

#[derive(Clone, Copy, Debug)]
struct OutputProjectionProbePlan {
    tensor_type: &'static str,
    hidden: u32,
    vocab: u32,
}

fn output_projection_probe_plan(profile: &ModelProfile) -> Option<OutputProjectionProbePlan> {
    let bytes = output_projection_probe_bytes(profile);
    if bytes == 0 {
        return None;
    }
    let hidden = profile
        .hidden_size
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            u32::try_from(profile.tensor_matmul.output.shape.max_input_width)
                .ok()
                .filter(|width| *width > 0)
        })?;
    let vocab = profile
        .tokenizer
        .vocab_size
        .filter(|vocab| *vocab > 0)
        .or_else(|| {
            u32::try_from(profile.tensor_matmul.output.shape.max_output_width)
                .ok()
                .filter(|width| *width > 0)
        })?;
    let tensor_type = output_projection_probe_tensor_type(profile)
        .or_else(|| dense_probe_tensor_type_from_quant(profile.quantization.as_deref()))?;
    Some(OutputProjectionProbePlan {
        tensor_type,
        hidden,
        vocab,
    })
}

fn output_projection_probe_bytes(profile: &ModelProfile) -> u64 {
    if profile.tensor_matmul.output.bytes > 0 || profile.tensor_group_bytes.output_bytes > 0 {
        return profile
            .tensor_matmul
            .output
            .bytes
            .max(profile.tensor_group_bytes.output_bytes);
    }
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::DenseTransformer
        | model_fit::ModelArchitectureClass::SparseMoeTransformer
        | model_fit::ModelArchitectureClass::Unknown => profile.tensor_group_bytes.embedding_bytes,
        _ => 0,
    }
}

fn output_projection_probe_tensor_type(profile: &ModelProfile) -> Option<&'static str> {
    if profile.tensor_matmul.output.bytes > 0 || profile.tensor_group_bytes.output_bytes > 0 {
        return dominant_supported_tensor_type(profile.tensor_matmul.output.type_bytes);
    }
    dominant_supported_tensor_type(profile.tensor_group_bytes.embedding_type_bytes)
}

fn dominant_supported_tensor_type(bytes: TensorTypeBytes) -> Option<&'static str> {
    let mut candidates = [
        ("f16", bytes.f16_bytes),
        ("q4_k", bytes.q4_k_bytes),
        ("q6_k", bytes.q6_k_bytes),
        ("q8_0", bytes.q8_0_bytes),
    ];
    candidates.sort_by(|(_, left), (_, right)| right.cmp(left));
    candidates
        .into_iter()
        .find_map(|(kind, bytes)| (bytes > 0).then_some(kind))
}

fn linear_attention_graph_probe_plans(
    profile: &ModelProfile,
) -> Vec<LinearAttentionGraphProbePlan> {
    if !has_recurrent_attention_profile(profile) {
        return Vec::new();
    }
    let recurrent = &profile.recurrent_attention;
    let Some(hidden) = profile.hidden_size.filter(|hidden| *hidden > 0) else {
        return Vec::new();
    };
    let Some(ffn) = profile.ffn_size.filter(|ffn| *ffn > 0).or_else(|| {
        u32::try_from(
            profile
                .tensor_matmul
                .feed_forward
                .shape
                .weighted_avg_output_width
                .max(profile.tensor_matmul.feed_forward.shape.max_output_width),
        )
        .ok()
        .filter(|width| *width > 0)
    }) else {
        return Vec::new();
    };
    let Some(qkv_width) = recurrent_projection_output_width(&recurrent.qkv_projection) else {
        return Vec::new();
    };
    let Some(gate_width) = recurrent_projection_output_width(&recurrent.gate_projection) else {
        return Vec::new();
    };
    let Some(state_width) = recurrent_projection_output_width(&recurrent.beta_projection).max(
        recurrent_projection_output_width(&recurrent.alpha_projection),
    ) else {
        return Vec::new();
    };
    let Some(output_input_width) = recurrent_projection_input_width(&recurrent.output_projection)
    else {
        return Vec::new();
    };
    if output_input_width > qkv_width {
        return Vec::new();
    }
    let recurrent_layers = recurrent.recurrent_layer_count.max(1);
    let full_attention_layers = profile
        .layer_count
        .unwrap_or(recurrent_layers)
        .saturating_sub(recurrent_layers);
    let mut tensor_types = dense_probe_tensor_types(profile);
    tensor_types.dedup();
    tensor_types
        .into_iter()
        .map(|tensor_type| LinearAttentionGraphProbePlan {
            tensor_type,
            hidden,
            qkv_width,
            gate_width,
            state_width,
            output_input_width,
            ffn,
            recurrent_layers,
            full_attention_layers,
            kv_width: dense_probe_kv_width(profile, hidden),
            graph_features: dense_probe_graph_features(profile),
            norm_head_width: dense_probe_norm_head_width(profile),
        })
        .collect()
}

fn has_recurrent_attention_profile(profile: &ModelProfile) -> bool {
    let recurrent = &profile.recurrent_attention;
    recurrent.recurrent_layer_count > 0
        && recurrent.qkv_projection.shape.tensor_count > 0
        && recurrent.gate_projection.shape.tensor_count > 0
        && recurrent.output_projection.shape.tensor_count > 0
}

fn recurrent_projection_input_width(group: &model_fit::TensorMatmulGroupProfile) -> Option<u32> {
    u32::try_from(group.shape.max_input_width)
        .ok()
        .filter(|width| *width > 0)
}

fn recurrent_projection_output_width(group: &model_fit::TensorMatmulGroupProfile) -> Option<u32> {
    u32::try_from(group.shape.max_output_width)
        .ok()
        .filter(|width| *width > 0)
}

fn dense_graph_probe_plans(args: &Args, profile: &ModelProfile) -> Vec<DenseGraphProbePlan> {
    let hidden = profile
        .hidden_size
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            let shape = profile.tensor_matmul.attention.shape;
            u32::try_from(shape.max_input_width.max(shape.max_output_width))
                .ok()
                .filter(|width| *width > 0)
        });
    let ffn = profile.ffn_size.filter(|ffn| *ffn > 0).or_else(|| {
        let shape = profile.tensor_matmul.feed_forward.shape;
        u32::try_from(
            shape
                .weighted_avg_output_width
                .max(shape.max_output_width)
                .max(shape.max_input_width),
        )
        .ok()
        .filter(|width| *width > 0)
    });
    let (Some(hidden), Some(ffn)) = (hidden, ffn) else {
        return Vec::new();
    };
    let mut tensor_types = dense_probe_tensor_types(profile);
    tensor_types.dedup();
    tensor_types
        .into_iter()
        .map(|tensor_type| DenseGraphProbePlan {
            tensor_type,
            hidden,
            kv_width: dense_probe_kv_width(profile, hidden),
            ffn,
            graph_features: dense_probe_graph_features(profile),
            norm_head_width: dense_probe_norm_head_width(profile),
            repeat_layers: dense_probe_repeat_layers(args, profile, tensor_type),
        })
        .collect()
}

fn dense_probe_norm_head_width(profile: &ModelProfile) -> u32 {
    profile
        .key_length
        .filter(|width| *width > 0)
        .or_else(|| {
            let hidden = profile.hidden_size?;
            let heads = profile.attention_heads.filter(|heads| *heads > 0)?;
            (hidden % heads == 0).then_some(hidden / heads)
        })
        .unwrap_or_default()
}

fn dense_probe_graph_features(profile: &ModelProfile) -> u32 {
    let mut features = 0;
    if profile.dense_graph_features.attention_q_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_Q_NORM;
    }
    if profile.dense_graph_features.attention_k_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_K_NORM;
    }
    if profile.dense_graph_features.attention_post_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_POST_NORM;
    }
    if profile.dense_graph_features.feed_forward_post_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_FFN_POST_NORM;
    }
    features
}

fn dense_probe_repeat_layers(args: &Args, profile: &ModelProfile, tensor_type: &str) -> Vec<u32> {
    if !tensor_type.eq_ignore_ascii_case("q4_k") && !tensor_type.eq_ignore_ascii_case("q8_0") {
        return vec![1];
    }

    let mut layers = match (
        args.dense_probe_depth,
        tensor_type.eq_ignore_ascii_case("q8_0"),
    ) {
        (DenseProbeDepth::Standard, true) => vec![1],
        (DenseProbeDepth::Deep, true) => vec![1, 4, 8],
        (DenseProbeDepth::Standard, false) => vec![1, 4, 8],
        (DenseProbeDepth::Deep, false) => vec![1, 4, 8, 16],
    };

    if let Some(model_layers) = profile.layer_count.filter(|count| *count > 0) {
        if args.dense_probe_depth == DenseProbeDepth::Standard
            && tensor_type.eq_ignore_ascii_case("q4_k")
        {
            add_standard_dense_depth_probes(&mut layers, model_layers);
        }
        if args.dense_probe_depth == DenseProbeDepth::Deep {
            // Deep validation is allowed to spend extra time building a
            // source-shaped synthetic graph whose layer count comes directly
            // from GGUF metadata. This is not an observed-throughput feedback
            // path: it does not load or run the real model weights, and it
            // does not consume the ABI/full-model tok/s result. Its purpose is
            // to falsify extrapolation from shallower graph probes to the
            // actual model depth.
            layers.push(model_layers);
        }
    }

    layers.sort_unstable();
    layers.dedup();
    layers
}

fn add_standard_dense_depth_probes(layers: &mut Vec<u32>, model_layers: u32) {
    // llama.cpp decode does not run one isolated matmul per layer. It submits a
    // whole one-token graph containing repeated attention/FFN matmuls, KV
    // reads/writes, normalization, residuals, output projection, backend graph
    // optimization, and command scheduling. Metal in particular can amortize
    // source-shaped graphs very differently between l8, l16, and full model
    // depth, while CUDA often stays closer to linear scaling. That difference
    // is a measured property of the backend graph, not a backend name rule.
    //
    // The default validator therefore collects enough synthetic graph depth to
    // falsify the old "scale l8 linearly to the whole model" assumption for
    // medium Q4_K dense models. These probes are still metadata-only: the
    // synthetic graph shape comes from GGUF fields such as layer count, hidden
    // width, KV width, FFN width, tensor type, and norm/head features. We do
    // not load the real model weights and we never feed observed tok/s back
    // into scoring.
    //
    // Q8_0 deliberately stays on the older shallow default until we have
    // broader held-out evidence. A small Q8 model can be dominated by runtime
    // and sampling overhead rather than transformer matmul depth, so admitting
    // a full-depth Q8 synthetic graph by default can overstate throughput even
    // though the synthetic graph itself is honest. The explicit `deep` mode
    // still collects those rows as diagnostics.
    if model_layers >= 16 {
        layers.push(16);
    }

    if model_layers <= 32 {
        layers.push(model_layers);
    }
}

fn dense_probe_kv_width(profile: &ModelProfile, hidden: u32) -> u32 {
    let key_width = dense_probe_kv_vector_width(profile, profile.key_length, hidden);
    let value_width = dense_probe_kv_vector_width(profile, profile.value_length, hidden);
    key_width.max(value_width).max(1)
}

fn dense_probe_kv_vector_width(
    profile: &ModelProfile,
    vector_length: Option<u32>,
    hidden: u32,
) -> u32 {
    match (profile.kv_heads, vector_length) {
        (Some(kv_heads), Some(length)) => kv_heads.saturating_mul(length).max(1),
        _ => hidden,
    }
}

fn dense_probe_tensor_types(profile: &ModelProfile) -> Vec<&'static str> {
    let bytes = add_tensor_type_bytes(
        profile.tensor_matmul.attention.type_bytes,
        profile.tensor_matmul.feed_forward.type_bytes,
    );
    let mut candidates = [
        ("f16", bytes.f16_bytes),
        ("q4_k", bytes.q4_k_bytes),
        ("q6_k", bytes.q6_k_bytes),
        ("q8_0", bytes.q8_0_bytes),
    ];
    candidates.sort_by(|(_, left), (_, right)| right.cmp(left));
    let mut tensor_types = candidates
        .into_iter()
        .filter_map(|(kind, bytes)| (bytes > 0).then_some(kind))
        .collect::<Vec<_>>();
    if let (true, Some(tensor_type)) = (
        tensor_types.is_empty(),
        dense_probe_tensor_type_from_quant(profile.quantization.as_deref()),
    ) {
        tensor_types.push(tensor_type);
    }
    tensor_types
}

fn dense_probe_tensor_type_from_quant(quantization: Option<&str>) -> Option<&'static str> {
    let quantization = quantization?.to_ascii_lowercase();
    if quantization.contains("q4_k") {
        Some("q4_k")
    } else if quantization.contains("q6_k") {
        Some("q6_k")
    } else if quantization.contains("q8_0") || quantization.contains("q8") {
        Some("q8_0")
    } else if quantization.contains("f16") {
        Some("f16")
    } else {
        None
    }
}

fn add_tensor_type_bytes(left: TensorTypeBytes, right: TensorTypeBytes) -> TensorTypeBytes {
    TensorTypeBytes {
        f32_bytes: left.f32_bytes.saturating_add(right.f32_bytes),
        f16_bytes: left.f16_bytes.saturating_add(right.f16_bytes),
        bf16_bytes: left.bf16_bytes.saturating_add(right.bf16_bytes),
        q4_0_bytes: left.q4_0_bytes.saturating_add(right.q4_0_bytes),
        q4_k_bytes: left.q4_k_bytes.saturating_add(right.q4_k_bytes),
        q5_k_bytes: left.q5_k_bytes.saturating_add(right.q5_k_bytes),
        q6_k_bytes: left.q6_k_bytes.saturating_add(right.q6_k_bytes),
        q8_0_bytes: left.q8_0_bytes.saturating_add(right.q8_0_bytes),
        iq_bytes: left.iq_bytes.saturating_add(right.iq_bytes),
        other_quantized_bytes: left
            .other_quantized_bytes
            .saturating_add(right.other_quantized_bytes),
        unknown_bytes: left.unknown_bytes.saturating_add(right.unknown_bytes),
    }
}

fn moe_graph_probe_plans(profile: &ModelProfile) -> Vec<MoeGraphProbePlan> {
    let Some(expert_count) = profile.expert_count.filter(|count| *count > 0) else {
        return Vec::new();
    };
    let experts_used = profile
        .expert_used_count
        .filter(|used| *used > 0)
        .unwrap_or(expert_count)
        .min(expert_count);
    let Some(hidden) = profile
        .hidden_size
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            let shape = profile.tensor_matmul.expert_feed_forward.shape;
            u32::try_from(shape.max_input_width.max(shape.max_output_width))
                .ok()
                .filter(|width| *width > 0)
        })
    else {
        return Vec::new();
    };
    let Some(expert_width) = profile.ffn_size.filter(|ffn| *ffn > 0).or_else(|| {
        let shape = profile.tensor_matmul.expert_feed_forward.shape;
        u32::try_from(shape.min_input_width.min(shape.min_output_width))
            .ok()
            .filter(|width| *width > 0)
    }) else {
        return Vec::new();
    };
    let kv_width = model_attention_kv_width(profile).min(u64::from(hidden));
    let Ok(kv_width) = u32::try_from(kv_width.max(1)) else {
        return Vec::new();
    };
    moe_probe_tensor_types(profile)
        .into_iter()
        .map(|tensor_type| MoeGraphProbePlan {
            tensor_type,
            expert_count,
            experts_used,
            expert_width,
            hidden,
            kv_width,
            repeat_layers: &[1, 4, 8],
        })
        .collect()
}

fn model_attention_kv_width(profile: &ModelProfile) -> u64 {
    let key_width = model_kv_width(profile, profile.key_length);
    let value_width = model_kv_width(profile, profile.value_length);
    key_width.max(value_width).max(1)
}

fn model_kv_width(profile: &ModelProfile, vector_length: Option<u32>) -> u64 {
    match (profile.kv_heads, vector_length) {
        (Some(kv_heads), Some(length)) => u64::from(kv_heads).saturating_mul(u64::from(length)),
        _ => u64::from(profile.hidden_size.unwrap_or(1)),
    }
}

fn moe_probe_tensor_types(profile: &ModelProfile) -> Vec<&'static str> {
    let bytes = profile.tensor_matmul.expert_feed_forward.type_bytes;
    let mut candidates = [("q4_k", bytes.q4_k_bytes), ("q6_k", bytes.q6_k_bytes)];
    candidates.sort_by(|(_, left), (_, right)| right.cmp(left));
    candidates
        .into_iter()
        .filter_map(|(kind, bytes)| (bytes > 0).then_some(kind))
        .collect()
}

fn gpu_bench_backend(backend: BackendKind) -> Option<mesh_llm_gpu_bench::BenchmarkBackend> {
    match backend {
        BackendKind::Metal => Some(mesh_llm_gpu_bench::BenchmarkBackend::Metal),
        BackendKind::Cuda => Some(mesh_llm_gpu_bench::BenchmarkBackend::Cuda),
        BackendKind::Rocm => Some(mesh_llm_gpu_bench::BenchmarkBackend::Hip),
        _ => None,
    }
}

fn fit_interpretation(recommendation: &ModelRecommendation) -> FitInterpretation {
    let local_accelerated_fit = matches!(
        recommendation.fit_status,
        FitStatus::FitsLocal | FitStatus::FitsWithWarning
    ) && recommendation.selected_backend != BackendKind::Cpu;
    let single_node_validation_allowed = matches!(
        recommendation.fit_status,
        FitStatus::FitsLocal | FitStatus::FitsWithWarning
    );
    let summary = match recommendation.fit_status {
        FitStatus::FitsLocal => "fits local selected backend".into(),
        FitStatus::FitsWithWarning => "fits local selected backend with warnings".into(),
        FitStatus::Rejected => "does not fit local selected backend".into(),
    };
    let mut details = Vec::new();
    match recommendation.fit_status {
        FitStatus::Rejected => {
            details.push(
                "validation is skipped by default because no local serving shape was selected"
                    .into(),
            );
        }
        _ => {
            details.push(format!(
                "selected backend {:?} is the local validation target",
                recommendation.selected_backend
            ));
        }
    }
    FitInterpretation {
        local_accelerated_fit,
        single_node_validation_allowed,
        summary,
        details,
    }
}

fn runtime_diagnostic(
    profile: &ModelProfile,
    recommendation: &ModelRecommendation,
    benchmarks: &[BenchmarkScenarioSummary],
) -> RuntimeDiagnostic {
    // This diagnostic records the Skippy single-stage shape used for
    // validation. It is intentionally separate from the fit estimate. When a
    // model misses, the first question is whether the observed runtime used a
    // different launch shape than the metadata estimator assumed: partial layer
    // loading, explicit CPU fallback, lower KV precision, flash-attention
    // override, or a batch/ubatch override. Capturing those knobs makes
    // anomalies reproducible without feeding benchmark results back into
    // model-fit scoring.
    let steady_decode_command = benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == "steady_decode")
        .and_then(|benchmark| benchmark.benchmark.observations.first())
        .map(|observation| observation.command.clone());
    RuntimeDiagnostic {
        validation_shape: "skippy-bench local-single full-model runtime-slice",
        selected_backend: format!("{:?}", recommendation.selected_backend),
        selected_accelerator: recommendation.selected_accelerator.clone(),
        layer_start: 0,
        layer_end: profile.layer_count,
        ctx_size: DEFAULT_CTX_SIZE,
        n_gpu_layers: -1,
        cache_type_k: "f16",
        cache_type_v: "f16",
        flash_attn_type: "auto",
        n_batch: None,
        n_ubatch: None,
        load_mode: "runtime-slice",
        filter_tensors_on_load: false,
        include_embeddings: true,
        include_output: true,
        steady_decode_command,
        notes: vec![
            "n_gpu_layers=-1 asks llama.cpp/Skippy to offload as much as the selected backend can support.".into(),
            "No validator-level n_batch or n_ubatch override is passed; defaults come from the native runtime.".into(),
            "Metal/CUDA kernel selection is not yet exposed in this report; use native GGML/llama logging or ABI hooks for that next layer.".into(),
        ],
    }
}

async fn prepare_model(
    args: &Args,
    repository: &HfModelRepository,
    input: &ModelInput,
    model_index: usize,
) -> Result<PreparedModel> {
    match input {
        ModelInput::Ref(model_ref) => {
            prepare_model_ref(args, repository, model_ref, model_index).await
        }
        ModelInput::Local(local) => prepare_local_model(args, local, model_index),
    }
}

async fn prepare_model_ref(
    args: &Args,
    repository: &HfModelRepository,
    model_ref: &str,
    model_index: usize,
) -> Result<PreparedModel> {
    heartbeat(
        Some(model_index),
        model_ref,
        "resolve_start",
        "resolving model artifact",
    );
    let artifact = {
        let _status = TerminalStatus::start(args.show_progress, format!("Resolving {model_ref}"));
        resolve_model_artifact_ref(model_ref, repository)
            .await
            .with_context(|| format!("resolve model ref {model_ref}"))?
    };
    heartbeat(
        Some(model_index),
        model_ref,
        "resolve_done",
        &format!("canonical_ref={}", artifact.canonical_ref),
    );
    if artifact.format != ModelFormat::Gguf {
        bail!(
            "{model_ref} resolved to {:?}, expected GGUF",
            artifact.format
        );
    }
    heartbeat(
        Some(model_index),
        model_ref,
        "download_start",
        "ensuring GGUF artifact is available",
    );
    let progress = download_progress(args, model_ref);
    let downloaded_paths = repository
        .download_artifact_files_with_progress(&artifact, progress)
        .await
        .with_context(|| format!("download model ref {model_ref}"))?;
    heartbeat(
        Some(model_index),
        model_ref,
        "download_done",
        &format!("files={}", downloaded_paths.len()),
    );
    let primary_gguf_path = primary_download_path(&artifact, &downloaded_paths)?;
    heartbeat(
        Some(model_index),
        model_ref,
        "profile_start",
        &format!("path={}", primary_gguf_path.display()),
    );
    let mut profile = {
        let _status = TerminalStatus::start(args.show_progress, format!("Profiling {model_ref}"));
        profile_gguf_path(&primary_gguf_path)?
    };
    heartbeat(
        Some(model_index),
        model_ref,
        "profile_done",
        &profile_summary(&profile),
    );
    profile.source.id = model_ref.to_string();
    profile.source.path = Some(primary_gguf_path.clone());

    Ok(PreparedModel {
        input_ref: model_ref.to_string(),
        resolved_ref: Some(artifact.canonical_ref.clone()),
        artifact: Some(artifact),
        downloaded_paths,
        primary_gguf_path,
        profile,
    })
}

fn prepare_local_model(
    args: &Args,
    local: &LocalModelInput,
    model_index: usize,
) -> Result<PreparedModel> {
    heartbeat(
        Some(model_index),
        &local.model_ref,
        "profile_start",
        &format!("path={}", local.gguf_path.display()),
    );
    let mut profile = {
        let _status =
            TerminalStatus::start(args.show_progress, format!("Profiling {}", local.model_ref));
        profile_gguf_path(&local.gguf_path)?
    };
    heartbeat(
        Some(model_index),
        &local.model_ref,
        "profile_done",
        &profile_summary(&profile),
    );
    profile.source.id = local.model_ref.clone();
    profile.source.path = Some(local.gguf_path.clone());
    Ok(PreparedModel {
        input_ref: local.model_ref.clone(),
        resolved_ref: None,
        artifact: None,
        downloaded_paths: vec![local.gguf_path.clone()],
        primary_gguf_path: local.gguf_path.clone(),
        profile,
    })
}

fn run_abi_decode_probe_for_recommendation(
    args: &Args,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
) -> AbiDecodeProbeSummary {
    if let Some(reason) = abi_decode_probe_skip_reason(args, recommendation) {
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "abi_probe_skip",
            &reason,
        );
        return skipped_abi_decode_probe(reason);
    }
    run_abi_decode_probe(args, model, model_index)
}

fn abi_decode_probe_skip_reason(
    args: &Args,
    recommendation: &ModelRecommendation,
) -> Option<String> {
    if !args.benchmark_all
        && !matches!(
            recommendation.fit_status,
            FitStatus::FitsLocal | FitStatus::FitsWithWarning
        )
    {
        return Some(format!(
            "fit status is {:?}; use --benchmark-all to force single-stage ABI decode probe",
            recommendation.fit_status
        ));
    }
    if !args.benchmark_all && recommendation.selected_backend == BackendKind::Cpu {
        return Some(
            "fit selected CPU backend; use --benchmark-all to force the single-stage ABI decode probe"
                .into(),
        );
    }
    None
}

fn skipped_abi_decode_probe(reason: String) -> AbiDecodeProbeSummary {
    AbiDecodeProbeSummary {
        attempted: false,
        skip_reason: Some(reason),
        tokens_per_second: None,
        elapsed_ms: None,
        llama_eval_tokens_per_second: None,
        llama_eval_ms: None,
        non_eval_overhead_ms: None,
        non_eval_overhead_pct: None,
        decode_call_tokens_per_second: None,
        decode_call_ms: None,
        sampling_tokens_per_second: None,
        sampling_ms: None,
        llama_eval_count: None,
        llama_graph_reuse_count: None,
        graph_node_count: None,
        graph_inventory_bucket_overflow_count: None,
        graph_inventory: Vec::new(),
        measured_tokens: None,
        prompt_token_count: None,
        command: Vec::new(),
        observations: Vec::new(),
        sample_count: 0,
        raw_sample_count: 0,
        min_tokens_per_second: None,
        max_tokens_per_second: None,
        spread_pct: None,
        raw_spread_pct: None,
        denoised_outlier_count: 0,
        error: None,
    }
}

fn run_abi_decode_probe(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
) -> AbiDecodeProbeSummary {
    let mut summary = AbiDecodeProbeSummary {
        attempted: true,
        skip_reason: None,
        tokens_per_second: None,
        elapsed_ms: None,
        llama_eval_tokens_per_second: None,
        llama_eval_ms: None,
        non_eval_overhead_ms: None,
        non_eval_overhead_pct: None,
        decode_call_tokens_per_second: None,
        decode_call_ms: None,
        sampling_tokens_per_second: None,
        sampling_ms: None,
        llama_eval_count: None,
        llama_graph_reuse_count: None,
        graph_node_count: None,
        graph_inventory_bucket_overflow_count: None,
        graph_inventory: Vec::new(),
        measured_tokens: None,
        prompt_token_count: None,
        command: Vec::new(),
        observations: Vec::new(),
        sample_count: 0,
        raw_sample_count: 0,
        min_tokens_per_second: None,
        max_tokens_per_second: None,
        spread_pct: None,
        raw_spread_pct: None,
        denoised_outlier_count: 0,
        error: None,
    };
    let Some(layer_count) = model.profile.layer_count else {
        summary.error = Some("model metadata did not include layer count".into());
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "abi_probe_skip",
            "model metadata did not include layer count",
        );
        return summary;
    };
    let command_args = vec![
        "abi-decode-probe".to_string(),
        "--model-path".to_string(),
        model.primary_gguf_path.display().to_string(),
        "--ctx-size".to_string(),
        DEFAULT_CTX_SIZE.to_string(),
        "--n-gpu-layers=-1".to_string(),
        "--layer-end".to_string(),
        layer_count.to_string(),
        "--prompt".to_string(),
        validation_prompt().to_string(),
        "--warmup-tokens".to_string(),
        DEFAULT_WARMUP_TOKENS.to_string(),
        "--measured-tokens".to_string(),
        DEFAULT_ABI_DECODE_MEASURED_TOKENS.to_string(),
    ];
    summary.command = command_display(&args.skippy_bench_bin, &command_args);
    heartbeat(
        Some(model_index),
        &model.input_ref,
        "abi_probe_start",
        &format!(
            "repeats={} measured_tokens={}",
            DEFAULT_ABI_DECODE_REPEATS, DEFAULT_ABI_DECODE_MEASURED_TOKENS
        ),
    );
    for repeat in 0..DEFAULT_ABI_DECODE_REPEATS {
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "abi_probe_repeat_start",
            &format!("repeat={}", repeat + 1),
        );
        summary.observations.push(run_abi_decode_probe_once(
            args,
            &model.input_ref,
            model_index,
            &command_args,
            repeat,
        ));
        if let Some(observation) = summary.observations.last() {
            heartbeat(
                Some(model_index),
                &model.input_ref,
                "abi_probe_repeat_done",
                &abi_probe_observation_detail(observation),
            );
            if fatal_abi_probe_observation(observation) {
                heartbeat(
                    Some(model_index),
                    &model.input_ref,
                    "abi_probe_repeats_abort",
                    "aborting ABI decode probe repeats after runtime startup failure",
                );
                break;
            }
        }
    }
    let summary = finalize_abi_decode_probe_summary(summary);
    heartbeat(
        Some(model_index),
        &model.input_ref,
        "abi_probe_done",
        &format!(
            "tok_s={} sample_count={} error={}",
            display_opt(summary.tokens_per_second),
            summary.sample_count,
            summary.error.as_deref().unwrap_or("-")
        ),
    );
    summary
}

fn run_abi_decode_probe_once(
    args: &Args,
    model_ref: &str,
    model_index: usize,
    command_args: &[String],
    repeat: usize,
) -> AbiDecodeProbeObservation {
    let command = command_display(&args.skippy_bench_bin, command_args);
    let output = Command::new(&args.skippy_bench_bin)
        .args(command_args)
        .output();
    match output {
        Ok(output) if output.status.success() => {
            match parse_abi_decode_probe_json(&output.stdout) {
                Ok(parsed) => AbiDecodeProbeObservation {
                    repeat,
                    command,
                    status_code: output.status.code(),
                    tokens_per_second: parsed.tokens_per_second,
                    elapsed_ms: parsed.elapsed_ms,
                    llama_eval_tokens_per_second: parsed.llama_eval_tokens_per_second,
                    llama_eval_ms: parsed.llama_eval_ms,
                    non_eval_overhead_ms: parsed.non_eval_overhead_ms,
                    decode_call_tokens_per_second: parsed.decode_call_tokens_per_second,
                    decode_call_ms: parsed.decode_call_ms,
                    sampling_tokens_per_second: parsed.sampling_tokens_per_second,
                    sampling_ms: parsed.sampling_ms,
                    llama_eval_count: parsed.llama_eval_count,
                    llama_graph_reuse_count: parsed.llama_graph_reuse_count,
                    graph_node_count: parsed.graph_node_count,
                    graph_inventory_bucket_overflow_count: parsed
                        .graph_inventory_bucket_overflow_count,
                    graph_inventory: parsed.graph_inventory,
                    measured_tokens: parsed.measured_tokens,
                    prompt_token_count: parsed.prompt_token_count,
                    stderr_tail: stderr_tail(&output.stderr),
                    error: None,
                },
                Err(err) => AbiDecodeProbeObservation {
                    repeat,
                    command,
                    status_code: output.status.code(),
                    tokens_per_second: None,
                    elapsed_ms: None,
                    llama_eval_tokens_per_second: None,
                    llama_eval_ms: None,
                    non_eval_overhead_ms: None,
                    decode_call_tokens_per_second: None,
                    decode_call_ms: None,
                    sampling_tokens_per_second: None,
                    sampling_ms: None,
                    llama_eval_count: None,
                    llama_graph_reuse_count: None,
                    graph_node_count: None,
                    graph_inventory_bucket_overflow_count: None,
                    graph_inventory: Vec::new(),
                    measured_tokens: None,
                    prompt_token_count: None,
                    stderr_tail: stderr_tail(&output.stderr),
                    error: Some(err),
                },
            }
        }
        Ok(output) => AbiDecodeProbeObservation {
            repeat,
            command,
            status_code: output.status.code(),
            tokens_per_second: None,
            elapsed_ms: None,
            llama_eval_tokens_per_second: None,
            llama_eval_ms: None,
            non_eval_overhead_ms: None,
            decode_call_tokens_per_second: None,
            decode_call_ms: None,
            sampling_tokens_per_second: None,
            sampling_ms: None,
            llama_eval_count: None,
            llama_graph_reuse_count: None,
            graph_node_count: None,
            graph_inventory_bucket_overflow_count: None,
            graph_inventory: Vec::new(),
            measured_tokens: None,
            prompt_token_count: None,
            stderr_tail: stderr_tail(&output.stderr),
            error: Some(format!(
                "abi decode probe exited with status {}",
                output.status.code().unwrap_or(-1)
            )),
        },
        Err(err) => {
            heartbeat(
                Some(model_index),
                model_ref,
                "abi_probe_start_error",
                &format!("repeat={} error={err}", repeat + 1),
            );
            AbiDecodeProbeObservation {
                repeat,
                command,
                status_code: None,
                tokens_per_second: None,
                elapsed_ms: None,
                llama_eval_tokens_per_second: None,
                llama_eval_ms: None,
                non_eval_overhead_ms: None,
                decode_call_tokens_per_second: None,
                decode_call_ms: None,
                sampling_tokens_per_second: None,
                sampling_ms: None,
                llama_eval_count: None,
                llama_graph_reuse_count: None,
                graph_node_count: None,
                graph_inventory_bucket_overflow_count: None,
                graph_inventory: Vec::new(),
                measured_tokens: None,
                prompt_token_count: None,
                stderr_tail: None,
                error: Some(format!("failed to start abi decode probe: {err}")),
            }
        }
    }
}

fn finalize_abi_decode_probe_summary(mut summary: AbiDecodeProbeSummary) -> AbiDecodeProbeSummary {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.tokens_per_second)
        .collect::<Vec<_>>();
    summary.raw_sample_count = samples.len();
    if samples.is_empty() {
        summary.error = Some("all abi decode probe repeats failed".into());
        return summary;
    }

    let stats = throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD);
    let median = stats.clean_median.expect("non-empty ABI sample stats");
    summary.tokens_per_second = Some(median);
    summary.sample_count = stats.clean_sample_count;
    summary.min_tokens_per_second = stats.clean_min;
    summary.max_tokens_per_second = stats.clean_max;
    summary.spread_pct = stats.clean_spread.map(|spread| spread * 100.0);
    summary.raw_spread_pct = stats.raw_spread.map(|spread| spread * 100.0);
    summary.denoised_outlier_count = stats.outlier_count;
    summary.measured_tokens = first_abi_measured_tokens(&summary);
    summary.prompt_token_count = first_abi_prompt_tokens(&summary);
    summary.elapsed_ms = summary
        .measured_tokens
        .map(|tokens| tokens as f64 * 1000.0 / median);
    summary.llama_eval_tokens_per_second = median_abi_llama_eval_tokens_per_second(&summary);
    summary.decode_call_tokens_per_second = median_abi_decode_call_tokens_per_second(&summary);
    summary.sampling_tokens_per_second = median_abi_sampling_tokens_per_second(&summary);
    summary.llama_eval_ms = match (
        summary.measured_tokens,
        summary.llama_eval_tokens_per_second,
    ) {
        (Some(tokens), Some(tps)) if tps > 0.0 => Some(tokens as f64 * 1000.0 / tps),
        _ => None,
    };
    summary.decode_call_ms = match (
        summary.measured_tokens,
        summary.decode_call_tokens_per_second,
    ) {
        (Some(tokens), Some(tps)) if tps > 0.0 => Some(tokens as f64 * 1000.0 / tps),
        _ => first_abi_decode_call_ms(&summary),
    };
    summary.sampling_ms = match (summary.measured_tokens, summary.sampling_tokens_per_second) {
        (Some(tokens), Some(tps)) if tps > 0.0 => Some(tokens as f64 * 1000.0 / tps),
        _ => first_abi_sampling_ms(&summary),
    };
    summary.non_eval_overhead_ms = match (
        summary.elapsed_ms,
        summary.decode_call_ms,
        summary.sampling_ms,
    ) {
        (Some(elapsed_ms), Some(decode_ms), Some(sampling_ms)) => {
            Some((elapsed_ms - decode_ms - sampling_ms).max(0.0))
        }
        _ => match (summary.elapsed_ms, summary.llama_eval_ms) {
            (Some(elapsed_ms), Some(llama_eval_ms)) if llama_eval_ms > 0.0 => {
                Some((elapsed_ms - llama_eval_ms).max(0.0))
            }
            _ => first_abi_non_eval_overhead_ms(&summary),
        },
    };
    summary.non_eval_overhead_pct = match (summary.non_eval_overhead_ms, summary.elapsed_ms) {
        (Some(overhead_ms), Some(elapsed_ms)) if elapsed_ms > 0.0 => {
            Some(overhead_ms / elapsed_ms * 100.0)
        }
        _ => None,
    };
    summary.llama_eval_count = first_abi_llama_eval_count(&summary);
    summary.llama_graph_reuse_count = first_abi_llama_graph_reuse_count(&summary);
    summary.graph_node_count = first_abi_graph_node_count(&summary);
    summary.graph_inventory_bucket_overflow_count =
        first_abi_graph_inventory_bucket_overflow_count(&summary);
    summary.graph_inventory = first_abi_graph_inventory(&summary).unwrap_or_default();
    summary.error = abi_decode_probe_error(&summary);
    summary
}

#[derive(Clone, Debug)]
struct ParsedAbiDecodeProbe {
    tokens_per_second: Option<f64>,
    elapsed_ms: Option<f64>,
    llama_eval_tokens_per_second: Option<f64>,
    llama_eval_ms: Option<f64>,
    non_eval_overhead_ms: Option<f64>,
    decode_call_tokens_per_second: Option<f64>,
    decode_call_ms: Option<f64>,
    sampling_tokens_per_second: Option<f64>,
    sampling_ms: Option<f64>,
    llama_eval_count: Option<u64>,
    llama_graph_reuse_count: Option<i64>,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    graph_inventory: Vec<AbiGraphInventoryBucket>,
    measured_tokens: Option<u64>,
    prompt_token_count: Option<u64>,
}

fn parse_abi_decode_probe_json(stdout: &[u8]) -> Result<ParsedAbiDecodeProbe, String> {
    let value = serde_json::from_slice::<Value>(stdout)
        .map_err(|err| format!("parse abi decode probe JSON: {err}"))?;
    let tokens_per_second = value.get("tokens_per_second").and_then(Value::as_f64);
    if tokens_per_second.is_none() {
        return Err("abi decode probe omitted tokens_per_second".into());
    }
    Ok(ParsedAbiDecodeProbe {
        tokens_per_second,
        elapsed_ms: value.get("elapsed_ms").and_then(Value::as_f64),
        llama_eval_tokens_per_second: value
            .get("llama_eval_tokens_per_second")
            .and_then(Value::as_f64),
        llama_eval_ms: value.get("llama_eval_ms").and_then(Value::as_f64),
        non_eval_overhead_ms: value.get("non_eval_overhead_ms").and_then(Value::as_f64),
        decode_call_tokens_per_second: value
            .get("decode_call_tokens_per_second")
            .and_then(Value::as_f64),
        decode_call_ms: value.get("decode_call_ms").and_then(Value::as_f64),
        sampling_tokens_per_second: value
            .get("sampling_tokens_per_second")
            .and_then(Value::as_f64),
        sampling_ms: value.get("sampling_ms").and_then(Value::as_f64),
        llama_eval_count: value.get("llama_eval_count").and_then(Value::as_u64),
        llama_graph_reuse_count: value.get("llama_graph_reuse_count").and_then(Value::as_i64),
        graph_node_count: value.get("graph_node_count").and_then(Value::as_u64),
        graph_inventory_bucket_overflow_count: value
            .get("graph_inventory_bucket_overflow_count")
            .and_then(Value::as_u64),
        graph_inventory: parse_abi_graph_inventory(&value),
        measured_tokens: value.get("measured_tokens").and_then(Value::as_u64),
        prompt_token_count: value.get("prompt_token_count").and_then(Value::as_u64),
    })
}

fn parse_abi_graph_inventory(value: &Value) -> Vec<AbiGraphInventoryBucket> {
    value
        .get("graph_inventory")
        .and_then(Value::as_array)
        .map(|buckets| {
            buckets
                .iter()
                .map(parse_abi_graph_inventory_bucket)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_abi_graph_inventory_bucket(value: &Value) -> AbiGraphInventoryBucket {
    AbiGraphInventoryBucket {
        family: value
            .get("family")
            .and_then(Value::as_str)
            .map(str::to_string),
        ggml_op: value.get("ggml_op").and_then(Value::as_i64),
        ggml_type: value.get("ggml_type").and_then(Value::as_u64),
        node_count: value.get("node_count").and_then(Value::as_u64),
        element_count: value.get("element_count").and_then(Value::as_u64),
        output_bytes: value.get("output_bytes").and_then(Value::as_u64),
        src0_bytes: value.get("src0_bytes").and_then(Value::as_u64),
        src1_bytes: value.get("src1_bytes").and_then(Value::as_u64),
        ne: value
            .get("ne")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_default(),
    }
}

fn first_abi_measured_tokens(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.measured_tokens)
}

fn first_abi_prompt_tokens(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.prompt_token_count)
}

fn median_abi_llama_eval_tokens_per_second(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.llama_eval_tokens_per_second)
        .collect::<Vec<_>>();
    (!samples.is_empty())
        .then(|| throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD))
        .and_then(|stats| stats.clean_median)
}

fn median_abi_decode_call_tokens_per_second(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.decode_call_tokens_per_second)
        .collect::<Vec<_>>();
    (!samples.is_empty())
        .then(|| throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD))
        .and_then(|stats| stats.clean_median)
}

fn median_abi_sampling_tokens_per_second(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.sampling_tokens_per_second)
        .collect::<Vec<_>>();
    (!samples.is_empty())
        .then(|| throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD))
        .and_then(|stats| stats.clean_median)
}

fn first_abi_non_eval_overhead_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.non_eval_overhead_ms)
}

fn first_abi_decode_call_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.decode_call_ms)
}

fn first_abi_sampling_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.sampling_ms)
}

fn first_abi_llama_eval_count(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.llama_eval_count)
}

fn first_abi_llama_graph_reuse_count(summary: &AbiDecodeProbeSummary) -> Option<i64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.llama_graph_reuse_count)
}

fn first_abi_graph_node_count(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.graph_node_count)
}

fn first_abi_graph_inventory_bucket_overflow_count(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.graph_inventory_bucket_overflow_count)
}

fn first_abi_graph_inventory(
    summary: &AbiDecodeProbeSummary,
) -> Option<Vec<AbiGraphInventoryBucket>> {
    summary
        .observations
        .iter()
        .find(|observation| !observation.graph_inventory.is_empty())
        .map(|observation| observation.graph_inventory.clone())
}

fn abi_decode_probe_error(summary: &AbiDecodeProbeSummary) -> Option<String> {
    let errors = summary
        .observations
        .iter()
        .filter_map(|observation| observation.error.as_deref())
        .collect::<Vec<_>>();
    (!errors.is_empty()).then(|| format!("{} abi decode repeats failed", errors.len()))
}

fn fatal_abi_probe_observation(observation: &AbiDecodeProbeObservation) -> bool {
    observation.error.is_some()
        && observation.status_code.is_some_and(|code| code != 0)
        && observation.tokens_per_second.is_none()
        && observation.elapsed_ms.is_none()
        && observation.llama_eval_ms.is_none()
        && observation.measured_tokens.is_none()
}

fn stderr_tail(stderr: &[u8]) -> Option<String> {
    let stderr = String::from_utf8_lossy(stderr);
    let tail = tail_lines(&stderr, 20);
    (!tail.trim().is_empty()).then_some(tail)
}

fn benchmark_model(
    args: &Args,
    hardware: &HardwareProfile,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
) -> Vec<BenchmarkScenarioSummary> {
    let scenarios = selected_benchmark_scenarios(args);
    let mut summaries = Vec::with_capacity(scenarios.len());
    let mut abort_reason = None;

    for (scenario_index, scenario) in scenarios.into_iter().enumerate() {
        if let Some(reason) = abort_reason.as_deref() {
            summaries.push(skipped_scenario_summary(
                scenario,
                &model.profile,
                recommendation,
                reason,
            ));
            continue;
        }
        let summary = benchmark_scenario(
            args,
            hardware,
            model,
            recommendation,
            model_index,
            scenario_index,
            scenario,
        );
        if fatal_benchmark_runtime_failure(&summary.benchmark) {
            abort_reason = Some(format!(
                "previous scenario {} failed to start the benchmark runtime",
                summary.scenario
            ));
        }
        summaries.push(summary);
    }

    summaries
}

fn benchmark_scenario(
    args: &Args,
    hardware: &HardwareProfile,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
    scenario_index: usize,
    scenario: BenchmarkScenarioSpec,
) -> BenchmarkScenarioSummary {
    let scenario = adapt_scenario_for_model(scenario, recommendation);
    heartbeat(
        Some(model_index),
        &model.input_ref,
        "scenario_start",
        &format!(
            "scenario={} max_new_tokens={} request_count={} reuse_session={}",
            scenario.name, scenario.max_new_tokens, scenario.request_count, scenario.reuse_session
        ),
    );
    let mut summary = BenchmarkSummary {
        verdict: "skipped".into(),
        ..BenchmarkSummary::default()
    };
    if let Some(reason) = benchmark_skip_reason(args, model, recommendation, scenario.kind) {
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "scenario_skip",
            &format!("scenario={} reason={reason}", scenario.name),
        );
        summary.skip_reason = Some(reason);
        let result = scenario_summary(scenario, &model.profile, recommendation, summary);
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "scenario_done",
            &scenario_summary_detail(&result),
        );
        return result;
    }

    let prediction = scenario_prediction(&scenario, recommendation);
    let range = scenario_prediction_range(&scenario, recommendation);
    let expected = BenchmarkExpected {
        predicted: prediction,
        range,
    };
    let initial = run_benchmark_repeats(
        args,
        model,
        model_index,
        scenario_index,
        &scenario,
        0,
        DEFAULT_REPEATS,
    );
    let summary = finalize_benchmark_summary(initial, &scenario, expected);
    let summary = remeasure_unstable_summary(
        args,
        model,
        model_index,
        scenario_index,
        &scenario,
        expected,
        summary,
    );
    let summary = confirm_stable_fit_mismatch_summary(
        args,
        model,
        model_index,
        scenario_index,
        &scenario,
        expected,
        summary,
    );
    let scenario_recommendation = benchmark_scenario_recommendation(
        hardware,
        &model.profile,
        recommendation,
        &scenario,
        &summary,
    );
    let result = scenario_summary(scenario, &model.profile, &scenario_recommendation, summary);
    heartbeat(
        Some(model_index),
        &model.input_ref,
        "scenario_done",
        &scenario_summary_detail(&result),
    );
    result
}

fn run_benchmark_repeats(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    scenario: &BenchmarkScenarioSpec,
    repeat_start: usize,
    repeat_count: usize,
) -> BenchmarkSummary {
    let mut summary = BenchmarkSummary {
        attempted: true,
        verdict: "skipped".into(),
        ..BenchmarkSummary::default()
    };
    for repeat_offset in 0..repeat_count {
        let repeat = repeat_start + repeat_offset;
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "benchmark_repeat_start",
            &format!(
                "scenario={} repeat={} batch_repeat={}/{}",
                scenario.name,
                repeat + 1,
                repeat_offset + 1,
                repeat_count
            ),
        );
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Benchmarking {} {} repeat {}/{}",
                model.input_ref,
                scenario.name,
                repeat_offset + 1,
                repeat_count
            ),
        );
        let observation =
            run_one_benchmark(args, model, model_index, scenario_index, repeat, scenario);
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "benchmark_repeat_done",
            &benchmark_observation_detail(&observation, scenario),
        );
        if let Some(error) = observation.error.as_ref() {
            summary.errors.push(error.clone());
        }
        let fatal = fatal_benchmark_observation(&observation, scenario);
        summary.observations.push(observation);
        if fatal {
            let reason = format!(
                "aborting {} repeats after runtime startup failure; later repeats would relaunch the same single-stage runtime",
                scenario.name
            );
            heartbeat(
                Some(model_index),
                &model.input_ref,
                "benchmark_repeats_abort",
                &reason,
            );
            summary.errors.push(reason);
            break;
        }
    }
    summary
}

fn skipped_scenario_summary(
    scenario: BenchmarkScenarioSpec,
    model: &ModelProfile,
    recommendation: &ModelRecommendation,
    reason: &str,
) -> BenchmarkScenarioSummary {
    let summary = BenchmarkSummary {
        skip_reason: Some(reason.into()),
        verdict: "skipped".into(),
        ..BenchmarkSummary::default()
    };
    scenario_summary(scenario, model, recommendation, summary)
}

fn benchmark_skip_reason(
    args: &Args,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    scenario: BenchmarkScenarioKind,
) -> Option<String> {
    if model.profile.layer_count.is_none() {
        return Some("model metadata did not include layer count".into());
    }
    if !args.benchmark_all
        && !matches!(
            recommendation.fit_status,
            FitStatus::FitsLocal | FitStatus::FitsWithWarning
        )
    {
        return Some(format!(
            "fit status is {:?}; use --benchmark-all to force single-stage benchmark",
            recommendation.fit_status
        ));
    }
    if scenario == BenchmarkScenarioKind::SteadyDecode
        && recommendation.estimated_decode_tokens_per_sec.is_none()
    {
        return Some("fit algorithm did not produce a decode tokens/sec estimate".into());
    }
    if scenario == BenchmarkScenarioKind::Prefill
        && recommendation.estimated_prefill_tokens_per_sec.is_none()
    {
        return Some("fit algorithm did not produce a prefill tokens/sec estimate".into());
    }
    if !args.benchmark_all && recommendation.selected_backend == BackendKind::Cpu {
        return Some(
            "fit selected CPU backend; use --benchmark-all to force the single-stage Skippy benchmark"
                .into(),
        );
    }
    None
}

fn adapt_scenario_for_model(
    mut scenario: BenchmarkScenarioSpec,
    recommendation: &ModelRecommendation,
) -> BenchmarkScenarioSpec {
    // Tiny models are fast enough that a short decode benchmark mostly measures
    // request/runtime jitter. Increase the steady-decode token window when the
    // fit estimate says the active byte footprint is small. This keeps the
    // validator honest: the fit algorithm still predicts from metadata alone,
    // while validation gives each model enough generated tokens for tok/s to be
    // a useful signal.
    if scenario.kind != BenchmarkScenarioKind::SteadyDecode {
        return scenario;
    }
    let Some(active_bytes) = recommendation.estimated_active_decode_bytes_per_token else {
        return scenario;
    };
    scenario.max_new_tokens = steady_decode_tokens_for_active_bytes(active_bytes);
    scenario
}

fn steady_decode_tokens_for_active_bytes(active_bytes: u64) -> usize {
    let gib = 1024 * 1024 * 1024;
    if active_bytes < gib / 2 {
        1024
    } else if active_bytes < 2 * gib {
        512
    } else {
        DEFAULT_MAX_NEW_TOKENS
    }
}

fn finalize_benchmark_summary(
    mut summary: BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
    expected: BenchmarkExpected,
) -> BenchmarkSummary {
    let samples = scenario_metric_samples(&summary, scenario);
    summary.raw_sample_count = samples.len();
    summary.successful_repeats = summary
        .observations
        .iter()
        .filter(|observation| observation.error.is_none())
        .count();
    if samples.is_empty() {
        summary.sample_count = 0;
        summary.verdict = if summary.attempted && benchmark_has_runtime_error(&summary) {
            "runtime-error".into()
        } else {
            "error".into()
        };
        return summary;
    }

    let stats = throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD);
    let median = stats.clean_median.expect("non-empty sample stats");
    let min = stats.clean_min.expect("non-empty sample stats");
    let max = stats.clean_max.expect("non-empty sample stats");
    let spread = stats.clean_spread.expect("non-empty sample stats");
    let observed_over_fit = if expected.predicted.is_some_and(|fit| fit > 0.0) {
        expected.predicted.map(|fit| median / fit)
    } else {
        None
    };

    summary.sample_count = stats.clean_sample_count;
    summary.median_tokens_per_sec = Some(median);
    summary.min_tokens_per_sec = Some(min);
    summary.max_tokens_per_sec = Some(max);
    summary.spread_pct = Some(spread * 100.0);
    summary.raw_median_tokens_per_sec = stats.raw_median;
    summary.raw_min_tokens_per_sec = stats.raw_min;
    summary.raw_max_tokens_per_sec = stats.raw_max;
    summary.raw_spread_pct = stats.raw_spread.map(|spread| spread * 100.0);
    summary.denoised_outlier_count = stats.outlier_count;
    summary.observed_over_fit = observed_over_fit;
    summary.verdict = benchmark_verdict(median, observed_over_fit, spread, expected.range);
    summary
}

fn remeasure_unstable_summary(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    scenario: &BenchmarkScenarioSpec,
    expected: BenchmarkExpected,
    mut initial: BenchmarkSummary,
) -> BenchmarkSummary {
    let Some(reason) = remeasure_reason(&initial, scenario) else {
        return initial;
    };
    thread::sleep(DEFAULT_REMEASURE_PAUSE);
    let remeasure = run_benchmark_repeats(
        args,
        model,
        model_index,
        scenario_index,
        scenario,
        DEFAULT_REPEATS,
        DEFAULT_REMEASURE_REPEATS,
    );
    let mut remeasure = finalize_benchmark_summary(remeasure, scenario, expected);
    if accepts_remeasure_summary(&remeasure) {
        remeasure.remeasured = true;
        remeasure.remeasure_reason = Some(reason);
        remeasure.initial_observations = initial.observations;
        return remeasure;
    }
    initial.remeasured = true;
    initial.remeasure_reason = Some(format!(
        "{reason}; remeasure was not stable enough to replace the initial pass"
    ));
    initial.rejected_remeasure_observations = remeasure.observations;
    initial
}

fn remeasure_reason(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<String> {
    if benchmark_has_runtime_error(summary) {
        return None;
    }
    let raw_spread = summary.raw_spread_pct? / 100.0;
    if raw_spread >= DEFAULT_REMEASURE_RAW_SPREAD {
        return Some(format!(
            "raw {} spread {:.1}% exceeded remeasure threshold {:.1}%",
            scenario.name,
            raw_spread * 100.0,
            DEFAULT_REMEASURE_RAW_SPREAD * 100.0
        ));
    }
    if scenario.kind != BenchmarkScenarioKind::SteadyDecode {
        return None;
    }
    let ordered_drop = ordered_sample_drop(summary, scenario)?;
    (ordered_drop >= DEFAULT_REMEASURE_ORDERED_DROP).then(|| {
        format!(
            "ordered steady-decode samples dropped {:.1}% across repeats",
            ordered_drop * 100.0
        )
    })
}

fn ordered_sample_drop(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<f64> {
    let samples = scenario_metric_samples(summary, scenario);
    let first = samples.first().copied().filter(|sample| *sample > 0.0)?;
    let last = samples.last().copied()?;
    (last < first).then_some((first - last) / first)
}

fn accepts_remeasure_summary(summary: &BenchmarkSummary) -> bool {
    if benchmark_has_runtime_error(summary) {
        return false;
    }
    if summary.successful_repeats < 2 {
        return false;
    }
    summary
        .spread_pct
        .is_some_and(|spread| spread <= DEFAULT_MAX_SPREAD * 100.0)
}

fn confirm_stable_fit_mismatch_summary(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    scenario: &BenchmarkScenarioSpec,
    expected: BenchmarkExpected,
    mut initial: BenchmarkSummary,
) -> BenchmarkSummary {
    let Some(reason) = stable_fit_mismatch_confirmation_reason(&initial, scenario) else {
        return initial;
    };
    thread::sleep(DEFAULT_REMEASURE_PAUSE);
    let confirmation = run_benchmark_repeats(
        args,
        model,
        model_index,
        scenario_index,
        scenario,
        DEFAULT_REPEATS + DEFAULT_REMEASURE_REPEATS,
        DEFAULT_CONFIRM_REPEATS,
    );
    let mut confirmation = finalize_benchmark_summary(confirmation, scenario, expected);
    if accepts_confirmation_summary(&initial, &confirmation) {
        confirmation.remeasured = true;
        confirmation.remeasure_reason = Some(reason);
        confirmation.initial_observations = preserved_initial_observations(initial);
        return confirmation;
    }
    initial.remeasured = true;
    initial.remeasure_reason = Some(format!(
        "{reason}; confirmation did not materially change the stable mismatch"
    ));
    initial.rejected_remeasure_observations = confirmation.observations;
    initial
}

fn stable_fit_mismatch_confirmation_reason(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<String> {
    if scenario.kind != BenchmarkScenarioKind::SteadyDecode || benchmark_has_runtime_error(summary)
    {
        return None;
    }
    if !is_stable_summary(summary) {
        return None;
    }
    let ratio = summary.observed_over_fit?;
    let outside_tolerance = (ratio - 1.0).abs() > DEFAULT_TOLERANCE;
    let mismatch_verdict = matches!(
        summary.verdict.as_str(),
        "slower-than-fit" | "faster-than-fit"
    );
    (outside_tolerance || mismatch_verdict).then(|| {
        format!(
            "stable steady-decode fit mismatch ratio {:.3} exceeded tolerance {:.1}%",
            ratio,
            DEFAULT_TOLERANCE * 100.0
        )
    })
}

fn accepts_confirmation_summary(
    initial: &BenchmarkSummary,
    confirmation: &BenchmarkSummary,
) -> bool {
    if !accepts_remeasure_summary(confirmation) {
        return false;
    }
    let Some(initial_ratio) = initial.observed_over_fit else {
        return false;
    };
    let Some(confirmation_ratio) = confirmation.observed_over_fit else {
        return false;
    };
    let Some(initial_median) = initial.median_tokens_per_sec else {
        return false;
    };
    let Some(confirmation_median) = confirmation.median_tokens_per_sec else {
        return false;
    };
    let observed_delta = relative_delta(initial_median, confirmation_median);
    let initial_error = (initial_ratio - 1.0).abs();
    let confirmation_error = (confirmation_ratio - 1.0).abs();
    observed_delta >= DEFAULT_CONFIRM_DELTA || confirmation_error + 0.05 < initial_error
}

fn is_stable_summary(summary: &BenchmarkSummary) -> bool {
    summary
        .spread_pct
        .is_some_and(|spread| spread <= DEFAULT_MAX_SPREAD * 100.0)
}

fn preserved_initial_observations(initial: BenchmarkSummary) -> Vec<BenchmarkObservation> {
    let mut observations = initial.initial_observations;
    observations.extend(initial.observations);
    observations
}

fn relative_delta(left: f64, right: f64) -> f64 {
    let baseline = left.abs().max(right.abs());
    if baseline <= f64::EPSILON {
        0.0
    } else {
        (left - right).abs() / baseline
    }
}

fn benchmark_has_runtime_error(summary: &BenchmarkSummary) -> bool {
    !summary.errors.is_empty()
        || summary
            .observations
            .iter()
            .any(|observation| observation.status_code.is_some_and(|code| code != 0))
        || summary
            .observations
            .iter()
            .any(|observation| observation.error.is_some())
}

fn fatal_benchmark_runtime_failure(summary: &BenchmarkSummary) -> bool {
    summary
        .observations
        .iter()
        .any(fatal_benchmark_observation_without_scenario)
}

fn fatal_benchmark_observation(
    observation: &BenchmarkObservation,
    scenario: &BenchmarkScenarioSpec,
) -> bool {
    fatal_benchmark_observation_without_scenario(observation)
        && benchmark_observation_metric(observation, scenario).is_none()
}

fn fatal_benchmark_observation_without_scenario(observation: &BenchmarkObservation) -> bool {
    // A non-zero `skippy-bench` exit with no request observations and no
    // measured aggregate metric means the stage runtime did not reach the
    // request path. Retrying the same model/scenario launches the same
    // single-stage runtime with the same metadata-derived config, so this is
    // not a throughput sample and should not consume another startup timeout.
    observation.error.is_some()
        && observation.status_code.is_some_and(|code| code != 0)
        && observation.generated_tokens_per_sec.is_none()
        && observation.text_request_elapsed_ms.is_none()
        && observation.request_results.is_empty()
}

fn scenario_metric_samples(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Vec<f64> {
    // Scenario sampling intentionally differs by workload shape.
    //
    // Steady decode should represent sustained token generation, so each repeat
    // contributes one aggregate decode-throughput sample. Aggregating generated
    // tokens over aggregate decode time denoises tiny models, where a single
    // request can swing just from scheduler/runtime jitter. This is not
    // "cheating" against the fit estimate: the prediction is unchanged and
    // metadata-only; we are only making the observation less noisy.
    //
    // Prefill is a separate metric from first-token latency. It uses Skippy's
    // request timing fields to compare prompt tokens / prefill elapsed time
    // against `estimated_prefill_tokens_per_sec`. That keeps the validation
    // falsifiable without using observed prefill speed as a scoring input.
    //
    // KV warm reuse cares about the reused final request, so it samples the last
    // request. First-token samples end-to-end request latency in milliseconds:
    // tokenize + prefill + the first decode step. That latency is deliberately
    // kept separate from prefill throughput because single-token decode after a
    // prompt can include graph/session/synchronization costs that sustained
    // steady decode does not see.
    let request_samples = match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode => summary
            .observations
            .iter()
            .filter_map(steady_decode_observation_tokens_per_sec)
            .collect::<Vec<_>>(),
        BenchmarkScenarioKind::Prefill => summary
            .observations
            .iter()
            .filter_map(prefill_observation_tokens_per_sec)
            .collect::<Vec<_>>(),
        BenchmarkScenarioKind::KvWarmReuse => summary
            .observations
            .iter()
            .filter_map(|observation| observation.request_results.last())
            .filter_map(|request| request.generated_tokens_per_sec)
            .collect::<Vec<_>>(),
        BenchmarkScenarioKind::FirstToken => summary
            .observations
            .iter()
            .filter_map(|observation| observation.text_request_elapsed_ms)
            .collect::<Vec<_>>(),
    };
    if !request_samples.is_empty() {
        return request_samples;
    }
    summary
        .observations
        .iter()
        .filter_map(|observation| observation.generated_tokens_per_sec)
        .collect()
}

fn steady_decode_observation_tokens_per_sec(observation: &BenchmarkObservation) -> Option<f64> {
    // Prefer decode-only timings from `/v1/text`, excluding tokenization and
    // prefill. If an older benchmark binary did not emit those fields, fall back
    // to the last request's reported throughput so historical validation JSON
    // can still be summarized.
    let generated_tokens = observation
        .request_results
        .iter()
        .map(|request| request.generated_token_count.unwrap_or_default())
        .sum::<u64>();
    let decode_elapsed_ms = observation
        .request_results
        .iter()
        .filter_map(|request| request.decode_elapsed_ms)
        .sum::<f64>();
    if generated_tokens > 0 && decode_elapsed_ms > 0.0 {
        return Some(generated_tokens as f64 / (decode_elapsed_ms / 1000.0));
    }
    observation.request_results.last().and_then(|request| {
        request
            .decode_tokens_per_sec
            .or(request.generated_tokens_per_sec)
    })
}

fn prefill_observation_tokens_per_sec(observation: &BenchmarkObservation) -> Option<f64> {
    let prompt_tokens = observation
        .request_results
        .iter()
        .map(|request| request.prompt_token_count.unwrap_or_default())
        .sum::<u64>();
    let prefill_elapsed_ms = observation
        .request_results
        .iter()
        .filter_map(|request| request.prefill_elapsed_ms)
        .sum::<f64>();
    if prompt_tokens > 0 && prefill_elapsed_ms > 0.0 {
        return Some(prompt_tokens as f64 / (prefill_elapsed_ms / 1000.0));
    }
    None
}

fn benchmark_scenario_recommendation(
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    fallback: &ModelRecommendation,
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> ModelRecommendation {
    if !matches!(
        scenario.kind,
        BenchmarkScenarioKind::SteadyDecode
            | BenchmarkScenarioKind::Prefill
            | BenchmarkScenarioKind::FirstToken
            | BenchmarkScenarioKind::KvWarmReuse
    ) {
        return fallback.clone();
    }
    let Some(context_tokens) = prediction_context_tokens(scenario, benchmark) else {
        return fallback.clone();
    };
    let mut workload = primary_workload_profile();
    workload.interaction.expected_prompt_tokens = Some(context_tokens);
    score_model(hardware, profile, &selection_config(&workload))
}

fn prediction_context_tokens(
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> Option<u32> {
    let prompt_tokens = median_prompt_token_count(benchmark)?;
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode => {
            let generated = median_generated_tokens_per_request(benchmark, scenario)?;
            Some(prompt_tokens.saturating_add(average_generated_prefix_tokens(generated)))
        }
        BenchmarkScenarioKind::KvWarmReuse => {
            let generated = median_generated_tokens_per_request(benchmark, scenario)?;
            let current_decode =
                prompt_tokens.saturating_add(average_generated_prefix_tokens(generated));
            if !scenario.reuse_session {
                return Some(current_decode);
            }
            let prior_requests = u32::try_from(scenario.request_count.saturating_sub(1)).ok()?;
            // `skippy-bench local-single` reuses the same session id for the
            // warm-reuse scenario. By the final request, previous requests have
            // already appended their prompt and generated tokens to the KV
            // state. The observed metric for this scenario is the last
            // request's decode throughput, so the metadata estimate should
            // charge the prior cached sequence plus the average prefix length
            // within the current generated run. This is a causal-attention
            // shape fact, not calibration against the measured tokens/sec.
            let prior_cached =
                prior_requests.saturating_mul(prompt_tokens.saturating_add(generated));
            Some(prior_cached.saturating_add(current_decode))
        }
        BenchmarkScenarioKind::Prefill | BenchmarkScenarioKind::FirstToken => Some(prompt_tokens),
    }
}

fn average_generated_prefix_tokens(generated_tokens: u32) -> u32 {
    // During sampled decode, generated token i attends to the prompt plus the
    // i previously emitted tokens. Averaged across an N-token generation that
    // is prompt + (N - 1) / 2. The scorer only accepts an integer context
    // proxy, so round the generated-prefix half up by one token for odd/even
    // boundaries instead of introducing a hidden fractional fudge factor.
    generated_tokens.saturating_sub(1).div_ceil(2)
}

fn median_generated_tokens_per_request(
    benchmark: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<u32> {
    let mut samples = benchmark
        .observations
        .iter()
        .flat_map(|observation| observation.request_results.iter())
        .filter_map(|request| request.generated_token_count)
        .filter_map(|count| u32::try_from(count).ok())
        .collect::<Vec<_>>();
    if samples.is_empty() {
        samples = benchmark
            .observations
            .iter()
            .filter_map(|observation| {
                let generated = observation.generated_token_count?;
                let request_count = observation
                    .request_count
                    .or_else(|| u64::try_from(scenario.request_count).ok())
                    .filter(|count| *count > 0)?;
                u32::try_from(generated / request_count).ok()
            })
            .collect();
    }
    if samples.is_empty() {
        return u32::try_from(scenario.max_new_tokens).ok();
    }
    samples.sort_unstable();
    Some(samples[samples.len() / 2])
}

fn median_prompt_token_count(benchmark: &BenchmarkSummary) -> Option<u32> {
    let mut samples = benchmark
        .observations
        .iter()
        .filter_map(|observation| observation.prompt_token_count)
        .filter_map(|count| u32::try_from(count).ok())
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable();
    Some(samples[samples.len() / 2])
}

fn scenario_summary(
    scenario: BenchmarkScenarioSpec,
    model: &ModelProfile,
    recommendation: &ModelRecommendation,
    mut benchmark: BenchmarkSummary,
) -> BenchmarkScenarioSummary {
    let predicted = scenario_prediction(&scenario, recommendation);
    let predicted_range = scenario_prediction_range(&scenario, recommendation);
    let observed = scenario_observed(&scenario, &benchmark);
    let observed_over_fit = match (observed, predicted) {
        (Some(observed), Some(predicted)) if predicted > 0.0 => Some(observed / predicted),
        _ => None,
    };
    let verdict = scenario_level_verdict(
        &scenario,
        &benchmark,
        observed,
        observed_over_fit,
        predicted_range,
    );
    // `BenchmarkSummary` is produced before scenario-specific rescoring is
    // possible, because the rescore needs the measured prompt-token count from
    // the benchmark itself. Keep the raw timing samples intact, but make the
    // embedded comparison fields match the scenario wrapper. Otherwise JSON
    // consumers and the Markdown table can accidentally report the generic
    // workload estimate while the scenario verdict was judged against the
    // prompt-shape-correct estimate.
    benchmark.observed_over_fit = observed_over_fit;
    benchmark.verdict = verdict.clone();
    BenchmarkScenarioSummary {
        scenario: scenario.name.into(),
        fit_metric: scenario.fit_metric.into(),
        predicted,
        predicted_range,
        prediction_context_tokens: prediction_context_tokens(&scenario, &benchmark),
        prediction_decode_cost_breakdown: recommendation.decode_cost_breakdown.clone(),
        observed,
        observed_over_fit,
        observed_over_abi: benchmark.observed_over_abi,
        first_token_breakdown: first_token_breakdown(&scenario, model, recommendation, &benchmark),
        verdict,
        benchmark,
    }
}

fn apply_observed_over_abi(
    benchmarks: &mut [BenchmarkScenarioSummary],
    abi_decode_probe: Option<&AbiDecodeProbeSummary>,
) {
    let abi_tokens_per_second = abi_decode_probe.and_then(|probe| probe.tokens_per_second);
    for benchmark in benchmarks {
        let observed_over_abi = ratio(benchmark.observed, abi_tokens_per_second);
        benchmark.observed_over_abi = observed_over_abi;
        benchmark.benchmark.observed_over_abi = observed_over_abi;
    }
}

fn first_token_breakdown(
    scenario: &BenchmarkScenarioSpec,
    model: &ModelProfile,
    recommendation: &ModelRecommendation,
    benchmark: &BenchmarkSummary,
) -> Option<FirstTokenBreakdown> {
    if scenario.kind != BenchmarkScenarioKind::FirstToken {
        return None;
    }
    let prompt_token_count = median_prompt_token_count(benchmark).map(u64::from);
    let observed_tokenize_ms =
        median_request_value(benchmark, |request| request.tokenize_elapsed_ms);
    let observed_prefill_ms = median_request_value(benchmark, |request| request.prefill_elapsed_ms);
    let observed_decode_ms = median_request_value(benchmark, |request| request.decode_elapsed_ms);
    let observed_total_ms =
        median_observation_value(benchmark, |observation| observation.text_request_elapsed_ms);
    let predicted_decode_ms = recommendation
        .estimated_first_token_decode_ms
        .map(f64::from);
    let predicted_overhead_ms = recommendation
        .estimated_first_token_overhead_ms
        .map(f64::from);
    let predicted_sampler_ms = recommendation
        .estimated_first_token_sampler_ms
        .map(f64::from);
    let predicted_sampled_decode_ms = predicted_decode_ms.map(|decode| {
        decode
            + predicted_overhead_ms.unwrap_or_default()
            + predicted_sampler_ms.unwrap_or_default()
    });
    // `/v1/text` measures the first decode call with sampling included. In
    // Skippy that call eventually reaches `skippy_decode_step_sampled()`,
    // which runs `skippy_sync_chat_sampling_history()` before applying the
    // sampler chain. The sync loop accepts each prompt token into the sampler
    // history on the first sampled decode after prefill; subsequent decode
    // steps only accept the newly generated token. We report the residual here
    // instead of hiding it in a tuned estimator constant so validation can show
    // whether first-token misses scale with prompt length and vocabulary size.
    let observed_sampled_decode_residual_ms =
        match (observed_decode_ms, predicted_sampled_decode_ms) {
            (Some(observed), Some(predicted)) => Some((observed - predicted).max(0.0)),
            _ => None,
        };
    let observed_sampled_decode_residual_us_per_prompt_token = observed_sampled_decode_residual_ms
        .zip(prompt_token_count)
        .and_then(|(residual_ms, tokens)| {
            (tokens > 0).then_some(residual_ms * 1000.0 / tokens as f64)
        });
    let observed_sum = observed_tokenize_ms.unwrap_or_default()
        + observed_prefill_ms.unwrap_or_default()
        + observed_decode_ms.unwrap_or_default();
    let observed_unattributed_ms = observed_total_ms.map(|total| (total - observed_sum).max(0.0));
    Some(FirstTokenBreakdown {
        prompt_token_count,
        tokenizer_vocab_size: model.tokenizer.vocab_size,
        chat_template_available: model.tokenizer.chat_template_available,
        predicted_prefill_ms: recommendation
            .estimated_first_token_prefill_ms
            .map(f64::from),
        predicted_decode_ms,
        predicted_overhead_ms,
        predicted_sampler_ms,
        predicted_sampled_decode_ms,
        observed_tokenize_ms,
        observed_prefill_ms,
        observed_decode_ms,
        observed_sampled_decode_residual_ms,
        observed_sampled_decode_residual_us_per_prompt_token,
        observed_unattributed_ms,
    })
}

fn median_request_value(
    benchmark: &BenchmarkSummary,
    value: impl Fn(&BenchmarkRequestObservation) -> Option<f64>,
) -> Option<f64> {
    let mut samples = benchmark
        .observations
        .iter()
        .flat_map(|observation| observation.request_results.iter())
        .filter_map(value)
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(median(&samples))
}

fn scenario_level_verdict(
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
    observed: Option<f64>,
    observed_over_fit: Option<f64>,
    predicted_range: Option<(f64, f64)>,
) -> String {
    if matches!(
        benchmark.verdict.as_str(),
        "skipped" | "error" | "runtime-error" | "inconclusive-noisy"
    ) {
        return benchmark.verdict.clone();
    }
    let Some(observed) = observed else {
        return "error".into();
    };
    let verdict = benchmark_verdict(
        observed,
        observed_over_fit,
        benchmark.spread_pct.unwrap_or_default() / 100.0,
        predicted_range,
    );
    if scenario.kind != BenchmarkScenarioKind::FirstToken {
        return verdict;
    }
    match verdict.as_str() {
        "slower-than-fit" => "faster-than-fit".into(),
        "faster-than-fit" => "slower-than-fit".into(),
        _ => verdict,
    }
}

fn scenario_observed(
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> Option<f64> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode => benchmark.median_tokens_per_sec,
        BenchmarkScenarioKind::Prefill => benchmark.median_tokens_per_sec,
        BenchmarkScenarioKind::FirstToken => {
            median_observation_value(benchmark, |observation| observation.text_request_elapsed_ms)
        }
        BenchmarkScenarioKind::KvWarmReuse => median_observation_value(benchmark, |observation| {
            observation
                .request_results
                .last()
                .and_then(|request| request.generated_tokens_per_sec)
        }),
    }
}

fn median_observation_value(
    benchmark: &BenchmarkSummary,
    value: impl Fn(&BenchmarkObservation) -> Option<f64>,
) -> Option<f64> {
    let mut samples = benchmark
        .observations
        .iter()
        .filter_map(value)
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(median(&samples))
}

fn scenario_prediction(
    scenario: &BenchmarkScenarioSpec,
    recommendation: &ModelRecommendation,
) -> Option<f64> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode | BenchmarkScenarioKind::KvWarmReuse => recommendation
            .estimated_decode_tokens_per_sec
            .map(f64::from),
        BenchmarkScenarioKind::Prefill => recommendation
            .estimated_prefill_tokens_per_sec
            .map(f64::from),
        BenchmarkScenarioKind::FirstToken => recommendation.estimated_first_token_ms.map(f64::from),
    }
}

fn scenario_prediction_range(
    scenario: &BenchmarkScenarioSpec,
    recommendation: &ModelRecommendation,
) -> Option<(f64, f64)> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode | BenchmarkScenarioKind::KvWarmReuse => recommendation
            .estimated_decode_tokens_per_sec_range
            .map(|range| (f64::from(range.lower), f64::from(range.upper))),
        BenchmarkScenarioKind::Prefill => None,
        BenchmarkScenarioKind::FirstToken => recommendation
            .estimated_first_token_ms_range
            .map(|range| (f64::from(range.lower_ms), f64::from(range.upper_ms))),
    }
}

fn run_one_benchmark(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    repeat: usize,
    scenario: &BenchmarkScenarioSpec,
) -> BenchmarkObservation {
    let run_id = benchmark_run_id(model_index, scenario.name, repeat);
    let report_json_path = std::env::temp_dir().join(format!("{run_id}-report.json"));
    let stdout_json_path = std::env::temp_dir().join(format!("{run_id}.json"));
    let mut observation = BenchmarkObservation {
        repeat,
        run_id: run_id.clone(),
        command: Vec::new(),
        status_code: None,
        wall_seconds: 0.0,
        prompt_token_count: None,
        generated_tokens_per_sec: None,
        generated_token_count: None,
        text_request_elapsed_ms: None,
        request_count: None,
        reuse_session: None,
        request_results: Vec::new(),
        stdout_json_path: Some(stdout_json_path.clone()),
        report_json_path: report_json_path.clone(),
        stderr_tail: None,
        error: None,
    };

    let layer_count = model
        .profile
        .layer_count
        .expect("benchmark skip reason checked layer count");
    let Ok(port_base) = benchmark_port_base(args, model_index, scenario_index, repeat) else {
        observation.error = Some("port allocation overflow".into());
        return observation;
    };
    let command_args = benchmark_command_args(
        args,
        model,
        layer_count,
        port_base,
        &run_id,
        &report_json_path,
        scenario,
    );
    observation.command = command_display(&args.skippy_bench_bin, &command_args);

    let started = Instant::now();
    let output = Command::new(&args.skippy_bench_bin)
        .args(&command_args)
        .output();
    observation.wall_seconds = started.elapsed().as_secs_f64();

    match output {
        Ok(output) => read_benchmark_output(output, stdout_json_path, &mut observation),
        Err(err) => observation.error = Some(format!("failed to start skippy-bench: {err}")),
    }
    observation
}

fn benchmark_run_id(model_index: usize, scenario: &str, repeat: usize) -> String {
    format!(
        "model-fit-validate-{}-{model_index}-{scenario}-{repeat}",
        std::process::id()
    )
}

fn read_benchmark_output(
    output: std::process::Output,
    stdout_json_path: PathBuf,
    observation: &mut BenchmarkObservation,
) {
    observation.status_code = output.status.code();
    if !output.stderr.is_empty() {
        observation.stderr_tail = Some(tail_lines(&String::from_utf8_lossy(&output.stderr), 40));
    }
    if let Err(err) = fs::write(&stdout_json_path, &output.stdout) {
        observation.error = Some(format!("write benchmark stdout: {err}"));
        return;
    }
    if !output.status.success() {
        observation.error = Some(format!(
            "benchmark exited with status {}",
            output.status.code().unwrap_or(-1)
        ));
        return;
    }
    match serde_json::from_slice::<Value>(&output.stdout) {
        Ok(value) => apply_benchmark_json(&value, observation),
        Err(err) => observation.error = Some(format!("parse skippy-bench output JSON: {err}")),
    }
}

fn apply_benchmark_json(value: &Value, observation: &mut BenchmarkObservation) {
    observation.generated_tokens_per_sec = value
        .get("generated_tokens_per_sec")
        .and_then(Value::as_f64);
    observation.generated_token_count = value.get("generated_token_count").and_then(Value::as_u64);
    observation.prompt_token_count = value
        .get("request_results")
        .and_then(Value::as_array)
        .and_then(|results| results.first())
        .and_then(|result| result.get("prompt_token_count"))
        .and_then(Value::as_u64);
    observation.text_request_elapsed_ms =
        value.get("text_request_elapsed_ms").and_then(Value::as_f64);
    observation.request_count = value.get("request_count").and_then(Value::as_u64);
    observation.reuse_session = value.get("reuse_session").and_then(Value::as_bool);
    observation.request_results = value
        .get("request_results")
        .and_then(Value::as_array)
        .map(|results| results.iter().map(request_observation_from_json).collect())
        .unwrap_or_default();
    if observation.generated_tokens_per_sec.is_none() {
        observation.error = Some("skippy-bench output omitted generated_tokens_per_sec".into());
    }
}

fn request_observation_from_json(value: &Value) -> BenchmarkRequestObservation {
    BenchmarkRequestObservation {
        request_id: string_field(value, "request_id"),
        session_id: string_field(value, "session_id"),
        elapsed_ms: value.get("elapsed_ms").and_then(Value::as_f64),
        tokenize_elapsed_ms: value.get("tokenize_elapsed_ms").and_then(Value::as_f64),
        prefill_elapsed_ms: value.get("prefill_elapsed_ms").and_then(Value::as_f64),
        decode_elapsed_ms: value.get("decode_elapsed_ms").and_then(Value::as_f64),
        prompt_token_count: value.get("prompt_token_count").and_then(Value::as_u64),
        generated_token_count: value.get("generated_token_count").and_then(Value::as_u64),
        generated_tokens_per_sec: value
            .get("generated_tokens_per_sec")
            .and_then(Value::as_f64),
        decode_tokens_per_sec: value.get("decode_tokens_per_sec").and_then(Value::as_f64),
    }
}

fn benchmark_command_args(
    args: &Args,
    model: &PreparedModel,
    layer_count: u32,
    port_base: u16,
    run_id: &str,
    report_json_path: &Path,
    scenario: &BenchmarkScenarioSpec,
) -> Vec<String> {
    let mut command_args = vec![
        "local-single".into(),
        "--metrics-server-bin".into(),
        args.metrics_server_bin.display().to_string(),
        "--stage-server-bin".into(),
        args.skippy_server_bin.display().to_string(),
        "--model-path".into(),
        model.primary_gguf_path.display().to_string(),
        "--model-id".into(),
        model.input_ref.clone(),
        "--ctx-size".into(),
        scenario.ctx_size.to_string(),
        "--n-gpu-layers=-1".into(),
        "--layer-end".into(),
        layer_count.to_string(),
        "--warmup-new-tokens".into(),
        scenario.warmup_tokens.to_string(),
        "--max-new-tokens".into(),
        scenario.max_new_tokens.to_string(),
        "--request-count".into(),
        scenario.request_count.to_string(),
        "--prompt".into(),
        scenario.prompt.clone(),
        "--run-id".into(),
        run_id.to_string(),
        "--metrics-http-addr".into(),
        format!("127.0.0.1:{port_base}"),
        "--metrics-otlp-grpc-addr".into(),
        format!("127.0.0.1:{}", port_base + 1000),
        "--stage-bind-addr".into(),
        format!("127.0.0.1:{}", port_base + 2000),
        "--output".into(),
        report_json_path.display().to_string(),
        "--startup-timeout-secs".into(),
        "300".into(),
    ];
    if scenario.reuse_session {
        command_args.push("--reuse-session".into());
    }
    command_args
}

fn benchmark_port_base(
    args: &Args,
    model_index: usize,
    scenario_index: usize,
    repeat: usize,
) -> Result<u16> {
    let repeats_per_scenario =
        DEFAULT_REPEATS + DEFAULT_REMEASURE_REPEATS + DEFAULT_CONFIRM_REPEATS;
    let scenario_count = selected_benchmark_scenarios(args).len().max(1);
    args.base_port
        .checked_add(
            (model_index * scenario_count * repeats_per_scenario
                + scenario_index * repeats_per_scenario
                + repeat) as u16
                * 10,
        )
        .context("port allocation overflow")
}

fn benchmark_verdict(
    median: f64,
    observed_over_fit: Option<f64>,
    spread: f64,
    predicted_range: Option<(f64, f64)>,
) -> String {
    if observed_over_fit.is_none() {
        return "observed-only".into();
    }
    if spread > DEFAULT_MAX_SPREAD {
        return "inconclusive-noisy".into();
    }
    let Some(ratio) = observed_over_fit else {
        return "error".into();
    };
    let within_tolerance = (ratio - 1.0).abs() <= DEFAULT_TOLERANCE;
    let within_range = predicted_range
        .map(|(lower, upper)| median >= lower && median <= upper)
        .unwrap_or(true);
    if within_tolerance && within_range {
        "match".into()
    } else if within_range && (ratio - 1.0).abs() <= DEFAULT_TOLERANCE + spread {
        "inconclusive-noisy".into()
    } else if ratio < 1.0 {
        "slower-than-fit".into()
    } else {
        "faster-than-fit".into()
    }
}

fn load_hardware_profile(args: &Args) -> Result<LoadedHardware> {
    heartbeat(
        None,
        "hardware",
        "hardware_start",
        "building hardware profile",
    );
    let survey = mesh_llm_system::hardware::survey();
    let (benchmark_outputs, facts, raw_json) = if let Some(path) = args.gpu_benchmark_json.as_ref()
    {
        heartbeat(
            None,
            "hardware",
            "gpu_benchmark_json_start",
            &format!("path={}", path.display()),
        );
        let bytes = read_json_input(path)?;
        let raw_json: Value = serde_json::from_slice(&bytes).context("parse GPU benchmark JSON")?;
        let (outputs, facts) = parse_gpu_benchmark_json(&raw_json, &survey)?;
        heartbeat(
            None,
            "hardware",
            "gpu_benchmark_json_done",
            &format!("outputs={}", outputs.len()),
        );
        (outputs, facts, raw_json)
    } else {
        let benchmark = run_local_gpu_benchmark(args, &survey)?;
        let outputs = benchmark.outputs;
        let facts = default_facts_with_backend(&survey, outputs.len(), benchmark.backend);
        let raw_json = json!({
            "source": "model-fit-validate:auto_gpu_benchmark",
            "runner_backend": benchmark.backend,
            "outputs": outputs,
        });
        (outputs, facts, raw_json)
    };
    let default_backend = facts
        .first()
        .and_then(|fact| fact.backend)
        .unwrap_or_else(|| infer_backend_from_survey(&survey));
    let profile = hardware_profile_from_gpu_benchmark(GpuBenchmarkHardwareInput {
        memory: memory_profile(&survey, &facts),
        cpu: cpu_profile(),
        default_backend,
        accelerators: facts,
        benchmark_outputs: benchmark_outputs.clone(),
    })?;
    heartbeat(
        None,
        "hardware",
        "hardware_done",
        &format!(
            "accelerators={} backend={:?}",
            profile.accelerators.len(),
            default_backend
        ),
    );
    Ok(LoadedHardware {
        profile,
        benchmark_outputs,
        raw_json,
    })
}

fn parse_gpu_benchmark_json(
    raw_json: &Value,
    survey: &HardwareSurvey,
) -> Result<(Vec<GpuBenchmarkOutput>, Vec<GpuBenchmarkAcceleratorFacts>)> {
    match serde_json::from_value::<Vec<GpuBenchmarkOutput>>(raw_json.clone()) {
        Ok(outputs) if !outputs.is_empty() => {
            return Ok((outputs.clone(), default_facts(survey, outputs.len())));
        }
        _ => {}
    }
    if let Some(outputs) = raw_json
        .get("outputs")
        .and_then(|raw_outputs| {
            serde_json::from_value::<Vec<GpuBenchmarkOutput>>(raw_outputs.clone()).ok()
        })
        .filter(|outputs| !outputs.is_empty())
    {
        return Ok((outputs.clone(), default_facts(survey, outputs.len())));
    }
    parse_gpus_command_json(raw_json, survey)
}

fn parse_gpus_command_json(
    raw_json: &Value,
    survey: &HardwareSurvey,
) -> Result<(Vec<GpuBenchmarkOutput>, Vec<GpuBenchmarkAcceleratorFacts>)> {
    let gpus = raw_json
        .get("gpus")
        .and_then(Value::as_array)
        .context("GPU benchmark JSON must be a raw BenchmarkOutput array or contain gpus[]")?;
    let mut outputs = Vec::new();
    let mut facts = Vec::new();
    for gpu in gpus {
        let Some(p90_gbps) = gpu.get("mem_bandwidth_gbps").and_then(Value::as_f64) else {
            continue;
        };
        if p90_gbps <= 0.0 {
            continue;
        }
        outputs.push(gpu_output_from_command_json(gpu, p90_gbps));
        facts.push(gpu_facts_from_command_json(gpu, survey));
    }
    if outputs.is_empty() {
        bail!("GPU benchmark JSON did not include any positive mem_bandwidth_gbps values");
    }
    Ok((outputs, facts))
}

fn gpu_output_from_command_json(gpu: &Value, p90_gbps: f64) -> GpuBenchmarkOutput {
    GpuBenchmarkOutput {
        device: string_field(gpu, "name").unwrap_or_else(|| "gpu".into()),
        buffer_mb: 0,
        runs: 0,
        p50_gbps: p90_gbps,
        p90_gbps,
        decode_effective_gbps: gpu.get("decode_effective_gbps").and_then(Value::as_f64),
        decode_fixed_overhead_ms: gpu.get("decode_fixed_overhead_ms").and_then(Value::as_f64),
        decode_runtime_overhead_ms: gpu
            .get("decode_runtime_overhead_ms")
            .and_then(Value::as_f64),
        post_prefill_decode_overhead_ms: gpu
            .get("post_prefill_decode_overhead_ms")
            .and_then(Value::as_f64),
        compute_tflops_fp32: gpu.get("compute_tflops_fp32").and_then(Value::as_f64),
        compute_tflops_fp16: gpu.get("compute_tflops_fp16").and_then(Value::as_f64),
        prefill_matmul_tflops_fp16: gpu
            .get("prefill_matmul_tflops_fp16")
            .and_then(Value::as_f64),
        prefill_ubatch_matmul_tflops_fp16: gpu
            .get("prefill_ubatch_matmul_tflops_fp16")
            .and_then(Value::as_f64),
        prefill_moe_matmul_tflops_fp16: gpu
            .get("prefill_moe_matmul_tflops_fp16")
            .and_then(Value::as_f64),
        sampler_history_us_per_token: gpu
            .get("sampler_history_us_per_token")
            .and_then(Value::as_f64),
        sampler_vocab_us_per_token: gpu
            .get("sampler_vocab_us_per_token")
            .and_then(Value::as_f64),
        decode_kernel_probes: gpu
            .get("decode_kernel_probes")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default(),
        noise_pct: 0.0,
        runtime_s: 0.0,
        rated_gbps: None,
        rated_estimated: None,
        efficiency_pct: None,
        bus_width_bits: None,
        mem_clock_mhz: None,
        gcn_arch: None,
        hbm: None,
    }
}

fn gpu_facts_from_command_json(
    gpu: &Value,
    survey: &HardwareSurvey,
) -> GpuBenchmarkAcceleratorFacts {
    let total_memory_bytes = gpu
        .get("vram_bytes")
        .and_then(Value::as_u64)
        .or_else(|| nonzero(survey.vram_bytes));
    let reserved_bytes = gpu.get("reserved_bytes").and_then(Value::as_u64);
    let unified_memory = gpu
        .get("unified_memory")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| survey.is_soc || survey.gpus.iter().any(|gpu| gpu.unified_memory));
    let name = string_field(gpu, "name").or_else(|| survey.gpu_name.clone());
    let backend = string_field(gpu, "backend_device")
        .as_deref()
        .map(infer_backend_from_device)
        .filter(|backend| *backend != BackendKind::Unknown)
        .or_else(|| Some(infer_backend_from_name(name.as_deref())));
    GpuBenchmarkAcceleratorFacts {
        name,
        kind: if unified_memory {
            AcceleratorKind::IntegratedGpu
        } else {
            AcceleratorKind::DiscreteGpu
        },
        backend,
        total_memory_bytes,
        available_memory_bytes: total_memory_bytes
            .map(|total| total.saturating_sub(reserved_bytes.unwrap_or(0))),
        unified_memory,
    }
}

fn default_facts(survey: &HardwareSurvey, count: usize) -> Vec<GpuBenchmarkAcceleratorFacts> {
    default_facts_with_backend(survey, count, BackendKind::Unknown)
}

fn default_facts_with_backend(
    survey: &HardwareSurvey,
    count: usize,
    runner_backend: BackendKind,
) -> Vec<GpuBenchmarkAcceleratorFacts> {
    (0..count)
        .map(|index| default_fact(survey, index, runner_backend))
        .collect()
}

fn default_fact(
    survey: &HardwareSurvey,
    index: usize,
    runner_backend: BackendKind,
) -> GpuBenchmarkAcceleratorFacts {
    let gpu = survey.gpus.get(index);
    let unified_memory = gpu
        .map(|gpu| gpu.unified_memory)
        .unwrap_or_else(|| survey.is_soc || survey.gpus.iter().any(|gpu| gpu.unified_memory));
    let total_memory_bytes = gpu
        .and_then(|gpu| nonzero(gpu.vram_bytes))
        .or_else(|| survey.gpu_vram.get(index).copied().and_then(nonzero))
        .or_else(|| nonzero(survey.vram_bytes));
    let reserved_bytes = gpu
        .and_then(|gpu| gpu.reserved_bytes)
        .or_else(|| survey.gpu_reserved.get(index).copied().flatten());
    let name = gpu
        .map(|gpu| gpu.display_name.clone())
        .filter(|name| !name.is_empty())
        .or_else(|| survey.gpu_name.clone());
    let backend = gpu
        .and_then(|gpu| gpu.backend_device.as_deref())
        .map(infer_backend_from_device)
        .filter(|backend| *backend != BackendKind::Unknown)
        .or_else(|| {
            let inferred = infer_backend_from_name(name.as_deref());
            (inferred != BackendKind::Unknown).then_some(inferred)
        })
        .or_else(|| (runner_backend != BackendKind::Unknown).then_some(runner_backend));
    GpuBenchmarkAcceleratorFacts {
        name,
        kind: if unified_memory {
            AcceleratorKind::IntegratedGpu
        } else {
            AcceleratorKind::DiscreteGpu
        },
        backend,
        total_memory_bytes,
        available_memory_bytes: total_memory_bytes
            .map(|total| total.saturating_sub(reserved_bytes.unwrap_or(0))),
        unified_memory,
    }
}

fn memory_profile(
    survey: &HardwareSurvey,
    facts: &[GpuBenchmarkAcceleratorFacts],
) -> MemoryProfile {
    let detected_unified_total = facts
        .iter()
        .filter(|fact| fact.unified_memory)
        .filter_map(|fact| fact.total_memory_bytes)
        .max();
    let detected_unified_available = facts
        .iter()
        .filter(|fact| fact.unified_memory)
        .filter_map(|fact| fact.available_memory_bytes)
        .max();
    let total = system_total_memory_bytes()
        .or(detected_unified_total)
        .or_else(|| nonzero(survey.vram_bytes));
    let available = system_available_memory_bytes()
        .or(detected_unified_available)
        .or(total);
    let has_unified = facts.iter().any(|fact| fact.unified_memory)
        || survey.is_soc
        || survey.gpus.iter().any(|gpu| gpu.unified_memory);
    MemoryProfile {
        total_system_bytes: total,
        available_system_bytes: available,
        total_unified_bytes: has_unified.then_some(total).flatten(),
        available_unified_bytes: has_unified.then_some(available).flatten(),
    }
}

fn cpu_profile() -> CpuProfile {
    CpuProfile {
        physical_cores: None,
        logical_cores: std::thread::available_parallelism()
            .ok()
            .and_then(|count| u32::try_from(count.get()).ok()),
        memory_bandwidth_bytes_per_sec: None,
        compute_tflops_fp16: None,
        post_prefill_decode_overhead_ms: None,
        prefill_matmul_tflops_fp16: None,
        prefill_ubatch_matmul_tflops_fp16: None,
        prefill_moe_matmul_tflops_fp16: None,
        sampler_history_us_per_token: None,
        sampler_vocab_us_per_token: None,
    }
}

fn run_local_gpu_benchmark(args: &Args, survey: &HardwareSurvey) -> Result<LocalGpuBenchmark> {
    heartbeat(
        None,
        "hardware",
        "gpu_benchmark_start",
        "running local GPU benchmark",
    );
    let _status = TerminalStatus::start(
        args.show_progress,
        "Benchmarking local GPU memory bandwidth".into(),
    );
    let runner = benchmark_runner_for_survey(survey)?;
    let runner_backend = backend_kind_from_runner(runner.backend);
    let outputs = mesh_llm_gpu_bench::run_benchmark_with_options(
        runner,
        Duration::from_secs(300),
        mesh_llm_gpu_bench::BenchmarkOptions {
            // Keep the validator's automatic machine profile focused on the
            // portable facts that every fit needs: measured memory bandwidth,
            // launch overhead, prefill compute probes, sampler overhead, VRAM,
            // and accelerator identity. Validation then appends model-shaped
            // probes derived from the GGUF being tested.
            //
            // Standard/deep GGML probe mode is useful for manual backend
            // analysis and for `mesh-llm gpus benchmark`, but it is intentionally
            // not the automatic validation path. Those modes sweep generic GGML
            // graph shapes, which can make a smoke validation spend minutes in
            // hardware profiling before it has even looked at the model. More
            // importantly, generic probes are not a better source of truth than
            // metadata-shaped probes below: for a sparse MoE model, for example,
            // we derive the expert count, active experts, expert width, hidden
            // width, tensor type, and repeated layer depth directly from GGUF
            // metadata and run that exact graph.
            //
            // This keeps the estimator honest:
            // - hardware facts come from observed benchmark data, not marketing
            //   bandwidth or backend-specific constants;
            // - model-shaped corrections come from source-faithful graph probes
            //   keyed by metadata dimensions, not filenames or observed model
            //   throughput;
            // - validation remains repeatable enough to run as a smoke check on
            //   CUDA/Metal hosts without burning cycles on unrelated shapes.
            probe_depth: mesh_llm_gpu_bench::ProbeDepth::HardwareOnly,
        },
    )
    .context("run local GPU benchmark")?;
    heartbeat(
        None,
        "hardware",
        "gpu_benchmark_done",
        &format!("outputs={}", outputs.len()),
    );
    Ok(LocalGpuBenchmark {
        outputs,
        backend: runner_backend,
    })
}

fn backend_kind_from_runner(backend: mesh_llm_gpu_bench::BenchmarkBackend) -> BackendKind {
    match backend {
        mesh_llm_gpu_bench::BenchmarkBackend::Metal => BackendKind::Metal,
        mesh_llm_gpu_bench::BenchmarkBackend::Cuda => BackendKind::Cuda,
        mesh_llm_gpu_bench::BenchmarkBackend::Hip => BackendKind::Rocm,
        mesh_llm_gpu_bench::BenchmarkBackend::Intel => BackendKind::Vulkan,
    }
}

fn benchmark_runner_for_survey(
    survey: &HardwareSurvey,
) -> Result<mesh_llm_gpu_bench::BenchmarkRunner> {
    let gpu_name = survey
        .gpu_name
        .as_deref()
        .or_else(|| survey.gpus.first().map(|gpu| gpu.display_name.as_str()));
    let gpu_count = if survey.gpu_count > 0 {
        survey.gpu_count
    } else {
        u8::try_from(survey.gpus.len()).unwrap_or(u8::MAX)
    };
    let is_soc = survey.is_soc || survey.gpus.iter().any(|gpu| gpu.unified_memory);
    mesh_llm_gpu_bench::runner_for(std::env::consts::OS, gpu_count, gpu_name, is_soc)
        .context("could not infer GPU benchmark backend from local hardware")
}

fn infer_backend_from_survey(survey: &HardwareSurvey) -> BackendKind {
    if survey
        .gpus
        .iter()
        .filter_map(|gpu| gpu.backend_device.as_deref())
        .any(|device| infer_backend_from_device(device) == BackendKind::Metal)
        || std::env::consts::OS == "macos"
    {
        return BackendKind::Metal;
    }
    infer_backend_from_name(survey.gpu_name.as_deref())
}

fn infer_backend_from_device(device: &str) -> BackendKind {
    let upper = device.to_ascii_uppercase();
    if upper.starts_with("MTL") || upper.contains("METAL") {
        BackendKind::Metal
    } else if upper.contains("CUDA") || upper.contains("NVIDIA") {
        BackendKind::Cuda
    } else if upper.contains("HIP") || upper.contains("ROCM") || upper.contains("AMD") {
        BackendKind::Rocm
    } else if upper.contains("VULKAN") {
        BackendKind::Vulkan
    } else {
        BackendKind::Unknown
    }
}

fn infer_backend_from_name(name: Option<&str>) -> BackendKind {
    let Some(name) = name else {
        return BackendKind::Unknown;
    };
    let upper = name.to_ascii_uppercase();
    if upper.contains("APPLE") || upper.contains("METAL") {
        BackendKind::Metal
    } else if upper.contains("NVIDIA") {
        BackendKind::Cuda
    } else if upper.contains("AMD") || upper.contains("RADEON") {
        BackendKind::Rocm
    } else {
        BackendKind::Unknown
    }
}

fn system_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        return command_u64("sysctl", &["-n", "hw.memsize"]);
    }
    #[cfg(target_os = "linux")]
    {
        return fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| meminfo_value_bytes(&text, "MemTotal:"));
    }
    #[allow(unreachable_code)]
    None
}

fn system_available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| meminfo_value_bytes(&text, "MemAvailable:"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn meminfo_value_bytes(text: &str, key: &str) -> Option<u64> {
    text.lines()
        .find(|line| line.starts_with(key))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib.saturating_mul(1024))
}

#[cfg(target_os = "macos")]
fn command_u64(program: &str, args: &[&str]) -> Option<u64> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then_some(output.stdout)
        .and_then(|stdout| String::from_utf8(stdout).ok())
        .and_then(|text| text.trim().parse::<u64>().ok())
}

fn nonzero(value: u64) -> Option<u64> {
    (value > 0).then_some(value)
}

fn selection_config(workload: &WorkloadProfile) -> SelectionConfig {
    let mut config = SelectionConfig {
        workload: workload.clone(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    config
}

fn score_workloads(
    hardware: &HardwareProfile,
    model: &ModelProfile,
) -> Vec<WorkloadRecommendation> {
    workload_profiles()
        .into_iter()
        .map(|(workload, profile)| WorkloadRecommendation {
            workload: workload.into(),
            recommendation: score_model(hardware, model, &selection_config(&profile)),
        })
        .collect()
}

fn workload_profiles() -> Vec<(&'static str, WorkloadProfile)> {
    vec![
        ("chat", WorkloadProfile::chat()),
        ("coding_agent", WorkloadProfile::coding_agent()),
        ("tool_calling", WorkloadProfile::tool_calling()),
        ("summarization", WorkloadProfile::summarization()),
        ("embedding", WorkloadProfile::embedding()),
        ("reranking", WorkloadProfile::reranking()),
        ("vision_chat", WorkloadProfile::vision_chat()),
        ("general_generation", WorkloadProfile::general_generation()),
    ]
}

fn primary_workload_label() -> &'static str {
    "chat"
}

fn primary_workload_profile() -> WorkloadProfile {
    WorkloadProfile::chat()
}

fn validation_prompt() -> &'static str {
    "You are validating local model throughput. Write a concise explanation of how memory bandwidth affects token generation speed."
}

fn benchmark_scenarios() -> Vec<BenchmarkScenarioSpec> {
    vec![
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::SteadyDecode,
            name: "steady_decode",
            fit_metric: "estimated_decode_tokens_per_sec",
            prompt: validation_prompt().into(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            warmup_tokens: DEFAULT_WARMUP_TOKENS,
            request_count: 3,
            reuse_session: false,
        },
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::Prefill,
            name: "prefill",
            fit_metric: "estimated_prefill_tokens_per_sec",
            prompt: first_token_prompt(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: FIRST_TOKEN_MAX_NEW_TOKENS,
            warmup_tokens: 0,
            request_count: 1,
            reuse_session: false,
        },
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::FirstToken,
            name: "first_token",
            fit_metric: "estimated_first_token_ms",
            prompt: first_token_prompt(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: FIRST_TOKEN_MAX_NEW_TOKENS,
            warmup_tokens: 0,
            request_count: 1,
            reuse_session: false,
        },
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::KvWarmReuse,
            name: "kv_warm_reuse",
            fit_metric: "warm_reuse_second_request_tokens_per_sec",
            prompt: kv_reuse_prompt(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: KV_WARM_REUSE_MAX_NEW_TOKENS,
            warmup_tokens: 0,
            request_count: 2,
            reuse_session: true,
        },
    ]
}

fn selected_benchmark_scenarios(args: &Args) -> Vec<BenchmarkScenarioSpec> {
    let scenarios = benchmark_scenarios();
    if args.benchmark_scenarios.is_empty() {
        return scenarios;
    }
    if args
        .benchmark_scenarios
        .iter()
        .any(|scenario| scenario == "all")
    {
        return scenarios;
    }
    let mut selected = Vec::new();
    for requested in &args.benchmark_scenarios {
        let scenario = scenarios
            .iter()
            .find(|scenario| scenario.name == requested)
            .cloned()
            .expect("scenario names are validated during argument parsing");
        if !selected
            .iter()
            .any(|existing: &BenchmarkScenarioSpec| existing.name == scenario.name)
        {
            selected.push(scenario);
        }
    }
    selected
}

fn fit_input_contract() -> FitInputContract {
    FitInputContract {
        hardware_fields_consumed: vec![
            "memory.available_system_bytes",
            "memory.available_unified_bytes",
            "accelerators.kind",
            "accelerators.backend",
            "accelerators.available_memory_bytes",
            "accelerators.memory_bandwidth_bytes_per_sec",
            "accelerators.decode_effective_bandwidth_bytes_per_sec",
            "accelerators.decode_fixed_overhead_ms",
            "accelerators.decode_runtime_overhead_ms",
            "accelerators.post_prefill_decode_overhead_ms",
            "accelerators.bandwidth_source",
            "accelerators.benchmark_noise_pct",
            "accelerators.compute_tflops_fp16",
            "accelerators.prefill_matmul_tflops_fp16",
            "accelerators.prefill_ubatch_matmul_tflops_fp16",
            "accelerators.prefill_moe_matmul_tflops_fp16",
            "accelerators.sampler_history_us_per_token",
            "accelerators.sampler_vocab_us_per_token",
            "accelerators.unified_memory",
            "cpu.memory_bandwidth_bytes_per_sec",
            "cpu.compute_tflops_fp16",
            "cpu.post_prefill_decode_overhead_ms",
            "cpu.prefill_matmul_tflops_fp16",
            "cpu.prefill_ubatch_matmul_tflops_fp16",
            "cpu.prefill_moe_matmul_tflops_fp16",
            "cpu.sampler_history_us_per_token",
            "cpu.sampler_vocab_us_per_token",
        ],
        model_fields_consumed: vec![
            "architecture",
            "architecture_class",
            "weight_coverage",
            "file_size_bytes",
            "tensor_bytes",
            "base_resident_bytes",
            "expert_tensor_bytes",
            "tensor_group_bytes",
            "tensor_matmul",
            "quantization",
            "layer_count",
            "hidden_size",
            "ffn_size",
            "attention_heads",
            "kv_heads",
            "key_length",
            "value_length",
            "context_length",
            "expert_count",
            "expert_used_count",
            "rope",
            "tokenizer",
            "capability_evidence",
        ],
        validation_backend: "skippy-bench local-single plus abi-decode-probe using skippy-server/llama.cpp full-model inference and the native skippy decode benchmark ABI",
        validation_note: "Validation observations exercise real GGML/llama.cpp model execution. They are reported as evidence only; observed model throughput and ABI probe throughput are not fed back into metadata-only fit scoring.",
    }
}

fn first_token_prompt() -> String {
    let seed = "Summarize this benchmark context into one operational takeaway: local inference fit depends on model bytes, active layers, KV cache pressure, backend overhead, and memory bandwidth.";
    std::iter::repeat_n(seed, 96).collect::<Vec<_>>().join("\n")
}

fn kv_reuse_prompt() -> String {
    "You are an agent inside a short tool loop. Inspect the same project context again and answer with the next concrete action.".into()
}

fn primary_download_path(
    artifact: &ResolvedModelArtifact,
    downloaded_paths: &[PathBuf],
) -> Result<PathBuf> {
    let primary = Path::new(&artifact.primary_file);
    downloaded_paths
        .iter()
        .find(|path| path.ends_with(primary))
        .or_else(|| downloaded_paths.first())
        .cloned()
        .with_context(|| format!("download produced no files for {}", artifact.canonical_ref))
}

fn download_progress(args: &Args, model_ref: &str) -> Option<ModelDownloadProgress> {
    args.show_progress.then(|| {
        let renderer = TerminalDownloadProgress::new(model_ref);
        ModelDownloadProgress::new(move |event| renderer.report(event))
    })
}

fn summarize(args: &Args, models: &[ModelValidationReport], tolerance: f64) -> ValidationSummary {
    let mut summary = ValidationSummary {
        model_count: models.len(),
        ..ValidationSummary::default()
    };
    let mut ratios = Vec::new();
    for model in models {
        count_model_summary(model, tolerance, &mut summary, &mut ratios);
    }
    ratios.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    summary.median_observed_over_fit = (!ratios.is_empty()).then(|| median(&ratios));
    summary.mean_observed_over_fit = mean(&ratios);
    summary.median_absolute_percent_error = median_absolute_percent_error(&ratios);
    summary.scenario_summaries = summarize_scenarios(args, models, tolerance);
    summary
}

fn summarize_scenarios(
    args: &Args,
    models: &[ModelValidationReport],
    tolerance: f64,
) -> Vec<ScenarioValidationSummary> {
    selected_benchmark_scenarios(args)
        .into_iter()
        .map(|scenario| summarize_scenario(models, scenario.name, tolerance))
        .collect()
}

fn summarize_scenario(
    models: &[ModelValidationReport],
    scenario: &str,
    tolerance: f64,
) -> ScenarioValidationSummary {
    let mut summary = ScenarioValidationSummary {
        scenario: scenario.into(),
        ..ScenarioValidationSummary::default()
    };
    let mut ratios = Vec::new();

    for model in models {
        let Some(benchmark) = model
            .benchmarks
            .iter()
            .find(|entry| entry.scenario == scenario)
        else {
            continue;
        };
        summary.sample_count += usize::from(benchmark.observed_over_fit.is_some());
        count_scenario_verdict(model, benchmark, tolerance, &mut summary, &mut ratios);
    }

    ratios.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    summary.median_observed_over_fit = (!ratios.is_empty()).then(|| median(&ratios));
    summary.mean_observed_over_fit = mean(&ratios);
    summary.median_absolute_percent_error = median_absolute_percent_error(&ratios);
    summary
}

fn count_scenario_verdict(
    model: &ModelValidationReport,
    benchmark: &BenchmarkScenarioSummary,
    tolerance: f64,
    summary: &mut ScenarioValidationSummary,
    ratios: &mut Vec<f64>,
) {
    match benchmark.verdict.as_str() {
        "match" => summary.matched_count += 1,
        "slower-than-fit" => summary.slower_than_fit_count += 1,
        "faster-than-fit" => summary.faster_than_fit_count += 1,
        "inconclusive-noisy" => summary.noisy_count += 1,
        "skipped" => summary.skipped_count += 1,
        "runtime-error" => summary.runtime_error_count += 1,
        "error" => summary.error_count += 1,
        _ => {}
    }
    if let Some(classification) = steady_decode_classification(model, benchmark) {
        count_decode_probe_classification(classification, summary);
    }
    if steady_decode_accuracy_exclusion(model, benchmark).is_some() {
        return;
    }
    if !accuracy_gated_verdict(&benchmark.verdict) {
        return;
    }
    if let Some(ratio) = benchmark.observed_over_fit {
        if (ratio - 1.0).abs() <= tolerance {
            summary.within_tolerance_count += 1;
        }
        ratios.push(ratio);
    }
}

fn steady_decode_accuracy_exclusion<'a>(
    model: &'a ModelValidationReport,
    benchmark: &BenchmarkScenarioSummary,
) -> Option<&'a str> {
    if benchmark.scenario != "steady_decode" || !accuracy_gated_verdict(&benchmark.verdict) {
        return None;
    }
    steady_decode_accuracy_exclusion_for_model(model)
}

fn steady_decode_classification<'a>(
    model: &'a ModelValidationReport,
    benchmark: &BenchmarkScenarioSummary,
) -> Option<&'a str> {
    if benchmark.scenario != "steady_decode" {
        return None;
    }
    steady_decode_classification_for_model(model)
}

fn steady_decode_classification_for_model(model: &ModelValidationReport) -> Option<&str> {
    model
        .decode_probe_diagnostic
        .as_ref()
        .map(|diagnostic| diagnostic.classification.as_str())
}

fn steady_decode_accuracy_exclusion_for_model(model: &ModelValidationReport) -> Option<&str> {
    let classification = steady_decode_classification_for_model(model)?;
    // The ±10% accuracy score is supposed to evaluate the metadata-only fit
    // estimate against a stable local decode path. When Skippy's observed
    // benchmark and the ABI decode probe disagree, that row is still valuable
    // evidence, but it is not clean estimator evidence: it mixes metadata cost,
    // probe representativeness, runtime/server path overhead, cache/session
    // behavior, and backend runtime state.
    //
    // Keep these cases visible as separate summary buckets instead of widening
    // the tolerance or silently counting them as fit failures. That follows the
    // repo-wide empirical rule: report residual misses honestly and avoid
    // tuning the estimator around a noisy local run.
    match classification {
        "steady_path_overhead_mismatch"
        | "runtime_path_mismatch"
        | "mixed_estimate_and_runtime_mismatch"
        | "abi_probe_noisy"
        | "probe_not_representative"
        | "unstable_probe_geometry" => Some(classification),
        _ => None,
    }
}

fn count_decode_probe_classification(
    classification: &str,
    summary: &mut ScenarioValidationSummary,
) {
    match classification {
        "metadata_estimate_miss" => summary.metadata_estimate_miss_count += 1,
        "steady_path_overhead_mismatch"
        | "runtime_path_mismatch"
        | "mixed_estimate_and_runtime_mismatch" => summary.runtime_path_mismatch_count += 1,
        "abi_probe_noisy" | "probe_not_representative" | "unstable_probe_geometry" => {
            summary.probe_mismatch_count += 1;
        }
        _ => {}
    }
}

fn count_decode_probe_classification_for_model(
    classification: &str,
    summary: &mut ValidationSummary,
) {
    match classification {
        "metadata_estimate_miss" => summary.metadata_estimate_miss_count += 1,
        "steady_path_overhead_mismatch"
        | "runtime_path_mismatch"
        | "mixed_estimate_and_runtime_mismatch" => summary.runtime_path_mismatch_count += 1,
        "abi_probe_noisy" | "probe_not_representative" | "unstable_probe_geometry" => {
            summary.probe_mismatch_count += 1;
        }
        _ => {}
    }
}

fn count_model_summary(
    model: &ModelValidationReport,
    tolerance: f64,
    summary: &mut ValidationSummary,
    ratios: &mut Vec<f64>,
) {
    match model.benchmark.verdict.as_str() {
        "match" => summary.matched_count += 1,
        "slower-than-fit" => summary.slower_than_fit_count += 1,
        "faster-than-fit" => summary.faster_than_fit_count += 1,
        "inconclusive-noisy" => summary.noisy_count += 1,
        "skipped" => summary.skipped_count += 1,
        "runtime-error" => summary.runtime_error_count += 1,
        "error" => summary.error_count += 1,
        _ => {}
    }
    if model.benchmark.attempted {
        summary.benchmarked_count += 1;
    }
    if let Some(classification) = steady_decode_classification_for_model(model) {
        count_decode_probe_classification_for_model(classification, summary);
    }
    if steady_decode_accuracy_exclusion_for_model(model).is_some() {
        return;
    }
    if !accuracy_gated_verdict(&model.benchmark.verdict) {
        return;
    }
    if let Some(ratio) = model.benchmark.observed_over_fit {
        if (ratio - 1.0).abs() <= tolerance {
            summary.within_tolerance_count += 1;
        }
        ratios.push(ratio);
    }
}

fn accuracy_gated_verdict(verdict: &str) -> bool {
    matches!(verdict, "match" | "slower-than-fit" | "faster-than-fit")
}

fn median_absolute_percent_error(ratios: &[f64]) -> Option<f64> {
    if ratios.is_empty() {
        return None;
    }
    let mut errors = ratios
        .iter()
        .map(|ratio| (ratio - 1.0).abs() * 100.0)
        .collect::<Vec<_>>();
    errors.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(median(&errors))
}

fn error_report(input_ref: String, error: String) -> ModelValidationReport {
    ModelValidationReport {
        input_ref,
        resolved_ref: None,
        artifact: None,
        downloaded_paths: Vec::new(),
        primary_gguf_path: None,
        model_profile: None,
        recommendation: None,
        fit_interpretation: None,
        runtime_diagnostic: None,
        recommendations: Vec::new(),
        abi_decode_probe: None,
        decode_probe_diagnostic: None,
        graph_inventory_diagnostic: None,
        operation_bucket_diagnostic: None,
        model_specific_decode_kernel_probes: Vec::new(),
        model_specific_probe_errors: Vec::new(),
        benchmarks: Vec::new(),
        benchmark: BenchmarkSummary {
            verdict: "error".into(),
            errors: vec![error.clone()],
            ..BenchmarkSummary::default()
        },
        errors: vec![error],
    }
}

fn print_markdown_table(rows: &[ModelValidationReport]) {
    println!(
        "| model_ref | fit | meaning | backend | est tok/s | abi tok/s | sample/sync tok/s | abi overhead | est range | steady median | steady/fit | steady/abi | decode diag | steady | first-token | kv-reuse |"
    );
    println!("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|");
    for row in rows {
        println!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.input_ref,
            fit_status(row),
            display_fit_meaning(row),
            display_selected_backend(row),
            display_steady_estimated_tps(row),
            display_abi_decode_probe(row),
            display_abi_sampling_probe(row),
            display_abi_non_eval_overhead(row),
            display_steady_estimated_range(row),
            display_steady_observed(row),
            display_steady_observed_over_fit(row),
            display_observed_over_abi(row),
            display_decode_probe_classification(row),
            scenario_verdict(row, "steady_decode"),
            scenario_verdict(row, "first_token"),
            scenario_verdict(row, "kv_warm_reuse"),
        );
    }
    print_dense_probe_ladder_table(rows);
    print_graph_inventory_diagnostic_table(rows);
    print_operation_bucket_diagnostic_table(rows);
}

fn print_graph_inventory_diagnostic_table(rows: &[ModelValidationReport]) {
    let rows_with_inventory = rows
        .iter()
        .filter(|row| {
            row.graph_inventory_diagnostic
                .as_ref()
                .is_some_and(|diagnostic| diagnostic.available)
        })
        .collect::<Vec<_>>();
    if rows_with_inventory.is_empty() {
        return;
    }

    println!();
    println!("Graph inventory diagnostic");
    println!(
        "| model_ref | status | graph nodes | selected probe layers | transformer src0/meta | transformer+unclassified src0/meta | unclassified matmul src0 | transformer nodes graph/meta | transformer ms / ABI ms | notes |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---|");
    for row in rows_with_inventory {
        let diagnostic = row
            .graph_inventory_diagnostic
            .as_ref()
            .expect("filtered row has graph inventory diagnostic");
        println!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.input_ref,
            diagnostic.status,
            display_option_u64(diagnostic.graph_node_count),
            display_option_u32(diagnostic.selected_transformer_probe_layers),
            display_option_ratio(diagnostic.graph_transformer_src0_over_metadata),
            display_option_ratio(diagnostic.graph_transformer_plus_unclassified_src0_over_metadata),
            diagnostic.graph_unclassified_matmul_src0_bytes,
            display_graph_node_ratio(diagnostic),
            display_option_ratio(diagnostic.estimated_transformer_over_abi),
            display_graph_inventory_notes(diagnostic),
        );
    }
}

fn print_operation_bucket_diagnostic_table(rows: &[ModelValidationReport]) {
    let rows_with_buckets = rows
        .iter()
        .filter(|row| {
            row.operation_bucket_diagnostic
                .as_ref()
                .is_some_and(|diagnostic| diagnostic.available)
        })
        .collect::<Vec<_>>();
    if rows_with_buckets.is_empty() {
        return;
    }

    println!();
    println!("Operation bucket diagnostic");
    println!(
        "| model_ref | bucket | source | est ms | est share | graph families | graph nodes | graph src0 | graph src1 | graph output | src0/meta | notes |"
    );
    println!("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|");
    for row in rows_with_buckets {
        let diagnostic = row
            .operation_bucket_diagnostic
            .as_ref()
            .expect("filtered row has operation bucket diagnostic");
        for bucket in &diagnostic.buckets {
            println!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                row.input_ref,
                bucket.bucket,
                bucket.source,
                display_option_ms(bucket.estimated_ms),
                display_option_ratio(bucket.estimated_share_of_selected_ms),
                bucket.graph_families.join(", "),
                bucket.graph_node_count,
                bucket.graph_src0_bytes,
                bucket.graph_src1_bytes,
                bucket.graph_output_bytes,
                display_option_ratio(bucket.graph_src0_over_metadata),
                display_operation_bucket_notes(bucket),
            );
        }
    }
}

fn print_dense_probe_ladder_table(rows: &[ModelValidationReport]) {
    let rows_with_dense_probes = rows
        .iter()
        .filter(|row| dense_probe_ladder_available(row))
        .collect::<Vec<_>>();
    if rows_with_dense_probes.is_empty() {
        return;
    }

    println!();
    println!("Dense probe ladder diagnostic");
    println!(
        "| model_ref | observed tok/s | fit tok/s | selected probe | l1 tok/s (GB/s) | l4 tok/s (GB/s) | l8 tok/s (GB/s) | l16 tok/s (GB/s) | full-depth tok/s (GB/s) |"
    );
    println!("|---|---:|---:|---|---:|---:|---:|---:|---:|");
    for row in rows_with_dense_probes {
        println!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.input_ref,
            display_steady_observed(row),
            display_steady_estimated_tps(row),
            display_selected_dense_probe(row),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(1)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(4)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(8)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(16)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::FullDepth),
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum DenseProbeLadderSlot {
    Layers(u32),
    FullDepth,
}

fn dense_probe_ladder_available(row: &ModelValidationReport) -> bool {
    selected_dense_transformer_group(row).is_some()
        && row
            .model_specific_decode_kernel_probes
            .iter()
            .any(is_dense_llama_graph_probe)
}

fn display_selected_dense_probe(row: &ModelValidationReport) -> String {
    selected_dense_transformer_group(row)
        .and_then(|group| group.probe_name.as_deref())
        .map(|name| format!("`{name}`"))
        .unwrap_or_else(|| "-".into())
}

fn display_dense_probe_ladder_cell(
    row: &ModelValidationReport,
    slot: DenseProbeLadderSlot,
) -> String {
    let Some(probe) = dense_probe_for_ladder_slot(row, slot) else {
        return "-".into();
    };
    let Some(implied_tps) = dense_probe_implied_tokens_per_second(row, probe) else {
        return format!("- ({:.0})", probe.effective_gbps);
    };
    format!("{implied_tps:.1} ({:.0})", probe.effective_gbps)
}

fn dense_probe_for_ladder_slot(
    row: &ModelValidationReport,
    slot: DenseProbeLadderSlot,
) -> Option<&DecodeKernelProbe> {
    let group = selected_dense_transformer_group(row)?;
    let target_layers = match slot {
        DenseProbeLadderSlot::Layers(layers) => layers,
        DenseProbeLadderSlot::FullDepth => row.model_profile.as_ref()?.layer_count?,
    };
    row.model_specific_decode_kernel_probes
        .iter()
        .filter(|probe| {
            is_dense_llama_graph_probe(probe)
                && dense_probe_layers(probe) == target_layers
                && group.probe_rows.is_none_or(|rows| probe.rows == rows)
                && group.probe_cols.is_none_or(|cols| probe.cols == cols)
                && probe.tensor_type.eq_ignore_ascii_case(&group.tensor_type)
        })
        .max_by(|left, right| {
            left.effective_gbps
                .partial_cmp(&right.effective_gbps)
                .unwrap_or(Ordering::Equal)
        })
}

fn dense_probe_implied_tokens_per_second(
    row: &ModelValidationReport,
    probe: &DecodeKernelProbe,
) -> Option<f64> {
    // This is a diagnostic, not an alternate scoring path. It asks:
    // "If this synthetic dense graph row supplied only the transformer-block
    // timing, and every other cost term from the already-produced recommendation
    // stayed the same, what tok/s would the model-fit estimate imply?"
    //
    // That framing lets the report compare l1/l4/l8/l16/full-depth probe rows
    // against observed steady decode without silently changing model-fit's
    // deterministic selector. It also exposes synthetic graph artifacts: if a
    // full-depth row whipsaws while observed tok/s is stable, the row is
    // validation evidence, not a better estimator.
    let recommendation = row.recommendation.as_ref()?;
    let breakdown = recommendation.decode_cost_breakdown.as_ref()?;
    let group = selected_dense_transformer_group(row)?;
    let model_layers = f64::from(row.model_profile.as_ref()?.layer_count?);
    let probe_layers = f64::from(dense_probe_layers(probe).max(1));
    let probe_elapsed_ms = probe.elapsed_ms?;
    let variable_probe_ms = (probe_elapsed_ms - f64::from(breakdown.fixed_overhead_ms)).max(0.0);
    let candidate_block_ms = variable_probe_ms * (model_layers / probe_layers);
    let original_block_ms = f64::from(group.bandwidth_ms);
    let other_bandwidth_ms = (f64::from(breakdown.bandwidth_ms) - original_block_ms).max(0.0);
    let candidate_bandwidth_ms = other_bandwidth_ms + candidate_block_ms;
    let overhead_ms = f64::from(breakdown.fixed_overhead_ms)
        + f64::from(breakdown.runtime_overhead_ms)
        + f64::from(breakdown.measured_graph_overhead_ms)
        + f64::from(breakdown.architecture_overhead_ms)
        + f64::from(breakdown.sampled_decode_sampler_ms);
    let candidate_total_ms =
        candidate_bandwidth_ms.max(f64::from(breakdown.compute_ms)) + overhead_ms;
    (candidate_total_ms > 0.0).then_some(1000.0 / candidate_total_ms)
}

fn selected_dense_transformer_group(
    row: &ModelValidationReport,
) -> Option<&model_fit::DecodeCostGroupBreakdown> {
    row.recommendation
        .as_ref()?
        .decode_cost_breakdown
        .as_ref()?
        .groups
        .iter()
        .find(|group| group.group == "transformer_block" && group.probe_name.is_some())
}

fn is_dense_llama_graph_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("llama_graph")
}

fn dense_probe_layers(probe: &DecodeKernelProbe) -> u32 {
    let name = probe.name.to_ascii_lowercase();
    let Some((_, suffix)) = name.split_once("_llama_graph_l") else {
        return 1;
    };
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse::<u32>().unwrap_or(1).max(1)
}

fn decode_probe_diagnostic(
    recommendation: &ModelRecommendation,
    abi_probe: Option<&AbiDecodeProbeSummary>,
    steady_benchmark: Option<&BenchmarkScenarioSummary>,
) -> Option<DecodeProbeDiagnostic> {
    if !matches!(
        recommendation.fit_status,
        FitStatus::FitsLocal | FitStatus::FitsWithWarning
    ) {
        return None;
    }

    let predicted = steady_benchmark
        .and_then(|benchmark| benchmark.predicted)
        .or_else(|| {
            recommendation
                .estimated_decode_tokens_per_sec
                .map(f64::from)
        });
    let abi = abi_probe.and_then(|probe| probe.tokens_per_second);
    let observed = steady_benchmark
        .and_then(|benchmark| benchmark.observed)
        .or_else(|| {
            steady_benchmark.and_then(|benchmark| benchmark.benchmark.median_tokens_per_sec)
        });
    let observed_over_fit = ratio(observed, predicted);
    let abi_over_fit = ratio(abi, predicted);
    let observed_over_abi = ratio(observed, abi);
    let observed_vs_fit = throughput_ratio_verdict(observed_over_fit);
    let abi_vs_fit = throughput_ratio_verdict(abi_over_fit);
    let observed_vs_abi = throughput_ratio_verdict(observed_over_abi);
    let abi_probe_noisy = abi_probe
        .and_then(|probe| probe.spread_pct)
        .is_some_and(|spread| spread > DEFAULT_MAX_SPREAD * 100.0);
    let classification = decode_probe_classification(
        predicted,
        abi,
        observed,
        abi_probe_noisy,
        observed_over_fit,
        abi_over_fit,
        observed_over_abi,
    );
    let notes = decode_probe_notes(
        observed_over_fit,
        abi_over_fit,
        observed_over_abi,
        &classification,
    );
    Some(DecodeProbeDiagnostic {
        predicted_tokens_per_second: predicted,
        abi_tokens_per_second: abi,
        observed_tokens_per_second: observed,
        observed_over_fit,
        abi_over_fit,
        observed_over_abi,
        observed_vs_fit,
        abi_vs_fit,
        observed_vs_abi,
        classification,
        notes,
    })
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0.0 => Some(numerator / denominator),
        _ => None,
    }
}

fn throughput_ratio_verdict(ratio: Option<f64>) -> String {
    let Some(ratio) = ratio else {
        return "missing".into();
    };
    if (ratio - 1.0).abs() <= DEFAULT_TOLERANCE {
        "match".into()
    } else if ratio < 1.0 {
        "slower-than-reference".into()
    } else {
        "faster-than-reference".into()
    }
}

fn decode_probe_classification(
    predicted: Option<f64>,
    abi: Option<f64>,
    observed: Option<f64>,
    abi_probe_noisy: bool,
    observed_over_fit: Option<f64>,
    abi_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
) -> String {
    if predicted.is_none() {
        return "missing_fit_estimate".into();
    }
    if observed.is_none() {
        return "missing_observed_benchmark".into();
    }
    if abi.is_none() {
        return "missing_abi_probe".into();
    }
    if abi_probe_noisy {
        return "abi_probe_noisy".into();
    }

    let fit_observed_matches = ratio_matches(observed_over_fit);
    let fit_abi_matches = ratio_matches(abi_over_fit);
    let abi_observed_matches = ratio_matches(observed_over_abi);
    match (fit_observed_matches, fit_abi_matches, abi_observed_matches) {
        (true, true, true) => "estimate_and_probe_agree".into(),
        (false, _, true) => "metadata_estimate_miss".into(),
        (false, true, false) => {
            if ratio_is_slower(observed_over_fit) && ratio_is_slower(observed_over_abi) {
                "steady_path_overhead_mismatch".into()
            } else {
                "runtime_path_mismatch".into()
            }
        }
        (false, false, false) => "mixed_estimate_and_runtime_mismatch".into(),
        (true, false, false) => "probe_not_representative".into(),
        (true, false, true) => "probe_differs_but_observed_matches_fit".into(),
        (true, true, false) => "unstable_probe_geometry".into(),
    }
}

fn ratio_matches(ratio: Option<f64>) -> bool {
    ratio.is_some_and(|ratio| (ratio - 1.0).abs() <= DEFAULT_TOLERANCE)
}

fn ratio_is_slower(ratio: Option<f64>) -> bool {
    ratio.is_some_and(|ratio| ratio < 1.0 - DEFAULT_TOLERANCE)
}

fn decode_probe_notes(
    observed_over_fit: Option<f64>,
    abi_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    classification: &str,
) -> Vec<String> {
    let mut notes = Vec::new();
    notes.push(
        "ABI probe is validation evidence only; it is not fed into metadata-only fit scoring."
            .into(),
    );
    if let Some(ratio) = observed_over_fit {
        notes.push(format!(
            "Observed steady decode is {:.1}% of the metadata estimate.",
            ratio * 100.0
        ));
    }
    if let Some(ratio) = abi_over_fit {
        notes.push(format!(
            "ABI decode probe is {:.1}% of the metadata estimate.",
            ratio * 100.0
        ));
    }
    if let Some(ratio) = observed_over_abi {
        notes.push(format!(
            "Observed steady decode is {:.1}% of the ABI decode probe.",
            ratio * 100.0
        ));
    }
    match classification {
        "metadata_estimate_miss" => notes.push(
            "ABI and observed decode agree, so the miss points at metadata cost modeling.".into(),
        ),
        "abi_probe_noisy" => notes.push(
            "The ABI decode probe exceeded the validator spread threshold, so this row is reported as probe-noise evidence rather than clean estimator accuracy evidence."
                .into(),
        ),
        "steady_path_overhead_mismatch" => notes.push(
            "ABI and metadata agree, but steady decode is slower; this points at validation/runtime path overhead such as server scheduling, sampler/session work, or benchmark scenario overhead rather than the metadata-only estimator."
                .into(),
        ),
        "runtime_path_mismatch" | "mixed_estimate_and_runtime_mismatch" => notes.push(
            "Observed decode diverges from the ABI probe, so runtime path, graph, cache, or benchmark shape needs inspection.".into(),
        ),
        "probe_not_representative" => notes.push(
            "The metadata estimate matched observed decode, but the ABI probe did not represent full runtime throughput.".into(),
        ),
        _ => {}
    }
    notes
}

fn graph_inventory_diagnostic(
    profile: &ModelProfile,
    recommendation: &ModelRecommendation,
    abi_probe: Option<&AbiDecodeProbeSummary>,
) -> Option<GraphInventoryDiagnostic> {
    let abi_probe = abi_probe?;
    let comparisons = graph_inventory_comparisons(profile, abi_probe);
    let metadata_transformer_weight_bytes = graph_metadata_transformer_bytes(profile);
    let graph_transformer_weight_src0_bytes = graph_transformer_src0_bytes(profile, abi_probe);
    let graph_unclassified_matmul_src0_bytes = graph_family_src0_bytes(abi_probe, "matmul");
    let graph_transformer_plus_unclassified_src0_bytes =
        graph_transformer_weight_src0_bytes.saturating_add(graph_unclassified_matmul_src0_bytes);
    let metadata_transformer_matmul_nodes = graph_metadata_transformer_nodes(profile);
    let graph_transformer_matmul_nodes = graph_transformer_node_count(profile, abi_probe);
    let transformer_cost_group = graph_transformer_cost_group(profile);
    let transformer_group = recommendation
        .decode_cost_breakdown
        .as_ref()
        .and_then(|breakdown| {
            breakdown
                .groups
                .iter()
                .find(|group| group.group == transformer_cost_group)
        });
    let selected_transformer_probe = transformer_group.and_then(|group| group.probe_name.clone());
    let selected_transformer_probe_layers = selected_transformer_probe
        .as_deref()
        .map(graph_probe_layers_from_name);
    let estimated_transformer_block_ms =
        transformer_group.map(|group| f64::from(group.bandwidth_ms));
    let abi_ms_per_token = match (abi_probe.measured_tokens, abi_probe.elapsed_ms) {
        (Some(tokens), Some(elapsed_ms)) if tokens > 0 => Some(elapsed_ms / tokens as f64),
        _ => abi_probe
            .tokens_per_second
            .filter(|tps| *tps > 0.0)
            .map(|tps| 1000.0 / tps),
    };
    let estimated_transformer_over_abi = ratio(estimated_transformer_block_ms, abi_ms_per_token);
    let graph_transformer_src0_over_metadata = ratio_u64(
        graph_transformer_weight_src0_bytes,
        metadata_transformer_weight_bytes,
    );
    let graph_transformer_plus_unclassified_src0_over_metadata = ratio_u64(
        graph_transformer_plus_unclassified_src0_bytes,
        metadata_transformer_weight_bytes,
    );
    let mut notes = graph_inventory_notes(
        metadata_transformer_weight_bytes,
        graph_transformer_weight_src0_bytes,
        graph_unclassified_matmul_src0_bytes,
        metadata_transformer_matmul_nodes,
        graph_transformer_matmul_nodes,
        selected_transformer_probe_layers,
        estimated_transformer_over_abi,
    );
    if abi_probe.graph_inventory_bucket_overflow_count.unwrap_or(0) > 0 {
        notes.push(
            "Graph inventory bucket overflowed; comparisons are partial and should not drive estimator changes."
                .into(),
        );
    }
    let available = !abi_probe.graph_inventory.is_empty();
    let status = graph_inventory_status(
        &comparisons,
        selected_transformer_probe_layers,
        graph_unclassified_matmul_src0_bytes,
        graph_transformer_plus_unclassified_src0_over_metadata,
    );
    Some(GraphInventoryDiagnostic {
        available,
        status,
        graph_node_count: abi_probe.graph_node_count,
        graph_inventory_bucket_overflow_count: abi_probe.graph_inventory_bucket_overflow_count,
        selected_transformer_probe,
        selected_transformer_probe_layers,
        metadata_transformer_matmul_nodes,
        graph_transformer_matmul_nodes,
        metadata_transformer_weight_bytes,
        graph_transformer_weight_src0_bytes,
        graph_unclassified_matmul_src0_bytes,
        graph_transformer_src0_over_metadata,
        graph_transformer_plus_unclassified_src0_over_metadata,
        estimated_transformer_block_ms,
        abi_ms_per_token,
        estimated_transformer_over_abi,
        comparisons,
        notes,
    })
}

fn graph_inventory_comparisons(
    profile: &ModelProfile,
    abi_probe: &AbiDecodeProbeSummary,
) -> Vec<GraphInventoryComparison> {
    let mut comparisons = vec![graph_inventory_comparison(
        "attention_matmul",
        profile.tensor_matmul.attention.bytes,
        profile.tensor_matmul.attention.shape.logical_matrix_count,
        abi_probe,
        "attention_matmul",
    )];

    if profile.architecture_class == model_fit::ModelArchitectureClass::SparseMoeTransformer {
        // llama.cpp does not lower sparse MoE expert FFN as the same graph
        // family as dense FFN. The routed expert path uses
        // GGML_OP_MUL_MAT_ID: one graph node per up/gate/down expert matrix
        // group for each layer, with the expert dimension packed behind the
        // tensor and selected by ids at runtime. Comparing GGUF
        // `expert_feed_forward` bytes against the dense `ffn_matmul` family
        // made the validator report a false inventory mismatch even when the
        // graph had exactly the routed expert bytes we expected.
        comparisons.push(graph_inventory_comparison(
            "expert_moe_matmul_id",
            profile.tensor_matmul.expert_feed_forward.bytes,
            sparse_moe_expected_expert_matmul_id_nodes(profile),
            abi_probe,
            "moe_matmul_id",
        ));
    } else {
        comparisons.push(graph_inventory_comparison(
            "ffn_matmul",
            profile.tensor_matmul.feed_forward.bytes,
            profile
                .tensor_matmul
                .feed_forward
                .shape
                .logical_matrix_count,
            abi_probe,
            "ffn_matmul",
        ));
    }

    comparisons.push(graph_inventory_comparison(
        "output_matmul",
        graph_expected_output_bytes(profile),
        graph_expected_output_nodes(profile),
        abi_probe,
        "output_matmul",
    ));

    comparisons
}

fn graph_transformer_cost_group(profile: &ModelProfile) -> &'static str {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => "sparse_transformer_block",
        _ => "transformer_block",
    }
}

fn sparse_moe_expected_expert_matmul_id_nodes(profile: &ModelProfile) -> u64 {
    if profile.tensor_matmul.expert_feed_forward.bytes == 0 {
        return 0;
    }

    profile
        .layer_count
        .filter(|layers| *layers > 0)
        .map(|layers| u64::from(layers).saturating_mul(3))
        .unwrap_or_else(|| {
            profile
                .tensor_matmul
                .expert_feed_forward
                .shape
                .logical_matrix_count
        })
}

fn graph_inventory_comparison(
    name: &'static str,
    metadata_weight_bytes: u64,
    metadata_node_count: u64,
    abi_probe: &AbiDecodeProbeSummary,
    family: &str,
) -> GraphInventoryComparison {
    let graph_weight_src0_bytes = graph_family_src0_bytes(abi_probe, family);
    let graph_node_count = graph_family_node_count(abi_probe, family);
    GraphInventoryComparison {
        name,
        metadata_weight_bytes,
        metadata_node_count,
        graph_weight_src0_bytes,
        graph_node_count,
        src0_over_metadata: ratio_u64(graph_weight_src0_bytes, metadata_weight_bytes),
        node_count_delta: i64::try_from(graph_node_count).unwrap_or(i64::MAX)
            - i64::try_from(metadata_node_count).unwrap_or(i64::MAX),
    }
}

fn graph_expected_output_bytes(profile: &ModelProfile) -> u64 {
    if profile.tensor_matmul.output.bytes > 0 || profile.tensor_group_bytes.output_bytes > 0 {
        return profile
            .tensor_matmul
            .output
            .bytes
            .max(profile.tensor_group_bytes.output_bytes);
    }
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::DenseTransformer
        | model_fit::ModelArchitectureClass::SparseMoeTransformer
        | model_fit::ModelArchitectureClass::Unknown => profile.tensor_group_bytes.embedding_bytes,
        _ => 0,
    }
}

fn graph_expected_output_nodes(profile: &ModelProfile) -> u64 {
    if graph_expected_output_bytes(profile) == 0 {
        0
    } else {
        profile
            .tensor_matmul
            .output
            .shape
            .logical_matrix_count
            .max(1)
    }
}

fn graph_family_src0_bytes(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.src0_bytes)
        .sum()
}

fn graph_family_node_count(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.node_count)
        .sum()
}

fn graph_inventory_status(
    comparisons: &[GraphInventoryComparison],
    selected_probe_layers: Option<u32>,
    graph_unclassified_matmul_src0_bytes: u64,
    graph_transformer_plus_unclassified_src0_over_metadata: Option<f64>,
) -> String {
    let inventory_mismatch = comparisons.iter().any(|comparison| {
        comparison
            .src0_over_metadata
            .is_some_and(|ratio| (ratio - 1.0).abs() > DEFAULT_TOLERANCE)
            || comparison.node_count_delta != 0
    });
    if inventory_mismatch
        && graph_unclassified_matmul_src0_bytes > 0
        && graph_transformer_plus_unclassified_src0_over_metadata
            .is_some_and(|ratio| (ratio - 1.0).abs() <= DEFAULT_TOLERANCE)
    {
        "metadata_inventory_has_unclassified_matmul".into()
    } else if inventory_mismatch
        && graph_transformer_plus_unclassified_src0_over_metadata
            .is_some_and(|ratio| ratio < 1.0 - DEFAULT_TOLERANCE)
    {
        "metadata_inventory_missing_transformer_matmul".into()
    } else if inventory_mismatch {
        "metadata_inventory_mismatch".into()
    } else if selected_probe_layers == Some(1) {
        "metadata_inventory_matches_probe_depth_risk".into()
    } else {
        "metadata_inventory_matches".into()
    }
}

fn graph_inventory_notes(
    metadata_transformer_weight_bytes: u64,
    graph_transformer_weight_src0_bytes: u64,
    graph_unclassified_matmul_src0_bytes: u64,
    metadata_transformer_matmul_nodes: u64,
    graph_transformer_matmul_nodes: u64,
    selected_probe_layers: Option<u32>,
    estimated_transformer_over_abi: Option<f64>,
) -> Vec<String> {
    let mut notes = Vec::new();
    if ratio_u64(
        graph_transformer_weight_src0_bytes,
        metadata_transformer_weight_bytes,
    )
    .is_some_and(|ratio| (ratio - 1.0).abs() <= DEFAULT_TOLERANCE)
        && metadata_transformer_matmul_nodes == graph_transformer_matmul_nodes
    {
        notes.push(
            "GGUF tensor inventory matches the llama.cpp transformer matmul graph; the miss is likely timing/probe representation, not tensor grouping."
                .into(),
        );
    }
    if selected_probe_layers == Some(1) && graph_transformer_matmul_nodes > 0 {
        notes.push(
            "The selected transformer timing comes from a one-layer synthetic graph while the native decode graph contains the full repeated-layer matmul inventory."
                .into(),
        );
    }
    if graph_unclassified_matmul_src0_bytes > 0 {
        notes.push(format!(
            "Native graph has {:.1} MiB of unclassified GGML_OP_MUL_MAT src0 bytes; inspect source node names before treating a known-family mismatch as missing model work.",
            graph_unclassified_matmul_src0_bytes as f64 / 1024.0 / 1024.0
        ));
    }
    if let Some(ratio) = estimated_transformer_over_abi {
        notes.push(format!(
            "Estimated transformer-block time is {:.1}% of total ABI ms/token.",
            ratio * 100.0
        ));
    }
    notes
}

fn operation_bucket_diagnostic(
    profile: &ModelProfile,
    recommendation: &ModelRecommendation,
    abi_probe: Option<&AbiDecodeProbeSummary>,
) -> Option<OperationBucketDiagnostic> {
    let abi_probe = abi_probe?;
    let breakdown = recommendation.decode_cost_breakdown.as_ref();
    let selected_ms = breakdown.map(|breakdown| f64::from(breakdown.selected_time_ms));
    let abi_ms = abi_ms_per_token(abi_probe);
    let buckets = operation_bucket_rows(profile, breakdown, abi_probe, selected_ms);
    let raw_graph_families = graph_operation_family_rows(abi_probe);
    let available = !buckets.is_empty() || !raw_graph_families.is_empty();
    let notes = operation_bucket_notes(breakdown.is_some());
    Some(OperationBucketDiagnostic {
        available,
        estimated_selected_ms_per_token: selected_ms,
        abi_ms_per_token: abi_ms,
        estimated_over_abi: ratio(selected_ms, abi_ms),
        buckets,
        raw_graph_families,
        notes,
    })
}

fn operation_bucket_rows(
    profile: &ModelProfile,
    breakdown: Option<&model_fit::DecodeCostBreakdown>,
    abi_probe: &AbiDecodeProbeSummary,
    selected_ms: Option<f64>,
) -> Vec<OperationBucketRow> {
    operation_bucket_specs(profile)
        .into_iter()
        .map(|spec| operation_bucket_row(spec, breakdown, abi_probe, selected_ms))
        .collect()
}

fn operation_bucket_specs(profile: &ModelProfile) -> Vec<OperationBucketSpec> {
    let mut specs = Vec::new();
    if profile.architecture_class == model_fit::ModelArchitectureClass::SparseMoeTransformer {
        specs.push(OperationBucketSpec {
            bucket: "sparse_transformer_block",
            graph_families: &["attention_matmul", "moe_matmul_id"],
            cost_group: "sparse_transformer_block",
            metadata_weight_bytes: graph_metadata_transformer_bytes(profile),
            note: "Sparse MoE scoring charges the llama.cpp token graph as attention plus routed expert GGML_OP_MUL_MAT_ID work; dense FFN buckets are not the right comparison for expert tensors.",
        });
        specs.push(OperationBucketSpec {
            bucket: "moe_router_and_runtime",
            graph_families: &["moe_runtime"],
            cost_group: "moe_router_and_runtime",
            metadata_weight_bytes: profile.tensor_matmul.feed_forward.bytes,
            note: "MoE router/gating work is already timed inside the sparse transformer block probe when that probe is selected; this row reports llama.cpp graph inventory only and must not borrow KV fallback timing or observed tok/s.",
        });
    } else {
        specs.push(OperationBucketSpec {
            bucket: "transformer_block",
            graph_families: &["attention_matmul", "ffn_matmul"],
            cost_group: "transformer_block",
            metadata_weight_bytes: graph_metadata_transformer_bytes(profile),
            note: "Scoring charges this as one scheduled llama.cpp token graph; the attention/FFN family split is diagnostic and must not become architecture-name logic.",
        });
    }
    specs.push(OperationBucketSpec {
        bucket: "output_matmul",
        graph_families: &["output_matmul"],
        cost_group: "output_matmul",
        metadata_weight_bytes: graph_expected_output_bytes(profile),
        note: "Output projection is separate from the repeated transformer block because vocab-sized logits can have a very different matrix shape.",
    });
    specs.push(OperationBucketSpec {
        bucket: "kv_and_activation",
        graph_families: &[
            "kv_cache",
            "attention_runtime",
            "ffn_runtime",
            "normalization",
        ],
        cost_group: "kv_and_activation",
        metadata_weight_bytes: 0,
        note: "Runtime buckets are source graph work, but model-fit currently estimates them as one metadata-derived non-weight group rather than independent per-op timings.",
    });
    specs.push(OperationBucketSpec {
        bucket: "unclassified_matmul",
        graph_families: &["matmul"],
        cost_group: "unclassified_matmul",
        metadata_weight_bytes: 0,
        note: "Unclassified matmul means the ABI graph saw GGML_OP_MUL_MAT nodes whose source names did not match the current diagnostic families; this is evidence to improve structural classification, not a model-family correction.",
    });
    specs
}

fn operation_bucket_row(
    spec: OperationBucketSpec,
    breakdown: Option<&model_fit::DecodeCostBreakdown>,
    abi_probe: &AbiDecodeProbeSummary,
    selected_ms: Option<f64>,
) -> OperationBucketRow {
    let group = breakdown.and_then(|breakdown| {
        breakdown
            .groups
            .iter()
            .find(|group| group.group == spec.cost_group)
    });
    let estimated_ms = group.map(|group| f64::from(group.bandwidth_ms));
    let estimated_traffic_bytes = group.map(|group| group.traffic_bytes).unwrap_or(0);
    OperationBucketRow {
        bucket: spec.bucket,
        source: group
            .map(|group| group.source.clone())
            .unwrap_or_else(|| "graph_inventory_only".into()),
        graph_families: spec.graph_families.to_vec(),
        estimated_ms,
        estimated_traffic_bytes,
        metadata_weight_bytes: spec.metadata_weight_bytes,
        graph_node_count: spec
            .graph_families
            .iter()
            .map(|family| graph_family_node_count(abi_probe, family))
            .sum(),
        graph_src0_bytes: spec
            .graph_families
            .iter()
            .map(|family| graph_family_src0_bytes(abi_probe, family))
            .sum(),
        graph_src1_bytes: spec
            .graph_families
            .iter()
            .map(|family| graph_family_src1_bytes(abi_probe, family))
            .sum(),
        graph_output_bytes: spec
            .graph_families
            .iter()
            .map(|family| graph_family_output_bytes(abi_probe, family))
            .sum(),
        graph_src0_over_metadata: ratio_u64(
            spec.graph_families
                .iter()
                .map(|family| graph_family_src0_bytes(abi_probe, family))
                .sum(),
            spec.metadata_weight_bytes,
        ),
        estimated_share_of_selected_ms: ratio(estimated_ms, selected_ms),
        notes: vec![spec.note.into()],
    }
}

fn graph_metadata_transformer_bytes(profile: &ModelProfile) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => profile
            .tensor_matmul
            .attention
            .bytes
            .saturating_add(profile.tensor_matmul.expert_feed_forward.bytes),
        _ => profile
            .tensor_matmul
            .attention
            .bytes
            .saturating_add(profile.tensor_matmul.feed_forward.bytes),
    }
}

fn graph_metadata_transformer_nodes(profile: &ModelProfile) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => profile
            .tensor_matmul
            .attention
            .shape
            .logical_matrix_count
            .saturating_add(sparse_moe_expected_expert_matmul_id_nodes(profile)),
        _ => profile
            .tensor_matmul
            .attention
            .shape
            .logical_matrix_count
            .saturating_add(
                profile
                    .tensor_matmul
                    .feed_forward
                    .shape
                    .logical_matrix_count,
            ),
    }
}

fn graph_transformer_src0_bytes(profile: &ModelProfile, abi_probe: &AbiDecodeProbeSummary) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => {
            graph_family_src0_bytes(abi_probe, "attention_matmul")
                .saturating_add(graph_family_src0_bytes(abi_probe, "moe_matmul_id"))
        }
        _ => graph_family_src0_bytes(abi_probe, "attention_matmul")
            .saturating_add(graph_family_src0_bytes(abi_probe, "ffn_matmul")),
    }
}

fn graph_transformer_node_count(profile: &ModelProfile, abi_probe: &AbiDecodeProbeSummary) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => {
            graph_family_node_count(abi_probe, "attention_matmul")
                .saturating_add(graph_family_node_count(abi_probe, "moe_matmul_id"))
        }
        _ => graph_family_node_count(abi_probe, "attention_matmul")
            .saturating_add(graph_family_node_count(abi_probe, "ffn_matmul")),
    }
}

fn graph_operation_family_rows(abi_probe: &AbiDecodeProbeSummary) -> Vec<GraphOperationFamilyRow> {
    let mut rows = BTreeMap::<String, GraphOperationFamilyRow>::new();
    for bucket in &abi_probe.graph_inventory {
        let family = bucket.family.clone().unwrap_or_else(|| "unknown".into());
        let row = rows
            .entry(family.clone())
            .or_insert_with(|| GraphOperationFamilyRow {
                family,
                node_count: 0,
                src0_bytes: 0,
                src1_bytes: 0,
                output_bytes: 0,
                element_count: 0,
            });
        row.node_count = row
            .node_count
            .saturating_add(bucket.node_count.unwrap_or(0));
        row.src0_bytes = row
            .src0_bytes
            .saturating_add(bucket.src0_bytes.unwrap_or(0));
        row.src1_bytes = row
            .src1_bytes
            .saturating_add(bucket.src1_bytes.unwrap_or(0));
        row.output_bytes = row
            .output_bytes
            .saturating_add(bucket.output_bytes.unwrap_or(0));
        row.element_count = row
            .element_count
            .saturating_add(bucket.element_count.unwrap_or(0));
    }
    rows.into_values().collect()
}

fn operation_bucket_notes(has_breakdown: bool) -> Vec<String> {
    let mut notes = vec![
        "Operation buckets are llama.cpp/GGML graph families, not model families or filename rules."
            .into(),
        "These rows are validation diagnostics; observed benchmark throughput is not fed back into metadata-only scoring."
            .into(),
    ];
    if !has_breakdown {
        notes.push(
            "No decode cost breakdown was available, so rows report graph inventory without estimated bucket timing."
                .into(),
        );
    }
    notes
}

fn ratio_u64(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn graph_family_src1_bytes(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.src1_bytes)
        .sum()
}

fn graph_family_output_bytes(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.output_bytes)
        .sum()
}

fn abi_ms_per_token(abi_probe: &AbiDecodeProbeSummary) -> Option<f64> {
    match (abi_probe.measured_tokens, abi_probe.elapsed_ms) {
        (Some(tokens), Some(elapsed_ms)) if tokens > 0 => Some(elapsed_ms / tokens as f64),
        _ => abi_probe
            .tokens_per_second
            .filter(|tps| *tps > 0.0)
            .map(|tps| 1000.0 / tps),
    }
}

fn graph_probe_layers_from_name(name: &str) -> u32 {
    for marker in [
        "_llama_graph_l",
        "_moe_block_graph_l",
        "_moe_graph_l",
        "_linear_attn_graph_r",
    ] {
        if let Some(layers) = graph_probe_layers_after_marker(name, marker) {
            return layers;
        }
    }
    1
}

fn graph_probe_layers_after_marker(name: &str, marker: &str) -> Option<u32> {
    let (_, suffix) = name.split_once(marker)?;
    Some(
        suffix
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse::<u32>()
            .unwrap_or(1)
            .max(1),
    )
}

fn display_observed_over_abi(row: &ModelValidationReport) -> String {
    row.decode_probe_diagnostic
        .as_ref()
        .and_then(|diagnostic| diagnostic.observed_over_abi)
        .map(|ratio| format!("{ratio:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_decode_probe_classification(row: &ModelValidationReport) -> String {
    row.decode_probe_diagnostic
        .as_ref()
        .map(|diagnostic| diagnostic.classification.clone())
        .unwrap_or_else(|| "-".into())
}

fn display_fit_meaning(row: &ModelValidationReport) -> String {
    row.fit_interpretation
        .as_ref()
        .map(|fit| fit.summary.clone())
        .unwrap_or_else(|| "-".into())
}

fn display_selected_backend(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .map(|rec| match rec.selected_accelerator.as_deref() {
            Some(accelerator) => format!("{:?} ({accelerator})", rec.selected_backend),
            None => format!("{:?}", rec.selected_backend),
        })
        .unwrap_or_else(|| "-".into())
}

fn display_abi_decode_probe(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.abi_decode_probe
        .as_ref()
        .and_then(|probe| probe.tokens_per_second)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_abi_sampling_probe(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.abi_decode_probe
        .as_ref()
        .and_then(|probe| probe.sampling_tokens_per_second)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_abi_non_eval_overhead(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.abi_decode_probe
        .as_ref()
        .and_then(|probe| probe.non_eval_overhead_pct)
        .map(|pct| format!("{pct:.1}%"))
        .unwrap_or_else(|| "-".into())
}

fn scenario_verdict(row: &ModelValidationReport, scenario: &str) -> String {
    scenario_summary_by_name(row, scenario)
        .map(|benchmark| benchmark.verdict.clone())
        .unwrap_or_else(|| "-".into())
}

fn scenario_summary_by_name<'a>(
    row: &'a ModelValidationReport,
    scenario: &str,
) -> Option<&'a BenchmarkScenarioSummary> {
    row.benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == scenario)
}

fn fit_status(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .map(|rec| format!("{:?}", rec.fit_status))
        .unwrap_or_else(|| "-".into())
}

fn display_estimated_tps(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.recommendation
        .as_ref()
        .and_then(|rec| rec.estimated_decode_tokens_per_sec)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_steady_estimated_tps(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.predicted)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| display_estimated_tps(row))
}

fn display_steady_estimated_range(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.predicted_range)
        .map(|range| format!("{:.1}-{:.1}", range.0, range.1))
        .unwrap_or_else(|| display_estimated_range(row))
}

fn display_steady_observed(row: &ModelValidationReport) -> String {
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.observed)
        .or(row.benchmark.median_tokens_per_sec)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_steady_observed_over_fit(row: &ModelValidationReport) -> String {
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.observed_over_fit)
        .or(row.benchmark.observed_over_fit)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_estimated_range(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.recommendation
        .as_ref()
        .and_then(|rec| rec.estimated_decode_tokens_per_sec_range)
        .map(|range| format!("{:.1}-{:.1}", range.lower, range.upper))
        .unwrap_or_else(|| "-".into())
}

fn row_is_local_fit(row: &ModelValidationReport) -> bool {
    row.recommendation.as_ref().is_some_and(|rec| {
        matches!(
            rec.fit_status,
            FitStatus::FitsLocal | FitStatus::FitsWithWarning
        )
    })
}

fn display_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_option_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_option_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_option_ratio(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_option_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "-".into())
}

fn display_graph_node_ratio(diagnostic: &GraphInventoryDiagnostic) -> String {
    format!(
        "{}/{}",
        diagnostic.graph_transformer_matmul_nodes, diagnostic.metadata_transformer_matmul_nodes
    )
}

fn display_graph_inventory_notes(diagnostic: &GraphInventoryDiagnostic) -> String {
    diagnostic
        .notes
        .first()
        .map(|note| note.replace('|', "/"))
        .unwrap_or_else(|| "-".into())
}

fn display_operation_bucket_notes(bucket: &OperationBucketRow) -> String {
    bucket
        .notes
        .first()
        .map(|note| note.replace('|', "/"))
        .unwrap_or_else(|| "-".into())
}

fn heartbeat(model_index: Option<usize>, model_ref: &str, phase: &str, detail: &str) {
    let index = model_index
        .map(|index| index.to_string())
        .unwrap_or_else(|| "-".into());
    let detail = detail.replace(['\r', '\n'], " ");
    eprintln!(
        "[model-fit-validate] model_index={index} phase={phase} model_ref={:?} {detail}",
        model_ref
    );
}

fn profile_summary(profile: &ModelProfile) -> String {
    format!(
        "architecture={} layers={} hidden={} ctx={} quant={} params={} file_bytes={}",
        profile.architecture.as_deref().unwrap_or("-"),
        display_u32(profile.layer_count),
        display_u32(profile.hidden_size),
        display_u32(profile.context_length),
        profile.quantization.as_deref().unwrap_or("-"),
        display_u64(profile.parameter_count),
        profile.file_size_bytes
    )
}

fn display_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn abi_probe_observation_detail(observation: &AbiDecodeProbeObservation) -> String {
    format!(
        "repeat={} status={} tok_s={} sample_sync_tok_s={} elapsed_ms={} decode_call_ms={} sampling_ms={} overhead_ms={} error={}",
        observation.repeat + 1,
        status_label(observation.status_code),
        display_opt(observation.tokens_per_second),
        display_opt(observation.sampling_tokens_per_second),
        display_opt(observation.elapsed_ms),
        display_opt(observation.decode_call_ms),
        display_opt(observation.sampling_ms),
        display_opt(observation.non_eval_overhead_ms),
        observation.error.as_deref().unwrap_or("-")
    )
}

fn benchmark_observation_detail(
    observation: &BenchmarkObservation,
    scenario: &BenchmarkScenarioSpec,
) -> String {
    format!(
        "scenario={} repeat={} status={} wall_s={:.2} metric={} error={}",
        scenario.name,
        observation.repeat + 1,
        status_label(observation.status_code),
        observation.wall_seconds,
        display_opt(benchmark_observation_metric(observation, scenario)),
        observation.error.as_deref().unwrap_or("-")
    )
}

fn benchmark_observation_metric(
    observation: &BenchmarkObservation,
    scenario: &BenchmarkScenarioSpec,
) -> Option<f64> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode => {
            steady_decode_observation_tokens_per_sec(observation)
        }
        BenchmarkScenarioKind::Prefill => prefill_observation_tokens_per_sec(observation),
        BenchmarkScenarioKind::FirstToken => observation.text_request_elapsed_ms,
        BenchmarkScenarioKind::KvWarmReuse => observation
            .request_results
            .last()
            .and_then(|request| request.generated_tokens_per_sec),
    }
}

fn scenario_summary_detail(summary: &BenchmarkScenarioSummary) -> String {
    format!(
        "scenario={} verdict={} observed={} observed_over_fit={}",
        summary.scenario,
        summary.verdict,
        display_opt(summary.observed),
        display_opt(summary.observed_over_fit)
    )
}

fn status_label(status_code: Option<i32>) -> String {
    status_code
        .map(|code| code.to_string())
        .unwrap_or_else(|| "-".into())
}

fn parse_local_model(value: &str) -> Result<LocalModelInput> {
    let mut fields = BTreeMap::new();
    for pair in value.split(',') {
        let Some((key, value)) = pair.split_once('=') else {
            bail!("model field must be key=value: {pair}");
        };
        fields.insert(key.trim(), value.trim());
    }
    Ok(LocalModelInput {
        model_ref: required_field(&fields, "ref")?.to_string(),
        gguf_path: PathBuf::from(required_field(&fields, "path")?),
    })
}

fn required_field<'a>(fields: &'a BTreeMap<&str, &str>, key: &str) -> Result<&'a str> {
    fields
        .get(key)
        .copied()
        .with_context(|| format!("missing model field {key}"))
}

fn input_label(input: &ModelInput) -> String {
    match input {
        ModelInput::Ref(model_ref) => model_ref.clone(),
        ModelInput::Local(local) => local.model_ref.clone(),
    }
}

fn read_json_input(path: &Path) -> Result<Vec<u8>> {
    if path == Path::new("-") {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::io::stdin()
            .read_to_end(&mut bytes)
            .context("read JSON from stdin")?;
        return Ok(bytes);
    }
    fs::read(path).with_context(|| format!("read {}", path.display()))
}

fn write_json_report(path: &Path, report: &ValidationReport) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn command_display(bin: &Path, args: &[String]) -> Vec<String> {
    std::iter::once(bin.display().to_string())
        .chain(args.iter().cloned())
        .collect()
}

#[derive(Clone)]
struct TerminalDownloadProgress {
    model_ref: Arc<String>,
    state: Arc<Mutex<TerminalDownloadProgressState>>,
}

#[derive(Default)]
struct TerminalDownloadProgressState {
    last_draw: Option<Instant>,
    active_line: bool,
}

impl TerminalDownloadProgress {
    fn new(model_ref: &str) -> Self {
        Self {
            model_ref: Arc::new(model_ref.to_string()),
            state: Arc::new(Mutex::new(TerminalDownloadProgressState::default())),
        }
    }

    fn report(&self, event: ModelDownloadProgressEvent) {
        match event {
            ModelDownloadProgressEvent::Ensuring {
                file,
                index,
                total_files,
                total_bytes,
            } => self.draw(
                format!(
                    "Ensuring {} [{}/{}] {}{}",
                    self.model_ref,
                    index,
                    total_files,
                    file,
                    total_bytes
                        .map(|bytes| format!(" ({})", format_bytes(bytes)))
                        .unwrap_or_default()
                ),
                true,
                false,
            ),
            ModelDownloadProgressEvent::Started {
                file, total_bytes, ..
            } => self.draw(
                format!(
                    "Downloading {} {}{}",
                    self.model_ref,
                    file,
                    total_bytes
                        .map(|bytes| format!(" ({})", format_bytes(bytes)))
                        .unwrap_or_default()
                ),
                true,
                false,
            ),
            ModelDownloadProgressEvent::Progress {
                file,
                downloaded_bytes,
                total_bytes,
                bytes_per_sec,
            } => self.draw(
                download_progress_line(
                    &self.model_ref,
                    &file,
                    downloaded_bytes,
                    total_bytes,
                    bytes_per_sec,
                ),
                false,
                false,
            ),
            ModelDownloadProgressEvent::Ready {
                file,
                index,
                total_files,
                size_bytes,
                ..
            } => self.draw(
                format!(
                    "Ready {} [{}/{}] {}{}",
                    self.model_ref,
                    index,
                    total_files,
                    file,
                    size_bytes
                        .map(|bytes| format!(" ({})", format_bytes(bytes)))
                        .unwrap_or_default()
                ),
                true,
                true,
            ),
            ModelDownloadProgressEvent::Complete { .. } => {}
        }
    }

    fn draw(&self, message: String, force: bool, finish_line: bool) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let now = Instant::now();
        if !force
            && state
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(150))
        {
            return;
        }
        state.last_draw = Some(now);
        state.active_line = !finish_line;
        eprint!("\r\x1b[2K{message}");
        if finish_line {
            eprintln!();
        }
        let _ = std::io::stderr().flush();
    }
}

impl Drop for TerminalDownloadProgress {
    fn drop(&mut self) {
        if Arc::strong_count(&self.state) != 1 {
            return;
        }
        match self.state.lock() {
            Ok(state) if state.active_line => eprintln!(),
            _ => {}
        }
    }
}

struct TerminalStatus {
    done: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl TerminalStatus {
    fn start(enabled: bool, message: String) -> Self {
        if !enabled {
            return Self {
                done: Arc::new(AtomicBool::new(true)),
                thread: None,
            };
        }
        let done = Arc::new(AtomicBool::new(false));
        let done_thread = Arc::clone(&done);
        let thread = thread::spawn(move || {
            let frames = ["|", "/", "-", "\\"];
            let mut index = 0usize;
            while !done_thread.load(AtomicOrdering::Relaxed) {
                eprint!("\r\x1b[2K{} {}", frames[index % frames.len()], message);
                let _ = std::io::stderr().flush();
                index += 1;
                thread::sleep(Duration::from_millis(120));
            }
        });
        Self {
            done,
            thread: Some(thread),
        }
    }

    fn finish(&mut self) {
        self.done.store(true, AtomicOrdering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}

impl Drop for TerminalStatus {
    fn drop(&mut self) {
        self.finish();
    }
}

fn download_progress_line(
    model_ref: &str,
    file: &str,
    downloaded: u64,
    total: Option<u64>,
    bytes_per_sec: Option<f64>,
) -> String {
    let speed = bytes_per_sec
        .filter(|speed| *speed > 0.0)
        .map(|speed| format!(" at {}/s", format_bytes(speed as u64)))
        .unwrap_or_default();
    if let Some(total) = total.filter(|total| *total > 0) {
        let percent = (downloaded.min(total) as f64 / total as f64) * 100.0;
        format!(
            "Downloading {model_ref} {file} {:>5.1}% ({}/{}){speed}",
            percent,
            format_bytes(downloaded),
            format_bytes(total)
        )
    } else {
        format!(
            "Downloading {model_ref} {file} {}{speed}",
            format_bytes(downloaded)
        )
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for candidate in &UNITS[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = candidate;
    }
    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn median(samples: &[f64]) -> f64 {
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

fn mean(samples: &[f64]) -> Option<f64> {
    (!samples.is_empty()).then(|| samples.iter().sum::<f64>() / samples.len() as f64)
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    next_value(args, name)?
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn print_usage() {
    eprintln!(
        "usage: model-fit-validate [--output-json report.json] [--models-file refs.txt] [--scenario steady_decode|prefill|first_token|kv_warm_reuse|all] [--dense-probe-depth standard|deep] [--benchmark-all] [--fit-only] [--no-progress] org/repo:Q4_K_M [org/repo:Q5_K_M ...]"
    );
}
