use anyhow::{Context, Result, bail};
use mesh_llm_system::hardware::HardwareSurvey;
use model_artifact::{ModelFormat, ResolvedModelArtifact, resolve_model_artifact_ref};
use model_fit::{
    AcceleratorKind, BackendKind, CpuProfile, FitStatus, GpuBenchmarkAcceleratorFacts,
    GpuBenchmarkHardwareInput, GpuBenchmarkOutput, HardwareProfile, MemoryProfile, ModelProfile,
    ModelRecommendation, SelectionConfig, WorkloadProfile, hardware_profile_from_gpu_benchmark,
    profile_gguf_path, score_model, throughput_sample_stats,
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
const DEFAULT_TOLERANCE: f64 = 0.10;
const DEFAULT_MAX_SPREAD: f64 = 0.10;
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
    base_port: u16,
    benchmark_all: bool,
    show_progress: bool,
    models: Vec<ModelInput>,
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
    hardware_profile: HardwareProfile,
    gpu_benchmark_outputs: Vec<GpuBenchmarkOutput>,
    gpu_benchmark_json: Value,
    selection_config: SelectionConfig,
    validation_config: ValidationConfig,
    models: Vec<ModelValidationReport>,
    summary: ValidationSummary,
}

#[derive(Debug, Serialize)]
struct ValidationConfig {
    ctx_size: u32,
    warmup_tokens: usize,
    max_new_tokens: usize,
    repeats: usize,
    tolerance: f64,
    max_spread: f64,
    benchmark_all: bool,
    show_progress: bool,
    prompt: String,
    primary_workload: String,
    scored_workloads: Vec<String>,
    benchmark_scenarios: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ModelValidationReport {
    input_ref: String,
    resolved_ref: Option<String>,
    artifact: Option<ResolvedModelArtifact>,
    downloaded_paths: Vec<PathBuf>,
    primary_gguf_path: Option<PathBuf>,
    model_profile: Option<ModelProfile>,
    recommendation: Option<ModelRecommendation>,
    recommendations: Vec<WorkloadRecommendation>,
    benchmarks: Vec<BenchmarkScenarioSummary>,
    benchmark: BenchmarkSummary,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct WorkloadRecommendation {
    workload: String,
    recommendation: ModelRecommendation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkScenarioKind {
    SteadyDecode,
    FirstToken,
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

#[derive(Clone, Debug, Serialize)]
struct BenchmarkScenarioSummary {
    scenario: String,
    fit_metric: String,
    predicted: Option<f64>,
    observed: Option<f64>,
    observed_over_fit: Option<f64>,
    verdict: String,
    benchmark: BenchmarkSummary,
}

#[derive(Clone, Debug, Default, Serialize)]
struct BenchmarkSummary {
    attempted: bool,
    skip_reason: Option<String>,
    observations: Vec<BenchmarkObservation>,
    successful_repeats: usize,
    sample_count: usize,
    raw_sample_count: usize,
    median_tokens_per_sec: Option<f64>,
    min_tokens_per_sec: Option<f64>,
    max_tokens_per_sec: Option<f64>,
    spread_pct: Option<f64>,
    raw_median_tokens_per_sec: Option<f64>,
    raw_min_tokens_per_sec: Option<f64>,
    raw_max_tokens_per_sec: Option<f64>,
    raw_spread_pct: Option<f64>,
    denoised_outlier_count: usize,
    observed_over_fit: Option<f64>,
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
    noisy_count: usize,
    skipped_count: usize,
    error_count: usize,
    runtime_error_count: usize,
    median_observed_over_fit: Option<f64>,
    mean_observed_over_fit: Option<f64>,
    median_absolute_percent_error: Option<f64>,
    within_tolerance_count: usize,
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
    }

    let summary = summarize(&models, DEFAULT_TOLERANCE);
    let report = ValidationReport {
        schema_version: 1,
        generated_at_unix_secs: unix_timestamp_secs(),
        command: std::env::args().collect(),
        hardware_profile: hardware.profile,
        gpu_benchmark_outputs: hardware.benchmark_outputs,
        gpu_benchmark_json: hardware.raw_json,
        selection_config,
        validation_config: ValidationConfig::from_args(&args),
        models,
        summary,
    };

    write_json_report(&args.output_json, &report)?;
    print_markdown_table(&report.models);
    eprintln!("wrote {}", args.output_json.display());
    Ok(())
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::env::args().skip(1);
        let mut parsed = Self {
            output_json: PathBuf::from("/tmp/model-fit-validation.json"),
            skippy_bench_bin: default_binary_path("skippy-bench"),
            skippy_server_bin: default_binary_path("skippy-server"),
            metrics_server_bin: default_binary_path("metrics-server"),
            gpu_benchmark_json: None,
            model_files: Vec::new(),
            base_port: 18400,
            benchmark_all: false,
            show_progress: true,
            models: Vec::new(),
        };

        while let Some(arg) = values.next() {
            parsed.parse_arg(arg, &mut values)?;
        }
        parsed.load_model_files()?;

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
            "--base-port" => self.base_port = parse_next(values, "--base-port")?,
            "--benchmark-all" => self.benchmark_all = true,
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
        Self {
            ctx_size: DEFAULT_CTX_SIZE,
            warmup_tokens: DEFAULT_WARMUP_TOKENS,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            repeats: DEFAULT_REPEATS,
            tolerance: DEFAULT_TOLERANCE,
            max_spread: DEFAULT_MAX_SPREAD,
            benchmark_all: args.benchmark_all,
            show_progress: args.show_progress,
            prompt: validation_prompt().into(),
            primary_workload: primary_workload_label().into(),
            scored_workloads: workload_profiles()
                .iter()
                .map(|(label, _)| (*label).to_string())
                .collect(),
            benchmark_scenarios: benchmark_scenarios()
                .iter()
                .map(|scenario| scenario.name.to_string())
                .collect(),
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
    match prepare_model(args, repository, input).await {
        Ok(prepared) => validate_prepared_model(args, hardware, prepared, model_index),
        Err(err) => error_report(input_label(input), format!("{err:#}")),
    }
}

fn validate_prepared_model(
    args: &Args,
    hardware: &HardwareProfile,
    prepared: PreparedModel,
    model_index: usize,
) -> ModelValidationReport {
    let recommendations = score_workloads(hardware, &prepared.profile);
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
    let benchmarks = benchmark_model(args, hardware, &prepared, &recommendation, model_index);
    let benchmark = benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == "steady_decode")
        .map(|benchmark| benchmark.benchmark.clone())
        .unwrap_or_else(|| BenchmarkSummary {
            verdict: "skipped".into(),
            skip_reason: Some("steady_decode scenario was not produced".into()),
            ..BenchmarkSummary::default()
        });
    ModelValidationReport {
        input_ref: prepared.input_ref,
        resolved_ref: prepared.resolved_ref,
        artifact: prepared.artifact,
        downloaded_paths: prepared.downloaded_paths,
        primary_gguf_path: Some(prepared.primary_gguf_path),
        model_profile: Some(prepared.profile),
        recommendation: Some(recommendation),
        recommendations,
        benchmarks,
        benchmark,
        errors: Vec::new(),
    }
}

async fn prepare_model(
    args: &Args,
    repository: &HfModelRepository,
    input: &ModelInput,
) -> Result<PreparedModel> {
    match input {
        ModelInput::Ref(model_ref) => prepare_model_ref(args, repository, model_ref).await,
        ModelInput::Local(local) => prepare_local_model(args, local),
    }
}

async fn prepare_model_ref(
    args: &Args,
    repository: &HfModelRepository,
    model_ref: &str,
) -> Result<PreparedModel> {
    let artifact = {
        let _status = TerminalStatus::start(args.show_progress, format!("Resolving {model_ref}"));
        resolve_model_artifact_ref(model_ref, repository)
            .await
            .with_context(|| format!("resolve model ref {model_ref}"))?
    };
    if artifact.format != ModelFormat::Gguf {
        bail!(
            "{model_ref} resolved to {:?}, expected GGUF",
            artifact.format
        );
    }
    let progress = download_progress(args, model_ref);
    let downloaded_paths = repository
        .download_artifact_files_with_progress(&artifact, progress)
        .await
        .with_context(|| format!("download model ref {model_ref}"))?;
    let primary_gguf_path = primary_download_path(&artifact, &downloaded_paths)?;
    let mut profile = {
        let _status = TerminalStatus::start(args.show_progress, format!("Profiling {model_ref}"));
        profile_gguf_path(&primary_gguf_path)?
    };
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

fn prepare_local_model(args: &Args, local: &LocalModelInput) -> Result<PreparedModel> {
    let mut profile = {
        let _status =
            TerminalStatus::start(args.show_progress, format!("Profiling {}", local.model_ref));
        profile_gguf_path(&local.gguf_path)?
    };
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

fn benchmark_model(
    args: &Args,
    hardware: &HardwareProfile,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
) -> Vec<BenchmarkScenarioSummary> {
    benchmark_scenarios()
        .into_iter()
        .enumerate()
        .map(|(scenario_index, scenario)| {
            benchmark_scenario(
                args,
                hardware,
                model,
                recommendation,
                model_index,
                scenario_index,
                scenario,
            )
        })
        .collect()
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
    let mut summary = BenchmarkSummary {
        verdict: "skipped".into(),
        ..BenchmarkSummary::default()
    };
    if let Some(reason) = benchmark_skip_reason(args, model, recommendation, scenario.kind) {
        summary.skip_reason = Some(reason);
        return scenario_summary(scenario, recommendation, summary);
    }

    summary.attempted = true;
    for repeat in 0..DEFAULT_REPEATS {
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Benchmarking {} {} repeat {}/{}",
                model.input_ref,
                scenario.name,
                repeat + 1,
                DEFAULT_REPEATS
            ),
        );
        let observation =
            run_one_benchmark(args, model, model_index, scenario_index, repeat, &scenario);
        if let Some(error) = observation.error.as_ref() {
            summary.errors.push(error.clone());
        }
        summary.observations.push(observation);
    }
    let prediction = scenario_prediction(&scenario, recommendation);
    let range = scenario_prediction_range(&scenario, recommendation);
    let summary = finalize_benchmark_summary(summary, &scenario, prediction, range);
    let scenario_recommendation = benchmark_scenario_recommendation(
        hardware,
        &model.profile,
        recommendation,
        &scenario,
        &summary,
    );
    scenario_summary(scenario, &scenario_recommendation, summary)
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
    if scenario == BenchmarkScenarioKind::SteadyDecode
        && recommendation.estimated_decode_tokens_per_sec.is_none()
    {
        return Some("fit algorithm did not produce a decode tokens/sec estimate".into());
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
    predicted: Option<f64>,
    predicted_range: Option<(f64, f64)>,
) -> BenchmarkSummary {
    let samples = throughput_samples(&summary, scenario);
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
    let observed_over_fit = if predicted.is_some_and(|fit| fit > 0.0) {
        predicted.map(|fit| median / fit)
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
    summary.verdict = benchmark_verdict(median, observed_over_fit, spread, predicted_range);
    summary
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

fn throughput_samples(summary: &BenchmarkSummary, scenario: &BenchmarkScenarioSpec) -> Vec<f64> {
    // Scenario sampling intentionally differs by workload shape.
    //
    // Steady decode should represent sustained token generation, so each repeat
    // contributes one aggregate decode-throughput sample. Aggregating generated
    // tokens over aggregate decode time denoises tiny models, where a single
    // request can swing just from scheduler/runtime jitter. This is not
    // "cheating" against the fit estimate: the prediction is unchanged and
    // metadata-only; we are only making the observation less noisy.
    //
    // KV warm reuse cares about the reused final request, so it samples the last
    // request. First-token falls back to the aggregate run metric because the
    // scenario is explicitly about the whole tokenize/prefill/first-decode path.
    let request_samples = match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode => summary
            .observations
            .iter()
            .filter_map(steady_decode_observation_tokens_per_sec)
            .collect::<Vec<_>>(),
        BenchmarkScenarioKind::KvWarmReuse => summary
            .observations
            .iter()
            .filter_map(|observation| observation.request_results.last())
            .filter_map(|request| request.generated_tokens_per_sec)
            .collect::<Vec<_>>(),
        BenchmarkScenarioKind::FirstToken => Vec::new(),
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

fn benchmark_scenario_recommendation(
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    fallback: &ModelRecommendation,
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> ModelRecommendation {
    if scenario.kind != BenchmarkScenarioKind::FirstToken {
        return fallback.clone();
    }
    let Some(prompt_tokens) = median_prompt_token_count(benchmark) else {
        return fallback.clone();
    };
    let mut workload = primary_workload_profile();
    workload.interaction.expected_prompt_tokens = Some(prompt_tokens);
    score_model(hardware, profile, &selection_config(&workload))
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
    benchmark.verdict = verdict.clone();
    BenchmarkScenarioSummary {
        scenario: scenario.name.into(),
        fit_metric: scenario.fit_metric.into(),
        predicted,
        observed,
        observed_over_fit,
        verdict,
        benchmark,
    }
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
    args.base_port
        .checked_add(
            (model_index * benchmark_scenarios().len() * DEFAULT_REPEATS
                + scenario_index * DEFAULT_REPEATS
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
    let survey = mesh_llm_system::hardware::survey();
    let (benchmark_outputs, facts, raw_json) = if let Some(path) = args.gpu_benchmark_json.as_ref()
    {
        let bytes = read_json_input(path)?;
        let raw_json: Value = serde_json::from_slice(&bytes).context("parse GPU benchmark JSON")?;
        let (outputs, facts) = parse_gpu_benchmark_json(&raw_json, &survey)?;
        (outputs, facts, raw_json)
    } else {
        let outputs = run_local_gpu_benchmark(args, &survey)?;
        let facts = default_facts(&survey, outputs.len());
        let raw_json = json!({
            "source": "model-fit-validate:auto_gpu_benchmark",
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
    if let Ok(outputs) = serde_json::from_value::<Vec<GpuBenchmarkOutput>>(raw_json.clone())
        && !outputs.is_empty()
    {
        return Ok((outputs.clone(), default_facts(survey, outputs.len())));
    }
    if let Some(raw_outputs) = raw_json.get("outputs")
        && let Ok(outputs) = serde_json::from_value::<Vec<GpuBenchmarkOutput>>(raw_outputs.clone())
        && !outputs.is_empty()
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
        compute_tflops_fp32: gpu.get("compute_tflops_fp32").and_then(Value::as_f64),
        compute_tflops_fp16: gpu.get("compute_tflops_fp16").and_then(Value::as_f64),
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
    (0..count)
        .map(|index| default_fact(survey, index))
        .collect()
}

fn default_fact(survey: &HardwareSurvey, index: usize) -> GpuBenchmarkAcceleratorFacts {
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
    }
}

fn run_local_gpu_benchmark(
    args: &Args,
    survey: &HardwareSurvey,
) -> Result<Vec<GpuBenchmarkOutput>> {
    let _status = TerminalStatus::start(
        args.show_progress,
        "Benchmarking local GPU memory bandwidth".into(),
    );
    let runner = benchmark_runner_for_survey(survey)?;
    mesh_llm_gpu_bench::run_benchmark(runner, Duration::from_secs(120))
        .context("run local GPU benchmark")
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
        return fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| meminfo_value_bytes(&text, "MemAvailable:"));
    }
    None
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

fn summarize(models: &[ModelValidationReport], tolerance: f64) -> ValidationSummary {
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
    summary
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
        recommendations: Vec::new(),
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
        "| model_ref | fit | est tok/s | est range | steady median | steady/fit | steady | first-token | kv-reuse |"
    );
    println!("|---|---|---:|---:|---:|---:|---|---|---|");
    for row in rows {
        println!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.input_ref,
            fit_status(row),
            display_estimated_tps(row),
            display_estimated_range(row),
            display_opt(row.benchmark.median_tokens_per_sec),
            display_opt(row.benchmark.observed_over_fit),
            row.benchmark.verdict,
            scenario_verdict(row, "first_token"),
            scenario_verdict(row, "kv_warm_reuse"),
        );
    }
}

fn scenario_verdict(row: &ModelValidationReport, scenario: &str) -> String {
    row.benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == scenario)
        .map(|benchmark| benchmark.verdict.clone())
        .unwrap_or_else(|| "-".into())
}

fn fit_status(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .map(|rec| format!("{:?}", rec.fit_status))
        .unwrap_or_else(|| "-".into())
}

fn display_estimated_tps(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .and_then(|rec| rec.estimated_decode_tokens_per_sec)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_estimated_range(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .and_then(|rec| rec.estimated_decode_tokens_per_sec_range)
        .map(|range| format!("{:.1}-{:.1}", range.lower, range.upper))
        .unwrap_or_else(|| "-".into())
}

fn display_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
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
        if let Ok(state) = self.state.lock()
            && state.active_line
        {
            eprintln!();
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
        "usage: model-fit-validate [--output-json report.json] [--models-file refs.txt] [--benchmark-all] [--no-progress] org/repo:Q4_K_M [org/repo:Q5_K_M ...]"
    );
}
