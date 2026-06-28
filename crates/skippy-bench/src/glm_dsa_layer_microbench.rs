use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::package::{PackagePart, PackageStageRequest, select_layer_package_parts};
use skippy_runtime::{
    ActivationDesc, ActivationFrame, FlashAttentionType, RuntimeActivationDType,
    RuntimeActivationLayout, RuntimeConfig, RuntimeLoadMode, StageModel, parse_cache_type,
    redirect_native_logs_to_file, restore_native_logs,
};

use crate::{
    cli::GlmDsaLayerMicrobenchArgs,
    glm_dsa_op_report::{
        DirectSparseDecisionRecord, MetalDispatchRecord, TimingGroupRecord, TimingRecord,
        parse_direct_sparse_decision_records, parse_metal_dispatch_records,
        parse_timing_group_records, parse_timing_records,
    },
};

const ACTIVATION_FLAG_GLM_DSA_TOP_K: u64 = 1 << 3;
const ENV_SYNTHETIC_TOP_K_SIDEBAND: &str = "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_SIDEBAND";
const ENV_SYNTHETIC_TOP_K_WIDTH: &str = "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_WIDTH";
const DEFAULT_SYNTHETIC_TOP_K_WIDTH: usize = 256;

pub fn glm_dsa_layer_microbench(args: GlmDsaLayerMicrobenchArgs) -> Result<()> {
    validate_args(&args)?;

    let selected = select_layer_package_parts(&package_request(&args))
        .context("select GLM-DSA layer package parts")?;
    let runtime_config = runtime_config(&args)?;
    let input = synthetic_activation_frame(&args)?;
    let token_ids = vec![1_i32; args.tokens];
    let positions = positions(args.tokens)?;
    let flags = MicrobenchFlags::from_args(&args);
    let comparison = if args.compare_dense_fallback {
        Some(run_dense_fallback_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            &input,
            &token_ids,
            &positions,
            flags,
        )?)
    } else if args.compare_cpu_direct_sparse {
        Some(run_cpu_direct_sparse_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            &input,
            &token_ids,
            &positions,
            flags,
        )?)
    } else {
        None
    };
    let case = match comparison.as_ref() {
        Some(comparison) => comparison.candidate.clone(),
        None => {
            let case = run_microbench_case(
                "candidate",
                &selected.absolute_paths,
                &runtime_config,
                &args,
                flags,
                &input,
                &token_ids,
                &positions,
                false,
            )?;
            case.as_case_summary()
        }
    };

    let direct_sparse_decision_summary =
        summarize_direct_sparse_decisions(&case.direct_sparse_decision_records);
    let report = MicrobenchReport {
        command: "glm-dsa-layer-microbench",
        model_id: args.model_id,
        stage_model: args.stage_model,
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        activation_width: args.activation_width,
        tokens: args.tokens,
        warmup: args.warmup,
        iterations: args.iterations,
        n_gpu_layers: args.n_gpu_layers,
        n_batch: runtime_config.n_batch,
        n_ubatch: runtime_config.n_ubatch,
        flags: case.flags,
        selected_parts: selected
            .selected_parts
            .iter()
            .map(package_part_summary)
            .collect(),
        input_payload_bytes: input.payload.len(),
        native_log_path: case.native_log_path,
        direct_sparse_decision_summary,
        direct_sparse_decision_records: case.direct_sparse_decision_records,
        metal_dispatch_records: case.metal_dispatch_records,
        op_timing_records: case.op_timing_records,
        group_timing_records: case.group_timing_records,
        comparison,
        timings: case.timings,
    };
    let parity_passed = report
        .comparison
        .as_ref()
        .is_none_or(|comparison| comparison.parity.passed);

    write_report(args.output.as_deref(), &report)?;
    if !parity_passed {
        bail!("GLM-DSA layer microbench parity comparison failed");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_dense_fallback_comparison(
    args: &GlmDsaLayerMicrobenchArgs,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    candidate_flags: MicrobenchFlags,
) -> Result<MicrobenchComparisonReport> {
    let baseline_flags = MicrobenchFlags {
        direct_sparse_attn: false,
        direct_sparse_prefill: false,
        ..candidate_flags
    };
    let baseline = run_microbench_case(
        "dense_fallback",
        selected_paths,
        runtime_config,
        args,
        baseline_flags,
        input,
        token_ids,
        positions,
        true,
    )?;
    let candidate = run_microbench_case(
        "candidate",
        selected_paths,
        runtime_config,
        args,
        candidate_flags,
        input,
        token_ids,
        positions,
        true,
    )?;
    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    Ok(MicrobenchComparisonReport {
        baseline: baseline.as_case_summary(),
        candidate: candidate.as_case_summary(),
        parity,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_cpu_direct_sparse_comparison(
    args: &GlmDsaLayerMicrobenchArgs,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    candidate_flags: MicrobenchFlags,
) -> Result<MicrobenchComparisonReport> {
    let mut baseline_config = runtime_config.clone();
    baseline_config.n_gpu_layers = 0;
    let baseline_flags = MicrobenchFlags {
        direct_sparse_attn: true,
        direct_sparse_prefill: true,
        ..candidate_flags
    };
    let baseline = run_microbench_case(
        "cpu_direct_sparse",
        selected_paths,
        &baseline_config,
        args,
        baseline_flags,
        input,
        token_ids,
        positions,
        true,
    )?;
    let candidate = run_microbench_case(
        "candidate",
        selected_paths,
        runtime_config,
        args,
        candidate_flags,
        input,
        token_ids,
        positions,
        true,
    )?;
    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    Ok(MicrobenchComparisonReport {
        baseline: baseline.as_case_summary(),
        candidate: candidate.as_case_summary(),
        parity,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_microbench_case(
    label: &'static str,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    args: &GlmDsaLayerMicrobenchArgs,
    flags: MicrobenchFlags,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    collect_outputs: bool,
) -> Result<MicrobenchCase> {
    configure_env_flags(flags);
    let native_logs = NativeLogCapture::start(flags.op_timing)?;
    let model = StageModel::open_from_parts(selected_paths, runtime_config)
        .with_context(|| format!("open GLM-DSA layer microbench model for {label}"))?;
    let mut timings = Vec::with_capacity(args.iterations);
    let mut outputs = Vec::with_capacity(if collect_outputs { args.iterations } else { 0 });
    let total_runs = args.warmup + args.iterations;
    for run_index in 0..total_runs {
        let mut session = model.create_session().context("create stage session")?;
        let started = Instant::now();
        let output = session
            .prefill_chunk_frame_with_positions(token_ids, positions, Some(input), 0)
            .with_context(|| format!("run microbench iteration {run_index}"))?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        if run_index >= args.warmup {
            timings.push(IterationTiming {
                iteration: run_index - args.warmup,
                elapsed_ms,
                output_payload_bytes: output.payload.len(),
                output_flags: output.desc.flags,
            });
            if collect_outputs {
                outputs.push(output);
            }
        }
    }
    let native_timings = native_logs.finish()?;
    Ok(MicrobenchCase {
        label,
        flags,
        n_gpu_layers: runtime_config.n_gpu_layers,
        native_log_path: native_timings.log_path,
        direct_sparse_decision_records: retain_case_decision_records(
            native_timings.direct_sparse_decision_records,
            args.tokens,
        ),
        metal_dispatch_records: native_timings.metal_dispatch_records,
        op_timing_records: skip_warmup_records(native_timings.op_timing_records, args.warmup),
        group_timing_records: skip_warmup_group_records(
            native_timings.group_timing_records,
            args.warmup,
        ),
        timings,
        outputs,
    })
}

fn validate_args(args: &GlmDsaLayerMicrobenchArgs) -> Result<()> {
    if args.layer_start >= args.layer_end {
        bail!("layer_start must be less than layer_end");
    }
    if args.layer_start == 0 {
        bail!(
            "glm-dsa-layer-microbench expects a nonzero layer_start and synthetic activation input"
        );
    }
    if args.tokens == 0 {
        bail!("tokens must be greater than zero");
    }
    if args.iterations == 0 {
        bail!("iterations must be greater than zero");
    }
    if args.activation_width == 0 {
        bail!("activation_width must be greater than zero");
    }
    if args.compare_dense_fallback && args.compare_cpu_direct_sparse {
        bail!("compare_dense_fallback and compare_cpu_direct_sparse are mutually exclusive");
    }
    Ok(())
}

fn configure_env_flags(flags: MicrobenchFlags) {
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_ATTN",
        flags.direct_sparse_attn,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_PREFILL",
        flags.direct_sparse_prefill,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_FUSED_SPARSE_MASK",
        flags.fused_sparse_mask,
    );
    set_env_flag(
        "LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER",
        flags.parallel_lightning_indexer,
    );
    set_env_flag("SKIPPY_GLM_DSA_OP_TIMING", flags.op_timing);
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_DIRECT_SPARSE_DECISIONS",
        flags.op_timing,
    );
    set_env_flag("SKIPPY_GLM_DSA_LOG_METAL_DISPATCH", flags.op_timing);
}

fn set_env_flag(name: &str, enabled: bool) {
    // This command is single-threaded and sets native runtime flags before opening llama.cpp.
    unsafe {
        std::env::set_var(name, if enabled { "1" } else { "0" });
    }
}

fn package_request(args: &GlmDsaLayerMicrobenchArgs) -> PackageStageRequest {
    PackageStageRequest {
        model_id: args.model_id.clone(),
        topology_id: "glm-dsa-layer-microbench".to_string(),
        package_ref: args.stage_model.to_string_lossy().to_string(),
        stage_id: format!("layers-{}-{}", args.layer_start, args.layer_end),
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        include_embeddings: false,
        include_output: false,
    }
}

fn runtime_config(args: &GlmDsaLayerMicrobenchArgs) -> Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: Some(args.n_batch.unwrap_or_else(|| bounded_u32(args.tokens))),
        n_ubatch: Some(args.n_ubatch.unwrap_or_else(|| bounded_u32(args.tokens))),
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        cache_type_k: parse_cache_type(&args.cache_type_k).context("parse cache_type_k")?,
        cache_type_v: parse_cache_type(&args.cache_type_v).context("parse cache_type_v")?,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::LayerPackage,
        projector_path: None,
        use_mmap: false,
        use_mmap_prefetch: false,
        use_mmap_buffer: false,
        include_embeddings: false,
        include_output: false,
        filter_tensors_on_load: true,
    })
}

fn bounded_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX).max(1)
}

fn synthetic_activation_frame(args: &GlmDsaLayerMicrobenchArgs) -> Result<ActivationFrame> {
    let width = usize::try_from(args.activation_width).context("activation_width exceeds usize")?;
    let top_k_sideband = synthetic_top_k_sideband_config()?;
    let value_count = args
        .tokens
        .checked_mul(width)
        .context("synthetic activation value count overflow")?;
    let payload_bytes = value_count
        .checked_mul(std::mem::size_of::<f32>())
        .context("synthetic activation payload size overflow")?;
    let sideband_bytes = top_k_sideband
        .as_ref()
        .map(|sideband| synthetic_top_k_sideband_bytes(args.tokens, sideband.width))
        .transpose()?
        .unwrap_or(0);
    let mut payload = Vec::with_capacity(payload_bytes);
    for token in 0..args.tokens {
        for dim in 0..width {
            let value = synthetic_activation_value(token, dim);
            payload.extend_from_slice(&value.to_ne_bytes());
        }
    }
    if let Some(sideband) = top_k_sideband {
        append_synthetic_top_k_sideband(&mut payload, args.tokens, sideband.width)?;
    }
    let flags = if sideband_bytes > 0 {
        ACTIVATION_FLAG_GLM_DSA_TOP_K
    } else {
        0
    };
    Ok(ActivationFrame {
        desc: ActivationDesc {
            version: 1,
            dtype: RuntimeActivationDType::F32,
            layout: RuntimeActivationLayout::TokenMajor,
            producer_stage_index: -1,
            layer_start: i32::try_from(args.layer_start.saturating_sub(1))
                .context("layer_start exceeds i32")?,
            layer_end: i32::try_from(args.layer_start).context("layer_start exceeds i32")?,
            token_count: u32::try_from(args.tokens).context("tokens exceeds u32")?,
            sequence_count: 1,
            payload_bytes: u64::try_from(payload.len()).context("payload length exceeds u64")?,
            flags,
        },
        payload,
    })
}

#[derive(Clone, Copy)]
struct SyntheticTopKSideband {
    width: usize,
}

fn synthetic_top_k_sideband_config() -> Result<Option<SyntheticTopKSideband>> {
    if !env_flag_enabled(ENV_SYNTHETIC_TOP_K_SIDEBAND) {
        return Ok(None);
    }
    let width = match std::env::var(ENV_SYNTHETIC_TOP_K_WIDTH) {
        Ok(value) if !value.trim().is_empty() => value
            .trim()
            .parse::<usize>()
            .with_context(|| format!("parse {ENV_SYNTHETIC_TOP_K_WIDTH}"))?,
        _ => DEFAULT_SYNTHETIC_TOP_K_WIDTH,
    };
    if width == 0 {
        bail!("{ENV_SYNTHETIC_TOP_K_WIDTH} must be greater than zero");
    }
    Ok(Some(SyntheticTopKSideband { width }))
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

fn synthetic_top_k_sideband_bytes(tokens: usize, width: usize) -> Result<usize> {
    tokens
        .checked_mul(width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
        .context("synthetic GLM-DSA top-k sideband size overflow")
}

fn append_synthetic_top_k_sideband(
    payload: &mut Vec<u8>,
    tokens: usize,
    width: usize,
) -> Result<()> {
    let bytes = synthetic_top_k_sideband_bytes(tokens, width)?;
    payload.reserve(bytes);
    for _token in 0..tokens {
        for i_top in 0..width {
            let index = i32::try_from(i_top).context("synthetic top-k index exceeds i32")?;
            payload.extend_from_slice(&index.to_ne_bytes());
        }
    }
    Ok(())
}

fn synthetic_activation_value(token: usize, dim: usize) -> f32 {
    let residue = ((token.wrapping_mul(31) + dim.wrapping_mul(17)) % 97) as f32;
    (residue / 97.0 - 0.5) * 0.02
}

fn positions(tokens: usize) -> Result<Vec<i32>> {
    (0..tokens)
        .map(|position| i32::try_from(position).context("position exceeds i32"))
        .collect()
}

fn package_part_summary(part: &PackagePart) -> PackagePartSummary {
    PackagePartSummary {
        role: part.role.clone(),
        layer_index: part.layer_index,
        path: part.path.clone(),
        artifact_bytes: part.artifact_bytes,
    }
}

fn write_report(output: Option<&Path>, report: &MicrobenchReport) -> Result<()> {
    let encoded = format!("{}\n", serde_json::to_string_pretty(report)?);
    if let Some(output) = output {
        fs::write(output, encoded).with_context(|| format!("write {}", output.display()))?;
    } else {
        print!("{encoded}");
    }
    Ok(())
}

struct NativeLogCapture {
    path: Option<PathBuf>,
    active: bool,
}

impl NativeLogCapture {
    fn start(enabled: bool) -> Result<Self> {
        if !enabled {
            return Ok(Self {
                path: None,
                active: false,
            });
        }
        let path = native_log_capture_path()?;
        redirect_native_logs_to_file(&path)?;
        Ok(Self {
            path: Some(path),
            active: true,
        })
    }

    fn finish(mut self) -> Result<NativeTimingCapture> {
        let Some(path) = self.path.clone() else {
            return Ok(NativeTimingCapture::default());
        };
        restore_native_logs();
        self.active = false;
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read native timing log {}", path.display()))?;
        Ok(NativeTimingCapture {
            log_path: Some(path),
            direct_sparse_decision_records: parse_direct_sparse_decision_records(&text)
                .context("parse native direct sparse decisions")?,
            metal_dispatch_records: parse_metal_dispatch_records(&text)
                .context("parse native Metal dispatch records")?,
            op_timing_records: parse_timing_records(&text).context("parse native op timings")?,
            group_timing_records: parse_timing_group_records(&text)
                .context("parse native group timings")?,
        })
    }
}

impl Drop for NativeLogCapture {
    fn drop(&mut self) {
        if self.active {
            restore_native_logs();
            self.active = false;
        }
    }
}

#[derive(Default)]
struct NativeTimingCapture {
    log_path: Option<PathBuf>,
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    op_timing_records: Vec<TimingRecord>,
    group_timing_records: Vec<TimingGroupRecord>,
}

fn native_log_capture_path() -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before unix epoch")?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "skippy-bench-glm-dsa-layer-microbench-{}-{nanos}.log",
        std::process::id()
    )))
}

fn skip_warmup_records(records: Vec<TimingRecord>, warmup: usize) -> Vec<TimingRecord> {
    records.into_iter().skip(warmup).collect()
}

fn skip_warmup_group_records(
    records: Vec<TimingGroupRecord>,
    warmup: usize,
) -> Vec<TimingGroupRecord> {
    records
        .into_iter()
        .filter_map(|mut record| {
            if record.record_index < warmup {
                return None;
            }
            record.record_index -= warmup;
            Some(record)
        })
        .collect()
}

fn retain_case_decision_records(
    records: Vec<DirectSparseDecisionRecord>,
    tokens: usize,
) -> Vec<DirectSparseDecisionRecord> {
    let Ok(tokens) = i64::try_from(tokens) else {
        return Vec::new();
    };
    records
        .into_iter()
        .filter(|record| record.ubatch_tokens == tokens)
        .collect()
}

fn summarize_direct_sparse_decisions(
    records: &[DirectSparseDecisionRecord],
) -> DirectSparseDecisionSummary {
    let mut summary = DirectSparseDecisionSummary {
        records: records.len(),
        ..DirectSparseDecisionSummary::default()
    };
    for record in records {
        if record.use_direct {
            summary.use_direct += 1;
        } else {
            summary.fallback += 1;
        }
        if record.decode_shape {
            summary.decode_shape += 1;
        }
        if record.prefill_shape {
            summary.prefill_shape += 1;
        }
        if record.token_shape_allowed {
            summary.token_shape_allowed += 1;
        }
    }
    summary
}

fn compare_case_outputs(
    baseline_outputs: &[ActivationFrame],
    candidate_outputs: &[ActivationFrame],
    args: &GlmDsaLayerMicrobenchArgs,
) -> Result<ParityComparison> {
    if baseline_outputs.len() != candidate_outputs.len() {
        bail!(
            "baseline output count {} did not match candidate output count {}",
            baseline_outputs.len(),
            candidate_outputs.len()
        );
    }
    let hidden_bytes = hidden_payload_bytes(args)?;
    let mut frames = Vec::with_capacity(baseline_outputs.len());
    let mut hidden_mismatches = 0usize;
    let mut sideband_mismatched_bytes = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    for (iteration, (baseline, candidate)) in baseline_outputs
        .iter()
        .zip(candidate_outputs.iter())
        .enumerate()
    {
        let frame = compare_activation_frames(
            iteration,
            baseline,
            candidate,
            hidden_bytes,
            args.parity_atol,
            args.parity_rtol,
        )?;
        hidden_mismatches += frame.hidden_mismatches;
        sideband_mismatched_bytes += frame.sideband_mismatched_bytes;
        max_abs_diff = max_abs_diff.max(frame.hidden_max_abs_diff);
        max_rel_diff = max_rel_diff.max(frame.hidden_max_rel_diff);
        frames.push(frame);
    }
    let passed = frames.iter().all(|frame| frame.passed);
    Ok(ParityComparison {
        passed,
        iterations: frames.len(),
        atol: args.parity_atol,
        rtol: args.parity_rtol,
        hidden_mismatches,
        sideband_mismatched_bytes,
        hidden_max_abs_diff: max_abs_diff,
        hidden_max_rel_diff: max_rel_diff,
        frames,
    })
}

fn hidden_payload_bytes(args: &GlmDsaLayerMicrobenchArgs) -> Result<usize> {
    let width = usize::try_from(args.activation_width).context("activation_width exceeds usize")?;
    args.tokens
        .checked_mul(width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("hidden activation payload size overflow")
}

fn compare_activation_frames(
    iteration: usize,
    baseline: &ActivationFrame,
    candidate: &ActivationFrame,
    hidden_bytes: usize,
    atol: f32,
    rtol: f32,
) -> Result<FrameParity> {
    ensure_hidden_payload("baseline", baseline, hidden_bytes)?;
    ensure_hidden_payload("candidate", candidate, hidden_bytes)?;
    let hidden = compare_hidden_payloads(
        &baseline.payload[..hidden_bytes],
        &candidate.payload[..hidden_bytes],
        atol,
        rtol,
    )?;
    let sideband = compare_sideband_payloads(
        &baseline.payload[hidden_bytes..],
        &candidate.payload[hidden_bytes..],
    );
    let output_flags_match = baseline.desc.flags == candidate.desc.flags;
    let payload_len_match = baseline.payload.len() == candidate.payload.len();
    let passed = output_flags_match
        && payload_len_match
        && hidden.mismatches == 0
        && sideband.mismatched_bytes == 0;
    Ok(FrameParity {
        iteration,
        passed,
        output_flags_match,
        baseline_output_flags: baseline.desc.flags,
        candidate_output_flags: candidate.desc.flags,
        payload_len_match,
        baseline_payload_bytes: baseline.payload.len(),
        candidate_payload_bytes: candidate.payload.len(),
        hidden_value_count: hidden.value_count,
        hidden_mismatches: hidden.mismatches,
        hidden_max_abs_diff: hidden.max_abs_diff,
        hidden_max_rel_diff: hidden.max_rel_diff,
        first_hidden_mismatch: hidden.first_mismatch,
        sideband_bytes: sideband.compared_bytes,
        sideband_mismatched_bytes: sideband.mismatched_bytes,
        first_sideband_mismatch: sideband.first_mismatch,
    })
}

fn ensure_hidden_payload(label: &str, frame: &ActivationFrame, hidden_bytes: usize) -> Result<()> {
    if frame.payload.len() < hidden_bytes {
        bail!(
            "{label} payload has {} bytes, expected at least {hidden_bytes} hidden bytes",
            frame.payload.len()
        );
    }
    Ok(())
}

fn compare_hidden_payloads(
    baseline: &[u8],
    candidate: &[u8],
    atol: f32,
    rtol: f32,
) -> Result<HiddenComparison> {
    if baseline.len() != candidate.len()
        || !baseline.len().is_multiple_of(std::mem::size_of::<f32>())
    {
        bail!("hidden payloads must be equal-sized f32 byte slices");
    }
    let mut mismatches = 0usize;
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut first_mismatch = None;
    for (index, (baseline_bytes, candidate_bytes)) in baseline
        .chunks_exact(std::mem::size_of::<f32>())
        .zip(candidate.chunks_exact(std::mem::size_of::<f32>()))
        .enumerate()
    {
        let baseline_value = f32::from_ne_bytes(
            baseline_bytes
                .try_into()
                .with_context(|| format!("read baseline f32 at {index}"))?,
        );
        let candidate_value = f32::from_ne_bytes(
            candidate_bytes
                .try_into()
                .with_context(|| format!("read candidate f32 at {index}"))?,
        );
        let abs_diff = (baseline_value - candidate_value).abs();
        let scale = baseline_value
            .abs()
            .max(candidate_value.abs())
            .max(f32::MIN_POSITIVE);
        let rel_diff = abs_diff / scale;
        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
        if !values_close(baseline_value, candidate_value, atol, rtol) {
            mismatches += 1;
            first_mismatch.get_or_insert(HiddenMismatch {
                index,
                baseline: baseline_value,
                candidate: candidate_value,
                abs_diff,
                rel_diff,
            });
        }
    }
    Ok(HiddenComparison {
        value_count: baseline.len() / std::mem::size_of::<f32>(),
        mismatches,
        max_abs_diff,
        max_rel_diff,
        first_mismatch,
    })
}

fn values_close(baseline: f32, candidate: f32, atol: f32, rtol: f32) -> bool {
    if baseline == candidate {
        return true;
    }
    if baseline.is_nan() || candidate.is_nan() {
        return baseline.is_nan() && candidate.is_nan();
    }
    let tolerance = atol + rtol * baseline.abs().max(candidate.abs());
    (baseline - candidate).abs() <= tolerance
}

fn compare_sideband_payloads(baseline: &[u8], candidate: &[u8]) -> SidebandComparison {
    let compared_bytes = baseline.len().min(candidate.len());
    let mut mismatched_bytes = baseline.len().abs_diff(candidate.len());
    let mut first_mismatch = None;
    for (index, (baseline_byte, candidate_byte)) in
        baseline.iter().zip(candidate.iter()).enumerate()
    {
        if baseline_byte != candidate_byte {
            mismatched_bytes += 1;
            first_mismatch.get_or_insert(index);
        }
    }
    if first_mismatch.is_none() && baseline.len() != candidate.len() {
        first_mismatch = Some(compared_bytes);
    }
    SidebandComparison {
        compared_bytes,
        mismatched_bytes,
        first_mismatch,
    }
}

struct MicrobenchCase {
    label: &'static str,
    flags: MicrobenchFlags,
    n_gpu_layers: i32,
    native_log_path: Option<PathBuf>,
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    op_timing_records: Vec<TimingRecord>,
    group_timing_records: Vec<TimingGroupRecord>,
    timings: Vec<IterationTiming>,
    outputs: Vec<ActivationFrame>,
}

impl MicrobenchCase {
    fn as_case_summary(&self) -> MicrobenchCaseSummary {
        MicrobenchCaseSummary {
            label: self.label,
            flags: self.flags,
            n_gpu_layers: self.n_gpu_layers,
            native_log_path: self.native_log_path.clone(),
            direct_sparse_decision_summary: summarize_direct_sparse_decisions(
                &self.direct_sparse_decision_records,
            ),
            direct_sparse_decision_records: self.direct_sparse_decision_records.clone(),
            metal_dispatch_records: self.metal_dispatch_records.clone(),
            op_timing_records: self.op_timing_records.clone(),
            group_timing_records: self.group_timing_records.clone(),
            timings: self.timings.clone(),
        }
    }
}

struct HiddenComparison {
    value_count: usize,
    mismatches: usize,
    max_abs_diff: f32,
    max_rel_diff: f32,
    first_mismatch: Option<HiddenMismatch>,
}

struct SidebandComparison {
    compared_bytes: usize,
    mismatched_bytes: usize,
    first_mismatch: Option<usize>,
}

#[derive(Serialize)]
struct MicrobenchReport {
    command: &'static str,
    model_id: String,
    stage_model: PathBuf,
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    activation_width: u32,
    tokens: usize,
    warmup: usize,
    iterations: usize,
    n_gpu_layers: i32,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    flags: MicrobenchFlags,
    selected_parts: Vec<PackagePartSummary>,
    input_payload_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_log_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "DirectSparseDecisionSummary::is_empty")]
    direct_sparse_decision_summary: DirectSparseDecisionSummary,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    op_timing_records: Vec<TimingRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    group_timing_records: Vec<TimingGroupRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<MicrobenchComparisonReport>,
    timings: Vec<IterationTiming>,
}

#[derive(Clone, Copy, Serialize)]
struct MicrobenchFlags {
    direct_sparse_attn: bool,
    direct_sparse_prefill: bool,
    fused_sparse_mask: bool,
    parallel_lightning_indexer: bool,
    op_timing: bool,
}

impl MicrobenchFlags {
    fn from_args(args: &GlmDsaLayerMicrobenchArgs) -> Self {
        Self {
            direct_sparse_attn: args.direct_sparse_attn,
            direct_sparse_prefill: args.direct_sparse_prefill,
            fused_sparse_mask: args.fused_sparse_mask,
            parallel_lightning_indexer: args.parallel_lightning_indexer,
            op_timing: args.op_timing,
        }
    }
}

#[derive(Clone, Serialize)]
struct MicrobenchCaseSummary {
    label: &'static str,
    flags: MicrobenchFlags,
    n_gpu_layers: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_log_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "DirectSparseDecisionSummary::is_empty")]
    direct_sparse_decision_summary: DirectSparseDecisionSummary,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    op_timing_records: Vec<TimingRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    group_timing_records: Vec<TimingGroupRecord>,
    timings: Vec<IterationTiming>,
}

#[derive(Clone, Copy, Default, Serialize)]
struct DirectSparseDecisionSummary {
    records: usize,
    use_direct: usize,
    fallback: usize,
    decode_shape: usize,
    prefill_shape: usize,
    token_shape_allowed: usize,
}

impl DirectSparseDecisionSummary {
    fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Serialize)]
struct MicrobenchComparisonReport {
    baseline: MicrobenchCaseSummary,
    candidate: MicrobenchCaseSummary,
    parity: ParityComparison,
}

#[derive(Serialize)]
struct ParityComparison {
    passed: bool,
    iterations: usize,
    atol: f32,
    rtol: f32,
    hidden_mismatches: usize,
    sideband_mismatched_bytes: usize,
    hidden_max_abs_diff: f32,
    hidden_max_rel_diff: f32,
    frames: Vec<FrameParity>,
}

#[derive(Serialize)]
struct FrameParity {
    iteration: usize,
    passed: bool,
    output_flags_match: bool,
    baseline_output_flags: u64,
    candidate_output_flags: u64,
    payload_len_match: bool,
    baseline_payload_bytes: usize,
    candidate_payload_bytes: usize,
    hidden_value_count: usize,
    hidden_mismatches: usize,
    hidden_max_abs_diff: f32,
    hidden_max_rel_diff: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_hidden_mismatch: Option<HiddenMismatch>,
    sideband_bytes: usize,
    sideband_mismatched_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_sideband_mismatch: Option<usize>,
}

#[derive(Serialize)]
struct HiddenMismatch {
    index: usize,
    baseline: f32,
    candidate: f32,
    abs_diff: f32,
    rel_diff: f32,
}

#[derive(Serialize)]
struct PackagePartSummary {
    role: String,
    layer_index: Option<u32>,
    path: PathBuf,
    artifact_bytes: u64,
}

#[derive(Clone, Serialize)]
struct IterationTiming {
    iteration: usize,
    elapsed_ms: f64,
    output_payload_bytes: usize,
    output_flags: u64,
}
