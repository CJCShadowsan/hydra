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
    glm_dsa_microbench_summary::{
        GlmDsaDispatchSummary, GlmDsaOpTimingSummary, RoutedMoeTimingSummary,
        TimingDistributionSummary, summarize_elapsed_ms, summarize_glm_dsa_op_timing,
        summarize_metal_dispatch, summarize_routed_moe_timing,
    },
    glm_dsa_op_report::{
        DirectSparseDecisionRecord, HotTensorRecord, MetalDispatchRecord, TimingGroupRecord,
        TimingRecord, parse_direct_sparse_decision_records, parse_hot_tensor_records,
        parse_metal_dispatch_records, parse_timing_group_records, parse_timing_records,
    },
};

const ACTIVATION_FLAG_GLM_DSA_TOP_K: u64 = 1 << 3;
const ENV_SYNTHETIC_TOP_K_SIDEBAND: &str = "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_SIDEBAND";
const ENV_SYNTHETIC_TOP_K_WIDTH: &str = "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_WIDTH";
const ENV_REAL_TOP_K_SOURCE_LAYER_START: &str =
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_SOURCE_LAYER_START";
const ENV_REAL_TOP_K_CACHE_DIR: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CACHE_DIR";
const ENV_REAL_TOP_K_REQUIRE_CACHE: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_REQUIRE_CACHE";
const ENV_REAL_TOP_K_CHAIN_SOURCES: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CHAIN_SOURCES";
const ENV_REAL_TOP_K_MAX_SOURCE_BYTES: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_MAX_SOURCE_BYTES";
const DEFAULT_SYNTHETIC_TOP_K_WIDTH: usize = 256;
const DEFAULT_REAL_TOP_K_MAX_SOURCE_BYTES: u64 = 110 * 1024 * 1024 * 1024;
const INPUT_FRAME_CACHE_MAGIC: &[u8; 16] = b"SKPGLMDSAFRM1\0\0\0";

pub fn glm_dsa_layer_microbench(args: GlmDsaLayerMicrobenchArgs) -> Result<()> {
    validate_args(&args)?;

    let selected = select_layer_package_parts(&package_request(&args))
        .context("select GLM-DSA layer package parts")?;
    let runtime_config = runtime_config(&args)?;
    let token_ids = vec![1_i32; args.tokens];
    let positions = positions(args.tokens)?;
    let flags = MicrobenchFlags::from_args(&args);
    let input = prepare_input_activation(&args, &token_ids, &positions, flags)?;
    let comparison = if args.compare_dense_fallback {
        Some(run_dense_fallback_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            &input.frame,
            &token_ids,
            &positions,
            flags,
        )?)
    } else if args.compare_cpu_direct_sparse {
        Some(run_cpu_direct_sparse_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            &input.frame,
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
                &input.frame,
                &token_ids,
                &positions,
                false,
            )?;
            case.as_case_summary()
        }
    };
    let optimized_dispatch_probe = if should_run_optimized_dispatch_probe(flags) {
        let probe_flags = MicrobenchFlags {
            op_timing: false,
            metal_dispatch_log: true,
            metal_topk_moe_route_fusion: true,
            ..flags
        };
        Some(
            run_microbench_case(
                "optimized_dispatch_probe",
                &selected.absolute_paths,
                &runtime_config,
                &args,
                probe_flags,
                &input.frame,
                &token_ids,
                &positions,
                false,
            )?
            .as_case_summary(),
        )
    } else {
        None
    };

    let direct_sparse_decision_summary =
        summarize_direct_sparse_decisions(&case.direct_sparse_decision_records);
    let timing_summary = case.timing_summary.clone();
    let metal_dispatch_summary = case.metal_dispatch_summary.clone();
    let op_timing_summary = case.op_timing_summary.clone();
    let routed_moe_timing_summary = case.routed_moe_timing_summary.clone();
    let profile_integrity = ProfileIntegrityReport::new(
        flags,
        &metal_dispatch_summary,
        &timing_summary,
        optimized_dispatch_probe.as_ref(),
    );
    let route_fusion_guard = args
        .require_optimized_route_fusion
        .then(|| build_route_fusion_guard(&case, optimized_dispatch_probe.as_ref()));
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
        input_source: input.report,
        selected_parts: selected
            .selected_parts
            .iter()
            .map(package_part_summary)
            .collect(),
        input_payload_bytes: input.frame.payload.len(),
        native_log_path: case.native_log_path,
        direct_sparse_decision_summary,
        timing_summary,
        metal_dispatch_summary,
        op_timing_summary,
        routed_moe_timing_summary,
        profile_integrity,
        route_fusion_guard,
        direct_sparse_decision_records: case.direct_sparse_decision_records,
        metal_dispatch_records: case.metal_dispatch_records,
        op_timing_records: case.op_timing_records,
        group_timing_records: case.group_timing_records,
        hot_tensor_records: case.hot_tensor_records,
        optimized_dispatch_probe,
        comparison,
        timings: case.timings,
    };
    let parity_passed = report
        .comparison
        .as_ref()
        .is_none_or(|comparison| comparison.parity.passed);

    write_report(args.output.as_deref(), &report)?;
    if let Some(guard) = &report.route_fusion_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA optimized route-fusion guard failed for {}: candidates={} skipped={} fused_dispatches={} reasons={}",
            guard.checked_case,
            guard.encode_candidate_records,
            guard.encode_skipped_candidate_records,
            guard.fused_dispatch_records,
            guard.reason_summary
        );
    }
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
    let native_logs = NativeLogCapture::start(flags.capture_native_logs())?;
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
        hot_tensor_records: skip_warmup_hot_tensor_records(
            native_timings.hot_tensor_records,
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
    if real_top_k_source_layer_start(args)?.is_some()
        && synthetic_top_k_sideband_config()?.is_some()
    {
        bail!(
            "{ENV_REAL_TOP_K_SOURCE_LAYER_START} cannot be combined with {ENV_SYNTHETIC_TOP_K_SIDEBAND}"
        );
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
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_METAL_DISPATCH",
        flags.metal_dispatch_log,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_METAL_TOPK_MOE_FUSION",
        flags.metal_topk_moe_route_fusion,
    );
}

fn set_env_flag(name: &str, enabled: bool) {
    // This command is single-threaded and sets native runtime flags before opening llama.cpp.
    unsafe {
        std::env::set_var(name, if enabled { "1" } else { "0" });
    }
}

fn package_request(args: &GlmDsaLayerMicrobenchArgs) -> PackageStageRequest {
    package_request_for_range(args, args.layer_start, args.layer_end)
}

fn package_request_for_range(
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
) -> PackageStageRequest {
    PackageStageRequest {
        model_id: args.model_id.clone(),
        topology_id: "glm-dsa-layer-microbench".to_string(),
        package_ref: args.stage_model.to_string_lossy().to_string(),
        stage_id: format!("layers-{layer_start}-{layer_end}"),
        layer_start,
        layer_end,
        include_embeddings: false,
        include_output: false,
    }
}

fn runtime_config(args: &GlmDsaLayerMicrobenchArgs) -> Result<RuntimeConfig> {
    runtime_config_for_range(args, args.layer_start, args.layer_end)
}

fn runtime_config_for_range(
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
) -> Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start,
        layer_end,
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

fn prepare_input_activation(
    args: &GlmDsaLayerMicrobenchArgs,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
) -> Result<PreparedInputActivation> {
    if let Some(source_layer_start) = real_top_k_source_layer_start(args)? {
        return real_top_k_activation_frame(args, token_ids, positions, flags, source_layer_start);
    }
    let top_k_sideband = synthetic_top_k_sideband_config()?;
    let frame = synthetic_activation_frame_for_layer(args, args.layer_start, top_k_sideband)?;
    let report = InputSourceReport::Synthetic {
        top_k_sideband: top_k_sideband.map(|sideband| sideband.width),
    };
    Ok(PreparedInputActivation { frame, report })
}

fn real_top_k_source_layer_start(args: &GlmDsaLayerMicrobenchArgs) -> Result<Option<u32>> {
    let Ok(value) = std::env::var(ENV_REAL_TOP_K_SOURCE_LAYER_START) else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "0" || trimmed.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let layer_start = trimmed
        .parse::<u32>()
        .with_context(|| format!("parse {ENV_REAL_TOP_K_SOURCE_LAYER_START}"))?;
    if layer_start >= args.layer_start {
        bail!(
            "{ENV_REAL_TOP_K_SOURCE_LAYER_START} must be less than target layer_start {}",
            args.layer_start
        );
    }
    Ok(Some(layer_start))
}

fn real_top_k_activation_frame(
    args: &GlmDsaLayerMicrobenchArgs,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
    source_layer_start: u32,
) -> Result<PreparedInputActivation> {
    let source_layer_end = args.layer_start;
    let generated = generate_real_top_k_frame(
        args,
        token_ids,
        positions,
        flags,
        source_layer_start,
        source_layer_end,
    )?;
    real_top_k_prepared_input(
        args,
        generated.frame,
        source_layer_start,
        source_layer_end,
        generated.selected_parts,
        generated.cache_path,
        generated.cache_hit,
    )
}

fn generate_real_top_k_frame(
    args: &GlmDsaLayerMicrobenchArgs,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
    source_layer_start: u32,
    source_layer_end: u32,
) -> Result<GeneratedTopKFrame> {
    let cache_path = real_top_k_cache_path(args, flags, source_layer_start, source_layer_end)?;
    if let Some(path) = cache_path.as_ref()
        && path.exists()
    {
        let frame = read_activation_frame_cache(path)
            .with_context(|| format!("read real top-k input cache {}", path.display()))?;
        validate_real_top_k_frame_for_range(args, &frame, source_layer_start, source_layer_end)?;
        return Ok(GeneratedTopKFrame {
            frame,
            selected_parts: Vec::new(),
            cache_path,
            cache_hit: true,
        });
    }
    if env_flag_enabled(ENV_REAL_TOP_K_REQUIRE_CACHE) {
        let cache = cache_path.as_ref().map_or_else(
            || "<disabled>".to_string(),
            |path| path.display().to_string(),
        );
        bail!("real top-k input cache is required but missing: {cache}");
    }

    let source_input =
        real_top_k_source_input(args, token_ids, positions, flags, source_layer_start)?;
    let source_request = package_request_for_range(args, source_layer_start, source_layer_end);
    let source_selected = select_layer_package_parts(&source_request)
        .context("select GLM-DSA real top-k source layer package parts")?;
    guard_real_top_k_source_size(&source_selected.selected_parts)
        .context("check GLM-DSA real top-k source span size")?;
    let source_config = runtime_config_for_range(args, source_layer_start, source_layer_end)?;
    let source_flags = MicrobenchFlags {
        direct_sparse_attn: false,
        direct_sparse_prefill: false,
        ..flags
    };
    configure_env_flags(source_flags);
    let source_model = StageModel::open_from_parts(&source_selected.absolute_paths, &source_config)
        .with_context(|| {
            format!("open GLM-DSA real top-k source model {source_layer_start}..{source_layer_end}")
        })?;
    let mut source_session = source_model
        .create_session()
        .context("create GLM-DSA real top-k source session")?;
    let frame = source_session
        .prefill_chunk_frame_with_positions(token_ids, positions, Some(&source_input), 0)
        .with_context(|| {
            format!("run GLM-DSA real top-k source {source_layer_start}..{source_layer_end}")
        })?;
    validate_real_top_k_frame_for_range(args, &frame, source_layer_start, source_layer_end)?;
    if let Some(path) = cache_path.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("create real top-k input cache dir {}", parent.display())
            })?;
        }
        write_activation_frame_cache(path, &frame)
            .with_context(|| format!("write real top-k input cache {}", path.display()))?;
    }
    Ok(GeneratedTopKFrame {
        frame,
        selected_parts: source_selected
            .selected_parts
            .iter()
            .map(package_part_summary)
            .collect(),
        cache_path,
        cache_hit: false,
    })
}

fn real_top_k_source_input(
    args: &GlmDsaLayerMicrobenchArgs,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
    source_layer_start: u32,
) -> Result<ActivationFrame> {
    let Some(chain_source_start) = chained_real_top_k_source_for(source_layer_start)? else {
        return synthetic_activation_frame_for_layer(args, source_layer_start, None);
    };
    if chain_source_start >= source_layer_start {
        bail!(
            "{ENV_REAL_TOP_K_CHAIN_SOURCES} selected invalid chain source {chain_source_start} for {source_layer_start}"
        );
    }
    generate_real_top_k_frame(
        args,
        token_ids,
        positions,
        flags,
        chain_source_start,
        source_layer_start,
    )
    .map(|generated| generated.frame)
}

fn chained_real_top_k_source_for(target_layer_start: u32) -> Result<Option<u32>> {
    let mut selected = None;
    for source in env_real_top_k_chain_sources()? {
        if source < target_layer_start && selected.is_none_or(|current| source > current) {
            selected = Some(source);
        }
    }
    Ok(selected)
}

fn env_real_top_k_chain_sources() -> Result<Vec<u32>> {
    let Ok(value) = std::env::var(ENV_REAL_TOP_K_CHAIN_SOURCES) else {
        return Ok(Vec::new());
    };
    parse_real_top_k_chain_sources(&value)
}

fn parse_real_top_k_chain_sources(value: &str) -> Result<Vec<u32>> {
    let mut sources = Vec::new();
    for raw in value.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        sources.push(
            trimmed
                .parse::<u32>()
                .with_context(|| format!("parse {ENV_REAL_TOP_K_CHAIN_SOURCES} entry {trimmed}"))?,
        );
    }
    Ok(sources)
}

fn guard_real_top_k_source_size(selected_parts: &[PackagePart]) -> Result<()> {
    let Some(max_bytes) = real_top_k_max_source_bytes()? else {
        return Ok(());
    };
    guard_real_top_k_source_size_with_limit(selected_parts, max_bytes)
}

fn guard_real_top_k_source_size_with_limit(
    selected_parts: &[PackagePart],
    max_bytes: u64,
) -> Result<()> {
    let artifact_bytes = selected_parts
        .iter()
        .try_fold(0_u64, |sum, part| sum.checked_add(part.artifact_bytes))
        .context("real top-k source artifact byte count overflow")?;
    if artifact_bytes > max_bytes {
        bail!(
            "real top-k source span selects {} bytes of layer artifacts, above {} byte limit; use {ENV_REAL_TOP_K_CHAIN_SOURCES} to split the source span or set {ENV_REAL_TOP_K_MAX_SOURCE_BYTES}=off to override",
            artifact_bytes,
            max_bytes
        );
    }
    Ok(())
}

fn real_top_k_max_source_bytes() -> Result<Option<u64>> {
    let Ok(value) = std::env::var(ENV_REAL_TOP_K_MAX_SOURCE_BYTES) else {
        return Ok(Some(DEFAULT_REAL_TOP_K_MAX_SOURCE_BYTES));
    };
    parse_real_top_k_max_source_bytes(&value)
}

fn parse_real_top_k_max_source_bytes(value: &str) -> Result<Option<u64>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Some(DEFAULT_REAL_TOP_K_MAX_SOURCE_BYTES));
    }
    if trimmed.eq_ignore_ascii_case("off") || trimmed == "0" {
        return Ok(None);
    }
    trimmed
        .parse::<u64>()
        .map(Some)
        .with_context(|| format!("parse {ENV_REAL_TOP_K_MAX_SOURCE_BYTES}"))
}

fn real_top_k_prepared_input(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: ActivationFrame,
    source_layer_start: u32,
    source_layer_end: u32,
    selected_parts: Vec<PackagePartSummary>,
    cache_path: Option<PathBuf>,
    cache_hit: bool,
) -> Result<PreparedInputActivation> {
    let report = InputSourceReport::RealTopK {
        layer_start: source_layer_start,
        layer_end: source_layer_end,
        output_flags: frame.desc.flags,
        output_payload_bytes: frame.payload.len(),
        sideband_bytes: real_top_k_sideband_bytes(args, &frame)?,
        cache_path,
        cache_hit,
        selected_parts,
    };
    Ok(PreparedInputActivation { frame, report })
}

fn real_top_k_cache_path(
    args: &GlmDsaLayerMicrobenchArgs,
    flags: MicrobenchFlags,
    source_layer_start: u32,
    source_layer_end: u32,
) -> Result<Option<PathBuf>> {
    let Ok(value) = std::env::var(ENV_REAL_TOP_K_CACHE_DIR) else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let n_batch = args.n_batch.unwrap_or_else(|| bounded_u32(args.tokens));
    let n_ubatch = args.n_ubatch.unwrap_or_else(|| bounded_u32(args.tokens));
    let file_name = format!(
        "real-topk-src{}-dst{}-tok{}-ctx{}-act{}-ngpu{}-nb{}-nub{}-pi{}.skippy-frame",
        source_layer_start,
        source_layer_end,
        args.tokens,
        args.ctx_size,
        args.activation_width,
        args.n_gpu_layers,
        n_batch,
        n_ubatch,
        u8::from(flags.parallel_lightning_indexer)
    );
    Ok(Some(PathBuf::from(trimmed).join(file_name)))
}

fn write_activation_frame_cache(path: &Path, frame: &ActivationFrame) -> Result<()> {
    let mut encoded = Vec::with_capacity(INPUT_FRAME_CACHE_MAGIC.len() + 64 + frame.payload.len());
    encoded.extend_from_slice(INPUT_FRAME_CACHE_MAGIC);
    push_u32(&mut encoded, frame.desc.version);
    push_i32(&mut encoded, frame.desc.dtype as i32);
    push_i32(&mut encoded, frame.desc.layout as i32);
    push_i32(&mut encoded, frame.desc.producer_stage_index);
    push_i32(&mut encoded, frame.desc.layer_start);
    push_i32(&mut encoded, frame.desc.layer_end);
    push_u32(&mut encoded, frame.desc.token_count);
    push_u32(&mut encoded, frame.desc.sequence_count);
    push_u64(&mut encoded, frame.desc.payload_bytes);
    push_u64(&mut encoded, frame.desc.flags);
    encoded.extend_from_slice(&frame.payload);
    fs::write(path, encoded).with_context(|| format!("write {}", path.display()))
}

fn read_activation_frame_cache(path: &Path) -> Result<ActivationFrame> {
    let encoded = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut cursor = CacheCursor::new(&encoded);
    cursor.expect_magic()?;
    let desc = ActivationDesc {
        version: cursor.read_u32("version")?,
        dtype: activation_dtype_from_i32(cursor.read_i32("dtype")?)?,
        layout: activation_layout_from_i32(cursor.read_i32("layout")?)?,
        producer_stage_index: cursor.read_i32("producer_stage_index")?,
        layer_start: cursor.read_i32("layer_start")?,
        layer_end: cursor.read_i32("layer_end")?,
        token_count: cursor.read_u32("token_count")?,
        sequence_count: cursor.read_u32("sequence_count")?,
        payload_bytes: cursor.read_u64("payload_bytes")?,
        flags: cursor.read_u64("flags")?,
    };
    let payload = cursor.remaining().to_vec();
    if u64::try_from(payload.len()).context("cached payload length exceeds u64")?
        != desc.payload_bytes
    {
        bail!(
            "cached activation payload has {} bytes, descriptor says {}",
            payload.len(),
            desc.payload_bytes
        );
    }
    Ok(ActivationFrame { desc, payload })
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(output: &mut Vec<u8>, value: i32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn activation_dtype_from_i32(value: i32) -> Result<RuntimeActivationDType> {
    match value {
        0 => Ok(RuntimeActivationDType::Unknown),
        1 => Ok(RuntimeActivationDType::F32),
        2 => Ok(RuntimeActivationDType::F16),
        3 => Ok(RuntimeActivationDType::Bf16),
        _ => bail!("cached activation frame has unsupported dtype {value}"),
    }
}

fn activation_layout_from_i32(value: i32) -> Result<RuntimeActivationLayout> {
    match value {
        0 => Ok(RuntimeActivationLayout::Opaque),
        1 => Ok(RuntimeActivationLayout::TokenMajor),
        _ => bail!("cached activation frame has unsupported layout {value}"),
    }
}

struct CacheCursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> CacheCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn expect_magic(&mut self) -> Result<()> {
        let magic = self.read_bytes(INPUT_FRAME_CACHE_MAGIC.len(), "magic")?;
        if magic != INPUT_FRAME_CACHE_MAGIC {
            bail!("cached activation frame has invalid magic");
        }
        Ok(())
    }

    fn read_u32(&mut self, field: &str) -> Result<u32> {
        let bytes = self.read_array::<4>(field)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self, field: &str) -> Result<i32> {
        let bytes = self.read_array::<4>(field)?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self, field: &str) -> Result<u64> {
        let bytes = self.read_array::<8>(field)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_array<const N: usize>(&mut self, field: &str) -> Result<[u8; N]> {
        self.read_bytes(N, field)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("cached activation frame field {field} had wrong size"))
    }

    fn read_bytes(&mut self, len: usize, field: &str) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .context("cached activation frame offset overflow")?;
        if end > self.data.len() {
            bail!("cached activation frame ended while reading {field}");
        }
        let bytes = &self.data[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn remaining(&self) -> &'a [u8] {
        &self.data[self.offset..]
    }
}

fn validate_real_top_k_frame_for_range(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: &ActivationFrame,
    source_layer_start: u32,
    source_layer_end: u32,
) -> Result<()> {
    if frame.desc.layer_end != i32::try_from(source_layer_end).context("layer_end exceeds i32")? {
        bail!(
            "real top-k source {}..{} produced layer_end {}, expected {}",
            source_layer_start,
            source_layer_end,
            frame.desc.layer_end,
            source_layer_end
        );
    }
    if (frame.desc.flags & ACTIVATION_FLAG_GLM_DSA_TOP_K) == 0 {
        bail!(
            "real top-k source {}..{} did not produce GLM-DSA top-k sideband",
            source_layer_start,
            source_layer_end
        );
    }
    let hidden_bytes = hidden_payload_bytes(args)?;
    if frame.payload.len() <= hidden_bytes {
        bail!(
            "real top-k source {}..{} payload has no top-k sideband: {} bytes <= {hidden_bytes}",
            source_layer_start,
            source_layer_end,
            frame.payload.len()
        );
    }
    Ok(())
}

fn real_top_k_sideband_bytes(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: &ActivationFrame,
) -> Result<usize> {
    let hidden_bytes = hidden_payload_bytes(args)?;
    if frame.payload.len() < hidden_bytes {
        bail!(
            "real top-k source payload has {} bytes, expected at least {hidden_bytes}",
            frame.payload.len()
        );
    }
    Ok(frame.payload.len() - hidden_bytes)
}

fn synthetic_activation_frame_for_layer(
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    top_k_sideband: Option<SyntheticTopKSideband>,
) -> Result<ActivationFrame> {
    let width = usize::try_from(args.activation_width).context("activation_width exceeds usize")?;
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
            layer_start: i32::try_from(layer_start.saturating_sub(1))
                .context("input layer_start exceeds i32")?,
            layer_end: i32::try_from(layer_start).context("input layer_start exceeds i32")?,
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

struct PreparedInputActivation {
    frame: ActivationFrame,
    report: InputSourceReport,
}

struct GeneratedTopKFrame {
    frame: ActivationFrame,
    selected_parts: Vec<PackagePartSummary>,
    cache_path: Option<PathBuf>,
    cache_hit: bool,
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
            hot_tensor_records: parse_hot_tensor_records(&text)
                .context("parse native hot tensor timings")?,
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
    hot_tensor_records: Vec<HotTensorRecord>,
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

fn skip_warmup_hot_tensor_records(
    records: Vec<HotTensorRecord>,
    warmup: usize,
) -> Vec<HotTensorRecord> {
    records
        .into_iter()
        .filter(|record| record.record_index >= warmup)
        .map(|mut record| {
            record.record_index -= warmup;
            record
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
    hot_tensor_records: Vec<HotTensorRecord>,
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
            timing_summary: summarize_elapsed_ms(
                self.timings.iter().map(|timing| timing.elapsed_ms),
            ),
            metal_dispatch_summary: summarize_metal_dispatch(&self.metal_dispatch_records),
            op_timing_summary: summarize_glm_dsa_op_timing(&self.op_timing_records),
            routed_moe_timing_summary: summarize_routed_moe_timing(&self.op_timing_records),
            direct_sparse_decision_records: self.direct_sparse_decision_records.clone(),
            metal_dispatch_records: self.metal_dispatch_records.clone(),
            op_timing_records: self.op_timing_records.clone(),
            group_timing_records: self.group_timing_records.clone(),
            hot_tensor_records: self.hot_tensor_records.clone(),
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
    input_source: InputSourceReport,
    selected_parts: Vec<PackagePartSummary>,
    input_payload_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_log_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "DirectSparseDecisionSummary::is_empty")]
    direct_sparse_decision_summary: DirectSparseDecisionSummary,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    timing_summary: TimingDistributionSummary,
    #[serde(skip_serializing_if = "GlmDsaDispatchSummary::is_empty")]
    metal_dispatch_summary: GlmDsaDispatchSummary,
    #[serde(skip_serializing_if = "GlmDsaOpTimingSummary::is_empty")]
    op_timing_summary: GlmDsaOpTimingSummary,
    #[serde(skip_serializing_if = "RoutedMoeTimingSummary::is_empty")]
    routed_moe_timing_summary: RoutedMoeTimingSummary,
    profile_integrity: ProfileIntegrityReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    route_fusion_guard: Option<RouteFusionGuardReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    op_timing_records: Vec<TimingRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    group_timing_records: Vec<TimingGroupRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    hot_tensor_records: Vec<HotTensorRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_dispatch_probe: Option<MicrobenchCaseSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<MicrobenchComparisonReport>,
    timings: Vec<IterationTiming>,
}

#[derive(Serialize)]
struct RouteFusionGuardReport {
    checked_case: &'static str,
    passed: bool,
    encode_candidate_records: usize,
    encode_fused_candidate_records: usize,
    encode_skipped_candidate_records: usize,
    fused_dispatch_records: usize,
    reason_summary: String,
}

fn build_route_fusion_guard(
    candidate: &MicrobenchCaseSummary,
    optimized_probe: Option<&MicrobenchCaseSummary>,
) -> RouteFusionGuardReport {
    let checked = optimized_probe.unwrap_or(candidate);
    let dispatch = &checked.metal_dispatch_summary;
    let encode_candidate_records = dispatch.topk_moe_route_encode_candidate_records;
    let encode_fused_candidate_records = dispatch.topk_moe_route_encode_fused_candidate_records;
    let encode_skipped_candidate_records = dispatch.topk_moe_route_encode_skipped_candidate_records;
    let fused_dispatch_records = dispatch.topk_moe_route_fused_records;
    let passed = encode_candidate_records > 0
        && encode_skipped_candidate_records == 0
        && fused_dispatch_records > 0;
    RouteFusionGuardReport {
        checked_case: checked.label,
        passed,
        encode_candidate_records,
        encode_fused_candidate_records,
        encode_skipped_candidate_records,
        fused_dispatch_records,
        reason_summary: summarize_route_fusion_reasons(dispatch),
    }
}

fn summarize_route_fusion_reasons(dispatch: &GlmDsaDispatchSummary) -> String {
    if dispatch.route_fusion_reasons.is_empty() {
        return "none".to_string();
    }
    dispatch
        .route_fusion_reasons
        .iter()
        .map(|reason| format!("{}:{}={}", reason.op, reason.reason, reason.records))
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum InputSourceReport {
    Synthetic {
        #[serde(skip_serializing_if = "Option::is_none")]
        top_k_sideband: Option<usize>,
    },
    RealTopK {
        layer_start: u32,
        layer_end: u32,
        output_flags: u64,
        output_payload_bytes: usize,
        sideband_bytes: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_path: Option<PathBuf>,
        cache_hit: bool,
        selected_parts: Vec<PackagePartSummary>,
    },
}

#[derive(Clone, Copy, Serialize)]
struct MicrobenchFlags {
    direct_sparse_attn: bool,
    direct_sparse_prefill: bool,
    fused_sparse_mask: bool,
    parallel_lightning_indexer: bool,
    op_timing: bool,
    metal_dispatch_log: bool,
    metal_topk_moe_route_fusion: bool,
}

impl MicrobenchFlags {
    fn from_args(args: &GlmDsaLayerMicrobenchArgs) -> Self {
        Self {
            direct_sparse_attn: args.direct_sparse_attn,
            direct_sparse_prefill: args.direct_sparse_prefill,
            fused_sparse_mask: args.fused_sparse_mask,
            parallel_lightning_indexer: args.parallel_lightning_indexer,
            op_timing: args.op_timing,
            metal_dispatch_log: args.metal_dispatch_log,
            metal_topk_moe_route_fusion: args.metal_topk_moe_route_fusion,
        }
    }

    fn capture_native_logs(self) -> bool {
        self.op_timing || self.metal_dispatch_log
    }
}

fn should_run_optimized_dispatch_probe(flags: MicrobenchFlags) -> bool {
    flags.op_timing && flags.metal_dispatch_log
}

#[derive(Serialize)]
struct ProfileIntegrityReport {
    op_timing_enabled: bool,
    metal_dispatch_log_enabled: bool,
    route_fusion_active: bool,
    route_fusion_encode_candidate_records: usize,
    route_fusion_encode_skipped_candidate_records: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_probe_route_fusion_active: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_probe_route_fusion_encode_candidate_records: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_probe_route_fusion_encode_skipped_candidate_records: Option<usize>,
    diagnostic_timing_may_disable_route_fusion: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostic_mean_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_probe_mean_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostic_slowdown_vs_optimized_probe: Option<f64>,
}

impl ProfileIntegrityReport {
    fn new(
        flags: MicrobenchFlags,
        dispatch: &GlmDsaDispatchSummary,
        timing: &TimingDistributionSummary,
        optimized_probe: Option<&MicrobenchCaseSummary>,
    ) -> Self {
        let route_fusion_active = dispatch.topk_moe_route_fused_records > 0;
        let route_fusion_encode_candidate_records =
            dispatch.topk_moe_route_encode_candidate_records;
        let route_fusion_encode_skipped_candidate_records =
            dispatch.topk_moe_route_encode_skipped_candidate_records;
        let optimized_probe_route_fusion_active = optimized_probe
            .map(|probe| probe.metal_dispatch_summary.topk_moe_route_fused_records > 0);
        let optimized_probe_route_fusion_encode_candidate_records = optimized_probe.map(|probe| {
            probe
                .metal_dispatch_summary
                .topk_moe_route_encode_candidate_records
        });
        let optimized_probe_route_fusion_encode_skipped_candidate_records =
            optimized_probe.map(|probe| {
                probe
                    .metal_dispatch_summary
                    .topk_moe_route_encode_skipped_candidate_records
            });
        let diagnostic_timing_may_disable_route_fusion =
            flags.op_timing && matches!(optimized_probe_route_fusion_active, Some(true));
        let diagnostic_mean_ms = timing.mean_ms;
        let optimized_probe_mean_ms =
            optimized_probe.and_then(|probe| probe.timing_summary.mean_ms);
        let diagnostic_slowdown_vs_optimized_probe =
            match (diagnostic_mean_ms, optimized_probe_mean_ms) {
                (Some(diagnostic), Some(optimized)) if optimized > f64::EPSILON => {
                    Some(diagnostic / optimized)
                }
                _ => None,
            };
        Self {
            op_timing_enabled: flags.op_timing,
            metal_dispatch_log_enabled: flags.metal_dispatch_log,
            route_fusion_active,
            route_fusion_encode_candidate_records,
            route_fusion_encode_skipped_candidate_records,
            optimized_probe_route_fusion_active,
            optimized_probe_route_fusion_encode_candidate_records,
            optimized_probe_route_fusion_encode_skipped_candidate_records,
            diagnostic_timing_may_disable_route_fusion,
            diagnostic_mean_ms,
            optimized_probe_mean_ms,
            diagnostic_slowdown_vs_optimized_probe,
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
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    timing_summary: TimingDistributionSummary,
    #[serde(skip_serializing_if = "GlmDsaDispatchSummary::is_empty")]
    metal_dispatch_summary: GlmDsaDispatchSummary,
    #[serde(skip_serializing_if = "GlmDsaOpTimingSummary::is_empty")]
    op_timing_summary: GlmDsaOpTimingSummary,
    #[serde(skip_serializing_if = "RoutedMoeTimingSummary::is_empty")]
    routed_moe_timing_summary: RoutedMoeTimingSummary,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    op_timing_records: Vec<TimingRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    group_timing_records: Vec<TimingGroupRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    hot_tensor_records: Vec<HotTensorRecord>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_fusion_guard_checks_optimized_probe_when_present() {
        let candidate = case_summary("candidate", 4, 4, 0);
        let optimized_probe = case_summary("optimized_dispatch_probe", 4, 0, 4);

        let guard = build_route_fusion_guard(&candidate, Some(&optimized_probe));

        assert!(guard.passed);
        assert_eq!(guard.checked_case, "optimized_dispatch_probe");
        assert_eq!(guard.encode_candidate_records, 4);
        assert_eq!(guard.encode_skipped_candidate_records, 0);
        assert_eq!(guard.fused_dispatch_records, 4);
    }

    #[test]
    fn route_fusion_guard_fails_without_fused_dispatches() {
        let candidate = case_summary("candidate", 4, 4, 0);

        let guard = build_route_fusion_guard(&candidate, None);

        assert!(!guard.passed);
        assert_eq!(guard.checked_case, "candidate");
        assert_eq!(guard.encode_candidate_records, 4);
        assert_eq!(guard.encode_skipped_candidate_records, 4);
        assert_eq!(guard.fused_dispatch_records, 0);
    }

    #[test]
    fn optimized_dispatch_probe_runs_for_diagnostic_reports() {
        let flags = MicrobenchFlags {
            direct_sparse_attn: true,
            direct_sparse_prefill: false,
            fused_sparse_mask: true,
            parallel_lightning_indexer: true,
            op_timing: true,
            metal_dispatch_log: true,
            metal_topk_moe_route_fusion: false,
        };

        assert!(should_run_optimized_dispatch_probe(flags));
    }

    #[test]
    fn parses_real_top_k_chain_sources() {
        assert_eq!(
            parse_real_top_k_chain_sources(" 30, 60 ,,").unwrap(),
            vec![30, 60]
        );
    }

    #[test]
    fn rejects_invalid_real_top_k_chain_source() {
        let error = parse_real_top_k_chain_sources("30, nope")
            .unwrap_err()
            .to_string();
        assert!(error.contains("SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CHAIN_SOURCES"));
    }

    #[test]
    fn parses_real_top_k_max_source_bytes() {
        assert_eq!(
            parse_real_top_k_max_source_bytes("").unwrap(),
            Some(DEFAULT_REAL_TOP_K_MAX_SOURCE_BYTES)
        );
        assert_eq!(parse_real_top_k_max_source_bytes("off").unwrap(), None);
        assert_eq!(parse_real_top_k_max_source_bytes("0").unwrap(), None);
        assert_eq!(parse_real_top_k_max_source_bytes("123").unwrap(), Some(123));
    }

    #[test]
    fn rejects_oversized_real_top_k_source_span() {
        let parts = vec![test_package_part(70), test_package_part(40)];

        let error = guard_real_top_k_source_size_with_limit(&parts, 100)
            .unwrap_err()
            .to_string();

        assert!(error.contains("real top-k source span selects 110 bytes"));
        assert!(error.contains("SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CHAIN_SOURCES"));
    }

    #[test]
    fn accepts_real_top_k_source_span_under_limit() {
        let parts = vec![test_package_part(70), test_package_part(30)];

        guard_real_top_k_source_size_with_limit(&parts, 100).unwrap();
    }

    fn test_package_part(artifact_bytes: u64) -> PackagePart {
        PackagePart {
            role: "layer".to_string(),
            layer_index: Some(0),
            path: PathBuf::from("layers/layer-000.gguf"),
            sha256: "test".to_string(),
            artifact_bytes,
        }
    }

    fn case_summary(
        label: &'static str,
        encode_candidate_records: usize,
        encode_skipped_candidate_records: usize,
        fused_dispatch_records: usize,
    ) -> MicrobenchCaseSummary {
        let dispatch = GlmDsaDispatchSummary {
            records: encode_candidate_records + fused_dispatch_records,
            topk_moe_route_encode_candidate_records: encode_candidate_records,
            topk_moe_route_encode_fused_candidate_records: encode_candidate_records
                - encode_skipped_candidate_records,
            topk_moe_route_encode_skipped_candidate_records: encode_skipped_candidate_records,
            topk_moe_route_fused_records: fused_dispatch_records,
            ..GlmDsaDispatchSummary::default()
        };
        MicrobenchCaseSummary {
            label,
            flags: MicrobenchFlags {
                direct_sparse_attn: true,
                direct_sparse_prefill: true,
                fused_sparse_mask: true,
                parallel_lightning_indexer: false,
                op_timing: false,
                metal_dispatch_log: true,
                metal_topk_moe_route_fusion: false,
            },
            n_gpu_layers: -1,
            native_log_path: None,
            direct_sparse_decision_summary: DirectSparseDecisionSummary::default(),
            timing_summary: TimingDistributionSummary::default(),
            metal_dispatch_summary: dispatch,
            op_timing_summary: GlmDsaOpTimingSummary::default(),
            routed_moe_timing_summary: RoutedMoeTimingSummary::default(),
            direct_sparse_decision_records: Vec::new(),
            metal_dispatch_records: Vec::new(),
            op_timing_records: Vec::new(),
            group_timing_records: Vec::new(),
            hot_tensor_records: Vec::new(),
            timings: Vec::new(),
        }
    }
}
