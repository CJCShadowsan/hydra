use std::{
    fs::{self, File},
    io::{BufReader, Cursor, Read},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
    activation_frame_flags_from_state_flags, activation_state_flags_from_frame_flags,
    encode_f32_activation_payload_with_state_flags, read_stage_message, state_flags,
    write_stage_message,
};
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
const ENV_STAGE_WIRE_ROUNDTRIP: &str = "SKIPPY_BENCH_GLM_DSA_STAGE_WIRE_ROUNDTRIP";
const DEFAULT_SYNTHETIC_TOP_K_WIDTH: usize = 256;
const DEFAULT_REAL_TOP_K_MAX_SOURCE_BYTES: u64 = 110 * 1024 * 1024 * 1024;
const GGUF_TENSOR_NAME_SCAN_CHUNK_BYTES: usize = 1024 * 1024;
const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;
const INPUT_FRAME_CACHE_MAGIC: &[u8; 16] = b"SKPGLMDSAFRM1\0\0\0";

pub fn glm_dsa_layer_microbench(args: GlmDsaLayerMicrobenchArgs) -> Result<()> {
    validate_args(&args)?;

    let selected = select_layer_package_parts(&package_request(&args))
        .context("select GLM-DSA layer package parts")?;
    let runtime_config = runtime_config(&args)?;
    let token_ids = vec![1_i32; args.tokens];
    let positions = positions(args.position_start, args.tokens)?;
    let flags = MicrobenchFlags::from_args(&args);
    let indexshare_policy = IndexSharePolicy::from_args_and_env(&args);
    let artifact_layer_role = artifact_layer_role_report(
        &selected.selected_parts,
        &selected.absolute_paths,
        args.layer_start,
    )
    .context("derive GLM-DSA artifact layer role")?;
    let input = prepare_input_activation(&args, &token_ids, &positions, flags)?;
    let stage_wire_roundtrip =
        maybe_stage_wire_roundtrip(&args, &input.frame, &token_ids, &positions)
            .context("round-trip GLM-DSA activation through Skippy stage wire")?;
    let input_frame = stage_wire_roundtrip
        .as_ref()
        .map_or(&input.frame, |roundtrip| &roundtrip.frame);
    let comparison = if args.compare_dense_fallback {
        Some(run_dense_fallback_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            input_frame,
            &token_ids,
            &positions,
            flags,
        )?)
    } else if args.compare_cpu_direct_sparse {
        Some(run_cpu_direct_sparse_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            input_frame,
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
                input_frame,
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
    let input_contract = activation_contract_report(&args, input_frame)?;
    let execution_contract = execution_contract_report(
        &args,
        &input.report,
        &input_contract,
        &indexshare_policy,
        artifact_layer_role,
    );
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
        position_start: args.position_start,
        warmup: args.warmup,
        iterations: args.iterations,
        n_gpu_layers: args.n_gpu_layers,
        n_batch: runtime_config.n_batch,
        n_ubatch: runtime_config.n_ubatch,
        flags: case.flags,
        indexshare_policy,
        input_source: input.report,
        selected_parts: selected
            .selected_parts
            .iter()
            .map(package_part_summary)
            .collect(),
        input_payload_bytes: input_frame.payload.len(),
        input_contract,
        stage_wire_roundtrip: stage_wire_roundtrip.map(|roundtrip| roundtrip.report),
        execution_contract,
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
    configure_env_flags(args, flags);
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
    if args.position_start < 0 {
        bail!("position_start must be greater than or equal to zero");
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

fn configure_env_flags(args: &GlmDsaLayerMicrobenchArgs, flags: MicrobenchFlags) {
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
    set_optional_env(
        "LLAMA_GLM_DSA_INDEXSHARE_FREQ",
        IndexSharePolicy::from_args_and_env(args)
            .freq
            .map(|freq| freq.to_string()),
    );
    set_optional_env(
        "LLAMA_GLM_DSA_INDEXSHARE_PATTERN",
        IndexSharePolicy::from_args_and_env(args).pattern,
    );
}

fn set_env_flag(name: &str, enabled: bool) {
    // This command is single-threaded and sets native runtime flags before opening llama.cpp.
    unsafe {
        std::env::set_var(name, if enabled { "1" } else { "0" });
    }
}

fn set_optional_env(name: &str, value: Option<String>) {
    // This command is single-threaded and sets native runtime flags before opening llama.cpp.
    unsafe {
        if let Some(value) = value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
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
        generated.report,
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
            report: GeneratedTopKReport {
                selected_parts: Vec::new(),
                source_start_artifact_role: None,
                cache_path,
                cache_hit: true,
            },
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
    let source_start_artifact_role = artifact_layer_role_report(
        &source_selected.selected_parts,
        &source_selected.absolute_paths,
        source_layer_start,
    )
    .context("derive GLM-DSA real top-k source artifact role")?;
    guard_real_top_k_source_start(&source_input, &source_start_artifact_role)
        .context("check GLM-DSA real top-k source start")?;
    let source_config = runtime_config_for_range(args, source_layer_start, source_layer_end)?;
    let source_flags = MicrobenchFlags {
        direct_sparse_attn: false,
        direct_sparse_prefill: false,
        ..flags
    };
    configure_env_flags(args, source_flags);
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
        report: GeneratedTopKReport {
            selected_parts: source_selected
                .selected_parts
                .iter()
                .map(package_part_summary)
                .collect(),
            source_start_artifact_role: Some(source_start_artifact_role),
            cache_path,
            cache_hit: false,
        },
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

fn guard_real_top_k_source_start(
    source_input: &ActivationFrame,
    artifact_role: &ArtifactLayerRoleReport,
) -> Result<()> {
    if (source_input.desc.flags & ACTIVATION_FLAG_GLM_DSA_TOP_K) != 0 {
        return Ok(());
    }
    if artifact_role.can_produce_top_k == Some(true) {
        return Ok(());
    }
    let source_layer_start = artifact_role.layer_index;
    bail!(
        "real top-k source layer {source_layer_start} has no input top-k sideband and cannot produce top-k from artifact role {:?}; choose a producer layer or set {ENV_REAL_TOP_K_CHAIN_SOURCES}",
        artifact_role.role
    )
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
    generated: GeneratedTopKReport,
) -> Result<PreparedInputActivation> {
    let report = InputSourceReport::RealTopK {
        layer_start: source_layer_start,
        layer_end: source_layer_end,
        output_flags: frame.desc.flags,
        output_payload_bytes: frame.payload.len(),
        sideband: Box::new(sideband_contract_report(
            args,
            &frame,
            Some(source_layer_start),
            source_layer_end,
            args.layer_start,
        )?),
        cache_path: generated.cache_path,
        cache_hit: generated.cache_hit,
        selected_parts: generated.selected_parts,
        source_start_artifact_role: generated.source_start_artifact_role,
    };
    Ok(PreparedInputActivation { frame, report })
}

fn maybe_stage_wire_roundtrip(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
) -> Result<Option<StageWireRoundTrip>> {
    if !env_flag_enabled(ENV_STAGE_WIRE_ROUNDTRIP) {
        return Ok(None);
    }
    stage_wire_roundtrip(args, frame, token_ids, positions).map(Some)
}

fn stage_wire_roundtrip(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
) -> Result<StageWireRoundTrip> {
    let hidden_bytes = hidden_payload_bytes(args)?;
    if frame.payload.len() < hidden_bytes {
        bail!(
            "activation payload has {} bytes, expected at least {hidden_bytes}",
            frame.payload.len()
        );
    }
    let token_count = i32::try_from(args.tokens).context("tokens exceeds i32")?;
    let activation_width =
        i32::try_from(args.activation_width).context("activation_width exceeds i32")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, WireActivationDType::F32);
    state.prompt_token_count = args.position_start.max(0);
    state.decode_step = args.tokens.saturating_sub(1).try_into().unwrap_or(i32::MAX);
    state.current_token = token_ids.last().copied().unwrap_or_default();
    state.source_stage_index = frame.desc.producer_stage_index.max(0);
    state.flags |= activation_state_flags_from_frame_flags(frame.desc.flags);
    let activation = encode_f32_activation_payload_with_state_flags(
        WireActivationDType::F32,
        token_count,
        activation_width,
        &frame.payload[..hidden_bytes],
        state.flags & !state_flags::GLM_DSA_TOP_K_SIDEBAND,
    )
    .context("encode activation payload for stage wire")?;
    let raw_bytes = if (state.flags & state_flags::GLM_DSA_TOP_K_SIDEBAND) != 0 {
        frame.payload[hidden_bytes..].to_vec()
    } else {
        Vec::new()
    };
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: args.position_start,
        token_count,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: token_ids.to_vec(),
        positions: positions.to_vec(),
        activation,
        raw_bytes,
    };
    let estimated_wire_bytes = message.estimated_wire_bytes();
    let mut encoded = Vec::with_capacity(estimated_wire_bytes);
    write_stage_message(&mut encoded, &message, WireActivationDType::F32)
        .context("encode Skippy stage wire message")?;
    let mut decoded = read_stage_message(Cursor::new(&encoded), activation_width)
        .context("decode Skippy stage wire message")?;
    let mut payload = decoded
        .take_activation_f32_payload(activation_width)
        .context("decode stage wire activation payload")?;
    let decoded_sideband_bytes = if (decoded.state.flags & state_flags::GLM_DSA_TOP_K_SIDEBAND) != 0
    {
        if !decoded
            .raw_bytes
            .len()
            .is_multiple_of(std::mem::size_of::<i32>())
        {
            bail!("decoded GLM-DSA top-k sideband payload is not i32-aligned");
        }
        payload.extend_from_slice(&decoded.raw_bytes);
        decoded.raw_bytes.len()
    } else {
        0
    };
    let mut desc = frame.desc;
    desc.producer_stage_index = decoded.state.source_stage_index;
    desc.token_count = decoded
        .token_count
        .try_into()
        .context("decoded token_count is negative")?;
    desc.sequence_count = if decoded.token_count > 0 { 1 } else { 0 };
    desc.payload_bytes = payload.len() as u64;
    desc.flags = activation_frame_flags_from_state_flags(decoded.state.flags);
    let decoded_frame = ActivationFrame { desc, payload };
    let payload_bytes_match = decoded_frame.payload == frame.payload;
    let flags_match = decoded_frame.desc.flags == frame.desc.flags;
    let sideband_bytes_match = decoded_sideband_bytes == message.raw_bytes.len();
    let sideband_checksum_match = fnv1a64(&decoded.raw_bytes) == fnv1a64(&message.raw_bytes);
    let passed =
        payload_bytes_match && flags_match && sideband_bytes_match && sideband_checksum_match;
    if !passed {
        bail!("Skippy stage wire round-trip did not preserve GLM-DSA activation payload");
    }
    let report = StageWireRoundTripReport {
        kind: format!("{:?}", message.kind),
        wire_dtype: format!("{:?}", WireActivationDType::F32),
        state_flags: decoded.state.flags,
        activation_flag_bits: decoded_frame.desc.flags,
        token_count: decoded.token_count,
        position_start: decoded.pos_start,
        token_sideband_count: decoded.tokens.len(),
        position_sideband_count: decoded.positions.len(),
        hidden_activation_bytes: hidden_bytes,
        raw_activation_wire_bytes: message.activation.len(),
        top_k_sideband_bytes: message.raw_bytes.len(),
        top_k_sideband_i32_count: message.raw_bytes.len() / std::mem::size_of::<i32>(),
        estimated_wire_bytes,
        encoded_wire_bytes: encoded.len(),
        decoded_payload_bytes: decoded_frame.payload.len(),
        decoded_payload_checksum: fnv1a64(&decoded_frame.payload),
        decoded_sideband_checksum: fnv1a64(&decoded.raw_bytes),
        payload_bytes_match,
        flags_match,
        sideband_bytes_match,
        sideband_checksum_match,
        passed,
    };
    Ok(StageWireRoundTrip {
        frame: decoded_frame,
        report,
    })
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
        "real-topk-src{}-dst{}-tok{}-pos{}-ctx{}-act{}-ngpu{}-nb{}-nub{}-pi{}.skippy-frame",
        source_layer_start,
        source_layer_end,
        args.tokens,
        args.position_start,
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

fn activation_contract_report(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: &ActivationFrame,
) -> Result<ActivationContractReport> {
    Ok(ActivationContractReport {
        dtype: format!("{:?}", frame.desc.dtype),
        layout: format!("{:?}", frame.desc.layout),
        producer_stage_index: frame.desc.producer_stage_index,
        layer_start: frame.desc.layer_start,
        layer_end: frame.desc.layer_end,
        consumer_layer_start: args.layer_start,
        consumer_layer_end: args.layer_end,
        token_count: frame.desc.token_count,
        sequence_count: frame.desc.sequence_count,
        position_start: args.position_start,
        position_end: position_end(args.position_start, args.tokens)?,
        payload_bytes: frame.payload.len(),
        descriptor_payload_bytes: frame.desc.payload_bytes,
        flags: frame.desc.flags,
        sideband: sideband_contract_report(
            args,
            frame,
            u32::try_from(frame.desc.layer_start).ok(),
            u32::try_from(frame.desc.layer_end).unwrap_or(args.layer_start),
            args.layer_start,
        )?,
    })
}

fn position_end(position_start: i32, tokens: usize) -> Result<i32> {
    let last_offset = tokens
        .checked_sub(1)
        .context("tokens must be greater than zero")?;
    let last_offset = i32::try_from(last_offset).context("position offset exceeds i32")?;
    position_start
        .checked_add(last_offset)
        .context("position exceeds i32")
}

fn sideband_contract_report(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: &ActivationFrame,
    source_layer_start: Option<u32>,
    source_layer_end: u32,
    consumer_layer_start: u32,
) -> Result<SidebandContractReport> {
    let hidden_bytes = hidden_payload_bytes(args)?;
    if frame.payload.len() < hidden_bytes {
        bail!(
            "activation payload has {} bytes, expected at least {hidden_bytes}",
            frame.payload.len()
        );
    }
    let sideband = &frame.payload[hidden_bytes..];
    let values = decode_i32_sideband(sideband)?;
    Ok(SidebandContractReport {
        present: (frame.desc.flags & ACTIVATION_FLAG_GLM_DSA_TOP_K) != 0,
        source_layer_start,
        source_layer_end,
        consumer_layer_start,
        position_start: args.position_start,
        position_end: position_end(args.position_start, args.tokens)?,
        hidden_bytes,
        sideband_bytes: sideband.len(),
        sideband_i32_count: values.len(),
        checksum: fnv1a64(sideband),
        min_index: values.iter().copied().min(),
        max_index: values.iter().copied().max(),
        unique_index_count: unique_i32_count(&values),
        sorted_ascending: values.windows(2).all(|pair| pair[0] <= pair[1]),
        negative_index_count: values.iter().filter(|value| **value < 0).count(),
        first_indices: values.iter().take(16).copied().collect(),
        last_indices: last_i32_values(&values, 16),
    })
}

fn decode_i32_sideband(sideband: &[u8]) -> Result<Vec<i32>> {
    if !sideband.len().is_multiple_of(std::mem::size_of::<i32>()) {
        bail!("GLM-DSA sideband payload is not i32-aligned");
    }
    sideband
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|chunk| {
            let bytes = chunk
                .try_into()
                .context("read GLM-DSA sideband i32 value")?;
            Ok(i32::from_ne_bytes(bytes))
        })
        .collect()
}

fn unique_i32_count(values: &[i32]) -> usize {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.dedup();
    values.len()
}

fn last_i32_values(values: &[i32], count: usize) -> Vec<i32> {
    let start = values.len().saturating_sub(count);
    values[start..].to_vec()
}

fn fnv1a64(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn artifact_layer_role_report(
    selected_parts: &[PackagePart],
    absolute_paths: &[PathBuf],
    layer_index: u32,
) -> Result<ArtifactLayerRoleReport> {
    let indexer_tensor_prefix = format!("blk.{layer_index}.indexer.");
    let Some((part, path)) = selected_parts
        .iter()
        .zip(absolute_paths.iter())
        .find(|(part, _)| part.role == "layer" && part.layer_index == Some(layer_index))
    else {
        return Ok(ArtifactLayerRoleReport {
            layer_index,
            role: None,
            basis: ArtifactLayerRoleBasis::NoMatchingLayerPart,
            part_path: None,
            indexer_tensor_prefix,
            can_produce_top_k: None,
        });
    };
    let can_produce_top_k = gguf_has_tensor_name_prefix(path, &indexer_tensor_prefix)
        .with_context(|| format!("scan {} for GLM-DSA indexer tensor names", path.display()))?;
    Ok(ArtifactLayerRoleReport {
        layer_index,
        role: Some(if can_produce_top_k {
            IndexShareRole::FullProducer
        } else {
            IndexShareRole::SharedConsumer
        }),
        basis: ArtifactLayerRoleBasis::TensorNameScan,
        part_path: Some(part.path.clone()),
        indexer_tensor_prefix,
        can_produce_top_k: Some(can_produce_top_k),
    })
}

fn gguf_has_tensor_name_prefix(path: &Path, prefix: &str) -> Result<bool> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(GGUF_TENSOR_NAME_SCAN_CHUNK_BYTES, file);
    let mut magic = [0_u8; 4];
    reader
        .read_exact(&mut magic)
        .with_context(|| format!("read GGUF magic from {}", path.display()))?;
    if &magic != GGUF_MAGIC {
        return file_contains_bytes(path, prefix.as_bytes());
    }
    let _version = read_u32_le(&mut reader)?;
    let tensor_count = read_u64_le(&mut reader)?;
    let metadata_count = read_u64_le(&mut reader)?;
    for _ in 0..metadata_count {
        skip_gguf_string(&mut reader)?;
        let value_type = read_u32_le(&mut reader)?;
        skip_gguf_value(&mut reader, value_type)?;
    }
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut reader)?;
        if name.starts_with(prefix) {
            return Ok(true);
        }
        let n_dims = read_u32_le(&mut reader)?;
        for _ in 0..n_dims {
            let _ = read_u64_le(&mut reader)?;
        }
        let _tensor_type = read_u32_le(&mut reader)?;
        let _offset = read_u64_le(&mut reader)?;
    }
    Ok(false)
}

fn read_u32_le(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).context("read GGUF u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_le(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).context("read GGUF u64")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_gguf_string(reader: &mut impl Read) -> Result<String> {
    let len = usize::try_from(read_u64_le(reader)?).context("GGUF string length exceeds usize")?;
    let mut bytes = vec![0_u8; len];
    reader
        .read_exact(&mut bytes)
        .context("read GGUF string bytes")?;
    String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
}

fn skip_gguf_string(reader: &mut impl Read) -> Result<()> {
    let len = read_u64_le(reader)?;
    skip_exact(reader, len)
}

fn skip_gguf_value(reader: &mut impl Read, value_type: u32) -> Result<()> {
    match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => skip_exact(reader, 1),
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => skip_exact(reader, 2),
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => skip_exact(reader, 4),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => skip_exact(reader, 8),
        GGUF_TYPE_STRING => skip_gguf_string(reader),
        GGUF_TYPE_ARRAY => {
            let element_type = read_u32_le(reader)?;
            let len = read_u64_le(reader)?;
            skip_gguf_array(reader, element_type, len)
        }
        _ => bail!("unsupported GGUF metadata value type {value_type}"),
    }
}

fn skip_gguf_array(reader: &mut impl Read, element_type: u32, len: u64) -> Result<()> {
    if let Some(width) = gguf_scalar_width(element_type) {
        return skip_exact(
            reader,
            len.checked_mul(width)
                .context("GGUF array byte count overflow")?,
        );
    }
    if element_type == GGUF_TYPE_STRING {
        for _ in 0..len {
            skip_gguf_string(reader)?;
        }
        return Ok(());
    }
    bail!("unsupported GGUF array element type {element_type}")
}

fn gguf_scalar_width(value_type: u32) -> Option<u64> {
    match value_type {
        GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => Some(1),
        GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => Some(2),
        GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => Some(4),
        GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn skip_exact(reader: &mut impl Read, len: u64) -> Result<()> {
    let mut limited = reader.take(len);
    std::io::copy(&mut limited, &mut std::io::sink()).context("skip GGUF bytes")?;
    if limited.limit() != 0 {
        bail!("unexpected EOF while skipping GGUF bytes");
    }
    Ok(())
}

fn file_contains_bytes(path: &Path, needle: &[u8]) -> Result<bool> {
    if needle.is_empty() {
        return Ok(true);
    }
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(GGUF_TENSOR_NAME_SCAN_CHUNK_BYTES, file);
    let mut chunk = vec![0_u8; GGUF_TENSOR_NAME_SCAN_CHUNK_BYTES];
    let mut carry = Vec::new();
    loop {
        let read = reader
            .read(&mut chunk)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            return Ok(false);
        }
        let mut search = Vec::with_capacity(carry.len() + read);
        search.extend_from_slice(&carry);
        search.extend_from_slice(&chunk[..read]);
        if contains_subslice(&search, needle) {
            return Ok(true);
        }
        let keep = needle.len().saturating_sub(1).min(search.len());
        carry.clear();
        carry.extend_from_slice(&search[search.len() - keep..]);
    }
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|candidate| candidate == needle)
}

fn execution_contract_report(
    args: &GlmDsaLayerMicrobenchArgs,
    input: &InputSourceReport,
    activation: &ActivationContractReport,
    policy: &IndexSharePolicy,
    artifact_layer_role: ArtifactLayerRoleReport,
) -> ExecutionContractReport {
    let policy_layer_role = indexshare_layer_role(args.layer_start, policy);
    let effective_layer_role = effective_layer_role(&policy_layer_role, &artifact_layer_role);
    let policy_artifact_compatible =
        policy_artifact_compatible(&policy_layer_role, &artifact_layer_role);
    let sideband_source = sideband_source_report(input);
    let sideband_required = matches!(effective_layer_role.role, IndexShareRole::SharedConsumer);
    let sideband_present = activation.sideband.sideband_bytes > 0 && activation.sideband.present;
    let proof_kind = proof_kind(
        effective_layer_role.role,
        sideband_source.kind,
        sideband_present,
    );
    ExecutionContractReport {
        proof_kind,
        policy_layer_role,
        artifact_layer_role,
        effective_layer_role,
        policy_artifact_compatible,
        sideband_source,
        sideband_required,
        sideband_present,
        sideband_contract_satisfied: !sideband_required || sideband_present,
        native_consumer_execution_proven: matches!(
            (proof_kind, sideband_present),
            (ExecutionProofKind::SharedConsumerWithRealTopK, true)
        ),
    }
}

fn effective_layer_role(
    policy_role: &IndexShareLayerRole,
    artifact_role: &ArtifactLayerRoleReport,
) -> EffectiveLayerRoleReport {
    if artifact_role.can_produce_top_k == Some(false)
        && matches!(policy_role.role, IndexShareRole::FullProducer)
    {
        return EffectiveLayerRoleReport {
            role: IndexShareRole::SharedConsumer,
            basis: EffectiveLayerRoleBasis::ArtifactNoIndexer,
        };
    }
    EffectiveLayerRoleReport {
        role: policy_role.role,
        basis: EffectiveLayerRoleBasis::Policy,
    }
}

fn policy_artifact_compatible(
    policy_role: &IndexShareLayerRole,
    artifact_role: &ArtifactLayerRoleReport,
) -> Option<bool> {
    artifact_role
        .can_produce_top_k
        .map(|can_produce| can_produce || !matches!(policy_role.role, IndexShareRole::FullProducer))
}

fn proof_kind(
    target_role: IndexShareRole,
    sideband_kind: SidebandSourceKind,
    sideband_present: bool,
) -> ExecutionProofKind {
    match (target_role, sideband_kind, sideband_present) {
        (IndexShareRole::SharedConsumer, SidebandSourceKind::RealTopK, true) => {
            ExecutionProofKind::SharedConsumerWithRealTopK
        }
        (IndexShareRole::SharedConsumer, SidebandSourceKind::SyntheticTopK, true) => {
            ExecutionProofKind::SharedConsumerWithSyntheticTopK
        }
        (IndexShareRole::SharedConsumer, _, _) => ExecutionProofKind::SharedConsumerMissingSideband,
        (IndexShareRole::FullProducer, SidebandSourceKind::None, false) => {
            ExecutionProofKind::FullProducerNoSideband
        }
        (IndexShareRole::FullProducer, SidebandSourceKind::RealTopK, true) => {
            ExecutionProofKind::FullProducerWithRealTopKInput
        }
        (IndexShareRole::FullProducer, SidebandSourceKind::SyntheticTopK, true) => {
            ExecutionProofKind::FullProducerWithSyntheticTopKInput
        }
        (IndexShareRole::FullProducer, _, _) => ExecutionProofKind::FullProducerOtherInput,
    }
}

fn indexshare_layer_role(layer_index: u32, policy: &IndexSharePolicy) -> IndexShareLayerRole {
    if let Some(pattern) = policy.pattern.as_deref()
        && let Some(role) = indexshare_pattern_role(layer_index, pattern)
    {
        return IndexShareLayerRole {
            role,
            basis: IndexShareRoleBasis::Pattern,
            freq: policy.freq,
            pattern: policy.pattern.clone(),
        };
    }
    let freq = policy.freq.unwrap_or(1).max(1);
    let role = if freq <= 1 || layer_index.is_multiple_of(freq) {
        IndexShareRole::FullProducer
    } else {
        IndexShareRole::SharedConsumer
    };
    IndexShareLayerRole {
        role,
        basis: IndexShareRoleBasis::Frequency,
        freq: Some(freq),
        pattern: policy.pattern.clone(),
    }
}

fn indexshare_pattern_role(layer_index: u32, pattern: &str) -> Option<IndexShareRole> {
    let mut current_layer = 0_u32;
    for value in pattern
        .chars()
        .filter_map(|ch| match ch.to_ascii_uppercase() {
            'F' => Some(IndexShareRole::FullProducer),
            'S' => Some(IndexShareRole::SharedConsumer),
            _ => None,
        })
    {
        if current_layer == layer_index {
            return Some(value);
        }
        current_layer = current_layer.saturating_add(1);
    }
    None
}

fn sideband_source_report(input: &InputSourceReport) -> SidebandSourceReport {
    match input {
        InputSourceReport::Synthetic { top_k_sideband } => SidebandSourceReport {
            kind: if top_k_sideband.is_some() {
                SidebandSourceKind::SyntheticTopK
            } else {
                SidebandSourceKind::None
            },
            source_layer_start: None,
            source_layer_end: None,
            top_k_width: *top_k_sideband,
        },
        InputSourceReport::RealTopK {
            layer_start,
            layer_end,
            sideband,
            ..
        } => SidebandSourceReport {
            kind: SidebandSourceKind::RealTopK,
            source_layer_start: Some(*layer_start),
            source_layer_end: Some(*layer_end),
            top_k_width: Some(sideband.sideband_i32_count),
        },
    }
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
    report: GeneratedTopKReport,
}

struct GeneratedTopKReport {
    selected_parts: Vec<PackagePartSummary>,
    source_start_artifact_role: Option<ArtifactLayerRoleReport>,
    cache_path: Option<PathBuf>,
    cache_hit: bool,
}

struct StageWireRoundTrip {
    frame: ActivationFrame,
    report: StageWireRoundTripReport,
}

fn positions(position_start: i32, tokens: usize) -> Result<Vec<i32>> {
    (0..tokens)
        .map(|offset| {
            let offset = i32::try_from(offset).context("position offset exceeds i32")?;
            position_start
                .checked_add(offset)
                .context("position exceeds i32")
        })
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
    position_start: i32,
    warmup: usize,
    iterations: usize,
    n_gpu_layers: i32,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    flags: MicrobenchFlags,
    #[serde(skip_serializing_if = "IndexSharePolicy::is_disabled")]
    indexshare_policy: IndexSharePolicy,
    input_source: InputSourceReport,
    selected_parts: Vec<PackagePartSummary>,
    input_payload_bytes: usize,
    input_contract: ActivationContractReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage_wire_roundtrip: Option<StageWireRoundTripReport>,
    execution_contract: ExecutionContractReport,
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

#[derive(Serialize)]
struct ActivationContractReport {
    dtype: String,
    layout: String,
    producer_stage_index: i32,
    layer_start: i32,
    layer_end: i32,
    consumer_layer_start: u32,
    consumer_layer_end: u32,
    token_count: u32,
    sequence_count: u32,
    position_start: i32,
    position_end: i32,
    payload_bytes: usize,
    descriptor_payload_bytes: u64,
    flags: u64,
    sideband: SidebandContractReport,
}

#[derive(Clone, Serialize)]
struct SidebandContractReport {
    present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_layer_start: Option<u32>,
    source_layer_end: u32,
    consumer_layer_start: u32,
    position_start: i32,
    position_end: i32,
    hidden_bytes: usize,
    sideband_bytes: usize,
    sideband_i32_count: usize,
    checksum: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_index: Option<i32>,
    unique_index_count: usize,
    sorted_ascending: bool,
    negative_index_count: usize,
    first_indices: Vec<i32>,
    last_indices: Vec<i32>,
}

#[derive(Serialize)]
struct StageWireRoundTripReport {
    kind: String,
    wire_dtype: String,
    state_flags: i32,
    activation_flag_bits: u64,
    token_count: i32,
    position_start: i32,
    token_sideband_count: usize,
    position_sideband_count: usize,
    hidden_activation_bytes: usize,
    raw_activation_wire_bytes: usize,
    top_k_sideband_bytes: usize,
    top_k_sideband_i32_count: usize,
    estimated_wire_bytes: usize,
    encoded_wire_bytes: usize,
    decoded_payload_bytes: usize,
    decoded_payload_checksum: String,
    decoded_sideband_checksum: String,
    payload_bytes_match: bool,
    flags_match: bool,
    sideband_bytes_match: bool,
    sideband_checksum_match: bool,
    passed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionProofKind {
    FullProducerNoSideband,
    FullProducerWithRealTopKInput,
    FullProducerWithSyntheticTopKInput,
    FullProducerOtherInput,
    SharedConsumerWithRealTopK,
    SharedConsumerWithSyntheticTopK,
    SharedConsumerMissingSideband,
}

#[derive(Serialize)]
struct ExecutionContractReport {
    proof_kind: ExecutionProofKind,
    policy_layer_role: IndexShareLayerRole,
    artifact_layer_role: ArtifactLayerRoleReport,
    effective_layer_role: EffectiveLayerRoleReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_artifact_compatible: Option<bool>,
    sideband_source: SidebandSourceReport,
    sideband_required: bool,
    sideband_present: bool,
    sideband_contract_satisfied: bool,
    native_consumer_execution_proven: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum IndexShareRole {
    FullProducer,
    SharedConsumer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum IndexShareRoleBasis {
    Pattern,
    Frequency,
}

#[derive(Serialize)]
struct IndexShareLayerRole {
    role: IndexShareRole,
    basis: IndexShareRoleBasis,
    #[serde(skip_serializing_if = "Option::is_none")]
    freq: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pattern: Option<String>,
}

#[derive(Clone, Serialize)]
struct ArtifactLayerRoleReport {
    layer_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<IndexShareRole>,
    basis: ArtifactLayerRoleBasis,
    #[serde(skip_serializing_if = "Option::is_none")]
    part_path: Option<PathBuf>,
    indexer_tensor_prefix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    can_produce_top_k: Option<bool>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ArtifactLayerRoleBasis {
    TensorNameScan,
    NoMatchingLayerPart,
}

#[derive(Clone, Copy, Serialize)]
struct EffectiveLayerRoleReport {
    role: IndexShareRole,
    basis: EffectiveLayerRoleBasis,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum EffectiveLayerRoleBasis {
    Policy,
    ArtifactNoIndexer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SidebandSourceKind {
    None,
    SyntheticTopK,
    RealTopK,
}

#[derive(Serialize)]
struct SidebandSourceReport {
    kind: SidebandSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_layer_start: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_layer_end: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k_width: Option<usize>,
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
        sideband: Box<SidebandContractReport>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_path: Option<PathBuf>,
        cache_hit: bool,
        selected_parts: Vec<PackagePartSummary>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source_start_artifact_role: Option<ArtifactLayerRoleReport>,
    },
}

#[derive(Clone, Serialize)]
struct IndexSharePolicy {
    #[serde(skip_serializing_if = "Option::is_none")]
    freq: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pattern: Option<String>,
}

impl IndexSharePolicy {
    fn from_args_and_env(args: &GlmDsaLayerMicrobenchArgs) -> Self {
        let freq = args
            .indexshare_freq
            .or_else(|| parse_env_u32("LLAMA_GLM_DSA_INDEXSHARE_FREQ"));
        let pattern = args.indexshare_pattern.clone().or_else(|| {
            std::env::var("LLAMA_GLM_DSA_INDEXSHARE_PATTERN")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });
        Self { freq, pattern }
    }

    fn is_disabled(&self) -> bool {
        self.freq.is_none() && self.pattern.is_none()
    }
}

fn parse_env_u32(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
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

    #[test]
    fn sideband_contract_reports_index_stats() {
        let mut args = test_args();
        args.tokens = 1;
        args.position_start = 255;
        let frame = synthetic_activation_frame_for_layer(
            &args,
            args.layer_start,
            Some(SyntheticTopKSideband { width: 4 }),
        )
        .unwrap();

        let report = activation_contract_report(&args, &frame).expect("activation contract report");

        assert!(report.sideband.present);
        assert_eq!(report.sideband.source_layer_start, Some(29));
        assert_eq!(report.sideband.source_layer_end, 30);
        assert_eq!(report.sideband.consumer_layer_start, 30);
        assert_eq!(report.sideband.position_start, 255);
        assert_eq!(report.sideband.position_end, 255);
        assert_eq!(report.sideband.hidden_bytes, 16);
        assert_eq!(report.sideband.sideband_bytes, 16);
        assert_eq!(report.sideband.sideband_i32_count, 4);
        assert_eq!(report.sideband.min_index, Some(0));
        assert_eq!(report.sideband.max_index, Some(3));
        assert_eq!(report.sideband.unique_index_count, 4);
        assert!(report.sideband.sorted_ascending);
        assert_eq!(report.sideband.negative_index_count, 0);
        assert_eq!(report.sideband.first_indices, vec![0, 1, 2, 3]);
        assert_eq!(report.sideband.last_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn stage_wire_roundtrip_preserves_glm_dsa_top_k_sideband() {
        let mut args = test_args();
        args.tokens = 1;
        args.position_start = 255;
        args.activation_width = 4;
        let frame = synthetic_activation_frame_for_layer(
            &args,
            args.layer_start,
            Some(SyntheticTopKSideband { width: 4 }),
        )
        .unwrap();
        let token_ids = vec![42];
        let positions = positions(args.position_start, args.tokens).unwrap();

        let roundtrip = stage_wire_roundtrip(&args, &frame, &token_ids, &positions).unwrap();

        assert_eq!(roundtrip.frame.payload, frame.payload);
        assert_eq!(roundtrip.frame.desc.flags, frame.desc.flags);
        assert!(roundtrip.report.passed);
        assert_eq!(roundtrip.report.hidden_activation_bytes, 16);
        assert_eq!(roundtrip.report.top_k_sideband_bytes, 16);
        assert_eq!(roundtrip.report.top_k_sideband_i32_count, 4);
        assert_eq!(
            roundtrip.report.state_flags & state_flags::GLM_DSA_TOP_K_SIDEBAND,
            state_flags::GLM_DSA_TOP_K_SIDEBAND
        );
    }

    #[test]
    fn positions_reject_overflow() {
        let error = positions(i32::MAX, 2).unwrap_err().to_string();

        assert!(error.contains("position exceeds i32"));
    }

    #[test]
    fn indexshare_frequency_marks_intervening_layers_shared() {
        let policy = IndexSharePolicy {
            freq: Some(4),
            pattern: None,
        };

        assert_eq!(
            indexshare_layer_role(28, &policy).role,
            IndexShareRole::FullProducer
        );
        assert_eq!(
            indexshare_layer_role(30, &policy).role,
            IndexShareRole::SharedConsumer
        );
    }

    #[test]
    fn indexshare_pattern_overrides_frequency() {
        let policy = IndexSharePolicy {
            freq: Some(1),
            pattern: Some("FSSS".to_string()),
        };

        assert_eq!(
            indexshare_layer_role(1, &policy).role,
            IndexShareRole::SharedConsumer
        );
        assert_eq!(
            indexshare_layer_role(4, &policy).role,
            IndexShareRole::FullProducer
        );
    }

    #[test]
    fn execution_contract_marks_real_top_k_shared_consumer_proof() {
        let mut args = test_args();
        args.layer_start = 30;
        args.layer_end = 31;
        args.position_start = 255;
        let frame = synthetic_activation_frame_for_layer(
            &args,
            args.layer_start,
            Some(SyntheticTopKSideband { width: 4 }),
        )
        .unwrap();
        let input_contract = activation_contract_report(&args, &frame).unwrap();
        let sideband = sideband_contract_report(&args, &frame, Some(26), 30, 30).unwrap();
        let input = InputSourceReport::RealTopK {
            layer_start: 26,
            layer_end: 30,
            output_flags: frame.desc.flags,
            output_payload_bytes: frame.payload.len(),
            sideband: Box::new(sideband),
            cache_path: None,
            cache_hit: false,
            selected_parts: Vec::new(),
            source_start_artifact_role: Some(artifact_role_for_test(26, Some(true))),
        };
        let policy = IndexSharePolicy {
            freq: Some(4),
            pattern: None,
        };
        let artifact_role = artifact_role_for_test(args.layer_start, Some(true));

        let report =
            execution_contract_report(&args, &input, &input_contract, &policy, artifact_role);

        assert_eq!(
            report.proof_kind,
            ExecutionProofKind::SharedConsumerWithRealTopK
        );
        assert_eq!(
            report.policy_layer_role.role,
            IndexShareRole::SharedConsumer
        );
        assert_eq!(
            report.effective_layer_role.role,
            IndexShareRole::SharedConsumer
        );
        assert_eq!(report.policy_artifact_compatible, Some(true));
        assert!(report.sideband_required);
        assert!(report.sideband_present);
        assert!(report.sideband_contract_satisfied);
        assert!(report.native_consumer_execution_proven);
    }

    #[test]
    fn artifact_role_detects_indexer_tensor_name() {
        let path = temp_test_file("glm-dsa-indexer-present.gguf");
        fs::write(&path, b"before blk.28.indexer.weight after").unwrap();
        let part = test_layer_package_part(28, 32);

        let role = artifact_layer_role_report(&[part], std::slice::from_ref(&path), 28).unwrap();

        assert_eq!(role.role, Some(IndexShareRole::FullProducer));
        assert_eq!(role.can_produce_top_k, Some(true));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn artifact_role_detects_missing_indexer_tensor_name() {
        let path = temp_test_file("glm-dsa-indexer-missing.gguf");
        fs::write(&path, b"before blk.28.attn_q.weight after").unwrap();
        let part = test_layer_package_part(28, 32);

        let role = artifact_layer_role_report(&[part], std::slice::from_ref(&path), 28).unwrap();

        assert_eq!(role.role, Some(IndexShareRole::SharedConsumer));
        assert_eq!(role.can_produce_top_k, Some(false));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn artifact_scan_detects_needle_across_chunk_boundary() {
        let path = temp_test_file("glm-dsa-indexer-boundary.gguf");
        let needle = b"blk.28.indexer.weight";
        let mut bytes = vec![b'a'; GGUF_TENSOR_NAME_SCAN_CHUNK_BYTES - 3];
        bytes.extend_from_slice(needle);
        fs::write(&path, bytes).unwrap();

        assert!(file_contains_bytes(&path, needle).unwrap());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn gguf_tensor_directory_detects_indexer_prefix() {
        let path = temp_test_file("glm-dsa-indexer-directory.gguf");
        fs::write(
            &path,
            minimal_gguf_with_tensor_names(&["blk.28.attn_q.weight", "blk.28.indexer.proj.weight"]),
        )
        .unwrap();

        assert!(gguf_has_tensor_name_prefix(&path, "blk.28.indexer.").unwrap());
        assert!(!gguf_has_tensor_name_prefix(&path, "blk.29.indexer.").unwrap());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn execution_contract_downgrades_policy_full_when_artifact_lacks_indexer() {
        let mut args = test_args();
        args.layer_start = 28;
        args.layer_end = 29;
        let frame = synthetic_activation_frame_for_layer(&args, args.layer_start, None).unwrap();
        let input_contract = activation_contract_report(&args, &frame).unwrap();
        let input = InputSourceReport::Synthetic {
            top_k_sideband: None,
        };
        let policy = IndexSharePolicy {
            freq: Some(4),
            pattern: None,
        };
        let artifact_role = artifact_role_for_test(args.layer_start, Some(false));

        let report =
            execution_contract_report(&args, &input, &input_contract, &policy, artifact_role);

        assert_eq!(report.policy_layer_role.role, IndexShareRole::FullProducer);
        assert_eq!(
            report.effective_layer_role.role,
            IndexShareRole::SharedConsumer
        );
        assert!(matches!(
            report.effective_layer_role.basis,
            EffectiveLayerRoleBasis::ArtifactNoIndexer
        ));
        assert_eq!(report.policy_artifact_compatible, Some(false));
        assert_eq!(
            report.proof_kind,
            ExecutionProofKind::SharedConsumerMissingSideband
        );
        assert!(report.sideband_required);
        assert!(!report.sideband_contract_satisfied);
        assert!(!report.native_consumer_execution_proven);
    }

    #[test]
    fn real_top_k_source_start_rejects_non_indexer_without_sideband_input() {
        let source_input = activation_frame_for_test(28, 0);
        let artifact_role = artifact_role_for_test(28, Some(false));

        let error = guard_real_top_k_source_start(&source_input, &artifact_role)
            .unwrap_err()
            .to_string();

        assert!(error.contains("real top-k source layer 28"));
        assert!(error.contains("cannot produce top-k"));
        assert!(error.contains("SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CHAIN_SOURCES"));
    }

    #[test]
    fn real_top_k_source_start_allows_non_indexer_with_sideband_input() {
        let source_input = activation_frame_for_test(28, ACTIVATION_FLAG_GLM_DSA_TOP_K);
        let artifact_role = artifact_role_for_test(28, Some(false));

        guard_real_top_k_source_start(&source_input, &artifact_role).unwrap();
    }

    fn artifact_role_for_test(
        layer_index: u32,
        can_produce_top_k: Option<bool>,
    ) -> ArtifactLayerRoleReport {
        ArtifactLayerRoleReport {
            layer_index,
            role: can_produce_top_k.map(|can_produce| {
                if can_produce {
                    IndexShareRole::FullProducer
                } else {
                    IndexShareRole::SharedConsumer
                }
            }),
            basis: can_produce_top_k.map_or(ArtifactLayerRoleBasis::NoMatchingLayerPart, |_| {
                ArtifactLayerRoleBasis::TensorNameScan
            }),
            part_path: can_produce_top_k
                .map(|_| PathBuf::from(format!("layers/layer-{layer_index:03}.gguf"))),
            indexer_tensor_prefix: format!("blk.{layer_index}.indexer."),
            can_produce_top_k,
        }
    }

    fn test_package_part(artifact_bytes: u64) -> PackagePart {
        test_layer_package_part(0, artifact_bytes)
    }

    fn test_layer_package_part(layer_index: u32, artifact_bytes: u64) -> PackagePart {
        PackagePart {
            role: "layer".to_string(),
            layer_index: Some(layer_index),
            path: PathBuf::from(format!("layers/layer-{layer_index:03}.gguf")),
            sha256: "test".to_string(),
            artifact_bytes,
        }
    }

    fn temp_test_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "skippy-bench-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn activation_frame_for_test(layer_end: u32, flags: u64) -> ActivationFrame {
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: -1,
                layer_start: i32::try_from(layer_end.saturating_sub(1)).unwrap(),
                layer_end: i32::try_from(layer_end).unwrap(),
                token_count: 1,
                sequence_count: 1,
                payload_bytes: 16,
                flags,
            },
            payload: vec![0; 16],
        }
    }

    fn minimal_gguf_with_tensor_names(names: &[&str]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(names.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        for name in names {
            push_gguf_string(&mut bytes, name);
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&1_u64.to_le_bytes());
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&0_u64.to_le_bytes());
        }
        bytes
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn test_args() -> GlmDsaLayerMicrobenchArgs {
        GlmDsaLayerMicrobenchArgs {
            stage_model: PathBuf::from("/tmp/glm52-layers"),
            model_id: "meshllm/GLM-5.2-Q2_K-MTP-Q8-layers".to_string(),
            layer_start: 30,
            layer_end: 31,
            ctx_size: 4096,
            activation_width: 4,
            tokens: 1,
            position_start: 0,
            iterations: 1,
            warmup: 0,
            n_gpu_layers: -1,
            n_batch: None,
            n_ubatch: None,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            direct_sparse_attn: true,
            direct_sparse_prefill: false,
            fused_sparse_mask: true,
            parallel_lightning_indexer: true,
            op_timing: true,
            metal_dispatch_log: false,
            metal_topk_moe_route_fusion: false,
            indexshare_freq: None,
            indexshare_pattern: None,
            require_optimized_route_fusion: false,
            compare_dense_fallback: false,
            compare_cpu_direct_sparse: false,
            parity_atol: 1.0e-3,
            parity_rtol: 1.0e-3,
            output: None,
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
