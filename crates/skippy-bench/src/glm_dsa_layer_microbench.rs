use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{BufReader, Cursor, Read, Write},
    path::{Path, PathBuf},
    process::Command,
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
    RuntimeActivationLayout, RuntimeConfig, RuntimeKvPageDesc, RuntimeLoadMode, StageModel,
    parse_cache_type, redirect_native_logs_to_file, restore_native_logs,
};

use crate::{
    cli::GlmDsaLayerMicrobenchArgs,
    glm_dsa_microbench_summary::{
        GlmDsaDispatchSummary, GlmDsaOpTimingSummary, RoutedMoeTimingSummary,
        TimingDistributionSummary, summarize_elapsed_ms, summarize_glm_dsa_op_timing,
        summarize_metal_dispatch, summarize_routed_moe_timing,
    },
    glm_dsa_op_report::{
        CompactFlashMaskRecord, CompactFlashPolicyRecord, ComputeBufferRecord,
        DirectSparseDecisionRecord, HotTensorRecord, IndexShareTraceSummary, MetalDispatchRecord,
        TimingGroupRecord, TimingRecord, parse_compact_flash_mask_records,
        parse_compact_flash_policy_records, parse_compute_buffer_records,
        parse_direct_sparse_decision_records, parse_hot_tensor_records,
        parse_indexshare_contract_records, parse_indexshare_trace_records,
        parse_metal_dispatch_records, parse_timing_group_records, parse_timing_records,
        summarize_indexshare_trace_records,
    },
};

const ACTIVATION_FLAG_GLM_DSA_TOP_K: u64 = 1 << 3;
const ENV_SYNTHETIC_TOP_K_SIDEBAND: &str = "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_SIDEBAND";
const ENV_SYNTHETIC_TOP_K_WIDTH: &str = "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_WIDTH";
const ENV_MALFORMED_TOP_K_BYTES: &str = "SKIPPY_BENCH_GLM_DSA_MALFORMED_TOP_K_BYTES";
const ENV_REAL_TOP_K_SOURCE_LAYER_START: &str =
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_SOURCE_LAYER_START";
const ENV_REAL_TOP_K_WARMUP_SOURCE_LAYER_START: &str =
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_WARMUP_SOURCE_LAYER_START";
const ENV_REAL_TOP_K_CACHE_DIR: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CACHE_DIR";
const ENV_REAL_TOP_K_REQUIRE_CACHE: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_REQUIRE_CACHE";
const ENV_REAL_TOP_K_CHAIN_SOURCES: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CHAIN_SOURCES";
const ENV_REAL_TOP_K_MAX_SOURCE_BYTES: &str = "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_MAX_SOURCE_BYTES";
const ENV_STAGE_WIRE_ROUNDTRIP: &str = "SKIPPY_BENCH_GLM_DSA_STAGE_WIRE_ROUNDTRIP";
const ENV_ALLOW_COMPACT_FLASH_AUTO: &str = "SKIPPY_GLM_DSA_BENCH_ALLOW_COMPACT_FLASH_AUTO";
const DEFAULT_SYNTHETIC_TOP_K_WIDTH: usize = 256;
const DEFAULT_REAL_TOP_K_MAX_SOURCE_BYTES: u64 = 110 * 1024 * 1024 * 1024;
const GGUF_TENSOR_NAME_SCAN_CHUNK_BYTES: usize = 1024 * 1024;
const SIDEBAND_DIFF_SAMPLE_LIMIT: usize = 8;
const SIDEBAND_TOKEN_DIFF_SAMPLE_LIMIT: usize = 16;
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
const GGML_TYPE_F32_ID: u32 = 0;
const GGML_TYPE_F16_ID: u32 = 1;
const GLM_DSA_F16_K_ROW_BYTES: u32 = 1152;

pub fn glm_dsa_layer_microbench(args: GlmDsaLayerMicrobenchArgs) -> Result<()> {
    validate_args(&args)?;
    let _single_run_guard = GlmDsaSingleRunGuard::acquire(&args)?;
    let mut deferred_model_drops = Vec::new();

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
    let input = prepare_input_activation(
        &args,
        &token_ids,
        &positions,
        flags,
        &mut deferred_model_drops,
    )?;
    let stage_wire_roundtrip =
        maybe_stage_wire_roundtrip(&args, &input.frame, &token_ids, &positions)
            .context("round-trip GLM-DSA activation through Skippy stage wire")?;
    let input_frame = stage_wire_roundtrip
        .as_ref()
        .map_or(&input.frame, |roundtrip| &roundtrip.frame);
    let candidate_warmup_source = native_indexshare_candidate_warmup_source(&args)?;
    let comparison = if args.compare_native_indexshare_producer_consumer {
        Some(run_native_indexshare_producer_consumer_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            input_frame,
            &token_ids,
            &positions,
            flags,
            &mut deferred_model_drops,
        )?)
    } else if args.compare_dense_fallback {
        Some(run_dense_fallback_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            input_frame,
            &token_ids,
            &positions,
            flags,
            &mut deferred_model_drops,
        )?)
    } else if args.compare_dense_flash_prefill {
        Some(run_dense_flash_prefill_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            input_frame,
            &token_ids,
            &positions,
            flags,
            &mut deferred_model_drops,
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
            &mut deferred_model_drops,
        )?)
    } else if args.compare_metal_sparse_attn_threads_baseline.is_some() {
        Some(run_metal_sparse_attn_threads_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            input_frame,
            &token_ids,
            &positions,
            flags,
            &mut deferred_model_drops,
        )?)
    } else if args.compare_moe_down_weighted_fusion
        || args.compare_moe_down_weighted_parallel
        || args.compare_moe_down_unweighted_slots
    {
        Some(run_moe_down_weighted_fusion_comparison(
            &args,
            &selected.absolute_paths,
            &runtime_config,
            input_frame,
            &token_ids,
            &positions,
            flags,
            &mut deferred_model_drops,
        )?)
    } else {
        None
    };
    let case = match comparison.as_ref() {
        Some(comparison) => comparison.candidate.as_case_summary(),
        None => {
            let case = run_microbench_case_with_warmup(
                "candidate",
                &selected.absolute_paths,
                &runtime_config,
                &args,
                args.layer_start,
                args.layer_end,
                flags,
                input_frame,
                &token_ids,
                &positions,
                false,
                candidate_warmup_source.as_ref(),
                allow_compact_flash_auto(&args),
                &mut deferred_model_drops,
            )?;
            case.as_case_summary()
        }
    };
    let effective_flags = case.flags;
    let optimized_dispatch_probe_case =
        if should_run_optimized_dispatch_probe(effective_flags, args.require_moe_motif_proof) {
            let probe_flags = MicrobenchFlags {
                op_timing: false,
                metal_dispatch_log: true,
                metal_topk_moe_route_fusion: effective_flags.metal_topk_moe_route_fusion,
                ..effective_flags
            };
            Some(run_microbench_case(
                "optimized_dispatch_probe",
                &selected.absolute_paths,
                &runtime_config,
                &args,
                args.layer_start,
                args.layer_end,
                probe_flags,
                &input.frame,
                &token_ids,
                &positions,
                true,
                &mut deferred_model_drops,
            )?)
        } else {
            None
        };
    let optimized_dispatch_probe = optimized_dispatch_probe_case
        .as_ref()
        .map(MicrobenchCase::as_case_summary);
    let optimized_dispatch_probe_parity =
        match (comparison.as_ref(), optimized_dispatch_probe_case.as_ref()) {
            (Some(comparison), Some(probe)) => Some(compare_case_outputs(
                &comparison.baseline.outputs,
                &probe.outputs,
                &args,
            )?),
            _ => None,
        };

    let compact_flash_policy_summary = summarize_compact_flash_policy(&case);
    let direct_sparse_decision_summary = summarize_direct_sparse_decisions(&case);
    let timing_summary = case.timing_summary.clone();
    let timing_breakdown = case.timing_breakdown.clone();
    let metal_dispatch_summary = case.metal_dispatch_summary.clone();
    let direct_sparse_spill_summary = summarize_direct_sparse_spill(&case.metal_dispatch_records);
    let op_timing_summary = case.op_timing_summary.clone();
    let routed_moe_timing_summary = case.routed_moe_timing_summary.clone();
    let indexshare_timing_summary = summarize_indexshare_timing(&case.group_timing_records);
    let input_contract = activation_contract_report(&args, input_frame)?;
    let execution_contract = execution_contract_report(
        &args,
        &input.report,
        &input_contract,
        &indexshare_policy,
        artifact_layer_role,
    );
    let profile_integrity = ProfileIntegrityReport::new(
        effective_flags,
        &metal_dispatch_summary,
        &timing_summary,
        optimized_dispatch_probe.as_ref(),
    );
    let representative_profile = RepresentativeProfileReport::new(
        &case,
        optimized_dispatch_probe.as_ref(),
        &profile_integrity,
    );
    let route_fusion_guard = args
        .require_optimized_route_fusion
        .then(|| build_route_fusion_guard(&case, optimized_dispatch_probe.as_ref()));
    let direct_sparse_prefill_guard = args
        .require_direct_sparse_prefill_proof
        .then(|| build_direct_sparse_prefill_guard(&case));
    let direct_sparse_decode_guard = args
        .require_direct_sparse_decode_proof
        .then(|| build_direct_sparse_decode_guard(&case));
    let partial_top_k_guard = if args.require_partial_top_k_proof {
        Some(build_partial_top_k_guard(
            &case,
            optimized_dispatch_probe.as_ref(),
            hidden_payload_bytes(&args)?,
            args.tokens,
        ))
    } else {
        None
    };
    let compact_flash_guard = args
        .require_compact_flash_proof
        .then(|| build_compact_flash_guard(&case));
    let moe_weighted_sum_guard = args
        .require_moe_weighted_sum_proof
        .then(|| build_moe_weighted_sum_guard(&case));
    let moe_motif_guard = args
        .require_moe_motif_proof
        .then(|| build_moe_motif_guard(&case, optimized_dispatch_probe.as_ref()));
    let native_indexshare_guard =
        args.require_native_indexshare_proof
            .then(|| match comparison.as_ref() {
                Some(comparison) if args.compare_native_indexshare_producer_consumer => {
                    build_native_indexshare_guard(&comparison.baseline.as_case_summary())
                }
                _ => build_native_indexshare_guard(&case),
            });
    let comparison_report = comparison.as_ref().map(MicrobenchComparison::as_report);
    let real_top_k_shared_consumer_guard =
        args.require_real_top_k_shared_consumer_proof.then(|| {
            build_real_top_k_shared_consumer_guard(
                &execution_contract,
                stage_wire_roundtrip
                    .as_ref()
                    .map(|roundtrip| &roundtrip.report),
            )
        });
    let report_kv_warmup_chunk_tokens = kv_warmup_chunk_tokens(&args);
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
        kv_warmup_tokens: args.kv_warmup_tokens,
        kv_warmup_chunk_tokens: report_kv_warmup_chunk_tokens,
        synthetic_kv_warmup: args.synthetic_kv_warmup,
        reuse_kv_warmup_checkpoint: args.reuse_kv_warmup_checkpoint,
        reuse_kv_warmup_stream: args.reuse_kv_warmup_stream,
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
        compact_flash_policy_summary,
        direct_sparse_decision_summary,
        timing_summary,
        timing_breakdown,
        metal_dispatch_summary,
        direct_sparse_spill_summary,
        op_timing_summary,
        routed_moe_timing_summary,
        indexshare_timing_summary,
        indexshare_trace_summary: case.indexshare_trace_summary.clone(),
        representative_profile,
        profile_integrity,
        route_fusion_guard,
        direct_sparse_prefill_guard,
        direct_sparse_decode_guard,
        partial_top_k_guard,
        compact_flash_guard,
        moe_weighted_sum_guard,
        moe_motif_guard,
        native_indexshare_guard,
        real_top_k_shared_consumer_guard,
        compact_flash_policy_records: case.compact_flash_policy_records,
        compact_flash_execution_policy_records: case.compact_flash_execution_policy_records,
        compact_flash_non_measured_policy_records: case.compact_flash_non_measured_policy_records,
        compact_flash_mask_records: case.compact_flash_mask_records,
        compact_flash_execution_mask_records: case.compact_flash_execution_mask_records,
        compact_flash_non_measured_mask_records: case.compact_flash_non_measured_mask_records,
        direct_sparse_decision_records: case.direct_sparse_decision_records,
        direct_sparse_execution_decision_records: case.direct_sparse_execution_decision_records,
        direct_sparse_non_measured_decision_records: case
            .direct_sparse_non_measured_decision_records,
        metal_dispatch_records: case.metal_dispatch_records,
        op_timing_records: case.op_timing_records,
        group_timing_records: case.group_timing_records,
        hot_tensor_records: case.hot_tensor_records,
        compute_buffer_records: case.compute_buffer_records,
        optimized_dispatch_probe,
        optimized_dispatch_probe_parity,
        comparison: comparison_report,
        timings: case.timings,
    };
    let parity_passed = report
        .comparison
        .as_ref()
        .is_none_or(|comparison| comparison.parity.passed)
        && report
            .optimized_dispatch_probe_parity
            .as_ref()
            .is_none_or(|parity| parity.passed);

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
    if let Some(guard) = &report.direct_sparse_prefill_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA direct sparse prefill proof failed for {}: prefill_direct={} large_prefill_direct={} sparse_mask_nodes={} dense_mask_dispatches={} dsa_nodes={} dsa_dispatches={} accepted_dispatches={} failures={}",
            guard.checked_case,
            guard.prefill_direct_decisions,
            guard.large_prefill_direct_decisions,
            guard.sparse_mask_nodes,
            guard.dense_sparse_mask_dispatches,
            guard.dsa_sparse_attn_nodes,
            guard.dsa_sparse_attn_dispatches,
            guard.accepted_prefill_dispatches,
            guard.failure_summary,
        );
    }
    if let Some(guard) = &report.direct_sparse_decode_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA direct sparse decode proof failed for {}: decode_direct={} fallback={} sparse_mask_nodes={} dense_mask_dispatches={} dsa_nodes={} dsa_dispatches={} accepted_dispatches={} failures={}",
            guard.checked_case,
            guard.decode_direct_decisions,
            guard.fallback_records,
            guard.sparse_mask_nodes,
            guard.dense_sparse_mask_dispatches,
            guard.dsa_sparse_attn_nodes,
            guard.dsa_sparse_attn_dispatches,
            guard.accepted_decode_dispatches,
            guard.failure_summary,
        );
    }
    if let Some(guard) = &report.partial_top_k_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA partial top-k proof failed for {}: partial_dispatches={} output_frames_with_expected_sideband={} failures={}",
            guard.checked_case,
            guard.partial_dsa_sparse_attn_dispatches,
            guard.output_frames_with_expected_sideband,
            guard.failure_summary,
        );
    }
    if let Some(guard) = &report.compact_flash_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA compact flash proof failed for {}: flash_glm_shape={} typed_get_rows={} compact_get_rows_fused={} promoted_get_rows={} dsa_sparse_attn={} mask_omission_records={} materialized_mla_kq_mask_records={} failures={}",
            guard.checked_case,
            guard.flash_attn_ext_glm_dsa_shape_records,
            guard.get_rows_typed_records,
            guard.dsa_compact_get_rows_fused_records,
            guard.get_rows_promote_records,
            guard.dsa_sparse_attn_records,
            guard.execution_mask_omission_records,
            guard.materialized_mla_kq_mask_records,
            guard.failure_summary,
        );
    }
    if let Some(guard) = &report.moe_weighted_sum_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA MoE weighted-sum proof failed for {}: weighted_sum={} f32x4={} failures={}",
            guard.checked_case,
            guard.moe_weighted_sum_records,
            guard.moe_weighted_sum_f32x4_records,
            guard.failure_summary,
        );
    }
    if let Some(guard) = &report.moe_motif_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA MoE motif proof failed for {}: candidates={} natural={} backend={} subgraph={} max_nodes={} failures={}",
            guard.checked_case,
            guard.motif_candidate_records,
            guard.natural_order_records,
            guard.backend_candidate_records,
            guard.subgraph_fusable_records,
            guard.max_motif_nodes,
            guard.failure_summary,
        );
    }
    if let Some(guard) = &report.native_indexshare_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA native IndexShare proof failed for {}: full_exec={} shared_exec={} top_k={} consume={} shared_missing_top_k={} failures={}",
            guard.checked_case,
            guard.full_exec_records,
            guard.shared_exec_records,
            guard.top_k_records,
            guard.consume_records,
            guard.shared_exec_missing_input_top_k,
            guard.failure_summary,
        );
    }
    if let Some(guard) = &report.real_top_k_shared_consumer_guard
        && !guard.passed
    {
        bail!(
            "GLM-DSA real top-k Shared-consumer proof failed: proof_kind={:?} sideband_source={:?} stage_wire_roundtrip_passed={:?} failures={}",
            guard.proof_kind,
            guard.sideband_source_kind,
            guard.stage_wire_roundtrip_passed,
            guard.failure_summary,
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
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchComparison> {
    let baseline_flags = MicrobenchFlags {
        direct_sparse_attn: false,
        compact_flash_attn: false,
        direct_sparse_prefill: false,
        enable_unproven_large_direct_sparse_prefill: false,
        direct_sparse_prefill_max_tokens: None,
        ..candidate_flags
    };
    let baseline = run_microbench_case_with_warmup(
        "dense_fallback",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        baseline_flags,
        input,
        token_ids,
        positions,
        true,
        None,
        false,
        deferred_model_drops,
    )?;
    let candidate = run_microbench_case(
        "candidate",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        candidate_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    Ok(MicrobenchComparison {
        baseline,
        candidate,
        parity,
        poisoned_candidate: None,
        poisoned_parity: None,
        sideband_sensitivity: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_dense_flash_prefill_comparison(
    args: &GlmDsaLayerMicrobenchArgs,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    candidate_flags: MicrobenchFlags,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchComparison> {
    let baseline_flags = MicrobenchFlags {
        direct_sparse_attn: true,
        compact_flash_attn: false,
        direct_sparse_prefill: false,
        enable_unproven_large_direct_sparse_prefill: false,
        direct_sparse_prefill_max_tokens: None,
        dense_sparse_mask_max_bytes: None,
        direct_sparse_prefill_min_kv_topk_ratio: None,
        ..candidate_flags
    };
    let candidate_flags = MicrobenchFlags {
        direct_sparse_attn: true,
        direct_sparse_prefill: true,
        ..candidate_flags
    };
    let baseline = run_microbench_case(
        "dense_flash_prefill",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        baseline_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let candidate = run_microbench_case(
        "direct_sparse_prefill",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        candidate_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    Ok(MicrobenchComparison {
        baseline,
        candidate,
        parity,
        poisoned_candidate: None,
        poisoned_parity: None,
        sideband_sensitivity: None,
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
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchComparison> {
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
        args.layer_start,
        args.layer_end,
        baseline_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let candidate = run_microbench_case(
        "candidate",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        candidate_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    Ok(MicrobenchComparison {
        baseline,
        candidate,
        parity,
        poisoned_candidate: None,
        poisoned_parity: None,
        sideband_sensitivity: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_metal_sparse_attn_threads_comparison(
    args: &GlmDsaLayerMicrobenchArgs,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    candidate_flags: MicrobenchFlags,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchComparison> {
    let baseline_threads = args
        .compare_metal_sparse_attn_threads_baseline
        .expect("validated metal sparse-attn thread baseline");
    let baseline_flags = MicrobenchFlags {
        direct_sparse_attn: true,
        direct_sparse_prefill: true,
        sparse_attn_threads: Some(baseline_threads),
        ..candidate_flags
    };
    let candidate_flags = MicrobenchFlags {
        direct_sparse_attn: true,
        direct_sparse_prefill: true,
        ..candidate_flags
    };
    let baseline = run_microbench_case(
        "metal_sparse_attn_threads_baseline",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        baseline_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let candidate = run_microbench_case(
        "candidate",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        candidate_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    Ok(MicrobenchComparison {
        baseline,
        candidate,
        parity,
        poisoned_candidate: None,
        poisoned_parity: None,
        sideband_sensitivity: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_moe_down_weighted_fusion_comparison(
    args: &GlmDsaLayerMicrobenchArgs,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    candidate_flags: MicrobenchFlags,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchComparison> {
    let baseline_flags = MicrobenchFlags {
        moe_motif_coencode: true,
        moe_down_weighted_fusion: false,
        moe_down_weighted_parallel: false,
        moe_down_unweighted_slots: false,
        ..candidate_flags
    };
    let candidate_flags = MicrobenchFlags {
        moe_motif_coencode: true,
        moe_down_weighted_fusion: args.compare_moe_down_weighted_fusion,
        moe_down_weighted_parallel: args.compare_moe_down_weighted_parallel,
        moe_down_unweighted_slots: args.compare_moe_down_unweighted_slots,
        ..candidate_flags
    };
    let baseline_label = if args.compare_moe_down_unweighted_slots {
        "moe_down_unweighted_slots_off"
    } else if args.compare_moe_down_weighted_parallel {
        "moe_down_weighted_parallel_off"
    } else {
        "moe_down_weighted_fusion_off"
    };
    let candidate_label = if args.compare_moe_down_unweighted_slots {
        "moe_down_unweighted_slots_on"
    } else if args.compare_moe_down_weighted_parallel {
        "moe_down_weighted_parallel_on"
    } else {
        "moe_down_weighted_fusion_on"
    };
    let baseline = run_microbench_case(
        baseline_label,
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        baseline_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let candidate = run_microbench_case(
        candidate_label,
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        candidate_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;
    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    Ok(MicrobenchComparison {
        baseline,
        candidate,
        parity,
        poisoned_candidate: None,
        poisoned_parity: None,
        sideband_sensitivity: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_native_indexshare_producer_consumer_comparison(
    args: &GlmDsaLayerMicrobenchArgs,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchComparison> {
    let target_layer_start = args
        .layer_start
        .checked_add(1)
        .context("native IndexShare target layer overflow")?;
    if target_layer_start >= args.layer_end {
        bail!(
            "--compare-native-indexshare-producer-consumer requires at least two layers; got {}..{}",
            args.layer_start,
            args.layer_end
        );
    }

    let trace_flags = MicrobenchFlags {
        native_indexshare_exec_log: true,
        ..flags
    };
    let baseline = run_microbench_case(
        "native_indexshare_full_span",
        selected_paths,
        runtime_config,
        args,
        args.layer_start,
        args.layer_end,
        trace_flags,
        input,
        token_ids,
        positions,
        true,
        deferred_model_drops,
    )?;

    let generated = generate_real_top_k_frame(
        args,
        token_ids,
        positions,
        trace_flags,
        args.layer_start,
        target_layer_start,
        deferred_model_drops,
    )
    .context("generate native IndexShare producer frame")?;
    let target_request = package_request_for_range(args, target_layer_start, args.layer_end);
    let target_selected = select_layer_package_parts(&target_request)
        .context("select GLM-DSA native IndexShare target layer package parts")?;
    let target_config = runtime_config_for_range(args, target_layer_start, args.layer_end)
        .context("build native IndexShare target runtime config")?;
    let warmup_source = NativeIndexShareWarmupSource {
        selected_paths: select_layer_package_parts(&package_request_for_range(
            args,
            args.layer_start,
            target_layer_start,
        ))
        .context("select GLM-DSA native IndexShare warmup source parts")?
        .absolute_paths,
        runtime_config: runtime_config_for_range(args, args.layer_start, target_layer_start)
            .context("build native IndexShare warmup source runtime config")?,
        layer_start: args.layer_start,
        layer_end: target_layer_start,
    };
    let candidate = run_microbench_case_with_warmup(
        "native_indexshare_source_target",
        &target_selected.absolute_paths,
        &target_config,
        args,
        target_layer_start,
        args.layer_end,
        trace_flags,
        &generated.frame,
        token_ids,
        positions,
        true,
        Some(&warmup_source),
        allow_compact_flash_auto(args),
        deferred_model_drops,
    )?;

    let parity = compare_case_outputs(&baseline.outputs, &candidate.outputs, args)?;
    let poisoned = poison_top_k_sideband(args, &generated.frame)
        .context("poison native IndexShare top-k sideband for sensitivity proof")?;
    let poisoned_candidate = run_microbench_case_with_warmup(
        "native_indexshare_poisoned_target",
        &target_selected.absolute_paths,
        &target_config,
        args,
        target_layer_start,
        args.layer_end,
        trace_flags,
        &poisoned.frame,
        token_ids,
        positions,
        true,
        Some(&warmup_source),
        allow_compact_flash_auto(args),
        deferred_model_drops,
    )?;
    let poisoned_parity =
        compare_case_outputs(&baseline.outputs, &poisoned_candidate.outputs, args)
            .context("compare poisoned native IndexShare target output")?;
    let sideband_sensitivity =
        build_sideband_sensitivity_report(&poisoned.report, &poisoned_parity);
    Ok(MicrobenchComparison {
        baseline,
        candidate,
        parity,
        poisoned_candidate: Some(poisoned_candidate),
        poisoned_parity: Some(poisoned_parity),
        sideband_sensitivity: Some(sideband_sensitivity),
    })
}

struct NativeIndexShareWarmupSource {
    selected_paths: Vec<PathBuf>,
    runtime_config: RuntimeConfig,
    layer_start: u32,
    layer_end: u32,
}

fn native_indexshare_candidate_warmup_source(
    args: &GlmDsaLayerMicrobenchArgs,
) -> Result<Option<NativeIndexShareWarmupSource>> {
    let Some(source_layer_start) = real_top_k_warmup_source_layer_start(args)? else {
        return Ok(None);
    };
    let target_layer_start = args.layer_start;
    let selected = select_layer_package_parts(&package_request_for_range(
        args,
        source_layer_start,
        target_layer_start,
    ))
    .context("select GLM-DSA native IndexShare candidate warmup source parts")?;
    Ok(Some(NativeIndexShareWarmupSource {
        selected_paths: selected.absolute_paths,
        runtime_config: runtime_config_for_range(args, source_layer_start, target_layer_start)
            .context("build native IndexShare candidate warmup source runtime config")?,
        layer_start: source_layer_start,
        layer_end: target_layer_start,
    }))
}

#[allow(clippy::too_many_arguments)]
fn run_microbench_case(
    label: &'static str,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
    flags: MicrobenchFlags,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    collect_outputs: bool,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchCase> {
    run_microbench_case_with_warmup(
        label,
        selected_paths,
        runtime_config,
        args,
        layer_start,
        layer_end,
        flags,
        input,
        token_ids,
        positions,
        collect_outputs,
        None,
        allow_compact_flash_auto(args),
        deferred_model_drops,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_microbench_case_with_warmup(
    label: &'static str,
    selected_paths: &[PathBuf],
    runtime_config: &RuntimeConfig,
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
    flags: MicrobenchFlags,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    collect_outputs: bool,
    warmup_source: Option<&NativeIndexShareWarmupSource>,
    allow_compact_flash_auto: bool,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<MicrobenchCase> {
    configure_env_flags(args, flags, allow_compact_flash_auto);
    let mut native_logs = Some(NativeLogCapture::start(flags.capture_native_logs())?);
    let model = match StageModel::open_from_parts(selected_paths, runtime_config) {
        Ok(model) => model,
        Err(error) => {
            let native_context = native_logs
                .take()
                .expect("native log capture is initialized")
                .finish_after_open_error();
            return Err(error).with_context(|| {
                format!("open GLM-DSA layer microbench model for {label}{native_context}")
            });
        }
    };
    let (timings, outputs) = if should_reuse_kv_warmup_stream(args) {
        run_microbench_iterations_with_streaming_kv(
            &model,
            args,
            layer_start,
            layer_end,
            warmup_source,
            flags,
            input,
            token_ids,
            deferred_model_drops,
            collect_outputs,
        )?
    } else if should_reuse_kv_warmup_checkpoint(args) {
        run_microbench_iterations_with_reused_kv(
            &model,
            args,
            layer_start,
            layer_end,
            warmup_source,
            flags,
            input,
            token_ids,
            positions,
            deferred_model_drops,
            collect_outputs,
        )?
    } else {
        run_microbench_iterations_with_fresh_sessions(
            &model,
            args,
            layer_start,
            layer_end,
            warmup_source,
            flags,
            input,
            token_ids,
            positions,
            deferred_model_drops,
            collect_outputs,
        )?
    };
    let native_timings = native_logs
        .take()
        .expect("native log capture is initialized")
        .finish()?;
    deferred_model_drops.push(model);
    let compact_flash_policy_records = retain_case_compact_policy_records(
        native_timings.compact_flash_policy_records,
        args.tokens,
    );
    let (compact_flash_non_measured_policy_records, compact_flash_execution_policy_records) =
        split_execution_compact_policy_records(compact_flash_policy_records.clone(), timings.len());
    let compact_flash_mask_records =
        retain_case_compact_mask_records(native_timings.compact_flash_mask_records, args.tokens);
    let (compact_flash_non_measured_mask_records, compact_flash_execution_mask_records) =
        split_execution_compact_mask_records(compact_flash_mask_records.clone(), timings.len());
    let direct_sparse_decision_records =
        retain_case_decision_records(native_timings.direct_sparse_decision_records, args.tokens);
    let (direct_sparse_non_measured_decision_records, direct_sparse_execution_decision_records) =
        split_execution_decision_records(direct_sparse_decision_records.clone(), timings.len());
    Ok(MicrobenchCase {
        label,
        flags,
        n_gpu_layers: runtime_config.n_gpu_layers,
        measured_tokens: args.tokens,
        native_log_path: native_timings.log_path,
        compact_flash_policy_records,
        compact_flash_execution_policy_records,
        compact_flash_non_measured_policy_records,
        compact_flash_mask_records,
        compact_flash_execution_mask_records,
        compact_flash_non_measured_mask_records,
        direct_sparse_decision_records,
        direct_sparse_execution_decision_records,
        direct_sparse_non_measured_decision_records,
        metal_dispatch_records: native_timings.metal_dispatch_records,
        op_timing_records: retain_measured_timing_records(
            native_timings.op_timing_records,
            args.tokens,
            args.warmup,
        ),
        group_timing_records: retain_measured_group_timing_records(
            native_timings.group_timing_records,
            args.tokens,
            args.warmup,
        ),
        indexshare_trace_summary: native_timings.indexshare_trace_summary,
        hot_tensor_records: retain_measured_hot_tensor_records(
            native_timings.hot_tensor_records,
            args.tokens,
            args.warmup,
        ),
        compute_buffer_records: native_timings.compute_buffer_records,
        timings,
        outputs,
    })
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_microbench_iterations_with_fresh_sessions(
    model: &StageModel,
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
    warmup_source: Option<&NativeIndexShareWarmupSource>,
    flags: MicrobenchFlags,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    deferred_model_drops: &mut Vec<StageModel>,
    collect_outputs: bool,
) -> Result<(Vec<IterationTiming>, Vec<ActivationFrame>)> {
    let mut timings = Vec::with_capacity(args.iterations);
    let mut outputs = Vec::with_capacity(if collect_outputs { args.iterations } else { 0 });
    let total_runs = args.warmup + args.iterations;
    for run_index in 0..total_runs {
        let mut session = model.create_session().context("create stage session")?;
        warm_session_kv_prefix_for_case(
            &mut session,
            args,
            flags,
            layer_start,
            layer_end,
            warmup_source,
            deferred_model_drops,
        )
        .with_context(|| format!("warm GLM-DSA KV prefix for iteration {run_index}"))?;
        run_timed_iteration(
            &mut session,
            args,
            input,
            token_ids,
            positions,
            collect_outputs,
            run_index,
            args.position_start,
            &mut timings,
            &mut outputs,
        )?;
    }
    Ok((timings, outputs))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_microbench_iterations_with_streaming_kv(
    model: &StageModel,
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
    warmup_source: Option<&NativeIndexShareWarmupSource>,
    flags: MicrobenchFlags,
    input: &ActivationFrame,
    token_ids: &[i32],
    deferred_model_drops: &mut Vec<StageModel>,
    collect_outputs: bool,
) -> Result<(Vec<IterationTiming>, Vec<ActivationFrame>)> {
    let mut session = model.create_session().context("create stage session")?;
    warm_session_kv_prefix_for_case(
        &mut session,
        args,
        flags,
        layer_start,
        layer_end,
        warmup_source,
        deferred_model_drops,
    )
    .context("warm GLM-DSA KV prefix for streaming decode")?;
    let mut timings = Vec::with_capacity(args.iterations);
    let mut outputs = Vec::with_capacity(if collect_outputs { args.iterations } else { 0 });
    let total_runs = args.warmup + args.iterations;
    for run_index in 0..total_runs {
        let run_position_start = streamed_iteration_position_start(args, run_index)?;
        let run_positions = positions(run_position_start, args.tokens)?;
        run_timed_iteration(
            &mut session,
            args,
            input,
            token_ids,
            &run_positions,
            collect_outputs,
            run_index,
            run_position_start,
            &mut timings,
            &mut outputs,
        )?;
    }
    Ok((timings, outputs))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn run_microbench_iterations_with_reused_kv(
    model: &StageModel,
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
    warmup_source: Option<&NativeIndexShareWarmupSource>,
    flags: MicrobenchFlags,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    deferred_model_drops: &mut Vec<StageModel>,
    collect_outputs: bool,
) -> Result<(Vec<IterationTiming>, Vec<ActivationFrame>)> {
    let mut session = model.create_session().context("create stage session")?;
    warm_session_kv_prefix_for_case(
        &mut session,
        args,
        flags,
        layer_start,
        layer_end,
        warmup_source,
        deferred_model_drops,
    )
    .context("warm GLM-DSA KV prefix for reusable checkpoint")?;
    let mut checkpoint = session
        .checkpoint()
        .context("checkpoint warmed GLM-DSA KV prefix")?;
    let mut timings = Vec::with_capacity(args.iterations);
    let mut outputs = Vec::with_capacity(if collect_outputs { args.iterations } else { 0 });
    let total_runs = args.warmup + args.iterations;
    for run_index in 0..total_runs {
        run_timed_iteration(
            &mut session,
            args,
            input,
            token_ids,
            positions,
            collect_outputs,
            run_index,
            args.position_start,
            &mut timings,
            &mut outputs,
        )?;
        session.restore_checkpoint(&checkpoint).with_context(|| {
            format!("restore GLM-DSA KV checkpoint after iteration {run_index}")
        })?;
        if run_index + 1 < total_runs {
            checkpoint = session.checkpoint().with_context(|| {
                format!("refresh GLM-DSA KV checkpoint after iteration {run_index}")
            })?;
        }
    }
    Ok((timings, outputs))
}

#[allow(clippy::too_many_arguments)]
fn run_timed_iteration(
    session: &mut skippy_runtime::StageSession,
    args: &GlmDsaLayerMicrobenchArgs,
    input: &ActivationFrame,
    token_ids: &[i32],
    positions: &[i32],
    collect_outputs: bool,
    run_index: usize,
    position_start: i32,
    timings: &mut Vec<IterationTiming>,
    outputs: &mut Vec<ActivationFrame>,
) -> Result<()> {
    let started = Instant::now();
    let output = session
        .prefill_chunk_frame_with_positions(token_ids, positions, Some(input), 0)
        .with_context(|| format!("run microbench iteration {run_index}"))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if run_index >= args.warmup {
        timings.push(IterationTiming {
            iteration: run_index - args.warmup,
            position_start,
            elapsed_ms,
            output_payload_bytes: output.payload.len(),
            output_flags: output.desc.flags,
        });
        if collect_outputs {
            outputs.push(output);
        }
    }
    Ok(())
}

fn should_reuse_kv_warmup_checkpoint(args: &GlmDsaLayerMicrobenchArgs) -> bool {
    args.reuse_kv_warmup_checkpoint && args.kv_warmup_tokens > 0
}

fn should_reuse_kv_warmup_stream(args: &GlmDsaLayerMicrobenchArgs) -> bool {
    args.reuse_kv_warmup_stream
}

fn streamed_iteration_position_start(
    args: &GlmDsaLayerMicrobenchArgs,
    run_index: usize,
) -> Result<i32> {
    let token_offset = run_index
        .checked_mul(args.tokens)
        .context("streaming decode position offset overflow")?;
    let token_offset =
        i32::try_from(token_offset).context("streaming decode position offset exceeds i32")?;
    args.position_start
        .checked_add(token_offset)
        .context("streaming decode position exceeds i32")
}

fn warm_session_kv_prefix_for_range(
    session: &mut skippy_runtime::StageSession,
    args: &GlmDsaLayerMicrobenchArgs,
    flags: MicrobenchFlags,
    layer_start: u32,
    layer_end: u32,
) -> Result<()> {
    if args.kv_warmup_tokens == 0 {
        return seed_session_position(session, args.position_start);
    }
    if args.synthetic_kv_warmup {
        return import_synthetic_kv_prefix_for_range(session, args, layer_start, layer_end);
    }
    let _diagnostic_guard = NativeDiagnosticFlagGuard::muted(flags);
    let chunk_tokens = kv_warmup_chunk_tokens(args)
        .min(args.kv_warmup_tokens)
        .max(1);
    let mut token_start = 0usize;
    while token_start < args.kv_warmup_tokens {
        let token_count = (args.kv_warmup_tokens - token_start).min(chunk_tokens);
        let token_ids = vec![1_i32; token_count];
        let position_start =
            i32::try_from(token_start).context("KV warmup token position exceeds i32")?;
        let positions = positions(position_start, token_count)?;
        let input =
            synthetic_activation_frame_for_layer_tokens(args, layer_start, token_count, None)?;
        session
            .prefill_chunk_frame_with_positions(&token_ids, &positions, Some(&input), 0)
            .with_context(|| {
                format!(
                    "run GLM-DSA KV warmup layer range {layer_start}..{layer_end} chunk {}..{}",
                    token_start,
                    token_start + token_count
                )
            })?;
        token_start += token_count;
    }
    Ok(())
}

fn warm_session_kv_prefix_for_case(
    session: &mut skippy_runtime::StageSession,
    args: &GlmDsaLayerMicrobenchArgs,
    flags: MicrobenchFlags,
    layer_start: u32,
    layer_end: u32,
    warmup_source: Option<&NativeIndexShareWarmupSource>,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<()> {
    if let Some(source) = warmup_source {
        return warm_session_kv_prefix_from_top_k_source(
            session,
            args,
            flags,
            layer_start,
            layer_end,
            source,
            deferred_model_drops,
        );
    }
    warm_session_kv_prefix_for_range(session, args, flags, layer_start, layer_end)
}

fn warm_session_kv_prefix_from_top_k_source(
    target_session: &mut skippy_runtime::StageSession,
    args: &GlmDsaLayerMicrobenchArgs,
    flags: MicrobenchFlags,
    target_layer_start: u32,
    target_layer_end: u32,
    source: &NativeIndexShareWarmupSource,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<()> {
    if source.layer_end != target_layer_start {
        bail!(
            "native IndexShare warmup source {}..{} must end at target layer_start {}",
            source.layer_start,
            source.layer_end,
            target_layer_start
        );
    }
    if args.kv_warmup_tokens == 0 {
        return seed_session_position(target_session, args.position_start);
    }

    let _diagnostic_guard = NativeDiagnosticFlagGuard::muted(flags);
    let source_model = StageModel::open_from_parts(&source.selected_paths, &source.runtime_config)
        .with_context(|| {
            format!(
                "open GLM-DSA native IndexShare warmup source {}..{}",
                source.layer_start, source.layer_end
            )
        })?;
    let mut source_session = source_model
        .create_session()
        .context("create GLM-DSA native IndexShare warmup source session")?;

    let chunk_tokens = kv_warmup_chunk_tokens(args)
        .min(args.kv_warmup_tokens)
        .max(1);
    let mut token_start = 0usize;
    while token_start < args.kv_warmup_tokens {
        let token_count = (args.kv_warmup_tokens - token_start).min(chunk_tokens);
        warm_top_k_target_chunk(
            &mut source_session,
            target_session,
            args,
            source.layer_start,
            target_layer_start,
            target_layer_end,
            token_start,
            token_count,
        )?;
        token_start += token_count;
    }
    drop(source_session);
    deferred_model_drops.push(source_model);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn warm_top_k_target_chunk(
    source_session: &mut skippy_runtime::StageSession,
    target_session: &mut skippy_runtime::StageSession,
    args: &GlmDsaLayerMicrobenchArgs,
    source_layer_start: u32,
    target_layer_start: u32,
    target_layer_end: u32,
    token_start: usize,
    token_count: usize,
) -> Result<()> {
    let token_ids = vec![1_i32; token_count];
    let position_start =
        i32::try_from(token_start).context("KV warmup token position exceeds i32")?;
    let positions = positions(position_start, token_count)?;
    let _malformed_sideband_guard = ScopedEnvRemoval::remove(ENV_MALFORMED_TOP_K_BYTES);
    let source_input =
        synthetic_activation_frame_for_layer_tokens(args, source_layer_start, token_count, None)?;
    let (_predicted_token, top_k_frame) = source_session
        .prefill_chunk_frame_sampled_with_positions(
            &token_ids,
            &positions,
            None,
            Some(&source_input),
            0,
        )
        .with_context(|| {
            format!(
                "run GLM-DSA native IndexShare warmup source layer {source_layer_start} chunk {}..{}",
                token_start,
                token_start + token_count
            )
        })?;
    target_session
        .prefill_chunk_frame_with_positions(&token_ids, &positions, Some(&top_k_frame), 0)
        .with_context(|| {
            format!(
                "run GLM-DSA native IndexShare warmup target range {target_layer_start}..{target_layer_end} chunk {}..{}",
                token_start,
                token_start + token_count
            )
        })?;
    Ok(())
}

fn import_synthetic_kv_prefix_for_range(
    session: &mut skippy_runtime::StageSession,
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    layer_end: u32,
) -> Result<()> {
    let layer_count = layer_end
        .checked_sub(layer_start)
        .context("invalid GLM-DSA layer range")?;
    let chunk_tokens = kv_warmup_chunk_tokens(args)
        .min(args.kv_warmup_tokens)
        .max(1);
    let mut token_start = 0usize;
    while token_start < args.kv_warmup_tokens {
        let token_count = (args.kv_warmup_tokens - token_start).min(chunk_tokens);
        let desc = synthetic_kv_page_desc(
            layer_start,
            layer_end,
            layer_count,
            token_start,
            token_count,
        )?;
        let payload = vec![
            0_u8;
            usize::try_from(desc.payload_bytes)
                .context("synthetic KV page exceeds usize")?
        ];
        session.import_kv_page(&desc, &payload).with_context(|| {
            format!(
                "import synthetic GLM-DSA KV page {}..{}",
                token_start,
                token_start + token_count
            )
        })?;
        token_start += token_count;
    }
    let token_count = u64::try_from(args.kv_warmup_tokens)
        .context("synthetic KV warmup token count exceeds u64")?;
    session
        .set_position(token_count)
        .context("set session position after synthetic KV import")?;
    Ok(())
}

fn synthetic_kv_page_desc(
    layer_start: u32,
    layer_end: u32,
    layer_count: u32,
    token_start: usize,
    token_count: usize,
) -> Result<RuntimeKvPageDesc> {
    let token_start = u64::try_from(token_start).context("synthetic KV token_start exceeds u64")?;
    let token_count = u64::try_from(token_count).context("synthetic KV token_count exceeds u64")?;
    let per_layer_bytes = token_count
        .checked_mul(u64::from(GLM_DSA_F16_K_ROW_BYTES))
        .context("synthetic KV page byte count overflow")?;
    let payload_bytes = per_layer_bytes
        .checked_mul(u64::from(layer_count))
        .context("synthetic KV page payload byte count overflow")?;
    Ok(RuntimeKvPageDesc {
        version: 1,
        layer_start: i32::try_from(layer_start).context("layer_start exceeds i32")?,
        layer_end: i32::try_from(layer_end).context("layer_end exceeds i32")?,
        token_start,
        token_count,
        layer_count,
        k_type: GGML_TYPE_F16_ID,
        v_type: GGML_TYPE_F32_ID,
        k_row_bytes: GLM_DSA_F16_K_ROW_BYTES,
        v_row_bytes: 0,
        v_element_bytes: 0,
        payload_bytes,
        flags: 0,
    })
}

struct NativeDiagnosticFlagGuard {
    flags: MicrobenchFlags,
}

impl NativeDiagnosticFlagGuard {
    fn muted(flags: MicrobenchFlags) -> Self {
        set_native_diagnostic_flags(flags, false);
        Self { flags }
    }
}

impl Drop for NativeDiagnosticFlagGuard {
    fn drop(&mut self) {
        set_native_diagnostic_flags(self.flags, true);
    }
}

struct ScopedEnvRemoval {
    name: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvRemoval {
    fn remove(name: &'static str) -> Self {
        let previous = std::env::var_os(name);
        clear_env(name);
        Self { name, previous }
    }
}

impl Drop for ScopedEnvRemoval {
    fn drop(&mut self) {
        // This command is single-threaded and temporarily scopes bench-only env knobs.
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.name, previous);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }
}

fn set_native_diagnostic_flags(flags: MicrobenchFlags, enabled: bool) {
    set_env_flag("SKIPPY_GLM_DSA_OP_TIMING", enabled && flags.op_timing);
    set_env_flag(
        "LLAMA_GLM_DSA_INDEXSHARE_EXEC_LOG",
        enabled && flags.native_indexshare_exec_log,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_DIRECT_SPARSE_DECISIONS",
        enabled && flags.op_timing,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_COMPACT_FLASH_POLICY",
        enabled && flags.metal_dispatch_log,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_METAL_DISPATCH",
        enabled && flags.metal_dispatch_log,
    );
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
    if args.ctx_size == 0 {
        bail!("ctx_size must be greater than zero");
    }
    let max_position_end = max_requested_position_end(args)?;
    if max_position_end >= i64::from(args.ctx_size) {
        bail!(
            "requested position_end {max_position_end} must be less than ctx_size {}; valid positions are 0..{}",
            args.ctx_size,
            args.ctx_size - 1
        );
    }
    if args.kv_warmup_tokens > 0 {
        let position_start =
            usize::try_from(args.position_start).context("position_start is negative")?;
        if args.kv_warmup_tokens != position_start {
            bail!(
                "kv_warmup_tokens must equal position_start so the synthetic KV prefix covers exactly 0..position_start"
            );
        }
    }
    if args.kv_warmup_chunk_tokens == Some(0) {
        bail!("kv_warmup_chunk_tokens must be greater than zero when set");
    }
    if args.synthetic_kv_warmup {
        if args.kv_warmup_tokens == 0 {
            bail!("synthetic_kv_warmup requires kv_warmup_tokens");
        }
        if !args.cache_type_k.eq_ignore_ascii_case("f16")
            || !args.cache_type_v.eq_ignore_ascii_case("f16")
        {
            bail!("synthetic_kv_warmup currently requires f16 cache_type_k and cache_type_v");
        }
    }
    if let Some(n_batch) = args.n_batch {
        let warmup_chunk = kv_warmup_chunk_tokens(args);
        if warmup_chunk > n_batch as usize {
            bail!("kv_warmup_chunk_tokens ({warmup_chunk}) must fit within n_batch ({n_batch})");
        }
    }
    if top_k_sideband_warmup_exports_chunk(args)?
        && let Some(n_ubatch) = args.n_ubatch
    {
        let warmup_chunk = kv_warmup_chunk_tokens(args);
        if warmup_chunk > n_ubatch as usize {
            bail!(
                "kv_warmup_chunk_tokens ({warmup_chunk}) must fit within n_ubatch ({n_ubatch}) when KV warmup exports GLM-DSA top-k sideband"
            );
        }
    }
    if args.reuse_kv_warmup_checkpoint && args.reuse_kv_warmup_stream {
        bail!("reuse_kv_warmup_checkpoint and reuse_kv_warmup_stream are mutually exclusive");
    }
    if args.reuse_kv_warmup_stream && real_top_k_source_layer_start(args)?.is_some() {
        bail!(
            "reuse_kv_warmup_stream cannot be combined with real top-k source input because the cached sideband is position-specific"
        );
    }
    if args.metal_topk_moe_route_fusion_native_default && !args.metal_topk_moe_route_fusion {
        bail!(
            "metal_topk_moe_route_fusion_native_default cannot be combined with metal_topk_moe_route_fusion=false"
        );
    }
    let comparison_count = usize::from(args.compare_dense_fallback)
        + usize::from(args.compare_dense_flash_prefill)
        + usize::from(args.compare_cpu_direct_sparse)
        + usize::from(args.compare_metal_sparse_attn_threads_baseline.is_some())
        + usize::from(args.compare_moe_down_weighted_fusion)
        + usize::from(args.compare_moe_down_weighted_parallel)
        + usize::from(args.compare_moe_down_unweighted_slots)
        + usize::from(args.compare_native_indexshare_producer_consumer);
    if comparison_count > 1 {
        bail!(
            "compare_dense_fallback, compare_dense_flash_prefill, compare_cpu_direct_sparse, compare_metal_sparse_attn_threads_baseline, compare_moe_down_weighted_fusion, compare_moe_down_weighted_parallel, compare_moe_down_unweighted_slots, and compare_native_indexshare_producer_consumer are mutually exclusive"
        );
    }
    if real_top_k_warmup_source_layer_start(args)?.is_some() {
        if comparison_count > 0 {
            bail!(
                "{ENV_REAL_TOP_K_WARMUP_SOURCE_LAYER_START} cannot be combined with comparison modes"
            );
        }
        if args.synthetic_kv_warmup {
            bail!(
                "{ENV_REAL_TOP_K_WARMUP_SOURCE_LAYER_START} cannot be combined with synthetic_kv_warmup"
            );
        }
    }
    if args.compare_native_indexshare_producer_consumer
        && args.layer_start.saturating_add(1) >= args.layer_end
    {
        bail!("compare_native_indexshare_producer_consumer requires at least two layers");
    }
    validate_sparse_attn_threads("sparse_attn_threads", args.sparse_attn_threads)?;
    validate_sparse_attn_group_heads("sparse_attn_group_heads", args.sparse_attn_group_heads)?;
    validate_lightning_indexer_threads(
        "lightning_indexer_threads",
        args.lightning_indexer_threads,
    )?;
    validate_sparse_attn_threads(
        "compare_metal_sparse_attn_threads_baseline",
        args.compare_metal_sparse_attn_threads_baseline,
    )?;
    if args.compare_metal_sparse_attn_threads_baseline.is_some()
        && args.sparse_attn_threads.is_none()
    {
        bail!(
            "compare_metal_sparse_attn_threads_baseline requires sparse_attn_threads for the candidate run"
        );
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

fn top_k_sideband_warmup_exports_chunk(args: &GlmDsaLayerMicrobenchArgs) -> Result<bool> {
    if args.kv_warmup_tokens == 0 || args.synthetic_kv_warmup {
        return Ok(false);
    }

    Ok(args.compare_native_indexshare_producer_consumer
        || real_top_k_source_layer_start(args)?.is_some()
        || real_top_k_warmup_source_layer_start(args)?.is_some())
}

fn max_requested_position_end(args: &GlmDsaLayerMicrobenchArgs) -> Result<i64> {
    let run_index = if args.reuse_kv_warmup_stream {
        args.warmup
            .checked_add(args.iterations)
            .and_then(|total| total.checked_sub(1))
            .context("streaming decode run count overflow")?
    } else {
        0
    };
    let run_token_offset = run_index
        .checked_mul(args.tokens)
        .context("streaming decode position offset overflow")?;
    let run_token_offset =
        i64::try_from(run_token_offset).context("streaming decode position offset exceeds i64")?;
    let token_offset = args
        .tokens
        .checked_sub(1)
        .context("tokens must be greater than zero")?;
    let token_offset = i64::try_from(token_offset).context("position offset exceeds i64")?;
    i64::from(args.position_start)
        .checked_add(run_token_offset)
        .and_then(|position| position.checked_add(token_offset))
        .context("position exceeds i64")
}

struct GlmDsaSingleRunGuard {
    lock_path: Option<PathBuf>,
}

impl GlmDsaSingleRunGuard {
    fn acquire(args: &GlmDsaLayerMicrobenchArgs) -> Result<Self> {
        if args.allow_concurrent {
            return Ok(Self { lock_path: None });
        }

        let current_pid = std::process::id();
        let active = active_glm_microbench_processes(current_pid)?;
        if !active.is_empty() {
            let summary = active
                .iter()
                .map(|process| format!("pid={} {}", process.pid, process.command))
                .collect::<Vec<_>>()
                .join("; ");
            bail!(
                "another glm-dsa-layer-microbench process is already running: {summary}. \
                 Stop it first or pass --allow-concurrent if this overlap is intentional"
            );
        }

        acquire_glm_microbench_lock(current_pid)
    }
}

impl Drop for GlmDsaSingleRunGuard {
    fn drop(&mut self) {
        if let Some(path) = &self.lock_path {
            let _ = fs::remove_file(path);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ActiveGlmMicrobenchProcess {
    pid: u32,
    command: String,
}

fn active_glm_microbench_processes(current_pid: u32) -> Result<Vec<ActiveGlmMicrobenchProcess>> {
    let output = Command::new("ps")
        .args(["-axo", "pid=,args="])
        .output()
        .context("scan process table for active GLM-DSA microbench runs")?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout
        .lines()
        .filter_map(|line| parse_active_glm_microbench_process(line, current_pid))
        .collect())
}

fn parse_active_glm_microbench_process(
    line: &str,
    current_pid: u32,
) -> Option<ActiveGlmMicrobenchProcess> {
    let line = line.trim();
    let (pid, command) = line.split_once(char::is_whitespace)?;
    let pid = pid.parse::<u32>().ok()?;
    if pid == current_pid || !is_glm_microbench_command(command.trim()) {
        return None;
    }
    Some(ActiveGlmMicrobenchProcess {
        pid,
        command: command.trim().to_string(),
    })
}

fn is_glm_microbench_command(command: &str) -> bool {
    let Some(executable) = command.split_whitespace().next() else {
        return false;
    };
    let Some(name) = Path::new(executable)
        .file_name()
        .and_then(|name| name.to_str())
    else {
        return false;
    };
    (name == "skippy-bench" || name.starts_with("skippy_bench-"))
        && command.contains("glm-dsa-layer-microbench")
}

fn acquire_glm_microbench_lock(current_pid: u32) -> Result<GlmDsaSingleRunGuard> {
    let lock_path = std::env::temp_dir().join("skippy-bench-glm-dsa-layer-microbench.lock");
    for _ in 0..2 {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                writeln!(file, "{current_pid}").context("write GLM-DSA microbench lock pid")?;
                return Ok(GlmDsaSingleRunGuard {
                    lock_path: Some(lock_path),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let lock_pid = read_lock_pid(&lock_path);
                if lock_pid.is_some_and(process_is_live) {
                    bail!(
                        "another glm-dsa-layer-microbench process owns {} with pid {}. \
                         Stop it first or pass --allow-concurrent if this overlap is intentional",
                        lock_path.display(),
                        lock_pid.unwrap()
                    );
                }
                fs::remove_file(&lock_path)
                    .with_context(|| format!("remove stale lock {}", lock_path.display()))?;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("create GLM-DSA microbench lock {}", lock_path.display())
                });
            }
        }
    }
    bail!(
        "failed to acquire GLM-DSA microbench lock at {} after removing stale state",
        lock_path.display()
    )
}

fn read_lock_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn process_is_live(pid: u32) -> bool {
    let pid = pid.to_string();
    Command::new("ps")
        .args(["-p", &pid])
        .output()
        .is_ok_and(|output| output.status.success())
}

fn configure_env_flags(
    args: &GlmDsaLayerMicrobenchArgs,
    flags: MicrobenchFlags,
    allow_compact_flash_auto: bool,
) {
    configure_direct_sparse_env(flags.direct_sparse_attn);
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_COMPACT_FLASH_ATTN",
        flags.compact_flash_attn,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_DISABLE_COMPACT_FLASH_ATTN",
        !flags.compact_flash_attn && !allow_compact_flash_auto,
    );
    if args.native_default_direct_sparse_prefill {
        clear_env("SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_PREFILL");
        clear_env("SKIPPY_GLM_DSA_DISABLE_DIRECT_SPARSE_PREFILL");
    } else {
        set_env_flag(
            "SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_PREFILL",
            flags.direct_sparse_prefill,
        );
        set_env_flag(
            "SKIPPY_GLM_DSA_DISABLE_DIRECT_SPARSE_PREFILL",
            !flags.direct_sparse_prefill,
        );
    }
    set_env_flag(
        "SKIPPY_GLM_DSA_ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL",
        flags.enable_unproven_large_direct_sparse_prefill,
    );
    set_optional_env(
        "SKIPPY_GLM_DSA_DIRECT_SPARSE_PREFILL_MAX_TOKENS",
        flags
            .direct_sparse_prefill_max_tokens
            .map(|max_tokens| max_tokens.to_string()),
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
        "LLAMA_GLM_DSA_INDEXSHARE_EXEC_LOG",
        flags.native_indexshare_exec_log,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_DIRECT_SPARSE_DECISIONS",
        flags.op_timing
            || flags.metal_dispatch_log
            || args.require_direct_sparse_prefill_proof
            || args.require_partial_top_k_proof,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_COMPACT_FLASH_POLICY",
        flags.metal_dispatch_log || args.require_compact_flash_proof,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_LOG_METAL_DISPATCH",
        flags.metal_dispatch_log,
    );
    configure_metal_topk_moe_route_fusion_env(flags);
    set_env_flag(
        "SKIPPY_GLM_DSA_MOE_MOTIF_COENCODE",
        flags.moe_motif_coencode,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_MOE_DOWN_WEIGHTED_FUSION",
        flags.moe_down_weighted_fusion,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_EXPERIMENTAL_Q3_DOWN_WEIGHTED_FUSION",
        flags.moe_down_weighted_fusion,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_EXPERIMENTAL_Q3_DOWN_WEIGHTED_PARALLEL",
        flags.moe_down_weighted_parallel,
    );
    set_env_flag(
        "SKIPPY_GLM_DSA_EXPERIMENTAL_Q3_DOWN_UNWEIGHTED_SLOTS",
        flags.moe_down_unweighted_slots,
    );
    set_optional_env(
        "SKIPPY_GLM_DSA_SPARSE_ATTN_THREADS",
        flags.sparse_attn_threads.map(|threads| threads.to_string()),
    );
    set_optional_env(
        "SKIPPY_GLM_DSA_SPARSE_ATTN_DECODE_GROUP_HEADS",
        flags.sparse_attn_group_heads.map(|heads| heads.to_string()),
    );
    set_optional_env(
        "LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER_THREADS",
        flags
            .lightning_indexer_threads
            .map(|threads| threads.to_string()),
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
    set_optional_env(
        "SKIPPY_GLM_DSA_DENSE_SPARSE_MASK_MAX_BYTES",
        flags
            .dense_sparse_mask_max_bytes
            .map(|max_bytes| max_bytes.to_string()),
    );
    set_optional_env(
        "SKIPPY_GLM_DSA_DIRECT_SPARSE_PREFILL_MIN_KV_TOPK_RATIO",
        flags
            .direct_sparse_prefill_min_kv_topk_ratio
            .map(|ratio| ratio.to_string()),
    );
}

fn configure_direct_sparse_env(enabled: bool) {
    if enabled {
        clear_env("SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_ATTN");
        clear_env("SKIPPY_GLM_DSA_DISABLE_DIRECT_SPARSE_ATTN");
        clear_env("SKIPPY_GLM_DSA_ALLOW_DENSE_SPARSE_MASK_FALLBACK");
    } else {
        set_env_flag("SKIPPY_GLM_DSA_ENABLE_DIRECT_SPARSE_ATTN", false);
        clear_env("SKIPPY_GLM_DSA_DISABLE_DIRECT_SPARSE_ATTN");
        set_env_flag("SKIPPY_GLM_DSA_ALLOW_DENSE_SPARSE_MASK_FALLBACK", true);
    }
}

fn configure_metal_topk_moe_route_fusion_env(flags: MicrobenchFlags) {
    match metal_topk_moe_route_fusion_env_plan(flags) {
        RouteFusionEnvPlan::NativeDefault => {
            clear_env("SKIPPY_GLM_DSA_ENABLE_METAL_TOPK_MOE_FUSION");
            clear_env("LLAMA_GLM_DSA_ENABLE_METAL_TOPK_MOE_FUSION");
            clear_env("LLAMA_GLM_DSA_DISABLE_METAL_TOPK_MOE_FUSION");
        }
        RouteFusionEnvPlan::LegacyOverride(enabled) => {
            set_env_flag("SKIPPY_GLM_DSA_ENABLE_METAL_TOPK_MOE_FUSION", enabled);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RouteFusionEnvPlan {
    NativeDefault,
    LegacyOverride(bool),
}

fn metal_topk_moe_route_fusion_env_plan(flags: MicrobenchFlags) -> RouteFusionEnvPlan {
    if flags.metal_topk_moe_route_fusion_native_default {
        RouteFusionEnvPlan::NativeDefault
    } else {
        RouteFusionEnvPlan::LegacyOverride(flags.metal_topk_moe_route_fusion)
    }
}

fn set_env_flag(name: &str, enabled: bool) {
    // This command is single-threaded and sets native runtime flags before opening llama.cpp.
    unsafe {
        std::env::set_var(name, if enabled { "1" } else { "0" });
    }
}

fn clear_env(name: &str) {
    // This command is single-threaded and sets native runtime flags before opening llama.cpp.
    unsafe {
        std::env::remove_var(name);
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

fn validate_sparse_attn_threads(label: &str, threads: Option<u32>) -> Result<()> {
    let Some(threads) = threads else {
        return Ok(());
    };
    match threads {
        32 | 64 | 128 | 256 => Ok(()),
        _ => bail!("{label} must be one of 32, 64, 128, or 256"),
    }
}

fn validate_sparse_attn_group_heads(label: &str, heads: Option<u32>) -> Result<()> {
    let Some(heads) = heads else {
        return Ok(());
    };
    match heads {
        2 | 4 => Ok(()),
        _ => bail!("{label} must be one of 2 or 4"),
    }
}

fn validate_lightning_indexer_threads(label: &str, threads: Option<u32>) -> Result<()> {
    let Some(threads) = threads else {
        return Ok(());
    };
    match threads {
        32 | 64 | 128 | 256 | 512 | 1024 => Ok(()),
        _ => bail!("{label} must be one of 32, 64, 128, 256, 512, or 1024"),
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
        n_batch: Some(
            args.n_batch
                .unwrap_or_else(|| bounded_u32(runtime_batch_tokens(args))),
        ),
        n_ubatch: Some(
            args.n_ubatch
                .unwrap_or_else(|| bounded_u32(runtime_batch_tokens(args))),
        ),
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

fn runtime_batch_tokens(args: &GlmDsaLayerMicrobenchArgs) -> usize {
    args.tokens.max(kv_warmup_chunk_tokens(args)).max(1)
}

fn kv_warmup_chunk_tokens(args: &GlmDsaLayerMicrobenchArgs) -> usize {
    args.kv_warmup_chunk_tokens
        .or_else(|| args.n_ubatch.map(|value| value as usize))
        .or_else(|| args.n_batch.map(|value| value as usize))
        .unwrap_or_else(|| conservative_kv_warmup_chunk_tokens(args))
        .max(1)
}

fn conservative_kv_warmup_chunk_tokens(args: &GlmDsaLayerMicrobenchArgs) -> usize {
    args.tokens.max(args.kv_warmup_tokens.min(128)).max(1)
}

fn prepare_input_activation(
    args: &GlmDsaLayerMicrobenchArgs,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<PreparedInputActivation> {
    if let Some(source_layer_start) = real_top_k_source_layer_start(args)? {
        return real_top_k_activation_frame(
            args,
            token_ids,
            positions,
            flags,
            source_layer_start,
            deferred_model_drops,
        );
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
    parse_real_top_k_source_layer_start(&value, args.layer_start, ENV_REAL_TOP_K_SOURCE_LAYER_START)
}

fn parse_real_top_k_source_layer_start(
    value: &str,
    target_layer_start: u32,
    env_name: &str,
) -> Result<Option<u32>> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let layer_start = trimmed
        .parse::<u32>()
        .with_context(|| format!("parse {env_name}"))?;
    if layer_start >= target_layer_start {
        bail!("{env_name} must be less than target layer_start {target_layer_start}",);
    }
    Ok(Some(layer_start))
}

fn real_top_k_warmup_source_layer_start(args: &GlmDsaLayerMicrobenchArgs) -> Result<Option<u32>> {
    let Ok(value) = std::env::var(ENV_REAL_TOP_K_WARMUP_SOURCE_LAYER_START) else {
        return Ok(None);
    };
    parse_real_top_k_source_layer_start(
        &value,
        args.layer_start,
        ENV_REAL_TOP_K_WARMUP_SOURCE_LAYER_START,
    )
}

fn real_top_k_activation_frame(
    args: &GlmDsaLayerMicrobenchArgs,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
    source_layer_start: u32,
    deferred_model_drops: &mut Vec<StageModel>,
) -> Result<PreparedInputActivation> {
    let source_layer_end = args.layer_start;
    let generated = generate_real_top_k_frame(
        args,
        token_ids,
        positions,
        flags,
        source_layer_start,
        source_layer_end,
        deferred_model_drops,
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
    deferred_model_drops: &mut Vec<StageModel>,
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

    let source_input = real_top_k_source_input(
        args,
        token_ids,
        positions,
        flags,
        source_layer_start,
        deferred_model_drops,
    )?;
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
        enable_unproven_large_direct_sparse_prefill: false,
        direct_sparse_prefill_max_tokens: None,
        ..flags
    };
    configure_env_flags(args, source_flags, allow_compact_flash_auto(args));
    let source_model = StageModel::open_from_parts(&source_selected.absolute_paths, &source_config)
        .with_context(|| {
            format!("open GLM-DSA real top-k source model {source_layer_start}..{source_layer_end}")
        })?;
    let mut source_session = source_model
        .create_session()
        .context("create GLM-DSA real top-k source session")?;
    warm_session_kv_prefix_for_range(
        &mut source_session,
        args,
        source_flags,
        source_layer_start,
        source_layer_end,
    )?;
    let frame = source_session
        .prefill_chunk_frame_with_positions(token_ids, positions, Some(&source_input), 0)
        .with_context(|| {
            format!("run GLM-DSA real top-k source {source_layer_start}..{source_layer_end}")
        })?;
    validate_real_top_k_frame_for_range(args, &frame, source_layer_start, source_layer_end)?;
    drop(source_session);
    deferred_model_drops.push(source_model);
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

fn seed_session_position(
    session: &mut skippy_runtime::StageSession,
    position_start: i32,
) -> Result<()> {
    let token_count = u64::try_from(position_start).context("position_start is negative")?;
    if token_count > 0 {
        session
            .set_position(token_count)
            .context("seed stage session position")?;
    }
    Ok(())
}

fn real_top_k_source_input(
    args: &GlmDsaLayerMicrobenchArgs,
    token_ids: &[i32],
    positions: &[i32],
    flags: MicrobenchFlags,
    source_layer_start: u32,
    deferred_model_drops: &mut Vec<StageModel>,
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
        deferred_model_drops,
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
    if !env_flag_enabled(ENV_STAGE_WIRE_ROUNDTRIP) && !args.require_real_top_k_shared_consumer_proof
    {
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
    let top_k_sideband_stats = summarize_top_k_sideband(
        &message.raw_bytes,
        positions,
        args.position_start,
        args.tokens,
    );
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
        top_k_sideband_stats,
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

fn summarize_top_k_sideband(
    raw_bytes: &[u8],
    positions: &[i32],
    position_start: i32,
    token_count: usize,
) -> TopKSidebandStats {
    let i32_aligned = raw_bytes.len().is_multiple_of(std::mem::size_of::<i32>());
    if !i32_aligned || raw_bytes.is_empty() || token_count == 0 {
        return TopKSidebandStats {
            i32_aligned,
            token_count,
            ..TopKSidebandStats::default()
        };
    }

    let values: Vec<i32> = raw_bytes
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("chunk size is checked")))
        .collect();
    let total_i32 = values.len();
    let width_per_token = total_i32
        .is_multiple_of(token_count)
        .then_some(total_i32 / token_count);
    let Some(width) = width_per_token else {
        return TopKSidebandStats {
            i32_aligned,
            token_count,
            total_i32,
            width_per_token,
            ..TopKSidebandStats::default()
        };
    };

    let mut stats = TopKSidebandStats {
        i32_aligned,
        token_count,
        total_i32,
        width_per_token,
        ..TopKSidebandStats::default()
    };
    let fallback_kv_len = position_start
        .max(0)
        .saturating_add(i32::try_from(token_count).unwrap_or(i32::MAX));
    let kv_len = positions
        .iter()
        .copied()
        .max()
        .map_or(fallback_kv_len, |position| {
            position.saturating_add(1).max(fallback_kv_len)
        })
        .max(0);

    for token_index in 0..token_count {
        let token_start = token_index * width;
        let token_end = token_start + width;
        let current_position = positions
            .get(token_index)
            .copied()
            .unwrap_or(position_start.saturating_add(token_index as i32))
            .max(0);
        let mut seen = HashSet::with_capacity(width.min(2048));
        let mut active_top_end = 0usize;

        for (offset, &value) in values[token_start..token_end].iter().enumerate() {
            if value < 0 {
                stats.negative_index_count += 1;
                continue;
            }
            if value >= kv_len {
                stats.out_of_range_index_count += 1;
                continue;
            }

            stats.valid_index_count += 1;
            if value <= current_position {
                stats.causal_visible_count += 1;
                active_top_end = offset + 1;
            } else {
                stats.future_index_count += 1;
            }
            if !seen.insert(value) {
                stats.duplicate_index_count += 1;
            }
        }

        stats.active_top_end_sum += active_top_end;
        stats.max_active_top_end = stats.max_active_top_end.max(active_top_end);
        stats.inactive_tail_count += width.saturating_sub(active_top_end);
    }

    stats.avg_active_top_end = nonzero_ratio(stats.active_top_end_sum, token_count);
    stats.active_prefix_ratio = nonzero_ratio(stats.active_top_end_sum, total_i32);
    stats.causal_visible_ratio = nonzero_ratio(stats.causal_visible_count, total_i32);
    stats.masked_future_in_active_prefix_count = stats
        .active_top_end_sum
        .saturating_sub(stats.causal_visible_count);

    stats
}

fn nonzero_ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
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
    let n_batch = args
        .n_batch
        .unwrap_or_else(|| bounded_u32(runtime_batch_tokens(args)));
    let n_ubatch = args
        .n_ubatch
        .unwrap_or_else(|| bounded_u32(runtime_batch_tokens(args)));
    let file_name = format!(
        "real-topk-src{}-dst{}-tok{}-pos{}-kv{}-ctx{}-act{}-ngpu{}-nb{}-nub{}-pi{}.skippy-frame",
        source_layer_start,
        source_layer_end,
        args.tokens,
        args.position_start,
        args.kv_warmup_tokens,
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

fn poison_top_k_sideband(
    args: &GlmDsaLayerMicrobenchArgs,
    frame: &ActivationFrame,
) -> Result<PoisonedTopKFrame> {
    let hidden_bytes = hidden_payload_bytes(args)?;
    if frame.payload.len() <= hidden_bytes {
        bail!("cannot poison GLM-DSA top-k sideband: frame has no sideband payload");
    }
    let sideband = &frame.payload[hidden_bytes..];
    if !sideband.len().is_multiple_of(std::mem::size_of::<i32>()) {
        bail!("cannot poison GLM-DSA top-k sideband: payload is not i32-aligned");
    }
    let sideband_i32_count = sideband.len() / std::mem::size_of::<i32>();
    let sideband_i32_per_token = sideband_i32_per_token(sideband_i32_count, args.tokens)
        .context("cannot poison GLM-DSA top-k sideband without per-token width")?;
    if sideband_i32_per_token < 2 {
        bail!("cannot poison GLM-DSA top-k sideband: width {sideband_i32_per_token} is too small");
    }

    let mut poisoned = frame.clone();
    let mut changed_i32_count = 0usize;
    let mut tokens_with_changes = 0usize;
    let mut original_unique_index_count = 0usize;
    {
        let sideband_bytes = &mut poisoned.payload[hidden_bytes..];
        for token_index in 0..args.tokens {
            let token_start = token_index
                .checked_mul(sideband_i32_per_token)
                .context("sideband token start overflow")?;
            let token_end = token_start
                .checked_add(sideband_i32_per_token)
                .context("sideband token end overflow")?;
            let token_values = read_sideband_i32_slice(sideband_bytes, token_start, token_end)?;
            let unique: HashSet<_> = token_values.iter().copied().collect();
            original_unique_index_count += unique.len();
            if unique.len() < 2 {
                continue;
            }

            let replacement = token_values[0];
            let mut token_changed = false;
            for (offset, original) in token_values.iter().copied().enumerate().skip(1) {
                if original == replacement {
                    continue;
                }
                write_sideband_i32(sideband_bytes, token_start + offset, replacement)?;
                changed_i32_count += 1;
                token_changed = true;
            }
            if token_changed {
                tokens_with_changes += 1;
            }
        }
    }

    if changed_i32_count == 0 {
        bail!("cannot poison GLM-DSA top-k sideband: no mutable top-k entries changed");
    }

    Ok(PoisonedTopKFrame {
        frame: poisoned,
        report: SidebandPoisonReport {
            poison_kind: "collapse_each_token_to_first_index",
            sideband_i32_count,
            sideband_i32_per_token,
            changed_i32_count,
            tokens_with_changes,
            original_unique_index_count,
        },
    })
}

fn read_sideband_i32_slice(sideband: &[u8], start_i32: usize, end_i32: usize) -> Result<Vec<i32>> {
    (start_i32..end_i32)
        .map(|index| {
            let byte_start = index
                .checked_mul(std::mem::size_of::<i32>())
                .context("sideband byte start overflow")?;
            let byte_end = byte_start
                .checked_add(std::mem::size_of::<i32>())
                .context("sideband byte end overflow")?;
            let bytes = sideband
                .get(byte_start..byte_end)
                .with_context(|| format!("read GLM-DSA sideband i32 at {index}"))?;
            Ok(i32::from_le_bytes(bytes.try_into().with_context(|| {
                format!("decode GLM-DSA sideband i32 at {index}")
            })?))
        })
        .collect()
}

fn write_sideband_i32(sideband: &mut [u8], index: usize, value: i32) -> Result<()> {
    let byte_start = index
        .checked_mul(std::mem::size_of::<i32>())
        .context("sideband byte start overflow")?;
    let byte_end = byte_start
        .checked_add(std::mem::size_of::<i32>())
        .context("sideband byte end overflow")?;
    let dst = sideband
        .get_mut(byte_start..byte_end)
        .with_context(|| format!("write GLM-DSA sideband i32 at {index}"))?;
    dst.copy_from_slice(&value.to_le_bytes());
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
        token_count: args.tokens,
        hidden_bytes,
        sideband_bytes: sideband.len(),
        sideband_i32_count: values.len(),
        sideband_i32_per_token: sideband_i32_per_token(values.len(), args.tokens),
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
            top_k_width: sideband.sideband_i32_per_token,
        },
    }
}

fn synthetic_activation_frame_for_layer(
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    top_k_sideband: Option<SyntheticTopKSideband>,
) -> Result<ActivationFrame> {
    synthetic_activation_frame_for_layer_tokens(args, layer_start, args.tokens, top_k_sideband)
}

fn synthetic_activation_frame_for_layer_tokens(
    args: &GlmDsaLayerMicrobenchArgs,
    layer_start: u32,
    tokens: usize,
    top_k_sideband: Option<SyntheticTopKSideband>,
) -> Result<ActivationFrame> {
    let width = usize::try_from(args.activation_width).context("activation_width exceeds usize")?;
    let value_count = tokens
        .checked_mul(width)
        .context("synthetic activation value count overflow")?;
    let payload_bytes = value_count
        .checked_mul(std::mem::size_of::<f32>())
        .context("synthetic activation payload size overflow")?;
    let sideband_bytes = top_k_sideband
        .as_ref()
        .map(|sideband| synthetic_top_k_sideband_bytes(tokens, sideband.width))
        .transpose()?
        .unwrap_or(0)
        .checked_add(malformed_top_k_sideband_bytes()?)
        .context("synthetic malformed GLM-DSA sideband size overflow")?;
    let mut payload = Vec::with_capacity(payload_bytes);
    for token in 0..tokens {
        for dim in 0..width {
            let value = synthetic_activation_value(token, dim);
            payload.extend_from_slice(&value.to_ne_bytes());
        }
    }
    if let Some(sideband) = top_k_sideband {
        append_synthetic_top_k_sideband(&mut payload, tokens, sideband.width)?;
    }
    append_malformed_top_k_sideband(&mut payload)?;
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
            token_count: u32::try_from(tokens).context("tokens exceeds u32")?,
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

fn malformed_top_k_sideband_bytes() -> Result<usize> {
    let Ok(value) = std::env::var(ENV_MALFORMED_TOP_K_BYTES) else {
        return Ok(0);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "0" || trimmed.eq_ignore_ascii_case("false") {
        return Ok(0);
    }
    trimmed
        .parse::<usize>()
        .with_context(|| format!("parse {ENV_MALFORMED_TOP_K_BYTES}"))
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

fn allow_compact_flash_auto(args: &GlmDsaLayerMicrobenchArgs) -> bool {
    args.allow_compact_flash_auto || env_flag_enabled(ENV_ALLOW_COMPACT_FLASH_AUTO)
}

fn synthetic_top_k_sideband_bytes(tokens: usize, width: usize) -> Result<usize> {
    tokens
        .checked_mul(width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
        .context("synthetic GLM-DSA top-k sideband size overflow")
}

fn sideband_i32_per_token(sideband_i32_count: usize, token_count: usize) -> Option<usize> {
    if token_count == 0 || !sideband_i32_count.is_multiple_of(token_count) {
        return None;
    }
    Some(sideband_i32_count / token_count)
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

fn append_malformed_top_k_sideband(payload: &mut Vec<u8>) -> Result<()> {
    let bytes = malformed_top_k_sideband_bytes()?;
    payload.reserve(bytes);
    for byte in 0..bytes {
        payload.push((byte & 0xff) as u8);
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

struct PoisonedTopKFrame {
    frame: ActivationFrame,
    report: SidebandPoisonReport,
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
            compact_flash_policy_records: parse_compact_flash_policy_records(&text)
                .context("parse native compact flash policy records")?,
            compact_flash_mask_records: parse_compact_flash_mask_records(&text)
                .context("parse native compact flash mask records")?,
            direct_sparse_decision_records: parse_direct_sparse_decision_records(&text)
                .context("parse native direct sparse decisions")?,
            metal_dispatch_records: parse_metal_dispatch_records(&text)
                .context("parse native Metal dispatch records")?,
            op_timing_records: parse_timing_records(&text).context("parse native op timings")?,
            group_timing_records: parse_timing_group_records(&text)
                .context("parse native group timings")?,
            indexshare_trace_summary: summarize_indexshare_trace_records(
                &parse_indexshare_trace_records(&text).context("parse native IndexShare trace")?,
                &parse_indexshare_contract_records(&text)
                    .context("parse native IndexShare contract trace")?,
            ),
            hot_tensor_records: parse_hot_tensor_records(&text)
                .context("parse native hot tensor timings")?,
            compute_buffer_records: parse_compute_buffer_records(&text)
                .context("parse native compute buffer records")?,
        })
    }

    fn finish_after_open_error(mut self) -> String {
        let Some(path) = self.path.clone() else {
            return String::new();
        };
        restore_native_logs();
        self.active = false;
        match fs::read_to_string(&path) {
            Ok(text) => format!(
                "; native log: {}; tail:\n{}",
                path.display(),
                tail_lines(&text, 80)
            ),
            Err(error) => format!("; native log: {}; read failed: {error}", path.display()),
        }
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
    compact_flash_policy_records: Vec<CompactFlashPolicyRecord>,
    compact_flash_mask_records: Vec<CompactFlashMaskRecord>,
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    op_timing_records: Vec<TimingRecord>,
    group_timing_records: Vec<TimingGroupRecord>,
    indexshare_trace_summary: IndexShareTraceSummary,
    hot_tensor_records: Vec<HotTensorRecord>,
    compute_buffer_records: Vec<ComputeBufferRecord>,
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
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

fn retain_measured_timing_records(
    records: Vec<TimingRecord>,
    measured_tokens: usize,
    warmup: usize,
) -> Vec<TimingRecord> {
    let measured_tokens = measured_tokens as u64;
    records
        .into_iter()
        .filter(|record| record.tokens == measured_tokens)
        .skip(warmup)
        .collect()
}

fn retain_measured_group_timing_records(
    records: Vec<TimingGroupRecord>,
    measured_tokens: usize,
    warmup: usize,
) -> Vec<TimingGroupRecord> {
    let measured_tokens = measured_tokens as u64;
    let records: Vec<_> = records
        .into_iter()
        .filter(|record| record.timing.tokens == measured_tokens)
        .collect();
    let record_index_map =
        measured_record_index_map(records.iter().map(|record| record.record_index), warmup);
    records
        .into_iter()
        .filter_map(|mut record| {
            record.record_index = *record_index_map.get(&record.record_index)?;
            Some(record)
        })
        .collect()
}

fn retain_measured_hot_tensor_records(
    records: Vec<HotTensorRecord>,
    measured_tokens: usize,
    warmup: usize,
) -> Vec<HotTensorRecord> {
    let measured_tokens = measured_tokens as u64;
    let records: Vec<_> = records
        .into_iter()
        .filter(|record| record.tokens == measured_tokens)
        .collect();
    let record_index_map =
        measured_record_index_map(records.iter().map(|record| record.record_index), warmup);
    records
        .into_iter()
        .filter_map(|mut record| {
            record.record_index = *record_index_map.get(&record.record_index)?;
            Some(record)
        })
        .collect()
}

fn measured_record_index_map(
    indices: impl Iterator<Item = usize>,
    warmup: usize,
) -> HashMap<usize, usize> {
    let mut unique_indices: Vec<_> = indices.collect();
    unique_indices.sort_unstable();
    unique_indices.dedup();
    unique_indices
        .into_iter()
        .skip(warmup)
        .enumerate()
        .map(|(new_index, old_index)| (old_index, new_index))
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

fn retain_case_compact_policy_records(
    records: Vec<CompactFlashPolicyRecord>,
    tokens: usize,
) -> Vec<CompactFlashPolicyRecord> {
    let Ok(tokens) = i64::try_from(tokens) else {
        return Vec::new();
    };
    records
        .into_iter()
        .filter(|record| record.ubatch_tokens == tokens)
        .collect()
}

fn retain_case_compact_mask_records(
    records: Vec<CompactFlashMaskRecord>,
    tokens: usize,
) -> Vec<CompactFlashMaskRecord> {
    let Ok(tokens) = i64::try_from(tokens) else {
        return Vec::new();
    };
    records
        .into_iter()
        .filter(|record| record.ubatch_tokens == tokens)
        .collect()
}

fn split_execution_decision_records(
    records: Vec<DirectSparseDecisionRecord>,
    measured_iterations: usize,
) -> (
    Vec<DirectSparseDecisionRecord>,
    Vec<DirectSparseDecisionRecord>,
) {
    if records.is_empty() || measured_iterations == 0 {
        return (records, Vec::new());
    }
    let execution_count = records.len().min(measured_iterations);
    let split_at = records.len() - execution_count;
    let non_measured = records[..split_at].to_vec();
    let execution = records[split_at..].to_vec();
    (non_measured, execution)
}

fn split_execution_compact_policy_records(
    records: Vec<CompactFlashPolicyRecord>,
    measured_iterations: usize,
) -> (Vec<CompactFlashPolicyRecord>, Vec<CompactFlashPolicyRecord>) {
    if records.is_empty() || measured_iterations == 0 {
        return (records, Vec::new());
    }
    let execution_count = records.len().min(measured_iterations);
    let split_at = records.len() - execution_count;
    let non_measured = records[..split_at].to_vec();
    let execution = records[split_at..].to_vec();
    (non_measured, execution)
}

fn split_execution_compact_mask_records(
    records: Vec<CompactFlashMaskRecord>,
    measured_iterations: usize,
) -> (Vec<CompactFlashMaskRecord>, Vec<CompactFlashMaskRecord>) {
    if records.is_empty() || measured_iterations == 0 {
        return (records, Vec::new());
    }
    let execution_count = records.len().min(measured_iterations);
    let split_at = records.len() - execution_count;
    let non_measured = records[..split_at].to_vec();
    let execution = records[split_at..].to_vec();
    (non_measured, execution)
}

fn summarize_direct_sparse_decisions(case: &MicrobenchCaseSummary) -> DirectSparseDecisionSummary {
    let mut summary = DirectSparseDecisionSummary {
        records: case.direct_sparse_decision_records.len(),
        execution_records: case.direct_sparse_execution_decision_records.len(),
        non_measured_records: case.direct_sparse_non_measured_decision_records.len(),
        ..DirectSparseDecisionSummary::default()
    };
    summarize_direct_sparse_decision_slice(
        &case.direct_sparse_decision_records,
        &mut summary,
        false,
    );
    summarize_direct_sparse_decision_slice(
        &case.direct_sparse_execution_decision_records,
        &mut summary,
        true,
    );
    summary
}

fn summarize_direct_sparse_decision_slice(
    records: &[DirectSparseDecisionRecord],
    summary: &mut DirectSparseDecisionSummary,
    execution: bool,
) {
    for record in records {
        if execution {
            if record.use_direct {
                summary.execution_use_direct += 1;
            } else {
                summary.execution_fallback += 1;
            }
        } else if record.use_direct {
            summary.use_direct += 1;
        } else {
            summary.fallback += 1;
        }
        if execution {
            *summary
                .execution_phases
                .entry(
                    record
                        .phase
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                )
                .or_default() += 1;
            *summary
                .execution_selector_reasons
                .entry(
                    record
                        .selector_reason
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                )
                .or_default() += 1;
            continue;
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
        *summary
            .phases
            .entry(
                record
                    .phase
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
            )
            .or_default() += 1;
    }
}

fn summarize_compact_flash_policy(case: &MicrobenchCaseSummary) -> CompactFlashPolicySummary {
    let mut summary = CompactFlashPolicySummary {
        records: case.compact_flash_policy_records.len(),
        execution_records: case.compact_flash_execution_policy_records.len(),
        non_measured_records: case.compact_flash_non_measured_policy_records.len(),
        ..CompactFlashPolicySummary::default()
    };
    summarize_compact_flash_policy_slice(&case.compact_flash_policy_records, &mut summary, false);
    summarize_compact_flash_policy_slice(
        &case.compact_flash_execution_policy_records,
        &mut summary,
        true,
    );
    summary
}

fn summarize_compact_flash_policy_slice(
    records: &[CompactFlashPolicyRecord],
    summary: &mut CompactFlashPolicySummary,
    execution: bool,
) {
    for record in records {
        if execution {
            if record.use_compact {
                summary.execution_use_compact += 1;
            } else {
                summary.execution_fallback += 1;
            }
        } else if record.use_compact {
            summary.use_compact += 1;
        } else {
            summary.fallback += 1;
        }
        if record.forced {
            summary.forced += 1;
        }
        if record.disabled {
            summary.disabled += 1;
        }
        if record.ratio_ok {
            summary.ratio_ok += 1;
        }
        if record.enabled {
            summary.enabled += 1;
        }
        if record.flash_attn {
            summary.flash_attn += 1;
        }
        if record.decode_shape {
            summary.decode_shape += 1;
        }
        if record.no_mask == Some(true) {
            summary.no_mask += 1;
        }
        if let Some(phase) = &record.phase {
            *summary.phases.entry(phase.clone()).or_default() += 1;
        }
        if execution && let Some(reason) = &record.selector_reason {
            *summary
                .execution_selector_reasons
                .entry(reason.clone())
                .or_default() += 1;
        }
    }
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
        sideband_token_count(baseline, candidate),
    )?;
    let output_flags_match = baseline.desc.flags == candidate.desc.flags;
    let payload_len_match = baseline.payload.len() == candidate.payload.len();
    let passed = output_flags_match
        && payload_len_match
        && hidden.mismatches == 0
        && sideband.semantic_match;
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
        sideband_exact_match: sideband.exact_match,
        sideband_semantic_match: sideband.semantic_match,
        sideband_bytes: sideband.compared_bytes,
        sideband_mismatched_bytes: sideband.mismatched_bytes,
        first_sideband_mismatch: sideband.first_mismatch,
        sideband_i32_diff: sideband.i32_diff,
    })
}

fn sideband_token_count(baseline: &ActivationFrame, candidate: &ActivationFrame) -> Option<usize> {
    let baseline_tokens = usize::try_from(baseline.desc.token_count).ok()?;
    let candidate_tokens = usize::try_from(candidate.desc.token_count).ok()?;
    (baseline_tokens == candidate_tokens && baseline_tokens > 0).then_some(baseline_tokens)
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

fn compare_sideband_payloads(
    baseline: &[u8],
    candidate: &[u8],
    token_count: Option<usize>,
) -> Result<SidebandComparison> {
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
    let i32_diff = compare_sideband_i32_payloads(baseline, candidate, token_count)?;
    let exact_match = mismatched_bytes == 0;
    let semantic_match = sideband_semantic_match(exact_match, &i32_diff);
    Ok(SidebandComparison {
        compared_bytes,
        mismatched_bytes,
        first_mismatch,
        exact_match,
        semantic_match,
        i32_diff,
    })
}

fn sideband_semantic_match(exact_match: bool, i32_diff: &Option<SidebandI32Diff>) -> bool {
    if exact_match {
        return true;
    }
    let Some(diff) = i32_diff else {
        return false;
    };
    if !diff.i32_aligned || diff.baseline_i32_count != diff.candidate_i32_count {
        return false;
    }
    diff.token_summary
        .as_ref()
        .is_some_and(|summary| summary.set_mismatched_tokens == 0)
}

fn compare_sideband_i32_payloads(
    baseline: &[u8],
    candidate: &[u8],
    token_count: Option<usize>,
) -> Result<Option<SidebandI32Diff>> {
    if baseline.is_empty() && candidate.is_empty() {
        return Ok(None);
    }
    let i32_aligned = baseline.len().is_multiple_of(std::mem::size_of::<i32>())
        && candidate.len().is_multiple_of(std::mem::size_of::<i32>());
    if !i32_aligned {
        return Ok(Some(SidebandI32Diff::unaligned(
            baseline.len(),
            candidate.len(),
        )));
    }
    let baseline_values = decode_i32_sideband(baseline)?;
    let candidate_values = decode_i32_sideband(candidate)?;
    let compared_i32 = baseline_values.len().min(candidate_values.len());
    let mut mismatched_i32 = baseline_values.len().abs_diff(candidate_values.len());
    let mut first_mismatches = Vec::new();
    for (index, (baseline, candidate)) in baseline_values
        .iter()
        .zip(candidate_values.iter())
        .enumerate()
    {
        if baseline != candidate {
            mismatched_i32 += 1;
            if first_mismatches.len() < SIDEBAND_DIFF_SAMPLE_LIMIT {
                first_mismatches.push(SidebandI32Mismatch {
                    index,
                    token_index: token_index_for_sideband(index, compared_i32, token_count),
                    offset_in_token: offset_in_token_for_sideband(index, compared_i32, token_count),
                    baseline: *baseline,
                    candidate: *candidate,
                });
            }
        }
    }
    let token_summary = token_count.and_then(|tokens| {
        sideband_token_diff_summary(&baseline_values, &candidate_values, tokens)
    });
    Ok(Some(SidebandI32Diff {
        i32_aligned,
        baseline_i32_count: baseline_values.len(),
        candidate_i32_count: candidate_values.len(),
        compared_i32,
        mismatched_i32,
        first_mismatches,
        token_summary,
    }))
}

fn token_index_for_sideband(
    index: usize,
    compared_i32: usize,
    token_count: Option<usize>,
) -> Option<usize> {
    sideband_width(compared_i32, token_count).map(|width| index / width)
}

fn offset_in_token_for_sideband(
    index: usize,
    compared_i32: usize,
    token_count: Option<usize>,
) -> Option<usize> {
    sideband_width(compared_i32, token_count).map(|width| index % width)
}

fn sideband_width(compared_i32: usize, token_count: Option<usize>) -> Option<usize> {
    let tokens = token_count?;
    (tokens > 0 && compared_i32.is_multiple_of(tokens)).then_some(compared_i32 / tokens)
}

fn sideband_token_diff_summary(
    baseline: &[i32],
    candidate: &[i32],
    token_count: usize,
) -> Option<SidebandTokenDiffSummary> {
    if token_count == 0
        || baseline.len() != candidate.len()
        || !baseline.len().is_multiple_of(token_count)
    {
        return None;
    }
    let width = baseline.len() / token_count;
    let mut exact_order_matching_tokens = 0usize;
    let mut set_equivalent_tokens = 0usize;
    let mut set_mismatched_tokens = 0usize;
    let mut first_set_mismatch = None;
    for token_index in 0..token_count {
        let start = token_index * width;
        let end = start + width;
        let baseline_token = &baseline[start..end];
        let candidate_token = &candidate[start..end];
        if baseline_token == candidate_token {
            exact_order_matching_tokens += 1;
            set_equivalent_tokens += 1;
            continue;
        }
        let set_diff = sideband_token_set_diff(baseline_token, candidate_token);
        if set_diff.set_equivalent {
            set_equivalent_tokens += 1;
        } else {
            set_mismatched_tokens += 1;
            first_set_mismatch.get_or_insert(SidebandTokenSetMismatch {
                token_index,
                baseline_only: set_diff.baseline_only,
                candidate_only: set_diff.candidate_only,
            });
        }
    }
    Some(SidebandTokenDiffSummary {
        token_count,
        width,
        exact_order_matching_tokens,
        set_equivalent_tokens,
        set_mismatched_tokens,
        first_set_mismatch,
    })
}

fn sideband_token_set_diff(baseline: &[i32], candidate: &[i32]) -> SidebandTokenSetDiff {
    let mut baseline_sorted = baseline.to_vec();
    let mut candidate_sorted = candidate.to_vec();
    baseline_sorted.sort_unstable();
    candidate_sorted.sort_unstable();
    let set_equivalent = baseline_sorted == candidate_sorted;
    if set_equivalent {
        return SidebandTokenSetDiff {
            set_equivalent,
            baseline_only: Vec::new(),
            candidate_only: Vec::new(),
        };
    }
    let (baseline_only, candidate_only) = sorted_multiset_diff(
        &baseline_sorted,
        &candidate_sorted,
        SIDEBAND_TOKEN_DIFF_SAMPLE_LIMIT,
    );
    SidebandTokenSetDiff {
        set_equivalent,
        baseline_only,
        candidate_only,
    }
}

fn sorted_multiset_diff(baseline: &[i32], candidate: &[i32], limit: usize) -> (Vec<i32>, Vec<i32>) {
    let mut baseline_only = Vec::new();
    let mut candidate_only = Vec::new();
    let mut baseline_index = 0usize;
    let mut candidate_index = 0usize;
    while baseline_index < baseline.len() || candidate_index < candidate.len() {
        match (baseline.get(baseline_index), candidate.get(candidate_index)) {
            (Some(baseline_value), Some(candidate_value)) if baseline_value == candidate_value => {
                baseline_index += 1;
                candidate_index += 1;
            }
            (Some(baseline_value), Some(candidate_value)) if baseline_value < candidate_value => {
                if baseline_only.len() < limit {
                    baseline_only.push(*baseline_value);
                }
                baseline_index += 1;
            }
            (Some(_), Some(candidate_value)) => {
                if candidate_only.len() < limit {
                    candidate_only.push(*candidate_value);
                }
                candidate_index += 1;
            }
            (Some(baseline_value), None) => {
                if baseline_only.len() < limit {
                    baseline_only.push(*baseline_value);
                }
                baseline_index += 1;
            }
            (None, Some(candidate_value)) => {
                if candidate_only.len() < limit {
                    candidate_only.push(*candidate_value);
                }
                candidate_index += 1;
            }
            (None, None) => break,
        }
    }
    (baseline_only, candidate_only)
}

struct MicrobenchCase {
    label: &'static str,
    flags: MicrobenchFlags,
    n_gpu_layers: i32,
    measured_tokens: usize,
    native_log_path: Option<PathBuf>,
    compact_flash_policy_records: Vec<CompactFlashPolicyRecord>,
    compact_flash_execution_policy_records: Vec<CompactFlashPolicyRecord>,
    compact_flash_non_measured_policy_records: Vec<CompactFlashPolicyRecord>,
    compact_flash_mask_records: Vec<CompactFlashMaskRecord>,
    compact_flash_execution_mask_records: Vec<CompactFlashMaskRecord>,
    compact_flash_non_measured_mask_records: Vec<CompactFlashMaskRecord>,
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    direct_sparse_execution_decision_records: Vec<DirectSparseDecisionRecord>,
    direct_sparse_non_measured_decision_records: Vec<DirectSparseDecisionRecord>,
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    op_timing_records: Vec<TimingRecord>,
    group_timing_records: Vec<TimingGroupRecord>,
    indexshare_trace_summary: IndexShareTraceSummary,
    hot_tensor_records: Vec<HotTensorRecord>,
    compute_buffer_records: Vec<ComputeBufferRecord>,
    timings: Vec<IterationTiming>,
    outputs: Vec<ActivationFrame>,
}

impl MicrobenchCase {
    fn as_case_summary(&self) -> MicrobenchCaseSummary {
        let mut summary = MicrobenchCaseSummary {
            label: self.label,
            flags: self.flags,
            n_gpu_layers: self.n_gpu_layers,
            native_log_path: self.native_log_path.clone(),
            compact_flash_policy_summary: CompactFlashPolicySummary::default(),
            compact_flash_policy_records: self.compact_flash_policy_records.clone(),
            compact_flash_execution_policy_records: self
                .compact_flash_execution_policy_records
                .clone(),
            compact_flash_non_measured_policy_records: self
                .compact_flash_non_measured_policy_records
                .clone(),
            compact_flash_mask_records: self.compact_flash_mask_records.clone(),
            compact_flash_execution_mask_records: self.compact_flash_execution_mask_records.clone(),
            compact_flash_non_measured_mask_records: self
                .compact_flash_non_measured_mask_records
                .clone(),
            direct_sparse_decision_summary: DirectSparseDecisionSummary::default(),
            timing_summary: summarize_elapsed_ms(
                self.timings.iter().map(|timing| timing.elapsed_ms),
            ),
            timing_breakdown: summarize_timing_breakdown(&self.timings, self.measured_tokens),
            metal_dispatch_summary: summarize_metal_dispatch(&self.metal_dispatch_records),
            direct_sparse_spill_summary: summarize_direct_sparse_spill(
                &self.metal_dispatch_records,
            ),
            op_timing_summary: summarize_glm_dsa_op_timing(&self.op_timing_records),
            routed_moe_timing_summary: summarize_routed_moe_timing(&self.op_timing_records),
            indexshare_timing_summary: summarize_indexshare_timing(&self.group_timing_records),
            indexshare_trace_summary: self.indexshare_trace_summary.clone(),
            direct_sparse_decision_records: self.direct_sparse_decision_records.clone(),
            direct_sparse_execution_decision_records: self
                .direct_sparse_execution_decision_records
                .clone(),
            direct_sparse_non_measured_decision_records: self
                .direct_sparse_non_measured_decision_records
                .clone(),
            metal_dispatch_records: self.metal_dispatch_records.clone(),
            op_timing_records: self.op_timing_records.clone(),
            group_timing_records: self.group_timing_records.clone(),
            hot_tensor_records: self.hot_tensor_records.clone(),
            compute_buffer_records: self.compute_buffer_records.clone(),
            timings: self.timings.clone(),
        };
        summary.compact_flash_policy_summary = summarize_compact_flash_policy(&summary);
        summary.direct_sparse_decision_summary = summarize_direct_sparse_decisions(&summary);
        summary
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
    exact_match: bool,
    semantic_match: bool,
    i32_diff: Option<SidebandI32Diff>,
}

#[derive(Clone, Serialize)]
struct SidebandI32Diff {
    i32_aligned: bool,
    baseline_i32_count: usize,
    candidate_i32_count: usize,
    compared_i32: usize,
    mismatched_i32: usize,
    first_mismatches: Vec<SidebandI32Mismatch>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_summary: Option<SidebandTokenDiffSummary>,
}

impl SidebandI32Diff {
    fn unaligned(baseline_bytes: usize, candidate_bytes: usize) -> Self {
        Self {
            i32_aligned: false,
            baseline_i32_count: baseline_bytes / std::mem::size_of::<i32>(),
            candidate_i32_count: candidate_bytes / std::mem::size_of::<i32>(),
            compared_i32: baseline_bytes.min(candidate_bytes) / std::mem::size_of::<i32>(),
            mismatched_i32: baseline_bytes.abs_diff(candidate_bytes) / std::mem::size_of::<i32>(),
            first_mismatches: Vec::new(),
            token_summary: None,
        }
    }
}

#[derive(Clone, Serialize)]
struct SidebandI32Mismatch {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    offset_in_token: Option<usize>,
    baseline: i32,
    candidate: i32,
}

#[derive(Clone, Serialize)]
struct SidebandTokenDiffSummary {
    token_count: usize,
    width: usize,
    exact_order_matching_tokens: usize,
    set_equivalent_tokens: usize,
    set_mismatched_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_set_mismatch: Option<SidebandTokenSetMismatch>,
}

#[derive(Clone, Serialize)]
struct SidebandTokenSetMismatch {
    token_index: usize,
    baseline_only: Vec<i32>,
    candidate_only: Vec<i32>,
}

struct SidebandTokenSetDiff {
    set_equivalent: bool,
    baseline_only: Vec<i32>,
    candidate_only: Vec<i32>,
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
    kv_warmup_tokens: usize,
    kv_warmup_chunk_tokens: usize,
    synthetic_kv_warmup: bool,
    reuse_kv_warmup_checkpoint: bool,
    reuse_kv_warmup_stream: bool,
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
    #[serde(skip_serializing_if = "CompactFlashPolicySummary::is_empty")]
    compact_flash_policy_summary: CompactFlashPolicySummary,
    #[serde(skip_serializing_if = "DirectSparseDecisionSummary::is_empty")]
    direct_sparse_decision_summary: DirectSparseDecisionSummary,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    timing_summary: TimingDistributionSummary,
    #[serde(skip_serializing_if = "TimingBreakdownSummary::is_empty")]
    timing_breakdown: TimingBreakdownSummary,
    #[serde(skip_serializing_if = "GlmDsaDispatchSummary::is_empty")]
    metal_dispatch_summary: GlmDsaDispatchSummary,
    #[serde(skip_serializing_if = "DirectSparseSpillSummary::is_empty")]
    direct_sparse_spill_summary: DirectSparseSpillSummary,
    #[serde(skip_serializing_if = "GlmDsaOpTimingSummary::is_empty")]
    op_timing_summary: GlmDsaOpTimingSummary,
    #[serde(skip_serializing_if = "RoutedMoeTimingSummary::is_empty")]
    routed_moe_timing_summary: RoutedMoeTimingSummary,
    #[serde(skip_serializing_if = "IndexShareTimingSummary::is_empty")]
    indexshare_timing_summary: IndexShareTimingSummary,
    #[serde(skip_serializing_if = "IndexShareTraceSummary::is_empty")]
    indexshare_trace_summary: IndexShareTraceSummary,
    representative_profile: RepresentativeProfileReport,
    profile_integrity: ProfileIntegrityReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    route_fusion_guard: Option<RouteFusionGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    direct_sparse_prefill_guard: Option<DirectSparsePrefillGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    direct_sparse_decode_guard: Option<DirectSparseDecodeGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    partial_top_k_guard: Option<PartialTopKGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compact_flash_guard: Option<CompactFlashGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_weighted_sum_guard: Option<MoeWeightedSumGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_motif_guard: Option<MoeMotifGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_indexshare_guard: Option<NativeIndexShareGuardReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    real_top_k_shared_consumer_guard: Option<RealTopKSharedConsumerGuardReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_policy_records: Vec<CompactFlashPolicyRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_execution_policy_records: Vec<CompactFlashPolicyRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_non_measured_policy_records: Vec<CompactFlashPolicyRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_mask_records: Vec<CompactFlashMaskRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_execution_mask_records: Vec<CompactFlashMaskRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_non_measured_mask_records: Vec<CompactFlashMaskRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_execution_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_non_measured_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    op_timing_records: Vec<TimingRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    group_timing_records: Vec<TimingGroupRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    hot_tensor_records: Vec<HotTensorRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compute_buffer_records: Vec<ComputeBufferRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_dispatch_probe: Option<MicrobenchCaseSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    optimized_dispatch_probe_parity: Option<ParityComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<MicrobenchComparisonReport>,
    timings: Vec<IterationTiming>,
}

#[derive(Clone, Default, Serialize)]
struct TimingBreakdownSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    measured_phase: Option<&'static str>,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    decode: TimingDistributionSummary,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    prefill: TimingDistributionSummary,
}

impl TimingBreakdownSummary {
    fn is_empty(summary: &Self) -> bool {
        TimingDistributionSummary::is_empty(&summary.decode)
            && TimingDistributionSummary::is_empty(&summary.prefill)
    }
}

fn summarize_timing_breakdown(
    timings: &[IterationTiming],
    tokens: usize,
) -> TimingBreakdownSummary {
    let measured = summarize_elapsed_ms(timings.iter().map(|timing| timing.elapsed_ms));
    if TimingDistributionSummary::is_empty(&measured) {
        return TimingBreakdownSummary::default();
    }
    match measured_timing_phase(tokens) {
        "decode" => TimingBreakdownSummary {
            measured_phase: Some("decode"),
            decode: measured,
            prefill: TimingDistributionSummary::default(),
        },
        "prefill" => TimingBreakdownSummary {
            measured_phase: Some("prefill"),
            decode: TimingDistributionSummary::default(),
            prefill: measured,
        },
        _ => unreachable!("measured timing phase is exhaustive"),
    }
}

fn measured_timing_phase(tokens: usize) -> &'static str {
    if tokens == 1 { "decode" } else { "prefill" }
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
struct DirectSparsePrefillGuardReport {
    checked_case: &'static str,
    passed: bool,
    direct_decision_records: usize,
    direct_use_records: usize,
    fallback_records: usize,
    prefill_decisions: usize,
    prefill_direct_decisions: usize,
    large_prefill_decisions: usize,
    large_prefill_direct_decisions: usize,
    sparse_mask_nodes: u64,
    dense_sparse_mask_dispatches: usize,
    dsa_sparse_attn_nodes: u64,
    dsa_sparse_attn_dispatches: usize,
    accepted_prefill_dispatches: usize,
    cached_topk_non_prefill_dispatches: usize,
    accepted_large_prefill_dispatches: usize,
    default_prefill_threads_x: u64,
    max_cached_topk_prefill_threads_x: u64,
    failure_summary: String,
}

#[derive(Serialize)]
struct DirectSparseDecodeGuardReport {
    checked_case: &'static str,
    passed: bool,
    direct_decision_records: usize,
    direct_use_records: usize,
    fallback_records: usize,
    decode_decisions: usize,
    decode_direct_decisions: usize,
    sparse_mask_nodes: u64,
    dense_sparse_mask_dispatches: usize,
    dsa_sparse_attn_nodes: u64,
    dsa_sparse_attn_dispatches: usize,
    accepted_decode_dispatches: usize,
    failure_summary: String,
}

#[derive(Serialize)]
struct PartialTopKGuardReport {
    checked_case: &'static str,
    passed: bool,
    dsa_sparse_attn_dispatches: usize,
    partial_dsa_sparse_attn_dispatches: usize,
    max_kv: Option<u64>,
    min_partial_top_k: Option<u64>,
    max_partial_top_k: Option<u64>,
    output_frames: usize,
    output_frames_with_sideband: usize,
    output_frames_with_expected_sideband: usize,
    expected_sideband_i32_per_token: Option<usize>,
    max_observed_sideband_i32_per_token: Option<usize>,
    native_shared_consume_records: usize,
    native_shared_consume_width_matches: usize,
    failure_summary: String,
}

#[derive(Serialize)]
struct CompactFlashGuardReport {
    checked_case: &'static str,
    passed: bool,
    flash_attn_ext_records: usize,
    flash_attn_ext_vec_records: usize,
    flash_attn_ext_tile_records: usize,
    flash_attn_ext_glm_dsa_shape_records: usize,
    get_rows_records: usize,
    get_rows_typed_records: usize,
    get_rows_promote_records: usize,
    compact_get_rows_records: usize,
    compact_get_rows_typed_records: usize,
    compact_get_rows_promote_records: usize,
    dsa_compact_get_rows_fused_records: usize,
    dsa_sparse_attn_records: usize,
    sparse_mask_nodes: u64,
    mask_omission_records: usize,
    execution_mask_omission_records: usize,
    omitted_mla_kq_mask_records: usize,
    materialized_mla_kq_mask_records: usize,
    policy_phase: Option<String>,
    policy_selector_reason: Option<String>,
    failure_summary: String,
}

#[derive(Serialize)]
struct MoeWeightedSumGuardReport {
    checked_case: &'static str,
    passed: bool,
    moe_weighted_sum_records: usize,
    moe_weighted_sum_f32x4_records: usize,
    moe_weighted_sum_already_weighted_records: usize,
    mul_mv_id_weighted_sum_fused_records: usize,
    mul_mv_id_weighted_sum_fused_q3_k_records: usize,
    mul_mv_id_weighted_slots_records: usize,
    mul_mv_id_weighted_slots_q3_k_records: usize,
    failure_summary: String,
}

#[derive(Serialize)]
struct MoeMotifGuardReport {
    checked_case: &'static str,
    passed: bool,
    motif_candidate_records: usize,
    natural_order_records: usize,
    backend_candidate_records: usize,
    subgraph_fusable_records: usize,
    coencoded_records: usize,
    max_motif_nodes: u64,
    failure_summary: String,
}

#[derive(Serialize)]
struct NativeIndexShareGuardReport {
    checked_case: &'static str,
    passed: bool,
    records: usize,
    exec_records: usize,
    full_exec_records: usize,
    shared_exec_records: usize,
    shared_exec_with_input_top_k: usize,
    shared_exec_missing_input_top_k: usize,
    top_k_records: usize,
    top_k_from_indexer: usize,
    top_k_from_full_visible: usize,
    consume_records: usize,
    min_consume_width: Option<i64>,
    max_consume_width: Option<i64>,
    full_layers: Vec<i32>,
    shared_layers: Vec<i32>,
    failure_summary: String,
}

#[derive(Serialize)]
struct RealTopKSharedConsumerGuardReport {
    passed: bool,
    proof_kind: ExecutionProofKind,
    policy_role: IndexShareRole,
    effective_role: IndexShareRole,
    sideband_source_kind: SidebandSourceKind,
    sideband_required: bool,
    sideband_present: bool,
    sideband_contract_satisfied: bool,
    native_consumer_execution_proven: bool,
    stage_wire_roundtrip_present: bool,
    stage_wire_roundtrip_passed: Option<bool>,
    stage_wire_sideband_bytes_match: Option<bool>,
    stage_wire_sideband_checksum_match: Option<bool>,
    failure_summary: String,
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
    token_count: usize,
    hidden_bytes: usize,
    sideband_bytes: usize,
    sideband_i32_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    sideband_i32_per_token: Option<usize>,
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

#[derive(Default, Serialize)]
struct TopKSidebandStats {
    i32_aligned: bool,
    token_count: usize,
    total_i32: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    width_per_token: Option<usize>,
    valid_index_count: usize,
    negative_index_count: usize,
    out_of_range_index_count: usize,
    causal_visible_count: usize,
    future_index_count: usize,
    duplicate_index_count: usize,
    active_top_end_sum: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_active_top_end: Option<f64>,
    max_active_top_end: usize,
    inactive_tail_count: usize,
    masked_future_in_active_prefix_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_prefix_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    causal_visible_ratio: Option<f64>,
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
    top_k_sideband_stats: TopKSidebandStats,
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

fn build_direct_sparse_prefill_guard(
    candidate: &MicrobenchCaseSummary,
) -> DirectSparsePrefillGuardReport {
    const DEFAULT_PREFILL_THREADS_X: u64 = 32;
    const MAX_CACHED_TOPK_PREFILL_THREADS_X: u64 = 256;

    let direct_decision_records = candidate.direct_sparse_decision_records.len();
    let direct_use_records = candidate
        .direct_sparse_decision_records
        .iter()
        .filter(|record| record.use_direct)
        .count();
    let fallback_records = direct_decision_records.saturating_sub(direct_use_records);
    let prefill_decisions = candidate
        .direct_sparse_decision_records
        .iter()
        .filter(|record| direct_sparse_decision_is_prefill(record))
        .count();
    let prefill_direct_decisions = candidate
        .direct_sparse_decision_records
        .iter()
        .filter(|record| direct_sparse_decision_is_prefill(record) && record.use_direct)
        .count();
    let large_prefill_decisions = candidate
        .direct_sparse_decision_records
        .iter()
        .filter(|record| record.large_prefill_shape == Some(true))
        .count();
    let large_prefill_direct_decisions = candidate
        .direct_sparse_decision_records
        .iter()
        .filter(|record| record.large_prefill_shape == Some(true) && record.use_direct)
        .count();
    let sparse_mask_nodes = candidate.op_timing_summary.sparse_mask.nodes;
    let dsa_sparse_attn_nodes = candidate.op_timing_summary.dsa_sparse_attn.nodes;
    let dense_sparse_mask_dispatches = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| record.op == "dsa_sparse_mask")
        .count();
    let dsa_sparse_attn_dispatches = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| record.op == "dsa_sparse_attn")
        .count();
    let accepted_prefill_dispatches = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| {
            record.op == "dsa_sparse_attn"
                && record.batch.is_some_and(|batch| batch > 1)
                && record.top_k.is_some_and(|top_k| top_k > 0)
                && prefill_dispatch_threads_accepted(record)
        })
        .count();
    let cached_topk_non_prefill_dispatches = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| {
            record.op == "dsa_sparse_attn"
                && dsa_sparse_attn_kernel_is_cached_topk(record.kernel.as_deref())
                && record.batch.is_none_or(|batch| batch <= 1)
        })
        .count();
    let accepted_large_prefill_dispatches = if large_prefill_direct_decisions > 0 {
        accepted_prefill_dispatches
    } else {
        0
    };
    let mut failures = Vec::new();
    if prefill_direct_decisions == 0 {
        failures.push("no_prefill_direct_decision");
    }
    if fallback_records > 0 {
        failures.push("fallback_decision_present");
    }
    if sparse_mask_nodes > 0 {
        failures.push("sparse_mask_nodes_present");
    }
    if dense_sparse_mask_dispatches > 0 {
        failures.push("dense_sparse_mask_dispatch_present");
    }
    if dsa_sparse_attn_nodes == 0 && dsa_sparse_attn_dispatches == 0 {
        failures.push("missing_dsa_sparse_attn_evidence");
    }
    if accepted_prefill_dispatches == 0 {
        failures.push("missing_accepted_prefill_dispatch");
    }
    if cached_topk_non_prefill_dispatches > 0 {
        failures.push("cached_topk_non_prefill_dispatch_present");
    }
    let passed = failures.is_empty();
    DirectSparsePrefillGuardReport {
        checked_case: candidate.label,
        passed,
        direct_decision_records,
        direct_use_records,
        fallback_records,
        prefill_decisions,
        prefill_direct_decisions,
        large_prefill_decisions,
        large_prefill_direct_decisions,
        sparse_mask_nodes,
        dense_sparse_mask_dispatches,
        dsa_sparse_attn_nodes,
        dsa_sparse_attn_dispatches,
        accepted_prefill_dispatches,
        cached_topk_non_prefill_dispatches,
        accepted_large_prefill_dispatches,
        default_prefill_threads_x: DEFAULT_PREFILL_THREADS_X,
        max_cached_topk_prefill_threads_x: MAX_CACHED_TOPK_PREFILL_THREADS_X,
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

fn build_direct_sparse_decode_guard(
    candidate: &MicrobenchCaseSummary,
) -> DirectSparseDecodeGuardReport {
    let execution_records = &candidate.direct_sparse_execution_decision_records;
    let direct_decision_records = execution_records.len();
    let direct_use_records = candidate
        .direct_sparse_execution_decision_records
        .iter()
        .filter(|record| record.use_direct)
        .count();
    let fallback_records = direct_decision_records.saturating_sub(direct_use_records);
    let decode_decisions = candidate
        .direct_sparse_execution_decision_records
        .iter()
        .filter(|record| record.decode_shape)
        .count();
    let decode_direct_decisions = candidate
        .direct_sparse_execution_decision_records
        .iter()
        .filter(|record| record.decode_shape && record.use_direct)
        .count();
    let sparse_mask_nodes = candidate.op_timing_summary.sparse_mask.nodes;
    let dsa_sparse_attn_nodes = candidate.op_timing_summary.dsa_sparse_attn.nodes;
    let dense_sparse_mask_dispatches = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| record.op == "dsa_sparse_mask")
        .count();
    let dsa_sparse_attn_dispatches = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| record.op == "dsa_sparse_attn")
        .count();
    let accepted_decode_dispatches = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| {
            record.op == "dsa_sparse_attn"
                && record.batch == Some(1)
                && record.top_k.is_some_and(|top_k| top_k > 0)
        })
        .count();

    let mut failures = Vec::new();
    if decode_direct_decisions == 0 {
        failures.push("no_decode_direct_decision");
    }
    if fallback_records > 0 {
        failures.push("fallback_decision_present");
    }
    if sparse_mask_nodes > 0 {
        failures.push("sparse_mask_nodes_present");
    }
    if dense_sparse_mask_dispatches > 0 {
        failures.push("dense_sparse_mask_dispatch_present");
    }
    if dsa_sparse_attn_nodes == 0 && dsa_sparse_attn_dispatches == 0 {
        failures.push("missing_dsa_sparse_attn_evidence");
    }
    if accepted_decode_dispatches == 0 {
        failures.push("missing_accepted_decode_dispatch");
    }

    DirectSparseDecodeGuardReport {
        checked_case: candidate.label,
        passed: failures.is_empty(),
        direct_decision_records,
        direct_use_records,
        fallback_records,
        decode_decisions,
        decode_direct_decisions,
        sparse_mask_nodes,
        dense_sparse_mask_dispatches,
        dsa_sparse_attn_nodes,
        dsa_sparse_attn_dispatches,
        accepted_decode_dispatches,
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

fn direct_sparse_decision_is_prefill(record: &DirectSparseDecisionRecord) -> bool {
    record.prefill_shape || record.large_prefill_shape == Some(true)
}

fn prefill_dispatch_threads_accepted(record: &MetalDispatchRecord) -> bool {
    const DEFAULT_PREFILL_THREADS_X: u64 = 32;
    const MAX_CACHED_TOPK_PREFILL_THREADS_X: u64 = 256;

    match record.kernel.as_deref() {
        Some("decode_vec" | "decode_vec_direct") => {
            record.threads_x <= MAX_CACHED_TOPK_PREFILL_THREADS_X
        }
        _ if dsa_sparse_attn_kernel_is_cached_topk(record.kernel.as_deref()) => {
            record.threads_x <= MAX_CACHED_TOPK_PREFILL_THREADS_X
        }
        _ if record.threads_x == DEFAULT_PREFILL_THREADS_X => true,
        _ => {
            record.threads_x <= MAX_CACHED_TOPK_PREFILL_THREADS_X
                && record
                    .batch
                    .zip(record.top_k)
                    .is_some_and(|(batch, top_k)| top_k <= batch)
        }
    }
}

fn dsa_sparse_attn_kernel_is_cached_topk(kernel: Option<&str>) -> bool {
    matches!(kernel, Some("cached_topk" | "cached_topk_v4"))
}

fn build_partial_top_k_guard(
    candidate: &MicrobenchCaseSummary,
    optimized_dispatch_probe: Option<&MicrobenchCaseSummary>,
    hidden_payload_bytes: usize,
    token_count: usize,
) -> PartialTopKGuardReport {
    let dispatch_case = optimized_dispatch_probe.unwrap_or(candidate);
    let dsa_dispatches: Vec<_> = candidate
        .metal_dispatch_records
        .iter()
        .filter(|record| record.op == "dsa_sparse_attn")
        .collect();
    let dispatch_case_dsa_dispatches: Vec<_> = dispatch_case
        .metal_dispatch_records
        .iter()
        .filter(|record| record.op == "dsa_sparse_attn")
        .collect();
    let partial_dispatches: Vec<_> = dsa_dispatches
        .iter()
        .copied()
        .filter(|record| dispatch_has_partial_top_k(record))
        .collect();
    let dispatch_case_partial_dispatches: Vec<_> = dispatch_case_dsa_dispatches
        .iter()
        .copied()
        .filter(|record| dispatch_has_partial_top_k(record))
        .collect();
    let proof_partial_dispatches = if dispatch_case_partial_dispatches.is_empty() {
        &partial_dispatches
    } else {
        &dispatch_case_partial_dispatches
    };
    let max_kv = dispatch_case_dsa_dispatches
        .iter()
        .filter_map(|record| record.kv)
        .max()
        .or_else(|| dsa_dispatches.iter().filter_map(|record| record.kv).max());
    let min_partial_top_k = proof_partial_dispatches
        .iter()
        .filter_map(|record| record.top_k)
        .min();
    let max_partial_top_k = proof_partial_dispatches
        .iter()
        .filter_map(|record| record.top_k)
        .max();
    let expected_sideband_i32_per_token = common_partial_top_k_width(proof_partial_dispatches)
        .and_then(|width| width.try_into().ok());
    let observed_sideband_widths: Vec<_> = candidate
        .timings
        .iter()
        .filter_map(|timing| {
            output_sideband_i32_per_token(
                timing.output_payload_bytes,
                hidden_payload_bytes,
                token_count,
            )
        })
        .collect();
    let output_frames_with_sideband = observed_sideband_widths
        .iter()
        .filter(|width| **width > 0)
        .count();
    let output_frames_with_expected_sideband =
        expected_sideband_i32_per_token.map_or(0, |expected| {
            observed_sideband_widths
                .iter()
                .filter(|width| **width == expected)
                .count()
        });
    let max_observed_sideband_i32_per_token = observed_sideband_widths.iter().copied().max();
    let native_shared_consume_width_matches =
        expected_sideband_i32_per_token.map_or(0, |expected| {
            let trace = &candidate.indexshare_trace_summary;
            let expected = expected as i64;
            let width_matches = trace.min_consume_width == Some(expected)
                && trace.max_consume_width == Some(expected);
            if width_matches && trace.shared_exec_missing_input_top_k == 0 {
                trace.consume_records
            } else {
                0
            }
        });
    let local_top_k_contract_satisfied =
        output_frames_with_expected_sideband > 0 || native_shared_consume_width_matches > 0;

    let mut failures = Vec::new();
    if proof_partial_dispatches.is_empty() {
        failures.push("missing_partial_top_k_dispatch");
    }
    if expected_sideband_i32_per_token.is_none() {
        failures.push("ambiguous_partial_top_k_width");
    }
    if !local_top_k_contract_satisfied {
        failures.push("missing_expected_top_k_width_proof");
    }
    let passed = failures.is_empty();
    PartialTopKGuardReport {
        checked_case: dispatch_case.label,
        passed,
        dsa_sparse_attn_dispatches: dispatch_case_dsa_dispatches.len(),
        partial_dsa_sparse_attn_dispatches: proof_partial_dispatches.len(),
        max_kv,
        min_partial_top_k,
        max_partial_top_k,
        output_frames: candidate.timings.len(),
        output_frames_with_sideband,
        output_frames_with_expected_sideband,
        expected_sideband_i32_per_token,
        max_observed_sideband_i32_per_token,
        native_shared_consume_records: candidate.indexshare_trace_summary.consume_records,
        native_shared_consume_width_matches,
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

fn dispatch_has_partial_top_k(record: &MetalDispatchRecord) -> bool {
    matches!(
        (record.kv, record.top_k),
        (Some(kv), Some(top_k)) if top_k > 0 && kv > top_k
    )
}

fn common_partial_top_k_width(records: &[&MetalDispatchRecord]) -> Option<u64> {
    let mut widths = records.iter().filter_map(|record| record.top_k);
    let first = widths.next()?;
    widths.all(|width| width == first).then_some(first)
}

fn build_compact_flash_guard(candidate: &MicrobenchCaseSummary) -> CompactFlashGuardReport {
    let dispatch = &candidate.metal_dispatch_summary;
    let compact_get_rows_records = compact_get_rows_records(&candidate.metal_dispatch_records);
    let compact_get_rows_typed_records = compact_get_rows_records
        .iter()
        .filter(|record| matches!(record.kernel.as_deref(), Some("typed" | "typed_vec4")))
        .count();
    let compact_get_rows_promote_records = compact_get_rows_records
        .iter()
        .filter(|record| record.kernel.as_deref() == Some("promote"))
        .count();
    let execution_mask_records = &candidate.compact_flash_execution_mask_records;
    let omitted_mla_kq_mask_records = execution_mask_records
        .iter()
        .filter(|record| record.omitted_mla_kq_mask)
        .count();
    let materialized_mla_kq_mask_records = execution_mask_records
        .len()
        .saturating_sub(omitted_mla_kq_mask_records);
    let execution_policy = candidate
        .compact_flash_execution_policy_records
        .last()
        .or_else(|| candidate.compact_flash_policy_records.last());
    let mut failures = Vec::new();
    if dispatch.flash_attn_ext_glm_dsa_shape_records == 0 {
        failures.push("missing_glm_shape_flash_attn_ext");
    }
    if dispatch.flash_attn_ext_vec_records == 0 {
        failures.push("missing_vec_flash_attn_ext");
    }
    if compact_get_rows_typed_records == 0 && dispatch.dsa_compact_get_rows_fused_records == 0 {
        failures.push("missing_compact_get_rows");
    }
    if compact_get_rows_promote_records > 0 {
        failures.push("promoted_get_rows_present");
    }
    if dispatch.dsa_sparse_attn_records > 0 {
        failures.push("dsa_sparse_attn_dispatch_present");
    }
    if candidate.op_timing_summary.sparse_mask.nodes > 0 {
        failures.push("sparse_mask_nodes_present");
    }
    if execution_mask_records.is_empty() {
        failures.push("missing_compact_mask_omission_evidence");
    }
    if materialized_mla_kq_mask_records > 0 {
        failures.push("mla_kq_mask_materialized");
    }
    let passed = failures.is_empty();
    CompactFlashGuardReport {
        checked_case: candidate.label,
        passed,
        flash_attn_ext_records: dispatch.flash_attn_ext_records,
        flash_attn_ext_vec_records: dispatch.flash_attn_ext_vec_records,
        flash_attn_ext_tile_records: dispatch.flash_attn_ext_tile_records,
        flash_attn_ext_glm_dsa_shape_records: dispatch.flash_attn_ext_glm_dsa_shape_records,
        get_rows_records: dispatch.get_rows_records,
        get_rows_typed_records: dispatch.get_rows_typed_records,
        get_rows_promote_records: dispatch.get_rows_promote_records,
        compact_get_rows_records: compact_get_rows_records.len(),
        compact_get_rows_typed_records,
        compact_get_rows_promote_records,
        dsa_compact_get_rows_fused_records: dispatch.dsa_compact_get_rows_fused_records,
        dsa_sparse_attn_records: dispatch.dsa_sparse_attn_records,
        sparse_mask_nodes: candidate.op_timing_summary.sparse_mask.nodes,
        mask_omission_records: candidate.compact_flash_mask_records.len(),
        execution_mask_omission_records: execution_mask_records.len(),
        omitted_mla_kq_mask_records,
        materialized_mla_kq_mask_records,
        policy_phase: execution_policy.and_then(|record| record.phase.clone()),
        policy_selector_reason: execution_policy.and_then(|record| record.selector_reason.clone()),
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

fn compact_get_rows_records(records: &[MetalDispatchRecord]) -> Vec<&MetalDispatchRecord> {
    records
        .iter()
        .filter(|record| {
            record.op == "get_rows"
                && (record.tensor.starts_with("dsa_compact_k_topk_rows")
                    || record.tensor.starts_with("dsa_compact_v_topk_rows")
                    || record.tensor.starts_with("dsa_compact_mask_topk_rows"))
        })
        .collect()
}

fn build_moe_weighted_sum_guard(candidate: &MicrobenchCaseSummary) -> MoeWeightedSumGuardReport {
    let dispatch = &candidate.metal_dispatch_summary;
    let mut failures = Vec::new();
    let standalone_weighted_sum_ok = dispatch.moe_weighted_sum_records > 0
        && dispatch.moe_weighted_sum_f32x4_records == dispatch.moe_weighted_sum_records;
    let fused_weighted_sum_ok = dispatch.mul_mv_id_weighted_sum_fused_records > 0
        && dispatch.mul_mv_id_weighted_sum_fused_q3_k_records
            == dispatch.mul_mv_id_weighted_sum_fused_records;
    let parallel_weighted_slots_ok = dispatch.mul_mv_id_weighted_slots_records > 0
        && dispatch.mul_mv_id_weighted_slots_q3_k_records
            == dispatch.mul_mv_id_weighted_slots_records
        && dispatch.moe_weighted_sum_already_weighted_records > 0;

    if !standalone_weighted_sum_ok && !fused_weighted_sum_ok && !parallel_weighted_slots_ok {
        if dispatch.moe_weighted_sum_records == 0 {
            failures.push("missing_moe_weighted_sum");
        }
        if dispatch.moe_weighted_sum_f32x4_records == 0
            && dispatch.moe_weighted_sum_already_weighted_records == 0
        {
            failures.push("missing_moe_weighted_sum_f32x4");
        }
        if dispatch.moe_weighted_sum_records > 0
            && dispatch.moe_weighted_sum_f32x4_records != dispatch.moe_weighted_sum_records
            && dispatch.moe_weighted_sum_already_weighted_records
                != dispatch.moe_weighted_sum_records
        {
            failures.push("non_f32x4_moe_weighted_sum_present");
        }
        if dispatch.mul_mv_id_weighted_sum_fused_records == 0 {
            failures.push("missing_mul_mv_id_weighted_sum_fused");
        }
        if dispatch.mul_mv_id_weighted_sum_fused_records > 0
            && dispatch.mul_mv_id_weighted_sum_fused_q3_k_records
                != dispatch.mul_mv_id_weighted_sum_fused_records
        {
            failures.push("non_q3_k_mul_mv_id_weighted_sum_fused_present");
        }
        if dispatch.mul_mv_id_weighted_slots_records == 0 {
            failures.push("missing_mul_mv_id_weighted_slots");
        }
        if dispatch.mul_mv_id_weighted_slots_records > 0
            && dispatch.mul_mv_id_weighted_slots_q3_k_records
                != dispatch.mul_mv_id_weighted_slots_records
        {
            failures.push("non_q3_k_mul_mv_id_weighted_slots_present");
        }
        if dispatch.mul_mv_id_weighted_slots_records > 0
            && dispatch.moe_weighted_sum_already_weighted_records == 0
        {
            failures.push("missing_already_weighted_moe_weighted_sum");
        }
    }
    let passed = failures.is_empty();
    MoeWeightedSumGuardReport {
        checked_case: candidate.label,
        passed,
        moe_weighted_sum_records: dispatch.moe_weighted_sum_records,
        moe_weighted_sum_f32x4_records: dispatch.moe_weighted_sum_f32x4_records,
        moe_weighted_sum_already_weighted_records: dispatch
            .moe_weighted_sum_already_weighted_records,
        mul_mv_id_weighted_sum_fused_records: dispatch.mul_mv_id_weighted_sum_fused_records,
        mul_mv_id_weighted_sum_fused_q3_k_records: dispatch
            .mul_mv_id_weighted_sum_fused_q3_k_records,
        mul_mv_id_weighted_slots_records: dispatch.mul_mv_id_weighted_slots_records,
        mul_mv_id_weighted_slots_q3_k_records: dispatch.mul_mv_id_weighted_slots_q3_k_records,
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

fn build_moe_motif_guard(
    candidate: &MicrobenchCaseSummary,
    optimized_probe: Option<&MicrobenchCaseSummary>,
) -> MoeMotifGuardReport {
    let checked = optimized_probe.unwrap_or(candidate);
    let dispatch = &checked.metal_dispatch_summary;
    let mut failures = Vec::new();
    if dispatch.glm_dsa_moe_motif_candidate_records == 0 {
        failures.push("missing_moe_motif_candidate");
    }
    if dispatch.glm_dsa_moe_motif_natural_order_records
        != dispatch.glm_dsa_moe_motif_candidate_records
    {
        failures.push("non_natural_order_motif_present");
    }
    if dispatch.glm_dsa_moe_motif_backend_candidate_records
        != dispatch.glm_dsa_moe_motif_candidate_records
    {
        failures.push("non_backend_candidate_motif_present");
    }
    if dispatch.glm_dsa_moe_motif_subgraph_fusable_records
        != dispatch.glm_dsa_moe_motif_candidate_records
    {
        failures.push("non_subgraph_fusable_motif_present");
    }
    if dispatch.glm_dsa_moe_motif_max_nodes < 4 {
        failures.push("motif_too_small");
    }
    let passed = failures.is_empty();
    MoeMotifGuardReport {
        checked_case: checked.label,
        passed,
        motif_candidate_records: dispatch.glm_dsa_moe_motif_candidate_records,
        natural_order_records: dispatch.glm_dsa_moe_motif_natural_order_records,
        backend_candidate_records: dispatch.glm_dsa_moe_motif_backend_candidate_records,
        subgraph_fusable_records: dispatch.glm_dsa_moe_motif_subgraph_fusable_records,
        coencoded_records: dispatch.glm_dsa_moe_motif_coencoded_records,
        max_motif_nodes: dispatch.glm_dsa_moe_motif_max_nodes,
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

fn output_sideband_i32_per_token(
    output_payload_bytes: usize,
    hidden_payload_bytes: usize,
    token_count: usize,
) -> Option<usize> {
    if token_count == 0 || output_payload_bytes < hidden_payload_bytes {
        return None;
    }
    let sideband_bytes = output_payload_bytes - hidden_payload_bytes;
    if !sideband_bytes.is_multiple_of(std::mem::size_of::<i32>()) {
        return None;
    }
    let sideband_i32 = sideband_bytes / std::mem::size_of::<i32>();
    sideband_i32
        .is_multiple_of(token_count)
        .then_some(sideband_i32 / token_count)
}

fn build_native_indexshare_guard(case: &MicrobenchCaseSummary) -> NativeIndexShareGuardReport {
    let trace = &case.indexshare_trace_summary;
    let mut failures = Vec::new();
    if trace.records == 0 {
        failures.push("missing_indexshare_trace_records");
    }
    if trace.full_exec_records == 0 {
        failures.push("missing_full_exec");
    }
    if trace.shared_exec_records == 0 {
        failures.push("missing_shared_exec");
    }
    if trace.top_k_records == 0 {
        failures.push("missing_top_k_production");
    }
    if trace.consume_records == 0 {
        failures.push("missing_shared_consume");
    }
    if trace.shared_exec_missing_input_top_k > 0 {
        failures.push("shared_exec_missing_input_top_k");
    }
    if trace.shared_exec_with_input_top_k < trace.shared_exec_records {
        failures.push("not_all_shared_execs_have_input_top_k");
    }
    let passed = failures.is_empty();
    NativeIndexShareGuardReport {
        checked_case: case.label,
        passed,
        records: trace.records,
        exec_records: trace.exec_records,
        full_exec_records: trace.full_exec_records,
        shared_exec_records: trace.shared_exec_records,
        shared_exec_with_input_top_k: trace.shared_exec_with_input_top_k,
        shared_exec_missing_input_top_k: trace.shared_exec_missing_input_top_k,
        top_k_records: trace.top_k_records,
        top_k_from_indexer: trace.top_k_from_indexer,
        top_k_from_full_visible: trace.top_k_from_full_visible,
        consume_records: trace.consume_records,
        min_consume_width: trace.min_consume_width,
        max_consume_width: trace.max_consume_width,
        full_layers: trace.full_layers.clone(),
        shared_layers: trace.shared_layers.clone(),
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

fn build_real_top_k_shared_consumer_guard(
    execution: &ExecutionContractReport,
    stage_wire_roundtrip: Option<&StageWireRoundTripReport>,
) -> RealTopKSharedConsumerGuardReport {
    let stage_wire_roundtrip_passed = stage_wire_roundtrip.map(|report| report.passed);
    let stage_wire_sideband_bytes_match =
        stage_wire_roundtrip.map(|report| report.sideband_bytes_match);
    let stage_wire_sideband_checksum_match =
        stage_wire_roundtrip.map(|report| report.sideband_checksum_match);
    let mut failures = Vec::new();
    if execution.proof_kind != ExecutionProofKind::SharedConsumerWithRealTopK {
        failures.push("not_shared_consumer_with_real_top_k");
    }
    if execution.effective_layer_role.role != IndexShareRole::SharedConsumer {
        failures.push("effective_role_not_shared_consumer");
    }
    if execution.sideband_source.kind != SidebandSourceKind::RealTopK {
        failures.push("sideband_source_not_real_top_k");
    }
    if !execution.sideband_required {
        failures.push("sideband_not_required");
    }
    if !execution.sideband_present {
        failures.push("sideband_missing");
    }
    if !execution.sideband_contract_satisfied {
        failures.push("sideband_contract_unsatisfied");
    }
    if !execution.native_consumer_execution_proven {
        failures.push("native_consumer_execution_not_proven");
    }
    if !matches!(stage_wire_roundtrip_passed, Some(true)) {
        failures.push("stage_wire_roundtrip_missing_or_failed");
    }
    if !matches!(stage_wire_sideband_bytes_match, Some(true)) {
        failures.push("stage_wire_sideband_bytes_mismatch");
    }
    if !matches!(stage_wire_sideband_checksum_match, Some(true)) {
        failures.push("stage_wire_sideband_checksum_mismatch");
    }
    let passed = failures.is_empty();
    RealTopKSharedConsumerGuardReport {
        passed,
        proof_kind: execution.proof_kind,
        policy_role: execution.policy_layer_role.role,
        effective_role: execution.effective_layer_role.role,
        sideband_source_kind: execution.sideband_source.kind,
        sideband_required: execution.sideband_required,
        sideband_present: execution.sideband_present,
        sideband_contract_satisfied: execution.sideband_contract_satisfied,
        native_consumer_execution_proven: execution.native_consumer_execution_proven,
        stage_wire_roundtrip_present: stage_wire_roundtrip.is_some(),
        stage_wire_roundtrip_passed,
        stage_wire_sideband_bytes_match,
        stage_wire_sideband_checksum_match,
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
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
    compact_flash_attn: bool,
    allow_compact_flash_auto: bool,
    direct_sparse_prefill: bool,
    native_default_direct_sparse_prefill: bool,
    enable_unproven_large_direct_sparse_prefill: bool,
    direct_sparse_prefill_max_tokens: Option<u32>,
    fused_sparse_mask: bool,
    parallel_lightning_indexer: bool,
    op_timing: bool,
    native_indexshare_exec_log: bool,
    metal_dispatch_log: bool,
    metal_topk_moe_route_fusion: bool,
    metal_topk_moe_route_fusion_native_default: bool,
    moe_motif_coencode: bool,
    moe_down_weighted_fusion: bool,
    moe_down_weighted_parallel: bool,
    moe_down_unweighted_slots: bool,
    sparse_attn_threads: Option<u32>,
    sparse_attn_group_heads: Option<u32>,
    lightning_indexer_threads: Option<u32>,
    dense_sparse_mask_max_bytes: Option<u64>,
    direct_sparse_prefill_min_kv_topk_ratio: Option<u32>,
}

impl MicrobenchFlags {
    fn from_args(args: &GlmDsaLayerMicrobenchArgs) -> Self {
        Self {
            direct_sparse_attn: args.direct_sparse_attn,
            compact_flash_attn: args.compact_flash_attn,
            allow_compact_flash_auto: args.allow_compact_flash_auto,
            direct_sparse_prefill: args.direct_sparse_prefill,
            native_default_direct_sparse_prefill: args.native_default_direct_sparse_prefill,
            enable_unproven_large_direct_sparse_prefill: args
                .enable_unproven_large_direct_sparse_prefill,
            direct_sparse_prefill_max_tokens: args.direct_sparse_prefill_max_tokens,
            fused_sparse_mask: args.fused_sparse_mask,
            parallel_lightning_indexer: args.parallel_lightning_indexer,
            op_timing: args.op_timing,
            native_indexshare_exec_log: args.require_native_indexshare_proof,
            metal_dispatch_log: args.metal_dispatch_log,
            metal_topk_moe_route_fusion: args.metal_topk_moe_route_fusion,
            metal_topk_moe_route_fusion_native_default: args
                .metal_topk_moe_route_fusion_native_default,
            moe_motif_coencode: args.moe_motif_coencode,
            moe_down_weighted_fusion: args.moe_down_weighted_fusion,
            moe_down_weighted_parallel: args.moe_down_weighted_parallel,
            moe_down_unweighted_slots: args.moe_down_unweighted_slots,
            sparse_attn_threads: args.sparse_attn_threads,
            sparse_attn_group_heads: args.sparse_attn_group_heads,
            lightning_indexer_threads: args.lightning_indexer_threads,
            dense_sparse_mask_max_bytes: args.dense_sparse_mask_max_bytes,
            direct_sparse_prefill_min_kv_topk_ratio: args.direct_sparse_prefill_min_kv_topk_ratio,
        }
    }

    fn capture_native_logs(self) -> bool {
        self.op_timing || self.native_indexshare_exec_log || self.metal_dispatch_log
    }
}

fn should_run_optimized_dispatch_probe(
    flags: MicrobenchFlags,
    require_moe_motif_proof: bool,
) -> bool {
    require_moe_motif_proof
        || (flags.op_timing && flags.metal_dispatch_log)
        || (flags.metal_topk_moe_route_fusion && !flags.metal_dispatch_log)
        || ((flags.moe_down_weighted_fusion
            || flags.moe_down_weighted_parallel
            || flags.moe_down_unweighted_slots)
            && !flags.metal_dispatch_log)
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

#[derive(Serialize)]
struct RepresentativeProfileReport {
    checked_case: &'static str,
    source: &'static str,
    diagnostic_timing_discarded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    warning: Option<&'static str>,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    timing_summary: TimingDistributionSummary,
    #[serde(skip_serializing_if = "TimingBreakdownSummary::is_empty")]
    timing_breakdown: TimingBreakdownSummary,
    #[serde(skip_serializing_if = "GlmDsaDispatchSummary::is_empty")]
    metal_dispatch_summary: GlmDsaDispatchSummary,
}

impl RepresentativeProfileReport {
    fn new(
        candidate: &MicrobenchCaseSummary,
        optimized_probe: Option<&MicrobenchCaseSummary>,
        integrity: &ProfileIntegrityReport,
    ) -> Self {
        let use_optimized_probe =
            integrity.diagnostic_timing_may_disable_route_fusion && optimized_probe.is_some();
        let checked = if use_optimized_probe {
            optimized_probe.expect("optimized probe is present")
        } else {
            candidate
        };
        Self {
            checked_case: checked.label,
            source: if use_optimized_probe {
                "optimized_dispatch_probe"
            } else {
                "candidate"
            },
            diagnostic_timing_discarded: use_optimized_probe,
            warning: use_optimized_probe.then_some(
                "op timing observes intermediate tensors and may disable graph fusion; representative profile uses the optimized probe",
            ),
            timing_summary: checked.timing_summary.clone(),
            timing_breakdown: checked.timing_breakdown.clone(),
            metal_dispatch_summary: checked.metal_dispatch_summary.clone(),
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
    #[serde(skip_serializing_if = "CompactFlashPolicySummary::is_empty")]
    compact_flash_policy_summary: CompactFlashPolicySummary,
    #[serde(skip_serializing_if = "DirectSparseDecisionSummary::is_empty")]
    direct_sparse_decision_summary: DirectSparseDecisionSummary,
    #[serde(skip_serializing_if = "TimingDistributionSummary::is_empty")]
    timing_summary: TimingDistributionSummary,
    #[serde(skip_serializing_if = "TimingBreakdownSummary::is_empty")]
    timing_breakdown: TimingBreakdownSummary,
    #[serde(skip_serializing_if = "GlmDsaDispatchSummary::is_empty")]
    metal_dispatch_summary: GlmDsaDispatchSummary,
    #[serde(skip_serializing_if = "DirectSparseSpillSummary::is_empty")]
    direct_sparse_spill_summary: DirectSparseSpillSummary,
    #[serde(skip_serializing_if = "GlmDsaOpTimingSummary::is_empty")]
    op_timing_summary: GlmDsaOpTimingSummary,
    #[serde(skip_serializing_if = "RoutedMoeTimingSummary::is_empty")]
    routed_moe_timing_summary: RoutedMoeTimingSummary,
    #[serde(skip_serializing_if = "IndexShareTimingSummary::is_empty")]
    indexshare_timing_summary: IndexShareTimingSummary,
    #[serde(skip_serializing_if = "IndexShareTraceSummary::is_empty")]
    indexshare_trace_summary: IndexShareTraceSummary,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_policy_records: Vec<CompactFlashPolicyRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_execution_policy_records: Vec<CompactFlashPolicyRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_non_measured_policy_records: Vec<CompactFlashPolicyRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_mask_records: Vec<CompactFlashMaskRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_execution_mask_records: Vec<CompactFlashMaskRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compact_flash_non_measured_mask_records: Vec<CompactFlashMaskRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_execution_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    direct_sparse_non_measured_decision_records: Vec<DirectSparseDecisionRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    metal_dispatch_records: Vec<MetalDispatchRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    op_timing_records: Vec<TimingRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    group_timing_records: Vec<TimingGroupRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    hot_tensor_records: Vec<HotTensorRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    compute_buffer_records: Vec<ComputeBufferRecord>,
    timings: Vec<IterationTiming>,
}

#[derive(Clone, Default, Serialize)]
struct DirectSparseSpillSummary {
    records: usize,
    decode_vec_records: usize,
    decode_vec_direct_records: usize,
    decode_vec_reduce_records: usize,
    rows: u64,
    partial_bytes: u64,
    softmax_bytes: u64,
    tmp_bytes: u64,
    max_tmp_bytes: u64,
    partial_mib: Option<f64>,
    softmax_mib: Option<f64>,
    tmp_mib: Option<f64>,
}

impl DirectSparseSpillSummary {
    fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[allow(clippy::field_reassign_with_default)]
fn summarize_direct_sparse_spill(records: &[MetalDispatchRecord]) -> DirectSparseSpillSummary {
    let mut summary = DirectSparseSpillSummary::default();
    summary.decode_vec_records = records
        .iter()
        .filter(|record| {
            record.op == "dsa_sparse_attn" && record.kernel.as_deref() == Some("decode_vec")
        })
        .count();
    summary.decode_vec_direct_records = records
        .iter()
        .filter(|record| {
            record.op == "dsa_sparse_attn" && record.kernel.as_deref() == Some("decode_vec_direct")
        })
        .count();
    summary.records += summary.decode_vec_direct_records;
    for record in records.iter().filter(|record| {
        record.op == "dsa_sparse_attn" && record.kernel.as_deref() == Some("decode_vec_direct")
    }) {
        summary.rows += record.rows.unwrap_or(0);
    }

    for record in records.iter().filter(|record| {
        record.op == "dsa_sparse_attn" && record.kernel.as_deref() == Some("decode_vec_reduce")
    }) {
        summary.records += 1;
        summary.decode_vec_reduce_records += 1;
        summary.rows += record.rows.unwrap_or(0);
        summary.partial_bytes += record.partial_bytes.unwrap_or(0);
        summary.softmax_bytes += record.softmax_bytes.unwrap_or(0);
        let tmp_bytes = record.tmp_bytes.unwrap_or(0);
        summary.tmp_bytes += tmp_bytes;
        summary.max_tmp_bytes = summary.max_tmp_bytes.max(tmp_bytes);
    }

    summary.partial_mib = bytes_to_mib(summary.partial_bytes);
    summary.softmax_mib = bytes_to_mib(summary.softmax_bytes);
    summary.tmp_mib = bytes_to_mib(summary.tmp_bytes);
    summary
}

fn bytes_to_mib(bytes: u64) -> Option<f64> {
    if bytes == 0 {
        None
    } else {
        Some(bytes as f64 / (1024.0 * 1024.0))
    }
}

#[derive(Clone, Default, Serialize)]
struct DirectSparseDecisionSummary {
    records: usize,
    execution_records: usize,
    non_measured_records: usize,
    use_direct: usize,
    fallback: usize,
    execution_use_direct: usize,
    execution_fallback: usize,
    decode_shape: usize,
    prefill_shape: usize,
    token_shape_allowed: usize,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    phases: BTreeMap<String, usize>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    execution_phases: BTreeMap<String, usize>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    execution_selector_reasons: BTreeMap<String, usize>,
}

impl DirectSparseDecisionSummary {
    fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Clone, Default, Serialize)]
struct CompactFlashPolicySummary {
    records: usize,
    execution_records: usize,
    non_measured_records: usize,
    use_compact: usize,
    fallback: usize,
    execution_use_compact: usize,
    execution_fallback: usize,
    forced: usize,
    disabled: usize,
    ratio_ok: usize,
    enabled: usize,
    flash_attn: usize,
    decode_shape: usize,
    no_mask: usize,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    phases: BTreeMap<String, usize>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    execution_selector_reasons: BTreeMap<String, usize>,
}

impl CompactFlashPolicySummary {
    fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Clone, Default, Serialize)]
struct IndexShareTimingSummary {
    records: usize,
    layer_groups: usize,
    producer_groups: usize,
    consumer_groups: usize,
    indexer_topk_nodes: u64,
    indexer_topk_us: u64,
    indexer_nodes: u64,
    indexer_us: u64,
    top_k_nodes: u64,
    top_k_us: u64,
    producer_total_us: u64,
    consumer_total_us: u64,
    indexer_share_of_indexer_topk: Option<f64>,
    top_k_share_of_indexer_topk: Option<f64>,
    producer_group_names: Vec<String>,
    consumer_group_names: Vec<String>,
}

impl IndexShareTimingSummary {
    fn is_empty(summary: &Self) -> bool {
        summary.records == 0
    }
}

#[derive(Default)]
struct IndexShareGroupTiming {
    records: usize,
    total_us: u64,
    indexer_topk_nodes: u64,
    indexer_topk_us: u64,
    indexer_nodes: u64,
    indexer_us: u64,
    top_k_nodes: u64,
    top_k_us: u64,
}

fn summarize_indexshare_timing(records: &[TimingGroupRecord]) -> IndexShareTimingSummary {
    let mut groups: HashMap<String, IndexShareGroupTiming> = HashMap::new();
    for record in records {
        if !record.group.starts_with("layer_") {
            continue;
        }
        let group = groups.entry(record.group.clone()).or_default();
        group.records += 1;
        group.total_us += record.timing.total_us;
        group.indexer_topk_nodes += record.timing.indexer_topk_nodes;
        group.indexer_topk_us += record.timing.indexer_topk_us;
        group.indexer_nodes += record.timing.indexer_nodes.unwrap_or(0);
        group.indexer_us += record.timing.indexer_us.unwrap_or(0);
        group.top_k_nodes += record.timing.top_k_nodes.unwrap_or(0);
        group.top_k_us += record.timing.top_k_us.unwrap_or(0);
    }

    let mut summary = IndexShareTimingSummary::default();
    summary.records = groups.values().map(|group| group.records).sum();
    summary.layer_groups = groups.len();
    summary.indexer_topk_nodes = groups.values().map(|group| group.indexer_topk_nodes).sum();
    summary.indexer_topk_us = groups.values().map(|group| group.indexer_topk_us).sum();
    summary.indexer_nodes = groups.values().map(|group| group.indexer_nodes).sum();
    summary.indexer_us = groups.values().map(|group| group.indexer_us).sum();
    summary.top_k_nodes = groups.values().map(|group| group.top_k_nodes).sum();
    summary.top_k_us = groups.values().map(|group| group.top_k_us).sum();
    summary.indexer_share_of_indexer_topk = ratio_u64(summary.indexer_us, summary.indexer_topk_us);
    summary.top_k_share_of_indexer_topk = ratio_u64(summary.top_k_us, summary.indexer_topk_us);

    let mut group_names: Vec<_> = groups.into_iter().collect();
    group_names.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    for (name, group) in group_names {
        if group.indexer_topk_nodes > 0 {
            summary.producer_groups += 1;
            summary.producer_total_us += group.total_us;
            summary.producer_group_names.push(name);
        } else {
            summary.consumer_groups += 1;
            summary.consumer_total_us += group.total_us;
            summary.consumer_group_names.push(name);
        }
    }
    summary
}

fn ratio_u64(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

struct MicrobenchComparison {
    baseline: MicrobenchCase,
    candidate: MicrobenchCase,
    parity: ParityComparison,
    poisoned_candidate: Option<MicrobenchCase>,
    poisoned_parity: Option<ParityComparison>,
    sideband_sensitivity: Option<SidebandSensitivityReport>,
}

impl MicrobenchComparison {
    fn as_report(&self) -> MicrobenchComparisonReport {
        MicrobenchComparisonReport {
            baseline: self.baseline.as_case_summary(),
            candidate: self.candidate.as_case_summary(),
            parity: self.parity.clone(),
            poisoned_candidate: self
                .poisoned_candidate
                .as_ref()
                .map(MicrobenchCase::as_case_summary),
            poisoned_parity: self.poisoned_parity.clone(),
            sideband_sensitivity: self.sideband_sensitivity.clone(),
        }
    }
}

#[derive(Serialize)]
struct MicrobenchComparisonReport {
    baseline: MicrobenchCaseSummary,
    candidate: MicrobenchCaseSummary,
    parity: ParityComparison,
    #[serde(skip_serializing_if = "Option::is_none")]
    poisoned_candidate: Option<MicrobenchCaseSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    poisoned_parity: Option<ParityComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sideband_sensitivity: Option<SidebandSensitivityReport>,
}

#[derive(Clone, Serialize)]
struct SidebandSensitivityReport {
    passed: bool,
    poison: SidebandPoisonReport,
    poisoned_hidden_mismatches: usize,
    poisoned_sideband_mismatched_bytes: usize,
    poisoned_hidden_max_abs_diff: f32,
    poisoned_hidden_max_rel_diff: f32,
    failure_summary: String,
}

#[derive(Clone, Serialize)]
struct SidebandPoisonReport {
    poison_kind: &'static str,
    sideband_i32_count: usize,
    sideband_i32_per_token: usize,
    changed_i32_count: usize,
    tokens_with_changes: usize,
    original_unique_index_count: usize,
}

fn build_sideband_sensitivity_report(
    poison: &SidebandPoisonReport,
    poisoned_parity: &ParityComparison,
) -> SidebandSensitivityReport {
    let mut failures = Vec::new();
    if poison.changed_i32_count == 0 {
        failures.push("no_sideband_indices_changed");
    }
    if poison.tokens_with_changes == 0 {
        failures.push("no_tokens_changed");
    }
    if poisoned_parity.passed {
        failures.push("poisoned_sideband_preserved_output_parity");
    }
    if poisoned_parity.hidden_mismatches == 0 && poisoned_parity.hidden_max_abs_diff == 0.0 {
        failures.push("poisoned_sideband_did_not_change_hidden_output");
    }
    SidebandSensitivityReport {
        passed: failures.is_empty(),
        poison: poison.clone(),
        poisoned_hidden_mismatches: poisoned_parity.hidden_mismatches,
        poisoned_sideband_mismatched_bytes: poisoned_parity.sideband_mismatched_bytes,
        poisoned_hidden_max_abs_diff: poisoned_parity.hidden_max_abs_diff,
        poisoned_hidden_max_rel_diff: poisoned_parity.hidden_max_rel_diff,
        failure_summary: if failures.is_empty() {
            "none".to_string()
        } else {
            failures.join(",")
        },
    }
}

#[derive(Clone, Serialize)]
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

#[derive(Clone, Serialize)]
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
    sideband_exact_match: bool,
    sideband_semantic_match: bool,
    sideband_bytes: usize,
    sideband_mismatched_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_sideband_mismatch: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sideband_i32_diff: Option<SidebandI32Diff>,
}

#[derive(Clone, Serialize)]
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
    position_start: i32,
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
    fn direct_sparse_prefill_guard_accepts_large_prefill_direct_path() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            1,
            &[dsa_sparse_attn_dispatch(64, 256, 32)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.prefill_direct_decisions, 1);
        assert_eq!(guard.large_prefill_direct_decisions, 1);
        assert_eq!(guard.sparse_mask_nodes, 0);
        assert_eq!(guard.dense_sparse_mask_dispatches, 0);
        assert_eq!(guard.dsa_sparse_attn_nodes, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_small_prefill_direct_path() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(false, true)],
            0,
            1,
            &[dsa_sparse_attn_dispatch(32, 256, 32)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.prefill_decisions, 1);
        assert_eq!(guard.prefill_direct_decisions, 1);
        assert_eq!(guard.large_prefill_direct_decisions, 0);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 0);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_default_kernel_wide_prefill_dispatch() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(false, true)],
            0,
            0,
            &[dsa_sparse_attn_dispatch(12, 12, 256)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.prefill_direct_decisions, 1);
        assert_eq!(guard.large_prefill_direct_decisions, 0);
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 0);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_dispatch_evidence_without_op_timing() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[dsa_sparse_attn_dispatch(64, 256, 32)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.dsa_sparse_attn_nodes, 0);
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_cached_topk_prefill_dispatch() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[cached_topk_dsa_sparse_attn_dispatch(2304, 2048, 256)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_small_cached_topk_prefill_dispatch() {
        let mut dispatch = dsa_sparse_attn_dispatch_with_kv(8, 256, 8, 256);
        dispatch.kernel = Some("cached_topk".to_string());
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(false, true)],
            0,
            0,
            &[dispatch],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.prefill_direct_decisions, 1);
        assert_eq!(guard.large_prefill_direct_decisions, 0);
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 0);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_cached_topk_v4_prefill_dispatch() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[cached_topk_v4_dsa_sparse_attn_dispatch(2304, 2048, 256)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_rejects_cached_topk_non_prefill_dispatch() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[
                cached_topk_dsa_sparse_attn_dispatch(64, 2048, 32),
                cached_topk_dsa_sparse_attn_dispatch(1, 2048, 32),
            ],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(!guard.passed);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.cached_topk_non_prefill_dispatches, 1);
        assert!(
            guard
                .failure_summary
                .contains("cached_topk_non_prefill_dispatch_present")
        );
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_decode_vec_prefill_dispatch() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[decode_vec_dsa_sparse_attn_dispatch(1024, 1024, 256)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_prefill_guard_accepts_decode_vec_direct_prefill_dispatch() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[decode_vec_direct_dsa_sparse_attn_dispatch(4096, 2048, 256)],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 1);
        assert_eq!(guard.accepted_large_prefill_dispatches, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_spill_summary_captures_decode_vec_reduce_tmp_bytes() {
        let mut reduce = decode_vec_dsa_sparse_attn_dispatch(4096, 2048, 32);
        reduce.kernel = Some("decode_vec_reduce".to_string());
        reduce.rows = Some(262_144);
        reduce.partial_bytes = Some(536_870_912);
        reduce.softmax_bytes = Some(4_194_304);
        reduce.tmp_bytes = Some(541_065_216);

        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[decode_vec_dsa_sparse_attn_dispatch(4096, 2048, 256), reduce],
        );

        let summary = candidate.direct_sparse_spill_summary;

        assert_eq!(summary.decode_vec_records, 1);
        assert_eq!(summary.decode_vec_reduce_records, 1);
        assert_eq!(summary.rows, 262_144);
        assert_eq!(summary.partial_bytes, 536_870_912);
        assert_eq!(summary.softmax_bytes, 4_194_304);
        assert_eq!(summary.tmp_bytes, 541_065_216);
        assert_eq!(summary.partial_mib, Some(512.0));
        assert_eq!(summary.softmax_mib, Some(4.0));
        assert_eq!(summary.tmp_mib, Some(516.0));
    }

    #[test]
    fn direct_sparse_spill_summary_captures_decode_vec_direct_zero_spill() {
        let mut direct = decode_vec_dsa_sparse_attn_dispatch(4096, 2048, 256);
        direct.kernel = Some("decode_vec_direct".to_string());
        direct.rows = Some(262_144);
        direct.partial_bytes = Some(0);
        direct.softmax_bytes = Some(0);
        direct.tmp_bytes = Some(0);

        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            0,
            0,
            &[direct],
        );

        let summary = candidate.direct_sparse_spill_summary;

        assert_eq!(summary.records, 1);
        assert_eq!(summary.decode_vec_records, 0);
        assert_eq!(summary.decode_vec_direct_records, 1);
        assert_eq!(summary.decode_vec_reduce_records, 0);
        assert_eq!(summary.rows, 262_144);
        assert_eq!(summary.partial_bytes, 0);
        assert_eq!(summary.softmax_bytes, 0);
        assert_eq!(summary.tmp_bytes, 0);
        assert_eq!(summary.tmp_mib, None);
    }

    #[test]
    fn direct_sparse_prefill_guard_rejects_dense_mask_or_uncapped_dispatch() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decision(true, true)],
            1,
            1,
            &[
                dsa_sparse_attn_dispatch(64, 256, 64),
                dsa_sparse_mask_dispatch(),
            ],
        );

        let guard = build_direct_sparse_prefill_guard(&candidate);

        assert!(!guard.passed);
        assert_eq!(guard.sparse_mask_nodes, 1);
        assert_eq!(guard.dense_sparse_mask_dispatches, 1);
        assert_eq!(guard.accepted_prefill_dispatches, 0);
        assert_eq!(guard.accepted_large_prefill_dispatches, 0);
        assert!(guard.failure_summary.contains("sparse_mask_nodes_present"));
        assert!(
            guard
                .failure_summary
                .contains("dense_sparse_mask_dispatch_present")
        );
        assert!(
            guard
                .failure_summary
                .contains("missing_accepted_prefill_dispatch")
        );
    }

    #[test]
    fn direct_sparse_decode_guard_accepts_direct_decode_path() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decode_decision(true)],
            0,
            3,
            &[dsa_sparse_attn_dispatch(1, 256, 256)],
        );

        let guard = build_direct_sparse_decode_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.decode_decisions, 1);
        assert_eq!(guard.decode_direct_decisions, 1);
        assert_eq!(guard.sparse_mask_nodes, 0);
        assert_eq!(guard.dense_sparse_mask_dispatches, 0);
        assert_eq!(guard.dsa_sparse_attn_nodes, 3);
        assert_eq!(guard.accepted_decode_dispatches, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_decode_guard_ignores_non_measured_fallback_decisions() {
        let mut candidate = direct_sparse_prefill_case_summary(
            &[
                direct_sparse_decode_decision(false),
                direct_sparse_decode_decision(true),
            ],
            0,
            3,
            &[dsa_sparse_attn_dispatch(1, 256, 256)],
        );
        candidate.direct_sparse_non_measured_decision_records =
            vec![direct_sparse_decode_decision(false)];
        candidate.direct_sparse_execution_decision_records =
            vec![direct_sparse_decode_decision(true)];
        candidate.direct_sparse_decision_summary = summarize_direct_sparse_decisions(&candidate);

        let guard = build_direct_sparse_decode_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.direct_decision_records, 1);
        assert_eq!(guard.decode_direct_decisions, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn direct_sparse_summary_reports_phase_and_selector_reason() {
        let candidate = direct_sparse_prefill_case_summary(
            &[
                direct_sparse_decode_decision(true),
                direct_sparse_decision(true, true),
            ],
            0,
            2,
            &[
                dsa_sparse_attn_dispatch(1, 256, 32),
                decode_vec_dsa_sparse_attn_dispatch(4096, 2048, 256),
            ],
        );
        let summary = candidate.direct_sparse_decision_summary;

        assert_eq!(summary.phases.get("decode"), Some(&1));
        assert_eq!(summary.phases.get("prefill"), Some(&1));
        assert_eq!(summary.execution_phases.get("decode"), Some(&1));
        assert_eq!(summary.execution_phases.get("prefill"), Some(&1));
        assert_eq!(
            summary
                .execution_selector_reasons
                .get("dense_mask_guard_large_prefill"),
            Some(&1)
        );
    }

    #[test]
    fn direct_sparse_decode_guard_rejects_dense_mask_fallback() {
        let candidate = direct_sparse_prefill_case_summary(
            &[direct_sparse_decode_decision(false)],
            1,
            0,
            &[dsa_sparse_mask_dispatch()],
        );

        let guard = build_direct_sparse_decode_guard(&candidate);

        assert!(!guard.passed);
        assert_eq!(guard.decode_decisions, 1);
        assert_eq!(guard.decode_direct_decisions, 0);
        assert_eq!(guard.sparse_mask_nodes, 1);
        assert_eq!(guard.dense_sparse_mask_dispatches, 1);
        assert_eq!(guard.accepted_decode_dispatches, 0);
        assert!(guard.failure_summary.contains("no_decode_direct_decision"));
        assert!(guard.failure_summary.contains("fallback_decision_present"));
        assert!(guard.failure_summary.contains("sparse_mask_nodes_present"));
        assert!(
            guard
                .failure_summary
                .contains("dense_sparse_mask_dispatch_present")
        );
        assert!(
            guard
                .failure_summary
                .contains("missing_dsa_sparse_attn_evidence")
        );
        assert!(
            guard
                .failure_summary
                .contains("missing_accepted_decode_dispatch")
        );
    }

    #[test]
    fn partial_top_k_guard_accepts_kv_larger_than_sideband_width() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate
            .metal_dispatch_records
            .push(dsa_sparse_attn_dispatch_with_kv(2304, 2304, 2048, 32));
        candidate.timings.push(IterationTiming {
            iteration: 0,
            position_start: 0,
            elapsed_ms: 1.0,
            output_payload_bytes: 16 + 2 * 2048 * std::mem::size_of::<i32>(),
            output_flags: ACTIVATION_FLAG_GLM_DSA_TOP_K,
        });

        let guard = build_partial_top_k_guard(&candidate, None, 16, 2);

        assert!(guard.passed);
        assert_eq!(guard.partial_dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.max_kv, Some(2304));
        assert_eq!(guard.expected_sideband_i32_per_token, Some(2048));
        assert_eq!(guard.output_frames_with_expected_sideband, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn timing_breakdown_marks_single_token_runs_as_decode() {
        let timings = vec![iteration_timing(0, 1.0), iteration_timing(1, 2.0)];

        let breakdown = summarize_timing_breakdown(&timings, 1);

        assert_eq!(breakdown.measured_phase, Some("decode"));
        assert_eq!(breakdown.decode.samples, 2);
        assert_eq!(breakdown.decode.mean_ms, Some(1.5));
        assert_eq!(breakdown.prefill.samples, 0);
    }

    #[test]
    fn timing_breakdown_marks_multi_token_runs_as_prefill() {
        let timings = vec![iteration_timing(0, 4.0), iteration_timing(1, 8.0)];

        let breakdown = summarize_timing_breakdown(&timings, 2304);

        assert_eq!(breakdown.measured_phase, Some("prefill"));
        assert_eq!(breakdown.decode.samples, 0);
        assert_eq!(breakdown.prefill.samples, 2);
        assert_eq!(breakdown.prefill.mean_ms, Some(6.0));
    }

    #[test]
    fn partial_top_k_guard_accepts_optimized_dispatch_probe() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.timings.push(IterationTiming {
            iteration: 0,
            position_start: 0,
            elapsed_ms: 1.0,
            output_payload_bytes: 16 + 2 * 2048 * std::mem::size_of::<i32>(),
            output_flags: ACTIVATION_FLAG_GLM_DSA_TOP_K,
        });
        let mut optimized_probe = case_summary("optimized_dispatch_probe", 0, 0, 0);
        optimized_probe
            .metal_dispatch_records
            .push(dsa_sparse_attn_dispatch_with_kv(2304, 2304, 2048, 32));

        let guard = build_partial_top_k_guard(&candidate, Some(&optimized_probe), 16, 2);

        assert!(guard.passed);
        assert_eq!(guard.checked_case, "optimized_dispatch_probe");
        assert_eq!(guard.dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.partial_dsa_sparse_attn_dispatches, 1);
        assert_eq!(guard.max_kv, Some(2304));
        assert_eq!(guard.expected_sideband_i32_per_token, Some(2048));
        assert_eq!(guard.output_frames_with_expected_sideband, 1);
    }

    #[test]
    fn partial_top_k_guard_accepts_native_shared_consume_width() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.indexshare_trace_summary = IndexShareTraceSummary {
            records: 4,
            exec_records: 2,
            full_exec_records: 0,
            shared_exec_records: 2,
            shared_exec_with_input_top_k: 2,
            shared_exec_missing_input_top_k: 0,
            top_k_records: 0,
            top_k_from_indexer: 0,
            top_k_from_full_visible: 0,
            consume_records: 2,
            min_consume_width: Some(2048),
            max_consume_width: Some(2048),
            full_layers: vec![],
            shared_layers: vec![31],
            ..IndexShareTraceSummary::default()
        };
        candidate.timings.push(IterationTiming {
            iteration: 0,
            position_start: 0,
            elapsed_ms: 1.0,
            output_payload_bytes: 16,
            output_flags: 0,
        });
        let mut optimized_probe = case_summary("optimized_dispatch_probe", 0, 0, 0);
        optimized_probe
            .metal_dispatch_records
            .push(dsa_sparse_attn_dispatch_with_kv(1, 4352, 2048, 32));

        let guard = build_partial_top_k_guard(&candidate, Some(&optimized_probe), 16, 1);

        assert!(guard.passed);
        assert_eq!(guard.output_frames_with_expected_sideband, 0);
        assert_eq!(guard.native_shared_consume_records, 2);
        assert_eq!(guard.native_shared_consume_width_matches, 2);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn partial_top_k_guard_rejects_degenerate_or_missing_sideband() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate
            .metal_dispatch_records
            .push(dsa_sparse_attn_dispatch_with_kv(2048, 2048, 2048, 32));
        candidate.timings.push(IterationTiming {
            iteration: 0,
            position_start: 0,
            elapsed_ms: 1.0,
            output_payload_bytes: 16,
            output_flags: 0,
        });

        let guard = build_partial_top_k_guard(&candidate, None, 16, 2);

        assert!(!guard.passed);
        assert_eq!(guard.partial_dsa_sparse_attn_dispatches, 0);
        assert_eq!(guard.output_frames_with_expected_sideband, 0);
        assert!(
            guard
                .failure_summary
                .contains("missing_partial_top_k_dispatch")
        );
        assert!(
            guard
                .failure_summary
                .contains("missing_expected_top_k_width_proof")
        );
    }

    #[test]
    fn compact_flash_guard_accepts_typed_glm_flash_path() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.metal_dispatch_summary.flash_attn_ext_records = 3;
        candidate.metal_dispatch_summary.flash_attn_ext_vec_records = 3;
        candidate
            .metal_dispatch_summary
            .flash_attn_ext_glm_dsa_shape_records = 3;
        candidate
            .metal_dispatch_records
            .push(compact_get_rows_dispatch(
                "dsa_compact_k_topk_rows-30",
                "typed",
            ));
        candidate
            .metal_dispatch_records
            .push(compact_get_rows_dispatch(
                "dsa_compact_v_topk_rows-30",
                "typed",
            ));
        candidate
            .compact_flash_mask_records
            .push(compact_flash_mask_record(true));
        candidate
            .compact_flash_execution_mask_records
            .push(compact_flash_mask_record(true));

        let guard = build_compact_flash_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.flash_attn_ext_glm_dsa_shape_records, 3);
        assert_eq!(guard.compact_get_rows_typed_records, 2);
        assert_eq!(guard.compact_get_rows_promote_records, 0);
        assert_eq!(guard.dsa_sparse_attn_records, 0);
        assert_eq!(guard.execution_mask_omission_records, 1);
        assert_eq!(guard.omitted_mla_kq_mask_records, 1);
        assert_eq!(guard.materialized_mla_kq_mask_records, 0);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn compact_flash_guard_accepts_native_policy_without_cli_flag() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.flags.compact_flash_attn = false;
        candidate.metal_dispatch_summary.flash_attn_ext_records = 1;
        candidate.metal_dispatch_summary.flash_attn_ext_vec_records = 1;
        candidate
            .metal_dispatch_summary
            .flash_attn_ext_glm_dsa_shape_records = 1;
        candidate
            .metal_dispatch_records
            .push(compact_get_rows_dispatch(
                "dsa_compact_k_topk_rows-30",
                "typed",
            ));
        candidate
            .metal_dispatch_records
            .push(compact_get_rows_dispatch(
                "dsa_compact_v_topk_rows-30",
                "typed",
            ));
        candidate
            .compact_flash_execution_policy_records
            .push(compact_flash_policy_record(
                "decode",
                "decode_compact",
                true,
            ));
        candidate
            .compact_flash_execution_mask_records
            .push(compact_flash_mask_record(true));

        let guard = build_compact_flash_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.failure_summary, "none");
        assert_eq!(guard.policy_phase.as_deref(), Some("decode"));
        assert_eq!(
            guard.policy_selector_reason.as_deref(),
            Some("decode_compact")
        );
    }

    #[test]
    fn compact_flash_guard_accepts_fused_compact_get_rows_path() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.flags.compact_flash_attn = true;
        candidate.metal_dispatch_summary.flash_attn_ext_records = 1;
        candidate.metal_dispatch_summary.flash_attn_ext_vec_records = 1;
        candidate
            .metal_dispatch_summary
            .flash_attn_ext_glm_dsa_shape_records = 1;
        candidate
            .metal_dispatch_summary
            .dsa_compact_get_rows_fused_records = 1;
        candidate
            .compact_flash_execution_mask_records
            .push(compact_flash_mask_record(true));

        let guard = build_compact_flash_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.compact_get_rows_typed_records, 0);
        assert_eq!(guard.dsa_compact_get_rows_fused_records, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn compact_flash_guard_accepts_vec4_typed_compact_get_rows_path() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.flags.compact_flash_attn = true;
        candidate.metal_dispatch_summary.flash_attn_ext_records = 1;
        candidate.metal_dispatch_summary.flash_attn_ext_vec_records = 1;
        candidate
            .metal_dispatch_summary
            .flash_attn_ext_glm_dsa_shape_records = 1;
        candidate
            .metal_dispatch_records
            .push(compact_get_rows_dispatch(
                "dsa_compact_k_topk_rows-30",
                "typed_vec4",
            ));
        candidate
            .compact_flash_execution_mask_records
            .push(compact_flash_mask_record(true));

        let guard = build_compact_flash_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.compact_get_rows_typed_records, 1);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn compact_flash_guard_rejects_fallback_or_promoted_path() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.flags.compact_flash_attn = true;
        candidate.metal_dispatch_summary.flash_attn_ext_records = 1;
        candidate
            .metal_dispatch_records
            .push(compact_get_rows_dispatch(
                "dsa_compact_k_topk_rows-30",
                "promote",
            ));
        candidate.metal_dispatch_summary.dsa_sparse_attn_records = 1;
        candidate.op_timing_summary.sparse_mask.nodes = 1;
        candidate
            .compact_flash_execution_mask_records
            .push(compact_flash_mask_record(false));

        let guard = build_compact_flash_guard(&candidate);

        assert!(!guard.passed);
        assert!(
            guard
                .failure_summary
                .contains("missing_glm_shape_flash_attn_ext")
        );
        assert!(guard.failure_summary.contains("missing_vec_flash_attn_ext"));
        assert!(guard.failure_summary.contains("missing_compact_get_rows"));
        assert!(guard.failure_summary.contains("promoted_get_rows_present"));
        assert!(
            guard
                .failure_summary
                .contains("dsa_sparse_attn_dispatch_present")
        );
        assert!(guard.failure_summary.contains("sparse_mask_nodes_present"));
        assert!(guard.failure_summary.contains("mla_kq_mask_materialized"));
    }

    #[test]
    fn compact_flash_guard_rejects_unrelated_get_rows_records() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.metal_dispatch_summary.flash_attn_ext_records = 1;
        candidate.metal_dispatch_summary.flash_attn_ext_vec_records = 1;
        candidate
            .metal_dispatch_summary
            .flash_attn_ext_glm_dsa_shape_records = 1;
        candidate
            .metal_dispatch_records
            .push(compact_get_rows_dispatch("ffn_moe_weights-30", "typed"));
        candidate
            .compact_flash_execution_mask_records
            .push(compact_flash_mask_record(true));

        let guard = build_compact_flash_guard(&candidate);

        assert!(!guard.passed);
        assert_eq!(guard.compact_get_rows_records, 0);
        assert!(guard.failure_summary.contains("missing_compact_get_rows"));
    }

    #[test]
    fn moe_weighted_sum_guard_accepts_f32x4_dispatches() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.metal_dispatch_summary.moe_weighted_sum_records = 4;
        candidate
            .metal_dispatch_summary
            .moe_weighted_sum_f32x4_records = 4;

        let guard = build_moe_weighted_sum_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.moe_weighted_sum_records, 4);
        assert_eq!(guard.moe_weighted_sum_f32x4_records, 4);
        assert_eq!(guard.mul_mv_id_weighted_sum_fused_records, 0);
        assert_eq!(guard.mul_mv_id_weighted_sum_fused_q3_k_records, 0);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn moe_weighted_sum_guard_accepts_q3_k_fused_dispatches() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate
            .metal_dispatch_summary
            .mul_mv_id_weighted_sum_fused_records = 4;
        candidate
            .metal_dispatch_summary
            .mul_mv_id_weighted_sum_fused_q3_k_records = 4;

        let guard = build_moe_weighted_sum_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.moe_weighted_sum_records, 0);
        assert_eq!(guard.moe_weighted_sum_f32x4_records, 0);
        assert_eq!(guard.mul_mv_id_weighted_sum_fused_records, 4);
        assert_eq!(guard.mul_mv_id_weighted_sum_fused_q3_k_records, 4);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn moe_weighted_sum_guard_accepts_q3_k_weighted_slots_dispatches() {
        let mut candidate = case_summary("candidate", 0, 0, 0);
        candidate.metal_dispatch_summary.moe_weighted_sum_records = 4;
        candidate
            .metal_dispatch_summary
            .moe_weighted_sum_already_weighted_records = 4;
        candidate
            .metal_dispatch_summary
            .mul_mv_id_weighted_slots_records = 4;
        candidate
            .metal_dispatch_summary
            .mul_mv_id_weighted_slots_q3_k_records = 4;

        let guard = build_moe_weighted_sum_guard(&candidate);

        assert!(guard.passed);
        assert_eq!(guard.moe_weighted_sum_records, 4);
        assert_eq!(guard.moe_weighted_sum_already_weighted_records, 4);
        assert_eq!(guard.mul_mv_id_weighted_slots_records, 4);
        assert_eq!(guard.mul_mv_id_weighted_slots_q3_k_records, 4);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn moe_weighted_sum_guard_rejects_missing_or_non_f32x4_dispatches() {
        let missing = case_summary("candidate", 0, 0, 0);

        let guard = build_moe_weighted_sum_guard(&missing);

        assert!(!guard.passed);
        assert!(guard.failure_summary.contains("missing_moe_weighted_sum"));
        assert!(
            guard
                .failure_summary
                .contains("missing_moe_weighted_sum_f32x4")
        );
        assert!(
            guard
                .failure_summary
                .contains("missing_mul_mv_id_weighted_sum_fused")
        );

        let mut scalar = case_summary("candidate", 0, 0, 0);
        scalar.metal_dispatch_summary.moe_weighted_sum_records = 4;
        scalar.metal_dispatch_summary.moe_weighted_sum_f32x4_records = 3;

        let guard = build_moe_weighted_sum_guard(&scalar);

        assert!(!guard.passed);
        assert!(
            guard
                .failure_summary
                .contains("non_f32x4_moe_weighted_sum_present")
        );

        let mut partial_fused = case_summary("candidate", 0, 0, 0);
        partial_fused
            .metal_dispatch_summary
            .mul_mv_id_weighted_sum_fused_records = 4;
        partial_fused
            .metal_dispatch_summary
            .mul_mv_id_weighted_sum_fused_q3_k_records = 3;

        let guard = build_moe_weighted_sum_guard(&partial_fused);

        assert!(!guard.passed);
        assert!(
            guard
                .failure_summary
                .contains("non_q3_k_mul_mv_id_weighted_sum_fused_present")
        );
    }

    #[test]
    fn moe_motif_guard_accepts_optimized_backend_candidates() {
        let candidate = case_summary("candidate", 0, 0, 0);
        let mut optimized_probe = case_summary("optimized_dispatch_probe", 0, 0, 0);
        optimized_probe
            .metal_dispatch_summary
            .glm_dsa_moe_motif_candidate_records = 4;
        optimized_probe
            .metal_dispatch_summary
            .glm_dsa_moe_motif_natural_order_records = 4;
        optimized_probe
            .metal_dispatch_summary
            .glm_dsa_moe_motif_backend_candidate_records = 4;
        optimized_probe
            .metal_dispatch_summary
            .glm_dsa_moe_motif_subgraph_fusable_records = 4;
        optimized_probe
            .metal_dispatch_summary
            .glm_dsa_moe_motif_max_nodes = 4;

        let guard = build_moe_motif_guard(&candidate, Some(&optimized_probe));

        assert!(guard.passed);
        assert_eq!(guard.checked_case, "optimized_dispatch_probe");
        assert_eq!(guard.motif_candidate_records, 4);
        assert_eq!(guard.natural_order_records, 4);
        assert_eq!(guard.backend_candidate_records, 4);
        assert_eq!(guard.subgraph_fusable_records, 4);
        assert_eq!(guard.max_motif_nodes, 4);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn moe_motif_guard_rejects_missing_or_partial_candidates() {
        let missing = case_summary("candidate", 0, 0, 0);

        let guard = build_moe_motif_guard(&missing, None);

        assert!(!guard.passed);
        assert!(
            guard
                .failure_summary
                .contains("missing_moe_motif_candidate")
        );
        assert!(guard.failure_summary.contains("motif_too_small"));

        let mut partial = case_summary("candidate", 0, 0, 0);
        partial
            .metal_dispatch_summary
            .glm_dsa_moe_motif_candidate_records = 2;
        partial
            .metal_dispatch_summary
            .glm_dsa_moe_motif_natural_order_records = 2;
        partial
            .metal_dispatch_summary
            .glm_dsa_moe_motif_backend_candidate_records = 1;
        partial
            .metal_dispatch_summary
            .glm_dsa_moe_motif_subgraph_fusable_records = 2;
        partial.metal_dispatch_summary.glm_dsa_moe_motif_max_nodes = 4;

        let guard = build_moe_motif_guard(&partial, None);

        assert!(!guard.passed);
        assert!(
            guard
                .failure_summary
                .contains("non_backend_candidate_motif_present")
        );
    }

    #[test]
    fn sideband_i32_diff_reports_order_and_set_mismatches() {
        let baseline = i32_sideband_bytes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let candidate = i32_sideband_bytes(&[0, 2, 1, 3, 4, 5, 8, 7]);

        let comparison = compare_sideband_payloads(&baseline, &candidate, Some(2)).unwrap();

        assert!(!comparison.exact_match);
        assert!(!comparison.semantic_match);
        let diff = comparison.i32_diff.expect("decoded i32 diff");
        assert!(diff.i32_aligned);
        assert_eq!(diff.baseline_i32_count, 8);
        assert_eq!(diff.candidate_i32_count, 8);
        assert_eq!(diff.mismatched_i32, 3);
        assert_eq!(diff.first_mismatches[0].index, 1);
        assert_eq!(diff.first_mismatches[0].token_index, Some(0));
        assert_eq!(diff.first_mismatches[0].offset_in_token, Some(1));
        assert_eq!(diff.first_mismatches[0].baseline, 1);
        assert_eq!(diff.first_mismatches[0].candidate, 2);

        let token_summary = diff.token_summary.expect("token summary");
        assert_eq!(token_summary.token_count, 2);
        assert_eq!(token_summary.width, 4);
        assert_eq!(token_summary.exact_order_matching_tokens, 0);
        assert_eq!(token_summary.set_equivalent_tokens, 1);
        assert_eq!(token_summary.set_mismatched_tokens, 1);

        let first_set_mismatch = token_summary
            .first_set_mismatch
            .expect("first set mismatch");
        assert_eq!(first_set_mismatch.token_index, 1);
        assert_eq!(first_set_mismatch.baseline_only, vec![6]);
        assert_eq!(first_set_mismatch.candidate_only, vec![8]);
    }

    #[test]
    fn sideband_i32_order_only_diff_matches_semantically() {
        let baseline = i32_sideband_bytes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let candidate = i32_sideband_bytes(&[1, 0, 3, 2, 7, 6, 5, 4]);

        let comparison = compare_sideband_payloads(&baseline, &candidate, Some(2)).unwrap();

        assert!(!comparison.exact_match);
        assert!(comparison.semantic_match);
        let token_summary = comparison
            .i32_diff
            .and_then(|diff| diff.token_summary)
            .expect("token summary");
        assert_eq!(token_summary.exact_order_matching_tokens, 0);
        assert_eq!(token_summary.set_equivalent_tokens, 2);
        assert_eq!(token_summary.set_mismatched_tokens, 0);
    }

    #[test]
    fn poison_top_k_sideband_collapses_each_token_and_requires_output_change() {
        let mut args = test_args();
        args.activation_width = 4;
        args.tokens = 1;
        let mut payload = vec![0_u8; 16];
        payload.extend_from_slice(&i32_sideband_bytes(&[3, 5, 8, 13]));
        let frame = ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 2,
                layer_end: 3,
                token_count: 1,
                sequence_count: 1,
                payload_bytes: payload.len() as u64,
                flags: ACTIVATION_FLAG_GLM_DSA_TOP_K,
            },
            payload,
        };

        let poisoned = poison_top_k_sideband(&args, &frame).unwrap();
        let sideband_values = decode_i32_sideband(&poisoned.frame.payload[16..]).unwrap();

        assert_eq!(sideband_values, vec![3, 3, 3, 3]);
        assert_eq!(poisoned.report.changed_i32_count, 3);
        assert_eq!(poisoned.report.tokens_with_changes, 1);

        let parity = ParityComparison {
            passed: false,
            iterations: 1,
            atol: args.parity_atol,
            rtol: args.parity_rtol,
            hidden_mismatches: 4,
            sideband_mismatched_bytes: 0,
            hidden_max_abs_diff: 0.25,
            hidden_max_rel_diff: 1.0,
            frames: Vec::new(),
        };
        let sensitivity = build_sideband_sensitivity_report(&poisoned.report, &parity);

        assert!(sensitivity.passed);
        assert_eq!(sensitivity.failure_summary, "none");
    }

    #[test]
    fn optimized_dispatch_probe_runs_for_diagnostic_reports() {
        let flags = MicrobenchFlags {
            direct_sparse_attn: true,
            compact_flash_attn: false,
            allow_compact_flash_auto: false,
            direct_sparse_prefill: false,
            native_default_direct_sparse_prefill: false,
            enable_unproven_large_direct_sparse_prefill: false,
            direct_sparse_prefill_max_tokens: None,
            fused_sparse_mask: true,
            parallel_lightning_indexer: true,
            op_timing: true,
            native_indexshare_exec_log: false,
            metal_dispatch_log: true,
            metal_topk_moe_route_fusion: false,
            metal_topk_moe_route_fusion_native_default: false,
            moe_motif_coencode: false,
            moe_down_weighted_fusion: false,
            moe_down_weighted_parallel: false,
            moe_down_unweighted_slots: false,
            sparse_attn_threads: None,
            sparse_attn_group_heads: None,
            lightning_indexer_threads: None,
            dense_sparse_mask_max_bytes: None,
            direct_sparse_prefill_min_kv_topk_ratio: None,
        };

        assert!(should_run_optimized_dispatch_probe(flags, false));
    }

    #[test]
    fn optimized_dispatch_probe_runs_for_clean_route_fusion_reports() {
        let flags = MicrobenchFlags {
            direct_sparse_attn: true,
            compact_flash_attn: false,
            allow_compact_flash_auto: false,
            direct_sparse_prefill: false,
            native_default_direct_sparse_prefill: false,
            enable_unproven_large_direct_sparse_prefill: false,
            direct_sparse_prefill_max_tokens: None,
            fused_sparse_mask: true,
            parallel_lightning_indexer: true,
            op_timing: false,
            native_indexshare_exec_log: false,
            metal_dispatch_log: false,
            metal_topk_moe_route_fusion: true,
            metal_topk_moe_route_fusion_native_default: false,
            moe_motif_coencode: false,
            moe_down_weighted_fusion: false,
            moe_down_weighted_parallel: false,
            moe_down_unweighted_slots: false,
            sparse_attn_threads: None,
            sparse_attn_group_heads: None,
            lightning_indexer_threads: None,
            dense_sparse_mask_max_bytes: None,
            direct_sparse_prefill_min_kv_topk_ratio: None,
        };

        assert!(should_run_optimized_dispatch_probe(flags, false));
    }

    #[test]
    fn optimized_dispatch_probe_skips_clean_non_fusion_reports() {
        let flags = MicrobenchFlags {
            direct_sparse_attn: true,
            compact_flash_attn: false,
            allow_compact_flash_auto: false,
            direct_sparse_prefill: false,
            native_default_direct_sparse_prefill: false,
            enable_unproven_large_direct_sparse_prefill: false,
            direct_sparse_prefill_max_tokens: None,
            fused_sparse_mask: true,
            parallel_lightning_indexer: true,
            op_timing: false,
            native_indexshare_exec_log: false,
            metal_dispatch_log: false,
            metal_topk_moe_route_fusion: false,
            metal_topk_moe_route_fusion_native_default: false,
            moe_motif_coencode: false,
            moe_down_weighted_fusion: false,
            moe_down_weighted_parallel: false,
            moe_down_unweighted_slots: false,
            sparse_attn_threads: None,
            sparse_attn_group_heads: None,
            lightning_indexer_threads: None,
            dense_sparse_mask_max_bytes: None,
            direct_sparse_prefill_min_kv_topk_ratio: None,
        };

        assert!(!should_run_optimized_dispatch_probe(flags, false));
        assert!(should_run_optimized_dispatch_probe(flags, true));
    }

    #[test]
    fn optimized_dispatch_probe_runs_for_clean_moe_down_weighted_reports() {
        let mut flags = test_microbench_flags();
        flags.op_timing = false;
        flags.metal_dispatch_log = false;
        flags.metal_topk_moe_route_fusion = false;
        flags.moe_down_weighted_fusion = true;

        assert!(should_run_optimized_dispatch_probe(flags, false));
    }

    #[test]
    fn route_fusion_env_plan_uses_legacy_override_by_default() {
        let mut flags = test_microbench_flags();
        flags.metal_topk_moe_route_fusion = true;
        flags.metal_topk_moe_route_fusion_native_default = false;
        assert_eq!(
            metal_topk_moe_route_fusion_env_plan(flags),
            RouteFusionEnvPlan::LegacyOverride(true)
        );

        flags.metal_topk_moe_route_fusion = false;
        assert_eq!(
            metal_topk_moe_route_fusion_env_plan(flags),
            RouteFusionEnvPlan::LegacyOverride(false)
        );
    }

    #[test]
    fn route_fusion_env_plan_can_prove_native_default() {
        let mut flags = test_microbench_flags();
        flags.metal_topk_moe_route_fusion = true;
        flags.metal_topk_moe_route_fusion_native_default = true;
        assert_eq!(
            metal_topk_moe_route_fusion_env_plan(flags),
            RouteFusionEnvPlan::NativeDefault
        );
    }

    #[test]
    fn representative_profile_uses_optimized_probe_when_op_timing_perturbs_fusion() {
        let mut candidate = case_summary("candidate", 4, 4, 0);
        candidate.timing_summary = single_sample_timing(153.0);
        let mut optimized_probe = case_summary("optimized_dispatch_probe", 4, 0, 4);
        optimized_probe.timing_summary = single_sample_timing(1.5);
        let mut flags = candidate.flags;
        flags.op_timing = true;
        flags.metal_dispatch_log = true;

        let integrity = ProfileIntegrityReport::new(
            flags,
            &candidate.metal_dispatch_summary,
            &candidate.timing_summary,
            Some(&optimized_probe),
        );
        let profile =
            RepresentativeProfileReport::new(&candidate, Some(&optimized_probe), &integrity);

        assert_eq!(profile.checked_case, "optimized_dispatch_probe");
        assert_eq!(profile.source, "optimized_dispatch_probe");
        assert!(profile.diagnostic_timing_discarded);
        assert_eq!(profile.timing_summary.mean_ms, Some(1.5));
        assert_eq!(
            profile.metal_dispatch_summary.topk_moe_route_fused_records,
            4
        );
    }

    #[test]
    fn representative_profile_uses_candidate_when_diagnostic_is_representative() {
        let mut candidate = case_summary("candidate", 4, 0, 4);
        candidate.timing_summary = single_sample_timing(1.5);
        let flags = candidate.flags;

        let integrity = ProfileIntegrityReport::new(
            flags,
            &candidate.metal_dispatch_summary,
            &candidate.timing_summary,
            None,
        );
        let profile = RepresentativeProfileReport::new(&candidate, None, &integrity);

        assert_eq!(profile.checked_case, "candidate");
        assert_eq!(profile.source, "candidate");
        assert!(!profile.diagnostic_timing_discarded);
        assert_eq!(profile.timing_summary.mean_ms, Some(1.5));
        assert_eq!(
            profile.metal_dispatch_summary.topk_moe_route_fused_records,
            4
        );
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
        assert_eq!(report.sideband.token_count, 1);
        assert_eq!(report.sideband.hidden_bytes, 16);
        assert_eq!(report.sideband.sideband_bytes, 16);
        assert_eq!(report.sideband.sideband_i32_count, 4);
        assert_eq!(report.sideband.sideband_i32_per_token, Some(4));
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
    fn validate_args_accepts_last_context_position() {
        let mut args = test_args();
        args.ctx_size = 4096;
        args.position_start = 4095;
        args.tokens = 1;

        validate_args(&args).unwrap();
        assert_eq!(max_requested_position_end(&args).unwrap(), 4095);
    }

    #[test]
    fn validate_args_rejects_one_past_context_position() {
        let mut args = test_args();
        args.ctx_size = 4096;
        args.position_start = 4096;
        args.tokens = 1;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(error.contains("requested position_end 4096 must be less than ctx_size 4096"));
        assert!(error.contains("valid positions are 0..4095"));
    }

    #[test]
    fn validate_args_checks_streamed_final_position() {
        let mut args = test_args();
        args.ctx_size = 10;
        args.position_start = 6;
        args.tokens = 2;
        args.warmup = 1;
        args.iterations = 2;
        args.reuse_kv_warmup_stream = true;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(error.contains("requested position_end 11 must be less than ctx_size 10"));
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
    fn real_top_k_shared_consumer_guard_accepts_stage_wire_proof() {
        let execution = shared_consumer_execution_contract_for_test();
        let stage_wire = stage_wire_roundtrip_report_for_test(true);

        let guard = build_real_top_k_shared_consumer_guard(&execution, Some(&stage_wire));

        assert!(guard.passed);
        assert_eq!(
            guard.proof_kind,
            ExecutionProofKind::SharedConsumerWithRealTopK
        );
        assert_eq!(guard.policy_role, IndexShareRole::SharedConsumer);
        assert_eq!(guard.effective_role, IndexShareRole::SharedConsumer);
        assert_eq!(guard.sideband_source_kind, SidebandSourceKind::RealTopK);
        assert_eq!(guard.stage_wire_roundtrip_passed, Some(true));
        assert_eq!(guard.stage_wire_sideband_bytes_match, Some(true));
        assert_eq!(guard.stage_wire_sideband_checksum_match, Some(true));
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn real_top_k_shared_consumer_guard_requires_stage_wire_proof() {
        let execution = shared_consumer_execution_contract_for_test();

        let guard = build_real_top_k_shared_consumer_guard(&execution, None);

        assert!(!guard.passed);
        assert!(guard.native_consumer_execution_proven);
        assert!(!guard.stage_wire_roundtrip_present);
        assert!(
            guard
                .failure_summary
                .contains("stage_wire_roundtrip_missing_or_failed")
        );
    }

    #[test]
    fn native_indexshare_guard_accepts_full_producer_shared_consumer_trace() {
        let case = microbench_case_with_indexshare_trace(IndexShareTraceSummary {
            records: 4,
            exec_records: 2,
            full_exec_records: 1,
            shared_exec_records: 1,
            shared_exec_with_input_top_k: 1,
            shared_exec_missing_input_top_k: 0,
            top_k_records: 1,
            top_k_from_indexer: 1,
            top_k_from_full_visible: 0,
            consume_records: 1,
            min_consume_width: Some(2048),
            max_consume_width: Some(2048),
            full_layers: vec![30],
            shared_layers: vec![31],
            ..IndexShareTraceSummary::default()
        });

        let guard = build_native_indexshare_guard(&case);

        assert!(guard.passed);
        assert_eq!(guard.checked_case, "candidate");
        assert_eq!(guard.full_layers, vec![30]);
        assert_eq!(guard.shared_layers, vec![31]);
        assert_eq!(guard.failure_summary, "none");
    }

    #[test]
    fn native_indexshare_guard_rejects_shared_exec_without_top_k() {
        let case = microbench_case_with_indexshare_trace(IndexShareTraceSummary {
            records: 2,
            exec_records: 2,
            full_exec_records: 1,
            shared_exec_records: 1,
            shared_exec_with_input_top_k: 0,
            shared_exec_missing_input_top_k: 1,
            top_k_records: 1,
            top_k_from_indexer: 1,
            top_k_from_full_visible: 0,
            consume_records: 0,
            min_consume_width: None,
            max_consume_width: None,
            full_layers: vec![30],
            shared_layers: vec![31],
            ..IndexShareTraceSummary::default()
        });

        let guard = build_native_indexshare_guard(&case);

        assert!(!guard.passed);
        assert!(
            guard
                .failure_summary
                .contains("shared_exec_missing_input_top_k")
        );
        assert!(guard.failure_summary.contains("missing_shared_consume"));
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

    fn timing_record(tokens: u64, total_us: u64) -> TimingRecord {
        TimingRecord {
            stage: 0,
            tokens,
            total_us,
            indexer_topk_nodes: 0,
            indexer_topk_us: 0,
            indexer_nodes: Some(0),
            indexer_us: Some(0),
            top_k_nodes: Some(0),
            top_k_us: Some(0),
            sparse_mask_nodes: 0,
            sparse_mask_us: 0,
            sparse_mask_fill_nodes: Some(0),
            sparse_mask_fill_us: Some(0),
            sparse_mask_topk_nodes: Some(0),
            sparse_mask_topk_us: Some(0),
            sparse_mask_add_nodes: Some(0),
            sparse_mask_add_us: Some(0),
            dsa_sparse_attn_nodes: Some(0),
            dsa_sparse_attn_us: Some(0),
            compact_get_rows_nodes: Some(0),
            compact_get_rows_us: Some(0),
            mla_attention_nodes: 0,
            mla_attention_us: 0,
            routed_moe_nodes: 0,
            routed_moe_us: 0,
            routed_moe_route_nodes: Some(0),
            routed_moe_route_us: Some(0),
            routed_moe_gate_up_nodes: Some(0),
            routed_moe_gate_up_us: Some(0),
            routed_moe_gate_nodes: Some(0),
            routed_moe_gate_us: Some(0),
            routed_moe_up_nodes: Some(0),
            routed_moe_up_us: Some(0),
            routed_moe_act_nodes: Some(0),
            routed_moe_act_us: Some(0),
            routed_moe_down_nodes: Some(0),
            routed_moe_down_us: Some(0),
            routed_moe_weighted_nodes: Some(0),
            routed_moe_weighted_us: Some(0),
            routed_moe_aggregate_nodes: Some(0),
            routed_moe_aggregate_us: Some(0),
            shared_expert_nodes: 0,
            shared_expert_us: 0,
        }
    }

    fn timing_group_record(record_index: usize, tokens: u64, total_us: u64) -> TimingGroupRecord {
        TimingGroupRecord {
            record_index,
            group: "layer_30".to_string(),
            timing: timing_record(tokens, total_us),
        }
    }

    fn hot_tensor_record(record_index: usize, tokens: u64, name: &str) -> HotTensorRecord {
        HotTensorRecord {
            record_index,
            stage: 0,
            tokens,
            rank: 1,
            op: "MUL_MAT".to_string(),
            kind: "test".to_string(),
            elapsed_us: 1,
            name: name.to_string(),
            ne0: 1,
            ne1: 1,
            ne2: 1,
            ne3: 1,
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
            kv_warmup_tokens: 0,
            kv_warmup_chunk_tokens: None,
            synthetic_kv_warmup: false,
            reuse_kv_warmup_checkpoint: false,
            reuse_kv_warmup_stream: false,
            iterations: 1,
            warmup: 0,
            n_gpu_layers: -1,
            n_batch: None,
            n_ubatch: None,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            direct_sparse_attn: true,
            compact_flash_attn: false,
            allow_compact_flash_auto: false,
            direct_sparse_prefill: false,
            native_default_direct_sparse_prefill: false,
            enable_unproven_large_direct_sparse_prefill: false,
            direct_sparse_prefill_max_tokens: None,
            fused_sparse_mask: true,
            parallel_lightning_indexer: true,
            op_timing: true,
            metal_dispatch_log: false,
            metal_topk_moe_route_fusion: true,
            metal_topk_moe_route_fusion_native_default: false,
            moe_motif_coencode: false,
            moe_down_weighted_fusion: false,
            moe_down_weighted_parallel: false,
            moe_down_unweighted_slots: false,
            sparse_attn_threads: None,
            sparse_attn_group_heads: None,
            lightning_indexer_threads: None,
            indexshare_freq: None,
            indexshare_pattern: None,
            dense_sparse_mask_max_bytes: None,
            direct_sparse_prefill_min_kv_topk_ratio: None,
            require_optimized_route_fusion: false,
            require_direct_sparse_prefill_proof: false,
            require_direct_sparse_decode_proof: false,
            require_partial_top_k_proof: false,
            require_compact_flash_proof: false,
            require_moe_weighted_sum_proof: false,
            require_moe_motif_proof: false,
            require_native_indexshare_proof: false,
            require_real_top_k_shared_consumer_proof: false,
            compare_dense_fallback: false,
            compare_dense_flash_prefill: false,
            compare_cpu_direct_sparse: false,
            compare_metal_sparse_attn_threads_baseline: None,
            compare_moe_down_weighted_fusion: false,
            compare_moe_down_weighted_parallel: false,
            compare_moe_down_unweighted_slots: false,
            compare_native_indexshare_producer_consumer: false,
            parity_atol: 1.0e-3,
            parity_rtol: 1.0e-3,
            allow_concurrent: false,
            output: None,
        }
    }

    #[test]
    fn glm_microbench_process_parser_matches_other_bench_process() {
        let process = parse_active_glm_microbench_process(
            "12345 ./target/release/skippy-bench glm-dsa-layer-microbench --stage-model /tmp/pkg",
            999,
        )
        .expect("expected active bench process");

        assert_eq!(process.pid, 12345);
        assert!(
            process
                .command
                .contains("glm-dsa-layer-microbench --stage-model")
        );
    }

    #[test]
    fn glm_microbench_process_parser_ignores_current_process() {
        let process = parse_active_glm_microbench_process(
            "12345 ./target/release/skippy-bench glm-dsa-layer-microbench --stage-model /tmp/pkg",
            12345,
        );

        assert_eq!(process, None);
    }

    #[test]
    fn glm_microbench_process_parser_ignores_shell_wrappers() {
        let process = parse_active_glm_microbench_process(
            "12345 /bin/zsh -c target/release/skippy-bench glm-dsa-layer-microbench --stage-model /tmp/pkg",
            999,
        );

        assert_eq!(process, None);
    }

    #[test]
    fn reuse_kv_warmup_checkpoint_requires_warmup_tokens() {
        let mut args = test_args();
        args.reuse_kv_warmup_checkpoint = true;
        args.kv_warmup_tokens = 0;

        assert!(!should_reuse_kv_warmup_checkpoint(&args));

        args.kv_warmup_tokens = 128;

        assert!(should_reuse_kv_warmup_checkpoint(&args));
    }

    #[test]
    fn real_top_k_source_layer_zero_is_valid_before_shared_consumer() {
        assert_eq!(
            parse_real_top_k_source_layer_start("0", 1, ENV_REAL_TOP_K_SOURCE_LAYER_START).unwrap(),
            Some(0)
        );
        assert_eq!(
            parse_real_top_k_source_layer_start("off", 1, ENV_REAL_TOP_K_SOURCE_LAYER_START)
                .unwrap(),
            None
        );
        assert!(
            parse_real_top_k_source_layer_start("1", 1, ENV_REAL_TOP_K_SOURCE_LAYER_START).is_err()
        );
    }

    #[test]
    fn real_top_k_warmup_source_layer_uses_env_name_in_errors() {
        let error =
            parse_real_top_k_source_layer_start("3", 3, ENV_REAL_TOP_K_WARMUP_SOURCE_LAYER_START)
                .expect_err("warmup source must be before target");
        assert!(
            error
                .to_string()
                .contains(ENV_REAL_TOP_K_WARMUP_SOURCE_LAYER_START)
        );
    }

    #[test]
    fn reuse_kv_warmup_stream_is_independent_of_checkpoint_mode() {
        let mut args = test_args();
        args.reuse_kv_warmup_stream = true;
        args.kv_warmup_tokens = 0;

        assert!(should_reuse_kv_warmup_stream(&args));
        assert!(!should_reuse_kv_warmup_checkpoint(&args));
    }

    #[test]
    fn streamed_iteration_positions_advance_by_token_count() {
        let mut args = test_args();
        args.position_start = 16_384;
        args.tokens = 2;

        assert_eq!(streamed_iteration_position_start(&args, 0).unwrap(), 16_384);
        assert_eq!(streamed_iteration_position_start(&args, 1).unwrap(), 16_386);
        assert_eq!(streamed_iteration_position_start(&args, 7).unwrap(), 16_398);
    }

    #[test]
    fn kv_warmup_chunk_uses_explicit_override_first() {
        let mut args = test_args();
        args.kv_warmup_tokens = 65_536;
        args.n_batch = Some(256);
        args.n_ubatch = Some(512);
        args.kv_warmup_chunk_tokens = Some(1024);

        assert_eq!(kv_warmup_chunk_tokens(&args), 1024);
    }

    #[test]
    fn kv_warmup_chunk_uses_runtime_batch_when_no_override() {
        let mut args = test_args();
        args.kv_warmup_tokens = 65_536;
        args.n_batch = Some(256);
        args.n_ubatch = Some(512);

        assert_eq!(kv_warmup_chunk_tokens(&args), 512);

        args.n_ubatch = None;

        assert_eq!(kv_warmup_chunk_tokens(&args), 256);

        args.n_batch = None;

        assert_eq!(kv_warmup_chunk_tokens(&args), 128);
    }

    #[test]
    fn synthetic_kv_page_desc_uses_glm_dsa_f16_layout() {
        let args = test_args();

        let desc = synthetic_kv_page_desc(args.layer_start, args.layer_end, 1, 1024, 256).unwrap();

        assert_eq!(desc.version, 1);
        assert_eq!(desc.layer_start, 30);
        assert_eq!(desc.layer_end, 31);
        assert_eq!(desc.token_start, 1024);
        assert_eq!(desc.token_count, 256);
        assert_eq!(desc.layer_count, 1);
        assert_eq!(desc.k_type, GGML_TYPE_F16_ID);
        assert_eq!(desc.v_type, GGML_TYPE_F32_ID);
        assert_eq!(desc.k_row_bytes, GLM_DSA_F16_K_ROW_BYTES);
        assert_eq!(desc.v_row_bytes, 0);
        assert_eq!(desc.v_element_bytes, 0);
        assert_eq!(desc.flags, 0);
        assert_eq!(desc.payload_bytes, 256 * u64::from(GLM_DSA_F16_K_ROW_BYTES));
    }

    #[test]
    fn validate_args_rejects_zero_kv_warmup_chunk_override() {
        let mut args = test_args();
        args.kv_warmup_chunk_tokens = Some(0);

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains("kv_warmup_chunk_tokens must be greater than zero"),
            "{error}"
        );
    }

    #[test]
    fn validate_args_rejects_synthetic_kv_without_warmup_tokens() {
        let mut args = test_args();
        args.synthetic_kv_warmup = true;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains("synthetic_kv_warmup requires kv_warmup_tokens"),
            "{error}"
        );
    }

    #[test]
    fn validate_args_rejects_top_k_warmup_chunk_larger_than_microbatch() {
        let mut args = test_args();
        args.compare_native_indexshare_producer_consumer = true;
        args.layer_start = 6;
        args.layer_end = 10;
        args.position_start = 256;
        args.kv_warmup_tokens = 256;
        args.kv_warmup_chunk_tokens = Some(64);
        args.n_batch = Some(64);
        args.n_ubatch = Some(4);

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains(
                "must fit within n_ubatch (4) when KV warmup exports GLM-DSA top-k sideband"
            ),
            "{error}"
        );
    }

    #[test]
    fn validate_args_allows_synthetic_kv_warmup_chunk_larger_than_microbatch() {
        let mut args = test_args();
        args.position_start = 256;
        args.kv_warmup_tokens = 256;
        args.kv_warmup_chunk_tokens = Some(64);
        args.n_batch = Some(64);
        args.n_ubatch = Some(4);
        args.synthetic_kv_warmup = true;

        validate_args(&args).unwrap();
    }

    #[test]
    fn validate_args_rejects_both_kv_reuse_modes() {
        let mut args = test_args();
        args.reuse_kv_warmup_checkpoint = true;
        args.reuse_kv_warmup_stream = true;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains("reuse_kv_warmup_checkpoint and reuse_kv_warmup_stream"),
            "{error}"
        );
    }

    #[test]
    fn validate_args_requires_two_layers_for_native_indexshare_comparison() {
        let mut args = test_args();
        args.layer_start = 30;
        args.layer_end = 31;
        args.compare_native_indexshare_producer_consumer = true;

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains(
                "compare_native_indexshare_producer_consumer requires at least two layers"
            ),
            "{error}"
        );
    }

    #[test]
    fn measured_timing_filters_drop_prefix_records_and_decode_warmup() {
        let records = vec![
            timing_record(128, 1000),
            timing_record(1, 10),
            timing_record(1, 20),
            timing_record(128, 2000),
            timing_record(1, 40),
        ];
        let measured = retain_measured_timing_records(records, 1, 1);

        assert_eq!(measured.len(), 2);
        assert_eq!(measured[0].total_us, 20);
        assert_eq!(measured[1].total_us, 40);

        let groups = vec![
            timing_group_record(0, 128, 1000),
            timing_group_record(1, 1, 10),
            timing_group_record(2, 1, 20),
            timing_group_record(3, 128, 2000),
            timing_group_record(4, 1, 40),
        ];
        let measured_groups = retain_measured_group_timing_records(groups, 1, 1);

        assert_eq!(measured_groups.len(), 2);
        assert_eq!(measured_groups[0].record_index, 0);
        assert_eq!(measured_groups[0].timing.total_us, 20);
        assert_eq!(measured_groups[1].record_index, 1);
        assert_eq!(measured_groups[1].timing.total_us, 40);

        let hot_tensors = vec![
            hot_tensor_record(0, 128, "prefix"),
            hot_tensor_record(1, 1, "decode-warmup"),
            hot_tensor_record(2, 1, "decode-0"),
            hot_tensor_record(4, 1, "decode-1"),
        ];
        let measured_hot_tensors = retain_measured_hot_tensor_records(hot_tensors, 1, 1);

        assert_eq!(measured_hot_tensors.len(), 2);
        assert_eq!(measured_hot_tensors[0].record_index, 0);
        assert_eq!(measured_hot_tensors[0].name, "decode-0");
        assert_eq!(measured_hot_tensors[1].record_index, 1);
        assert_eq!(measured_hot_tensors[1].name, "decode-1");
    }

    #[test]
    fn indexshare_timing_summary_classifies_producers_and_consumers() {
        let mut producer = timing_group_record(0, 1, 20_000);
        producer.group = "layer_30".to_string();
        producer.timing.indexer_topk_nodes = 25;
        producer.timing.indexer_topk_us = 8_000;
        producer.timing.indexer_nodes = Some(10);
        producer.timing.indexer_us = Some(5_000);
        producer.timing.top_k_nodes = Some(15);
        producer.timing.top_k_us = Some(3_000);
        let mut consumer_a = timing_group_record(0, 1, 11_000);
        consumer_a.group = "layer_31".to_string();
        let mut consumer_b = timing_group_record(0, 1, 12_000);
        consumer_b.group = "layer_32".to_string();
        let mut consumer_c = timing_group_record(0, 1, 13_000);
        consumer_c.group = "layer_33".to_string();

        let summary = summarize_indexshare_timing(&[consumer_c, producer, consumer_a, consumer_b]);

        assert_eq!(summary.records, 4);
        assert_eq!(summary.layer_groups, 4);
        assert_eq!(summary.producer_groups, 1);
        assert_eq!(summary.consumer_groups, 3);
        assert_eq!(summary.indexer_topk_nodes, 25);
        assert_eq!(summary.indexer_topk_us, 8_000);
        assert_eq!(summary.indexer_nodes, 10);
        assert_eq!(summary.indexer_us, 5_000);
        assert_eq!(summary.top_k_nodes, 15);
        assert_eq!(summary.top_k_us, 3_000);
        assert_eq!(summary.indexer_share_of_indexer_topk, Some(0.625));
        assert_eq!(summary.top_k_share_of_indexer_topk, Some(0.375));
        assert_eq!(summary.producer_total_us, 20_000);
        assert_eq!(summary.consumer_total_us, 36_000);
        assert_eq!(summary.producer_group_names, ["layer_30"]);
        assert_eq!(
            summary.consumer_group_names,
            ["layer_31", "layer_32", "layer_33"]
        );
    }

    #[test]
    fn validate_args_requires_candidate_threads_for_metal_thread_comparison() {
        let mut args = test_args();
        args.compare_metal_sparse_attn_threads_baseline = Some(32);

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains(
                "compare_metal_sparse_attn_threads_baseline requires sparse_attn_threads"
            ),
            "{error}"
        );
    }

    #[test]
    fn validate_args_rejects_unsupported_sparse_attn_thread_width() {
        let mut args = test_args();
        args.sparse_attn_threads = Some(96);

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains("sparse_attn_threads must be one of 32, 64, 128, or 256"),
            "{error}"
        );
    }

    #[test]
    fn validate_args_rejects_unsupported_lightning_indexer_thread_width() {
        let mut args = test_args();
        args.lightning_indexer_threads = Some(96);

        let error = validate_args(&args).unwrap_err().to_string();

        assert!(
            error.contains(
                "lightning_indexer_threads must be one of 32, 64, 128, 256, 512, or 1024"
            ),
            "{error}"
        );
    }

    fn shared_consumer_execution_contract_for_test() -> ExecutionContractReport {
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
        execution_contract_report(&args, &input, &input_contract, &policy, artifact_role)
    }

    fn stage_wire_roundtrip_report_for_test(passed: bool) -> StageWireRoundTripReport {
        StageWireRoundTripReport {
            kind: "DecodeEmbd".to_string(),
            wire_dtype: "F32".to_string(),
            state_flags: 1,
            activation_flag_bits: ACTIVATION_FLAG_GLM_DSA_TOP_K,
            token_count: 1,
            position_start: 0,
            token_sideband_count: 1,
            position_sideband_count: 1,
            hidden_activation_bytes: 16,
            raw_activation_wire_bytes: 16,
            top_k_sideband_bytes: 16,
            top_k_sideband_i32_count: 4,
            top_k_sideband_stats: TopKSidebandStats {
                i32_aligned: true,
                token_count: 1,
                total_i32: 4,
                width_per_token: Some(4),
                valid_index_count: 4,
                negative_index_count: 0,
                out_of_range_index_count: 0,
                causal_visible_count: 4,
                future_index_count: 0,
                duplicate_index_count: 0,
                active_top_end_sum: 4,
                avg_active_top_end: Some(4.0),
                max_active_top_end: 4,
                inactive_tail_count: 0,
                masked_future_in_active_prefix_count: 0,
                active_prefix_ratio: Some(1.0),
                causal_visible_ratio: Some(1.0),
            },
            estimated_wire_bytes: 32,
            encoded_wire_bytes: 32,
            decoded_payload_bytes: 32,
            decoded_payload_checksum: "payload".to_string(),
            decoded_sideband_checksum: "sideband".to_string(),
            payload_bytes_match: passed,
            flags_match: passed,
            sideband_bytes_match: passed,
            sideband_checksum_match: passed,
            passed,
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
                compact_flash_attn: false,
                allow_compact_flash_auto: false,
                direct_sparse_prefill: true,
                native_default_direct_sparse_prefill: false,
                enable_unproven_large_direct_sparse_prefill: false,
                direct_sparse_prefill_max_tokens: None,
                fused_sparse_mask: true,
                parallel_lightning_indexer: false,
                op_timing: false,
                native_indexshare_exec_log: false,
                metal_dispatch_log: true,
                metal_topk_moe_route_fusion: false,
                metal_topk_moe_route_fusion_native_default: false,
                moe_motif_coencode: false,
                moe_down_weighted_fusion: false,
                moe_down_weighted_parallel: false,
                moe_down_unweighted_slots: false,
                sparse_attn_threads: None,
                sparse_attn_group_heads: None,
                lightning_indexer_threads: None,
                dense_sparse_mask_max_bytes: None,
                direct_sparse_prefill_min_kv_topk_ratio: None,
            },
            n_gpu_layers: -1,
            native_log_path: None,
            compact_flash_policy_summary: CompactFlashPolicySummary::default(),
            compact_flash_policy_records: Vec::new(),
            compact_flash_execution_policy_records: Vec::new(),
            compact_flash_non_measured_policy_records: Vec::new(),
            compact_flash_mask_records: Vec::new(),
            compact_flash_execution_mask_records: Vec::new(),
            compact_flash_non_measured_mask_records: Vec::new(),
            direct_sparse_decision_summary: DirectSparseDecisionSummary::default(),
            timing_summary: TimingDistributionSummary::default(),
            timing_breakdown: TimingBreakdownSummary::default(),
            metal_dispatch_summary: dispatch,
            direct_sparse_spill_summary: DirectSparseSpillSummary::default(),
            op_timing_summary: GlmDsaOpTimingSummary::default(),
            routed_moe_timing_summary: RoutedMoeTimingSummary::default(),
            indexshare_timing_summary: IndexShareTimingSummary::default(),
            indexshare_trace_summary: IndexShareTraceSummary::default(),
            direct_sparse_decision_records: Vec::new(),
            direct_sparse_execution_decision_records: Vec::new(),
            direct_sparse_non_measured_decision_records: Vec::new(),
            metal_dispatch_records: Vec::new(),
            op_timing_records: Vec::new(),
            group_timing_records: Vec::new(),
            hot_tensor_records: Vec::new(),
            compute_buffer_records: Vec::new(),
            timings: Vec::new(),
        }
    }

    fn microbench_case_with_indexshare_trace(
        indexshare_trace_summary: IndexShareTraceSummary,
    ) -> MicrobenchCaseSummary {
        MicrobenchCase {
            label: "candidate",
            flags: MicrobenchFlags {
                direct_sparse_attn: true,
                compact_flash_attn: false,
                allow_compact_flash_auto: false,
                direct_sparse_prefill: false,
                native_default_direct_sparse_prefill: false,
                enable_unproven_large_direct_sparse_prefill: false,
                direct_sparse_prefill_max_tokens: None,
                fused_sparse_mask: true,
                parallel_lightning_indexer: false,
                op_timing: true,
                native_indexshare_exec_log: true,
                metal_dispatch_log: false,
                metal_topk_moe_route_fusion: true,
                metal_topk_moe_route_fusion_native_default: false,
                moe_motif_coencode: false,
                moe_down_weighted_fusion: false,
                moe_down_weighted_parallel: false,
                moe_down_unweighted_slots: false,
                sparse_attn_threads: None,
                sparse_attn_group_heads: None,
                lightning_indexer_threads: None,
                dense_sparse_mask_max_bytes: None,
                direct_sparse_prefill_min_kv_topk_ratio: None,
            },
            n_gpu_layers: -1,
            measured_tokens: 1,
            native_log_path: None,
            compact_flash_policy_records: Vec::new(),
            compact_flash_execution_policy_records: Vec::new(),
            compact_flash_non_measured_policy_records: Vec::new(),
            compact_flash_mask_records: Vec::new(),
            compact_flash_execution_mask_records: Vec::new(),
            compact_flash_non_measured_mask_records: Vec::new(),
            direct_sparse_decision_records: Vec::new(),
            direct_sparse_execution_decision_records: Vec::new(),
            direct_sparse_non_measured_decision_records: Vec::new(),
            metal_dispatch_records: Vec::new(),
            op_timing_records: Vec::new(),
            group_timing_records: Vec::new(),
            indexshare_trace_summary,
            hot_tensor_records: Vec::new(),
            compute_buffer_records: Vec::new(),
            timings: Vec::new(),
            outputs: Vec::new(),
        }
        .as_case_summary()
    }

    fn single_sample_timing(mean_ms: f64) -> TimingDistributionSummary {
        TimingDistributionSummary {
            samples: 1,
            mean_ms: Some(mean_ms),
            min_ms: Some(mean_ms),
            p50_ms: Some(mean_ms),
            p90_ms: Some(mean_ms),
            p95_ms: Some(mean_ms),
            p99_ms: Some(mean_ms),
            max_ms: Some(mean_ms),
            stdev_ms: Some(0.0),
            coefficient_of_variation: Some(0.0),
            slow_outlier_count: 0,
            slow_outlier_threshold_ms: Some(mean_ms * 1.25),
        }
    }

    fn iteration_timing(iteration: usize, elapsed_ms: f64) -> IterationTiming {
        IterationTiming {
            iteration,
            position_start: 0,
            elapsed_ms,
            output_payload_bytes: 16,
            output_flags: 0,
        }
    }

    fn direct_sparse_prefill_case_summary(
        decisions: &[DirectSparseDecisionRecord],
        sparse_mask_nodes: u64,
        dsa_sparse_attn_nodes: u64,
        dispatches: &[MetalDispatchRecord],
    ) -> MicrobenchCaseSummary {
        let mut case = case_summary("candidate", 0, 0, 0);
        case.direct_sparse_decision_records = decisions.to_vec();
        case.direct_sparse_execution_decision_records = decisions.to_vec();
        case.direct_sparse_decision_summary = summarize_direct_sparse_decisions(&case);
        case.op_timing_summary.records = 1;
        case.op_timing_summary.sparse_mask.nodes = sparse_mask_nodes;
        case.op_timing_summary.dsa_sparse_attn.nodes = dsa_sparse_attn_nodes;
        case.metal_dispatch_records = dispatches.to_vec();
        case.direct_sparse_spill_summary =
            summarize_direct_sparse_spill(&case.metal_dispatch_records);
        case
    }

    fn direct_sparse_decision(
        large_prefill_shape: bool,
        use_direct: bool,
    ) -> DirectSparseDecisionRecord {
        DirectSparseDecisionRecord {
            layer: 30,
            ubatch_tokens: 64,
            sparse_batch: 64,
            sparse_streams: 1,
            prefill_cap: 32,
            sparse_kv: Some(2048),
            sparse_top_k: Some(64),
            min_kv_topk_ratio: Some(32),
            kv_topk_ratio: Some(32),
            dense_mask_bytes: Some(524288),
            dense_mask_limit: Some(1),
            phase: Some("prefill".to_string()),
            selector_reason: Some(if use_direct {
                if large_prefill_shape {
                    "dense_mask_guard_large_prefill".to_string()
                } else {
                    "short_prefill".to_string()
                }
            } else {
                "fallback".to_string()
            }),
            direct_enabled: true,
            prefill_enabled: true,
            decode_shape: false,
            prefill_shape: !large_prefill_shape,
            large_prefill_shape: Some(large_prefill_shape),
            token_shape_allowed: true,
            kq_b_ok: true,
            sinks_ok: true,
            alibi_ok: true,
            soft_cap_ok: true,
            use_direct,
        }
    }

    fn direct_sparse_decode_decision(use_direct: bool) -> DirectSparseDecisionRecord {
        DirectSparseDecisionRecord {
            layer: 30,
            ubatch_tokens: 1,
            sparse_batch: 1,
            sparse_streams: 1,
            prefill_cap: 32,
            sparse_kv: Some(2048),
            sparse_top_k: Some(256),
            min_kv_topk_ratio: Some(32),
            kv_topk_ratio: Some(8),
            dense_mask_bytes: Some(8192),
            dense_mask_limit: Some(536_870_912),
            phase: Some("decode".to_string()),
            selector_reason: Some(if use_direct {
                "decode".to_string()
            } else {
                "fallback".to_string()
            }),
            direct_enabled: true,
            prefill_enabled: false,
            decode_shape: true,
            prefill_shape: false,
            large_prefill_shape: Some(false),
            token_shape_allowed: true,
            kq_b_ok: true,
            sinks_ok: true,
            alibi_ok: true,
            soft_cap_ok: true,
            use_direct,
        }
    }

    fn dsa_sparse_attn_dispatch(batch: u64, top_k: u64, threads_x: u64) -> MetalDispatchRecord {
        dsa_sparse_attn_dispatch_with_kv(batch, 256, top_k, threads_x)
    }

    fn cached_topk_dsa_sparse_attn_dispatch(
        batch: u64,
        top_k: u64,
        threads_x: u64,
    ) -> MetalDispatchRecord {
        let mut record = dsa_sparse_attn_dispatch_with_kv(batch, batch, top_k, threads_x);
        record.kernel = Some("cached_topk".to_string());
        record
    }

    fn cached_topk_v4_dsa_sparse_attn_dispatch(
        batch: u64,
        top_k: u64,
        threads_x: u64,
    ) -> MetalDispatchRecord {
        let mut record = dsa_sparse_attn_dispatch_with_kv(batch, batch, top_k, threads_x);
        record.kernel = Some("cached_topk_v4".to_string());
        record
    }

    fn decode_vec_dsa_sparse_attn_dispatch(
        batch: u64,
        top_k: u64,
        threads_x: u64,
    ) -> MetalDispatchRecord {
        let mut record = dsa_sparse_attn_dispatch_with_kv(batch, batch * 97, top_k, threads_x);
        record.kernel = Some("decode_vec".to_string());
        record
    }

    fn decode_vec_direct_dsa_sparse_attn_dispatch(
        batch: u64,
        top_k: u64,
        threads_x: u64,
    ) -> MetalDispatchRecord {
        let mut record = dsa_sparse_attn_dispatch_with_kv(batch, batch * 97, top_k, threads_x);
        record.kernel = Some("decode_vec_direct".to_string());
        record
    }

    fn dsa_sparse_mask_dispatch() -> MetalDispatchRecord {
        let mut record = dsa_sparse_attn_dispatch_with_kv(64, 256, 256, 32);
        record.op = "dsa_sparse_mask".to_string();
        record.kernel = Some("set".to_string());
        record.tensor = "dsa_sparse_mask-30".to_string();
        record
    }

    fn compact_get_rows_dispatch(tensor: &str, kernel: &str) -> MetalDispatchRecord {
        let mut record = dsa_sparse_attn_dispatch_with_kv(1, 2048, 2048, 576);
        record.op = "get_rows".to_string();
        record.kernel = Some(kernel.to_string());
        record.tensor = tensor.to_string();
        record
    }

    fn compact_flash_policy_record(
        phase: &str,
        selector_reason: &str,
        use_compact: bool,
    ) -> CompactFlashPolicyRecord {
        CompactFlashPolicyRecord {
            layer: 30,
            ubatch_tokens: 1,
            visible_kv: 4096,
            top_k: 2048,
            kv_topk_ratio: 2,
            min_kv_topk_ratio: 2,
            forced: false,
            disabled: false,
            ratio_ok: true,
            enabled: true,
            flash_attn: true,
            phase: Some(phase.to_string()),
            decode_shape: phase == "decode",
            kq_b_ok: true,
            sinks_ok: true,
            alibi_ok: true,
            soft_cap_ok: true,
            no_mask: Some(use_compact),
            use_compact,
            selector_reason: Some(selector_reason.to_string()),
        }
    }

    fn compact_flash_mask_record(omitted_mla_kq_mask: bool) -> CompactFlashMaskRecord {
        CompactFlashMaskRecord {
            layer: 30,
            omitted_mla_kq_mask,
            visible_kv: 4096,
            ubatch_tokens: 1,
            streams: 1,
            max_top_k: 2048,
        }
    }

    fn dsa_sparse_attn_dispatch_with_kv(
        batch: u64,
        kv: u64,
        top_k: u64,
        threads_x: u64,
    ) -> MetalDispatchRecord {
        MetalDispatchRecord {
            op: "dsa_sparse_attn".to_string(),
            kernel: Some("default".to_string()),
            tensor: "dsa_sparse_attn-30".to_string(),
            next: None,
            next_op: None,
            shared_gate: None,
            shared_up: None,
            weighted_sum: None,
            weighted_sum_op: None,
            reason: None,
            shared_branch: None,
            weighted_sum_uses_down: None,
            natural_order: None,
            backend_candidate: None,
            pair_fusable: None,
            subgraph_fusable: None,
            motif_nodes: None,
            fusion_outputs: None,
            filtered_gap: None,
            graph_gap: None,
            weighted_sum_gap: None,
            weighted_sum_graph_gap: None,
            parallel: None,
            q_type: Some("f32".to_string()),
            k_type: Some("f16".to_string()),
            v_type: Some("f16".to_string()),
            mask_type: Some("f16".to_string()),
            top_k_type: Some("i32".to_string()),
            src_type: None,
            dst_type: Some("f32".to_string()),
            q_width: Some(576),
            v_width: Some(512),
            batch: Some(batch),
            heads: Some(64),
            stream: Some(1),
            kv: Some(kv),
            top_k: Some(top_k),
            top_stream: Some(1),
            selected_keys: None,
            q_read_bytes: None,
            k_read_bytes: None,
            v_read_bytes: None,
            mask_read_bytes: None,
            top_k_read_bytes: None,
            scratch_per_tg_bytes: None,
            score_fma: None,
            value_fma: None,
            reduction_strategy: None,
            rows: None,
            partial_bytes: None,
            softmax_bytes: None,
            tmp_bytes: None,
            nwg: None,
            tmp_f16: None,
            dst_partial: None,
            grid_x: 64,
            grid_y: batch,
            grid_z: 1,
            threads_x,
            threads_y: Some(1),
        }
    }

    fn test_microbench_flags() -> MicrobenchFlags {
        MicrobenchFlags {
            direct_sparse_attn: true,
            compact_flash_attn: false,
            allow_compact_flash_auto: false,
            direct_sparse_prefill: true,
            native_default_direct_sparse_prefill: false,
            enable_unproven_large_direct_sparse_prefill: false,
            direct_sparse_prefill_max_tokens: None,
            fused_sparse_mask: true,
            parallel_lightning_indexer: false,
            op_timing: false,
            native_indexshare_exec_log: false,
            metal_dispatch_log: true,
            metal_topk_moe_route_fusion: true,
            metal_topk_moe_route_fusion_native_default: false,
            moe_motif_coencode: false,
            moe_down_weighted_fusion: false,
            moe_down_weighted_parallel: false,
            moe_down_unweighted_slots: false,
            sparse_attn_threads: None,
            sparse_attn_group_heads: None,
            lightning_indexer_threads: None,
            dense_sparse_mask_max_bytes: None,
            direct_sparse_prefill_min_kv_topk_ratio: None,
        }
    }

    fn i32_sideband_bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }
}
