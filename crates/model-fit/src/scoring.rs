use crate::{
    AcceleratorKind, AcceleratorProfile, BackendKind, CapabilityEvidence, DecodeEstimateRange,
    EstimateConfidence, FirstTokenEstimateRange, FitStatus, HardwareProfile, KvCacheKind,
    MatmulShapeProfile, MeasurementSource, ModelArchitectureClass, ModelProfile,
    ModelRecommendation, Requirement, ScoreWeights, SelectionConfig, SplitCandidateEstimate,
    TensorGroupBytes, TensorMatmulGroupProfile, TensorTypeBytes, WeightCoverage, WorkloadTask,
};
use std::cmp::Ordering;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const LLAMA_DEFAULT_UBATCH_TOKENS: u32 = 512;

#[derive(Clone, Debug)]
struct ExecutionBudget {
    backend: BackendKind,
    accelerator_name: Option<String>,
    accelerator_kind: AcceleratorKind,
    usable_memory_bytes: u64,
    memory_bandwidth_bytes_per_sec: Option<u64>,
    decode_effective_bandwidth_bytes_per_sec: Option<u64>,
    decode_fixed_overhead_ms: Option<f32>,
    post_prefill_decode_overhead_ms: Option<f32>,
    bandwidth_source: MeasurementSource,
    benchmark_noise_pct: Option<f32>,
    compute_tflops_fp16: Option<f32>,
    prefill_matmul_tflops_fp16: Option<f32>,
    prefill_moe_matmul_tflops_fp16: Option<f32>,
    unified_memory: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeMemoryEstimate {
    pub runtime_bytes: u64,
    pub kv_cache_bytes: u64,
    pub resident_weight_bytes: u64,
    pub scratch_bytes: u64,
    pub backend_overhead_bytes: u64,
}

pub fn estimate_kv_cache_bytes(model: &ModelProfile, config: &SelectionConfig) -> u64 {
    // Public KV estimate for the configured target context. See
    // `kv_cache_bytes_for_context` for the shape assumptions. This is exposed
    // because callers often want to show users how much of the memory budget is
    // model weights vs context.
    let context_tokens = target_context_tokens(model, config);
    kv_cache_bytes_for_context(model, config, context_tokens)
}

pub fn estimate_runtime_memory_bytes(model: &ModelProfile, config: &SelectionConfig) -> u64 {
    runtime_memory_estimate(model, config).runtime_bytes
}

pub fn score_model(
    hardware: &HardwareProfile,
    model: &ModelProfile,
    config: &SelectionConfig,
) -> ModelRecommendation {
    let mut recommendations = execution_budgets(hardware)
        .into_iter()
        .map(|budget| score_for_budget(model, config, &budget))
        .collect::<Vec<_>>();

    if recommendations.is_empty() {
        let budget = ExecutionBudget {
            backend: BackendKind::Unknown,
            accelerator_name: None,
            accelerator_kind: AcceleratorKind::Unknown,
            usable_memory_bytes: 0,
            memory_bandwidth_bytes_per_sec: None,
            decode_effective_bandwidth_bytes_per_sec: None,
            decode_fixed_overhead_ms: None,
            post_prefill_decode_overhead_ms: None,
            bandwidth_source: MeasurementSource::Unknown,
            benchmark_noise_pct: None,
            compute_tflops_fp16: None,
            prefill_matmul_tflops_fp16: None,
            prefill_moe_matmul_tflops_fp16: None,
            unified_memory: false,
        };
        return score_for_budget(model, config, &budget);
    }

    recommendations.sort_by(compare_execution_budget_recommendations);
    recommendations.remove(0)
}

pub fn rank_models(
    hardware: &HardwareProfile,
    models: &[ModelProfile],
    config: &SelectionConfig,
) -> Vec<ModelRecommendation> {
    let mut recommendations = models
        .iter()
        .map(|model| score_model(hardware, model, config))
        .collect::<Vec<_>>();
    recommendations.sort_by(compare_recommendations);
    recommendations
}

fn score_for_budget(
    model: &ModelProfile,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
) -> ModelRecommendation {
    let memory = runtime_memory_estimate(model, config);
    let active_decode_bytes = active_decode_bytes_per_token(model, config);
    let active_decode_flops = active_decode_flops_per_token(model);
    let estimated_decode_tps = decode_tokens_per_sec(
        active_decode_bytes,
        active_decode_flops,
        budget,
        config,
        model,
    );
    let estimated_decode_range =
        decode_tokens_per_sec_range(estimated_decode_tps, active_decode_bytes, model, budget);
    let estimated_prefill_tps = prefill_tokens_per_sec(model, config, budget, estimated_decode_tps);
    let first_token =
        first_token_estimate(estimated_prefill_tps, estimated_decode_tps, config, budget);
    let estimated_first_token_range = first_token_ms_range(first_token.total_ms, model, budget);
    let memory_limit = memory_limit_with_margin(budget.usable_memory_bytes, config.safety_margin);
    let mut warnings = Vec::new();
    let mut reasons = Vec::new();

    let (workload_score, workload_reject) =
        workload_score(model, config, &mut reasons, &mut warnings);
    let fit_status = fit_status(
        model,
        &memory,
        budget,
        memory_limit,
        workload_reject,
        &mut reasons,
        &mut warnings,
    );
    let memory_score = memory_score(memory.runtime_bytes, memory_limit);
    let context_score = context_score(model, config, &mut reasons, &mut warnings);
    let decode_score = decode_score(estimated_decode_tps, config);
    let prefill_score = prefill_score(model, config, budget);
    let total_score = total_score(
        config.weights,
        memory_score,
        context_score,
        decode_score,
        prefill_score,
        workload_score,
        fit_status,
    );

    if budget.memory_bandwidth_bytes_per_sec.is_none() {
        warnings
            .push("memory bandwidth is missing; decode score uses a conservative fallback".into());
    }
    if measured_gpu_budget(budget) && budget.decode_effective_bandwidth_bytes_per_sec.is_none() {
        warnings.push(
            "measured GPU profile is missing decode-shaped bandwidth; decode estimate falls back to raw bandwidth"
                .into(),
        );
    }
    if measured_gpu_budget(budget) && budget.decode_fixed_overhead_ms.is_none() {
        warnings.push(
            "measured GPU profile is missing fixed decode overhead; no measured fixed overhead is applied"
                .into(),
        );
    }
    if budget.unified_memory {
        reasons.push("using unified-memory budget for model weights, KV cache, and scratch".into());
    }
    reasons.push(format!(
        "runtime estimate includes {:.1} GiB scratch and {:.1} GiB backend overhead",
        gib(memory.scratch_bytes),
        gib(memory.backend_overhead_bytes)
    ));
    add_decode_estimate_reason(model, budget, config, &mut reasons);
    add_prefill_estimate_reason(first_token.total_ms, config, &mut reasons);
    add_architecture_warnings(model, &mut warnings);

    ModelRecommendation {
        source: model.source.clone(),
        selected_backend: budget.backend,
        selected_accelerator: budget.accelerator_name.clone(),
        architecture_class: model.architecture_class,
        estimate_confidence: estimate_confidence(model, budget),
        fit_status,
        total_score,
        memory_score,
        context_score,
        decode_score,
        prefill_score,
        workload_score,
        estimated_runtime_memory_bytes: memory.runtime_bytes,
        estimated_kv_cache_bytes: memory.kv_cache_bytes,
        estimated_active_decode_bytes_per_token: active_decode_bytes,
        estimated_decode_tokens_per_sec: estimated_decode_tps,
        estimated_decode_tokens_per_sec_range: estimated_decode_range,
        estimated_prefill_tokens_per_sec: estimated_prefill_tps,
        estimated_first_token_prefill_ms: first_token.prefill_ms,
        estimated_first_token_decode_ms: first_token.decode_ms,
        estimated_first_token_overhead_ms: first_token.overhead_ms,
        estimated_first_token_ms: first_token.total_ms,
        estimated_first_token_ms_range: estimated_first_token_range,
        split_candidate: split_candidate(model, &memory, budget, memory_limit, fit_status),
        capability_evidence: model.capability_evidence.clone(),
        reasons,
        warnings,
    }
}

fn execution_budgets(hardware: &HardwareProfile) -> Vec<ExecutionBudget> {
    let mut budgets = hardware
        .accelerators
        .iter()
        .map(|accelerator| accelerator_budget(hardware, accelerator))
        .collect::<Vec<_>>();
    if let Some(memory) = hardware.memory.available_system_bytes {
        budgets.push(ExecutionBudget {
            backend: BackendKind::Cpu,
            accelerator_name: Some("CPU".into()),
            accelerator_kind: AcceleratorKind::Cpu,
            usable_memory_bytes: memory,
            memory_bandwidth_bytes_per_sec: hardware.cpu.memory_bandwidth_bytes_per_sec,
            decode_effective_bandwidth_bytes_per_sec: None,
            decode_fixed_overhead_ms: None,
            bandwidth_source: hardware
                .cpu
                .memory_bandwidth_bytes_per_sec
                .map(|_| MeasurementSource::Measured)
                .unwrap_or(MeasurementSource::Unknown),
            benchmark_noise_pct: None,
            post_prefill_decode_overhead_ms: hardware.cpu.post_prefill_decode_overhead_ms,
            compute_tflops_fp16: hardware.cpu.compute_tflops_fp16,
            prefill_matmul_tflops_fp16: hardware.cpu.prefill_matmul_tflops_fp16,
            prefill_moe_matmul_tflops_fp16: hardware.cpu.prefill_moe_matmul_tflops_fp16,
            unified_memory: false,
        });
    }
    budgets
}

fn accelerator_budget(
    hardware: &HardwareProfile,
    accelerator: &AcceleratorProfile,
) -> ExecutionBudget {
    let usable_memory_bytes = if accelerator.unified_memory {
        accelerator
            .available_memory_bytes
            .or(hardware.memory.available_unified_bytes)
            .or(hardware.memory.available_system_bytes)
            .or(accelerator.total_memory_bytes)
            .or(hardware.memory.total_unified_bytes)
            .unwrap_or(0)
    } else {
        accelerator
            .available_memory_bytes
            .or(accelerator.total_memory_bytes)
            .unwrap_or(0)
    };

    ExecutionBudget {
        backend: accelerator.backend,
        accelerator_name: accelerator.name.clone().or_else(|| {
            (accelerator.kind != AcceleratorKind::Unknown)
                .then(|| format!("{:?}", accelerator.kind))
        }),
        accelerator_kind: accelerator.kind,
        usable_memory_bytes,
        memory_bandwidth_bytes_per_sec: accelerator.memory_bandwidth_bytes_per_sec,
        decode_effective_bandwidth_bytes_per_sec: accelerator
            .decode_effective_bandwidth_bytes_per_sec,
        decode_fixed_overhead_ms: accelerator.decode_fixed_overhead_ms,
        post_prefill_decode_overhead_ms: accelerator.post_prefill_decode_overhead_ms,
        bandwidth_source: accelerator.bandwidth_source,
        benchmark_noise_pct: accelerator.benchmark_noise_pct,
        compute_tflops_fp16: accelerator.compute_tflops_fp16,
        prefill_matmul_tflops_fp16: accelerator.prefill_matmul_tflops_fp16,
        prefill_moe_matmul_tflops_fp16: accelerator.prefill_moe_matmul_tflops_fp16,
        unified_memory: accelerator.unified_memory,
    }
}

fn runtime_memory_estimate(
    model: &ModelProfile,
    config: &SelectionConfig,
) -> RuntimeMemoryEstimate {
    // Runtime memory is more than the GGUF file. The resident weights are the
    // largest term, but context-heavy workloads can add a large KV cache and
    // every backend needs scratch/activation space. We keep this estimate
    // explainable instead of trying to reproduce llama.cpp allocation internals:
    //
    //   resident weights + KV cache + scratch + backend overhead
    //
    // The selector applies a configurable safety margin to this value before it
    // ranks a model as a local fit. That margin absorbs allocator
    // fragmentation, OS pressure, backend work buffers, and imperfect metadata.
    let resident_weight_bytes = resident_weight_bytes(model);
    let kv_cache_bytes = estimate_kv_cache_bytes(model, config);
    let scratch_bytes = scratch_bytes(model, resident_weight_bytes);
    let backend_overhead_bytes = 256 * MIB + resident_weight_bytes / 100;
    let runtime_bytes = resident_weight_bytes
        .saturating_add(kv_cache_bytes)
        .saturating_add(scratch_bytes)
        .saturating_add(backend_overhead_bytes);
    RuntimeMemoryEstimate {
        runtime_bytes,
        kv_cache_bytes,
        resident_weight_bytes,
        scratch_bytes,
        backend_overhead_bytes,
    }
}

fn resident_weight_bytes(model: &ModelProfile) -> u64 {
    model
        .tensor_bytes
        .or_else(|| {
            Some(
                model
                    .base_resident_bytes?
                    .saturating_add(model.expert_tensor_bytes.unwrap_or(0)),
            )
        })
        .unwrap_or(model.file_size_bytes)
}

fn scratch_bytes(model: &ModelProfile, resident_weight_bytes: u64) -> u64 {
    let minimum = match model.architecture_class {
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            256 * MIB
        }
        _ => 512 * MIB,
    };
    minimum.max(resident_weight_bytes / 20)
}

fn kv_cache_bytes_for_context(
    model: &ModelProfile,
    config: &SelectionConfig,
    context_tokens: u32,
) -> u64 {
    // KV cache is the memory term that scales with requested context rather
    // than file size. The exact layout is backend/model dependent, but GGUF
    // normally gives enough architecture metadata to estimate the dominant
    // shape:
    //
    //   (K row + V row) * layer_count * context_tokens
    //
    // If grouped-query attention metadata is present, `kv_width` uses
    // kv_heads * key/value_length. Otherwise it falls back to hidden_size, which
    // is conservative for arbitrary GGUFs. Cache quantization is modeled with
    // ggml block row sizes so Q8/Q4 KV settings reduce memory in roughly the
    // same proportions llama.cpp will use.
    if !uses_transformer_kv_cache(model.architecture_class) {
        return 0;
    }
    let Some(layers) = model.layer_count else {
        return fallback_kv_cache_bytes(model, config, context_tokens);
    };
    let k_width = kv_width(model, model.key_length);
    let v_width = kv_width(model, model.value_length);
    let k_bytes = row_size(config.kv_cache_type.k, k_width).saturating_mul(u64::from(layers));
    let v_bytes = row_size(config.kv_cache_type.v, v_width).saturating_mul(u64::from(layers));
    k_bytes
        .saturating_add(v_bytes)
        .saturating_mul(u64::from(context_tokens))
}

fn fallback_kv_cache_bytes(
    model: &ModelProfile,
    config: &SelectionConfig,
    context_tokens: u32,
) -> u64 {
    let Some(hidden) = model.hidden_size else {
        return 0;
    };
    let layers = model.layer_count.unwrap_or(1);
    let k = row_size(config.kv_cache_type.k, u64::from(hidden));
    let v = row_size(config.kv_cache_type.v, u64::from(hidden));
    k.saturating_add(v)
        .saturating_mul(u64::from(layers))
        .saturating_mul(u64::from(context_tokens))
}

fn kv_width(model: &ModelProfile, vector_length: Option<u32>) -> u64 {
    match (model.kv_heads, vector_length) {
        (Some(kv_heads), Some(length)) => u64::from(kv_heads).saturating_mul(u64::from(length)),
        _ => u64::from(model.hidden_size.unwrap_or_default()),
    }
}

fn row_size(kind: KvCacheKind, elements: u64) -> u64 {
    let (block_elements, block_bytes) = match kind {
        KvCacheKind::F16 => (1, 2),
        KvCacheKind::Q8_0 => (32, 34),
        KvCacheKind::Q4_0 => (32, 18),
    };
    elements.div_ceil(block_elements) * block_bytes
}

fn active_decode_bytes_per_token(model: &ModelProfile, config: &SelectionConfig) -> Option<u64> {
    // Decode is usually the user-visible bottleneck for local chat/agent loops:
    // it runs once per generated token, and on common GGUF backends it is often
    // closer to memory-bandwidth-bound than compute-bound. The core predictor is
    // therefore "how many bytes must be touched for one token?"
    //
    // For dense models that is mostly resident transformer weights. For MoE
    // models it is base/resident weights plus only the routed experts, because a
    // token does not execute every expert. When tensor group sizes are available
    // from GGUF inspection we use them directly; otherwise we fall back to the
    // coarser resident/active byte estimates.
    //
    // KV reads and a small activation proxy are included so that context length
    // and layer/hidden width can still affect decode ranking. This is not a full
    // llama.cpp execution trace. It is an explainable first-pass "active byte
    // pressure" estimate that can be produced from GGUF metadata alone.
    let active_weights = match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => active_moe_decode_weight_traffic(model),
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            return None;
        }
        _ => active_dense_decode_weight_traffic(model),
    };
    let context = config
        .workload
        .interaction
        .expected_prompt_tokens
        .unwrap_or_else(|| target_context_tokens(model, config) / 2);
    let kv_read_bytes = kv_cache_bytes_for_context(model, config, context)
        .saturating_mul((config.kv_read_scale * 1000.0).round() as u64)
        / 1000;
    let activation_overhead = activation_overhead_bytes(model);
    Some(
        active_weights
            .saturating_add(kv_read_bytes)
            .saturating_add(activation_overhead),
    )
}

fn active_dense_decode_weight_traffic(model: &ModelProfile) -> u64 {
    if let Some(bytes) = dense_matmul_traffic_bytes(model) {
        return bytes;
    }
    let groups = model.tensor_group_bytes;
    let storage_bytes = if tensor_groups_available(groups) {
        groups
            .attention_bytes
            .saturating_add(groups.feed_forward_bytes)
            .saturating_add(groups.output_bytes)
            .saturating_add(groups.normalization_bytes)
            .saturating_add(groups.other_bytes)
    } else {
        resident_weight_bytes(model)
    };
    decode_weight_traffic_bytes(model, storage_bytes)
}

fn active_moe_decode_weight_traffic(model: &ModelProfile) -> u64 {
    if let Some(bytes) = moe_matmul_traffic_bytes(model) {
        return bytes;
    }
    let groups = model.tensor_group_bytes;
    let storage_bytes = if tensor_groups_available(groups) {
        let active_expert_bytes = active_expert_bytes(
            groups.expert_feed_forward_bytes,
            model.expert_count,
            model.expert_used_count,
        );
        groups
            .attention_bytes
            .saturating_add(groups.feed_forward_bytes)
            .saturating_add(active_expert_bytes)
            .saturating_add(groups.output_bytes)
            .saturating_add(groups.normalization_bytes)
            .saturating_add(groups.other_bytes)
    } else {
        active_moe_weight_bytes(model)
    };
    decode_weight_traffic_bytes(model, storage_bytes)
}

fn decode_weight_traffic_bytes(model: &ModelProfile, storage_bytes: u64) -> u64 {
    // GGUF tensor bytes are a resident-storage fact. For a decode-token
    // GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID path, those bytes are the only
    // source-grounded value we currently have for arbitrary GGUFs unless the
    // profile also carries per-tensor type/shape data and a llama.cpp matmul
    // kernel cost model.
    //
    // Do not apply validation-derived quantization multipliers here. Q8_0,
    // Q4_K, IQ, and MoE tensors do take different llama.cpp kernels, but the
    // right representation is an explicit matmul model derived from GGML tensor
    // block layout and backend kernel traits, not a storage-to-traffic scale
    // tuned until one local validation set looks good. Until that model exists,
    // decode prediction should remain conservative and the validator should
    // report misses honestly.
    let _ = model;
    storage_bytes
}

fn dense_matmul_traffic_bytes(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped = matmul
        .attention
        .kernel_traffic_bytes()
        .saturating_add(matmul.feed_forward.kernel_traffic_bytes())
        .saturating_add(matmul.output.kernel_traffic_bytes());
    if grouped > 0 {
        Some(grouped)
    } else {
        let base = tensor_type_kernel_traffic_bytes(matmul.base_type_bytes);
        if base > 0 {
            Some(base)
        } else {
            (matmul.base_bytes > 0).then_some(matmul.base_bytes)
        }
    }
}

fn moe_matmul_traffic_bytes(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped_base = matmul
        .attention
        .kernel_traffic_bytes()
        .saturating_add(matmul.feed_forward.kernel_traffic_bytes())
        .saturating_add(matmul.output.kernel_traffic_bytes());
    let grouped_expert = matmul.expert_feed_forward.kernel_traffic_bytes();
    if grouped_base > 0 || grouped_expert > 0 {
        return Some(grouped_base.saturating_add(active_expert_bytes(
            grouped_expert,
            model.expert_count,
            model.expert_used_count,
        )));
    }
    if matmul.base_bytes > 0 || matmul.expert_bytes > 0 {
        let base = tensor_type_kernel_traffic_bytes(matmul.base_type_bytes);
        let base = if base > 0 { base } else { matmul.base_bytes };
        let expert = tensor_type_kernel_traffic_bytes(matmul.expert_type_bytes);
        let expert = if expert > 0 {
            expert
        } else {
            matmul.expert_bytes
        };
        return Some(base.saturating_add(active_expert_bytes(
            expert,
            model.expert_count,
            model.expert_used_count,
        )));
    }
    None
}

trait MatmulKernelTraffic {
    fn kernel_traffic_bytes(&self) -> u64;
}

impl MatmulKernelTraffic for TensorMatmulGroupProfile {
    fn kernel_traffic_bytes(&self) -> u64 {
        let traffic = tensor_type_kernel_traffic_bytes(self.type_bytes);
        if traffic > 0 { traffic } else { self.bytes }
    }
}

fn tensor_type_kernel_traffic_bytes(bytes: TensorTypeBytes) -> u64 {
    // This is not a filename/model-family boost. It is the first source-derived
    // bridge between GGUF resident storage bytes and the decode loop that
    // llama.cpp actually executes.
    //
    // `scan_gguf_tensor_byte_profile()` tells us, per matmul tensor group, how
    // many bytes are stored as each GGML tensor type. llama.cpp then dispatches
    // those tensors into different GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID
    // kernels. The stored byte count is therefore a real memory-capacity fact,
    // but it is not always the best decode-time cost unit:
    //
    // - Q8_0 stores about 34 bytes per 32 weights (`block_q8_0`). Its Metal
    //   mat-vec kernel reads that block directly (`qs` plus `d`), and CUDA's
    //   MMVQ/MMQ path uses q8-specific `vec_dot_q8_0_q8_1` helpers. The source
    //   does not justify counting less than the resident Q8_0 block bytes as
    //   decode traffic, so Q8_0 is charged at its stored bytes.
    // - Q4_K stores far fewer bytes, but its `block_q4_K` super-block carries
    //   packed nibbles, scales/mins, and a q4x4 dequant path. The physical
    //   bytes are lower, while the useful bandwidth-equivalent work is not
    //   proportionally lower.
    //
    // The multipliers below produce "bandwidth-equivalent traffic": bytes of
    // resident matmul storage adjusted by GGML tensor-type kernel structure.
    // They deliberately apply to every GGUF with the same tensor type mix. They
    // are not backend labels, not architecture labels, and not model names.
    //
    // Source anchors in the pinned llama.cpp tree:
    // - ggml.c defines block size/type size for Q8_0 and K-quants.
    // - ggml-metal.metal instantiates separate matmul kernels for Q8_0 and
    //   Q4_K/Q5_K/Q6_K instead of a single generic bytes-only path.
    // - ggml-metal-ops.cpp dispatches Q8_0 with the f32/f16/bf16 row-grouping
    //   path for generic mat-vec, not with the K-quant grouping branch.
    // - ggml-cuda uses q8-specific MMVQ/MMQ vec-dot helpers and q8-specific
    //   MUL_MAT_ID templates.
    //
    // Earlier versions discounted Q8_0 storage bytes because Q8 has a simpler
    // block layout than K-quants. Broader validation falsified that as a
    // portable traffic model: Q8_0 was over-predicted on both Metal and CUDA.
    // The source-grounded rule is now simpler and stricter: count stored bytes
    // for Q4_K/Q5_K/Q6_K/Q8_0 and let measured hardware bandwidth, graph shape,
    // and compute floors decide the final rate.
    scaled_type_bytes(bytes.f32_bytes, 1.00)
        .saturating_add(scaled_type_bytes(bytes.f16_bytes, 1.00))
        .saturating_add(scaled_type_bytes(bytes.bf16_bytes, 1.00))
        .saturating_add(scaled_type_bytes(bytes.q4_0_bytes, 1.18))
        .saturating_add(scaled_type_bytes(bytes.q4_k_bytes, 1.00))
        .saturating_add(scaled_type_bytes(bytes.q5_k_bytes, 1.00))
        .saturating_add(scaled_type_bytes(bytes.q6_k_bytes, 1.00))
        .saturating_add(scaled_type_bytes(bytes.q8_0_bytes, 1.00))
        .saturating_add(scaled_type_bytes(bytes.iq_bytes, 1.18))
        .saturating_add(scaled_type_bytes(bytes.other_quantized_bytes, 1.05))
        .saturating_add(bytes.unknown_bytes)
}

fn scaled_type_bytes(bytes: u64, factor: f32) -> u64 {
    ((bytes as f64) * f64::from(factor)).round() as u64
}

fn active_decode_flops_per_token(model: &ModelProfile) -> Option<u64> {
    match model.architecture_class {
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::Unknown => {
            dense_matmul_flops(model)
        }
        ModelArchitectureClass::SparseMoeTransformer => moe_matmul_flops(model),
        _ => None,
    }
}

fn dense_matmul_flops(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped = matmul
        .attention
        .flops_per_token
        .saturating_add(matmul.feed_forward.flops_per_token)
        .saturating_add(matmul.output.flops_per_token);
    if grouped > 0 {
        Some(grouped)
    } else {
        (matmul.base_flops_per_token > 0).then_some(matmul.base_flops_per_token)
    }
}

fn moe_matmul_flops(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped_base = matmul
        .attention
        .flops_per_token
        .saturating_add(matmul.feed_forward.flops_per_token)
        .saturating_add(matmul.output.flops_per_token);
    let grouped_expert = matmul.expert_feed_forward.flops_per_token;
    if grouped_base > 0 || grouped_expert > 0 {
        return Some(grouped_base.saturating_add(active_expert_bytes(
            grouped_expert,
            model.expert_count,
            model.expert_used_count,
        )));
    }
    if matmul.base_flops_per_token > 0 || matmul.expert_flops_per_token > 0 {
        return Some(
            matmul
                .base_flops_per_token
                .saturating_add(active_expert_bytes(
                    matmul.expert_flops_per_token,
                    model.expert_count,
                    model.expert_used_count,
                )),
        );
    }
    None
}

fn tensor_groups_available(groups: TensorGroupBytes) -> bool {
    groups.attention_bytes > 0
        || groups.feed_forward_bytes > 0
        || groups.expert_feed_forward_bytes > 0
        || groups.output_bytes > 0
}

fn active_moe_weight_bytes(model: &ModelProfile) -> u64 {
    let base = model.base_resident_bytes.unwrap_or(0);
    let expert = model.expert_tensor_bytes.unwrap_or(0);
    base.saturating_add(active_expert_bytes(
        expert,
        model.expert_count,
        model.expert_used_count,
    ))
}

fn active_expert_bytes(
    expert_bytes: u64,
    expert_count: Option<u32>,
    expert_used_count: Option<u32>,
) -> u64 {
    let Some(expert_count) = expert_count.filter(|count| *count > 0) else {
        return expert_bytes;
    };
    let active = expert_used_count.unwrap_or(expert_count).min(expert_count);
    expert_bytes.saturating_mul(u64::from(active)) / u64::from(expert_count)
}

fn activation_overhead_bytes(model: &ModelProfile) -> u64 {
    let layer_width = u64::from(model.layer_count.unwrap_or(1))
        .saturating_mul(u64::from(model.hidden_size.unwrap_or_default()));
    layer_width.saturating_mul(16)
}

fn decode_tokens_per_sec(
    active_decode_bytes: Option<u64>,
    active_decode_flops: Option<u64>,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    model: &ModelProfile,
) -> Option<f32> {
    // Decode estimate shape:
    //
    //   tok/s = 1000 / (active_bytes / effective_bandwidth + overhead_ms)
    //
    // `active_bytes / effective_bandwidth` gives the slope for models that are
    // mostly streaming weights and KV state. The overhead terms then account for
    // costs that do not scale linearly with model bytes: backend/runtime fixed
    // cost, MoE routing/dispatch, and small-model inefficiency. Keeping those
    // pieces separate is important for ranking: a 70B model should mostly be
    // limited by memory traffic, while a 135M model can be dominated by launch
    // and scheduling costs even though it barely touches memory.
    //
    // This function must not use filename/catalog reputation. It only consumes
    // metadata, measured hardware facts, and explicit config knobs so validation
    // can run against any GGUF and explain misses.
    let bytes = active_decode_bytes?;
    if bytes == 0 {
        return None;
    }
    let base_bandwidth = decode_base_bandwidth_bytes_per_sec(budget, config);
    let architecture_factor = match model.architecture_class {
        ModelArchitectureClass::Unknown => 0.75,
        ModelArchitectureClass::RecurrentOrStateSpace => 0.85,
        _ => 1.0,
    };
    let quantization_factor = quantization_efficiency_factor(model.quantization.as_deref());
    let shape_factor = decode_shape_bandwidth_factor(model, bytes);
    let effective_bandwidth =
        base_bandwidth as f32 * architecture_factor * quantization_factor * shape_factor;
    let bandwidth_ms = bytes as f32 / effective_bandwidth.max(1.0) * 1000.0;
    let compute_ms = decode_compute_ms(active_decode_flops, budget).unwrap_or(0.0);
    let overhead_ms = fixed_decode_overhead_ms(budget, config)
        + measured_decode_graph_overhead_ms(model, budget)
        + architecture_decode_overhead_ms(model, budget, config);
    Some(1000.0 / (bandwidth_ms.max(compute_ms) + overhead_ms).max(0.001))
}

fn decode_compute_ms(active_decode_flops: Option<u64>, budget: &ExecutionBudget) -> Option<f32> {
    let flops = active_decode_flops? as f32;
    let tflops = budget.compute_tflops_fp16.filter(|value| *value > 0.0)?;
    Some(flops / (tflops * 1_000_000_000_000.0) * 1000.0)
}

fn decode_shape_bandwidth_factor(model: &ModelProfile, active_decode_bytes: u64) -> f32 {
    // Wide dense models can achieve slightly better effective streaming
    // bandwidth than tiny/narrow models because there is enough per-token work
    // to keep the backend occupied. This is intentionally a small factor and
    // only applies when both the hidden width and active byte footprint look
    // like a normal medium/large dense transformer. It should improve the slope
    // for 8B-ish dense models without letting architecture labels dominate the
    // selector.
    if !matches!(
        model.architecture_class,
        ModelArchitectureClass::DenseTransformer
    ) {
        return 1.0;
    }
    let Some(hidden) = model.hidden_size else {
        return 1.0;
    };
    let active_gib = active_decode_bytes as f32 / GIB as f32;
    let width = hidden as f32;
    if active_gib < 4.0 || width < 3072.0 {
        return 1.0;
    }
    let width_term = ((width - 3072.0) / 1024.0).clamp(0.0, 1.0);
    let active_term = ((active_gib - 4.0) / 1.0).clamp(0.0, 1.0);
    let ffn_term = ffn_expansion_occupancy_term(model);
    1.0 + 0.12 * width_term.max(active_term * 0.70) + ffn_term
}

fn ffn_expansion_occupancy_term(model: &ModelProfile) -> f32 {
    // Large FFN matrices mean the decode graph spends more of its time in a few
    // large feed-forward matmuls rather than in many smaller surrounding
    // operations. That tends to improve occupancy for active-byte-heavy dense
    // models because each FFN kernel has more useful multiply-add work to
    // amortize dispatch, dequant, and memory scheduling.
    //
    // Prefer the GGUF tensor shape summary over metadata widths. GGUF stores
    // matmul tensors in GGML `ne[]` order, so model-artifact records the
    // element-weighted input/output widths for the actual FFN matrices that
    // llama.cpp will feed to GGML_OP_MUL_MAT. That handles different-sized
    // matmuls directly: a model with a very expanded FFN, a compact FFN, or
    // mixed grouped tensors receives the term implied by its real tensor shapes.
    if let Some(term) = ffn_shape_occupancy_term(model.tensor_matmul.feed_forward.shape) {
        return term;
    }
    let Some(hidden) = model.hidden_size.filter(|hidden| *hidden > 0) else {
        return 0.0;
    };
    let Some(ffn) = model.ffn_size else {
        return 0.0;
    };
    let ratio = ffn as f32 / hidden as f32;
    if ratio <= 4.0 {
        return 0.0;
    }
    ((ratio - 4.0) / 1.5 * 0.14).clamp(0.0, 0.14)
}

fn ffn_shape_occupancy_term(shape: MatmulShapeProfile) -> Option<f32> {
    let expansion = ffn_shape_expansion_ratio(shape)?;
    if expansion <= 4.0 {
        return Some(0.0);
    }
    Some(((expansion - 4.0) / 1.5 * 0.14).clamp(0.0, 0.14))
}

fn ffn_shape_expansion_ratio(shape: MatmulShapeProfile) -> Option<f32> {
    if shape.logical_matrix_count == 0
        || shape.weighted_avg_input_width == 0
        || shape.weighted_avg_output_width == 0
    {
        return None;
    }
    // Use the range across the FFN group rather than only the aggregate average:
    // `up/gate` and `down` matrices are transposes in shape terms, so averaging
    // input/output widths together would hide the expansion that determines how
    // large each individual FFN matmul is.
    let narrow = shape
        .min_input_width
        .min(shape.min_output_width)
        .min(shape.weighted_avg_input_width)
        .min(shape.weighted_avg_output_width) as f32;
    let wide = shape
        .max_input_width
        .max(shape.max_output_width)
        .max(shape.weighted_avg_input_width)
        .max(shape.weighted_avg_output_width) as f32;
    if narrow <= 0.0 {
        return None;
    }
    Some(wide / narrow)
}

fn measured_decode_graph_overhead_ms(model: &ModelProfile, budget: &ExecutionBudget) -> f32 {
    // The GPU benchmark reports two independent hardware facts:
    //
    // - `decode_effective_bandwidth`: a decode-shaped stream that already pays
    //   for eight sequential kernel/command-buffer submissions while moving a
    //   large weight-like byte range.
    // - `decode_fixed_overhead_ms`: the measured cost of one empty backend
    //   submission on this machine.
    //
    // llama.cpp decode is not one monolithic kernel. `llama-graph.cpp` builds a
    // repeated per-layer GGML graph and the backends dispatch `GGML_OP_MUL_MAT`
    // / `GGML_OP_MUL_MAT_ID` plus attention, normalization, elementwise, and
    // copy/view kernels around those matmuls. The GGUF metadata tells us the
    // number of transformer blocks, and `model-artifact` records the logical
    // matmul count from actual tensor shapes. Those are source-derived shape
    // facts, not model names.
    //
    // To avoid double-counting the benchmark's own decode loop, only add the
    // measured fixed overhead for graph work beyond the eight dispatches already
    // present in the decode-shaped bandwidth measurement. This is deliberately
    // backend-neutral: CUDA may report a very small fixed submission cost,
    // Metal may report a larger one, and model-fit simply consumes the measured
    // hardware profile. If the benchmark did not measure fixed overhead, this
    // term is zero and the recommendation carries the existing warning.
    if !measured_gpu_budget(budget) {
        return 0.0;
    }
    let Some(fixed_ms) = budget.decode_fixed_overhead_ms.filter(|value| *value > 0.0) else {
        return 0.0;
    };
    let Some(layers) = model.layer_count.filter(|layers| *layers > 0) else {
        return 0.0;
    };
    let logical_matmuls = dense_logical_matmul_count(model);
    if logical_matmuls == 0 {
        return 0.0;
    }
    const GPU_BENCH_DECODE_DISPATCHES: f32 = 8.0;
    let graph_dispatch_groups = if fixed_ms < 0.01 {
        // On very low-latency submission paths, an empty dispatch no longer
        // captures the visible per-token graph work. The source trace shows a
        // decode layer is not just matmuls: llama.cpp builds matmul nodes plus
        // normalization, RoPE, attention, activation, copy/view, and elementwise
        // nodes around them. The graph-node multiplier is therefore based on
        // tensor type and logical matmul count, not backend name or model
        // identity. Q8_0 uses a simpler block/dequant path than K-quants, so it
        // gets fewer surrounding work units.
        logical_matmuls as f32 * low_latency_graph_node_multiplier(model)
    } else {
        layers as f32 + expanded_ffn_sequential_stage_groups(model, layers, logical_matmuls)
    };
    let extra_dispatch_groups = (graph_dispatch_groups - GPU_BENCH_DECODE_DISPATCHES).max(0.0);
    extra_dispatch_groups * fixed_ms
}

fn expanded_ffn_sequential_stage_groups(
    model: &ModelProfile,
    layers: u32,
    logical_matmuls: u64,
) -> f32 {
    // For measured-GPU paths with non-trivial fixed submission cost, the base
    // graph term charges roughly one backend submission per transformer layer.
    // That is a useful lower bound for a backend that can encode many GGML
    // nodes into one command buffer, but it misses a source-level shape:
    // expanded FFNs are not one opaque operation. llama.cpp builds separate
    // attention projection/output matmuls, FFN gate/up/down matmuls,
    // activation/multiply nodes, norms, RoPE, views/copies, and attention nodes
    // around them. The FFN path is especially sequential because down
    // projection depends on the gate/up activation.
    //
    // GGUF gives us the portable inputs for this without looking at backend
    // names, model IDs, or validation throughput: actual logical matmul count,
    // layer count, hidden width, and FFN expansion from tensor dimensions. This
    // returns additional sequential graph groups, later multiplied by the
    // measured fixed dispatch cost for the selected hardware profile.
    let Some(hidden) = model.hidden_size.filter(|hidden| *hidden > 0) else {
        return 0.0;
    };
    let Some(ffn_expansion) = ffn_expansion_ratio(model) else {
        return 0.0;
    };
    if layers == 0 || logical_matmuls == 0 {
        return 0.0;
    }

    let sequential_stage_pressure =
        ((logical_matmuls as f32).sqrt() - (layers as f32).sqrt()).max(0.0);
    let width_visibility = ((hidden as f32 - 576.0) / (1024.0 - 576.0)).clamp(0.0, 1.0);
    let expansion_visibility = ((ffn_expansion - 2.0) / 1.0).clamp(0.0, 1.0);
    sequential_stage_pressure * width_visibility * expansion_visibility
}

fn ffn_expansion_ratio(model: &ModelProfile) -> Option<f32> {
    if let Some(ratio) = ffn_shape_expansion_ratio(model.tensor_matmul.feed_forward.shape) {
        return Some(ratio);
    }
    let hidden = model.hidden_size.filter(|hidden| *hidden > 0)? as f32;
    let ffn = model.ffn_size? as f32;
    Some(ffn / hidden)
}

fn low_latency_graph_node_multiplier(model: &ModelProfile) -> f32 {
    let types = aggregate_matmul_type_bytes(&model.tensor_matmul);
    let q8 = types.q8_0_bytes;
    let quantized = types
        .q4_0_bytes
        .saturating_add(types.q4_k_bytes)
        .saturating_add(types.q5_k_bytes)
        .saturating_add(types.q6_k_bytes)
        .saturating_add(types.q8_0_bytes)
        .saturating_add(types.iq_bytes)
        .saturating_add(types.other_quantized_bytes);
    if quantized > 0 && q8.saturating_mul(2) >= quantized {
        3.0
    } else {
        4.0
    }
}

fn aggregate_matmul_type_bytes(matmul: &crate::TensorMatmulProfile) -> TensorTypeBytes {
    add_type_bytes(
        add_type_bytes(
            add_type_bytes(matmul.attention.type_bytes, matmul.feed_forward.type_bytes),
            matmul.output.type_bytes,
        ),
        matmul.expert_feed_forward.type_bytes,
    )
}

fn add_type_bytes(left: TensorTypeBytes, right: TensorTypeBytes) -> TensorTypeBytes {
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

fn raw_memory_bandwidth_bytes_per_sec(budget: &ExecutionBudget) -> u64 {
    budget
        .memory_bandwidth_bytes_per_sec
        .unwrap_or(match budget.backend {
            BackendKind::Cpu => 80_000_000_000,
            _ => 200_000_000_000,
        })
}

fn decode_base_bandwidth_bytes_per_sec(budget: &ExecutionBudget, config: &SelectionConfig) -> u64 {
    if measured_gpu_budget(budget) {
        return budget
            .decode_effective_bandwidth_bytes_per_sec
            .or(budget.memory_bandwidth_bytes_per_sec)
            .unwrap_or(50_000_000_000);
    }
    (raw_memory_bandwidth_bytes_per_sec(budget) as f32
        * fallback_backend_efficiency(budget.backend, config)) as u64
}

fn prefill_tokens_per_sec(
    model: &ModelProfile,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
    decode_tokens_per_sec: Option<f32>,
) -> Option<f32> {
    if !uses_transformer_kv_cache(model.architecture_class) {
        return None;
    }
    let prompt_tokens = config.workload.interaction.expected_prompt_tokens?;
    if prompt_tokens == 0 {
        return None;
    }
    let roofline = prefill_roofline_tokens_per_sec(model, config, budget, prompt_tokens);
    let fallback = legacy_prefill_tokens_per_sec(model, decode_tokens_per_sec);
    if model.architecture_class == ModelArchitectureClass::SparseMoeTransformer {
        return match (roofline, fallback) {
            (Some(roofline), Some(fallback)) => Some(roofline.min(fallback)),
            (Some(roofline), None) => Some(roofline),
            (None, fallback) => fallback,
        };
    }
    roofline.or(fallback)
}

fn prefill_roofline_tokens_per_sec(
    model: &ModelProfile,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
    prompt_tokens: u32,
) -> Option<f32> {
    // llama.cpp builds a different graph for prompt processing than for
    // one-token decode. `llama-context.cpp` splits the prompt into ubatches
    // (`n_ubatch` defaults to 512 in `llama_context_default_params`), and
    // the backend receives batched `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID`
    // work instead of a long stream of single-token matvecs. That gives prefill
    // a compute-shaped roofline:
    //
    //   total_ms = max(prompt_matmul_flops / measured_compute,
    //                  ubatches * active_weight_bytes / measured_bandwidth)
    //              + ubatches * graph_overhead
    //
    // This is intentionally not a CUDA/Metal/ROCm branch. The scorer consumes
    // only the GGUF-derived matmul FLOPs/bytes and the measured hardware facts
    // that `mesh-llm gpus benchmark` already places in `HardwareProfile`.
    //
    // Very narrow models are the awkward corner. Their prompt graph is often
    // dominated by non-matmul kernels, scheduling, logits/pooling work, and
    // launch latency that our current generic hardware benchmark does not
    // measure directly. For those, keep the older decode-correlated fallback
    // until we add a measured prefill-shaped hardware probe.
    if !prefill_roofline_has_enough_matmul_shape(model) {
        return None;
    }
    let flops_per_token = active_decode_flops_per_token(model)? as f32;
    let active_weight_bytes = active_prefill_pressure_bytes(model)? as f32;
    let ubatches = prefill_ubatch_count(prompt_tokens) as f32;
    let compute_ms = prefill_compute_ms(flops_per_token, prompt_tokens, model, budget)?;
    let bandwidth_ms = prefill_weight_stream_ms(active_weight_bytes, ubatches, budget)?;
    let overhead_ms = prefill_graph_overhead_ms(model, budget, config, ubatches);
    let total_ms = compute_ms.max(bandwidth_ms) + overhead_ms;
    (total_ms > 0.0).then_some(prompt_tokens as f32 / total_ms * 1000.0)
}

fn prefill_roofline_has_enough_matmul_shape(model: &ModelProfile) -> bool {
    if model.architecture_class == ModelArchitectureClass::SparseMoeTransformer {
        return true;
    }
    let Some(hidden) = model.hidden_size else {
        return false;
    };
    hidden >= 2048
}

fn prefill_ubatch_count(prompt_tokens: u32) -> u32 {
    prompt_tokens.div_ceil(LLAMA_DEFAULT_UBATCH_TOKENS).max(1)
}

fn prefill_compute_ms(
    flops_per_token: f32,
    prompt_tokens: u32,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<f32> {
    let tflops = prefill_compute_tflops(model, budget)?;
    let total_flops = flops_per_token * prompt_tokens as f32;
    Some(total_flops / (tflops * 1_000_000_000_000.0) * 1000.0)
}

fn prefill_compute_tflops(model: &ModelProfile, budget: &ExecutionBudget) -> Option<f32> {
    // Dense and sparse-MoE prefill are intentionally separate measured hardware
    // facts. Dense prompt processing maps to batched GEMM. Sparse MoE routes
    // through expert selection and `GGML_OP_MUL_MAT_ID`, so it only uses the
    // MoE-shaped probe when that probe is present. If the MoE probe is absent,
    // return `None` so the caller falls back to the older MoE-aware estimate
    // instead of overpredicting MoE with the dense matmul probe.
    let measured = match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => budget.prefill_moe_matmul_tflops_fp16,
        _ => budget
            .prefill_matmul_tflops_fp16
            .or(budget.compute_tflops_fp16),
    };
    measured.filter(|value| *value > 0.0)
}

fn prefill_weight_stream_ms(
    active_weight_bytes: f32,
    ubatches: f32,
    budget: &ExecutionBudget,
) -> Option<f32> {
    let bandwidth = raw_memory_bandwidth_bytes_per_sec(budget) as f32;
    (bandwidth > 0.0).then_some(active_weight_bytes * ubatches / bandwidth * 1000.0)
}

fn prefill_graph_overhead_ms(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    ubatches: f32,
) -> f32 {
    ubatches
        * (measured_decode_graph_overhead_ms(model, budget)
            + architecture_decode_overhead_ms(model, budget, config))
}

fn legacy_prefill_tokens_per_sec(
    model: &ModelProfile,
    decode_tokens_per_sec: Option<f32>,
) -> Option<f32> {
    // This fallback is still useful for tiny and narrow models where the
    // matmul roofline has the wrong dominant term. It is deliberately isolated
    // from the main prefill path so future work can replace it with a measured
    // prefill-shaped hardware probe instead of spreading more special cases
    // through the scorer.
    let decode_tokens_per_sec = decode_tokens_per_sec?;
    let parallelism = prefill_decode_parallelism_factor(model)?;
    let prefill_tokens_per_sec = decode_tokens_per_sec * parallelism;
    Some(prefill_tokens_per_sec.max(0.0))
}

fn prefill_decode_parallelism_factor(model: &ModelProfile) -> Option<f32> {
    // This factor estimates how much more parallelism prefill gets compared to
    // decode. Smaller hidden sizes and smaller active-byte footprints generally
    // let prefill run many prompt tokens in parallel, so their factor is higher.
    // Large active models get less of a boost because weight traffic and memory
    // pressure still dominate.
    //
    // Q8 gets a higher prefill factor for non-tiny models because validation
    // showed that its decode path can be relatively slower than its batched
    // prefill path. This is a metadata-level quantization adjustment, not a
    // model identity boost.
    let hidden = model.hidden_size.filter(|hidden| *hidden > 0)? as f32;
    let active_gib = active_prefill_pressure_bytes(model)? as f32 / GIB as f32;
    let width_ratio = (1024.0 / hidden).min(1.0);
    let width_term = 26.0 * width_ratio.powf(1.35);
    let active_pressure = 1.0 + (active_gib - 1.0).max(0.0) * 0.22;
    let mut factor = 9.0 + width_term / active_pressure;

    if matches!(
        model.architecture_class,
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::Unknown
    ) && (1536.0..=2304.0).contains(&hidden)
    {
        factor += 3.0;
    }
    if (800.0..=1536.0).contains(&hidden) {
        factor += 3.0;
    }
    if hidden < 800.0 {
        factor *= 1.80;
    }
    factor *= 1.12;
    if quantization_is_q8(model.quantization.as_deref()) && hidden >= 800.0 {
        factor *= 1.70;
    }
    Some(factor.clamp(8.0, 72.0))
}

fn active_prefill_pressure_bytes(model: &ModelProfile) -> Option<u64> {
    match model.architecture_class {
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::Unknown => {
            Some(active_dense_decode_weight_traffic(model))
        }
        ModelArchitectureClass::SparseMoeTransformer => {
            let dense_active = active_moe_decode_weight_traffic(model);
            let groups = model.tensor_group_bytes;
            let resident = groups
                .attention_bytes
                .saturating_add(groups.feed_forward_bytes)
                .saturating_add(groups.expert_feed_forward_bytes)
                .saturating_add(groups.output_bytes)
                .saturating_add(groups.normalization_bytes)
                .saturating_add(groups.other_bytes);
            Some(dense_active.max(resident / 3))
        }
        _ => None,
    }
}

fn quantization_is_q8(quantization: Option<&str>) -> bool {
    let Some(quantization) = quantization else {
        return false;
    };
    let lower = quantization.to_ascii_lowercase();
    lower.contains("q8_0") || gguf_file_type_is(&lower, 7)
}

#[derive(Clone, Copy, Debug, Default)]
struct FirstTokenEstimate {
    prefill_ms: Option<f32>,
    decode_ms: Option<f32>,
    overhead_ms: Option<f32>,
    total_ms: Option<f32>,
}

fn first_token_estimate(
    prefill_tps: Option<f32>,
    decode_tps: Option<f32>,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
) -> FirstTokenEstimate {
    let prefill_ms = config
        .workload
        .interaction
        .expected_prompt_tokens
        .zip(prefill_tps)
        .map(|(prompt_tokens, tps)| prompt_tokens as f32 / tps.max(0.001) * 1000.0);
    let decode_ms = decode_tps.map(|tps| 1000.0 / tps.max(0.001));
    // This is a lower-bound hardware/runtime transition fact: it measures the
    // backend cost of issuing decode-shaped work immediately after
    // prefill-shaped matmul work without loading a GGUF. It deliberately does
    // not try to cover llama.cpp graph construction, KV/session bookkeeping,
    // tokenization, HTTP, or sampling. Validation keeps those residuals visible
    // so we do not smuggle model-benchmark observations into metadata-only fit.
    let overhead_value_ms = budget.post_prefill_decode_overhead_ms.unwrap_or(0.0);
    let overhead_ms = Some(overhead_value_ms);
    let total_ms =
        prefill_ms.map(|prefill| prefill + decode_ms.unwrap_or_default() + overhead_value_ms);
    FirstTokenEstimate {
        prefill_ms,
        decode_ms,
        overhead_ms,
        total_ms,
    }
}

fn first_token_ms_range(
    point: Option<f32>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<FirstTokenEstimateRange> {
    let point = point?;
    let mut uncertainty = 0.35f32;
    if budget.memory_bandwidth_bytes_per_sec.is_none() {
        uncertainty += 0.15;
    }
    if !tensor_groups_available(model.tensor_group_bytes) {
        uncertainty += 0.10;
    }
    if model.hidden_size.is_none() || model.layer_count.is_none() {
        uncertainty += 0.10;
    }
    let uncertainty = uncertainty.clamp(0.25, 0.70);
    Some(FirstTokenEstimateRange {
        lower_ms: point * (1.0 - uncertainty),
        upper_ms: point * (1.0 + uncertainty),
    })
}

fn dense_logical_matmul_count(model: &ModelProfile) -> u64 {
    let matmul = &model.tensor_matmul;
    matmul
        .attention
        .shape
        .logical_matrix_count
        .saturating_add(matmul.feed_forward.shape.logical_matrix_count)
        .saturating_add(matmul.output.shape.logical_matrix_count)
}

fn quantization_efficiency_factor(quantization: Option<&str>) -> f32 {
    let Some(quantization) = quantization else {
        return 0.90;
    };
    let lower = quantization.to_ascii_lowercase();
    if lower.contains("bf16") || gguf_file_type_is(&lower, 32) {
        0.45
    } else if lower.contains("f16") || gguf_file_type_is(&lower, 1) {
        0.60
    } else if lower.contains("q8_0") || gguf_file_type_is(&lower, 7) {
        1.0
    } else if lower.contains("q6_k") || gguf_file_type_is(&lower, 18) {
        0.88
    } else if lower.contains("q5_k")
        || gguf_file_type_is(&lower, 16)
        || gguf_file_type_is(&lower, 17)
    {
        0.95
    } else if lower.contains("iq") || gguf_file_type_is(&lower, 19) {
        0.85
    } else {
        1.0
    }
}

fn gguf_file_type_is(quantization: &str, expected: u32) -> bool {
    quantization
        .strip_prefix("gguf_file_type_")
        .and_then(|value| value.parse::<u32>().ok())
        == Some(expected)
}

fn decode_tokens_per_sec_range(
    point: Option<f32>,
    active_decode_bytes: Option<u64>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<DecodeEstimateRange> {
    let point = point?;
    let uncertainty = decode_uncertainty(active_decode_bytes, model, budget);
    Some(DecodeEstimateRange {
        lower: point * (1.0 - uncertainty),
        upper: point * (1.0 + uncertainty),
    })
}

fn decode_uncertainty(
    active_decode_bytes: Option<u64>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> f32 {
    let mut uncertainty = 0.25f32;
    if budget.memory_bandwidth_bytes_per_sec.is_none() {
        uncertainty += 0.20;
    }
    if !tensor_groups_available(model.tensor_group_bytes) {
        uncertainty += 0.15;
    }
    if matches!(
        model.architecture_class,
        ModelArchitectureClass::SparseMoeTransformer | ModelArchitectureClass::Unknown
    ) {
        uncertainty += 0.10;
    }
    if active_decode_bytes.is_some_and(|bytes| bytes < 2 * 1024 * 1024 * 1024) {
        uncertainty += 0.10;
    }
    if model
        .quantization
        .as_deref()
        .is_none_or(quantization_is_uncertain)
    {
        uncertainty += 0.05;
    }
    uncertainty.clamp(0.20, 0.60)
}

fn quantization_is_uncertain(quantization: &str) -> bool {
    let lower = quantization.to_ascii_lowercase();
    lower.contains("iq") || gguf_file_type_is(&lower, 19) || gguf_file_type_is(&lower, 20)
}

fn decode_bandwidth_efficiency(budget: &ExecutionBudget, config: &SelectionConfig) -> f32 {
    // There are two modes:
    //
    // 1. Measured bandwidth: use one portable decode-efficiency knob plus a
    //    small noise adjustment. The benchmark has already exercised the actual
    //    backend path, so backend-specific multipliers would double-count local
    //    assumptions.
    // 2. Estimated/manual bandwidth: fall back to backend defaults because a
    //    hand-authored profile needs some prior about Metal vs CUDA vs CPU.
    if budget.bandwidth_source == MeasurementSource::Measured {
        return benchmark_noise_factor(budget);
    }
    fallback_backend_efficiency(budget.backend, config)
}

fn benchmark_noise_factor(budget: &ExecutionBudget) -> f32 {
    // Noise is not used as a hard rejection. It is a mild pessimism term: if the
    // benchmark itself had meaningful spread, the selector should not rank a
    // model as if every decode token will get the best sustained bandwidth.
    let Some(noise_pct) = budget.benchmark_noise_pct else {
        return 1.0;
    };
    (1.0 - noise_pct.max(0.0) / 100.0 * 0.5).clamp(0.85, 1.0)
}

fn fallback_backend_efficiency(backend: BackendKind, config: &SelectionConfig) -> f32 {
    match backend {
        BackendKind::Metal => config.backend_efficiency.metal,
        BackendKind::Cuda => config.backend_efficiency.cuda,
        BackendKind::Rocm => config.backend_efficiency.rocm,
        BackendKind::Vulkan => config.backend_efficiency.vulkan,
        BackendKind::Cpu => config.backend_efficiency.cpu,
        BackendKind::Unknown => config.backend_efficiency.unknown,
    }
}

fn fixed_decode_overhead_ms(budget: &ExecutionBudget, config: &SelectionConfig) -> f32 {
    // Fixed overhead matters most when active bytes are small. For measured GPU
    // hardware this must come from the benchmark profile. A universal measured
    // GPU constant would encode backend/runtime behavior in model-fit instead
    // of observing it on the target machine.
    if measured_gpu_budget(budget) {
        return budget.decode_fixed_overhead_ms.unwrap_or(0.0);
    }
    let backend = budget.backend;
    match backend {
        BackendKind::Metal => config.decode_overhead.metal_fixed_ms,
        BackendKind::Cuda => config.decode_overhead.cuda_fixed_ms,
        BackendKind::Rocm => config.decode_overhead.rocm_fixed_ms,
        BackendKind::Vulkan => config.decode_overhead.vulkan_fixed_ms,
        BackendKind::Cpu => config.decode_overhead.cpu_fixed_ms,
        BackendKind::Unknown => config.decode_overhead.unknown_fixed_ms,
    }
}

fn measured_gpu_budget(budget: &ExecutionBudget) -> bool {
    budget.bandwidth_source == MeasurementSource::Measured
        && budget.backend != BackendKind::Cpu
        && budget.accelerator_kind != AcceleratorKind::Cpu
}

fn architecture_decode_overhead_ms(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
) -> f32 {
    match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => {
            model.layer_count.unwrap_or_default() as f32
                * moe_dispatch_overhead_ms_per_layer(budget, config)
        }
        _ => 0.0,
    }
}

fn moe_dispatch_overhead_ms_per_layer(budget: &ExecutionBudget, config: &SelectionConfig) -> f32 {
    // MoE decode adds source-visible graph work beyond dense FFN decode:
    // router logits/probabilities/top-k, GGML_OP_MUL_MAT_ID expert matmuls,
    // weighting, and expert aggregation. That work is real, but it is not a
    // backend-independent latency constant.
    //
    // On measured hardware, use the benchmark's fixed submission cost as the
    // per-layer dispatch unit and cap it at the fallback MoE prior. This keeps
    // the estimator portable:
    //
    // - a low-latency measured path with MMVQ/MMQ and fusion support in the
    //   llama.cpp backend is not forced to pay a large hand-written MoE
    //   constant;
    // - a higher-latency measured path still pays visible extra graph work;
    // - unmeasured hardware keeps the conservative config prior and carries the
    //   existing calibration warnings.
    //
    // The rule consumes only GGUF architecture class plus measured hardware
    // facts. It does not branch on backend names, model names, or observed model
    // throughput from validation.
    if measured_gpu_budget(budget) {
        return budget
            .decode_fixed_overhead_ms
            .filter(|value| *value > 0.0)
            .map(|fixed| fixed.min(config.decode_overhead.moe_dispatch_ms_per_layer))
            .unwrap_or(0.0);
    }
    config.decode_overhead.moe_dispatch_ms_per_layer
}

fn fit_status(
    model: &ModelProfile,
    memory: &RuntimeMemoryEstimate,
    budget: &ExecutionBudget,
    memory_limit: u64,
    workload_reject: bool,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> FitStatus {
    if !standalone_weight_coverage(model, reasons, warnings) {
        return FitStatus::Rejected;
    }
    if workload_reject {
        reasons.push("model capability evidence conflicts with required workload".into());
        return FitStatus::Rejected;
    }
    if memory.runtime_bytes <= memory_limit {
        reasons.push(format!(
            "estimated runtime memory fits within safety-adjusted budget ({:.1} GiB <= {:.1} GiB)",
            gib(memory.runtime_bytes),
            gib(memory_limit)
        ));
        if memory.runtime_bytes > memory_limit.saturating_mul(9) / 10 {
            warnings.push("model fits but leaves little memory headroom".into());
            FitStatus::FitsWithWarning
        } else {
            FitStatus::FitsLocal
        }
    } else if split_viable(model, memory, budget) {
        warnings.push("model does not fit locally but may be a Skippy split candidate".into());
        FitStatus::SplitCandidate
    } else {
        reasons.push(format!(
            "estimated runtime memory exceeds safety-adjusted budget ({:.1} GiB > {:.1} GiB)",
            gib(memory.runtime_bytes),
            gib(memory_limit)
        ));
        FitStatus::Rejected
    }
}

fn standalone_weight_coverage(
    model: &ModelProfile,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> bool {
    match model.weight_coverage {
        WeightCoverage::Full | WeightCoverage::Unknown => true,
        WeightCoverage::PartialTransformer {
            present_layers,
            expected_layers,
        } => {
            reasons.push(format!(
                "GGUF tensor coverage is partial ({present_layers}/{expected_layers} transformer blocks)"
            ));
            warnings.push(
                "partial GGUF artifacts are not ranked as standalone local model files".into(),
            );
            false
        }
        WeightCoverage::MetadataOnly => {
            reasons.push(
                "GGUF has model metadata but no standalone transformer weight coverage".into(),
            );
            warnings.push(
                "metadata-only or tokenizer/package GGUF artifacts are not standalone model candidates"
                    .into(),
            );
            false
        }
    }
}

fn split_viable(
    model: &ModelProfile,
    memory: &RuntimeMemoryEstimate,
    budget: &ExecutionBudget,
) -> bool {
    uses_transformer_kv_cache(model.architecture_class)
        && budget.usable_memory_bytes > 0
        && memory.resident_weight_bytes > budget.usable_memory_bytes / 2
}

fn split_candidate(
    model: &ModelProfile,
    memory: &RuntimeMemoryEstimate,
    _budget: &ExecutionBudget,
    memory_limit: u64,
    fit_status: FitStatus,
) -> Option<SplitCandidateEstimate> {
    if fit_status != FitStatus::SplitCandidate || memory_limit == 0 {
        return None;
    }
    let stages = memory.runtime_bytes.div_ceil(memory_limit).max(2);
    Some(SplitCandidateEstimate {
        estimated_stages: stages.min(u64::from(u32::MAX)) as u32,
        per_stage_memory_budget_bytes: memory_limit,
        warning: format!(
            "activation transfer depends on hidden_size={:?}, layers={:?}, and network bandwidth",
            model.hidden_size, model.layer_count
        ),
    })
}

fn memory_limit_with_margin(usable_memory_bytes: u64, safety_margin: f32) -> u64 {
    let margin = safety_margin.clamp(0.0, 0.9);
    (usable_memory_bytes as f32 * (1.0 - margin)) as u64
}

fn memory_score(runtime_bytes: u64, memory_limit: u64) -> f32 {
    if memory_limit == 0 || runtime_bytes > memory_limit {
        return 0.0;
    }
    let headroom = (memory_limit - runtime_bytes) as f32 / memory_limit as f32;
    (0.35 + headroom * 1.30).clamp(0.0, 1.0)
}

fn context_score(
    model: &ModelProfile,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> f32 {
    let required = required_context_tokens(config);
    let Some(native) = model.context_length else {
        warnings.push("model native context length is unknown".into());
        return 0.50;
    };
    if native >= required {
        reasons.push(format!(
            "native context {native} meets requested {required} tokens"
        ));
        return rope_context_penalty(model, required, warnings);
    }
    warnings.push(format!(
        "native context {native} is below requested {required} tokens"
    ));
    (native as f32 / required as f32).clamp(0.0, 0.80)
}

fn rope_context_penalty(model: &ModelProfile, required: u32, warnings: &mut Vec<String>) -> f32 {
    if model
        .rope
        .original_context_length
        .is_some_and(|original| original < required)
        && model.rope.finetuned != Some(true)
    {
        warnings.push("requested context appears to rely on unconfirmed rope scaling".into());
        return 0.80;
    }
    1.0
}

fn decode_score(estimated_decode_tps: Option<f32>, config: &SelectionConfig) -> f32 {
    if config.weights.decode == 0.0 {
        return 1.0;
    }
    let Some(tps) = estimated_decode_tps else {
        return 0.0;
    };
    let minimum = config
        .workload
        .preferences
        .minimum_decode_tps
        .unwrap_or(1.0);
    let preferred = config
        .workload
        .preferences
        .preferred_decode_tps
        .unwrap_or(minimum.max(1.0));
    if tps < minimum {
        return (tps / minimum * 0.40).clamp(0.0, 0.40);
    }
    (0.40 + (tps - minimum) / (preferred - minimum).max(1.0) * 0.60).clamp(0.0, 1.0)
}

fn prefill_score(model: &ModelProfile, config: &SelectionConfig, budget: &ExecutionBudget) -> f32 {
    let active = match model.architecture_class {
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            resident_weight_bytes(model)
        }
        ModelArchitectureClass::SparseMoeTransformer => active_moe_weight_bytes(model),
        _ => resident_weight_bytes(model),
    };
    let bandwidth = budget
        .memory_bandwidth_bytes_per_sec
        .unwrap_or(120_000_000_000) as f32
        * decode_bandwidth_efficiency(budget, config);
    if active == 0 {
        return 0.50;
    }
    let pressure = active as f32 / bandwidth.max(1.0);
    (1.0 / (1.0 + pressure)).clamp(0.0, 1.0)
}

fn workload_score(
    model: &ModelProfile,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> (f32, bool) {
    let requirements = &config.workload.requirements;
    let checks = [
        (
            requirements.chat_template,
            has(model, CapabilityEvidence::ChatTemplatePresent),
            "chat template",
        ),
        (
            requirements.system_messages,
            has(model, CapabilityEvidence::SystemRoleInChatTemplate),
            "system-message template support",
        ),
        (
            requirements.tool_calling,
            has(model, CapabilityEvidence::ToolUseTemplateMarkers),
            "tool-call template markers",
        ),
        (
            requirements.fill_in_middle,
            has(model, CapabilityEvidence::FillInMiddleTokensPresent),
            "fill-in-middle tokens",
        ),
        (
            requirements.embeddings,
            has(model, CapabilityEvidence::EmbeddingModel),
            "embedding model evidence",
        ),
        (
            requirements.reranking,
            has(model, CapabilityEvidence::ClassifierOrReranker),
            "reranker/classifier evidence",
        ),
        (
            requirements.vision,
            has(model, CapabilityEvidence::MultimodalProjector),
            "vision/multimodal evidence",
        ),
    ];

    let mut total = 0.0f32;
    let mut weight = 0.0f32;
    let mut reject = false;
    for (requirement, present, label) in checks {
        let (score, check_weight, failed) = requirement_score(requirement, present);
        if check_weight > 0.0 {
            if present {
                reasons.push(format!("workload evidence matched: {label}"));
            } else if requirement == Requirement::Required {
                warnings.push(format!("required workload evidence missing: {label}"));
            }
        }
        total += score * check_weight;
        weight += check_weight;
        reject |= failed;
    }

    let tag_score = explicit_tag_score(model, config);
    total += tag_score * 0.5;
    weight += 0.5;

    let score = if weight == 0.0 { 0.70 } else { total / weight };
    (score.clamp(0.0, 1.0), reject)
}

fn requirement_score(requirement: Requirement, present: bool) -> (f32, f32, bool) {
    match requirement {
        Requirement::Required => (if present { 1.0 } else { 0.0 }, 1.0, !present),
        Requirement::Preferred => (if present { 1.0 } else { 0.45 }, 0.75, false),
        Requirement::Neutral => (0.70, 0.0, false),
        Requirement::Penalize => (if present { 0.25 } else { 0.80 }, 0.50, false),
        Requirement::Reject => (if present { 0.0 } else { 0.80 }, 1.0, present),
    }
}

fn explicit_tag_score(model: &ModelProfile, config: &SelectionConfig) -> f32 {
    let tags = model
        .capability_evidence
        .iter()
        .filter_map(|evidence| match evidence {
            CapabilityEvidence::ExplicitGeneralTag(tag) => Some(tag.to_ascii_lowercase()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if tags.is_empty() {
        return 0.55;
    }
    let wanted = match config.workload.task {
        WorkloadTask::Coding => ["code", "coding", "programming", "fim"].as_slice(),
        WorkloadTask::ToolCalling => ["tool", "function", "agent"].as_slice(),
        WorkloadTask::Embedding => ["embedding", "sentence-transformers"].as_slice(),
        WorkloadTask::Reranking => ["rerank", "reranker", "classifier"].as_slice(),
        WorkloadTask::Summarization => ["summarization", "summary"].as_slice(),
        _ => ["chat", "instruct"].as_slice(),
    };
    if tags
        .iter()
        .any(|tag| wanted.iter().any(|wanted| tag.contains(wanted)))
    {
        1.0
    } else {
        0.60
    }
}

fn has(model: &ModelProfile, evidence: CapabilityEvidence) -> bool {
    model.capability_evidence.contains(&evidence)
}

fn total_score(
    weights: ScoreWeights,
    memory_score: f32,
    context_score: f32,
    decode_score: f32,
    prefill_score: f32,
    workload_score: f32,
    fit_status: FitStatus,
) -> f32 {
    if fit_status == FitStatus::Rejected {
        return 0.0;
    }
    let weight_sum =
        weights.memory + weights.context + weights.decode + weights.prefill + weights.workload;
    if weight_sum <= 0.0 {
        return 0.0;
    }
    let score = weights.memory * memory_score
        + weights.context * context_score
        + weights.decode * decode_score
        + weights.prefill * prefill_score
        + weights.workload * workload_score;
    let status_factor = match fit_status {
        FitStatus::FitsLocal => 1.0,
        FitStatus::FitsWithWarning => 0.85,
        FitStatus::SplitCandidate => 0.55,
        FitStatus::Rejected => 0.0,
    };
    (score / weight_sum * status_factor).clamp(0.0, 1.0)
}

fn target_context_tokens(model: &ModelProfile, config: &SelectionConfig) -> u32 {
    let required = required_context_tokens(config);
    let model_context = model.context_length.unwrap_or(required);
    round_up_context(required.min(model_context.max(required.min(4_096))))
}

fn required_context_tokens(config: &SelectionConfig) -> u32 {
    let from_requirements = config.workload.requirements.min_context_tokens;
    let from_interaction = config.workload.interaction.expected_prompt_tokens;
    from_requirements
        .into_iter()
        .chain(from_interaction)
        .max()
        .unwrap_or(4_096)
        .max(1)
}

fn round_up_context(tokens: u32) -> u32 {
    tokens.div_ceil(256) * 256
}

fn uses_transformer_kv_cache(class: ModelArchitectureClass) -> bool {
    matches!(
        class,
        ModelArchitectureClass::DenseTransformer
            | ModelArchitectureClass::SparseMoeTransformer
            | ModelArchitectureClass::Unknown
    )
}

fn add_architecture_warnings(model: &ModelProfile, warnings: &mut Vec<String>) {
    match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => warnings.push(
            "MoE decode estimate uses active experts, but resident memory may still require all experts"
                .into(),
        ),
        ModelArchitectureClass::Unknown => {
            warnings.push("unknown architecture; estimates use conservative full-tensor assumptions".into());
        }
        ModelArchitectureClass::RecurrentOrStateSpace => warnings.push(
            "recurrent/state-space architecture has approximate context and decode estimates".into(),
        ),
        _ => {}
    }
}

fn add_decode_estimate_reason(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
) {
    if !uses_transformer_kv_cache(model.architecture_class) {
        return;
    }
    let matmul = &model.tensor_matmul;
    let grouped_matmul_bytes = matmul
        .attention
        .bytes
        .saturating_add(matmul.feed_forward.bytes)
        .saturating_add(matmul.output.bytes)
        .saturating_add(matmul.expert_feed_forward.bytes);
    if grouped_matmul_bytes > 0 {
        reasons.push(format!(
            "decode estimate uses GGUF matmul groups ({:.1} GiB attention, {:.1} GiB FFN, {:.1} GiB output, {:.1} GiB expert FFN pool)",
            gib(matmul.attention.bytes),
            gib(matmul.feed_forward.bytes),
            gib(matmul.output.bytes),
            gib(matmul.expert_feed_forward.bytes)
        ));
        if let Some(flops) = active_decode_flops_per_token(model) {
            reasons.push(format!(
                "decode estimate includes {:.1} GFLOP/token matmul compute floor from GGUF tensor shapes",
                flops as f32 / 1_000_000_000.0
            ));
        }
        add_matmul_shape_reason(matmul, reasons);
    } else if model.tensor_matmul.base_bytes > 0 || model.tensor_matmul.expert_bytes > 0 {
        reasons.push(format!(
            "decode estimate uses GGUF matmul tensors ({:.1} GiB base, {:.1} GiB expert pool) and active MoE experts when present",
            gib(model.tensor_matmul.base_bytes),
            gib(model.tensor_matmul.expert_bytes)
        ));
        if let Some(flops) = active_decode_flops_per_token(model) {
            reasons.push(format!(
                "decode estimate includes {:.1} GFLOP/token matmul compute floor from GGUF tensor shapes",
                flops as f32 / 1_000_000_000.0
            ));
        }
        add_matmul_shape_reason(matmul, reasons);
    } else if tensor_groups_available(model.tensor_group_bytes) {
        reasons.push(
            "decode estimate uses GGUF tensor groups for attention, FFN, experts, output, and KV pressure"
                .into(),
        );
    }
    let fixed_ms = fixed_decode_overhead_ms(budget, config);
    if measured_gpu_budget(budget)
        && let Some(bytes_per_sec) = budget.decode_effective_bandwidth_bytes_per_sec
    {
        reasons.push(format!(
            "decode estimate uses measured decode-shaped bandwidth ({:.1} GB/s)",
            bytes_per_sec as f32 / 1_000_000_000.0
        ));
    }
    let arch_ms = architecture_decode_overhead_ms(model, budget, config);
    let graph_ms = measured_decode_graph_overhead_ms(model, budget);
    let shape_adjustment = active_decode_bytes_per_token(model, config)
        .map(|bytes| (decode_shape_bandwidth_factor(model, bytes), graph_ms));
    if arch_ms > 0.0 {
        reasons.push(format!(
            "decode estimate adds {:.1} ms/token backend overhead, {:.1} ms/token measured graph overhead, and {:.1} ms/token architecture overhead from GGUF metadata",
            fixed_ms, graph_ms, arch_ms
        ));
    } else {
        reasons.push(format!(
            "decode estimate adds {:.1} ms/token backend overhead and {:.1} ms/token measured graph overhead",
            fixed_ms, graph_ms
        ));
    }
    match shape_adjustment {
        Some((shape_factor, graph_ms)) if shape_factor != 1.0 || graph_ms > 0.0 => {
            reasons.push(format!(
                "decode estimate applies {:.2}x shape bandwidth factor and {:.1} ms/token measured graph overhead from GGUF layer/matmul shape metadata",
                shape_factor, graph_ms
            ));
        }
        _ => {}
    }
}

fn add_matmul_shape_reason(matmul: &crate::TensorMatmulProfile, reasons: &mut Vec<String>) {
    let ffn = matmul.feed_forward.shape;
    if ffn.logical_matrix_count == 0 {
        return;
    }
    reasons.push(format!(
        "decode estimate uses FFN matmul shape summary ({} logical matrices, avg {}x{}, width range {}..{} by {}..{})",
        ffn.logical_matrix_count,
        ffn.weighted_avg_output_width,
        ffn.weighted_avg_input_width,
        ffn.min_output_width,
        ffn.max_output_width,
        ffn.min_input_width,
        ffn.max_input_width
    ));
}

fn add_prefill_estimate_reason(
    estimated_first_token_ms: Option<f32>,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
) {
    let Some(estimated_first_token_ms) = estimated_first_token_ms else {
        return;
    };
    let prompt_tokens = config
        .workload
        .interaction
        .expected_prompt_tokens
        .unwrap_or_default();
    reasons.push(format!(
        "prefill estimate predicts {:.0} ms first-token latency for about {prompt_tokens} prompt tokens",
        estimated_first_token_ms
    ));
}

fn estimate_confidence(model: &ModelProfile, budget: &ExecutionBudget) -> EstimateConfidence {
    if model.weight_coverage != WeightCoverage::Full {
        return EstimateConfidence::Low;
    }
    if model.architecture_class == ModelArchitectureClass::Unknown
        || budget.memory_bandwidth_bytes_per_sec.is_none()
        || (measured_gpu_budget(budget)
            && budget.decode_effective_bandwidth_bytes_per_sec.is_none())
    {
        return EstimateConfidence::Low;
    }
    if model.tensor_bytes.is_some()
        && model.layer_count.is_some()
        && model.hidden_size.is_some()
        && model.context_length.is_some()
    {
        EstimateConfidence::High
    } else {
        EstimateConfidence::Medium
    }
}

fn compare_recommendations(left: &ModelRecommendation, right: &ModelRecommendation) -> Ordering {
    status_rank(left.fit_status)
        .cmp(&status_rank(right.fit_status))
        .then_with(|| compare_f32_desc(left.total_score, right.total_score))
        .then_with(|| {
            compare_option_f32_desc(
                left.estimated_decode_tokens_per_sec,
                right.estimated_decode_tokens_per_sec,
            )
        })
        .then_with(|| compare_f32_desc(left.context_score, right.context_score))
        .then_with(|| {
            left.estimated_runtime_memory_bytes
                .cmp(&right.estimated_runtime_memory_bytes)
        })
        .then_with(|| left.source.id.cmp(&right.source.id))
}

fn compare_execution_budget_recommendations(
    left: &ModelRecommendation,
    right: &ModelRecommendation,
) -> Ordering {
    // This comparator is deliberately different from cross-model ranking.
    //
    // When we are choosing *how this one model should run on this one machine*,
    // memory headroom should decide only after the execution path is viable and
    // the estimated throughput is known. Otherwise a CPU budget with lots of
    // spare RAM can beat a measured GPU budget, causing the model's published
    // fit estimate to describe the wrong execution path. The Qwen2.5-Coder 7B
    // validation run caught exactly that shape: the model fit in GPU VRAM, but
    // the old comparator selected the CPU memory budget because its memory
    // score was higher, dropping the predicted decode rate by almost an order of
    // magnitude.
    //
    // Also prefer estimates backed by measured hardware facts over estimates
    // that came from fallback bandwidth constants. A fallback CPU budget should
    // not outrank a measured Metal/CUDA/ROCm budget just because the fallback
    // has no measured graph overhead. If a future CPU has measured bandwidth
    // and genuinely predicts faster decode, it can still win on the same rule.
    //
    // This is not a CUDA/Metal/backend preference and it is not model-specific.
    // It says: for a single model, among budgets with the same fit status and
    // estimate evidence quality, prefer the budget whose metadata-only estimate
    // predicts faster decode.
    status_rank(left.fit_status)
        .cmp(&status_rank(right.fit_status))
        .then_with(|| estimate_evidence_rank(left).cmp(&estimate_evidence_rank(right)))
        .then_with(|| {
            compare_option_f32_desc(
                left.estimated_decode_tokens_per_sec,
                right.estimated_decode_tokens_per_sec,
            )
        })
        .then_with(|| compare_f32_desc(left.total_score, right.total_score))
        .then_with(|| compare_f32_desc(left.memory_score, right.memory_score))
        .then_with(|| {
            left.estimated_runtime_memory_bytes
                .cmp(&right.estimated_runtime_memory_bytes)
        })
        .then_with(|| left.source.id.cmp(&right.source.id))
}

fn estimate_evidence_rank(recommendation: &ModelRecommendation) -> u8 {
    match recommendation.estimate_confidence {
        EstimateConfidence::High => 0,
        EstimateConfidence::Medium => 1,
        EstimateConfidence::Low => 2,
    }
}

fn status_rank(status: FitStatus) -> u8 {
    match status {
        FitStatus::FitsLocal => 0,
        FitStatus::FitsWithWarning => 1,
        FitStatus::SplitCandidate => 2,
        FitStatus::Rejected => 3,
    }
}

fn compare_f32_desc(left: f32, right: f32) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn compare_option_f32_desc(left: Option<f32>, right: Option<f32>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => compare_f32_desc(left, right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn gib(bytes: u64) -> f32 {
    bytes as f32 / 1024.0 / 1024.0 / 1024.0
}
