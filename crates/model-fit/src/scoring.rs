use crate::{
    AcceleratorKind, AcceleratorProfile, BackendKind, CapabilityEvidence, DecodeEstimateRange,
    EstimateConfidence, FirstTokenEstimateRange, FitStatus, HardwareProfile, KvCacheKind,
    MeasurementSource, ModelArchitectureClass, ModelProfile, ModelRecommendation, Requirement,
    ScoreWeights, SelectionConfig, SplitCandidateEstimate, TensorGroupBytes, WeightCoverage,
    WorkloadTask,
};
use std::cmp::Ordering;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

#[derive(Clone, Debug)]
struct ExecutionBudget {
    backend: BackendKind,
    accelerator_name: Option<String>,
    accelerator_kind: AcceleratorKind,
    usable_memory_bytes: u64,
    memory_bandwidth_bytes_per_sec: Option<u64>,
    bandwidth_source: MeasurementSource,
    benchmark_noise_pct: Option<f32>,
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
            bandwidth_source: MeasurementSource::Unknown,
            benchmark_noise_pct: None,
            unified_memory: false,
        };
        return score_for_budget(model, config, &budget);
    }

    recommendations.sort_by(compare_recommendations);
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
    let estimated_decode_tps = decode_tokens_per_sec(active_decode_bytes, budget, config, model);
    let estimated_decode_range =
        decode_tokens_per_sec_range(estimated_decode_tps, active_decode_bytes, model, budget);
    let estimated_prefill_tps = prefill_tokens_per_sec(model, config, estimated_decode_tps);
    let estimated_first_token_ms =
        first_token_ms(estimated_prefill_tps, estimated_decode_tps, config);
    let estimated_first_token_range = first_token_ms_range(estimated_first_token_ms, model, budget);
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
    if budget.unified_memory {
        reasons.push("using unified-memory budget for model weights, KV cache, and scratch".into());
    }
    reasons.push(format!(
        "runtime estimate includes {:.1} GiB scratch and {:.1} GiB backend overhead",
        gib(memory.scratch_bytes),
        gib(memory.backend_overhead_bytes)
    ));
    add_decode_estimate_reason(model, budget, config, &mut reasons);
    add_prefill_estimate_reason(estimated_first_token_ms, config, &mut reasons);
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
        estimated_first_token_ms,
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
            bandwidth_source: hardware
                .cpu
                .memory_bandwidth_bytes_per_sec
                .map(|_| MeasurementSource::Measured)
                .unwrap_or(MeasurementSource::Unknown),
            benchmark_noise_pct: None,
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
        bandwidth_source: accelerator.bandwidth_source,
        benchmark_noise_pct: accelerator.benchmark_noise_pct,
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
        ModelArchitectureClass::SparseMoeTransformer => active_moe_decode_weight_bytes(model),
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            return None;
        }
        _ => active_dense_decode_weight_bytes(model),
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

fn active_dense_decode_weight_bytes(model: &ModelProfile) -> u64 {
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

fn active_moe_decode_weight_bytes(model: &ModelProfile) -> u64 {
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
    // GGUF tensor bytes are a resident-storage fact, not a perfect proxy for the
    // effective memory traffic seen by one llama.cpp decode token. The selector
    // still starts from tensor-group storage because it is the durable metadata
    // available for arbitrary GGUFs, then applies only quantization/structure
    // corrections that come from the file metadata itself.
    //
    // The important case is Q8_0. Q8_0 stores about twice as many bytes as a
    // Q4_K model, but it is a simple block format with less dequant bookkeeping
    // than the K-quants. In validation, treating Q8 resident bytes as literal
    // per-token traffic consistently underpredicted dense Q8 decode throughput.
    // This is not a backend or model-name boost: any GGUF with the same
    // quantization and dense/MoE structure gets the same storage-to-traffic
    // adjustment. Memory fit still uses resident bytes; only the decode traffic
    // slope is adjusted.
    let scale = decode_weight_traffic_scale(model);
    ((storage_bytes as f64) * f64::from(scale)).round() as u64
}

fn decode_weight_traffic_scale(model: &ModelProfile) -> f32 {
    if !quantization_is_q8(model.quantization.as_deref()) {
        return 1.0;
    }
    match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => 0.78,
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::Unknown => 0.60,
        _ => 1.0,
    }
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
    let raw_bandwidth = raw_memory_bandwidth_bytes_per_sec(budget);
    let efficiency = decode_bandwidth_efficiency(budget, config);
    let architecture_factor = match model.architecture_class {
        ModelArchitectureClass::Unknown => 0.75,
        ModelArchitectureClass::RecurrentOrStateSpace => 0.85,
        _ => 1.0,
    };
    let quantization_factor = quantization_efficiency_factor(model.quantization.as_deref());
    let shape_factor = decode_shape_bandwidth_factor(model, bytes);
    let effective_bandwidth = raw_bandwidth as f32
        * efficiency
        * architecture_factor
        * quantization_factor
        * shape_factor;
    let bandwidth_ms = bytes as f32 / effective_bandwidth.max(1.0) * 1000.0;
    let overhead_ms = fixed_decode_overhead_ms(budget, config)
        + architecture_decode_overhead_ms(model, config)
        + dense_medium_width_decode_overhead_ms(model, bytes)
        + low_active_decode_overhead_ms(model, budget, bytes)
        + small_width_decode_overhead_ms(model, budget, bytes);
    Some(1000.0 / (bandwidth_ms + overhead_ms).max(0.001))
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
    if hidden >= 4096 && active_gib >= 4.0 {
        1.12
    } else {
        1.0
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

fn prefill_tokens_per_sec(
    model: &ModelProfile,
    config: &SelectionConfig,
    decode_tokens_per_sec: Option<f32>,
) -> Option<f32> {
    // Prefill and decode are not the same operation. Prefill processes many
    // prompt tokens at once and can expose much more parallelism; decode is a
    // repeated one-token loop and is commonly memory-bandwidth-bound. In early
    // validation, trying to independently predict prefill from raw bandwidth,
    // attention proxies, and prompt length produced unstable results from the
    // small amount of metadata we can rely on for arbitrary GGUFs.
    //
    // The current first-pass model derives prefill throughput from the decode
    // estimate multiplied by a shape-based parallelism factor. That keeps the
    // two estimates correlated through the same hardware/profile facts while
    // still allowing prefill to be much faster for narrow or small active-byte
    // models. The first-token estimator then combines prompt_tokens / prefill
    // with one decode step.
    if !uses_transformer_kv_cache(model.architecture_class) {
        return None;
    }
    let prompt_tokens = config.workload.interaction.expected_prompt_tokens?;
    if prompt_tokens == 0 {
        return None;
    }
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
            Some(active_dense_decode_weight_bytes(model))
        }
        ModelArchitectureClass::SparseMoeTransformer => {
            let dense_active = active_moe_decode_weight_bytes(model);
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

fn first_token_ms(
    prefill_tps: Option<f32>,
    decode_tps: Option<f32>,
    config: &SelectionConfig,
) -> Option<f32> {
    let prompt_tokens = config.workload.interaction.expected_prompt_tokens?;
    let prefill_tps = prefill_tps?;
    let prefill_ms = prompt_tokens as f32 / prefill_tps.max(0.001) * 1000.0;
    let decode_ms = decode_tps.map(|tps| 1000.0 / tps.max(0.001)).unwrap_or(0.0);
    Some(prefill_ms + decode_ms)
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

fn small_width_decode_overhead_ms(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    active_decode_bytes: u64,
) -> f32 {
    // Narrow transformer shapes tend to under-deliver against a pure
    // bytes/bandwidth estimate. There is less work per layer to amortize backend
    // fixed costs, and the execution path can become dominated by kernel
    // scheduling, synchronization, and tensor-shape overheads. The effect is
    // strongest for local-sized models, then fades for very large active byte
    // footprints where streaming weights dominates again.
    //
    // This is based only on hidden width and active bytes. It deliberately does
    // not special-case particular small models; the same rule should apply to a
    // future tiny GGUF with similar geometry.
    if !matches!(
        model.architecture_class,
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::SparseMoeTransformer
    ) {
        return 0.0;
    }
    let Some(hidden) = model.hidden_size.filter(|hidden| *hidden > 0) else {
        return 0.0;
    };
    let width_pressure = (4096.0 / hidden as f32 - 1.0).max(0.0);
    if width_pressure == 0.0 {
        return 0.0;
    }
    let active_gib = active_decode_bytes as f32 / GIB as f32;
    let active_factor = if active_gib < 0.25 {
        0.05
    } else if active_gib < 0.50 {
        0.22
    } else if active_gib < 1.0 {
        0.42
    } else if active_gib < 2.0 {
        0.70
    } else if active_gib < 6.0 {
        1.0
    } else if active_gib < 10.0 {
        0.50
    } else {
        0.0
    };
    let backend_factor = width_overhead_factor(budget);
    (width_pressure * active_factor * backend_factor).clamp(0.0, 3.0)
}

fn width_overhead_factor(budget: &ExecutionBudget) -> f32 {
    // Once bandwidth is measured, prefer a backend-neutral overhead factor. The
    // measured profile already encodes the actual device/backend path well
    // enough for this first-pass selector, and keeping the measured path neutral
    // avoids overfitting this crate to one machine's Metal/CUDA behavior.
    if measured_gpu_budget(budget) {
        return 3.0;
    }
    match budget.backend {
        BackendKind::Metal => 3.0,
        BackendKind::Cuda => 1.0,
        BackendKind::Rocm => 1.5,
        BackendKind::Vulkan => 2.0,
        BackendKind::Cpu | BackendKind::Unknown => 0.5,
    }
}

fn dense_medium_width_decode_overhead_ms(model: &ModelProfile, active_decode_bytes: u64) -> f32 {
    // Medium-width dense models below roughly 1 GiB active bytes occupy an
    // awkward middle ground: they are large enough that fixed overhead alone is
    // not the whole story, but not large enough to behave like clean streaming
    // bandwidth tests. This small additive term keeps that region from being
    // overestimated without penalizing larger 7B/8B dense models.
    if !matches!(
        model.architecture_class,
        ModelArchitectureClass::DenseTransformer
    ) {
        return 0.0;
    }
    let Some(hidden) = model.hidden_size else {
        return 0.0;
    };
    let active_gib = active_decode_bytes as f32 / GIB as f32;
    if (1536..=2304).contains(&hidden) && active_gib < 1.0 {
        1.15
    } else {
        0.0
    }
}

fn low_active_decode_overhead_ms(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    active_decode_bytes: u64,
) -> f32 {
    // Tiny active-byte models are the main place where "bytes divided by memory
    // bandwidth" lies. A 100-600M class model may touch so little memory per
    // token that constant runtime costs become a large share of latency. In
    // validation this showed up as stable overprediction for small GGUFs even
    // after using decode-only timing and longer token windows.
    //
    // Apply this only to measured GPU profiles and only below 0.5 GiB active
    // bytes. For unmeasured profiles we leave uncertainty wider rather than
    // pretending this calibrated overhead is portable. The hidden-width term
    // increases the penalty for very narrow models, where there is even less
    // useful work to amortize per-token overhead.
    if !measured_gpu_budget(budget) {
        return 0.0;
    }
    let Some(hidden) = model.hidden_size.filter(|hidden| *hidden > 0) else {
        return 0.0;
    };
    let active_gib = active_decode_bytes as f32 / GIB as f32;
    if active_gib >= 0.50 {
        return 0.0;
    }
    let active_deficit = (0.50 - active_gib) / 0.50;
    let width_factor = (1024.0 / hidden as f32).max(1.0).sqrt();
    (active_deficit * width_factor * 1.60).clamp(0.0, 1.80)
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
        return config.measured_decode_efficiency * benchmark_noise_factor(budget);
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
    // hardware we use a single measured-GPU value to keep the primary path
    // portable across backend labels. For unmeasured profiles, backend defaults
    // remain useful priors.
    if measured_gpu_budget(budget) {
        return config.decode_overhead.measured_gpu_fixed_ms;
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

fn architecture_decode_overhead_ms(model: &ModelProfile, config: &SelectionConfig) -> f32 {
    match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => {
            model.layer_count.unwrap_or_default() as f32
                * config.decode_overhead.moe_dispatch_ms_per_layer
        }
        _ => 0.0,
    }
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
    if let Some(original) = model.rope.original_context_length
        && original < required
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
    if tensor_groups_available(model.tensor_group_bytes) {
        reasons.push(
            "decode estimate uses GGUF tensor groups for attention, FFN, experts, output, and KV pressure"
                .into(),
        );
    }
    let fixed_ms = fixed_decode_overhead_ms(budget, config);
    let arch_ms = architecture_decode_overhead_ms(model, config);
    let shape_adjustment = active_decode_bytes_per_token(model, config).map(|bytes| {
        (
            decode_shape_bandwidth_factor(model, bytes),
            dense_medium_width_decode_overhead_ms(model, bytes),
            low_active_decode_overhead_ms(model, budget, bytes),
        )
    });
    if arch_ms > 0.0 {
        reasons.push(format!(
            "decode estimate adds {:.1} ms/token backend overhead and {:.1} ms/token architecture overhead from GGUF metadata",
            fixed_ms, arch_ms
        ));
    } else {
        reasons.push(format!(
            "decode estimate adds {:.1} ms/token backend overhead",
            fixed_ms
        ));
    }
    if let Some((shape_factor, medium_width_ms, low_active_ms)) = shape_adjustment
        && (shape_factor != 1.0 || medium_width_ms > 0.0 || low_active_ms > 0.0)
    {
        reasons.push(format!(
            "decode estimate applies {:.2}x shape bandwidth factor, {:.1} ms/token dense-width overhead, and {:.1} ms/token low-active overhead from hidden size and active bytes",
            shape_factor, medium_width_ms, low_active_ms
        ));
    }
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
