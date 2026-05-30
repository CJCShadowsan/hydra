use crate::{
    AcceleratorKind, AcceleratorProfile, BackendKind, CapabilityEvidence, EstimateConfidence,
    FitStatus, HardwareProfile, KvCacheKind, ModelArchitectureClass, ModelProfile,
    ModelRecommendation, Requirement, ScoreWeights, SelectionConfig, SplitCandidateEstimate,
    WeightCoverage, WorkloadTask,
};
use std::cmp::Ordering;

const MIB: u64 = 1024 * 1024;

#[derive(Clone, Debug)]
struct ExecutionBudget {
    backend: BackendKind,
    accelerator_name: Option<String>,
    usable_memory_bytes: u64,
    memory_bandwidth_bytes_per_sec: Option<u64>,
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
            usable_memory_bytes: 0,
            memory_bandwidth_bytes_per_sec: None,
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
            usable_memory_bytes: memory,
            memory_bandwidth_bytes_per_sec: hardware.cpu.memory_bandwidth_bytes_per_sec,
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
        usable_memory_bytes,
        memory_bandwidth_bytes_per_sec: accelerator.memory_bandwidth_bytes_per_sec,
        unified_memory: accelerator.unified_memory,
    }
}

fn runtime_memory_estimate(
    model: &ModelProfile,
    config: &SelectionConfig,
) -> RuntimeMemoryEstimate {
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
    let active_weights = match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => active_moe_weight_bytes(model),
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            return None;
        }
        _ => resident_weight_bytes(model),
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

fn active_moe_weight_bytes(model: &ModelProfile) -> u64 {
    let base = model.base_resident_bytes.unwrap_or(0);
    let expert = model.expert_tensor_bytes.unwrap_or(0);
    let Some(expert_count) = model.expert_count.filter(|count| *count > 0) else {
        return resident_weight_bytes(model);
    };
    let active = model
        .expert_used_count
        .unwrap_or(expert_count)
        .min(expert_count);
    base.saturating_add(expert.saturating_mul(u64::from(active)) / u64::from(expert_count))
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
    let bytes = active_decode_bytes?;
    if bytes == 0 {
        return None;
    }
    let raw_bandwidth = budget
        .memory_bandwidth_bytes_per_sec
        .unwrap_or(match budget.backend {
            BackendKind::Cpu => 80_000_000_000,
            _ => 200_000_000_000,
        });
    let efficiency = backend_efficiency(budget.backend, config);
    let architecture_factor = match model.architecture_class {
        ModelArchitectureClass::Unknown => 0.75,
        ModelArchitectureClass::RecurrentOrStateSpace => 0.85,
        _ => 1.0,
    };
    let effective_bandwidth = raw_bandwidth as f32 * efficiency * architecture_factor;
    let bandwidth_ms = bytes as f32 / effective_bandwidth.max(1.0) * 1000.0;
    let overhead_ms = fixed_decode_overhead_ms(budget.backend, config)
        + architecture_decode_overhead_ms(model, config);
    Some(1000.0 / (bandwidth_ms + overhead_ms).max(0.001))
}

fn backend_efficiency(backend: BackendKind, config: &SelectionConfig) -> f32 {
    match backend {
        BackendKind::Metal => config.backend_efficiency.metal,
        BackendKind::Cuda => config.backend_efficiency.cuda,
        BackendKind::Rocm => config.backend_efficiency.rocm,
        BackendKind::Vulkan => config.backend_efficiency.vulkan,
        BackendKind::Cpu => config.backend_efficiency.cpu,
        BackendKind::Unknown => config.backend_efficiency.unknown,
    }
}

fn fixed_decode_overhead_ms(backend: BackendKind, config: &SelectionConfig) -> f32 {
    match backend {
        BackendKind::Metal => config.decode_overhead.metal_fixed_ms,
        BackendKind::Cuda => config.decode_overhead.cuda_fixed_ms,
        BackendKind::Rocm => config.decode_overhead.rocm_fixed_ms,
        BackendKind::Vulkan => config.decode_overhead.vulkan_fixed_ms,
        BackendKind::Cpu => config.decode_overhead.cpu_fixed_ms,
        BackendKind::Unknown => config.decode_overhead.unknown_fixed_ms,
    }
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
        * backend_efficiency(budget.backend, config);
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
    let fixed_ms = fixed_decode_overhead_ms(budget.backend, config);
    let arch_ms = architecture_decode_overhead_ms(model, config);
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
