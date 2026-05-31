use crate::{
    CapabilityEvidence, MatmulShapeProfile, ModelArchitectureClass, ModelProfile, ModelSource,
    RopeProfile, TensorGroupBytes, TensorMatmulGroupProfile, TensorMatmulProfile, TensorTypeBytes,
    TokenizerProfile, WeightCoverage,
};
use anyhow::{Context, Result};
use model_artifact::gguf::{
    scan_gguf_compact_meta, scan_gguf_fit_meta, scan_gguf_tensor_byte_profile,
};
use std::path::Path;

pub fn profile_gguf_path(path: impl AsRef<Path>) -> Result<ModelProfile> {
    let path = path.as_ref();
    let compact = scan_gguf_compact_meta(path)
        .with_context(|| format!("scan compact GGUF metadata from {}", path.display()))?;
    let fit = scan_gguf_fit_meta(path)
        .with_context(|| format!("scan fit GGUF metadata from {}", path.display()))?;
    let tensor_profile = scan_gguf_tensor_byte_profile(path)
        .with_context(|| format!("scan GGUF tensor profile from {}", path.display()))?;
    let file_size_bytes = std::fs::metadata(path)
        .with_context(|| format!("stat GGUF {}", path.display()))?
        .len();

    let mut capability_evidence = capability_evidence(&compact, &fit);
    capability_evidence.sort_by_key(evidence_sort_key);
    capability_evidence.dedup();

    let architecture_class = architecture_class(&compact, &fit, &capability_evidence);

    Ok(ModelProfile {
        source: ModelSource {
            id: path.display().to_string(),
            path: Some(path.to_path_buf()),
            metadata_name: fit.general_name.clone(),
        },
        architecture: non_empty(compact.architecture.clone()),
        architecture_class,
        weight_coverage: weight_coverage(&compact, architecture_class, &tensor_profile),
        file_size_bytes,
        tensor_bytes: Some(tensor_profile.full_model_bytes),
        base_resident_bytes: Some(tensor_profile.base_resident_bytes),
        expert_tensor_bytes: Some(tensor_profile.expert_tensor_bytes),
        tensor_group_bytes: TensorGroupBytes {
            attention_bytes: tensor_profile.group_bytes.attention_bytes,
            feed_forward_bytes: tensor_profile.group_bytes.feed_forward_bytes,
            expert_feed_forward_bytes: tensor_profile.group_bytes.expert_feed_forward_bytes,
            embedding_bytes: tensor_profile.group_bytes.embedding_bytes,
            output_bytes: tensor_profile.group_bytes.output_bytes,
            normalization_bytes: tensor_profile.group_bytes.normalization_bytes,
            other_bytes: tensor_profile.group_bytes.other_bytes,
        },
        tensor_matmul: TensorMatmulProfile {
            base_bytes: tensor_profile.matmul.base_bytes,
            expert_bytes: tensor_profile.matmul.expert_bytes,
            base_flops_per_token: tensor_profile.matmul.base_flops_per_token,
            expert_flops_per_token: tensor_profile.matmul.expert_flops_per_token,
            base_type_bytes: tensor_type_bytes(tensor_profile.matmul.base_type_bytes),
            expert_type_bytes: tensor_type_bytes(tensor_profile.matmul.expert_type_bytes),
            attention: tensor_matmul_group(tensor_profile.matmul.attention),
            feed_forward: tensor_matmul_group(tensor_profile.matmul.feed_forward),
            expert_feed_forward: tensor_matmul_group(tensor_profile.matmul.expert_feed_forward),
            output: tensor_matmul_group(tensor_profile.matmul.output),
        },
        parameter_count: parameter_count_from_size_label(compact.parameter_size.as_deref()),
        quantization: fit
            .file_type
            .map(|file_type| format!("gguf_file_type_{file_type}")),
        layer_count: non_zero(compact.layer_count),
        hidden_size: non_zero(compact.embedding_size),
        ffn_size: non_zero(compact.feed_forward_length),
        attention_heads: non_zero(compact.head_count),
        kv_heads: compact.effective_kv_head_count(),
        key_length: non_zero(compact.key_length),
        value_length: non_zero(compact.value_length),
        context_length: non_zero(compact.context_length),
        expert_count: non_zero(compact.expert_count),
        expert_used_count: non_zero(compact.expert_used_count),
        rope: RopeProfile {
            scale: non_default_f32(compact.rope_scale),
            freq_base: non_default_f32(compact.rope_freq_base),
            scaling_type: fit.rope_scaling_type.clone(),
            scaling_factor: fit.rope_scaling_factor,
            original_context_length: fit.rope_scaling_original_context_length,
            finetuned: fit.rope_scaling_finetuned,
        },
        tokenizer: TokenizerProfile {
            model: non_empty(compact.tokenizer_model_name.clone()),
            chat_template_available: fit.has_chat_template(),
        },
        capability_evidence,
    })
}

fn tensor_matmul_group(
    group: model_artifact::gguf::GgufMatmulGroupProfile,
) -> TensorMatmulGroupProfile {
    TensorMatmulGroupProfile {
        bytes: group.bytes,
        flops_per_token: group.flops_per_token,
        type_bytes: tensor_type_bytes(group.type_bytes),
        shape: matmul_shape(group.shape),
    }
}

fn matmul_shape(shape: model_artifact::gguf::GgufMatmulShapeProfile) -> MatmulShapeProfile {
    MatmulShapeProfile {
        tensor_count: shape.tensor_count,
        logical_matrix_count: shape.logical_matrix_count,
        total_elements: shape.total_elements,
        min_input_width: shape.min_input_width,
        max_input_width: shape.max_input_width,
        min_output_width: shape.min_output_width,
        max_output_width: shape.max_output_width,
        weighted_avg_input_width: shape.weighted_avg_input_width,
        weighted_avg_output_width: shape.weighted_avg_output_width,
    }
}

fn tensor_type_bytes(bytes: model_artifact::gguf::GgufTensorTypeByteProfile) -> TensorTypeBytes {
    TensorTypeBytes {
        f32_bytes: bytes.f32_bytes,
        f16_bytes: bytes.f16_bytes,
        bf16_bytes: bytes.bf16_bytes,
        q4_0_bytes: bytes.q4_0_bytes,
        q4_k_bytes: bytes.q4_k_bytes,
        q5_k_bytes: bytes.q5_k_bytes,
        q6_k_bytes: bytes.q6_k_bytes,
        q8_0_bytes: bytes.q8_0_bytes,
        iq_bytes: bytes.iq_bytes,
        other_quantized_bytes: bytes.other_quantized_bytes,
        unknown_bytes: bytes.unknown_bytes,
    }
}

fn weight_coverage(
    compact: &model_artifact::gguf::GgufCompactMeta,
    architecture_class: ModelArchitectureClass,
    tensor_profile: &model_artifact::gguf::GgufTensorByteProfile,
) -> WeightCoverage {
    if tensor_profile.tensor_count == 0 {
        return WeightCoverage::MetadataOnly;
    }
    if !matches!(
        architecture_class,
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::SparseMoeTransformer
    ) {
        return WeightCoverage::Full;
    }
    let expected_layers = compact.layer_count;
    let present_layers = tensor_profile.distinct_block_count;
    if expected_layers == 0 {
        return WeightCoverage::Unknown;
    }
    if present_layers == expected_layers {
        WeightCoverage::Full
    } else if present_layers > 0 {
        WeightCoverage::PartialTransformer {
            present_layers,
            expected_layers,
        }
    } else {
        WeightCoverage::MetadataOnly
    }
}

fn capability_evidence(
    compact: &model_artifact::gguf::GgufCompactMeta,
    fit: &model_artifact::gguf::GgufFitMeta,
) -> Vec<CapabilityEvidence> {
    let mut evidence = Vec::new();
    if fit.has_chat_template() {
        evidence.push(CapabilityEvidence::ChatTemplatePresent);
    }

    let chat_template = fit.chat_template_text().to_ascii_lowercase();
    if chat_template.contains("system") {
        evidence.push(CapabilityEvidence::SystemRoleInChatTemplate);
    }
    if chat_template.contains("tool_calls")
        || chat_template.contains("tool call")
        || chat_template.contains("tools")
        || chat_template.contains("function")
    {
        evidence.push(CapabilityEvidence::ToolUseTemplateMarkers);
    }
    if fit.has_fill_in_middle_tokens() {
        evidence.push(CapabilityEvidence::FillInMiddleTokensPresent);
    }
    for tag in &fit.general_tags {
        let tag = tag.trim();
        if !tag.is_empty() {
            evidence.push(CapabilityEvidence::ExplicitGeneralTag(tag.to_string()));
        }
    }
    if compact.context_length > 0 {
        evidence.push(CapabilityEvidence::NativeContextAtLeast(
            compact.context_length,
        ));
    }
    if is_embedding_metadata(fit) {
        evidence.push(CapabilityEvidence::EmbeddingModel);
    }
    if is_classifier_metadata(fit) {
        evidence.push(CapabilityEvidence::ClassifierOrReranker);
    }
    if fit.clip_projector_type.is_some()
        || fit.clip_has_vision_encoder == Some(true)
        || fit.clip_has_audio_encoder == Some(true)
    {
        evidence.push(CapabilityEvidence::MultimodalProjector);
    }
    evidence
}

fn architecture_class(
    compact: &model_artifact::gguf::GgufCompactMeta,
    fit: &model_artifact::gguf::GgufFitMeta,
    evidence: &[CapabilityEvidence],
) -> ModelArchitectureClass {
    if evidence.contains(&CapabilityEvidence::MultimodalProjector) {
        return ModelArchitectureClass::MultimodalProjector;
    }
    if evidence.contains(&CapabilityEvidence::ClassifierOrReranker) {
        return ModelArchitectureClass::RerankerOrClassifier;
    }
    if evidence.contains(&CapabilityEvidence::EmbeddingModel) {
        return ModelArchitectureClass::Embedding;
    }
    if compact.expert_count > 0 {
        return ModelArchitectureClass::SparseMoeTransformer;
    }
    let arch = compact.architecture.to_ascii_lowercase();
    if is_recurrent_architecture(&arch) {
        return ModelArchitectureClass::RecurrentOrStateSpace;
    }
    if compact.layer_count > 0 && compact.embedding_size > 0 {
        return ModelArchitectureClass::DenseTransformer;
    }
    if is_embedding_metadata(fit) {
        return ModelArchitectureClass::Embedding;
    }
    ModelArchitectureClass::Unknown
}

fn is_recurrent_architecture(arch: &str) -> bool {
    arch.contains("rwkv")
        || arch.contains("mamba")
        || arch.contains("jamba")
        || arch.contains("recurrent")
        || arch.contains("ssm")
        || arch.contains("falcon_h1")
}

fn is_embedding_metadata(fit: &model_artifact::gguf::GgufFitMeta) -> bool {
    fit.pooling_type.is_some()
        || fit
            .general_type
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("embedding"))
}

fn is_classifier_metadata(fit: &model_artifact::gguf::GgufFitMeta) -> bool {
    !fit.classifier_output_labels.is_empty()
        || fit.general_type.as_deref().is_some_and(|value| {
            value.eq_ignore_ascii_case("classifier") || value.eq_ignore_ascii_case("reranker")
        })
}

fn non_empty(value: String) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn non_zero(value: u32) -> Option<u32> {
    (value > 0).then_some(value)
}

fn non_default_f32(value: f32) -> Option<f32> {
    (value > 0.0).then_some(value)
}

fn parameter_count_from_size_label(value: Option<&str>) -> Option<u64> {
    let value = value?.trim();
    let lower = value.to_ascii_lowercase();
    let suffix = lower.chars().last()?;
    let number = lower.strip_suffix(suffix)?.parse::<f64>().ok()?;
    match suffix {
        'b' => Some((number * 1_000_000_000.0) as u64),
        'm' => Some((number * 1_000_000.0) as u64),
        _ => None,
    }
}

fn evidence_sort_key(evidence: &CapabilityEvidence) -> String {
    format!("{evidence:?}")
}
