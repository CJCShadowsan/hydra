use crate::*;

const GIB: u64 = 1024 * 1024 * 1024;

fn m1_ultra() -> HardwareProfile {
    HardwareProfile {
        memory: MemoryProfile {
            total_system_bytes: Some(128 * GIB),
            available_system_bytes: Some(110 * GIB),
            total_unified_bytes: Some(128 * GIB),
            available_unified_bytes: Some(110 * GIB),
        },
        accelerators: vec![AcceleratorProfile {
            name: Some("Apple M1 Ultra".into()),
            kind: AcceleratorKind::IntegratedGpu,
            backend: BackendKind::Metal,
            total_memory_bytes: Some(128 * GIB),
            available_memory_bytes: Some(110 * GIB),
            memory_bandwidth_bytes_per_sec: Some(800_000_000_000),
            bandwidth_source: MeasurementSource::Measured,
            benchmark_noise_pct: Some(1.0),
            bandwidth_efficiency_pct: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory: true,
        }],
        cpu: CpuProfile {
            physical_cores: Some(20),
            logical_cores: Some(20),
            memory_bandwidth_bytes_per_sec: Some(200_000_000_000),
        },
    }
}

fn dense_model(id: &str, bytes: u64, layers: u32, hidden: u32, context: u32) -> ModelProfile {
    ModelProfile {
        source: ModelSource {
            id: id.into(),
            path: None,
            metadata_name: None,
        },
        architecture: Some("llama".into()),
        architecture_class: ModelArchitectureClass::DenseTransformer,
        weight_coverage: WeightCoverage::Full,
        file_size_bytes: bytes,
        tensor_bytes: Some(bytes),
        base_resident_bytes: Some(bytes),
        expert_tensor_bytes: Some(0),
        tensor_group_bytes: TensorGroupBytes {
            attention_bytes: bytes / 3,
            feed_forward_bytes: bytes / 2,
            expert_feed_forward_bytes: 0,
            embedding_bytes: bytes / 12,
            output_bytes: bytes / 12,
            normalization_bytes: bytes / 100,
            other_bytes: bytes
                .saturating_sub(bytes / 3)
                .saturating_sub(bytes / 2)
                .saturating_sub(bytes / 12)
                .saturating_sub(bytes / 12)
                .saturating_sub(bytes / 100),
        },
        parameter_count: None,
        quantization: Some("Q4_K_M".into()),
        layer_count: Some(layers),
        hidden_size: Some(hidden),
        ffn_size: Some(hidden * 4),
        attention_heads: Some(32),
        kv_heads: Some(8),
        key_length: Some(hidden / 32),
        value_length: Some(hidden / 8),
        context_length: Some(context),
        expert_count: None,
        expert_used_count: None,
        rope: RopeProfile::default(),
        tokenizer: TokenizerProfile {
            model: Some("gpt2".into()),
            chat_template_available: true,
        },
        capability_evidence: vec![
            CapabilityEvidence::ChatTemplatePresent,
            CapabilityEvidence::SystemRoleInChatTemplate,
            CapabilityEvidence::NativeContextAtLeast(context),
        ],
    }
}

#[test]
fn dense_14b_beats_dense_70b_for_latency_sensitive_chat() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let models = vec![
        dense_model("dense-14b", 9 * GIB, 40, 5120, 32_768),
        dense_model("dense-70b", 42 * GIB, 80, 8192, 32_768),
    ];

    let ranked = rank_models(&hardware, &models, &config);

    assert_eq!(ranked[0].source.id, "dense-14b");
    assert!(ranked[0].estimated_decode_tokens_per_sec > ranked[1].estimated_decode_tokens_per_sec);
}

#[test]
fn coding_agent_prefers_explicit_fim_and_tool_evidence() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::coding_agent(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut coding = dense_model("opaque-id-a", 18 * GIB, 48, 6144, 65_536);
    coding.capability_evidence.extend([
        CapabilityEvidence::ToolUseTemplateMarkers,
        CapabilityEvidence::FillInMiddleTokensPresent,
    ]);
    let plain = dense_model("opaque-id-b", 18 * GIB, 48, 6144, 65_536);

    let ranked = rank_models(&hardware, &[plain, coding], &config);

    assert_eq!(ranked[0].source.id, "opaque-id-a");
    assert!(ranked[0].workload_score > ranked[1].workload_score);
}

#[test]
fn embedding_workload_accepts_embedding_model_and_rejects_chat_model() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::embedding(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let chat = dense_model("chat", 5 * GIB, 32, 4096, 8_192);
    let mut embedding = dense_model("embedding", GIB, 12, 768, 512);
    embedding.architecture_class = ModelArchitectureClass::Embedding;
    embedding.capability_evidence = vec![CapabilityEvidence::EmbeddingModel];

    let ranked = rank_models(&hardware, &[chat, embedding], &config);

    assert_eq!(ranked[0].source.id, "embedding");
    assert_eq!(ranked[1].fit_status, FitStatus::Rejected);
}

#[test]
fn chat_workload_rejects_embedding_model() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut embedding = dense_model("embedding", GIB, 12, 768, 512);
    embedding.architecture_class = ModelArchitectureClass::Embedding;
    embedding.capability_evidence = vec![CapabilityEvidence::EmbeddingModel];

    let rec = score_model(&hardware, &embedding, &config);

    assert_eq!(rec.fit_status, FitStatus::Rejected);
}

#[test]
fn moe_decode_uses_active_expert_bytes() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut moe = dense_model("moe", 60 * GIB, 48, 6144, 32_768);
    moe.architecture_class = ModelArchitectureClass::SparseMoeTransformer;
    moe.base_resident_bytes = Some(12 * GIB);
    moe.expert_tensor_bytes = Some(48 * GIB);
    moe.tensor_group_bytes.attention_bytes = 8 * GIB;
    moe.tensor_group_bytes.feed_forward_bytes = 4 * GIB;
    moe.tensor_group_bytes.expert_feed_forward_bytes = 48 * GIB;
    moe.expert_count = Some(16);
    moe.expert_used_count = Some(2);

    let rec = score_model(&hardware, &moe, &config);

    assert!(rec.estimated_active_decode_bytes_per_token.unwrap() < 30 * GIB);
    assert!(
        rec.warnings
            .iter()
            .any(|warning| warning.contains("active experts"))
    );
}

#[test]
fn filename_like_identifier_does_not_create_coding_suitability() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::coding_agent(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model(
        "qwen-coder-tool-instruct-name-only.gguf",
        6 * GIB,
        32,
        4096,
        32_768,
    );
    let rec = score_model(&hardware, &model, &config);

    assert!(rec.workload_score < 0.75);
    assert!(
        !rec.reasons
            .iter()
            .any(|reason| reason.contains("fill-in-middle") || reason.contains("tool-call"))
    );
}

#[test]
fn oversized_dense_model_becomes_split_candidate() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::summarization(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("dense-180b", 120 * GIB, 96, 12_288, 32_768);

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.fit_status, FitStatus::SplitCandidate);
    assert!(rec.split_candidate.is_some());
}

#[test]
fn partial_transformer_gguf_is_not_ranked_as_standalone_model() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut model = dense_model("stage-artifact", 4 * GIB, 36, 4096, 32_768);
    model.weight_coverage = WeightCoverage::PartialTransformer {
        present_layers: 18,
        expected_layers: 36,
    };

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.fit_status, FitStatus::Rejected);
    assert!(rec.reasons.iter().any(|reason| reason.contains("partial")));
}

#[test]
fn decode_estimate_reports_uncertainty_range() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("dense", 4 * GIB, 32, 4096, 32_768);

    let rec = score_model(&hardware, &model, &config);
    let point = rec
        .estimated_decode_tokens_per_sec
        .expect("decode estimate should exist");
    let range = rec
        .estimated_decode_tokens_per_sec_range
        .expect("decode range should exist");

    assert!(range.lower < point);
    assert!(range.upper > point);
}

#[test]
fn prefill_estimate_reports_first_token_latency_range() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("dense", 4 * GIB, 32, 4096, 32_768);

    let rec = score_model(&hardware, &model, &config);
    let point = rec
        .estimated_first_token_ms
        .expect("first-token estimate should exist");
    let range = rec
        .estimated_first_token_ms_range
        .expect("first-token range should exist");

    assert!(point > 0.0);
    assert!(range.lower_ms < point);
    assert!(range.upper_ms > point);
    assert!(rec.estimated_prefill_tokens_per_sec.unwrap() > 0.0);
}

#[test]
fn decode_estimate_penalizes_small_hidden_width_efficiency() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let narrow = dense_model("narrow", 3 * GIB, 32, 2048, 32_768);
    let wide = dense_model("wide", 3 * GIB, 32, 4096, 32_768);

    let narrow_rec = score_model(&hardware, &narrow, &config);
    let wide_rec = score_model(&hardware, &wide, &config);

    assert!(narrow_rec.estimated_decode_tokens_per_sec < wide_rec.estimated_decode_tokens_per_sec);
}

#[test]
fn measured_gpu_bandwidth_uses_backend_neutral_efficiency() {
    let metal = m1_ultra();
    let mut cuda = metal.clone();
    cuda.accelerators[0].backend = BackendKind::Cuda;
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("portable", 4 * GIB, 32, 4096, 32_768);

    let metal_tps = score_model(&metal, &model, &config)
        .estimated_decode_tokens_per_sec
        .expect("measured metal estimate should exist");
    let cuda_tps = score_model(&cuda, &model, &config)
        .estimated_decode_tokens_per_sec
        .expect("measured cuda estimate should exist");

    assert!((metal_tps - cuda_tps).abs() < 0.001);
}

#[test]
fn q8_decode_traffic_uses_quantization_shape_not_resident_bytes() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let q4 = dense_model("q4", 8 * GIB, 32, 4096, 32_768);
    let mut q8 = q4.clone();
    q8.source.id = "q8".into();
    q8.quantization = Some("Q8_0".into());
    q8.file_size_bytes = 16 * GIB;
    q8.tensor_bytes = Some(16 * GIB);
    q8.base_resident_bytes = Some(16 * GIB);
    q8.tensor_group_bytes.attention_bytes *= 2;
    q8.tensor_group_bytes.feed_forward_bytes *= 2;
    q8.tensor_group_bytes.output_bytes *= 2;
    q8.tensor_group_bytes.embedding_bytes *= 2;
    q8.tensor_group_bytes.normalization_bytes *= 2;
    q8.tensor_group_bytes.other_bytes *= 2;

    let q4_rec = score_model(&hardware, &q4, &config);
    let q8_rec = score_model(&hardware, &q8, &config);
    let q4_active = q4_rec.estimated_active_decode_bytes_per_token.unwrap();
    let q8_active = q8_rec.estimated_active_decode_bytes_per_token.unwrap();

    assert!(q8_active < q4_active * 2);
    assert!(
        q8_rec.estimated_decode_tokens_per_sec.unwrap()
            > q4_rec.estimated_decode_tokens_per_sec.unwrap() * 0.70
    );
}

#[test]
fn hardware_profile_uses_mesh_gpu_benchmark_output_as_measured_bandwidth() {
    let hardware = hardware_profile_from_gpu_benchmark(GpuBenchmarkHardwareInput {
        memory: MemoryProfile {
            total_system_bytes: Some(128 * GIB),
            available_system_bytes: Some(110 * GIB),
            total_unified_bytes: Some(128 * GIB),
            available_unified_bytes: Some(110 * GIB),
        },
        cpu: CpuProfile::default(),
        default_backend: BackendKind::Metal,
        accelerators: vec![GpuBenchmarkAcceleratorFacts {
            name: Some("Apple M1 Ultra".into()),
            kind: AcceleratorKind::IntegratedGpu,
            backend: Some(BackendKind::Metal),
            total_memory_bytes: Some(128 * GIB),
            available_memory_bytes: Some(110 * GIB),
            unified_memory: true,
        }],
        benchmark_outputs: vec![GpuBenchmarkOutput {
            device: "Apple M1 Ultra".into(),
            buffer_mb: 1024,
            runs: 7,
            p50_gbps: 710.0,
            p90_gbps: 737.0,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            noise_pct: 1.0,
            runtime_s: 0.25,
            rated_gbps: None,
            rated_estimated: None,
            efficiency_pct: None,
            bus_width_bits: None,
            mem_clock_mhz: None,
            gcn_arch: None,
            hbm: None,
        }],
    })
    .expect("benchmark output should build hardware profile");

    let accelerator = &hardware.accelerators[0];
    assert_eq!(accelerator.bandwidth_source, MeasurementSource::Measured);
    assert_eq!(
        accelerator.memory_bandwidth_bytes_per_sec,
        Some(737_000_000_000)
    );
    assert_eq!(accelerator.benchmark_noise_pct, Some(1.0));
    assert_eq!(accelerator.available_memory_bytes, Some(110 * GIB));
}
