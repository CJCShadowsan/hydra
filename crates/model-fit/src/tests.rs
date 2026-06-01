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
            decode_effective_bandwidth_bytes_per_sec: Some(320_000_000_000),
            decode_fixed_overhead_ms: Some(1.25),
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
    let attention_bytes = bytes / 3;
    let feed_forward_bytes = bytes / 2;
    let output_bytes = bytes / 12;
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
            attention_bytes,
            feed_forward_bytes,
            expert_feed_forward_bytes: 0,
            embedding_bytes: bytes / 12,
            output_bytes,
            normalization_bytes: bytes / 100,
            other_bytes: bytes
                .saturating_sub(bytes / 3)
                .saturating_sub(bytes / 2)
                .saturating_sub(bytes / 12)
                .saturating_sub(bytes / 12)
                .saturating_sub(bytes / 100),
        },
        tensor_matmul: TensorMatmulProfile {
            base_bytes: attention_bytes + feed_forward_bytes + output_bytes,
            expert_bytes: 0,
            base_flops_per_token: 0,
            expert_flops_per_token: 0,
            base_type_bytes: TensorTypeBytes {
                q4_k_bytes: attention_bytes + feed_forward_bytes + output_bytes,
                ..TensorTypeBytes::default()
            },
            expert_type_bytes: TensorTypeBytes::default(),
            attention: synthetic_matmul_group(attention_bytes, layers * 4, hidden, hidden),
            feed_forward: synthetic_matmul_group(
                feed_forward_bytes,
                layers * 3,
                hidden,
                hidden * 4,
            ),
            output: synthetic_matmul_group(output_bytes, 1, hidden, hidden),
            expert_feed_forward: TensorMatmulGroupProfile::default(),
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

fn synthetic_matmul_group(
    bytes: u64,
    logical_matrix_count: u32,
    input_width: u32,
    output_width: u32,
) -> TensorMatmulGroupProfile {
    TensorMatmulGroupProfile {
        bytes,
        type_bytes: TensorTypeBytes {
            q4_k_bytes: bytes,
            ..TensorTypeBytes::default()
        },
        shape: MatmulShapeProfile {
            tensor_count: u64::from(logical_matrix_count),
            logical_matrix_count: u64::from(logical_matrix_count),
            total_elements: u64::from(logical_matrix_count)
                .saturating_mul(u64::from(input_width))
                .saturating_mul(u64::from(output_width)),
            min_input_width: u64::from(input_width.min(output_width)),
            max_input_width: u64::from(input_width.max(output_width)),
            min_output_width: u64::from(input_width.min(output_width)),
            max_output_width: u64::from(input_width.max(output_width)),
            weighted_avg_input_width: u64::from(input_width),
            weighted_avg_output_width: u64::from(output_width),
        },
        ..TensorMatmulGroupProfile::default()
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
    moe.tensor_matmul.base_bytes = 12 * GIB;
    moe.tensor_matmul.expert_bytes = 48 * GIB;
    moe.tensor_matmul.attention.bytes = 8 * GIB;
    moe.tensor_matmul.feed_forward.bytes = 4 * GIB;
    moe.tensor_matmul.expert_feed_forward.bytes = 48 * GIB;
    moe.tensor_matmul.base_type_bytes = TensorTypeBytes {
        q4_k_bytes: 12 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.expert_type_bytes = TensorTypeBytes {
        q4_k_bytes: 48 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.attention.type_bytes = TensorTypeBytes {
        q4_k_bytes: 8 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.feed_forward.type_bytes = TensorTypeBytes {
        q4_k_bytes: 4 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.expert_feed_forward.type_bytes = TensorTypeBytes {
        q4_k_bytes: 48 * GIB,
        ..TensorTypeBytes::default()
    };
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
fn measured_moe_dispatch_overhead_uses_submission_cost() {
    let mut low_overhead = m1_ultra();
    low_overhead.accelerators[0].decode_fixed_overhead_ms = Some(0.002);
    let mut high_overhead = m1_ultra();
    high_overhead.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut moe = dense_model("measured-moe", 4 * GIB, 16, 2048, 4096);
    moe.architecture_class = ModelArchitectureClass::SparseMoeTransformer;
    moe.expert_count = Some(64);
    moe.expert_used_count = Some(8);
    moe.tensor_group_bytes.expert_feed_forward_bytes = 3 * GIB;
    moe.tensor_matmul.expert_bytes = 3 * GIB;
    moe.tensor_matmul.expert_feed_forward.bytes = 3 * GIB;
    moe.tensor_matmul.expert_feed_forward.type_bytes = TensorTypeBytes {
        q4_k_bytes: 3 * GIB,
        ..TensorTypeBytes::default()
    };

    let low_rec = score_model(&low_overhead, &moe, &config);
    let high_rec = score_model(&high_overhead, &moe, &config);

    assert!(
        low_rec.estimated_decode_tokens_per_sec.unwrap()
            > high_rec.estimated_decode_tokens_per_sec.unwrap()
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
fn decode_estimate_uses_measured_graph_overhead_for_deeper_shapes() {
    let mut hardware = m1_ultra();
    hardware.memory.available_system_bytes = None;
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let shallow = dense_model("shallow", 3 * GIB, 16, 4096, 32_768);
    let deep = dense_model("deep", 3 * GIB, 40, 4096, 32_768);

    let shallow_rec = score_model(&hardware, &shallow, &config);
    let deep_rec = score_model(&hardware, &deep, &config);

    assert!(deep_rec.estimated_decode_tokens_per_sec < shallow_rec.estimated_decode_tokens_per_sec);
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
fn budget_selection_prefers_faster_measured_gpu_over_cpu_headroom() {
    let hardware = HardwareProfile {
        memory: MemoryProfile {
            total_system_bytes: Some(32 * GIB),
            available_system_bytes: Some(30 * GIB),
            total_unified_bytes: None,
            available_unified_bytes: None,
        },
        accelerators: vec![AcceleratorProfile {
            name: Some("Measured CUDA GPU".into()),
            kind: AcceleratorKind::DiscreteGpu,
            backend: BackendKind::Cuda,
            total_memory_bytes: Some(16 * GIB),
            available_memory_bytes: Some(15 * GIB),
            memory_bandwidth_bytes_per_sec: Some(900_000_000_000),
            decode_effective_bandwidth_bytes_per_sec: Some(850_000_000_000),
            decode_fixed_overhead_ms: Some(0.002),
            bandwidth_source: MeasurementSource::Measured,
            benchmark_noise_pct: Some(0.5),
            bandwidth_efficiency_pct: Some(90.0),
            compute_tflops_fp32: None,
            compute_tflops_fp16: Some(50.0),
            unified_memory: false,
        }],
        cpu: CpuProfile {
            physical_cores: Some(8),
            logical_cores: Some(16),
            memory_bandwidth_bytes_per_sec: None,
        },
    };
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("fits-gpu-and-cpu", 7 * GIB, 28, 3584, 32_768);

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.selected_backend, BackendKind::Cuda);
    assert!(
        !rec.warnings
            .iter()
            .any(|warning| warning.contains("memory bandwidth is missing"))
    );
    assert!(rec.estimated_decode_tokens_per_sec.unwrap() > 100.0);
}

#[test]
fn q8_decode_uses_ggml_type_kernel_traffic() {
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
    q8.tensor_matmul.base_bytes *= 2;
    q8.tensor_matmul.base_type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.base_type_bytes.q8_0_bytes = q8.tensor_matmul.base_bytes;
    q8.tensor_matmul.attention.bytes *= 2;
    q8.tensor_matmul.feed_forward.bytes *= 2;
    q8.tensor_matmul.output.bytes *= 2;
    q8.tensor_matmul.attention.type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.feed_forward.type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.output.type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.attention.type_bytes.q8_0_bytes = q8.tensor_matmul.attention.bytes;
    q8.tensor_matmul.feed_forward.type_bytes.q8_0_bytes = q8.tensor_matmul.feed_forward.bytes;
    q8.tensor_matmul.output.type_bytes.q8_0_bytes = q8.tensor_matmul.output.bytes;

    let q4_rec = score_model(&hardware, &q4, &config);
    let q8_rec = score_model(&hardware, &q8, &config);
    let q4_active = q4_rec.estimated_active_decode_bytes_per_token.unwrap();
    let q8_active = q8_rec.estimated_active_decode_bytes_per_token.unwrap();

    assert!(q8_active > q4_active);
    assert!(q8_active > q4_active * 16 / 10);
    assert!(
        q8_rec.estimated_decode_tokens_per_sec.unwrap()
            < q4_rec.estimated_decode_tokens_per_sec.unwrap()
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
            decode_effective_gbps: Some(295.0),
            decode_fixed_overhead_ms: Some(1.25),
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
