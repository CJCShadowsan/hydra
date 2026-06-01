use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub memory: MemoryProfile,
    pub accelerators: Vec<AcceleratorProfile>,
    pub cpu: CpuProfile,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub total_system_bytes: Option<u64>,
    pub available_system_bytes: Option<u64>,
    pub total_unified_bytes: Option<u64>,
    pub available_unified_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AcceleratorProfile {
    pub name: Option<String>,
    pub kind: AcceleratorKind,
    pub backend: BackendKind,
    pub total_memory_bytes: Option<u64>,
    pub available_memory_bytes: Option<u64>,
    /// Bandwidth used by the fit model, expressed as raw device/system bytes
    /// per second before model-fit's decode-efficiency factor is applied.
    ///
    /// Prefer this value from `mesh-llm gpus benchmark` instead of marketing
    /// bandwidth. The benchmark exercises the actual host/backend path that
    /// local inference will use, which makes it a better predictor of decode
    /// throughput while still remaining model-independent.
    pub memory_bandwidth_bytes_per_sec: Option<u64>,
    /// Effective bandwidth measured by a decode-shaped benchmark.
    ///
    /// This differs from raw memory bandwidth. Raw bandwidth measures a large,
    /// sustained streaming kernel. Decode runs many smaller kernels and pays
    /// scheduling/synchronization costs between them. `model-fit` uses this
    /// field when available so measured hardware profiles do not need hidden
    /// backend-specific efficiency constants.
    #[serde(default)]
    pub decode_effective_bandwidth_bytes_per_sec: Option<u64>,
    /// Fixed per-token dispatch overhead measured by the GPU benchmark.
    ///
    /// This is a runtime fact, not a model fact. If the benchmark did not
    /// measure it, model-fit leaves the measured-GPU fixed overhead out of the
    /// equation and reports that the decode calibration is incomplete.
    #[serde(default)]
    pub decode_fixed_overhead_ms: Option<f32>,
    /// Lower-bound cost of issuing the first decode-shaped command after a
    /// prefill-shaped matmul.
    ///
    /// The benchmark measures this without a GGUF model, so it captures backend
    /// queue/command transition cost rather than llama.cpp graph/session work.
    /// First-token validation keeps the remaining residual visible instead of
    /// hiding it inside this field.
    #[serde(default)]
    pub post_prefill_decode_overhead_ms: Option<f32>,
    pub bandwidth_source: MeasurementSource,
    /// Run-to-run spread from the GPU bandwidth benchmark.
    ///
    /// Model-fit uses this as a small pessimism adjustment for measured
    /// bandwidth. A noisy bandwidth benchmark usually means the machine is not
    /// delivering a perfectly stable memory path, so the decoder should not
    /// treat the measured p90 as a guaranteed sustained rate.
    #[serde(default)]
    pub benchmark_noise_pct: Option<f32>,
    /// Optional diagnostic output from the GPU benchmark.
    ///
    /// These fields are intentionally stored on the hardware profile even when
    /// the first-pass heuristic does not consume all of them. They make
    /// validation JSON self-contained, and they give later fit passes room to
    /// distinguish memory-bound decode from compute-heavy prefill without
    /// changing the profile schema again.
    #[serde(default)]
    pub bandwidth_efficiency_pct: Option<f32>,
    #[serde(default)]
    pub compute_tflops_fp32: Option<f32>,
    #[serde(default)]
    pub compute_tflops_fp16: Option<f32>,
    /// Optional prefill-shaped FP16 matrix-multiply throughput from the GPU
    /// benchmark.
    ///
    /// This is separate from scalar/vector FP16 throughput. Prompt prefill in
    /// llama.cpp is dominated by batched `GGML_OP_MUL_MAT` work, so a generic
    /// dense matmul probe is a better hardware fact for the prefill roofline
    /// than decode bandwidth or scalar FMA throughput. The value is still
    /// model-independent: it must come from `mesh-llm gpus benchmark`, not from
    /// benchmarking the GGUF being fitted.
    #[serde(default)]
    pub prefill_matmul_tflops_fp16: Option<f32>,
    /// Optional ubatch-shaped FP16 matrix-multiply throughput from the GPU
    /// benchmark.
    ///
    /// llama.cpp prompt ingestion is not one square GEMM. It splits prompt
    /// tokens into ubatches, commonly 512 tokens, and feeds those skinny
    /// activation batches through the model weight matrices. This measured fact
    /// lets first-token/prefill estimates use that source shape instead of
    /// assuming peak square-GEMM throughput.
    #[serde(default)]
    pub prefill_ubatch_matmul_tflops_fp16: Option<f32>,
    /// Optional MoE-prefill-shaped FP16 matrix-multiply throughput from the GPU
    /// benchmark.
    ///
    /// Sparse MoE prompt processing does not behave like one large dense GEMM:
    /// llama.cpp routes expert work through `GGML_OP_MUL_MAT_ID`, with expert
    /// selection, id mapping, and aggregation around many expert matmuls. A
    /// separate probe lets the fit model consume that hardware fact without
    /// hard-coding backend names.
    #[serde(default)]
    pub prefill_moe_matmul_tflops_fp16: Option<f32>,
    pub unified_memory: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CpuProfile {
    pub physical_cores: Option<u32>,
    pub logical_cores: Option<u32>,
    pub memory_bandwidth_bytes_per_sec: Option<u64>,
    /// Optional CPU FP16 throughput measured by a hardware benchmark.
    ///
    /// This is deliberately optional. Many CPUs either do not expose a native
    /// FP16 matrix path or do not use it for GGML quantized inference. When a
    /// future CPU benchmark can measure the same semantic facts as the GPU
    /// benchmark, model-fit can consume them without adding backend-name
    /// branches.
    #[serde(default)]
    pub compute_tflops_fp16: Option<f32>,
    #[serde(default)]
    pub post_prefill_decode_overhead_ms: Option<f32>,
    #[serde(default)]
    pub prefill_matmul_tflops_fp16: Option<f32>,
    #[serde(default)]
    pub prefill_ubatch_matmul_tflops_fp16: Option<f32>,
    #[serde(default)]
    pub prefill_moe_matmul_tflops_fp16: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum AcceleratorKind {
    IntegratedGpu,
    DiscreteGpu,
    Cpu,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendKind {
    Metal,
    Cuda,
    Rocm,
    Vulkan,
    Cpu,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum MeasurementSource {
    Measured,
    Estimated,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelSource {
    pub id: String,
    pub path: Option<PathBuf>,
    pub metadata_name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelProfile {
    pub source: ModelSource,
    pub architecture: Option<String>,
    pub architecture_class: ModelArchitectureClass,
    pub weight_coverage: WeightCoverage,
    pub file_size_bytes: u64,
    pub tensor_bytes: Option<u64>,
    pub base_resident_bytes: Option<u64>,
    pub expert_tensor_bytes: Option<u64>,
    pub tensor_group_bytes: TensorGroupBytes,
    pub tensor_matmul: TensorMatmulProfile,
    pub parameter_count: Option<u64>,
    pub quantization: Option<String>,
    pub layer_count: Option<u32>,
    pub hidden_size: Option<u32>,
    pub ffn_size: Option<u32>,
    pub attention_heads: Option<u32>,
    pub kv_heads: Option<u32>,
    pub key_length: Option<u32>,
    pub value_length: Option<u32>,
    pub context_length: Option<u32>,
    pub expert_count: Option<u32>,
    pub expert_used_count: Option<u32>,
    pub rope: RopeProfile,
    pub tokenizer: TokenizerProfile,
    pub capability_evidence: Vec<CapabilityEvidence>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorGroupBytes {
    pub attention_bytes: u64,
    pub feed_forward_bytes: u64,
    pub expert_feed_forward_bytes: u64,
    pub embedding_bytes: u64,
    pub output_bytes: u64,
    pub normalization_bytes: u64,
    pub other_bytes: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorMatmulProfile {
    pub base_bytes: u64,
    pub expert_bytes: u64,
    pub base_flops_per_token: u64,
    pub expert_flops_per_token: u64,
    pub base_type_bytes: TensorTypeBytes,
    pub expert_type_bytes: TensorTypeBytes,
    pub attention: TensorMatmulGroupProfile,
    pub feed_forward: TensorMatmulGroupProfile,
    pub expert_feed_forward: TensorMatmulGroupProfile,
    pub output: TensorMatmulGroupProfile,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorMatmulGroupProfile {
    pub bytes: u64,
    pub flops_per_token: u64,
    pub type_bytes: TensorTypeBytes,
    pub shape: MatmulShapeProfile,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct MatmulShapeProfile {
    pub tensor_count: u64,
    pub logical_matrix_count: u64,
    pub total_elements: u64,
    pub min_input_width: u64,
    pub max_input_width: u64,
    pub min_output_width: u64,
    pub max_output_width: u64,
    pub weighted_avg_input_width: u64,
    pub weighted_avg_output_width: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorTypeBytes {
    pub f32_bytes: u64,
    pub f16_bytes: u64,
    pub bf16_bytes: u64,
    pub q4_0_bytes: u64,
    pub q4_k_bytes: u64,
    pub q5_k_bytes: u64,
    pub q6_k_bytes: u64,
    pub q8_0_bytes: u64,
    pub iq_bytes: u64,
    pub other_quantized_bytes: u64,
    pub unknown_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum WeightCoverage {
    Full,
    PartialTransformer {
        present_layers: u32,
        expected_layers: u32,
    },
    MetadataOnly,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RopeProfile {
    pub scale: Option<f32>,
    pub freq_base: Option<f32>,
    pub scaling_type: Option<String>,
    pub scaling_factor: Option<f32>,
    pub original_context_length: Option<u32>,
    pub finetuned: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenizerProfile {
    pub model: Option<String>,
    pub chat_template_available: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum ModelArchitectureClass {
    DenseTransformer,
    SparseMoeTransformer,
    RecurrentOrStateSpace,
    Embedding,
    RerankerOrClassifier,
    MultimodalProjector,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CapabilityEvidence {
    ChatTemplatePresent,
    SystemRoleInChatTemplate,
    ToolUseTemplateMarkers,
    FillInMiddleTokensPresent,
    ExplicitGeneralTag(String),
    NativeContextAtLeast(u32),
    EmbeddingModel,
    ClassifierOrReranker,
    MultimodalProjector,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionConfig {
    pub safety_margin: f32,
    pub kv_cache_type: KvCacheType,
    pub backend_efficiency: BackendEfficiencyConfig,
    pub decode_overhead: DecodeOverheadConfig,
    pub workload: WorkloadProfile,
    pub weights: ScoreWeights,
    pub kv_read_scale: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KvCacheType {
    pub k: KvCacheKind,
    pub v: KvCacheKind,
}

impl Default for KvCacheType {
    fn default() -> Self {
        Self {
            k: KvCacheKind::F16,
            v: KvCacheKind::F16,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum KvCacheKind {
    #[default]
    F16,
    Q8_0,
    Q4_0,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BackendEfficiencyConfig {
    pub metal: f32,
    pub cuda: f32,
    pub rocm: f32,
    pub vulkan: f32,
    pub cpu: f32,
    pub unknown: f32,
}

impl Default for BackendEfficiencyConfig {
    fn default() -> Self {
        // Default to a neutral multiplier. Hardware profiles produced from
        // `mesh-llm gpus benchmark` should carry decode-shaped bandwidth
        // directly, so model-fit does not need backend-specific "Metal is X%"
        // or "CUDA is Y%" assumptions to predict measured machines. Callers may
        // still override these fields for hypothetical/manual profiles, but the
        // crate default keeps missing benchmark data visible instead of hiding
        // it behind magic backend constants.
        Self {
            metal: 1.0,
            cuda: 1.0,
            rocm: 1.0,
            vulkan: 1.0,
            cpu: 1.0,
            unknown: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DecodeOverheadConfig {
    pub metal_fixed_ms: f32,
    pub cuda_fixed_ms: f32,
    pub rocm_fixed_ms: f32,
    pub vulkan_fixed_ms: f32,
    pub cpu_fixed_ms: f32,
    pub unknown_fixed_ms: f32,
    pub moe_dispatch_ms_per_layer: f32,
}

impl Default for DecodeOverheadConfig {
    fn default() -> Self {
        Self {
            // Fixed decode overhead is highly runtime/backend/hardware shaped.
            // Measured GPU profiles get this from `mesh-llm gpus benchmark`.
            // For unmeasured profiles the honest default is no fixed overhead
            // plus a warning from scoring that decode calibration is missing,
            // not a backend-specific constant chosen from local observations.
            metal_fixed_ms: 0.0,
            cuda_fixed_ms: 0.0,
            rocm_fixed_ms: 0.0,
            vulkan_fixed_ms: 0.0,
            cpu_fixed_ms: 0.0,
            unknown_fixed_ms: 0.0,
            moe_dispatch_ms_per_layer: 0.11,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ScoreWeights {
    pub memory: f32,
    pub context: f32,
    pub decode: f32,
    pub prefill: f32,
    pub workload: f32,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            memory: 0.25,
            context: 0.20,
            decode: 0.25,
            prefill: 0.10,
            workload: 0.20,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub task: WorkloadTask,
    pub interaction: InteractionProfile,
    pub requirements: CapabilityRequirements,
    pub preferences: WorkloadPreferences,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum WorkloadTask {
    Chat,
    Coding,
    Summarization,
    Extraction,
    ToolCalling,
    Embedding,
    Reranking,
    Classification,
    MultimodalUnderstanding,
    #[default]
    GeneralGeneration,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InteractionProfile {
    pub expected_prompt_tokens: Option<u32>,
    pub expected_output_tokens: Option<u32>,
    pub latency_sensitive: bool,
    pub multi_turn: bool,
    pub agent_loop: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CapabilityRequirements {
    pub chat_template: Requirement,
    pub system_messages: Requirement,
    pub tool_calling: Requirement,
    pub fill_in_middle: Requirement,
    pub embeddings: Requirement,
    pub reranking: Requirement,
    pub vision: Requirement,
    pub audio: Requirement,
    pub min_context_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum Requirement {
    Required,
    Preferred,
    #[default]
    Neutral,
    Penalize,
    Reject,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkloadPreferences {
    pub prefer_quality_over_speed: f32,
    pub prefer_context_over_speed: f32,
    pub minimum_decode_tps: Option<f32>,
    pub preferred_decode_tps: Option<f32>,
}

impl Default for WorkloadPreferences {
    fn default() -> Self {
        Self {
            prefer_quality_over_speed: 0.5,
            prefer_context_over_speed: 0.5,
            minimum_decode_tps: Some(4.0),
            preferred_decode_tps: Some(16.0),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub source: ModelSource,
    pub selected_backend: BackendKind,
    pub selected_accelerator: Option<String>,
    pub architecture_class: ModelArchitectureClass,
    pub estimate_confidence: EstimateConfidence,
    pub fit_status: FitStatus,
    pub total_score: f32,
    pub memory_score: f32,
    pub context_score: f32,
    pub decode_score: f32,
    pub prefill_score: f32,
    pub workload_score: f32,
    pub estimated_runtime_memory_bytes: u64,
    pub estimated_kv_cache_bytes: u64,
    pub estimated_active_decode_bytes_per_token: Option<u64>,
    pub estimated_decode_tokens_per_sec: Option<f32>,
    pub estimated_decode_tokens_per_sec_range: Option<DecodeEstimateRange>,
    pub estimated_prefill_tokens_per_sec: Option<f32>,
    pub estimated_first_token_prefill_ms: Option<f32>,
    pub estimated_first_token_decode_ms: Option<f32>,
    pub estimated_first_token_overhead_ms: Option<f32>,
    pub estimated_first_token_ms: Option<f32>,
    pub estimated_first_token_ms_range: Option<FirstTokenEstimateRange>,
    pub split_candidate: Option<SplitCandidateEstimate>,
    pub capability_evidence: Vec<CapabilityEvidence>,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DecodeEstimateRange {
    pub lower: f32,
    pub upper: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FirstTokenEstimateRange {
    pub lower_ms: f32,
    pub upper_ms: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FitStatus {
    FitsLocal,
    FitsWithWarning,
    SplitCandidate,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EstimateConfidence {
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitCandidateEstimate {
    pub estimated_stages: u32,
    pub per_stage_memory_budget_bytes: u64,
    pub warning: String,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        let workload = WorkloadProfile::general_generation();
        Self {
            safety_margin: 0.20,
            kv_cache_type: KvCacheType::default(),
            backend_efficiency: BackendEfficiencyConfig::default(),
            decode_overhead: DecodeOverheadConfig::default(),
            weights: workload.default_weights(),
            workload,
            kv_read_scale: 0.25,
        }
    }
}
