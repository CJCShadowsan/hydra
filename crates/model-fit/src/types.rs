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
    pub memory_bandwidth_bytes_per_sec: Option<u64>,
    pub bandwidth_source: MeasurementSource,
    pub unified_memory: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CpuProfile {
    pub physical_cores: Option<u32>,
    pub logical_cores: Option<u32>,
    pub memory_bandwidth_bytes_per_sec: Option<u64>,
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
        Self {
            metal: 0.45,
            cuda: 0.55,
            rocm: 0.45,
            vulkan: 0.35,
            cpu: 0.20,
            unknown: 0.30,
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
            metal_fixed_ms: 4.0,
            cuda_fixed_ms: 1.5,
            rocm_fixed_ms: 2.0,
            vulkan_fixed_ms: 3.0,
            cpu_fixed_ms: 0.5,
            unknown_fixed_ms: 3.0,
            moe_dispatch_ms_per_layer: 0.16,
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
    pub split_candidate: Option<SplitCandidateEstimate>,
    pub capability_evidence: Vec<CapabilityEvidence>,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
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
