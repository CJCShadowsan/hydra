mod gpu_benchmark;
mod hf_cache;
mod profile;
mod scoring;
mod types;
mod workload;

pub use gpu_benchmark::{
    GpuBenchmarkAcceleratorFacts, GpuBenchmarkHardwareInput, GpuBenchmarkOutput,
    hardware_profile_from_gpu_benchmark,
};
pub use hf_cache::{HfCacheModelProfile, profile_hf_cache};
pub use profile::profile_gguf_path;
pub use scoring::{
    estimate_kv_cache_bytes, estimate_runtime_memory_bytes, rank_models, score_model,
};
pub use types::{
    AcceleratorKind, AcceleratorProfile, BackendEfficiencyConfig, BackendKind, CapabilityEvidence,
    CapabilityRequirements, CpuProfile, DecodeOverheadConfig, EstimateConfidence, FitStatus,
    HardwareProfile, InteractionProfile, KvCacheKind, KvCacheType, MeasurementSource,
    MemoryProfile, ModelArchitectureClass, ModelProfile, ModelRecommendation, ModelSource,
    Requirement, RopeProfile, ScoreWeights, SelectionConfig, SplitCandidateEstimate,
    TokenizerProfile, WeightCoverage, WorkloadPreferences, WorkloadProfile, WorkloadTask,
};

#[cfg(test)]
mod tests;
