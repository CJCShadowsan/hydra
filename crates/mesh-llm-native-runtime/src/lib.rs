//! Shared native runtime manifest, resolution, and cache policy.

mod cache;
mod flavor;
mod host;
mod manifest;
mod resolver;

pub use cache::{
    CachePrunePlan, InstalledNativeRuntime, NativeRuntimeCache, NativeRuntimeCacheRoot,
    NativeRuntimePruneMode, native_runtime_cache_root,
};
pub use flavor::{NativeRuntimeFlavor, NativeRuntimeFlavorParseError};
pub use host::{HostGpuProfile, HostRuntimeProfile};
pub use manifest::{
    NativeRuntimeArtifact, NativeRuntimeManifest, NativeRuntimeReleaseManifest,
    NativeRuntimeRequirement,
};
pub use resolver::{
    CandidateEvaluation, CandidateRejection, NativeRuntimeResolution, NativeRuntimeResolver,
    NativeRuntimeSource, RuntimeSelection, select_native_runtime,
};
