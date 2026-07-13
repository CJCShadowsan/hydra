#![forbid(unsafe_code)]

pub mod network_cost;
pub mod placement;
pub mod scheduler;
pub mod vast;

pub use network_cost::{
    NetworkAdvisoryHint, NetworkCostCollector, NetworkCostConfig, NetworkCostObservation,
    NetworkCostSnapshot, NetworkCostStatusSnapshot, TargetNetworkKey,
};
pub use placement::{
    ArtifactKind, ArtifactPlacementProvider, PlacementCacheSnapshot, PlacementCompatibility,
    PlacementManager, PlacementManifest, PlacementOperationSnapshot, PlacementPrefetchRequest,
    PlacementProviderConfig, PosixNamespaceProvider, S3NamespaceProvider,
};
pub use scheduler::{
    SchedulerConfig, SchedulerDecision, SchedulerMode, SchedulerStatusSnapshot,
    SchedulerTargetCandidate, TargetKind,
};
pub use vast::{
    VastProviderLocation, VastTriggerConfig, VastTriggerMode, VastTriggerRequest,
    VastTriggerResponse, send_vast_trigger,
};
