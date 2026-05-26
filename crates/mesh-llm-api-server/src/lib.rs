#![forbid(unsafe_code)]

mod discover;
mod node;

pub use discover::{
    AutoConnectResult, AutoNodeResult, create_auto_client, create_auto_node, discover_public_meshes,
};
pub use mesh_llm_api_client::events;
pub use mesh_llm_api_client::{
    ChatMessage, ChatRequest, ClientBuilder, ClientConfig, InviteToken, MAX_RECONNECT_ATTEMPTS,
    MeshApiError, MeshClient, Model, OwnerKeypair, PublicMesh, PublicMeshQuery, RequestId,
    ResponsesRequest, Status,
};
pub use mesh_llm_node::serving::ServingController;

/// Run the full mesh-llm runtime in-process — the same code path the
/// `mesh-llm` binary runs. Only available with the `host-runtime` feature.
///
/// This is the SDK entry point for embedders who want their Rust app to
/// act exactly like running `mesh-llm serve` or `mesh-llm client` —
/// with auto-discovery, election, tunnel manager, OpenAI HTTP proxy,
/// management console, and local model serving (when configured) —
/// without spawning the binary as a subprocess.
///
/// See [`mesh_llm_host_runtime::host_node::run_serve`] for the full
/// documentation.
#[cfg(feature = "host-runtime")]
pub use mesh_llm_host_runtime::host_node::{run_serve, MeshServeSpec};
pub use node::{
    CapabilityLevel, CleanupPolicy, CleanupResult, DeleteModelOptions, DeleteModelResult,
    DevicePolicy, DownloadId, DownloadOptions, DownloadedModel, InstalledModel, LoadModelOptions,
    MeshEvents, MeshInference, MeshModels, MeshNode, MeshNodeBuilder, MeshNodeConfig, MeshQuicBind,
    MeshRole, MeshServing, MeshStatusApi, ModelCacheStatus, ModelCapabilities, ModelDetails,
    ModelKind, ModelSearchQuery, ModelSource, ModelSummary, PrunePolicy, PruneResult, ServedModel,
    ServingModelState, ServingStatus, UnloadModelOptions, UnloadTarget,
};
