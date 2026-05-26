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
/// # Example
///
/// ```no_run
/// use mesh_llm_api_server::{run_serve, MeshServeSpec};
///
/// # async fn run() -> anyhow::Result<()> {
/// run_serve(MeshServeSpec {
///     // Same flags `mesh-llm serve` / `mesh-llm client` accept.
///     client: true,                       // false (default) = serve role
///     auto: true,                         // == --auto
///     relays: vec!["https://public.example/".into()],
///     port: Some(9337),                   // OpenAI HTTP proxy port
///     console_port: Some(3131),           // management API / web console
///     headless: true,                     // skip embedded web UI
///     max_vram_gb: Some(0.0),             // client-only, no VRAM advert
///     ..MeshServeSpec::default()
/// })
/// .await?;
/// # Ok(())
/// # }
/// ```
///
/// (Gated iroh-relay support — per-relay bearer tokens via
/// `--relay-auth URL=TOKEN` and a `relay_auths` field on
/// `MeshServeSpec` — lives on the separate gated-relay PR.)
///
/// The future blocks until the runtime exits. The runtime is not
/// currently `Send`-clean; if you need concurrent work, run on a
/// `tokio::task::LocalSet` rather than `tokio::spawn`.
///
/// For finer-grained control — composing pieces without running the
/// whole orchestration — see [`MeshNodeBuilder`] instead.
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
