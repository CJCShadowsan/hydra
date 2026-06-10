//! MLX backend integration.
//!
//! Makes `mesh-mlx` usable as a local inference backend, mirroring the Skippy
//! HTTP handle shape: load a model and serve the OpenAI API on an ephemeral
//! local port; expose `port()` + `shutdown()` so the host can route OpenAI
//! traffic to it like any other local backend.
//!
//! MLX is **local-only** and **Apple-Silicon-only**. The selection helpers
//! ([`mlx_supported`], [`plan_parallelism`]) always compile; the actual serving
//! requires the `mlx-backend` feature (which links the native MLX Metal engine).
//! Without that feature, [`MlxModelHandle::load`] returns an error so callers
//! degrade gracefully to the Skippy/llama.cpp lane.

use anyhow::Result;

// Re-export the mesh-facing MLX decision types so callers can plan placement
// through this module. (Some are only referenced under the `mlx-backend`
// feature or in tests, but they are part of this module's public surface.)
#[allow(unused_imports)]
pub use mesh_mlx::{
    LatencySample, MlxBackendKind, MlxOrchestrator, NodeEndpoint, ParallelismMode, ParallelismPlan,
    TransportPlan, mlx_supported,
};

use crate::mesh::{self, PeerInfo};

/// The default TCP base port for MLX's ring backend. Each rank listens on
/// `base + connection_index`; mesh assigns the same base to every node so the
/// rank-ordered hostfile is consistent.
pub const MLX_RING_BASE_PORT: u16 = 5680;

/// A discovered MLX group: the rank-ordered endpoints, the per-link latency
/// samples, and the chosen parallelism + transport plan. `local_rank` is this
/// node's index in the ring.
#[derive(Debug, Clone)]
pub struct MlxGroupPlan {
    pub local_rank: usize,
    pub endpoints: Vec<NodeEndpoint>,
    pub samples: Vec<LatencySample>,
    pub parallelism: ParallelismPlan,
    pub transport: TransportPlan,
}

impl MlxGroupPlan {
    /// Render the rank-ordered JSON hostfile MLX's `load_nodes()` consumes.
    pub fn hostfile(&self) -> String {
        self.transport.render_hostfile()
    }

    /// Whether this is a real multi-node group (vs. a single local node).
    pub fn is_distributed(&self) -> bool {
        self.endpoints.len() > 1
    }
}

/// Plan tensor-vs-pipeline parallelism + transport for a candidate MLX group
/// from measured inter-node latency. Pure decision logic mesh owns; usable
/// without the native engine. Exercised by unit tests and used by
/// [`plan_group_from_peers`].
pub fn plan_parallelism(
    nodes: Vec<NodeEndpoint>,
    samples: &[LatencySample],
) -> (ParallelismPlan, TransportPlan) {
    MlxOrchestrator::default().plan(nodes, samples)
}

/// Whether a discovered peer is eligible to join an MLX group: Apple Silicon
/// (so it can run the Metal engine) and directly routable (has at least one
/// non-loopback IP address — MLX opens its own TCP/RDMA sockets and cannot use
/// mesh's relay/QUIC transport).
fn peer_is_mlx_eligible(peer: &PeerInfo) -> bool {
    let apple_silicon = peer.is_soc == Some(true)
        || peer
            .gpu_name
            .as_deref()
            .map(|g| g.contains("Apple"))
            .unwrap_or(false);
    apple_silicon && peer_direct_ips(peer).next().is_some()
}

/// This peer's directly-routable IPs (loopback filtered out).
fn peer_direct_ips(peer: &PeerInfo) -> impl Iterator<Item = std::net::SocketAddr> + '_ {
    peer.addr
        .ip_addrs()
        .copied()
        .filter(|sa| !sa.ip().is_loopback())
}

/// Build a [`NodeEndpoint`] for a peer, attaching MLX's ring port to each IP.
/// `rdma` is left empty here; Thunderbolt RDMA device maps are discovered
/// separately (via the JACCL setup) and merged in when available.
fn peer_endpoint(ssh: String, ips: impl Iterator<Item = std::net::IpAddr>) -> NodeEndpoint {
    NodeEndpoint {
        ssh,
        ips: ips.map(|ip| format!("{ip}:{MLX_RING_BASE_PORT}")).collect(),
        rdma: Vec::new(),
    }
}

/// Form an MLX group plan from mesh's discovered peers.
///
/// This is the discovery → MLX handoff: mesh *finds and selects* the peers (here
/// we filter its gossiped peer list to Apple-Silicon, directly-routable nodes
/// and read its measured RTT), and produces the rank-ordered hostfile + plan
/// that MLX then uses to open its own TCP ring (or JACCL/RDMA). MLX traffic does
/// **not** flow through mesh — mesh only supplies the addresses.
///
/// Returns `None` when there are no eligible peers (→ run single-node).
///
/// Rank order: the local node is rank 0, then eligible peers sorted by endpoint
/// id for a stable, identical ordering on every node (so all nodes build the
/// same ring).
pub async fn plan_group_from_peers(node: &mesh::Node) -> Option<MlxGroupPlan> {
    if !MlxModelHandle::available() {
        return None;
    }

    let peers = node.peers().await;
    let mut eligible: Vec<&PeerInfo> = peers.iter().filter(|p| peer_is_mlx_eligible(p)).collect();
    if eligible.is_empty() {
        return None;
    }
    // Stable, deterministic ordering shared by all nodes.
    eligible.sort_by_key(|p| p.id.to_string());

    // Rank 0 is the local node; its loopback endpoint is replaced by its real
    // LAN address by the launcher, so advertise the ring port on 0.0.0.0 here.
    let local_id = node.id().to_string();
    let mut endpoints = Vec::with_capacity(eligible.len() + 1);
    endpoints.push(peer_endpoint(
        local_id,
        std::iter::once(std::net::IpAddr::from([0, 0, 0, 0])),
    ));

    let mut samples = Vec::new();
    for (i, peer) in eligible.iter().enumerate() {
        let rank = i + 1;
        let ips: Vec<std::net::IpAddr> = peer_direct_ips(peer).map(|sa| sa.ip()).collect();
        endpoints.push(peer_endpoint(peer.id.to_string(), ips.into_iter()));
        // RTT from mesh's measurements feeds the tensor-vs-pipeline decision.
        if let Some(rtt_ms) = peer.current_direct_rtt_ms() {
            samples.push(LatencySample::new(
                0,
                rank,
                std::time::Duration::from_millis(rtt_ms as u64),
            ));
        }
    }

    let (parallelism, transport) = plan_parallelism(endpoints.clone(), &samples);
    Some(MlxGroupPlan {
        local_rank: 0,
        endpoints,
        samples,
        parallelism,
        transport,
    })
}

/// Distributed setup for an MLX backend node: the rank-ordered hostfile, this
/// node's rank, the MLX backend, and the chosen parallelism mode.
///
/// These fields are consumed by `load_distributed` only under the `mlx-backend`
/// feature (which links the engine); without it they're carried but unread.
#[cfg_attr(not(feature = "mlx-backend"), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct MlxDistributedSetup {
    pub hostfile_json: String,
    pub rank: usize,
    pub backend: MlxBackendKind,
    pub mode: ParallelismMode,
}

/// Options for loading an MLX model as a local backend.
#[derive(Debug, Clone)]
pub struct MlxModelLoadOptions {
    /// Hugging Face repo id (safetensors; bf16/fp16 or quantized 4-bit).
    pub model_id: String,
    /// Address to bind the OpenAI server to. Use `127.0.0.1:0` for an ephemeral
    /// port (the local-backend convention).
    pub bind_addr: std::net::SocketAddr,
    /// When set, join an MLX distributed group (multi-node). When `None`, serve
    /// single-node.
    pub distributed: Option<MlxDistributedSetup>,
}

impl MlxModelLoadOptions {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            bind_addr: "127.0.0.1:0".parse().expect("static addr parses"),
            distributed: None,
        }
    }

    /// Attach a distributed setup derived from an [`MlxGroupPlan`].
    pub fn with_group(mut self, plan: &MlxGroupPlan) -> Self {
        if plan.is_distributed() {
            self.distributed = Some(MlxDistributedSetup {
                hostfile_json: plan.hostfile(),
                rank: plan.local_rank,
                backend: plan.transport.backend,
                mode: plan.parallelism.mode,
            });
        }
        self
    }
}

/// A running MLX backend: an OpenAI server on a local port.
pub struct MlxModelHandle {
    #[cfg(feature = "mlx-backend")]
    server: mesh_mlx::ServerHandle,
    port: u16,
    model_id: String,
}

impl MlxModelHandle {
    /// The local port the OpenAI server is bound to.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The served model id.
    #[allow(dead_code)]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The base URL mesh routes OpenAI requests to.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    /// Whether this host can run the MLX backend (Apple Silicon + the
    /// `mlx-backend` feature compiled in).
    pub fn available() -> bool {
        cfg!(feature = "mlx-backend") && mlx_supported()
    }

    /// Load a model and start serving. Requires the `mlx-backend` feature.
    #[cfg(feature = "mlx-backend")]
    pub async fn load(options: MlxModelLoadOptions) -> Result<Self> {
        use mesh_mlx::{Engine, ModelRef, ServerState, spawn};

        if !mlx_supported() {
            anyhow::bail!("MLX backend requires Apple Silicon (macOS aarch64)");
        }

        // Distributed: join the MLX group mesh discovered. Loading shards the
        // model per rank (pipeline → this stage's layers; tensor → sliced
        // projections) and brings up MLX's own TCP ring / JACCL to the peers.
        if let Some(dist) = options.distributed.clone() {
            return Self::load_distributed(options, dist).await;
        }

        let engine = Engine::load_single(&ModelRef::new(&options.model_id))
            .await
            .map_err(|e| anyhow::anyhow!("load MLX model {}: {e}", options.model_id))?;
        let state = ServerState::new(engine, options.model_id.clone());
        let server = spawn(state, options.bind_addr)
            .await
            .map_err(|e| anyhow::anyhow!("start MLX OpenAI server: {e}"))?;
        let port = server.port();
        tracing::info!(
            model = %options.model_id,
            port,
            "MLX backend serving OpenAI API"
        );
        Ok(Self {
            server,
            port,
            model_id: options.model_id,
        })
    }

    /// Bring up a distributed MLX node. Joins the group (writes the hostfile,
    /// sets `MLX_HOSTFILE`/`MLX_RANK`, inits the ring/JACCL backend) and loads
    /// the sharded model for this rank.
    ///
    /// Rank 0 serves the OpenAI API; the chat path drives the group in
    /// lock-step so the other ranks participate in every generation. (The
    /// multi-rank worker loop and rank-0 fan-out are the piece that needs a
    /// live 2+ node rig to validate end to end.)
    #[cfg(feature = "mlx-backend")]
    async fn load_distributed(
        options: MlxModelLoadOptions,
        dist: MlxDistributedSetup,
    ) -> Result<Self> {
        use mesh_mlx::{DistributedEngine, JoinParams, ModelRef, ServerState, spawn};

        let join = JoinParams {
            hostfile_json: dist.hostfile_json,
            rank: dist.rank,
            backend: dist.backend.to_backend(),
            mode: dist.mode,
        };
        let dengine = DistributedEngine::join(&ModelRef::new(&options.model_id), join)
            .await
            .map_err(|e| anyhow::anyhow!("join MLX group for {}: {e}", options.model_id))?;

        // The distributed engine owns the live group; the server's chat path
        // drives the group in lock-step. Every rank's process holds the engine
        // so collectives stay synchronised.
        let state = ServerState::distributed(dengine, options.model_id.clone());
        let server = spawn(state, options.bind_addr)
            .await
            .map_err(|e| anyhow::anyhow!("start MLX OpenAI server: {e}"))?;
        let port = server.port();
        tracing::info!(
            model = %options.model_id,
            rank = dist.rank,
            mode = ?dist.mode,
            port,
            "MLX distributed backend serving OpenAI API"
        );
        Ok(Self {
            server,
            port,
            model_id: options.model_id,
        })
    }

    /// Without the `mlx-backend` feature, the engine isn't linked; report it so
    /// callers fall back to another lane.
    #[cfg(not(feature = "mlx-backend"))]
    pub async fn load(options: MlxModelLoadOptions) -> Result<Self> {
        anyhow::bail!(
            "MLX backend not compiled in (model {} on {}); build with --features mlx-backend on Apple Silicon",
            options.model_id,
            options.bind_addr
        )
    }

    /// Stop the OpenAI server.
    #[cfg(feature = "mlx-backend")]
    pub async fn shutdown(self) -> Result<()> {
        self.server.shutdown().await;
        Ok(())
    }

    /// No-op shutdown when the engine isn't linked.
    #[cfg(not(feature = "mlx-backend"))]
    pub async fn shutdown(self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn load_options_default_to_ephemeral_port() {
        let o = MlxModelLoadOptions::new("mlx-community/Qwen2.5-0.5B-Instruct-4bit");
        assert_eq!(o.bind_addr.port(), 0);
        assert!(o.bind_addr.ip().is_loopback());
    }

    #[test]
    fn availability_requires_feature_and_apple_silicon() {
        // Without the mlx-backend feature this is always false; with it, it
        // tracks the host arch. Either way it must not panic.
        let _ = MlxModelHandle::available();
    }

    #[test]
    fn planner_routes_low_latency_to_tensor() {
        let nodes = vec![
            NodeEndpoint {
                ssh: "mac-0".into(),
                ips: vec!["10.0.0.1".into()],
                rdma: vec![],
            },
            NodeEndpoint {
                ssh: "mac-1".into(),
                ips: vec!["10.0.0.2".into()],
                rdma: vec![],
            },
        ];
        let (plan, transport) = plan_parallelism(
            nodes,
            &[LatencySample::new(0, 1, Duration::from_micros(700))],
        );
        assert_eq!(plan.mode, ParallelismMode::Tensor);
        assert_eq!(transport.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn peer_endpoint_attaches_ring_port_to_ips() {
        let ep = peer_endpoint(
            "peer-x".into(),
            [
                std::net::IpAddr::from([192, 168, 1, 10]),
                std::net::IpAddr::from([192, 168, 1, 11]),
            ]
            .into_iter(),
        );
        assert_eq!(ep.ssh, "peer-x");
        assert_eq!(
            ep.ips,
            vec![
                format!("192.168.1.10:{MLX_RING_BASE_PORT}"),
                format!("192.168.1.11:{MLX_RING_BASE_PORT}"),
            ]
        );
        assert!(ep.rdma.is_empty());
    }

    #[test]
    fn group_plan_single_node_is_not_distributed() {
        let endpoints = vec![peer_endpoint(
            "self".into(),
            std::iter::once(std::net::IpAddr::from([0, 0, 0, 0])),
        )];
        let (parallelism, transport) = plan_parallelism(endpoints.clone(), &[]);
        let plan = MlxGroupPlan {
            local_rank: 0,
            endpoints,
            samples: vec![],
            parallelism,
            transport,
        };
        assert!(!plan.is_distributed());
        // with_group on a single-node plan attaches no distributed setup.
        let opts = MlxModelLoadOptions::new("m").with_group(&plan);
        assert!(opts.distributed.is_none());
    }

    #[test]
    fn group_plan_multi_node_builds_hostfile_and_setup() {
        let endpoints = vec![
            peer_endpoint(
                "self".into(),
                std::iter::once(std::net::IpAddr::from([0, 0, 0, 0])),
            ),
            peer_endpoint(
                "peer-1".into(),
                std::iter::once(std::net::IpAddr::from([10, 0, 0, 2])),
            ),
        ];
        let (parallelism, transport) = plan_parallelism(
            endpoints.clone(),
            &[LatencySample::new(0, 1, Duration::from_millis(8))],
        );
        let plan = MlxGroupPlan {
            local_rank: 0,
            endpoints,
            samples: vec![LatencySample::new(0, 1, Duration::from_millis(8))],
            parallelism,
            transport,
        };
        assert!(plan.is_distributed());
        // High RTT → pipeline.
        assert_eq!(plan.parallelism.mode, ParallelismMode::Pipeline);
        let hf = plan.hostfile();
        assert!(hf.contains("10.0.0.2"));

        let opts = MlxModelLoadOptions::new("m").with_group(&plan);
        let dist = opts.distributed.expect("multi-node attaches setup");
        assert_eq!(dist.rank, 0);
        assert_eq!(dist.mode, ParallelismMode::Pipeline);
        assert!(dist.hostfile_json.contains("10.0.0.2"));
    }
}
