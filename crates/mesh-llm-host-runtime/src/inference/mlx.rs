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
    ParallelismPreference, TransportPlan, TransportPreference, detect_rdma_devices, mlx_supported,
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
    /// Render the rank-ordered ring hostfile MLX's `load_nodes()` consumes
    /// (`[["ip:port", ...], ...]`).
    pub fn hostfile(&self) -> String {
        self.transport.render_hostfile()
    }

    /// Render the JACCL NxN devices-matrix JSON (the `MLX_IBV_DEVICES` file),
    /// when the backend is JACCL and every node advertises a complete RDMA row.
    pub fn jaccl_devices(&self) -> Option<String> {
        if self.transport.backend != MlxBackendKind::Jaccl {
            return None;
        }
        self.transport.render_jaccl_devices()
    }

    /// The rank-0 coordinator address JACCL bootstraps over: the first IP of
    /// the rank-0 endpoint. `None` if rank 0 has no advertised IP.
    pub fn coordinator(&self) -> Option<String> {
        self.endpoints.first().and_then(|e| e.ips.first().cloned())
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
#[cfg_attr(not(test), allow(dead_code))]
pub fn plan_parallelism(
    nodes: Vec<NodeEndpoint>,
    samples: &[LatencySample],
) -> (ParallelismPlan, TransportPlan) {
    MlxOrchestrator::default().plan(nodes, samples)
}

/// Whether a discovered peer is eligible to join an MLX group: Apple Silicon
/// (so it can run the Metal engine), directly routable (has at least one
/// non-loopback IP address — MLX opens its own TCP/RDMA sockets and cannot use
/// mesh's relay/QUIC transport), **and** sharing interest in the same model.
///
/// The model check is what keeps the group correct: an MLX group is a fixed
/// formation where every member must be running the *same* model. A peer that
/// is Apple-Silicon but serving a different model (or a GGUF on Skippy) must
/// not be pulled into the group, or the leader would block at `Group::init`
/// waiting for a rank that never joins. We treat a peer as a group member when
/// it is serving, has requested, or has explicit interest in `model_id` — i.e.
/// the operator launched the same model on it (the `mlx.launch` model).
fn peer_is_mlx_eligible(peer: &PeerInfo, model_id: &str) -> bool {
    let apple_silicon = peer.is_soc == Some(true)
        || peer
            .gpu_name
            .as_deref()
            .map(|g| g.contains("Apple"))
            .unwrap_or(false);
    apple_silicon && peer_direct_ips(peer).next().is_some() && peer_shares_model(peer, model_id)
}

/// Whether `peer` is running (or wants to run) the same model, by any of the
/// gossiped model-interest signals.
fn peer_shares_model(peer: &PeerInfo, model_id: &str) -> bool {
    peer.serving_models
        .iter()
        .chain(peer.requested_models.iter())
        .chain(peer.explicit_model_interests.iter())
        .chain(peer.models.iter())
        .any(|m| model_ids_match(m, model_id))
}

/// Whether two model references denote the same model, compared
/// case-insensitively against the full id and the trailing repo/name component
/// so `org/Repo` and `Repo` agree (mesh stores HF snapshots under repo paths,
/// so a peer may advertise either form).
fn model_ids_match(a: &str, b: &str) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    if a == b {
        return true;
    }
    let tail = |s: &str| s.rsplit('/').next().unwrap_or(s).to_string();
    tail(&a) == tail(&b)
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
/// Rank order: **all** group members — the local node and the eligible peers —
/// are sorted together by node id, so every node computes the identical rank
/// order and the rings agree. `local_rank` is this node's position in that
/// shared order.
/// Whether the operator opted into distributed MLX (`MESH_LLM_MLX_DISTRIBUTED`).
///
/// Off by default: forming a fixed tensor/pipeline group is an explicit
/// operator decision (you launch the same model on each Apple-Silicon node),
/// not something to trigger from demand. Accepts `1`/`true`/`yes`/`on`.
fn distributed_mlx_enabled() -> bool {
    matches!(
        std::env::var("MESH_LLM_MLX_DISTRIBUTED")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// The rendezvous window: how long a node waits for its expected MLX peers to
/// appear via gossip before forming the group. Override the expected peer count
/// with `MESH_LLM_MLX_GROUP_SIZE` (total nodes including self); otherwise the
/// node forms a group with whatever eligible peers it can see after the wait.
fn mlx_rendezvous() -> (std::time::Duration, Option<usize>) {
    let secs = std::env::var("MESH_LLM_MLX_RENDEZVOUS_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30);
    let expected = std::env::var("MESH_LLM_MLX_GROUP_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n >= 2);
    (std::time::Duration::from_secs(secs), expected)
}

/// Wait up to the rendezvous window for MLX-eligible peers to appear.
///
/// If `MESH_LLM_MLX_GROUP_SIZE` is set, returns as soon as that many total
/// nodes (self + eligible peers) are visible, or when the window elapses.
/// Without it, polls until at least one eligible peer is seen (then returns)
/// or the window elapses. Returns the eligible peers found (empty → single-node).
async fn wait_for_mlx_peers(node: &mesh::Node, model_id: &str) -> Vec<PeerInfo> {
    let (window, expected) = mlx_rendezvous();
    let deadline = std::time::Instant::now() + window;
    let poll = std::time::Duration::from_millis(500);
    loop {
        let eligible: Vec<PeerInfo> = node
            .peers()
            .await
            .into_iter()
            .filter(|p| peer_is_mlx_eligible(p, model_id))
            .collect();
        let have = eligible.len() + 1; // + self
        match expected {
            Some(n) if have >= n => return eligible,
            None if !eligible.is_empty() => return eligible,
            _ => {}
        }
        if std::time::Instant::now() >= deadline {
            if !eligible.is_empty() {
                tracing::warn!(
                    found = eligible.len() + 1,
                    expected = ?expected,
                    "MLX rendezvous window elapsed; forming group with peers seen so far"
                );
            }
            return eligible;
        }
        tokio::time::sleep(poll).await;
    }
}

pub async fn plan_group_from_peers(node: &mesh::Node, model_id: &str) -> Option<MlxGroupPlan> {
    if !MlxModelHandle::available() {
        return None;
    }

    // Distributed MLX is opt-in. An MLX group is a *fixed* tensor/pipeline
    // formation decided at init — it is not a demand-driven, dynamically-joined
    // pool. So a node only forms a group when the operator explicitly asked for
    // it (and is expected to start the same model on each Apple-Silicon node,
    // the `mlx.launch` model). Default off → always serve single-node, no
    // surprise multi-node formation.
    if !distributed_mlx_enabled() {
        return None;
    }

    // Rendezvous window: the operator launches the same model on each node, but
    // gossip discovery is asynchronous, so a node may reach here before it has
    // seen its peers. Wait briefly for the expected peers (sharing this model)
    // to appear so all nodes converge on the same group rather than racing to
    // serve single-node.
    let eligible_owned = wait_for_mlx_peers(node, model_id).await;
    if eligible_owned.is_empty() {
        return None;
    }
    let eligible: Vec<&PeerInfo> = eligible_owned.iter().collect();

    // The MLX ring backend needs ONE hostfile, identical on every rank, with
    // each node's real routable address. So the local node must contribute the
    // same IP its peers see — not 0.0.0.0. If we have no routable IP, we cannot
    // be reached for a ring/JACCL group, so fall back to single-node.
    let local_ips: Vec<std::net::IpAddr> = node
        .self_direct_ips()
        .into_iter()
        .map(|sa| sa.ip())
        .collect();
    if local_ips.is_empty() {
        tracing::warn!(
            "MLX: no routable local IP to advertise for a distributed group; serving single-node"
        );
        return None;
    }

    // Stable, deterministic ordering shared by all nodes: the local node sorts
    // *with* the peers by id, so every member derives the same ring.
    let local_id = node.id().to_string();
    let (members, local_rank) = rank_order(
        local_id,
        eligible.iter().map(|p| (p.id.to_string(), Some(*p))),
    );

    let mut endpoints = Vec::with_capacity(members.len());
    let mut samples = Vec::new();
    for (rank, (id, peer)) in members.iter().enumerate() {
        match peer {
            // The local node advertises its real routable IP(s) — the same
            // address peers see — so the shared hostfile is consistent.
            None => endpoints.push(peer_endpoint(id.clone(), local_ips.iter().copied())),
            Some(p) => {
                let ips: Vec<std::net::IpAddr> = peer_direct_ips(p).map(|sa| sa.ip()).collect();
                endpoints.push(peer_endpoint(id.clone(), ips.into_iter()));
                // RTT from mesh's measurements feeds the tensor-vs-pipeline
                // decision (measured local → peer).
                if let Some(rtt_ms) = p.current_direct_rtt_ms() {
                    samples.push(LatencySample::new(
                        local_rank,
                        rank,
                        std::time::Duration::from_millis(rtt_ms as u64),
                    ));
                }
            }
        }
    }

    // Transport preference (MESH_LLM_MLX_TRANSPORT=auto|ring|jaccl) decides
    // whether we attempt JACCL (RDMA/Thunderbolt) or stay on the TCP ring.
    let pref = TransportPreference::from_env();
    apply_local_rdma_row(&mut endpoints, local_rank, pref);

    // Parallelism preference (MESH_LLM_MLX_PARALLELISM=auto|tensor|pipeline)
    // can force the mode — e.g. tensor over plain Ethernet — otherwise the
    // latency-aware planner decides from measured RTT.
    let parallelism = MlxOrchestrator::default().planner.plan_with_preference(
        endpoints.len(),
        &samples,
        ParallelismPreference::from_env(),
    );
    let transport = select_transport(parallelism.mode, &endpoints, pref);

    Some(MlxGroupPlan {
        local_rank,
        endpoints,
        samples,
        parallelism,
        transport,
    })
}

/// Sort the local node and the eligible peers into the shared rank order.
///
/// Every group member runs this with the same set of ids, so all members
/// derive the identical ring; the returned index is the local node's rank in
/// that order. Pure so the ordering invariant is unit-testable.
fn rank_order<'p>(
    local_id: String,
    peers: impl Iterator<Item = (String, Option<&'p PeerInfo>)>,
) -> (Vec<(String, Option<&'p PeerInfo>)>, usize) {
    let mut members: Vec<(String, Option<&'p PeerInfo>)> = vec![(local_id.clone(), None)];
    members.extend(peers);
    members.sort_by(|a, b| a.0.cmp(&b.0));
    let local_rank = members
        .iter()
        .position(|(id, _)| *id == local_id)
        .expect("local id is in members");
    (members, local_rank)
}

/// Populate the local node's RDMA device row from locally-detected devices
/// when JACCL is wanted (auto/jaccl).
///
/// We can detect *this* node's devices via `ibv_devices`; the peers' device
/// names must be carried by mesh gossip, which is not yet wired (`PeerInfo` has
/// no RDMA field). So this fills the local rank's row best-effort; a full JACCL
/// mesh engages once peer device maps are gossiped. Until then `Auto`
/// detects-but-falls-back (safe), and explicit `Jaccl` without devices warns.
fn apply_local_rdma_row(
    endpoints: &mut [NodeEndpoint],
    local_rank: usize,
    pref: TransportPreference,
) {
    if !matches!(pref, TransportPreference::Jaccl | TransportPreference::Auto) {
        return;
    }
    let local_rdma = detect_rdma_devices();
    if local_rdma.is_empty() {
        if pref == TransportPreference::Jaccl {
            tracing::warn!(
                "MESH_LLM_MLX_TRANSPORT=jaccl but no local RDMA devices detected \
                 (ibv_devices empty); JACCL requires macOS 26.2+, `rdma_ctl enable`, \
                 and a Thunderbolt-5 mesh. Falling back per planner."
            );
        }
        return;
    }
    let n = endpoints.len();
    let dev = local_rdma.first().cloned();
    // Diagonal (self) is null; reuse the first device for each peer link until
    // per-link mapping is gossiped.
    let row: Vec<Option<String>> = (0..n)
        .map(|j| if j == local_rank { None } else { dev.clone() })
        .collect();
    endpoints[local_rank].rdma = row;
    tracing::info!(devices = ?local_rdma, "MLX detected local RDMA devices for JACCL");
}

/// Pick the transport, falling back to ring (loudly) if explicit JACCL can't be
/// satisfied across the group.
fn select_transport(
    mode: ParallelismMode,
    endpoints: &[NodeEndpoint],
    pref: TransportPreference,
) -> TransportPlan {
    match TransportPlan::recommend_with(mode, endpoints.to_vec(), pref) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("MLX transport: {e}; falling back to TCP ring");
            TransportPlan::recommend_with(mode, endpoints.to_vec(), TransportPreference::Ring)
                .expect("ring never errors")
        }
    }
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
    /// JACCL NxN devices-matrix JSON + rank-0 coordinator `ip:port`. `None` for
    /// the ring backend.
    pub jaccl: Option<MlxJacclSetup>,
}

/// JACCL-specific distributed inputs (only present for the JACCL backend).
#[cfg_attr(not(feature = "mlx-backend"), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct MlxJacclSetup {
    pub devices_json: String,
    pub coordinator: String,
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
            // For JACCL, carry the NxN devices matrix + coordinator. If the
            // matrix can't be rendered (incomplete RDMA mesh) the setup falls
            // back to no JACCL inputs — `Group::init` would then fail loudly
            // rather than run on a half-configured fabric.
            let jaccl = match (plan.jaccl_devices(), plan.coordinator()) {
                (Some(devices_json), Some(coordinator)) => Some(MlxJacclSetup {
                    devices_json,
                    coordinator,
                }),
                _ => None,
            };
            self.distributed = Some(MlxDistributedSetup {
                hostfile_json: plan.hostfile(),
                rank: plan.local_rank,
                backend: plan.transport.backend,
                mode: plan.parallelism.mode,
                jaccl,
            });
        }
        self
    }
}

/// A running MLX backend.
///
/// On a single node or the distributed **leader** (rank 0) this owns the OpenAI
/// server. On a distributed **worker** (rank != 0) there is no server — the
/// node owns a [`mesh_mlx::WorkerHandle`] running the lock-step worker loop, and
/// mesh routes inference to the leader, not here.
pub struct MlxModelHandle {
    #[cfg(feature = "mlx-backend")]
    role: MlxNodeRole,
    port: u16,
    model_id: String,
}

/// The serving role of an MLX node within its (possibly single-node) group.
#[cfg(feature = "mlx-backend")]
enum MlxNodeRole {
    /// Single node or distributed leader: owns the OpenAI server. The optional
    /// `ServerState` clone is present for a distributed leader so shutdown can
    /// broadcast the worker-exit sentinel before stopping the server.
    Leader {
        server: mesh_mlx::ServerHandle,
        state: Option<mesh_mlx::ServerState>,
    },
    /// Distributed worker: owns the lock-step worker loop, no HTTP server.
    Worker(mesh_mlx::WorkerHandle),
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
            role: MlxNodeRole::Leader {
                server,
                state: None,
            },
            port,
            model_id: options.model_id,
        })
    }

    /// Bring up a distributed MLX node. Joins the group (writes the hostfile,
    /// sets `MLX_HOSTFILE`/`MLX_RANK`, inits the ring/JACCL backend) and loads
    /// the sharded model for this rank.
    ///
    /// **Leader (rank 0)** serves the OpenAI API; its chat path broadcasts each
    /// request to the workers and then drives the lock-step generation.
    ///
    /// **Workers (rank != 0)** do not serve OpenAI. They run the lock-step
    /// worker loop ([`mesh_mlx::WorkerHandle`]), parked until the leader
    /// broadcasts a request, then running the matching generation so the
    /// group's collectives stay synchronised. Without this, the leader would
    /// deadlock on its first request waiting for workers that never entered the
    /// collectives.
    #[cfg(feature = "mlx-backend")]
    async fn load_distributed(
        options: MlxModelLoadOptions,
        dist: MlxDistributedSetup,
    ) -> Result<Self> {
        use mesh_mlx::{
            DistributedEngine, JacclParams, JoinParams, ModelRef, ServerState, WorkerHandle, spawn,
        };

        let join = JoinParams {
            hostfile_json: dist.hostfile_json,
            jaccl: dist.jaccl.map(|j| JacclParams {
                devices_json: j.devices_json,
                coordinator: j.coordinator,
            }),
            rank: dist.rank,
            backend: dist.backend.to_backend(),
            mode: dist.mode,
        };
        let dengine = DistributedEngine::join(&ModelRef::new(&options.model_id), join)
            .await
            .map_err(|e| anyhow::anyhow!("join MLX group for {}: {e}", options.model_id))?;

        // Worker ranks run the lock-step loop and never serve OpenAI; mesh
        // routes inference to the leader. Use a fixed sentinel port so the
        // handle has a stable (unused) value.
        if !dengine.is_leader() {
            let rank = dengine.rank();
            tracing::info!(
                model = %options.model_id,
                rank,
                mode = ?dist.mode,
                "MLX distributed worker running lock-step loop (no local OpenAI server)"
            );
            let worker = WorkerHandle::spawn(dengine);
            return Ok(Self {
                role: MlxNodeRole::Worker(worker),
                port: 0,
                model_id: options.model_id,
            });
        }

        // Leader: serve OpenAI. The chat path broadcasts each request to the
        // workers and drives the group in lock-step. Keep a `ServerState` clone
        // so shutdown can broadcast the worker-exit sentinel.
        let state = ServerState::distributed(dengine, options.model_id.clone());
        let server = spawn(state.clone(), options.bind_addr)
            .await
            .map_err(|e| anyhow::anyhow!("start MLX OpenAI server: {e}"))?;
        let port = server.port();
        tracing::info!(
            model = %options.model_id,
            rank = dist.rank,
            mode = ?dist.mode,
            port,
            "MLX distributed leader serving OpenAI API"
        );
        Ok(Self {
            role: MlxNodeRole::Leader {
                server,
                state: Some(state),
            },
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

    /// Stop the backend. A distributed leader first broadcasts the worker-exit
    /// sentinel so worker ranks leave their lock-step loop cleanly, then stops
    /// its OpenAI server. A worker waits for its loop to finish.
    #[cfg(feature = "mlx-backend")]
    pub async fn shutdown(self) -> Result<()> {
        match self.role {
            MlxNodeRole::Leader { server, state } => {
                if let Some(state) = state {
                    state.signal_worker_shutdown().await;
                }
                server.shutdown().await;
            }
            MlxNodeRole::Worker(worker) => {
                // The leader's sentinel (or its exit) ends the loop; join it.
                if let Err(e) = worker.join().await {
                    tracing::warn!("MLX worker loop ended with error: {e}");
                }
            }
        }
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
    fn model_ids_match_full_and_tail_case_insensitive() {
        // Exact and case-insensitive.
        assert!(model_ids_match(
            "mlx-community/Qwen2.5-0.5B",
            "MLX-Community/qwen2.5-0.5b"
        ));
        // Repo-path vs bare name (mesh stores HF snapshots under repo paths).
        assert!(model_ids_match(
            "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            "Qwen2.5-0.5B-Instruct-bf16"
        ));
        // Different models do not match.
        assert!(!model_ids_match(
            "mlx-community/Qwen2.5-0.5B",
            "mlx-community/Llama-3.2-3B"
        ));
        // Same tail, different org still matches (tail is the identity we have).
        assert!(model_ids_match("org-a/Model-X", "org-b/Model-X"));
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
    fn rank_order_is_identical_on_every_node() {
        // Three nodes, each planning with itself as "local" and the other two
        // as peers: all must derive the same member order, and each must find
        // itself at its sorted position.
        let ids = ["node-b", "node-a", "node-c"];
        let mut orders = Vec::new();
        for local in ids {
            let peers = ids
                .iter()
                .filter(|id| **id != local)
                .map(|id| (id.to_string(), None));
            let (members, local_rank) = rank_order(local.to_string(), peers);
            let order: Vec<String> = members.into_iter().map(|(id, _)| id).collect();
            assert_eq!(order[local_rank], local);
            orders.push(order);
        }
        assert_eq!(orders[0], orders[1]);
        assert_eq!(orders[1], orders[2]);
        assert_eq!(orders[0], vec!["node-a", "node-b", "node-c"]);
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
        // Ring hostfile is the MLX runtime format: array of arrays of ip:port,
        // one row per rank — not the {ssh,ips,rdma} launch-tooling shape.
        let rows: Vec<Vec<String>> = serde_json::from_str(&hf).expect("ring hostfile parses");
        assert_eq!(rows.len(), 2);
        assert!(rows[1].iter().any(|s| s.contains("10.0.0.2:5680")));
        assert!(!hf.contains("ssh"));
        // Ring backend → no JACCL devices matrix.
        assert!(plan.jaccl_devices().is_none());

        let opts = MlxModelLoadOptions::new("m").with_group(&plan);
        let dist = opts.distributed.expect("multi-node attaches setup");
        assert_eq!(dist.rank, 0);
        assert_eq!(dist.mode, ParallelismMode::Pipeline);
        assert!(dist.hostfile_json.contains("10.0.0.2"));
    }
}
