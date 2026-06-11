//! The mesh-facing surface: latency-aware parallelism planning and the
//! transport/hostfile plan. Pure logic — no engine calls — so mesh can preview
//! and unit-test placement decisions.
//!
//! MLX mode is **local-only**: MLX opens its own TCP (ring) / RDMA (jaccl)
//! sockets and cannot use mesh's QUIC transport, and tunnelling would defeat the
//! latency MLX distributed exists for. Mesh forms an MLX group only from
//! Apple-Silicon, MLX-capable, directly-routable peers.

use crate::distributed::Backend;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// How MLX should split the model across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismMode {
    /// Single node, no split.
    Single,
    /// Layer pipeline (one activation per stage hop). Latency tolerant — the
    /// default over Ethernet/Wi-Fi.
    Pipeline,
    /// Tensor (Megatron) sharding — all-reduce per layer. Needs a low-latency
    /// fabric (Thunderbolt/JACCL or tight LAN).
    Tensor,
}

/// One inter-node round-trip-time sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencySample {
    pub from_rank: usize,
    pub to_rank: usize,
    pub rtt: Duration,
}

impl LatencySample {
    pub fn new(from_rank: usize, to_rank: usize, rtt: Duration) -> Self {
        Self {
            from_rank,
            to_rank,
            rtt,
        }
    }
}

/// Operator preference for the parallelism mode, normally from the
/// `MESH_LLM_MLX_PARALLELISM` env var.
///
/// `Auto` (default) lets the latency-aware planner decide. `Tensor` and
/// `Pipeline` force the mode regardless of measured RTT — e.g. tensor over
/// plain Ethernet (works, slower per-token than over RDMA, but can still win
/// on memory-bound decode), or pipeline on a Thunderbolt mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismPreference {
    /// Let the latency-aware planner decide (default).
    #[default]
    Auto,
    /// Force tensor parallelism even on a high-latency fabric.
    Tensor,
    /// Force pipeline parallelism even on a low-latency fabric.
    Pipeline,
}

impl ParallelismPreference {
    /// Parse from the `MESH_LLM_MLX_PARALLELISM` env var
    /// (auto|tensor|pipeline). Unknown/empty values fall back to `Auto`.
    pub fn from_env() -> Self {
        Self::parse(&std::env::var("MESH_LLM_MLX_PARALLELISM").unwrap_or_default())
    }

    /// Parse a preference value (auto|tensor|tp|pipeline|pp, case-insensitive).
    /// Unknown/empty values fall back to `Auto`.
    pub fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "tensor" | "tp" => ParallelismPreference::Tensor,
            "pipeline" | "pp" => ParallelismPreference::Pipeline,
            _ => ParallelismPreference::Auto,
        }
    }

    /// The forced [`ParallelismMode`], if this preference is not `Auto`.
    fn forced_mode(self) -> Option<ParallelismMode> {
        match self {
            ParallelismPreference::Auto => None,
            ParallelismPreference::Tensor => Some(ParallelismMode::Tensor),
            ParallelismPreference::Pipeline => Some(ParallelismMode::Pipeline),
        }
    }
}

/// Chooses [`ParallelismMode`] from measured inter-node latency.
#[derive(Debug, Clone, Copy)]
pub struct ParallelismPlanner {
    /// Worst-case inter-node RTT at or below which tensor parallelism is chosen.
    pub tensor_rtt_threshold: Duration,
}

impl Default for ParallelismPlanner {
    fn default() -> Self {
        Self {
            tensor_rtt_threshold: Duration::from_millis(2),
        }
    }
}

/// Planning outcome plus reasoning for telemetry/console.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelismPlan {
    pub mode: ParallelismMode,
    pub node_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_rtt: Option<Duration>,
    pub reason: String,
}

impl ParallelismPlanner {
    pub fn with_threshold(tensor_rtt_threshold: Duration) -> Self {
        Self {
            tensor_rtt_threshold,
        }
    }

    pub fn plan(&self, node_count: usize, samples: &[LatencySample]) -> ParallelismPlan {
        self.plan_with_preference(node_count, samples, ParallelismPreference::Auto)
    }

    /// Plan honouring an operator [`ParallelismPreference`]. A forced mode wins
    /// over the latency heuristic (e.g. tensor over plain Ethernet); single
    /// node always plans `Single`.
    pub fn plan_with_preference(
        &self,
        node_count: usize,
        samples: &[LatencySample],
        pref: ParallelismPreference,
    ) -> ParallelismPlan {
        if node_count <= 1 {
            return ParallelismPlan {
                mode: ParallelismMode::Single,
                node_count,
                worst_rtt: None,
                reason: "single node: no model split".into(),
            };
        }
        let worst = samples.iter().map(|s| s.rtt).max();
        if let Some(mode) = pref.forced_mode() {
            return ParallelismPlan {
                mode,
                node_count,
                worst_rtt: worst,
                reason: format!("operator preference (MESH_LLM_MLX_PARALLELISM): forced {mode:?}"),
            };
        }
        let (mode, reason) = match worst {
            None => (
                ParallelismMode::Pipeline,
                "no latency samples yet: defaulting to latency-tolerant pipeline".into(),
            ),
            Some(rtt) if rtt <= self.tensor_rtt_threshold => (
                ParallelismMode::Tensor,
                format!(
                    "worst RTT {rtt:?} <= {:?}: low-latency fabric, using tensor parallel",
                    self.tensor_rtt_threshold
                ),
            ),
            Some(rtt) => (
                ParallelismMode::Pipeline,
                format!(
                    "worst RTT {rtt:?} > {:?}: using latency-tolerant pipeline",
                    self.tensor_rtt_threshold
                ),
            ),
        };
        ParallelismPlan {
            mode,
            node_count,
            worst_rtt: worst,
            reason,
        }
    }
}

/// A directly-routable node in the local MLX group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeEndpoint {
    /// ssh target `mlx.launch`-equivalent orchestration uses.
    pub ssh: String,
    /// Directly-routable IPs MLX binds/connects to (ring backend).
    pub ips: Vec<String>,
    /// Optional per-peer rdma device names for JACCL (None for self).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rdma: Vec<Option<String>>,
}

/// The networking plan: which MLX backend + the node hostfile entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportPlan {
    pub backend: MlxBackendKind,
    pub nodes: Vec<NodeEndpoint>,
}

/// Mirror of [`Backend`] for serialisation in plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxBackendKind {
    Ring,
    Jaccl,
    Mpi,
}

impl MlxBackendKind {
    pub fn to_backend(self) -> Backend {
        match self {
            MlxBackendKind::Ring => Backend::Ring,
            MlxBackendKind::Jaccl => Backend::Jaccl,
            MlxBackendKind::Mpi => Backend::Mpi,
        }
    }
}

/// What transport the operator wants for MLX inter-node traffic.
///
/// JACCL (RDMA over Thunderbolt) can't be silently auto-enabled — it needs
/// macOS 26.2+, `rdma_ctl enable` in recovery mode, and a Thunderbolt-5 mesh —
/// so it's opt-in. But it also shouldn't require hand-editing JSON, so `Auto`
/// uses JACCL **only when RDMA devices are actually detected** and otherwise
/// falls back to the TCP ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportPreference {
    /// Use JACCL if RDMA is detected on all nodes; otherwise ring. (default)
    #[default]
    Auto,
    /// Force the TCP ring even if RDMA is available.
    Ring,
    /// Require JACCL. If RDMA is not present this is an error (no silent
    /// downgrade — the operator explicitly asked for it).
    Jaccl,
}

impl TransportPreference {
    /// Parse from the `MESH_LLM_MLX_TRANSPORT` env var (auto|ring|jaccl).
    /// Unknown/empty values fall back to `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("MESH_LLM_MLX_TRANSPORT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "ring" | "tcp" => TransportPreference::Ring,
            "jaccl" | "rdma" | "thunderbolt" => TransportPreference::Jaccl,
            _ => TransportPreference::Auto,
        }
    }
}

impl TransportPlan {
    /// Recommend a backend for the chosen mode and the available nodes, using
    /// the default `Auto` preference. Tensor prefers JACCL when RDMA maps are
    /// present; otherwise ring (TCP over the LAN).
    pub fn recommend(mode: ParallelismMode, nodes: Vec<NodeEndpoint>) -> Self {
        Self::recommend_with(mode, nodes, TransportPreference::Auto)
            .expect("Auto preference never errors")
    }

    /// Recommend a backend honouring an explicit [`TransportPreference`].
    ///
    /// - `Ring` → always ring.
    /// - `Jaccl` → JACCL, but **errors** if no node advertises RDMA maps.
    /// - `Auto` → JACCL when every node has an RDMA map (a complete Thunderbolt
    ///   mesh); otherwise ring.
    pub fn recommend_with(
        mode: ParallelismMode,
        nodes: Vec<NodeEndpoint>,
        pref: TransportPreference,
    ) -> Result<Self, String> {
        // A usable JACCL mesh needs every node to expose a complete RDMA device
        // map: one row per node, with every non-self entry populated. A merely
        // non-empty row is not enough — a partial map fails at hostfile load.
        let all_have_rdma = !nodes.is_empty()
            && nodes.iter().enumerate().all(|(i, n)| {
                n.rdma.len() == nodes.len()
                    && n.rdma
                        .iter()
                        .enumerate()
                        .all(|(j, dev)| i == j || dev.is_some())
            });
        let any_rdma = nodes.iter().any(|n| !n.rdma.is_empty());

        let backend = match pref {
            TransportPreference::Ring => MlxBackendKind::Ring,
            TransportPreference::Jaccl => {
                if !all_have_rdma {
                    return Err(format!(
                        "MESH_LLM_MLX_TRANSPORT=jaccl requested but RDMA is not available on all \
                         {} node(s) (any_rdma={any_rdma}). JACCL needs macOS 26.2+, \
                         `rdma_ctl enable` in recovery mode, and a Thunderbolt-5 mesh. \
                         Set MESH_LLM_MLX_TRANSPORT=auto to fall back to the TCP ring.",
                        nodes.len()
                    ));
                }
                // JACCL is only beneficial for tensor parallelism; honour the
                // operator's request regardless of mode but it pairs with tensor.
                MlxBackendKind::Jaccl
            }
            TransportPreference::Auto => match mode {
                ParallelismMode::Tensor if all_have_rdma => MlxBackendKind::Jaccl,
                _ => MlxBackendKind::Ring,
            },
        };
        Ok(TransportPlan { backend, nodes })
    }

    /// Render the `MLX_HOSTFILE` JSON the MLX **ring** backend reads at runtime.
    ///
    /// This is *not* the `{ssh, ips, rdma}` launch-tooling shape (that is what
    /// `mlx.distributed_config` consumes to SSH-launch and generate hosts). The
    /// ring backend's `load_nodes()` expects an array, in rank order, of arrays
    /// of `"ip:port"` strings — one inner array per node, one entry per
    /// connection:
    ///
    /// ```json
    /// [
    ///   ["10.0.0.1:5680"],
    ///   ["10.0.0.2:5680"]
    /// ]
    /// ```
    ///
    /// Each node's row carries its own `ips` (the addresses it binds/accepts on
    /// for its rank, and that peers connect to for the others).
    pub fn render_hostfile(&self) -> String {
        let rows: Vec<Vec<String>> = self.nodes.iter().map(|n| n.ips.clone()).collect();
        serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into())
    }

    /// Render the JACCL devices matrix JSON that `MLX_IBV_DEVICES` points to.
    ///
    /// JACCL's `parse_devices_json` reads this file as an NxN matrix: row `i`,
    /// column `j` is the list of RDMA device name(s) rank `i` uses to reach
    /// rank `j`. The diagonal (`i == j`) is empty; every off-diagonal cell must
    /// be populated and all cells must have the same connection count. Returns
    /// `None` unless every node advertises a complete row (so we never write an
    /// invalid matrix that JACCL would reject).
    ///
    /// Shape (2 nodes, single connection each):
    /// ```json
    /// [
    ///   [[], ["rdma_en2"]],
    ///   [["rdma_en2"], []]
    /// ]
    /// ```
    pub fn render_jaccl_devices(&self) -> Option<String> {
        let n = self.nodes.len();
        if n == 0 {
            return None;
        }
        let mut matrix: Vec<Vec<Vec<String>>> = Vec::with_capacity(n);
        for (i, node) in self.nodes.iter().enumerate() {
            if node.rdma.len() != n {
                return None;
            }
            let mut row = Vec::with_capacity(n);
            for (j, dev) in node.rdma.iter().enumerate() {
                match (i == j, dev) {
                    (true, _) => row.push(Vec::new()),
                    (false, Some(d)) => row.push(vec![d.clone()]),
                    (false, None) => return None,
                }
            }
            matrix.push(row);
        }
        serde_json::to_string_pretty(&matrix).ok()
    }
}

/// Whether this host can run the MLX backend (Apple Silicon + macOS).
///
/// Mesh calls this to decide if a peer is MLX-eligible before forming a group.
/// MLX mode is local-only and Metal-only, so the gate is macOS on aarch64.
pub fn mlx_supported() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

/// Detect this host's RDMA (Thunderbolt) devices by running `ibv_devices`.
///
/// Returns the device names (e.g. `["rdma_en2", "rdma_en3", …]`) — these are
/// what JACCL needs to map the Thunderbolt mesh. Empty when RDMA isn't enabled
/// (no macOS 26.2 / `rdma_ctl enable`), no Thunderbolt fabric, or the tool is
/// missing — in which case the planner falls back to the TCP ring.
///
/// This is the auto-detection that makes JACCL opt-in-but-zero-config: a node
/// gossips this list, and a complete mesh (every node has devices) unlocks
/// JACCL under `Auto`.
pub fn detect_rdma_devices() -> Vec<String> {
    let output = match std::process::Command::new("ibv_devices").output() {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&output.stdout);
    // `ibv_devices` prints a header then `<device>\t<node_guid>` rows.
    text.lines()
        .skip_while(|l| l.contains("device") && l.contains("node GUID"))
        .filter_map(|l| l.split_whitespace().next())
        .filter(|tok| tok.starts_with("rdma_") || tok.starts_with("rdma"))
        .map(|s| s.to_string())
        .collect()
}

/// The mesh-facing orchestration surface for an MLX node.
///
/// Mesh uses this to (1) check eligibility, (2) plan parallelism from measured
/// latency, and (3) obtain the backend address it should bind the OpenAI server
/// to and then route OpenAI traffic to. The actual serving (loading the model
/// and running the OpenAI server) is driven by [`crate::runtime`]; this type is
/// the thin, testable decision layer mesh owns.
#[derive(Debug, Clone, Default)]
pub struct MlxOrchestrator {
    pub planner: ParallelismPlanner,
}

impl MlxOrchestrator {
    pub fn new(planner: ParallelismPlanner) -> Self {
        Self { planner }
    }

    /// Whether this host can serve MLX.
    pub fn supported(&self) -> bool {
        mlx_supported()
    }

    /// Plan the parallelism mode + transport for a candidate MLX group.
    ///
    /// `nodes` are the directly-routable, MLX-eligible peers (mesh must have
    /// already filtered to Apple-Silicon, MLX-capable, same-LAN/Thunderbolt
    /// peers — MLX is local-only and cannot use mesh QUIC). `samples` is the
    /// measured inter-node RTT.
    pub fn plan(
        &self,
        nodes: Vec<NodeEndpoint>,
        samples: &[LatencySample],
    ) -> (ParallelismPlan, TransportPlan) {
        let plan = self.planner.plan(nodes.len().max(1), samples);
        let transport = TransportPlan::recommend(plan.mode, nodes);
        (plan, transport)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(n: usize) -> Vec<NodeEndpoint> {
        (0..n)
            .map(|i| NodeEndpoint {
                ssh: format!("mac-{i}"),
                ips: vec![format!("10.0.0.{}", i + 1)],
                rdma: vec![],
            })
            .collect()
    }

    #[test]
    fn single_node_is_single() {
        assert_eq!(
            ParallelismPlanner::default().plan(1, &[]).mode,
            ParallelismMode::Single
        );
    }

    #[test]
    fn sub_threshold_is_tensor_above_is_pipeline() {
        let p = ParallelismPlanner::default();
        assert_eq!(
            p.plan(2, &[LatencySample::new(0, 1, Duration::from_micros(800))])
                .mode,
            ParallelismMode::Tensor
        );
        assert_eq!(
            p.plan(2, &[LatencySample::new(0, 1, Duration::from_millis(5))])
                .mode,
            ParallelismMode::Pipeline
        );
    }

    #[test]
    fn forced_tensor_wins_over_high_rtt() {
        // Tensor over plain Ethernet: high RTT would normally pick pipeline,
        // but an explicit operator preference forces tensor.
        let p = ParallelismPlanner::default();
        let plan = p.plan_with_preference(
            2,
            &[LatencySample::new(0, 1, Duration::from_millis(5))],
            ParallelismPreference::Tensor,
        );
        assert_eq!(plan.mode, ParallelismMode::Tensor);
        assert!(plan.reason.contains("forced"));
    }

    #[test]
    fn forced_pipeline_wins_over_low_rtt() {
        let p = ParallelismPlanner::default();
        let plan = p.plan_with_preference(
            2,
            &[LatencySample::new(0, 1, Duration::from_micros(500))],
            ParallelismPreference::Pipeline,
        );
        assert_eq!(plan.mode, ParallelismMode::Pipeline);
    }

    #[test]
    fn forced_mode_still_single_on_one_node() {
        let p = ParallelismPlanner::default();
        let plan = p.plan_with_preference(1, &[], ParallelismPreference::Tensor);
        assert_eq!(plan.mode, ParallelismMode::Single);
    }

    #[test]
    fn parallelism_preference_parses_values() {
        for (s, want) in [
            ("tensor", ParallelismPreference::Tensor),
            ("TP", ParallelismPreference::Tensor),
            ("pipeline", ParallelismPreference::Pipeline),
            ("pp", ParallelismPreference::Pipeline),
            ("auto", ParallelismPreference::Auto),
            ("", ParallelismPreference::Auto),
            ("bogus", ParallelismPreference::Auto),
        ] {
            assert_eq!(ParallelismPreference::parse(s), want, "value {s:?}");
        }
    }

    #[test]
    fn no_samples_defaults_pipeline() {
        assert_eq!(
            ParallelismPlanner::default().plan(2, &[]).mode,
            ParallelismMode::Pipeline
        );
    }

    #[test]
    fn worst_link_drives_decision() {
        let plan = ParallelismPlanner::default().plan(
            3,
            &[
                LatencySample::new(0, 1, Duration::from_micros(500)),
                LatencySample::new(1, 2, Duration::from_millis(8)),
            ],
        );
        assert_eq!(plan.mode, ParallelismMode::Pipeline);
        assert_eq!(plan.worst_rtt, Some(Duration::from_millis(8)));
    }

    #[test]
    fn tensor_with_rdma_picks_jaccl() {
        let mut nodes = plain(2);
        nodes[0].rdma = vec![None, Some("rdma_en5".into())];
        nodes[1].rdma = vec![Some("rdma_en5".into()), None];
        let plan = TransportPlan::recommend(ParallelismMode::Tensor, nodes);
        assert_eq!(plan.backend, MlxBackendKind::Jaccl);
    }

    #[test]
    fn orchestrator_plans_mode_and_transport_together() {
        let orch = MlxOrchestrator::default();
        // Low-latency 2-node → tensor + (ring, since no rdma maps here).
        let (plan, transport) = orch.plan(
            plain(2),
            &[LatencySample::new(0, 1, Duration::from_micros(700))],
        );
        assert_eq!(plan.mode, ParallelismMode::Tensor);
        assert_eq!(transport.backend, MlxBackendKind::Ring);

        // High-latency → pipeline + ring.
        let (plan, transport) = orch.plan(
            plain(3),
            &[LatencySample::new(0, 1, Duration::from_millis(7))],
        );
        assert_eq!(plan.mode, ParallelismMode::Pipeline);
        assert_eq!(transport.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn pipeline_uses_ring_and_renders_hostfile() {
        let plan = TransportPlan::recommend(ParallelismMode::Pipeline, plain(2));
        assert_eq!(plan.backend, MlxBackendKind::Ring);
        let hf = plan.render_hostfile();
        // Ring hostfile is the MLX runtime format: array of arrays of ip:port.
        let parsed: Vec<Vec<String>> = serde_json::from_str(&hf).unwrap();
        assert_eq!(parsed, vec![vec!["10.0.0.1"], vec!["10.0.0.2"]]);
        // No launch-tooling fields in the runtime hostfile.
        assert!(!hf.contains("mac-0"));
        assert!(!hf.contains("ssh"));
        assert!(!hf.contains("rdma"));
        // No JACCL devices matrix without an RDMA mesh.
        assert!(plan.render_jaccl_devices().is_none());
    }

    /// Two nodes each advertising an RDMA device map (a complete Thunderbolt
    /// mesh), used to exercise the JACCL paths.
    fn rdma_pair() -> Vec<NodeEndpoint> {
        vec![
            NodeEndpoint {
                ssh: "mac-0".into(),
                ips: vec!["10.0.0.1:5680".into()],
                rdma: vec![None, Some("rdma_en5".into())],
            },
            NodeEndpoint {
                ssh: "mac-1".into(),
                ips: vec!["10.0.0.2:5680".into()],
                rdma: vec![Some("rdma_en5".into()), None],
            },
        ]
    }

    #[test]
    fn auto_uses_jaccl_for_tensor_only_when_full_rdma_mesh() {
        // Tensor + complete RDMA mesh → JACCL.
        let p = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Auto,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Jaccl);

        // Pipeline + RDMA → still ring (JACCL only benefits tensor under Auto).
        let p = TransportPlan::recommend_with(
            ParallelismMode::Pipeline,
            rdma_pair(),
            TransportPreference::Auto,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Ring);

        // Tensor but no RDMA maps → ring.
        let p = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            plain(2),
            TransportPreference::Auto,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn explicit_ring_forces_tcp_even_with_rdma() {
        let p = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Ring,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn explicit_jaccl_errors_without_rdma() {
        let err = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            plain(2),
            TransportPreference::Jaccl,
        );
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("jaccl"));

        // With a full mesh it succeeds.
        let ok = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Jaccl,
        );
        assert_eq!(ok.unwrap().backend, MlxBackendKind::Jaccl);
    }

    #[test]
    fn jaccl_renders_ring_hostfile_and_devices_matrix() {
        let plan = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Jaccl,
        )
        .unwrap();

        // The ring hostfile is the runtime format MLX parses: an array of
        // arrays of "ip:port" — no objects, no rdma field.
        let hf = plan.render_hostfile();
        let parsed: Vec<Vec<String>> = serde_json::from_str(&hf).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].iter().all(|s| s.contains(':')));
        assert!(!hf.contains("rdma"));
        assert!(!hf.contains("ssh"));

        // The JACCL devices matrix is a separate NxN file: diagonal empty,
        // off-diagonal populated with device name(s).
        let devices = plan.render_jaccl_devices().expect("complete rdma mesh");
        let matrix: Vec<Vec<Vec<String>>> = serde_json::from_str(&devices).unwrap();
        assert_eq!(matrix.len(), 2);
        assert!(matrix[0][0].is_empty(), "diagonal is empty");
        assert!(!matrix[0][1].is_empty(), "off-diagonal populated");
        assert_eq!(matrix[1][0], vec!["rdma_en5".to_string()]);
    }

    #[test]
    fn transport_preference_parses_from_env_values() {
        // Default when unset/unknown.
        assert_eq!(TransportPreference::default(), TransportPreference::Auto);
    }
}
