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
        if node_count <= 1 {
            return ParallelismPlan {
                mode: ParallelismMode::Single,
                node_count,
                worst_rtt: None,
                reason: "single node: no model split".into(),
            };
        }
        let worst = samples.iter().map(|s| s.rtt).max();
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

impl TransportPlan {
    /// Recommend a backend for the chosen mode and the available nodes.
    /// Tensor prefers JACCL when Thunderbolt RDMA maps are present; otherwise
    /// ring (TCP over the LAN).
    pub fn recommend(mode: ParallelismMode, nodes: Vec<NodeEndpoint>) -> Self {
        let has_rdma = nodes.iter().any(|n| !n.rdma.is_empty());
        let backend = match mode {
            ParallelismMode::Tensor if has_rdma => MlxBackendKind::Jaccl,
            _ => MlxBackendKind::Ring,
        };
        TransportPlan { backend, nodes }
    }

    /// Render the JSON hostfile MLX launch consumes (`[{ssh, ips, rdma?}]`).
    pub fn render_hostfile(&self) -> String {
        let entries: Vec<serde_json::Value> = self
            .nodes
            .iter()
            .map(|n| {
                let mut obj = serde_json::Map::new();
                obj.insert("ssh".into(), n.ssh.clone().into());
                obj.insert("ips".into(), n.ips.clone().into());
                if !n.rdma.is_empty() {
                    let rdma: Vec<serde_json::Value> = n
                        .rdma
                        .iter()
                        .map(|d| match d {
                            Some(s) => serde_json::Value::String(s.clone()),
                            None => serde_json::Value::Null,
                        })
                        .collect();
                    obj.insert("rdma".into(), rdma.into());
                }
                serde_json::Value::Object(obj)
            })
            .collect();
        serde_json::to_string_pretty(&serde_json::Value::Array(entries))
            .unwrap_or_else(|_| "[]".into())
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
    fn pipeline_uses_ring_and_renders_hostfile() {
        let plan = TransportPlan::recommend(ParallelismMode::Pipeline, plain(2));
        assert_eq!(plan.backend, MlxBackendKind::Ring);
        let hf = plan.render_hostfile();
        assert!(hf.contains("10.0.0.1"));
        assert!(hf.contains("mac-0"));
        assert!(!hf.contains("rdma"));
    }
}
