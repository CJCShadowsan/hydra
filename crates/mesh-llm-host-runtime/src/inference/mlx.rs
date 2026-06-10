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

/// Plan tensor-vs-pipeline parallelism + transport for a candidate MLX group
/// from measured inter-node latency. Pure decision logic mesh owns; usable
/// without the native engine.
///
/// This is the entry point for **multi-node** MLX group formation. Single-node
/// serving (wired into `runtime::local`) does not need it; it is called once a
/// group of MLX-eligible peers is being assembled. Exercised by unit tests.
#[allow(dead_code)]
pub fn plan_parallelism(
    nodes: Vec<NodeEndpoint>,
    samples: &[LatencySample],
) -> (ParallelismPlan, TransportPlan) {
    MlxOrchestrator::default().plan(nodes, samples)
}

/// Options for loading an MLX model as a local backend.
#[derive(Debug, Clone)]
pub struct MlxModelLoadOptions {
    /// Hugging Face repo id (safetensors; bf16/fp16 or quantized 4-bit).
    pub model_id: String,
    /// Address to bind the OpenAI server to. Use `127.0.0.1:0` for an ephemeral
    /// port (the local-backend convention).
    pub bind_addr: std::net::SocketAddr,
}

impl MlxModelLoadOptions {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            bind_addr: "127.0.0.1:0".parse().expect("static addr parses"),
        }
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
}
