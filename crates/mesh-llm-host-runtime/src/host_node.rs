//! In-process mesh node entry point for the published SDK.
//!
//! This is the bridge between [`mesh-llm-api-server`][api-server] (the
//! public Rust SDK) and the real iroh-backed mesh node implementation in
//! `crate::mesh::Node`. Without this module the SDK can only run an
//! HTTP-shim "client" that flips `connected = true` and emits an event;
//! with it, a Rust application can `cargo add mesh-llm-api-server --features
//! host-runtime` and run an actual mesh peer that does gossip, relay
//! registration (including [`--relay-auth`][relay-auth]), invite tokens,
//! and QUIC peer connections — the same things the `mesh-llm` binary
//! does.
//!
//! [api-server]: https://docs.rs/mesh-llm-api-server
//! [relay-auth]: https://github.com/Mesh-LLM/mesh-llm/pull/641
//!
//! ## Scope
//!
//! This module deliberately exposes only what the SDK needs:
//!
//! - [`HostNodeSpec`] — what the SDK passes in (role, relays, relay auths,
//!   QUIC bind, VRAM cap, enumerate-host flag).
//! - [`HostNode`] — the handle the SDK gets back (`invite_token`,
//!   `start_accepting`, `id`, `shutdown`).
//! - [`start_host_node`] — the entry point.
//!
//! It does not expose the full `mesh::Node` API. Internals stay
//! `pub(crate)` so we can keep refactoring without breaking SDK
//! consumers.
//!
//! ## What this does not do (yet)
//!
//! - It does not start local model serving. That requires plugging an
//!   `EmbeddedServingController` from [`crate::sdk`] into the SDK's
//!   `MeshNodeBuilder`. For client-only embedders (no GPU) this is
//!   not needed — the OpenAI proxy still routes requests to remote
//!   mesh peers serving the model.
//! - It does not start auto-discovery (`--auto`) of public meshes.
//!   Consumers can call [`HostNode::join`] with an invite token they
//!   obtained out of band (e.g. through Nostr).

use crate::api;
use crate::inference::election;
use crate::mesh::{self, NodeRole, QuicBindSelection};
use crate::network::affinity::AffinityRouter;
use crate::network::openai::ingress::api_proxy;
use anyhow::{Context, Result};
use std::net::SocketAddr;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;

/// Configuration for [`start_host_node`].
///
/// Field shape mirrors the slice of `mesh-llm`'s CLI flags that
/// `crate::mesh::Node::start` consumes. New fields here track new CLI
/// flags as they get added.
///
/// Gated iroh-relay (per-relay bearer token) support lives behind the
/// separate `--relay-auth` flag tracked on its own PR; this struct will
/// gain a `relay_auths` field once that lands.
#[derive(Clone, Debug, Default)]
pub struct HostNodeSpec {
    /// Mesh role.
    pub role: NodeRole,
    /// iroh relay URLs (empty = use bundled defaults).
    pub relays: Vec<String>,
    /// Local QUIC bind selection (IP and/or port).
    pub quic_bind: QuicBindSelection,
    /// VRAM cap in GB. `Some(0.0)` for client-only nodes that should not
    /// advertise any VRAM.
    pub max_vram_gb: Option<f64>,
    /// Whether to publish a hardware survey to gossip.
    pub enumerate_host: bool,
}

/// A running mesh node started by [`start_host_node`].
///
/// Drop the handle (or call [`HostNode::shutdown`]) to stop the iroh
/// endpoint and tear down background tasks.
#[derive(Clone)]
pub struct HostNode {
    inner: mesh::Node,
}

impl HostNode {
    /// Start accepting incoming mesh connections.
    ///
    /// The iroh endpoint binds in [`start_host_node`], but the accept
    /// loop waits for this call so the embedder can finish wiring (set a
    /// display name, advertise models) before the node is reachable.
    pub fn start_accepting(&self) {
        self.inner.start_accepting();
    }

    /// Hex-formatted endpoint ID, suitable for logging.
    pub fn id(&self) -> String {
        format!("{:?}", self.inner.id())
    }

    /// An invite token that other nodes can use to join this one.
    pub fn invite_token(&self) -> String {
        self.inner.invite_token()
    }

    /// Join an existing mesh via an invite token produced elsewhere.
    pub async fn join(&self, invite_token: &str) -> Result<()> {
        self.inner.join(invite_token).await
    }

    /// Set a human-readable display name advertised to peers.
    pub async fn set_display_name(&self, name: String) {
        self.inner.set_display_name(name).await;
    }

    /// Replace the set of models this node advertises.
    pub async fn set_models(&self, models: Vec<String>) {
        self.inner.set_models(models).await;
    }

    /// Current set of advertised models.
    pub async fn models(&self) -> Vec<String> {
        self.inner.models().await
    }

    /// Shut the node down (best-effort).
    pub async fn shutdown(&self) {
        self.inner.shutdown_control_listener().await;
    }
}

/// Bring an in-process mesh node online with the given spec.
///
/// Equivalent to the iroh-endpoint slice of `mesh-llm serve` / `mesh-llm
/// client`: binds the iroh endpoint, attaches relay-auth tokens, waits
/// briefly for the home relay to come online, and returns a handle. The
/// caller is responsible for any further wiring (calling
/// [`HostNode::start_accepting`], setting models / display name, joining
/// other meshes via [`HostNode::join`]).
pub async fn start_host_node(spec: HostNodeSpec) -> Result<HostNode> {
    let (node, _channels) = mesh::Node::start(
        spec.role,
        &spec.relays,
        spec.quic_bind,
        spec.max_vram_gb,
        spec.enumerate_host,
        None, // owner control config — not currently exposed to SDK
        None, // config file — not relevant to SDK consumers
    )
    .await?;
    Ok(HostNode { inner: node })
}

// Re-export the types embedders need to express a spec. Hidden inside
// the curated `host_node` namespace, NOT at the crate root, so we can
// keep refactoring the underlying `mesh` module.
pub use mesh::{NodeRole as MeshNodeRole, QuicBindSelection as MeshQuicBindSelection};

/// Handle to a running in-process OpenAI HTTP proxy started by
/// [`start_openai_proxy`].
///
/// Drop the handle (or call [`OpenAiProxyHandle::shutdown`]) to stop the
/// proxy. While alive, requests against
/// `http://{bound_addr}/v1/{chat/completions,models,…}` route to mesh
/// peers via the underlying `HostNode`'s gossip + QUIC transport.
pub struct OpenAiProxyHandle {
    addr: SocketAddr,
    task: JoinHandle<()>,
    /// Held so the no-op runtime-control receiver isn't dropped while the
    /// proxy is alive (the proxy sends control requests on this channel;
    /// dropping the receiver would close the sender and surface spurious
    /// errors). Drained on a background task.
    _control_drain: JoinHandle<()>,
}

impl OpenAiProxyHandle {
    /// The local address the proxy is bound to. When the embedder asks for
    /// port 0 this is the OS-assigned ephemeral port.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Base URL suitable for OpenAI-compatible client libraries.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Stop the proxy task. Idempotent; safe to call after drop().
    pub fn shutdown(&self) {
        self.task.abort();
        self._control_drain.abort();
    }
}

impl Drop for OpenAiProxyHandle {
    fn drop(&mut self) {
        self.task.abort();
        self._control_drain.abort();
    }
}

/// Start an OpenAI-compatible HTTP proxy that fronts a [`HostNode`].
///
/// Equivalent to the `--port` slice of `mesh-llm serve` / `mesh-llm
/// client`: binds a TCP listener, accepts HTTP connections, parses
/// requests, and routes them to mesh peers that advertise the requested
/// model in gossip. Suitable for client-only embedders (no local
/// serving) and for embedders that have plugged a `ServingController`
/// into their `MeshNode` for local inference.
///
/// Returns once the listener is bound; the proxy keeps running in a
/// background task until [`OpenAiProxyHandle::shutdown`] or drop.
///
/// The `port = 0` case is supported and asks the OS for an ephemeral
/// port; read it from [`OpenAiProxyHandle::local_addr`] after this call
/// returns.
pub async fn start_openai_proxy(
    node: &HostNode,
    port: u16,
    listen_all: bool,
) -> Result<OpenAiProxyHandle> {
    let bind_addr = if listen_all {
        format!("0.0.0.0:{port}")
    } else {
        format!("127.0.0.1:{port}")
    };
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("binding OpenAI proxy to {bind_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("reading OpenAI proxy local addr")?;

    // Targets watch channel: starts empty. Routing to remote peers does
    // not read this — it reads `node.hosts_for_model()` at request time.
    // The channel only matters if the embedder later wires local serving
    // through `crate::sdk::EmbeddedServingController`.
    let (_target_tx, target_rx) = watch::channel(election::ModelTargets::default());

    // Runtime-control channel: the proxy sends model load/unload commands
    // here. Embedders without local serving have no one to handle these,
    // so spawn a background drain that just logs and discards.
    let (control_tx, mut control_rx) = mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    let control_drain = tokio::spawn(async move {
        // Discard with a single line per request. We deliberately don't
        // {:?} the request because RuntimeControlRequest doesn't impl
        // Debug and is an internal type the SDK shouldn't widen for a
        // background log line.
        while control_rx.recv().await.is_some() {
            tracing::debug!(
                "SDK-mode OpenAI proxy received a runtime-control request with no handler attached; discarding"
            );
        }
    });

    let affinity = AffinityRouter::new();
    let node_for_proxy = node.inner.clone();
    let task = tokio::spawn(async move {
        api_proxy(
            node_for_proxy,
            local_addr.port(),
            target_rx,
            control_tx,
            Some(listener),
            listen_all,
            affinity,
        )
        .await;
    });

    Ok(OpenAiProxyHandle {
        addr: local_addr,
        task,
        _control_drain: control_drain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::endpoint::{presets, Endpoint, RelayMode};
    use iroh::SecretKey;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
    use std::time::Duration;

    fn free_local_udp_port() -> u16 {
        let socket = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .expect("allocate local UDP port");
        socket.local_addr().expect("read local UDP port").port()
    }

    async fn probe_quic_port_released(port: u16) -> anyhow::Result<()> {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        let mut last_error = None;

        for _ in 0..20 {
            match Endpoint::builder(presets::Minimal)
                .secret_key(SecretKey::generate())
                .relay_mode(RelayMode::Disabled)
                .bind_addr(bind_addr)?
                .bind()
                .await
            {
                Ok(endpoint) => {
                    endpoint.close().await;
                    return Ok(());
                }
                Err(err) => {
                    last_error = Some(err);
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            }
        }

        Err(anyhow::anyhow!(
            "host-node shutdown should release UDP port {port}: {:?}",
            last_error
        ))
    }

    #[tokio::test]
    async fn id_returns_bare_hex_endpoint_id() -> anyhow::Result<()> {
        let inner = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let expected = inner.id().to_string();
        let node = HostNode { inner };

        assert_eq!(node.id(), expected);
        assert!(!node.id().contains("PublicKey"));

        node.shutdown().await;
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_closes_the_mesh_endpoint() -> anyhow::Result<()> {
        let inner = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let node = HostNode { inner };

        node.shutdown().await;

        assert!(node.inner.endpoint_is_closed_for_tests());
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_releases_fixed_quic_bind() -> anyhow::Result<()> {
        let quic_port = free_local_udp_port();
        let node = start_host_node(HostNodeSpec {
            role: MeshNodeRole::Client,
            quic_bind: MeshQuicBindSelection {
                ip: Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
                port: Some(quic_port),
            },
            max_vram_gb: Some(0.0),
            enumerate_host: false,
            ..HostNodeSpec::default()
        })
        .await?;

        node.start_accepting();
        node.shutdown().await;
        drop(node);

        probe_quic_port_released(quic_port).await
    }
}
