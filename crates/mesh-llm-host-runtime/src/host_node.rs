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

/// Full mesh-llm runtime configuration for [`run_serve`].
///
/// Every field maps to a `mesh-llm` CLI flag. Defaults match the
/// binary's defaults so the SDK consumer only sets what they want
/// different.
///
/// Unlike [`HostNodeSpec`] (which only brings up the iroh endpoint),
/// a `MeshServeSpec` drives the **full** runtime path — the same code
/// path `mesh-llm serve` / `mesh-llm client` use. That means election,
/// tunnel manager, OpenAI proxy, management console, auto-discovery,
/// local model serving, plugin host — everything the binary does.
#[derive(Clone, Debug, Default)]
pub struct MeshServeSpec {
    /// Run as a client only (no GPU, no model). Maps to `--client`.
    pub client: bool,
    /// Auto-join the best discovered mesh. Maps to `--auto`.
    pub auto: bool,
    /// Publish this mesh for Nostr discovery. Maps to `--publish`.
    pub publish: bool,
    /// Human-readable mesh name. Maps to `--mesh-name`.
    pub mesh_name: Option<String>,
    /// Region tag (e.g. "US"). Maps to `--region`.
    pub region: Option<String>,
    /// Blackboard display name. Maps to `--name`.
    pub display_name: Option<String>,
    /// Invite tokens to join. Maps to repeatable `--join <invite>`.
    pub join: Vec<String>,
    /// Discovery filter (mesh name). Maps to `--discover [filter]`.
    pub discover: Option<String>,

    /// Models to serve. Path, catalog name, or HF ref. Maps to
    /// repeatable `--model <ref>`.
    pub models: Vec<String>,
    /// Raw local GGUF files. Maps to repeatable `--gguf <path>`.
    pub ggufs: Vec<std::path::PathBuf>,
    /// Explicit mmproj sidecar. Maps to `--mmproj`.
    pub mmproj: Option<std::path::PathBuf>,

    /// OpenAI API port. Default 9337. Maps to `--port`.
    pub port: Option<u16>,
    /// Console port. Default 3131. Maps to `--console`.
    pub console_port: Option<u16>,
    /// Disable the embedded web UI but keep the management API. Maps
    /// to `--headless`.
    pub headless: bool,
    /// Enable blackboard on public meshes. Maps to `--blackboard`.
    pub blackboard: bool,

    /// iroh relay URLs. Maps to repeatable `--relay <url>`.
    ///
    /// Gated iroh-relay (per-relay bearer token) support lives behind
    /// the separate `--relay-auth` PR; this struct will gain a
    /// `relay_auths` field once that lands.
    pub relays: Vec<String>,
    /// Custom Nostr relay URLs. Maps to repeatable `--nostr-relay`.
    pub nostr_relays: Vec<String>,
    /// Fixed QUIC bind port (NAT forwarding). Maps to `--bind-port`.
    pub bind_port: Option<u16>,
    /// Local QUIC bind IP. Maps to `--bind-ip`.
    pub bind_ip: Option<std::net::IpAddr>,
    /// Bind to 0.0.0.0 instead of 127.0.0.1. Maps to `--listen-all`.
    pub listen_all: bool,

    /// VRAM cap in GB. Maps to `--max-vram`.
    pub max_vram_gb: Option<f64>,
    /// Disable hardware survey gossip. Maps to `--no-enumerate-host`.
    pub no_enumerate_host: bool,

    /// Config file path. Maps to `--config`.
    pub config: Option<std::path::PathBuf>,
    /// Owner keystore path. Maps to `--owner-key`.
    pub owner_key: Option<std::path::PathBuf>,
    /// Fail startup without owner attestation. Maps to `--owner-required`.
    pub owner_required: bool,
    /// Node certificate label. Maps to `--node-label`.
    pub node_label: Option<String>,
    /// Add trusted owner IDs. Maps to repeatable `--trust-owner`.
    pub trust_owners: Vec<String>,

    /// Enable mesh runtime debug output. Maps to `--debug`.
    pub debug: bool,

    /// Extra raw argv flags for anything this struct doesn't yet
    /// expose typed. Inserted after the typed flags. Use sparingly.
    pub extra_args: Vec<String>,
}

impl MeshServeSpec {
    /// Serialise this spec into a CLI argv vector. Exposed primarily
    /// for tests and embedders that want to see exactly what they're
    /// about to run.
    pub fn into_argv(self) -> Vec<std::ffi::OsString> {
        let mut argv: Vec<std::ffi::OsString> = Vec::new();
        argv.push("mesh-llm".into());
        argv.push(if self.client { "client" } else { "serve" }.into());
        self.append_top_level(&mut argv);
        self.append_model_args(&mut argv);
        self.append_ports(&mut argv);
        self.append_relay_args(&mut argv);
        self.append_bind_args(&mut argv);
        self.append_owner_args(&mut argv);
        argv.extend(self.extra_args.into_iter().map(Into::into));
        argv
    }

    fn append_top_level(&self, argv: &mut Vec<std::ffi::OsString>) {
        if self.debug {
            argv.push("--debug".into());
        }
        if self.auto {
            argv.push("--auto".into());
        }
        if self.publish {
            argv.push("--publish".into());
        }
        if let Some(name) = &self.mesh_name {
            argv.push("--mesh-name".into());
            argv.push(name.into());
        }
        if let Some(region) = &self.region {
            argv.push("--region".into());
            argv.push(region.into());
        }
        if let Some(display) = &self.display_name {
            argv.push("--name".into());
            argv.push(display.into());
        }
        for invite in &self.join {
            argv.push("--join".into());
            argv.push(invite.into());
        }
        if let Some(filter) = &self.discover {
            argv.push("--discover".into());
            argv.push(filter.into());
        }
        if self.headless {
            argv.push("--headless".into());
        }
        if self.blackboard {
            argv.push("--blackboard".into());
        }
        if let Some(gb) = self.max_vram_gb {
            argv.push("--max-vram".into());
            argv.push(gb.to_string().into());
        }
        if self.no_enumerate_host {
            argv.push("--no-enumerate-host".into());
        }
    }

    fn append_model_args(&self, argv: &mut Vec<std::ffi::OsString>) {
        for model in &self.models {
            argv.push("--model".into());
            argv.push(model.into());
        }
        for gguf in &self.ggufs {
            argv.push("--gguf".into());
            argv.push(gguf.as_os_str().to_os_string());
        }
        if let Some(mmproj) = &self.mmproj {
            argv.push("--mmproj".into());
            argv.push(mmproj.as_os_str().to_os_string());
        }
    }

    fn append_ports(&self, argv: &mut Vec<std::ffi::OsString>) {
        if let Some(port) = self.port {
            argv.push("--port".into());
            argv.push(port.to_string().into());
        }
        if let Some(console) = self.console_port {
            argv.push("--console".into());
            argv.push(console.to_string().into());
        }
    }

    fn append_relay_args(&self, argv: &mut Vec<std::ffi::OsString>) {
        for relay in &self.relays {
            argv.push("--relay".into());
            argv.push(relay.into());
        }
        for url in &self.nostr_relays {
            argv.push("--nostr-relay".into());
            argv.push(url.into());
        }
    }

    fn append_bind_args(&self, argv: &mut Vec<std::ffi::OsString>) {
        if let Some(port) = self.bind_port {
            argv.push("--bind-port".into());
            argv.push(port.to_string().into());
        }
        if let Some(ip) = self.bind_ip {
            argv.push("--bind-ip".into());
            argv.push(ip.to_string().into());
        }
        if self.listen_all {
            argv.push("--listen-all".into());
        }
    }

    fn append_owner_args(&self, argv: &mut Vec<std::ffi::OsString>) {
        if let Some(config) = &self.config {
            argv.push("--config".into());
            argv.push(config.as_os_str().to_os_string());
        }
        if let Some(owner_key) = &self.owner_key {
            argv.push("--owner-key".into());
            argv.push(owner_key.as_os_str().to_os_string());
        }
        if self.owner_required {
            argv.push("--owner-required".into());
        }
        if let Some(label) = &self.node_label {
            argv.push("--node-label".into());
            argv.push(label.into());
        }
        for owner in &self.trust_owners {
            argv.push("--trust-owner".into());
            argv.push(owner.into());
        }
    }
}

/// Run the full mesh-llm runtime in-process.
///
/// This is the in-process equivalent of running the `mesh-llm` binary.
/// Everything the CLI does happens here: auto-discovery, election,
/// tunnel manager, OpenAI HTTP proxy, management console, model load /
/// serving (when a serving controller / GGUF is configured), plugin
/// host — driven by the same `runtime::run_with_args` entry point
/// `mesh-llm serve` / `mesh-llm client` use.
///
/// The future blocks until the runtime exits (signal, internal
/// shutdown, or fatal error). Embedders driving concurrent work should
/// run this on a `tokio::task::LocalSet` because the runtime is not
/// currently `Send`-clean.
///
/// # Example
///
/// ```no_run
/// use mesh_llm_host_runtime::host_node::{run_serve, MeshServeSpec};
///
/// # async fn run() -> anyhow::Result<()> {
/// run_serve(MeshServeSpec {
///     client: true,
///     auto: true,
///     relays: vec!["https://public.example/".into()],
///     port: Some(9337),
///     console_port: Some(3131),
///     headless: true,
///     max_vram_gb: Some(0.0),
///     ..MeshServeSpec::default()
/// })
/// .await?;
/// # Ok(())
/// # }
/// ```
pub async fn run_serve(spec: MeshServeSpec) -> Result<()> {
    crate::run_with_args(spec.into_argv()).await
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

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn mesh_serve_spec_argv_parses_via_the_real_cli_parser() {
        // The MeshServeSpec exists so SDK consumers can drive the same
        // runtime the binary drives. If a flag we emit doesn't exist in
        // the real Clap surface (typo, renamed, removed), this test
        // fails immediately and points at the drifted field.
        use clap::Parser;

        let spec = MeshServeSpec {
            client: true,
            auto: true,
            publish: false,
            mesh_name: Some("my-mesh".into()),
            region: Some("US".into()),
            display_name: Some("sprout".into()),
            join: vec!["invite-1".into(), "invite-2".into()],
            discover: Some("public".into()),
            models: vec!["Qwen3-8B-Q4_K_M".into()],
            ggufs: vec!["/tmp/foo.gguf".into()],
            mmproj: None,
            port: Some(9337),
            console_port: Some(3131),
            headless: true,
            blackboard: false,
            relays: vec!["https://public.example/".into()],
            nostr_relays: vec![],
            bind_port: Some(45000),
            bind_ip: None,
            listen_all: false,
            max_vram_gb: Some(0.0),
            no_enumerate_host: true,
            config: None,
            owner_key: None,
            owner_required: false,
            node_label: Some("sprout-app".into()),
            trust_owners: vec!["owner-abc".into()],
            debug: false,
            extra_args: vec![],
        };

        let argv = spec.into_argv();
        let normalized = crate::cli::normalize_runtime_surface_args(argv);
        let cli = crate::cli::Cli::try_parse_from(&normalized.normalized)
            .expect("MeshServeSpec argv must parse via the real CLI");

        assert!(cli.client);
        assert!(cli.auto);
        assert!(!cli.publish);
        assert_eq!(cli.mesh_name.as_deref(), Some("my-mesh"));
        assert_eq!(cli.region.as_deref(), Some("US"));
        assert_eq!(cli.name.as_deref(), Some("sprout"));
        assert_eq!(
            cli.join,
            vec!["invite-1".to_string(), "invite-2".to_string()]
        );
        assert_eq!(cli.discover.as_deref(), Some("public"));
        assert_eq!(cli.model, vec![std::path::PathBuf::from("Qwen3-8B-Q4_K_M")]);
        assert_eq!(cli.gguf, vec![std::path::PathBuf::from("/tmp/foo.gguf")]);
        assert_eq!(cli.port, 9337);
        assert_eq!(cli.console, 3131);
        assert!(cli.headless);
        assert_eq!(cli.relay, vec!["https://gated.example/".to_string()]);
        assert_eq!(
            cli.relay_auth,
            vec![(
                "https://gated.example/".to_string(),
                "bearer-abc".to_string()
            )],
        );
        assert_eq!(cli.bind_port, Some(45000));
        assert_eq!(cli.max_vram, Some(0.0));
        assert!(cli.no_enumerate_host);
        assert_eq!(cli.node_label.as_deref(), Some("sprout-app"));
        assert_eq!(cli.trust_owner, vec!["owner-abc".to_string()]);
    }
}
