//! End-to-end test: SDK consumer drives a real iroh-backed mesh node
//! through the published `MeshNodeBuilder` API, registers with a gated
//! iroh relay using `--relay-auth`-equivalent setters, and reaches
//! `online()`.
//!
//! Gated on the `host-runtime` feature — only meaningful when the SDK
//! actually starts a mesh node, not the HTTP-shim default.

#![cfg(feature = "host-runtime")]

use futures_util::StreamExt;
use iroh::endpoint::{presets, Endpoint, RelayMode};
use iroh::test_utils::run_relay_server_with_access;
use iroh::{RelayConfig as IrohRelayConfig, RelayMap, SecretKey, Watcher};
use iroh_relay::server::{Access, AccessConfig};
use iroh_relay::tls::CaRootsConfig;
use mesh_llm_api_server::{InviteToken, MeshNode, MeshQuicBind, MeshRole, OwnerKeypair};
use mesh_llm_host_runtime::host_node::{start_host_node, HostNodeSpec, MeshNodeRole};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::time::Duration;

/// Spawn an in-process iroh-relay that only admits `expected_token`.
async fn spawn_gated_relay(expected_token: &'static str) -> (String, iroh_relay::server::Server) {
    let access = AccessConfig::Restricted(Box::new(move |request| {
        Box::pin(async move {
            if request.auth_token().as_deref() == Some(expected_token) {
                Access::Allow
            } else {
                Access::Deny
            }
        })
    }));
    let (_relay_map, relay_url, server) = run_relay_server_with_access(false, access)
        .await
        .expect("spawn gated relay");
    (relay_url.to_string(), server)
}

/// Start an anchor `HostNode` pointed at the given relay so the test
/// stays offline / deterministic. Without an explicit relay, the anchor
/// would fall back to the bundled public relays and depend on the
/// internet being reachable. Optionally takes an auth token if the
/// target relay is gated (`AccessConfig::Restricted`).
async fn anchor_on_relay(
    relay_url: &str,
    relay_token: Option<&str>,
) -> (String, mesh_llm_host_runtime::host_node::HostNode) {
    let mut relay_auths = std::collections::HashMap::new();
    if let Some(token) = relay_token {
        relay_auths.insert(relay_url.to_string(), token.to_string());
    }
    let anchor = start_host_node(HostNodeSpec {
        role: MeshNodeRole::Client,
        relays: vec![relay_url.to_string()],
        relay_auths,
        max_vram_gb: Some(0.0),
        enumerate_host: false,
        ..HostNodeSpec::default()
    })
    .await
    .expect("anchor host node should start");
    anchor.start_accepting();
    (anchor.invite_token(), anchor)
}

fn free_local_udp_port() -> u16 {
    let socket =
        UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("allocate local UDP port");
    socket.local_addr().expect("read local UDP port").port()
}

async fn probe_quic_port_released(port: u16) {
    let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let mut last_error = String::new();

    for _ in 0..20 {
        match Endpoint::builder(presets::Minimal)
            .secret_key(SecretKey::generate())
            .relay_mode(RelayMode::Disabled)
            .bind_addr(bind_addr)
            .expect("configure probe bind address")
            .bind()
            .await
        {
            Ok(endpoint) => {
                endpoint.close().await;
                return;
            }
            Err(err) => {
                last_error = format!("{err:#}");
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        }
    }

    panic!("host-node cleanup should release UDP port {port}: {last_error}");
}

#[tokio::test]
async fn mesh_node_builder_threads_relay_auth_to_real_iroh_endpoint() {
    const TOKEN: &str = "secret-bearer-token";
    let (gated_url, _server) = spawn_gated_relay(TOKEN).await;
    let (invite, anchor) = anchor_on_relay(&gated_url, Some(TOKEN)).await;
    let invite_token: InviteToken = invite.parse().expect("parse invite");

    let node = MeshNode::builder()
        .identity(OwnerKeypair::generate())
        .join(invite_token)
        .role(MeshRole::Client)
        .relay(&gated_url)
        .relay_auth(&gated_url, TOKEN)
        .max_vram_gb(0.0)
        .build()
        .expect("builder");

    // start() should bring the underlying iroh endpoint online via the
    // gated relay (admitted because we passed the matching token). We
    // don't reach Ok unless the host-runtime path is actually used.
    tokio::time::timeout(Duration::from_secs(10), node.start())
        .await
        .expect("MeshNode.start() should resolve within 10s")
        .expect("MeshNode.start() should succeed when relay_auth matches");

    // Invite token from the running mesh node is non-empty.
    let invite = node
        .invite_token()
        .await
        .expect("invite_token should be populated after start");
    assert!(!invite.is_empty(), "invite token must not be empty");

    node.stop().await.expect("stop");
    anchor.shutdown().await;
}

#[tokio::test]
async fn mesh_node_start_fails_when_host_runtime_join_token_is_invalid() {
    let quic_port = free_local_udp_port();
    let invite_token: InviteToken = "mesh-test:not-base64".parse().expect("parse invite");

    let node = MeshNode::builder()
        .identity(OwnerKeypair::generate())
        .join(invite_token)
        .role(MeshRole::Client)
        .quic_bind(MeshQuicBind {
            ip: Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            port: Some(quic_port),
        })
        .max_vram_gb(0.0)
        .build()
        .expect("builder");

    let start = tokio::time::timeout(Duration::from_secs(10), node.start())
        .await
        .expect("MeshNode.start() should resolve within 10s");

    assert!(
        start.is_err(),
        "host-runtime start must attempt the configured join token"
    );

    assert!(
        node.invite_token().await.is_none(),
        "failed host-runtime starts must not store a node"
    );
    probe_quic_port_released(quic_port).await;
}

#[tokio::test]
async fn wrong_relay_token_is_rejected_by_gated_relay() {
    // We can't observe relay-level denial through the SDK's surface
    // yet, so verify the underlying iroh wire path directly: build an
    // endpoint with the same relay map shape the SDK would build,
    // but with the WRONG token, and assert iroh surfaces
    // `not authorized` via `home_relay_status`.
    //
    // This is the property defended on the runtime side; it's what
    // keeps the SDK's relay_auth setter honest end-to-end.
    const TOKEN: &str = "secret-bearer-token";
    let (gated_url, _server) = spawn_gated_relay(TOKEN).await;

    let parsed: iroh::RelayUrl = gated_url.parse().expect("parse url");
    let cfg = IrohRelayConfig::new(parsed, None).with_auth_token("wrong-token");
    let map: RelayMap = RelayMap::from_iter([cfg]);

    let ep = Endpoint::builder(presets::Minimal)
        .secret_key(SecretKey::generate())
        .relay_mode(RelayMode::Custom(map))
        .ca_roots_config(CaRootsConfig::insecure_skip_verify())
        .bind()
        .await
        .expect("endpoint bind");

    let mut stream = ep.home_relay_status().stream();
    let auth_err = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(status) = stream.next().await {
            if let Some(err) = status.iter().filter_map(|s| s.last_error()).next() {
                return Some(format!("{err:#}"));
            }
        }
        None
    })
    .await
    .expect("home relay status within 5s")
    .expect("home relay status should yield an error");
    assert!(
        auth_err.contains("not authorized"),
        "wrong token must be denied by gated relay, got: {auth_err}"
    );
}
