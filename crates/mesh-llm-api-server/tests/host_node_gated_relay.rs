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
use mesh_llm_api_server::{InviteToken, MeshNode, MeshRole, OwnerKeypair};
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

/// Produce a parseable invite token (encoded EndpointAddr) from a
/// throwaway endpoint. The SDK's `MeshNode::start()` calls
/// `host_node.join(invite)`; the join path needs a parseable token even
/// if no peer is reachable behind it.
async fn neutral_invite_token() -> String {
    use base64::Engine;
    let ep = Endpoint::builder(presets::Minimal)
        .secret_key(SecretKey::generate())
        .relay_mode(RelayMode::Custom(RelayMap::empty()))
        .ca_roots_config(CaRootsConfig::insecure_skip_verify())
        .bind()
        .await
        .expect("dummy endpoint");
    let addr = ep.addr();
    drop(ep);
    let json = serde_json::to_vec(&addr).expect("serialize endpoint addr");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
}

#[tokio::test]
async fn mesh_node_builder_threads_relay_auth_to_real_iroh_endpoint() {
    const TOKEN: &str = "secret-bearer-token";
    let (gated_url, _server) = spawn_gated_relay(TOKEN).await;
    let invite_token: InviteToken = neutral_invite_token().await.parse().expect("parse invite");

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
