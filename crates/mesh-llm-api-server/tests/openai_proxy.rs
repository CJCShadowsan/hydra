//! End-to-end test: SDK consumer asks `MeshNodeBuilder` to spin up an
//! OpenAI HTTP proxy alongside the in-process mesh node, and we hit it
//! over real TCP/HTTP.
//!
//! Gated on the `host-runtime` feature.

#![cfg(feature = "host-runtime")]

use iroh::test_utils::run_relay_server_with_access;
use iroh_relay::server::AccessConfig;
use mesh_llm_api_server::{InviteToken, MeshNode, MeshRole, OwnerKeypair};
use mesh_llm_host_runtime::host_node::{start_host_node, HostNode, HostNodeSpec, MeshNodeRole};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Spin up an in-process open relay so the test runs offline and
/// deterministic. Returns the relay URL string and the server (held by
/// the caller so it isn't dropped mid-test).
async fn spawn_open_relay() -> (String, iroh_relay::server::Server) {
    let (_relay_map, relay_url, server) =
        run_relay_server_with_access(false, AccessConfig::Everyone)
            .await
            .expect("spawn open relay");
    (relay_url.to_string(), server)
}

/// Start an anchor `HostNode` that the SDK node can join. Pointed at
/// the in-process relay so we don't reach the bundled defaults.
async fn anchor_on_relay(relay_url: &str) -> (String, HostNode) {
    let anchor = start_host_node(HostNodeSpec {
        role: MeshNodeRole::Client,
        relays: vec![relay_url.to_string()],
        max_vram_gb: Some(0.0),
        enumerate_host: false,
        ..HostNodeSpec::default()
    })
    .await
    .expect("anchor host node should start");
    anchor.start_accepting();
    (anchor.invite_token(), anchor)
}

/// Minimal HTTP GET against `host:port` returning the response status line.
/// Avoids pulling reqwest into dev-deps just for this smoke test.
async fn http_get(host_port: &str, path: &str) -> String {
    let mut stream = TcpStream::connect(host_port)
        .await
        .expect("connect to proxy");
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");
    let mut buf = Vec::with_capacity(1024);
    stream.read_to_end(&mut buf).await.expect("read response");
    String::from_utf8_lossy(&buf).to_string()
}

#[tokio::test]
async fn openai_proxy_binds_and_serves_v1_models_over_http() {
    let (relay_url, _relay_server) = spawn_open_relay().await;
    let (invite, anchor) = anchor_on_relay(&relay_url).await;
    let invite_token: InviteToken = invite.parse().expect("parse invite");

    let node = MeshNode::builder()
        .identity(OwnerKeypair::generate())
        .join(invite_token)
        .role(MeshRole::Client)
        .relay(&relay_url) // keep the SDK node on our in-process relay too
        .max_vram_gb(0.0)
        // Port 0 → OS-assigned ephemeral. The handle reports the real one.
        .openai_port(0)
        .build()
        .expect("builder");

    tokio::time::timeout(Duration::from_secs(60), node.start())
        .await
        .expect("MeshNode.start() should resolve within 60s")
        .expect("MeshNode.start() should succeed");

    let base = node
        .openai_base_url()
        .await
        .expect("openai_base_url should be populated after start with openai_port");
    assert!(base.starts_with("http://127.0.0.1:"), "base url: {base}");

    // host:port for our raw TCP probe.
    let host_port = base
        .strip_prefix("http://")
        .expect("base url has http:// prefix");

    // Hit /v1/models — should return 200 with a JSON body containing
    // the OpenAI shape. With no peers serving anything, `data` should be
    // an empty array but the endpoint itself must respond.
    let response = http_get(host_port, "/v1/models").await;
    let status_line = response.lines().next().unwrap_or_default();
    assert!(
        status_line.starts_with("HTTP/1.1 200"),
        "expected 200 OK from /v1/models, got status line {status_line:?}\n\nFull response:\n{response}"
    );
    let body_start = response
        .find("\r\n\r\n")
        .expect("response has body separator")
        + 4;
    let body = &response[body_start..];
    assert!(
        body.contains("\"data\""),
        "expected JSON body containing `data` field, got: {body}"
    );

    node.stop().await.expect("stop");

    // After stop(), the port should no longer answer.
    let connect_after_stop = TcpStream::connect(host_port).await;
    assert!(
        connect_after_stop.is_err()
            || tokio::time::timeout(
                Duration::from_secs(1),
                connect_after_stop.unwrap().read_u8(),
            )
            .await
            .is_ok(), // EOF on a half-shut connection is fine too.
        "OpenAI proxy port {host_port} should be closed after MeshNode::stop()"
    );

    // Explicitly shut the anchor down so its iroh endpoint + accept
    // loop don't leak into the next test. Dropping the handle alone is
    // explicitly not the shutdown contract for `HostNode`.
    anchor.shutdown().await;
}

#[tokio::test]
async fn start_is_idempotent_on_repeat_calls() {
    // Regression: a second .start() used to spawn a second iroh
    // endpoint + OpenAI proxy and orphan the first (stop() only knew
    // about the most recent). Now it's a no-op.
    let (relay_url, _relay_server) = spawn_open_relay().await;
    let (invite, anchor) = anchor_on_relay(&relay_url).await;
    let invite_token: InviteToken = invite.parse().expect("parse invite");

    let node = MeshNode::builder()
        .identity(OwnerKeypair::generate())
        .join(invite_token)
        .role(MeshRole::Client)
        .relay(&relay_url)
        .max_vram_gb(0.0)
        .openai_port(0)
        .build()
        .expect("builder");

    tokio::time::timeout(Duration::from_secs(60), node.start())
        .await
        .expect("first start within 60s")
        .expect("first start ok");

    let base_before = node.openai_base_url().await.expect("first base url");

    // Second start() must be a no-op: same proxy URL, no second bind.
    tokio::time::timeout(Duration::from_secs(5), node.start())
        .await
        .expect("second start should return immediately")
        .expect("second start ok");

    let base_after = node.openai_base_url().await.expect("second base url");
    assert_eq!(
        base_before, base_after,
        "second start() must not replace the running OpenAI proxy"
    );

    node.stop().await.expect("stop");
    anchor.shutdown().await;
}
