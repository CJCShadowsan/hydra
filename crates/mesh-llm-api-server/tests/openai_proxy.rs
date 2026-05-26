//! End-to-end test: SDK consumer asks `MeshNodeBuilder` to spin up an
//! OpenAI HTTP proxy alongside the in-process mesh node, and we hit it
//! over real TCP/HTTP.
//!
//! Gated on the `host-runtime` feature.

#![cfg(feature = "host-runtime")]

use mesh_llm_api_server::{InviteToken, MeshNode, MeshRole, OwnerKeypair};
use mesh_llm_host_runtime::host_node::{start_host_node, HostNodeSpec, MeshNodeRole};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

async fn anchor_invite_token() -> (String, mesh_llm_host_runtime::host_node::HostNode) {
    let anchor = start_host_node(HostNodeSpec {
        role: MeshNodeRole::Client,
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
async fn http_get_status(host_port: &str, path: &str) -> String {
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
    let text = String::from_utf8_lossy(&buf).to_string();
    text
}

#[tokio::test]
async fn openai_proxy_binds_and_serves_v1_models_over_http() {
    let (invite, _anchor) = anchor_invite_token().await;
    let invite_token: InviteToken = invite.parse().expect("parse invite");

    let node = MeshNode::builder()
        .identity(OwnerKeypair::generate())
        .join(invite_token)
        .role(MeshRole::Client)
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
    let response = http_get_status(host_port, "/v1/models").await;
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
    // Body is OpenAI-style models response. Strip optional chunked-encoding
    // length lines and accept either `{"object":"list",…}` or `[…]`.
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
                connect_after_stop.unwrap().read_u8()
            )
            .await
            .is_ok(), // EOF on a half-shut connection is fine too.
        "OpenAI proxy port {host_port} should be closed after MeshNode::stop()"
    );
}
