use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use skippy_server::{EmbeddedServerHandle, EmbeddedState};

pub(super) async fn wait_for_binary_stage_ready(
    server: Option<&EmbeddedServerHandle>,
    bind_addr: SocketAddr,
    timeout: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    let mut last_error = None;
    while Instant::now() < deadline {
        if let Some(error) = server_startup_error(server) {
            return Err(error.context(format!(
                "binary stage did not become ready at {bind_addr} before startup failed"
            )));
        }
        match probe_binary_stage_ready_once(bind_addr).await {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out waiting for binary stage ready at {bind_addr}"))
        .context(format!(
            "binary stage did not become ready at {bind_addr} before timeout"
        )))
}

async fn probe_binary_stage_ready_once(bind_addr: SocketAddr) -> Result<()> {
    tokio::task::spawn_blocking(move || probe_binary_stage_ready_blocking(bind_addr))
        .await
        .context("join binary stage readiness probe")?
}

fn probe_binary_stage_ready_blocking(bind_addr: SocketAddr) -> Result<()> {
    let mut stream = std::net::TcpStream::connect_timeout(&bind_addr, Duration::from_secs(1))
        .with_context(|| format!("connect binary stage listener at {bind_addr}"))?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(2))).ok();
    skippy_protocol::binary::recv_ready(&mut stream).context("binary stage ready handshake failed")
}

fn server_startup_error(server: Option<&EmbeddedServerHandle>) -> Option<anyhow::Error> {
    let status = server?.status();
    if status.state != EmbeddedState::Failed {
        return None;
    }
    Some(anyhow!(
        "{} failed during startup: {}",
        status.name,
        status.last_error.as_deref().unwrap_or("unknown error")
    ))
}
