use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};
use tokio::net::TcpStream;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    match (method, path) {
        ("POST", "/api/placement/prefetch") => handle_prefetch(stream, state, body).await,
        ("GET", "/api/placement/cache") => handle_cache(stream, state).await,
        ("POST", "/api/placement/pin") => handle_pin(stream, state, body).await,
        ("POST", "/api/placement/evict") => handle_evict(stream, state, body).await,
        ("GET", path) if path.starts_with("/api/placement/status/") => {
            handle_status(stream, state, path).await
        }
        _ => respond_error(stream, 404, "Not found").await,
    }
}

async fn handle_prefetch(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let request: hydra::PlacementPrefetchRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(err) => return respond_error(stream, 400, &format!("Invalid JSON body: {err}")).await,
    };
    match state.placement_prefetch(request).await {
        Ok(snapshot) => respond_json(stream, 202, &snapshot).await,
        Err(err) => respond_error(stream, 502, &err.to_string()).await,
    }
}

async fn handle_status(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some(operation_id) = decode_path_suffix(path, "/api/placement/status/") else {
        return respond_error(stream, 400, "Missing operation id").await;
    };
    match state.placement_status(&operation_id).await {
        Some(snapshot) => respond_json(stream, 200, &snapshot).await,
        None => respond_error(stream, 404, "Placement operation not found").await,
    }
}

async fn handle_cache(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    respond_json(stream, 200, &state.placement_cache().await).await
}

async fn handle_pin(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let request: hydra::PlacementPinRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(err) => return respond_error(stream, 400, &format!("Invalid JSON body: {err}")).await,
    };
    respond_json(stream, 200, &state.placement_pin(request).await).await
}

async fn handle_evict(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let request: hydra::PlacementEvictRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(err) => return respond_error(stream, 400, &format!("Invalid JSON body: {err}")).await,
    };
    match state.placement_evict(request).await {
        Ok(snapshot) => respond_json(stream, 200, &snapshot).await,
        Err(err) => respond_error(stream, 502, &err.to_string()).await,
    }
}

fn decode_path_suffix(path: &str, prefix: &str) -> Option<String> {
    let suffix = path.strip_prefix(prefix)?.trim();
    if suffix.is_empty() {
        return None;
    }
    urlencoding::decode(suffix)
        .ok()
        .map(|decoded| decoded.into_owned())
}
