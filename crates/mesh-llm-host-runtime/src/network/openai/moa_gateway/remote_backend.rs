use crate::mesh;
use mesh_mixture_of_agents as moa;

type RemoteJoinSet = tokio::task::JoinSet<(iroh::EndpointId, Result<serde_json::Value, String>)>;

pub(in crate::network::openai::moa_gateway) struct RemoteModelBackend {
    node: mesh::Node,
    peer_ids: Vec<iroh::EndpointId>,
    hedge_delay: std::time::Duration,
}

impl RemoteModelBackend {
    pub(in crate::network::openai::moa_gateway) fn new(
        node: mesh::Node,
        peer_ids: Vec<iroh::EndpointId>,
        hedge_delay: std::time::Duration,
    ) -> Self {
        Self {
            node,
            peer_ids,
            hedge_delay,
        }
    }
}

#[async_trait::async_trait]
impl moa::ModelBackend for RemoteModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let raw = build_chat_completion_request(model, messages, tools, max_tokens, sampling)?;
        call_remote_candidates(
            self.node.clone(),
            self.peer_ids.clone(),
            raw,
            timeout,
            self.hedge_delay,
        )
        .await
    }
}

fn build_chat_completion_request(
    model: &str,
    messages: &[serde_json::Value],
    tools: Option<&serde_json::Value>,
    max_tokens: u32,
    sampling: moa::SamplingParams,
) -> Result<Vec<u8>, String> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "stream": false,
        "mesh_hooks": false,
    });
    if let Some(tools) = tools {
        body.as_object_mut()
            .expect("chat completion body should be an object")
            .insert("tools".to_string(), tools.clone());
        body.as_object_mut()
            .expect("chat completion body should be an object")
            .insert("parallel_tool_calls".to_string(), serde_json::json!(false));
    }
    moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
    let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("serialize: {e}"))?;
    let http_request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         \r\n",
        body_bytes.len()
    );
    let mut raw = http_request.into_bytes();
    raw.extend_from_slice(&body_bytes);
    Ok(raw)
}

async fn call_remote_candidates(
    node: mesh::Node,
    peer_ids: Vec<iroh::EndpointId>,
    raw: Vec<u8>,
    timeout: std::time::Duration,
    hedge_delay: std::time::Duration,
) -> Result<serde_json::Value, String> {
    if peer_ids.is_empty() {
        return Err("remote: no peer candidates".to_string());
    }

    let mut remaining = peer_ids.into_iter();
    let mut join_set = tokio::task::JoinSet::new();
    let mut last_err = None;
    let mut attempts = 0u32;
    let mut remaining_exhausted = false;

    if let Some(peer_id) = remaining.next() {
        spawn_remote_attempt(&mut join_set, node.clone(), peer_id, raw.clone(), timeout);
        attempts += 1;
    }

    while !join_set.is_empty() {
        if remaining_exhausted {
            match next_remote_event(&mut join_set, true).await {
                RemoteAttemptEvent::Success(body) => {
                    finish_remote_attempts(&mut join_set).await;
                    return Ok(body);
                }
                RemoteAttemptEvent::Failure(err) => last_err = Some(err),
                RemoteAttemptEvent::Closed => break,
            }
            continue;
        }

        let hedge_sleep = tokio::time::sleep(hedge_delay);
        tokio::pin!(hedge_sleep);
        tokio::select! {
            event = next_remote_event(&mut join_set, false) => {
                match event {
                    RemoteAttemptEvent::Success(body) => {
                        finish_remote_attempts(&mut join_set).await;
                        return Ok(body);
                    }
                    RemoteAttemptEvent::Failure(err) => {
                        last_err = Some(err);
                        remaining_exhausted = !try_spawn_next_remote(
                            &mut join_set,
                            &node,
                            &mut remaining,
                            &raw,
                            timeout,
                            &mut attempts,
                        );
                    }
                    RemoteAttemptEvent::Closed => break,
                }
            }
            _ = &mut hedge_sleep => {
                remaining_exhausted = !try_spawn_next_remote(
                    &mut join_set,
                    &node,
                    &mut remaining,
                    &raw,
                    timeout,
                    &mut attempts,
                );
            }
        }
    }

    Err(format!(
        "remote: all {attempts} peer candidate(s) failed: {}",
        last_err.unwrap_or_else(|| "unknown error".to_string())
    ))
}

enum RemoteAttemptEvent {
    Success(serde_json::Value),
    Failure(String),
    Closed,
}

async fn next_remote_event(join_set: &mut RemoteJoinSet, exhausted: bool) -> RemoteAttemptEvent {
    match join_set.join_next().await {
        Some(Ok((_, Ok(body)))) => RemoteAttemptEvent::Success(body),
        Some(Ok((peer_id, Err(err)))) => {
            log_remote_failure(peer_id, &err, exhausted);
            RemoteAttemptEvent::Failure(err)
        }
        Some(Err(err)) => {
            tracing::warn!("MoA: remote backend task join error: {err}");
            RemoteAttemptEvent::Failure(err.to_string())
        }
        None => RemoteAttemptEvent::Closed,
    }
}

fn log_remote_failure(peer_id: iroh::EndpointId, err: &str, exhausted: bool) {
    if exhausted {
        tracing::warn!(
            "MoA: remote backend {} failed after hedge exhaustion: {err}",
            peer_id.fmt_short()
        );
    } else {
        tracing::warn!(
            "MoA: remote backend {} failed: {err}, trying next host",
            peer_id.fmt_short()
        );
    }
}

fn try_spawn_next_remote(
    join_set: &mut RemoteJoinSet,
    node: &mesh::Node,
    remaining: &mut impl Iterator<Item = iroh::EndpointId>,
    raw: &[u8],
    timeout: std::time::Duration,
    attempts: &mut u32,
) -> bool {
    let Some(peer_id) = remaining.next() else {
        return false;
    };
    spawn_remote_attempt(join_set, node.clone(), peer_id, raw.to_vec(), timeout);
    *attempts += 1;
    true
}

fn spawn_remote_attempt(
    join_set: &mut RemoteJoinSet,
    node: mesh::Node,
    peer_id: iroh::EndpointId,
    raw: Vec<u8>,
    timeout: std::time::Duration,
) {
    tracing::info!("MoA: remote backend candidate {}", peer_id.fmt_short());
    join_set.spawn(async move {
        let result = call_remote_peer(node, peer_id, raw, timeout).await;
        (peer_id, result)
    });
}

async fn finish_remote_attempts(join_set: &mut RemoteJoinSet) {
    join_set.abort_all();
    while join_set.join_next().await.is_some() {}
}

async fn call_remote_peer(
    node: mesh::Node,
    peer_id: iroh::EndpointId,
    raw: Vec<u8>,
    timeout: std::time::Duration,
) -> Result<serde_json::Value, String> {
    tokio::time::timeout(timeout, async {
        let (mut send, mut recv) = node
            .open_http_tunnel(peer_id)
            .await
            .map_err(|e| format!("tunnel {}: {e}", peer_id.fmt_short()))?;
        send.write_all(&raw)
            .await
            .map_err(|e| format!("send {}: {e}", peer_id.fmt_short()))?;
        send.finish()
            .map_err(|e| format!("finish {}: {e}", peer_id.fmt_short()))?;
        let response = recv
            .read_to_end(4 * 1024 * 1024)
            .await
            .map_err(|e| format!("recv {}: {e}", peer_id.fmt_short()))?;
        parse_quic_http_response(&response)
    })
    .await
    .map_err(|_| {
        format!(
            "remote {} timeout after {}s",
            peer_id.fmt_short(),
            timeout.as_secs()
        )
    })?
}

fn parse_quic_http_response(response: &[u8]) -> Result<serde_json::Value, String> {
    let s = String::from_utf8_lossy(response);
    let header_end = s
        .find("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line = s[..header_end].lines().next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status != 200 {
        return Err(format!("HTTP {status}: {}", moa::truncate_chars(&s, 200)));
    }
    let body = &s[header_end + 4..];
    serde_json::from_str(body).map_err(|e| format!("parse: {e}"))
}
