//! Minimal OpenAI-compatible HTTP server backed by an [`Engine`].
//!
//! Exposes `/v1/models` and `/v1/chat/completions` (non-streaming) so mesh can
//! route OpenAI traffic to an MLX node exactly as it does for other backends.
//! Kept self-contained (own request/response types) so `mesh-mlx` doesn't couple
//! to internal API crates.
//!
//! Generation runs on the engine's single GPU stream, so requests are serialised
//! behind a mutex — one MLX node serves one request at a time (correct, simple;
//! batching is a later optimisation).

use crate::runtime::Engine;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// The inference backend behind the server: single-node or a distributed group.
enum Backend {
    Single(Engine),
    Distributed(crate::runtime::DistributedEngine),
}

impl Backend {
    fn chat(&self, system: Option<&str>, user: &str, max_tokens: usize) -> crate::Result<String> {
        match self {
            Backend::Single(e) => e.chat(system, user, max_tokens),
            Backend::Distributed(d) => d.chat(system, user, max_tokens),
        }
    }
}

/// Shared server state: the loaded backend (serialised) + the served model id.
#[derive(Clone)]
pub struct ServerState {
    backend: Arc<Mutex<Backend>>,
    model_id: String,
}

impl ServerState {
    /// Single-node backend.
    pub fn new(engine: Engine, model_id: impl Into<String>) -> Self {
        ServerState {
            backend: Arc::new(Mutex::new(Backend::Single(engine))),
            model_id: model_id.into(),
        }
    }

    /// Distributed backend (this node's rank participates in a group). The
    /// chat path drives the group in lock-step.
    pub fn distributed(
        engine: crate::runtime::DistributedEngine,
        model_id: impl Into<String>,
    ) -> Self {
        ServerState {
            backend: Arc::new(Mutex::new(Backend::Distributed(engine))),
            model_id: model_id.into(),
        }
    }
}

/// Build the OpenAI-compatible router.
pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(|| async { "ok" }))
        .with_state(state)
}

/// Serve the OpenAI API on `addr` until the process exits.
pub async fn serve(state: ServerState, addr: std::net::SocketAddr) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router(state))
        .await
        .map_err(std::io::Error::other)
}

/// A running server: the bound address (with the real port when `:0` was used)
/// plus a shutdown trigger.
pub struct ServerHandle {
    pub addr: std::net::SocketAddr,
    shutdown: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl ServerHandle {
    /// The bound local address (host:port).
    pub fn addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    /// The bound port.
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// Signal shutdown and wait for the server task to finish.
    pub async fn shutdown(self) {
        let _ = self.shutdown.send(());
        let _ = self.task.await;
    }
}

/// Bind and spawn the server in the background, returning a handle with the
/// real bound address. Pass `127.0.0.1:0` to get an ephemeral port (the pattern
/// mesh's local backends use).
pub async fn spawn(
    state: ServerState,
    addr: std::net::SocketAddr,
) -> std::io::Result<ServerHandle> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, router(state))
            .with_graceful_shutdown(async {
                let _ = rx.await;
            })
            .await;
    });
    Ok(ServerHandle {
        addr: bound,
        shutdown: tx,
        task,
    })
}

// ---- OpenAI wire types (minimal subset) ----

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
struct Choice {
    index: usize,
    message: ChatMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
}

#[derive(Debug, Serialize)]
struct ApiError {
    error: ApiErrorBody,
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    message: String,
}

async fn list_models(State(state): State<ServerState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model",
        }],
    })
}

async fn chat_completions(
    State(state): State<ServerState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    // Split the most recent user message + any system message for our minimal
    // chat template.
    let system = req
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone());
    let user = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let backend = state.backend.lock().await;
    let result = backend.chat(system.as_deref(), &user, req.max_tokens);
    drop(backend);

    match result {
        Ok(text) => Json(ChatResponse {
            id: format!("chatcmpl-{}", short_id()),
            object: "chat.completion",
            model: if req.model.is_empty() {
                state.model_id.clone()
            } else {
                req.model
            },
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: text,
                },
                finish_reason: "stop",
            }],
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError {
                error: ApiErrorBody {
                    message: e.to_string(),
                },
            }),
        )
            .into_response(),
    }
}

/// A short pseudo-unique id (timestamp-based; ids need not be globally unique).
fn short_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_parses_messages_and_defaults() {
        let json = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.messages[0].role, "user");
    }

    #[test]
    fn response_serialises_openai_shape() {
        let resp = ChatResponse {
            id: "chatcmpl-x".into(),
            object: "chat.completion",
            model: "m".into(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "hello".into(),
                },
                finish_reason: "stop",
            }],
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hello");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }
}
