#![recursion_limit = "256"]

mod api;
mod capture;
mod cli;
pub mod crypto;
mod inference;
mod mesh;
mod models;
mod network;
mod plugin;
mod plugins;
mod protocol;
mod runtime;
mod runtime_data;
mod system;

pub mod host_node;
pub mod sdk;

pub mod proto {
    pub use mesh_llm_protocol::proto::*;
}

pub(crate) use plugins::blackboard;

use anyhow::Result;
use std::time::Duration;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub async fn run() -> Result<()> {
    runtime::run().await
}

/// Run the full mesh-llm runtime with a caller-supplied argv.
///
/// Equivalent to `run()` except the argv comes from the caller instead
/// of `std::env::args_os()`. This is the SDK entry point for embedders
/// who want to run the same code path the binary runs — full
/// `mesh-llm serve` / `mesh-llm client` behaviour, including auto-discover,
/// local model serving (when configured), election, tunnel manager,
/// OpenAI proxy, and management console — from inside their own Rust
/// application.
///
/// Build the argv from a typed config via
/// [`host_node::MeshServeSpec`][crate::host_node::MeshServeSpec] for
/// type-safety, or pass a `Vec<&str>` directly if you want raw control.
///
/// The future returned blocks until the runtime exits. Embedders
/// driving concurrent work should use `tokio::task::LocalSet` (the
/// runtime is not currently `Send`-clean).
pub async fn run_with_args<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString>,
{
    runtime::run_with_args(args).await
}

pub async fn run_main() -> i32 {
    match run().await {
        Ok(()) => 0,
        Err(err) => {
            let _ = cli::output::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}

#[cfg(test)]
include!("exact_test_wrappers.rs");
