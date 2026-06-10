//! # mesh-mlx
//!
//! A native Rust MLX runtime for Apple Silicon — **no Python, no Swift**. It
//! links the MLX C++/Metal engine through its C API (`mlx-c`, via
//! [`mesh_mlx_sys`]) and implements LLM inference in Rust: model forward passes,
//! safetensors loading, tokenization, generation, and **distributed
//! (pipeline / tensor) execution** using MLX's own collectives.
//!
//! The MLX C++ engine does all compute and all networking (ring/TCP, JACCL/RDMA
//! over Thunderbolt). Rust is the orchestration layer — the same role Python
//! `mlx-lm` plays — but compiled, single-language, and embeddable in the mesh
//! binary.
//!
//! ## Layout
//! - [`array`], [`ops`], [`nn`] — safe RAII wrappers + transformer building
//!   blocks over the engine.
//! - [`distributed`] — process [`distributed::Group`] + collectives, and
//!   [`distributed::Pipeline`] layer assignment.
//! - [`models`] — config + forward passes (Llama / Mistral / Qwen2 / Qwen3).
//! - [`loader`], [`download`] — selective safetensors download + load.
//! - [`runtime`] — tokenizer, generation, and the high-level [`runtime::Engine`].
//! - [`mesh`] — latency-aware parallelism planner + transport plan for mesh
//!   orchestration (local-only; MLX cannot use mesh QUIC).
//!
//! ## Features
//! - `link-mlx` — build and link the native MLX engine (Apple Silicon).
//!   Without it the crate type-checks and unit-tests pure logic in CI without a
//!   Metal build; real inference requires the feature.

pub mod array;
pub mod distributed;
pub mod download;
pub mod loader;
pub mod mesh;
pub mod models;
pub mod nn;
pub mod ops;
pub mod runtime;

pub use array::{Array, Dtype, Stream};
pub use distributed::{Backend, Group, Pipeline};
pub use download::ModelRef;
pub use mesh::{
    LatencySample, MlxBackendKind, NodeEndpoint, ParallelismMode, ParallelismPlan,
    ParallelismPlanner, TransportPlan,
};
pub use models::{Family, ModelConfig};
pub use runtime::Engine;

/// Errors from the MLX runtime.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    /// A native MLX engine op returned a non-zero status.
    #[error("mlx engine error: {0}")]
    Engine(String),

    /// A tensor shape was invalid for the requested op.
    #[error("shape error: {0}")]
    Shape(String),

    /// A required weight tensor was not found in the loaded model.
    #[error("missing weight: {0}")]
    MissingWeight(String),

    /// Model load / config parse failure.
    #[error("load error: {0}")]
    Load(String),

    /// Hugging Face download failure.
    #[error("download error: {0}")]
    Download(String),

    /// Tokenizer load / encode / decode failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Distributed init / collective failure.
    #[error("distributed error: {0}")]
    Distributed(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, MlxError>;
