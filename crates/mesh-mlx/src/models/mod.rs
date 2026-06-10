//! Model configs and forward passes.
//!
//! Initial family coverage: Llama / Mistral / Qwen2 / Qwen3 — the common
//! dense-transformer shape that covers most small/mid models worth running on a
//! Mac mesh. Additional families are added by transcribing their reference
//! forward pass (see `llama.rs` for the pattern).

mod config;
mod llama;

pub use config::{Family, ModelConfig};
pub use llama::{LayerCache, LlamaModel};
