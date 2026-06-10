//! Real end-to-end single-node inference — **no Python, no Swift**.
//!
//! Gated behind `link-mlx` because it builds/links the native MLX Metal engine
//! and downloads a small safetensors model from Hugging Face on first run.
//!
//! Run on an Apple Silicon Mac (or the macOS CI runner):
//! ```bash
//! cargo test -p mesh-mlx --features link-mlx --test live_single_node -- --nocapture
//! ```
//!
//! Overrides:
//!   - `MLX_TEST_MODEL` — HF repo id (default: a tiny bf16 Qwen2.5).

#![cfg(feature = "link-mlx")]

use mesh_mlx::{Engine, ModelRef};

fn model_id() -> String {
    // Default to an unquantized (bf16) model — quantized 4-bit loading is a
    // follow-up. bf16/fp16 models work today.
    std::env::var("MLX_TEST_MODEL")
        .unwrap_or_else(|_| "mlx-community/Qwen2.5-0.5B-Instruct-bf16".to_string())
}

#[tokio::test]
async fn downloads_and_generates_real_tokens() {
    let model = ModelRef::new(model_id());

    // Selective download + load (single node => full repo) and build the engine.
    let engine = Engine::load_single(&model)
        .await
        .expect("load MLX model from HF");

    assert!(
        engine.config.num_hidden_layers > 0,
        "config should report layers"
    );

    // Generate a short greedy completion entirely in-process via MLX.
    let out = engine
        .chat(Some("You are concise."), "Say hello in one word.", 8)
        .expect("generation should succeed");

    assert!(
        !out.trim().is_empty(),
        "expected non-empty completion, got {out:?}"
    );
    eprintln!("MLX completion: {out:?}");
}
