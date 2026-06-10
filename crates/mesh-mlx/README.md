# mesh-mlx

Native **Rust** MLX runtime for Apple Silicon — **no Python, no Swift**. Links
the MLX C++/Metal engine through its C API (`mlx-c`) and implements LLM
inference in Rust: model forward passes, safetensors loading, tokenization,
generation, and distributed (pipeline / tensor) primitives.

The MLX C++ engine does all compute and all networking (ring/TCP, JACCL/RDMA
over Thunderbolt). Rust is the orchestration layer — the same role Python
`mlx-lm` plays — but compiled, single-language, and embeddable in mesh-llm.

## Why (research summary)

The MLX distribution machinery (collectives, send/recv, ring/JACCL transports)
and all kernels live in the C++ core and are exposed via `mlx-c`. Python is not
in the hot path — a sharded layer's forward is literally `x @ Wᵀ;
all_sum(x)`, three C dispatches. Model forward passes are mechanical
transcriptions shared between Python `mlx-lm` and Swift `mlx-swift-lm`. So we do
it all in Rust over `mlx-c`: reuse the engine + collectives, transcribe a short
list of forward passes, write the small distributed wiring once. Full evidence:
`docs/design/MLX_PARALLELISM_RESEARCH.md`; architecture: `docs/design/MESH_MLX.md`.

## Layout

- `array`, `ops`, `nn` — safe RAII wrappers + transformer building blocks.
- `distributed` — process `Group` + collectives; `Pipeline` layer assignment.
- `models` — config + forward passes (Llama / Mistral / Qwen2 / Qwen3).
- `loader`, `download` — selective safetensors download + load.
- `runtime` — tokenizer, generate, high-level `Engine`.
- `mesh` — latency-aware parallelism planner + transport plan (local-only;
  MLX cannot use mesh QUIC).

## Features

- `link-mlx` — build and link the native MLX engine (Apple Silicon) for real
  inference. Without it, `mesh-mlx-sys` provides panicking stubs so the crate
  links and pure-logic unit tests run on any platform in CI (no Metal build).

## Try it (Apple Silicon)

```bash
xcodebuild -downloadComponent MetalToolchain   # one-time
cargo test -p mesh-mlx --features link-mlx --test live_single_node -- --nocapture
```

Downloads a small bf16 model from Hugging Face and generates tokens entirely in
Rust + MLX on Metal.

## Status

Single-node inference works end-to-end (verified). Quantized 4-bit weights and
multi-node pipeline/tensor execution are in progress — see
`docs/design/MESH_MLX.md`.
