# mesh-mlx — native Rust MLX runtime (no Python, no Swift)

Status: **working single-node** on branch `micn/mlx-distributed`. Verified
end-to-end on Apple Silicon: downloads a safetensors model from Hugging Face,
loads it into the native MLX Metal engine, and generates tokens entirely from
Rust over `mlx-c` — no Python, no Swift. Distributed (pipeline/tensor) wiring
and quantized-weight support are the next steps (see "Status & roadmap" below).

`mesh-mlx` lets an all-Mac mesh run inference on **MLX** entirely from Rust. It
links the MLX C++ engine through its C API (`mlx-c`) and implements model
forward passes, weight loading, generation, and **distributed (pipeline /
tensor) inference** directly in Rust. There is **no Python and no Swift at build
or runtime** — the only native dependency is the MLX C++/Metal engine, the same
engine Python `mlx-lm` and Swift `mlx-swift-lm` sit on top of.

## Why this shape (the research that led here)

See `MLX_PARALLELISM_RESEARCH.md` for the full evidence. The load-bearing facts:

1. **The engine is C++ and language-agnostic.** Collectives (`all_sum`,
   `all_gather`), point-to-point (`send`, `recv`, `recv_like`), the ring/JACCL
   transports, every matmul/RoPE/SDPA kernel — all live in MLX's C++ core and
   are exposed through the stable **C API** (`mlx/c/distributed.h`,
   `distributed_group.h`, etc.). Confirmed in `ml-explore/mlx-c` v0.6.0.
2. **Python is not in the hot path.** A Python sharded layer's forward is
   literally `x @ W.T; mx.distributed.all_sum(x)` — three lines that each
   dispatch into C++. Python only *describes* the op sequence; the engine does
   all compute and all networking. Swift `mlx-swift-lm` is the same: thin glue
   over the same kernels.
3. **Forward passes are mechanical transcription.** Python `mlx-lm` and Swift
   `mlx-swift-lm` define the *same* models line-for-line in two languages
   (verified on Llama attention/MLP). A third transcription into Rust over
   `mlx-c` is rote translation with two reference implementations to copy.
4. **Distribution is small and not done outside Python.** Only ~18 Python models
   implement tensor `shard()` and ~7 implement `PipelineMixin`; Swift's
   distributed path is stubbed (mlx-swift even *excludes* `distributed.cpp` from
   its build). So whatever language we pick, the distributed wiring is ours to
   write — and it is small, because each collective is a single C call.

Conclusion: do it all in Rust over `mlx-c`. Reuse the C++ engine + C distributed
primitives; transcribe a short list of forward passes for the families worth
running; write the distributed wiring once.

## Crates

```
crates/mesh-mlx-sys/   FFI bindings to mlx-c + native build/link (build.rs)
crates/mesh-mlx/       safe Rust API: arrays, NN ops, distributed group/
                       collectives, model zoo (Llama/Qwen…), safetensors
                       loader, tokenizer, generate, OpenAI server, and the
                       mesh-facing backend (latency-aware planner + transport)
```

### `mesh-mlx-sys`
- Raw `extern "C"` declarations mirroring the `mlx-c` headers (array, stream,
  ops, fast, random, io, distributed, distributed_group).
- `build.rs` clones/builds MLX + mlx-c (CMake, Metal) and links the static libs,
  **gated behind the `link-mlx` feature** so the bindings crate type-checks in
  CI without a 30-minute Metal build. The native build is an opt-in artifact,
  matching how the repo treats the patched llama.cpp ABI.

### `mesh-mlx`
- `array`, `ops`, `nn` — safe wrappers over the sys layer (RAII for
  `mlx_array`/`mlx_stream`, matmul/SDPA/RoPE/silu/rms_norm/etc.).
- `distributed/` — `Group` (init/rank/size/split), `all_sum`, `all_gather`,
  `send`, `recv_like`; the **sharded linear** (tensor) and **pipeline** (layer
  split + send/recv) building blocks.
- `models/` — `Model` trait + per-family forward passes (start: Llama, Qwen3),
  each with optional `shard()` (tensor) and `pipeline()` (pipeline) like the
  Python references.
- `loader/` — safetensors selective download from HF (only the shards a stage
  needs, mirroring `mlx-lm.sharded_load`), config parsing, weight mapping.
- `runtime/` — tokenizer, sampling, KV cache, `generate`, OpenAI-compatible
  HTTP server (single process; rank 0 serves, workers run the pipeline/tensor
  group).
- `mesh/` — the mesh-facing surface: latency-aware `ParallelismPlanner`
  (tensor when worst inter-node RTT ≤ threshold, else pipeline), `TransportPlan`
  (LAN ring vs Thunderbolt JACCL), typed config. **Local-only** — MLX cannot use
  mesh QUIC and tunnelling would defeat its latency goal, so mesh forms a group
  only from Apple-Silicon, MLX-capable, directly-routable peers.

## Distributed model

- **Pipeline** (default over Ethernet): split layers contiguously across ranks;
  each rank `recv_like`s the activation from the next rank, runs its layers,
  `send`s to the previous rank; rank 0 finishes with `all_gather`. One activation
  per stage boundary — latency tolerant.
- **Tensor** (needs JACCL/Thunderbolt): sharded linears — `AllToSharded`
  (split output dim) and `ShardedToAll` (split input dim + `all_sum`), two
  all-reduces per transformer layer — latency bound.
- Mode chosen by the latency-aware planner from mesh's measured inter-node RTT.

## Networking

MLX opens its own TCP (ring) or RDMA (JACCL) sockets from a hostfile;
`mx.distributed.init` only accepts `{any, mpi, ring, nccl, jaccl}`. So mesh
supplies a hostfile of directly-routable peers and stays out of the activation
path. JACCL (RDMA over Thunderbolt 5) is required for good tensor parallel;
ring (TCP) over the LAN is the pipeline path.

## Build & test strategy

- Pure Rust logic (planner, transport, config, loader plumbing, model graph
  construction) compiles and unit-tests in CI **without** the native engine.
- The `link-mlx` feature builds the MLX engine and enables real inference; the
  end-to-end test (download a tiny safetensors model from HF, run single-node
  generate, assert non-empty output) runs on the macOS CI runner / a dev Mac
  under that feature. No Python.
- Without `link-mlx`, `mesh-mlx-sys` provides panicking stubs for the FFI
  symbols so the whole crate links and the pure-logic unit tests run on any
  platform in CI. The native Metal build only happens under `link-mlx`.

## Verified

`cargo test -p mesh-mlx --features link-mlx --test live_single_node` on an
Apple Silicon Mac downloads `mlx-community/Qwen2.5-0.5B-Instruct-bf16`, builds
the MLX Metal engine via `build.rs` (CMake FetchContent of `mlx-c` + `mlx`),
loads the safetensors weights, runs the Rust Llama/Qwen forward pass on Metal,
and returns a non-empty completion. Requires the Metal Toolchain
(`xcodebuild -downloadComponent MetalToolchain`).

## Status & roadmap

Done:
- `mesh-mlx-sys` FFI + gated native build/link (verified linking real engine).
- Safe array/ops/nn layer; Llama / Qwen2 / Qwen3 forward pass.
- Selective safetensors download + load; tokenizer; greedy generate.
- Single-node inference verified end-to-end on Metal.
- Latency-aware planner + transport plan (pure logic, unit-tested).
- Pipeline layer-assignment + collectives wrappers (unit-tested; multi-node
  execution wiring pending).

Next:
- Quantized weights (4-bit MLX-community models): dequantize / use MLX quantized
  matmul so the common `*-4bit` repos load. (bf16/fp16 work today.)
- Multi-node execution: drive the pipeline send/recv loop across ranks and the
  tensor sharded-linear path with a live `Group`.
- Full Jinja `chat_template` support (currently a ChatML-compatible framing).
- Sampling beyond greedy (temperature / top-p).
