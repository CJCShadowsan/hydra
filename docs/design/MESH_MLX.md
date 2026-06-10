# mesh-mlx — native Rust MLX runtime (no Python, no Swift)

Status: **code-complete** on branch `micn/mlx-distributed` (single-node verified;
multi-node execution pending hardware testing). Verified end-to-end on Apple
Silicon: downloads a safetensors model from Hugging Face, loads it into the
native MLX Metal engine, and generates coherent tokens entirely from Rust over
`mlx-c` — no Python, no Swift. Both **bf16 and quantized 4-bit** models produce
correct output ("What is the capital of France?" → "The capital of France is
Paris."). Pipeline + tensor parallel paths and the OpenAI server are implemented;
distributed execution is wired and unit-tested, awaiting a multi-node test rig.

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

mesh-llm-host-runtime/src/inference/mlx.rs   the backend integration: loads a
                       model and serves OpenAI on a local port; MlxModelHandle
                       mirrors the Skippy handle so mesh routes to it identically
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

Done (code-complete):
- `mesh-mlx-sys` FFI + gated native build/link (verified linking real engine).
- Safe array/ops/nn layer; Llama / Mistral / Qwen2 / Qwen3 forward pass.
- **Quantized 4-bit** weights: quantized matmul for linears + gather-then-
  dequantize for embeddings. bf16/fp16 and 4-bit both verified coherent.
- Selective safetensors download + load; tokenizer; greedy generate.
- **Single-node inference verified end-to-end on Metal** (correct answers).
- **OpenAI-compatible server** (`/v1/chat/completions`, `/v1/models`) + the
  `mlx-serve` binary mesh spawns/targets.
- **Pipeline-parallel** generate loop (`generate_distributed`: embed → recv →
  layers → send → head → broadcast) wired over the `Group` collectives.
- **Tensor-parallel** path: per-rank weight slicing (`shard_tensor_parallel`)
  + sharded attention/MLP with `all_sum` on row-parallel projections.
- Latency-aware planner + transport plan + `MlxOrchestrator` (mesh-facing
  decision surface). All pure logic unit-tested.

Wired into mesh (usable as a backend):
- `mesh-llm-host-runtime` depends on `mesh-mlx` and has an
  `inference::mlx::MlxModelHandle` that loads a model and serves the OpenAI API
  on an ephemeral local port (mirrors the Skippy HTTP handle: `port()` +
  `shutdown()`).
- `LocalRuntimeBackendHandle::Mlx` is a first-class backend variant; all handle
  methods (`pid`, `shutdown`, status, guardrails) handle it.
- `runtime::local::start_runtime_local_model` routes to
  `start_runtime_mlx_model` when `MlxModelHandle::available()` (Apple Silicon +
  `mlx-backend` feature) **and** the model is a safetensors directory
  (`is_mlx_safetensors_model`). GGUF / layer packages fall through to Skippy.
- Gated by the host-runtime `mlx-backend` feature → `mesh-mlx/link-mlx`. Without
  it the backend reports unavailable and mesh uses the Skippy lane; the
  selection code still compiles (no Metal build in normal CI).

Discovery → MLX handoff (wired):
- `inference::mlx::plan_group_from_peers(node)` turns mesh's gossiped peer list
  into an MLX group: it filters to Apple-Silicon, directly-routable peers
  (`is_soc`/`gpu_name` + non-loopback `EndpointAddr` IPs), reads mesh's measured
  `current_direct_rtt_ms()` into `LatencySample`s, assigns a stable rank order
  (local = rank 0, peers sorted by id), and produces the rank-ordered hostfile
  (`ip:MLX_RING_BASE_PORT`) + parallelism/transport plan. mesh *finds and
  selects* the peers; MLX then opens its **own** TCP ring / JACCL to those
  addresses — mesh traffic never carries MLX data.
- `start_runtime_mlx_model` consults it: when a distributed group is found it
  passes the setup via `MlxModelLoadOptions::with_group`.
- `MlxModelHandle::load_distributed` → `mesh_mlx::DistributedEngine::join`:
  writes the hostfile, sets `MLX_HOSTFILE`/`MLX_RANK` (read by the ring/jaccl
  backends), inits the `Group`, and loads the model sharded per mode (pipeline =
  this stage's layers; tensor = sliced projections). The OpenAI server's chat
  path drives the group in lock-step.

Pending (needs multi-node hardware):
- Validate the pipeline/tensor execution across a live 2+ node `Group` (Ethernet
  ring / Thunderbolt JACCL) — throughput + correctness. All the upstream
  machinery (discovery, planning, hostfile, group init, sharded load, generate
  loops) is implemented and unit-tested; the rank-0 fan-out / non-rank-0 worker
  coordination is the piece that can only be exercised on a real rig.

Polish (non-blocking):
- Full Jinja `chat_template` (currently a ChatML-compatible framing that works
  for Qwen/Llama-style models).
- Sampling beyond greedy (temperature / top-p).
- Quantized row-parallel tensor sharding (currently dense-only; quantized
  models shard column-parallel projections only — correct, less memory saving).
