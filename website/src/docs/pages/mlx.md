# MLX runtime on Apple Silicon

Mesh LLM can optionally serve Hugging Face/MLX safetensors models through a native Rust MLX runtime (`mesh-mlx`). This path is for Apple Silicon Macs and is separate from the default Skippy/GGUF runtime.

The MLX runtime is behind an opt-in cargo feature because it builds and links MLX/Metal native code. Normal `just build` and release builds are unchanged.

## Current status

What works today:

- Single-node Apple Silicon serving for MLX/safetensors model repos.
- Native MLX C/Metal linking through `mesh-mlx-sys` (`link-mlx`).
- A multi-turn OpenAI-compatible local server (full message history preserved) used by mesh routing.
- Latency-aware distributed planning for directly-routable Mac peers.
- MLX TCP ring (`ring`) transport over Ethernet/Wi-Fi.
- JACCL/RDMA hostfile support when every node advertises complete RDMA device rows.
- Operator overrides for tensor-vs-pipeline and ring-vs-JACCL.

Important limitations:

- Multi-node MLX needs real hardware validation before it should be treated as production-ready.
- MLX distributed traffic does **not** use mesh QUIC tunnels. MLX opens its own TCP/RDMA sockets and therefore requires direct node-to-node reachability.
- The OpenAI surface is non-streaming and text-only (no tool calls, no vision/audio), with a multi-turn ChatML-style template. The MLX backend's advertised capabilities are clamped to match, so the mesh never routes tool/streaming/multimodal traffic to it.
- MLX models are safetensors/MLX-style Hugging Face repos, not GGUF layer packages.
- Mixed MLX + Skippy stages in one split are not supported.

## Build

On Apple Silicon macOS:

```bash
just build-mlx
```

For a release binary:

```bash
just release-build-mlx
```

These recipes set `MESH_LLM_FEATURES=mlx`, which enables:

```text
mesh-llm/mlx
  -> mesh-llm-host-runtime/mlx-backend
  -> mesh-mlx/link-mlx
  -> mesh-mlx-sys/link-mlx
```

The first native build may be slow because `mesh-mlx-sys` builds MLX C/C++ code. The build script supports:

| Variable | Meaning |
|---|---|
| `MLX_C_DIR` | Path to a checked-out/prebuilt `mlx-c` tree with a `CMakeLists.txt`. |
| `MLX_C_TAG` | `mlx-c` git tag to fetch when `MLX_C_DIR` is not set. |

## Single-node serving

Build with MLX enabled, then serve an MLX/safetensors model repo or local model directory. For example:

```bash
just build-mlx
./target/debug/mesh-llm serve --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 --log-format json
```

Then call the local OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 32
  }'
```

You can also run the crate-level native smoke test directly:

```bash
cargo test -p mesh-mlx --features link-mlx --test live_single_node -- --nocapture
```

That test downloads a tiny MLX model and verifies a real generation path through the native engine.

## Multi-node planning

When MLX is enabled, mesh can form an MLX group from Apple Silicon peers that have direct IP endpoints. Mesh supplies the peer addresses and rank order, then MLX owns its own transport:

- `ring`: TCP sockets, works over Ethernet/Wi-Fi.
- `jaccl`: RDMA over Thunderbolt/JACCL, requires the OS and hardware setup MLX expects.

Mesh deliberately does not tunnel MLX traffic over QUIC because the latency profile would defeat MLX distributed's purpose.

### How distributed serving works

MLX is SPMD: every node in the group runs the same lock-step generation, but only one node receives the OpenAI request.

- **Rank 0 (leader)** serves the OpenAI API. Every node computes the same rank order from the shared peer set, so all nodes agree on who the leader is. For each request the leader broadcasts the rendered prompt and token budget to the workers, then runs the generation.
- **Other ranks (workers)** do not serve OpenAI. They run a lock-step worker loop, parked until the leader broadcasts a request, then run the identical generation so the group's collectives stay in step. Greedy decoding is deterministic, so every rank derives the same token and stops on the same step. When the leader shuts down it broadcasts a sentinel so workers exit cleanly.

This is why a single mesh-advertised model id works across the group: clients only ever talk to the leader's endpoint.

### Parallelism mode

Use `MESH_LLM_MLX_PARALLELISM` to choose how the model is split:

| Value | Behavior |
|---|---|
| `auto` / unset | Choose from measured RTT. Worst RTT <= 2ms uses tensor; otherwise pipeline. |
| `pipeline` / `pp` | Force pipeline parallelism. Recommended over normal Ethernet. |
| `tensor` / `tp` | Force tensor parallelism, including over Ethernet. Works as an operator choice, but can be slow on higher-latency links. |

Examples:

```bash
# Practical Ethernet default: pipeline over TCP ring.
MESH_LLM_MLX_PARALLELISM=pipeline \
MESH_LLM_MLX_TRANSPORT=ring \
./target/debug/mesh-llm serve --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 --log-format json

# Try tensor parallelism over Ethernet/TCP ring. This is allowed but latency-sensitive.
MESH_LLM_MLX_PARALLELISM=tensor \
MESH_LLM_MLX_TRANSPORT=ring \
./target/debug/mesh-llm serve --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 --log-format json

# Prefer tensor over JACCL/RDMA when the full RDMA mesh is available.
MESH_LLM_MLX_PARALLELISM=tensor \
MESH_LLM_MLX_TRANSPORT=jaccl \
./target/debug/mesh-llm serve --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 --log-format json
```

### Transport mode

Use `MESH_LLM_MLX_TRANSPORT` to choose the MLX backend transport:

| Value | Behavior |
|---|---|
| `auto` / unset | Use JACCL only when every node has a complete RDMA device map; otherwise use ring. |
| `ring` / `tcp` | Force TCP ring. Use this for Ethernet/Wi-Fi. |
| `jaccl` / `rdma` / `thunderbolt` | Require JACCL/RDMA. If the group does not have complete RDMA metadata, mesh logs the error and falls back to ring for safety. |

## Ethernet vs RDMA guidance

This mirrors MLX/MLX-LM's distributed shape:

| Fabric | Recommended mode | Why |
|---|---|---|
| Ethernet/Wi-Fi TCP ring | Pipeline | Pipeline sends activations between stages and tolerates higher latency better. |
| Low-latency LAN | Auto or tensor | Auto picks tensor when measured RTT is <= 2ms. |
| Thunderbolt/JACCL/RDMA | Tensor | Tensor parallelism performs all-reduces every layer and benefits most from RDMA latency. |

Tensor over Ethernet is intentionally supported as an operator override. It is not forbidden; it is simply latency-sensitive. Pipeline remains the practical default for ordinary Ethernet because it communicates at stage boundaries instead of doing per-layer all-reduces.

## CI coverage

PR CI now has two layers:

1. Stub/no-engine unit tests for `mesh-mlx-sys` and `mesh-mlx`, which run quickly and cover planning, parsing, and Rust-side validation.
2. A macOS native MLX smoke job that builds the `link-mlx` path and runs `live_single_node` so CI exercises real MLX/Metal generation instead of only stubs.

If the native job fails, check the MLX CMake build first, then the Hugging Face model download, then the generated output assertion.

## Relationship to Skippy

Skippy remains the default GGUF/layer-package runtime. MLX is an Apple Silicon safetensors lane. The two runtimes are intentionally not mixed inside a single split because their activation formats and model artifact formats differ.
