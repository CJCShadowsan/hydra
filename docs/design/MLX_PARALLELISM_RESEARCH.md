# MLX Distributed Parallelism — Tensor vs. Pipeline

Research notes on how MLX splits LLM workloads across machines, the choice between
tensor and pipeline parallelism, behavior over Ethernet, and how weights are
distributed/downloaded. Evidence drawn from a lite clone of the `ml-explore` repos
(`mlx`, `mlx-lm`, `mlx-swift`, `mlx-swift-examples`, `mlx-examples`).

## TL;DR

- MLX supports **both tensor parallelism (TP) and pipeline parallelism (PP)**.
  The split logic lives in the **Python stack** (`mlx` core + `mlx-lm`). The Swift
  repos only expose low-level C distributed bindings — no high-level sharding.
- **It's the user's choice** at launch time (`--pipeline` flag), gated by what the
  model supports. Default is tensor parallel.
- **Each node holds only ~1/N of the model**, not the full weights — that's how a
  cluster runs a model too big for one Mac.
- **Over Ethernet, pipeline is recommended** (not required). TP wants ultra-low
  latency (Thunderbolt + JACCL/RDMA).
- **On disk, the schemes differ sharply:** pipeline downloads only its layer shards;
  tensor downloads the whole model to every node.

## Where it lives

| Repo | Parallelism content |
|------|--------------------|
| `mlx` (core) | Backends `mlx/distributed/{mpi,ring,nccl,jaccl}`; sharded layers in `python/mlx/nn/layers/distributed.py` |
| `mlx-lm` | `models/pipeline.py` (PP mixin), per-model `shard()` (TP), `utils.sharded_load`, `examples/sharded_generate.py` |
| `mlx-swift` | Only C bindings (`Cmlx/include/mlx/c/distributed.h`) — no Swift sharding |
| `mlx-swift-examples` | No distributed code |
| `mlx-examples` (legacy) | Only **data parallelism** (`average_gradients`/`all_sum`) |

## Choosing TP vs PP

Selected at runtime; `sharded_load` errors if the model doesn't support the chosen mode.

```python
# mlx-lm/mlx_lm/examples/sharded_generate.py
pipeline_group = group if args.pipeline else None      # --pipeline flag
tensor_group   = group if not args.pipeline else None  # default = tensor parallel
model, tok = sharded_load(args.model, pipeline_group, tensor_group)
```

- TP requires the model class to define `shard()`.
- PP requires `model.model` to have `pipeline()` (the `PipelineMixin`).
- Not all models support both (e.g. `llama.py` has `shard()` but no PP mixin;
  `deepseek_v3` has the PP mixin). The `mlx-lm` CLI uses one or the other, not both.

## How the work is split (compute)

| | Pipeline | Tensor |
|---|---|---|
| Split axis | Across **layers** (depth) | Within **each layer** (width) |
| Data flow | Activation flows node→node, one `send`/`recv` per stage | All nodes compute same layer, sync via `all_sum` |
| Comm per layer | 1 send/recv at stage boundary | 2 all-reduces (after attention, after MLP) |
| Concurrency | Sequential handoff (assembly line) | Lockstep, constant chatter |

**Pipeline** (`mlx-lm/models/pipeline.py` + `deepseek_v3.py`): layers split contiguously;
forward pass is point-to-point.

```python
# PipelineMixin: drop layers this rank doesn't own
self.layers = self.layers[: self.end_idx]
self.layers[: self.start_idx] = [None] * self.start_idx

# forward: recv from next stage, compute my layers, send to prev stage
if pipeline_rank < pipeline_size - 1:
    h = mx.distributed.recv_like(h, (pipeline_rank + 1))
for l, c in zip(self.pipeline_layers, cache):
    h = l(h, mask, cache=c)
if pipeline_rank != 0:
    h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
if pipeline_size > 1:
    h = mx.distributed.all_gather(h)[: h.shape[0]]
```

**Tensor** (`python/mlx/nn/layers/distributed.py`): Megatron-style sharded linears.
Each node keeps only its slice of every matrix.

```python
# AllToShardedLinear: weight is (output_dims // N, input_dims)   -> split output dim
# ShardedToAllLinear: weight is (output_dims, input_dims // N)    -> split input dim,
#                     then all-reduce so every node has full result
def __call__(self, x):                 # ShardedToAllLinear
    x = x @ self["weight"].T
    x = mx.distributed.all_sum(x, group=self.group)
    ...
```

`mlx-lm/models/llama.py` `shard()`: q/k/v/gate/up = all-to-sharded, o_proj/down = sharded-to-all,
and `n_heads //= N`, `n_kv_heads //= N`.

## Disk / download behavior (the key practical difference)

`sharded_load` downloads via HuggingFace Hub (`snapshot_download`) in **two passes**.

**Pass 1 — metadata only (every node):** config, tokenizer, and
`model.safetensors.index.json` (the tensor→file map). No weights. Then lazy-build
the model to know parameter names.

**Pass 2 — weights — diverges by scheme:**

```python
# PIPELINE: download only the shard files holding MY layers
if pipeline_group is not None:
    model.model.pipeline(pipeline_group)
    weight_index = json.load(...index.json...)["weight_map"]
    local_files = set()
    for k, _ in tree_flatten(model.parameters()):  # only my layers' params
        local_files.add(weight_index[k])
    _download(repo, allow_patterns=local_files)
# TENSOR: download EVERYTHING
else:
    _download(repo)
```

| | Pipeline | Tensor |
|---|---|---|
| Downloaded to disk per node | Only its layers' `.safetensors` files (~1/N) | **Whole model** (full weights) |
| Disk footprint per node | ~1/N of model size | Full model size |
| Resident memory per node | ~1/N | ~1/N (slices full files lazily at load) |
| Constraint | MLX-converted models only (needs index weight map) | Any model with `shard()` |

So with TP each node downloads/stores the full model on disk but keeps only 1/N
resident in RAM; with PP each node downloads/stores only its ~1/N slice.

## Ethernet

Backends (`mlx/docs/src/usage/distributed.rst`):

| Backend | Transport | Ethernet? | For |
|---------|-----------|-----------|-----|
| Ring | TCP sockets | ✅ Yes (default) | all-reduce / all-gather |
| MPI | TCP | ✅ Yes | general collectives |
| JACCL | RDMA over Thunderbolt 5 (macOS 26.2+) | ❌ No | **tensor parallelism** (low latency) |
| NCCL | NVLink/IB/TCP | CUDA only | CUDA multi-GPU |

- **Ethernet path = Ring (TCP)**; build hostfile via `mlx.distributed_config --over ethernet` (uses each node's `en0` IP).
- Docs call JACCL *"necessary for things like tensor parallelism"* — TP's two
  all-reduces per layer are latency-bound, so TP over Ethernet works but is slow.
- **Pipeline is the practical Ethernet choice**: one activation send/recv per stage
  boundary tolerates higher latency, and each node only downloads its slice.
- Ethernet tuning: Ring `--connections-per-ip N`; MPI `--mca btl_tcp_links N`,
  `--mca btl_tcp_if_include en0`.

### Required vs recommended

You are **not forced** to use pipeline on Ethernet. Both run. But for good
performance over plain Ethernet, pipeline is recommended; TP performance needs a
Thunderbolt + JACCL interconnect.

## Example launch

```bash
# Pipeline over Ethernet (Ring/TCP)
mlx.launch --backend ring --hostfile hosts.json \
    python sharded_generate.py --pipeline -p "Hello world"

# Tensor parallel over Thunderbolt (JACCL/RDMA)
mlx.launch --backend jaccl --env MLX_METAL_FAST_SYNCH=1 --hostfile hosts.json \
    python sharded_generate.py -p "Hello world"
```
