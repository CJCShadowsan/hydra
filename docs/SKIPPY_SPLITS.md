# Running Big Models With Skippy Splits

Skippy is Mesh LLM's embedded staged runtime. It lets the mesh run models that
do not fit on one machine by loading package-backed layer stages across
selected peers.

## Mental model

1. The coordinator resolves the requested model or layer package.
2. The topology planner picks peers and contiguous layer ranges.
3. Downstream/final stages load first.
4. Stage 0 becomes routable only after every required stage reports ready.
5. OpenAI clients keep using the normal mesh endpoint at
   `http://localhost:9337/v1`.

If one node can load the full model, Mesh LLM prefers the single-node path.
Splitting is used when the model physically needs a split or when an explicit
split run asks for it.

## Use a published layer package

Layer packages are durable Hugging Face repos with a `model-package.json`
manifest and GGUF fragments. Prefer immutable refs for production runs:

```bash
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@<revision> --split
```

Named or moving refs are useful while testing:

```bash
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers --split
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers:main --split
```

Other peers join the mesh normally:

```bash
mesh-llm serve --join <token>
```

## Two-node split smoke test

Use the same layer-package model on every serving node. Each node resolves the
package and downloads only the shared artifacts plus the layer files needed for
its assigned stage.

Package-owned generation defaults travel with the layer package. If
`model-package.json` declares `generation.policy` or `generation.thresholds`,
the coordinator resolves those defaults before stage launch and passes the
resolved policy into the embedded llama runtime. Per-run overrides are allowed
for experiments, but the runtime must log the package recommendation and the
selected override or fallback. GLM-DSA packages use
`generation.policy.profile = "glm-dsa-v1"`; they must not require
Skippy-specific prompt, thinking, sparse attention, or IndexShare behavior that
is absent from llama.cpp itself.

For GLM-DSA split runs, treat `generation.policy` as a package-owned execution
hint and llama.cpp as the source of truth for correctness. A package may prefer
decode `compact-flash`, short-prefill `dense`, long-prefill `sparse-chunked`,
and IndexShare `required`, but Skippy should only select those paths when the
embedded llama runtime reports support and should log any fallback.
Skippy-specific compatibility code must not override prompt parsing, thinking
mode, sparse attention, or IndexShare semantics; those behaviors come from the
embedded llama runtime. Skippy's job is to pass the resolved package policy and
the package layers into that runtime, then surface resolver decisions in logs
or telemetry.

Current Metal backend evidence for one-token decode shows compact selected-row
flash at `63.40 us/run` versus direct sparse at `106.57 us/run` and dense
masked flash at `71.72 us/run` on the `kv=257,top_k=64` fixture family.
Exact-boundary
fixtures where selected KV equals visible KV measured compact selected-row
flash at `57.99-61.90 us/run` across `kv=128..513`, while direct sparse
measured `137.15-711.10 us/run` on the same shapes. The literal
`top_k >= visible_kv` boundary is handled by llama.cpp's all-KV flash bypass,
but ordinary one-token decode after prefill still takes the compact selected-KV
route because IndexShare top-k is selected from the previous KV state while
attention sees previous+current KV. For short phase shapes, dense masked flash
measured `68.58-70.80 us/run` versus
`461.98-473.75 us/run` for direct sparse at 4-16 tokens, so Skippy should treat
short prefill and verification as dense defaults unless the embedded llama
runtime resolves an explicit, measured sparse override.
At the configured GLM-5.2 `top_k=768` width, compact selected-row flash measured
`55.11-55.62 us/run` for `kv=1024..2048`, while direct sparse measured
`984.50-988.95 us/run` on the same shapes.
After those phase decisions, measured GLM-5.2 FFN decode cost is dominated by
MoE expert execution, not route/top-k overhead. The gated Metal fixture
estimates `391.61 us` per routed FFN decode layer, with `375.83 us` (`96.0%`)
in routed gate/up/down matmuls, `5.24 us` (`1.3%`) in routed fused SwiGLU, and
only `10.54 us` (`2.7%`) in route/top-k plus weighted sum. A
production-shaped fused GLU shared expert plus final add measured `410.19 us`,
making the routed+shared FFN estimate `801.80 us`; shared expert execution is
`51.2%` of that estimate. The isolated shared fused SwiGLU split row is only
`4.37 us`; the earlier unfused activation/mul diagnostic measured `296.90 us`,
but that path does not represent the normal llama.cpp shared expert graph
because `build_ffn()` already uses `ggml_swiglu_split()`. The remaining MoE
optimization target is therefore routed/shared expert matmul and whole-graph
execution, not a reason to add a Skippy-specific generation schema.
The extended fixture measured a merged q2_K routed gate/up shape at `1.02x`
faster for the routed estimate, a merged shared gate/up fused GLU shape at
`1.02x` faster for the shared expert, a weighted-down MoE graph shape at
`1.02x` on the small quantized whole-graph fixture, and a q2_K down-projection
alternative at `1.14x` faster before quality is measured. That keeps the
split-layer contract unchanged and points local llama.cpp work at expert matmul
kernels, shared-expert whole execution, and controlled down-projection quant
experiments. Merged shared gate/up may be recorded as an evidence-gated
`generation.policy.experimental` option by packages that actually contain that
validated artifact shape.
The Phase E report can be made a hard evidence gate with
`GLM52_PHASE_E_REQUIRE_GATES=1`; when the optional kernel sweep is enabled it
also proves the dispatch alternatives are present. That sweep showed generic
dispatch tuning is not the lever: forcing one-token q3_K routed down through
`mul_mm_id` measured `927.49 us` versus `164.63 us` on the default `mul_mv_id`
path, q3_K `mul_mv_id` simdgroup tuning stayed within measurement noise
(`164.63 us` default versus `163.63 us` best), and q3_K row-height tuning found
no faster row than default `nr0=4` at `165.11 us`.

The main split-serving implication is that Skippy should pass through the
resolved `generation.policy` and `generation.thresholds` contract, not add a
separate GLM-DSA policy surface. With a GLM-5.2-style `top_k` width of `768`,
IndexShare costs `3 KiB/token` (`6 MiB` for 2048 tokens), while a dense sparse
mask at `128k` visible KV costs `512 KiB/token` (`1 GiB` for 2048 tokens). That
gap is the reason the package contract requires Shared-layer IndexShare and
long-prefill sparse/chunked execution once llama.cpp has the native path.

```bash
# node A: starts the private mesh and becomes the coordinator
mesh-llm serve \
  --model meshllm/Qwen3-8B-Q4_K_M-layers \
  --split \
  --max-vram 5 \
  --bind-port 7842 \
  --port 9447 \
  --console 3232

# node B: joins with the token printed by node A
mesh-llm serve \
  --model meshllm/Qwen3-8B-Q4_K_M-layers \
  --split \
  --max-vram 5 \
  --join <token> \
  --bind-port 7843 \
  --port 9447 \
  --console 3232
```

For hosts with more than one network interface, add `--bind-ip <lan-ip>` on
each node so the invite token and gossip advertise the routable address.

Once both stages are ready:

```bash
curl -sS http://127.0.0.1:3232/api/status | jq '{state:.node_state, ready:.llama_ready, peers:(.peers|length), stages:.runtime.stages}'
curl -sS http://127.0.0.1:9447/v1/models | jq '.data[].id'
curl -sS http://127.0.0.1:9447/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer mesh' \
  -d '{"model":"meshllm/Qwen3-8B-Q4_K_M-layers","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":16}'
```

## Use a local GGUF

Direct GGUFs still work:

```bash
mesh-llm serve --gguf ~/models/model.gguf
```

Internally, direct GGUF serving materializes through the same package-backed
stage machinery as a synthetic single-stage package. That keeps the runtime path
consistent without requiring you to publish a package repository first.

## Check readiness

```bash
curl -s http://localhost:3131/api/status | jq .
curl -s http://localhost:9337/v1/models | jq '.data[].id'
mesh-llm doctor split --model-ref meshllm/Qwen3-8B-Q4_K_M-layers --port 3131
```

The stage runtime status is exposed through the management API and web console.
The OpenAI model list should include the full model id once stage 0 is ready.
The split doctor explains which peers are eligible, which peers were excluded,
and the exact next step when the coordinator sees only itself as a valid split
participant.

For maintainer debugging, add `--output-dir <dir>`. The doctor bundle includes
`split-readiness.json`, management API snapshots for runtime/stage/llama status,
plugin startup/provider/endpoint snapshots, `skippy-diagnostics.json`, and the
active instance's `skippy-native.log` when the local runtime directory can be
matched to the console port.

On Windows, collect a shareable diagnostic bundle from already-running nodes:

```powershell
.\contrib\windows\CollectSplitDiagnostics.ps1 `
  -Model meshllm/Qwen3-8B-Q4_K_M-layers `
  -ConsoleUrls http://127.0.0.1:3131 `
  -ApiUrls http://127.0.0.1:9337/v1
```

## Cache behavior

Mesh sets Skippy materialization under the Mesh LLM cache by default:

```text
<user-cache>/mesh-llm/skippy-stages
```

Layer-package downloads use Skippy's Hugging Face package cache unless
`SKIPPY_HF_PACKAGE_CACHE` overrides it. Materialized stage GGUFs are derived
cache, not the durable package format.

Preview cache cleanup:

```bash
mesh-llm models prune
```

Apply cleanup:

```bash
mesh-llm models prune --yes
```

`models prune` protects active or pinned materialized stages and removes only
eligible derived cache entries.

## Verify a package before rollout

For a brand-new model family or a large sharded GGUF candidate, start with the
[new model onboarding checklist](skippy/NEW_MODEL_ONBOARDING.md) before adding a
support-matrix entry.

Package-only verification checks resolution, artifact integrity, and local stage
materialization:

```bash
mesh-llm models certify hf://meshllm/Qwen3-8B-Q4_K_M-layers --package-only --report-out cert.json
```

Use package-only certification as the rollout preflight for published package
refs. It should fail before a split model becomes routable when package
resolution, manifest shape, artifact size/SHA, missing stage files,
tokenizer/projector sidecars, or local materialization are not clean enough for
serving. For a local package directory, run the package-local preflight first:

```bash
skippy-model-package preflight ./model-package --stages 2 --verify-sha256
```

Runtime verification additionally checks a running OpenAI-compatible endpoint:

```bash
mesh-llm models certify hf://meshllm/Qwen3-8B-Q4_K_M-layers \
  --api-base http://127.0.0.1:9337 \
  --json
```

Runtime certification hits `/v1/models`, `/v1/chat/completions`, and
`/v1/responses` and requires real text-bearing responses.

## Peer artifact transfer

For split runs, a worker may fetch missing package artifacts from the
coordinating mesh node before falling back to normal local/Hugging Face package
resolution. This is not a discovery protocol and does not gossip local package
inventory.

Peer artifact transfer is disabled by default on public meshes. Use it only for
trusted or lab deployments:

```bash
MESH_LLM_ARTIFACT_TRANSFER=trusted mesh-llm serve --model hf://meshllm/<repo>@<revision> --split
MESH_LLM_ARTIFACT_TRANSFER=open mesh-llm serve --model hf://meshllm/<repo>@<revision> --split
```

Only immutable `hf://namespace/repo@revision` package refs are eligible for peer
transfer. Received artifacts are size/SHA-256 verified and installed atomically.

## More details

- [LAYER_PACKAGE_REPOS.md](LAYER_PACKAGE_REPOS.md) explains how to contribute packages.
- [specs/layer-package-repos.md](specs/layer-package-repos.md) is the manifest spec.
- [skippy/FAMILY_STATUS.md](skippy/FAMILY_STATUS.md) lists certified families.
- [skippy/TOPOLOGY_PLANNER.md](skippy/TOPOLOGY_PLANNER.md) documents topology planning, including latency-aware physical stage counts.
