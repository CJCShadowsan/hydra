<h1 align="center">Hydra</h1>

Hydra is a downstream fork of [Mesh LLM](https://github.com/Mesh-LLM/mesh-llm)
for network-aware, low-latency inference. It keeps the upstream mesh runtime,
then adds Hydra-owned scheduling, network telemetry, artifact placement, and
VAST-backed cache/model movement. The name is intentional: if one head is cut
off, the system keeps going and grows back.

Use **Hydra**. The primary command is `hydra`; the upstream-compatible
`mesh-llm` binary may still exist for compatibility, but new operator flows in
this fork should be written with `hydra`.

Hydra preserves upstream Mesh LLM attribution and tracks upstream changes while
allowing product-level divergence. See [NOTICE](NOTICE), [CHANGES.md](CHANGES.md),
and [docs/hydra/UPSTREAM_SYNC.md](docs/hydra/UPSTREAM_SYNC.md).

## What Hydra Adds

- Passive network cost tracking for RTT, queue wait, TTFT, ITL/TPOT,
  tokens/sec, cache hit rate, KV transfer, artifact materialization, jitter,
  bandwidth estimate, and failures.
- Shadow or active SLO-aware target scheduling that can choose lower-latency
  routes after upstream capability, health, media, and context filters.
- POSIX and S3-compatible artifact placement for model weights, layer packages,
  KV state, recurrent state, and activation-frame cache.
- VAST DataSpace/DataEngine trigger support through configurable webhook
  payloads after placement manifests commit.
- Exact cache compatibility checks before remote KV/recurrent/activation cache
  can be reused.
- Upstream sync tooling so Hydra can diverge without missing Mesh LLM changes.

## Install Hydra

Until Hydra has packaged release installers, build the `hydra` binary from this
repository:

```bash
git clone git@github.com:CJCShadowsan/hydra.git
cd hydra
cargo build -p mesh-llm --bin hydra --release
```

Run it directly:

```bash
./target/release/hydra --help
./target/release/hydra setup
```

Or install it into your Cargo bin directory:

```bash
cargo install --path crates/mesh-llm --bin hydra
hydra setup
```

Hydra currently uses the upstream Mesh LLM config and data directories for
compatibility. Expect paths such as `~/.mesh-llm` until the fork grows its own
migration path.

## Quick Start

Join the public mesh and start serving:

```bash
hydra serve --auto
```

That command chooses a backend flavor, downloads a suitable model if needed,
joins the best discovered public mesh, starts the OpenAI-compatible API on
`http://localhost:9337/v1`, and starts the web console/management API on
`http://localhost:3131`.

Check available models:

```bash
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

Send an OpenAI-compatible request:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-4.7-Flash-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

For server deployments, add `--headless` to hide the web UI while keeping the
management API on the `--console` port:

```bash
hydra serve --auto --headless
```

## Enable Hydra Scheduling

Hydra defaults to conservative behavior. Start with scheduler shadow mode so
operators can inspect decisions without changing request routing:

```toml
[scheduler]
mode = "shadow"
ttft_budget_ms = 450
tpot_budget_ms = 40
affinity_override_threshold_ms = 100
stale_after_ms = 15000
cache_affinity_credit_ms = 80
failure_penalty_ms = 500
unknown_remote_penalty_ms = 75
```

Run with the config:

```bash
hydra serve --auto --config hydra.toml
```

Inspect Hydra network cost and scheduler state:

```bash
curl -s http://localhost:3131/api/status | jq '.network_costs, .scheduler'
```

After shadow decisions look correct, switch to active mode:

```toml
[scheduler]
mode = "active"
```

## Place Artifacts

Hydra can publish artifacts into a mounted namespace, including a VAST
DataSpace mounted as POSIX:

```bash
hydra placement prefetch layer_package qwen3-stage-0 /local/cache/stage-0 \
  --posix-root /vast/global/hydra
```

Check operation state:

```bash
hydra placement cache
hydra placement status <operation-id>
```

Pin or evict placement records:

```bash
hydra placement pin qwen3-stage-0
hydra placement evict qwen3-stage-0 --kind layer_package --posix-root /vast/global/hydra
```

## Trigger VAST Movement

Hydra can publish an artifact manifest, then trigger a VAST DataEngine/webhook
workflow to move or materialize the artifact at another site:

```bash
hydra placement prefetch layer_package qwen3-stage-0 /local/cache/stage-0 \
  --posix-root /vast/global/hydra \
  --vast-trigger-endpoint https://vast-dataengine.example.internal/hydra/ship \
  --vast-tenant acme-ai \
  --vast-dataspace prod-dataspace \
  --vast-source-namespace /vast/global/hydra \
  --vast-destination-namespace /vast/site-b/hydra \
  --vast-target-site site-b
```

The trigger payload includes the committed manifest, checksum, byte size,
artifact kind, compatibility identity, provider location, and target-site hints.
If the namespace publish succeeds but the trigger fails, Hydra records the
operation as failed with the manifest attached so operators can recover from the
committed artifact.

See [docs/hydra/VAST_PLACEMENT.md](docs/hydra/VAST_PLACEMENT.md) for the JSON
API form and VAST deployment notes.

## Common Workflows

| Goal | Command | Full guide |
|---|---|---|
| Try the public mesh | `hydra serve --auto` | [docs/MESHES.md](docs/MESHES.md) |
| Start a private mesh | `hydra serve --model Qwen3-8B-Q4_K_M` | [docs/MESHES.md](docs/MESHES.md) |
| Publish your own mesh | `hydra serve --model Qwen3-8B-Q4_K_M --publish` | [docs/MESHES.md](docs/MESHES.md) |
| Join by invite token | `hydra serve --join <token>` | [docs/MESHES.md](docs/MESHES.md) |
| Run an API-only client | `hydra client --auto` | [docs/MESHES.md](docs/MESHES.md) |
| Run a big model with splits | `hydra serve --model hf://meshllm/<repo>@<rev> --split` | [docs/SKIPPY_SPLITS.md](docs/SKIPPY_SPLITS.md) |
| Place model/layer/cache artifacts | `hydra placement prefetch ...` | [docs/hydra/VAST_PLACEMENT.md](docs/hydra/VAST_PLACEMENT.md) |
| Use Goose, OpenCode, Claude Code, or Pi | `hydra goose`, `hydra opencode`, `hydra claude`, `hydra pi` | [docs/AGENTS.md](docs/AGENTS.md) |
| Build or contribute | `cargo build -p mesh-llm --bin hydra --release` | [CONTRIBUTING.md](CONTRIBUTING.md) |

## How Hydra Works

- **Upstream mesh runtime.** Hydra inherits Mesh LLM's distributed runtime,
  OpenAI-compatible API, public/private mesh discovery, model routing, and
  Skippy stage splits.
- **Hydra network telemetry.** Each node records bounded, local-only route and
  artifact cost observations. `/api/status` exposes summaries and compact
  advisory hints.
- **Hydra scheduler.** Requests pass through upstream eligibility filters first.
  Hydra then scores remaining targets by queue, network, prefill/cache miss,
  KV transfer, cold start, decode pressure, artifact readiness, cache affinity,
  and recent failure cost.
- **Hydra placement.** Artifacts publish through staging paths, checksum
  validation, manifest commit, and atomic publish where available.
- **Hydra cache safety.** KV/recurrent/activation state is reusable only when
  exact identity fields match. Recompute remains the safe fallback.
- **VAST integration.** Hydra treats VAST DataSpace as a POSIX or S3-compatible
  global namespace and uses explicit webhook/DataEngine triggers for site
  movement.

For upstream Mesh LLM behavior that Hydra still inherits, see
[docs/USAGE.md](docs/USAGE.md) and [docs/CLI.md](docs/CLI.md). Those inherited
docs may still show `mesh-llm`; in this fork, use `hydra` for the same commands
unless a section is explicitly about upstream compatibility.

## Mixture-of-Agents (`model: "mesh"`) — experimental

> ⚠️ **Experimental.** The MoA gateway is new in this release. Behavior,
> routing heuristics, error shapes, and tuning knobs may change between
> versions while we tune it. Treat `model: "mesh"` as a preview feature
> rather than a stable production path; use a specific model id when you
> need stable semantics.

Send a request with `"model": "mesh"` and the proxy fans it out to every
model available in the mesh in parallel, arbitrates their responses with
deterministic logic, and returns one OpenAI-compatible reply. The arbiter
runs in code (not as another model call) and only escalates to a reducer
LLM on genuine conflict. Tool calls flow through the full pipeline.

```bash
curl http://localhost:9337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mesh","messages":[{"role":"user","content":"What is the capital of Japan?"}]}'
```

Requires at least two distinct models in the mesh. See
[docs/design/MOA_GATEWAY.md](docs/design/MOA_GATEWAY.md) for the
architecture, arbitration rules, and tuning knobs.



## Supported model families

Hydra inherits Mesh LLM's Skippy runtime, which tracks llama.cpp family parity
with reviewed GGUF representatives. The current reviewed support set covers 72
P0/P1 family rows, with 89 certified rows in the full parity inventory,
including Qwen, Llama, Gemma, Mistral, DeepSeek, GLM, MiniMax, Phi, Granite,
Hunyuan, EXAONE, Cohere, Falcon, RWKV, and many others.

Split multimodal serving is certified for Qwen2-VL, Qwen3-VL,
Qwen3-VL-MoE, HunyuanOCR/Hunyuan-VL, and DeepSeek-OCR using real GGUF plus
projector fixtures. DeepSeek3 and EXAONE-MoE use package-backed stages because
the full GGUFs are too large for the cheap local baseline.

See [docs/skippy/FAMILY_STATUS.md](docs/skippy/FAMILY_STATUS.md) for the full
artifact, split, wire dtype, cache policy, and exception matrix. See
[docs/skippy/LLAMA_PARITY.md](docs/skippy/LLAMA_PARITY.md) for the remaining
llama.cpp parity queue.

## Install and build notes

Hydra packaged releases are not published yet. Build the `hydra` binary from
source:

```bash
git clone git@github.com:CJCShadowsan/hydra.git
cd hydra
cargo build -p mesh-llm --bin hydra --release
```

Hydra source builds require Rust, `cmake`, and Node.js 24 + npm. Full repository
maintenance workflows also use `just`. CUDA builds need `nvcc`, ROCm builds
need ROCm/HIP, and Vulkan builds need Vulkan dev files plus `glslc`.

The upstream `mesh-llm` release process includes packaged binary attestation.
Hydra will need its own release signing process before Hydra release binaries
can make equivalent claims. Local source builds report as unstamped development
builds.

## Documentation hub

| Doc | Use it for |
|---|---|
| [docs/MESHES.md](docs/MESHES.md) | Private meshes, public discovery, publishing, invite tokens, API-only clients |
| [docs/SKIPPY_SPLITS.md](docs/SKIPPY_SPLITS.md) | Running big models with package-backed Skippy stage splits |
| [docs/LAYER_PACKAGE_REPOS.md](docs/LAYER_PACKAGE_REPOS.md) | Contributing and publishing layer package repositories |
| [docs/AGENTS.md](docs/AGENTS.md) | Goose, Claude Code, OpenCode, Pi, curl, and blackboard |
| [docs/EXO_COMPARISON.md](docs/EXO_COMPARISON.md) | Balanced comparison with Exo |
| [docs/CLI.md](docs/CLI.md) | Command reference and JSON automation |
| [docs/USAGE.md](docs/USAGE.md) | Longer operational usage guide, runtime control, owner-control operator flows |
| [docs/design/TESTING.md](docs/design/TESTING.md) | Testing playbook, mixed-version QA, remote deploy checks |
| [docs/plugins/flash-moe.md](docs/plugins/flash-moe.md) | Optional Flash-MoE SSD expert streaming backend setup |
| [docs/skippy/FAMILY_STATUS.md](docs/skippy/FAMILY_STATUS.md) | Certified Skippy model-family status |
| [docs/specs/layer-package-repos.md](docs/specs/layer-package-repos.md) | Manifest and artifact format spec |
| [docs/specs/mesh-setup-installer.md](docs/specs/mesh-setup-installer.md) | Installer/bootstrap and setup command behavior spec |

## Community

Mesh LLM is experimental distributed-systems software. When you report bugs,
include the command you ran, platform/backend flavor, `/api/status` output if
available, and whether the node was private, published, or joined with `--auto`.
