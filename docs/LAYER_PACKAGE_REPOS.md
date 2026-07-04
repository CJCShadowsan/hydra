# Contributing Layer Package Repositories

Layer package repositories let Mesh LLM run very large models with Skippy stage
splits. A package repository is a durable Hugging Face repo containing one
`model-package.json` manifest plus GGUF fragments for shared tensors, per-layer
tensors, and optional multimodal projectors.

Use this page for contributor workflow. The exact schema lives in
[specs/layer-package-repos.md](specs/layer-package-repos.md).

## Repository shape

```text
model-package.json
shared/
  metadata.gguf
  embeddings.gguf
  output.gguf
layers/
  layer-00000.gguf
  layer-00001.gguf
  ...
projectors/
  mmproj-model-f16.gguf
README.md
```

Required rules:

- `model-package.json` must be at the repo root.
- `schema_version` must be `1`.
- `format` must be `layer-package`.
- Each manifest artifact path must be relative to the repo root.
- Paths must not be absolute and must not escape with `..`.
- Every artifact entry must include size and SHA-256.
- Production refs should use immutable `hf://namespace/repo@revision` pins.

Model-specific generation policy defaults belong under the manifest
`generation` section. Use `generation.policy` for package-validated execution
semantics and `generation.thresholds` for numeric runtime resolver hints. Do
not add model-family-specific objects such as `generation.glm_dsa`; model
families use stable policy profiles instead.

The split matters. Policy values are portable names for phase behavior:
`decode`, `short_prefill`, `long_prefill`, `verify`, and `indexshare`.
Threshold values are numbers the resolver uses to decide whether a policy is
appropriate on the current request/backend. A package should not encode Metal,
CUDA, or Skippy implementation names in either place.

For GLM-DSA packages, set `generation.policy.profile` to `glm-dsa-v1` and
declare phase choices such as decode `compact-flash`, short-prefill `dense`,
long-prefill `sparse-chunked`, and IndexShare `required`. The policy values
name semantic execution paths, not backend kernel implementations. Runtime
config may override these values for experiments, but the package manifest is
the source of truth for validated defaults and consumers must log any override
or fallback. See
[specs/layer-package-repos.md](specs/layer-package-repos.md#generation-defaults).

Minimal GLM-DSA shape:

```json
{
  "generation": {
    "policy": {
      "profile": "glm-dsa-v1",
      "decode": "compact-flash",
      "short_prefill": "dense",
      "long_prefill": "sparse-chunked",
      "verify": "auto",
      "indexshare": "required",
      "experimental": {
        "selected_row_flash": "evidence-gated"
      }
    },
    "thresholds": {
      "short_prefill_max_tokens": 2048,
      "compact_flash_min_kv": 1,
      "dense_mask_max_bytes": 268435456
    }
  }
}
```

The threshold values should be grounded in tensor sizes. For example, a
GLM-DSA top-k sideband costs `tokens * top_k * 4` bytes, while a dense sparse
mask costs roughly `tokens * visible_kv * 4` bytes. With a representative
GLM-5.2 split package sideband width of `768` i32 values per token, the
IndexShare sideband is `3 KiB/token`: `384 KiB` for a 128-token chunk,
`1.5 MiB` for 512 tokens, `6 MiB` for 2048 tokens, and `384 MiB` for a full
128k-token window. At `128k` visible KV, the corresponding dense mask is
`512 KiB/token`: `64 MiB` for 128 tokens, `256 MiB` for 512 tokens, and `1 GiB`
for 2048 tokens. That size gap is why GLM-DSA packages should record
dense-mask and compact-flash thresholds instead of leaving sparse policy
implicit.

For current GLM-DSA decode tuning, the llama.cpp Metal fixtures measured
compact selected-row flash at `63.40 us/run` for
`GLM_DSA_SELECTED_ROW_FLASH(kv=257,top_k=64)`, direct sparse attention at
`106.57 us/run`, and dense masked flash at `71.72 us/run` on the comparable
one-token shape. That is the evidence behind preferring
`decode: "compact-flash"` once parity is proven on the target package/backend.
For exact-boundary fixtures where selected KV equals visible KV, compact
selected-row flash measured `61.90 us` at `kv=128`, `60.71 us` at `kv=256`,
`61.24 us` at `kv=257`, and `57.99 us` at `kv=513`; direct sparse measured
`137.15 us`, `209.97 us`, `249.30 us`, and `711.10 us` for the same shapes.
The literal `top_k >= visible_kv` boundary is handled by llama.cpp's all-KV
flash bypass, but ordinary one-token decode after prefill is different:
IndexShare top-k is selected from the previous KV state while attention sees
previous+current KV. Even at short history, that route still exercises compact
selected-KV flash. That is why native GLM-DSA decode should prefer compact
flash over direct sparse by default when flash attention is available.
At GLM-5.2's configured `top_k=768` width, compact selected-row flash measured
`55.11-55.62 us/run` for `kv=1024..2048`, while direct sparse measured
`984.50-988.95 us/run` on the same shapes. That is roughly an `18x` win for
compact selected-KV flash over direct sparse once visible KV exceeds the
selected-row window.
The same fixture family measured dense masked flash much faster than direct
sparse for short phase shapes (`68.58-70.80 us/run` versus
`461.98-473.75 us/run` for 4-16 tokens), so GLM-DSA packages should keep
short prefill and verification dense by default unless a backend-specific sparse
path has its own evidence.
After those phase gates, the next measured local bottleneck is routed expert
decode rather than top-k routing. The current Metal MoE fixture estimates one
GLM-5.2 routed FFN decode layer at `386.81 us`, with expert matmuls accounting
for `376.28 us` (`97.3%`). Route/top-k plus weighted sum is only `10.53 us`
(`2.7%`). That evidence should inform runtime optimization order, but it does
not add new manifest schema: package policy still belongs under
`generation.policy`, and numeric resolver hints still belong under
`generation.thresholds`.

In practice, this means the package's `generation` block is the phase-aware
contract: decode prefers compact flash, short prefill prefers dense, long
prefill avoids dense masks, verification remains `auto`, and Shared-layer
IndexShare is required unless an explicit fallback is selected and logged.

## Local package tooling

`skippy-model-package` is the local inspection and writing tool. Current
subcommands are:

```bash
skippy-model-package inspect <model.gguf>
skippy-model-package plan <model.gguf> --stages 4
skippy-model-package write <model.gguf> --layers 0..12 --out ./stage-0.gguf
skippy-model-package write-stages <model.gguf> --stages 4 --out-dir ./stages
skippy-model-package write-package <model.gguf> --out-dir ./package
skippy-model-package validate <model.gguf> ./stages/stage-*.gguf
skippy-model-package validate-package <model.gguf> ./package
skippy-model-package preflight ./package --stages 4 --verify-sha256
```

Validate before publishing:

```bash
skippy-model-package validate-package <model.gguf> ./package
skippy-model-package preflight ./package --stages 4 --verify-sha256
```

## Queue a Hugging Face package job

Mesh LLM includes a spend-bearing HF Jobs helper for package generation. It is
dry-run by default and must be confirmed explicitly before submitting jobs:

```bash
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --dry-run
mesh-llm models package unsloth/Qwen3-8B-GGUF:Q4_K_M --confirm --follow
```

The hidden compatibility alias is `mesh-llm model-package`; prefer
`mesh-llm models package` in docs and scripts.

Important options:

- `--target <repo>`: destination Hugging Face package repo.
- `--model-id <id>`: OpenAI-facing package model id.
- `--timeout <duration>`: HF Jobs timeout, defaulting to `1h` unless raised by
  size-based estimates.
- `--dry-run`: print the resolved package plan and maximum cost without side effects.
- `--confirm`: submit the job.
- `--follow`: wait and stream job progress.
- `--status <job-id>`, `--logs <job-id>`, `--cancel <job-id>`, `--list`: inspect
  or manage submitted jobs.
- `--update-script`: refresh the bucket script when needed.

The source model should stay in colon-selector form, for example
`unsloth/Qwen3-8B-GGUF:Q4_K_M`. Do not split the quant into a separate `--quant`
argument for generated job inputs.

## Publishing flow

The HF Jobs script performs the publishing work:

1. clone mesh-llm,
2. build `skippy-model-package`,
3. run `write-package`,
4. validate the manifest,
5. upload package artifacts incrementally,
6. upload `model-package.json`,
7. write a package model card,
8. update `meshllm/catalog`,
9. print the suggested run command.

The printed run command follows this shape:

```bash
mesh-llm serve --model <target-repo> --split
```

For package refs in hand-written docs and configs, prefer the explicit package
scheme:

```text
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@<revision>
```

## After publishing

Run package-only certification first:

```bash
mesh-llm models certify hf://namespace/repo@revision --package-only --report-out cert.json
```

Use the immutable published ref for certification, not only the local package
directory. Fix package-local preflight diagnostics and published-ref
package-only certification failures before moving on to a live endpoint smoke.

Then run a live endpoint smoke once the mesh is serving it:

```bash
mesh-llm models certify hf://namespace/repo@revision --api-base http://127.0.0.1:9337 --json
```

If the package is intended for public meshes, keep peer artifact transfer off by
default. Enable `MESH_LLM_ARTIFACT_TRANSFER=trusted` only for same-owner or
explicitly trusted-owner deployments, and `open` only in lab environments.
