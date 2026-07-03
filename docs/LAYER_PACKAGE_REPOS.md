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

For GLM-DSA packages, set `generation.policy.profile` to `glm-dsa-v1` and
declare phase choices such as decode `compact-flash`, short-prefill `dense`,
long-prefill `sparse-chunked`, and IndexShare `required`. The policy values
name semantic execution paths, not backend kernel implementations. Runtime
config may override these values for experiments, but the package manifest is
the source of truth for validated defaults and consumers must log any override
or fallback. See
[specs/layer-package-repos.md](specs/layer-package-repos.md#generation-defaults).

The threshold values should be grounded in tensor sizes. For example, a
GLM-DSA top-k sideband costs `tokens * top_k * 4` bytes, while a dense sparse
mask costs roughly `tokens * visible_kv * 4` bytes. With a representative
GLM-5.2 split package sideband width of `768` i32 values per token, the
IndexShare sideband is `3 KiB/token`; at `128k` visible KV, the corresponding
dense mask is `512 KiB/token`. That size gap is why GLM-DSA packages should
record dense-mask and compact-flash thresholds instead of leaving sparse policy
implicit.

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
