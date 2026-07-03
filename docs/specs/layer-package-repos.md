# Layer Package Repository Spec

Status: draft

Layer package repositories are durable model artifacts for skippy-backed stage
serving. A repository contains one `model-package.json` manifest plus GGUF
fragments that can be selected by layer range and loaded by a stage without
requiring every peer to store or materialize the full source model.

The current runtime accepts local package directories and Hugging Face package
references of the form `hf://namespace/repo`, `hf://namespace/repo:revision`,
or `hf://namespace/repo@revision`.

## Goals

- Let mesh nodes fetch only the model pieces needed for their assigned layer
  range.
- Keep package identity tied to a real source model coordinate, revision, and
  artifact set.
- Make package validation deterministic before a stage is launched.
- Keep package manifests additive so older runtimes can reject unsupported
  packages clearly instead of loading incompatible tensor layouts.
- Treat per-stage materialized GGUFs as derived cache, not as the durable
  package format.

## Repository Identity

A layer package repository SHOULD be named after the source model and
distribution it contains, for example:

```text
meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
meshllm/DeepSeek-V3.2-UD-Q4_K_XL-layers
```

The repository identity is not enough to prove compatibility. Consumers MUST
read `model-package.json` and use the manifest fields below as the source of
truth.

Package references use `hf://` so runtime code can distinguish package repos
from model coordinates and local paths:

```text
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers:8f4c2d1
hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@main
```

Publishers SHOULD point production configs at an immutable commit hash or tag,
not a moving branch.

## Repository Layout

The root of the repository MUST contain `model-package.json`.

Recommended layout:

```text
model-package.json
shared/
  metadata.gguf
  embeddings.gguf
  output.gguf
layers/
  layer-00000.gguf
  layer-00001.gguf
  layer-00002.gguf
  ...
projectors/
  mmproj-model-f16.gguf
README.md
```

Required artifacts:

| Artifact | Purpose |
| --- | --- |
| `shared/metadata.gguf` | Shared GGUF metadata and tokenizer state required by every stage. |
| `shared/embeddings.gguf` | Input-boundary tensors required by the first stage. |
| `shared/output.gguf` | Output-boundary tensors required by the final stage. |
| `layers/layer-NNNNN.gguf` | Owned tensors for one transformer layer. |

Optional artifacts:

| Artifact | Purpose |
| --- | --- |
| `projectors/*.gguf` | Multimodal projector GGUFs, currently `kind: "mmproj"`, used by stage 0 or single-stage serving. |

Artifact paths in the manifest MUST be relative to the repository root. They
MUST NOT be absolute paths and MUST NOT escape the package root with `..`.
Consumers MUST reject unsafe paths.

Each owned tensor from the source model MUST appear in exactly one package
artifact. Shared metadata and tokenizer values may be repeated only where the
GGUF writer requires them to keep each fragment loadable.

Projector artifacts are package-level companions, not transformer-layer
fragments. They MUST NOT be counted as owned layer tensors and MUST NOT be
merged into per-stage GGUF materializations.

## Peer Cache Transfer

For split Skippy runs, a worker may fetch missing Hugging Face package artifacts
from the coordinating mesh node over admitted mesh `STREAM_SUBPROTOCOL`
transport before falling back to normal local/HF package resolution. This
transfer path is an optimization for already-selected split participants, not a
package discovery protocol.

Privacy and compatibility boundaries:

- Nodes advertise only `skippy-stage/2` subprotocol feature support, including
  `artifact-transfer`. They do not gossip local package inventory, artifact
  paths, cache roots, or tokens.
- Mesh owns only the subprotocol open envelope; Skippy owns the artifact
  request/response schema, authorization semantics, and byte framing.
- Peer cache transfer uses the mesh `STREAM_SUBPROTOCOL` envelope. Generation-3
  Skippy stage peers are a compatibility break, so older chained-reply
  subprotocol peers are not mixed into new split topologies.
- Only `hf://namespace/repo@revision` package refs are eligible for peer
  transfer.
- The serving node checks the active split topology and only serves artifacts
  needed by the requesting node's assigned stage range. Stage 0 may fetch input
  boundary files and projector artifacts; final stages may fetch output boundary
  files.
- `model-package.json` is capped at 16 MiB for peer transfer.
- Non-manifest artifacts must match the manifest-declared relative path, byte
  size, and SHA-256 digest.
- Received artifacts are written to a fresh hidden partial file and installed
  atomically only after size and SHA-256 verification.
- Peer artifact transfer is not advertised or served by default on public mesh
  nodes. Set `MESH_LLM_ARTIFACT_TRANSFER=trusted` to enable same-owner or
  explicitly trusted-owner transfer, or `MESH_LLM_ARTIFACT_TRANSFER=open` for
  lab deployments that intentionally allow any peer.

## Manifest Schema

The manifest file is UTF-8 JSON. The current schema version is `1`.

Minimal shape:

```json
{
  "schema_version": 1,
  "model_id": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
  "source_model": {
    "path": "/cache/Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
    "sha256": "<64 hex chars>",
    "repo": "Qwen/Qwen3-235B-A22B-GGUF",
    "revision": "<source commit>",
    "primary_file": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
    "canonical_ref": "Qwen/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL",
    "distribution_id": "UD-Q4_K_XL",
    "files": [
      {
        "path": "Qwen3-235B-A22B-UD-Q4_K_XL.gguf",
        "size_bytes": 123,
        "sha256": "<64 hex chars>"
      }
    ]
  },
  "format": "layer-package",
  "layer_count": 94,
  "activation_width": 8192,
  "shared": {
    "metadata": {
      "path": "shared/metadata.gguf",
      "tensor_count": 0,
      "tensor_bytes": 0,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    },
    "embeddings": {
      "path": "shared/embeddings.gguf",
      "tensor_count": 4,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    },
    "output": {
      "path": "shared/output.gguf",
      "tensor_count": 4,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    }
  },
  "generation": {
    "policy": {
      "profile": "glm-dsa-v1",
      "decode": "compact-flash",
      "short_prefill": "dense",
      "long_prefill": "sparse-chunked",
      "verify": "auto",
      "indexshare": "required",
      "experimental": {
        "selected_row_flash": "off"
      }
    },
    "thresholds": {
      "short_prefill_max_tokens": 2048,
      "compact_flash_min_kv": 256,
      "dense_mask_max_bytes": 268435456
    },
    "speculative_decoding": {
      "default": "native-mtp-n1",
      "strategies": {
        "native-mtp-n1": {
          "type": "native-mtp",
          "prediction_depth": 1,
          "layer_indices": [47],
          "window_policy": {
            "default": "fixed",
            "initial_window": 1,
            "min_window": 1,
            "max_window": 1
          }
        }
      }
    }
  },
  "layers": [
    {
      "layer_index": 0,
      "path": "layers/layer-00000.gguf",
      "tensor_count": 32,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    }
  ],
  "projectors": [
    {
      "kind": "mmproj",
      "path": "projectors/mmproj-model-f16.gguf",
      "tensor_count": 128,
      "tensor_bytes": 123,
      "artifact_bytes": 123,
      "sha256": "<64 hex chars>"
    }
  ],
  "skippy_abi_version": "1.2.3",
  "created_at_unix_secs": 1790000000
}
```

Required top-level fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | MUST be `1` for the current format. |
| `model_id` | MUST be a non-empty model coordinate, not a filesystem-derived name. |
| `source_model` | MUST identify the source artifact used to build the package. |
| `format` | MUST be `layer-package`. |
| `layer_count` | MUST match the source model's transformer layer count. |
| `activation_width` | SHOULD be present; routing and topology planning rely on it. |
| `generation` | MAY declare package-owned generation defaults, including native speculative decoding strategies. |
| `shared` | MUST include `metadata`, `embeddings`, and `output` artifacts. |
| `layers` | MUST include exactly one entry for each layer index `0..layer_count`. |
| `projectors` | MAY include package-level projector artifacts; currently only `kind: "mmproj"` is defined. |
| `skippy_abi_version` | MUST describe the llama/skippy ABI used to write the fragments. |
| `created_at_unix_secs` | SHOULD be set by the package writer for provenance. |

Each artifact entry MUST include:

- `path`: repository-relative artifact path.
- `tensor_count`: number of tensors in the fragment.
- `tensor_bytes`: total bytes for tensor payloads in the fragment.
- `artifact_bytes`: exact file size in bytes.
- `sha256`: lowercase or uppercase 64-character SHA-256 hex digest.

`artifact_bytes` MUST be greater than zero. `tensor_bytes` MUST be zero when
`tensor_count` is zero and greater than zero when `tensor_count` is greater
than zero.

`projectors` is optional and defaults to an empty list. When present, each
projector entry uses the same artifact fields plus:

- `kind`: projector type. The current schema defines `mmproj`; consumers MUST
  reject unknown kinds unless they explicitly support them.
- `path`: repository-relative projector GGUF path, usually under `projectors/`.

Projector entries MUST have non-empty, safe relative paths, positive
`artifact_bytes`, and valid 64-character SHA-256 digests. A package MAY declare
multiple projectors for future model variants, but the current serving default
is to use the first declared `mmproj` when no explicit projector path is
configured.

### Generation Defaults

`generation` is optional and defaults to no package-owned generation policy.
When present, it may declare package-authored runtime defaults. The package owns
defaults that are specific to the artifact distribution, such as quant layout,
preserved native tensors, validated sparse-attention paths, and native
speculative decoding strategy.

The `generation` object has two separate responsibilities:

- `generation.policy` names the semantic execution profile and phase choices
  that were validated for this artifact.
- `generation.thresholds` supplies numeric resolver hints used to decide when a
  phase choice applies.

Keep these responsibilities separate. Policy fields should answer "which
execution path is intended for this phase?" Threshold fields should answer
"when should the runtime choose or reject that path?" For example,
`decode: "compact-flash"` belongs under `generation.policy`, while
`compact_flash_min_kv: 256` belongs under `generation.thresholds`.

The `generation` object is intentionally generic. Do not add
model-family-specific sub-objects such as `generation.glm_dsa`; use a stable
`generation.policy.profile` instead. This keeps the manifest shape reusable for
future sparse attention, native prediction, and verifier policies without
creating one schema branch per model family.

Runtime config and explicit CLI/environment overrides MAY override these
defaults for experiments, but consumers SHOULD log the final resolved policy and
the package recommendation that was overridden. If a consumer cannot execute the
package-recommended path, it MUST choose a correctness-preserving fallback and
emit the fallback reason.

Package generation defaults are not a substitute for model correctness
metadata. Architecture-specific GGUF metadata and tensor layout still define
whether a runtime may execute the model at all; `generation.policy` only
chooses among valid execution paths for that artifact.

#### Execution Policy

Packages MAY declare `generation.policy` to describe the package-validated
execution profile. This profile is a model execution policy, not a backend
implementation detail. It should use stable semantic path names such as
`compact-flash` rather than Metal/CUDA kernel names.

The current proposed shape is:

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
        "selected_row_flash": "off"
      }
    },
    "thresholds": {
      "short_prefill_max_tokens": 2048,
      "compact_flash_min_kv": 256,
      "dense_mask_max_bytes": 268435456
    }
  }
}
```

`profile` names the policy family and version. For GLM-DSA packages, use
`glm-dsa-v1` until a later profile intentionally changes the meaning of the
phase fields or thresholds. Other model families SHOULD use their own stable
profile names instead of adding model-family-specific top-level objects under
`generation`.

The profile is also the compatibility boundary for package tooling. Writers may
infer a known profile from GGUF metadata and tensors, but they must not invent
backend-specific field names for one model family. If a later GLM-DSA package
needs different phase semantics, create a new profile such as `glm-dsa-v2`
instead of changing the meaning of `glm-dsa-v1`.

Policy values are intentionally phase-specific:

- `decode`: preferred one-token generation path. For GLM-DSA this is expected
  to become `compact-flash` when compact selected-KV attention has parity on
  the package.
- `short_prefill`: preferred path below the short-prefill threshold. Packages
  MAY select `dense` when sparse/indexer overhead is known to dominate.
- `long_prefill`: preferred path above the short-prefill threshold. Packages
  SHOULD avoid policies that materialize dense sparse masks for long context.
- `verify`: preferred path for speculative verification spans. It MAY remain
  `auto` until verifier-specific parity and performance are measured.
- `indexshare`: whether Shared GLM-DSA layers require local IndexShare/top-k
  state. `required` means a consumer must not silently recompute shared-layer
  indexers unless an explicit fallback policy is selected and logged.
- `experimental.selected_row_flash`: controls selected-row flash fusion. Use
  `off` until the package has reproducible wins for that path.

Suggested semantic path values are:

- `auto`: runtime chooses using package thresholds and backend capability.
- `dense`: dense attention path; useful for short prefill when measured faster.
- `direct-sparse`: direct GLM-DSA sparse attention.
- `compact-flash`: compact selected K/V followed by flash attention.
- `sparse-chunked`: chunked sparse prefill path for long prompts.
- `fallback`: named correctness fallback when a native sparse backend is not
  available.

`generation.thresholds` are package recommendations. Consumers SHOULD treat
them as input to the runtime policy resolver, not as hard schema limits. Every
policy decision SHOULD emit telemetry containing the policy profile, phase,
selected path, rejected path or fallback reason, `n_kv`, `top_k` when present,
IndexShare role when present, backend, and any dense sparse-mask allocation
avoided.

Thresholds must be named for the resolver decision they inform, not for an
implementation detail. For example, prefer `dense_mask_max_bytes` over a
backend-specific allocation flag. This keeps the same package usable across
Metal, CUDA, CPU, and future backends while still allowing each backend to make
an evidence-based decision.

#### Policy Resolution

Consumers resolve generation policy in this order:

1. Request/runtime override, when explicitly configured for an experiment.
2. Package `generation.policy` and `generation.thresholds`.
3. Runtime built-in default for the architecture.
4. Correctness fallback when the preferred path is unsupported.

The resolved policy is a runtime contract. Package tools may infer and write
the policy from GGUF tensor shape, but serving code must not re-infer a
different policy silently. For example, a GLM-DSA package with split
`attn_k_b`, `attn_v_b`, and `attn_kv_a_mqa` tensors may advertise
`glm-dsa-v1`; the runtime may still fall back from `compact-flash` to `dense`
on a backend that lacks compact selected-KV attention, but it must log that
fallback.

For `glm-dsa-v1`, the current phase intent is:

| Phase | Recommended value | Intent |
| --- | --- | --- |
| `decode` | `compact-flash` | Avoid dense sparse-mask materialization during one-token decode. |
| `short_prefill` | `dense` | Avoid paying sparse/indexer overhead when prompts are below the package threshold. |
| `long_prefill` | `sparse-chunked` | Keep long-context prefill away from huge dense sparse masks. |
| `verify` | `auto` | Let the runtime select a verifier path until verifier-specific parity is proven. |
| `indexshare` | `required` | Reuse Full-layer top-k/index state for Shared GLM-DSA layers instead of silent recompute. |
| `experimental.selected_row_flash` | `off` | Keep experimental selected-row flash disabled until package evidence enables it. |

For `glm-dsa-v1`, the current threshold intent is:

| Threshold | Meaning |
| --- | --- |
| `short_prefill_max_tokens` | Maximum prompt/window length that should prefer the short-prefill policy. |
| `compact_flash_min_kv` | Minimum KV length where compact selected-KV flash attention is worth considering. |
| `dense_mask_max_bytes` | Maximum dense sparse-mask allocation the runtime should permit before forcing a sparse fallback. |

For GLM-DSA threshold tuning, consumers SHOULD reason from the package tensor
sizes rather than from model names alone. The relevant first-order estimates
are:

| Quantity | Formula | Example |
| --- | --- | --- |
| Hidden activation bytes | `tokens * hidden_width * activation_bytes` | GLM-5.2 width `6144`: `12 KiB/token` at f16 or `24 KiB/token` at f32. |
| IndexShare sideband bytes | `tokens * top_k * 4` | Width `768`: `3 KiB/token`, `384 KiB` for a 128-token chunk. |
| Dense sparse-mask bytes | `tokens * visible_kv * 4` | At `128k` visible KV: `512 KiB/token`, `64 MiB` for a 128-token chunk, `1 GiB` for a 2048-token chunk. |

These numbers are intentionally policy inputs, not schema requirements. A
package with a different `attention.indexer.top_k`, hidden width, activation
wire dtype, or context target will produce different thresholds. The important
contract is that packages expose enough policy and threshold information for
the runtime to explain why it selected dense, direct sparse, compact-flash, or
fallback execution.

#### Speculative Decoding

When present, `generation` may also declare `speculative_decoding` defaults:

- `default`: the strategy id the package recommends for this distribution.
- `strategies`: a map of strategy id to strategy configuration.

The current native MTP strategy shape is:

```json
{
  "type": "native-mtp",
  "prediction_depth": 1,
  "layer_indices": [47],
  "window_policy": {
    "default": "fixed",
    "initial_window": 1,
    "min_window": 1,
    "max_window": 1
  }
}
```

Native MTP strategy rules:

- `type` MUST be `native-mtp`.
- `prediction_depth` MUST be `1` for the current Skippy native MTP path.
- `layer_indices` MUST list package layer indices containing native MTP/NextN
  tensors, usually the final `blk.N.nextn.*` block emitted by GLM GGUF
  conversion.
- `window_policy` SHOULD be fixed to `1` until runtimes support wider native
  MTP heads.

Draft-model speculation may use the same strategy map with `type:
"draft-model"` and fields such as `draft_model` and adaptive `window_policy`.
Consumers that do not recognize a strategy type MUST ignore it unless it is the
declared default for a request they are trying to serve.

Operators may override the package recommendation in `config.toml` with
`speculative.strategy`. Supported values are `auto` (use package/runtime
defaults), `native-mtp-n1` (force the current native MTP strategy), and
`disabled` (disable native MTP for the configured model/default scope).

## Layer Selection

For a stage with `layer_start..layer_end`, consumers select:

1. `shared.metadata`
2. `shared.embeddings` only when the stage owns the input boundary
3. every `layers[]` entry with `layer_index` in `layer_start..layer_end`
4. `shared.output` only when the stage owns the final output boundary

The selected parts are sufficient to load the stage. A normal package-backed
runtime SHOULD load selected parts directly. If a runtime composes those parts
into a per-stage GGUF file, that file is derived cache and MUST NOT become the
published repository format.

Projectors are selected independently from layer parts. For a package-backed
multimodal model:

1. an explicit stage `projector_path` wins;
2. otherwise, stage 0 or a single-stage runtime uses the first declared
   `projectors[]` entry with `kind: "mmproj"`;
3. downstream stages do not load projector artifacts.

Consumers MUST NOT infer projector identity from sibling files or filename
stems. If a package needs a projector, the manifest must declare it explicitly.

## Publishing Rules

Package creation SHOULD use:

```bash
skippy-model-package write-package org/repo:distribution --out-dir model-package/
```

Multimodal packages SHOULD declare projector artifacts at write time:

```bash
skippy-model-package write-package org/repo:distribution \
  --projector mmproj-model-f16.gguf \
  --out-dir model-package/
```

The package writer copies declared projectors into `projectors/`, records their
checksums and sizes in `model-package.json`, and keeps them as durable package
artifacts.

Local source GGUF paths are allowed only with explicit provenance:

```bash
skippy-model-package write-package ./model.gguf \
  --out-dir model-package/ \
  --model-id org/repo:distribution \
  --source-revision <commit> \
  --source-file model.gguf
```

Before publishing, run package validation against the source model:

```bash
skippy-model-package validate-package /path/to/source.gguf model-package/
```

A published repository SHOULD include a short `README.md` with:

- source model coordinate and revision;
- source artifact filename and checksum;
- package manifest checksum;
- projector artifact filenames and checksums, when present;
- layer count and activation width;
- skippy ABI version used to write the package;
- package generation defaults such as native MTP strategy id and prediction
  depth, when declared;
- validation command and result;
- any model-family certification notes, such as supported activation wire dtype
  or exact-cache policy.

## Consumer Validation

Before a stage starts, consumers MUST validate:

- `model-package.json` parses as schema version `1`;
- `format` is `layer-package`;
- `skippy_abi_version` is compatible with the runtime ABI;
- `model_id` and source identity fields are non-empty;
- requested layer range is non-empty and within `layer_count`;
- requested layers exist and are not duplicated in the manifest;
- selected artifact paths are relative, safe, and files;
- selected artifact sizes match `artifact_bytes`;
- declared projector paths are relative, safe, and files;
- declared projector sizes match `artifact_bytes`.

Consumers MUST verify SHA-256 checksums for selected artifacts before using an
`hf://` layer package, including cache-hit resolutions. Local package
directories MAY keep checksum verification behind `SKIPPY_VERIFY_PACKAGE_SHA`
for development workflows.

Metadata-only inspection for inventory, identity discovery, or prepare planning
MAY validate only the manifest and shared metadata artifact, and must not
require a non-empty stage layer range. A real stage load still requires the
non-empty range and selected-artifact checks above.

Implementations MAY cache successful checksum verification results. Cache keys
and records should be derived from manifest and artifact identity plus file
metadata, and should not store raw local package paths.

Checksum verification SHOULD include declared projectors when the package is
first downloaded, when the package is validated by tooling, or when a stage is
about to use a projector.

## Compatibility

Manifest schema changes MUST be additive when possible. New optional fields may
be added to schema version `1` if old runtimes can ignore them safely.

`projectors` is an additive schema-version-1 field. Old packages without it
remain valid. Runtimes that do not understand projectors may still serve text
stages from the package, but they MUST reject multimodal serving requests that
require an undeclared or unsupported projector.

Changes that alter tensor ownership, layer indexing, path semantics, ABI
compatibility, or required fields MUST use a new schema version or a new
`format` value. Runtimes MUST reject unknown schema versions and incompatible
ABI versions rather than attempting best-effort loading.

Package repositories do not change mesh gossip or wire protocol semantics by
themselves. When package metadata is advertised through mesh status or planning,
new fields must follow the normal additive compatibility rules for mesh
protocol data.

## Open Questions

- Whether package repos should include an optional signed manifest or checksum
  sidecar for deployments that do not trust the hosting backend.
- Whether large repositories should publish one branch/tag per source revision
  and quantization, or one immutable repository per distribution.
- Whether package validation results should be machine-readable artifacts in
  the repo or remain documented in `README.md`.
