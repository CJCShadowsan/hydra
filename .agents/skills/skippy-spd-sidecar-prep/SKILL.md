---
name: skippy-spd-sidecar-prep
description: Use this skill when preparing, training, exporting, packaging, or validating SPD sidecar heads for Skippy, including topology/tap selection, local MPS/CUDA proof runs, Hugging Face job planning, safetensors export, parity fixtures, and request-path SPD smoke validation.
metadata:
  short-description: Prepare Skippy SPD sidecar heads
---

# skippy-spd-sidecar-prep

Use this skill for SPD sidecar preparation for Skippy. This covers deciding the
sidecar topology, running the reference trainer/evaluator, exporting a Skippy
serving artifact, and validating that the head is usable from Rust and live
Skippy taps.

## Critical Rules

- Treat SPD sidecars as topology-bound artifacts, not generic draft models. A
  head is tied to the base model/tokenizer, chat template, hidden size, logical
  SPD stage count, selected hidden-state taps, projection layout, draft vocab,
  and spec-layer count.
- Treat the serving sidecar as a coordinator-owned companion bundle. Worker
  stages should only need the derived `spd_tap_return_hf_indices` allowlist;
  they should not need the sidecar weights, fixture, or local trainer outputs.
- Choose the target Skippy split topology before serious training. Physical
  stage placement can differ only if it exposes the same logical hidden-state
  taps required by the sidecar manifest.
- Do not claim real distributed speedup from single-host smokes. Separate model
  quality, Rust/Python parity, live tap correctness, request-path correctness,
  and distributed performance evidence.
- Do not replace real training/eval with unit-test-only evidence. Unit tests are
  useful gates, but SPD sidecar preparation needs real checkpoints, real hidden
  taps, parity fixtures, and live Skippy smoke runs.
- Hugging Face Jobs are spend-bearing. Default to dry-run planning, print the
  model, dataset, topology, hardware flavor, timeout, output repo, and maximum
  cost, and require explicit confirmation before submitting.

## Related Skills

- Use `hf-layer-package-jobs` when turning SPD training into a Hugging Face job
  flow or any other spend-bearing HF automation.
- Use `skippy-model-package` when preparing GGUF stage artifacts or validating
  physical split boundaries.
- Use `skippy-correctness` for full-model versus staged-execution parity.
- Use `skippy-bench` for SPD request-path smoke reports and benchmark summaries.

## Repo Entry Points

- `evals/spd/hf_train_eval_qwen06.py` trains/evaluates a real SPD speculation
  head by cloning and patching the reference SPD repository. It can also prepare
  an existing checkpoint from a local path or Hugging Face model repo.
- `evals/spd/export_spd_head.py` converts `speculation_head_final.pt` plus
  `skippy-spd-head.json` into Rust-readable `spd-head.safetensors`.
- `evals/spd/export_parity_fixture.py` exports real Python/reference hidden-tap
  rows, logits, top-k proposals, and cache fixtures for Rust parity checks.
- `evals/spd/README.md` is the live progress log and command cookbook for the
  current SPD proof.

## Mesh-Native Sidecar Bundles

Use `[defaults.speculative] spd_bundle_ref = "..."`
or a per-model `speculative.spd_bundle_ref` when proving SPD through Mesh rather
than a lower-level smoke harness. The ref may be a local bundle directory, a
direct local `skippy-spd-head.json`, or `hf://namespace/repo[@revision]`.

The bundle must contain:

- `skippy-spd-head.json`
- the manifest-declared serving checkpoint, normally `spd-head.safetensors`
- `spd-parity-fixture.safetensors`

The base model should remain a normal Mesh/Skippy model reference or layer
package so each node materializes its assigned stage through the existing
resolver. When stage 0 is configured with a resolved local Skippy layer package,
the SPD proposal source can replay boundary taps from package parts and read h0
from the package embedding part, so a coordinator-side full GGUF override is no
longer required for that shape. `spd_model_path` remains useful as an explicit
full-GGUF override for lower-level smokes. Do not use
`spd-openai-smoke --rsync-model-artifacts` as product proof; that is a
harness-only shortcut.

## Topology Checklist

Before training or evaluating a sidecar, write down:

- Base model repo/ref and revision.
- Target GGUF artifact, quant, and layer count if this is meant for Skippy.
- Tokenizer and chat template settings; for Qwen, explicitly decide thinking
  versus no-thinking template behavior.
- Logical SPD stage count and `stage_layer_boundaries`.
- Explicit `shallow_hidden_layer_indices` if the reference trainer needs taps
  that do not match simple stage boundaries.
- `num_spec_layers`, draft vocab choice, and `draft_top_k` for evaluation.
- Physical Skippy split boundaries that expose every required hidden tap.

For the current pretrained Qwen3.5-4B S4/L4 proof, the tap-aligned physical
split is `8,10,16,20,24,31`, which exposes ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`.

## First Real-Node Target

Use the pretrained `Qwen/Qwen3.5-4B` S4/L4 sidecar before training a new head.
It is the first target because it already has strong reference acceptance,
Rust/Python parity, live Skippy tap parity, and a known tap-aligned split.

Expected local artifact paths in the current proof workspace:

- GGUF:
  `.artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf`
- Sidecar manifest:
  `/private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/skippy-spd-head.json`
- Serving checkpoint:
  `/private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-head.safetensors`
- Parity fixture:
  `/private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-parity-fixture.safetensors`

Keep stage 0, the OpenAI frontend, and the SPD sidecar on the coordinator. Put
downstream physical stages on worker nodes or devices. With one worker node,
start with a no-launch preflight:

```bash
target/release/skippy-bench spd-openai-smoke \
  --stage-server-bin target/release/skippy-server \
  --manifest /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/skippy-spd-head.json \
  --fixture /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-parity-fixture.safetensors \
  --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --ctx-size 128 \
  --n-gpu-layers=-1 \
  --stage-hosts local,<worker>,<worker>,<worker>,<worker>,<worker>,<worker> \
  --endpoint-host-map local=<coordinator-lan-ip-or-name>,<worker>=<worker-lan-ip-or-name> \
  --remote-model-path-map <worker>=/path/on/worker/Qwen3.5-4B-Q4_K_M.gguf \
  --max-tokens 1 \
  --repeat-count 1 \
  --preflight-only \
  --output /tmp/spd-qwen35-first-remote-preflight.json
```

Only after the preflight validates artifacts, tap coverage, endpoint maps, and
remote model paths, remove `--preflight-only` and run the smoke:

```bash
target/release/skippy-bench spd-openai-smoke \
  --stage-server-bin target/release/skippy-server \
  --manifest /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/skippy-spd-head.json \
  --fixture /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-parity-fixture.safetensors \
  --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --ctx-size 128 \
  --n-gpu-layers=-1 \
  --stage-hosts local,<worker>,<worker>,<worker>,<worker>,<worker>,<worker> \
  --endpoint-host-map local=<coordinator-lan-ip-or-name>,<worker>=<worker-lan-ip-or-name> \
  --remote-model-path-map <worker>=/path/on/worker/Qwen3.5-4B-Q4_K_M.gguf \
  --max-tokens 1 \
  --repeat-count 1 \
  --output /tmp/spd-qwen35-first-remote-openai.json
```

Do not report speedup from this first remote smoke unless the report has enough
tokens/repeats and the hardware placement actually lets stages overlap. The
first purpose is to prove stage launch, hidden-tap return, sidecar proposal,
target verification, and cleanup across a real node boundary.

Current checkpoint: the first real-node proof has completed for the pretrained
Qwen3.5-4B sidecar. The clean paired LAN report
`/private/tmp/spd-lan-count-paired.json` matched baseline/SPD content, accepted
`23 / 23` proposals, reached `max_in_flight=4`, had `0` oldest rejections,
`0` younger drains, and `0` tap failures. The SPD-only LAN sweep
`/private/tmp/spd-lan-mini-sweep.json` exercised reset behavior with
`57 / 59` accepted, one oldest rejection, three younger drains, `0` tap
failures, and `0` out-of-order replay proposals. Treat these as KV/transport
correctness evidence, not speed evidence.

## Product Two-Stage Target

For the product-shaped two-node Skippy test, use exactly two physical stages:
`0..16` on the coordinator and `16..32` on the worker. The current pretrained
Qwen3.5-4B S4/L4 sidecar is not compatible with that topology because it needs
non-boundary taps from the `8,10,16,20,24,31` split. Do not use it for a
two-stage SPD-vs-baseline speed claim.

The first real two-stage baseline checkpoint is
`/private/tmp/skippy-two-stage-baseline.json`: one coordinator stage plus one
worker stage, `--splits 16 --layer-end 32`, `--n-gpu-layers=-1`, `24` generated
tokens, baseline decode `1293.2ms`, wall `1678.9ms`, stage-0 compute
`253.0ms`, downstream wait `990.2ms`, and zero tap/record/ignored-tap
failures. This proves the ordinary two-stage Metal split shape and cleanup, not
SPD speed.

Train the matching sidecar with logical boundaries that match the physical
split. The trainer derives the required tap rows as `0,16,32;0,16`, where `0`
is the embedding row and nonzero rows are stage-boundary hidden states:

```bash
python3 evals/spd/hf_train_eval_qwen06.py \
  --work-dir /tmp/skippy-spd-qwen35-4b-s2-16 \
  --model-name Qwen/Qwen3.5-4B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-rows 8192 \
  --eval-rows-per-set 32 \
  --num-stages 2 \
  --stage-layer-boundaries 16,32 \
  --num-spec-layers 4 \
  --max-length 512 \
  --max-new-tokens 64 \
  --draft-top-k 4 \
  --device mps \
  --model-torch-dtype float16 \
  --upload-repo ''
```

Use a smaller row count only to debug trainer/export plumbing. For the real
artifact, use a larger local run or a confirmed Hugging Face job, export
`spd-head.safetensors`, export a parity fixture, run fixture parity, run live
tap parity on `--splits 16 --layer-end 32`, and only then run
`spd-openai-smoke --run-baseline true --run-spd true --spd-rolling-executor`
on the same two-node topology.

Current package-backed checkpoint: local release `spd-openai-smoke` passed with
`--model-path` set to a Skippy layer package directory, generated both stages as
`load_mode=layer-package`, and logged `spd_model_source=layer_package` with no
full-GGUF `spd_model_path`. The report
`/private/tmp/spd-qwen35-s2-openai-package-local-4-rerun.json` matched
baseline/SPD content and recorded `0` tap failures, but accepted `0 / 3`
proposals from the tiny S2 debug sidecar. Treat this as request-path/package
source correctness evidence, not speed or sidecar-quality evidence.

Current Mesh-native Qwen3-8B product checkpoint: a real two-node run with the
exact immutable `meshllm/Qwen3-8B-Q4_K_M-layers` package ref completed through
Mesh resolver/download/stage-control and answered via the local OpenAI proxy.
Mesh elected the worker as stage-0 coordinator and placed the local M4 as
downstream `stage-1` with `layer_range=23..36`. Treat this as product split
serving evidence. It is not SPD evidence yet. For an SPD run on this exact
topology, train/export a `Qwen/Qwen3-8B` sidecar with `num_stages=2` and
`stage_layer_boundaries=23,36`, or add a product planner constraint so an SPD
manifest can force/reject incompatible Mesh stage boundaries before serving.
Use the trainer dry-run before spending time or HF money:

```bash
python3 evals/spd/hf_train_eval_qwen06.py \
  --dry-run-topology \
  --model-name Qwen/Qwen3-8B \
  --manifest-base-model-path Qwen/Qwen3-8B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-rows 8192 \
  --eval-rows-per-set 32 \
  --num-stages 2 \
  --stage-layer-boundaries 23,36 \
  --num-spec-layers 4 \
  --max-length 512 \
  --max-new-tokens 64 \
  --draft-top-k 4 \
  --device mps \
  --upload-repo ''
```

This must report `physical_split_boundaries=[23]`, `layer_end=36`, tap rows
`0,23,36;0,23`, and worker tap-return allowlist `[23,36]`. For local MPS
Qwen3-8B plumbing, keep `--model-torch-dtype float16`; leave `auto` or use
`bfloat16` on CUDA/HF jobs after confirming the target hardware.

Current local Qwen3-8B debug checkpoint: a 2-row MPS plumbing run trained
`Qwen/Qwen3-8B` with `num_stages=2`, `stage_layer_boundaries=23,36`,
`num_spec_layers=4`, `max_length=64`, and `--model-torch-dtype float16`. It
produced `speculation_head_final.pt`, `skippy-spd-head.json`, a BF16
`spd-head.safetensors` export with `56` tensors, and a one-prompt parity
fixture under `/private/tmp/skippy-spd-qwen3-8b-s2-23-debug-20260618-100141`.
Rust external manifest validation, external fixture validation, and
`skippy-bench spd-fixture-parity` passed for the BF16 export. Do not use this
artifact for quality or speed claims: it used only two rows. Do not export this
head as F16 for Rust today; current SPD safetensors reads reject F16 head
tensors. No acceptance rate exists yet for the Qwen3-8B S2 `23,36` target:
reference eval failed with `KeyError: 23` during custom-tap pipeline fill, so
acceptance must come from a fixed reference eval or a Rust request-path smoke.

## First Larger Training Target

Use `Qwen/Qwen3-8B` for the first larger dense sidecar training proof. Keep the
initial training topology conservative:

- `num_stages=4`
- `num_spec_layers=4`
- greedy/no-thinking eval
- draft vocab capped at `32k`
- UltraChat rows first, then broaden the corpus after export/parity works

Inspect the target HF config/GGUF metadata before selecting
`stage_layer_boundaries`. Derive logical boundaries from the actual target layer
count, then use the Skippy tap planner to determine physical tap-aligned split
boundaries. Do not reuse the pretrained Qwen3.5-4B `8,10,16,20,24,31` physical
split as the training topology for Qwen3-8B.

## Local Proof Flow

Use the M4/MPS local path for a small proof or overfit/debug run. Do not treat
it as the final 4B-quality training path.

```bash
python3 evals/spd/hf_train_eval_qwen06.py \
  --work-dir /tmp/skippy-spd-qwen06-proof \
  --model-name Qwen/Qwen3-0.6B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-rows 1024 \
  --eval-rows-per-set 8 \
  --num-stages 2 \
  --num-spec-layers 4 \
  --max-length 256 \
  --max-new-tokens 64 \
  --draft-top-k 4 \
  --device mps \
  --upload-repo ''
```

Use `--device cuda` on a GPU host. Keep `--upload-repo ''` for local dry runs
unless artifact upload is explicitly wanted.

## Export Flow

After training or downloading a reference checkpoint, export it to Skippy
serving format:

```bash
python3 evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

The expected serving bundle is:

- `speculation_head_final.pt`
- `skippy-spd-head.json`
- `spd-head.safetensors`
- `spd-parity-fixture.safetensors`
- eval summaries and raw per-sample acceptance traces

## Validation Flow

Validate in increasing order of realism:

```bash
SKIPPY_SPD_MANIFEST=/path/to/skippy-spd-head.json \
  cargo test -p skippy-runtime validates_external_manifest_when_skippy_spd_manifest_is_set

SKIPPY_SPD_PARITY_FIXTURE=/path/to/spd-parity-fixture.safetensors \
  cargo test -p skippy-runtime validates_external_parity_fixture_when_skippy_spd_parity_fixture_is_set

SKIPPY_SPD_MANIFEST=/path/to/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/path/to/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set

cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /path/to/skippy-spd-head.json \
  --fixture /path/to/spd-parity-fixture.safetensors \
  --top-k 8
```

Then validate live Skippy taps with `skippy-bench spd-live-tap-parity`, followed
by `skippy-bench spd-openai-smoke` on the physical split topology that exposes
the manifest-required taps. Use release binaries for request-path timing.

## Evidence To Report

When reporting SPD sidecar status, include:

- Base model, revision, tokenizer/template mode, and GGUF/quant if applicable.
- Logical topology and physical Skippy split boundaries.
- Training dataset, row count, max length, epochs, batch/accumulation, learning
  rate, and draft vocab.
- Eval acceptance, equivalent accept length, theoretical gain, and generated
  token count.
- Rust/Python fixture parity, live tap parity, accepted/proposed counts, tap
  failures, and content-match status.
- Timing broken down into baseline decode, SPD decode, downstream wait, sidecar
  cache prefill, decoder layers, final norm, lm head/top-k, and head total when
  those fields are available.
