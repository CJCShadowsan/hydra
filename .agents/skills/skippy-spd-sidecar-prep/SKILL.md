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
- Treat the reusable artifact key as the logical layer-boundary/tap topology,
  not the physical hostname list. If a deployment packs adjacent logical stages
  onto a larger node, the same sidecar can be reused only when the runtime still
  exposes every manifest-required boundary tap. This is the path to avoid
  training every possible physical grouping: precompute a small set of canonical
  logical topologies, then clump contiguous logical stages at placement time.
- Treat the serving sidecar as a coordinator-owned companion bundle. Worker
  stages should only need the derived `spd_tap_return_hf_indices` allowlist;
  they should not need the sidecar weights, fixture, or local trainer outputs.
- Choose the target Skippy split topology before serious training. Physical
  stage placement can differ only if it exposes the same logical hidden-state
  taps required by the sidecar manifest.
- Do not claim real distributed speedup from single-host smokes. Separate model
  quality, Rust/Python parity, live tap correctness, request-path correctness,
  and distributed performance evidence.
- For the current product SPD proof, report `paper_pipeline_estimate` and
  `paper_like_speedup_vs_serial_split` as pipeline-fill economics only:
  accepted/proposed proposals, saved/unsaved candidate token round trips,
  max-in-flight, tap failures, and content equality. Do not call them measured
  speedup without a matched baseline/SPD wall-clock run that already clears the
  acceptance and round-trip-savings gate.
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
- `skippy-bench spd-live-tap-parity --product-corpus-dir <dir>` exports live
  Skippy/product activation rows (`rows.f32` plus `rows.jsonl`) for the current
  topology. Add `--product-native-teacher-logits` to write native Q4/product
  verifier logits over the SPD draft vocabulary in the same directory.
  `evals/spd/prepare_product_activation_corpus.py` converts product tap rows
  into safetensors, and
  `evals/spd/prepare_native_product_teacher_logits.py` converts native verifier
  logits into the teacher-logit safetensors used for quant-specific KL
  fine-tuning.
- `evals/spd/augment_product_activation_teacher_logits.py` attaches frozen HF
  teacher logits to product-captured rows by rerunning the same context tokens
  through the base model and saving logits aligned to `query_row_index` /
  `target_position`. Treat this as fallback/debug HF-teacher KL data over
  product tap inputs, not production native quantized-verifier logits.
- `evals/spd/train_product_activation_head.py` fine-tunes an existing reference
  `speculation_head_final.pt` on product `cur_in` safetensors plus aligned
  teacher logits. For Q4 product proofs, prefer native verifier teacher logits
  over HF BF16 teacher logits.
- `skippy-bench spd-product-corpus-capture` captures raw product tap rows and
  native verifier logits from a Skippy layer package using topology only; it
  does not require an existing SPD manifest or parity fixture.
  If the planner or script emits the native-teacher option explicitly, use
  `--product-native-teacher-logits true`; this command's Clap argument uses
  `ArgAction::Set`, unlike the older `spd-live-tap-parity` bool flag.
  For very large package captures, prefer `--stream-live-tap-stages` plus an
  explicit `--stage-backend-devices` map. This keeps the full native verifier
  session as the teacher while opening only one live-tap stage model at a time;
  do not substitute terminal-stage output logits for the full verifier unless a
  separate parity gate proves the teacher argmax is unchanged.
- `evals/spd/train_product_activation_head_only.py` and
  `evals/spd/score_product_activation_head_only.py` train/score a fresh SPD
  head from raw product tensors while loading AutoConfig only, not full base
  model weights. Use this for large native-package-first lanes such as
  Qwen3-Coder-480B S8.
- `evals/spd/export_product_serving_fixture.py` exports a serving-only fixture
  carrying row metadata and final norm weights for Rust request-path smoke. It
  is not a Python/reference parity fixture.
- `evals/spd/plan_hf_spd_qualification.py --qualification-mode
  native-package-fresh` plans the capped native package flow: package download,
  topology-only capture, conversion, head-only train/score, serving export,
  package-backed smoke, latency simulation, and upload.
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

For a `native-package-fresh` qualification run, a temporary
`spd-serving-fixture.safetensors` is acceptable for request-path smoke because
the server needs row metadata and final norm weights. Do not treat that file as
a replacement for `spd-parity-fixture.safetensors` in a final bundle or parity
claim.

The base model should remain a normal Mesh/Skippy model reference or layer
package so each node materializes its assigned stage through the existing
resolver. When stage 0 is configured with a resolved local Skippy layer package,
the SPD proposal source can replay boundary taps from package parts and read h0
from the package embedding part, including Q4_K and Q6_K token embeddings, so a
coordinator-side full GGUF override is no longer required for that shape.
`spd_model_path` remains useful as an explicit full-GGUF override for
lower-level smokes. Do not use `spd-openai-smoke --rsync-model-artifacts` as
product proof; that is a harness-only shortcut.

## Topology Checklist

Before training or evaluating a sidecar, write down:

- Base model repo/ref and revision.
- Target GGUF artifact, quant, and layer count if this is meant for Skippy.
- Tokenizer and chat template settings; for Qwen, explicitly decide thinking
  versus no-thinking template behavior.
- Logical SPD stage count and `stage_layer_boundaries`.
- Physical placement plan: one logical stage per node for the first proof, or
  explicit contiguous clumping of logical stages onto larger nodes. Clumped
  nodes must still return internal logical-boundary taps required by the
  manifest, and timing evidence must call out reduced physical overlap.
- Explicit `shallow_hidden_layer_indices` if the reference trainer needs taps
  that do not match simple stage boundaries.
- `num_spec_layers`, draft vocab choice, and `draft_top_k` for evaluation.
- Physical Skippy split boundaries that expose every required hidden tap.

## Quality Validation Without a Physical Split

Do not spend real-node time to discover whether a predictor is low quality.
Validate sidecar quality and artifact correctness locally before using Ethernet
or multi-node Mesh:

1. Run reference/HF held-out eval and report acceptance rate, equivalent
   accepted length, top-k target coverage, and theoretical saved decoder steps.
2. Export `spd-head.safetensors` and a parity fixture, then run Rust fixture
   parity to prove Python and Rust proposals match on fixed hidden-tap rows.
3. Run local live-tap parity with localhost stages for the logical split to
   prove Skippy returns every manifest-required tap.
4. Run local package-backed baseline versus SPD smoke and require nonzero
   accepted proposals plus nonzero saved candidate token round trips before
   moving to real nodes.

Treat the physical split as distributed-system validation: endpoint placement,
QUIC/LAN latency, per-stage KV cleanup, tap transport, and measured timing.

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
The current product-corpus smoke for this exact package/split is
`/tmp/spd-qwen3-8b-product-corpus-smoke`: `sample_count=1`, `row_count=2`,
`hidden_size=4096`, `rows_f32_bytes=32768`, and the old BF16-trained sidecar
still rejects the first product proposal (`proposal=9914`, `target=23`). Use
this as corpus-export evidence only, not quality or speed evidence.
That corpus converts to `/tmp/spd-qwen3-8b-product-corpus-smoke.safetensors`.
The matching one-sample HF teacher augmentation writes
`/tmp/spd-qwen3-8b-product-teacher-smoke.safetensors`; its teacher top-1 is
token `23`, matching the product greedy target for the captured row. A one-step
MPS BF16 fine-tune smoke writes
`/tmp/spd-qwen3-8b-product-finetune-smoke/speculation_head_final.pt` from the
existing LR `1e-4` Qwen3-8B sidecar. Treat this as data/training-bridge
evidence only: the teacher logits come from HF `Qwen/Qwen3-8B`, not native
Q4_K_M verifier logits, and the sample count is `1`.

Current exact-prompt Qwen3-8B product bridge: the unfine-tuned LR `1e-4`
checkpoint matched content on all `9 / 9` exact reference prompts but accepted
only `8 / 63` product proposals
(`/tmp/spd-qwen3-8b-identical-prompts-product-nt9.json`). Target tokens were
mostly aligned between reference and product (`60 / 63` positions, `7 / 9`
prompts exact), but proposal parity was poor (`10 / 63` proposal-token matches).
The product-corpus path now supports tokenized prompt JSONL and exported
`/tmp/spd-qwen3-8b-product-corpus-nt9.safetensors` from those exact prompts:
`72` samples, `9` prompts, `8` verify steps, `row_count=2`, `hidden_size=4096`.
HF teacher augmentation wrote
`/tmp/spd-qwen3-8b-product-teacher-nt9.safetensors` with draft-width BF16 logits
for all `72` samples; `71` labels were in draft-vocab scope. This is
KL-compatible product-tap data with an HF teacher, not native Q4_K_M verifier
logits.

Current product-distribution debug sidecar: a stronger local BF16 fine-tune
over the 72 product rows wrote
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/speculation_head_final.pt`
from the LR `1e-4` checkpoint using batch `8`, `10` epochs, LR `2e-5`, and
reached product-row argmax accuracy `0.875`. The BF16 serving export SHA is
`3b87a779034fd2974da76e3c368ee0000b5bbec5a735f3c7a7d3fec65c3d8866`, and Rust
fixture parity passes. Local package-backed serving on the same 9 prompts
accepted `42 / 63` proposals without the rolling executor and `44 / 59` with
the rolling executor, with exact content and `0` tap failures. The rolling
report is
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/openai-product-nt9-rolling.json`:
`54` rolling launches, `9` no-proposal launch misses, `max_in_flight=2`, `39`
oldest accepts, `15` oldest rejections, and `15` drained younger replies. Treat
this as a successful product-path training bridge and overfit debug sidecar, not
a final generalizing artifact.

Current one-worker LAN checkpoint for that debug sidecar: the no-launch
preflight at
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/openai-lan-preflight.json`
validated the package-backed `23,36` split and `[23,36]` tap allowlist. The
remote worker cache initially had only stage-0 package parts; copying the
selected downstream layer parts `23..35` and `shared/output.gguf` fixed package
materialization. The full 9-prompt paired LAN rolling report at
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/openai-lan-nt9-rolling.json`
matched content on all `9 / 9`, accepted `44 / 59` proposals, committed `39`
optimistic tokens, recorded `0` tap failures and `0` rolling launch misses, and
reached `max_in_flight=2`. Baseline decode mean was `554.0ms`; SPD decode mean
was `1702.7ms` (`0.325x`). The paper-style estimate was positive (`1.49x`,
`44` saved / `15` unsaved token round trips), so the remaining gap is concrete
overhead: sidecar head mean `69.5ms`, normal downstream wait mean `150.7ms`,
optimistic downstream wait mean `115.7ms`, and chained hidden-wait mean
`108.9ms`.
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
Qwen3-8B training beyond tiny plumbing, use `--model-torch-dtype bfloat16` on
this M4. A matching 64-row float16 run completed but produced all-non-finite
head tensors; the bfloat16 run stayed finite. Leave `auto` only for older small
proof heads or use `bfloat16` on CUDA/HF jobs after confirming the target
hardware.

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
tensors. The reference eval patch now propagates explicit
`stage_layer_boundaries` into the Python pipeline simulator, so the old
`KeyError: 23` custom-tap fill failure is fixed for boundary-derived rows. A
tiny debug eval of this 2-row head reported `24` generated tokens, `42` decode
steps, aggregate acceptance `0.5714`, equivalent accept length `1.1429`,
theoretical throughput gain `14.29%`, and `3 / 24` accepted draft flags. Treat
that as plumbing acceptance only; the real Qwen3-8B artifact still needs a
larger training run and same-topology baseline/SPD request-path comparison.
The package-backed local release smoke for the same `23,36` topology now opens
Q4_K h0 rows from the Qwen3-8B layer package, rejects wrong activation widths
before launching stages, matches baseline/SPD content, and records clean tap
counters. The paired report
`/private/tmp/spd-qwen3-8b-s2-23-debug-local-openai-paired-8.json` proposed
`7`, accepted `0`, rejected `7`, used `7` inline package taps with `0` replay
fallbacks, and measured `134.3ms` baseline decode versus `586.4ms` SPD decode.
This is request-path plumbing evidence only. With `0 / 7` accepted proposals,
there are zero critical-path token round trips saved, so the slowdown is
expected from sidecar overhead.

Current local Qwen3-8B finite training checkpoint: the bfloat16 64-row MPS run
at `/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-train64-20260618-104718/artifacts/20260618-104718`
uses the exact `23,36` topology, exported finite BF16 serving weights and a
finite parity fixture, and passed `skippy-bench spd-fixture-parity`
mechanically. Reference eval over `24` prompts / `384` generated tokens
reported aggregate acceptance `0.5378`, equivalent accept length `1.0756`, and
theoretical gain `7.59%`. Local package-backed `spd-openai-smoke` matched
baseline/SPD content but accepted `0 / 15` proposals on the default prompt,
with `15` inline package taps, `0` replay fallbacks, and `0` tap failures.
Treat this as finite training/export/request-path evidence, not speed evidence.
The export scripts now reject non-finite checkpoint or fixture tensors.
`spd-openai-smoke` reports candidate, saved, and unsaved token round trips under
`summary.paper_pipeline_estimate`; for this checkpoint the accepted-token round
trip math is `15` candidates, `0` saved, and `15` unsaved.

Current larger local Qwen3-8B checkpoint: the bfloat16 512-row MPS run at
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-train512-20260618-105916/artifacts/20260618-105916`
also used the exact `23,36` topology. It trained for `10.37min` with
`train_loss=28.96`, exported finite BF16 serving weights, and passed fixture
parity mechanically with tight tap-input reconstruction, but reference eval was
still weak: aggregate acceptance `0.5306`, equivalent accept length `1.0611`,
theoretical gain `6.21%`, and `135 / 1536` accepted draft flags. Package-backed
serving accepted `0 / 15` proposals on the default prompt and `0 / 90` across a
six-prompt code/math/writing sweep, with clean content and tap counters. Do not
run a two-node speed comparison with this head; it saves zero token round trips.
The next real sidecar step is a better or larger training recipe, likely a
confirmed HF-scale bfloat16/CUDA run or a training/config fix, until local
package-backed serving shows nonzero saved token round trips.

Current HF-scale Qwen3-8B checkpoint: HF job
`meshllm/6a33e49bef9220ea67d991c2` trained `Qwen/Qwen3-8B` for the exact S2
`23,36` topology on UltraChat `train_sft` with `15997` usable rows, BF16,
`max_length=2048`, `epochs=1`, LR `1e-4`, `num_spec_layers=4`, and draft
top-k `4`. Artifacts are in
`meshllm/skippy-spd-qwen3-8b-s2-23/runs/20260618-122936`. Reference held-out
eval reported aggregate acceptance `0.7013`, equivalent accept length
`1.4026`, and theoretical gain `41.0%` over `6123` generated tokens. Serving
export and Rust/Python fixture parity pass, but native Q4 live-tap strict
parity against the BF16 fixture does not; use package-backed top-1 acceptance
as the product gate.

Current request-path gate for that HF-scale head: local package-backed rolling
smoke matched baseline/SPD content on `6 / 6`, had `0` tap failures, proposed
`90`, accepted `17`, and rejected `73`. A real two-stage worker smoke over a
direct low-latency link matched content on `6 / 6`, had `0` tap failures,
proposed `89`, accepted `16`, rejected `73`, and committed `12` optimistic
tokens. Treat this as end-to-end mechanics evidence for the real split, not a
speed claim: native Q4 top-1 acceptance is only about `18%`, and the paper-like
round-trip estimate remains below break-even.

Current native-Q4 adaptation gate: the native verifier-logit capture path is
implemented and smoke-validated. The larger current gate is
`/tmp/spd-native-teacher-train137-v4`: `548` product-tap rows from `137` train
prompts and `4` verify steps, with native Q4 teacher logits over `32000` draft
tokens. `545 / 548` train labels are in scope and native teacher argmax matches
the Q4 target on `545 / 545`. The matching held-out capture
`/tmp/spd-native-teacher-heldout16-v4` has `64` rows; `61 / 64` labels are in
scope and native teacher argmax matches target on `61 / 61`. The unadapted 16k
head scored `51 / 545` train top-1 and `5 / 61` held-out top-1. The first
conservative native-Q4 warm start at
`/tmp/spd-native-q4-adapt-train137-v4-e3-lr1e5-hard025/` reached `401 / 545`
train top-1 and `496 / 545` train top-4, with held-out `20 / 61` top-1 and
`33 / 61` top-4. A short regularization sweep improved held-out top-1 to
`22 / 61` at `/tmp/spd-native-q4-adapt-train137-v4-e5-lr2e6-hard01/`
(`5` epochs, LR `2e-6`, weight decay `1e-2`, hard-label weight `0.1`), with
held-out top-4 still `33 / 61`. The top-4-best candidate was
`/tmp/spd-native-q4-adapt-train137-v4-e3-lr5e6-hard01/` with held-out
`20 / 61` top-1 and `34 / 61` top-4. Treat this as a material quant-specific
improvement but still not enough quality for a speed run. Next gate: broaden
native-Q4 train rows and tune regularization against the same held-out gate
before exporting or serving.

Current broadened reference-pool gate: `build_product_prompt_tokens.py` can
exclude a frozen held-out prompt-token file when raising context limits. The
train-only prompt set
`/private/tmp/spd-qwen3-8b-product-prompts-paper3-train-all-heldout16-frozen-max512`
contains `224` prompts under `512` tokens with the original `16` held-out
prompts excluded. Release `skippy-bench spd-live-tap-parity` captured
`/tmp/spd-native-teacher-train224-v8`: `1792` native-Q4 rows, exact native-logit
bytes, `1763 / 1792` labels in draft scope, and native teacher argmax matching
the Q4 target on `1763 / 1763` in-scope labels. The original 16k head scored
`253 / 1763` train top-1 and `583 / 1763` train top-4. The conservative
warm-start at `/tmp/spd-native-q4-adapt-train224-v8-e5-lr2e6-hard01/` improved
train to `1013 / 1763` top-1 and `1287 / 1763` top-4, but scored only
`21 / 61` held-out top-1 and `33 / 61` held-out top-4. The prior top-4 recipe
rerun at `/tmp/spd-native-q4-adapt-train224-v8-e3-lr5e6-hard01/` tied held-out
top-1 at `22 / 61` but regressed held-out top-4 to `32 / 61`. Treat this as
evidence that more rows from the small reference-eval pool are saturated.

Current serving-shaped UltraChat-native gate: `build_hf_prompt_tokens.py`
created `/private/tmp/spd-qwen3-8b-ultrachat-serving-v1-max480`, a fixed
`HuggingFaceH4/ultrachat_200k` `train_sft` shard with `1024` train prompts and
`256` held-out prompts rendered through the Qwen no-thinking chat template.
The held-out capture
`/tmp/spd-native-teacher-ultrachat-serving-v1-heldout256-v4-ctx1024` has
`1024` rows, `983 / 1024` labels in draft scope, and native teacher argmax
matching the Q4 target on `983 / 983`. On this gate, the original 16k head
scores `106 / 983` top-1 and `208 / 983` top-4; the reference-pool best scores
only `147 / 983` top-1 and `284 / 983` top-4; the 512-prompt UltraChat-native
adaptation scores `346 / 983` top-1 and `541 / 983` top-4; and the mixed
reference+UltraChat adaptation scores `332 / 983` top-1 and `540 / 983`
top-4. Treat the old `61`-row reference held-out gate as distribution-specific
debug evidence, not the serving-shaped quality gate.

Current scaled UltraChat-native checkpoint:
`/tmp/spd-native-teacher-ultrachat-serving-v1-train1024-v4-ctx1024` captured
`4096` native-Q4 train rows, with `3934 / 4096` labels in draft scope and
native teacher argmax matching target on `3934 / 3934`. The current best
warm-start is
`/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e5-lr2e6-hard01/`:
`5` epochs, LR `2e-6`, weight decay `1e-2`, hard-label weight `0.1`. It scores
`383 / 983` top-1 and `574 / 983` top-4 on the larger UltraChat held-out gate,
with train `1347 / 3934` top-1 and `2132 / 3934` top-4. This is a material
native-Q4 improvement, but still not a speed claim. The serving export
`/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e5-lr2e6-hard01/spd-head.safetensors`
has SHA `cab69fd4a9405819dc1a51afe058f1617995d0858702a2510313d600158349fe`,
and Rust fixture parity exits successfully for the exported manifest/fixture.
The bounded local package-backed rolling smoke on the first `16` UltraChat
held-out prompts matched content on `16 / 16`, had `0` tap failures, proposed
`41`, accepted `19`, and rejected `22`. The paper-style estimate is
`0.9268x` (`19` saved versus `22` unsaved candidate token round trips), so it
is still below break-even. Treat this as an acceptance and candidate-round-trip
savings gate for keeping the pipeline full, not measured wall-clock speed
evidence for the small local/two-node shape.

Do not use `--init-mode fresh` with the current projected product-corpus
safetensors. Those rows store `cur_in` after the manifest `stage_projs`
projection, so they are tied to the checkpoint projection basis that produced
them. A same-recipe fresh native-Q4 control scored `413 / 983` top-1 and
`523 / 983` top-4 offline on the old projected held-out rows, but accepted
`0 / 48` proposals in local serving because live taps were projected by the
fresh head's different `stage_projs`. The current projected-corpus path is valid
for checkpoint-mode warm-start adaptation from the same checkpoint
`stage_projs` basis, not merely the same logical topology. Proper direct
native-Q4 training requires raw terminal-normalized tap-concat rows before any
`stage_projs` projection, plus a trainer path that applies and trains
`stage_projs`.

Current raw-corpus implementation checkpoint: `spd-live-tap-parity` product
corpora now write both projected `rows.f32` and raw `raw_rows.f32`. The raw rows
are terminal-final-normed tap concatenations before `stage_projs`, with packed
row widths/offsets in `manifest.json`. `prepare_product_activation_corpus.py`
exports `raw_tap_concat`, `raw_tap_offsets`, and `raw_tap_widths`.
`train_product_activation_head.py --input-mode raw` projects raw rows through
`g0_proj` / `stage_projs` in the training graph; `--input-mode auto` keeps
checkpoint warm starts on projected rows and uses raw rows for fresh mode when
available. `score_product_activation_head.py --input-mode raw` scores through
the same projection path. A live raw smoke at
`/tmp/spd-raw-corpus-smoke-20260619` captured `3` samples with exact raw byte
counts and native Q4 teacher logits, converted to safetensors, ran a one-step
fresh raw train, and a tiny hard-label overfit reached `1 / 3` top-1 and
`3 / 3` top-4. Treat this as direct-training plumbing evidence only; next use a
larger disjoint raw train/held-out gate before export or serving. The first
disjoint raw gate at `/tmp/spd-raw-gate-20260619` has train16 and heldout16
corpora with `64` rows each; train has `60 / 64` labels in draft scope and
held-out has `59 / 64`, with native teacher argmax matching Q4 target on all
in-scope rows. Fresh raw training on train16 scored only `4 / 59` held-out
top-1 and `5 / 59` top-4. The existing current-best checkpoint scores that
held-out raw gate at `24 / 59` top-1 and `41 / 59` top-4, and a tiny raw-mode
checkpoint adaptation left it unchanged. Treat checkpoint raw adaptation as the
near-term production path; fresh raw is a future large-corpus research arm. The
scaled train64 raw adaptation from the current-best checkpoint improved the
frozen heldout16 top-1 to `28 / 59` and exported as
`/tmp/spd-raw-checkpoint-adapt-train64-v4-e3-lr5e6-hard01-20260619/bundle/`.
Rust fixture parity passed, and local package-backed rolling smoke matched
content on `16 / 16`, had `0` tap failures, proposed `39`, accepted `22`,
rejected `17`, and crossed the pipeline-fill gate with `22` saved versus `17`
unsaved candidate token round trips (`paper_like_speedup_vs_serial_split=1.1282x`).
Treat this as the first local acceptance/round-trip-savings win, not measured
distributed speedup.

Current broader raw gate: heldout64 at `/tmp/spd-raw-gate-20260619` has
`256` rows from `64` UltraChat held-out prompts, `241 / 256` labels in draft
scope, native teacher argmax matching Q4 target on every in-scope row, and zero
token-line overlap with the train shards. Scores on heldout64: original 16k
`23 / 241` top-1 and `49 / 241` top-4; current-best warm start `89 / 241` and
`140 / 241`; train128 raw `101 / 241` and `146 / 241`; train256 raw
`107 / 241` and `148 / 241`. The train128 bundle passed Rust fixture parity
and local package-backed rolling smoke matched content on `64 / 64` with
`0` tap failures, but accepted only `62 / 168` proposals: `62` saved versus
`106` unsaved candidate token round trips
(`paper_like_speedup_vs_serial_split=0.7381x`). The first `16` prompts were
barely positive (`21` saved / `18` unsaved), so heldout16 is only a smoke/debug
subset now. Do not run a real split until a broader held-out package-backed gate
clears saved > unsaved candidate token round trips with margin.

For larger SPD work, a Hugging Face job can be a valid pre-LAN gate: raw
product-tap capture, native-Q4 teacher-logit conversion, raw-mode checkpoint
adaptation or fresh head-only training, held-out scoring, serving export, and
local package-backed multi-stage smoke can all run on one machine. Fixture
parity is required when a true parity fixture exists; it is not provided by the
first `native-package-fresh` serving fixture. It is still not a full Mesh split
speed claim. Dry-run the model/package ref, dataset shard, topology, row cap,
hardware, timeout, output repo, and max cost before any spend-bearing submit.
The job should also emit a deterministic pipeline-economics report using
`evals/spd/simulate_latency.py --openai-report ...`. Sweep realistic physical
stage costs, LAN hop assumptions, and sidecar latency. Use `--sidecar-ms 0`
only as the paper's ideal hidden-sidecar scenario; otherwise include measured
`probe_head_total_ms` or an explicit sidecar budget. A candidate is not ready
for a real split if broad held-out acceptance fails the estimated break-even
under realistic physical placement.

Predigest SPD sidecars by canonical logical topology, not physical host count.
The sidecar manifest owns required tap boundaries. Mesh may fit contiguous
logical stages onto fewer physical nodes, but only if those internal taps remain
observable. Pipeline-fill estimates must use the fitted physical node buckets:
ten logical SPD stages colocated on three nodes are compatibility-valid only
with all taps exposed, and speed estimates have three physical compute buckets.
Base layer-package artifacts should remain per-stage-node downloads through the
normal Mesh/Skippy resolver. The SPD sidecar bundle is coordinator-owned by
default; workers should need only the manifest-derived tap-return allowlist.
Do not require every worker node to download or run the SPD predictor unless a
later design explicitly chooses distributed sidecar execution.

## First Larger Training Target

Use Qwen3-Coder-480B S8 for the next larger sidecar qualification proof. The
exact package target is
`meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`; its package
metadata reports `62` layers, activation width `6144`, and `256.98 GiB` of
package artifacts.

Start with logical S8 topology:

- `stage_layer_boundaries=8,16,24,32,40,48,55,62`
- required taps `[0,8,16,24,32,40,48,55,62]`
- `num_spec_layers=4`
- draft vocab capped at `32k`
- vocab size `151936` unless package metadata starts publishing `vocab_size`
- UltraChat rows first, then broaden after serving export and package smoke work

SPD must reuse the normal Skippy layer package. Skippy owns physical layer
ranges and layer package materialization; the SPD sidecar owns logical tap
requirements and proposal weights. The coordinator runs the predictor. Workers
only need the manifest-derived tap-return allowlist. If Mesh colocates
contiguous logical SPD stages onto fewer physical nodes, those nodes must still
return every internal logical-boundary tap required by the manifest.

Run this on Hugging Face, not on the local M4. Realistic lanes from 2026-06-19
HF Jobs rates:

- first capped native-package lane: `rtx-pro-6000x4` for `4.5h`, planned
  maximum `$49.50`
- memory-tighter alternative: `h200x2` for `5h`, planned maximum `$50`, but
  only `282GB` VRAM for a `256.98 GiB` package plus runtime/KV buffers
- memory-safer short lane: `h200x4` for `2.5h`, planned maximum `$50`
- longer quality decision: likely above `$50` unless setup/capture/training is
  already optimized

Do not use the existing full-HF-reference trainer as the Qwen480 path. The job
must be native-package-first: Skippy runs the Q4 layer package to capture taps
and native verifier logits; sidecar training must then train the head from those
tensors without loading `Qwen/Qwen3-Coder-480B-A35B-Instruct` through
Transformers.

Current checkpoint: topology-only capture, head-only train/score, serving
fixture export, and the `native-package-fresh` planner path are implemented and
locally syntax/compile checked. The Qwen480 dry run resolves the package shape,
S8 tap topology, `rtx-pro-6000x4`, `4.5h`, and max `$49.49991`, and the
generated command graph has no full-base train/score path. The setup path
installs build prerequisites, detects CUDA architecture, builds the CUDA ABI
with `just build-runtime`, and builds release `skippy-bench` / `skippy-server`.
The current capture plan emits
`--stage-backend-devices CUDA0,CUDA0,CUDA1,CUDA1,CUDA2,CUDA2,CUDA3,CUDA3`,
`--stream-live-tap-stages`, and `--product-native-teacher-logits true`. The
streamed mode is intended to fix the `55..62` CUDA3 OOM seen after full package
download by reducing tap-stage residency while preserving the full native Q4
verifier for teacher tokens/logits. Measure repeated stage-open time before
deciding whether the reload churn fits the capped lane.
The next Qwen480 retry must also use the two-phase product-corpus capture path:
record verifier target tokens and native draft-vocab logits first, drop the full
verifier model/session, then open streamed tap stages and replay the recorded
contexts to write product rows. This is required because the first streamed
retry reached capture but still OOMed opening stage `0..8` while the full
verifier was resident. A small cached Qwen3-0.6B package smoke passed this
two-phase path with streamed taps and native teacher logits.
If the local branch is not pushed, upload a patch artifact and set
`MESH_LLM_PATCH_PATH` so the job applies it after cloning. Remaining risk for
the first capped job is runtime compatibility with the Qwen480 MoE config and
package-backed capture startup under the timeout. True Rust/Python fixture
parity is still skipped until native parity fixture export exists.

Do not submit spend until the dry run prints model/package ref, dataset shard,
prompt counts, topology, hardware flavor, timeout, output repo, and max cost.

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
