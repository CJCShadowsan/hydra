# Next Goal: Resubmit Qwen480 S8 SPD Smoke-Existing With Prefill Tap Fix

This file is disposable. Durable evidence belongs in `evals/spd/README.md` and
`docs/skippy/speculative_decoding.md`.

## One-Line Goal

Resubmit the capped Hugging Face smoke-existing run for the already-uploaded
Qwen3-Coder-480B S8 SPD sidecar artifact, using the patched Skippy request path
that preserves stage-0 prefill taps through initial SPD source reset.

## Immediate Target

- Runner: Hugging Face Jobs in the `meshllm` org.
- Spend cap for the next submitted job: timeout `1.5h` on `rtx-pro-6000x4`,
  about `$16.50` maximum.
- No heavy local M4 training/capture loops for this target.
- Base/checkpoint family: `Qwen/Qwen3-Coder-480B-A35B-Instruct`.
- Exact package: `meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`.
- Package model id: `unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:UD-Q4_K_XL`.
- Package size from `model-package.json`: `256.98 GiB`.
- Package shape from `model-package.json`: `62` layers, activation width `6144`.
- Vocab size for native capture: `151936` (`--vocab-size 151936`).
- Logical SPD topology: S8,
  `stage_layer_boundaries=8,16,24,32,40,48,55,62`.
- Required SPD taps: `[0,8,16,24,32,40,48,55,62]`.
- Approx layer-package bytes by logical stage:
  `33.3,32.4,32.7,32.4,32.7,33.6,28.4,30.2 GiB`.
- Existing sidecar artifact:
  `meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/runs/native-package-fresh`.
- Smoke map: `CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`.
- Do not recapture/retrain unless the uploaded artifact is unusable.

## Skippy Layers vs SPD Topology

SPD must reuse the normal Skippy layer package. It does not get a separate copy
of model layers.

- Skippy physical stages own actual layer ranges and package materialization.
- The SPD sidecar owns logical tap requirements and proposal weights.
- Workers need only the manifest-derived tap-return allowlist.
- The coordinator owns the SPD bundle and runs the predictor by default.
- If Mesh packs multiple logical SPD stages onto one physical node, that node
  must still return the internal logical-boundary taps required by the manifest.

This lets us predigest sidecars for canonical logical topologies such as S4,
S6, and S8 instead of training every possible physical host grouping.

## Cost-Capped First Lane

The first submitted lane should do as much real native-package work as possible
under `$50`, but it must not load the full HF base model through Transformers.
It should either produce a usable first sidecar or fail/expire with enough logs
to identify the next bottleneck.

Candidate HF flavors from the current Jobs hardware list:

- `rtx-pro-6000x4`: `384GB` VRAM, `1024GB` RAM, about `$11/hr`; cap around
  `4h30m` to stay under `$50`.
- `h200x2`: `282GB` VRAM, `512GB` RAM, about `$10/hr`; cap `5h`, but this is
  tight for a `256.98 GiB` package plus KV/runtime buffers.
- `h200x4`: `564GB` VRAM, `1024GB` RAM, about `$20/hr`; cap `2h30m`, more
  memory-safe but probably too short for a useful train/capture cycle.

First submitted target was `rtx-pro-6000x4`, timeout `4h30m`, max cost `$50`.
The next retry is narrower: `rtx-pro-6000x4`, timeout `1.5h`, max cost about
`$16.50`, because capture/train/export already completed and the uploaded
artifact should be reused.

## Steps

1. Done: add a topology-only native-capture/bootstrap mode.
   - Inputs: package ref, S8 boundaries, hidden size, draft vocab, prompt shard.
   - Output: raw tap rows, native-Q4 teacher logits, row metadata, and a minimal
     topology shape.
   - It must not require an existing trained SPD head.
   - Implemented as `skippy-bench spd-product-corpus-capture`.
2. Done: add or wire a head-only trainer/scorer for captured native rows/logits.
   - It builds the SPD predictor from topology/config only.
   - It trains `stage_projs`, SPD layers, and draft-vocab head from raw rows.
   - Final norm is captured from the package and fixed, matching serving.
   - It must not instantiate `Qwen/Qwen3-Coder-480B-A35B-Instruct` with
     Transformers.
   - Implemented as `train_product_activation_head_only.py` and
     `score_product_activation_head_only.py`; both load AutoConfig only.
3. Done for serving, not parity: export and validate artifacts.
   - Emits `skippy-spd-head.json`, `spd-head.safetensors`, and
     `spd-serving-fixture.safetensors`.
   - Rust request-path serving can use the serving fixture for row metadata and
     final norm.
   - True Rust fixture parity is skipped for `native-package-fresh` until a
     native parity fixture exporter exists.
4. Next inside the capped job: rerun package-backed local-on-HF smoke.
   - Use the same Qwen480 package directory and S8 split.
   - Verify stage-0 hf `8` prompt-window rows are no longer missing after the
     initial SPD source reset.
   - Check content match, tap failures, accepted/proposed counts, and
     saved/unsaved candidate-token round trips.
5. Next inside the capped job: emit pipeline economics.
   - Run latency simulation for S8 clumped to 4 physical buckets and hop
     assumptions `0.2,1,5,10ms`.
   - Treat this as HF-side predictor/economics evidence, not LAN wall-clock
     proof.

## Follow-On HF Meshlet Spike

Do not dispatch a separate HF meshlet while the Qwen480 sidecar job is still
trying to produce its first usable artifacts. The meshlet becomes the next
short spike only after the current lane has enough evidence to carry:

- held-out native teacher summaries exist and have coherent in-scope target
  coverage;
- training/scoring completed or failed with an actionable quality result;
- the serving bundle exported;
- package-backed rolling `spd-openai-smoke` matched baseline content, recorded
  zero tap failures, and reported useful accepted/proposed plus saved/unsaved
  candidate-token round trips.

When those gates clear, run the first HF meshlet as one HF Job, not multiple HF
Jobs: start a coordinator, stage-server processes, SPD sidecar, and OpenAI
frontend as separate local processes inside the job, with optional synthetic
per-stage latency. Success for that spike is process lifecycle, package
materialization, tap return, SPD proposal/verification, rolling cleanup, and a
repeatable pipeline-economics report. Multiple HF Jobs with exposed ports are a
later transport spike, not the first end-to-end validation path.

## First Capped Run

Planner dry run used before submission:

```bash
python3 evals/spd/plan_hf_spd_qualification.py \
  --base-model Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --package-ref meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers \
  --qualification-mode native-package-fresh \
  --num-stages 8 \
  --stage-layer-boundaries 8,16,24,32,40,48,55,62 \
  --num-spec-layers 4 \
  --draft-top-k 4 \
  --draft-vocab-size 32000 \
  --vocab-size 151936 \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-prompts 512 \
  --heldout-prompts 64 \
  --max-prompt-tokens 480 \
  --verify-steps 4 \
  --ctx-size 1024 \
  --physical-node-count 4 \
  --logical-stage-ms 40 \
  --hop-ms 0.2,1,5,10 \
  --flavor rtx-pro-6000x4 \
  --timeout 4.5h \
  --max-cost-usd 50 \
  --output-repo meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8 \
  --out /tmp/spd-qwen480-s8-native-package-fresh-plan.json \
  --json
```

Latest dry-run checkpoint after the capture-arg fix: this command resolves the package metadata as
`62` layers, activation width `6144`, package model id
`unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:UD-Q4_K_XL`, hardware
`rtx-pro-6000x4`, timeout `16200s`, and max cost `$49.49991`. The generated
native-package command graph contains no `AutoModelForCausalLM`, no
`hf_train_eval_qwen06.py`, no `spd-live-tap-parity`, and no warm-start path.
The generated `spd-product-corpus-capture` command now emits
`--product-native-teacher-logits true`, matching the native capture CLI's
`ArgAction::Set` boolean shape. The generated HF setup no longer asks pip to
upgrade/install `torch`; the PyTorch CUDA base image supplies it.
The current local dry run also emits
`--stage-backend-devices CUDA0,CUDA0,CUDA1,CUDA1,CUDA2,CUDA2,CUDA3,CUDA3` and
`--stream-live-tap-stages` for capture. Streaming preserves the existing full
native Q4 verifier session for teacher logits and opens only one tap-stage
model at a time, which is the planned fix for the `55..62` CUDA3 OOM.
The setup path now installs Rust/`just`/build prerequisites, detects the CUDA
architecture, builds the CUDA stage ABI with `just build-runtime`, and then
builds `target/release/skippy-bench` plus `target/release/skippy-server`.
Dispatch should use an HF-uploaded patch artifact via `MESH_LLM_PATCH_PATH`
instead of requiring a GitHub push from this machine.
`evals/spd/bootstrap_qwen480_s8_native_job.sh` is the job entrypoint for this:
it downloads the uploaded patch, applies it to a bootstrap clone, regenerates
the reviewed plan, then runs `run_hf_spd_qualification_plan.py`.

Latest failed job used the same parameters through an HF-uploaded
patch/bootstrap artifact, because the local branch was not pushed to GitHub from
this machine:

```bash
id=6a3535603093dba73ce2a264
url=https://huggingface.co/jobs/meshllm/6a3535603093dba73ce2a264
run_id=20260619T122507Z-b843a851
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T122507Z-b843a851
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T122507Z-b843a851/
upload_commit=80014e284aa1e727a305c3ff5c44fb2ca82659d6
patch_sha256=7ec74581ee16e30ce4d56b99a5b0092eb8fc513b92c36acae8fbf8a93d952436
bootstrap_sha256=378a4bc91ff2c4aadeffa2a501180aafba44bedc2df377598bc0a3f3ce8ab6d6
dry_run_plan_sha256=542b9b61a118ee5a4a6a68d103ab1614bc020371129e233fe1d8e8bc93c4e7c6
```

Failed capture-argument resubmission used the same parameters through the fixed
uploaded patch/bootstrap artifact:

```bash
id=6a353b9d3093dba73ce2a2bf
url=https://huggingface.co/jobs/meshllm/6a353b9d3093dba73ce2a2bf
run_id=20260619T125208Z-22663dd2
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T125208Z-22663dd2
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T125208Z-22663dd2/
upload_commit=da3c7956783e86c3e50368ddbd32c00286f263df
patch_revision=da3c7956783e86c3e50368ddbd32c00286f263df
patch_sha256=450002e81f41b6adaf72c997ecad28700e29f2faf191c7c93d1aceb06e76757f
bootstrap_sha256=378a4bc91ff2c4aadeffa2a501180aafba44bedc2df377598bc0a3f3ce8ab6d6
dry_run_plan_sha256=dcce197cb092662ae7048df92f65356833fcb6d60b3c4630613942deb739f78a
```

Failed streamed-capture resubmission used the same `$50` capped lane with the
streamed live-tap capture patch:

```bash
id=6a354843953ed90bfb944848
url=https://huggingface.co/jobs/meshllm/6a354843953ed90bfb944848
run_id=20260619T134535Z-595b67cb
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T134535Z-595b67cb
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T134535Z-595b67cb/
upload_commit=9198f2468ae69dbb13c0d0a16f7b99c0e3e7dd5d
patch_revision=9198f2468ae69dbb13c0d0a16f7b99c0e3e7dd5d
patch_sha256=717b871d6668ad895869013f8a20168160bc46557b927ade1473258dea369c61
bootstrap_sha256=378a4bc91ff2c4aadeffa2a501180aafba44bedc2df377598bc0a3f3ce8ab6d6
dry_run_plan_sha256=e33d546eb7b3b3d639441fca7331f85fc0addf85d00623bb3cb7fb7b5966d9de
```

It failed after 209 seconds, before model work, because
`crates/skippy-server/src/frontend/spd.rs` still initialized
`SpdLiveTapRunnerConfig` without the new `stage_backend_devices` and
`stream_stages` fields. Commit `d19da20d` fixed that initializer. Commit
`f23e28ba` made the job timeout configurable so the retry could run shorter
than the original `4.5h` lane.

Failed streamed-capture retry used the fixed server config and a shorter
timeout:

```bash
id=6a354a2f953ed90bfb94486f
url=https://huggingface.co/jobs/meshllm/6a354a2f953ed90bfb94486f
run_id=20260619T135416Z-f23e28ba
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T135416Z-f23e28ba
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T135416Z-f23e28ba/
upload_commit=fc2f95dd543955f1e821c7036bebd0e48501974f
patch_revision=fc2f95dd543955f1e821c7036bebd0e48501974f
patch_sha256=58cfa43179ce251784014955f43f02092ae0936cb7df75a1a4e545f9f2c8b6bc
bootstrap_sha256=30d27fa808c08df2f3ca1613381de1ca0a828694e66448f3bc03e55b2610cb05
dry_run_plan_sha256=61ffa3a560948536e9fc4df7e7dd4c178f36ab4309dbe340e37c63de02d5a9d5
```

It failed after reaching `capture[0]` because CUDA0 could not open streamed
tap stage `0..8` while the full verifier was still resident. Commit `3d1442f8`
changed `spd-product-corpus-capture` to a two-phase flow: record verifier
targets/native draft-vocab logits first, drop the verifier, then replay
contexts through live tap stages.

The first two-phase retry used the `3d1442f8` patch artifact and the same
shorter timeout:

```bash
id=6a35536b3093dba73ce2a377
url=https://huggingface.co/jobs/meshllm/6a35536b3093dba73ce2a377
run_id=20260619T143116Z-3d1442f8
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T143116Z-3d1442f8
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T143116Z-3d1442f8/
upload_commit=abaefe222379e5bd6f949ebec7ca37de79faf715
patch_revision=abaefe222379e5bd6f949ebec7ca37de79faf715
patch_sha256=9f623c5d3f6d5f9aa34b10e72b9849a435794634faecc497d363c3e05bd0afe1
bootstrap_sha256=30d27fa808c08df2f3ca1613381de1ca0a828694e66448f3bc03e55b2610cb05
dry_run_plan_sha256=61ffa3a560948536e9fc4df7e7dd4c178f36ab4309dbe340e37c63de02d5a9d5
```

It was manually canceled after proving the old allocator failure point but
before timeout. It completed release `skippy-bench`/`skippy-server` builds,
downloaded the full Qwen480 layer-package snapshot (`276G / 276G`, `69`
files), built the prompt dataset shards, entered native capture, and logged the
streamed stage `0..8` allocation as
`CUDA0 model buffer size = 34051.88 MiB` instead of failing with `cudaMalloc`.
The reason for canceling was not a correctness failure: code inspection showed
`--stream-live-tap-stages` reopens all eight tap stages for each prompt/step,
so `512` train prompts plus `64` held-out prompts at `4` verify steps would
burn the cap on repeated stage-open churn before training or smoke.

Next retry: keep the two-phase verifier drop but run resident tap stages, not
streamed tap stages. The previous resident OOM happened while the full verifier
was still loaded; after phase 1 exits, the resident S8 tap stages should fit on
`rtx-pro-6000x4` with the existing device map. The planner now defaults to
resident tap stages and only emits `--stream-live-tap-stages` when explicitly
requested.

First artifact-producing profile for the next retry:

```bash
TRAIN_PROMPTS=32
HELDOUT_PROMPTS=8
VERIFY_STEPS=1
STREAM_LIVE_TAP_STAGES=false
JOB_TIMEOUT=2h
```

The matching dry run resolves resident tap capture, `rtx-pro-6000x4`, timeout
`7200s`, max cost about `$22`, no `AutoModelForCausalLM`, no
`hf_train_eval_qwen06.py`, no `spd-live-tap-parity`, and no
`--stream-live-tap-stages`. Treat this as a mechanics/artifact lane, not final
sidecar quality. If resident stages fit and this reaches capture/train/export
smoke, raise the row count in a later capped quality lane.

Submitted resident-small retry:

```bash
id=6a3563743093dba73ce2a4ab
url=https://huggingface.co/jobs/meshllm/6a3563743093dba73ce2a4ab
run_id=20260619T154157Z-52a4ffdf
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T154157Z-52a4ffdf
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T154157Z-52a4ffdf/
upload_commit=509ed659ccfb237892e9329887f1a6261f352bf1
patch_revision=509ed659ccfb237892e9329887f1a6261f352bf1
patch_sha256=57bb1e3180923531641384af307bfb41e6ee05fb0c75795916e93cd2ddc25645
bootstrap_sha256=c8e2efa7ec104ec6c38d6b8584cd8851ea84692b73ff3df40dcb5fe08e79a022
dry_run_plan_sha256=3936027d9c94951a8667a9d7c58e186064c58df49e1d4ca51a3563b3bbd6e5e4
```

The job is labeled `spd-qwen480-resident-small` and was submitted with
`TRAIN_PROMPTS=32`, `HELDOUT_PROMPTS=8`, `VERIFY_STEPS=1`,
`STREAM_LIVE_TAP_STAGES=false`, and `JOB_TIMEOUT=2h`.

Observed resident-small result: it failed after `1424s` running, but only after
the important mechanics gates cleared. The job completed release build,
downloaded the full `69`-file / `276G` package snapshot, built prompts, loaded
the full Qwen480 verifier across four RTX PRO 6000 GPUs, phase-separated
verifier capture from tap replay, opened resident S8 tap stages, converted
train and held-out native corpora, trained the head-only predictor, scored
held-out, and exported an `8.72GB` BF16 serving head. Train had `32` samples
with `31 / 32` labels in draft scope; held-out had `8 / 8` labels in scope.
Head-only training reported `base_model_load=skipped`, final hard-label
accuracy `1.0` on the tiny train set, and held-out `2 / 8` top-1 plus `5 / 8`
top-4. The failure was a generated shell quoting bug in the
`rust_fixture_parity` skip command: the script ran `echo ...; Rust fixture ...`,
so Bash tried to execute `Rust` and exited `127`.

Next retry: resubmit the same resident-small profile with the generator fixed
to emit the parity-skip message as one `printf` command. Local validation
passes for `python3 -m py_compile` on the planner/runner, the same 32/8/1 dry
run, and `run_hf_spd_qualification_plan.py --groups rust_fixture_parity`. The
first new gate is package-backed rolling smoke and upload, because capture,
train, score, and export already worked once.

Submitted fixed resident-small retry:

```bash
id=6a356b6d3093dba73ce2a5da
url=https://huggingface.co/jobs/meshllm/6a356b6d3093dba73ce2a5da
run_id=20260619T161546Z-a6dae908
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T161546Z-a6dae908
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T161546Z-a6dae908/
upload_commit=f57a5053d8c1ff20ca74798dd076fcb317a6038a
patch_revision=f57a5053d8c1ff20ca74798dd076fcb317a6038a
patch_sha256=b432270244c258d9621f05afe1a8de455ba3d540b445d6a158e56f33f4d3bc25
bootstrap_sha256=c8e2efa7ec104ec6c38d6b8584cd8851ea84692b73ff3df40dcb5fe08e79a022
dry_run_plan_sha256=2493c61db82bc1e0d77b684dc606256ead3bbc54e3679a6e66c5345e01bf763f
```

The job is labeled `spd-qwen480-resident-small-fixed` and uses the same
`TRAIN_PROMPTS=32`, `HELDOUT_PROMPTS=8`, `VERIFY_STEPS=1`,
`STREAM_LIVE_TAP_STAGES=false`, and `JOB_TIMEOUT=2h` profile. At submission it
was scheduling. First gate: it should replay the already-proven capture path,
then continue past `rust_fixture_parity` into package smoke and upload.

Observed fixed resident-small result: the job is `ERROR` after `1383s` running,
but it passed the prior parity-skip failure and repeated the useful native
mechanics gates. It again completed build, full `69`-file / `276G` package
download, Qwen480 verifier load, two-phase verifier capture plus resident tap
replay, native train/held-out conversion, head-only training, held-out scoring,
and BF16 serving export. Held-out score was again `2 / 8` top-1 and `5 / 8`
top-4. The export SHA was
`3fcdb93eeea5d23c4ae3df3dc39e10e70f59564a2ab20820f09aa0a7a5fe3f9d`.
The new failure is package-backed smoke readiness: `baseline OpenAI frontend
did not become ready`, with `127.0.0.1:<port>/v1/models` returning connection
refused. Because upload was still after smoke, the expensive sidecar artifact
was not durably uploaded by that job.

Local fix checkpoint for the next retry: `spd-openai-smoke` now collects and
prints bounded stage-log tails after OpenAI readiness failure, the
`native-package-fresh` plan uploads artifacts in `upload_pre_smoke` before
package smoke, and native-package smoke now uses
`--startup-timeout-secs 600 --request-timeout-secs 600` with
`--work-dir /workspace/spd-qualification/artifact/openai-smoke-work`. The local
32/8/1 dry run resolves `rtx-pro-6000x4`, timeout `7200s`, max cost
`$21.99996`, S8 boundaries `8,16,24,32,40,48,55,62`, no
`AutoModelForCausalLM`, no `hf_train_eval_qwen06.py`, no `spd-live-tap-parity`,
and no `--stream-live-tap-stages`. The generated smoke command includes the
600s timeouts and artifact-owned work dir, and the command graph includes
`upload_pre_smoke` before `package_smoke`.

Submitted observable resident-small retry:

```bash
id=6a3575be3093dba73ce2a692
url=https://huggingface.co/jobs/meshllm/6a3575be3093dba73ce2a692
run_id=20260619T165954Z-76662252
local_artifact_dir=/tmp/spd-qwen480-native-job-20260619T165954Z-76662252
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T165954Z-76662252/
upload_commit=83a6631a29fcb534057d34353d9e78a2d248cbf3
patch_revision=83a6631a29fcb534057d34353d9e78a2d248cbf3
patch_sha256=dc33c52493b3c6bc2bcf478052e57b4700ecaaf77fe3b323d7bfcc612bf37c10
bootstrap_sha256=c8e2efa7ec104ec6c38d6b8584cd8851ea84692b73ff3df40dcb5fe08e79a022
dry_run_plan_sha256=d04bb5d3bbe785e7ca572dabee59c9eed1cc817e7bdeb9084fc2c199962217bf
```

This job is labeled `spd-qwen480-resident-small-observable`, uses
`rtx-pro-6000x4`, `TRAIN_PROMPTS=32`, `HELDOUT_PROMPTS=8`, `VERIFY_STEPS=1`,
`STREAM_LIVE_TAP_STAGES=false`, and `JOB_TIMEOUT=2h`. First gates: reach
`upload_pre_smoke` after export, then either pass package smoke or fail with
stage-log tails showing why the baseline OpenAI frontend did not bind.

Observed result: the job is `ERROR` after `1898s`, but it reached and uploaded
the real sidecar artifact before smoke. It repeated release build, full Qwen480
package download, verifier load, two-phase native target/logit capture,
resident tap replay, native conversion, head-only train/score, serving export,
and `upload_pre_smoke`. Train conversion had `31 / 32` labels in draft scope;
held-out had `8 / 8`; held-out quality was `2 / 8` top-1 and `5 / 8` top-4.
The uploaded `spd-head.safetensors` is `8,723,214,136` bytes with SHA
`f77dbfb1f83a1c3a79446b983c7de3e77f63c22f4bacbd8ae0d92efbeef3fc75`.
Artifacts are available under
`meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/runs/native-package-fresh`.

The package smoke failure is now understood: stage `1` was already listening on
CUDA0, then stage `0` tried to allocate a `34051.88 MiB` CUDA0 buffer and
failed with `cudaMalloc failed: out of memory`; stages `1..7` otherwise
started cleanly. This is a smoke placement bug. It does not invalidate the
native capture, head-only training, export, or upload gates.

Current local patch for the next run: `spd-openai-smoke` accepts
`--stage-backend-devices`; the HF planner accepts
`--smoke-stage-backend-devices`; `bootstrap_qwen480_s8_native_job.sh` passes
`SMOKE_STAGE_BACKEND_DEVICES`; and
`bootstrap_qwen480_s8_smoke_existing_job.sh` can hydrate the uploaded artifact
and run only package smoke, latency simulation, and upload.

Next immediate run should be smoke-existing, not recapture/retrain:

```bash
SMOKE_STAGE_BACKEND_DEVICES=CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3
ARTIFACT_RUN_PATH=runs/native-package-fresh
JOB_TIMEOUT=1.5h
```

The dry run for this shape resolves `rtx-pro-6000x4`, timeout `5400s`, planned
max `$16.49997`, package `62` layers / width `6144`, S8 boundaries
`8,16,24,32,40,48,55,62`, and emits a package-smoke command with the explicit
CPU/GPU map. The selected smoke-existing command groups avoid
`AutoModelForCausalLM`, `hf_train_eval_qwen06.py`, and the old reference path.

Submitted smoke-existing retry:

```bash
id=6a3581b9953ed90bfb944dd3
url=https://huggingface.co/jobs/meshllm/6a3581b9953ed90bfb944dd3
run_id=20260619T175052Z-588fe084
local_artifact_dir=/tmp/spd-qwen480-smoke-existing-job-20260619T175052Z-588fe084
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T175052Z-588fe084/
upload_commit=1dce6e58644cf957e517f430ac653df6257fc455
patch_revision=1dce6e58644cf957e517f430ac653df6257fc455
patch_sha256=e9947039a0f2cbe0ba127793dbd0fa054bd6f1cbb23b996059ed501e422d89f1
bootstrap_sha256=969e6005f7e9e108c2bca0ad0b50c430cc0cf7b9b78db577beee9859209c07d8
dry_run_plan_sha256=7fba2ddb364e6ad8eab8bbd4f78ac01b4e359614cf124643a4771da1325a5012
```

This job failed after `1169s` running, but it reached the important
smoke-existing preflight gates: release build, full package download,
deterministic prompt rebuild, and hydration of the uploaded
`runs/native-package-fresh` artifact all completed. It failed before launching
models because `package_smoke` ran as a second generated script from the
bootstrap checkout, where `target/release/skippy-bench` did not exist. This is
a split-script cwd bug, not a package, sidecar, or Qwen480 runtime failure.

Fixed in local commit `bf682379`: after hydrating the artifact,
`bootstrap_qwen480_s8_smoke_existing_job.sh` now `cd`s into
`$WORK_DIR/mesh-llm`, where setup built the release binaries and wrote
`physical-stage-ms.txt`, and fails fast if those files are missing.

Submitted fixed smoke-existing retry:

```bash
id=6a35894f953ed90bfb944e49
url=https://huggingface.co/jobs/meshllm/6a35894f953ed90bfb944e49
run_id=20260619T182322Z-bf682379
local_artifact_dir=/tmp/spd-qwen480-smoke-existing-job-20260619T182322Z-bf682379
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T182322Z-bf682379/
upload_commit=ea52905865b07ad17a5fdd7519d27a07ad4f689c
patch_revision=ea52905865b07ad17a5fdd7519d27a07ad4f689c
patch_sha256=af3c6353916a96b9a6568004c11ad32b0b788d06cf31182458b8a3cdee29799d
bootstrap_sha256=173a370c1f7b9f0b2bac501d866fd835e26c397be39024817a717d7ba3c325e7
dry_run_plan_sha256=7fba2ddb364e6ad8eab8bbd4f78ac01b4e359614cf124643a4771da1325a5012
```

This job is labeled `spd-qwen480-smoke-existing`, uses `rtx-pro-6000x4`,
`JOB_TIMEOUT=1.5h`, `ARTIFACT_RUN_PATH=runs/native-package-fresh`, and
`SMOKE_STAGE_BACKEND_DEVICES=CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`. First
poll showed it past bootstrap download and into generated setup. First gates
remain: patch apply, selected-group setup/package download, artifact hydration,
then package smoke and latency simulation.

Result: `ERROR` after `1621s`, but the cwd fix and package-smoke launch were
proved. The job completed release build, full package download, prompt rebuild,
artifact hydration, and package-backed baseline/SPD smoke. Tap transport was
clean (`0` return failures, `0` record failures, `0` ignored taps), and
downstream taps returned for `16,24,32,40,48,55,62`; stage 0 recorded hf `8`.
The request-path bug was that every SPD case proposed `0` tokens because hf `8`
prompt-window rows such as `40..45` / `58..63` were missing from the proposal
cache. Root cause is fixed locally: initial `reset_to_context(prompt)` now
preserves prefilled tap rows when the source context was empty. The latency
simulator also now accepts the intended four-physical-bucket clumped model
against an eight-stage smoke report.

Submitted current-patch smoke-existing retry:

```bash
id=6a3593cf3093dba73ce2a78f
url=https://huggingface.co/jobs/meshllm/6a3593cf3093dba73ce2a78f
run_id=20260619T190753Z-6abc8370
local_artifact_dir=/tmp/spd-qwen480-smoke-existing-job-20260619T190753Z-6abc8370
output_repo=meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8
input_prefix=job-inputs/20260619T190753Z-6abc8370/
upload_commit=43940c19fefce860f58c37ebe0517a13d32f8419
patch_revision=43940c19fefce860f58c37ebe0517a13d32f8419
patch_sha256=4bca067da32ae42067845663ac345e515feeae8abd57aa0ad883de6b9458f15a
bootstrap_sha256=173a370c1f7b9f0b2bac501d866fd835e26c397be39024817a717d7ba3c325e7
dry_run_plan_sha256=7fba2ddb364e6ad8eab8bbd4f78ac01b4e359614cf124643a4771da1325a5012
```

An immediately previous submission `6a3593b5953ed90bfb944ef8` failed in `7s`
with exit `126` because the HF CLI command omitted the `--` terminator and
parsed `-lc` as a job flag. The corrected job above inspected as
`['bash', '-lc', ...]` and started in `SCHEDULING`.

Startup attempts before the latest two-phase retry:

- `meshllm/6a35304a953ed90bfb9446a8` failed in 3 seconds with exit `126`
  because the HF CLI invocation passed the multiline script to `bash` as a
  filename.
- `meshllm/6a3531c3953ed90bfb9446e2` failed the same way; root cause was
  missing the HF CLI `--` option terminator before container command flags, so
  `-lc` was parsed as a job label instead of a Bash argument.
- CPU canary `meshllm/6a3531e9953ed90bfb9446e4` proved the corrected
  `hf jobs run ... -- <image> bash -lc <script>` form and printed
  `hf-command-ok`.
- `meshllm/6a3532083093dba73ce2a204` reached the bootstrap script, cloned the
  repo, and failed in 16 seconds because `BOOTSTRAP_DIR` was not exported to
  the Python patch downloader. Fixed in commit `bcec1f4f`.
- `meshllm/6a35325c3093dba73ce2a206` reached patch apply and plan
  construction, then failed after 17 seconds because
  `/workspace/spd-qualification` did not exist before writing
  `native-package-fresh-plan.json`. Fixed in commit `861c2450`; the planner now
  creates `--out` parents, and bootstrap also creates `$WORK_DIR`.
- `meshllm/6a353427953ed90bfb944722` reached generated setup and failed after
  90 seconds at
  `MESH_LLM_SKIP_UI=1 MESH_LLM_BUILD_PROFILE=release just build-runtime backend=cuda cuda_arch="$CUDA_ARCH"`.
  The repo's `just build-runtime` recipe takes positional parameters, so the
  generated command must be `just build-runtime cuda "$CUDA_ARCH"`. Local dry
  run now emits the corrected command.
- `meshllm/6a3535603093dba73ce2a264` ran for 1189 seconds, passed CUDA
  `build-runtime`, built the Rust release binaries, downloaded the full
  Qwen480 package snapshot (`69` files, about `276G`), and built disjoint
  UltraChat prompt-token files (`512` train prompts, `64` held-out prompts,
  train mean `101.3` tokens, held-out mean `103.8` tokens). It failed at the
  first `spd-product-corpus-capture` command because the planner emitted
  `--product-native-teacher-logits` without the required `true` value. Local
  dry run now emits `--product-native-teacher-logits true`.
- `meshllm/6a353b9d3093dba73ce2a2bf` ran for 1249 seconds and cost about
  `$3.82` at the planned `$11/hr` rate. The command graph used the fixed
  `--product-native-teacher-logits true` argument, built CUDA/Rust release
  binaries, downloaded the full package, generated the `512`/`64` prompt shards,
  and reached `capture[0]`. It failed while opening topology-only package stage
  `55..62`: CUDA3 could not allocate a `30905.58 MiB` model buffer. This is a
  live-runner memory-residency issue, not another planner/reference-path issue.

Older status check on 2026-06-19: job
`meshllm/6a3535603093dba73ce2a264` is `ERROR` with exit code `1`. It reached
real package download and prompt construction but did not start capture rows,
training, scoring, export, or smoke because of the fixed capture boolean
argument issue.

Status check on 2026-06-20 local time: streamed-capture job
`meshllm/6a354a2f953ed90bfb94486f` is `ERROR` after 1226 seconds. It cleared
inline bootstrap download/startup, CUDA ABI build, Rust release builds, Python
dependency setup, full 69-file Qwen480 layer-package download (`276G / 276G`),
and prompt-token shard building, then reached `capture[0]`. It failed opening
streamed tap stage `0..8`: CUDA0 could not allocate a `34051.88 MiB` buffer
while the full verifier remained resident.

Local fix checkpoint: `spd-product-corpus-capture` now runs target generation
and native draft-vocab logit capture first, drops the full verifier model, then
opens the streamed tap runner and replays the recorded contexts to write final
product rows. This preserves target/logit semantics while avoiding verifier and
tap-stage model residency overlap. A real cached package smoke passed against
`meshllm_Qwen3-0_6B-Q4_K_M-layers-test-main` with `--splits 14 --layer-end 28`,
`--stream-live-tap-stages`, one prompt, one verify step, native teacher logits,
and matching product row byte counts:
`/tmp/spd-two-phase-smoke-report.json`.

Cost status on 2026-06-20: the latest artifact-producing observable job ran
`1898s`, about `$5.80` at the planned `$11/hr` rate. Including the prior
serious GPU retries puts completed Qwen480 GPU spend roughly in the mid-to-high
`$20s`. The next smoke-existing run is timeout-capped at `1.5h`, about
`$16.50` max on `rtx-pro-6000x4`, and avoids repeating capture/train.

Prior-job inspection commands:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --from huggingface_hub hf jobs inspect meshllm/6a3535603093dba73ce2a264
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --from huggingface_hub hf jobs logs meshllm/6a3535603093dba73ce2a264
```

Latest failed-job inspection commands:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --from huggingface_hub hf jobs inspect meshllm/6a3575be3093dba73ce2a692
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --from huggingface_hub hf jobs logs --tail 6400 meshllm/6a3575be3093dba73ce2a692
```

## Remaining Risks During Run

- Smoke-existing still has to prove that the uploaded serving bundle can drive
  the request path with the package-backed S8 split.
- `native-package-fresh` exports a serving fixture, not a true Python/reference
  parity fixture. Do not claim Rust/Python fixture parity for this lane yet.
- The default smoke map uses four CPU stages to avoid GPU overcommit. This is
  acceptable for a mechanics smoke, but it may be slow and is not a measured
  speedup shape.
- If smoke still fails readiness, use the uploaded `openai-smoke-work` logs to
  distinguish stage launch, package materialization, frontend bind, and request
  timeout failures.
- If smoke passes, the next quality gate must broaden held-out prompts before
  treating the sidecar as useful; the current `2 / 8` top-1 and `5 / 8` top-4
  score is only a tiny-lane signal.
- Because the run uses an uploaded patch artifact rather than a pushed branch,
  keep the run id, upload commit, and patch SHA with every report.

## Pass/Fail Gate

Pass for the capped lane means:

- the planner resolves the exact package, layer count, activation width, and S8
  tap topology;
- the chosen HF flavor and timeout stay under `$50`;
- train and held-out prompt-token shards are disjoint;
- package-backed staged load/capture starts on the S8 split;
- native teacher logits are captured over the SPD draft vocab;
- sidecar training starts from captured rows/logits without full HF base-model
  loading;
- exported artifacts pass serving-fixture validation if the run reaches export;
- Rust fixture parity is not expected in this lane until native parity fixture
  export is added;
- package-backed rolling smoke matches baseline content and has zero tap
  failures if the run reaches smoke;
- the report records accepted/proposed, saved/unsaved candidate-token round
  trips, sidecar timing, and latency simulation.

Do not call this a speedup or final sidecar-quality proof unless broad held-out
package-backed serving shows saved candidate-token round trips exceeding
unsaved round trips with margin.
