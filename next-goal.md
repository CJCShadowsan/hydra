# Next Goal: Resubmit Qwen3-Coder-480B S8 SPD HF Run

This file is disposable. Durable evidence belongs in `evals/spd/README.md` and
`docs/skippy/speculative_decoding.md`.

## One-Line Goal

Resubmit the capped Hugging Face native-package run for a Qwen3-Coder-480B S8
SPD sidecar after fixing the capture boolean argument, using the exact MeshLLM
Skippy layer package for teacher capture and training only the SPD predictor
from captured taps/logits.

## Immediate Target

- Runner: Hugging Face Jobs in the `meshllm` org.
- Spend cap for the first submitted job: `$50` maximum.
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

Submitted first target: `rtx-pro-6000x4`, timeout `4h30m`, max cost `$50`.

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
4. Planned inside the capped job: run package-backed local-on-HF smoke.
   - Use the same Qwen480 package directory and S8 split.
   - Check content match, tap failures, accepted/proposed counts, and
     saved/unsaved candidate-token round trips.
5. Planned inside the capped job: emit pipeline economics.
   - Run latency simulation for S8 clumped to 4 physical buckets and hop
     assumptions `0.2,1,5,10ms`.
   - Treat this as HF-side predictor/economics evidence, not LAN wall-clock
     proof.

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

The timeout is the spending backstop. At the current checked rate for
`rtx-pro-6000x4`, `4.5h` plans at about `$49.50`; the job should finish, fail,
or be killed by HF at timeout.

Startup attempts before the current live job:

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

Latest status check on 2026-06-19: job
`meshllm/6a3535603093dba73ce2a264` is `ERROR` with exit code `1`. It reached
real package download and prompt construction but did not start capture rows,
training, scoring, export, or smoke because of the fixed capture boolean
argument issue.

Prior-job inspection commands:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --from huggingface_hub hf jobs inspect meshllm/6a3535603093dba73ce2a264
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --from huggingface_hub hf jobs logs meshllm/6a3535603093dba73ce2a264
```

## Remaining Risks During Run

- The first HF run may expose a reference-code compatibility issue between
  `SpeculationHeadTransformer` and the Qwen3-Coder-480B MoE config. The
  current scripts deliberately load AutoConfig only, so this should fail early
  without loading the full model if unsupported.
- `native-package-fresh` exports a serving fixture, not a true Python/reference
  parity fixture. Do not claim Rust/Python fixture parity for this lane yet.
- Setup/build time runs inside the `4.5h` cap. If the job expires before useful
  capture, the next lane should reduce the first-run scope or use prebuilt
  runtime artifacts before increasing spend.
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
