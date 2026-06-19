# Skippy SPD Proof Notes

This directory is the public, reproducible handoff for the Skippy Speculative
Pipeline Decoding (SPD) proof.

SPD is treated here as a separate trained sidecar head. It proposes draft tokens
from selected target-model hidden states; the target model still verifies every
accepted token. The work in this directory proves the training/evaluation path
and records the artifact contract Skippy needs before serving the head from
Rust.

## Current Immediate Goal: Qwen3-Coder-480B S8 Native Package Qualification

The immediate target is now Qwen3-Coder-480B S8 on Hugging Face. The package is
`meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`, with package model
id `unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF:UD-Q4_K_XL`. Its
`model-package.json` reports `62` layers, activation width `6144`, and
`256.98 GiB` of package artifacts.

The first logical SPD topology is S8:
`stage_layer_boundaries=8,16,24,32,40,48,55,62`, with required taps
`[0,8,16,24,32,40,48,55,62]`. Approximate layer-package bytes by logical stage
are `33.3,32.4,32.7,32.4,32.7,33.6,28.4,30.2 GiB`.

SPD should reuse the ordinary Skippy layer package: Skippy owns physical layer
ranges and package materialization, while the sidecar owns logical tap
requirements and proposal weights. The coordinator runs the SPD predictor by
default; workers need only the manifest-derived tap-return allowlist. If Mesh
colocates multiple logical SPD stages on one physical node, that node must
still return every internal logical-boundary tap required by the manifest.

The first HF run must be native-package-first: build disjoint train/held-out
prompt shards, download the Qwen3-Coder-480B layer package, capture raw
tap-concat rows and native quant verifier logits/top-k over the SPD draft
vocab, train the sidecar from those rows, score held-out top-1/top-4, export
the serving bundle, run package-backed rolling `spd-openai-smoke`, and emit
latency simulation JSON from `evals/spd/simulate_latency.py`.

Current implementation checkpoint: `skippy-bench spd-product-corpus-capture`
provides topology-only native capture without an existing SPD manifest or
fixture. `train_product_activation_head_only.py` and
`score_product_activation_head_only.py` train/score the sidecar from captured
raw rows and native teacher logits while loading AutoConfig only, not the full
base model. `export_product_serving_fixture.py` emits the serving-only fixture
needed by the Rust request path for row metadata and final norm. The
`native-package-fresh` planner path now generates only these native-package
commands; it does not call `hf_train_eval_qwen06.py`, `spd-live-tap-parity`, or
the old full-base product trainer/scorer.

Important remaining gap: the native fresh lane exports a serving fixture, not a
true Python/reference parity fixture. Do not claim Rust/Python fixture parity
for Qwen480 S8 until a native parity fixture exporter exists. The current
capped runs have not exposed a Qwen3-Coder-480B MoE/AutoConfig compatibility
blocker; the immediate unproven gate is package-backed smoke placement and
request-path behavior for the uploaded bundle.

HF Jobs rates checked on 2026-06-19 put `rtx-pro-6000x4` at about `$11/hr`,
`h200x2` at about `$10/hr`, `h200x4` at about `$20/hr`, and
`rtx-pro-6000x8` at about `$22/hr`. The first capped lane used
`rtx-pro-6000x4` with a `4.5h` timeout for a planned maximum of `$49.50`. The
latest retry used the same flavor with a `3.5h` timeout, planned around
`$38.50`, so aggregate spend remains under the original `$50` intent after
earlier failures. These lanes are enough for serious package/capture/smoke
attempts if setup and download are fast enough; they are not guaranteed to
finish a final-quality sidecar. `h200x2` can run for `5h` under `$50` but is
tighter on VRAM for a `256.98 GiB` package; `h200x4` is safer on memory but only
buys about `2.5h` under `$50`.

Dry-run checkpoint on 2026-06-19:
`plan_hf_spd_qualification.py --qualification-mode native-package-fresh` with
`--vocab-size 151936`, `rtx-pro-6000x4`, and `4.5h` resolves the package as
`62` layers / width `6144`, plans max cost `$49.49991`, and emits no
`AutoModelForCausalLM`, no `hf_train_eval_qwen06.py`, no `spd-live-tap-parity`,
and no warm-start dependency in the generated native command graph. The setup
commands install Rust/`just`/build prerequisites, detect the CUDA architecture,
build the CUDA stage ABI with `just build-runtime`, and build
`target/release/skippy-bench` plus `target/release/skippy-server`. After job
`meshllm/6a3535603093dba73ce2a264`, the dry run now also verifies that
`spd-product-corpus-capture` emits `--product-native-teacher-logits true` and
that the HF dependency setup does not ask pip to upgrade/install `torch` over
the PyTorch CUDA base image. Dispatch can
carry local unpushed changes by uploading a patch artifact to the HF output repo
and setting `MESH_LLM_PATCH_PATH` before executing the generated plan.
`evals/spd/bootstrap_qwen480_s8_native_job.sh` is the intended HF job
entrypoint for this capped lane.

First serious submitted checkpoint on 2026-06-19: HF Job
`meshllm/6a3535603093dba73ce2a264` using `rtx-pro-6000x4` and timeout `4.5h`;
the job URL is `https://huggingface.co/jobs/meshllm/6a3535603093dba73ce2a264`.
The spending backstop is the HF timeout, still planned at about `$49.50`.
Because the local branch is not pushed from this machine, the job bootstraps
from an uploaded patch artifact in
`meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8`.

The current submitted artifact is `job-inputs/20260619T122507Z-b843a851/`,
upload commit `80014e284aa1e727a305c3ff5c44fb2ca82659d6`, local artifacts
`/tmp/spd-qwen480-native-job-20260619T122507Z-b843a851`, patch SHA256
`7ec74581ee16e30ce4d56b99a5b0092eb8fc513b92c36acae8fbf8a93d952436`,
bootstrap SHA256
`378a4bc91ff2c4aadeffa2a501180aafba44bedc2df377598bc0a3f3ce8ab6d6`, and
dry-run plan SHA256
`542b9b61a118ee5a4a6a68d103ab1614bc020371129e233fe1d8e8bc93c4e7c6`.
Earlier startup attempts failed before model work: two `bash` invocations
missed the HF CLI `--` option terminator and treated the script as a filename,
then one corrected invocation exposed an unexported `BOOTSTRAP_DIR` variable in
the bootstrap script, then one run reached plan construction and failed because
`/workspace/spd-qualification` was missing before writing the generated plan.
CPU canary `meshllm/6a3531e9953ed90bfb9446e4` verified the corrected CLI form.
Job `meshllm/6a353427953ed90bfb944722` reached generated setup: it downloaded
the bootstrap, applied the patch, generated the native-package plan, installed
build prerequisites, installed Rust, installed `just`, cloned the patched repo
for execution, and then failed at the generated
`just build-runtime backend=cuda cuda_arch="$CUDA_ARCH"` command. The repo's
`build-runtime` recipe takes positional parameters, so the planner now emits
`just build-runtime cuda "$CUDA_ARCH"`. Job
`meshllm/6a3535603093dba73ce2a264` passed that failure, built the CUDA
llama.cpp stage runtime, built the Rust release binaries, downloaded the full
Qwen480 package snapshot (`69` files, about `276G`), and generated the
UltraChat prompt-token shards (`512` train prompts, `64` held-out prompts,
train mean `101.3` tokens, held-out mean `103.8` tokens). It then failed at the
first `spd-product-corpus-capture` invocation because
`--product-native-teacher-logits` was emitted without its required boolean
value. The planner now emits `--product-native-teacher-logits true`, so the
next resubmission should start actual capture rows unless a runtime/model issue
appears.

Resubmission checkpoint on 2026-06-19: HF Job
`meshllm/6a353b9d3093dba73ce2a2bf` used the same `rtx-pro-6000x4` / `4.5h`
cap. It used fixed artifact
`job-inputs/20260619T125208Z-22663dd2/` from upload commit
`da3c7956783e86c3e50368ddbd32c00286f263df`, with patch SHA256
`450002e81f41b6adaf72c997ecad28700e29f2faf191c7c93d1aceb06e76757f` and
dry-run plan SHA256
`dcce197cb092662ae7048df92f65356833fcb6d60b3c4630613942deb739f78a`. HF
registered the pinned `PATCH_REVISION` and fixed artifact paths. The job ran
for `1249` seconds, costing about `$3.82` at the planned `$11/hr` rate. It
passed CUDA/Rust release builds, package download, prompt-token generation, and
the fixed `--product-native-teacher-logits true` command shape, then failed at
actual `capture[0]` startup: topology-only package stage `55..62` could not
allocate a `30905.58 MiB` CUDA3 model buffer. The two serious Qwen480 HF jobs
cost about `$7.45` combined; including the shorter startup failures keeps total
GPU spend for this lane under about `$8`.

Local fix checkpoint after that OOM: `spd-product-corpus-capture` now has a
`--stream-live-tap-stages` mode. In that mode the capture path keeps the
unchanged full native Q4 verifier session for greedy target tokens and draft
vocab teacher logits, while live-tap replay opens one logical tap-stage model,
captures its boundary frame, drops it, and then opens the next stage. The
Qwen480 native dry run now emits both
`--stage-backend-devices CUDA0,CUDA0,CUDA1,CUDA1,CUDA2,CUDA2,CUDA3,CUDA3` and
`--stream-live-tap-stages`, plus the fixed
`--product-native-teacher-logits true`. This should reduce peak VRAM without
changing teacher semantics. The risk to measure on the next HF run is repeated
stage-open churn inside the current timeout cap.

Streamed-capture submission on 2026-06-19: HF Job
`meshllm/6a354843953ed90bfb944848` used the same `rtx-pro-6000x4` /
`4.5h` timeout cap and input artifact
`job-inputs/20260619T134535Z-595b67cb/` from upload commit
`9198f2468ae69dbb13c0d0a16f7b99c0e3e7dd5d`. The patch SHA256 is
`717b871d6668ad895869013f8a20168160bc46557b927ade1473258dea369c61`; the
bootstrap SHA256 is
`378a4bc91ff2c4aadeffa2a501180aafba44bedc2df377598bc0a3f3ce8ab6d6`; the
dry-run plan SHA256 is
`e33d546eb7b3b3d639441fca7331f85fc0addf85d00623bb3cb7fb7b5966d9de`.
It failed after `209` seconds, before model work, because
`crates/skippy-server/src/frontend/spd.rs` still initialized
`SpdLiveTapRunnerConfig` without the new `stage_backend_devices` and
`stream_stages` fields. Commit `d19da20d` fixed that server initializer, and
commit `f23e28ba` made the HF job timeout configurable for a shorter retry.

Current streamed-capture retry on 2026-06-20 local time: HF Job
`meshllm/6a354a2f953ed90bfb94486f` ran on `rtx-pro-6000x4` with a
`3.5h` timeout cap. It uses input artifact
`job-inputs/20260619T135416Z-f23e28ba/` from upload commit
`fc2f95dd543955f1e821c7036bebd0e48501974f`. The patch SHA256 is
`58cfa43179ce251784014955f43f02092ae0936cb7df75a1a4e545f9f2c8b6bc`; the
bootstrap SHA256 is
`30d27fa808c08df2f3ca1613381de1ca0a828694e66448f3bc03e55b2610cb05`; the
dry-run plan SHA256 is
`61ffa3a560948536e9fc4df7e7dd4c178f36ab4309dbe340e37c63de02d5a9d5`. It
failed after `1226` seconds. It cleared inline bootstrap download/startup, CUDA
ABI build, Rust release builds, Python dependency setup, full 69-file Qwen480
layer-package download (`276G / 276G`), and prompt-token shard building, then
reached `capture[0]`. It failed opening streamed tap stage `0..8`: CUDA0 could
not allocate a `34051.88 MiB` buffer while the full verifier remained resident.
At about `$11/hr`, this run added about `$3.75`, bringing completed GPU spend
for the Qwen480 lane to roughly `$12` to `$13`.

Local fix after that capture-residency failure: `spd-product-corpus-capture`
now phase-separates topology-only capture. Phase 1 runs the full target verifier
to record target tokens and native draft-vocab logits for every prompt/step,
then drops the verifier model. Phase 2 opens the streamed tap runner and replays
the recorded contexts to write `rows.f32`, `raw_rows.f32`, and `rows.jsonl` in
the original prompt/step order. This keeps native teacher semantics unchanged
while avoiding full-verifier plus tap-stage GPU residency overlap. A real cached
package smoke passed with
`meshllm_Qwen3-0_6B-Q4_K_M-layers-test-main`, `--splits 14 --layer-end 28`,
`--stream-live-tap-stages`, one prompt, one verify step, native teacher logits,
and matching product row byte counts; report:
`/tmp/spd-two-phase-smoke-report.json`.

Two-phase retry on 2026-06-20 local time: HF Job
`meshllm/6a35536b3093dba73ce2a377` ran on `rtx-pro-6000x4` with a `3.5h`
timeout cap. It used input artifact
`job-inputs/20260619T143116Z-3d1442f8/` from upload commit
`abaefe222379e5bd6f949ebec7ca37de79faf715`. The patch SHA256 is
`9f623c5d3f6d5f9aa34b10e72b9849a435794634faecc497d363c3e05bd0afe1`; the
bootstrap SHA256 is
`30d27fa808c08df2f3ca1613381de1ca0a828694e66448f3bc03e55b2610cb05`; the
dry-run plan SHA256 is
`61ffa3a560948536e9fc4df7e7dd4c178f36ab4309dbe340e37c63de02d5a9d5`.

Observed gate update: this run passed the previous OOM point. It completed the
release `skippy-bench`/`skippy-server` builds, Python dependency setup, the
full Qwen480 package snapshot download (`276G / 276G`, `69` files), and prompt
dataset processing, then entered native capture. Logs show streamed stage
`0..8` allocating `CUDA0 model buffer size = 34051.88 MiB`; the prior retry
failed at that exact allocation. The job was manually canceled after this gate
to avoid burning the rest of the cap on an implementation-shaped bottleneck:
`--stream-live-tap-stages` reopens all eight tap stages for every prompt/step,
so the full `512` train / `64` held-out / `4` verify-step lane was not going
to reach training or smoke in useful time. Treat this as allocator/startup
evidence only, not a completed capture/train/smoke result.

Next retry plan: keep the two-phase verifier drop but use resident tap stages
instead of streamed tap stages. The earlier resident-stage OOM happened while
the full verifier was still loaded; after phase 1 exits, the resident S8 tap
stages should fit on `rtx-pro-6000x4` with the two-stages-per-GPU device map.
`plan_hf_spd_qualification.py` now defaults to resident tap stages and emits
`--stream-live-tap-stages` only when explicitly requested. The bootstrap also
accepts `TRAIN_PROMPTS`, `HELDOUT_PROMPTS`, `VERIFY_STEPS`, and
`STREAM_LIVE_TAP_STAGES` overrides. The first artifact-producing retry profile
is `TRAIN_PROMPTS=32`, `HELDOUT_PROMPTS=8`, `VERIFY_STEPS=1`,
`STREAM_LIVE_TAP_STAGES=false`, and `JOB_TIMEOUT=2h`; its dry run plans about
`$22` on `rtx-pro-6000x4` and still contains no old reference-train path.

Resident-small retry submitted on 2026-06-20 local time: HF Job
`meshllm/6a3563743093dba73ce2a4ab`, URL
`https://huggingface.co/jobs/meshllm/6a3563743093dba73ce2a4ab`, using input
artifact `job-inputs/20260619T154157Z-52a4ffdf/` from upload commit
`509ed659ccfb237892e9329887f1a6261f352bf1`. Patch SHA256 is
`57bb1e3180923531641384af307bfb41e6ee05fb0c75795916e93cd2ddc25645`;
bootstrap SHA256 is
`c8e2efa7ec104ec6c38d6b8584cd8851ea84692b73ff3df40dcb5fe08e79a022`; dry-run
plan SHA256 is
`3936027d9c94951a8667a9d7c58e186064c58df49e1d4ca51a3563b3bbd6e5e4`.
The first gate is resident S8 tap-stage allocation after verifier drop. The
second gate is train and held-out capture summaries, followed by
conversion/training/export/package smoke if the smaller lane reaches them.

Resident-small result: the job failed after `1424s` running, but after the
native-package mechanics gates cleared. It completed release build, downloaded
the full `69`-file / `276G` package snapshot, built prompt shards, loaded the
full Qwen480 verifier across four RTX PRO 6000 GPUs, split verifier
target/logit capture from tap replay, opened resident S8 tap stages, converted
train and held-out corpora, trained the head-only predictor, scored held-out,
and exported a BF16 serving head. Train conversion had `32` samples with
`31 / 32` labels in draft scope; held-out had `8 / 8`. The head-only train
summary reported `base_model_load=skipped`, final train hard-label accuracy
`1.0`, and the held-out score was `2 / 8` top-1 and `5 / 8` top-4. The serving
export wrote `spd-head.safetensors` with SHA
`96a520d81a13bb586897e607767081b98a0d587119a63ae3e4eb7ce15732d15d` and
`8,723,214,136` bytes.

The failure was not allocator, MoE startup, capture, or training. The generated
`native-package-fresh` parity-skip command was
`echo native-package-fresh exports a serving fixture only; Rust fixture parity
is skipped...`, so Bash treated `Rust` as a command and exited `127`.
`plan_hf_spd_qualification.py` now emits the skip as a single `printf` command.
Local validation passes: Python compile for the planner/runner, the same
resident 32/8/1 dry run, and
`run_hf_spd_qualification_plan.py --groups rust_fixture_parity`. The next retry
should reuse the same resident-small profile; the first new gate is
package-backed rolling smoke plus upload.

Fixed resident-small retry submitted on 2026-06-20 local time: HF Job
`meshllm/6a356b6d3093dba73ce2a5da`, URL
`https://huggingface.co/jobs/meshllm/6a356b6d3093dba73ce2a5da`, using input
artifact `job-inputs/20260619T161546Z-a6dae908/` from upload commit
`f57a5053d8c1ff20ca74798dd076fcb317a6038a`. Patch SHA256 is
`b432270244c258d9621f05afe1a8de455ba3d540b445d6a158e56f33f4d3bc25`;
bootstrap SHA256 is
`c8e2efa7ec104ec6c38d6b8584cd8851ea84692b73ff3df40dcb5fe08e79a022`; dry-run
plan SHA256 is
`2493c61db82bc1e0d77b684dc606256ead3bbc54e3679a6e66c5345e01bf763f`. The job is
labeled `spd-qwen480-resident-small-fixed` and uses the same
`TRAIN_PROMPTS=32`, `HELDOUT_PROMPTS=8`, `VERIFY_STEPS=1`,
`STREAM_LIVE_TAP_STAGES=false`, and `JOB_TIMEOUT=2h` profile. At submission it
was scheduling. The first new gate is passing the now-fixed
`rust_fixture_parity` skip and reaching package-backed rolling smoke plus
upload.

Observed fixed resident-small result: the job is `ERROR` after `1383s` running,
but it passed the parity-skip gate and repeated the important native-package
mechanics. It completed build, full Qwen480 package download, verifier load,
two-phase verifier target/logit capture, resident tap replay, native
train/held-out conversion, head-only train/score, and BF16 serving export.
Held-out score remained tiny-lane evidence only: `2 / 8` top-1 and `5 / 8`
top-4. The exported `spd-head.safetensors` was `8,723,214,136` bytes with SHA
`3fcdb93eeea5d23c4ae3df3dc39e10e70f59564a2ab20820f09aa0a7a5fe3f9d`. The new
failure was package smoke readiness: the baseline OpenAI frontend did not
become ready and `/v1/models` returned connection refused. Since upload still
ran after smoke, that run did not preserve the exported sidecar in the output
repo.

Local fix checkpoint after that smoke failure: `spd-openai-smoke` now stops
stages and prints bounded stage-log tails when OpenAI readiness fails, so the
next HF failure should show whether stage 0 was still loading, crashed, or
failed to start the embedded frontend. The `native-package-fresh` planner now
adds an `upload_pre_smoke` group before package smoke, runs Qwen480 smoke with
`--startup-timeout-secs 600 --request-timeout-secs 600`, and puts smoke work
under `/workspace/spd-qualification/artifact/openai-smoke-work` so successful
smokes upload their stage logs. The current 32/8/1 resident dry run is still
native-package-first: `rtx-pro-6000x4`, timeout `7200s`, max cost
`$21.99996`, no `AutoModelForCausalLM`, no `hf_train_eval_qwen06.py`, no
`spd-live-tap-parity`, no `--stream-live-tap-stages`, and command groups ordered
with `upload_pre_smoke` before `package_smoke`.

Observable resident-small retry submitted on 2026-06-20 local time: HF Job
`meshllm/6a3575be3093dba73ce2a692`, URL
`https://huggingface.co/jobs/meshllm/6a3575be3093dba73ce2a692`, using input
artifact `job-inputs/20260619T165954Z-76662252/` from upload commit
`83a6631a29fcb534057d34353d9e78a2d248cbf3`. Patch SHA256 is
`dc33c52493b3c6bc2bcf478052e57b4700ecaaf77fe3b323d7bfcc612bf37c10`;
bootstrap SHA256 is
`c8e2efa7ec104ec6c38d6b8584cd8851ea84692b73ff3df40dcb5fe08e79a022`; dry-run
plan SHA256 is
`d04bb5d3bbe785e7ca572dabee59c9eed1cc817e7bdeb9084fc2c199962217bf`. The job is
labeled `spd-qwen480-resident-small-observable` and uses the same 32/8/1
resident profile on `rtx-pro-6000x4` with `JOB_TIMEOUT=2h`. Its first new gates
are pre-smoke artifact upload, then package smoke with longer readiness/request
timeouts and stage-log tails on failure.

Observable resident-small result: the job is `ERROR` after `1898s`, but it
completed the artifact-producing path and preserved the sidecar before smoke.
It repeated the native-package mechanics gates: release build, full `69`-file /
`276G` package download, full Qwen480 verifier load/offload, two-phase native
target/logit capture, resident tap replay, native train/held-out conversion,
head-only train/score with `base_model_load=skipped`, serving export, and
`upload_pre_smoke`. Train conversion had `31 / 32` labels in draft scope;
held-out had `8 / 8`. Held-out score was `2 / 8` top-1 and `5 / 8` top-4,
which is tiny-lane quality evidence only. The uploaded serving head is
`8,723,214,136` bytes with SHA
`f77dbfb1f83a1c3a79446b983c7de3e77f63c22f4bacbd8ae0d92efbeef3fc75`.
Artifacts are under
`meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/runs/native-package-fresh`.

Package smoke failed for a concrete placement reason, not because capture,
training, export, or upload failed. Stage logs showed stage `1` listening on
CUDA0, then stage `0` tried to allocate a `34051.88 MiB` CUDA0 buffer and
failed with `cudaMalloc failed: out of memory`. Stages `1..7` had already
started cleanly. The immediate fix is to run package smoke with an explicit
per-stage backend map so two large logical stages are not resident on the same
GPU during smoke. `spd-openai-smoke` now supports
`--stage-backend-devices`, `plan_hf_spd_qualification.py` accepts
`--smoke-stage-backend-devices`, and
`bootstrap_qwen480_s8_native_job.sh` passes `SMOKE_STAGE_BACKEND_DEVICES`
through to the planner for any full retry.

Next spend-bearing step: use
`evals/spd/bootstrap_qwen480_s8_smoke_existing_job.sh` to hydrate the uploaded
`runs/native-package-fresh` artifact, regenerate the deterministic held-out
prompt shard, download the same Qwen480 package, and run only package smoke,
latency simulation, and upload. Default map:
`CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`; default timeout `1.5h`, planned max
about `$16.50` on `rtx-pro-6000x4`. This is a mechanics/economics smoke, not a
speed claim, and CPU-mapped stages may be slow. A full recapture/retrain retry
is now lower priority unless the uploaded artifact proves unusable.

Smoke-existing retry submitted on 2026-06-20 local time: HF Job
`meshllm/6a3581b9953ed90bfb944dd3`, URL
`https://huggingface.co/jobs/meshllm/6a3581b9953ed90bfb944dd3`, label
`spd-qwen480-smoke-existing`. It uses `rtx-pro-6000x4`, timeout `1.5h`,
`PATCH_REVISION=1dce6e58644cf957e517f430ac653df6257fc455`,
`PATCH_PATH_IN_REPO=job-inputs/20260619T175052Z-588fe084/mesh-llm.patch`,
`ARTIFACT_RUN_PATH=runs/native-package-fresh`, and
`SMOKE_STAGE_BACKEND_DEVICES=CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`. The
uploaded job input patch SHA256 is
`e9947039a0f2cbe0ba127793dbd0fa054bd6f1cbb23b996059ed501e422d89f1`; the
bootstrap SHA256 is
`969e6005f7e9e108c2bca0ad0b50c430cc0cf7b9b78db577beee9859209c07d8`; the
dry-run plan SHA256 is
`7fba2ddb364e6ad8eab8bbd4f78ac01b4e359614cf124643a4771da1325a5012`. At first
poll, the job was still scheduling; first gates are bootstrap download, patch
apply, selected-group setup/package download, artifact hydration, and package
smoke with the explicit map.

That retry failed after `1169s` running, but not on model launch or sidecar
quality. It completed release build, full `276G` package download, deterministic
prompt rebuild, and hydrated the uploaded `runs/native-package-fresh` artifact.
The failure was a split-script cwd bug: `package_smoke` ran as a second
generated script from the bootstrap checkout, where
`target/release/skippy-bench` did not exist. Commit `bf682379` fixes the
smoke-existing bootstrap by re-entering `$WORK_DIR/mesh-llm` before
`package_smoke`, `latency_simulation`, and `upload`, and by failing fast if the
release binaries or `physical-stage-ms.txt` are missing.

Fixed smoke-existing retry submitted on 2026-06-20 local time: HF Job
`meshllm/6a35894f953ed90bfb944e49`, URL
`https://huggingface.co/jobs/meshllm/6a35894f953ed90bfb944e49`, label
`spd-qwen480-smoke-existing`. It uses the same `rtx-pro-6000x4` / `1.5h` cap
and CPU/GPU smoke map, with
`PATCH_REVISION=ea52905865b07ad17a5fdd7519d27a07ad4f689c`,
`PATCH_PATH_IN_REPO=job-inputs/20260619T182322Z-bf682379/mesh-llm.patch`, and
`ARTIFACT_RUN_PATH=runs/native-package-fresh`. The uploaded job input patch
SHA256 is
`af3c6353916a96b9a6568004c11ad32b0b788d06cf31182458b8a3cdee29799d`; the
bootstrap SHA256 is
`173a370c1f7b9f0b2bac501d866fd835e26c397be39024817a717d7ba3c325e7`; the
dry-run plan SHA256 remains
`7fba2ddb364e6ad8eab8bbd4f78ac01b4e359614cf124643a4771da1325a5012`. First poll
showed bootstrap download complete and generated setup running.

That fixed retry ended `ERROR` after `1621s` running, but it proved the cwd fix
and reached package-backed baseline/SPD smoke. It completed release build, full
package download, deterministic prompt rebuild, artifact hydration, and stage
launch with the CPU/GPU smoke map. The smoke cases recorded downstream taps
for hf indices `16,24,32,40,48,55,62` and local stage-0 hf `8` tap records
with `0` tap return failures, `0` tap record failures, and `0` ignored taps.
However every SPD case reported `spd_proposal_total_proposed=0` after
`38` or `44` proposal attempts because the proposal cache was missing hf `8`
rows for prompt-window positions such as `40..45` and `58..63`. Root cause:
the initial SPD `reset_to_context(prompt)` ran after prefill and retained zero
tap rows when the source context was still empty; downstream stages could
recover prompt rows through first-decode context sideband replay, but stage 0
does not re-run the whole prompt during decode. The local patch now preserves
prefill tap rows on that initial source reset. The same job then failed in
latency simulation because the report had eight stage processes while
`physical-stage-ms.txt` modeled four clumped physical buckets; `simulate_latency.py`
now accepts that clumped what-if shape and records both modeled and report
physical stage counts.

Next retry remains smoke-existing only: upload the current patch, hydrate
`runs/native-package-fresh`, reuse the same prompt shard and Qwen480 package,
and rerun package smoke plus latency simulation. Do not recapture or retrain
unless the uploaded artifact itself becomes unusable.

Current patch bundle for that retry is
`job-inputs/20260619T190753Z-6abc8370/`, upload commit
`43940c19fefce860f58c37ebe0517a13d32f8419`. Patch SHA256:
`4bca067da32ae42067845663ac345e515feeae8abd57aa0ad883de6b9458f15a`;
bootstrap SHA256:
`173a370c1f7b9f0b2bac501d866fd835e26c397be39024817a717d7ba3c325e7`;
dry-run plan SHA256:
`7fba2ddb364e6ad8eab8bbd4f78ac01b4e359614cf124643a4771da1325a5012`.
An initial submission `meshllm/6a3593b5953ed90bfb944ef8` failed in `7s`
with exit `126` because the HF CLI command omitted the `--` terminator and
parsed `-lc` as a job flag. Corrected retry:
`meshllm/6a3593cf3093dba73ce2a78f`
(`https://huggingface.co/jobs/meshllm/6a3593cf3093dba73ce2a78f`), using the
same `rtx-pro-6000x4` / `1.5h` cap, patch revision, artifact path, and
CPU/GPU smoke map. First inspect showed the corrected command as
`['bash', '-lc', ...]` and status `SCHEDULING`.

If this Qwen480 lane clears the sidecar quality and package-backed smoke gates,
the next HF validation spike should be a single-job meshlet: one HF Job starts
the coordinator, stage servers, SPD sidecar, and OpenAI frontend as separate
local processes, with optional artificial per-stage latency. That would test
package download, stage placement, tap returns, SPD proposal/verification,
rolling executor cleanup, and pipeline economics repeatably. A multi-HF-job
node spike is lower priority until the transport story is explicit, because HF
job port exposure is not the same as a normal low-latency Mesh LAN.

Dispatch gate for that meshlet: do not spend on it until the current Qwen480
lane has produced held-out native teacher summaries, completed training/scoring
or failed with an actionable sidecar-quality result, exported the serving
bundle, and run package-backed rolling `spd-openai-smoke` with matched baseline
content, zero tap failures, and useful accepted/proposed plus saved/unsaved
candidate-token round-trip counts. The meshlet is a validation layer for a
candidate sidecar, not a replacement for producing one.

Pass criteria: train/held-out prompt-token shards have zero overlap, native
teacher argmax matches the quant verifier target on in-scope rows, serving
artifacts export if training reaches export, package-backed rolling smoke
matches baseline content with zero tap failures, broad held-out saved
candidate-token round trips exceed unsaved round trips with margin, and latency
simulation stays positive under realistic stage costs, link latency, and
sidecar latency.

## Prior Handoff: Qwen3-8B `23,36`

`Qwen/Qwen3-8B` against the immutable
`meshllm/Qwen3-8B-Q4_K_M-layers` package with logical SPD boundaries `23,36`
is now harness evidence, not the immediate target. The topology dry run is
clean: `physical_split_boundaries=[23]`,
`layer_end=36`, tap rows `0,23,36;0,23`, and product tap-return allowlist
`[23,36]`. Keep this as a hard constraint for any future Qwen3-8B training
artifact or speed claim.

Current evidence still points to train/serve activation-distribution mismatch,
but the latest product sidecar is the first checkpoint that clears the local
held-out paper-style break-even gate on this exact product topology:

- the 512-row BF16 reference-trained `Qwen3-8B` S2 `23,36` head is
  topology-correct and parity-clean, but product Q4 package serving still
  accepts poorly;
- exact-prompt diagnostics show target-token parity is mostly aligned while
  proposal-token parity is poor;
- a 72-row product-tap fine-tune recovers high product-path acceptance on its
  training prompts, proving the mechanics can learn the product activation
  distribution but not proving generalization;
- a broader product-distribution bridge now trains on `48` prompts and evaluates
  on `24` held-out prompts across MT-Bench, GSM8K, and HumanEval.

The current prompt-token seed set is
`/tmp/spd-qwen3-8b-product-prompts-paper3-train16-heldout8`, rendered with the
Qwen chat template and `enable_thinking=false`. The exact product split produced
`384` train rows and `192` held-out rows from the Q4 layer package. HF teacher
augmentation wrote draft-width BF16 logits over the top-32k draft vocabulary
(`378 / 384` train labels and `190 / 192` held-out labels inside scope). The HF
teacher is not native Q4_K_M verifier KL, but it is aligned enough for this
bridge: teacher top-1 matches the product Q4 target on `363 / 384` train rows
and `180 / 192` held-out rows; teacher top-4 contains the Q4 target on
`378 / 384` train rows and `190 / 192` held-out rows.

The 5-epoch BF16 MPS fine-tune from the LR `1e-4` checkpoint reached train
argmax accuracy `0.875`. The serving export is
`/tmp/spd-qwen3-8b-product-finetune-paper3-train16-e5-lr2e5/`, with BF16
`spd-head.safetensors` SHA
`43501aa95fd191ad087af1396f6b1909cb2bc72e83391085d23997978427531b`.
`skippy-bench spd-fixture-parity` exits successfully for that export.

2026-06-18 larger max120 product-corpus check: the no-spend path was expanded
to `/tmp/spd-qwen3-8b-product-prompts-paper3-train32-heldout16-max120`, which
keeps prompts short enough for the current live-tap `n_batch=128` limit. It
captured `712` train rows from `89` prompts and `256` held-out rows from `32`
disjoint prompts. HF teacher alignment on the train rows remained strong
(`668 / 712` teacher top-1 matches native Q4 target; `702 / 712` top-4 contains
target). The 5-epoch BF16 HF-KL fine-tune exported and passed Rust fixture
parity at
`/tmp/spd-qwen3-8b-product-finetune-paper3-train32-max120-e5-lr2e5/`, but
held-out live-tap acceptance was only `91 / 256` with exact greedy output.
Held-out attribution shows the problem is sidecar generalization, not
teacher/Q4 disagreement: the HF teacher matched the native Q4 target on
`245 / 256`, while the sidecar matched the teacher on only `95 / 256`.

A native-hard-label experiment on the same `712` rows also exported and passed
Rust fixture parity at
`/tmp/spd-qwen3-8b-product-finetune-paper3-train32-max120-hard-e5-lr2e5/`.
It improved the same held-out gate only to `95 / 256`, again with exact greedy
output and terminal-final-normed parity clean. This is not enough for a
request-path speed smoke. The grounded next no-spend step is a larger disjoint
short-prompt product corpus (for example the generated
`/tmp/spd-qwen3-8b-product-prompts-paper3-train56-heldout8-max120`, `137`
train prompts / `16` held-out prompts) plus less-overfit training and the same
held-out attribution. The native Q4_K_M verifier-logit tooling gap is now
closed enough for the next training gate.

2026-06-19 native-Q4 verifier-logit checkpoint: the llama.cpp ABI now exposes
`skippy_session_copy_current_logits`, Rust can copy current verifier logits over
the SPD draft vocabulary, and `skippy-bench spd-live-tap-parity
--product-native-teacher-logits` writes native product verifier logits alongside
product tap rows. `evals/spd/prepare_native_product_teacher_logits.py` converts
those rows into the teacher safetensors consumed by
`train_product_activation_head.py`. A two-prompt reuse smoke at
`/tmp/spd-native-teacher-reuse-smoke` passed after moving the full Q4 target
model load out of the per-prompt loop; it wrote `2` product rows and `2` native
teacher-logit rows with exact byte counts.

The first native-Q4 train/held-out adaptation gate used the 16k UltraChat
checkpoint at
`/private/tmp/skippy-spd-qwen3-8b-s2-23-16k-hf/runs/20260618-122936/train/speculation_head_final.pt`.
Held-out capture `/tmp/spd-native-teacher-smoke` wrote `16` rows, draft logit
width `32000`, exact byte counts, and `15 / 16` labels in draft scope; the
native teacher argmax matched the in-scope Q4 target on `15 / 15`. Train capture
`/tmp/spd-native-teacher-train16-v4` wrote `64` rows from the first `16` train
prompts with `4` verify steps, exact byte counts, and `64 / 64` labels in draft
scope with native teacher argmax matching the Q4 target on `64 / 64`.
Converted tensors are:

- train product corpus:
  `/tmp/spd-native-teacher-train16-v4-corpus.safetensors`
- train native teacher logits:
  `/tmp/spd-native-teacher-train16-v4-teacher.safetensors`
- held-out product corpus:
  `/tmp/spd-native-teacher-smoke-corpus.safetensors`
- held-out native teacher logits:
  `/tmp/spd-native-teacher-smoke-teacher.safetensors`

Before adaptation, the 16k head scored `7 / 64` train top-1 and `12 / 64`
train top-4 against native labels, but `0 / 15` held-out top-1 and `0 / 15`
held-out top-4 in scope. A conservative native-Q4 warm start at
`/tmp/spd-native-q4-adapt-train16-v4-e3-lr1e5-hard025/` trained for `3` epochs,
batch `8`, LR `1e-5`, weight decay `1e-2`, KL weight `1.0`, and hard-label
weight `0.25`. It overfit the tiny native train set (`49 / 64` train top-1,
`55 / 64` train top-4), while held-out improved only to `2 / 15` top-1 and
`4 / 15` top-4 in scope. Treat this as proof that native quant-specific
supervision is wired and learnable, not as enough quality to scale speed tests.

The larger native-Q4 gate reused the same target-logit path on the full
short-prompt train file. `/tmp/spd-native-teacher-train137-v4` captured `548`
rows from `137` train prompts with `4` verify steps; byte counts were exact,
`545 / 548` labels were inside the draft logit scope, and native teacher argmax
matched the in-scope Q4 target on `545 / 545`. The matching held-out capture
`/tmp/spd-native-teacher-heldout16-v4` captured `64` rows from the `16` held-out
prompts; `61 / 64` labels were in scope and native teacher argmax matched the
Q4 target on `61 / 61`. Converted tensors are:

- train product corpus:
  `/tmp/spd-native-teacher-train137-v4-corpus.safetensors`
- train native teacher logits:
  `/tmp/spd-native-teacher-train137-v4-teacher.safetensors`
- held-out product corpus:
  `/tmp/spd-native-teacher-heldout16-v4-corpus.safetensors`
- held-out native teacher logits:
  `/tmp/spd-native-teacher-heldout16-v4-teacher.safetensors`

On this larger gate, the original 16k head scored `51 / 545` train top-1 and
`98 / 545` train top-4, with held-out `5 / 61` top-1 and `14 / 61` top-4. The
native-Q4 warm start at
`/tmp/spd-native-q4-adapt-train137-v4-e3-lr1e5-hard025/` used the same recipe
as the small gate (`3` epochs, batch `8`, LR `1e-5`, weight decay `1e-2`, KL
weight `1.0`, hard-label weight `0.25`). It reached `401 / 545` train top-1 and
`496 / 545` train top-4, while held-out improved materially to `20 / 61` top-1
and `33 / 61` top-4. This confirms that quant-specific native supervision is
the right direction, but `32.8%` held-out top-1 is still below a speed-candidate
threshold. Do not export or run request-path speed with this checkpoint yet;
next quality work should broaden native-Q4 train rows and tune regularization
against the same held-out gate.

A short regularization sweep on the same native-Q4 tensors found a slightly
better local recipe. The best held-out top-1 run was
`/tmp/spd-native-q4-adapt-train137-v4-e5-lr2e6-hard01/`: `5` epochs, batch `8`,
LR `2e-6`, weight decay `1e-2`, KL weight `1.0`, hard-label weight `0.1`. It
scored `311 / 545` train top-1, `400 / 545` train top-4, `22 / 61` held-out
top-1, and `33 / 61` held-out top-4. The top-4-best sweep candidate was
`/tmp/spd-native-q4-adapt-train137-v4-e3-lr5e6-hard01/`: `344 / 545` train
top-1, `459 / 545` train top-4, `20 / 61` held-out top-1, and `34 / 61`
held-out top-4. KL-only/high-temperature and heavier weight decay did not
improve the held-out gate. The conclusion stays the same: native-Q4 adaptation
is materially better than the original 16k head, but the current data/recipe
still does not justify export or speed testing.

The next scale attempt broadened the reference-prompt pool without changing the
frozen held-out gate. `build_product_prompt_tokens.py` now accepts
`--exclude-prompt-token-file`, so
`/private/tmp/spd-qwen3-8b-product-prompts-paper3-train-all-heldout16-frozen-max512`
contains `224` train prompts under `512` tokens while explicitly excluding the
same `16` held-out prompts. Release `skippy-bench spd-live-tap-parity` captured
`/tmp/spd-native-teacher-train224-v8`: `1792` rows from `224` prompts with `8`
verify steps, exact `229376000` native-logit bytes, `1763 / 1792` labels inside
the draft scope, and native teacher argmax matching the Q4 target on
`1763 / 1763` in-scope labels. Converted tensors are:

- train product corpus:
  `/tmp/spd-native-teacher-train224-v8-corpus.safetensors`
- train native teacher logits:
  `/tmp/spd-native-teacher-train224-v8-teacher.safetensors`

The unadapted 16k head scored `253 / 1763` train top-1 and `583 / 1763` train
top-4 on this broadened corpus. The conservative broad-corpus warm start at
`/tmp/spd-native-q4-adapt-train224-v8-e5-lr2e6-hard01/` reached
`1013 / 1763` train top-1 and `1287 / 1763` train top-4, but scored only
`21 / 61` held-out top-1 and `33 / 61` held-out top-4 on the frozen held-out
gate. The prior top-4 recipe rerun at
`/tmp/spd-native-q4-adapt-train224-v8-e3-lr5e6-hard01/` tied the old best
held-out top-1 at `22 / 61`, but regressed held-out top-4 to `32 / 61`. This
means more rows from the small reference-eval prompt pool do not lift the
candidate-set ceiling. The next useful scale step is a genuinely broader
product-token shard, starting with UltraChat prompts rendered through the same
Qwen no-thinking template, not more sweeps on these `224` examples.

The broader UltraChat-native gate changes the interpretation. The reproducible
builder `evals/spd/build_hf_prompt_tokens.py` now creates HF prompt-token
shards with fixed train/held-out splits, source indices, and Qwen
`enable_thinking=false` chat-template rendering. The first serving-shaped shard
is `/private/tmp/spd-qwen3-8b-ultrachat-serving-v1-max480`: `1024` train
prompts and `256` held-out prompts from `HuggingFaceH4/ultrachat_200k`
`train_sft`, shuffled with seed `23`, capped at `480` prompt tokens. The
held-out capture
`/tmp/spd-native-teacher-ultrachat-serving-v1-heldout256-v4-ctx1024` wrote
`1024` product rows and `1024` native-teacher rows with exact byte counts;
`983 / 1024` labels are inside the 32k draft scope and native teacher argmax
matches the in-scope Q4 target on `983 / 983`. Converted tensors are:

- held-out product corpus:
  `/tmp/spd-native-teacher-ultrachat-serving-v1-heldout256-v4-ctx1024-corpus.safetensors`
- held-out native teacher logits:
  `/tmp/spd-native-teacher-ultrachat-serving-v1-heldout256-v4-ctx1024-teacher.safetensors`

On this larger UltraChat held-out gate, the original 16k head scored
`106 / 983` top-1 and `208 / 983` top-4. The reference-pool best
`/tmp/spd-native-q4-adapt-train137-v4-e5-lr2e6-hard01/` only scored
`147 / 983` top-1 and `284 / 983` top-4, confirming that the earlier
`61`-row reference held-out gate was too narrow and distribution-specific for
serving-shaped decisions. The UltraChat-only native-Q4 warm start trained from
`512` prompts at
`/tmp/spd-native-q4-adapt-ultrachat512-v4-ctx1024-e5-lr2e6-hard01/` scored
`346 / 983` top-1 and `541 / 983` top-4 on the same held-out gate. The mixed
reference+UltraChat warm start
`/tmp/spd-native-q4-adapt-mix-ref224-ultra512-v1-e3-lr2e6-hard01/` scored
`332 / 983` top-1 and `540 / 983` top-4. Treat this as evidence that
distribution-matched native-Q4 rows matter more than more reference-pool
sweeps.

The scaled UltraChat-native train capture
`/tmp/spd-native-teacher-ultrachat-serving-v1-train1024-v4-ctx1024` wrote
`4096` product rows and `4096` native-teacher rows with exact byte counts;
`3934 / 4096` labels are in the 32k draft scope and native teacher argmax
matches the in-scope Q4 target on `3934 / 3934`. Converted tensors are:

- train product corpus:
  `/tmp/spd-native-teacher-ultrachat-serving-v1-train1024-v4-ctx1024-corpus.safetensors`
- train native teacher logits:
  `/tmp/spd-native-teacher-ultrachat-serving-v1-train1024-v4-ctx1024-teacher.safetensors`

The 3-epoch scaled warm start
`/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e3-lr2e6-hard01/`
scored `360 / 983` top-1 and `543 / 983` top-4 on the larger UltraChat
held-out gate. The 5-epoch variant
`/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e5-lr2e6-hard01/`
is the current best native-Q4 adaptation: `383 / 983` top-1 and `574 / 983`
top-4 held-out, with train `1347 / 3934` top-1 and `2132 / 3934` top-4. This
materially improves over the original 16k head (`106 / 983`, `208 / 983`) and
the earlier 512-prompt UltraChat adaptation (`346 / 983`, `541 / 983`). It is
still not enough to claim speedup. Treat the request-path number as an
acceptance/round-trip-savings gate for whether SPD can keep the split pipeline
full, not as measured wall-clock speed evidence for this small local/two-node
shape.

A direct fresh native-Q4 control on the existing safetensors corpus is not a
valid serving comparison: these product rows already contain the sidecar input
after the manifest's `stage_projs` projection, so the rows are tied to the
checkpoint projection basis that captured them. Offline scoring of a fresh head
against those projected rows can look better, but serving will project live taps
with the fresh head's different projection weights. A same-recipe fresh control
confirmed that failure mode: offline held-out scored `413 / 983` top-1 and
`523 / 983` top-4, but local serving accepted `0 / 48` proposals with clean tap
counters. The current projected-corpus path is therefore valid for warm-start
adaptation from the same checkpoint `stage_projs` basis, not merely the same
logical topology, and not for direct sidecar training from scratch. Proper
direct native-Q4 training needs raw terminal-normalized tap-concat rows before
any `stage_projs` projection, plus a trainer path that applies and trains
`stage_projs`.

2026-06-19 raw-corpus training-path checkpoint: the product corpus writer now
emits `raw_rows.f32` beside the existing projected `rows.f32`. `raw_rows.f32`
stores terminal-final-normed tap concatenations before any sidecar
`stage_projs` projection, with per-row widths and offsets recorded in
`manifest.json`. `prepare_product_activation_corpus.py` preserves those rows as
`raw_tap_concat`, `raw_tap_offsets`, and `raw_tap_widths` in safetensors.
`train_product_activation_head.py` now supports `--input-mode raw`, which
projects each packed row through `g0_proj` / `stage_projs` inside the training
graph, so direct/fresh training updates the projection weights consistently.
`--input-mode auto` keeps checkpoint-mode warm starts on projected rows and
uses raw rows for `--init-mode fresh` when available; projected fresh training
remains rejected. `score_product_activation_head.py` also supports
`--input-mode raw` so offline scoring cannot silently reuse the old projection
basis. Validation now includes a live raw smoke, not just code-level checks.
The smoke at `/tmp/spd-raw-corpus-smoke-20260619` wrote `3` samples, `2` SPD
rows, raw row width `16384` with widths `[12288,4096]`, exact raw/projected
byte counts, and native Q4 teacher logits over the `32000`-token draft scope.
Conversion wrote `/tmp/spd-raw-corpus-smoke-20260619-corpus.safetensors` and
`/tmp/spd-raw-corpus-smoke-20260619-teacher.safetensors`. A one-step fresh raw
train completed, and a hard-label overfit smoke at
`/tmp/spd-raw-direct-overfit3-20260619/` reduced loss from `19.19` to `1.52`;
raw scoring reached `1 / 3` top-1 and `3 / 3` top-4 against native Q4 targets.
Treat this as live-data plumbing evidence that direct raw training can learn
through `stage_projs`, not as a sidecar-quality result. The next real gate is a
larger disjoint raw native-Q4 train/held-out corpus for the same `23,36`
topology.

The first disjoint raw gate is
`/tmp/spd-raw-gate-20260619`: train16 and heldout16 prompt-token subsets from
the UltraChat serving-shaped shard, each captured with `4` verify steps. The
train conversion wrote
`/tmp/spd-raw-gate-20260619/train16-v4-corpus.safetensors` and
`/tmp/spd-raw-gate-20260619/train16-v4-teacher.safetensors`, with `64` rows and
`60 / 64` labels in draft scope. The held-out conversion wrote
`/tmp/spd-raw-gate-20260619/heldout16-v4-corpus.safetensors` and
`/tmp/spd-raw-gate-20260619/heldout16-v4-teacher.safetensors`, with `64` rows
and `59 / 64` labels in draft scope. Native teacher argmax matches the Q4
target on every in-scope row in both sets.

Fresh raw training on only `64` rows is too weak to use as the production path:
`/tmp/spd-raw-direct-train16-v4-hardonly-e10-lr5e4-20260619/` scored
`4 / 59` held-out top-1 and `5 / 59` top-4 in scope. The existing current-best
checkpoint scores the same held-out raw gate at `24 / 59` top-1 and `41 / 59`
top-4 in scope, proving the raw path is consistent with the checkpoint
projection basis. A small raw-mode checkpoint adaptation from that current-best
checkpoint,
`/tmp/spd-raw-checkpoint-adapt-train16-v4-e5-lr2e6-hard01-20260619/`, left the
held-out gate unchanged at `24 / 59` top-1 and `41 / 59` top-4. The next
production-oriented move is scaled raw-mode checkpoint adaptation on more
UltraChat serving-shaped raw rows, not more tiny fresh-from-random training.

The scaled train64 raw gate then crossed the local package-backed
pipeline-fill threshold. Release `skippy-bench` captured
`/tmp/spd-raw-gate-20260619/train64-v4-corpus`: `256` rows from `64` train
prompts, exact raw/projected byte counts, native Q4 teacher logits, and
`247 / 256` labels in draft scope. Conversion wrote
`/tmp/spd-raw-gate-20260619/train64-v4-corpus.safetensors` and
`/tmp/spd-raw-gate-20260619/train64-v4-teacher.safetensors`. The original 16k
checkpoint scored `11 / 59` top-1 and `15 / 59` top-4 on frozen heldout16; raw
adaptation from the original 16k checkpoint improved that to `15 / 59` top-1
and `24 / 59` top-4. The stronger raw-mode adaptation from the current-best
checkpoint,
`/tmp/spd-raw-checkpoint-adapt-train64-v4-e3-lr5e6-hard01-20260619/`, scored
`28 / 59` heldout top-1 and `40 / 59` top-4.

That train64 candidate exported as a BF16 serving bundle at
`/tmp/spd-raw-checkpoint-adapt-train64-v4-e3-lr5e6-hard01-20260619/bundle/`
with serving checkpoint SHA
`69166291b4b9d433d73564d8035908cb9db9d3638dd7238136afa79a525d5a96`, and
`target/release/skippy-bench spd-fixture-parity` passed. Local package-backed
rolling smoke wrote
`/tmp/spd-raw-checkpoint-adapt-train64-v4-e3-lr5e6-hard01-20260619/openai-heldout16-local-rolling.json`:
content matched `16 / 16`, tap return/record/ignored failures were all `0`,
SPD proposed `39`, accepted `22`, rejected `17`, and committed `21`
optimistic tokens. The pipeline-fill estimate is now above break-even:
`22` saved versus `17` unsaved candidate token round trips, save rate `56.4%`,
and `paper_like_speedup_vs_serial_split=1.1282x`. Measured local decode remains
slower (`0.171x`) due same-machine contention and sidecar/rolling overhead; do
not report that as distributed speedup.

The broader heldout64 gate invalidated the narrow heldout16 promotion signal.
The heldout64 corpus at `/tmp/spd-raw-gate-20260619/heldout64-v4-corpus` uses
the first `64` UltraChat held-out prompts, has `256` rows, `241 / 256` labels
in draft scope, and has zero overlap with the train shards. Native teacher
argmax matches the Q4 target on all in-scope rows. Offline heldout64 scores:
original 16k `23 / 241` top-1 and `49 / 241` top-4; current-best warm start
`89 / 241` and `140 / 241`; train64 stronger raw adaptation `92 / 241` and
`138 / 241`; train128 stronger raw adaptation `101 / 241` and `146 / 241`;
train256 stronger raw adaptation `107 / 241` and `148 / 241`.

The train128 candidate exported and passed Rust fixture parity, but the full
heldout64 package-backed rolling smoke failed the pipeline-fill gate:
`/tmp/spd-raw-checkpoint-adapt-train128-v4-e3-lr5e6-hard01-20260619/openai-heldout64-local-rolling.json`
matched content on `64 / 64`, had `0` tap failures, proposed `168`, accepted
`62`, rejected `106`, and reported `62` saved versus `106` unsaved candidate
token round trips (`paper_like_speedup_vs_serial_split=0.7381x`). The first
`16` prompts were barely positive (`21` saved / `18` unsaved), so heldout16 is
now only a quick smoke/debug subset. Do not promote train128 to a real-node run.

The current best offline raw candidate is train256:
`/tmp/spd-raw-checkpoint-adapt-train256-v4-e3-lr5e6-hard01-20260619/`. Its
train corpus has `1024` rows and `986 / 1024` labels in draft scope. It scores
`107 / 241` heldout64 top-1 and `148 / 241` top-4, which is a real improvement
over train128 but still below the rough >50% acceptance level needed for a
two-stage round-trip ledger to clear break-even with margin. The next quality
step is more raw native-Q4 rows and/or a better recipe, not real-node timing.

The current best warm start exports cleanly for Rust serving:

- manifest:
  `/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e5-lr2e6-hard01/skippy-spd-head.json`
- serving checkpoint:
  `/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e5-lr2e6-hard01/spd-head.safetensors`
- serving checkpoint SHA:
  `cab69fd4a9405819dc1a51afe058f1617995d0858702a2510313d600158349fe`
- parity fixture:
  `/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e5-lr2e6-hard01/spd-parity-fixture.safetensors`

`target/release/skippy-bench spd-fixture-parity` exits successfully for that
bundle. Cached fixture parity is exact; direct fixture parity has a close-logit
rank swap in the tail of top-8 but passes the harness gate. The bounded local
package-backed rolling smoke on the first `16` UltraChat held-out prompts wrote
`/tmp/spd-native-q4-adapt-ultrachat-serving-v1-train1024-v4-ctx1024-e5-lr2e6-hard01/openai-heldout16-local-rolling.json`.
It matched baseline/SPD content on `16 / 16`, recorded `0` tap return failures,
`0` tap record failures, and `0` ignored taps, proposed `41`, accepted `19`,
and rejected `22`. The paper-style estimate is still below break-even:
`19` saved versus `22` unsaved candidate token round trips, save rate
`46.3%`, and `paper_like_speedup_vs_serial_split=0.9268x`. Measured all-local
decode remains slower (`0.173x`) because stage work and the sidecar contend on
one machine. Treat this as request-path correctness plus a near-miss
pipeline-fill quality gate, not measured SPD speed evidence.

The held-out live-tap check at
`/tmp/spd-qwen3-8b-product-finetune-paper3-train16-e5-lr2e5/live-tap-heldout8.json`
matched non-SPD greedy output on all `24` prompts and accepted `110 / 192`
proposals (`57.3%`). The old product head on the same held-out corpus accepted
`102 / 192`, so this is a real but modest quality improvement.

The all-local rolling OpenAI request-path smoke at
`/tmp/spd-qwen3-8b-product-finetune-paper3-train16-e5-lr2e5/openai-heldout8-rolling.json`
matched baseline/SPD content on all `24 / 24` held-out prompts, accepted
`81 / 160` proposals (`50.6%`), committed `74` optimistic tokens, saved `81`
candidate token round trips, left `79` unsaved, and recorded `0` tap return
failures, `0` tap record failures, and `0` ignored taps. Its idealized
two-stage `paper_pipeline_estimate` is `1.0125x` versus serial split. Measured
all-local decode is still only `0.140x` of baseline because local stages and
the sidecar contend on the same machine. Mean probe head time is `62.5ms`,
normal downstream wait is `104.0ms`, optimistic downstream wait is `91.4ms`,
and chained hidden wait is `33.1ms`.

A real two-node direct-cable smoke then exercised the same package, sidecar,
and `23,36` split with stage 0 on the coordinator and stage 1 on the worker.
The cable route measured about `1ms` ping, versus much higher and jitterier
ordinary LAN latency. One stale-native rerun produced exact text but `0`
proposals because the Rust release binary had been relinked without rebuilding
the Metal llama stage ABI after the final-stage tap patch. The fix was to run
`scripts/build-llama.sh` for the Metal stage ABI dir and then rebuild
`skippy-server` / `skippy-bench` against the refreshed native library.

With the refreshed native library, a one-prompt direct-cable smoke returned
HF36 taps, proposed `7`, accepted `1`, and had `0` tap failures or ignored
taps. The three-prompt paired regression set matched content on `3 / 3`,
accepted `8 / 18`, and also kept tap failures and ignored taps at `0`.

The full paired rolling report
`/tmp/spd-qwen3-8b-product-finetune-paper3-train16-e5-lr2e5/openai-heldout8-rolling-direct-nativefresh.json`
matched baseline/SPD content on all `24 / 24` prompts, accepted `78 / 156`
proposals (`50.0%`), launched `137` rolling verifier entries, reached
`max_in_flight=2`, accepted `65` oldest entries, rejected `58` oldest entries,
and drained `72` younger entries. It recorded `0` tap return failures, `0` tap
record failures, and `0` ignored taps. The paper estimate is exactly break-even
at `1.0x` (`78` saved / `78` unsaved), and measured decode is `0.321x` of
baseline (`439.0ms` baseline decode mean versus `1366.6ms` SPD decode mean).
Mean probe head time is `64.7ms`, normal downstream wait is `144.0ms`,
optimistic downstream wait is `66.5ms`, and chained hidden wait is `77.1ms`.

This is training-bridge, Rust parity, live-tap, request-path correctness, and
real two-node mechanics evidence. It is still not a speedup result. The sidecar
quality margin is too thin: local is barely above the paper threshold and the
real split is exactly break-even. The next spend should therefore be either
more product-distribution training data for the same `23,36` topology or native
Q4_K_M verifier logits for paper-faithful KL, not a larger generic
reference-distribution run and not a speed claim. This keeps the work aligned
with `spd.pdf`: speculation latency has to be hidden under target pipeline work,
and theoretical `L'_acc = N/K*n` must stay separate from measured wall-clock
speed.

## What Works

- A real SPD head can be trained locally for `Qwen/Qwen3-0.6B` with the paper's
  reference implementation.
- A real pretrained SPD head for `Qwen/Qwen3.5-4B` reaches high acceptance on
  local eval prompts.
- Real per-sample SPD eval traces can be fed into a Skippy split-stage latency
  model to estimate how much pipeline bubble/activation-hop latency SPD can
  hide.
- `skippy-runtime` can parse and validate the SPD head manifest/checkpoint
  binding, including a Rust-readable safetensors serving checkpoint and
  selected tensor payload reads.
- `skippy-runtime` can run the pretrained `Qwen/Qwen3.5-4B` SPD head over a
  recorded Python fixture and match Python top-k draft candidates.
- `skippy-runtime` exposes `SpdQwen3Head`, a reusable loaded-head boundary for
  repeated Qwen SPD proposals without reopening the manifest/checkpoint each
  time.
- `skippy-runtime` can read GGUF `token_embd.weight` rows directly for the
  SPD hidden-state-index `0` embedding tap on the current Qwen3.5-4B proof
  model, including Q8_0 rows used by the Qwen3-8B diagnostics.
- `skippy-runtime` can also read the SPD h0 embedding tap from selected Skippy
  layer-package parts, including quantized Q4_K and Q6_K embeddings, so
  package-backed stage-0 SPD replay no longer has to open an invalid `0..0`
  layer stage or carry a coordinator-side full GGUF.
- `skippy-runtime` can reconstruct the SPD `cur_in` rows from raw recorded
  hidden-state tap inputs using `g0_proj` and `stage_projs.*`.
- `skippy-model-package` can plan, write, and preflight explicit tap-aligned
  layer splits for the `Qwen/Qwen3.5-4B` S4/L4 proof head.
- `skippy-bench local-split-chain-binary` can run the `Qwen/Qwen3.5-4B` GGUF
  through the full tap-aligned seven-stage Skippy binary chain locally, using
  `CPU0` to bypass local Metal auto-selection.
- `skippy-bench spd-live-tap-parity` can assemble the pretrained Qwen3.5-4B
  SPD head input from live Skippy activation frames, including an
  embedding-only side tap for hidden-state index `0`, run the Rust SPD head
  from those live taps, and verify repeated live top-1 proposals with the
  Skippy target verifier.
- `skippy-bench spd-live-tap-parity --product-corpus-dir <dir>` can now write
  product live-tap SPD rows for sidecar preparation. The corpus contains
  terminal-final-normed `cur_in` rows in `rows.f32`, verifier/proposal metadata
  in `rows.jsonl`, and a `manifest.json` that records topology, model package,
  split, row hf-indices, and the explicit limitation that current labels are
  product verifier greedy top-1 tokens rather than full teacher logits. The
  helper `evals/spd/prepare_product_activation_corpus.py` converts that corpus
  to a safetensors tensor dataset and maps labels into `draft_token_ids` when
  the SPD manifest provides a draft vocab.
- `evals/spd/augment_product_activation_teacher_logits.py` can attach frozen HF
  teacher logits to product-captured rows, aligned by `query_row_index` and
  target position. This is KL-compatible training data for the reference head,
  but it is still HF-teacher data, not native Q4_K_M verifier logits.
- `evals/spd/build_product_prompt_tokens.py` can render the reference
  MT-Bench, GSM8K, and HumanEval prompt JSONL files through the target
  tokenizer chat template, split them into train and held-out prompt-token
  files, and write the exact JSONL accepted by
  `skippy-bench spd-live-tap-parity --prompt-token-file`.
- `evals/spd/diagnose_product_teacher_alignment.py` compares product Q4 target
  tokens, HF teacher top-k tokens, corpus-captured proposals, and optional live
  head proposals so sidecar-quality work can separate teacher/verifier mismatch
  from a head that simply has not learned the available teacher.
- `evals/spd/train_product_activation_head.py` can fine-tune an existing
  `speculation_head_final.pt` on product `cur_in` rows plus aligned teacher
  logits and save a reference-compatible checkpoint.
- `skippy-server` has a request-path speculative proposal-source boundary in
  front of the existing target verify/repair/rollback loop. The current draft
  model path uses it, and an experimental `spd-replay` source can load the
  pretrained Qwen3.5-4B head from `--openai-spd-manifest` /
  `--openai-spd-fixture`.
- A bounded local OpenAI request through `skippy-server` has exercised the
  pretrained head in the live serving path: four `spd-replay` proposals, two
  accepted, two rejected, and the same greedy text as the no-SPD baseline.
- A release `skippy-server` request-path smoke now runs the pretrained head
  from inline returned Skippy taps plus direct GGUF h0 embeddings without
  `--openai-spd-replay-fallback`. A no-thinking prompt/template smoke first
  reproduced the exact HF reference target stream
  `[71093, 12305, 198, 727, 884, 2784, 11, 292]`; the Python reference
  `generate(..., draft_top_k=1)` accepted `7 / 8` tokens on that same prompt,
  proving the low-acceptance serving run was not a head-quality issue. The
  serving bug was token-position alignment: rolling observer positions were
  using the previous-token predictor position, so post-first proposal rows read
  the current token from the wrong slot and repeatedly proposed it. After
  switching live rolling positions to actual token indices, the same bounded
  OpenAI smoke kept exact output, proposed `8` tokens, accepted `8`, rejected
  `0`, committed `3 / 3` optimistic target decodes, and recorded `0` tap
  return failures, `0` tap record failures, and `0` ignored taps. Baseline was
  `626ms` wall / `198ms` decode; SPD was still slower at `3521ms` wall /
  `2921ms` decode on the local CPU proof, so this is an acceptance/scheduling
  fix rather than a speedup claim. The report's paper estimate for the logical
  `S=4` head showed a `4.0x` paper-like speedup versus serial split from the
  observed `1.0` accept rate, while the current implementation remained about
  `59x` slower than that estimate because proposal and stage scheduling are
  still local proof code.
- A pre-patch ungated optimistic diagnostic that did not request returned taps
  was faster (`2591ms` wall / `2004ms` decode with exact output), but it only
  produced `3` proposals, committed `2` optimistic tokens, and the rolling
  replay reported `5` missing proposal positions plus `2` out-of-order
  proposals. Treat that as evidence that tap-return cost is real, not as a
  correct serving mode.
- The current request path asks for SPD tap returns whenever optimistic SPD
  decode starts. The latest ungated no-thinking smoke kept exact output,
  proposed and accepted `8 / 8`, committed `4 / 4` optimistic decodes, recorded
  `0` tap failures and `0` ignored taps, and the rolling replay had `7`
  inserted drafts with `0` missing and `0` out-of-order proposals. Baseline was
  `629ms` wall / `201ms` decode; SPD was `4154ms` wall / `2919ms` decode. This
  is the most faithful current serving proof, and it points directly at
  tap-return transport plus fully overlapped rolling execution as the remaining
  bottleneck.
- With `25ms` injected downstream-stage delay, the same current always-tap
  request path still kept exact output, accepted `8 / 8`, committed `4 / 4`
  optimistic decodes, and kept rolling replay ordered. The latency gap narrowed
  but did not cross over: baseline was `2807ms` wall / `1938ms` decode; SPD was
  `4210ms` wall / `3105ms` decode (`0.667x` wall, `0.624x` decode). The paper
  estimate for the same trace was `484ms` decode at the baseline stage cost, so
  current SPD is still about `6.4x` slower than the paper-shaped rolling
  schedule even when artificial hop latency is present.
- 2026-06-17 row-specific tap collection removed a serving blocker where
  proposal assembly treated every required hidden-state tap as required for
  every row. A `[4, 4, 4, 0]` proposal window no longer waits for downstream
  taps on the `g_0` row. The fixed no-delay smoke at
  `/private/tmp/spd-openai-sparse-rows-smoke1/report.json` preserved exact
  output, accepted `5 / 5`, and moved optimistic probes earlier than `h31`
  (`trigger_hf_index=20,20,16` after bootstrap). The 8-token rerun at
  `/private/tmp/spd-openai-sparse-rows-smoke8/report.json` accepted `8 / 8`
  with exact output, but SPD remained slower (`205ms` baseline decode versus
  `5631ms` SPD decode). With `25ms` downstream delay, the sparse-row smoke at
  `/private/tmp/spd-openai-sparse-rows-delay25-smoke1/report.json` kept exact
  output and `5 / 5` acceptance and narrowed decode speed to `0.353x`
  SPD-vs-baseline. This proves the missing-tap gate is fixed; the remaining
  speed gap is rolling executor/direct-return scheduling, not proposal row
  availability.
- 2026-06-17 rolling-preferred serving now keeps SPD optimistic decode on the
  direct-return path instead of letting the older primary `VerifySpan` branch
  consume the remainder after the first burst. The no-delay smoke at
  `/private/tmp/spd-openai-rolling-prefer-smoke8/report.json` preserved exact
  output, accepted `8 / 8`, committed `6 / 6` optimistic verifier results
  (`4 / 4` chained), emitted two pre-target bursts plus six
  optimistic-commit probes, and left rolling replay with `0` missing or
  out-of-order proposals. It is still not a speed result (`207ms` baseline
  decode versus `15210ms` SPD decode), which isolates the next blocker to
  verifier overlap/hidden waits rather than sidecar quality or rolling-row
  availability. The same 8-token smoke with `25ms` downstream delay at
  `/private/tmp/spd-openai-rolling-prefer-delay25-smoke8/report.json` also
  preserved exact output, accepted `8 / 8`, committed `6 / 6` optimistic
  verifier results, and reached only `0.242x` decode speed versus baseline
  (`1997ms` baseline decode, `8247ms` SPD decode).
- 2026-06-17 one-token verifier execution now routes single-token
  `VerifySpan` messages through the normal decode-frame runtime path while
  preserving the `VerifySpan` wire shape, direct-return reply shape, and
  rollback checkpoints. The comparable no-delay smoke at
  `/private/tmp/spd-openai-single-token-decode-smoke8/report.json` preserved
  exact output, accepted `8 / 8`, and committed `6 / 6` optimistic verifier
  results (`4 / 4` chained), but only improved SPD decode from `15210ms` to
  `13516ms` (`224ms` baseline). A temporary unshipped diagnostic that skipped
  verifier checkpoints
  (`/private/tmp/spd-openai-skip-verify-checkpoint-smoke8/report.json`) still
  took `9527ms` SPD decode versus `215ms` baseline. Per-stage logs show the
  remaining long calls overlap across local stage processes on the same M4
  Metal device; baseline is serial, while rolling SPD creates true concurrent
  stage compute. This local single-GPU smoke is therefore a correctness and
  scheduler-shape test, not a fair speed oracle. The next decisive benchmark
  needs stage placement across distinct devices/nodes.
- 2026-06-17 `skippy-bench spd-openai-smoke` can now run the same OpenAI SPD
  request-path smoke with explicit stage placement. `--stage-hosts` cycles
  stage placement across `local` plus remote SSH targets, stage 0 remains local
  so the OpenAI frontend and sidecar stay on the coordinator, and
  `--endpoint-host-map local=<reachable-stage0-host>` makes the direct-return
  topology usable from remote stages. `--remote-model-path-map` can point each
  remote target at an existing GGUF, or `--rsync-model-artifacts` can copy the
  model into the run directory. The post-refactor local regression smoke at
  `/private/tmp/spd-openai-remote-refactor-local-smoke2b/report.json`
  preserved exact output, accepted `2 / 2` SPD proposals, committed one
  optimistic token, and recorded `0` tap failures. This validates the benchmark
  path after the placement extraction; the speed question still requires a
  distinct-device run.
- 2026-06-17 `spd-openai-smoke --preflight-only` now validates first-node SPD
  run inputs without launching stages. The Qwen3.5-4B preflight at
  `/private/tmp/spd-qwen35-first-remote-preflight.json` checked the release
  `skippy-server` binary, the 2.74 GB GGUF, the sidecar manifest, `66` serving
  checkpoint tensors, `28` parity fixture tensors, logical `S=4`, physical
  split `8,10,16,20,24,31`, tap returns `8,10,16,20,24,31`, local stage port
  `20031`, and a complete stage-0-local plus worker endpoint plan with no
  warnings.
- 2026-06-17 the local CPU multi-token repeat at
  `/private/tmp/spd-local-multitoken-repeat-cpu.json` preserved exact output for
  `3 / 3` measured baseline/SPD pairs, accepted `24 / 24` SPD proposals,
  committed `18` optimistic tokens with `12` chained commits, and kept rolling
  replay ordered (`21` inserted drafts, `15` accepted windows, `0` missing,
  `0` out-of-order). It is still a negative speed result: baseline decode mean
  was `219.3ms`, SPD decode mean was `13964.2ms`, while the paper estimate from
  the observed trace was `54.8ms`. The timing splits point away from a missing
  sidecar cache port: proposal cache prefill averaged `16.8ms` over `24`
  probes, sidecar head total averaged `45.9ms`, normal downstream wait averaged
  `2681.2ms`, and optimistic hidden wait averaged `2169.6ms`.
- 2026-06-17 the matching local Metal repeat at
  `/private/tmp/spd-local-multitoken-repeat-metal.json` preserved the same
  exactness shape (`3 / 3` content matches, `24 / 24` accepted proposals,
  `18` optimistic commits, `12` chained commits, `0` tap failures, `0` missing
  or out-of-order rolling proposals). Metal reduced SPD decode from the CPU
  run's `13964.2ms` mean to `1652.6ms` mean and cut optimistic hidden wait from
  `2169.6ms` to `90.5ms`, but it was still slower than the `201.0ms` baseline
  decode (`0.122x`). The paper estimate from the same accepted trace was
  `50.2ms`, so the remaining gap is still the native rolling executor and
  same-machine stage contention, not sidecar acceptance.
- 2026-06-17 an opt-in native rolling executor now runs inside the
  `skippy-server` OpenAI SPD request path behind
  `--openai-spd-rolling-executor`, and `skippy-bench spd-openai-smoke` passes
  it with `--spd-rolling-executor`. The first local preflight at
  `/private/tmp/spd-rolling-executor-local-preflight.json` validated the same
  Qwen3.5-4B S4/L4 seven-stage split and tap plan without launching stages. The
  paired local smoke at `/private/tmp/spd-rolling-executor-local-paired-final.json`
  preserved exact baseline/SPD output for a six-token request, launched `5`
  executor-owned speculative verifies from direct-return tap callbacks, reached
  the logical `S=4` max in-flight depth, committed `3` oldest entries, rejected
  `0` oldest entries, drained `0` younger entries, and recorded `0` tap
  failures. This closes the earlier diagnostic-only rolling scheduler gap for
  a request-path smoke. It is still a negative speed result on one local debug
  machine (`170.5ms` baseline decode versus `25149.1ms` SPD decode), so the
  next proof needs real split placement on distinct hardware rather than more
  same-machine timing.
- 2026-06-17 follow-up rolling-executor work moved speculative direct-return
  taps into a pending cache that is overlaid for rolling proposals, promoted
  only when the accepted context reaches those positions, and cleared on
  verified-context reset. The executor target observer now drains every ready
  oldest scheduler commit after a target token arrives instead of checking only
  once. Focused SPD tests, `cargo clippy -p skippy-server --all-targets -- -D
  warnings`, and `cargo test -p skippy-server --lib` pass. The current debug
  Metal smoke at
  `/private/tmp/spd-rolling-executor-metal-smoke8-commit-drain.json` preserves
  exact output, reaches `max_in_flight=4`, and keeps rolling replay clean
  (`0` missing / `0` out-of-order), but it still proposes `8`, accepts `7`,
  rejects `1`, and is far slower than baseline (`229.1ms` baseline decode
  versus `23301.0ms` SPD decode). This is not the final paper-shaped executor
  yet: the request path still processes younger chained verifier replies before
  the rolling executor owns commit/restore. A deeper-row launch gate experiment
  was not retained because it starved the executor (`max_in_flight=3`), reduced
  acceptance to `5 / 7`, and reintroduced missing replay proposals.
- 2026-06-17 native rolling-executor recovery now uses the existing
  request-scoped `Stop` reset path before replaying the canonical prefix after
  a rolling rejection, instead of trying to repair dirty stage sessions with a
  trim-only replay. The replay path also always resends `ConfigureGeneration`
  after reset so downstream final stages reopen their direct-return stream even
  when the request has no chat sampling metadata. Rolling rejection no longer
  disables future rolling launches for the rest of the request; the executor
  now has a regression test proving it can drain younger work, reset to the
  corrected prefix, and accept fresh verifier launches. Code-level gates pass:
  `cargo fmt --all`; `cargo test -p skippy-server --lib spd::`;
  `cargo test -p skippy-server --lib generation_config_message_without_metadata_still_configures_generation`;
  `cargo check -p skippy-server`;
  `cargo clippy -p skippy-server --all-targets -- -D warnings`;
  `cargo test -p skippy-server --lib -- --skip accepted_binary_stage_connection_is_blocking`;
  `cargo test -p skippy-bench spd_openai`;
  `cargo check -p skippy-bench`;
  `cargo clippy -p skippy-bench --all-targets -- -D warnings`; and
  `cargo build -p skippy-server -p skippy-bench`. The pretrained Qwen3.5
  artifact path is still healthy: `skippy-bench spd-fixture-parity` matched the
  recorded Python top-k token ids, the external `skippy-runtime` manifest,
  fixture, and Qwen3 fixture-forward tests pass with the real manifest/fixture,
  and the rebuilt `spd-openai-smoke --preflight-only --spd-rolling-executor`
  report at `/private/tmp/spd-rolling-executor-reset-smoke24-preflight.json`
  validates the GGUF, sidecar checkpoint, parity fixture, tap coverage, and
  `8,10,16,20,24,31` split. `skippy-bench spd-openai-check` now provides an
  offline report gate for the first real smoke: by default it requires exact
  baseline/SPD content, at least `24` accepted SPD tokens, `max_in_flight >= 4`,
  `0` oldest rolling rejections, `0` drained younger replies, `0` tap failures,
  `0` missing/out-of-order rolling replay proposals, and a verified rolling
  prefix that matches the target. The required follow-up is still a
  model-backed `spd-openai-smoke --spd-rolling-executor` run with real stage
  ports; this checkpoint is not yet a content-match or speed claim.
- 2026-06-17 the first real LAN split checkpoint is now model-backed and
  content-matched. A one-worker CPU correctness run at
  `/private/tmp/spd-lan-cpu-spd8.json` kept stage 0, the OpenAI frontend,
  and the SPD sidecar on the coordinator, placed physical stages 1-6 on one LAN
  worker, used `--n-gpu-layers 0 --spd-n-gpu-layers 0`, matched baseline
  output, accepted `7 / 7` SPD proposals, reached `max_in_flight=4`, rejected
  `0` oldest entries, drained `0` younger replies, and passed
  `spd-openai-check` with one allowed tail missing proposal. The stage logs had
  no connection errors, no Metal OOM lines, and no `llama_decode failed` lines.
  This proves the explicit stage placement, direct-return tap transport,
  rolling executor, and Skippy KV/session path can complete across the LAN.
  It is not a speed claim: CPU baseline decode was `1501.4ms` and CPU SPD
  decode was `4991.3ms`. The first all-Metal one-worker attempt failed because
  six remote Metal-backed stage processes oversubscribed the worker and stage 3
  hit out-of-memory on the first `DecodeEmbd`; future speed gates need distinct
  devices/workers or fewer Metal-backed stage processes per worker.
- 2026-06-18 the product-shaped two-stage baseline now works on the real
  M4-plus-worker split. The report
  `/private/tmp/skippy-two-stage-baseline.json` used exactly two physical
  stages, `--splits 16 --layer-end 32`, with stage 0 on the coordinator and
  stage 1 on the worker, `--n-gpu-layers=-1`, and a 24-token bounded prompt.
  Baseline wall time was `1678.9ms`, decode was `1293.2ms`, stage-0 compute was
  `253.0ms`, downstream wait was `990.2ms`, and tap counters stayed clean. This
  is a baseline split proof only: the current pretrained Qwen3.5 S4/L4 sidecar
  requires the physical tap split `8,10,16,20,24,31` and must not be used for a
  two-stage speed claim. The matching sidecar topology is `num_stages=2` with
  `stage_layer_boundaries=16,32`, deriving tap rows `0,16,32;0,16`.
- 2026-06-18 Mesh-native two-node Qwen3-8B layer-package proof now completes
  through the product resolver/download/stage-control path. Both nodes used the
  exact immutable package ref for `meshllm/Qwen3-8B-Q4_K_M-layers`, with HF
  credentials visible on the worker and artifact transfer enabled. Mesh elected
  the worker as stage-0 coordinator and placed the local M4 as downstream
  `stage-1` with `layer_range=23..36`; the worker downloaded the missing stage
  package artifacts, advertised the model, and a local OpenAI chat request
  through the proxy succeeded. This is a real two-node Skippy serving proof,
  not an SPD proof or speed claim. For SPD, the important result is the
  topology: a sidecar for this run must be trained/exported for
  `num_stages=2`, `stage_layer_boundaries=23,36` on `Qwen/Qwen3-8B`, or the
  split planner must be constrained by the SPD manifest before using a sidecar
  trained for different boundaries. The post-run Ctrl-C shutdown hit a
  ggml-metal cleanup assert after the successful inference; that was not an
  inference-path failure.

Topology preflight for that exact product split is now a no-spend trainer mode.
It exits before cloning the reference repo or downloading model data:

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
  --model-torch-dtype float16 \
  --upload-repo ''
```

The expected topology output is `physical_split_boundaries=[23]`,
`layer_end=36`, `shallow_hidden_layer_indices="0,23,36;0,23"`, and worker
tap-return allowlist `[23,36]`. A real sidecar training run should use the same
topology arguments without `--dry-run-topology`; use smaller `--train-rows`
only for plumbing, not for the quality artifact. On this local M4/MPS path,
use `--model-torch-dtype bfloat16` for Qwen3-8B runs that take more than a tiny
debug step count. A 64-row float16 run completed but every exported head tensor
was non-finite, while the matching bfloat16 run stayed finite. The runner's
default `auto` dtype preserves the older float32 MPS behavior used by smaller
proof heads.

2026-06-18 local Qwen3-8B S2 `23,36` sidecar plumbing checkpoint: a tiny
2-row MPS run loaded the full `Qwen/Qwen3-8B` HF weights on the local M4 with
`--model-torch-dtype float16`, trained `num_stages=2`,
`stage_layer_boundaries=23,36`, `num_spec_layers=4`, `max_length=64`,
`gradient_accumulation_steps=1`, and wrote a topology-bound
`speculation_head_final.pt` plus `skippy-spd-head.json`. This proves the local
trainer can load the first product topology and produce a real checkpoint, but
it is not a quality artifact: it used only 2 UltraChat rows, reported
`train_loss=7.961`, and should not be used for a speed or acceptance claim.

The same debug checkpoint exported successfully to a Rust-readable BF16 serving
artifact: `spd-head.safetensors`, `56` tensors, about `2.12 GiB`, with manifest
topology `hidden_size=4096`, `draft_vocab_size=32000`, `num_stages=2`,
`stage_layer_boundaries=[23,36]`, and tap rows `[[0,23,36],[0,23]]`. An F16
export was rejected by the current Rust parity runner (`unsupported f32 dtype
F16`), so use BF16 or F32 for Skippy SPD serving exports until F16 tensor reads
are implemented intentionally. Rust external manifest and fixture validation
passed for the debug bundle. `skippy-bench spd-fixture-parity` passed against a
one-prompt parity fixture: tap-input reconstruction was effectively exact
(`max_abs_diff=0.000061`), the non-cached Rust forward top-8 token ids matched
Python exactly, and the cached diagnostic returned the same top-token set with
one rank swap. Treat this as training/export/Rust-forward plumbing evidence
only.

The reference eval patch now propagates explicit `stage_layer_boundaries` into
the Python pipeline simulator, so the Qwen3-8B S2 head runs target stages as
`0..23` and `23..36` instead of the old equal `0..18` and `18..36` split. The
previous `KeyError: 23` custom-tap fill failure is fixed for boundary-derived
rows. A tiny debug eval of the 2-row head with `3` one-row eval sets,
`max_new_tokens=8`, greedy decoding, and `draft_top_k=4` reported
`24` generated tokens, `42` decode steps, aggregate acceptance `0.5714`,
equivalent accept length `1.1429`, theoretical throughput gain `14.29%`, and
`3 / 24` accepted draft flags. This is only a plumbing acceptance signal for
the tiny debug head; the real artifact still needs a larger training run and a
same-topology baseline/SPD request-path comparison.

The package-backed Qwen3-8B request path now reaches the local release smoke
with the same `23,36` topology and `--activation-width 4096`. The stage-0 SPD
h0 reader can open Q4_K `token_embd.weight` from the Skippy layer package's
`embeddings.gguf`, so the smoke no longer needs a coordinator-side full-GGUF
override only for embedding rows. A preflight guard now rejects
`--activation-width` values that do not match the manifest hidden size; the old
Qwen3-8B `2560` default mistake fails before launching stages. The paired local
report `/private/tmp/spd-qwen3-8b-s2-23-debug-local-openai-paired-8.json`
matched baseline/SPD content, proposed `7`, accepted `0`, rejected `7`,
recorded `0` tap return/record/ignored failures, and used inline package taps
for all proposals (`7` inline hits, `0` replay fallbacks). Baseline decode was
`134.3ms`; SPD decode was `586.4ms`, with about `426ms` spent in the sidecar
forward/head path. This proves package-backed Qwen3-8B SPD request-path
plumbing for the exact product topology. It is not speed or quality evidence:
with `0 / 7` accepted proposals, there are no future verifier/stage round trips
to remove from the critical path, so the expected ideal speedup is effectively
`1.0x` before overhead and the measured result is necessarily slower.

2026-06-18 local Qwen3-8B S2 `23,36` 64-row training checkpoint: a float16 MPS
run with `train_rows=64`, `max_length=128`, `gradient_accumulation_steps=4`,
and `draft_top_k=4` produced an all-non-finite checkpoint despite completing
training/eval. The export scripts now reject non-finite checkpoint and fixture
tensors so this cannot silently become a serving bundle. Repeating the same
shape with `--model-torch-dtype bfloat16` produced a finite checkpoint at
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-train64-20260618-104718/artifacts/20260618-104718`.
Reference eval over `24` prompts and `384` generated tokens reported aggregate
acceptance `0.5378`, equivalent accept length `1.0756`, theoretical gain
`7.59%`, and `30 / 384` accepted draft flags. BF16 export produced a
Rust-readable `spd-head.safetensors` with `56` tensors; parity fixture export
produced finite logits and real top-token ids. `skippy-bench
spd-fixture-parity` passed mechanically; tap reconstruction was close
(`max_abs_diff=0.000488`), the uncached Rust/Python top-8 token set matched
with rank/logit drift, and the cached diagnostic still diverged more. Local
package-backed `spd-openai-smoke` on the same `23,36` topology matched
baseline/SPD content, proposed `15`, accepted `0`, rejected `15`, recorded
`0` tap failures, used `15` inline package taps with `0` replay fallbacks, and
measured `243.4ms` baseline decode versus `1187.8ms` SPD decode. This is a
finite local training/export/request-path checkpoint, not a speed candidate:
serving acceptance on the default prompt is still zero, and sidecar forward/head
time was about `879ms` for `16` generated tokens.
`spd-openai-smoke` now records the same bottom line directly in
`summary.paper_pipeline_estimate`: candidate token round trips are proposed SPD
tokens, saved token round trips are accepted SPD tokens, and unsaved token round
trips are rejected proposals. For this checkpoint, that math is `15`
candidates, `0` saved, and `15` unsaved, so there is no ideal critical-path
speedup to claim before overhead.

2026-06-18 local Qwen3-8B S2 `23,36` 512-row training checkpoint: a larger
bfloat16 MPS run used the same exact product topology with `train_rows=512`,
`max_length=256`, `max_new_tokens=32`, `batch_size=1`,
`gradient_accumulation_steps=4`, `epochs=1`, `learning_rate=1e-5`,
`num_spec_layers=4`, `draft_top_k=4`, and the reference UltraChat Qwen top-32k
draft vocab. Training completed in `10.37min` with `train_loss=28.96`; the
checkpoint lives under
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-train512-20260618-105916/artifacts/20260618-105916`.
Reference eval over `48` prompts and `1536` generated tokens reported
aggregate acceptance `0.5306`, equivalent accept length `1.0611`,
theoretical gain `6.21%`, and `135 / 1536` accepted draft flags at
`draft_top_k=4`, so it did not improve over the 64-row proof. BF16 serving
export passed the non-finite guard with `56` tensors. After fixing Rust serving
to use the Qwen3 rotary config (`rope_theta=1000000`, `rotary_dim=128`) and the
Qwen3 final RMSNorm style, the refreshed serving checkpoint SHA is
`0f101bbc5b14b928b2328a7dc38d4ffdc6c0e03e635ffb1d7a78ee86ac421898`; parity
fixture export produced finite logits and SHA
`46790b94455d2b421d7fe4a8606d3e29a7163be4407171e74ec5647cb15430c4`.
`skippy-bench spd-fixture-parity` now matches the Python top-k order exactly on
the forward and cached fixture paths. Remaining numeric drift is BF16-scale:
tap reconstruction `max_abs_diff=0.000122`, forward `spec_query_max_abs_diff`
`0.0234`, forward `final_hidden_max_abs_diff=0.1875`, cached
`logits_max_abs_diff=0.125`. The pretrained Qwen3.5-4B fixture still matches
top-k exactly on the legacy Qwen3.5 rotary/final-norm defaults, so the fix did
not regress the earlier sidecar.

The reference wrapper now also patches the single-chain evaluator's stage
stepping path, so serving-equivalent `draft_top_k=1` eval works for the custom
S2 `23,36` topology instead of failing with `KeyError: 23`. The fresh top-1
report
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-top1-eval-20260618-continued2/artifacts/20260618-114228/eval/summary/pipeline_eval__train__speculation_head_final__nt12__summary.json`
processed `12` prompts and `384` generated tokens. It accepted only `12 / 384`
draft flags, with equivalent accept length `1.0323` and theoretical gain
`3.23%`. This is the serving-aligned quality metric for this checkpoint; it is
nonzero but too weak to justify a two-node speed run.

2026-06-18 LR diagnostic: the same 512-row BF16 local MPS run repeated with the
paper/reference learning rate `1e-4` instead of `1e-5`, still on Qwen3-8B S2
`23,36`, `max_length=256`, `num_spec_layers=4`, and `draft_top_k=1` eval. The
run lives under
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-train512-lr1e4-20260618/artifacts/20260618-114627`.
Training completed in `10.35min`, `train_loss=27.14`, and the end-of-run train
log accuracy rose to about `0.099`. Reference top-1 eval on the same 12-prompt
mini benchmark set improved to `41 / 384` accepted draft flags, equivalent
accept length `1.0741`, and theoretical gain `7.50%`. BF16 serving export SHA
is `04629f50d1499a8714451e711ca5bc65087e311f82a03ff1f69b63d43ed26054`; parity
fixture SHA is `fef2b8be57e3385c0f2c9d0f716e5d0e38db074171ba22f453964b2636fb8682`.
Rust fixture parity matches the forward and cached top-4, with a BF16-scale
rank swap only at forward top-8. Package-backed serving on the six original
code/math/writing prompts still accepted `0 / 90`, but a targeted GSM8K
mini-eval smoke with `ctx_size=256` accepted `2 / 120`, matched content on all
4 prompts, and kept tap counters clean. This proves the improved head can
accept through Rust/package-backed serving, but the product acceptance rate is
still far below the reference eval.

The Qwen3 no-thinking reference wrapper now matches the product/HF native
`enable_thinking=false` prompt surface, including the assistant
`<think></think>` prefill. A fresh eval-only run reused the LR `1e-4`
checkpoint with the aligned template:
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-lr1e4-template-aligned-eval-20260618/artifacts/20260618-120902/eval/summary/pipeline_eval__train__speculation_head_final__nt36__summary.json`.
Across `36` prompts and `1152` generated tokens it accepted `140 / 1152`
draft flags, equivalent accept length `1.0817`, and theoretical gain `8.25%`.
Dataset split was GSM8K `50 / 384`, MT-Bench `51 / 384`, and HumanEval
`39 / 384`. Therefore the product-smoke gap is **not** explained by the prior
chat-template mismatch alone. The next gate is an identical-prompt parity
diagnostic: render the exact product OpenAI prompt tokens, run reference and
Rust/package-backed SPD over those same token ids/prompts, and compare
proposal token ids, tap rows, and target verification decisions position by
position before spending on larger training.

A follow-up token-trace reference eval added `prompt_token_ids`,
`generated_token_ids`, and `token_acceptance` to the raw JSONL:
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-lr1e4-token-trace-eval-20260618/artifacts/20260618-121348/eval/raw/pipeline_eval__train__speculation_head_final__nt12__per_sample.jsonl`.
The first four GSM8K prompts have the same prompt-token lengths in product and
reference (`51`, `117`, `55`, `76`), but the generated target tokens only match
all `32` positions for prompts `0` and `2`; prompt `1` diverges at generated
token index `6`, and prompt `3` diverges immediately at generated token index
`0`. This exposes a real hidden assumption: the reference eval is using HF
Qwen3-8B weights, while the product smoke verifies against the Q4_K_M GGUF
layer package. On the comparable identical-output prompts, product still trails
reference for prompt `2`: reference has `5 / 31` speculative accepts after
excluding the always-true first generated token, while product accepts `2 / 27`.
Prompt `0` has `0 / 31` speculative accepts in both. Do not treat the aggregate
reference count as product acceptance until the sidecar is evaluated against
the same quantized target behavior or the product smoke is rerun on a
higher-precision GGUF target.

The corrected parity did **not** make the 512-row Qwen3-8B sidecar a product
candidate. The paired local package-backed OpenAI sweep
`/private/tmp/spd-qwen3-8b-s2-23-bf16-train512-local-openai-sweep6-16-after-parity.json`
matched baseline/SPD content on all `6` code/math/writing prompts and kept
clean taps, but accepted `0 / 90` top-1 proposals. The paper estimate therefore
remains `90` candidate token round trips, `0` saved, and `90` unsaved. Mean
baseline decode was `251.3ms`; mean SPD decode was `1226.7ms`; mean sidecar
head time for proposed tokens was `59.6ms`. Do not run a two-node speed
comparison with this head; it would only prove overhead. The next real sidecar
step is reference/product proposal parity plus training scale/config/top-1
quality, not LAN orchestration: use a confirmed HF-scale bfloat16/CUDA job or
change the training recipe only after package-backed serving and reference eval
agree on the same prompt surface.

The LR diagnostic shows the direction: `1e-4` improves top-1 quality, but this
is still not enough. Do not spend an HF-scale job until the reference evaluator
and product smoke agree on proposal/verification behavior for identical prompts.

2026-06-18 identical-prompt/product-distribution checkpoint: the Qwen3-8B S2
`23,36` sidecar was evaluated on the same 9 reference prompts through the
package-backed product request path. The unfine-tuned LR `1e-4` checkpoint
matched baseline/SPD content on all `9 / 9` prompts but accepted only
`8 / 63` proposals
(`/tmp/spd-qwen3-8b-identical-prompts-product-nt9.json`). The per-prompt
reference/product comparator showed mostly aligned target streams (`60 / 63`
target-token positions, `7 / 9` prompts exact) but poor proposal parity
(`10 / 63` proposal-token matches, no prompt with full proposal parity), so the
gap was not just prompt rendering. `skippy-bench spd-live-tap-parity` now
accepts a tokenized prompt JSONL and exported a 72-row product activation
corpus from those same 9 prompts:
`/tmp/spd-qwen3-8b-product-corpus-nt9.safetensors`. HF teacher augmentation
over `Qwen/Qwen3-8B` produced
`/tmp/spd-qwen3-8b-product-teacher-nt9.safetensors` with draft-width BF16
logits for all 72 samples; 71 labels were inside the draft-vocab scope. This is
KL-compatible product-tap data, but still uses HF teacher logits rather than
native Q4_K_M verifier logits.

Two local product fine-tune bridges proved the product-path proposal source can
move. A 2-epoch, batch-8 BF16 pass wrote
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8/speculation_head_final.pt`, improving
product-row argmax accuracy from `0.0` to `0.25` and local package-backed
acceptance from `8 / 63` to `25 / 63`. A stronger 10-epoch, batch-8, LR
`2e-5` debug pass wrote
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/speculation_head_final.pt`,
reached product-row argmax accuracy `0.875`, exported finite BF16 serving
weights with SHA
`3b87a779034fd2974da76e3c368ee0000b5bbec5a735f3c7a7d3fec65c3d8866`, and
passed Rust fixture parity. Local package-backed serving on the exact 9 prompts
matched content on all `9 / 9`, accepted `42 / 63` proposals without the
rolling executor, and accepted `44 / 59` with the rolling executor. The rolling
report
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/openai-product-nt9-rolling.json`
recorded `0` tap failures, `54` rolling launches, `9` no-proposal launch
misses, `max_in_flight=2`, `39` oldest accepts, `15` oldest rejections, and `15`
drained younger replies. The paper-style two-stage estimate is now positive
(`44` saved token round trips, `15` unsaved, `1.49x` versus serial split), but
the current same-machine implementation remains slower (`0.156x` decode)
because proposal/head work and stage waits are not hidden.

The same stronger debug sidecar has now run through a real one-worker LAN split
with one physical stage per machine. A no-launch preflight validated the
package-backed `23,36` split and tap allowlist in
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/openai-lan-preflight.json`.
The remote worker cache initially lacked downstream package parts; copying only
the required `23..35` layer parts plus `shared/output.gguf` fixed that
materialization gap. The full 9-prompt paired LAN rolling report at
`/tmp/spd-qwen3-8b-product-finetune-nt9-b8-e10-lr2e5/openai-lan-nt9-rolling.json`
matched content on all `9 / 9`, accepted `44 / 59` proposals, committed `39`
optimistic tokens, recorded `0` tap failures, and had `0` rolling launch
misses with `max_in_flight=2`. Mean baseline decode was `554.0ms`; mean SPD
decode was `1702.7ms` (`0.325x`). The paper-style round-trip estimate remained
positive (`1.49x`, `44` saved / `15` unsaved), so the current negative wall
speed is now concrete overhead evidence: mean sidecar head time was `69.5ms`,
normal downstream wait averaged `150.7ms`, optimistic downstream wait averaged
`115.7ms`, and chained hidden waits averaged `108.9ms`. Treat this as real
two-node correctness plus overhead decomposition for an overfit debug head, not
a final generalizing sidecar.

2026-06-18 product-distribution held-out checkpoint: the next non-overfit
bridge used
`/tmp/spd-qwen3-8b-product-prompts-paper3-train8-heldout4`, with `24` train
prompts and `12` held-out prompts drawn evenly from MT-Bench, GSM8K, and
HumanEval. Product row capture against the exact
`meshllm/Qwen3-8B-Q4_K_M-layers` package and `--splits 23 --layer-end 36`
wrote `192` train rows to
`/tmp/spd-qwen3-8b-product-corpus-paper3-train8` and `96` held-out rows to
`/tmp/spd-qwen3-8b-product-corpus-paper3-heldout4`. HF teacher augmentation
over `Qwen/Qwen3-8B` wrote
`/tmp/spd-qwen3-8b-product-teacher-paper3-train8.safetensors` and
`/tmp/spd-qwen3-8b-product-teacher-paper3-heldout4.safetensors` with
draft-width BF16 logits; this remains HF-teacher KL data aligned to product
tap inputs, not native Q4_K_M verifier logits. A 5-epoch BF16 MPS fine-tune
from the LR `1e-4` S2 `23,36` checkpoint wrote
`/tmp/spd-qwen3-8b-product-finetune-paper3-train8-e5-lr2e5/speculation_head_final.pt`,
reached train argmax accuracy `0.875`, exported finite BF16 serving weights
with SHA
`e1928134128ffb0f05bdb20a6d635ecba2d890b09c7257f98304e27a2fe80130`, and passed
Rust fixture parity.

The held-out live-tap report
`/tmp/spd-qwen3-8b-product-finetune-paper3-train8-e5-lr2e5/live-tap-heldout4.json`
matched ordinary greedy output on all `12` prompts and accepted `39 / 96`
top-1 proposals (`40.6%`). The all-local rolling OpenAI request-path report
`/tmp/spd-qwen3-8b-product-finetune-paper3-train8-e5-lr2e5/openai-heldout4-rolling.json`
matched baseline/SPD content on all `12 / 12` held-out prompts, accepted
`30 / 82` proposals (`36.6%`), committed `29` optimistic tokens, and recorded
`0` tap return failures, `0` tap record failures, and `0` ignored taps. The
request-path run saved `30` candidate token round trips but left `52` unsaved,
so the idealized two-stage `paper_pipeline_estimate` is still below break-even
at `0.73x`. Observed all-local decode is `0.13x` of baseline, with mean probe
head time `63.5ms`, normal downstream wait `111.5ms`, optimistic downstream
wait `94.3ms`, and chained hidden wait `39.8ms`. This is the first honest
held-out generalization signal for product taps, but not a speed proof; do not
run a two-node speed claim with this head until held-out package-backed serving
clears `paper_pipeline_estimate > 1.0`.

- 2026-06-17 the first model-backed 24-token rolling-executor smoke after the
  replay reset cleanup is
  `/private/tmp/spd-rolling-executor-real-local-smoke24-4.json`. It restores
  exact baseline/SPD content and keeps tap transport healthy (`0` tap record
  failures, `0` tap return failures), but it does **not** pass the paper gate:
  the pretrained Qwen3.5 sidecar accepted `20 / 24` proposals, the rolling
  executor observed `1` oldest rejection and drained `3` younger replies,
  rolling trace replay still has `9` missing proposals, and debug local SPD
  decode was `49911.5ms` versus `529.7ms` baseline. `skippy-bench
  spd-openai-check --max-spd-decode-ms 1652.6` correctly fails this report.
  The concrete rejection is target position `38`, where the rolling sidecar
  proposed token `198` and the target produced `5423`. This proves the reset
  path is content-correct after rejection, but the request path is still not the
  paper/reference executor: it can recover from a miss, but it is not yet a
  continuously full oldest-commit pipeline with clean replay and speedup.
- `skippy-runtime::spd::SpdRollingScheduler` now codifies the paper/reference
  rolling scheduler state transitions in Rust: newest-first in-flight entries,
  evicted-prefix speculation rows on acceptance, oldest-entry verification
  after fill, and reset-to-corrected-token behavior on rejection.
  `SpdRollingTraceReplay` replays observed target/proposal traces through that
  same runtime primitive and reports the final target-verified prefix.
  `SpdRollingObserver` is the live token/position observer used by
  `skippy-server` diagnostics. It now exposes `take_verified_delta()` so inline
  probes can report newly target-verified token spans; the latest fixed smoke
  emitted deltas `[71093]`, `[12305]`, `[198]`, `[727]`, `[884]`, and
  `[2784]`.
  It also exposes `speculation_rows()` with `row_positions` and `row_i_stages`
  so serving can assemble the reference row roles instead of using only a
  sliding context window. `SpdRollingObserver::draft_plan()` now clones the
  verified scheduler for proposal generation. The server advances that draft
  plan locally while it proposes the next window, so later proposals use the
  paper-shaped rolling rows without mutating the live observer until the target
  verifier accepts or rejects them. The runtime scheduler reports nominal paper
  layout roles; the serving proposal path resolves them through the manifest.
  For the Qwen3.5-4B artifact, `trained_with_use_deepest=true`, and fixture
  parity confirms the exported rows use inference roles `[4, 4, 4, 0]` for
  positions `[9, 10, 11, 12]`. The latest fixed smoke populated rows from the
  first probe (`[23, 24, 25, 26]` / `[4, 4, 4, 0]`) and kept them moving after
  accepted evictions (`[30, 30, 31, 32, 33]` / `[4, 3, 3, 3, 0]` at step `7`).
  A bounded primary-`VerifySpan` smoke with `max_tokens=8`,
  `--optimistic-decode false`, and replay fallback preserved exact greedy
  output, ran two SPD windows, accepted `8 / 8` proposals, inserted `7` rolling
  drafts, verified `5` filled rolling windows, and reported `0` missing or
  out-of-order proposals. `skippy-bench` now carries primary-verify-only cases
  into the aggregate rolling summary from `cases[].decode.rolling`: a rerun at
  `/private/tmp/spd-openai-primary-rolling-report-smoke/report.json` reports
  `cases_replayed=0`, `live_cases_observed=1`, `inserted_drafts=7`,
  `verified_windows=5`, and `0` missing/out-of-order proposals. It remained
  deliberately slow proof code: baseline `636ms` wall / `204ms` decode versus
  SPD `3817ms` wall / `3228ms` decode, with `2643ms` spent proposing. The
  paper-style estimate from the same `1.0` accept rate is `4.0x` versus serial
  split, or about `51ms` decode at that run's baseline stage cost, so current
  serving is about `63x` slower than the schedule it is trying to realize. A
  proposal-breakdown rerun at
  `/private/tmp/spd-openai-proposal-breakdown-smoke/report.json` preserved exact
  output and accepted `8 / 8`, but all `8` proposals came from replay fallback:
  `inline_tap_hits=0`, `replay_fallbacks=8`, `tap_collect_ms=2205ms`,
  `cur_in_ms=51ms`, and `forward_ms=509ms` inside `2766ms` of total proposal
  time. This confirms the next speed-path requirement is in-flight/direct-return
  rolling hidden states, not more local replay tuning.
- A no-replay optimistic inline-probe breakdown rerun at
  `/private/tmp/spd-openai-overlap-probe-smoke/report.json` preserved exact
  output, proposed and accepted `8 / 8`, committed `4 / 4` optimistic decodes,
  and showed every measured `pre_target_reply` and `optimistic_commit` probe
  using `tap_source=inline`. The accepted optimistic-commit probes now run
  during the in-flight optimistic `VerifySpan` reply wait, with
  `trigger_hf_index=31` and about `0.001ms` wait-after-probe. The final decode
  event carries the same source evidence: `inline_tap_hits=8`,
  `replay_fallbacks=0`, `tap_collect_ms=2.39ms`, `cur_in_ms=115.2ms`, and
  `forward_ms=528.5ms`. This proves optimistic probes can consume direct-return
  taps without replay fallback and can overlap target waits; it is still not a
  speedup (`202ms` baseline decode versus `2800ms` SPD decode) because the
  current path remains a bounded one-token proof instead of the full overlapped
  rolling schedule.
- A no-thinking chainability rerun at
  `/private/tmp/spd-openai-chainability-summary-smoke/report.json` preserved
  exact output, accepted `8 / 8`, committed `4 / 4` optimistic decodes, and
  split `optimistic_commit` probes out in `summary.pipeline_gap`: `4 / 4`
  commit probes proposed, `4 / 4` were accepted, and their mean
  wait-after-probe was about `0.001ms`. That is the clearest current evidence
  that the next speed-path blocker is safe chained/rolling target execution,
  not sidecar proposal availability on this prompt. The run was still slower:
  `201ms` baseline decode versus `2873ms` SPD decode.
- A follow-up no-thinking chained optimistic execution smoke at
  `/private/tmp/spd-openai-chained-optimistic-smoke8/report.json` preserved
  exact output and turned accepted optimistic-commit proposals into real target
  work. It proposed `6` SPD tokens, accepted `4`, rejected `2`, committed
  `4 / 4` optimistic target tokens, and committed `2 / 2` through a bounded
  one-step chained optimistic `VerifySpan` while the previous optimistic
  verifier was still in flight. Tap return failures, tap record failures, and
  ignored taps were all `0`, and the report preserves `chain=true` on the two
  chained `DecodeEmbdOptimistic` token events. Primary `VerifySpan` commits now
  also emit token events, so replay sees the full target stream
  `[71093, 12305, 198, 727, 884, 2784, 11, 292]` and verifies it matches.
  Baseline decode was `203.1ms`; SPD decode was `2820.5ms`, so this is
  execution-structure evidence, not a speedup. The rolling trace replay is now
  conservative when target token events are missing: it reports
  missing/out-of-order proposal positions instead of zero-filling an unobserved
  verified prefix.
  A recursive in-flight chain experiment was not retained: without
  per-message direct-return correlation, launching another `PredictedTokens`
  verifier from inside a previous chain's return path could consume or wait on
  the wrong same-kind reply.
- Direct-return prediction replies now have an opt-in origin header on the
  direct-return stream, and stage 0 buffers unmatched final replies until the
  requested origin arrives. A current release smoke at
  `/private/tmp/spd-openai-origin-aware-smoke2/report.json` preserved exact
  output, proposed `6` SPD tokens, accepted `4`, rejected `2`, committed
  `4 / 4` optimistic target tokens, committed `2 / 2` chained optimistic target
  tokens, and recorded `0` tap return failures, `0` tap record failures, and
  `0` ignored taps. Baseline decode was `239.1ms`; SPD decode was `2692.0ms`.
  This removes the reply-ownership blocker for several same-kind in-flight
  verifiers, but it is still a bounded one-step chain. The next full-rolling
  step is a scheduler/executor that owns the in-flight entries, launch order,
  and rollback/restore contract.
- Checkpoint ownership is now generation-addressed. Speculative `VerifySpan`
  messages carry a nonzero `checkpoint_generation` derived from target
  position, direct-return origins include that generation, and embedded plus
  downstream stages checkpoint/restore by session plus generation. The current
  release smoke at `/private/tmp/spd-openai-checkpoint-gen-smoke1/report.json`
  preserved exact output, proposed `6` SPD tokens, accepted `4`, rejected `2`,
  committed `4 / 4` optimistic target tokens, committed `2 / 2` chained
  optimistic target tokens, and recorded `0` tap return failures, `0` tap record
  failures, and `0` ignored taps. Baseline decode was `203.7ms`; SPD decode was
  `2796.9ms`. This removes the checkpoint-overwrite blocker for multiple
  in-flight entries, but it is still not the paper's continuously full rolling
  pipeline.
- The request path now uses an origin-matched rolling queue rather than one
  hardcoded chained verifier. Accepted optimistic entries advance the sidecar
  context, wait for their own direct-return reply by origin, and may launch one
  deeper verifier from returned taps while capped by the logical SPD stage
  count. The current release smoke at
  `/private/tmp/spd-openai-hidden-wait-smoke1/report.json` preserved exact
  output, reached `max_optimistic_chain_depth=2`, proposed `8` SPD tokens,
  accepted `6`, rejected `2`, committed `5 / 7` optimistic target tokens,
  committed `2 / 4` chained optimistic target tokens, and recorded `0` tap
  return failures, `0` tap record failures, and `0` ignored taps. Baseline
  decode was `202.3ms`; SPD decode was `9275.4ms`. The depth-2 verifier entries
  launched and restored correctly, but both rejected on this prompt. Derived
  hidden-wait from the same run was about `6.72s` total, almost all from chained
  rows; the two rejected depth-2 rows hid about `1.69s` and `5.03s` behind older
  verifier work. The report exposes this as `hidden_wait_ms` on
  `optimistic_decodes[]` and as hidden-wait summaries under
  `summary.pipeline_gap`. This proves the queue is overlapping latency, but
  proposal quality and rollback cost still prevent a speedup.
- A stage-role audit showed that fixture `row_i_stages` are tap/projection
  roles, not the Qwen spec head's internal fixed-memory roles. Passing
  `row_i_stages=[4,4,4,0]` as fixed-stage ids made parity much worse
  (`/private/tmp/spd-fixture-parity-topk4-stageids.json`: forward final-hidden
  max diff `9.75`, spec-query max diff `28.4375`). The corrected native
  contract leaves fixed-stage ids unset for live proposal windows so Qwen uses
  the reference `_infer_stage_ids(q_len)` schedule, while cache prefill can
  explicitly mark completed prefix rows as deepest-stage rows. The corrected
  fixture run at
  `/private/tmp/spd-fixture-parity-topk4-fixedstage-default.json` restored
  matching top-k and the prior small drift (`0.125` forward final-hidden max
  diff, `0.0625` cached final-hidden/logit max diff). A fresh OpenAI smoke at
  `/private/tmp/spd-openai-fixedstage-default-smoke1/report.json` preserved
  exact output, proposed `8`, accepted `6`, rejected `2`, reached depth `2`,
  and recorded no tap failures or ignored taps, but remained much slower
  (`203.8ms` baseline decode versus `13912.7ms` SPD decode).
- Proposal-row telemetry then found a real serving-context bug behind the
  repeated depth-2 proposals. The diagnostic run at
  `/private/tmp/spd-openai-proposal-rows-smoke1/report.json` showed step 2
  assembled its proposal from stale rows `[23,24,25,26]` with
  `next_draft_position=27` even though the accepted optimistic token had moved
  the target position to 28, so the sidecar proposed the previous token again.
  The server now observes accepted optimistic-commit probes into
  `SpdRollingObserver` immediately before a deeper chained verifier requests
  rows. The follow-up release smoke at
  `/private/tmp/spd-openai-rolling-observe-smoke1/report.json` preserved exact
  output, proposed `5`, accepted `5`, rejected `0`, committed `3 / 3`
  optimistic decodes and `2 / 2` chained optimistic decodes, reached depth `2`,
  and recorded no tap failures. Step 2 now proposes token `198` from rows
  `[24,25,26,27]` with `next_draft_position=28`. The stale-row repeat is fixed,
  but SPD is still slower (`222.2ms` baseline decode versus `2667.6ms` SPD
  decode) and rolling replay still reports `3` missing proposal positions
  starting at position 30 after the chain boundary. A follow-up
  miss-diagnostic smoke at
  `/private/tmp/spd-openai-tap-position-diagnostics-smoke1/report.json`
  preserved exact output, proposed `5`, accepted all `5`, committed `3 / 3`
  optimistic decodes and `2 / 2` chained optimistic decodes, and recorded no tap
  return or record failures. It made the post-target probe empties concrete:
  probe steps `4`, `5`, and `6` report `missing_inline_taps` for position `28`;
  h0 is no longer included in those inline requirements, and tap-position
  telemetry corrected the first diagnosis: the non-h0 rows had already been
  recorded before the probes, then a shorter accepted-prefix commit pruned them
  because SPD's sidecar context had advanced ahead of emitted tokens. The prefix-ack
  fix treats those shorter prefix-compatible accepted-context updates as
  acknowledgements instead of resets. The follow-up smoke at
  `/private/tmp/spd-openai-prefix-ack-smoke1/report.json` preserved exact
  output, proposed and accepted `8 / 8`, committed `6 / 6` optimistic verifier
  results including `4 / 4` chained results, reported `0` post-target empty
  probes, and kept rolling replay at `0` missing/out-of-order proposals. It is
  still slower (`219.7ms` baseline decode versus `2795.3ms` SPD decode), so the
  remaining task is performance and full rolling execution, not sidecar
  topology, h0, fixed-stage, or head-quality.
- A thinking-mode rejection rerun at
  `/private/tmp/spd-openai-overlap-rejection-clean-smoke/report.json` preserved
  exact output while accepting only `1 / 8` proposals, rejecting `7 / 8`, and
  committing `1 / 3` optimistic decodes. Rollback stayed correct with no tap
  failures or ignored taps, but the final live `decode.rolling` snapshot still
  reported one out-of-order proposal at the frontier. The live observer now
  keeps early proposals pending and promotes them after accepted context
  catches up. The follow-up rejection smoke at
  `/private/tmp/spd-openai-pending-promote-rejection-smoke/report.json`
  preserved exact output, proposed `8` tokens, accepted `3`, rejected `5`,
  committed `2 / 4` optimistic decodes, and ended with
  `decode.rolling.out_of_order_proposals=0`. It remains slower
  (`207ms` baseline decode versus `3475ms` SPD decode), so mixed
  inline/primary reporting is now cleaner but full rolling scheduling remains
  open.
  The request path now derives tap returns from the topology
  (`[8, 10, 16, 20, 24, 31]` for this head after stripping h0) and waits only
  for row-specific taps while assembling each proposal.
  `skippy-bench` uses the same observer path through the runtime-owned replay
  for reports; serving still needs to use it for actual stage execution.
- `skippy-server --openai-spd-optimistic-decode` can use an accepted inline SPD
  proposal to start and commit one optimistic target decode in the real
  seven-stage request path. The current proof is gated to deterministic
  sampling and uses a checkpointing one-token `VerifySpan` plus restore for
  rollback.
- Mesh-native Skippy config now has an experimental default-off
  `[speculative] mode = "spd"` path that passes manifest, fixture, model,
  window, top-k, GPU-layer, replay-fallback, and optimistic-decode settings
  into embedded stage-0 OpenAI serving. The resolver rejects mixed draft/SPD
  sources and keeps SPD staged-only.
- `llama-spec-bench` can run a real target/draft speculative-decoding
  diagnostic after opening the target with enough execution lanes for verifier
  and projection sessions.
- 2026-06-17 native rolling SPD now has an SPD-owned shadow/snapshot KV path
  for the real OpenAI request path. The Rust protocol added `CopySession` and
  `DropSession` controls, the llama.cpp stage ABI added
  `skippy_session_copy_prefix`, and rolling launches now refuse to seed a fresh
  shadow unless canonical KV is materialized at exactly the requested prefix.
  This matters for recurrent/hybrid Qwen stages: copying an older or future
  prefix from canonical is invalid, and earlier local smokes exposed both
  failure modes.
- The current rejection-tolerant model-backed local smoke at
  `/private/tmp/spd-rolling-shadow-sky8.json` preserves exact baseline/SPD
  content for the Qwen3.5-4B S4/L4 seven-stage split, reaches
  `max_in_flight=4`, accepts `11 / 16` SPD proposals, observes one oldest
  rejection, drains three younger verifier replies, and records `0` tap return
  failures, `0` tap record failures, and `0` ignored taps. The explicit gate
  passes with
  `skippy-bench spd-openai-check --min-accepted 8 --max-rejected-oldest 1 --max-drained-younger 3 --max-rolling-trace-missing-proposals 9`.
  This is a correctness/recovery checkpoint, not a speed claim: same-machine
  CPU debug SPD decode was `53405.2ms` versus `452.7ms` baseline, and the next
  performance proof still needs real stage placement across distinct hardware.
- 2026-06-17 KV/rolling follow-up: release `spd-openai-smoke` at
  `/private/tmp/spd-local-rolling-kv-counters-smoke24.json` preserved exact
  baseline/SPD content and passed `spd-openai-check` with `19 / 23` accepted SPD
  proposals, `max_in_flight=4`, one oldest rejection, and three drained younger
  verifier replies. The new rolling launch-miss breakdown is actionable:
  `no_proposal=49`, `shadow_missing_view=40`, `in_flight_full=14`,
  `shadow_not_seedable=2`, `no_rows=0`, with `2` successful exact canonical
  shadow reseeds. A trial that copied an older canonical prefix into a shadow
  lane failed on this Qwen path with
  `recurrent session copy requires source at the copied prefix`, so older-prefix
  copy is not a portable fix. The next executor change should retain or seed
  shadow snapshots at the paper scheduler positions around rejection/recovery,
  then verify/evict only the oldest completed entry.
- 2026-06-17 idle rolling-executor catch-up now realigns an empty executor to
  the accepted canonical context after rejection/drain or other idle gaps,
  instead of continuing to request stale shadow KV views. The release
  `spd-openai-smoke` at `/private/tmp/spd-local-idle-catchup-smoke24.json`
  preserved exact baseline/SPD content and passed
  `spd-openai-check --require-rolling-executor true --min-accepted 21
  --max-rejected-oldest 1 --max-drained-younger 3
  --max-rolling-trace-missing-proposals 3`. Compared with the previous KV
  report, shadow-missing launch attempts fell from `40` to `5`, rolling
  launches rose from `15` to `22`, accepted SPD proposals rose from `19 / 23`
  to `21 / 22`, and replay missing proposals fell from `9` to `3`. This is
  still below the paper gate: there is one oldest rejection, three drained
  younger replies, three replay-missing proposals, and local SPD decode is
  still slower than the baseline because the request path waits on chained
  verifier replies instead of running the full zero-bubble rolling executor.
- 2026-06-17 hybrid checkpoint restore now has a native Skippy/llama.cpp
  patch-queue fix. A pre-patch non-rolling CPU smoke failed with
  `failed to trim hybrid memory suffix`: `skippy_restore_session_checkpoint`
  was trimming the hybrid memory wrapper before restoring the saved recurrent
  checkpoint lane, which asked Qwen's recurrent state to roll back a speculative
  suffix outside its rollback window. Patch `0094` keeps explicit
  `skippy_trim_session` semantics unchanged, but checkpoint restore now trims
  only the attention KV suffix for hybrid/hybrid-ISWA memory and then restores
  recurrent state from the checkpoint. The post-patch non-rolling CPU smoke at
  `/private/tmp/spd-local-nonrolling-cpu-smoke24-v2.json` completes and accepts
  `20 / 24` proposals; the comparable local rolling smoke at
  `/private/tmp/spd-local-rolling-cpu-smoke24-v2.json` preserves exact content,
  reaches `max_in_flight=4`, records `0` tap failures, and accepts `21 / 22`;
  the one-worker LAN split rerun at `/private/tmp/spd-lan-cpu-spd24-v2.json`
  has the same content-correct `21 / 22` acceptance shape with `0` tap failures.
  The surviving oldest rejection is the same target-position `38` mismatch seen
  in non-rolling verification, so it is sidecar/top-1 quality for this prompt,
  not rolling executor, LAN transport, or Skippy KV corruption. This is still
  below the paper gate: one oldest rejection drains three younger replies, replay
  still misses three proposals, and LAN CPU SPD decode is `14392.6ms` versus
  `4502.5ms` baseline (`0.313x`). The strict `spd-openai-check` failure is
  exactly those four gates (`21 < 24` accepted, one oldest rejection, three
  drained younger replies, three missing replay proposals); the relaxed
  correctness gate passed at
  `/private/tmp/spd-lan-cpu-spd24-v2-check-relaxed.json`.
- 2026-06-18 the same request-path rolling executor has a clean 24-token LAN
  split case and a rejection/reset sweep on one worker. The paired clean-count
  report at `/private/tmp/spd-lan-count-paired.json` matches baseline/SPD
  content, accepts `23 / 23` SPD proposals, reaches `max_in_flight=4`, records
  `21` oldest accepts, `0` oldest rejections, `0` drained younger replies, `0`
  tap return failures, `0` tap record failures, and `0` ignored taps. Its
  focused gate passes at `/private/tmp/spd-lan-count-paired-check.json`, with
  one allowed terminal replay miss at the `max_tokens` boundary and a verified
  24-token prefix matching target. The three-prompt SPD-only LAN sweep at
  `/private/tmp/spd-lan-mini-sweep.json` exercises both clean acceptance and
  rejection recovery over the real split path: aggregate `57 / 59` accepted,
  `2` rejected, `max_in_flight=4`, one oldest rejection, three younger drains,
  `0` tap failures, `0` ignored taps, `0` out-of-order replay proposals, and a
  verified prefix matching target. The SPD-only checker must run with
  `--require-content-match false` because there is no baseline half; the
  bounded gate passed at `/private/tmp/spd-lan-mini-sweep-check.json`. These
  runs strengthen the KV/shadow-session claim: accepted speculative taps are
  promoted correctly, rejection drains reset without corrupting Skippy state,
  and the surviving negative speed result is still scheduling/resource
  placement, not evidence of KV corruption. The paired LAN speed signal remains
  negative: baseline decode `4426.1ms`, SPD decode `13458.4ms` (`0.329x`).

## What Does Not Work Yet

- The replay fallback collects taps by replaying the current context through
  local `StageModel` slices. It is a correctness bridge into live serving, not
  the optimized inline hidden-tap transport needed for real speed.
- The no-replay request path can now start and commit optimistic target decodes
  and can reach high acceptance on the bounded proof prompt, but the current
  local CPU path is still slower than the normal split baseline. The native
  sidecar cache is faithful to the Python `spec_past_kv` path on the cached
  fixture (`cache_prefix_len=20`, exact Rust/Python cached top-k ids, full
  cached-logit max diff `0.0625`), and the fixed request-path smoke accepted
  `8 / 8` proposals with exact greedy output. The next bottleneck is therefore
  concrete: proposal probes still cost tens of milliseconds, normal downstream
  waits dominate wall time, and the serving path is still a bounded
  one-token/optimistic proof rather than the paper's fully overlapped rolling
  schedule.
- The binary stage transport can already send prediction replies on a direct
  return stream, so stage 0 does not fundamentally need to wait for each final
  prediction before writing the next stage message. The missing serving piece is
  a rolling executor that keeps multiple speculative stage messages in flight,
  verifies the oldest completed entry, and rolls back to the rejected target
  position. A trim-to-position rollback shortcut was tested for SPD optimistic
  rejection and was not retained: an `enable_thinking=true` rejection smoke
  changed deterministic output casing (`Thinking Process` versus
  `Thinking process`). Current serving therefore keeps checkpoint/restore for
  exactness. The restored checkpoint path reran the same thinking-mode smoke
  with exact output, `1 / 8` accepted proposals, `6` optimistic requests, `5`
  rejected optimistic decodes, `1` committed optimistic token, `0` tap failures,
  about `0.038ms` total optimistic checkpoint time, and about `20.8ms` total
  restore time. Generation-addressed checkpoints now provide the rollback
  primitive; rolling execution still needs a scheduler that keeps several
  entries in flight and restores the matching generation only when the oldest
  completed verifier rejects. The smoke benchmark now fails by default on paired baseline/SPD
  content mismatch after writing its JSON report, so this kind of rollback
  regression is not counted as a passing smoke; `--allow-content-mismatch` is
  reserved for exploratory sweeps.
- Optimistic target messages now request SPD tap returns whenever optimistic SPD
  decode starts, including ungated runs. The accepted-context lifecycle filter
  drops stale future taps, but a production path should still buffer speculative
  taps and promote them only after acceptance so rejected speculative work
  cannot pollute the inline tap cache or delay rollback.
- Primary SPD `VerifySpan` windows now use that same lifecycle: every input
  position in the span is marked pending before the target stages run, then
  fully accepted spans promote those rows and rejected spans reset to the
  verified prefix. This prevents multi-token verified SPD windows from dropping
  returned taps as stale future rows before the accept/reject decision is known.
  `skippy-bench spd-openai-smoke` exposes `--spd-replay-fallback`, which passes
  `--openai-spd-replay-fallback` through to the embedded stage-0 server; pair
  it with `--optimistic-decode false` to force primary SPD `VerifySpan` windows
  for correctness evidence. A bounded `max_tokens=4` release smoke
  (`/private/tmp/spd-openai-primary-rolling-smoke/report.json`) preserved exact
  output, ran one primary SPD window, accepted `4 / 4` proposals, recorded `0`
  ignored taps, and had no optimistic commits. The primary verifier now feeds
  the committed target span into the rolling observer: `inserted_drafts=3`,
  `verified_windows=1`, `accepted_windows=1`, and no missing or out-of-order
  proposals. This is deliberately slow replay-fallback evidence: baseline
  `502ms` wall / `125ms` decode versus SPD `1837ms` wall / `1297ms` decode. A
  separate bounded `max_tokens=4`
  release smoke
  (`/private/tmp/spd-openai-pending-verify-taps-smoke/report.json`) preserved
  exact output, accepted `4 / 4` SPD proposals, committed `2 / 2` optimistic
  target decodes, recorded `0` ignored taps, and kept rolling replay ordered.
  It was still slower: baseline `541ms` wall / `146ms` decode versus SPD
  `2047ms` wall / `1453ms` decode.
- The live request-path proof is bounded; it still needs a larger acceptance
  and latency sweep.
- The `.pt` checkpoint is a proof/training artifact. Export it to
  `spd-head.safetensors` before Rust-side serving work.
- SPD sidecars are tied to the base model/tokenizer and logical tap topology
  they were trained/exported for: number of logical SPD stages, selected hidden
  tap layers, projection layout, hidden size, draft vocab, and spec-layer
  count. Physical Skippy placement can differ only if it exposes the same
  logical taps; otherwise a matching hidden-tap ABI or a topology-specific
  sidecar is required. The Qwen head also has internal fixed-memory stage roles;
  those are inferred by the spec module for the live proposal window and should
  not be confused with tap `row_i_stages`.
- Real distributed speedup is still unproven. The earlier Python/reference eval
  and Skippy latency model are useful because they use real acceptance traces,
  but they remain theoretical/simulated speed evidence. Current local
  request-path smokes are correctness and scheduler-shape evidence; running
  all stages on one machine/device is not a fair SPD speed oracle because it
  adds true concurrent stage work on shared resources.
- The latest repeated CPU run shows the overhead is not primarily a missing
  reference sidecar cache path. Cache reuse and cached logits have parity
  evidence, and the live request path reported cache hits rather than misses.
  The remaining gap is native serving scheduling: direct-return tap plumbing,
  downstream wait, hidden verifier wait, and the missing continuously full
  rolling executor.
- The matching Metal repeat narrows those waits substantially but still loses
  to baseline, which makes distinct-device placement the next required speed
  experiment.

## Open Training Data

The local Qwen3-0.6B proof uses:

- dataset: `HuggingFaceH4/ultrachat_200k`
- split: `train_sft`
- rows: first `1024` rows for the recorded local proof

The reference SPD repository lists the intended training corpus family as:

- UltraChat-200k
- ShareGPT
- SmolTalk
- SmolTalk-Chinese

MT-Bench, HumanEval, and GSM8K prompts are used here only for evaluation.

## Reproduce Qwen3-0.6B Training

This is the smallest useful proof that the training path and artifact shape
work. It trains a real head from open data.

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

Use `--device cuda` on a GPU host. The runner also supports HF Jobs, but that is
only a convenience wrapper; the proof is ordinary Python plus open data.

Recorded local result:

| Model | Head | Eval draft top-k | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B` | locally trained, 4 spec layers | 4 | 1536 | 326 / 1536 | 0.5628 | 1.1257 | 12.67% |

This proves the training/export path, but it is not the high-gain target.

## Reproduce Qwen3.5-4B Pretrained Head Eval

This is the strongest current model-quality signal. It uses an author-published
SPD head and evaluates it locally against the reference verifier.

```bash
python3 evals/spd/hf_train_eval_qwen06.py \
  --work-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4 \
  --model-name Qwen/Qwen3.5-4B \
  --spec-head-repo yuyijiong/speculative_pipeline_decoding \
  --spec-head-file Qwen3.5-4B_s4_l4.pt \
  --manifest-base-model-path Qwen/Qwen3.5-4B \
  --skip-train \
  --device mps \
  --eval-rows-per-set 8 \
  --max-new-tokens 64 \
  --draft-top-k 4 \
  --upload-repo ''
```

Use `--device cuda` on a GPU host. The first run downloads the base model and
the SPD head.

Recorded local result:

| Model | Head | Eval draft top-k | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-4B` | pretrained, 4 stages / 4 spec layers | 4 | 1536 | 1230 / 1536 | 0.6176 | 2.4704 | 163.39% |

The accepted-flags count and aggregate acceptance use different denominators in
the reference output. `1230 / 1536` is the draft-flag count; `0.6176` is the
reference aggregate acceptance metric used for equivalent accept length.

Per-dataset theoretical gains from the same run:

| Dataset | Acceptance | Equivalent accept length | Theoretical gain |
| --- | ---: | ---: | ---: |
| MT-Bench | 0.4918 | 1.9673 | 98.42% |
| HumanEval | 0.8797 | 3.5189 | 254.18% |
| GSM8K | 0.5926 | 2.3704 | 137.58% |

## Latency Simulation From Real Traces

`simulate_latency.py` consumes the raw `eval/raw/*per_sample.jsonl` file emitted
by the reference evaluator. It does not invent acceptance; it uses the real
`new_tokens`, `decode_loop_steps`, and accepted-flag counters from the run.

```bash
python3 evals/spd/simulate_latency.py \
  --raw /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/eval/raw/pipeline_eval__train__speculation_head_final__nt24__per_sample.jsonl \
  --stage-ms 4,4,4,4 \
  --hop-ms 0,1,5,10,25
```

Recorded Qwen3.5-4B trace with a four-stage `4ms,4ms,4ms,4ms` model:

| Hop ms | Serial split tok/s | SPD pipeline tok/s | SPD vs serial split | Paper-like gain | P50 serial ms | P50 SPD ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 62.50 | 617.61 | 9.882x | 2.470x | 1024.00 | 106.50 |
| 1 | 52.63 | 494.09 | 9.388x | 2.470x | 1216.00 | 133.12 |
| 5 | 32.26 | 274.49 | 8.509x | 2.470x | 1984.00 | 239.62 |
| 10 | 21.74 | 176.46 | 8.117x | 2.470x | 2944.00 | 372.75 |
| 25 | 10.99 | 85.19 | 7.752x | 2.470x | 5824.00 | 772.12 |

The `paper-like gain` column is based on the SPD trace alone. The `SPD vs serial
split` column models a Skippy-specific comparison where ordinary split serving
must traverse every stage/hop for each generated token before the next target
token is known.

The simulator's aggregate-cycle formula reports the same equivalent accept
length as `2.470x` (`+147.04%`). The reference eval summary separately reports
a token-weighted theoretical gain of `163.39%`.

## Export the Serving Checkpoint

After training or downloading a reference SPD head, export the PyTorch
checkpoint to a Rust-readable serving artifact:

```bash
python3 evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

The exporter writes `spd-head.safetensors` next to the manifest and adds an
optional `serving_checkpoint` section to `skippy-spd-head.json`. The original
`.pt` checkpoint remains referenced for provenance.

For the pretrained `Qwen/Qwen3.5-4B` S4/L4 head, the tap-aligned Skippy proof
split is:

```bash
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir .artifacts/spd/qwen35-4b-gguf/
skippy-model-package plan .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --splits 8,10,16,20,24,31
skippy-model-package write-stages .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --splits 8,10,16,20,24,31 \
  --out-dir /tmp/qwen35-spd-tap-slices/
skippy-model-package validate .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  /tmp/qwen35-spd-tap-slices/stage-*.gguf
```

Those split boundaries produce ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`, exposing every hidden
state required by the pretrained head as a stage boundary for the local proof.
The recorded local artifact validation used `Qwen3.5-4B-Q4_K_M.gguf` and found
all `426` owned tensors exactly once across the seven slices.

The same split shape has also been exercised through live Skippy binary stage
transport against the full GGUF:

```bash
cargo run -p skippy-bench -- local-split-chain-binary \
  --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --ctx-size 128 \
  --n-gpu-layers 0 \
  --selected-backend-device CPU0 \
  --stage-bind-base-port 19131 \
  --prompt Hello
```

Recorded result: stage ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`, activation width
`2560`, first boundary payload `10240` bytes / `5120` f16 wire bytes, prompt
token id `9419`, predicted token `11`. A `local-split-compare` run on the same
GGUF and prompt matched the unsplit full-model token `11`.

Validate an exported local head through Rust with:

```bash
SKIPPY_SPD_MANIFEST=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  cargo test -p skippy-runtime validates_external_manifest_when_skippy_spd_manifest_is_set
```

## Export a Rust/Python Parity Fixture

Rust top-k parity uses the same trained head and the same real hidden-state
inputs as Python. Export a fixture with:

```bash
python3 evals/spd/export_parity_fixture.py \
  --reference-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/speculative_pipeline_decoding \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --base-model-path Qwen/Qwen3.5-4B \
  --out /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --device mps \
  --top-k 8
```

This writes real SPD inference rows, raw hidden-state tap rows, position ids,
base final-norm weight, Python intermediate states, Python logits, Python top-k
draft indices, and Python top-k full token ids. When the prompt leaves prefix
rows before the rolling SPD window, it also writes `cached_prefill_cur_in`,
`cached_prefill_position_ids`, and Python cached `spec_past_kv` logits/top-k
for the same proposal rows. Validate the fixture container through Rust with:

```bash
SKIPPY_SPD_PARITY_FIXTURE=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  cargo test -p skippy-runtime validates_external_parity_fixture_when_skippy_spd_parity_fixture_is_set
```

Validate the real Rust/Python top-k parity path in release mode:

```bash
SKIPPY_SPD_MANIFEST=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set

SKIPPY_SPD_MANIFEST=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_cached_fixture_forward_matches_python_topk_when_env_is_set
```

Or run the combined bench report:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --top-k 8
```

Recorded parity result from the regenerated `Hello` fixture:

- tap input reconstruction max absolute diff: `7.62939453125e-6`
- Rust matched Python top-k draft indices
  `[7728, 15014, 38999, 10036, 11235, 13293, 15953, 0]`
- full token ids
  `[9419, 21251, 109266, 12675, 14556, 18103, 23066, 0]`
- spec-query max absolute diff: `0.03125`
- final-hidden max absolute diff: `0.125`

Recorded cached parity result from the Qwen3.5-4B fixture with `20` prefix
rows:

- Rust/Python cached top-k draft indices matched
  `[23, 17, 24, 21, 16, 22, 660, 19]`
- cached full token ids matched `[23, 17, 24, 21, 16, 22, 760, 19]`
- cached spec-query max absolute diff: `0.03125`
- cached final-hidden max absolute diff: `0.0625`
- full cached-logit max absolute diff: `0.0625`

## Run the Live Skippy Tap Proof

After exporting the parity fixture and building the patched native Skippy ABI,
run the pretrained head from real Skippy activation frames:

```bash
cargo run -p skippy-bench -- spd-live-tap-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --ctx-size 128 \
  --n-gpu-layers 0 \
  --selected-backend-device CPU0 \
  --top-k 8 \
  --verify-steps 8
```

Recorded local result:

- live taps captured: `0,8,10,16,20,24,31`
- each tap frame: `13` tokens, `133120` bytes, hidden width `2560`
- live `cur_in` max absolute diff vs HF fixture: `0.3134765625`
- g0 row max absolute diff vs HF fixture: `0.00103759765625`
- live Skippy top-1 token id: `9419`
- fixture Python/Rust top-1 token id: `9419`
- live top-8 token ids:
  `[9419, 21251, 109266, 14556, 23066, 18103, 12675, 0]`
- fixture top-8 token ids:
  `[9419, 21251, 109266, 12675, 14556, 18103, 23066, 0]`
- target verifier input token: `271`
- target verifier predicted token: `9419`
- accepted live SPD top-1 proposal: `true`
- verifier checkpoint restored to token count `12`
- ordinary non-SPD greedy token: `9419`
- verified committed output matches ordinary non-SPD greedy output: `true`

Recorded repeated verifier run with `--verify-steps 8`:

- generated committed tokens: `[9419, 0, 2500, 628, 353, 1438, 488, 3242]`
- accepted live SPD top-1 proposals: `7 / 8`
- rejected proposals: `1 / 8`
- top-1 acceptance rate for this diagnostic prompt: `0.875`
- every target verifier window rewound to the pre-verify token count: `true`
- every committed token matched ordinary non-SPD greedy decoding: `true`
- total elapsed: about `1697ms`
- average step timing: about `212ms` total, `128ms` tap replay, `5ms`
  assembling `cur_in`, `41ms` SPD head, `21ms` target verify, and `17ms`
  ordinary greedy decode

The live proof uses the Q4_K_M GGUF, while the fixture was exported from the HF
BF16 model. The deeper-row drift is therefore expected; the current result says
the Skippy tap/head plumbing works and the best proposal survives quantization
for this prompt. It also proves repeated real target-verifier acceptance
windows in a diagnostic harness, but does not yet measure request-path SPD
serving throughput.

`skippy-server serve-binary --openai-bind-addr` now has an experimental
request-path source for the same head:

```bash
skippy-server serve-binary \
  --config stage0.json \
  --topology topology.json \
  --activation-width 2560 \
  --openai-bind-addr 127.0.0.1:9337 \
  --openai-spd-manifest /path/to/skippy-spd-head.json \
  --openai-spd-fixture /path/to/spd-parity-fixture.safetensors \
  --openai-spd-model-path /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --openai-speculative-window 1
```

That path feeds real SPD proposals into the normal Skippy `VerifySpan`
verify/repair/rollback loop.

For a reproducible local request-path smoke, prefer the benchmark wrapper. It
launches the local binary stages, runs a baseline request and an SPD request,
derives the selective tap-return allowlist from the fixture, and writes a JSON
summary of OpenAI decode telemetry:

```bash
skippy-bench spd-openai-smoke \
  --stage-server-bin target/release/skippy-server \
  --manifest /path/to/skippy-spd-head.json \
  --fixture /path/to/spd-parity-fixture.safetensors \
  --model-path /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --activation-width 2560 \
  --activation-wire-dtype f16 \
  --max-tokens 8 \
  --temperature 0.0 \
  --output /tmp/spd-openai-smoke-report.json
```

Add `--downstream-wire-delay-ms <ms>` and optionally
`--downstream-wire-mbps <mbps>` to run the same smoke with local downstream
wire conditioning. Add `--prompt-file <jsonl>` and `--prompt-limit <n>` to run
the same baseline/SPD shape over a prompt set. Prompt files accept non-empty
plain-text lines, JSON string lines, JSON objects with `prompt`, `text`, or
`content`, chat-style `messages`, or `turns` arrays. `messages` are sent to the
OpenAI chat endpoint unchanged; `turns` are joined into one user message. `id`,
`label`, or `prompt_id` are used as report labels when present. The report's
aggregate `summary` records paired content matches, baseline/SPD wall and
decode means, speedup ratios, total accept/reject counts, optimistic commits,
tap failures, per-prompt comparisons, and `summary.pipeline_gap`. The
pipeline-gap block rolls up pre-target, optimistic-commit, and post-target
inline probes, empty post-target probe rate, probe and wait-after-probe timing,
normal versus optimistic downstream wait time, and whether optimistic verifies
requested reusable SPD tap returns. Optimistic-commit probe counts show whether
the sidecar produced an accepted next-token proposal while the already-started
optimistic verifier was still in flight. It also reports pre-target proposals
without tap returns, accepted/rejected tap-return requests, and the tap-return
acceptance rate for margin-gate tuning. `cases[].decode.rolling` and
`cases[].inline_probes[].rolling` expose the live request-path rolling state.
When the observer can form paper-style speculation rows, the same rolling block
includes `row_positions`, resolved inference `row_i_stages`,
`row_evicted_prefix_position`,
`row_newest_position`, and `row_next_draft_position`; empty row arrays mean no
row snapshot was available for that event, not that the whole request lacked
paper-shaped rows. The scheduler owns nominal paper layout roles; the server
resolves those roles through the sidecar manifest before reporting the row
stages used for proposal assembly.
`cases[].inline_probes[].rolling_verified_delta` is present when the observer
advanced the target-verified prefix and includes the start position,
verified-up-to frontier, tokens, and token count for that newly verified span.
`cases[].inline_probes[].tap_source`, `tap_collect_ms`, `cur_in_ms`, and
`forward_ms` report whether an optimistic inline probe used direct-return taps
or replay fallback, plus the time spent collecting taps, assembling `cur_in`,
and running the sidecar forward. These fields cover `optimistic_commit`
diagnostics as well as normal pre-target inline probes.
`summary.paper_pipeline_estimate` projects the
observed accept rate onto the paper/reference rolling pipeline schedule using
the manifest's logical SPD stage count, while still reporting the physical
tap-aligned Skippy stage count. For the current product proof, interpret this as
pipeline-fill economics, not measured speedup: accepted speculative tokens are
the first-order count of future split-stage verifier trips that can be removed
from the critical path only if the rolling executor overlaps those verifier
trips. The simple benchmark math is: count baseline emitted tokens and
split-stage trips, count accepted speculative tokens/windows from the same
prompt set, derive the paper-style critical-path reduction from the rolling
trace or `paper_pipeline_estimate`, then compare that upper bound with the
measured `decode_speedup_spd_vs_baseline` only as an overhead diagnostic.
Sidecar forward/head time, tap-collection time, transport latency, rejected
windows, and rollback drains come directly out of the same report and explain
the gap between the pipeline-fill estimate and measured wall/decode speed. If
accepted proposals are zero, the round-trip savings are zero and the only valid
expectation is slowdown from SPD overhead. `summary.rolling_trace_replay` replays observed
pre-target and diagnostic `optimistic_commit` proposals through
`SpdRollingScheduler` when token/proposal traces exist, and otherwise falls
back to final live `cases[].decode.rolling` telemetry for primary-verify-only
smokes. `cases_replayed` counts trace replays; `live_cases_observed` counts
those live final rolling summaries. Replayed traces also report final
target-verified prefix tokens and `verified_prefix_matches_target`, which is
the exactness guard for future scheduler-driven serving changes.
`cases[].decode.spd_proposal_total_*` reports SPD proposal-source totals from
the final decode event across primary proposal windows and inline probe
attempts: requested/attempted/proposed counts, inline-tap hits, replay
fallbacks, cache hits/misses, and time spent collecting taps, assembling
`cur_in`, and running the sidecar head. Use those fields to separate head cost
from replay-fallback hidden-tap cost.

`SpdRollingScheduler` and `SpdRollingTraceReplay` in `skippy-runtime::spd` are
the Rust contract for the next serving rewrite. They are intentionally
token/position-only today; hidden states, direct-return taps, and runtime
checkpoints still live in `skippy-server`.

Recorded bounded local OpenAI request-path proof:

- topology: seven tap-aligned local CPU stages,
  `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`
- model: `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`
- prompt: Humaneval eval row `index=8`, `max_tokens=4`, `temperature=0`
- SPD source: `spd-replay`
- SPD proposals: `4`
- accepted proposals: `2`
- rejected proposals: `2`
- emitted text: `<think>\nThe user wants me`
- no-SPD baseline emitted the same text
- SPD replay wall time: about `101.5s`; no-SPD baseline wall time: about
  `1.28s`

That request-path result proves correct integration with target verification,
not speed. The next engineering step is to schedule proposal generation around
freshly returned inline taps and then run ordinary split serving and SPD serving
against a larger shared prompt set with injected and real hop latency.

Current inline-tap progress: embedded stage-0 serving records stage-0 boundary
activation rows into an SPD-positioned tap cache, downstream binary stages can
return tap frames over the direct-return side channel for SPD-marked requests,
and `spd-replay` overlays complete cached boundary frames before falling back to
local replay. A one-token Qwen3.5-4B smoke on seven local CPU stages returned
the required `10`, `20`, and `31` rows with no tap-return failures. The
proposal source now skips local downstream replay when all required non-h0 rows
are present and reads h0 from GGUF `token_embd.weight` when possible. Recorded
release no-replay smoke for a one-token request:

- topology: seven tap-aligned local CPU stages,
  `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`
- binary: `target/release/skippy-server`
- prompt: `Write a Python function named add that returns the sum of two integers.`
- response content for `max_tokens=4`: `<think>\nThinking Process:\n\n`
- no-SPD baseline emitted the same text
- inline probe phase: `pre_target_reply`
- inline probe trigger: returned `hf_index=31` tap from producer stage `5`
- inline probe elapsed: about `389ms` to `393ms` each
- target wait after probe: about `0ms`
- inline verified SPD windows: `4`
- accepted proposals: `1`
- rejected proposals: `3`
- inline accept rate on this prompt: `0.25`
- target-verified proposal sequence:
  - proposed `8160`, target `90700`, rejected
  - proposed `264`, target `8340`, rejected
  - proposed `25`, target `25`, accepted
  - proposed `25`, target `271`, rejected
- SPD request wall time: about `3.39s`
- no-SPD same-topology wall time for the same four-token request: about `0.57s`
- SPD decode time: about `2.69s`, including about `1.56s` of head proposal time
- no-SPD decode time: about `145ms`

That result proves the real pretrained head runs from inline Skippy request
taps without replay fallback and can start before the final target reply is
consumed by stage 0. It also proves those proposals are verified against normal
target decode and ordinary greedy output is preserved. The first four-token
run was not a live speedup because it predated the Qwen serving-head fast path
and accepted only one of four proposals. Later release `spd-live-tap-parity`
runs matched ordinary greedy output and rewound every verifier window. The
first release timing sample accepted `3 / 3` live top-1 proposals and averaged
about `248ms` per step: `42ms` in the SPD head, `58ms` assembling `cur_in`, and
`107ms` in tap replay. Keeping sidecar projection weights resident cut
assembly to about `41ms`; parallelizing tap projection cut it to about `5ms`.
The latest eight-step release live-tap sample accepted `7 / 8` proposals and
averaged about `212ms` per step.

The real OpenAI request path was then rerun with optimistic decode, selective
tap returns, resident projection weights, and parallel tap projection. For the
same eight-token prompt, the latest no-SPD baseline was about `0.65s` wall /
`209ms` decode. The in-repo `skippy-bench spd-openai-smoke` command now
reproduces this local seven-stage request-path flow and derives the selective
tap-return allowlist from the sidecar topology. Earlier filtered SPD returned
only hidden taps `10`, `20`, and `31`; the current topology-derived default for
the Qwen3.5-4B S4/L4 head is `[8, 10, 16, 20, 24, 31]` after stripping h0, so
paper-shaped rolling rows can request their required taps.
Row-specific collection keeps fixture-shaped probes from waiting on those
future-row taps. The latest topology-derived run produced the same text,
proposed `8`, accepted `3`, rejected `5`, committed two optimistic tokens, and
ran in about `3.52s` wall / `2.96s` decode versus a `547ms` wall / `212ms`
baseline. It is slower than the narrower fixture-tap run because downstream now
returns more tap frames, but those frames are required for true paper rows.
Before the token-position fix, a follow-up run produced exact same text and
kept live/replay rolling ordered but accepted only `1 / 8` proposals: the live
rows were shifted by one token after the first accepted proposal, so the sidecar
kept reading the previous token as the newest row. The Python reference
generator on the same no-thinking prompt accepted `7 / 8`, which isolated the
problem to serving row alignment rather than sidecar quality.
An earlier pre-cache resolved-role diagnostic matched the exported fixture
roles (`[4, 4, 4, 0]` for full Qwen3.5 snapshots) and preserved exact greedy
text, but accepted `0 / 8` proposals. That run kept live/replay rolling ordered
and emitted resolved rows from the first probe, but slowed to about `9.93s`
wall / `9.23s` decode versus the same-run baseline at about `633ms` wall /
`201ms` decode. The follow-up native-cache smoke added the first stateful path:
Rust now lazily prefills complete `g_S` prefix rows from inline prefill taps,
stores sidecar K/V per spec layer, crops to the minimum rolling row position
before each proposal, and emits `cache_used` / `cache_prefix_len` on inline
probe reports.
The cache was active on every proposal (`cache_prefix_len` moved from `20` to
`29`) and exact text still matched baseline, but the shifted row positions made
acceptance collapse. The cached Python fixture closed the cache-fidelity
question: with `20` prefix rows, Rust and Python cached top-k token ids matched
exactly (`[23, 17, 24, 21, 16, 22, 760, 19]`), with `0.0625` full-logit max
diff. After switching live rolling positions to actual token indices, the same
bounded OpenAI smoke accepted `8 / 8` proposals, committed `3 / 3` optimistic
target decodes, and kept exact greedy output. Baseline was `626ms` wall /
`198ms` decode; SPD was still slower at `3521ms` wall / `2921ms` decode.
Treat the remaining issue as serving latency/scheduling evidence, not
cache-logit mismatch or low-head-quality evidence.

The next fixed-stage-default smoke preserved exact output but accepted only
`6 / 8` proposals, and both failures were depth-2 optimistic commits. New
proposal-row telemetry showed those failures were not model quality: the step-2
proposal reused stale rows `[23,24,25,26]` after the accepted optimistic token
should have advanced proposal assembly to target position 28. Observing
accepted optimistic-commit probes into the live rolling observer immediately
fixed that stale context. The follow-up smoke at
`/private/tmp/spd-openai-rolling-observe-smoke1/report.json` preserved exact
output, proposed `5`, accepted all `5`, committed `3 / 3` optimistic decodes
and `2 / 2` chained optimistic decodes, and step 2 proposed token `198` from
rows `[24,25,26,27]`. It still ran slower than baseline (`222ms` baseline
decode versus `2668ms` SPD decode) and ended with `3` missing rolling proposal
positions after the chain boundary, so the next work is executor scheduling,
not the Qwen fixed-stage contract or stale accepted-context rows.

The miss-diagnostic smoke at
`/private/tmp/spd-openai-tap-position-diagnostics-smoke1/report.json` narrowed
that executor gap further. It preserved exact output, accepted `5 / 5`
proposals, and committed all optimistic/chained optimistic verifier results,
but post-target probe steps `4`, `5`, and `6` were empty with
`missing_inline_taps` for position `28`. h0 is now synthesized from embeddings
instead of treated as an inline tap. Tap-position telemetry showed position
`28` was recorded for all required non-h0 heads before those probes, then lost
during a shorter accepted-prefix commit from the token-emission path. The
serving path now treats prefix-compatible accepted-context updates as
acknowledgements when SPD is already ahead, so it advances lifecycle state
without pruning future rows. The follow-up smoke at
`/private/tmp/spd-openai-prefix-ack-smoke1/report.json` preserved exact output,
accepted `8 / 8`, committed `6 / 6` optimistic verifier results, eliminated
post-target empty probes, and left rolling replay with `0` missing or
out-of-order proposals. The native path still needs the paper-shaped rolling
executor and faster sidecar/head execution to become a speed result.

Earlier filtered SPD
produced the same text, proposed `4`, accepted `1`, rejected `3`, committed one
optimistic token, and ran in about `1.92s` wall / `1.38s` decode. Switching
optimistic target work from `CheckpointSession + DecodeEmbd` to a checkpointing
one-token `VerifySpan` preserved exact text and the same proposal counts, cut
optimistic checkpoint telemetry to about `0.017ms`, and measured about `1.95s`
wall / `1.39s` decode on the same prompt. The previous
unfiltered SPD run was about `3.19s` wall / `2.60s` decode; filtered SPD before
the projection fast path was about `2.22s` wall / `1.63s` decode. Proposal time
is now about `239ms`, down from about `478ms` before the cache/parallel path.
Local CPU SPD still needs higher accepted proposal coverage and lower target
wait before it beats the normal split path. Treat this as a
regression/performance smoke, not native mesh config evidence, because
`skippy-bench` writes stage JSON itself.

A one-prompt prompt-file smoke against
`crates/skippy-bench/corpora/speculative_coding_prompts.jsonl` with
`--prompt-limit 1` validated the aggregate report in the real staged OpenAI
path. Prompt `spec-code-001` matched baseline text exactly, proposed `2`, accepted
`1`, rejected `1`, committed `0` optimistic tokens, and had no tap failures.
The speed result was still negative: baseline was about `805ms` wall / `50.7ms`
decode, while SPD was about `1338ms` wall / `438ms` decode (`0.602x` wall,
`0.116x` decode). Use this as report-shape evidence and a starting point for a
larger prompt sweep, not as proof of SPD acceleration.

A two-prompt smoke against `crates/skippy-bench/corpora/chat_corpus_fixture.jsonl`
with `--prompt-limit 2` verified that `spd-openai-smoke` preserves true
chat-style `messages` rows when constructing the OpenAI request. The prompt set
covered one flat prompt and one `{system,user}` message prompt. Baseline and
SPD emitted matching two-token text for both prompts. The aggregate summary
reported `prompt_pairs = 2`, `matching_content = 2`, SPD proposed `4`, accepted
`1`, rejected `3`, committed `0` optimistic tokens, and had no tap failures.
The mean baseline timing was about `451ms` wall / `53.3ms` decode; mean SPD
timing was about `975ms` wall / `447ms` decode (`0.462x` wall speedup and
`0.119x` decode speedup). Use this as chat-corpus benchmark evidence, not as
proof of SPD acceleration.

The benchmark can also inject bounded local downstream-stage latency through
the native `serve-binary` wire conditioner. With `--downstream-wire-delay-ms 10`
the same eight-token run preserved exact output and narrowed the local gap:
baseline was about `2.51s` wall / `1.92s` decode, while SPD was about `2.80s`
wall / `2.14s` decode with `4` proposals, `1` accepted proposal, and one
committed optimistic token. At `25ms`, the current optimistic path did not cross
over: baseline was about `2.76s` wall / `1.94s` decode, while SPD was about
`5.08s` wall / `4.13s` decode. Rejected optimistic target decodes pay delayed
downstream work too, so the current low-acceptance Qwen proof head is correct
but not yet a latency speed path. Disabling optimistic decode at `25ms` produced
`3` accepted probes out of `8` but still took about `3.65s` wall / `2.67s`
decode because proposals are not committed without the optimistic path.

The serving path now emits inline probe top-1 logits and top-1/top-2 logit
margins, and `skippy-bench spd-openai-smoke` exposes
`--optimistic-min-logit-margin` for gated optimistic decode. On the same `25ms`
run with `--spd-top-k 2`, rejected optimistic proposals had margins `0.125` and
`1.0`; the accepted proposal had margin `2.5`. With
`--optimistic-min-logit-margin 1.5`, the paired current-code smoke preserved
exact output, skipped the two rejected optimistic decodes, committed the accepted
token, and measured baseline at about `2.76s` wall / `1.91s` decode versus gated
SPD at about `3.70s` wall / `2.83s` decode. This proves the gate removes bad
optimistic work, not that the current head is fast enough.

The first tap-return implementation requested taps only after a margin-gated
optimistic decode, which proved accepted optimistic target decodes could
preserve their tap rows and improve proposal coverage after a committed
optimistic token. With `--downstream-wire-delay-ms 25` and
`--optimistic-min-logit-margin 2.5`, SPD measured `6` proposals, `3` accepted
probes, `2` committed optimistic tokens, `0` rejected optimistic decodes, and
exact output. It was still slower than the same-run baseline: about `4.03s`
wall / `3.02s` decode versus baseline `2.89s` wall / `2.07s` decode. At `10ms`,
the same gate measured baseline `1.86s` wall / `1.10s` decode and SPD `2.52s`
wall / `1.81s` decode. Current code applies that tap-return behavior to every
optimistic SPD verify that is actually started; the margin gate remains only a
work-start filter. This is a coverage fix and a useful tuning surface; it is
not yet a speed result for the current CPU proof head.

The request path now keeps inline tap-cache rows for the common token prefix
when the SPD source resets, dropping only rows at or after the first divergent
token. A pre-patch ungated no-tap diagnostic was faster, but it starved the
rolling rows after accepted optimistic tokens. The retained behavior requests
optimistic taps for every started optimistic SPD verify, drops future rows on
rejection, preserves accepted-extension rows after verification, and treats
shorter accepted-prefix commits as acknowledgements when SPD's context is
already ahead. That keeps the rolling replay ordered in the current smoke, but
tap-return transport and sidecar/head work still dominate local latency.

Current mesh-native config status: the same experimental SPD knobs are
available through `[defaults.speculative]` and per-model `speculative`
configuration (`mode = "spd"`, `spd_bundle_ref`, `spd_manifest_path`,
`spd_fixture_path`, `spd_model_path`, `spd_max_tokens`, `spd_top_k`, `spd_gpu_layers`,
`spd_replay_fallback`, `spd_optimistic_decode`,
`spd_rolling_executor`, `spd_optimistic_min_logit_margin`). These settings now resolve and propagate
into staged embedded OpenAI args. Native staged config also derives
the SPD tap-return allowlist from the sidecar topology and carries it through
stage-control load requests so workers return every logical-row tap except h0;
the host-runtime split-load test asserts the derived
`[8, 10, 16, 20, 24, 31]` list reaches the worker `StageLoadRequest`. This is
config plumbing only; it does not choose a compatible tap topology or train a
sidecar automatically. The current Mesh resource-aware split planner also does
not consume the SPD manifest when choosing stage boundaries. Until that hook
exists, a product SPD sidecar must be trained for the actual planned Mesh
topology, not for a convenient or previously tested split.

The reusable artifact key is the logical layer-boundary topology, not the
hostname list. For the first Qwen3-8B two-node proof, the sidecar is trained for
`23,36`, which corresponds to Skippy ranges `0..23` and `23..36`. The same
sidecar can move between M4+mini, two cloud nodes, or local stage servers if
those ranges and taps are preserved. A future planner can avoid a combinatorial
explosion by precomputing a small set of canonical logical SPD topologies and
packing adjacent logical stages onto a larger node, provided the runtime still
returns every manifest-required logical boundary tap. Such clumping changes the
amount of physical overlap available, so it needs separate timing evidence, but
it should not require retraining when the logical taps are unchanged.

`spd_bundle_ref` is the product-shaped sidecar input: the coordinator resolves a
local sidecar directory/manifest or `hf://namespace/repo[@revision]` containing
`skippy-spd-head.json`, the manifest-declared serving checkpoint
(`spd-head.safetensors` in current exports), and
`spd-parity-fixture.safetensors`. Worker stages still receive only the derived
tap-return allowlist. The SPD replay model source can now be either an explicit
full GGUF through `spd_model_path` or the resolved local Skippy layer package
used by stage 0. In the package-backed shape, live taps open selected package
parts and h0 comes from the package embedding part, so the coordinator does not
need an extra full GGUF only for SPD replay.

2026-06-18 package-backed SPD request-path checkpoint: release
`spd-openai-smoke` passed with `--model-path /private/tmp/skippy-qwen35-4b-package-s2`,
`--splits 16 --layer-end 32`, and the two-stage S2 sidecar bundle. Generated
stage configs used
`load_mode=layer-package` for `0..16` and `16..32`; stage-0 logs reported
`llama_stage.spd_model_source="layer_package"` and `spd_model_path=null`.
Report `/private/tmp/spd-qwen35-s2-openai-package-local-4-rerun.json` matched
baseline/SPD content, proposed `3`, accepted `0`, rejected `3`, and recorded
`0` tap return/record failures. This proves package-backed request-path
correctness for the two-stage shape; it is not speed evidence and not a quality
claim for the tiny debug sidecar.

2026-06-18 Qwen3-8B S2/23 checkpoint: a topology-matched
`Qwen/Qwen3-8B` sidecar was trained locally with `num_stages=2` and
`stage_layer_boundaries=23,36`, exported to BF16 safetensors, and passed Rust
fixture parity. Reference eval over the same GSM8K rows used by the product
prompt file reported nonzero acceptance, but release `spd-openai-smoke` on the
GGUF request path did not. The initial Q4/Q8 product smokes proposed from
shallow `[1,0]` rows when HF36 was absent; this was not true to a manifest with
`trained_with_use_deepest=true`. The runtime now treats those deepest rows as
required: without replay fallback, the strict diagnostic report
`/private/tmp/spd-qwen3-8b-s2-23-lr1e4-direct-q4-gsm8k-prompt2-strict-no-replay.json`
made `0` proposals and explicitly reported missing HF36 rows instead of
silently using shallower taps.

The same strict diagnostic with `--spd-replay-fallback` proved the remaining
gap is not just row-role routing. Q4 replay
`/private/tmp/spd-qwen3-8b-s2-23-lr1e4-direct-q4-gsm8k-prompt2-strict-replay.json`
used deepest `[2,0]` rows for every proposal but accepted `0 / 8`; the first
post-target proposal was `48146` while the Python reference trace proposed the
target token `594` at the same target position. Q8 replay
`/private/tmp/spd-qwen3-8b-s2-23-lr1e4-direct-q8-gsm8k-prompt2-strict-replay.json`
also used deepest rows and accepted `0 / 8`, with first post-target proposal
`7570`. Q8 therefore does not explain the reference/product acceptance gap.

`skippy-bench spd-live-tap-parity` now uses the shared runtime live-tap runner
instead of the older benchmark-local embedding-only `0..0` stage path, so the
Qwen3-8B Q4/Q8 checks no longer abort in native llama.cpp. It also fails closed:
the JSON report is written first, then the command exits nonzero when the actual
terminal-final-normed serving path diverges from the fixture. The Q8 diagnostic
`/private/tmp/spd-qwen3-8b-s2-23-lr1e4-live-tap-q8-gamma-diagnostic-v2.json`
shows the raw terminal h36 mismatch is a semantic pre-final-norm vs
post-final-norm mismatch: raw h36 has `max_abs=121.346748`, mean abs
`5.174249`, and cosine `0.883039` against the fixture, while applying Qwen final
RMSNorm with GGUF `output_norm.weight` moves it to `max_abs=0.654507`, mean abs
`0.053620`, and cosine `0.999539`. A robust inferred-gamma check on the same
row reports cosine `0.991195` against GGUF `output_norm.weight` after skipping
near-zero denominators. The normalized path still fails the strict gate:
`cur_in_max_abs_diff=0.09765625`, rank-paired logit diff `0.0625` within
tolerance, top-1 token matching the Rust fixture (`9914`), but exact top-k rank
drift remains. Target verification rejects that first normalized proposal
because the Q8 target greedy token is `23`, not `9914`.

Current release request-path smoke on Q4 with strict deepest rows and no replay
fallback is
`/private/tmp/spd-qwen3-8b-q4-current-gsm8k-prompt2-inline-finaltap-v5.json`.
It matched baseline/SPD content, recorded `0` tap return/record failures, used
inline terminal HF36 taps (`tap_returns_by_hf_index={"36":12}` and
`tap_records_by_hf_index={"23":13,"36":8}`), and made every proposal from
inline deepest rows: `inline_tap_hits=7`, `replay_fallbacks=0`, row stages
`[2,0]`, `missing_proposals=0`, and `out_of_order_proposals=0`. This closes the
local product request-path gap for terminal HF36 delivery. The native side adds
`SKIPPY_ACTIVATION_FLAG_FINAL_NORMED` and enables llama embeddings only for
final filtered stages that actually reserve activation-output capacity, so
ordinary final-stage baseline decode remains a no-op for activation return.

The same report proposed `7`, accepted `1`, rejected `6`, and saved `1 / 7`
candidate token round trips. Baseline decode was `120.9ms`; SPD decode was
`1055.7ms`, with proposal/head work dominating (`head_total_ms=455.7ms`,
`cur_in_ms=40.0ms`) and same-machine stage contention inflating downstream
wait. This is mechanics evidence, not speed evidence. Also keep acceptance
metrics honest: reference eval with `draft_top_k=4` is an oracle-style metric and
cannot be compared directly to serving greedy top-1 verification. For speed
work, the serving-path metric that matters for this checkpoint is `1 / 7`.

Fresh identical-prompt proposal parity now has a repeatable comparator:
`evals/spd/compare_reference_product_spd.py`. A reference eval-only rerun with
the same LR `1e-4` BF16 head, `draft_top_k=1`, `max_new_tokens=8`, and the
exact GSM8K Indras prompt emitted proposal traces at
`/private/tmp/skippy-spd-qwen3-8b-s2-23-bf16-lr1e4-proposal-trace-eval-uv-20260618-141057/artifacts/20260618-141057/eval/raw/pipeline_eval__train__speculation_head_final__nt9__per_sample.jsonl`.
Comparing that trace with
`/private/tmp/spd-qwen3-8b-q4-current-gsm8k-prompt2-inline-finaltap-v5.json`
produced
`/private/tmp/spd-qwen3-8b-q4-current-gsm8k-prompt2-reference-product-comparison.json`.
The target token stream matches exactly for the first eight generated tokens
(`10061,594,1438,1495,279,3491,3019,553`), but proposal parity is not clean:
only `2 / 7` proposal tokens match the reference. The first reference proposal
for target position `56` is token `594` and is accepted; the product inline Q4
path proposes `7570` for that same target position from row roles `[2,0]` and
rejects it. Do not start a larger training run solely from aggregate reference
acceptance until this exact-prompt reference/product proposal divergence is
explained or accepted as quantized-target training drift.

Current live-tap reruns narrow that decision. `spd-live-tap-parity` now routes a
local directory containing `model-package.json` through the existing
layer-package tap runner instead of treating it as a direct GGUF. The
package-shaped Q4 report
`/private/tmp/spd-qwen3-8b-s2-23-lr1e4-live-tap-q4-layerpackage-current-rerun.json`
successfully opens the Mesh-resolved
`meshllm/Qwen3-8B-Q4_K_M-layers` package, collects h0/HF23/HF36 live taps, and
matches target verification against the non-SPD decoder. It fails the same
final-normed parity gate as the direct Q4 control: `cur_in_max_abs_diff=0.49609375`,
rank-paired logit diff `0.3125`, and exact top-k drift. Direct GGUF controls
using the same current harness show the quantization gradient. The Q8 report
`/private/tmp/spd-qwen3-8b-s2-23-lr1e4-live-tap-q8-direct-current-rerun.json`
uses the terminal-final-normed path, matches target verification against the
non-SPD decoder, and reduces the gate to `cur_in_max_abs_diff=0.09765625` with
rank-paired logit diff `0.0625` within tolerance, but exact top-k still drifts.
The Q4 report
`/private/tmp/spd-qwen3-8b-s2-23-lr1e4-live-tap-q4-direct-current-rerun.json`
is worse on the same path: `cur_in_max_abs_diff=0.49609375`, rank-paired logit
diff `0.3125`, and exact top-k drift. Raw terminal h36 remains much worse in
both reports, confirming that final-normed terminal h36 is the serving
convention closest to the BF16 training fixture. The practical read is that the
current product-side mechanics are good enough to expose the real issue: a BF16
reference-trained head is being evaluated on quantized GGUF/Metal hidden-state
distributions, especially Q4. Before spending on a bigger generic BF16 run,
train/evaluate on product-like quantized activations for the exact `23,36`
split rather than only scaling the same BF16-reference recipe.

Bounded local `llama-spec-bench` status for ordinary target/draft speculative
decoding, separate from SPD:

- target: `Qwen3-4B-Q4_K_M.gguf`
- draft: `Qwen3-0.6B-Q4_K_M.gguf`
- prompt count: `8`
- `max_new_tokens=8`, `speculative_window=4`, `ctx_size=512`, `n_gpu_layers=0`
- tokenizer match: `true` for every prompt
- speculative output matched baseline for every prompt
- generated tokens: `64`
- speculative windows: `39`
- accept rate: `22.8%` (`29` accepted / `127` draft tokens, `35` rejected)
- mean accepted tokens per window: `0.74`
- target baseline: `79.31 tok/s`
- current serial speculative path: `53.69 tok/s`
- projected batched rollback path: `50.03 tok/s`
- projected scratch verification path: `13.00 tok/s`

That run proves the target/draft benchmark harness is usable on real GGUF pairs
after the target lane-count fix. It is not evidence of SPD speedup, and this
particular target/draft pair is not a serving-speed candidate as configured.

## Validate Hidden Tap Compatibility

`skippy-runtime` includes a Rust tap planner that converts the manifest's
hidden-state requirements into concrete Skippy stage ownership. The reference
index convention is `0 = embedding output`; `k >= 1` means output after decoder
layer `k - 1`.

For the pretrained `Qwen/Qwen3.5-4B` S4/L4 head, required tap groups are:

```text
g4: [0, 10, 20, 31]
g3: [0, 8, 16, 24]
g2: [0, 8, 16]
g1: [0, 8]
```

The checked-in tests show that a normal four-way split `0..8, 8..16, 16..24,
24..32` still needs internal taps `10,20,31`. A tap-aligned proof split
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32` can expose every required
tap as an ordinary stage boundary.

## Artifact Contract

The proof runner writes:

- `train/speculation_head_final.pt`
- `train/spd-head.safetensors` after export
- `train/spd-parity-fixture.safetensors` after fixture export
- `train/skippy-spd-head.json`
- `eval/raw/*.jsonl`
- `eval/summary/*.json`

The manifest schema is `skippy-spd-head/v1`. It binds a head checkpoint to:

- base model path/id
- checkpoint format/version
- checkpoint byte size and sha256
- hidden size
- base vocab size
- draft vocab size and optional draft token ids
- number of target stages and optional logical stage layer boundaries
- number of spec layers
- shallow hidden-layer tap indices
- optional rotary metadata (`rope_theta`, `rotary_dim`) used by Rust serving
  for Qwen-family sidecars; Qwen3.5 legacy manifests may omit it, but new
  Qwen3 sidecars should include it
- optional safetensors serving checkpoint path, size, checksum, tensor count,
  and dtype

Rust validation lives in `crates/skippy-runtime/src/spd.rs`.
Safetensors parsing and BF16/F32/I64 payload reads live in
`crates/skippy-runtime/src/spd/safetensors.rs`.
The constrained Qwen fixture forward path lives in
`crates/skippy-runtime/src/spd/qwen.rs`.
The tap-row-to-`cur_in` projection bridge lives in
`crates/skippy-runtime/src/spd/tap_input.rs`.

## Current Qwen3-8B S2/23 HF-Scale Checkpoint

2026-06-18 HF job `meshllm/6a33e49bef9220ea67d991c2` trained a
topology-matched `Qwen/Qwen3-8B` SPD sidecar for the real two-stage
`23,36` logical split. The job used `HuggingFaceH4/ultrachat_200k`
`train_sft`, `15997` usable rows from the requested `16k`, BF16 on
`a100-large`, `max_length=2048`, `epochs=1`, batch `1`, gradient accumulation
`8`, LR `1e-4`, `num_spec_layers=4`, draft top-k `4`, and uploaded artifacts
to `meshllm/skippy-spd-qwen3-8b-s2-23` under `runs/20260618-122936`.

Reference held-out eval completed on `96` prompts / `6123` generated tokens:
aggregate acceptance `0.7013`, equivalent accept length `1.4026`, and
theoretical throughput gain `41.0%`. Per split: MT-Bench `0.7041`, HumanEval
`0.7299`, GSM8K `0.6724`. Treat this as reference BF16 quality evidence, not
proof of native Q4 top-1 speed, because the reference run used draft top-k `4`
and HF weights.

Serving export and fixed-fixture parity passed. The downloaded checkpoint SHA
matched the manifest
(`2337ab591beded04ad088d46ef0256c4e569cfcf3bfb940d618e872aa635519a`);
`export_spd_head.py` produced BF16 `spd-head.safetensors` with `56` tensors
and SHA `6fa55a0c836dc53fb59bc78a2e942d4f668659c65a3b1af20349e08c6674fccc`;
`export_parity_fixture.py` produced fixture SHA
`b7aee917d1ca407efae3fb4d9904ee5da29b4beadd7d616440bef98c53918ac7`.
`skippy-bench spd-fixture-parity` matched Python/Rust top-8 token IDs on both
direct and cached fixture paths.

Live package-backed tap parity exposed the expected BF16-versus-Q4 serving
gap. The split returned required taps `[0,23,36]`, verifier greedy output
matched the non-SPD baseline, and an 8-step verified generation accepted
`2 / 8`, but strict fixture parity failed against the HF BF16 rows
(`terminal_final_normed cur_in max diff 0.5635`, rank logit diff `1.0`, top-k
order changed). Treat this as quantized-serving drift, not an export/tap wiring
failure.

Local package-backed OpenAI smoke with rolling executor on six prompts matched
baseline/SPD content on `6 / 6`, had `0` tap failures, proposed `90`, accepted
`17`, rejected `73`, and committed `14` optimistic tokens. The paper-style
round-trip math was `17` saved and `73` unsaved candidate token round trips,
with mean baseline decode `252.5ms`, mean SPD decode `2502.7ms`, and mean SPD
head total `63.7ms`. The corresponding real two-stage worker smoke over a
direct low-latency link also matched content on `6 / 6`, had `0` tap failures,
proposed `89`, accepted `16`, rejected `73`, and committed `12` optimistic
tokens. Mean baseline decode was `736.5ms`, mean SPD decode was `3466.2ms`,
and the measured ICMP link latency immediately before the run was `0.9-2.0ms`
with `1.16ms` average over 10 packets.

Bottom line: the end-to-end SPD mechanics now work for the real two-stage
Qwen3-8B split with a real trained sidecar and real worker placement. It is not
a speed candidate yet. Native Q4 top-1 acceptance is only about `18%`, so the
sidecar saves some token round trips but not enough to beat added sidecar and
rolling-executor overhead.

## HF Pre-LAN Split-Economics Gate

Larger SPD work should move off the M4 into a Hugging Face qualification job
before another real-node speed attempt. The HF job can run raw product-tap
capture, native-Q4 teacher-logit conversion, raw-mode training/adaptation,
held-out scoring, serving export, and a single-machine package-backed
`spd-openai-smoke`. That is enough to prove predictor quality, artifact
readiness for serving, and request-path mechanics. It is not a distributed
wall-clock speed claim. For `native-package-fresh`, true Rust/Python fixture
parity is explicitly skipped until native parity fixture export exists.

Use `evals/spd/simulate_latency.py` as the deterministic bridge from HF/local
smoke evidence to LAN plausibility. For a Skippy OpenAI smoke report, it reads
observed accepted/proposed candidate-token round trips and sweeps assumed
physical stage costs and inter-node hop latencies:

```bash
python3 evals/spd/simulate_latency.py \
  --openai-report /path/to/spd-openai-smoke.json \
  --stage-ms 40,40 \
  --hop-ms 0.2,1,5,10
```

Pass `--sidecar-ms 0` to model the paper's ideal hidden-sidecar condition; omit
it to use the report's measured `probe_head_total_ms` when present. The gate is
intentionally strict: broad held-out content must match, tap failures must be
zero, sidecar latency must fit under the slowest physical pipeline slot or be
explicitly accounted for, and estimated `spd_vs_serial_saved_tokens` must clear
`1.0` with margin over realistic LAN assumptions.

This also defines how predigested SPD splits should work. Sidecars are trained
for canonical logical topologies and required tap boundaries. Mesh can fit those
logical stages onto fewer physical nodes by colocating contiguous logical
stages, but the runtime still has to return every manifest-required tap. The
economics model must use the fitted physical placement, not the raw logical
stage count: ten logical SPD stages colocated on three nodes may be functional,
but the speed estimate has only three physical compute buckets.

Artifact distribution should stay modular. The base model remains a normal
Mesh/Skippy layer package, so each physical stage node downloads or materializes
only the layer parts it owns. The SPD predictor bundle is coordinator-owned by
default: the coordinator needs `skippy-spd-head.json`, `spd-head.safetensors`,
and either a parity fixture for fixture-parity proofs or a serving fixture for
request-path smoke; worker nodes only need the derived tap-return allowlist and
must return the hidden states requested by the manifest. Running the SPD
predictor on every node should be treated as a future optimization/design
change, not a requirement for the current proof.

The dry-run helper for this flow is:

```bash
python3 evals/spd/plan_hf_spd_qualification.py --json
```

For the current Qwen3-Coder-480B target, the dry-run/planner path should
resolve `meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`, keep S8
logical boundaries `8,16,24,32,40,48,55,62`, require taps
`[0,8,16,24,32,40,48,55,62]`, and model contiguous physical clumping such as
`[[0,1],[2,3],[4,5],[6,7]]` for four physical buckets. The first capped HF lane
used `rtx-pro-6000x4` for `4.5h`, planned at `$49.49991`; the latest retry used
the same flavor for `3.5h`, planned around `$38.50`. The next retry should use
the two-phase capture patch and try package download, staged package load,
native tap/logit capture, head-only training, small held-out scoring, serving
export, package-backed smoke, and latency simulation. The current dry run uses
`--vocab-size 151936` and emits no full-base train/score command. The next
action is to upload the patch artifact, resubmit the capped job, and patch only
the next concrete blocker if it fails.

The older `Qwen3-8B` raw-Q4 path remains useful as harness evidence: it proves
package-backed mechanics, tap return, Rust sidecar loading, rolling
verification, and the M4/mini split path. It is no longer the immediate scaling
target because predictor quality was not proven on broad held-out prompts.

The current local reports demonstrate why this gate matters. The train128 raw
candidate on broad heldout64 matches content and has zero tap failures, but
accepts only `62 / 168` proposals (`62` saved versus `106` unsaved), so it
fails even under ideal hidden-sidecar assumptions. The earlier train64
heldout16 report is only a narrow debug win (`22` saved versus `17` unsaved);
with the measured local sidecar latency around `64ms`, it is still negative
unless the real target pipeline slot is at least that slow or the sidecar is
optimized/offloaded.

## Next Engineering Steps

1. Move the immediate larger-model SPD goal to Qwen3-Coder-480B S8 on HF. Use
   `meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`, logical S8
   boundaries `8,16,24,32,40,48,55,62`, and native package taps/logits as the
   distillation source. Reuse the normal Skippy layer package for physical
   stage material; the SPD sidecar owns logical tap requirements and proposal
   weights only. The topology-only capture and head-only train/score path now
   exists, and the latest `3.5h` capped retry proved package download and prompt
   building but failed on verifier/tap-stage VRAM overlap. The next action is
   to resubmit with the two-phase capture patch, watching for Qwen480 MoE config
   compatibility, CUDA allocator release between phases, and streamed
   stage-open timing under the timeout.
2. Do not run the current Qwen3-8B S2/23 HF-scale head as a speed proof yet.
   The terminal h36 semantic mismatch is understood: Skippy's terminal boundary
   is pre-final-norm and the HF fixture/training row is post-final-norm, so
   serving now applies the Qwen final RMSNorm before projecting unflagged
   terminal rows. Inline package taps and the rolling executor work on the real
   two-stage split, but the current 16k UltraChat head accepts only `16 / 89`
   proposals on the worker smoke. The next sidecar-quality gate is therefore
   not another mechanics comparison; improve native Q4 top-1 acceptance for the
   exact `23,36` product split until the package-backed paper estimate clears
   `1.0` with margin.

   The concrete quality path is now native-Q4 product-row training: expand the
   captured rows beyond the first `64` train / `16` held-out native-logit gate,
   keep held-out prompts separate, fine-tune from the 16k checkpoint, and score
   train plus held-out rows before export. Also keep a build provenance check in
   every speed run: after llama.cpp patch changes, rebuild the native stage ABI
   before relinking release `skippy-server` / `skippy-bench`.

3. Train or fetch a topology-matched sidecar for each real two-stage product
   split before making any speed claim. For Qwen3.5-4B that means
   `num_stages=2`, `stage_layer_boundaries=16,32`, and tap rows
   `0,16,32;0,16`; the current pretrained S4/L4 sidecar is intentionally
   excluded from this test. For the Mesh-native Qwen3-8B split, the logical
   target is `Qwen/Qwen3-8B` with `num_stages=2` and
   `stage_layer_boundaries=23,36`. The current 16k UltraChat head is a real
   trained checkpoint and proves the topology end to end, but native Q4 top-1
   acceptance is still too low for a speed claim.
4. Run the larger native-Q4 product-row training gate on HF, not by thrashing
   the M4. The dry run must print model/package ref, dataset shard, logical
   topology, row cap, hardware, timeout, output repo, and maximum cost. The job
   should emit held-out scores, the serving bundle, package smoke, and
   latency-simulation JSON. Fixture parity is not part of the first
   `native-package-fresh` lane until native parity fixture export exists.
5. Only after held-out package-backed serving clears a `1.0` paper estimate,
   rerun `spd-openai-smoke` with explicit `--stage-hosts`,
   pre-materialized package directories or Mesh-resolved package artifacts, and
   ordinary split baseline/SPD pairs to test real wall-clock speed. Stage 0 plus
   the sidecar should stay on the coordinator; stage 1 should be on the worker.
6. Use injected downstream delay only as a bounded diagnostic; do not report it
   as distributed speedup.
7. Add an SPD sidecar package workflow around the Python reference trainer:
   plan logical tap topology, train, eval `L'_acc`, export safetensors/manifest,
   validate Rust parity when a real parity fixture exists, then publish sidecar
   metadata alongside Skippy model artifacts.

## Next Research Steps

1. Train a head for `Qwen/Qwen3-8B` as the first larger dense Qwen-family
   scaling proof. Use the matching GGUF quant intended for Skippy split serving
   after the HF config/GGUF metadata has been inspected.
2. Start with `num_stages=4`, `num_spec_layers=4`, greedy/no-thinking eval, and
   a draft vocab capped at `32k`. Derive the logical stage boundaries from the
   inspected target layer count; do not copy the Qwen3.5-4B
   `8,10,16,20,24,31` physical tap split into a new sidecar.
3. For the first spend-bearing HF job, dry-run the exact model, dataset, row
   count, hardware flavor, timeout, output repo, and max cost first. A useful
   first real training scale is `8k` to `16k` UltraChat rows before spending on
   a larger corpus.
4. Treat the trained sidecar as a Hugging Face/package artifact with explicit
   base-model, tokenizer, logical stage, tap-layer, spec-layer, draft-vocab, and
   checksum metadata.
5. Record acceptance, equivalent accept length, and latency simulation from the
   same eval prompts before attempting native speed claims.
6. Only after that, evaluate custom large MoE targets. Very large MoE models
   need activation-capture support and are not the right first scaling proof.
