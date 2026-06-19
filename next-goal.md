# Next Goal: Improve Qwen480 S8 SPD Sidecar Quality

This file is disposable. Durable evidence belongs in `evals/spd/README.md` and
`docs/skippy/speculative_decoding.md`.

## One-Line Goal

Train and qualify a larger Qwen3-Coder-480B S8 native SPD sidecar on Hugging
Face, then only move to an HF meshlet if held-out package-backed serving shows
real candidate-token round-trip savings under the same logical topology.

## Current Checkpoint

- Completed HF Job: `meshllm/6a3593cf3093dba73ce2a78f`.
- Artifact repo/path:
  `meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/runs/native-package-fresh`.
- Exact package:
  `meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`.
- Logical topology: S8,
  `stage_layer_boundaries=8,16,24,32,40,48,55,62`.
- Required taps: `[0,8,16,24,32,40,48,55,62]`.
- Smoke map used: `CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`.
- Downloaded reports:
  `/private/tmp/spd-qwen480-smoke-existing-completed/runs/native-package-fresh/`.

## Evidence From The Completed Smoke

- Baseline/SPD content matched on `8 / 8` prompts.
- Tap return failures: `0`.
- Tap record failures: `0`.
- Ignored taps: `0`.
- Inline probes: `32 / 32` produced proposals.
- Proposal miss reasons: all `null`.
- Missing taps: none.
- Rolling proposals: `32` proposed, `0` accepted, `32` rejected.
- Optimistic tokens committed: `0`.
- Pipeline/economics: `0` saved versus `32` unsaved candidate-token round
  trips; `paper_like_speedup_vs_serial_split=0`.
- Held-out scorer for the tiny artifact: `2 / 8` top-1 and `5 / 8` top-4.
- Train summary overfit the tiny train set: `final_argmax_acc=1.0`,
  `sample_count=32`.
- Mean sidecar head time for pre-target probes: about `399.9ms`.
- Mean normal downstream wait in the smoke: about `1515.6ms`.

Conclusion: the Qwen480 S8 request path, tap transport, package source, rolling
executor integration, and latency simulation path are alive. The blocker is
sidecar quality from too little native training data, not missing taps or
package-smoke mechanics.

## Success Gate

This goal is done only when a capped HF quality lane produces a larger trained
Qwen480 S8 sidecar and the package-backed held-out smoke shows:

- matched baseline/SPD content;
- zero tap return, tap record, and ignored-tap failures;
- nonzero accepted rolling proposals;
- more saved than unsaved candidate-token round trips.

If it clears that gate, the next goal becomes a short HF meshlet spike: run a
coordinator, local stage servers, the SPD sidecar, and OpenAI frontend as
separate processes inside one HF Job, optionally with artificial latency, to
validate lifecycle and pipeline economics before spending time on multi-HF-job
transport.

## Immediate Next Work

1. Submit the larger native-package-fresh quality profile only with explicit
   spend approval:
   - run on HF, not local M4;
   - same Qwen480 package and S8 topology;
   - train prompts at least `256`, preferably `512` if the cap still makes
     sense;
   - held-out prompts at least `64`;
   - verify steps `4`;
   - keep native package-first training and no full Qwen480 Transformers load;
   - print package, topology, prompt counts, hardware, timeout, output repo,
     and max cost.
   - Done: `/tmp/spd-qwen480-s8-quality-native-package-fresh-plan.json`,
     SHA256
     `563f142b265067cdda806a9f1ff29fa8743deddca51d48f2e9c829bc93972465`.
     The plan uses `512` train prompts, `64` held-out prompts, `4` verify
     steps, `rtx-pro-6000x4`, timeout `4.5h`, and max cost `$49.49991`.
2. Inspect the generated command graph and confirm it still avoids:
   - `AutoModelForCausalLM`;
   - `hf_train_eval_qwen06.py`;
   - `spd-live-tap-parity`;
   - warm-start/full-HF reference paths.
   - Done for the dry run above: `rg` found no matches for
     `AutoModelForCausalLM`, `hf_train_eval_qwen06`, `spd-live-tap-parity`, or
     `from_pretrained(`.
3. Before submitting spend, report the planned cap and expected risk:
   - `rtx-pro-6000x4` has worked mechanically and costs about `$11/hr`;
   - a `4.5h` timeout is about `$49.50`;
   - the risk is whether the larger capture/train/smoke reaches completion
     under the cap, not whether the old reference path is being used.
   - Submitted HF Job `meshllm/6a35cdc03093dba73ce2a9ad` after explicit
     spend approval. It is capped by the `4.5h` HF timeout on
     `rtx-pro-6000x4`, for planned max cost `$49.49991`.
   - Job input bundle:
     `job-inputs/20260619T231327Z-5b98182e/`, Hub revision
     `cf9e608e208f14708ebd826ce663dc607147fe6f`.
   - Patch SHA256:
     `9dda3b489a4a17b0cec69d892125613cd7a0b63177e5bc925e406d4d9af4c0bb`.
   - Launch env includes
     `SMOKE_STAGE_BACKEND_DEVICES=CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`
     to avoid repeating the old two-stages-per-GPU package-smoke OOM.
4. Do not dispatch an HF meshlet yet. Meshlet is a follow-on only after
   package-backed held-out serving saves more candidate-token round trips than
   it wastes, with matched content and zero tap failures.

## Why Not Meshlet Yet

The first single-job HF meshlet should validate process lifecycle, package
materialization, tap return, SPD proposal/verification, rolling cleanup, and
pipeline economics in one HF Job. It is useful only once there is a candidate
sidecar. The current sidecar proposes but accepts nothing, so a meshlet would
mostly re-measure known bad sidecar quality.
