# Next Goal: Improve Qwen480 S8 SPD Sidecar Quality

This file is disposable. Durable evidence belongs in `evals/spd/README.md` and
`docs/skippy/speculative_decoding.md`.

## One-Line Goal

Train and qualify a larger Qwen3-Coder-480B S8 native SPD sidecar on Hugging
Face, then only move to an HF meshlet if held-out package-backed serving shows
real candidate-token round-trip savings under the same logical topology.

## Current Checkpoint

- Active HF Job to monitor: `meshllm/6a3611dd953ed90bfb945575`, created
  2026-06-20 04:06:53 UTC, label `spd-qwen480-quality-8k-diagnostic`,
  `rtx-pro-6000x4`, timeout `3.9h`, max planned cost `$42.899922`.
- Latest HF Job: `meshllm/6a35fb70953ed90bfb94547c`, created
  2026-06-20 02:31:12 UTC, label `spd-qwen480-quality-8k-bounded`, run
  `20260620T023047Z-594c0d00`. It ended `ERROR` after about `3975s` on
  `rtx-pro-6000x4`, with estimated spend about `$12.15`, under its
  `$42.899922` timeout cap.
- It passed bootstrap, pinned checkout, patch apply, CUDA/Rust release build,
  full Qwen480 package download, bounded mixed prompt build, native capture,
  train/held-out conversion, head-only training, held-out scoring, serving
  export, product parity fixture export, and serving fixture export.
- It failed before package-backed smoke at `rust_fixture_parity[0]` with
  `failed to reconstruct SPD fixture cur_in from tap inputs`. That means the
  8k head has not produced a served acceptance result yet.
- Active job input bundle:
  `meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/job-inputs/20260620T023047Z-594c0d00/`,
  uploaded at Hub commit `8d5cd9141a88ac12b300b26c55a2dd5a2680aeba`.
- Patch base/head: base `f87e69bf9daf88a0b48040c32fd0a06fffea4029`,
  head `d4c12243db1fab71b38716979a4ba2d04563130d`.
  Patch SHA256:
  `d20f6eb5235a4f549356417459f541b284cab990740d3bfb070514f24d9dde02`.
- Submitted pinned plan SHA256:
  `c5692cc64cf753ae8091a89cefd95ec8879c89fe059ba9f79a9e6f7d30e8e5b7`.
  Replacement dry-run plan:
  `/tmp/spd-qwen480-s8-quality-8k-native-package-fresh-mixed-balanced-bounded-plan.json`,
  SHA256
  `91d09809c79ddd0db0a126c659cc2de124cbdeaa21f8fa26e0495b95071fa426`.
- Current diagnostic retry has been submitted and is the active job to monitor:
  HF Job `meshllm/6a3611dd953ed90bfb945575`, created
  2026-06-20 04:06:53 UTC, label
  `spd-qwen480-quality-8k-diagnostic`. It was last observed `RUNNING` at
  2026-06-20 04:56 UTC. It has passed setup/release build, downloaded the full
  `69`-file / `276G` Qwen480 package, and entered the mixed-data plan. The HF
  log endpoint currently returns no useful tail lines, but `hf jobs stats`
  showed real work rather than idle time: about `309.8GB / 1.0TB` host memory
  and roughly `69GB` allocated on each of four RTX PRO 6000 GPUs, with live GPU
  utilization. No fixed-row parity, package-smoke, or acceptance result has
  appeared yet. Do not submit a duplicate while this job is active. The
  refreshed dry-run plan is
  `/tmp/spd-qwen480-s8-quality-8k-native-package-fresh-mixed-balanced-bounded-diagnostic-plan.json`,
  SHA256
  `44a92e8c759af9304f790a72bb02f196443b0ab1ebe112dc0b6589ca8f0db244`.
  Uploaded no-compute input bundle:
  `meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/job-inputs/20260620T040137Z-diagnostic/`,
  Hub commit `9683bc4cf28df5b8dbb5b14ffa8428aaded664dd`. Bundle manifest was
  downloaded back and verified; uploaded plan SHA256 is still
  `44a92e8c759af9304f790a72bb02f196443b0ab1ebe112dc0b6589ca8f0db244`, patch
  SHA256 is
  `62a3d29e0c4793c629e77a7a50d8156b8bbf1f4bfa97ce5d5001585c7126724f`, and
  bootstrap SHA256 is
  `89c389d02b4c10a9932b07fd4782a518e4a0bb4341c24894beaa0c7a81dc5946`.
  `evals/spd/bootstrap_qwen480_s8_native_job.sh` can now consume an uploaded
  pinned plan via `PLAN_PATH_IN_REPO`, so the next retry does not accidentally
  regenerate the older UltraChat-only default plan inside HF.
  Local confirm-gated submit helper:
  `/tmp/submit-spd-qwen480-8k-diagnostic.py`. Its dry run prints
  `rtx-pro-6000x4`, timeout `3.9h`, and max planned cost `$42.899922`; it was
  submitted with `--confirm` after the spend-capped goal resumed.
- Bounded 8k quality signal: `8192` native-Q4 train samples, `512` held-out
  samples, `7979 / 8192` train labels in draft-vocab scope, and
  `493 / 512` held-out labels in scope. Final train hard-label accuracy was
  `0.25`. Held-out native-teacher top-1/top-4 was `168 / 512` and
  `249 / 512`; serving-target top-1/top-4 for in-scope full-vocab targets was
  `167 / 493` and `247 / 493`. This is a materially better offline signal than
  the earlier lane, but not a qualified sidecar until fixed-row parity and
  package-backed served acceptance clear.
- Canceled HF Job: `meshllm/6a35f141953ed90bfb945409`, submitted
  2026-06-20 01:47:45 UTC. It completed bootstrap, pinned checkout, patch
  apply, CUDA/Rust release build, and full Qwen480 package download, then
  entered prompt-token building. It was canceled intentionally because
  `build_hf_prompt_tokens.py` was reading/tokenizing all source rows before
  selection. Estimated running cost at cancellation was about `$6.64`; together
  with the replacement cap, the lane stays within the original `$50` intent.
- Latest fully completed broad package-backed smoke remains HF Job
  `meshllm/6a35cdc03093dba73ce2a9ad`.
- Artifact repo/path:
  `meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/runs/native-package-fresh`.
- Exact package:
  `meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`.
- Logical topology: S8,
  `stage_layer_boundaries=8,16,24,32,40,48,55,62`.
- Required taps: `[0,8,16,24,32,40,48,55,62]`.
- Smoke map used: `CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`.
- Downloaded final reports:
  `/private/tmp/spd-qwen480-quality-final-json/`.
- Completed-smoke training scale: `512` train prompts x `4` verify steps = `2048` native-Q4
  train samples; `64` held-out prompts = `256` held-out samples.
- Held-out offline score: native-teacher top-1 `96 / 256`, top-4 `129 / 256`.
- Serving head: `8,723,214,136` bytes, SHA256
  `5cf3c15c54919414809cf409d252c5c4b0fa2b5ec084d91d4966e54976e75936`.

## Evidence From The Latest Broad Smoke

- Baseline/SPD content matched on `64 / 64` prompts.
- Tap return failures: `0`.
- Tap record failures: `0`.
- Ignored taps: `0`.
- Rolling proposals: `256` proposed, `0` accepted, `256` rejected.
- Optimistic tokens committed: `0`.
- Pipeline/economics: `0` saved versus `256` unsaved candidate-token round
  trips; latency simulation reports `paper_like_speedup_vs_serial_split=0.0`.
- Mean sidecar cost used by the latency simulator: about `395.8ms`.

Conclusion: the Qwen480 S8 request path, tap transport, package source, rolling
executor integration, and latency simulation path work at broad held-out scale.
The blocker is acceptance, but do not reduce that to "more rows" yet. The
current contradiction is serious: offline held-out scoring reported useful
top-1 signal, while package-backed serving accepted `0 / 256`. Before spending
again, close the structural acceptance checks below: serving/Python parity on
fixed native rows, corpus-derived draft-vocab coverage, and then data scale.
Do not dispatch a meshlet for this sidecar.

## Prior Tiny Smoke Evidence

Previous HF Job: `meshllm/6a3593cf3093dba73ce2a78f`. Downloaded reports:
`/private/tmp/spd-qwen480-smoke-existing-completed/runs/native-package-fresh/`.

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

## Acceptance-Rate Focus

Do not spend the next iteration re-proving taps, package loading, or meshlet
lifecycle unless a new request-path failure appears.
The paper trains the frozen-target SPD speculation module with KL distillation
over about `1.2M` filtered samples from ShareGPT, UltraChat, SmolTalk, and
SmolTalk-Chinese, with max length `2048`, LR `1e-4`, linear decay, one epoch,
and simulated pipeline occupancy over the same multi-depth layout used at
inference. The completed Qwen480 broad-smoke lane used only `2048` native-Q4
train samples; the latest bounded 8k lane used `8192`. Both are still tiny
relative to the paper and should be treated as production-path quality signals,
not enough evidence that SPD quality fails at proper scale.

The next goal remains the same topology but with a more paper-faithful
native-Q4 KD/data lane: broaden beyond UltraChat-only, preserve a frozen
token-line-disjoint held-out gate, train from native verifier logits rather
than BF16 full-model teachers, use a frequency-built `32k` draft vocabulary
from the selected training conversations instead of token IDs `0..31999`, and
judge readiness by broad held-out package-backed acceptance/economics before
any HF meshlet.

Second-opinion review agreed that the acceptance focus is correct, but warned
that the earlier `96 / 256` native-teacher top-1 versus `0 / 256` served
acceptance gap is not explained by scale alone. The latest 8k lane improves the
offline picture (`167 / 493` serving-target top-1, `247 / 493` top-4), but it
failed before smoke at fixture tap-input reconstruction. The next evidence must
separate:

- `teacher_top1`: agreement with the draft-restricted native-teacher argmax;
- `serving_target_top1`: agreement with the full-vocab greedy target when that
  target is inside the draft vocabulary;
- actual package-backed accepted/proposed proposals.

If `serving_target_top1` looks good on fixed rows but serving still accepts
zero, the missing piece is native Rust/Python fixture parity or live-row
alignment, not training data.

Second-opinion acceptance gate: first fix the failed fixed-row parity
diagnostic so the 8k head can reach package-backed smoke. If it then serves
`0` accepted proposals, do not jump directly to `16k`/`64k`/paper-scale data.
Run a tiny Qwen480 S8 overfit-to-serving-prompts proof on the exact package
topology. If an intentionally overfit head accepts nonzero proposals in
package-backed serving, the request path is aligned and data scale is the next
lever. If even the overfit head accepts `0`, the blocker is row/projection/live
tap alignment or Rust/Python forward parity, not insufficient data.

Second-opinion update after the diagnostic patch: keep treating the
`cur_in` reconstruction failure as potentially the visible form of the same
alignment bug that made the earlier broad smoke serve `0` accepted proposals.
The retry must prove small fixed-row reconstruction diff, not merely produce a
more verbose error. If the failure localizes to Rust reconstruction or fixture
export, prefer a cheaper artifact/corpus reuse path over repeating native
capture/train. Only scale rows after parity is clean and an overfit
serving-prompts control can accept nonzero proposals on the exact served
topology.

Second-opinion update after submitting `meshllm/6a3611dd953ed90bfb945575`:
the active diagnostic is a measurement instrument, not a quality fix by
itself. Do not scale rows while fixed-row `cur_in` reconstruction is failing.
Use the diagnostic context to fix raw tap offsets/widths, stage-HF-index
mapping, or projection-basis alignment first. Only after fixed-row parity
passes and package-backed smoke still serves `0` should the
overfit-to-serving-prompts control run; only after served acceptance clears
with saved candidate-token round trips should `16k`/`64k`/paper-scale data be
submitted.

Second-opinion update at 2026-06-20 04:56 UTC: active job stats show real GPU
work, so keep waiting for the diagnostic result rather than canceling or
launching a duplicate. If fixed-row parity passes but served acceptance is
still `0`, the next spend gate is the already prepared
`--overfit-serving-prompts` proof, not a larger data jump. Nonzero served
acceptance from that overfit proof means request-path alignment is good and
data scale is the lever; zero served acceptance from an overfit head means the
blocker is still row/projection/live-tap alignment.

Second-opinion update at 2026-06-20 05:11 UTC: keep treating sufficient data as
the likely eventual quality lever, but not the next spend gate. The active 8k
head has not served yet, so comparing its improved offline score to the older
`0 / 256` served broad-smoke result is not a controlled data-scale test. The
next decisive evidence is fixed-row Rust/Python parity and, if served
acceptance is still zero after that, a tiny overfit-to-serving-prompts control
on the exact Qwen480 S8 package topology. A nonzero served acceptance result
from that overfit control makes `16k`/`64k`/paper-scale data the right lever; a
zero result means data scale is a red herring until row/projection/live-tap
alignment is fixed.

Raw-log update at 2026-06-20 05:03 UTC: the HF CLI log tail was empty, but the
underlying SSE endpoint
`https://huggingface.co/api/jobs/meshllm/6a3611dd953ed90bfb945575/logs`
returned buffered events when called with `HF_TOKEN`. The job reached
`capture[0]` at 2026-06-20 04:29:01 UTC after prompt building completed with
`2048` train prompts, `128` held-out prompts, and
`draft_vocab_unique_train_tokens=45576`. The latest raw log timestamp was
2026-06-20 05:03:07 UTC inside native capture, with repeated
`init: embeddings required but some input tokens were not marked as outputs ->
overriding` messages. No conversion, training, fixed-row parity, package
smoke, or acceptance result has appeared yet.

Raw-log/stats update at 2026-06-20 05:11 UTC: a fresh SSE sample still showed
the latest log event inside native capture at 2026-06-20 05:09:48 UTC, with no
command boundary for conversion, training, parity, package smoke, or
acceptance. `hf jobs stats` continued to show the job running with about
`305`-`314GB / 1.0TB` host memory and live GPU memory/use cycling through the
capture phase. Do not submit a duplicate while this job is active.

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

1. Do not dispatch an HF meshlet or another smoke-existing retry. The mechanics
   are already clean; the current blocker is the Qwen480 S8 sidecar acceptance
   rate.
2. Fixed locally: the fixed-row parity diagnostic now exposes the underlying
   row/projection/tensor context instead of only
   `failed to reconstruct SPD fixture cur_in from tap inputs`. It reports the
   row, position, stage id, HF tap indices, concat tensor shape/length, hidden
   size, stage boundaries, projection tensor name, input width, and weight
   shape. Local gates passed for the focused tap-input tests, `cargo check -p
   skippy-runtime`, `cargo clippy -p skippy-runtime --all-targets -- -D
   warnings`, `bash -n` on the Qwen480 bootstraps, `git diff --check`, and a
   clean patch-apply check against `origin/codex/skippy-spd-proof`.
3. Then rerun or resume the fixed-row native parity gate for the Qwen480
   native-package-fresh lane:
   - compare Python head-only top-k on saved native rows with Rust serving
     proposals for the same rows;
   - report `teacher_top1`, `serving_target_top1`, draft-vocab target coverage,
     and actual served accepted/proposed separately;
   - if fixed-row Python predicts the served target but Rust rejects all rows,
     fix row alignment/forward parity before buying more training data.
   - This gate is necessary but not sufficient: if it passes and package smoke
     still accepts zero, add a live-row reconstruction check that compares
     request-time tap-projected `cur_in` against the saved native corpus row for
     the same context.
   - Implemented for the next HF lane: `export_product_parity_fixture.py`
     writes `spd-product-parity-fixture.safetensors`, and
     `plan_hf_spd_qualification.py --qualification-mode native-package-fresh`
     now runs `skippy-bench spd-fixture-parity` before package smoke. The first
     Qwen480 execution of this gate failed at tap-input reconstruction, so
     diagnostics must come before data scale.
4. The next spend-bearing candidate should be the refreshed diagnostic retry,
   not a larger data jump yet. It keeps the same Qwen480 S8 package/topology,
   uses `8192` native-Q4 train samples and `128` held-out prompts, builds a
   corpus-frequency `32k` draft vocabulary from selected training
   conversations, trains KL-only against captured native verifier logits, caps
   source-row preprocessing at `12000` rows per dataset, and caps planned cost
   at `$42.899922` on `rtx-pro-6000x4`. If fixed-row parity fails again, use
   the new row/projection context to fix alignment. If parity passes and
   package smoke still accepts `0`, run the overfit-to-serving-prompts control
   before buying `16k`, `64k`, or paper-scale data.
   The input bundle has been uploaded and the capped retry has been submitted
   as HF Job `meshllm/6a3611dd953ed90bfb945575`. Next action is to monitor
   bootstrap, pinned-plan download, patch apply, capture/train/score,
   fixed-row parity, and package-backed smoke; do not submit another retry
   unless this one exits or is intentionally canceled.
5. If the 8k run has clean mechanics and low but nonzero acceptance, scale the
   same recipe to `16k`, then `64k`, and only then toward the paper's mixed-data
   scale. If the 8k run reaches smoke and still has `0` served acceptance, first
   run the tiny Qwen480 overfit existence proof above. The paper's reported run
   is about `1M` selected conversations, `1.2M` filtered samples, max length
   `2048`, one epoch, LR `1e-4`, linear decay, and KL against the frozen target.
   Our current Qwen480 lanes are only `2048` and `8192` native-Q4 samples, so
   they cannot answer whether SPD quality works at paper scale.
6. Acceptance is the gate. A run is useful only if it reports all of:
   `full_vocab_target_in_draft_scope`, `serving_target_top1/top4`,
   package-backed accepted/proposed proposals, saved/unsaved candidate-token
   round trips, and a latency simulation using measured sidecar cost.
7. The overfit fallback is now an explicit no-spend dry run, not an implicit
   data-size hack. `plan_hf_spd_qualification.py` accepts
   `--overfit-serving-prompts` for `native-package-fresh`, training on the same
   held-out product rows used by package-backed smoke. Verified dry-run:
   `/tmp/spd-qwen480-s8-overfit-serving-control-plan.json`, capped at about
   `$22.00` on `rtx-pro-6000x4` for `2h`, Qwen480 S8 topology, `8` held-out
   prompts x `4` verify steps, and no `AutoModelForCausalLM`,
   `hf_train_eval_qwen06.py`, `spd-live-tap-parity`, or `from_pretrained(`
   command-graph matches.

## Historical Work Log

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

5. `meshllm/6a35cdc03093dba73ce2a9ad` completed and failed the
   acceptance/economics gate while mechanics stayed clean. Final downloaded
   reports:
   `/private/tmp/spd-qwen480-quality-final-json/openai-heldout-rolling.json`
   and
   `/private/tmp/spd-qwen480-quality-final-json/latency-simulation.json`.
   Evidence:
   - baseline/SPD content matched on `64 / 64` prompts;
   - tap return failures `0`, tap record failures `0`, ignored taps `0`;
   - rolling windows proposed `256`, accepted `0`, rejected `256`;
   - optimistic requests `64`, accepted `0`, committed `0`;
   - saved candidate-token round trips `0`, unsaved `256`;
   - latency simulation totals report `paper_like_speedup_vs_serial_split=0`;
   - mean measured sidecar cost used by the simulator was about `395.8ms`.
   Conclusion: this proves package-backed request mechanics at broad held-out
   scale, but the sidecar is still not qualified. Next spend must be a
   data/recipe scale-up, not a smoke-existing rerun or meshlet:
   - first larger target: at least `8k` to `16k` native-Q4 train samples if it
     fits a capped lane, with `64` to `256` held-out prompts;
   - longer target: move toward paper-scale mixed data if early scaling improves
     broad held-out acceptance;
   - generate and pass a corpus-frequency `draft-token-ids.json` into native
     capture so the draft vocab is not the arbitrary `0..31999` token range;
   - keep KL against captured native draft-vocab logits as the core objective;
   - report `serving_target_top1/top4`, draft-vocab target coverage,
     package-backed accepted/proposed, saved/unsaved candidate-token round
     trips, and realistic latency simulation.
   - No-spend fallback dry run prepared:
     `/tmp/spd-qwen480-s8-quality-8k-native-package-fresh-paperlike-plan.json`,
     SHA256
     `5d59c3b025e457d979437171044823b9a92b4220a99678baac800292918a2816`.
     It uses the same Qwen480 S8 package/topology, `2048` train prompts with
     `4` verify steps (`8192` native-Q4 train samples), `128` held-out prompts,
     `physical-node-count=4`, capture map
     `CUDA0,CUDA0,CUDA1,CUDA1,CUDA2,CUDA2,CUDA3,CUDA3`, package-smoke map
     `CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3`, one epoch, LR `1e-4`,
     KL-only (`hard_label_weight=0`), `rtx-pro-6000x4`, `4.5h`, and planned
     max cost `$49.49991`. `rg` found no `AutoModelForCausalLM`,
     `hf_train_eval_qwen06`, `spd-live-tap-parity`, or `from_pretrained(` in
     the plan. Do not submit it until spend is explicitly approved.
   - The prompt-token builder and planner now accept comma-separated
     `--dataset`, `--dataset-split`, and optional `--dataset-config` values, so
     the next dry run can move toward the paper's mixed
     ShareGPT/UltraChat/SmolTalk data rather than staying UltraChat-only.

6. Updated no-spend, paper-aligned dry run prepared:
   `/tmp/spd-qwen480-s8-quality-8k-native-package-fresh-mixed-balanced-paperlike-plan.json`,
   SHA256
   `24e9d55378acc68f82f098dab0c954d23b68c0acda0e6bfdd4e804dfbd5ecc0c`.
   It keeps the same Qwen480 S8 package/topology, `2048` train prompts with
   `4` verify steps (`8192` native-Q4 train samples), `128` held-out prompts,
   `ctx_size=2048`, one epoch, LR `1e-4`, KL-only native teacher training,
   `rtx-pro-6000x4`, `4.5h`, and planned max cost `$49.49991`.
   Data sources are balanced round-robin across:
   `HuggingFaceH4/ultrachat_200k:train_sft`,
   `HuggingFaceTB/smoltalk:all/train`,
   `opencsg/smoltalk-chinese:train`, and
   `mlabonne/WizardLM_evol_instruct_70k-ShareGPT:train`.
   The prompt builder writes `draft-token-ids.json` from selected training
   conversations and native capture passes it with `--draft-token-ids-file`.
   `rg` found no `AutoModelForCausalLM`, `hf_train_eval_qwen06`,
   `spd-live-tap-parity`, or `from_pretrained(` in the plan. Do not submit it
   until spend is explicitly approved and the native parity/serving-target
   checks above are acknowledged.

7. Mixed-data 8k lane submitted after the spend-capped goal resumed, then
   canceled for prompt-preprocessing cost control:
   HF Job `meshllm/6a35f141953ed90bfb945409`, created
   2026-06-20 01:47:45 UTC, label `spd-qwen480-quality-8k`, run
   `20260620T014653Z-724af833`.
   Input bundle:
   `job-inputs/20260620T014653Z-724af833/`, upload commit
   `a297f50747afa0c15e5840b8e88d7410a1346fb7`.
   Local bundle:
   `/tmp/spd-qwen480-native-job-20260620T014653Z-724af833`.
   Patch SHA256:
   `55d002d14f77aab050edc0d13da3a08a84c8df5055ae3c0c860b5a50fb6c6704`;
   bootstrap SHA256:
   `39a62b2dfed65b3885d5e716b9e4b2316542e8ce0f42b671a13db73800e7b9ae`;
   submitted pinned plan SHA256:
   `bb7ab5c3816857df9bd97fd2ecc7ccc5e616bd70c4f904d03dbb9acd876e3b32`.
   The input bundle was token-fetch verified before submit, and the patch was
   checked locally with `git apply --check` against the exact pinned base
   `f87e69bf9daf88a0b48040c32fd0a06fffea4029`. It later completed bootstrap,
   pinned checkout, patch apply, CUDA/Rust release build, full package
   download, and entered `build_prompts[0]`. It was canceled because that
   prompt-builder invocation had no source-row cap and was tokenizing millions
   of source rows before selecting `2048` train prompts and `128` held-out
   prompts. This was not a capture, training, parity, or smoke failure.

8. Bounded replacement result:
   HF Job `meshllm/6a35fb70953ed90bfb94547c`, created
   2026-06-20 02:31:12 UTC, label `spd-qwen480-quality-8k-bounded`, run
   `20260620T023047Z-594c0d00`.
   Input bundle:
   `job-inputs/20260620T023047Z-594c0d00/`, upload commit
   `8d5cd9141a88ac12b300b26c55a2dd5a2680aeba`.
   Local bundle:
   `/tmp/spd-qwen480-native-job-20260620T023047Z-594c0d00`.
   Replacement plan:
   `/tmp/spd-qwen480-s8-quality-8k-native-package-fresh-mixed-balanced-bounded-plan.json`,
   SHA256
   `91d09809c79ddd0db0a126c659cc2de124cbdeaa21f8fa26e0495b95071fa426`.
   It kept the same Qwen480 S8 package/topology and 8k native-Q4 sample target,
   added `--max-source-rows 12000`, reduced timeout to `3.9h`, and capped
   planned cost at `$42.899922`. With the canceled run's estimated `$6.64`
   running cost, this stayed under the original `$50` intent. It passed
   bootstrap, patch apply, CUDA/Rust release build, full package download,
   bounded prompt build, native capture, conversion, head-only train/score,
   serving export, product fixture export, and serving fixture export. It ended
   `ERROR` at `rust_fixture_parity[0]` after about `3975s`, with estimated spend
   about `$12.15`, before package-backed smoke. Offline held-out improved to
   `167 / 493` serving-target top-1 and `247 / 493` top-4, but there is no
   served acceptance result for this 8k head.

## Why Not Meshlet Yet

The first single-job HF meshlet should validate process lifecycle, package
materialization, tap return, SPD proposal/verification, rolling cleanup, and
pipeline economics in one HF Job. It is useful only once there is a candidate
sidecar. The current sidecar proposes but accepts nothing, so a meshlet would
mostly re-measure known bad sidecar quality.
