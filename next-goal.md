# Next Goal: Improve Qwen480 S8 SPD Sidecar Quality

This file is disposable. Durable evidence belongs in `evals/spd/README.md` and
`docs/skippy/speculative_decoding.md`.

## One-Line Goal

Train and qualify a larger Qwen3-Coder-480B S8 native SPD sidecar on Hugging
Face, then only move to an HF meshlet if held-out package-backed serving shows
real candidate-token round-trip savings under the same logical topology.

## Current Checkpoint

- Active HF Job: `meshllm/6a35fb70953ed90bfb94547c`, created
  2026-06-20 02:31:12 UTC. It is the bounded mixed-data 8k native-Q4 quality
  lane on `rtx-pro-6000x4`, timeout `3.9h`, planned max cost `$42.899922`.
  Latest observed status at 2026-06-20 02:53:44 UTC: `RUNNING`,
  `runningSecs=1084`, estimated running cost about `$3.31`.
  It has passed bootstrap, pinned checkout, patch apply, CUDA/Rust release
  build, full 69-file Qwen480 package download, and bounded prompt build, and
  is now in native CUDA capture logs.
- Active job input bundle:
  `meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8/job-inputs/20260620T023047Z-594c0d00/`,
  uploaded at Hub commit `8d5cd9141a88ac12b300b26c55a2dd5a2680aeba`.
- Patch base/head: base `f87e69bf9daf88a0b48040c32fd0a06fffea4029`,
  head `d4c12243db1fab71b38716979a4ba2d04563130d`.
  Patch SHA256:
  `d20f6eb5235a4f549356417459f541b284cab990740d3bfb070514f24d9dde02`.
- Submitted pinned plan SHA256:
  `c5692cc64cf753ae8091a89cefd95ec8879c89fe059ba9f79a9e6f7d30e8e5b7`.
  The plan is the paper-aligned Qwen480 S8 mixed-data run plus
  `--max-source-rows 12000`, so prompt preparation cannot burn the GPU cap by
  tokenizing every row in million-row source datasets before selecting prompts.
- Canceled HF Job: `meshllm/6a35f141953ed90bfb945409`, submitted
  2026-06-20 01:47:45 UTC. It completed bootstrap, pinned checkout, patch
  apply, CUDA/Rust release build, and full Qwen480 package download, then
  entered prompt-token building. It was canceled intentionally because
  `build_hf_prompt_tokens.py` was reading/tokenizing all source rows before
  selection. Estimated running cost at cancellation was about `$6.64`; together
  with the replacement cap, the lane stays within the original `$50` intent.
- Latest completed HF Job: `meshllm/6a35cdc03093dba73ce2a9ad`.
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
- Training scale: `512` train prompts x `4` verify steps = `2048` native-Q4
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
SmolTalk-Chinese, with max length `2048`, LR `1e-4`, linear decay, and one
epoch. The completed Qwen480 quality lane is only `2048` native-Q4 train samples,
so it is a first production-path quality signal, not sufficient paper-scale
evidence.

The next goal remains the same topology but with a more paper-faithful
native-Q4 KD/data lane: broaden beyond UltraChat-only, preserve a frozen
token-line-disjoint held-out gate, train from native verifier logits rather
than BF16 full-model teachers, use a frequency-built `32k` draft vocabulary
from the selected training conversations instead of token IDs `0..31999`, and
judge readiness by broad held-out package-backed acceptance/economics before
any HF meshlet.

Second-opinion review agreed that the acceptance focus is correct, but warned
that the `96 / 256` native-teacher top-1 versus `0 / 256` served acceptance
gap is not explained by scale alone. The next evidence must separate:

- `teacher_top1`: agreement with the draft-restricted native-teacher argmax;
- `serving_target_top1`: agreement with the full-vocab greedy target when that
  target is inside the draft vocabulary;
- actual package-backed accepted/proposed proposals.

If `serving_target_top1` looks good on fixed rows but serving still accepts
zero, the missing piece is native Rust/Python fixture parity or live-row
alignment, not training data.

Second-opinion acceptance gate: if this bounded 8k lane still serves `0`
accepted proposals, do not jump directly to `16k`/`64k`/paper-scale data. First
run a tiny Qwen480 S8 overfit-to-serving-prompts proof on the exact package
topology. If an intentionally overfit head accepts nonzero proposals in
package-backed serving, the request path is aligned and data scale is the next
lever. If even the overfit head accepts `0`, the blocker is row/projection/live
tap alignment or Rust/Python forward parity, not insufficient data.

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
2. Before spending again, add or run a fixed-row native parity gate for the
   Qwen480 native-package-fresh lane:
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
     now runs `skippy-bench spd-fixture-parity` before package smoke. This has
     not yet been run on Qwen480 artifacts because the previous local download
     retained only final JSONs and the serving bundle, not held-out corpus and
     teacher tensors.
3. The next spend-bearing candidate is running as bounded HF Job
   `meshllm/6a35fb70953ed90bfb94547c`. The replacement dry-run plan is:
   `/tmp/spd-qwen480-s8-quality-8k-native-package-fresh-mixed-balanced-bounded-plan.json`,
   SHA256
   `91d09809c79ddd0db0a126c659cc2de124cbdeaa21f8fa26e0495b95071fa426`.
   It keeps the same Qwen480 S8 package/topology, uses `8192` native-Q4 train
   samples and `128` held-out prompts, builds a corpus-frequency `32k` draft
   vocabulary from selected training conversations, trains KL-only against
   captured native verifier logits, caps source-row preprocessing at `12000`
   rows per dataset, and caps planned cost at `$42.899922` on
   `rtx-pro-6000x4`. First checks after start: bootstrap script fetch,
   checkout of patch base `f87e69bf9daf88a0b48040c32fd0a06fffea4029`, patch
   apply, CUDA build, package capture, `export_product_parity_fixture.py`,
   `skippy-bench spd-fixture-parity`, then package-backed smoke.
4. If the 8k run has clean mechanics and low but nonzero acceptance, scale the
   same recipe to `16k`, then `64k`, and only then toward the paper's mixed-data
   scale. If the 8k run still has `0` served acceptance, first run the tiny
   Qwen480 overfit existence proof above. The paper's reported run is about
   `1M` selected conversations, `1.2M` filtered samples, max length `2048`, one
   epoch, LR `1e-4`, linear decay, and KL against the frozen target. Our
   current Qwen480 run is only `2048` native-Q4 samples, so it cannot answer
   whether SPD quality works at paper scale.
5. Acceptance is the gate. A run is useful only if it reports all of:
   `full_vocab_target_in_draft_scope`, `serving_target_top1/top4`,
   package-backed accepted/proposed proposals, saved/unsaved candidate-token
   round trips, and a latency simulation using measured sidecar cost.

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

8. Bounded replacement submitted:
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
   It keeps the same Qwen480 S8 package/topology and 8k native-Q4 sample
   target, adds `--max-source-rows 12000`, reduces timeout to `3.9h`, and caps
   planned cost at `$42.899922`. With the canceled run's estimated `$6.64`
   running cost, this stays under the original `$50` intent. Observed gate:
   bootstrap, patch apply, CUDA/Rust release build, full package download
   (`69 / 69` files in about `3.5min`), and bounded prompt build all passed.
   At 2026-06-20 02:53:44 UTC the job was running native CUDA capture logs.
   Next checks are train/held-out capture completion, conversion, head-only
   train/score, product fixture parity, and package-backed
   acceptance/economics.

## Why Not Meshlet Yet

The first single-job HF meshlet should validate process lifecycle, package
materialization, tap return, SPD proposal/verification, rolling cleanup, and
pipeline economics in one HF Job. It is useful only once there is a candidate
sidecar. The current sidecar proposes but accepts nothing, so a meshlet would
mostly re-measure known bad sidecar quality.
