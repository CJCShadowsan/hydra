# SPD Progress Brief

Updated: 2026-06-20 05:32 UTC / 15:32 AEST.

## Bottom Line

SPD is mechanically alive in Skippy, but Qwen480 S8 is not proven useful yet.
The current blocker is acceptance rate from the trained sidecar, with one
important guardrail: do not spend more data until fixed-row Rust/Python parity
and package-backed smoke prove that serving is using the trained head correctly.

The paper points in the same direction: the predictor quality comes from a
frozen-target KL training run over much more data than we have tried so far.
Our Qwen480 runs are useful product-path evidence, but they are still tiny
compared with the paper-scale recipe.

## What Has Worked

- Native Skippy SPD request path exists: tap return, rolling proposals,
  verification, rejection/reset handling, and exact-content checks have run.
- Pretrained Qwen3.5/Qwen3-family SPD mechanics worked well enough to validate
  the executor shape, including local and LAN-style rolling runs with clean tap
  behavior.
- The real Qwen3-8B two-stage `23,36` path was exercised with native Q4
  serving: content matched, tap failures were zero, and package-backed/two-node
  smokes accepted some proposals (`17 / 90` local, `16 / 89` worker split).
- The Qwen480 HF path can now download the full layer package, capture native
  Q4 rows/logits, train head-only without loading the full base model through
  Transformers, export the serving head, and run most of the package-backed
  qualification flow.
- Qwen480 package-backed mechanics have been clean: baseline/SPD content
  matched and tap return/record/ignored failures were zero in the broad smoke.

## Data Tried So Far

- Qwen3-8B `23,36` BF16 reference train: about `15997` usable UltraChat rows,
  max length `2048`, one epoch, LR `1e-4`, draft top-k `4`. Reference eval was
  strong (`0.7013` aggregate acceptance), but native Q4 serving acceptance was
  much lower.
- Qwen480 S8 tiny lane: `32` native-Q4 train samples and `8` held-out samples.
  It proved plumbing only; held-out top-1/top-4 was `2 / 8` and `5 / 8`, and
  package serving accepted `0 / 32`.
- Qwen480 S8 broad lane: `512` train prompts x `4` verify steps =
  `2048` native-Q4 train samples; `64` held-out prompts =
  `256` held-out samples. Offline native-teacher top-1/top-4 was `96 / 256`
  and `129 / 256`; package-backed serving accepted `0 / 256`.
- Qwen480 S8 mixed-data bounded lane: `2048` train prompts x `4` verify steps =
  `8192` native-Q4 train samples; `128` held-out prompts =
  `512` held-out samples. Offline signal improved:
  `serving_target_top1=167 / 493`, `serving_target_top4=247 / 493`,
  `teacher_top1=168 / 512`, `teacher_top4=249 / 512`,
  `final_argmax_acc=0.25`. It failed before package smoke because Rust wrongly
  required sorted draft-token IDs.

Paper reference point: the paper used about `1.2M` filtered samples from mixed
chat data, max length `2048`, one epoch, LR `1e-4`, linear decay, and KL
against the frozen target. Our `2048` and `8192` Qwen480 samples are nowhere
near that scale.

## Current Run

Active HF Job: `meshllm/6a36251f3093dba73ce2ab39`
(`spd-qwen480-quality-8k-draft-vocab-fix`).

It reruns the 8k Qwen480 S8 plan with the Rust manifest fix that permits
frequency-ordered unique `draft_token_ids`. That matters because the head's
logits are positional over this draft vocab; sorting token IDs after training
would corrupt the mapping.

Latest checked state: still `RUNNING`; `hf jobs logs --tail 120` showed Rust
release compilation at 2026-06-20 05:38 UTC. It has not yet reached capture,
training, fixed-row parity, package-backed smoke, or acceptance reporting.
At 2026-06-20 05:42 UTC it had advanced to `setup[23]` and was downloading the
69-file Qwen480 layer package snapshot.

## What Is Not Proven

- No Qwen480 S8 sidecar has yet shown nonzero package-backed served acceptance
  on a broad held-out run.
- The improved 8k offline result has not yet reached package smoke.
- We do not yet know whether Qwen480's zero served acceptance was mainly too
  little data, a row/projection/live-tap alignment bug, an underfit training
  recipe, or some combination of those.
- We do not have a real Qwen480 speedup claim. The right near-term metric is
  accepted/proposed proposals and saved versus unsaved candidate-token round
  trips, not wall-clock speed.

## Next Gate

1. Let the current 8k draft-vocab-fix HF run reach fixed-row parity.
2. If parity fails, fix alignment before buying more data.
3. If parity passes, run a live-row alignment gate before package smoke. The
   next retry plan now has `live_row_parity`, using the real product-row
   context tokens in the parity fixture plus `spd-live-tap-parity
   --skip-target-verification` with per-stage device placement.
4. Require package-backed smoke to report accepted/proposed proposals plus
   saved/unsaved candidate-token round trips.
5. If package smoke still accepts `0`, run the overfit-to-serving-prompts
   control on the exact Qwen480 S8 topology. Nonzero acceptance there means data
   scale is the next lever; zero means serving alignment is still wrong.
6. Only after that, scale the same native-Q4 mixed-data recipe to `16k`,
   `64k`, then closer to paper-scale if the signal keeps improving.
