# Skippy SPD Proof Notes

This directory is the public, reproducible handoff for the Skippy Speculative
Pipeline Decoding (SPD) proof.

SPD is treated here as a separate trained sidecar head. It proposes draft tokens
from selected target-model hidden states; the target model still verifies every
accepted token. The work in this directory proves the training/evaluation path
and records the artifact contract Skippy needs before serving the head from
Rust.

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
  model.
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
tap-aligned Skippy stage count. For speed interpretation, treat the accepted
token count as the first-order count of future split-stage verifier trips that
can be removed from the critical path only if the rolling executor overlaps
those verifier trips. The simple benchmark math is: count baseline emitted
tokens and split-stage trips, count accepted speculative tokens/windows from
the same prompt set, derive the paper-style critical-path reduction from the
rolling trace or `paper_pipeline_estimate`, then compare that upper bound with
the measured `decode_speedup_spd_vs_baseline`. Sidecar forward/head time,
tap-collection time, transport latency, rejected windows, and rollback drains
come directly out of the same report and explain the gap between the paper
estimate and measured wall/decode speed. If accepted proposals are zero, the
round-trip savings are zero and the only valid expectation is slowdown from
SPD overhead. `summary.rolling_trace_replay` replays observed
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
- optional safetensors serving checkpoint path, size, checksum, tensor count,
  and dtype

Rust validation lives in `crates/skippy-runtime/src/spd.rs`.
Safetensors parsing and BF16/F32/I64 payload reads live in
`crates/skippy-runtime/src/spd/safetensors.rs`.
The constrained Qwen fixture forward path lives in
`crates/skippy-runtime/src/spd/qwen.rs`.
The tap-row-to-`cur_in` projection bridge lives in
`crates/skippy-runtime/src/spd/tap_input.rs`.

## Next Engineering Steps

1. Train or fetch a topology-matched sidecar for the real two-stage product
   split before making the next speed claim. For Qwen3.5-4B that means
   `num_stages=2`, `stage_layer_boundaries=16,32`, and tap rows
   `0,16,32;0,16`; the current pretrained S4/L4 sidecar is intentionally
   excluded from this test. For the Mesh-native Qwen3-8B split just observed,
   the sidecar target is `Qwen/Qwen3-8B` with `num_stages=2` and
   `stage_layer_boundaries=23,36` unless a manifest-driven topology constraint
   is added first.
2. Move from the completed one-worker CPU LAN correctness proofs, the completed
   two-stage Metal baseline, and the completed local package-backed SPD smoke to
   a real speed gate: Mesh-resolved package materialization on both nodes, stage
   0 plus sidecar on the coordinator, stage 1 on the worker, paired
   baseline/SPD content and timing, and no one-worker all-Metal
   oversubscription. Do not call it a speed proof until rolling replay is back
   to `0` missing / `0` out-of-order proposals on the measured prompt set,
   oldest rejection drains are explained as sidecar-quality misses, and the
   report shows useful overlap rather than same-worker stage contention.
3. Run a larger local `spd-openai-smoke --prompt-file ...` sweep to measure
   acceptance distribution, rollback frequency, rolling gaps, and
   `summary.paper_pipeline_estimate` across prompt types. Report accepted
   speculative tokens/windows, estimated critical-path split-stage round trips
   saved, measured decode speedup, and the sidecar/tap/transport overhead that
   remains after those savings.
4. Use injected downstream delay only as a bounded diagnostic while no separate
   worker is available; do not report it as distributed speedup.
5. When another worker is available, rerun `spd-openai-smoke` with explicit
   `--stage-hosts`, pre-materialized package directories or Mesh-resolved
   package artifacts, and ordinary split baseline/SPD pairs to test real
   wall-clock speed.
6. Add an SPD sidecar package workflow around the Python reference trainer:
   plan logical tap topology, train, eval `L'_acc`, export safetensors/manifest,
   validate Rust parity, then publish sidecar metadata alongside Skippy model
   artifacts.

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
