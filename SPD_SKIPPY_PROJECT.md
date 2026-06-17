# SPD for Skippy Project Handoff

This branch captures the current public proof and next implementation steps for
running Speculative Pipeline Decoding (SPD) in Skippy.

It intentionally excludes private lab hosts, credentials, local IPs, and
machine-specific notes. Use it as a research/implementation handoff that another
engineer can reproduce from open models, open data, and the checked-in scripts.

## Source Paper

Paper: **Speculative Pipeline Decoding: Higher-Accuracy and Zero-Bubble
Speculation via Pipeline Parallelism**

- arXiv: `https://arxiv.org/abs/2605.30852`
- Reference code: `https://github.com/yuyijiong/speculative_pipeline_decoding`

The core idea is to combine pipeline parallel target-model execution with a
trained speculation module. The target model is partitioned into `n` pipeline
stages. While the target pipeline is processing one token per stage, the SPD
head consumes selected intermediate hidden states from the pipeline and proposes
future draft token(s). The target model still verifies the draft tokens, so with
verification enabled the output follows the base model's decoding path.

## Why This Matters for Skippy

Skippy already splits a model across staged runtimes. Ordinary split decoding is
sensitive to stage and network latency because each generated token must traverse
the full stage chain before the next target token is known.

SPD is interesting because it can fill the pipeline and amortize that stage/hop
latency across accepted speculative work. The quality question is whether the
sidecar head accepts enough tokens. The engineering question is whether Skippy
can expose the required hidden-state taps and verify proposals without breaking
target-model equivalence.

Headline current result: the pretrained `Qwen/Qwen3.5-4B` SPD head reported
`1230 / 1536` accepted draft flags on the local reference eval. The reference
summary reports aggregate acceptance `0.6176`, equivalent accept length
`2.4704`, and token-weighted theoretical gain `163.39%`; the branch latency
simulator reports the same equivalent accept length as a paper-like `2.470x`
ratio (`+147.04%`) under its aggregate-cycle formula. Feeding that same real
trace into a four-stage Skippy latency model with `4ms` per stage estimated
`9.882x` SPD-vs-serial split throughput at `0ms` hop, `8.117x` at `10ms` hop,
and `7.752x` at `25ms` hop. The Rust live-tap harness now also proves three
consecutive Qwen3.5-4B top-1 SPD proposals from live Skippy activation frames,
and a later eight-step release run accepted `7 / 8` top-1 proposals. Every
verifier window rewound and the committed token stream matched ordinary
non-SPD greedy decoding in both runs. The
Skippy OpenAI request path can run the pretrained head from inline returned
taps without local replay fallback. The branch now also has an opt-in
`--openai-spd-optimistic-decode` path for deterministic sampling
(`temperature <= 0.0`): once a pre-target SPD proposal is ready, stage 0 starts
a checkpointing one-token `VerifySpan` from the proposed token, commits that
optimistic target result if the proposal is accepted, or restores the checkpoint
after draining the stale optimistic result if the proposal is rejected. A
release smoke produced the same eight-token greedy output as the no-SPD
baseline and committed one token from the optimistic target path. This is a
real serving-path proof of SPD scheduling and rollback mechanics, but it is not
a local speedup yet. A release live-tap parity run with the optimized serving
head accepted `7 / 8` live top-1 proposals, matched ordinary greedy output,
rewound every verifier window, and averaged about `212ms` per step:
`41ms` in the SPD head, `5ms` assembling `cur_in`, and `128ms` in tap replay.
The `cur_in` win came from keeping sidecar projection weights resident and
parallelizing the tap projection matmul. An earlier real OpenAI request-path
rerun with selective tap returns produced the same eight-token greedy output as
the no-SPD baseline, proposed `4` tokens, accepted `1`, rejected `3`, and
committed one optimistic token. The same-topology baseline was about `0.70s`;
filtered SPD is about `1.95s`, down from about `3.19s` unfiltered and `2.22s`
before the projection fast path. That `spd-openai-smoke` summary made the
remaining gap explicit: pre-target probes proposed `4 / 4` tokens with `1 / 4`
accepted, post-target probes were empty `3 / 3` times, pre-target probes
averaged `56.9ms`, normal downstream waits averaged `190.9ms`, the one
optimistic downstream wait was `21.0ms`, and that earlier ungated optimistic
path requested `0` reusable tap returns. The next target became reducing target
wait and increasing accepted proposal coverage enough for accepted SPD work to
beat the normal target pipeline, not proving the head can be called. A
margin-gated tap-return smoke (`--spd-top-k 2 --optimistic-min-logit-margin 2`) removed the
empty post-target probe gap and committed two optimistic tokens, but it remained
slower: baseline was `579ms` wall / `201ms` decode, while SPD was `2348ms` wall
/ `1871ms` decode. The report showed `6 / 6` pre-target proposals, `3 / 6`
accepted, `3` optimistic tap-return requests, `2` accepted tap-return requests,
`1` rejected tap-return request, and about `173ms` mean optimistic downstream
wait. The request path now also tracks accepted context length plus the single
pending optimistic tap position, so late returned taps for rejected future
positions are ignored instead of being cached. The first lifecycle-filter smoke
kept exact output, `6 / 6` pre-target proposals, `3 / 6` accepted, two
optimistic commits, and `0` ignored taps in that run; it was still slower at
`2351ms` wall / `1869ms` decode versus a `569ms` wall / `206ms` baseline. That
points at direct-return scheduling and true rolling pipeline state, not another
replay bridge. The current request path asks for SPD tap returns on every
started optimistic SPD verify, gated or ungated. A pre-patch no-tap ungated
diagnostic was faster (`2591ms` wall / `2004ms` decode) but starved the rolling
rows (`5` missing proposal positions and `2` out-of-order proposals). The
current always-tap ungated smoke preserved exact output, accepted `8 / 8`
proposals, committed `4 / 4` optimistic decodes, had no missing or out-of-order
rolling replay entries, and showed the cost honestly: baseline `629ms` wall /
`201ms` decode versus SPD `4154ms` wall / `2919ms` decode. With `25ms`
injected downstream-stage delay, the same current path still preserved exact
output, accepted `8 / 8`, committed `4 / 4` optimistic decodes, and kept
rolling replay ordered, but it still did not cross over: baseline was `2807ms`
wall / `1938ms` decode; SPD was `4210ms` wall / `3105ms` decode (`0.667x`
wall, `0.624x` decode). The paper estimate for that trace was `484ms` decode
at baseline stage cost, so the current request path is about `6.4x` slower than
the paper-shaped rolling schedule even with artificial hop latency.
`spd-openai-smoke`
now also emits a paper-style rolling-pipeline
estimate from the actual accept rate, using the manifest's logical SPD stage
count instead of the physical tap-aligned Skippy slice count. On the same
seven-slice proof topology, the head manifest is logical `S=4`; one
topology-row smoke accepted `3 / 8` proposals overall and implied a `1.5x`
paper-style speedup versus serial split, about `141ms` decode at that run's
baseline stage cost. The request path now also emits live
`llama_stage.spd_rolling.*` telemetry on inline probes and decode completion.
Follow-up diagnostic `optimistic_commit` probes use the accepted optimistic
token's returned taps to ask the SPD sidecar for the committed optimistic
position; this is diagnostic-only and does not change commit behavior. A
pre-fix run kept live/replay rolling ordered but accepted only `1 / 8`
proposals. The same no-thinking prompt in the Python reference accepted
`7 / 8`, proving the low acceptance was a serving row-alignment bug rather than
a head-quality problem. The fix is to use actual token indices for live rolling
positions. After that change, the bounded OpenAI request-path smoke accepted
`8 / 8` proposals, committed `3 / 3` optimistic target decodes, preserved exact
greedy output, and recorded `0` tap return failures, `0` tap record failures,
and `0` ignored taps. Baseline was `626ms` wall / `198ms` decode; SPD remained
slower at `3521ms` wall / `2921ms` decode, so this is an acceptance/scheduling
fix rather than a speedup claim.
The latest smoke also derives downstream tap returns from the sidecar topology,
requesting `[8, 10, 16, 20, 24, 31]` instead of only the fixture's
`[10, 20, 31]`. Proposal assembly now waits only for the row-specific taps it
is assembling, so this restores pre-target proposals while still making
the extra topology taps available for future rolling rows. The runtime
scheduler now also reports the nominal paper row roles needed to build
speculation inputs: before an evicted prefix it is `g_{S-1}..g_0`; after an
accepted eviction it is `g_S^evicted,g_{S-1}..g_0`. The serving proposal path
then resolves those nominal roles through the manifest. For the Qwen3.5-4B
artifact, `trained_with_use_deepest=true`, so full snapshots with all required
tap blocks upgrade to the deepest trained row. Fixture parity confirms the
exported rows use positions `[9, 10, 11, 12]` with inference roles
`[4, 4, 4, 0]`, not nominal `[3, 2, 1, 0]`.

2026-06-17 follow-up: proposal tap collection is now sparse by row role instead
of dense over every row position. The previous dense collection forced a
`[4, 4, 4, 0]` window to wait for `h31` on the `g_0` row even though that row
only consumes `h0`. The fixed no-delay smoke
(`/private/tmp/spd-openai-sparse-rows-smoke1/report.json`) preserved exact
output, accepted `5 / 5`, and moved ready optimistic probes earlier than the
final tap (`trigger_hf_index=20,20,16` after bootstrap). A comparable 8-token
smoke (`/private/tmp/spd-openai-sparse-rows-smoke8/report.json`) preserved
exact output and accepted `8 / 8`, but only `3` optimistic decodes were
committed and the decode path remained slower (`205ms` baseline decode versus
`5631ms` SPD decode). With `25ms` injected downstream delay
(`/private/tmp/spd-openai-sparse-rows-delay25-smoke1/report.json`), exact output
and `5 / 5` acceptance held and the decode gap narrowed to `0.353x`
SPD-vs-baseline, but it still did not cross over. This removes the missing-tap
blocker; the next gap is the rolling executor and direct-return wait ownership,
not sidecar row availability.

2026-06-17 rolling follow-up: when SPD optimistic decode is enabled under
deterministic sampling, stage-0 serving now prefers the direct-return rolling
path instead of falling back into the primary `VerifySpan` speculative branch
after the first optimistic burst. The no-delay smoke at
`/private/tmp/spd-openai-rolling-prefer-smoke8/report.json` preserved exact
output, accepted `8 / 8`, committed `6 / 6` optimistic verifier results
(`4 / 4` chained), emitted two pre-target bursts plus six optimistic-commit
probes, and ended with `0` missing or out-of-order rolling proposals. It is
still much slower than baseline (`207ms` baseline decode versus `15210ms` SPD
decode), so this is correctness/scheduling evidence, not a speed win. The next
work is shrinking hidden waits and turning this into a persistent paper-shaped
rolling executor rather than proving the Qwen sidecar again. The delayed-link
rerun at `/private/tmp/spd-openai-rolling-prefer-delay25-smoke8/report.json`
also preserved exact output, accepted `8 / 8`, committed `6 / 6` optimistic
verifier results, and reached `0.242x` decode speed versus baseline (`1997ms`
baseline decode, `8247ms` SPD decode).

2026-06-17 verifier follow-up: single-token `VerifySpan` now executes through
the normal decode-frame runtime path while keeping the `VerifySpan` wire/reply
contract and rollback checkpoints. The matching smoke at
`/private/tmp/spd-openai-single-token-decode-smoke8/report.json` preserved
exact output, accepted `8 / 8`, and committed `6 / 6` optimistic verifier
results (`4 / 4` chained), but only improved SPD decode to `13516ms` versus
`224ms` baseline. A temporary unshipped checkpoint-skip diagnostic at
`/private/tmp/spd-openai-skip-verify-checkpoint-smoke8/report.json` still took
`9527ms` SPD decode versus `215ms` baseline. The per-stage logs show overlapping
stage compute on the same M4 Metal device inflating individual stage calls from
single-digit milliseconds to hundreds or thousands of milliseconds. That makes
the current local smoke a correctness/scheduler probe, not a speed verdict; the
next useful benchmark has to place stages on distinct devices or nodes.

`skippy-bench spd-openai-smoke` now has the minimal native placement surface for
that benchmark. `--stage-hosts` cycles stage placement across `local` and remote
SSH targets, while requiring stage 0 to remain local so the OpenAI frontend,
sidecar, and request/report ownership stay on the coordinator. Remote
direct-return links use `--endpoint-host-map local=<reachable-stage0-host>` and
per-target endpoint mappings; model access can come from
`--remote-model-path-map` or from `--rsync-model-artifacts`. The refactor was
validated with a short local release smoke at
`/private/tmp/spd-openai-remote-refactor-local-smoke2b/report.json`: exact
baseline/SPD content match, `2 / 2` SPD proposals accepted, one optimistic token
committed, and `0` tap return or record failures. This proves the benchmark path
still works after adding placement support, not that SPD is fast locally.

The request path now has a first native sidecar KV cache implementation: during
proposal assembly it lazily prefills `g_S` prefix rows from complete inline
prefill taps, stores rotated K plus V per spec layer, crops the cache at the
minimum rolling row position before each proposal, and runs the query row
against prefix-plus-current cached attention. A cached Python fixture compares
the native cache against the reference `spec_past_kv` path on the same
Qwen3.5-4B head: `cached_prefill_rows=20`, `cache_prefix_len=20`,
Rust/Python cached top-k token ids match exactly
`[23, 17, 24, 21, 16, 22, 760, 19]`, and the full cached logit max absolute
diff is `0.0625` (`spec_query` diff `0.03125`, `final_hidden` diff `0.0625`).
That closes the cache-fidelity question. The remaining gap is serving
latency/scheduling: proposal probes still cost tens of milliseconds and the
current request path is still a bounded one-token/optimistic proof, not the
fully overlapped paper rolling schedule.

`skippy-runtime::spd::SpdRollingScheduler` now captures the paper/reference
rolling pipeline state machine as a small Rust primitive. It tracks the
newest-first in-flight speculative entries, emits the hidden-state row
positions needed for the speculation head, verifies the oldest completed entry
once the pipeline is full, preserves the evicted-prefix row on acceptance, and
resets the pipeline to the corrected target token on rejection.
`SpdRollingTraceReplay` replays observed target/proposal traces through that
same runtime primitive and reports the final target-verified prefix tokens.
`SpdRollingObserver` is the live token/position observer used by `skippy-server`
diagnostics; its `take_verified_delta()` method releases only newly
target-verified token spans and its `speculation_rows()` method exposes the
position plus `g_i` row roles needed to assemble SPD sidecar input without
sliding-context guessing. `SpdRollingObserver::draft_plan()` now clones the
verified scheduler for proposal generation. The server advances that draft plan
locally while it proposes the next window, so later proposals can use the
paper-shaped rolling rows without mutating the live observer until the target
verifier accepts or rejects them. A bounded primary-`VerifySpan` smoke with
`max_tokens=8`, `--optimistic-decode false`, and replay fallback preserved exact
greedy output, ran two SPD windows, accepted `8 / 8` proposals, inserted `7`
rolling drafts, verified `5` filled rolling windows, and reported `0` missing
or out-of-order proposals. The second window's stage-0 telemetry showed the
accepted-eviction rows `[30, 30, 31, 32, 33]` with resolved roles
`[4, 3, 3, 3, 0]`. `skippy-bench` now carries primary-verify-only cases into
the aggregate rolling summary from `cases[].decode.rolling`: a rerun at
`/private/tmp/spd-openai-primary-rolling-report-smoke/report.json` reports
`cases_replayed=0`, `live_cases_observed=1`, `inserted_drafts=7`,
`verified_windows=5`, and `0` missing/out-of-order proposals. It is still
deliberately slow proof code: that rerun was baseline `636ms` wall / `204ms`
decode versus SPD `3817ms` wall / `3228ms` decode, with `2643ms` spent
proposing. The paper-style estimate from the same `1.0` accept rate is `4.0x`
versus serial split, or about `51ms` decode at that run's baseline stage cost,
so current serving is about `63x` slower than the schedule it is trying to
realize. The latest primary proposal-breakdown rerun at
`/private/tmp/spd-openai-proposal-breakdown-smoke/report.json` preserved exact
output and accepted `8 / 8`, but all `8` proposals came from replay fallback:
`inline_tap_hits=0`, `replay_fallbacks=8`, `tap_collect_ms=2205ms`,
`cur_in_ms=51ms`, and `forward_ms=509ms` inside `2766ms` of total proposal
time. The follow-up no-replay optimistic inline-probe rerun at
`/private/tmp/spd-openai-overlap-probe-smoke/report.json` preserved exact
output, proposed and accepted `8 / 8`, committed `4 / 4` optimistic decodes,
and every measured inline probe reported `tap_source=inline` with no replay
fallback. Optimistic-commit probes now run during the in-flight optimistic
`VerifySpan` reply wait when the preceding target proposal is accepted, instead
of being recomputed after the reply. Those probes reported
`trigger_hf_index=31` and about `0.001ms` wait-after-probe, proving the sidecar
work moved into the target wait. The final decode event now agrees with the
probe evidence: `inline_tap_hits=8`, `replay_fallbacks=0`,
`tap_collect_ms=2.39ms`, `cur_in_ms=115.2ms`, and `forward_ms=528.5ms`.
Baseline decode was `202ms`; SPD decode was `2800ms` (previous comparable run:
`2834ms`). A thinking-mode rejection smoke at
`/private/tmp/spd-openai-overlap-rejection-clean-smoke/report.json` preserved
exact output with `1 / 8` proposals accepted, `7 / 8` rejected, and `1 / 3`
optimistic decodes committed, but left one live `decode.rolling`
out-of-order proposal at the frontier. The live observer now retains early
proposals and promotes them once accepted context catches up. The follow-up
rejection smoke at
`/private/tmp/spd-openai-pending-promote-rejection-smoke/report.json`
preserved exact output, proposed `8` tokens, accepted `3`, rejected `5`,
committed `2 / 4` optimistic decodes, and ended with live
`decode.rolling.out_of_order_proposals=0`. Baseline decode was `207ms`; SPD
decode was `3475ms`. A no-thinking chainability rerun at
`/private/tmp/spd-openai-chainability-summary-smoke/report.json` preserved
exact output, accepted `8 / 8`, committed `4 / 4` optimistic decodes, and now
reports the wait-overlapped commit-probe signal separately: `4 / 4`
`optimistic_commit` probes proposed, `4 / 4` were accepted, and their mean
wait-after-probe was about `0.001ms`. That says the sidecar is ready often
enough on this prompt; the missing speed path is safe chained/rolling target
execution. The same run was still slower, `201ms` baseline decode versus
`2873ms` SPD decode. `skippy-bench` still uses runtime-owned
`SpdRollingTraceReplay` when token/proposal traces exist, and primary
`VerifySpan` commits now emit token events for that trace. Reports without
per-token traces can still fall back to live final rolling state. The serving
path does not yet execute the full rolling schedule. This remains the
implementation contract the serving path needs to move from bounded side
verification to the paper's `n`-slot rolling schedule.

The first bounded chained optimistic target execution is now in the real
serving path. A no-thinking release smoke at
`/private/tmp/spd-openai-chained-optimistic-smoke8/report.json` preserved exact
greedy output, proposed `6` SPD tokens, accepted `4`, rejected `2`, committed
`4 / 4` optimistic target tokens, and committed `2 / 2` of those through a
one-step chained optimistic `VerifySpan` while the previous optimistic verifier
was still in flight. Tap return failures, tap record failures, and ignored taps
were all `0`. The report preserves `chain=true` on the two chained
`DecodeEmbdOptimistic` token events and now emits token events for primary
`VerifySpan` commits too, so rolling trace replay sees the full emitted target
stream `[71093, 12305, 198, 727, 884, 2784, 11, 292]` and verifies it matches.
The smoke remained slower, with `203.1ms` baseline decode versus `2820.5ms` SPD
decode. This is the first serving-path proof that accepted optimistic-commit
proposals can become real target messages, but it is still a bounded one-step
chain, not recursive paper SPD. `SpdRollingTraceReplay` is now conservative
when target token events are missing: it reports missing or out-of-order
proposal positions instead of fabricating zero-filled verified prefix tokens.
An attempted recursive chain that started another same-kind `VerifySpan` from a
prior chain's direct-return callback was not retained. It preserved exact text
in one exploratory smoke but launched from partial early taps and produced
rejected duplicate proposals; adding a final-tap gate then exposed that
same-kind direct-return replies needed explicit origin metadata before several
`PredictedTokens` verifiers could be in flight safely.

Direct-return prediction replies now carry a direct-return-only origin header
when the `PredictionReturnOpen` stream opts in, and the stage-0 receiver can
buffer out-of-order final replies until the requested `(kind, pos_start,
decode_step, token_count, prompt_token_count, checkpoint_generation)` arrives.
Legacy direct-return streams without that flag still decode as untagged
replies. A current release smoke at
`/private/tmp/spd-openai-origin-aware-smoke2/report.json` preserved exact
greedy output, proposed `6` SPD tokens, accepted `4`, rejected `2`, committed
`4 / 4` optimistic target tokens, committed `2 / 2` chained optimistic target
tokens, and recorded `0` tap return failures, `0` tap record failures, and `0`
ignored taps. Baseline decode was `239.1ms`; SPD decode was `2692.0ms`, so the
change removes the reply-ownership blocker but is still not a speedup. The next
full paper-style step is a rolling executor that decides which verified
pipeline entry to launch next and when to reset/restore, not another untagged
nested callback.

Checkpoint ownership is now generation-addressed instead of session-only. Stage
0 stamps speculative `VerifySpan` messages with `checkpoint_generation` derived
from the target position, the direct-return origin includes that generation, and
both embedded stage 0 and downstream binary stages checkpoint/restore by
`(session_id, checkpoint_generation)`. The current release smoke at
`/private/tmp/spd-openai-checkpoint-gen-smoke1/report.json` preserved exact
greedy output, proposed `6` SPD tokens, accepted `4`, rejected `2`, committed
`4 / 4` optimistic target tokens, committed `2 / 2` chained optimistic target
tokens, and recorded `0` tap return failures, `0` tap record failures, and `0`
ignored taps. Baseline decode was `203.7ms`; SPD decode was `2796.9ms`. This
removes the checkpoint-overwrite blocker for multiple in-flight speculative
entries, but the serving loop is still the bounded one-step chain rather than
the paper's continuously full rolling pipeline.

The serving path now uses a small origin-matched rolling queue instead of a
single hardcoded chained verifier. Accepted optimistic verifiers advance the
sidecar context, wait for their own direct-return reply by origin, and may launch
one deeper verifier from returned taps while staying capped by the logical SPD
stage count. A current release smoke at
`/private/tmp/spd-openai-hidden-wait-smoke1/report.json` preserved exact
greedy output, reached `max_optimistic_chain_depth=2`, proposed `8` SPD tokens,
accepted `6`, rejected `2`, committed `5 / 7` optimistic target tokens,
committed `2 / 4` chained optimistic target tokens, and recorded `0` tap return
failures, `0` tap record failures, and `0` ignored taps. Baseline decode was
`202.3ms`; SPD decode was `9275.4ms`. The depth-2 verifier entries launched and
rolled back cleanly, but both depth-2 proposals rejected on this prompt. The
raw timing already shows real latency hiding: derived hidden-wait was about
`6.72s` total, almost entirely from chained rows, with the two depth-2 rows
hiding about `1.69s` and `5.03s` behind earlier verifier work. The benchmark
report now carries these as `hidden_wait_ms` on `optimistic_decodes[]` and as
`summary.pipeline_gap.optimistic_decode_hidden_wait_ms` /
`chained_optimistic_decode_hidden_wait_ms`. This is the first serving proof of
more than one chained in-flight verifier; the next gap is making the rolling
scheduler/proposal rows productive enough to keep the pipeline full and
accepted, then removing the serial wait points that dominate the local CPU
split.

Stage-role audit: the Qwen SPD reference has two different stage concepts.
`row_i_stages` selects which target hidden taps and `stage_projs.*` projection
build each `cur_in` row. The spec head's internal fixed-memory `stage_ids`
default to `_infer_stage_ids(q_len)` unless explicitly overridden. A Rust run
that fed fixture `row_i_stages=[4,4,4,0]` directly into the fixed-stage
projection path made parity much worse
(`/private/tmp/spd-fixture-parity-topk4-stageids.json`: forward final-hidden
max diff `9.75`, spec-query max diff `28.4375`). The native contract now keeps
those concepts separate: live proposal rows use `row_i_stages` only for tap
assembly, Qwen forward defaults to inferred fixed-stage roles, and cache
prefill can explicitly mark completed prefix rows as deepest-stage rows. The
corrected parity run
(`/private/tmp/spd-fixture-parity-topk4-fixedstage-default.json`) returned to
the known bf16-sized drift: forward final-hidden max diff `0.125`, spec-query
max diff `0.25`, cached final-hidden/logit max diff `0.0625`, and matching
Rust/Python top-k. A fresh release OpenAI smoke
(`/private/tmp/spd-openai-fixedstage-default-smoke1/report.json`) preserved
exact output, proposed `8`, accepted `6`, rejected `2`, reached
`max_optimistic_chain_depth=2`, and recorded `0` tap failures/ignored taps.
It was still slow (`203.8ms` baseline decode versus `13912.7ms` SPD decode),
so the repeated depth-2 rejections are not explained by a fixed-stage row-id
bug; the next blocker remains proposal quality/rolling context and rollback
cost.

Proposal-row telemetry then isolated one serving-context bug. A diagnostic run
(`/private/tmp/spd-openai-proposal-rows-smoke1/report.json`) showed the bad
depth-2 proposal at step 2 was assembled from stale rows `[23,24,25,26]`
with `next_draft_position=27`, even though the accepted optimistic token had
already moved the target position to 28; the sidecar therefore proposed the
previous token again. The serving path now observes accepted optimistic-commit
probes into `SpdRollingObserver` immediately, before a deeper chained verifier
asks for proposal rows. The follow-up release smoke
(`/private/tmp/spd-openai-rolling-observe-smoke1/report.json`) preserved exact
output, proposed `5`, accepted `5`, rejected `0`, committed `3 / 3`
optimistic decodes plus `2 / 2` chained optimistic decodes, reached depth `2`,
and recorded `0` tap failures. Step 2 now proposes token `198` from rows
`[24,25,26,27]` with `next_draft_position=28`, so the stale-row repeat is
gone. It is still slower (`222.2ms` baseline decode versus `2667.6ms` SPD
decode) and the final rolling replay reports `3` missing proposal positions
starting at position 30, which keeps the next task focused on the executor
that keeps rolling rows filled after a chain boundary.

The latest miss-diagnostic smoke
(`/private/tmp/spd-openai-tap-position-diagnostics-smoke1/report.json`)
preserved exact greedy output, proposed `5`, accepted `5`, rejected `0`,
committed `3 / 3` optimistic decodes plus `2 / 2` chained optimistic decodes,
and recorded `0` tap return or record failures. It is still slower
(`204.6ms` baseline decode versus `2657.3ms` SPD decode). The new probe
diagnostics report `missing_inline_taps` for post-target probe steps `4`, `5`,
and `6`; after filtering h0 out of inline requirements, those misses are only
non-h0 topology taps. Tap-position telemetry then corrected the first
diagnosis: the required position-`28` rows were recorded before the empty
post-target probes, but a later accepted-prefix acknowledgement from the token
emission path pruned future tap rows because SPD's sidecar context had already
advanced ahead of the emitted prefix. The serving path now treats shorter
prefix-compatible accepted-context updates as acknowledgements, not resets.
The follow-up release smoke
(`/private/tmp/spd-openai-prefix-ack-smoke1/report.json`) preserved exact
output, proposed and accepted `8 / 8`, committed `6 / 6` optimistic verifier
results including `4 / 4` chained optimistic results, reported `0`
post-target empty probes, and kept rolling replay at `0` missing and `0`
out-of-order proposals. It is still slower (`219.7ms` baseline decode versus
`2795.3ms` SPD decode), so the remaining work is performance and the fully
overlapped rolling executor, not sidecar topology, h0 synthesis, or the Qwen
fixed-stage contract.

## What Works Today

### 1. Real Small-Model Training Proof

`evals/spd/hf_train_eval_qwen06.py` trains a real SPD head using the paper's
reference code.

Recorded local proof:

| Model | Training data | Head | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B` | `HuggingFaceH4/ultrachat_200k`, split `train_sft`, first 1024 rows | 4 spec layers, 2 stages | 1536 | 326 / 1536 | 0.5628 | 1.1257 | 12.67% |

This proves the train/eval/export path. It is not the high-gain target.

### 2. Strong Modest-Model Acceptance Signal

The author-published `Qwen3.5-4B_s4_l4.pt` SPD head evaluates well with the
reference verifier.

Recorded local proof:

| Model | Head | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-4B` | pretrained, 4 stages / 4 spec layers | 1536 | 1230 / 1536 | 0.6176 | 2.4704 | 163.39% |

The accepted-flags count and aggregate acceptance use different denominators in
the reference output. `1230 / 1536` is the draft-flag count; `0.6176` is the
reference aggregate acceptance metric used for equivalent accept length.

Per-dataset theoretical gains from the same run:

| Dataset | Acceptance | Equivalent accept length | Theoretical gain |
| --- | ---: | ---: | ---: |
| MT-Bench | 0.4918 | 1.9673 | 98.42% |
| HumanEval | 0.8797 | 3.5189 | 254.18% |
| GSM8K | 0.5926 | 2.3704 | 137.58% |

This is the main reason to keep pursuing SPD for Skippy.

### 3. Trace-Based Skippy Latency Model

`evals/spd/simulate_latency.py` consumes real per-sample SPD eval traces and
models split-stage latency. It does not invent acceptance.

Recorded Qwen3.5-4B trace with four target stages at `4ms,4ms,4ms,4ms`:

| Hop ms | Serial split tok/s | SPD pipeline tok/s | SPD vs serial split | Paper-like gain | P50 serial ms | P50 SPD ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 62.50 | 617.61 | 9.882x | 2.470x | 1024.00 | 106.50 |
| 1 | 52.63 | 494.09 | 9.388x | 2.470x | 1216.00 | 133.12 |
| 5 | 32.26 | 274.49 | 8.509x | 2.470x | 1984.00 | 239.62 |
| 10 | 21.74 | 176.46 | 8.117x | 2.470x | 2944.00 | 372.75 |
| 25 | 10.99 | 85.19 | 7.752x | 2.470x | 5824.00 | 772.12 |

The paper-like gain is from the SPD trace itself. The Skippy comparison models
ordinary split serving as requiring each generated token to traverse all
stages/hops before the next target token is known.

### 4. Rust Serving Artifact Validation

`crates/skippy-runtime/src/spd.rs` adds a manifest parser and validator for SPD
head artifacts:

- schema: `skippy-spd-head/v1`
- checkpoint path, byte size, sha256
- base model path/id
- checkpoint format/version
- hidden size
- vocab size
- draft vocab size and optional draft token ids
- number of target stages
- number of spec layers
- shallow hidden-layer tap indices
- optional safetensors serving checkpoint path, size, checksum, tensor count,
  and dtype

`evals/spd/export_spd_head.py` exports the reference `.pt` checkpoint into
`spd-head.safetensors` and updates the manifest with a `serving_checkpoint`
section. Skippy can inspect the serving artifact and read tensor payloads. The
constrained Rust Qwen fixture path can reconstruct SPD input rows from recorded
hidden-state taps and execute the head against recorded fixtures. The live-tap
bench path can also assemble those rows from real Skippy activation frames for
the tap-aligned Qwen3.5-4B proof split.

`skippy-runtime::spd::SpdQwen3Head` is the current Rust hosting boundary for
the pretrained Qwen head. It opens the manifest and serving checkpoint once,
validates the tensor shapes, caches the serving weights, keeps the runtime
shape, and exposes repeated `forward()` calls for proposal generation. The live
verifier harness and request-path probe now use that loaded head instead of
reopening the artifact on every proposal. This is the shape a Skippy SPD
sidecar should use, but the current Qwen forward remains a straightforward Rust
reference path, expands BF16 weights to `f32`, and is still CPU-bound. The
serving `forward()` path now skips diagnostic trace collection, computes top-k
directly from the LM head instead of materializing/sorting full vocab logits,
and parallelizes large dense rows. Fixture diagnostics still use the full-logit
trace path so parity evidence is not weakened by the serving fast path.

### 5. Rust Hidden-Tap Planning

`crates/skippy-runtime/src/spd/tap_plan.rs` translates the manifest's hidden
state requirements into concrete Skippy stage ownership. The reference
convention is:

- HF hidden-state index `0` is the embedding output
- HF hidden-state index `k >= 1` is the output after decoder layer `k - 1`

For the pretrained `Qwen/Qwen3.5-4B` S4/L4 head, the required tap groups are:

```text
g4: [0, 10, 20, 31]
g3: [0, 8, 16, 24]
g2: [0, 8, 16]
g1: [0, 8]
```

The new Rust planner confirms:

| Candidate Skippy layer ranges | Result |
| --- | --- |
| `0..8, 8..16, 16..24, 24..32` | ordinary four-way split exposes boundary indices `8,16,24,32`; SPD still needs internal taps `10,20,31` |
| `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32` | tap-aligned proof split can expose every required tap as a stage boundary |

This gives two implementation options: add a narrow internal hidden-tap ABI for
normal four-stage serving, or use a tap-aligned over-split as the fastest local
proof that the pretrained head can drive live Skippy proposals.

`skippy-model-package` now supports explicit split boundaries for this proof:

```bash
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir .artifacts/spd/qwen35-4b-gguf/
skippy-model-package plan model.gguf --splits 8,10,16,20,24,31
skippy-model-package write-stages model.gguf \
  --splits 8,10,16,20,24,31 \
  --out-dir /tmp/qwen35-spd-tap-slices/
skippy-model-package validate model.gguf /tmp/qwen35-spd-tap-slices/stage-*.gguf
skippy-model-package preflight model-package/ --splits 8,10,16,20,24,31
```

For a 32-layer Qwen3.5 model, those boundaries materialize the ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`. This does not yet make
SPD live in Skippy by itself; it removes the artifact-generation blocker for a
tap-aligned local proof that can use normal stage-boundary activation frames
before adding a production hidden-tap ABI.

Recorded local artifact result with
`unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf`:

- source size: `2.6G`
- source sha256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
- plan: `32` layers, `7` tap-aligned stages
- validation: all `426` owned tensors present exactly once across the seven
  slices; no missing or duplicate owned tensors

The local live-chain smoke now runs the same tap-aligned shape through real
Skippy binary stage transport by selecting the CPU backend explicitly:

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

Recorded result: all seven ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32` forwarded over the
binary chain, activation width `2560`, first boundary payload `10240` bytes /
`5120` f16 wire bytes, prompt token id `9419`, predicted token `11`. A
`local-split-compare` run against the same GGUF and prompt matched the unsplit
full-model predicted token `11`.

The Rust SPD fixture path now validates the missing bridge between Skippy taps
and the pretrained head input. `evals/spd/export_parity_fixture.py` records raw
tap-row tensors (`tap_row_*_concat` plus `tap_row_*_hf_indices`) before Python
applies `g0_proj` or `stage_projs.*`; `skippy-runtime` reconstructs the same
`cur_in` from those rows. Recorded real-artifact command:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --top-k 8
```

Recorded result for the regenerated `Hello` fixture:

- tap input reconstruction max abs diff: `7.62939453125e-6`
- Rust/Python draft indices:
  `[7728,15014,38999,10036,11235,13293,15953,0]`
- full token ids:
  `[9419,21251,109266,12675,14556,18103,23066,0]`
- spec-query max abs diff: `0.03125`
- final-hidden max abs diff: `0.125`

The branch now also proves the trained head can consume live Skippy-produced
tap frames for the same Qwen3.5-4B fixture. The proof command drives the
fixture prompt token ids through tap-aligned runtime slices, adds an
embedding-only side tap for HF hidden-state index `0`, assembles `cur_in` from
real activation frames, runs the Rust Qwen SPD head, and verifies the live
top-1 proposal with Skippy's target verifier:

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

Recorded local live-tap result against `Qwen3.5-4B-Q4_K_M.gguf`:

- live taps captured: `0,8,10,16,20,24,31`
- each tap frame: `13` tokens, `133120` bytes, hidden width `2560`
- live `cur_in` max abs diff vs HF fixture: `0.3134765625`
- g0 row max abs diff vs HF fixture: `0.00103759765625`
- live Skippy top-1 token id: `9419`
- fixture Python/Rust top-1 token id: `9419`
- live top-8 token ids:
  `[9419,21251,109266,14556,23066,18103,12675,0]`
- fixture top-8 token ids:
  `[9419,21251,109266,12675,14556,18103,23066,0]`
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

The top-8 set is the same, with lower-ranked candidates reordered by GGUF
quantization/runtime drift. This is a real Skippy tap/head/target-verifier
proof over repeated proposal windows, but it is still a diagnostic harness and
not a serving throughput measurement. `skippy-server` now has an experimental
request-path SPD source that can load the same pretrained head. By default it
uses inline Skippy taps when they are complete and otherwise returns no
proposal. Passing `--openai-spd-replay-fallback` enables the older slow local
tap replay mode for correctness proofs.

Recorded local request-path smoke with `skippy-server serve-binary` and
`--openai-spd-replay-fallback`:

- topology: seven tap-aligned local CPU stages,
  `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`
- model: `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`
- prompt: Humaneval eval row `index=8`, `max_tokens=4`, `temperature=0`
- SPD source: `spd-replay`
- SPD proposals: `4`
- accepted proposals: `2`
- rejected proposals: `2`
- accepted proposal windows: `2 / 4`
- emitted text: `<think>\nThe user wants me`
- no-SPD baseline emitted the same text for the same request
- SPD replay wall time: about `101.5s`; logged decode time: about `100.6s`
- no-SPD Skippy baseline wall time: about `1.28s`; logged decode time: about
  `90ms`

This proves the live OpenAI serving path can call a real pretrained SPD head,
feed proposals into Skippy's verifier, accept/reject per token, and preserve
ordinary greedy output. It is still not a serving throughput measurement:
`spd-replay` recomputes taps through local `StageModel` slices and runs the head
on CPU for each proposal when replay fallback is enabled. Use the trace latency
simulator for current speedup estimates until proposal scheduling consumes
inline taps without replay.

Inline tap transport now works for the tap-aligned local proof topology. During
embedded stage-0 serving, Skippy records stage-0 boundary activation frames into
an SPD-positioned tap cache keyed by hidden-state index and token position.
Downstream binary stages can return SPD tap frames over the direct-return side
channel when stage 0 marks an SPD request. A one-token Qwen3.5-4B smoke on
seven local CPU stages returned and recorded required hidden-state rows for
`10`, `20`, and `31` with no tap-return failures:

- response content: `<think>\nThinking`
- wall time: `23.646s`
- stage-0 local tap records: `hf_index=8`, rows `17` and `1`, `required=false`
- downstream required tap records:
  - `hf_index=10`, producer stage `1`, rows `17` and `1`, `required=true`
  - `hf_index=20`, producer stage `3`, rows `17` and `1`, `required=true`
  - `hf_index=31`, producer stage `5`, rows `17` and `1`, `required=true`
- downstream non-required tap records: `16` and `24`, rows `17` and `1`

Recorded local release request-path smoke without replay fallback, before
optimistic commit was enabled:

- topology: seven tap-aligned local CPU stages,
  `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`
- binary: `target/release/skippy-server`
- prompt: `Write a Python function named add that returns the sum of two integers.`
- response content for `max_tokens=4`: `<think>\nThinking Process:\n\n`
- no-SPD baseline emitted the same text for the same request
- SPD source: `spd-replay`, with `--openai-spd-replay-fallback` disabled
- inline tap state: required downstream taps `10`, `20`, and `31` returned and
  recorded for prompt rows and all four generated-token rows; no tap-return
  failures
- h0 source: direct GGUF `token_embd.weight` rows
- SPD head hosting: cached pretrained Qwen3.5-4B serving weights
- inline probe phase: `pre_target_reply` for all four proposals
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

This proves the real pretrained head can run in the Skippy OpenAI request path
from inline Skippy taps plus direct GGUF h0 embeddings, without replaying local
stage slices. It also proves stage 0 can start that head before consuming the
final target token reply, count accepted/rejected SPD proposals against normal
target decode, and preserve ordinary greedy output. It is not a speedup yet:
the current unoptimized CPU Rust head is much slower than the remaining
final-stage work in this local topology, and this prompt accepted only one of
four proposals.

Recorded local release request-path smoke with
`--openai-spd-optimistic-decode` and `temperature=0.0`:

- topology: same seven tap-aligned local CPU stages
- binary: `target/release/skippy-server`
- prompt: `Write a Python function named add that returns the sum of two integers.`
- response content for `max_tokens=8`: `<think>\nThinking Process:\n\n1.  **`
- no-SPD baseline emitted the same text for the same request
- no-SPD baseline wall time: about `0.55s`; decode time: about `197ms`
- optimistic SPD wall time: about `3.31s`; decode time: about `2.76s`
- inline SPD windows: `2`
- accepted proposals: `1`
- rejected proposals: `1`
- committed optimistic target tokens: `1`
- optimistic checkpoint time: about `11.7ms` total
- optimistic target-decode time: about `53.4ms` total
- optimistic target-reply wait time: about `41.7ms` total
- rollback restore time for the rejected proposal: about `3.4ms`
- target-verified optimistic sequence:
  - proposed `8160`, target `90700`, rejected; stale optimistic next token
    `579` was drained before restore
  - proposed `16`, target `16`, accepted; optimistic next token `13` was
    committed as the next generated token
- tap failures: none

This proves Skippy can use a real pretrained SPD proposal to start actual
next-token target work in the serving path and preserve exact greedy output
through accept/reject. It is slower locally because proposal generation is
currently the bottleneck in that request-path smoke.

After the Qwen serving-head fast path, bounded release `spd-live-tap-parity`
runs matched ordinary non-SPD greedy output and rewound every verifier window.
The first release timing sample accepted all `3 / 3` live top-1 proposals and
averaged about `248ms` per verified step: `42ms` in the SPD head, `58ms`
assembling `cur_in`, and `107ms` in tap replay. Keeping sidecar projection
weights resident cut `cur_in` assembly to about `41ms`. Parallelizing the tap
projection matmul cut it again to about `5ms`, with the same accepted tokens
and greedy output. The latest eight-step release live-tap sample accepted
`7 / 8` top-1 proposals, kept exact greedy output, and averaged about `212ms`
per verified step: `41ms` in the SPD head, `5ms` assembling `cur_in`, and
`128ms` in tap replay.

The real OpenAI request path was then rerun in release mode on the same
tap-aligned seven-stage CPU topology with `max_tokens=8`,
`--openai-spd-optimistic-decode`, and selective downstream tap returns. Stage
configs for that run initially set `spd_tap_return_hf_indices = [10, 20, 31]`,
so downstream stages returned only the required taps for the active `g4`
fixture row instead of also sending non-required `16` and `24` tap frames. That
was too narrow for paper-shaped rolling rows. The in-repo
`skippy-bench spd-openai-smoke` command now reproduces this flow with a
topology-derived allowlist: it launches the local binary stages, starts
embedded stage-0 OpenAI serving, runs baseline and SPD requests, derives
`[8, 10, 16, 20, 24, 31]` from the sidecar topology after stripping h0, and
emits the telemetry summary. It can also run prompt files and now emits an
aggregate `summary` with paired content matches, mean baseline/SPD wall and
decode times, speedup ratios, total accept/reject counts, optimistic commits,
tap failures, and per-prompt comparisons. Native staged config also derives
that allowlist from the sidecar topology and carries it through the stage-load
protobuf into `StageConfig`; an empty allowlist preserves legacy "return all
downstream taps" behavior. Prompt files can now exercise plain prompts,
chat-style `messages`, and `turns` rows; `messages` are sent to the OpenAI chat
endpoint unchanged.

Recorded filtered release request-path progression:

- topology: same seven tap-aligned local CPU stages
- response content: `<think>\nThinking Process:\n\n1.  **`
- no-SPD baseline emitted the same text for the same request
- previous unfiltered optimistic SPD wall time: about `3.19s`; decode time:
  about `2.60s`
- first filtered optimistic SPD wall time: about `2.10s`; decode time: about
  `1.60s`
- filtered SPD after resident projection cache: about `2.22s` wall / `1.63s`
  decode; proposal time `398ms`, down from `478ms` before the cache
- filtered SPD after parallel tap projection: about `1.92s` wall / `1.38s`
  decode; proposal time `239ms`
- filtered SPD after switching optimistic target work from
  `CheckpointSession + DecodeEmbd` to a checkpointing one-token `VerifySpan`:
  about `1.95s` wall / `1.39s` decode; optimistic checkpoint telemetry fell to
  about `0.017ms`, with the same emitted text, `4` proposals, `1` accepted
  proposal, `3` rejected proposals, and `1` committed optimistic token
- latest same-topology no-SPD baseline: about `0.65s` wall / `209ms` decode
- tap-return filter proof: unfiltered returned `10`, `16`, `20`, `24`, and
  `31` eight times each; filtered returned only `10`, `20`, and `31` eight
  times each
- stage-0 tap records after filtering: local `8` plus downstream `10`, `20`,
  and `31`; no `16` or `24` returned records
- inline SPD windows: `4`
- accepted proposals: `1`
- rejected proposals: `3`
- committed optimistic target tokens: `1`
- latest inline probe elapsed times: about `70ms`, `55ms`, `53ms`, and `62ms`
  for the four ready proposals
- latest optimistic target-decode time: about `81ms` total
- latest optimistic target-reply wait time: about `64ms` total
- latest token downstream waits after filtering: about `230ms`, `185ms`,
  `185ms`, `20ms` for the committed optimistic token, then `141ms`, `159ms`,
  `156ms`, and `211ms`
- earlier aggregate pipeline gap: `4 / 4` pre-target probes proposed, `1 / 4`
  accepted, `3 / 3` post-target probes were empty, pre-target probes averaged
  `56.9ms`, normal downstream waits averaged `190.9ms`, the one optimistic
  downstream wait was `21.0ms`, and the earlier ungated no-tap path requested
  `0` reusable tap returns
- margin-gated tap-return smoke with `--spd-top-k 2
  --optimistic-min-logit-margin 2`: baseline `579ms` wall / `201ms` decode,
  SPD `2348ms` wall / `1871ms` decode, exact same emitted text, `6 / 6`
  pre-target proposals, `3 / 6` accepted, no empty post-target probes, `3`
  optimistic tap-return requests, `2` accepted tap-return requests, `1`
  rejected tap-return request, and `173ms` mean optimistic downstream wait
- lifecycle-filter rerun: exact same emitted text, `6 / 6` pre-target
  proposals, `3 / 6` accepted, `2` optimistic commits, `0` ignored stale taps,
  SPD `2351ms` wall / `1869ms` decode, baseline `569ms` wall / `206ms` decode
- topology-derived tap-return plus row-specific collection rerun: exact same
  emitted text, tap allowlist `[8, 10, 16, 20, 24, 31]`, downstream returned
  `10/16/20/24/31` while stage 0 recorded local boundary tap `8`, `8`
  proposals observed, `3` accepted, `5` rejected, `2` optimistic commits,
  baseline `547ms` wall / `212ms` decode, SPD `3523ms` wall / `2961ms` decode
- absolute row-position plus accepted-context catch-up rerun: exact same
  emitted text, tap allowlist `[8, 10, 16, 20, 24, 31]`, `8` proposals
  observed, `1` accepted, `7` rejected, `1` optimistic commit, live rolling and
  replay both reported `7` inserted drafts, `0` missing proposals, `0`
  out-of-order proposals, `2` rejected rolling windows, first rejected target
  position `24`, verified-up-to `30`, and verified prefix
  `[90700, 8340, 25, 271, 16, 13, 220]`. That run was timing-noisy
  (`2157ms` baseline wall / `1721ms` baseline decode, `3185ms` SPD wall /
  `2509ms` SPD decode), so use it as scheduler evidence, not a benchmark.
- paper-style rolling-pipeline estimate in `spd-openai-smoke`: logical SPD
  stage-count `4`, physical tap-aligned stage-count `7`, accepted proposal rate
  `0.375`, paper-like speedup versus serial split `1.5x`, estimated decode at
  baseline stage cost `141ms`, current SPD decode `2961ms`,
  current/paper-estimate slowdown about `20.9x`
- pipeline-gap telemetry for that same topology-derived run: `6 / 6`
  pre-target probes proposed, `3 / 6` accepted, `0` post-target probes, mean
  pre-target probe `63.9ms`, mean normal downstream wait `333.7ms`, mean
  optimistic downstream wait `272.6ms`, `3` optimistic tap-return requests, `2`
  accepted tap-return requests, `1` rejected tap-return request, and `0` ignored
  stale taps. The report now separates `optimistic_commit` probe counts and
  timings from ordinary pre-target probes so future smokes can quantify whether
  wait-overlapped commit probes are accepted often enough to justify the
  checkpoint/executor work needed to chain them into real target messages.
- pre-fix live rolling telemetry plus rolling trace replay in `spd-openai-smoke`:
  observed `pre_target_reply` and diagnostic `optimistic_commit` proposal order
  through `SpdRollingScheduler`, inserted `7` drafts, saw `0` missing proposal
  positions, kept `0` proposals out of order after accepted-context catch-up,
  verified `2` filled windows, rejected first at target position `24`, and
  ended with pipeline length `2` / verified-up-to `30`; the verified output
  prefix is `[90700, 8340, 25, 271, 16, 13, 220]` and
  `verified_prefix_matches_target=true`
- pre-fix rolling verified-delta telemetry in `spd-openai-smoke`: the live observer
  released target-verified deltas `[90700]`, `[8340]`, `[25, 271]`, `[16]`, and
  `[13, 220]` as the accepted-context catch-up advanced the verified frontier;
  this is telemetry only and the OpenAI path still emits through the existing
  target/optimistic token flow
- rolling row-role telemetry in `spd-openai-smoke`: inline probe reports now
  include `row_positions` and resolved inference `row_i_stages`. The runtime
  scheduler contract is nominal `g_{S-1}..g_0` before evicted-prefix acceptance
  and `g_S^evicted,g_{S-1}..g_0` after acceptance, but the serving proposal
  path resolves those roles through the manifest. For the current Qwen3.5-4B
  artifact, `trained_with_use_deepest=true`, so full snapshots upgrade to
  `[4, 4, 4, 0]`. The pre-fix reference-role smoke populated rows from the first
  probe (`[20, 21, 22, 23]` / `[4, 4, 4, 0]`) and kept them moving after
  catch-up (`[27, 28, 29, 30]` / `[4, 4, 4, 0]` at step `7`)
- row-specific tap collection: `skippy-server` now preloads projection weights
  for every topology row role, but each proposal waits only for the hidden taps
  required by the rows it is assembling; this keeps the old fixture-shaped
  probes from blocking on future rolling-row taps while still allowing paper
  rows to use them once the scheduler reaches those rows
- proposal-source breakdown telemetry: final `stage.openai_decode` reports
  `llama_stage.spd_proposal.total.*` so smokes distinguish inline tap hits from
  replay fallback. The first proposal-breakdown smoke accepted `8 / 8` with
  exact output but used replay fallback for all `8` proposals; tap collection
  consumed about `2205ms`, `cur_in` assembly about `51ms`, and sidecar forward
  about `509ms`, so the request path is bottlenecked on replayed hidden taps.
- inline-probe source/timing telemetry: `stage.spd_inline_probe` now reports
  `llama_stage.spd_inline_probe_tap_source`,
  `llama_stage.spd_inline_probe_tap_collect_ms`,
  `llama_stage.spd_inline_probe_cur_in_ms`, and
  `llama_stage.spd_inline_probe_forward_ms`. The no-replay optimistic rerun
  accepted `8 / 8`, committed `4 / 4` optimistic decodes, preserved exact
  output, and showed every pre-target and optimistic-commit probe using
  `tap_source=inline`. Final decode proposal totals now include those inline
  probe attempts (`inline_tap_hits=8`, `replay_fallbacks=0`), so the summary no
  longer hides direct-return work behind primary-window-only counters. The next
  overlap rerun moved accepted optimistic-commit probes into the in-flight
  optimistic reply wait (`trigger_hf_index=31`, about `0.001ms`
  wait-after-probe), preserving exact output and measuring `2800ms` SPD decode
  versus `202ms` baseline decode. This is real overlap evidence, but still not
  a speedup.
- Rust rolling-scheduler primitive: accepts the oldest completed entry after
  pipeline fill, emits the duplicated evicted-prefix/speculation row positions
  used by the reference loop, resets to the corrected target token on rejection,
  and now backs live request-path rolling telemetry through `SpdRollingObserver`
  plus its `take_verified_delta()` and `speculation_rows()` contracts;
  `skippy-bench`'s `summary.rolling_trace_replay` report is backed by
  `SpdRollingTraceReplay` for token/proposal traces and by final live
  `decode.rolling` telemetry for primary-verify-only cases
- token-position fix rerun: the same no-thinking prompt/template first
  reproduced the Python reference target tokens
  `[71093, 12305, 198, 727, 884, 2784, 11, 292]`; the HF reference accepted
  `7 / 8`, while the fixed Skippy request path accepted `8 / 8` against the
  GGUF target, committed `3 / 3` optimistic decodes, preserved exact output,
  and had no tap failures or ignored taps. Baseline was `626ms` wall / `198ms`
  decode; SPD was `3521ms` wall / `2921ms` decode.
- current always-tap optimistic rerun: exact output, accepted `8 / 8`, committed
  `4 / 4` optimistic decodes, `0` tap failures, `0` ignored taps, and rolling
  trace replay with `7` inserted drafts, `0` missing proposal positions, and
  `0` out-of-order proposals. Baseline was `629ms` wall / `201ms` decode; SPD
  was `4154ms` wall / `2919ms` decode. The pre-patch no-tap ungated diagnostic
  was faster (`2591ms` wall / `2004ms` decode) but left `5` missing proposal
  positions, so it is not a correct paper-row serving mode.
- current always-tap `25ms` downstream-delay rerun: exact output, accepted
  `8 / 8`, committed `4 / 4` optimistic decodes, `0` tap failures, `0` ignored
  taps, and rolling trace replay with `0` missing or out-of-order proposals.
  Baseline was `2807ms` wall / `1938ms` decode; SPD was `4210ms` wall /
  `3105ms` decode. The paper estimate for the same accepted trace was `484ms`
  decode, so this is still a serving-scheduler gap rather than an acceptance
  gap.
- tap failures: none

Resolving the trained Qwen head to reference fixture roles and adding the
native-cache path are both now validated pieces of the serving proof. The
cached fixture/export proves cache-logit parity with the Python reference on the
same pretrained Qwen3.5-4B head: after `20` prefix rows, Rust and Python cached
top-k token ids match exactly (`[23, 17, 24, 21, 16, 22, 760, 19]`) and full
cached-logit max diff is `0.0625`. The earlier `0 / 8` serving result was a
live token-position bug, not a cache mismatch. After fixing row positions, the
request path accepts the bounded proof prompt; the remaining gap is serving
latency and fully overlapped rolling scheduling.

The earlier nominal-role path was a material serving-path improvement from tap
filtering, projection execution, and removing the separate optimistic checkpoint
control message, but it was still slower than the ordinary split baseline on
local CPU. The remaining bottleneck is no longer `cur_in` assembly or
checkpoint-control overhead; it is the combination of proposal probe cost,
inflated downstream waits, and the fact that this is still a one-token
side-verify schedule rather than the paper's fully overlapped rolling loop.
Margin-gated tap returns prove accepted optimistic work can keep later inline
probes fed, but the tap-return path makes each optimistic verify wait roughly a
full downstream stage chain and rejected tap-return work is costly. This is
still a one-token side verify, not the paper/reference loop's rolling `n`-slot
pipeline where the oldest completed speculative entry is verified and evicted
while a new entry is inserted. The accepted-context lifecycle filter is a
correctness guard for late stale future taps; it does not change the fundamental
scheduling shape. The paper-style estimate makes that concrete: the aggregate
accept rate is high enough to imply speedup under the reference rolling
scheduler, while the current side-verify schedule is dominated by proposal and
downstream waits. The live rolling telemetry and offline replay give the next
rewrite an exact position/prefix contract to preserve.
The `SpdRollingScheduler` primitive makes the next serving rewrite concrete:
stage
0 needs to keep these scheduler entries in flight, produce a proposal for the
next scheduler position every pipeline step, route returned taps by scheduler
position, verify the oldest completed entry, and only reset the runtime/session
state when that oldest verification rejects. The direct smoke still injects
stage JSON explicitly through `skippy-bench`; native mesh propagation is covered
separately by host-runtime tests.
The existing direct-return prediction stream means stage 0 can write multiple
generation messages without waiting for each final prediction reply; the binary
stages can then naturally overlap work across processes. Rollback depth is no
longer session-only: `RuntimeState` stores checkpoints by session plus
`checkpoint_generation`, and the OpenAI path derives nonzero generations from
speculative target positions while reserving generation `0` for the legacy
single-checkpoint path. A
trim-to-position shortcut for SPD optimistic rejection was tested and not kept:
an `enable_thinking=true` rejection smoke exercised five rejected optimistic
decodes with `0` checkpoint time and no tap failures, but exact output changed
case (`Thinking Process` versus `Thinking process`). Current serving therefore
keeps checkpoint/restore for exactness. The restored checkpoint path reran the
same thinking-mode smoke with exact output, `1 / 8` accepted proposals, `6`
optimistic requests, `5` rejected optimistic decodes, `1` committed optimistic
token, no tap failures, about `0.038ms` total optimistic checkpoint time, and
about `20.8ms` total restore time. Generation-addressed checkpoints remove the
overwrite hazard; rolling execution still needs the scheduler that owns several
in-flight entries, verifies the oldest completed entry first, and restores the
matching generation only on rejection. The smoke benchmark now fails by default on
paired baseline/SPD content mismatch after writing its JSON report, so this kind
of rollback regression is not counted as a passing smoke;
`--allow-content-mismatch` is reserved for exploratory sweeps.

A one-prompt corpus smoke using
`crates/skippy-bench/corpora/speculative_coding_prompts.jsonl` with
`--prompt-limit 1` verified the new aggregate report shape against real staged
OpenAI serving. For `spec-code-001`, baseline and SPD emitted the same
two-token text. The summary reported `prompt_pairs = 1`, `matching_content = 1`,
SPD proposed `2`, accepted `1`, rejected `1`, committed `0` optimistic tokens,
and had no tap failures. It remained slower: baseline was about `805ms` wall /
`50.7ms` decode, while SPD was about `1338ms` wall / `438ms` decode
(`0.602x` wall speedup and `0.116x` decode speedup). This confirms the corpus
benchmark can collect honest evidence; it is not a speedup result.

A two-prompt smoke using `crates/skippy-bench/corpora/chat_corpus_fixture.jsonl`
with `--prompt-limit 2` verified that `spd-openai-smoke` preserves true
chat-style `messages` rows when constructing the OpenAI request. The prompt set
covered one flat prompt and one `{system,user}` message prompt. Baseline and
SPD emitted matching two-token text for both prompts. The aggregate summary
reported `prompt_pairs = 2`, `matching_content = 2`, SPD proposed `4`, accepted
`1`, rejected `3`, committed `0` optimistic tokens, and had no tap failures.
The mean baseline timing was about `451ms` wall / `53.3ms` decode; mean SPD
timing was about `975ms` wall / `447ms` decode (`0.462x` wall speedup and
`0.119x` decode speedup). This validates the chat-corpus benchmark path, not a
speedup.

The same `skippy-bench spd-openai-smoke` wrapper can pass native
`serve-binary` downstream wire conditioning to every launched stage. A `10ms`
downstream-message delay preserved exact output and narrowed the local gap:
baseline was about `2.51s` wall / `1.92s` decode, while SPD was about `2.80s`
wall / `2.14s` decode with `4` proposals, `1` accepted proposal, and one
committed optimistic token. At `25ms`, the current optimistic path became worse:
baseline was about `2.76s` wall / `1.94s` decode, while SPD was about `5.08s`
wall / `4.13s` decode. The reason is mechanical: rejected optimistic target
decodes also traverse the delayed downstream path before rollback. With
optimistic decode disabled at `25ms`, SPD accepted `3 / 8` probes but still took
about `3.65s` wall / `2.67s` decode because those accepted probes were measured,
not committed. The current local proof is therefore correctness-positive and
latency-instrumented, but still needs better acceptance/gating before it is a
speed path.

The request path now carries the SPD head's top-1 logit and top-1/top-2 logit
margin through inline probe telemetry and can gate optimistic target decode on
that margin. In the `25ms` latency calibration with `spd_top_k = 2`, the two
rejected optimistic proposals had margins `0.125` and `1.0`; the accepted
proposal had margin `2.5`. Running the same smoke with
`spd_optimistic_min_logit_margin = 1.5` skipped the rejected optimistic decodes
and still committed the accepted token. Paired current-code result: baseline
about `2.76s` wall / `1.91s` decode; gated SPD about `3.70s` wall / `2.83s`
decode; same emitted text; `4` measured proposals, `1` accepted, `0` rejected
optimistic decodes, and no restore cost. This is a real mitigation for bad
optimistic work, but it is still not a speedup with the current low-acceptance
proof head and local CPU stages.

The first optimistic tap-return implementation requested SPD tap returns only
after a margin gate allowed the optimistic target decode. That fixed the
post-accept coverage gap: after accepting the step-2 proposal, the next run
produced pre-target probes at steps `4`, `5`, and `6` instead of post-target
empty probes. With `25ms` downstream delay and a tighter `2.5` margin gate, SPD
measured `6` proposals, `3` accepted probes, `2` committed optimistic tokens,
and `0` rejected optimistic decodes. It was still slower than baseline: about
`4.03s` wall / `3.02s` decode versus baseline `2.89s` wall / `2.07s` decode.
At `10ms` delay the same gate measured baseline `1.86s` wall / `1.10s` decode
and SPD `2.52s` wall / `1.81s` decode. Current code requests those taps for
every optimistic SPD verify that is actually started, while the margin gate
only controls whether the optimistic work starts. That keeps accepted
optimistic tokens useful to later rolling rows, but tap-return and downstream
wait overhead still dominate for this CPU proof.

The request path now keeps inline tap-cache rows for the common token prefix
when the SPD source resets, dropping only rows at or after the first divergent
token. A pre-patch ungated no-tap diagnostic was faster because optimistic
target work did not return tap payloads, but it starved later rolling rows.
The retained behavior requests optimistic taps for every started optimistic SPD
verify, drops future rows when that optimistic decode is rejected, and preserves
accepted-extension rows after verification. Prefix reset remains conservative:
when the SPD source is reset to a prefix-compatible context, cached tap rows
before the first divergent token are preserved and divergent/future rows are
dropped. Accepted-prefix acknowledgements are now handled separately: if the
accepted context is already a prefix of the sidecar's longer verified context,
the tap lifecycle advances the accepted length without pruning rows that later
rolling proposals still need.

2026-06-17 checkpoint: the first local multi-token/repeat CPU run and the
first remote-run preflight are now recorded. The CPU repeat at
`/private/tmp/spd-local-multitoken-repeat-cpu.json` used the pretrained
Qwen3.5-4B S4/L4 sidecar, `--max-tokens 8`, `--warmup-count 1`, and
`--repeat-count 3`. It preserved exact output for `3 / 3` measured pairs,
accepted `24 / 24` proposals, committed `18` optimistic target tokens with
`12` chained commits, and kept rolling replay ordered (`21` inserted drafts,
`15` accepted windows, `0` missing, `0` out-of-order). It is still not a speed
result: baseline decode averaged `219.3ms`, SPD decode averaged `13964.2ms`,
and the paper-shaped estimate from the accepted trace was `54.8ms`. The timing
split is important: sidecar cache prefill averaged `16.8ms` over `24` probes
and sidecar head total averaged `45.9ms`, while normal downstream wait averaged
`2681.2ms` and optimistic hidden wait averaged `2169.6ms`. That makes the
remaining overhead a native serving/scheduler gap, not evidence that the paper
or reference sidecar cache mechanics failed to port.

The matching Metal repeat at
`/private/tmp/spd-local-multitoken-repeat-metal.json` preserved the same
correctness shape (`3 / 3` content matches, `24 / 24` accepted proposals,
`18` optimistic commits, `12` chained commits, and ordered rolling replay).
Metal reduced mean SPD decode from `13964.2ms` to `1652.6ms` and optimistic
hidden wait from `2169.6ms` to `90.5ms`, but it still lost to the `201.0ms`
baseline decode (`0.122x`) and remained about `32.9x` slower than the
paper-shaped estimate. The next speed evidence must come from distinct-device
or real-node placement, not another same-machine repeat.

The first no-launch remote preflight at
`/private/tmp/spd-qwen35-first-remote-preflight.json` validated the release
`skippy-server` binary, GGUF, manifest, `66` serving checkpoint tensors, `28`
parity fixture tensors, logical `S=4`, physical split `8,10,16,20,24,31`, tap
returns `8,10,16,20,24,31`, local stage port `20031`, and a complete
stage-0-local plus worker endpoint plan. Use
`spd-openai-smoke --preflight-only` before a real node launch, then remove the
flag for the smoke once model paths and reachable endpoint maps are real.

## What Does Not Work Yet

- The `spd-replay` request path has a correctness fallback, not a final speed
  path. `--openai-spd-replay-fallback` replays taps through local stage slices
  before feeding proposals into the existing verify/repair/rollback loop.
- Inline hidden-tap capture, direct-return transport, and opt-in optimistic
  next-token decode work for the local tap-aligned proof. The latest always-tap
  request-path release smoke accepts the bounded proof prompt and keeps rolling
  replay ordered, but it is still slower than baseline: `4154ms` wall /
  `2919ms` decode versus `629ms` wall / `201ms` decode on the same local CPU
  topology, and `4210ms` wall / `3105ms` decode versus `2807ms` wall / `1938ms`
  decode with `25ms` downstream delay. Optimistic tap returns keep proposal
  coverage after accepted optimistic tokens, but the current path still waits on
  normal downstream target work and only commits one-token optimistic verifies.
  The request path still needs to turn the row/delta diagnostics into the
  paper/reference rolling execution schedule instead of only observing it.
- Optimistic target messages now request SPD tap returns whenever optimistic
  SPD work starts. The accepted-context lifecycle filter prevents stale future
  taps from polluting the inline cache. A production path still needs
  cancellation, side-channel draining, or another way to promote speculative
  taps only after acceptance without delaying rollback.
- Primary SPD `VerifySpan` windows now mark all input positions in the span as
  pending tap positions before target verification. Fully accepted spans promote
  those rows with the accepted context; rejected spans reset and drop divergent
  future rows. This fixes a coverage gap where verified multi-token SPD windows
  could request taps but then ignore them as stale future rows before the target
  decision was known. `skippy-bench spd-openai-smoke` now exposes
  `--spd-replay-fallback`, which passes `--openai-spd-replay-fallback` through
  to the embedded stage-0 server; pair it with `--optimistic-decode false` to
  force primary SPD `VerifySpan` windows for correctness evidence. A bounded
  `max_tokens=4` release smoke
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
- Request-path acceptance has been proven on bounded four- and eight-token
  smokes, but a larger local request-path acceptance/latency sweep is still
  needed.
- No larger-than-4B head has been trained by us yet.
- No real distributed SPD speedup has been measured yet. The paper/reference
  result and the earlier Skippy latency model are strong evidence that high
  acceptance can translate into large split-serving gains, but they are not a
  native wall-clock result. Current local Skippy smokes run several stage
  processes on the same machine/device, which can make SPD appear much slower
  because speculative work creates real concurrent stage compute on shared
  resources. Until a second worker is available again, treat local runs as
  correctness, row-schedule, rollback, and instrumentation validation. Use
  injected downstream delay only as a bounded diagnostic, not as a substitute
  for a distinct-device benchmark.

## Correctness Contract

SPD should be treated as a verified speculative path.

For greedy decoding:

1. SPD proposes a token.
2. The target model computes the verified logits for that position.
3. The token is accepted only if it equals the target argmax.
4. On rejection, Skippy rolls back speculative state and emits the target token.

For sampling:

1. SPD proposes from draft distribution `q`.
2. Target distribution `p` is computed by the base model.
3. Standard speculative rejection sampling accepts with the corrected
   probability and otherwise samples from residual `max(0, p - q)`.

Do not ship unverified/lossy SPD as the default path. Lossy SPD should only be a
separate explicit experiment because wrong accepted tokens change the future
context.

## Practical Skippy Hosting Model

Treat the SPD head as a sidecar artifact attached to a Skippy stage topology.
It should not be exposed as a separate OpenAI model and should not mutate the
base GGUF/layer-package weights.

Recommended first implementation:

- one SPD sidecar runtime per active Skippy topology/session group
- host it in one Skippy process first, likely coordinator or final-stage side
- other stages expose/send selected hidden-state taps
- SPD implements the `skippy-server` speculative proposal-source boundary
- normal Skippy stages verify every emitted token through the existing
  verify/repair/rollback path

Topology binding matters. An SPD sidecar is bound to the base model/tokenizer
and to the logical hidden-tap topology it was trained/exported for: number of
target stages, optional `stage_layer_boundaries`, tap indices, and projection
weights. Physical Skippy placement can change only if it exposes the same
logical taps to the sidecar. For example, the current pretrained Qwen3.5 S4/L4
head can be served by a seven-slice tap-aligned proof topology because that
topology exposes the required logical taps `0,8,10,16,20,24,31`. A production
four-stage topology would either need an internal hidden-tap ABI for the missing
internal taps or a sidecar trained and exported for that topology.

The mesh-native config path now has an experimental default-off SPD surface in
`[defaults.speculative]` / per-model `speculative`: `mode = "spd"`,
`spd_manifest_path`, `spd_fixture_path`, `spd_model_path`, `spd_max_tokens`,
`spd_top_k`, `spd_gpu_layers`, `spd_replay_fallback`, and
`spd_optimistic_decode`, plus `spd_optimistic_min_logit_margin` for optimistic
decode margin gating. The resolver rejects mixed draft-model and SPD
sources, treats SPD as staged-only, and passes these fields through to the
embedded stage-0 OpenAI runtime. It also derives the downstream tap-return
allowlist from the sidecar topology, strips hidden-state index `0`, and
forwards that list through mesh stage-control load requests so workers return
every tap required by the paper row roles without returning h0. Focused
host-runtime coverage now asserts that `split_generation_load_settings` derives
`[8, 10, 16, 20, 24, 31]` and that `split_runtime_stage_load_request` carries it
into the worker load request. This is config plumbing for native mesh serving,
not topology auto-selection or sidecar training.

Current experimental serving hook:

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

The default source uses inline tap cache rows only. Add
`--openai-spd-replay-fallback` to run the older slow local `StageModel` replay
path for request-path correctness proofing; do not use that fallback as the
final performance architecture.

Distributed SPD execution across all stage nodes may become useful later, but it
is not the first proof path.

## Sidecar Training and Package Plan

The paper/reference repository already contains the training path (`train.py`),
the inference/eval path (`pipeline_inference.py`, `eval.py`), and the model
definition (`pipeline_model.py`). Skippy should not reimplement training in Rust
first. The practical path is to wrap the Python trainer/evaluator and make the
result look like a Skippy-resolvable sidecar artifact.

Recommended workflow:

1. `plan`: inspect the base model config/GGUF metadata and choose a logical SPD
   topology: logical stage count, tap layer groups, spec-layer count, draft
   vocab, and whether internal taps are required.
2. `train`: run the reference trainer for that base model/topology and freeze
   the base model exactly as the paper does.
3. `eval`: run the reference eval on the bundled MT-Bench, GSM8K, and HumanEval
   style prompts; record acceptance, `L'_acc = (generated_tokens / steps) * S`,
   and paper-style theoretical gain.
4. `export`: convert `speculation_head_final.pt` into Rust-readable
   `spd-head.safetensors`, `skippy-spd-head.json`, draft-vocab metadata, and a
   parity fixture.
5. `validate`: run Rust fixture parity, cached fixture parity, live-tap parity,
   and `spd-openai-smoke`.
6. `publish`: upload the sidecar artifact bundle to a Hugging Face repo or
   model-package sidecar directory with explicit compatibility metadata.

The sidecar package must bind to the base model/tokenizer identity and logical
tap topology, not to a physical node count. Physical Skippy placement is valid
only when it exposes the same logical taps to the sidecar. A future Skippy
resolver should therefore match a requested base model plus available physical
split plan against compatible SPD sidecar manifests, then reject or re-plan when
required taps cannot be produced.

## Spec-Decoding Benchmark Baseline

The imported `llama-spec-bench` target/draft diagnostic now runs after fixing
its target lane count: speculative generation needs both a target verifier
session and a projection session. A bounded local run used
`Qwen3-4B-Q4_K_M.gguf` as target and `Qwen3-0.6B-Q4_K_M.gguf` as draft with
`max_new_tokens=8`, `speculative_window=4`, `ctx_size=512`, and
`n_gpu_layers=0`.

Recorded one-prompt result:

- tokenizer match: `true`
- speculative output matched target baseline: `true`
- draft tokens: `13`
- accepted tokens: `4`
- rejected tokens: `4`
- accept rate: `30.8%`
- target baseline: `84.54 tok/s`
- current serial speculative path: `57.06 tok/s`
- projected batched rollback: `59.76 tok/s`
- projected scratch verification: `18.75 tok/s`

Recorded release eight-prompt corpus sweep against the first eight
`kv_mixed_prompts.jsonl` rows:

- prompt count: `8`
- tokenizer match: `true` for every prompt
- speculative output matched target baseline for every prompt
- prompt tokens: `138`
- generated tokens: `64`
- speculative windows: `39`
- draft tokens: `127`
- accepted tokens: `29`
- rejected tokens: `35`
- accept rate: `22.8%`
- mean accepted tokens per window: `0.74`
- target baseline: `79.31 tok/s`
- current serial speculative path: `53.69 tok/s`
- projected batched rollback: `50.03 tok/s`
- projected scratch verification: `13.00 tok/s`

This is not an SPD result and not a speed win. It is a useful native benchmark
baseline showing the target/draft speculative harness can now measure tokenizer
agreement, acceptance, correctness, and projected verifier strategies on real
GGUF pairs. On this target/draft pair, broader prompt coverage made the
acceptance problem clearer: the draft is useful diagnostically but not a
serving-speed candidate as configured.

## llama.cpp / Stage Runtime Dependencies

The current proof branch does not require James's GLM/MTP work to reproduce the
Python SPD results or validate the Rust manifest. The live Skippy path will,
however, need additional staged-runtime/llama-side capability.

Likely required:

- hidden-state tap export from selected layers/stages, with token position,
  dtype, shape, and stage ownership metadata
- enough sideband transport to return those taps to the SPD sidecar without
  changing ordinary generation output
- verification support that can run proposed SPD tokens through the real target
  stages and return the target-model decision
- rollback/session-trim support for rejected speculative tokens
- ABI version bumps and Rust `skippy-ffi` mirrors for any new staged-runtime
  calls

Adjacent work that may help:

- native MTP verification work has similar concerns around speculative proposal,
  target verification, sideband data, and rollback
- GLM/MTP branches may contain useful patterns for verifier plumbing, but they
  are not SPD themselves and should not be merged wholesale just to start SPD
- package-declared draft speculation work may be useful later for advertising
  optional SPD artifacts in model/layer packages

First Skippy implementation should add the minimum SPD-specific stage-runtime
surface needed for Qwen3.5-4B parity: capture required hidden taps, run the SPD
head, and verify proposed tokens. Pull reusable verifier/rollback patterns from
MTP work only after confirming they apply cleanly to SPD.

## Artifact Layout Target

Layer packages should eventually support optional SPD artifacts:

```text
package/
  manifest.json
  parts/
    ...
  spd/
    skippy-spd-head.json
    spd-head.safetensors
    draft-vocab.json
```

The manifest must bind the head to:

- base model digest/path
- tokenizer/vocab identity
- hidden size
- split topology
- stage count
- layer taps
- draft vocab
- head tensor checksum

## Reproduction Commands

### Train Small Qwen3-0.6B Head

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

Use `--device cuda` on a CUDA host.

### Evaluate Pretrained Qwen3.5-4B Head

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

### Export Serving Checkpoint

```bash
python3 evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

### Export Rust/Python Parity Fixture

```bash
python3 evals/spd/export_parity_fixture.py \
  --reference-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/speculative_pipeline_decoding \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --base-model-path Qwen/Qwen3.5-4B \
  --out /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --device mps \
  --top-k 8
```

The fixture includes raw hidden-state tap rows used to build SPD `cur_in`.
Use `skippy-bench spd-fixture-parity` for one JSON report covering tap-row
projection parity and Qwen SPD forward/top-k parity:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --top-k 8
```

### Simulate Split Latency From Trace

```bash
python3 evals/spd/simulate_latency.py \
  --raw /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/eval/raw/pipeline_eval__train__speculation_head_final__nt24__per_sample.jsonl \
  --stage-ms 4,4,4,4 \
  --hop-ms 0,1,5,10,25
```

## Engineering Next Steps

### Milestone 1: Tensor Export

Goal: produce a Rust-serving artifact from the `.pt` checkpoint.

Tasks:

1. Add/export a `safetensors` writer for the SPD checkpoint. Done in
   `evals/spd/export_spd_head.py`.
2. Preserve tensor names, shapes, dtype, draft vocab ids, and config. Done via
   the safetensors file and `skippy-spd-head.json`.
3. Extend `skippy-spd-head.json` to reference the serving checkpoint. Done with
   the optional `serving_checkpoint` section.
4. Add a small shape/checksum inspection command or test fixture. Done in
   `skippy-runtime` tests.
5. Add a minimal safetensors payload reader for BF16/F32/I64 tensors. Done in
   `crates/skippy-runtime/src/spd/safetensors.rs`.

Exit criteria:

- Qwen3.5-4B SPD head exports deterministically.
- Rust can validate the manifest, enumerate expected tensors, and read selected
  tensor payloads from the serving checkpoint or parity fixture.
- To validate a local exported head through Rust, set
  `SKIPPY_SPD_MANIFEST=/tmp/.../train/skippy-spd-head.json` and run:

```bash
cargo test -p skippy-runtime validates_external_manifest_when_skippy_spd_manifest_is_set
```

### Milestone 2: Rust Forward Pass Parity

Goal: Rust computes the same draft candidates as Python for recorded inputs.

Tasks:

1. Record hidden-state tap fixtures from Python reference execution. Fixture
   export is implemented in `evals/spd/export_parity_fixture.py`.
   It now records raw tap-row tensors before Python applies `g0_proj` or
   `stage_projs.*`.
2. Implement the Qwen3.5-4B SPD head forward pass in Rust. Done for the
   recorded fixture path in `crates/skippy-runtime/src/spd/qwen.rs`.
3. Compare Rust top-k draft candidates against Python top-k on the same hidden
   states. Done for the pretrained Qwen3.5-4B fixture.
4. Add focused tests with small fixture tensors and opt-in real-artifact tests.
   Done in `skippy-runtime` tests.
5. Reconstruct SPD `cur_in` from raw tap rows in Rust before running the head.
   Done in `crates/skippy-runtime/src/spd/tap_input.rs` and exposed through
   `skippy-bench spd-fixture-parity`.

Exit criteria:

- Rust top-k proposals match Python within tolerance on recorded fixtures.
- No Skippy serving integration is required for this milestone.
- Recorded real-artifact parity:
  - Tap input reconstruction max absolute diff: `7.62939453125e-6`
  - Rust/Python draft indices:
    `[7728, 15014, 38999, 10036, 11235, 13293, 15953, 0]`
  - Full token ids:
    `[9419, 21251, 109266, 12675, 14556, 18103, 23066, 0]`
  - Spec-query max absolute diff: `0.03125`
  - Final-hidden max absolute diff: `0.125`

```bash
SKIPPY_SPD_MANIFEST=/tmp/.../train/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/tmp/.../train/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set
```

Or run the single bench report:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/.../train/skippy-spd-head.json \
  --fixture /tmp/.../train/spd-parity-fixture.safetensors \
  --top-k 8
```

### Milestone 3: Skippy Hidden-State Taps

Goal: Skippy can expose the hidden states the SPD head needs.

Tasks:

1. Identify the target layer taps from `skippy-spd-head.json`. Done in
   `skippy-runtime` tap-planning tests.
2. Decide proof topology:
   - fastest proof: tap-aligned over-split so required taps are ordinary stage
     boundaries. Done for the local Qwen3.5 seven-stage binary chain.
   - production path: add an internal hidden-tap ABI so ordinary four-stage
     serving can expose taps `10,20,31`
3. Add a hidden-state sideband/tap path in the staged runtime if not using the
   over-split proof path.
4. Validate dtype, shape, token position, and stage ownership.
5. Write a correctness test that compares tapped hidden states against a known
   reference for a small prompt.

Exit criteria:

- Skippy can capture the required taps for a live prompt without changing
  normal generation output.

### Milestone 4: Live Verified SPD in Skippy

Goal: Skippy uses SPD proposals during generation and verifies every token.

Tasks:

1. Add a proposal-source boundary in `skippy-server`. Done for the current
   draft-model path and reused by SPD.
2. Wire SPD proposal generation into that boundary. Experimental replay-tap
   source is in place behind `--openai-spd-manifest` / `--openai-spd-fixture`.
3. Feed proposals into the existing target verification path. Done by sharing
   the same `VerifySpan` verifier/repair loop as the draft-model source.
4. Roll back speculative KV/session state on rejection. Done through the
   existing verifier reset path for any rejecting proposal source.
5. Emit metrics for proposals, accepted tokens, rejected tokens, equivalent
   accept length, and decode-loop steps. Basic request-path proposal/accept
   telemetry is in place; equivalent accept length and sweep reporting still
   need promotion into the benchmark/report layer.
6. Run ordinary split serving and SPD serving against the same prompts. Done
   for one bounded local smoke; still needed as a broader sweep.
7. Replace replayed local tap collection with inline hidden-tap capture and
   transport so performance can match the SPD pipeline design. Stage-0
   positioned tap cache/overlay and downstream direct-return tap transport are
   in place for the tap-aligned local proof. The request path has been rerun
   after the Qwen head fast path with selective tap returns; broader sweeps and
   proposal/return overhead reduction remain.

Exit criteria:

- Greedy outputs match ordinary target-model decoding.
- Acceptance and equivalent accept length are non-zero and close to reference
  trace behavior.
- Latency improves under injected hop/stage delay.

### Milestone 5: Larger-Model Training Proof

Goal: prove the SPD head generation pipeline scales beyond the pretrained 4B
artifact.

Recommended first target:

- a larger Qwen-family model with architecture/tokenizer support close to the
  reference implementation
- avoid custom huge MoE targets for the first scaling proof

Tasks:

1. Train with open conversation data mix.
2. Keep draft vocab capped at 32k or 50k.
3. Evaluate on the same MT-Bench/HumanEval/GSM8K prompt sets.
4. Record acceptance, equivalent accept length, and latency simulation.
5. Publish only artifact manifests, scripts, and aggregate metrics unless model
   licensing allows the trained head to be shared.

Exit criteria:

- Larger-model head reaches useful equivalent accept length.
- Training process and artifact production are reproducible by another engineer.

## Branch Scope

This branch should stay focused on SPD proof and handoff material:

- `SPD_SKIPPY_PROJECT.md`
- `evals/spd/`
- `crates/skippy-runtime/src/spd.rs`
- `crates/skippy-runtime/src/spd/`
- minimal module export from `skippy-runtime`

Avoid mixing in unrelated MTP, GLM, packaging, branch-reconciliation, or private
lab automation work. Those can be inputs later, but the purpose of this branch
is to make the SPD path clear and reproducible.

## Validation

Run:

```bash
python3 -m py_compile evals/spd/hf_train_eval_qwen06.py evals/spd/simulate_latency.py evals/spd/export_spd_head.py evals/spd/export_parity_fixture.py
cargo fmt --all -- --check
cargo test -p skippy-runtime spd
SKIPPY_SPD_MANIFEST=/tmp/.../train/skippy-spd-head.json SKIPPY_SPD_PARITY_FIXTURE=/tmp/.../train/spd-parity-fixture.safetensors cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set
cargo run -p skippy-bench -- spd-fixture-parity --manifest /tmp/.../train/skippy-spd-head.json --fixture /tmp/.../train/spd-parity-fixture.safetensors --top-k 8
cargo clippy -p skippy-runtime --all-targets -- -D warnings
```

Before publishing or handing off, run the repo's normal secret scan and also
check the diff for private hostnames, private IPs, access tokens, credentials,
and absolute developer-machine paths.
