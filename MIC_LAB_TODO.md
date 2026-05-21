# Lab / overnight TODO (running list)

Captured from the conversation so I don't goldfish them. Updated as
items are done.

## In flight

- [x] **PR #620 — opinionated MoA no-think.** Flipped default to
  no-think; escape hatch via reasoning knobs preserved. 4 new unit
  tests, live smoke 3/3 clean, PR title + description updated.
  **CI green (19/19). 5h+ lab data confirms identical performance to
  explicit no_think probe (3175ms vs 3328ms).**

- [x] **Public-mesh per-model tok/s probe.** 183 samples over 5h+.
  Confirms Mic's intuition: Qwen3-8B is bimodal (0.28 - 24.47 tok/s,
  86x spread). Qwen3-32B never succeeds. Qwen2.5-3B consistent slow.
  Data is exactly the input PR #596 + a small routing layer needs.

- [x] **Overnight report**: `MIC_LAB_REPORT.md` written + pushed.

## Queued (in priority order Mic asked them)

- [ ] **Public-mesh per-model tok/s probe.** Join the public mesh as
  client, walk `/v1/models`, hit each model with the same prompt, log
  first-token latency + tok/s + success rate. Output table. Pure
  observation — doesn't disturb the private lab.

- [ ] **MoA streaming (Issue #618).** Make MoA emit incremental
  `response.output_text.delta` events instead of one-shot delta on
  `/v1/responses`. Three options sketched in the issue; start with C
  (chunk the already-buffered winner) for fastest visual feedback,
  consider A (splice the winner's stream once arbiter commits) if C
  works. Lab here is the right place to A/B it.

- [ ] **Health-aware routing.** Speculation but worth a look: peers
  that just timed out shouldn't be picked again for N seconds. Check
  Issue #596 (per-peer tok/s gossip — open already?) to see if the
  routing layer can take advantage. Don't over-build; small layer on
  top.

- [ ] **General stability / lab continued.** Keep the 3-node mesh
  probe running overnight, watch for new `LastOpenPath` patterns,
  watch the `mesh_chat` vs `mesh_no_think` A/B mature, watch for any
  regression from the same-target retry + no-think changes.

- [ ] **Tomorrow morning report.** Concrete numbers + screenshots,
  what shipped, what blocked, what's still open.

## Stability concerns to keep watching while working

- Did anything regress with the no-think change? (Tool path, reducer,
  failure modes.)
- Same-target QUIC retry from `micn/quic-retry-on-connection-lost`:
  no-think runs are faster, so less window for path teardown. Should
  be fine but worth watching.
- Connection events overnight — count + which peer involved.

## Done

- [x] PR #615 — chat redirect + sole-answer grace + Responses-API
  adapter — **MERGED**.
- [x] PR #619 — comment AUTO_BACKEND_MODEL as one-line flip point.
- [x] Issue #617 — opened, will be closed by PR #620.
- [x] Issue #618 — opened (MoA streaming).
- [x] PR #620 — feat(moa): propagate enable_thinking. Honest testing
  comment posted. Now needs the "opinionated default" change.
- [x] Lab v3 → v4 with new binary. 3-node mesh restored. Probe and
  watch-conn running.
- [x] Lab notes captured in `MIC_LAB_NOTES.md`.
- [x] PR #615 binary deployed to fly app and verified live (v0.66.0).
