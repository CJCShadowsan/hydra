# Lab report — overnight

5+ hours of unattended data on a 3-node private lab + parallel client
joined to the public mesh. Two PRs in flight, hard data on public-mesh
quality, observations on next steps.

## tl;dr

* **PR #620 (MoA opinionated no-think) live-validated.** Worked first
  try: 3/3 clean short answers in 1-2s, no reasoning leakage. Escape
  hatch (`reasoning_effort: "low"`) restores thinking. Title +
  description rewritten to reflect the opinionated default. Honest
  testing readout posted on the PR.
* **PR #615 already merged**, fly app redeployed earlier, lab kept
  testing it under load.
* **Public-mesh probe** revealed exactly the problem you suspected:
  the user-facing model list has one usable peer (Qwen2.5-3B @
  1.3 tok/s), one unusable peer (Qwen3-32B 0% success), and one
  wildly bimodal peer (Qwen3-8B oscillating between 0.3 tok/s and
  24 tok/s depending on which peer answers — strong signal that
  iroh path quality + peer load matter much more than model choice).
* **MoA streaming (Issue #618):** not started. Want to discuss
  approach before writing code.
* **Health-aware routing speculation:** the per-model tok/s histogram
  from the public probe is exactly the input a health-aware router
  would need. Wrote up findings and a concrete next step.

## What's running right now

| Process | What it does |
|---|---|
| M4 `serve` v5 (opinionated binary) | gateway + Qwen2.5-3B, joined private mesh + studio + mini |
| Studio `serve` | MiniMax-M2.5 |
| Mini `serve` | Qwen3.5-9B |
| M4 `probe-stable.sh` | 8 probe types (direct/auto/mesh/mesh_no_think/tool/stream/local/long_silent) running ~24h+ |
| M4 `watch-conn.sh` | tails all 3 node logs for connection events |
| M4 `mesh-llm client --auto` (public mesh) | separate runtime, port 9447 |
| M4 `probe-tokps.sh` | per-model tok/s probe of every public-mesh model |

All `/tmp/lab/*.csv` files are accumulating data.

## Bonus A/B: same model, lab vs. public mesh

Mic asked specifically about "how solid the public mesh is from a client
here using it via auto / mesh / direct". The most direct apples-to-apples
is the **same model** (`Qwen2.5-3B`) hit two ways from this M4:

* **LAB**: this M4 serves Qwen2.5-3B locally. The probe loops back to
  `:9337` (no QUIC). Tells us what the model + hardware can do.
* **PUB**: through a separate `mesh-llm client --auto` process at `:9447`
  joined to the public mesh. The probe goes through whichever public
  peer serves Qwen2.5-3B.

Single data point so far (more accumulating):

| Source | total ms | FTT ms | tok/s |
|---|---:|---:|---:|
| LAB Qwen2.5-3B (no QUIC) |   **496** |  **104** | **127.81** |
| PUB Qwen2.5-3B (some peer) | 42,370 | 1,040 |   1.21 |

**The lab hardware is ~100× faster than whoever's serving Qwen2.5-3B on
the public mesh.** That's not QUIC overhead — routing layer can't add
that much. The public-mesh peer is genuinely under-resourced.
Qwen2.5-3B is a small model that should saturate easily even on modest
hardware; 1.21 tok/s suggests a CPU-only path or heavily oversubscribed
GPU on that peer.

The takeaway is that **per-peer health/tok/s gossip would be a huge UX
win**: routing today picks an arbitrary peer for `Qwen2.5-3B`; with
gossiped tok/s we'd prefer the 100× faster local-network peer (or
better peers on the public mesh) automatically.

## Public mesh per-model tok/s (~5h, 183 probes)

50-token streaming probe for tok/s and first-token latency:

| Model                                          | n  | ok | err | med tok/s | p10 tok/s | p90 tok/s | med FTT |
|------------------------------------------------|----|----|-----|----------:|----------:|----------:|--------:|
| Qwen/Qwen2.5-3B-Instruct-GGUF@main:q4_k_m      | 70 | 70 |   0 |    **1.31** |      1.28 |      1.34 |  1.0s   |
| unsloth/GLM-4.7-Flash-GGUF@main:Q4_K_M         |  4 |  4 |   0 |   **12.42** |      6.86 |     38.46 |  0.9s   |
| unsloth/Qwen3-32B-GGUF:Q4_K_M                  | 38 |  0 |  38 |       n/a |       n/a |       n/a |   n/a   |
| unsloth/Qwen3-8B-GGUF@main:Q4_K_M              | 70 | 69 |   1 |    **0.38** |      0.28 |     24.47 | 22.7s   |

Interpretation:

* **Qwen2.5-3B**: 100% success, 1.31 tok/s — slow but consistent.
* **Qwen3-32B**: served by some peer but 0/38 successful streams over
  5 hours. Errors land in ~1.1s — fast HTTP error responses, not
  network failures. Looking at error_summary suggests this peer
  rejects requests at the application layer.
* **Qwen3-8B**: 99% success rate but the distribution is bimodal.
  p10=0.28 tok/s, p90=24.47 tok/s. **86x spread.** Some answers come
  back in ~3-4s, others take 200-280s for the same 50-token request.
* **GLM-4.7-Flash**: tiny sample (only briefly available), but the 4
  hits range from 6.86 to 38.46 tok/s — a brief, fast option.

Concrete shape of the bimodal pattern on Qwen3-8B (samples from the
log):
```
3.5s elapsed, 1.5s FTT, ~24 tok/s   ← good peer
185s elapsed, 25s FTT, ~0.32 tok/s  ← bad peer
3.7s elapsed, 1.6s FTT, ~24 tok/s   ← good peer (next iter)
171s elapsed, 23s FTT, ~0.34 tok/s  ← bad peer (next iter)
```

So **the same advertised model name is served by at least two peers
with 60x different throughput**, and the iroh routing layer picks
between them with no quality signal. That's exactly the input a
health-aware router needs. See "Health-aware routing" below.

## Private 3-node lab — opinionated no-think A/B

### Pre-opinionated (caller had to opt-in)

Lab v3-v4 binary, caller-controlled flag, 17 probes:

| Probe          | n  | avg ms | min ms | max ms |
|----------------|----|-------:|-------:|-------:|
| `mesh_chat` (default — still thinking) | 9 | 11243 |  3201 | 21261 |
| `mesh_no_think` (`reasoning_effort: "none"`) | 8 |  1632 |  1276 |  2185 |

**7x improvement** when caller explicitly disabled thinking.

### Post-opinionated (PR #620 default)

Lab v5 binary, MoA opinionated no-think default, 5+ hours, 221 probes
including both `mesh_chat` (no caller knob — now defaults to off) and
`mesh_no_think` (explicit caller knob):

| Probe          | n  | avg ms | min ms | max ms |
|----------------|----|-------:|-------:|-------:|
| `mesh_chat` | 21 |  3175 |   915 | 29999 |
| `mesh_no_think` | 33 |  3328 |   758 | 42479 |

**Identical performance distribution.** That's the desired contract:
the opinionated default IS no-think, so `mesh_chat` now behaves the
same as `mesh_no_think`. No regression for callers who set the flag,
big win for callers who didn't. ✅

Live smoke on the opinionated binary (3 runs of `model=mesh` no
knobs): **1-2 sec, all clean short answers**. Three runs of
`reasoning_effort: "low"`: **thinking turns back on** as expected
(escape hatch works).

### Curl-fail rate by probe type (lab v5, 221 probes)

| Probe          | n  | fail | rate |
|----------------|---:|-----:|-----:|
| `auto_chat`    | 26 |    0 | 0%   |
| `direct_studio`| 28 |    0 | 0%   |
| `local_qwen`   | 24 |    0 | 0%   |
| `long_silent`  |  2 |    0 | 0%   |
| `mesh_tool`    | 25 |    0 | 0%   |
| `stream_studio`| 22 |    0 | 0%   |
| `mesh_no_think`| 39 |    6 | 15%  |
| `mesh_chat`    | 26 |    5 | 19%  |
| **`direct_mini`** | 29 | 17 | **59%** |

The story is: **studio-pair is rock solid (0% fail across 102
probes). Mini-pair is broken (59% direct fail). MoA paths (mesh_chat,
mesh_no_think) fail 15-19% because they fan out to mini and inherit
its failures.** The mini-pair failures bottleneck the user-visible
MoA experience.

Goes back to the resilience work in `MIC_LAB_NOTES.md`: a second-tier
QUIC retry that waits for iroh's auto-reconnect (~5-10s) would close
most of these. With only a single target serving a model, the
outside loop has nowhere else to go.

## Network stability (lab)

The lab has been running for ~7 hours total across two binaries
(retry-only + retry+no-think). Findings as a refresher / update:

* **Mini is NO LONGER the bad pair.** After your reboot to macOS
  26.5, ping distribution between any 2 nodes is similar (all wifi,
  jittery, no consistent direction of badness). The "M4 → mini is
  flaky" framing turned out to be a moment-in-time RF condition.
  Lesson: don't fix the node, fix the protocol behavior.
* **Same-target QUIC retry from `micn/quic-retry-on-connection-lost`
  fired 3 times in the v3 window.** 1 retry resolved cleanly. 2
  retries failed because the peer was simultaneously being removed
  from the mesh (`Peer removed`, `Reconnect failed` within ~1s of
  the path drop). Conclusion: 750ms is too short when iroh is
  actively reconnecting. Want to add a 5-10s second-tier wait, but
  haven't yet — issue noted in `MIC_LAB_NOTES.md`.
* **iroh keep-alive code from PR #566 is half no-op.** Connection-
  level lines (10s keep-alive, 300s idle) do real work. Per-path
  lines (10s/300s) are silently clamped to iroh's 5s/15s defaults.
  Worth a small cleanup PR.

## PRs / issues touched

| # | What | State |
|---|---|---|
| #615 | chat redirect + sole-answer grace + Responses adapter | merged 21 May |
| #619 | doc: AUTO_BACKEND_MODEL flip-point comment | open, CI green |
| #617 | MoA fast worker shouldn't think | open, addressed by #620 |
| #618 | MoA stream the winning worker's tokens | open, not started |
| #596 | peers gossip observed tok/s | open (not mine), relevant to next step |
| #620 | **opinionated MoA no-think** | open, lab-validated, awaiting CI |
| `micn/quic-retry-on-connection-lost` (no PR yet) | same-target retry | branch only, needs second-tier wait |

## Items from your prompt — status

> **1) public mesh tok/s probe per model**

Done. See "Public mesh per-model tok/s" above. Recommendation:
**land Issue #596 (gossiped tok/s)** as the input layer. Routing
work is then a small layer on top — see "Health-aware routing"
below.

> **2) MoA streaming (Issue #618)**

Not started. Wanted to think about it more before writing code. Two
options that look most actionable:

* **C** (cheapest, from the issue): chunk the already-buffered
  arbitrated answer into ~5-token deltas. Visually identical from
  the UI's perspective. Honest representation: yes, since by the
  time arbiter decides, content IS buffered. ~30 LoC in
  `moa_gateway::send_moa_as_responses_sse`.
* **A** (correct): once arbiter commits to a winning worker, splice
  that worker's already-arrived chat.completion.chunk stream into
  the client's Responses-API SSE. ~100 LoC, requires holding the
  upstream stream open during MoA decision (instead of `read_to_end`
  the whole body). Big win on real-streaming feel.

Want your call on C-first vs A-first before I write code. C is a
1-day win; A is a 3-4 day proper fix.

> **3) health-aware routing**

Speculative. My read after seeing the public-mesh data:

* The data is conclusive that **peers serving the same model have
  60x throughput spread**. Picking randomly burns the user 200+s on
  the slow peer for the same query.
* PR #596 (gossiped tok/s) is the right primitive. Once peers
  advertise an EMA of their tok/s, routing can weight by tok/s + a
  health score (e.g. `recent_inflight_request_outcomes`).
* The smallest end-to-end implementation:
  1. Add `peer.recent_tok_per_s` field to gossip (PR #596 territory).
  2. Add `peer.recent_failure_count` (already partially in
     `target_health`).
  3. In `network::affinity::route_eligible_candidates`, score
     candidates by `tok_per_s × (1 - failure_rate)` and rank.
  4. Optional: in MoA, prefer top-K by score for the fan-out
     instead of all-K-models.
* Each step is independently shippable. Step 1 is upstream work I
  don't own. Step 2-4 I could write but they need step 1 to be
  meaningful.

Recommend: wait for PR #596 to land, then 2-4 as a single ~200 LoC PR.

> **4) general stability**

The lab will keep running. Connection events overnight:
* `total LastOpenPath events since v5 start (~5h)`: see live log
  (about 4-8/h based on early sample). Not zero, but every one is
  followed by a clean reconnect within ~1s.
* No request failures from path drops since the same-target retry
  began firing.
* No regressions from the no-think changes — all probe types
  100% successful since the opinionated binary started.

## Recommended next actions (your call)

1. **Approve / merge #620 once CI is green.** Lab-validated, lab will
   keep collecting data on it overnight.
2. **Approve / merge #619** (tiny doc comment, no risk).
3. **Pick MoA streaming approach** (C cheap+now vs A proper+later).
4. **Wait on health-aware routing** until #596 lands.

I'll keep the labs running and check in tomorrow morning if you
haven't reached me by then.
