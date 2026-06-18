# Speculative Decoding Outstanding Work

This note tracks open work for n-gram speculative decoding and SPD sidecar
serving. Broader staged-serving design lives in [`../SKIPPY.md`](../SKIPPY.md),
and benchmark command/report guidance lives in
[`../../crates/skippy-bench/README.md`](../../crates/skippy-bench/README.md).

## Current State

### N-Gram

N-gram speculative decoding is implemented and useful, especially for repeated
coding/editing sessions. It is model-free: the pool observes accepted target
tokens, proposes continuations when a context suffix repeats, and the staged
target verifies every proposed token through `VerifySpan`.

Current policy:

- Use n-gram speculation for coding-shaped sessions and repeated edit loops.
- Do not expect large wins on cold, one-shot, open-ended chat.
- Keep the default n-gram confidence policy flat at 55% until the verifier path
  is redesigned around actual verifier cost.
- Treat n-gram pooling as independent from KV/full-state cache. It remains safe
  for recurrent families such as Qwen3.6 because it does not restore model
  state.

### SPD Sidecar

Status as of 2026-06-18: SPD is a real native request-path proof, but not a
speedup proof yet.

For the current `Qwen/Qwen3-8B` product target, use the exact two-stage
`23,36` topology observed through Mesh and keep the sidecar tied to that split.
The immediate blocker is not ordinary LAN orchestration. A topology-correct
512-row BF16 reference-trained head has weak product Q4 package acceptance,
while a tiny product-tap fine-tune recovers high acceptance only on its training
prompts. The newest product-distribution bridge trains on `48` prompts and
evaluates on `24` held-out prompts across MT-Bench, GSM8K, and HumanEval. It is
the first same-topology product head to clear the all-local held-out
paper-style break-even gate: live-tap validation accepts `110 / 192` proposals
with matched greedy output, and the all-local rolling OpenAI request path
matches baseline/SPD content on all `24 / 24` held-out prompts with `0` tap
failures, `0` ignored taps, and `81 / 160` accepted proposals.

That is correctness and first-quality evidence, not speed evidence. After
refreshing the native Metal stage ABI build so the final-stage tap patch was
actually linked, the real two-node direct-cable gate at
`/tmp/spd-qwen3-8b-product-finetune-paper3-train16-e5-lr2e5/openai-heldout8-rolling-direct-nativefresh.json`
matched baseline/SPD content on all `24 / 24` held-out prompts, returned HF36
taps, accepted `78 / 156` proposals, and recorded `0` tap return failures, `0`
tap record failures, and `0` ignored taps. Its idealized two-stage
`paper_pipeline_estimate` is exactly break-even at `1.0x` (`78` saved / `78`
unsaved candidate token round trips), while measured decode is still only
`0.321x` of baseline because sidecar/native overhead dominates this two-node
topology. Do not spend on or claim a larger generic reference-distribution
sidecar until the train/serve activation-distribution gap and
HF-teacher-versus-native-logit question are addressed. This keeps the work
aligned with the SPD paper: speed requires speculation to be hidden under the
target pipeline step, and theoretical `L'_acc = N/K*n` must remain separate from
measured wall-clock speed. The next gate is a robust same-topology product
sidecar whose held-out package-backed serving clears `paper_pipeline_estimate >
1.0` with margin on the real split.

A later no-spend max120 product-corpus check confirmed that simply fitting the
small HF-teacher bridge harder is not enough. The
`/tmp/spd-qwen3-8b-product-prompts-paper3-train32-heldout16-max120` split
captured `712` train rows and `256` disjoint held-out rows without the earlier
live-tap `n_batch=128` assertion. HF teacher alignment stayed high
(`245 / 256` held-out teacher top-1 matches native Q4 target), but the
5-epoch HF-KL head accepted only `91 / 256` held-out proposals and the native
hard-label variant only `95 / 256`; both were fixture-parity clean and
content-exact. Do not run release/request-path timing for those heads. The next
grounded step is a larger disjoint short-prompt product corpus plus
less-overfit training; if that still fails, expose native Q4_K_M verifier
top-k/logits rather than spending on larger generic HF-teacher KL.

What is working:

- Real `skippy-bench spd-openai-smoke` can launch local binary stages, start the
  embedded stage-0 OpenAI frontend, load a trained Qwen3.5-4B SPD sidecar
  manifest/checkpoint, collect live hidden-state taps, run the Rust sidecar head,
  and verify accepted tokens through the target staged runtime.
- The Rust sidecar path has fixture parity coverage, live-tap parity coverage,
  OpenAI smoke report coverage, warmup/repeat reporting, and phase timing for
  tap collection, `cur_in` assembly, sidecar cache prefill, fixed projections,
  sidecar decoder layers, final norm, and LM-head/top-k.
- The target model remains the source of truth. SPD proposals only commit after
  target verification accepts them.
- The runtime rejects topologies that do not provide the hidden-state tap
  boundaries required by the sidecar manifest, which prevents silently running a
  trained sidecar against an incompatible physical split.
- The opt-in native rolling executor path (`--openai-spd-rolling-executor` /
  `skippy-bench spd-openai-smoke --spd-rolling-executor`) can now keep a
  logical `S=4` rolling queue in flight from live direct-return taps, verify the
  oldest completed entry, and report oldest-commit/drain counters from the
  request path.

Latest native evidence:

| Field | Value |
| --- | --- |
| Report set | `/private/tmp/spd-local-nonrolling-cpu-smoke24-v2.json`, `/private/tmp/spd-local-rolling-cpu-smoke24-v2.json`, `/private/tmp/spd-lan-cpu-spd24-v2.json` |
| Model | Qwen3.5-4B Q4_K_M GGUF |
| Sidecar | pretrained Qwen3.5-4B SPD manifest + serving checkpoint |
| Host/device | local M4 CPU stages plus one-worker LAN CPU split for transport/KV proof |
| Command shape | `spd-openai-smoke --splits 8,10,16,20,24,31 --max-tokens 24 --spd-rolling-executor --n-gpu-layers 0 --spd-n-gpu-layers 0` |
| Logical SPD stages | 4 |
| Physical stages needed by this artifact | 7 (`0..8 | 8..10 | 10..16 | 16..20 | 20..24 | 24..31 | 31..32`) |
| Native KV finding | llama.cpp patch `0094` fixes hybrid checkpoint restore by trimming attention KV only before restoring recurrent checkpoint state |
| Non-rolling control | completed after the patch; accepted 20 / 24 proposals, proving the target-position 38 mismatch is sidecar/top-1 behavior rather than rolling KV corruption |
| Local rolling | exact content, 21 / 22 accepted, max in flight 4, one oldest rejection, three younger drains, 0 tap failures |
| LAN rolling | exact content, 21 / 22 accepted, max in flight 4, one oldest rejection, three younger drains, 0 tap failures |
| Clean LAN rolling case | `/private/tmp/spd-lan-count-paired.json`: exact content, 23 / 23 accepted, max in flight 4, 21 oldest accepts, 0 oldest rejections, 0 younger drains, 0 tap failures |
| LAN reset sweep | `/private/tmp/spd-lan-mini-sweep.json`: three SPD-only prompts, 57 / 59 accepted, one oldest rejection, three younger drains, 0 tap failures, 0 out-of-order replay proposals |
| LAN speed signal | negative correctness result: baseline decode 4502.5 ms, SPD decode 14392.6 ms (`0.313x`) |

First real-node split target:

- Use the pretrained `Qwen/Qwen3.5-4B` S4/L4 SPD sidecar first. It is the only
  current artifact with strong reference acceptance evidence, Rust/Python
  parity, live Skippy tap parity, and a known tap-aligned physical split.
- Target GGUF: `.artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf`
  (`unsloth/Qwen3.5-4B-GGUF:Q4_K_M`).
- Sidecar bundle:
  `/private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/`
  with `skippy-spd-head.json`, `spd-head.safetensors`, and
  `spd-parity-fixture.safetensors`.
- Required physical split for this artifact: `8,10,16,20,24,31`, exposing taps
  `0,8,10,16,20,24,31`. Do not try a clean four-stage split with this sidecar;
  it will miss required hidden-state rows.
- Keep stage 0, the OpenAI frontend, and the SPD sidecar on the coordinator.
  Place downstream physical stages on the worker node or worker devices.

Readiness check on 2026-06-17:

| Check | Result |
| --- | --- |
| Live tap parity report | `/private/tmp/spd-real-node-ready-live-tap.json` |
| Live taps | `0,8,10,16,20,24,31` |
| Live tap verification | 2 / 2 proposals accepted, 0 rejected |
| Live tap output | matched ordinary non-SPD greedy output |
| OpenAI smoke report | `/private/tmp/spd-real-node-ready-openai.json` |
| OpenAI topology | seven local CPU stages, same tap-aligned split |
| OpenAI content match | 1 / 1 baseline/SPD pair matched |
| OpenAI accepted/proposed | 1 / 1 |
| OpenAI tap failures | 0 return, 0 record, 0 ignored |
| OpenAI measured decode | baseline 26.2 ms, SPD 268.1 ms |
| OpenAI downstream wait | 261.7 ms |
| OpenAI sidecar cache/head | 123.2 ms cache prefill, 52.1 ms head total |

Latest multi-token overhead sweeps on 2026-06-17:

| Field | Value |
| --- | --- |
| CPU report | `/private/tmp/spd-local-multitoken-repeat-cpu.json` |
| Metal report | `/private/tmp/spd-local-multitoken-repeat-metal.json` |
| Command shape | `spd-openai-smoke --max-tokens 8 --warmup-count 1 --repeat-count 3 --run-baseline true --run-spd true` |
| Host/device | local M4 node, local binary stage processes; CPU used `CPU0`, Metal used `MTL0` |
| Content match | 3 / 3 baseline/SPD pairs on both CPU and Metal |
| SPD accepted/proposed | 24 / 24 on both CPU and Metal |
| Optimistic commits | 18 total, 12 chained on both CPU and Metal |
| Max optimistic chain depth | 2 |
| Rolling replay | 21 inserted drafts, 15 accepted windows, 0 missing, 0 out-of-order, verified prefix matched target on both runs |
| Tap failures | 0 return, 0 record, 0 ignored on both runs |
| CPU mean decode | baseline 219.3 ms, SPD 13964.2 ms |
| Metal mean decode | baseline 201.0 ms, SPD 1652.6 ms |
| CPU mean waits | normal downstream 2681.2 ms, optimistic hidden wait 2169.6 ms |
| Metal mean waits | normal downstream 262.9 ms, optimistic hidden wait 90.5 ms |
| Sidecar cache/head | CPU 16.8 ms cache prefill / 45.9 ms head total; Metal 18.7 ms cache prefill / 48.9 ms head total |
| Paper estimate from observed trace | 4.0x versus serial split; 54.8 ms CPU-estimated decode, 50.2 ms Metal-estimated decode |

These runs answer the current overhead question: the trained head, cache reuse,
tap rows, and verification mechanics are native enough to preserve output and
reach 100% acceptance on the bounded prompt. The large local slowdown is not a
missing sidecar-cache port from the paper/reference implementation. It is the
gap between the paper/reference rolling pipeline schedule and the current
Skippy serving shape: local stage processes contend for the same machine, and
bounded optimistic verifier work still spends seconds in downstream and hidden
waits on CPU, or hundreds of milliseconds on Metal, instead of keeping a full
rolling queue continuously useful. Metal narrows the gap by about 8.4x versus
CPU SPD decode, but the run still reaches only `0.122x` SPD-vs-baseline decode
speed and remains about `32.9x` slower than the paper-shaped estimate.

First remote preflight on 2026-06-17:

| Check | Result |
| --- | --- |
| Report | `/private/tmp/spd-qwen35-first-remote-preflight.json` |
| Mode | `spd-openai-preflight`; no stages launched |
| Artifact checks | `skippy-server` stat succeeded; GGUF stat succeeded; serving checkpoint has 66 tensors; parity fixture has 28 tensors |
| Logical/physical topology | logical SPD stages 4; physical stages 7 |
| Split/tap check | `8,10,16,20,24,31` exposes tap returns `8,10,16,20,24,31` |
| Remote plan | stage 0 local with checked port 20031, stages 1-6 assigned to one worker endpoint plan |
| Warnings | none for the complete fake endpoint/model-path map |

Native rolling-executor local smoke on 2026-06-17:

| Check | Result |
| --- | --- |
| Preflight report | `/private/tmp/spd-rolling-executor-local-preflight.json` |
| Paired smoke report | `/private/tmp/spd-rolling-executor-local-paired-final.json` |
| Build shape | current debug `skippy-server` / `skippy-bench`, not release timing binaries |
| Command shape | `spd-openai-smoke --splits 8,10,16,20,24,31 --max-tokens 6 --run-baseline true --run-spd true --optimistic-decode true --spd-rolling-executor` |
| Content match | 1 / 1 baseline/SPD pair matched |
| Rolling executor launches | 5 |
| Rolling executor max in flight | 4 |
| Rolling executor oldest commits | 3 accepted, 0 rejected |
| Younger drains | 0 |
| Optimistic commits | 5 total, 4 chained |
| Tap failures | 0 return, 0 record, 0 ignored |
| Current speed signal | negative local/debug result; baseline decode 170.5 ms, SPD decode 25149.1 ms |

This closes the earlier "diagnostic-only rolling scheduler" gap for the
request path: the server can launch younger verifier work from every eligible
tap callback and maintain a filled logical rolling queue. It is still a local
same-machine proof. It does not show the paper's distributed overlap regime and
should not be used as speedup evidence.

Latest KV/rolling diagnostic on 2026-06-17:

| Check | Result |
| --- | --- |
| Reports | `/private/tmp/spd-local-nonrolling-cpu-smoke24-v2.json`, `/private/tmp/spd-local-rolling-cpu-smoke24-v2.json` |
| Content match | 1 / 1 baseline/SPD pair matched |
| SPD proposals | 21 accepted / 22 proposed |
| Rolling executor | 22 launches, max in flight 4, 17 oldest accepts, 1 oldest rejection, 3 younger replies drained |
| Launch-miss breakdown | 72 no proposal, 5 missing shadow view, 17 in-flight full, 2 shadow not seedable, 0 no rows |
| Rolling replay | 20 inserted drafts, 3 missing proposals, 0 out-of-order proposals, verified 24-token prefix matched target |
| Native KV finding | patch `0094` prevents checkpoint restore from trimming recurrent/hybrid memory before restoring the saved checkpoint lane; explicit trim still uses the recurrent owner |
| Sidecar-quality control | non-rolling CPU verification accepted 20 / 24 and hit the same target-position 38 mismatch as rolling, so the remaining rejection is not a rolling/LAN KV artifact |
| Speed signal | still negative locally: baseline decode 620.5 ms, SPD decode 8839.2 ms (`0.070x`); this is correctness/scheduler evidence, not a speedup claim |

First real LAN split checkpoint on 2026-06-17:

| Check | Result |
| --- | --- |
| Baseline-only report | `/private/tmp/spd-lan-cpu-baseline1.json` |
| First paired SPD report | `/private/tmp/spd-lan-cpu-spd8.json` |
| Latest 24-token paired report | `/private/tmp/spd-lan-cpu-spd24-v2.json` |
| Placement | stage 0, OpenAI frontend, and SPD sidecar on the coordinator; physical stages 1-6 on one LAN worker |
| Runtime device choice | `--n-gpu-layers 0 --spd-n-gpu-layers 0` for the transport proof |
| Content match | 1 / 1 baseline/SPD pair matched |
| SPD proposals | first run: 7 / 7 accepted; latest 24-token run: 21 / 22 accepted with one oldest rejection |
| Rolling executor | latest run: 22 launches, max in flight 4, 17 oldest accepts, 1 oldest rejection, 3 younger drains |
| Rolling replay | latest run: 20 inserted drafts, 3 missing proposals, 0 out-of-order proposals, verified 24-token prefix matched target |
| Stage logs | no post-ready KV, decode, tap-return, or Metal OOM errors; only transient startup readiness retries on the latest run |
| Checker | the 8-token checkpoint passed; the 24-token run fails the strict paper gate for `21 < 24` accepted proposals, one oldest rejection, three drained younger replies, and three missing replay proposals; the relaxed correctness gate passed at `/private/tmp/spd-lan-cpu-spd24-v2-check-relaxed.json` |
| Speed signal | negative by design: latest CPU baseline decode 4502.5 ms, SPD decode 14392.6 ms (`0.313x`) |

Follow-up LAN split evidence on 2026-06-18:

| Check | Result |
| --- | --- |
| Clean paired report | `/private/tmp/spd-lan-count-paired.json` |
| Clean paired checker | `/private/tmp/spd-lan-count-paired-check.json` |
| Placement | same one-worker LAN CPU split: stage 0, OpenAI frontend, and SPD sidecar on the coordinator; physical stages 1-6 on one worker |
| Content match | 1 / 1 baseline/SPD pair matched |
| SPD proposals | 23 / 23 accepted, 0 rejected |
| Rolling executor | 23 launches, max in flight 4, 21 oldest accepts, 0 oldest rejections, 0 younger drains |
| Rolling replay | one terminal missing proposal at the max-token boundary, 0 out-of-order proposals, verified 24-token prefix matched target |
| Tap/KV evidence | 0 tap return failures, 0 tap record failures, 0 ignored taps |
| Speed signal | still negative: baseline decode 4426.1 ms, SPD decode 13458.4 ms (`0.329x`) |
| SPD-only reset sweep | `/private/tmp/spd-lan-mini-sweep.json` plus `/private/tmp/spd-lan-mini-sweep-check.json` |
| Sweep result | three prompts, 57 / 59 accepted, two rejected proposals, one oldest rejection, three younger drains, max in flight 4, 0 tap failures, 0 out-of-order replay proposals |

The clean paired case proves the happy-path shadow-KV promotion behavior over
real stage transport. The SPD-only sweep proves rejection/drain recovery over
the same split path. The checker for SPD-only reports must pass
`--require-content-match false` because no baseline half exists in that report.

Product-shaped two-stage baseline on 2026-06-18:

| Check | Result |
| --- | --- |
| Report | `/private/tmp/skippy-two-stage-baseline.json` |
| Preflight | `/private/tmp/skippy-two-stage-baseline-preflight.json` |
| Topology | two physical stages, `--splits 16 --layer-end 32`, ranges `0..16` and `16..32` |
| Placement | stage 0 and OpenAI frontend on the coordinator; stage 1 on one worker |
| Runtime device choice | `--n-gpu-layers=-1`; both inspected stage logs selected Metal |
| Baseline output | 24-token bounded count prompt, finish reason `length` |
| Baseline timing | wall `1678.9ms`, decode `1293.2ms`, stage-0 compute `253.0ms`, downstream wait `990.2ms` |
| Tap/KV evidence | no tap return failures, no tap record failures, no ignored taps |
| Cleanup | local and worker stage ports were free after the run; no worker `skippy-server` process remained |

This is the first ordinary two-stage Skippy split proof for the product shape
the speed gate should use. It is baseline-only because the current pretrained
Qwen3.5-4B S4/L4 sidecar does not match this topology. That artifact requires
the tap-aligned split `8,10,16,20,24,31`; a true two-stage split exposes only
hidden-state boundaries `16` and `32` plus the embedding row `0`. The matching
sidecar must be trained for `num_stages=2` and `stage_layer_boundaries=16,32`,
which derives tap rows `0,16,32;0,16`.

The first attempted LAN smoke with all remote stages using Metal
(`--n-gpu-layers -1`) failed before a full token because one worker was running
six Metal-backed stage processes. Stage 3 hit Metal out-of-memory on the first
`DecodeEmbd`. That failure is a placement/resource issue, not a Skippy KV or
binary-transport correctness failure. For one-worker correctness gates, use CPU
stages. For speed gates, spread stages across distinct devices/workers or reduce
the number of Metal-backed processes per worker.

Paper fidelity:

- The mechanism is paper-shaped: hidden states from target stages are converted
  into sidecar rows, the sidecar proposes a draft token, and the target verifies
  before commit.
- The sidecar is topology-bound in practice. A trained artifact can require
  hidden-state taps that do not line up with a simple `N` physical-stage split.
  The current Qwen3.5-4B proof required all tap boundaries
  `8,10,16,20,24,31`, even though the sidecar's logical topology has four SPD
  stages.
- The performance claim is not proven. The current proof is CPU-heavy and either
  local or one-worker LAN placement; it does not yet reproduce the paper's useful
  overlap regime where target pipeline work and sidecar work hide each other on
  genuinely parallel hardware.
- The overhead delta is now a distributed-execution and scheduling-quality gap,
  not evidence that the paper or reference sidecar mechanics are absent. The
  reference loop keeps an `n`-slot rolling pipeline, runs target stage work and
  speculation in parallel, reuses `spec_past_kv`, and verifies/evicts the
  oldest completed entry. Skippy now has the Rust scheduler primitive, inline
  taps, cache parity, bounded chained verification evidence, and an opt-in
  request-path rolling executor, but still needs a real split run that proves
  useful overlap on distinct hardware.

## Outstanding Work

### SPD Speedup Validation

The next SPD milestone is not more unit coverage; it is an end-to-end speedup
run with enough instrumentation to explain the result.

Open items:

- Run the multi-token baseline-vs-SPD sweep on distinct devices/nodes; both CPU
  and Metal local repeats passed correctness, but same-machine repeats are not
  speed oracles.
- For the two-node product shape, train a Qwen3.5-4B sidecar for
  `num_stages=2` and `stage_layer_boundaries=16,32` before attempting
  baseline-vs-SPD. The pretrained S4/L4 sidecar is not a valid artifact for the
  two-stage split.
- Use a topology-compatible artifact and record both logical SPD stage count and
  physical tap-aligned stage count.
- Use distinct devices or nodes so target stage work and sidecar work can
  overlap instead of competing for the same local CPU/memory bandwidth.
- Keep reporting downstream wait, sidecar cache prefill, sidecar head total,
  decoder-layer timing, accept rate, rolling gaps, and content equality.
- Treat any speedup claim as invalid unless the report includes the command,
  commit SHA, model identity, sidecar artifact identity, topology, hardware, and
  raw JSON report path.

### SPD Runtime Cost Reduction

The current local native proof is dominated by costs that are not hidden in the
one-token local run.

Open items:

- Keep the stateful sidecar cache path, but do not treat cache prefill as the
  primary blocker: the 8-token CPU repeat had 24 proposal cache hits, no cache
  misses, and only 16.8 ms mean prefill after the first warm row.
- Reduce or hide downstream wait and verifier hidden wait; the 8-token CPU
  repeat spent 2681.2 ms mean normal downstream wait and 2169.6 ms mean
  optimistic hidden wait.
- Keep tightening the opt-in rolling executor until it is safe as the normal
  SPD serving path: keep multiple speculative entries useful across longer
  prompts, route returned taps by scheduler position, verify the oldest
  completed entry, and reset only on the oldest rejection.
- Add a server-side reuse path for warmup/measured requests only after request
  attribution is robust; the current benchmark intentionally isolates stage
  processes per iteration so logs are unambiguous.
- Investigate whether the required tap-boundary topology should be materialized
  as extra lightweight tap stages, fused into neighboring stages, or retrained
  for cleaner physical stage splits.

### Immediate SPD Next Runs

Run these before making any speedup claim:

1. Local all-tap baseline comparison:

   CPU status: completed on 2026-06-17 at
   `/private/tmp/spd-local-multitoken-repeat-cpu.json`. Metal status:
   completed on 2026-06-17 at
   `/private/tmp/spd-local-multitoken-repeat-metal.json`. Both preserved exact
   output and accepted `24 / 24` proposals, but SPD was still slower than
   baseline. Metal reduced SPD decode from `13964.2ms` to `1652.6ms`, which
   confirms CPU contention was real while leaving a native scheduler gap.

   ```bash
   target/release/skippy-bench spd-openai-smoke \
     --stage-server-bin target/release/skippy-server \
     --manifest <spd-head.json> \
     --fixture <spd-parity-fixture.safetensors> \
     --model-path <target.gguf> \
     --model-id local/spd-qwen35-4b \
     --splits 8,10,16,20,24,31 \
     --layer-end 32 \
     --ctx-size 128 \
     --n-gpu-layers=-1 \
     --selected-backend-device MTL0 \
     --max-tokens 8 \
     --warmup-count 1 \
     --repeat-count 3 \
     --optimistic-decode true \
     --spd-rolling-executor \
     --output /tmp/spd-openai-smoke-local-mtl-repeat.json
   ```

   This is still a contention-heavy local run, but it gives measured
   baseline/SPD pairing, repeated samples, and multi-token accept/rolling data.

2. Distinct-device or multi-node all-tap run:

   Status: one-worker CPU-backed LAN transport proofs completed on 2026-06-17
   and 2026-06-18. The latest clean paired report
   `/private/tmp/spd-lan-count-paired.json` matched baseline, accepted
   `23 / 23` SPD proposals, reached `max_in_flight=4`, had no oldest rejection
   or younger drain, and kept tap failures at zero. The follow-up SPD-only sweep
   `/private/tmp/spd-lan-mini-sweep.json` accepted `57 / 59` proposals and
   exercised one oldest rejection plus three younger drains with zero tap
   failures. This is correctness and placement evidence only; it is not a
   speedup claim.

   - keep stage 0 and the SPD sidecar on the coordinator;
   - place physical stages on distinct devices/nodes where available;
   - keep `--splits 8,10,16,20,24,31` for the current Qwen3.5-4B sidecar
     artifact unless a cleaner topology-specific sidecar is trained;
   - compare baseline/SPD decode time, downstream wait, sidecar cache prefill,
     sidecar head total, accept rate, rolling gaps, and content equality.
   - do not launch all six downstream physical stages on one worker with
     `--n-gpu-layers -1`; that oversubscribed Metal and failed with
     out-of-memory on the first decode step. Use CPU for one-worker correctness,
     or use multiple devices/workers for a real speed gate.

   First remote command shape when one worker is available and already has the
   same GGUF path. Run it once with `--preflight-only` first; that report should
   validate artifacts, splits, tap coverage, endpoint mapping, and remote model
   paths before any SSH process launch.

   ```bash
   target/release/skippy-bench spd-openai-smoke \
     --stage-server-bin target/release/skippy-server \
     --manifest /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/skippy-spd-head.json \
     --fixture /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-parity-fixture.safetensors \
     --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
     --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
     --splits 8,10,16,20,24,31 \
     --layer-end 32 \
     --ctx-size 128 \
     --n-gpu-layers=0 \
     --spd-n-gpu-layers=0 \
     --stage-hosts local,<worker>,<worker>,<worker>,<worker>,<worker>,<worker> \
     --endpoint-host-map local=<coordinator-lan-ip-or-name>,<worker>=<worker-lan-ip-or-name> \
     --remote-model-path-map <worker>=/path/on/worker/Qwen3.5-4B-Q4_K_M.gguf \
     --max-tokens 1 \
     --repeat-count 1 \
     --preflight-only \
     --output /tmp/spd-qwen35-first-remote-preflight.json
   ```

   Then remove `--preflight-only` and write the real smoke report:

   ```bash
   target/release/skippy-bench spd-openai-smoke \
     --stage-server-bin target/release/skippy-server \
     --manifest /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/skippy-spd-head.json \
     --fixture /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-parity-fixture.safetensors \
     --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
     --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
     --splits 8,10,16,20,24,31 \
     --layer-end 32 \
     --ctx-size 128 \
     --n-gpu-layers=0 \
     --spd-n-gpu-layers=0 \
     --stage-hosts local,<worker>,<worker>,<worker>,<worker>,<worker>,<worker> \
     --endpoint-host-map local=<coordinator-lan-ip-or-name>,<worker>=<worker-lan-ip-or-name> \
     --remote-model-path-map <worker>=/path/on/worker/Qwen3.5-4B-Q4_K_M.gguf \
     --max-tokens 8 \
     --repeat-count 1 \
     --optimistic-decode true \
     --spd-rolling-executor \
     --output /tmp/spd-qwen35-first-remote-openai.json
   ```

   Use `--rsync-model-artifacts` only if copying the 2.6 GB GGUF for the run is
   acceptable; otherwise stage the GGUF once on the worker and use
   `--remote-model-path-map`.

3. Sidecar/topology training check:

   - Use `evals/spd/hf_train_eval_qwen06.py` for the first topology-specific
     proof. It already clones/patches the reference repo, trains/evaluates,
     exports the Skippy manifest, and supports `--stage-layer-boundaries` plus
     explicit `--shallow-hidden-layer-indices`.
   - For the two-node product split, train a Qwen3.5-4B S2/L4 sidecar before
     running SPD. The command shape is:

     ```bash
     python3 evals/spd/hf_train_eval_qwen06.py \
       --work-dir /tmp/skippy-spd-qwen35-4b-s2-16 \
       --model-name Qwen/Qwen3.5-4B \
       --dataset HuggingFaceH4/ultrachat_200k \
       --dataset-split train_sft \
       --train-rows 8192 \
       --eval-rows-per-set 32 \
       --num-stages 2 \
       --stage-layer-boundaries 16,32 \
       --num-spec-layers 4 \
       --max-length 512 \
       --max-new-tokens 64 \
       --draft-top-k 4 \
       --device mps \
       --upload-repo ''
     ```

     After training, export `spd-head.safetensors`, export a parity fixture,
     run fixture parity, run live-tap parity with `--splits 16 --layer-end 32`,
     and only then run the paired two-node
     `spd-openai-smoke --run-baseline true --run-spd true
     --spd-rolling-executor` gate.
   - Do not blindly reuse the pretrained Qwen3.5 S4/L4 tap layout for a cleaner
     physical split. For any intended Skippy split, write the exact tap rows
     first, then train/evaluate the sidecar against those rows.
   - For the current larger dense proof, use `Qwen/Qwen3-8B` with the
     product-shaped two-stage topology already observed through Mesh:
     `num_stages=2`, `stage_layer_boundaries=23,36`, `num_spec_layers=4`,
     greedy/no-thinking eval, and a `32k` draft vocab cap. The manifest/export
     path must carry the Qwen3 rotary metadata (`rope_theta=1000000`,
     `rotary_dim=128`) so Rust serving uses the same head math as the
     reference. Do not reuse the pretrained Qwen3.5-4B `8,10,16,20,24,31`
     physical split as a training topology.
   - The first 512-row local BF16 Qwen3-8B S2 `23,36` head is a plumbing and
     parity checkpoint, not a speed candidate. After fixing the reference
     single-chain evaluator for custom boundaries, serving-equivalent
     `draft_top_k=1` eval accepted only `12 / 384` draft flags
     (`3.23%` theoretical gain), and package-backed serving accepted `0 / 90`
     proposals across the six-prompt smoke. Scale or change the training recipe
     before spending a two-node speed run on this topology.
   - A repeat 512-row run with the paper/reference LR `1e-4` improved
     reference top-1 acceptance to `41 / 384` flags (`7.50%` theoretical gain)
     and produced the first nonzero Qwen3-8B package-backed serving acceptance
     on matched GSM8K prompts: `2 / 120` proposals accepted with clean taps.
     It still accepted `0 / 90` on the original code/math/writing sweep, so it
     is evidence that the recipe direction is better, not a speed candidate.
   - The first broader product-row bridge captured `192` train rows and `96`
     held-out rows from the exact Q4 layer package, attached HF teacher logits,
     fine-tuned from the LR `1e-4` checkpoint, and exported a BF16 serving
     bundle that passes Rust fixture parity. Held-out live-tap acceptance is
     `39 / 96`; held-out all-local OpenAI rolling acceptance is `30 / 82` with
     exact content and `0` tap failures. This is a real generalization signal,
     but it is still below two-stage break-even: `paper_pipeline_estimate` is
     `0.73x`. The next gate is more same-topology product data and/or native
     Q4_K_M verifier logits, not a speed run.
   - Run a local or dry-run HF job on `Qwen/Qwen3-0.6B` only when debugging the
     trainer/export path itself. It is no longer the next scaling target now
     that the Qwen3.5-4B request path has real LAN KV evidence.
   - Record the sidecar manifest's required hidden-state indices alongside every
     benchmark topology. Prefer a sidecar whose required taps match the intended
     mesh stage layout; otherwise the runtime must either create extra tap
     stages or fuse tap collection into neighboring stages.

### Batched Target Verification

Verification is still the governor. Warm n-gram runs show useful acceptance, but
the live staged path still spends too much wall time in target verification,
stage forwarding, and repair bookkeeping.

Open items:

- Investigate true batched target verification for multi-token n-gram spans.
- Reduce per-window protocol round trips and per-stage bookkeeping overhead.
- Compare block verification against tree-style verification before adding a
  larger public protocol surface.
- Keep measuring `verify_wall_ms`, verifier compute, downstream wait, protocol
  request count, protocol token count, max span, and average span.

### Rejection Repair

Early rejection still hurts n-gram more than proposal quality alone suggests.
The first-token early-reject fast path exists, but wider windows still pay too
much restore/reverify overhead.

Open items:

- Make repair decisions cost-aware, not only confidence/window-size aware.
- Preserve the tail-reject fast path.
- Avoid repair `VerifySpan` when a normal decode step is cheaper.
- Track repair cost by task type, not only globally.

### Pool Policy And Lifetime

N-gram pools are valuable while the user is iterating in the same context. They
are less valuable after a project/session has gone cold.

Open items:

- Add explicit pool TTL and LRU eviction policy.
- Keep pools in memory by default; avoid disk persistence until there is a clear
  reproducibility or resume requirement.
- Consider separate retention classes for session pools, project pools, and
  tenant-wide warm pools.
- Expose pool memory usage and candidate counts in telemetry.

### Concurrent Sessions

The server path needs to be boringly reliable under many prompt workers.

Open items:

- Stress-test `ngram-pool-server` with many concurrent session IDs.
- Shard or partition pool locks if contention appears.
- Verify pool keys include model, tokenizer, tenant, project, session, explicit
  pool ID, and n-gram size.
- Ensure failed or cancelled requests only observe accepted target tokens.

### Routing Policy

The OpenAI-compatible frontend should eventually route coding-shaped requests to
n-gram speculation before draft speculation when the session/project pool is
warm enough.

Open items:

- Add a conservative coding-prompt detector for file paths, fenced code, diffs,
  compiler errors, stack traces, tests, symbols, and tool logs.
- Use acceptance and verifier-cost telemetry to disable or shrink n-gram windows
  when a session is not benefiting.
- Keep routing as a performance policy only; correctness must always come from
  target verification.

### Benchmark Coverage

The current numbers are useful but not enough to lock policy.

Open items:

- Continue using HF-sourced benchmark corpora instead of checked-in large
  prompt bodies.
- Keep smoke and long tiers for all benchmark modes, not only speculation.
- Run warm coding-loop confirmation regularly because that is the expected
  n-gram win case.
- Report by task type, especially coding versus chat/instruction.
- Preserve raw logs under `target/prompt-spec-corpus/<timestamp>` for audit.

## Done Criteria For Promotion

N-gram should become an automatic first-choice coding strategy only after:

- warm coding-loop runs show consistent speedup over baseline;
- verifier wall time decreases, not just acceptance rate increasing;
- concurrent session stress runs do not show lock contention or pool bleed;
- telemetry can explain why a session enabled, shrank, or disabled n-gram;
- regression runs include at least smoke and long corpus tiers.
