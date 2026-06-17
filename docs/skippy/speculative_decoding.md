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

Status as of 2026-06-17: SPD is a real native request-path proof, but not a
speedup proof yet.

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

Latest native evidence:

| Field | Value |
| --- | --- |
| Commit | `c0298e54` |
| Report | `/private/tmp/spd-openai-smoke-repeat-telemetry-cpu-1tok.json` |
| Model | Qwen3.5-4B Q4_K_M GGUF |
| Sidecar | pretrained Qwen3.5-4B SPD manifest + serving checkpoint |
| Host/device | local M4 node, `CPU0`, local binary stage processes |
| Command shape | `spd-openai-smoke --splits 8,10,16,20,24,31 --max-tokens 1 --warmup-count 1 --repeat-count 1 --run-baseline false` |
| Logical SPD stages | 4 |
| Physical stages needed by this artifact | 7 (`0..8 | 8..10 | 10..16 | 16..20 | 20..24 | 24..31 | 31..32`) |
| Measured accepted/proposed | 1 / 1 |
| Measured wall/decode | 914.2 ms / 276.8 ms |
| Measured downstream wait | 269.9 ms |
| Measured sidecar cache prefill | 119.8 ms |
| Measured sidecar head total | 47.6 ms |
| Measured sidecar decoder layers | 34.1 ms |
| Tap failures | 0 |

Paper fidelity:

- The mechanism is paper-shaped: hidden states from target stages are converted
  into sidecar rows, the sidecar proposes a draft token, and the target verifies
  before commit.
- The sidecar is topology-bound in practice. A trained artifact can require
  hidden-state taps that do not line up with a simple `N` physical-stage split.
  The current Qwen3.5-4B proof required all tap boundaries
  `8,10,16,20,24,31`, even though the sidecar's logical topology has four SPD
  stages.
- The performance claim is not proven. The local proof is single-machine,
  process-heavy, mostly CPU-bound, and one-token. It does not yet reproduce the
  paper's useful overlap regime where target pipeline work and sidecar work hide
  each other on genuinely parallel hardware.

## Outstanding Work

### SPD Speedup Validation

The next SPD milestone is not more unit coverage; it is an end-to-end speedup
run with enough instrumentation to explain the result.

Open items:

- Run baseline-vs-SPD with `--repeat-count` over multi-token prompts, not only a
  one-token smoke.
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

- Reduce or hide sidecar cache prefill; the measured CPU proof spent about
  120 ms there for a one-token proposal.
- Reduce downstream wait and stage handoff overhead; the measured CPU proof
  spent about 270 ms waiting downstream.
- Add a server-side reuse path for warmup/measured requests only after request
  attribution is robust; the current benchmark intentionally isolates stage
  processes per iteration so logs are unambiguous.
- Investigate whether the required tap-boundary topology should be materialized
  as extra lightweight tap stages, fused into neighboring stages, or retrained
  for cleaner physical stage splits.

### Immediate SPD Next Runs

Run these before making any speedup claim:

1. Local all-tap baseline comparison:

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
     --n-gpu-layers -1 \
     --selected-backend-device MTL0 \
     --max-tokens 8 \
     --warmup-count 1 \
     --repeat-count 3 \
     --output /tmp/spd-openai-smoke-local-mtl-repeat.json
   ```

   This is still a contention-heavy local run, but it gives measured
   baseline/SPD pairing, repeated samples, and multi-token accept/rolling data.

2. Distinct-device or multi-node all-tap run:

   - keep stage 0 and the SPD sidecar on the coordinator;
   - place physical stages on distinct devices/nodes where available;
   - keep `--splits 8,10,16,20,24,31` for the current Qwen3.5-4B sidecar
     artifact unless a cleaner topology-specific sidecar is trained;
   - compare baseline/SPD decode time, downstream wait, sidecar cache prefill,
     sidecar head total, accept rate, rolling gaps, and content equality.

3. Sidecar/topology training check:

   - inspect whether the reference training/export path can train heads for
     cleaner physical split plans, not only the current all-tap artifact;
   - record the sidecar manifest's required hidden-state indices alongside every
     benchmark topology;
   - prefer a sidecar whose required taps match the intended mesh stage layout,
     otherwise the runtime must either create extra tap stages or fuse tap
     collection into neighboring stages.

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
