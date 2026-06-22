---
name: skippy-shard-tree-validation
description: Use when validating Shard-style tree speculative decoding in mesh-llm/Skippy, including target/draft GGUF compatibility, greedy-equivalence checks, split-stage mesh serving, Hugging Face worker experiments, or benchmarks comparing sync draft, pipelined linear speculation, and tree speculation.
---

# Skippy Shard Tree Validation

Use this for evidence-gathering, not implementation. Pair it with
`skippy-spec-bench` for target/draft compatibility, `hf-layer-package-jobs` for
HF-hosted layer package prep, and `hf-mesh-jobs` for Hugging Face worker mesh
proofs. Do not use lab/studio hosts for this workflow unless the user explicitly
restores that option.

## Ground Rules

- Benchmark tree speculation with deterministic target sampling first:
  `temperature = 0`. Do not claim stochastic distribution-preserving correctness.
- Do not use a release build, HF proof, or WAN proof as the first validation
  step for new Shard scheduler or transport work. First run focused debug/unit
  coverage for the changed primitive, then run the local proof, and only then
  spend on release/HF evidence. A release rebuild before the missing primitive
  has test coverage is process failure, not proof.
- Use one fixed prompt set, target model, draft model, context size, and split
  topology across all modes.
- Keep the draft model standard: a compatible GGUF with the same tokenizer. No
  special draft-serving protocol is required.
- Treat greedy equivalence as the correctness gate: non-spec greedy output and
  tree-spec greedy output must token-match before performance numbers matter.
- Do not hard-code hostnames, IPs, private tokens, mesh join tokens, or Hugging
  Face secrets in committed scripts or docs.
- Re-check the Shard reference source before changing the scheduler. The
  authoritative files are `../shard/phase0/specpipe.py`,
  `../shard/phase0/tree.py`, `../shard/phase0/fastverify.py`, and the GLM WAN
  paths in `../shard/research/glm_swarm_nvfp4_pipe.py` and
  `../shard/research/glm_swarm_nvfp4_cg.py`. For measured WAN context, also
  read `../shard/docs/research/wan-speculative-decoding.md` and the receipts
  under `../shard/docs/receipts/`.

## Shard Reference Invariants

Do not reduce Shard to generic speculative decoding. The WAN mechanism is:

- Direct return: the tail sends verify results straight to the coordinator
  instead of relaying back through every stage.
- Fixed verify chunk: send `[current_token] + K draft tokens`, so the target
  verifies K futures in one stage-chain traversal.
- Pipelined coordinator: keep `depth` fixed-size verify chunks in flight and run
  the next draft request while target verify chunks cross the WAN.
- Full accept commits only the K draft tokens. The target bonus token is folded
  into the next overlapping chunk so positions stay aligned by K.
- Reject commits accepted prefix plus the target correction, discards remaining
  in-flight verify results as stale, drops stale draft output, and re-primes from
  the corrected prefix.
- Tree mode verifies a flattened candidate tree with ancestor attention,
  accepts the target-greedy path, and lazily gathers the accepted-path KV on the
  next verify.

The GLM reference uses a normal tokenizer-compatible draft model
(`GLM-4-9B` in the Shard scripts) plus target verification. The performance
claim depends on direct return, pipelined depth, KV-cached stages, and a draft
acceptance rate high enough that stale work does not dominate.

Do not conflate the GLM headline with tree speculation. The GLM pipe path is
linear `[current] + K draft` verification with async in-flight windows. Tree
mode is a separate multiplier: it raises tokens per traversal, but Shard's own
gpt-oss notes show it can regress wall-clock while target verify remains
compute-bound. Prove the linear latency-hiding scheduler first, then fast
draft/verify, then tree under a latency-bound verify regime.

For larger quantized/CUDA runs, be precise about equivalence. Shard's receipts
prove within-engine losslessness; cross-regime comparisons can diverge at
floating-point near-ties when K, batching, or kernels differ. Full-target
reference matching is a hard gate for small deterministic proof prompts, but the
GLM/gpt-oss class proof should prefer same-engine/same-regime token hashes and
should not treat every near-tie divergence as a scheduler bug.

## Validation Matrix

Run the same workload in these modes:

```toml
[defaults.speculative]
mode = "draft"
draft_model_path = "/path/to/draft.gguf"
draft_selection_policy = "manual"
draft_max_tokens = 8
pipelined_depth = 1
```

```toml
[defaults.speculative]
mode = "shard-pipeline"
draft_model_path = "/path/to/draft.gguf"
draft_selection_policy = "manual"
draft_max_tokens = 8
pipelined_depth = 3
```

```toml
[defaults.speculative]
mode = "tree"
draft_model_path = "/path/to/draft.gguf"
draft_selection_policy = "manual"
draft_max_tokens = 8
```

Compare:

- Baseline greedy target-only output.
- Sync linear speculative output.
- Pipelined linear speculative output.
- Tree speculative output.

`mode = "shard-pipeline"` is an operator-facing alias for the linear draft
verifier with Shard-style pipelined defaults. The resolver should canonicalize
it back to the internal linear `draft` execution path after applying the depth
defaults and guards; downstream runtime code should not grow a separate
`shard-pipeline` branch.

## Local Correctness

1. Build with the native ABI the serving path will use:

```bash
scripts/prepare-llama.sh pinned
LLAMA_STAGE_BACKEND=metal scripts/build-llama.sh
just build
```

Use the appropriate `LLAMA_STAGE_BACKEND` for the host.

2. Verify target/draft compatibility:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
cargo test -p skippy-server --lib
cargo test -p mesh-llm-host-runtime --lib inference::skippy::resolver
```

3. Run greedy requests with `temperature: 0` and compare emitted token IDs or
detokenized text between target-only and tree mode. A mismatch is a correctness
bug, not a benchmark result.

For a cheap same-engine q_len check before an endpoint/HF run, use
`skippy-bench verify-span-local` with the target GGUF, `--split-layer`,
`--post-prefill-seed true`, and `--chat-template true`. The report should show
`full_batched_serial_primary_match`, `split_batched_matches_full_batched`, and
`split_serial_matches_full_serial` as `true`. This validates the batched
VerifySpan primitive under the proof runner's chat-template shape, but it does
not replace a full OpenAI endpoint versus split-serving parity pass.

## Distributed Run Shape

For a two-node or HF-worker test:

- Package or reuse the target model as Skippy layer packages.
- Run stage 0 with the draft GGUF available locally.
- Split target stages across workers using the same stage topology as the
  baseline split run.
- Ensure direct prediction return is enabled when measuring latency benefit.
- For intentional WAN validation, set
  `MESH_LLM_ALLOW_SLOW_DIRECT_STAGE_PATHS=1` on every participating node so
  measured high-RTT direct paths are admitted and priced by the topology planner
  instead of rejected by the default production guard.
- When proving latency amortization, use
  `MESH_LLM_SPLIT_FORCE_BOUNDARIES=<comma-separated layer boundaries>` on the
  coordinator if the production planner collapses a two-node proof into a
  near-solo placement such as `0..35` + `35..36`. Keep the same forced topology
  across target-only, sync draft, pipelined draft, and tree runs.
- Confirm `/v1/models` and console status show all split stages before starting
  benchmarks.

When using Hugging Face workers, use `hf-mesh-jobs`. The target can be split
into layer package repos; the draft model remains a normal GGUF on the
coordinator/stage-0 worker unless intentionally testing remote draft placement.

For the repeatable one-local-coordinator plus one-HF-worker WAN proof, use:

```bash
scripts/skippy-shard-hf-wan-proof.sh <target-model-ref> <draft-gguf>
```

The script requires `MESH_SHARD_HF_ARTIFACT_REPO`,
`MESH_SHARD_HF_ARTIFACT_PATH`, and `MESH_SHARD_HF_ARTIFACT_SHA` for the Linux
worker tarball. It writes evidence under `/tmp`, records owned HF job IDs in a
scratch ledger, cancels only jobs it launched, and fails when Shard-specific
proof gates are missing: direct stage path, direct prediction return,
speculation engagement, `[current] + draft` verify-window shape, greedy output
match, draft rollback after rejection, stale-window KV recovery when stale
work is observed, FIFO return accounting, committed-plus-stale window accounting,
and pipelined engagement for pipelined mode.

When a full non-split target is available, set
`MESH_SHARD_HF_REFERENCE_BASE_URL` plus `MESH_SHARD_HF_REFERENCE_MODEL`, or pass
`MESH_SHARD_HF_REFERENCE_RESULTS_JSON`. The HF proof runner then treats matching
that canonical reference as a hard gate for target, sync draft, and pipelined
draft outputs.

If no reference endpoint is already running, create a reusable reference JSON
with:

```bash
MESH_SHARD_REFERENCE_TARGET_ID=sha256:<canonical-target-source-sha> \
scripts/skippy-shard-reference-capture.sh \
  ./target/release/mesh-llm \
  /path/to/full-target.gguf \
  /tmp/full-target-reference.json
```

Use the same canonical target id when proving with a layer-package target that
was derived from that full GGUF:
`MESH_SHARD_HF_REFERENCE_TARGET_ID` for the HF runner or
`MESH_SHARD_PROOF_REFERENCE_TARGET_ID` for the persistent mesh runner. In strict
reference mode, the runner rejects a reused reference JSON before launch if the
prompt ids/text/order, `max_tokens`, greedy temperature metadata, or target id
do not match. Do not reuse a reference captured for a different target or prompt
shape.

For strict HF evidence, also set `MESH_SHARD_HF_REQUIRE_REFERENCE=1` and
`MESH_SHARD_HF_REQUIRE_ADVERSARIAL=1`. The HF runner propagates synthetic wire
delay/jitter and `SKIPPY_SPEC_DRAFT_FAULT_*` hooks to both the local coordinator
and the owned HF worker, then fails if rollback or stale-window recovery evidence
is missing under adversarial pressure.

Prefer two HF passes:

- clean pass: `MESH_SHARD_HF_REQUIRE_REFERENCE=1`,
  `MESH_SHARD_HF_REQUIRE_ADVERSARIAL=0`, and a real speedup floor;
- adversarial pass: `MESH_SHARD_HF_REQUIRE_REFERENCE=1`,
  `MESH_SHARD_HF_REQUIRE_ADVERSARIAL=1`,
  `SKIPPY_SPEC_DRAFT_FAULT_EVERY=<n>`, and a relaxed speedup floor.

Use the clean pass for the throughput claim and the adversarial pass for
rollback/stale-window correctness. A forced-fault run can intentionally waste
work, so it is not by itself a fair speedup result.

Also plan a late-window/reordered-return adversarial pass. Shard's ring usually
returns results FIFO on one return channel, but delayed verify results and return
channel identity races are load-bearing WAN failure modes. For the current mesh
FIFO direct-return stream, set `SKIPPY_SPEC_RETURN_DELAY_EVERY` and
`SKIPPY_SPEC_RETURN_DELAY_MS` to force delayed-but-not-dropped verify returns.
Set `SKIPPY_SPEC_RETURN_RECONNECT_EVERY=<n>` for a separate return-channel churn
pass: the tail writer forces a reconnect after every `n` successful replies,
waits for the coordinator's replacement upstream-opened sink for the same
request/session, and sends the queued reply on that replacement stream.
This hook is only meaningful after both halves are covered by focused tests:
the tail must swap queued replies onto a replacement sink, and the coordinator
must reopen and accept a replacement sink after the current return stream ends.
The proof should show sent pipelined windows equal FIFO-accounted return windows
and zero decode-step order violations. It must also show
`llama_stage.spec.pipelined_identity_violations == 0`; missing identity metrics
mean the run used an older binary and is not a current Shard proof. It should
also show max in-flight windows above one and sent windows equal committed plus
stale windows; otherwise the run may be FIFO-correct but effectively serial.
When the reconnect hook is set, require `direct_return_reconnect_observed`.
If mesh transport can later deliver returns out of order, rerun with a
reordered-return adversary before treating unordered delivery as supported.

Finally, fixed K is the deterministic proof shape, not the whole WAN policy.
Add a K sweep or adaptive-K run when evaluating topology changes so higher RTT
shows higher useful draft depth instead of only one hand-picked K.
Use `scripts/skippy-shard-sweep.sh --kind mesh` or `--kind hf` to run repeated
proof passes over `MESH_SHARD_SWEEP_KS` and `MESH_SHARD_SWEEP_DEPTHS`. The
aggregate `sweep.json` should show pipelined throughput and
`tokens_per_s_ratio_vs_sync_draft` improving with useful depth under a
latency-bound setup; otherwise the run is not proving Shard's latency-hiding
claim even if FIFO correctness passes.

The local persistent-host proof runner,
`scripts/skippy-shard-mesh-proof.sh`, accepts the analogous
`MESH_SHARD_PROOF_REFERENCE_BASE_URL`,
`MESH_SHARD_PROOF_REFERENCE_MODEL`, and
`MESH_SHARD_PROOF_REFERENCE_RESULTS_JSON` variables. Use them before spending on
HF when a local full-target server or precomputed reference is available.

Reference matching by itself is a happy-path gate. For Shard-style latency
hiding, also run an adversarial pass with synthetic wire delay/jitter and
`SKIPPY_SPEC_DRAFT_FAULT_EVERY` so stale windows, draft rollback, target-session
recovery, and re-prime behavior are exercised while still matching the
canonical greedy reference.

For that local adversarial pass, set
`MESH_SHARD_PROOF_REQUIRE_REFERENCE=1` and
`MESH_SHARD_PROOF_REQUIRE_ADVERSARIAL=1`. The local runner fails on missing
`proof_gates`: direct stage path, direct prediction return, speculation
engagement, `K+1` linear verify-window shape, post-reject draft reset,
stale-window recovery, pipelined depth, pipelined speedup, and canonical
reference match. Pipelined mode also gates on zero identified-reply violations.
Tree mode is gated by tree activity metrics rather than the linear `K+1` window
shape.

## Metrics To Capture

Capture per mode:

- Output token match against greedy baseline.
- Tokens/sec and first-token/steady-state latency.
- `llama_stage.spec.accept_rate`.
- `llama_stage.spec.primary_verify_elapsed_ms`.
- `llama_stage.spec.draft_propose_ms`.
- `llama_stage.spec.pipelined_stale_windows`.
- `llama_stage.spec.pipelined_async_draft_wait_ms`.
- `llama_stage.spec.tree_windows`.
- `llama_stage.spec.tree_nodes`.
- `llama_stage.spec.tree_gather_ms`.

A useful report table is:

```text
mode | topology | target | draft | prompt_set | temp | tok/s | accept_rate | verify_ms | draft_ms | stale_windows | tree_gather_ms | token_match
```

## Failure Triage

- Token mismatch in tree mode: inspect tree acceptance/gather first, then
  tokenizer compatibility.
- Full reject at root: likely weak draft pairing or mismatched sampling.
- Mismatch after accepted branch: suspect KV gather path or downstream session
  alignment.
- Good correctness but no speedup: inspect direct-return availability, verify
  time versus draft time, stale windows, and tree gather cost.
