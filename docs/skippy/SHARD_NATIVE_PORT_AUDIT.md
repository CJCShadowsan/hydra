# Shard Native Port: State and Plan

This is the single source of truth for the Shard-native Skippy effort. Keep
status, evidence, and next steps here; other docs should only point here.

## Goal

Port the Shard approach into native mesh-llm/Skippy:

- direct tail-to-coordinator verify return;
- fixed `[current_token] + K draft tokens` target verify windows;
- multiple verify windows in flight to hide WAN latency;
- full-accept, reject-correction, stale-window discard, and draft re-prime
  semantics matching Shard;
- rollback-safe stage KV and return-channel recovery;
- later: Shard `FastVerify`-style fixed-shape/static-KV verify and GLM-5.2
  certification.

This is not just a new mode name. `speculative.mode = "shard-pipeline"` is an
operator alias for the linear Shard scheduler path, and the implementation has
to preserve Shard's scheduler and transport semantics.

## State Of Play

| Area | State | Remaining gate |
|---|---|---|
| Linear Shard scheduler | Ported for sync and pipelined draft paths. It uses fixed `[current] + K draft` verify windows, depth, stale discard, reject correction, and draft re-prime. Proven across two real machines (local Metal stage-0 + HF CUDA stage-1, ~240 ms WAN RTT): greedy-identical to target-only, an absolute speculative-over-baseline win (sync 1.36x, pipelined 1.57x vs target-only on natural prompts), and pipelined beats serial sync-draft (1.04x-1.16x). | Larger latency-bound K/depth sweeps and GLM_DSA proof. |
| Direct tail return | Ported as first-class binary transport with identified reply envelopes. Multi-node WAN proof shows it cuts coordinator primary-verify downstream wait under pipelined depth. | Unordered/multiplexed return transport is not supported yet. |
| Return-channel reconnect | Ported for the direct-return split path. Tail sends explicit `PredictionReturnReconnect`, coordinator reopens a replacement upstream-opened sink, tail sends the queued reply there. Proven on the real WAN split: `direct_return_reconnect_observed` with output still token-exact under forced reconnect churn. | Unordered/multiplexed return transport is still out of scope. |
| Rollback/KV repair | Present via trim/replay and forced downstream repair replies; stale direct-return replies are discarded after correction. Proven token-exact under forced faults plus return-channel reconnect churn on the real WAN split. | Cheap checkpoint/branch handles and static-KV overwrite path. |
| Fast verify | Not ported. Skippy still uses llama.cpp eager/batched verify, not Shard `FastVerify`. | Design and implement fixed-shape verify/capture, static KV overwrite rollback, and live-context bucketing. |
| Tree speculation | ABI and mechanics exist; greedy output matches the target-only reference both same-machine and across the real WAN split (3B target + 1.5B draft, `tree_speculation_engaged`, all gates green). The earlier "deeper tree diverged" symptom was an emit-loop truncation bug (the tree commit loop pre-seeded its stop flag from `decision.reached_stop`, so a stop-window emitted only the first accepted token and dropped the rest, including the trailing EOS), not a mask/gather fault. Fixed in `embedded_generation.rs` and locked with a `speculative.rs` regression test. | Latency-bound depth/width throughput sweep, then evaluate as a multiplier under high RTT. |
| GLM-5.2 | Patch queue has partial GLM_DSA graph/indexer/trim support and safe-boundary policy. | No GLM_DSA single-stage parity, split parity, rollback proof, or GLM draft pairing proof yet. |

## Evidence

Current useful evidence is small/modest-model proof. It proves the scheduler
shape and a real multi-node WAN win (correct greedy output plus an absolute
speculative-over-baseline speedup on two machines with small models). It is the
pilot, not the GLM-scale headline; GLM-class pairing should raise acceptance and
the latency-hiding margin further.

| Proof | Result |
|---|---|
| Focused direct-return tests | `cargo test -p skippy-server --lib binary_transport::direct_return`: 12 tests passed on 2026-06-22. Covers tail replacement, generation-tagged stale-stream ignore, coordinator reopen, and forced reconnect no-loss/no-dup. |
| Local strict identity proof | `/tmp/mesh-shard-proof-identity-qwen25-20260622T065106Z/results/summary.json`: Qwen2.5 3B target, Qwen2.5 0.5B draft, two local split stages, `K=6`, depth 6. Target, sync draft, and pipelined draft token IDs matched canonical reference. |
| Local adversarial rollback proof | `/tmp/mesh-shard-proof-identity-adv-qwen25-20260622T065255Z/results/summary.json`: forced every 3rd draft token off-greedy; target, sync draft, and pipelined draft still matched reference; stale and rejected windows were accounted. |
| Local forced reconnect split proof | `/tmp/mesh-shard-proof.J6Cnsd/results/summary.json`: `SKIPPY_SPEC_RETURN_RECONNECT_EVERY=2`; target, sync draft, and pipelined draft all observed reconnect request and reconnect completion, matched canonical token IDs, and pipelined mode reached depth 6 with FIFO and identity violations 0. This is correctness under churn, not throughput. |
| Local latency proof | `/tmp/mesh-shard-proof-ref-qwen25-20260622T041611Z/results/summary.json`: `K=2`, depth 6, 1s synthetic direct-return delay. Pipelined draft was 1.217x sync draft while matching the canonical target reference. |
| HF/WAN Qwen2.5 proof | `/tmp/mesh-shard-hf-shard-hf-qwen25-identity-20260622T072929Z/results/summary.json`: local coordinator plus owned HF `t4-small` worker, forced `0..18 / 18..36`, RTT about 227-304 ms. Pipelined draft was 1.08x sync draft with strict reference match and identity violations 0. |
| HF/WAN Qwen3 depth sweep | `/tmp/mesh-shard-sweep-qwen3-06b-depth-20260621T233957Z/sweep.json`: fixed `K=2`, depths 2/4/6 improved pipelined-vs-sync throughput as depth rose, with FIFO violations 0. |
| Local tree greedy-parity proof | `/tmp/tree-proof-out/results/summary.json`: local two-stage Metal split, Qwen2.5-1.5B target, Qwen2.5-0.5B draft, `mode = "tree"`, `draft_max_tokens = 6`, `temperature = 0`. All three deterministic prompts produced tree-mode token IDs identical to the target-only greedy run, including the trailing EOS, with `proof_gates.output_matches_target = true`. This regression-tests the tree emit-loop truncation fix; it is a correctness gate, not a throughput claim. |
| Local four-mode greedy-parity proof | `/tmp/full-proof-out/results/summary.json`: local two-stage Metal split, Qwen2.5-1.5B target, Qwen2.5-0.5B draft, 40 ms synthetic downstream wire delay, `temperature = 0`, modes `target sync-draft pipelined-draft tree`. All speculative modes produced token IDs identical to the target-only reference and passed every Shard mechanism gate (direct stage path, direct prediction return, `[current]+K` verify-chunk shape, pipelined depth engaged, FIFO return accounting, `pipelined_identity_violations = 0`, stale-window KV recovery, post-reject draft recovery). The only failing gates were `pipelined_speedup_ok` / `pipelined_speedup_vs_sync_ok`; see the single-GPU throughput note below. |
| Multi-node WAN absolute-win proof | `/tmp/wan-proof-natural/results/summary.json`: real two-machine split (local Metal stage-0 + HF `t4-small` CUDA stage-1, ~240 ms direct RTT), worker built from this worktree (HF build `6a390c84953ed90bfb94803a`), target `meshllm/skippy-shard-qwen25-3b-q4-k-m-layers-proof-20260621`, same-family Qwen2.5-1.5B draft, natural prompts, `temperature = 0`. All gates passed (`WAN-PROOF exit=0`). Sync-draft and pipelined-draft were token-identical to target-only. Absolute speculative-over-baseline win: sync-draft 3.150 vs target 2.319 tok/s (1.36x), pipelined-draft 3.642 tok/s (1.57x). Pipelined beat serial sync-draft 1.156x by holding in-flight windows and overlapping verify round-trips across the WAN. This is the headline pilot result: Shard split speculation is correct and faster than baseline on real multi-node hardware with small models. |
| Multi-node WAN tree-parity proof | `/tmp/wan-proof-tree/results/summary.json`: real two-machine WAN split (local Metal stage-0 + HF CUDA stage-1, ~240 ms RTT), 3B target + 1.5B draft, `mode = "tree"`, natural prompts, `temperature = 0`. All gates passed (`WAN-TREE exit=0`). Tree output was token-identical to target-only on every prompt with tree speculation engaged (3-4 tree windows, 39-52 tree nodes per request). This proves the tree path (and the emit-loop truncation fix) holds across the real split, not just same-machine. All owned HF jobs cancelled; no orphaned spend. |
| Multi-node WAN adversarial rollback proof | `/tmp/wan-proof-adv/results/summary.json`: same two-machine WAN split, 3B target + 1.5B draft, natural prompts, `temperature = 0`, with forced faults (`SKIPPY_SPEC_DRAFT_FAULT_EVERY=3`) and return-channel churn (`SKIPPY_SPEC_RETURN_RECONNECT_EVERY=4`). Pipelined output stayed token-identical to target-only on every prompt while exercising the full Shard recovery machinery: 5-13 rejections, 26-44 stale windows discarded, 5-13 KV recovery restores per prompt, with `pipelined_stale_kv_recovery_observed`, `post_reject_draft_recovery_observed`, `direct_return_reconnect_observed`, and `pipelined_identity_match` all true. Only `pipelined_speedup_vs_sync_ok` was false, which is expected for a forced-fault correctness pass (intentionally wasted work is not a fair speedup measurement). This proves rollback, stale-window discard, re-prime, and return-channel reconnect under WAN churn. All owned HF jobs cancelled; no orphaned spend. |
| Multi-node WAN mechanism proof | `/tmp/wan-proof-out2/results/summary.json`: same two-machine split with the weaker Qwen2.5-0.5B draft and adversarial "return exactly" prompts, `temperature = 0`. Sync-draft and pipelined-draft were token-identical to target-only and pipelined beat serial sync-draft (1.355 vs 1.305 tok/s, ~1.04x; `pipelined_speedup_vs_sync_ok = true`) by holding up to 4 in-flight windows and cutting coordinator downstream wait across the WAN. Both speculative modes were below target-only here (0.85x / 0.82x) because that pairing/prompt shape accepts too few drafts; this isolates that the absolute baseline win is gated on acceptance, not the scheduler. All owned HF jobs cancelled; no orphaned spend. |
| Single-GPU depth sweep | `/tmp/shard-sweep-out/sweep.json`: local single-Metal-GPU split, K=6, self-draft (~100% accept), 300 ms delay, depths 2/4/6. Output token-identical to target at every depth. Pipelined/sync degraded with depth (0.811x, 0.720x, 0.660x), confirming that on one shared GPU deeper pipelining only adds verify-window contention. This is the expected single-device inverse of the multi-node result (pipelined 1.16x over sync at depth 6), and isolates depth-scaling as a multi-accelerator property. |
| Latency-bound pipelined mechanism proof | `/tmp/selfdraft-proof-out/results/summary.json`: same two-stage Metal split with 300 ms synthetic downstream wire delay and self-draft (draft == target) to drive near-full acceptance and isolate wire-wait hiding. Pipelined direct return hid the wire round-trip (`primary_verify_downstream_wait_ms` dropped from sync's 581 ms to 12 ms on the long prompt) while matching target token IDs and passing all mechanism gates. Wall-clock still favored sync-draft because stale/in-flight verify windows contend for the single shared Metal GPU; `pipelined_speedup_vs_sync_ok = false`. This proves the latency-hiding mechanism, not a single-node throughput win. |

Known noise: some local proof runs exposed a Metal backend shutdown assert after
SIGTERM. The proof processes exited 0; track that as cleanup separately, not as
scheduler correctness evidence.

### Multi-node WAN proof (real two-machine pilot)

The throughput claim was moved off a single shared GPU onto two real machines: a
static local Metal coordinator (built with `MESH_LLM_DYNAMIC_NATIVE_RUNTIME=0`,
embedding the branch-local Skippy ABI) running stage-0 (layers 0..18), plus a
Hugging Face `t4-small` CUDA worker running stage-1 (layers 18..36), built from
this exact worktree via `scripts/skippy-shard-hf-build-artifact.sh` (HF build
job `6a390c84953ed90bfb94803a`, COMPLETED). Target was the catalog layer-package
`meshllm/skippy-shard-qwen25-3b-q4-k-m-layers-proof-20260621`, draft was a local
Qwen2.5-0.5B GGUF, `temperature = 0`, run via
`scripts/skippy-shard-hf-wan-proof.sh`.

The mesh formed across the WAN with a real ~240 ms direct RTT, both stages
reported `ready`, and all three modes (`target`, `sync-draft`,
`pipelined-draft`) ran to completion. Evidence at
`/tmp/wan-proof-out2/results/summary.json`:

- Greedy correctness held across the split: sync-draft and pipelined-draft
  output was token-identical to the target-only reference on every prompt,
  including the trailing EOS.
- Pipelined direct return beat serial sync-draft: 1.355 vs 1.305 tok/s overall
  (~1.04x), with `pipelined_speedup_vs_sync_ok = true`. The mechanism is visible
  per prompt: pipelined held up to 4 in-flight windows and cut the coordinator's
  primary-verify downstream wait (for example exact-1: sync 1165 ms vs pipelined
  890 ms) by overlapping verify round-trips across the ~240 ms WAN.
- The only failing gate was `pipelined_speedup_ok` (pipelined vs target-only):
  both speculative modes were slower than target-only here (0.85x and 0.82x),
  because this 3B-target / 0.5B-draft pairing accepts too few tokens
  (committed accept rate well under 0.5 on the short proof prompts) to overcome
  per-window verify overhead. That is a draft-pairing/acceptance economics
  result, not a scheduler or transport fault; a stronger same-family draft or a
  higher-acceptance workload is the lever, and the GLM-class pairing is the
  intended high-acceptance, latency-bound regime.

This proves the Shard latency-hiding scheduler works spread across two real
machines on small models: correct greedy output, direct return, fixed
`[current]+K` windows, pipelined in-flight depth, and a real
pipelined-over-sync win under WAN RTT. The remaining lever for an absolute
speculative-over-baseline win is draft acceptance, which is exactly what scales
up at GLM size. All owned HF jobs were cancelled or completed; no orphaned spend
remains.

### Depth-scaling sweep (single-GPU vs multi-node)

A fixed-K=6 depth sweep (`scripts/skippy-shard-sweep.sh --kind mesh`, depths
2/4/6, self-draft for ~100% acceptance, 300 ms synthetic delay,
`/tmp/shard-sweep-out/sweep.json`) makes the single-GPU limitation explicit:
pipelined throughput *degrades* as depth rises (pipelined/sync 0.811x at depth 2,
0.720x at depth 4, 0.660x at depth 6), because deeper pipelining issues more
ahead-of-acceptance verify windows that all contend for the one shared Metal GPU.
Output stayed token-identical to target at every depth. This is the inverse of
the latency-hiding signature the validation skill looks for, and it is expected:
on a single device, depth cannot help. The real two-machine WAN run shows the
opposite direction at depth 6 (pipelined 1.16x *over* sync), because the
in-flight verify windows execute on the peer's GPU concurrently. The depth-scaling
improvement curve is therefore a genuinely multi-accelerator property; capturing
the rising pipelined-vs-sync curve across depths would require a multi-node
(HF-worker) sweep, not the local single-GPU one.

### Single-GPU throughput limitation

The local two-node proofs run both split stages on one physical Metal GPU, so
every stale and in-flight verify window competes for the same device. The
pipelined scheduler correctly hides the wire round-trip (downstream wait drops
to near zero under synthetic delay), but the verify compute it issues ahead of
acceptance serializes on the single GPU, so wall-clock throughput does not beat
serial sync-draft locally. This is expected: Shard's latency-hiding win is a
multi-machine claim. The throughput gate only becomes meaningful when each stage
owns a separate accelerator, so the in-flight verify windows execute on a peer's
GPU concurrently with the coordinator instead of contending for one device. The
single-machine proofs are therefore scoped to mechanism and greedy correctness;
the throughput headline must come from a true multi-node (or HF-worker) run.

## Plan

1. **Stabilize the linear Shard path.**
   Keep the current scheduler/direct-return/reconnect code gated by focused
   tests and strict local proof. No release/HF run should be used as first
   evidence for a changed primitive.

2. **Run the reconnect proof on release/HF.**
   Use `SKIPPY_SPEC_RETURN_RECONNECT_EVERY=2` with the same Qwen2.5 target,
   draft, prompts, reference, and forced topology. This should prove the local
   reconnect seam survives the real worker artifact and WAN path.

3. **Implement Shard `FastVerify` equivalent.**
   Add a native fixed-shape verify path in the staged llama.cpp ABI:
   static/checkpointed KV, overwrite-at-start rollback, fixed verify shapes,
   CUDA/Metal feasibility by backend, and live-context bucketing. This is the
   main missing performance primitive.

4. **Certify GLM_DSA before any GLM-5.2 claim.**
   Prove target-only GLM_DSA greedy/logit parity, safe split boundaries,
   rollback under forced faults, and a tokenizer/vocab-compatible draft pairing
   such as the Shard GLM-4-9B pairing. Only then run GLM WAN.

5. **Scale proof.**
   After fast verify and GLM_DSA gates, run larger K/depth sweeps on a
   latency-bound split target, then evaluate tree speculation as a multiplier.

## Reference Sources

- `../shard/phase0/specpipe.py`: linear scheduler, direct return, stale-window
  discard, tree coordinator.
- `../shard/phase0/tree.py`: flattened tree, ancestor attention, accepted-path
  KV gather.
- `../shard/phase0/fastverify.py`: static KV, fixed-shape verify, overwrite
  rollback, live-context bucketing.
- `../shard/research/glm_swarm_nvfp4_pipe.py`: GLM-5.2 WAN path with GLM-4-9B
  draft.
- `../shard/research/glm_swarm_nvfp4_cg.py`: GLM draft CUDA-graph experiment
  and rollback lesson.
- `../shard/docs/research/wan-speculative-decoding.md` and receipts: headline
  WAN claims and exact run evidence.
