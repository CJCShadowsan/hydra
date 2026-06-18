# Next Goal: Improve Native Q4 Top-1 Acceptance

This file is disposable. Durable status, evidence, caveats, and follow-up notes
belong in `evals/spd/README.md` and `docs/skippy/speculative_decoding.md`.

## Current Checkpoint

- Target model: `Qwen/Qwen3-8B`.
- Target Skippy package: `meshllm/Qwen3-8B-Q4_K_M-layers`.
- First real layout: two nodes, one Skippy stage per node.
- Layer split: coordinator `0..23`, worker `23..36`.
- SPD topology: `num_stages=2`, `stage_layer_boundaries=23,36`.
- Required sidecar tap rows: `0,23,36;0,23`.
- Worker tap-return indices: `[23,36]`.
- HF job `meshllm/6a33e49bef9220ea67d991c2` completed under the `$50` cap and
  uploaded `runs/20260618-122936` to
  `meshllm/skippy-spd-qwen3-8b-s2-23`.
- Training used UltraChat `train_sft`, `15997` usable rows, BF16,
  `max_length=2048`, `epochs=1`, LR `1e-4`, `num_spec_layers=4`, and draft
  top-k `4`.
- Reference held-out eval on `96` prompts / `6123` generated tokens reported
  aggregate acceptance `0.7013`, equivalent accept length `1.4026`, and
  theoretical gain `41.0%`.
- BF16 serving export and Rust/Python fixture parity passed.
- Live package-backed strict fixture parity did not pass against native Q4
  hidden states, but required taps were present and greedy verifier output
  matched baseline. Treat that as quantization/serving drift, not broken tap
  wiring.
- Local package-backed rolling smoke matched content on `6 / 6`, had `0` tap
  failures, proposed `90`, accepted `17`, rejected `73`.
- Real two-stage worker smoke over a direct low-latency link matched content on
  `6 / 6`, had `0` tap failures, proposed `89`, accepted `16`, rejected `73`,
  and committed `12` optimistic tokens.

## Immediate Objective

Improve native Q4 top-1 acceptance for the exact `23,36` split before any speed
claim.

The current head proves the training/export/request path, but native top-1
serving acceptance is only about `18%`. That saves some token round trips, but
not enough to beat sidecar and rolling-executor overhead.

## Quality Gate

Do not use another real two-node run as the first quality test. The predictor
can be trained and scored before any physical split:

1. Build a product-tap corpus from broader prompt rows for the same `23,36`
   topology; keep held-out prompts separate.
2. Attach HF teacher logits to captured product rows, or expose native target
   logits if that becomes available.
3. Fine-tune from the 16k UltraChat checkpoint on product-tap rows.
4. Re-export `spd-head.safetensors` and a parity fixture.
5. Run Rust fixture parity, local package-backed smoke, then the real worker
   smoke only if local package-backed native acceptance improves.

## Logical Topology Rule

Train sidecars for logical layer-boundary topologies, not hostnames. With one
Skippy stage per node, the first two-node Qwen3-8B run uses the `23,36`
sidecar. If a future deployment packs adjacent logical stages onto one larger
node, it may reuse the same sidecar only if the runtime still exposes every
logical boundary tap the manifest requires.

This avoids a full combinatorial explosion: precompute a small set of canonical
logical topologies, then clump contiguous logical stages during placement. The
tradeoff is that clumped logical stages may lose some physical overlap, so they
must be benchmarked honestly, but they do not require retraining if the tap
topology is unchanged.

## Next Actions

1. Build or reuse a held-out product prompt set for native package-backed
   capture.
2. Capture product tap rows from the current 16k head/topology, keeping train
   and held-out prompts separate.
3. Attach teacher logits to captured rows.
4. Fine-tune from the 16k checkpoint and rerun export, fixture parity, local
   smoke, and worker smoke.
5. Gate on native package-backed top-1 acceptance and paper-style saved versus
   unsaved round trips. Do not run another speed comparison until the estimate
   clears `1.0` with margin.
6. Consider a 64k HF follow-up only after the product-tap gate shows the recipe
   improves native acceptance rather than just reference top-k coverage.
