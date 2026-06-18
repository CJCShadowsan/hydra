# Next Goal: Robust Qwen3-8B SPD Real-Split Gate

This file is disposable and should be deleted when this immediate gate is done.
Durable status, evidence, caveats, and follow-up notes belong in
`evals/spd/README.md` and `docs/skippy/speculative_decoding.md`.

## Current Checkpoint

- Topology: `Qwen/Qwen3-8B`, `meshllm/Qwen3-8B-Q4_K_M-layers`, logical
  boundaries `23,36`.
- Current sidecar:
  `/tmp/spd-qwen3-8b-product-finetune-paper3-train16-e5-lr2e5/`.
- Held-out live tap: `110 / 192` accepted, exact greedy output on `24 / 24`.
- All-local OpenAI rolling: exact content on `24 / 24`, `0` tap failures,
  `0` ignored taps, `81 / 160` accepted, `81` saved / `79` unsaved token round
  trips, `paper_pipeline_estimate=1.0125x`.
- Real two-node direct-cable OpenAI rolling after refreshing the native Metal
  stage ABI: exact content on `24 / 24`, `0` tap return failures, `0` tap
  record failures, `0` ignored taps, `78 / 156` accepted, `78` saved / `78`
  unsaved token round trips, `paper_pipeline_estimate=1.0x`.
- Timing from that real split: baseline decode mean `439.0ms`, SPD decode mean
  `1366.6ms` (`0.321x`), mean probe head time `64.7ms`, normal downstream
  wait `144.0ms`, optimistic downstream wait `66.5ms`, and chained hidden wait
  `77.1ms`.
- Larger max120 product-corpus check:
  - `/tmp/spd-qwen3-8b-product-prompts-paper3-train32-heldout16-max120`
    captured `712` train rows and `256` disjoint held-out rows without the
    earlier `n_batch=128` assertion.
  - HF teacher alignment is not the dominant held-out blocker: teacher top-1
    matched native Q4 on `245 / 256`, while the HF-KL sidecar matched teacher
    on only `95 / 256`.
  - HF-KL train32 head:
    `/tmp/spd-qwen3-8b-product-finetune-paper3-train32-max120-e5-lr2e5/`,
    fixture parity clean, held-out live tap `91 / 256`.
  - Native-hard-label train32 head:
    `/tmp/spd-qwen3-8b-product-finetune-paper3-train32-max120-hard-e5-lr2e5/`,
    fixture parity clean, held-out live tap `95 / 256`.
- Bottom line: mechanics work on the real split, but quality is only marginal
  and the latest larger-product fine-tunes are worse on held-out prompts. This
  is still not a speedup result.

## Immediate Objective

Make the same `23,36` product sidecar robust enough for a real split handoff:

- exact baseline/SPD content on held-out prompts;
- `0` tap return failures, `0` tap record failures, `0` ignored taps;
- `summary.paper_pipeline_estimate.paper_like_speedup_vs_serial_split > 1.0`
  with margin on the real two-node split, not just all-local;
- report saved/unsaved token round trips and mean sidecar/head, downstream-wait,
  optimistic-wait, and chained-hidden-wait timings.

## Next Actions

1. Do not run another request-path or release timing smoke until held-out
   live-tap acceptance recovers with margin; the latest train32 HF-KL and
   hard-label heads are not speed candidates.
2. Capture a larger disjoint short-prompt corpus before more training. The
   generated next split is
   `/tmp/spd-qwen3-8b-product-prompts-paper3-train56-heldout8-max120`
   (`137` train prompts, `16` held-out prompts, still `max_prompt_tokens=120`
   for the current live-tap batch limit).
3. Train less-overfit candidates on that larger corpus and rerun the held-out
   attribution: sidecar-vs-native, sidecar-vs-HF-teacher, and HF-teacher-vs-Q4.
4. If larger/diverse product rows still fail, expose native Q4_K_M verifier
   top-k/logits for paper-faithful supervision instead of spending on larger
   generic HF-teacher KL.
5. Repeat all-local live-tap and OpenAI rolling only as a quick filter once
   held-out live-tap quality clears; treat the real two-node direct-cable run as
   the acceptance gate after that.
6. Before another request-path smoke, confirm the native stage ABI library and
   scoped release `skippy-server` / `skippy-bench` binaries are newer than the
   llama.cpp patch checkout; a stale native library can hide final-stage tap
   behavior even when Rust was rebuilt.

No spend-bearing HF/CUDA job without a dry-run plan and explicit approval.
