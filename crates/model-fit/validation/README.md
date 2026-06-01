# model-fit validation corpora

These manifests define repeatable GGUF sets for `model-fit-validate`.

The lists are stratified by estimator behavior rather than popularity:

- tiny dense models check fixed decode overhead and low-active-byte behavior
- small dense models check the transition into memory-bandwidth-bound decode
- 7B/8B and coder models check common local serving shapes
- quant pairs check Q4/Q5/Q6/Q8 slope changes without changing architecture
- MoE models check active expert bytes instead of total expert bytes
- embedding/reranker models check workload suitability in metadata reports

Use the smoke set for self-hosted PR validation:

```bash
target/release/model-fit-validate \
  --no-progress \
  --models-file crates/model-fit/validation/smoke-models.txt \
  --output-json /tmp/model-fit-validation.json

target/release/model-fit-check-validation \
  --min-models 8 \
  /tmp/model-fit-validation.json

target/release/model-fit-check-validation \
  --scenario all \
  --markdown-out /tmp/model-fit-validation.md \
  /tmp/model-fit-validation.json
```

Use the deep set for manual or nightly validation on high-memory runners:

```bash
target/release/model-fit-validate \
  --no-progress \
  --models-file crates/model-fit/validation/deep-models.txt \
  --output-json /tmp/model-fit-validation-deep.json
```

`model-fit-validate --models-file` ignores blank lines and `#` comments.

Use the Q8/MoE focused set when changing tensor-type traffic or active-expert
dispatch modeling:

```bash
target/release/model-fit-validate \
  --no-progress \
  --models-file crates/model-fit/validation/q8-moe-models.txt \
  --output-json /tmp/model-fit-validation-q8-moe.json
```

## ABI decode validation

The validator can run Skippy's single-stage benchmark and the Skippy decode ABI
probe for each GGUF. The ABI probe exercises llama.cpp's decode graph directly
and reports a denoised median over repeated observations; it is meant to check
whether the metadata-only fit estimate lands near the runtime's actual decode
path without feeding observed throughput back into scoring.

The steady-decode estimator is source-grounded rather than model-name grounded:

- GGUF tensor profiles identify attention, FFN, output, and expert matmul bytes.
- Tensor type profiles distinguish Q8_0, K-quants, f16/f32, IQ, and unknown
  groups because llama.cpp dispatches different GGML matmul kernels for those
  tensor types.
- Layer count and logical matmul shape counts approximate the per-token graph
  shape around `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`.
- `mesh-llm gpus benchmark` supplies measured decode-shaped bandwidth and fixed
  backend submission overhead; model-fit consumes those hardware facts instead
  of assuming Metal, CUDA, or ROCm behavior.

Representative two-machine broader validation after the ABI decode probe,
source-grounded graph-overhead estimator, prefill scenario split, and
single-model execution-budget selector:

| machine | backend | scenario | median abs error | notable result |
|---|---|---|---:|---|
| Mac Studio M1 Ultra | Metal | steady_decode | 9.8% | 16 benchmarked samples; tiny models now select measured Metal instead of optimistic CPU fallback, with Q5/Q6/Q8 coder and reranker misses still visible. |
| white.local | CUDA | steady_decode | 8.6% | 16 benchmarked samples; Qwen2.5-Coder Q4/Q5/Q6 matched, while Q8, OLMoE, and reranker behavior remain honest misses. |
| Mac Studio M1 Ultra | Metal | prefill | 24.6% | Prefill is validated as prompt tokens divided by Skippy's `prefill_elapsed_ms`; it is separate from first-token latency and not yet within the steady-decode band. |
| white.local | CUDA | prefill | 108.1% | CUDA prefill is much faster than the current metadata prediction for several dense models, so this remains future source-work rather than a tuned constant. |
| Mac Studio M1 Ultra | Metal | first_token | 19.2% | First-token latency includes tokenize, prefill, first decode, and request overhead; larger coder quants are still slower than fit. |
| white.local | CUDA | first_token | 58.9% | First-token prediction still needs prompt-shape and backend scheduling work. |
| Mac Studio M1 Ultra | Metal | kv_warm_reuse | 13.7% | KV reuse is close but still over-predicts Qwen2.5-Coder Q4/Q5 and bge reranker. |
| white.local | CUDA | kv_warm_reuse | 16.4% | CUDA KV reuse is stable but remains slower than fit on several dense and reranker samples. |

These numbers are validation evidence, not calibration inputs. Do not loosen
thresholds, add model-specific exceptions, or use observed throughput in the
metadata-only estimator to make this table pass. Treat misses as hypotheses to
test against source behavior, hardware facts, or broader held-out models.

Focused Q8/MoE follow-up after removing the Q8_0 stored-byte discount and
making measured MoE dispatch overhead use measured fixed submission cost:

| machine | backend | scenario | samples | median abs error | notable result |
|---|---|---|---:|---:|---|
| Mac Studio M1 Ultra | Metal | steady_decode | 6 | 8.8% | Qwen2.5-Coder Q8 moved from slower-than-fit to a 1.06 observed/fit match; OLMoE remained close but noisy; bge reranker stayed slower-than-fit. |
| white.local | CUDA | steady_decode | 6 | 8.3% | Qwen2.5-Coder Q8 improved from 0.60 to 0.86 observed/fit, and OLMoE improved from 1.97 faster-than-fit to 0.90 match. |
| Mac Studio M1 Ultra | Metal | prefill | 5 | 12.2% | Qwen2.5-Coder Q4/Q5/Q6 and OLMoE are close; Q8 prefill remains noisy/slower-than-fit. |
| white.local | CUDA | prefill | 5 | 223.6% | CUDA prefill remains a known residual miss and should not be hidden by decode changes. |
| Mac Studio M1 Ultra | Metal | kv_warm_reuse | 6 | 14.3% | Qwen2.5-Coder Q8 now matches KV reuse; Q4/Q5 and bge reranker remain slower-than-fit/noisy. |
| white.local | CUDA | kv_warm_reuse | 6 | 17.5% | Qwen2.5-Coder and bge reranker KV reuse remain slower than steady-decode fit. |
