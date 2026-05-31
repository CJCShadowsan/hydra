# model-fit validation corpora

These manifests define repeatable GGUF sets for `model-fit-validate`.

The lists are stratified by estimator behavior rather than popularity:

- tiny dense models check fixed decode overhead and low-active-byte behavior
- small dense models check the transition into memory-bandwidth-bound decode
- 7B/8B and coder models check common local serving shapes
- quant pairs check Q4/Q8 slope changes without changing architecture
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

Representative two-machine five-model validation after the ABI decode probe and
source-grounded graph-overhead estimator:

| machine | backend | scenario | median abs error | notable result |
|---|---|---|---:|---|
| Mac Studio M1 Ultra | Metal | steady_decode | 8.9% | Qwen3 8B matched at 0.98 observed/fit; Qwen3 0.6B remained a 0.79 miss and two samples were noisy. |
| white.local | CUDA | steady_decode | 5.8% | All five steady-decode samples matched within the 10% target band. |
| Mac Studio M1 Ultra | Metal | first_token | 12.1% | First-token latency is close but still misses on Qwen3 0.6B, Llama 3.2 3B, and Qwen3 8B. |
| white.local | CUDA | first_token | 59.7% | First-token prediction still needs separate prefill and prompt-shape work. |
| Mac Studio M1 Ultra | Metal | kv_warm_reuse | 18.8% | KV reuse has noisy small-model samples and slower-than-fit misses on 3B/8B. |
| white.local | CUDA | kv_warm_reuse | 14.0% | CUDA KV reuse is stable but remains slower than fit on several models. |

These numbers are validation evidence, not calibration inputs. Do not loosen
thresholds, add model-specific exceptions, or use observed throughput in the
metadata-only estimator to make this table pass. Treat misses as hypotheses to
test against source behavior, hardware facts, or broader held-out models.
