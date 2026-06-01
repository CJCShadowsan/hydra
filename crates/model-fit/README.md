# model-fit

`model-fit` ranks GGUF model artifacts against a local hardware profile.

The crate is intentionally metadata-first and deterministic. It consumes:

- hardware facts and measured GPU bandwidth, decode overhead, scalar compute,
  dense prefill matmul throughput, and MoE-shaped prefill matmul throughput
  from `mesh-llm gpus benchmark`
- GGUF-derived model metadata such as tensor bytes, layer count, hidden width,
  KV dimensions, context length, architecture class, quantization, tokenizer
  metadata, and workload capability evidence
- workload preferences for chat, coding, tool use, summarization, embedding,
  reranking, and related local inference shapes

The selector estimates runtime memory, KV cache size, active decode bytes,
decode throughput, prefill throughput, first-token latency, workload fit, and
split candidacy. It does not use model filenames or catalog reputation as a
performance signal.

## Validation

The crate includes two validation manifests:

- `validation/smoke-models.txt` for self-hosted PR smoke testing
- `validation/deep-models.txt` for manual or nightly high-memory validation
- `validation/q8-moe-models.txt` for focused Q8, dense prefill, and MoE checks

Run a local validation pass:

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

The validation report is JSON so later agents can analyze hardware facts, model
profiles, recommendations, benchmark observations, the estimator input
contract, and scenario-level agreement.
