# Skippy SPD Proof Notes

This directory is the public, reproducible handoff for the Skippy Speculative
Pipeline Decoding (SPD) proof.

SPD is treated here as a separate trained sidecar head. It proposes draft tokens
from selected target-model hidden states; the target model still verifies every
accepted token. The work in this directory proves the training/evaluation path
and records the artifact contract Skippy needs before serving the head from
Rust.

## What Works

- A real SPD head can be trained locally for `Qwen/Qwen3-0.6B` with the paper's
  reference implementation.
- A real pretrained SPD head for `Qwen/Qwen3.5-4B` reaches high acceptance on
  local eval prompts.
- Real per-sample SPD eval traces can be fed into a Skippy split-stage latency
  model to estimate how much pipeline bubble/activation-hop latency SPD can
  hide.
- `skippy-runtime` can parse and validate the SPD head manifest/checkpoint
  binding, including a Rust-readable safetensors serving checkpoint. It does
  not execute the head yet.

## What Does Not Work Yet

- Skippy/Rust does not yet run the SPD head forward pass.
- Skippy does not yet expose the live hidden-state taps required by the head.
- No live Skippy generation request has used trained SPD proposals yet.
- The `.pt` checkpoint is a proof/training artifact. Export it to
  `spd-head.safetensors` before Rust-side serving work.

## Open Training Data

The local Qwen3-0.6B proof uses:

- dataset: `HuggingFaceH4/ultrachat_200k`
- split: `train_sft`
- rows: first `1024` rows for the recorded local proof

The reference SPD repository lists the intended training corpus family as:

- UltraChat-200k
- ShareGPT
- SmolTalk
- SmolTalk-Chinese

MT-Bench, HumanEval, and GSM8K prompts are used here only for evaluation.

## GLM 4.7 Frontload Smoke Path

`glm47_frontload.py` inspects a local GLM 4.7 checkpoint and writes a tiny
manifest-compatible SPD smoke artifact before any long training run. This is
for frontloading integration risk: GLM model metadata, non-uniform stage
boundaries, hidden-state tap rows, and Rust manifest validation. The generated
weights are shape fixtures, not a trained SPD head.

The default local checkpoint path points at the cached `zai-org/GLM-4.7-Flash`
snapshot when present. Override it with `--model-path` on another machine.

```bash
python evals/spd/glm47_frontload.py \
  --model-path /path/to/GLM-4.7-Flash \
  --work-dir /tmp/skippy-spd-glm47-frontload \
  --num-stages 3 \
  --stage-layer-boundaries 15,31,47 \
  --num-spec-layers 1 \
  --draft-vocab-size 8 \
  --write-smoke-artifacts
```

The command writes:

- `glm47-spd-frontload.json` — checkpoint inspection and derived topology
- `speculation_head_final.pt` — placeholder provenance file
- `spd-head.safetensors` — tiny Rust-readable serving shape fixture
- `skippy-spd-head.json` — manifest with `stage_layer_boundaries`

Validate the smoke manifest without building native llama.cpp:

```bash
SKIPPY_SPD_MANIFEST=/tmp/skippy-spd-glm47-frontload/glm47-spd-frontload/skippy-spd-head.json \
  cargo test -p skippy-runtime --features dynamic-native-runtime \
  validates_external_manifest_when_skippy_spd_manifest_is_set
```

The inspected local GLM 4.7 Flash checkpoint currently reports
`model_type = glm4_moe_lite`, architecture `Glm4MoeLiteForCausalLM`,
`num_hidden_layers = 47`, `hidden_size = 2048`, `vocab_size = 154880`, and
layer-47 auxiliary tensors `eh_proj`, `enorm`, and `hnorm`. Because 47 target
layers do not divide evenly into the old equal-stage assumptions, GLM SPD
manifests carry explicit `stage_layer_boundaries`.

### GLM Training Smoke Command

After the frontload smoke passes, run a tiny real training smoke with a
tokenizer-specific draft vocab. This is not a model-quality run; it verifies
that the reference trainer can load GLM, accept the non-uniform tap topology,
produce a real `speculation_head_final.pt`, and write a Skippy manifest.

```bash
python evals/spd/hf_train_eval_qwen06.py \
  --work-dir /tmp/skippy-spd-glm47-train-smoke \
  --model-name /path/to/GLM-4.7-Flash \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-rows 8 \
  --skip-eval \
  --num-stages 3 \
  --stage-layer-boundaries 15,31,47 \
  --num-spec-layers 1 \
  --max-length 128 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --build-draft-vocab-size 1024 \
  --draft-vocab-json '' \
  --device cuda \
  --upload-repo ''
```

`--stage-layer-boundaries` derives the reference trainer's
`--shallow_hidden_layer_indices` as `0,15,31,47;0,15,31;0,15`. You can override
that directly with `--shallow-hidden-layer-indices` when testing another tap
layout. `--build-draft-vocab-size` builds a GLM-tokenizer draft vocab from the
loaded training rows and passes the generated JSON into the reference trainer.

## Reproduce Qwen3-0.6B Training

This is the smallest useful proof that the training path and artifact shape
work. It trains a real head from open data.

```bash
python evals/spd/hf_train_eval_qwen06.py \
  --work-dir /tmp/skippy-spd-qwen06-proof \
  --model-name Qwen/Qwen3-0.6B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-rows 1024 \
  --eval-rows-per-set 8 \
  --num-stages 2 \
  --num-spec-layers 4 \
  --max-length 256 \
  --max-new-tokens 64 \
  --draft-top-k 4 \
  --device mps \
  --upload-repo ''
```

Use `--device cuda` on a GPU host. The runner also supports HF Jobs, but that is
only a convenience wrapper; the proof is ordinary Python plus open data.

Recorded local result:

| Model | Head | Eval draft top-k | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B` | locally trained, 4 spec layers | 4 | 1536 | 326 / 1536 | 0.5628 | 1.1257 | 12.67% |

This proves the training/export path, but it is not the high-gain target.

## Reproduce Qwen3.5-4B Pretrained Head Eval

This is the strongest current model-quality signal. It uses an author-published
SPD head and evaluates it locally against the reference verifier.

```bash
python evals/spd/hf_train_eval_qwen06.py \
  --work-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4 \
  --model-name Qwen/Qwen3.5-4B \
  --spec-head-repo yuyijiong/speculative_pipeline_decoding \
  --spec-head-file Qwen3.5-4B_s4_l4.pt \
  --manifest-base-model-path Qwen/Qwen3.5-4B \
  --skip-train \
  --device mps \
  --eval-rows-per-set 8 \
  --max-new-tokens 64 \
  --draft-top-k 4 \
  --upload-repo ''
```

Use `--device cuda` on a GPU host. The first run downloads the base model and
the SPD head.

Recorded local result:

| Model | Head | Eval draft top-k | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-4B` | pretrained, 4 stages / 4 spec layers | 4 | 1536 | 1230 / 1536 | 0.6176 | 2.4704 | 163.39% |

Per-dataset theoretical gains from the same run:

| Dataset | Acceptance | Equivalent accept length | Theoretical gain |
| --- | ---: | ---: | ---: |
| MT-Bench | 0.4918 | 1.9673 | 98.42% |
| HumanEval | 0.8797 | 3.5189 | 254.18% |
| GSM8K | 0.5926 | 2.3704 | 137.58% |

## Latency Simulation From Real Traces

`simulate_latency.py` consumes the raw `eval/raw/*per_sample.jsonl` file emitted
by the reference evaluator. It does not invent acceptance; it uses the real
`new_tokens`, `decode_loop_steps`, and accepted-flag counters from the run.

```bash
python evals/spd/simulate_latency.py \
  --raw /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/eval/raw/pipeline_eval__train__speculation_head_final__nt24__per_sample.jsonl \
  --stage-ms 4,4,4,4 \
  --hop-ms 0,1,5,10,25
```

Recorded Qwen3.5-4B trace with a four-stage `4ms,4ms,4ms,4ms` model:

| Hop ms | Serial split tok/s | SPD pipeline tok/s | SPD vs serial split | Paper-like gain | P50 serial ms | P50 SPD ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 62.50 | 617.61 | 9.882x | 2.470x | 1024.00 | 106.50 |
| 1 | 52.63 | 494.09 | 9.388x | 2.470x | 1216.00 | 133.12 |
| 5 | 32.26 | 274.49 | 8.509x | 2.470x | 1984.00 | 239.62 |
| 10 | 21.74 | 176.46 | 8.117x | 2.470x | 2944.00 | 372.75 |
| 25 | 10.99 | 85.19 | 7.752x | 2.470x | 5824.00 | 772.12 |

The `paper-like gain` column is based on the SPD trace alone. The `SPD vs serial
split` column models a Skippy-specific comparison where ordinary split serving
must traverse every stage/hop for each generated token before the next target
token is known.

## Export the Serving Checkpoint

After training or downloading a reference SPD head, export the PyTorch
checkpoint to a Rust-readable serving artifact:

```bash
python evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

The exporter writes `spd-head.safetensors` next to the manifest and adds an
optional `serving_checkpoint` section to `skippy-spd-head.json`. The original
`.pt` checkpoint remains referenced for provenance.

Validate an exported local head through Rust with:

```bash
SKIPPY_SPD_MANIFEST=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  cargo test -p skippy-runtime validates_external_manifest_when_skippy_spd_manifest_is_set
```

## Artifact Contract

The proof runner writes:

- `train/speculation_head_final.pt`
- `train/spd-head.safetensors` after export
- `train/skippy-spd-head.json`
- `eval/raw/*.jsonl`
- `eval/summary/*.json`

The manifest schema is `skippy-spd-head/v1`. It binds a head checkpoint to:

- base model path/id
- checkpoint format/version
- checkpoint byte size and sha256
- hidden size
- base vocab size
- draft vocab size and optional draft token ids
- number of target stages
- number of spec layers
- shallow hidden-layer tap indices
- optional safetensors serving checkpoint path, size, checksum, tensor count,
  and dtype

Rust validation lives in `crates/skippy-runtime/src/spd.rs`.

## Next Engineering Steps

1. Add a tensor loader for the SPD head weights and draft vocab mapping.
2. Implement the Qwen3.5-4B SPD forward pass in Rust for the recorded topology.
3. Capture Skippy hidden-state taps and compare Rust top-k proposals to the
   Python reference on the same taps.
4. Wire live proposal generation into `skippy-server`.
5. Verify every accepted token through the normal target stages.
6. Benchmark against ordinary split serving with both injected hop latency and a
   real multi-node topology.

## Next Research Steps

1. Train a head for a larger Qwen-family model to prove scaling beyond the
   pretrained 4B artifact.
2. Keep the draft vocab capped at 32k or 50k first.
3. Record acceptance, equivalent accept length, and latency simulation from the
   same eval prompts.
4. Only after that, evaluate custom large MoE targets. Very large MoE models
   need activation-capture support and are not the right first scaling proof.
