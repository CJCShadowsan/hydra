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
  binding, including a Rust-readable safetensors serving checkpoint and
  selected tensor payload reads.
- `skippy-runtime` can run the pretrained `Qwen/Qwen3.5-4B` SPD head over a
  recorded Python fixture and match Python top-k draft candidates.

## What Does Not Work Yet

- Skippy does not yet run the SPD head from live staged hidden-state taps.
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

## Export a Rust/Python Parity Fixture

Rust top-k parity uses the same trained head and the same real hidden-state
inputs as Python. Export a fixture with:

```bash
python evals/spd/export_parity_fixture.py \
  --reference-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/speculative_pipeline_decoding \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --base-model-path Qwen/Qwen3.5-4B \
  --out /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --device mps \
  --top-k 8
```

This writes real SPD inference rows, position ids, base final-norm weight,
Python intermediate states, Python logits, Python top-k draft indices, and
Python top-k full token ids. Validate the fixture container through Rust with:

```bash
SKIPPY_SPD_PARITY_FIXTURE=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  cargo test -p skippy-runtime validates_external_parity_fixture_when_skippy_spd_parity_fixture_is_set
```

Validate the real Rust/Python top-k parity path in release mode:

```bash
SKIPPY_SPD_MANIFEST=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set
```

Recorded parity result: Rust matched Python top-k draft indices
`[135, 23, 17, 21, 16, 22, 24, 2598]`, which map to full token ids
`[220, 23, 17, 21, 16, 22, 24, 2972]`.

## Artifact Contract

The proof runner writes:

- `train/speculation_head_final.pt`
- `train/spd-head.safetensors` after export
- `train/spd-parity-fixture.safetensors` after fixture export
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
Safetensors parsing and BF16/F32/I64 payload reads live in
`crates/skippy-runtime/src/spd/safetensors.rs`.
The constrained Qwen fixture forward path lives in
`crates/skippy-runtime/src/spd/qwen.rs`.

## Next Engineering Steps

1. Capture Skippy hidden-state taps and compare Rust top-k proposals to the
   Python reference on the same taps.
2. Wire live proposal generation into `skippy-server`.
3. Verify every accepted token through the normal target stages.
4. Benchmark against ordinary split serving with both injected hop latency and a
   real multi-node topology.

## Next Research Steps

1. Train a head for a larger Qwen-family model to prove scaling beyond the
   pretrained 4B artifact.
2. Keep the draft vocab capped at 32k or 50k first.
3. Record acceptance, equivalent accept length, and latency simulation from the
   same eval prompts.
4. Only after that, evaluate custom large MoE targets. Very large MoE models
   need activation-capture support and are not the right first scaling proof.
