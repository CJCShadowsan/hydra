# SPD for Skippy Project Handoff

This branch captures the current public proof and next implementation steps for
running Speculative Pipeline Decoding (SPD) in Skippy.

It intentionally excludes private lab hosts, credentials, local IPs, and
machine-specific notes. Use it as a research/implementation handoff that another
engineer can reproduce from open models, open data, and the checked-in scripts.

## Source Paper

Paper: **Speculative Pipeline Decoding: Higher-Accuracy and Zero-Bubble
Speculation via Pipeline Parallelism**

- arXiv: `https://arxiv.org/abs/2605.30852`
- Reference code: `https://github.com/yuyijiong/speculative_pipeline_decoding`

The core idea is to combine pipeline parallel target-model execution with a
trained speculation module. The target model is partitioned into `n` pipeline
stages. While the target pipeline is processing one token per stage, the SPD
head consumes selected intermediate hidden states from the pipeline and proposes
future draft token(s). The target model still verifies the draft tokens, so with
verification enabled the output follows the base model's decoding path.

## Why This Matters for Skippy

Skippy already splits a model across staged runtimes. Ordinary split decoding is
sensitive to stage and network latency because each generated token must traverse
the full stage chain before the next target token is known.

SPD is interesting because it can fill the pipeline and amortize that stage/hop
latency across accepted speculative work. The quality question is whether the
sidecar head accepts enough tokens. The engineering question is whether Skippy
can expose the required hidden-state taps and verify proposals without breaking
target-model equivalence.

Headline current result: the pretrained `Qwen/Qwen3.5-4B` SPD head accepted
`1230 / 1536` draft flags on the local reference eval, with equivalent accept
length `2.4704` and token-weighted theoretical gain `163.39%`. Feeding that
same real trace into a four-stage Skippy latency model with `4ms` per stage
estimated `9.882x` SPD-vs-serial split throughput at `0ms` hop, `8.117x` at
`10ms` hop, and `7.752x` at `25ms` hop.

## What Works Today

### 1. Real Small-Model Training Proof

`evals/spd/hf_train_eval_qwen06.py` trains a real SPD head using the paper's
reference code.

Recorded local proof:

| Model | Training data | Head | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-0.6B` | `HuggingFaceH4/ultrachat_200k`, split `train_sft`, first 1024 rows | 4 spec layers, 2 stages | 1536 | 326 / 1536 | 0.5628 | 1.1257 | 12.67% |

This proves the train/eval/export path. It is not the high-gain target.

### 2. Strong Modest-Model Acceptance Signal

The author-published `Qwen3.5-4B_s4_l4.pt` SPD head evaluates well with the
reference verifier.

Recorded local proof:

| Model | Head | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-4B` | pretrained, 4 stages / 4 spec layers | 1536 | 1230 / 1536 | 0.6176 | 2.4704 | 163.39% |

Per-dataset theoretical gains from the same run:

| Dataset | Acceptance | Equivalent accept length | Theoretical gain |
| --- | ---: | ---: | ---: |
| MT-Bench | 0.4918 | 1.9673 | 98.42% |
| HumanEval | 0.8797 | 3.5189 | 254.18% |
| GSM8K | 0.5926 | 2.3704 | 137.58% |

This is the main reason to keep pursuing SPD for Skippy.

### 3. Trace-Based Skippy Latency Model

`evals/spd/simulate_latency.py` consumes real per-sample SPD eval traces and
models split-stage latency. It does not invent acceptance.

Recorded Qwen3.5-4B trace with four target stages at `4ms,4ms,4ms,4ms`:

| Hop ms | Serial split tok/s | SPD pipeline tok/s | SPD vs serial split | Paper-like gain | P50 serial ms | P50 SPD ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 62.50 | 617.61 | 9.882x | 2.470x | 1024.00 | 106.50 |
| 1 | 52.63 | 494.09 | 9.388x | 2.470x | 1216.00 | 133.12 |
| 5 | 32.26 | 274.49 | 8.509x | 2.470x | 1984.00 | 239.62 |
| 10 | 21.74 | 176.46 | 8.117x | 2.470x | 2944.00 | 372.75 |
| 25 | 10.99 | 85.19 | 7.752x | 2.470x | 5824.00 | 772.12 |

The paper-like gain is from the SPD trace itself. The Skippy comparison models
ordinary split serving as requiring each generated token to traverse all
stages/hops before the next target token is known.

### 4. Rust Serving Artifact Validation

`crates/skippy-runtime/src/spd.rs` adds a manifest parser and validator for SPD
head artifacts:

- schema: `skippy-spd-head/v1`
- checkpoint path, byte size, sha256
- base model path/id
- checkpoint format/version
- hidden size
- vocab size
- draft vocab size and optional draft token ids
- number of target stages
- number of spec layers
- shallow hidden-layer tap indices
- optional safetensors serving checkpoint path, size, checksum, tensor count,
  and dtype

`evals/spd/export_spd_head.py` exports the reference `.pt` checkpoint into
`spd-head.safetensors` and updates the manifest with a `serving_checkpoint`
section. Skippy can inspect the serving artifact and read tensor payloads. The
constrained Rust Qwen fixture path can execute the head against recorded
fixtures, but live Skippy hidden-state integration is not wired yet.

## What Does Not Work Yet

- Skippy does not yet expose live hidden-state taps for SPD.
- No live Skippy request has used trained SPD proposals.
- No larger-than-4B head has been trained by us yet.

## Correctness Contract

SPD should be treated as a verified speculative path.

For greedy decoding:

1. SPD proposes a token.
2. The target model computes the verified logits for that position.
3. The token is accepted only if it equals the target argmax.
4. On rejection, Skippy rolls back speculative state and emits the target token.

For sampling:

1. SPD proposes from draft distribution `q`.
2. Target distribution `p` is computed by the base model.
3. Standard speculative rejection sampling accepts with the corrected
   probability and otherwise samples from residual `max(0, p - q)`.

Do not ship unverified/lossy SPD as the default path. Lossy SPD should only be a
separate explicit experiment because wrong accepted tokens change the future
context.

## Practical Skippy Hosting Model

Treat the SPD head as a sidecar artifact attached to a Skippy stage topology.
It should not be exposed as a separate OpenAI model and should not mutate the
base GGUF/layer-package weights.

Recommended first implementation:

- one SPD sidecar runtime per active Skippy topology/session group
- host it in one Skippy process first, likely coordinator or final-stage side
- other stages expose/send selected hidden-state taps
- SPD proposes token candidates
- normal Skippy stages verify every emitted token

Distributed SPD execution across all stage nodes may become useful later, but it
is not the first proof path.

## llama.cpp / Stage Runtime Dependencies

The current proof branch does not require James's GLM/MTP work to reproduce the
Python SPD results or validate the Rust manifest. The live Skippy path will,
however, need additional staged-runtime/llama-side capability.

Likely required:

- hidden-state tap export from selected layers/stages, with token position,
  dtype, shape, and stage ownership metadata
- enough sideband transport to return those taps to the SPD sidecar without
  changing ordinary generation output
- verification support that can run proposed SPD tokens through the real target
  stages and return the target-model decision
- rollback/session-trim support for rejected speculative tokens
- ABI version bumps and Rust `skippy-ffi` mirrors for any new staged-runtime
  calls

Adjacent work that may help:

- native MTP verification work has similar concerns around speculative proposal,
  target verification, sideband data, and rollback
- GLM/MTP branches may contain useful patterns for verifier plumbing, but they
  are not SPD themselves and should not be merged wholesale just to start SPD
- package-declared draft speculation work may be useful later for advertising
  optional SPD artifacts in model/layer packages

First Skippy implementation should add the minimum SPD-specific stage-runtime
surface needed for Qwen3.5-4B parity: capture required hidden taps, run the SPD
head, and verify proposed tokens. Pull reusable verifier/rollback patterns from
MTP work only after confirming they apply cleanly to SPD.

## Artifact Layout Target

Layer packages should eventually support optional SPD artifacts:

```text
package/
  manifest.json
  parts/
    ...
  spd/
    skippy-spd-head.json
    spd-head.safetensors
    draft-vocab.json
```

The manifest must bind the head to:

- base model digest/path
- tokenizer/vocab identity
- hidden size
- split topology
- stage count
- layer taps
- draft vocab
- head tensor checksum

## Reproduction Commands

### Train Small Qwen3-0.6B Head

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

Use `--device cuda` on a CUDA host.

### Evaluate Pretrained Qwen3.5-4B Head

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

### Export Serving Checkpoint

```bash
python evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

### Export Rust/Python Parity Fixture

```bash
python evals/spd/export_parity_fixture.py \
  --reference-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/speculative_pipeline_decoding \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --base-model-path Qwen/Qwen3.5-4B \
  --out /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --device mps \
  --top-k 8
```

### Simulate Split Latency From Trace

```bash
python evals/spd/simulate_latency.py \
  --raw /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/eval/raw/pipeline_eval__train__speculation_head_final__nt24__per_sample.jsonl \
  --stage-ms 4,4,4,4 \
  --hop-ms 0,1,5,10,25
```

## Engineering Next Steps

### Milestone 1: Tensor Export

Goal: produce a Rust-serving artifact from the `.pt` checkpoint.

Tasks:

1. Add/export a `safetensors` writer for the SPD checkpoint. Done in
   `evals/spd/export_spd_head.py`.
2. Preserve tensor names, shapes, dtype, draft vocab ids, and config. Done via
   the safetensors file and `skippy-spd-head.json`.
3. Extend `skippy-spd-head.json` to reference the serving checkpoint. Done with
   the optional `serving_checkpoint` section.
4. Add a small shape/checksum inspection command or test fixture. Done in
   `skippy-runtime` tests.
5. Add a minimal safetensors payload reader for BF16/F32/I64 tensors. Done in
   `crates/skippy-runtime/src/spd/safetensors.rs`.

Exit criteria:

- Qwen3.5-4B SPD head exports deterministically.
- Rust can validate the manifest, enumerate expected tensors, and read selected
  tensor payloads from the serving checkpoint or parity fixture.
- To validate a local exported head through Rust, set
  `SKIPPY_SPD_MANIFEST=/tmp/.../train/skippy-spd-head.json` and run:

```bash
cargo test -p skippy-runtime validates_external_manifest_when_skippy_spd_manifest_is_set
```

### Milestone 2: Rust Forward Pass Parity

Goal: Rust computes the same draft candidates as Python for recorded inputs.

Tasks:

1. Record hidden-state tap fixtures from Python reference execution. Fixture
   export is implemented in `evals/spd/export_parity_fixture.py`.
2. Implement the Qwen3.5-4B SPD head forward pass in Rust. Done for the
   recorded fixture path in `crates/skippy-runtime/src/spd/qwen.rs`.
3. Compare Rust top-k draft candidates against Python top-k on the same hidden
   states. Done for the pretrained Qwen3.5-4B fixture.
4. Add focused tests with small fixture tensors and opt-in real-artifact tests.
   Done in `skippy-runtime` tests.

Exit criteria:

- Rust top-k proposals match Python within tolerance on recorded fixtures.
- No Skippy serving integration is required for this milestone.
- Recorded real-artifact parity:
  - Rust/Python draft indices: `[135, 23, 17, 21, 16, 22, 24, 2598]`
  - Full token ids: `[220, 23, 17, 21, 16, 22, 24, 2972]`
  - Spec-query max absolute diff: `0.03125`
  - Final-hidden max absolute diff: `0.0625`

```bash
SKIPPY_SPD_MANIFEST=/tmp/.../train/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/tmp/.../train/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set
```

### Milestone 3: Skippy Hidden-State Taps

Goal: Skippy can expose the hidden states the SPD head needs.

Tasks:

1. Identify the target layer taps from `skippy-spd-head.json`.
2. Add a hidden-state sideband/tap path in the staged runtime.
3. Validate dtype, shape, token position, and stage ownership.
4. Write a correctness test that compares tapped hidden states against a known
   reference for a small prompt.

Exit criteria:

- Skippy can capture the required taps for a live prompt without changing
  normal generation output.

### Milestone 4: Live Verified SPD in Skippy

Goal: Skippy uses SPD proposals during generation and verifies every token.

Tasks:

1. Wire SPD proposal generation into `skippy-server`.
2. Feed proposals into the existing target verification path.
3. Roll back speculative KV/session state on rejection.
4. Emit metrics for proposals, accepted tokens, rejected tokens, equivalent
   accept length, and decode-loop steps.
5. Run ordinary split serving and SPD serving against the same prompts.

Exit criteria:

- Greedy outputs match ordinary target-model decoding.
- Acceptance and equivalent accept length are non-zero and close to reference
  trace behavior.
- Latency improves under injected hop/stage delay.

### Milestone 5: Larger-Model Training Proof

Goal: prove the SPD head generation pipeline scales beyond the pretrained 4B
artifact.

Recommended first target:

- a larger Qwen-family model with architecture/tokenizer support close to the
  reference implementation
- avoid custom huge MoE targets for the first scaling proof

Tasks:

1. Train with open conversation data mix.
2. Keep draft vocab capped at 32k or 50k.
3. Evaluate on the same MT-Bench/HumanEval/GSM8K prompt sets.
4. Record acceptance, equivalent accept length, and latency simulation.
5. Publish only artifact manifests, scripts, and aggregate metrics unless model
   licensing allows the trained head to be shared.

Exit criteria:

- Larger-model head reaches useful equivalent accept length.
- Training process and artifact production are reproducible by another engineer.

## Branch Scope

This branch should stay focused on SPD proof and handoff material:

- `SPD_SKIPPY_PROJECT.md`
- `evals/spd/`
- `crates/skippy-runtime/src/spd.rs`
- minimal module export from `skippy-runtime`

Avoid mixing in unrelated MTP, GLM, packaging, branch-reconciliation, or private
lab automation work. Those can be inputs later, but the purpose of this branch
is to make the SPD path clear and reproducible.

## Validation

Run:

```bash
python3 -m py_compile evals/spd/hf_train_eval_qwen06.py evals/spd/simulate_latency.py evals/spd/export_spd_head.py evals/spd/export_parity_fixture.py
cargo fmt --all -- --check
cargo test -p skippy-runtime spd
SKIPPY_SPD_MANIFEST=/tmp/.../train/skippy-spd-head.json SKIPPY_SPD_PARITY_FIXTURE=/tmp/.../train/spd-parity-fixture.safetensors cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set
cargo clippy -p skippy-runtime --all-targets -- -D warnings
```

Before publishing or handing off, run the repo's normal secret scan and also
check the diff for private hostnames, private IPs, access tokens, credentials,
and absolute developer-machine paths.
