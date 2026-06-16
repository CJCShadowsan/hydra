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

Headline current result: the pretrained `Qwen/Qwen3.5-4B` SPD head reported
`1230 / 1536` accepted draft flags on the local reference eval. The reference
summary reports aggregate acceptance `0.6176`, equivalent accept length
`2.4704`, and token-weighted theoretical gain `163.39%`; the branch latency
simulator reports the same equivalent accept length as a paper-like `2.470x`
ratio (`+147.04%`) under its aggregate-cycle formula. Feeding that same real
trace into a four-stage Skippy latency model with `4ms` per stage estimated
`9.882x` SPD-vs-serial split throughput at `0ms` hop, `8.117x` at `10ms` hop,
and `7.752x` at `25ms` hop. The Rust live-tap harness now also proves three
consecutive Qwen3.5-4B top-1 SPD proposals from live Skippy activation frames:
all three were accepted by the target verifier, every verifier window rewound,
and the committed token stream matched ordinary non-SPD greedy decoding. The
Skippy OpenAI request path can run the pretrained head from inline returned
taps without local replay fallback. In the current release smoke that probe was
ready and ran in about `400ms`, but it ran after the normal target token
returned and the proposal was rejected. That is a real request-path proof, not
yet a live serving speedup.

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

The accepted-flags count and aggregate acceptance use different denominators in
the reference output. `1230 / 1536` is the draft-flag count; `0.6176` is the
reference aggregate acceptance metric used for equivalent accept length.

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
constrained Rust Qwen fixture path can reconstruct SPD input rows from recorded
hidden-state taps and execute the head against recorded fixtures. The live-tap
bench path can also assemble those rows from real Skippy activation frames for
the tap-aligned Qwen3.5-4B proof split.

`skippy-runtime::spd::SpdQwen3Head` is the current Rust hosting boundary for
the pretrained Qwen head. It opens the manifest and serving checkpoint once,
validates the tensor shapes, caches the serving weights, keeps the runtime
shape, and exposes repeated `forward()` calls for proposal generation. The live
verifier harness and request-path probe now use that loaded head instead of
reopening the artifact on every proposal. This is the shape a Skippy SPD
sidecar should use, but the current Qwen forward remains a straightforward Rust
reference path, expands BF16 weights to `f32`, and is not yet optimized for
request-path throughput.

### 5. Rust Hidden-Tap Planning

`crates/skippy-runtime/src/spd/tap_plan.rs` translates the manifest's hidden
state requirements into concrete Skippy stage ownership. The reference
convention is:

- HF hidden-state index `0` is the embedding output
- HF hidden-state index `k >= 1` is the output after decoder layer `k - 1`

For the pretrained `Qwen/Qwen3.5-4B` S4/L4 head, the required tap groups are:

```text
g4: [0, 10, 20, 31]
g3: [0, 8, 16, 24]
g2: [0, 8, 16]
g1: [0, 8]
```

The new Rust planner confirms:

| Candidate Skippy layer ranges | Result |
| --- | --- |
| `0..8, 8..16, 16..24, 24..32` | ordinary four-way split exposes boundary indices `8,16,24,32`; SPD still needs internal taps `10,20,31` |
| `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32` | tap-aligned proof split can expose every required tap as a stage boundary |

This gives two implementation options: add a narrow internal hidden-tap ABI for
normal four-stage serving, or use a tap-aligned over-split as the fastest local
proof that the pretrained head can drive live Skippy proposals.

`skippy-model-package` now supports explicit split boundaries for this proof:

```bash
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir .artifacts/spd/qwen35-4b-gguf/
skippy-model-package plan model.gguf --splits 8,10,16,20,24,31
skippy-model-package write-stages model.gguf \
  --splits 8,10,16,20,24,31 \
  --out-dir /tmp/qwen35-spd-tap-slices/
skippy-model-package validate model.gguf /tmp/qwen35-spd-tap-slices/stage-*.gguf
skippy-model-package preflight model-package/ --splits 8,10,16,20,24,31
```

For a 32-layer Qwen3.5 model, those boundaries materialize the ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`. This does not yet make
SPD live in Skippy by itself; it removes the artifact-generation blocker for a
tap-aligned local proof that can use normal stage-boundary activation frames
before adding a production hidden-tap ABI.

Recorded local artifact result with
`unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf`:

- source size: `2.6G`
- source sha256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
- plan: `32` layers, `7` tap-aligned stages
- validation: all `426` owned tensors present exactly once across the seven
  slices; no missing or duplicate owned tensors

The local live-chain smoke now runs the same tap-aligned shape through real
Skippy binary stage transport by selecting the CPU backend explicitly:

```bash
cargo run -p skippy-bench -- local-split-chain-binary \
  --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --ctx-size 128 \
  --n-gpu-layers 0 \
  --selected-backend-device CPU0 \
  --stage-bind-base-port 19131 \
  --prompt Hello
```

Recorded result: all seven ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32` forwarded over the
binary chain, activation width `2560`, first boundary payload `10240` bytes /
`5120` f16 wire bytes, prompt token id `9419`, predicted token `11`. A
`local-split-compare` run against the same GGUF and prompt matched the unsplit
full-model predicted token `11`.

The Rust SPD fixture path now validates the missing bridge between Skippy taps
and the pretrained head input. `evals/spd/export_parity_fixture.py` records raw
tap-row tensors (`tap_row_*_concat` plus `tap_row_*_hf_indices`) before Python
applies `g0_proj` or `stage_projs.*`; `skippy-runtime` reconstructs the same
`cur_in` from those rows. Recorded real-artifact command:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --top-k 8
```

Recorded result for the regenerated `Hello` fixture:

- tap input reconstruction max abs diff: `7.62939453125e-6`
- Rust/Python draft indices:
  `[7728,15014,38999,10036,11235,13293,15953,0]`
- full token ids:
  `[9419,21251,109266,12675,14556,18103,23066,0]`
- spec-query max abs diff: `0.03125`
- final-hidden max abs diff: `0.125`

The branch now also proves the trained head can consume live Skippy-produced
tap frames for the same Qwen3.5-4B fixture. The proof command drives the
fixture prompt token ids through tap-aligned runtime slices, adds an
embedding-only side tap for HF hidden-state index `0`, assembles `cur_in` from
real activation frames, runs the Rust Qwen SPD head, and verifies the live
top-1 proposal with Skippy's target verifier:

```bash
cargo run -p skippy-bench -- spd-live-tap-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --ctx-size 128 \
  --n-gpu-layers 0 \
  --selected-backend-device CPU0 \
  --top-k 8 \
  --verify-steps 3
```

Recorded local live-tap result against `Qwen3.5-4B-Q4_K_M.gguf`:

- live taps captured: `0,8,10,16,20,24,31`
- each tap frame: `13` tokens, `133120` bytes, hidden width `2560`
- live `cur_in` max abs diff vs HF fixture: `0.3134765625`
- g0 row max abs diff vs HF fixture: `0.00103759765625`
- live Skippy top-1 token id: `9419`
- fixture Python/Rust top-1 token id: `9419`
- live top-8 token ids:
  `[9419,21251,109266,14556,23066,18103,12675,0]`
- fixture top-8 token ids:
  `[9419,21251,109266,12675,14556,18103,23066,0]`
- target verifier input token: `271`
- target verifier predicted token: `9419`
- accepted live SPD top-1 proposal: `true`
- verifier checkpoint restored to token count `12`
- ordinary non-SPD greedy token: `9419`
- verified committed output matches ordinary non-SPD greedy output: `true`

Recorded repeated verifier run with `--verify-steps 3`:

- generated committed tokens: `[9419,0,2500]`
- accepted live SPD top-1 proposals: `3 / 3`
- rejected proposals: `0 / 3`
- top-1 acceptance rate for this diagnostic prompt: `1.0`
- every target verifier window rewound to the pre-verify token count: `true`
- every committed token matched ordinary non-SPD greedy decoding: `true`

The top-8 set is the same, with lower-ranked candidates reordered by GGUF
quantization/runtime drift. This is a real Skippy tap/head/target-verifier
proof over repeated proposal windows, but it is still a diagnostic harness and
not a serving throughput measurement. `skippy-server` now has an experimental
request-path SPD source that can load the same pretrained head. By default it
uses inline Skippy taps when they are complete and otherwise returns no
proposal. Passing `--openai-spd-replay-fallback` enables the older slow local
tap replay mode for correctness proofs.

Recorded local request-path smoke with `skippy-server serve-binary` and
`--openai-spd-replay-fallback`:

- topology: seven tap-aligned local CPU stages,
  `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`
- model: `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`
- prompt: Humaneval eval row `index=8`, `max_tokens=4`, `temperature=0`
- SPD source: `spd-replay`
- SPD proposals: `4`
- accepted proposals: `2`
- rejected proposals: `2`
- accepted proposal windows: `2 / 4`
- emitted text: `<think>\nThe user wants me`
- no-SPD baseline emitted the same text for the same request
- SPD replay wall time: about `101.5s`; logged decode time: about `100.6s`
- no-SPD Skippy baseline wall time: about `1.28s`; logged decode time: about
  `90ms`

This proves the live OpenAI serving path can call a real pretrained SPD head,
feed proposals into Skippy's verifier, accept/reject per token, and preserve
ordinary greedy output. It is still not a serving throughput measurement:
`spd-replay` recomputes taps through local `StageModel` slices and runs the head
on CPU for each proposal when replay fallback is enabled. Use the trace latency
simulator for current speedup estimates until proposal scheduling consumes
inline taps without replay.

Inline tap transport now works for the tap-aligned local proof topology. During
embedded stage-0 serving, Skippy records stage-0 boundary activation frames into
an SPD-positioned tap cache keyed by hidden-state index and token position.
Downstream binary stages can return SPD tap frames over the direct-return side
channel when stage 0 marks an SPD request. A one-token Qwen3.5-4B smoke on
seven local CPU stages returned and recorded required hidden-state rows for
`10`, `20`, and `31` with no tap-return failures:

- response content: `<think>\nThinking`
- wall time: `23.646s`
- stage-0 local tap records: `hf_index=8`, rows `17` and `1`, `required=false`
- downstream required tap records:
  - `hf_index=10`, producer stage `1`, rows `17` and `1`, `required=true`
  - `hf_index=20`, producer stage `3`, rows `17` and `1`, `required=true`
  - `hf_index=31`, producer stage `5`, rows `17` and `1`, `required=true`
- downstream non-required tap records: `16` and `24`, rows `17` and `1`

Recorded local release request-path smoke without replay fallback:

- topology: seven tap-aligned local CPU stages,
  `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`
- binary: `target/release/skippy-server`
- prompt: `Write a Python function named add that returns the sum of two integers.`
- response content for `max_tokens=1`: `<think>\nThinking`
- SPD source: `spd-replay`, with `--openai-spd-replay-fallback` disabled
- inline tap state: required downstream taps `10`, `20`, and `31` returned and
  recorded for prompt rows and the current-token row; no tap-return failures
- h0 source: direct GGUF `token_embd.weight` rows
- SPD head hosting: cached pretrained Qwen3.5-4B serving weights
- inline probe elapsed: about `400ms`
- inline probe result: ready, proposed token `8160`, target token `90700`,
  accepted `false`
- SPD request wall time: about `1.23s`
- no-SPD same-topology wall time for the same one-token request: about `0.46s`

This proves the real pretrained head can run in the Skippy OpenAI request path
from inline Skippy taps plus direct GGUF h0 embeddings, without replaying local
stage slices. It is not a speedup yet: the probe currently runs after the target
token has already traversed the full pipeline, and the first sampled proposal
in this smoke was rejected.

This still does not make `spd-replay` a full speed path. Proposal generation at
the top of the decode loop starts before the in-flight current-token target pass
has produced the taps that the next proposal needs, so the default inline source
returns no proposal there. When all required non-h0 rows are present, it skips
local downstream replay and runs only the embedding-only h0 tap before the SPD
head. The normal decode path now has the right place to probe the head after the
target pass returns the current-token taps; the next serving milestone is using
that probe to drive ahead-of-final-stage verification and then measuring
ordinary split serving against inline-tap SPD serving.

## What Does Not Work Yet

- The `spd-replay` request path has a correctness fallback, not a final speed
  path. `--openai-spd-replay-fallback` replays taps through local stage slices
  before feeding proposals into the existing verify/repair/rollback loop.
- Inline hidden-tap capture and direct-return transport work for the local
  tap-aligned proof. Proposal scheduling still needs to turn freshly returned
  current-token taps into useful ahead-of-final-stage work.
- The current no-replay request-path probe is post-target-decode telemetry. It
  proves readiness and measures head cost, but it adds latency instead of
  hiding pipeline bubbles.
- Request-path acceptance has been proven on a bounded four-token smoke, but a
  larger local request-path acceptance/latency sweep is still needed.
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
- SPD implements the `skippy-server` speculative proposal-source boundary
- normal Skippy stages verify every emitted token through the existing
  verify/repair/rollback path

Current experimental serving hook:

```bash
skippy-server serve-binary \
  --config stage0.json \
  --topology topology.json \
  --activation-width 2560 \
  --openai-bind-addr 127.0.0.1:9337 \
  --openai-spd-manifest /path/to/skippy-spd-head.json \
  --openai-spd-fixture /path/to/spd-parity-fixture.safetensors \
  --openai-spd-model-path /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --openai-speculative-window 1
```

The default source uses inline tap cache rows only. Add
`--openai-spd-replay-fallback` to run the older slow local `StageModel` replay
path for request-path correctness proofing; do not use that fallback as the
final performance architecture.

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
python3 evals/spd/hf_train_eval_qwen06.py \
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
python3 evals/spd/hf_train_eval_qwen06.py \
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
python3 evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

### Export Rust/Python Parity Fixture

```bash
python3 evals/spd/export_parity_fixture.py \
  --reference-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/speculative_pipeline_decoding \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --base-model-path Qwen/Qwen3.5-4B \
  --out /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --device mps \
  --top-k 8
```

The fixture includes raw hidden-state tap rows used to build SPD `cur_in`.
Use `skippy-bench spd-fixture-parity` for one JSON report covering tap-row
projection parity and Qwen SPD forward/top-k parity:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --top-k 8
```

### Simulate Split Latency From Trace

```bash
python3 evals/spd/simulate_latency.py \
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
   It now records raw tap-row tensors before Python applies `g0_proj` or
   `stage_projs.*`.
2. Implement the Qwen3.5-4B SPD head forward pass in Rust. Done for the
   recorded fixture path in `crates/skippy-runtime/src/spd/qwen.rs`.
3. Compare Rust top-k draft candidates against Python top-k on the same hidden
   states. Done for the pretrained Qwen3.5-4B fixture.
4. Add focused tests with small fixture tensors and opt-in real-artifact tests.
   Done in `skippy-runtime` tests.
5. Reconstruct SPD `cur_in` from raw tap rows in Rust before running the head.
   Done in `crates/skippy-runtime/src/spd/tap_input.rs` and exposed through
   `skippy-bench spd-fixture-parity`.

Exit criteria:

- Rust top-k proposals match Python within tolerance on recorded fixtures.
- No Skippy serving integration is required for this milestone.
- Recorded real-artifact parity:
  - Tap input reconstruction max absolute diff: `7.62939453125e-6`
  - Rust/Python draft indices:
    `[7728, 15014, 38999, 10036, 11235, 13293, 15953, 0]`
  - Full token ids:
    `[9419, 21251, 109266, 12675, 14556, 18103, 23066, 0]`
  - Spec-query max absolute diff: `0.03125`
  - Final-hidden max absolute diff: `0.125`

```bash
SKIPPY_SPD_MANIFEST=/tmp/.../train/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/tmp/.../train/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set
```

Or run the single bench report:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/.../train/skippy-spd-head.json \
  --fixture /tmp/.../train/spd-parity-fixture.safetensors \
  --top-k 8
```

### Milestone 3: Skippy Hidden-State Taps

Goal: Skippy can expose the hidden states the SPD head needs.

Tasks:

1. Identify the target layer taps from `skippy-spd-head.json`. Done in
   `skippy-runtime` tap-planning tests.
2. Decide proof topology:
   - fastest proof: tap-aligned over-split so required taps are ordinary stage
     boundaries. Done for the local Qwen3.5 seven-stage binary chain.
   - production path: add an internal hidden-tap ABI so ordinary four-stage
     serving can expose taps `10,20,31`
3. Add a hidden-state sideband/tap path in the staged runtime if not using the
   over-split proof path.
4. Validate dtype, shape, token position, and stage ownership.
5. Write a correctness test that compares tapped hidden states against a known
   reference for a small prompt.

Exit criteria:

- Skippy can capture the required taps for a live prompt without changing
  normal generation output.

### Milestone 4: Live Verified SPD in Skippy

Goal: Skippy uses SPD proposals during generation and verifies every token.

Tasks:

1. Add a proposal-source boundary in `skippy-server`. Done for the current
   draft-model path and reused by SPD.
2. Wire SPD proposal generation into that boundary. Experimental replay-tap
   source is in place behind `--openai-spd-manifest` / `--openai-spd-fixture`.
3. Feed proposals into the existing target verification path. Done by sharing
   the same `VerifySpan` verifier/repair loop as the draft-model source.
4. Roll back speculative KV/session state on rejection. Done through the
   existing verifier reset path for any rejecting proposal source.
5. Emit metrics for proposals, accepted tokens, rejected tokens, equivalent
   accept length, and decode-loop steps. Basic request-path proposal/accept
   telemetry is in place; equivalent accept length and sweep reporting still
   need promotion into the benchmark/report layer.
6. Run ordinary split serving and SPD serving against the same prompts. Done
   for one bounded local smoke; still needed as a broader sweep.
7. Replace replayed local tap collection with inline hidden-tap capture and
   transport so performance can match the SPD pipeline design. Stage-0
   positioned tap cache/overlay and downstream direct-return tap transport are
   in place for the tap-aligned local proof; in-flight proposal scheduling
   remains.

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
- `crates/skippy-runtime/src/spd/`
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
cargo run -p skippy-bench -- spd-fixture-parity --manifest /tmp/.../train/skippy-spd-head.json --fixture /tmp/.../train/spd-parity-fixture.safetensors --top-k 8
cargo clippy -p skippy-runtime --all-targets -- -D warnings
```

Before publishing or handing off, run the repo's normal secret scan and also
check the diff for private hostnames, private IPs, access tokens, credentials,
and absolute developer-machine paths.
