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
- `skippy-runtime` exposes `SpdQwen3Head`, a reusable loaded-head boundary for
  repeated Qwen SPD proposals without reopening the manifest/checkpoint each
  time.
- `skippy-runtime` can reconstruct the SPD `cur_in` rows from raw recorded
  hidden-state tap inputs using `g0_proj` and `stage_projs.*`.
- `skippy-model-package` can plan, write, and preflight explicit tap-aligned
  layer splits for the `Qwen/Qwen3.5-4B` S4/L4 proof head.
- `skippy-bench local-split-chain-binary` can run the `Qwen/Qwen3.5-4B` GGUF
  through the full tap-aligned seven-stage Skippy binary chain locally, using
  `CPU0` to bypass local Metal auto-selection.
- `skippy-bench spd-live-tap-parity` can assemble the pretrained Qwen3.5-4B
  SPD head input from live Skippy activation frames, including an
  embedding-only side tap for hidden-state index `0`, run the Rust SPD head
  from those live taps, and verify repeated live top-1 proposals with the
  Skippy target verifier.
- `skippy-server` has a request-path speculative proposal-source boundary in
  front of the existing target verify/repair/rollback loop. The current draft
  model path uses it, and an experimental `spd-replay` source can load the
  pretrained Qwen3.5-4B head from `--openai-spd-manifest` /
  `--openai-spd-fixture`.
- A bounded local OpenAI request through `skippy-server` has exercised the
  pretrained head in the live serving path: four `spd-replay` proposals, two
  accepted, two rejected, and the same greedy text as the no-SPD baseline.

## What Does Not Work Yet

- The `spd-replay` source collects taps by replaying the current context through
  local `StageModel` slices. It is a correctness bridge into live serving, not
  the optimized inline hidden-tap transport needed for real speed.
- The live request-path proof is bounded; it still needs a larger acceptance
  and latency sweep.
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

Use `--device cuda` on a GPU host. The first run downloads the base model and
the SPD head.

Recorded local result:

| Model | Head | Eval draft top-k | Generated tokens | Accepted flags | Acceptance | Equivalent accept length | Theoretical gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-4B` | pretrained, 4 stages / 4 spec layers | 4 | 1536 | 1230 / 1536 | 0.6176 | 2.4704 | 163.39% |

The accepted-flags count and aggregate acceptance use different denominators in
the reference output. `1230 / 1536` is the draft-flag count; `0.6176` is the
reference aggregate acceptance metric used for equivalent accept length.

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
python3 evals/spd/simulate_latency.py \
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

The simulator's aggregate-cycle formula reports the same equivalent accept
length as `2.470x` (`+147.04%`). The reference eval summary separately reports
a token-weighted theoretical gain of `163.39%`.

## Export the Serving Checkpoint

After training or downloading a reference SPD head, export the PyTorch
checkpoint to a Rust-readable serving artifact:

```bash
python3 evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

The exporter writes `spd-head.safetensors` next to the manifest and adds an
optional `serving_checkpoint` section to `skippy-spd-head.json`. The original
`.pt` checkpoint remains referenced for provenance.

For the pretrained `Qwen/Qwen3.5-4B` S4/L4 head, the tap-aligned Skippy proof
split is:

```bash
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir .artifacts/spd/qwen35-4b-gguf/
skippy-model-package plan .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --splits 8,10,16,20,24,31
skippy-model-package write-stages .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --splits 8,10,16,20,24,31 \
  --out-dir /tmp/qwen35-spd-tap-slices/
skippy-model-package validate .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  /tmp/qwen35-spd-tap-slices/stage-*.gguf
```

Those split boundaries produce ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`, exposing every hidden
state required by the pretrained head as a stage boundary for the local proof.
The recorded local artifact validation used `Qwen3.5-4B-Q4_K_M.gguf` and found
all `426` owned tensors exactly once across the seven slices.

The same split shape has also been exercised through live Skippy binary stage
transport against the full GGUF:

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

Recorded result: stage ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`, activation width
`2560`, first boundary payload `10240` bytes / `5120` f16 wire bytes, prompt
token id `9419`, predicted token `11`. A `local-split-compare` run on the same
GGUF and prompt matched the unsplit full-model token `11`.

Validate an exported local head through Rust with:

```bash
SKIPPY_SPD_MANIFEST=/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  cargo test -p skippy-runtime validates_external_manifest_when_skippy_spd_manifest_is_set
```

## Export a Rust/Python Parity Fixture

Rust top-k parity uses the same trained head and the same real hidden-state
inputs as Python. Export a fixture with:

```bash
python3 evals/spd/export_parity_fixture.py \
  --reference-dir /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/speculative_pipeline_decoding \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --base-model-path Qwen/Qwen3.5-4B \
  --out /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --device mps \
  --top-k 8
```

This writes real SPD inference rows, raw hidden-state tap rows, position ids,
base final-norm weight, Python intermediate states, Python logits, Python top-k
draft indices, and Python top-k full token ids. Validate the fixture container
through Rust with:

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

Or run the combined bench report:

```bash
cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --fixture /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/spd-parity-fixture.safetensors \
  --top-k 8
```

Recorded parity result from the regenerated `Hello` fixture:

- tap input reconstruction max absolute diff: `7.62939453125e-6`
- Rust matched Python top-k draft indices
  `[7728, 15014, 38999, 10036, 11235, 13293, 15953, 0]`
- full token ids
  `[9419, 21251, 109266, 12675, 14556, 18103, 23066, 0]`
- spec-query max absolute diff: `0.03125`
- final-hidden max absolute diff: `0.125`

## Run the Live Skippy Tap Proof

After exporting the parity fixture and building the patched native Skippy ABI,
run the pretrained head from real Skippy activation frames:

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

Recorded local result:

- live taps captured: `0,8,10,16,20,24,31`
- each tap frame: `13` tokens, `133120` bytes, hidden width `2560`
- live `cur_in` max absolute diff vs HF fixture: `0.3134765625`
- g0 row max absolute diff vs HF fixture: `0.00103759765625`
- live Skippy top-1 token id: `9419`
- fixture Python/Rust top-1 token id: `9419`
- live top-8 token ids:
  `[9419, 21251, 109266, 14556, 23066, 18103, 12675, 0]`
- fixture top-8 token ids:
  `[9419, 21251, 109266, 12675, 14556, 18103, 23066, 0]`
- target verifier input token: `271`
- target verifier predicted token: `9419`
- accepted live SPD top-1 proposal: `true`
- verifier checkpoint restored to token count `12`
- ordinary non-SPD greedy token: `9419`
- verified committed output matches ordinary non-SPD greedy output: `true`

Recorded repeated verifier run with `--verify-steps 3`:

- generated committed tokens: `[9419, 0, 2500]`
- accepted live SPD top-1 proposals: `3 / 3`
- rejected proposals: `0 / 3`
- top-1 acceptance rate for this diagnostic prompt: `1.0`
- every target verifier window rewound to the pre-verify token count: `true`
- every committed token matched ordinary non-SPD greedy decoding: `true`

The live proof uses the Q4_K_M GGUF, while the fixture was exported from the HF
BF16 model. The deeper-row drift is therefore expected; the current result says
the Skippy tap/head plumbing works and the best proposal survives quantization
for this prompt. It also proves repeated real target-verifier acceptance
windows in a diagnostic harness, but does not yet measure request-path SPD
serving throughput.

`skippy-server serve-binary --openai-bind-addr` now has an experimental
request-path source for the same head:

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

That path feeds real SPD proposals into the normal Skippy `VerifySpan`
verify/repair/rollback loop.

Recorded bounded local OpenAI request-path proof:

- topology: seven tap-aligned local CPU stages,
  `0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`
- model: `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`
- prompt: Humaneval eval row `index=8`, `max_tokens=4`, `temperature=0`
- SPD source: `spd-replay`
- SPD proposals: `4`
- accepted proposals: `2`
- rejected proposals: `2`
- emitted text: `<think>\nThe user wants me`
- no-SPD baseline emitted the same text
- SPD replay wall time: about `101.5s`; no-SPD baseline wall time: about
  `1.28s`

That request-path result proves correct integration with target verification,
not speed. The next engineering step is to schedule proposal generation around
freshly returned inline taps and then run ordinary split serving and SPD serving
against a larger shared prompt set with injected and real hop latency.

Current inline-tap progress: embedded stage-0 serving records stage-0 boundary
activation rows into an SPD-positioned tap cache, downstream binary stages can
return tap frames over the direct-return side channel for SPD-marked requests,
and `spd-replay` overlays complete cached boundary frames before falling back to
local replay. A one-token Qwen3.5-4B smoke on seven local CPU stages returned
the required `10`, `20`, and `31` rows with no tap-return failures. The
proposal source now skips local downstream replay when all required non-h0 rows
are present and runs only the embedding-only h0 tap. The remaining production
work is scheduling proposal generation after the in-flight current-token taps
exist, then measuring ordinary split serving against inline-tap SPD serving.

## Validate Hidden Tap Compatibility

`skippy-runtime` includes a Rust tap planner that converts the manifest's
hidden-state requirements into concrete Skippy stage ownership. The reference
index convention is `0 = embedding output`; `k >= 1` means output after decoder
layer `k - 1`.

For the pretrained `Qwen/Qwen3.5-4B` S4/L4 head, required tap groups are:

```text
g4: [0, 10, 20, 31]
g3: [0, 8, 16, 24]
g2: [0, 8, 16]
g1: [0, 8]
```

The checked-in tests show that a normal four-way split `0..8, 8..16, 16..24,
24..32` still needs internal taps `10,20,31`. A tap-aligned proof split
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32` can expose every required
tap as an ordinary stage boundary.

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
The tap-row-to-`cur_in` projection bridge lives in
`crates/skippy-runtime/src/spd/tap_input.rs`.

## Next Engineering Steps

1. Capture Skippy hidden-state taps and compare Rust top-k proposals to the
   Python reference on the same taps.
2. For the first live proof, prefer the tap-aligned over-split unless the
   hidden-tap ABI is already available.
3. Replace `spd-replay` with inline tap capture in `skippy-server`. Stage-0
   positioned tap caching and downstream direct-return tap transport exist for
   the tap-aligned local proof; in-flight proposal scheduling remains.
4. Verify every accepted token through the normal target stages.
5. Benchmark against ordinary split serving with both injected hop latency and a
   real multi-node topology.

## Next Research Steps

1. Train a head for a larger Qwen-family model to prove scaling beyond the
   pretrained 4B artifact.
2. Keep the draft vocab capped at 32k or 50k first.
3. Record acceptance, equivalent accept length, and latency simulation from the
   same eval prompts.
4. Only after that, evaluate custom large MoE targets. Very large MoE models
   need activation-capture support and are not the right first scaling proof.
