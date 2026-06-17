# GLM 4.7 SPD Execution Plan

This plan combines the local GLM 4.7 checkpoint, the GLM llama.cpp/Skippy work
on `feat/jianyang-glm-llama-patches`, the Skippy SPD proof handoff in PR #859,
and the reference implementation at `yuyijiong/speculative_pipeline_decoding`.

The goal is to train a GLM 4.7 SPD sidecar model that can act as a wider
drafting oracle than native GLM MTP, then use the Speedy benchmark to compare
vanilla GLM decode against GLM decode with the verified SPD sidecar enabled.

## Scope

This effort is about trained SPD on top of the current native-MTP branch. Native
GLM MTP remains the verifier/correctness foundation and implementation base, but
its `N=1` proposal width is likely too narrow to amortize multi-stage Skippy
latency. SPD is the candidate `N>1` drafting oracle. It should not be treated as
a competing unverified benchmark lane.

PR #860 is the compact GLM SPD donor branch for training, export, manifest, and
latency-model plumbing. PR #859 remains useful proof archaeology, but its broad
live-serving/protocol changes should only be revisited after offline GLM sidecar
quality justifies serving integration. The reference SPD repo remains the
training/evaluation source for the sidecar head.

## Phase 1: Inspect GLM

Inspect the local checkpoint before training or benchmarking:

- architecture and `model_type`
- tokenizer identity and chat template presence
- `num_hidden_layers`
- `hidden_size`
- vocab size
- any GLM-specific auxiliary tensors preserved by llama.cpp patches

The local `zai-org/GLM-4.7-Flash` snapshot reports:

- architecture: `Glm4MoeLiteForCausalLM`
- model type: `glm4_moe_lite`
- target layers: `47`
- hidden size: `2048`
- vocab size: `154880`
- auxiliary layer-47 tensors: `eh_proj`, `enorm`, `hnorm`

## Phase 2: Frontload Code Risk

Before long GPU training jobs, make the GLM SPD path executable enough to expose
integration gaps:

- add GLM model metadata inspection
- support explicit non-uniform `stage_layer_boundaries`
- derive GLM hidden-state tap rows from those boundaries
- generate a small GLM-tokenizer draft vocabulary for training smoke runs
- patch the reference trainer so explicit tap rows can bypass equal-stage
  assumptions
- write manifest-compatible GLM SPD smoke artifacts
- validate the smoke manifest through `skippy-runtime`
- keep the smoke artifacts clearly separate from trained weights

`evals/spd/glm47_frontload.py` owns this first executable surface. The default
GLM 4.7 Flash topology uses stage boundaries `15,31,47`, matching the 47-layer
target model without relying on equal layer division.

## Phase 3: Speedy Baseline

After the GLM code path exists, establish the vanilla GLM baseline:

1. Run the Speedy benchmark against vanilla GLM decode from the local checkpoint
   or Skippy package.
2. Freeze Speedy prompt set, generation settings, tokenizer, temperature, max
   tokens, hardware, and runtime build.
3. Record Speedy throughput, latency distribution, generated token counts, and
   output text.
4. If using Skippy, validate vanilla split correctness against non-split GLM.

This baseline is the only performance comparison target for SPD.

## Phase 4: Training Smoke

Before a model-quality run, run a tiny real training smoke:

- frozen local GLM 4.7 base model
- explicit stage boundaries `15,31,47`
- derived hidden tap rows `0,15,31,47;0,15,31;0,15`
- small generated GLM-tokenizer draft vocab
- a few rows from the training corpus
- `--skip-eval` unless the training checkpoint is produced successfully

The success criterion is artifact flow, not acceptance quality:

- `speculation_head_final.pt`
- `skippy-spd-head.json`
- optional `spd-head.safetensors` after export
- Rust SPD manifest validation

The initial tiny GLM 4.7 smoke passed and the artifacts are stored in the
private Hugging Face model repo `meshllm/skippy-spd-glm47-train-smoke`. The
repo contains the manifest, serving safetensors export, and original reference
checkpoint for reproducing the Skippy SPD manifest and serving-artifact path.

## Phase 5: Train And Evaluate SPD

Build a GLM tokenizer-specific draft vocabulary. Do not reuse Qwen draft vocab.
Start with 32k tokens, then try 50k if vocab coverage limits acceptance.

Train only the SPD sidecar head against the frozen GLM base model. Evaluate with
verification enabled and record accepted draft flags, acceptance rate,
equivalent accept length, theoretical gain, summaries, and raw traces.

Compare these paths, with tok/s as the hard optimization target:

- vanilla llama.cpp GLM target decode with zero MTP
- current Skippy GLM native-MTP verifier path
- Skippy GLM target decode with verified SPD sidecar, reported by draft width
  `N=1,2,4,8`

## Phase 6: Serving Decision

Only wire the trained SPD head into live Skippy if the Speedy benchmark shows
it is materially faster than vanilla GLM while preserving target-equivalent
verified output.

Serving integration then needs:

- Rust safetensors loading for the SPD head
- GLM SPD forward pass for the recorded topology
- Skippy hidden-state taps or transport
- Python/Rust proposal parity on fixed taps
- verified proposal generation
- rollback/session trim for rejected proposals
- SPD metrics for draft, accept, reject, proposal time, verification time, and
  end-to-end throughput

If the trained sidecar is weak, keep the result as research evidence and do not
wire it into serving.
