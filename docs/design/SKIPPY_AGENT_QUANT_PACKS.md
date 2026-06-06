# Skippy Agent Quant Packs

Status: design draft

Skippy Agent Quant Packs are Skippy-native quantized model packages. They start
from a source model and produce layer package artifacts whose quantization
layout, stage-planning hints, latency profiles, and certification evidence are
optimized for Skippy's staged runtime and coding-agent workloads.

The goal is not to make one smaller GGUF or to merely profile existing quants.
The goal is to quantize source models so they run really well on Skippy: lower
stage latency, better memory fit, more balanced splits, and preserved coding
agent behavior such as tool-call validity, patch quality, long-context recall,
and stable repeated-prefix execution.

Profiling is an input to this pipeline, not the product. The product is a
repeatable path from source model to validated Skippy quant pack.

`skippy-bench` is the runtime evidence spine for the post-profiler work. The
quant-pack tooling should generate reproducible `skippy-bench` plans and then
consume the resulting reports; it should not grow a second benchmark runner.

## Problem

Whole-model quantization treats the model as one artifact with one quality and
latency tradeoff. Skippy execution does not work that way. Skippy runs a model
as ordered stages, each with its own compute, memory, KV-cache pressure, and
activation-transfer cost. A quant that is reasonable for single-process
llama.cpp can still be poorly shaped for staged serving.

For staged serving, the planner needs to understand:

- which contiguous layer ranges fit each node;
- how expensive each layer range is during prefill and decode;
- which tensors or layer bands are sensitive to lower precision;
- whether a split plan leaves enough KV/cache headroom for agent loops;
- whether activation transfer cost overwhelms the compute saved by splitting.

For quantization, the pack builder also needs to decide where to spend
precision:

- embeddings and output tensors may need higher precision to protect token
  identity and logits;
- early and late layers may be more quality-sensitive than middle layers;
- latency-heavy layer bands may be worth lowering if quality holds;
- stage boundaries may need quant layouts that make the slowest stage less
  dominant;
- MoE routers, experts, and attention/FFN tensors may need different policies
  by family.

Coding-agent workloads make this sharper. They often combine large stable
prefixes, repo context, tool definitions, short decode bursts, JSON/function
arguments, and many repeated turns. A quant that looks good on generic
throughput can still be a poor agent model if it breaks structured outputs or
small patch details.

## Goals

- Define a package-level design for Skippy-native mixed quantization.
- Generate quantized layer package artifacts from source models.
- Keep quant evidence, native layer latency, and agent certification attached to
  package identity.
- Let topology planning score stage layouts by measured latency and quality
  evidence, not only layer count.
- Preserve compatibility with existing layer-package consumers by making new
  metadata additive.
- Create a repeatable path from base model to certified agent-optimized package.

## Non-Goals

- Do not require a mesh protocol change for the first version.
- Do not make older nodes understand quant-layout semantics.
- Do not replace existing family certification, package validation, or Skippy
  correctness gates.
- Do not treat agent-pack certification as a universal quality claim for all
  chat, reasoning, or multimodal workloads.
- Do not introduce lossy activation-wire defaults without family/split evidence.

## Definitions

| Term | Meaning |
| --- | --- |
| Source model | The original GGUF model coordinate, revision, file list, and checksums. |
| Quant layout | The quantization format applied to each tensor group or layer band. |
| Skippy quant pack | A quantized layer package optimized for Skippy staged execution. |
| Native layer latency | Measured per-layer prefill/decode latency for a model artifact on a backend/device. |
| Agent pack | A layer package with agent-focused quant layout, stage hints, profiles, and evidence. |
| Certification evidence | Machine-readable reports proving package validity, staged correctness, agent behavior, and cache stability. |

## Workload Model

Agent packs optimize for requests shaped like coding-agent traffic:

- large system prompts and tool definitions;
- repo or task context with long shared prefixes;
- many turns with same-prefix reuse;
- short decode bursts between tool calls;
- strict JSON/schema output for OpenAI-style `tool_calls`;
- patch/diff generation where identifiers and whitespace matter;
- recovery turns after tool results or failed edits;
- routing through `model: auto` and `model: mesh` as well as direct model ids.

The certification suite should measure these behaviors directly. Generic
perplexity and tokens/sec remain useful diagnostics, but they are not sufficient
promotion gates for an agent pack.

## Package Shape

An agent pack is still a normal Skippy layer package. Existing consumers should
be able to reject or ignore unknown optional metadata without confusing tensor
ownership, layer indexing, or artifact paths.

Durable package identity still comes from:

- `model-package.json`;
- source model coordinate and revision;
- source artifact checksums;
- package artifact checksums;
- Skippy ABI compatibility.

Agent-pack metadata is additive:

```json
{
  "agent_pack": {
    "schema_version": 1,
    "profile": "coding-agent",
    "base_model_id": "Qwen/Qwen3-Coder-30B-A3B-GGUF:Q4_K_M",
    "pack_id": "qwen3-coder-30b-skippy-agent-v1",
    "quant_layout": {
      "strategy": "stage-aware-mixed",
      "default": "Q4_K_M",
      "groups": [
        {
          "name": "embedding-and-output",
          "tensors": ["token_embd", "output"],
          "quant": "Q6_K"
        },
        {
          "name": "early-layers",
          "layers": [0, 7],
          "quant": "Q5_K_M"
        },
        {
          "name": "middle-layers",
          "layers": [8, 55],
          "quant": "Q4_K_M"
        },
        {
          "name": "late-layers",
          "layers": [56, 63],
          "quant": "Q5_K_M"
        }
      ]
    },
    "certification": {
      "status": "candidate",
      "reports": []
    }
  }
}
```

The exact manifest location can be decided during implementation. The
compatibility rule is more important than the first field name: this metadata
must not change required schema-version-1 behavior unless the layer-package spec
is explicitly revised.

## Native Layer Profiles

Each agent pack should carry or reference measurements with this shape:

```json
{
  "native_layer_profile": {
    "schema_version": 1,
    "model_artifact_sha256": "<package or source checksum>",
    "backend": "metal",
    "device": {
      "stable_id": "metal:apple-m3-ultra",
      "memory_bytes": 274877906944
    },
    "runtime": {
      "mesh_llm_version": "0.x.y",
      "skippy_abi": "x.y.z",
      "llama_cpp_revision": "<revision>"
    },
    "request_shape": {
      "phase": "decode",
      "existing_kv_tokens": 8192,
      "generated_tokens": 1,
      "batch_size": 1,
      "kv_type": "f16"
    },
    "layers": [
      {
        "index": 0,
        "mean_ms": 1.7,
        "p95_ms": 2.1,
        "samples": 50
      }
    ]
  }
}
```

Profiles are measurements, not immutable model truth. The planner should prefer
fresh local measurements when available and fall back to package-published
profiles when local data is missing.

Native profiles should separate at least:

- prefill latency;
- decode latency;
- KV/cache memory pressure;
- stage materialization size;
- activation transfer bytes per boundary;
- backend/device/runtime version.

## Decode-First Profiling

Agent packs should optimize decode first once repeated-prefix caching is healthy.
The first turn of a coding-agent session may still be prefill-heavy, but later
tool loops usually become:

```text
small suffix prefill + generated_tokens * decode_ms_per_token
```

That makes decode latency the steady-state bottleneck for coding agents. The
profiler should therefore treat warm-KV, single-token decode as the primary
measurement lane:

```text
layer_decode_ms[token=1, batch=1, warm_kv]
```

The next decode lanes should show how layer cost changes under agent pressure:

```text
layer_decode_ms[token=1, batch=N]
layer_decode_ms[ctx=8k, warm_kv]
layer_decode_ms[ctx=32k, warm_kv]
layer_decode_ms[ctx=64k, warm_kv]
layer_decode_ms[cache_pressure=true]
```

Context length matters because decode is not constant. Attention cost and
memory behavior change as KV length grows, so a quant layout that is fast at
2k context may be a poor choice for 32k or 64k agent sessions.

The profiler should still measure prefill, suffix-prefill, and cache replay as
guardrails. Decode wins only translate into better agent latency when prefix
reuse remains stable and suffix-prefill does not regress.

The planner-facing summary should make the decode estimate explicit:

```text
total_decode_ms_per_token =
  sum(layer_decode_ms)
+ sampling_overhead
+ kv_cache_overhead
+ stage_transfer_overhead
+ scheduler_overhead

estimated_tokens_per_second = 1000 / total_decode_ms_per_token
```

For split serving, the planner also needs the slowest-stage estimate:

```text
pipeline_decode_ms_per_token =
  max(stage_decode_ms)
+ boundary_transfer_ms
+ scheduler_overhead
```

Single-request latency remains constrained by the ordered stage pipeline. The
pipeline estimate is most useful for finding unbalanced stages and for
predicting aggregate throughput under concurrent agent traffic.

## Quantization Strategy

Mixed quantization should be generated and evaluated by layer or tensor group.
The first candidate matrix should include:

| Candidate | Purpose |
| --- | --- |
| Whole-model baseline | Establish the current quality/latency baseline. |
| Higher precision embeddings/output | Protect token identity and final logits. |
| Higher precision first/last bands | Test common sensitivity around boundary and output behavior. |
| Lower precision latency-heavy bands | Reduce the ranges that dominate native latency. |
| Stage-balanced layout | Tune layer bands so planned stages finish at similar times. |
| Agent-sensitive layout | Raise precision only where agent evals show regressions. |

The selected layout should optimize for:

```text
agent_score / (decode_latency + transfer_cost + memory_pressure)
```

where `agent_score` includes structured-output reliability and edit quality,
not only text similarity.

Quantization should improve Skippy serving in four concrete ways:

- **Lower stage latency**: reduce decode and prefill cost in expensive layer
  bands, especially the slowest planned stage.
- **Better memory fit**: reduce model and KV/cache pressure so stages fit on
  more nodes and leave headroom for long coding-agent sessions.
- **Better stage balance**: choose layer/tensor precision so planned stages
  finish closer together instead of one stage dominating the pipeline.
- **Preserved quality**: spend higher precision on tensors or bands where lower
  precision harms tool calls, code edits, long-context recall, or split
  correctness.

The pack builder should start with deterministic layout families before trying
automatic search:

| Layout family | Initial policy |
| --- | --- |
| Baseline | Match the source quant across all tensors, then package and certify it. |
| Boundary protected | Raise embeddings, output, first band, and last band by one quant tier. |
| Middle compressed | Lower the middle layer band while keeping boundaries protected. |
| FFN compressed, attention protected | Lower middle-band FFN tensors while keeping attention tensors at a safer tier for long-context recall. |
| Stage balanced | Lower precision in the currently slowest planned stage until stage times converge or quality fails. |
| Quality repaired | Raise precision only in layer/tensor groups implicated by failed agent or correctness evals. |

Protection is source-aware. If the input GGUF is already Q4, a protected group
should usually stay Q4 so later broad compression rules cannot lower it further;
it should not be up-quantized to Q6/Q5 and spend memory without recovering
quality. Higher-precision sources such as F16, Q8, or Q6 can still use Q6/Q5
protection tiers because those are real down-quant choices.

For MoE coder models, the initial candidate matrix should not treat expert and
router tensors as ordinary middle-layer weights. Router tensors such as
`ffn_gate_inp` should be protected at a higher tier, and expert tensors such as
`ffn_*_exps` should have explicit tensor-name selectors that appear before broad
layer-range compression rules in the quantizer override file. Expert protection
should be a source-aware floor: it should preserve Q6/Q5 source tiers and never
drop below the initial Q4_K_M protection tier. This keeps stage-latency
experiments from silently spending precision in the wrong place.

The first implementation does not need a perfect optimizer. A deterministic
candidate matrix plus measured ranking is enough to start producing useful
packs and evidence.

## End-to-End Pipeline

The pack flow should be source-model driven:

```text
source model
  -> inspect tensors, layers, architecture, and family policy
  -> generate candidate quant layouts
  -> quantize tensors by layout
  -> write Skippy layer package artifacts
  -> validate package ownership and staged correctness
  -> profile Skippy serving shapes
  -> run coding-agent certification
  -> rank candidates and publish the best pack with evidence
```

The corresponding CLI surface can grow in this order:

```bash
skippy-model-package quant-pack source-plan unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF \
  --revision main \
  --local-dir /Volumes/External/models/qwen3-coder-480b \
  --allow-pattern 'UD-Q4_K_XL/*.gguf' \
  --quant-pack-out-dir target/skippy-quant-packs/qwen3-coder-480b \
  --expected-download-bytes 275600000000 \
  --min-free-bytes 330000000000 \
  --hf-jobs-workload-out qwen-hf-job-workload.sh \
  --hf-jobs-submit-json-out qwen-hf-job-submit.json \
  --hf-jobs-image ghcr.io/<owner>/skippy-quant-pack-job:cpu \
  --hf-jobs-timeout 36h \
  --hf-jobs-upload-repo <owner>/<repo> \
  --out qwen-source-plan.json \
  --script-out fetch-qwen-source.sh

skippy-model-package quant-pack hf-jobs-validate qwen-hf-job-submit.json \
  --expected-image ghcr.io/<owner>/skippy-quant-pack-job:cpu \
  --expected-upload-repo <owner>/<repo>

skippy-model-package quant-plan <source.gguf> \
  --profile coding-agent \
  --stages 2 \
  --out quant-plan.json

skippy-model-package quantize <source.gguf> \
  --plan quant-plan.json \
  --candidate middle-compressed \
  --out-dir quantize-run/ \
  --llama-quantize /path/to/llama-quantize

skippy-model-package quant-pack finalize quantize-run/quantize-run.json \
  --out-dir candidate-pack-run/ \
  --model-id org/model:middle-compressed \
  --source-revision <revision> \
  --source-file <source.gguf> \
  --stages 2 \
  --reuse-package-if-present

skippy-model-package quant-pack build <source.gguf> \
  --profile coding-agent \
  --stages 2 \
  --candidate middle-compressed \
  --llama-quantize /path/to/llama-quantize \
  --out-dir candidate-pack-run/ \
  --model-id org/model:middle-compressed \
  --source-revision <revision> \
  --source-file <source.gguf> \
  --decode-profile

skippy-model-package quant-pack build-all <source.gguf> \
  --profile coding-agent \
  --stages 2 \
  --llama-quantize /path/to/llama-quantize \
  --out-dir candidate-sweep/ \
  --model-id-prefix org/model \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --activation-wire-dtype f16 \
  --decode-profile

skippy-model-package quant-pack rank candidate-*/ \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --activation-wire-dtype f16 \
  --out quant-pack-rank.json

skippy-model-package quant-pack evidence-plan candidate-pack/ \
  --hosts host-a,host-b \
  --splits 20 \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --activation-wire-dtype f16 \
  --remote-root /tmp/skippy-runtime-bench \
  --remote-root-map host-b=/Volumes/External/skippy-runtime-bench \
  --remote-shared-root-map host-a=/Volumes/External/skippy-runtime-bench \
  --endpoint-host-map host-b=192.168.0.4 \
  --metrics-otlp-grpc-url http://host-a:14317 \
  --rsync-model-artifacts \
  --evidence-dir candidate-pack/evidence \
  --out candidate-pack/evidence-plan.json \
  --script-out candidate-pack/run-evidence.sh

skippy-model-package quant-pack evidence-plan-all candidate-sweep/ \
  --hosts host-a,host-b \
  --splits 20 \
  --top-ranked 2 \
  --out candidate-sweep/evidence-plan-all.json \
  --script-out candidate-sweep/run-evidence.sh

skippy-model-package quant-pack certify candidate-pack/ \
  --skippy-bench-report evidence/focused-runtime-report.json \
  --skippy-bench-report evidence/chat-corpus.json \
  --skippy-bench-report evidence/long-context-chat-corpus.json \
  --skippy-bench-report evidence/prompt-lengths-summary.json \
  --quality-evidence evidence/agent-tool-call-results.jsonl \
  --quality-evidence evidence/kv-tool-loop-stability/summary.json \
  --require-skippy-bench \
  --require-quality-evidence \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --activation-wire-dtype f16 \
  --out candidate-pack/certification.json
```

The generated `source-plan` script runs `hf download`, discovers the first
downloaded `.gguf`, derives `--source-file` from its basename, and prints the
follow-on `quant-pack build-all` command rather than launching a huge
quantization sweep automatically. Pass `--source-file <downloaded.gguf>` when a
known first shard should be pinned instead of discovered. Do not use Skippy
materialized stage/tokenizer slices as source inputs; those are derived cache
artifacts. For large sources, pass `--expected-download-bytes` and
`--min-free-bytes` so the runbook records the dry-run size and fails before
download when the target volume cannot fit the source. For 480B-scale
candidates, also pass `--hf-jobs-workload-out` and submit the generated
workload to Hugging Face Jobs or another remote runner. That workload downloads
the source inside the job, runs `quant-pack build-all`, and can upload the
output directory when `HF_UPLOAD_REPO` is set, keeping Studio out of the
high-memory quantize/package path.
Pass `--hf-jobs-submit-json-out` with `--hf-jobs-image` when the source plan
should also emit a reviewable Hugging Face Jobs `run` payload. That payload
captures the image, flavor, timeout, command, `HF_TOKEN` secret, and optional
`HF_UPLOAD_REPO` value; it is an operator handoff artifact, not an automatic
paid job submission. The job image must provide `skippy-model-package`,
`llama-quantize`, `hf`, and the backend libraries needed by the selected
quantization path.
The CPU image for this handoff is built with `just docker-build-quant-pack-job
ghcr.io/<owner>/skippy-quant-pack-job:cpu` and pushed with
`just docker-push-quant-pack-job ghcr.io/<owner>/skippy-quant-pack-job:cpu`.
The current Qwen 480B remote build handoff is tracked in
`docs/runbooks/QWEN480_SKIPPY_QUANT_PACK_HF_JOB.md`.
Validate generated submit payloads with `quant-pack hf-jobs-validate` before
submission. The validator checks the HF Jobs `run` envelope, known hardware
flavor, timeout, detached execution, `HF_TOKEN` secret, source download,
`quant-pack build-all`, output repo creation, and upload command.
Its report also includes `hf_jobs_cli.shell`, an equivalent `hf jobs run ...`
command for operators who submit through the Hugging Face CLI.

The initial Qwen Coder source target has been verified with `hf models info` and
`hf download --dry-run`:

- repo: `unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF`;
- revision: `b86deeefd82f1a3374c5536dfc1dd0ce27ac092d`;
- source include: `UD-Q4_K_XL/*.gguf`;
- shard count: 6;
- dry-run size: about 275.6G;
- first shard:
  `UD-Q4_K_XL/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-00001-of-00006.gguf`.

`quant-plan` should emit candidate layouts, not mutate model bytes. Each layout
records:

- source model identity and checksum;
- target profile such as `coding-agent`;
- layer count, tensor groups, and quant policy;
- intended stage count and preferred split boundaries;
- quality-risk notes such as protected embeddings/output or boundary bands;
- expected artifact identity fields used later by package validation.

`quantize` should turn one layout into a quantized GGUF plus the exact tensor
override and `agent-pack.json` metadata needed to reproduce it. The metadata
should include the source GGUF path/hash, inferred source quant, selected layout
hash, and tensor groups.

`quant-pack finalize` should be the resumable bridge from an existing
`quantize-run.json` to the normal candidate build manifest. It should package
the recorded quantized GGUF with the adjacent agent-pack metadata, run preflight,
and write `quant-pack-build.json` without rerunning quantization. For
Qwen-sized artifacts, it should also support reusing an already-materialized
package while still regenerating preflight and manifest state, so interrupted or
supervised runs can continue into `rank`, `evidence-plan`, and `certify`.

`quant-pack build` should be the one-shot selected-candidate path. It should
write the quant plan, run the quantizer, create ordinary Skippy layer package
artifacts with additive quant metadata, run preflight, optionally attach a
decode-first profile, and emit a build manifest that points at every
intermediate artifact. The manifest should also serialize the resolved source
identity, quantizer path, thread and split-output policy, package validation
policy, and decode-profile request shape so the candidate can be audited without
recovering shell history. Existing Skippy consumers can ignore the metadata;
new tooling can use it for planning, certification, and ranking.

`quant-pack build-all` should run the same selected-candidate path across every
candidate in a quant plan, or across a requested subset. It should produce one
run directory per candidate plus a top-level rank report so the operator can
immediately see which candidates deserve quality certification. The sweep should
record the context size, GPU-layer policy, KV cache dtypes, and activation wire
dtype used for first-pass ranking so those assumptions match the later
`skippy-bench` and certification commands. It should also record the shared
quantizer, split-output, package-validation, and decode-profile assumptions
once at the sweep level, while each candidate manifest records its resolved
source/package identity and concrete artifacts. The sweep manifest should
summarize candidate artifact readiness, including the quantized model, package,
preflight output, decode profile, future evidence directory, and certification
path, so missing local build outputs are visible before lab hardware is used. It
should also carry a `next_steps.evidence_plan_all` command template that points
at stable evidence-plan and runbook output paths, with only lab-specific host
selection left for the operator to fill in.

`quant-pack evidence-plan` should turn a chosen candidate build into an
operator-ready evidence manifest. It should read the package `model_id`,
quantized GGUF path, layer count, stage count, and package path, infer split
boundaries by default or accept explicit `--splits`, record the activation wire
dtype, and emit the exact corpus-prep, `skippy-bench`, QA, and
`quant-pack certify` commands with stable output paths. It should end by
rerunning `quant-pack rank` into a stable `rank-after-evidence.json` path so
operators immediately see the post-certification score. It should carry
focused-runtime deployment knobs such as remote root maps, shared roots,
endpoint host maps, remote metrics collector URL, selected stage/metrics
binaries, model artifact rsync policy, and keep-remote policy into the generated
`skippy-bench focused-runtime` command. Before the measured remote run, it
should emit a `focused-runtime --schema-smoke` command with the same topology
and runtime-shape arguments so operators can validate split boundaries,
layer-end, context, KV cache, activation-wire, corpus, and lab-option plumbing
locally. It should also be able to emit an executable shell runbook from the
same command list so operators can run the evidence pass without copying
commands by hand. The script should check each command's declared outputs so
missing reports fail close to the command that was supposed to produce them,
and multi-candidate scripts should hoist shared corpus-preparation commands so
a sweep does not redo shared setup once per candidate. This is the
reproducibility bridge from candidate ranking to real hardware evidence.
Plans should also be able to include an optional local split-chain evidence lane
for Studio-scale proving before the lab is available. For topologies with at
least two split boundaries, for example a 48-layer Qwen coder proxy with
`--splits 16,32` or a four-stage Qwen-scale package with `--splits 16,32,47`,
the plan can insert `skippy-bench local-split-chain-binary` and write
`evidence/local-split-chain.json`. That report should record the predicted
token, exact first-boundary activation payload/wire byte counts, and
same-shape transfer estimates for later decode boundaries. This proves local
stage slicing, stage chaining, and transfer-size shape for a candidate, and
gives `rank`/`certify` another optional evidence report to attach. It is not a
replacement for measured distributed `focused-runtime`, because it does not
cover multi-node scheduling, network contention, remote backend differences, or
tail latency under concurrent agent traffic.
It is also not the safe default for 480B-scale direct GGUFs: a local chain can
start one process per stage, and each process may map or load a large slice on
the same machine. For large Qwen coder candidates, Studio should produce plans,
token audits, schema smoke, and small-model/proxy local proofs; quantize,
package, profile, and focused-runtime evidence should run on lab nodes or
Hugging Face Jobs. Direct-GGUF local chain runs above the safety guard should
require an explicit operator override.

Current Studio-local proxy evidence uses
`unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` as a safe Qwen Coder family
stand-in for the local split-chain lane. The static proxy profile at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/evidence/static-profile.json`
records a 28-layer, three-stage split `10,19` with stage artifact bytes
`1,707,802,624`, `1,234,151,424`, and `1,735,165,952`. The local split-chain
report at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/evidence/local-split-chain.json`
successfully returned a predicted token and measured a first-boundary f16 wire
payload of `7,168` bytes for activation width `3,584`, with the second boundary
recorded as a same-shape estimate. A separate direct-GGUF local-stage decode
profile at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/evidence/local-stage-decode-profile.json`
measured the full 7B proxy as a single stage with `existing_kv_tokens=128`,
`samples=3`, and mean decode latency `14.145417 ms`. This proxy evidence proves
the local stage-chain and transfer accounting path; it is not certification
evidence for the 480B Qwen Coder pack.

Artifact hashes:

- `7a10f02dd143aeb7e59c4107ff8bc2f0b708a06698e7ac438be3c4ac2860151c`
  `static-profile.json`
- `8ecf18839f9650f1a571cd3f2ce0634c1b76b8cbeec99567587f2e11668e8a6a`
  `local-split-chain.json`
- `e233ae0db2387e8b13f73faaf151e7cd0fe1715b608dc19b367d20865d0d5bf8`
  `local-stage-decode-profile.json`

The first real Studio-local requantized proxy candidate is
`ffn-compressed-attention-protected`, built from the same
`unsloth/Qwen2.5-Coder-7B-Instruct-GGUF` Q4_K_M source into:

```text
/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep/ffn-compressed-attention-protected
```

This candidate keeps the source quant layout by default and lowers only the
middle-band FFN weight tensors for layers `4..=23` to Q3_K_M, using layout hash
`9533097889696c9c0f722ecd089be4a63d1a0359757c3e37e28e02237d0ec17b`.
It is intentionally attention-protected: middle-band attention tensors stay at
the source quant tier for the first proxy experiment so long-context recall is
not made worse before the evidence loop exists.

The candidate GGUF is
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep/ffn-compressed-attention-protected/ffn-compressed-attention-protected.gguf`,
with SHA-256
`8e1f7e9fa707dfdd9daa897c958a3949b5503f7714cdc89ceb869e045bf873aa`.
The source Q4_K_M blob is `4,683,073,504` bytes and the candidate GGUF is
`4,019,503,072` bytes, so this first mixed-quant proxy saves about `663 MB`
without changing attention tensors.

Preflight validates the three-stage package and shows the stage byte effect:

| Stage | Layers | Source proxy bytes | Candidate bytes |
| --- | --- | ---: | ---: |
| 0 | `0..10` | `1,707,802,624` | `1,581,701,888` |
| 1 | `10..19` | `1,234,151,424` | `997,520,736` |
| 2 | `19..28` | `1,735,165,952` | `1,630,182,176` |

That is useful memory and stage-balance evidence, especially for the middle
stage, but it is not yet a win on decode latency. The finalized direct local
stage decode profile is measured with `existing_kv_tokens=128`,
`warmup_samples=1`, and `samples=3`; it reports mean decode latency
`15.891708 ms` and p95 `15.939875 ms`. The source proxy baseline from the
earlier direct local-stage profile was `14.145417 ms` mean under the same tiny
sample shape, so this candidate demonstrates the pipeline and memory movement
but should not be promoted as the fast candidate.

The candidate local split-chain evidence proves the actual requantized GGUF can
run through the `10,19` stage chain, return predicted token `48298`, and carry
f16 activation wire payloads of `7,168` bytes at each boundary for activation
width `3,584`. After rerunning `quant-pack rank` with the finalized decode
profile attached, the candidate is `measured: true` and the rank uses
`direct_local_split_chain` transfer evidence. It remains uncertified:
`certification_status` is absent, no agent-quality evidence is attached, and
the rank notes correctly say quality is unproven.

Requantized proxy artifact hashes:

- `798abe557fc72fa0260209cfa01564496e2e5d3ea8a583ccfc83bdfb8c83edbf`
  `quant-pack-build.json`
- `e1f95b26aa1e83ea4dff282a70e3b8ef44b2324356b76d4fce3e4298bf73949a`
  `quantize/quantize-run.json`
- `8b06a5b7a079d41f5bb42a1412e42e952643ad5694aa1258ffed581ba8bafff1`
  `preflight.json`
- `5e79e8318752e6e8ec55915c5330e09baf54397ce4b64082baa3a04ead215939`
  `decode-profile.json`
- `886927bdebc99b833b484bf340cabdc34837f75d8e1441ace49caa2e1cf86728`
  `evidence/local-split-chain.json`
- `c54db85cb7a5d41f8832f49428bf546c10976cf80787fa678d7dbfbfe4b9dadd`
  `evidence/rank-after-local-split.json`

A follow-on Studio-local proxy sweep added comparable measured runs over the
packaged source baseline and progressively narrower stage-balance candidates:
`stage-balanced-proxy`, which lowers layers `19..24` in the largest byte stage
to Q3_K_M; `stage-balanced-ffn-proxy`, which lowers only FFN tensors in that
same stage band while keeping attention at the source quant; per-projection FFN
variants that lower only `ffn_down`, `ffn_gate`, or `ffn_up` tensors in layers
`19..23`; and a `stage-balanced-ffn-gate-up-proxy` variant that combines the
two promising single-projection lanes. The first sweep used the same small
local decode profile shape (`existing_kv_tokens=128`, `warmup_samples=1`,
`samples=3`) and the same local split-chain lane (`splits=10,19`,
`ctx_size=1024`, f16 activation wire). The normalized confirmation pass
reprofiled every candidate with `existing_kv_tokens=128`, `warmup_samples=3`,
and `samples=20`, so the current rank table compares all rows under the same
local decode measurement shape. The refreshed quant plan lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/quant-plan.json`
with SHA-256
`fcd823cce5ac16b425b380aef682b107a154d5586b28efa6173b629cb4fdd86e`.
The current rank report lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep/rank-after-proxy-candidates.json`
with SHA-256
`482a8439758cd667986eefef10fe483c31449b1f95832943a5292c413a03a2a8`.

| Rank | Candidate | Decode mean ms | Package bytes | Slowest stage bytes | Stage imbalance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `baseline-source-quant` | `13.912183` | `4,872,975,232` | `1,800,450,848` | `1.391920` |
| 2 | `stage-balanced-ffn-down-proxy` | `14.025075` | `4,792,880,000` | `1,779,022,592` | `1.375354` |
| 3 | `stage-balanced-ffn-gate-up-proxy` | `14.307750` | `4,782,801,792` | `1,779,022,592` | `1.375354` |
| 4 | `stage-balanced-proxy` | `14.476614` | `4,682,263,424` | `1,779,022,592` | `1.375354` |
| 5 | `ffn-compressed-attention-protected` | `15.761321` | `4,209,404,800` | `1,630,182,176` | `1.634234` |
| 6 | `stage-balanced-ffn-gate-proxy` | `15.934977` | `4,827,888,512` | `1,779,022,592` | `1.375354` |
| 7 | `stage-balanced-ffn-up-proxy` | `16.152862` | `4,827,888,512` | `1,779,022,592` | `1.375354` |
| 8 | `stage-balanced-ffn-proxy` | `17.183527` | `4,702,706,560` | `1,779,022,592` | `1.375354` |

This is the most important proxy result so far: the unchanged packaged source
quant still wins on local decode. The normalized 20-sample pass also changed
the earlier single-projection conclusion: `ffn_down` is now the best compressed
proxy and is only `0.112892` ms slower than baseline on mean decode while
reducing package bytes and the largest stage. The combined `ffn_gate_up`
candidate remains useful as a second compressed lane, but `ffn_gate` and
`ffn_up` alone show worse mean and tail latency under the stronger profile. The
broader `stage-balanced-proxy` variant remains a byte-pressure experiment with
moderate decode cost, while `stage-balanced-ffn-proxy` looks too slow after
normalization. The older `ffn-compressed-attention-protected` candidate is a
larger memory win but not a decode win. These are valid candidate-pack evidence
runs because they prove the builder, package, preflight, split-chain, rank, and
audit hash flow. They are not final winners because quality is still unproven
and baseline remains the fastest decode profile.

The next layer-sensitivity pass now emits one-layer-at-a-time `ffn_down` and
`ffn_gate_up` probes inside the largest unprotected stage. For the 7B proxy
shape, the refreshed plan added layer probes for layers `19..23` after the
coarse FFN candidates and before the broad `stage-balanced-proxy` candidate.
The plan lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/quant-plan-layer-sensitivity.json`
with SHA-256
`6093f5d8787f24a01a6b02980e18ee9ed91973ea311e398e14b07c11e247216c`.
The focused 10-candidate sweep built each probed GGUF, packaged it, preflighted
it, and attached local decode profiles with `existing_kv_tokens=128`,
`warmup_samples=3`, `samples=20`, `ctx_size=1024`, `n_gpu_layers=0`, and f16 KV
and activation wire settings. Every decode profile reported
`measurement_status.status: measured` with 20 samples. The rank report lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep-layer-sensitivity/quant-pack-rank.json`
with SHA-256
`d0b1e69ab5e9898b9b8de852cfaf9478da7313e2bbe5e0f0eab36bfeaca29b07`.

| Rank | Candidate | Decode mean ms | Decode p95 ms | Package bytes | Slowest stage bytes | Stage imbalance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `stage-balanced-layer-22-ffn-gate-up-proxy` | `13.545296` | `13.595750` | `4,854,940,544` | `1,782,416,160` | `1.377977` |
| 2 | `stage-balanced-layer-22-ffn-down-proxy` | `13.597048` | `15.153250` | `4,863,957,888` | `1,791,433,504` | `1.384949` |
| 3 | `stage-balanced-layer-19-ffn-gate-up-proxy` | `13.632567` | `14.051625` | `4,854,940,544` | `1,782,416,160` | `1.377977` |
| 4 | `stage-balanced-layer-20-ffn-down-proxy` | `13.937233` | `14.013042` | `4,846,453,632` | `1,779,022,592` | `1.375354` |
| 5 | `stage-balanced-layer-21-ffn-down-proxy` | `13.936871` | `14.040083` | `4,863,957,888` | `1,791,433,504` | `1.384949` |
| 9 | `stage-balanced-layer-23-ffn-down-proxy` | `22.331746` | `29.860750` | `4,846,453,632` | `1,779,022,592` | `1.375354` |
| 10 | `stage-balanced-layer-23-ffn-gate-up-proxy` | `22.880223` | `27.191958` | `4,854,940,544` | `1,782,416,160` | `1.377977` |

This narrows the next candidate construction step: combine the best low-latency
single-layer probes, starting with layer 22 `ffn_gate`/`ffn_up` and a small
number of `ffn_down` probes from layers 20, 21, or 22, while treating layer 23
as latency-sensitive under this local proxy shape. The combined candidates
should be compared against both the unchanged packaged baseline and the best
coarse compressed lanes. Only candidates that hold decode latency while taking
meaningful memory pressure out of the package should graduate to the expensive
agent-quality and lab/HF evidence lanes.

A follow-on mixed-candidate sweep added `quant-pack compose-candidate` as the
bridge from measured probes to buildable combined layouts. The command appends
an evidence-composed candidate to an existing quant plan by combining selected
candidate groups, deduping shared groups, and recomputing the layout hash with
the same quant-layout hash scheme used by the planner. `quant-pack build` and
`quant-pack build-all` now also accept `--plan`, so composed plans can flow
through the same quantize, package, preflight, decode-profile, and rank path as
planner-native candidates. The composed 7B proxy plan lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/quant-plan-mixed-layer-candidates.json`
with SHA-256
`0540df673691a2ede2bfa7f7b738f8936f346ed1c103cd5295cc37c6ac37f68b`.
The 3-candidate sweep lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep-mixed-layer-candidates/`;
its build-all manifest has SHA-256
`49aa75c88871d5cbbac011d00fa46dec79858031482bd075f5c5e73a67b81ed9`, and its
rank report has SHA-256
`e5a612950e3bb7e9c15fe04da7f6774a47cff84bd4704b8474799fbc34003fe1`.

| Rank | Candidate | Decode mean ms | Decode p95 ms | Package bytes | Slowest stage bytes | Stage imbalance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `mixed-layer-22-20-21-down-gate-up-proxy` | `13.522375` | `13.622666` | `4,810,384,256` | `1,779,022,592` | `1.375354` |
| 2 | `mixed-layer-22-gate-up-20-21-down-proxy` | `14.098735` | `14.315041` | `4,819,401,600` | `1,779,022,592` | `1.375354` |
| 3 | `mixed-layer-22-gate-up-20-down-proxy` | `53.905646` | `85.157000` | `4,828,418,944` | `1,779,022,592` | `1.375354` |

All three mixed candidates were measured with 20 local decode samples. The best
row combines layer 22 `ffn_gate`/`ffn_up` with `ffn_down` on layers 20, 21, and
22. Under this proxy shape it is slightly faster than the previous single-layer
winner and smaller than the coarse `stage-balanced-ffn-gate-up-proxy` package.
The two-group row is a strong warning that apparent single-probe wins do not
compose monotonically; it should not graduate.

The combined normalized comparison then ranked the surviving mixed candidate
against the unchanged packaged baseline, the best coarse compressed lanes, and
the best single-layer probe using the measured 20-sample decode-profile
artifacts as the source of truth. The combined rank report lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/combined-normalized-ranking/quant-pack-rank.json`
with SHA-256
`25bb4a953d371371ec9041cec213c62d209bdb4fedfbedbccc21d887914835b9`.

| Rank | Candidate | Decode mean ms | Decode p95 ms | Package bytes | Slowest stage bytes | Stage imbalance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `mixed-layer-22-20-21-down-gate-up-proxy` | `13.522375` | `13.622666` | `4,810,384,256` | `1,779,022,592` | `1.375354` |
| 2 | `stage-balanced-layer-22-ffn-gate-up-proxy` | `13.545296` | `13.595750` | `4,854,940,544` | `1,782,416,160` | `1.377977` |
| 3 | `baseline-source-quant` | `13.912183` | `13.993625` | `4,872,975,232` | `1,800,450,848` | `1.391920` |
| 4 | `stage-balanced-ffn-down-proxy` | `14.025075` | `14.081625` | `4,792,880,000` | `1,779,022,592` | `1.375354` |
| 5 | `stage-balanced-ffn-gate-up-proxy` | `14.307750` | `14.376167` | `4,782,801,792` | `1,779,022,592` | `1.375354` |
| 6 | `stage-balanced-proxy` | `14.476614` | `14.668208` | `4,682,263,424` | `1,779,022,592` | `1.375354` |

This is the current proxy graduation gate. The mixed candidate is now the best
measured local proxy row and also improves package bytes, slowest-stage bytes,
and stage imbalance versus the packaged source baseline. It should graduate to
coding-agent quality evidence and larger-model/HF evidence. It is still not a
final Skippy-optimized Qwen pack because certification, tool-call/JSON checks,
code-edit quality, long-context recall, repeated-turn KV behavior, and
larger-model transfer/runtime evidence remain unproven.

The graduated mixed candidate now has a concrete evidence runbook instead of
only an oral next step. The runbook lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep-mixed-layer-candidates/mixed-layer-22-20-21-down-gate-up-proxy/run-evidence.sh`,
with its machine-readable plan at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep-mixed-layer-candidates/mixed-layer-22-20-21-down-gate-up-proxy/evidence-plan.json`.
The evidence plan SHA-256 is
`fd120822deb54f8e00fdd18727484d751c011021b32ae2b8231bec87a4a75175`, and the
runbook SHA-256 is
`ef07d24a95f041d52ac7361d26e2f8f931ebb31d3452b31c879f3695a618002f`.
As of the current evidence status audit, `7` of `14` generated commands are
complete. The complete lane covers evidence directory creation, default corpus
materialization, token-length audit, focused-runtime schema smoke, and a
Studio-local split-chain proof. The next missing command is the real
`skippy-bench focused-runtime --execute-remote` report; the runbook still uses
placeholder hosts `host-0,host-1,host-2`, so this step must run on actual lab
hosts or a Hugging Face job lane before certification can pass.

The same candidate also has a job-path evidence plan generated with
`--execution-run-dir` so the local planning artifacts can be inspected on
Studio while the resulting commands target a lab/HF filesystem. The job-path
plan lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep-mixed-layer-candidates/mixed-layer-22-20-21-down-gate-up-proxy/evidence-plan-job-path.json`,
with SHA-256
`79a080170d3712356f3f090f310a1d5addf6cf85e3dbf0d7b15ed7ddc42c7aae`; its
runbook lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep-mixed-layer-candidates/mixed-layer-22-20-21-down-gate-up-proxy/run-evidence-job-path.sh`,
with SHA-256
`1a8cd42010389df0d1ac8936b2c3326b1b11c3021a2b741203ef5c3e1fbe299e`. This plan
records `source_run_dir` as the local `/Volumes/External/...` candidate and
rebases `run_dir`, `package`, `quantized_model`, and `evidence_dir` under
`/tmp/skippy-quant-evidence/input/mixed-layer-22-20-21-down-gate-up-proxy`,
with `runbook_cwd=/workspace/mesh-llm`. On Studio, `evidence-status` correctly
reports this plan as fully missing with toolchain warnings because the target
paths and binaries are job-environment contracts. It is a handoff artifact, not
a completed evidence report.

The token-length audit is now configured for the evidence lane's real context
shape rather than the early proxy profiling shape: `ctx_size=8192`,
`generation_limit=512`, `enable_thinking=false`. It reports `565` corpus rows,
`565` fitting rows, `0` context overflows, p95 prompt length `734` tokens, p99
prompt length `1316` tokens, max prompt length `3187` tokens, and max requested
length `3699` tokens. The summary report SHA-256 is
`923b0babf9907cc769b5c6b95f4e5249ca880008116836de651e1c7781dc7336`.

The local split-chain proof for the same mixed candidate uses splits `10,19`,
`layer_end=28`, `ctx_size=8192`, `n_gpu_layers=0`, and f16 activation wire. It
records `mode: local-split-chain-binary`, activation width `3584`, predicted
token `48298`, token id `7985`, and two f16 boundary transfers: stage `0 -> 1`
at layer boundary `10` with `7168` wire payload bytes, and stage `1 -> 2` at
layer boundary `19` with `7168` wire payload bytes. The first boundary is
observed from the driver; the second is estimated from the first boundary
shape. The split-chain report SHA-256 is
`eb10d9da353812cc7f5f2ce0cecb4e235907275768286aba504e684a5b359387`.

The focused-runtime schema-smoke report verifies the intended stage topology
shape before real remote launch: three stages, placeholder hosts
`host-0,host-1,host-2`, splits `10,19`, stage ranges `0..10`, `10..19`, and
`19..28`, package stage-load mode, `ctx_size=8192`, `n_gpu_layers=0`, f16 KV,
and f16 activation wire. Its SHA-256 is
`8e4a635b4640871dfc8c6a64a98c35ea37f7894710032fe4a3ff03a5cffefe0b`. This is
only a command/topology shape proof; it must not count as measured runtime
evidence.

The missing half of the runbook is the expensive and quality-sensitive half:
measured focused runtime on real hosts or HF infrastructure, coding-loop
OpenAI-compatible chat corpus, long-context chat corpus, tool-call/JSON
reliability, repeated-turn KV/tool-loop stability, certification, and
post-evidence ranking. Until those exist and `quant-pack certify` binds their
hashes to the candidate artifacts, this mixed candidate remains a graduated
proxy candidate, not a trusted Skippy-optimized Qwen Coder release.

Additional proxy artifact hashes:

- `0dc4883de30d028bcc29947fd5ca44f3b3359049337b58db9bcd3258364301fd`
  `baseline-source-quant/baseline-source-quant.gguf`
- `c6227faef8cf109a1469cfa365ea24fcc42f6bfbeb65d9ef8ca612a6be835a2e`
  `baseline-source-quant/decode-profile.json`
- `39e56ca6a3e25b8d8cc291fb98c9fe781f568e3f7f11f77cf58c4d3dd936bcae`
  `baseline-source-quant/evidence/local-split-chain.json`
- `680bb6a8125d38a01db402aed7ae94c6085f5a79efd97e0bf250331e7140b9b6`
  `stage-balanced-proxy/stage-balanced-proxy.gguf`
- `386f7c781413ff7a09bcec0186682e270b5f5359b438da5a217a681cf1c97ee0`
  `stage-balanced-proxy/decode-profile.json`
- `e3345ae08b85ff1c328ec4d9697e06d97c176de1b1f070bddde9bbfebff9946c`
  `stage-balanced-proxy/evidence/local-split-chain.json`
- `1193892ea3135f8b7bc9b02bd0730eec2c3b3c7cb9fda8db7d1fe5d5d9edaf89`
  `stage-balanced-ffn-proxy/stage-balanced-ffn-proxy.gguf`
- `cbc4acdc28a63852d7111836899f9baa3c21112930408c86a44e45a8d46eb031`
  `stage-balanced-ffn-proxy/decode-profile.json`
- `324871217f334c98a9cfcf53c003298117f84d6756f7c4d0ad528e2f0644fba0`
  `stage-balanced-ffn-proxy/evidence/local-split-chain.json`
- `83928ee2564b1afb630cbc7dd24c3e26ee87ddac8f9b715cd2f9b7695c9343fa`
  `stage-balanced-ffn-gate-up-proxy/stage-balanced-ffn-gate-up-proxy.gguf`
- `90be96aeb503b9e8a03aa0179c0285df594a221d9db8f8f0f794bc9b79bcef08`
  `stage-balanced-ffn-gate-up-proxy/decode-profile.json`
- `6c3ada06c556d8c315256b4a5242cf80912a9ee5f71b9e123cd40139e06289a1`
  `stage-balanced-ffn-gate-up-proxy/evidence/local-split-chain.json`
- `8d59df6e962c9ea07328377344e86a9da6af83f24bbc1c74a306f0025caf67d1`
  `stage-balanced-ffn-down-proxy/stage-balanced-ffn-down-proxy.gguf`
- `9dc734980b35d2b1b13b3d20ed0401f99d97644d67a45abffae8e2b22763ed4f`
  `stage-balanced-ffn-down-proxy/decode-profile.json`
- `5911f576295c0f27c86fae4d0daa93e2c7a5d4220b3339e032a7f06bd8960b48`
  `stage-balanced-ffn-down-proxy/evidence/local-split-chain.json`
- `6b5c0b96922dc1bf30233100b45c7ac44825055bfc05aa4a8a4ccc727403e6de`
  `stage-balanced-ffn-gate-proxy/stage-balanced-ffn-gate-proxy.gguf`
- `2b68a91ba8203285337e18e4845cc9243d8ef2690f2499065149b9fe10719b48`
  `stage-balanced-ffn-gate-proxy/decode-profile.json`
- `e83cc8d685d6c66dc8d68cd9940d6b6ab7072524b8fae7e12b12738fe8c3586b`
  `stage-balanced-ffn-gate-proxy/evidence/local-split-chain.json`
- `24e58dbd2c7c2b45adb5de689ac00b058a9bb268623a2ecb07f2bfc377fbe0ec`
  `stage-balanced-ffn-up-proxy/stage-balanced-ffn-up-proxy.gguf`
- `285a604aab0b654ecea0f59fcd33293ddab1f4dae1739b87185e80dbab2dfd5c`
  `stage-balanced-ffn-up-proxy/decode-profile.json`
- `cc1c1aa1690a269d229d538cbbae5a8383d4161a60a9c9fb3bf3ce7dd49ff8c7`
  `stage-balanced-ffn-up-proxy/evidence/local-split-chain.json`
- `8e1f7e9fa707dfdd9daa897c958a3949b5503f7714cdc89ceb869e045bf873aa`
  `ffn-compressed-attention-protected/ffn-compressed-attention-protected.gguf`
- `5e79e8318752e6e8ec55915c5330e09baf54397ce4b64082baa3a04ead215939`
  `ffn-compressed-attention-protected/decode-profile.json`
- `886927bdebc99b833b484bf340cabdc34837f75d8e1441ace49caa2e1cf86728`
  `ffn-compressed-attention-protected/evidence/local-split-chain.json`

Plans should optionally include a lab preflight script before the measured
focused-runtime command. For Qwen-scale runs this makes SSH reachability, stale
stage processes, lab-port listeners, and free-space checks part of the declared
evidence state instead of an oral runbook step. The preflight should declare a
success marker that is written only after the checker exits cleanly, so a failed
preflight log cannot accidentally satisfy resume/status checks. It should also
make host roles explicit: `skippy-bench focused-runtime --hosts` are SSH
targets for remote stage launch, while `--endpoint-host-map` carries separate
stage fabric/IP addresses. The SSH preflight host list may check additional or
equivalent host strings, but if it differs from runtime `--hosts`, the plan
should warn that the runtime hosts still need to resolve over SSH. SSH options
such as port, user, identity, and timeout policy should also be serializable
into the evidence plan for both measured runtime launch and lab preflight
instead of relying on the operator's ambient shell environment. Plans should
warn when only the preflight has SSH options, because that can make the
preflight pass while the measured `skippy-bench` remote launch still uses
default SSH.
`quant-pack evidence-status` should read those generated plans back, check the
declared outputs, and report the next missing command so interrupted Qwen-scale
evidence runs can resume from the actual artifact state instead of from memory.
It should distinguish fully missing commands from partial commands where some
diagnostics exist but a required success marker is absent. For partial text
logs, it should surface an `observed_failure` line when possible so operators
can tell whether they are blocked on SSH reachability, dirty remote processes,
low disk, or a runtime command failure. It should also inspect known generated
JSON outputs and stay partial when measured `focused-runtime` is not
`mode: executed` with generated throughput and decode p50 latency,
`chat-corpus` or `long-context-chat-corpus` reports request errors,
`token-lengths` reports context overflow, rank outputs are malformed or
internally inconsistent, or `certify` writes a
failed, malformed, or status-less `certification.json`, because file existence
alone does not prove evidence success. The
`focused-runtime-schema-smoke` report remains useful as a topology and command
shape check, but it must not satisfy measured runtime evidence or
certification. Generated commands should also carry a stable `evidence_type`
lane label, for example `skippy-bench-focused-runtime`,
`skippy-bench-chat-corpus`, `skippy-bench-long-context-chat-corpus`, and
`skippy-bench-local-split-chain`, and `skippy-bench-token-lengths`, so
automation can identify report semantics without relying only on human-oriented
command ids. Older plans without the label remain valid by falling back to
command ids.
For multi-candidate plans, status should also track the sweep-level final rank
output and include it in aggregate complete/missing command counts.
It should also audit serialized local toolchain paths and warn when a declared
`skippy-bench`, `skippy-model-package`, QA script, or focused-runtime helper
binary/script is missing or non-executable, so the existing warning gate can
stop bad runbooks before remote hardware is touched.
The evidence manifest should also serialize the local `skippy-bench` and
`skippy-model-package` binary paths, plus the agent tool-call and KV tool-loop
QA script paths, plus the local runbook working directory used to generate those
commands. Generated runbooks should `cd` into that directory before running
warning gates, corpus prep, and relative paths. That keeps runbooks reproducible
across shells, background jobs, and resumed lab sessions where `target/debug`, a
release bundle, or the repo-local `scripts/` directory may not be on `PATH` or
relative to the caller's current working directory.

`quant-pack evidence-plan-all` should apply that same bridge to a `build-all`
sweep. It should inherit the sweep's runtime-shape assumptions, optionally
filter to explicit candidate ids or the top valid candidates from the rank
report, and emit per-candidate evidence directories and command plans without
launching the benchmarks itself.
When split boundaries come from quant-plan stage hints, the generated
`skippy-bench focused-runtime` command should opt into uneven stage ranges. Raw
layer-count balance is a useful guard for legacy split experiments, but
Skippy-native quant packs may need latency-, byte-, or memory-balanced stage
ranges that do not have identical layer counts.
Generated evidence runbooks should be resumable: before running a command, the
script should check that command's declared outputs and skip it only when every
output already exists and `quant-pack evidence-status --command-complete`
reports that the command is semantically complete. Partial commands must rerun,
which lets a failed preflight log remain useful diagnostics without letting it
satisfy the missing success marker, and prevents stale or failed JSON reports
from being silently accepted. Multi-candidate evidence runbooks should also
finish with a sweep-level rank report across all selected candidate run
directories, so the operator gets one post-certification comparison artifact
instead of stitching per-candidate reports by hand.

`quant-pack certify` should bind package validation, profiler output,
`skippy-bench` reports, agent harnesses, and cache stability checks to the exact
candidate build. Certification should fail closed: a candidate can stay
`experimental` with useful measurements, but it should not become
`certified_agent` without evidence. When quality evidence is required,
certification should require both the agent/tool-call reliability lane and the
KV/tool-loop stability lane to pass; extra ad hoc quality evidence may be
summarized, but it should not alone certify the pack for coding-agent use.
Agent-quality status should require that same complete quality coverage even
when the CLI is not failing the command on missing quality evidence.
Focused-runtime evidence should also prove that the measured split topology
matches the candidate's quant-plan stage hints. A latency report for the right
model on the right hosts is not enough if it used different layer boundaries.
The certification report should hash the build manifest, quantized GGUF,
package manifest, agent-pack metadata, preflight output, quantize-run manifest,
and all attached evidence reports so later audits can prove exactly which
candidate artifacts were certified. Evidence files that live inside the
candidate run should be recorded relative to the run directory, while outside
evidence should be recorded by absolute path, so rank/audit tools can rehash
in-run evidence after the candidate directory is moved or resumed elsewhere.

`rank` should choose among candidates by request shape. The initial ranker can
be a deterministic score:

```text
score =
  quality_score
- slowest_stage_decode_penalty
- memory_pressure_penalty
- transfer_penalty
- cache_instability_penalty
- uncertified_boundary_penalty
```

This makes profiler work useful without letting it become the main project:
profiling supplies latency and memory terms; certification supplies quality and
correctness terms; ranking decides which quant pack is worth using.

`quant-pack rank` is the early evidence ranker for local candidate runs. Before
full certification exists, it should compare preflight validity, attached decode
profile timing, slowest-stage model plus estimated KV cache bytes for the
selected context/cache dtype shape, stage-size imbalance, and a first-order
activation-transfer estimate from activation width and the selected wire dtype.
Its output can guide which candidates deserve expensive coding-agent quality checks.
It should also read the standard generated `evidence/` outputs directly, so a
sweep can rerank as soon as `skippy-bench focused-runtime`, `chat-corpus`,
`token-lengths`, or `local-split-chain` reports appear. Direct evidence remains
provisional until certification binds it to artifact hashes, and rank should
only credit direct reports that pass basic semantics: measured focused-runtime
must be `mode: executed`, chat corpus must have zero request errors, token
lengths must fit context, and local split-chain reports must have a predicted
token plus positive boundary wire bytes. A usable local split-chain report can
replace the crude activation-width transfer estimate in provisional ranking.
Rank output should record the transfer-cost source explicitly, for example
`certified_local_split_chain`, `direct_local_split_chain`, or
`preflight_estimate`, so operators know whether a candidate won on measured
Studio evidence or on a fallback estimate.
Schema-smoke or failed reports should remain useful diagnostics without
improving a candidate's provisional rank, and rank notes should name the usable
direct evidence lanes that counted and explain which direct evidence reports
were ignored.
Once `certification.json` exists beside a candidate build, or at the generated
`evidence/certification.json` path, ranking should become quality-aware and
should prefer the generated evidence copy when both paths exist. Agent-quality
certified candidates should beat merely fast uncertified candidates,
measurement-only candidates should stay provisional, and failed certifications
should sink the candidate even when its decode profile is attractive. Attached
certified evidence reports should only contribute rank measurements and counts
when their summarized status is `pass`; failed attached reports remain audit
evidence without improving the candidate's score. Ranking
should require certification to be verifiable by checking subject hashes against
the current build manifest, quantized GGUF, package manifest, agent-pack
metadata, preflight output, quantize-run manifest when available, and attached
`skippy-bench` or quality evidence report files. If those hashes no longer
match, or if the certification is missing the hashes needed to audit freshness,
the certification must be scored like a failed certification, and its attached
runtime measurements must not influence rank scoring.
Certified candidates should use focused-runtime generated-token throughput from
the attached `skippy-bench` report as stronger ranking evidence than local
single-stage decode triage, because it measures the actual staged runtime
topology.

## Reproducible Toolchain Requirements

The Qwen Coder pack is the first target, not a special case. The same toolchain
should work for other supported families by changing source model, family
policy, and certification profile.

Every generated plan, package, and evidence bundle should record enough inputs
to reproduce or audit the result:

- source model coordinate, revision, file list, and checksums;
- source quant or base precision;
- tool versions, Skippy ABI version, and llama.cpp revision;
- quant layout candidate id, layout hash, tensor selectors, and quant targets;
- stage count, split hints, activation dtype policy, and KV/cache shape;
- backend/device/runtime used for measurements;
- exact certification commands, prompt fixtures, and report paths;
- ranker version, scoring weights, and selected candidate reason.

The builder should avoid hidden defaults for decisions that affect artifacts.
Defaults are acceptable for the first Qwen Coder candidate, but they must be
serialized into the plan so a later run can explain why a tensor group was
protected, compressed, repaired, or left unchanged.

## Certification Gates

Promotion from candidate to certified should require evidence in these lanes.

| Lane | Required evidence |
| --- | --- |
| Package | Manifest validation, source checksum, artifact checksums, layer coverage, no duplicate owned tensors. |
| Correctness | Single-stage parity, representative 2-stage split parity, multi-stage chain parity, and activation dtype policy. |
| Agent behavior | Tool-call validity, streamed tool-call handling, tool-result continuation, patch generation, and direct `model` ids. |
| Cache stability | Same-prefix cache reuse, suffix-prefill behavior, repeated tool-loop stability, and native-log scan when available. |
| Performance | Native layer decode profile, prefill/cache guardrails, stage-latency balance, transfer overhead, TTFT, prompt time, decode throughput, and memory headroom. |
| Routing | Direct model, `auto`, and `mesh` behavior when the pack is available alongside ordinary models. |

The initial certification commands should reuse existing harnesses where
possible:

```bash
skippy-model-package preflight <package-dir> --stages 2
just bench-corpus long
just bench-corpus coding-loop
just bench-corpus long-context
skippy-bench token-lengths \
  --model-path <quantized.gguf> \
  --prompt-corpus target/bench-corpora/long/corpus.jsonl \
  --ctx-size 8192 \
  --generation-limit 512 \
  --layer-end <layer-count> \
  --enable-thinking false \
  --summary-json target/bench-corpora/long/prompt-lengths-summary.json
skippy-bench focused-runtime \
  --stage-model <package-dir> \
  --model-id <model-id> \
  --hosts <stage0-host>,<stage1-host> \
  --splits <layer-boundary> \
  --layer-end <layer-count> \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --activation-wire-dtype f16 \
  --prompt-corpus target/bench-corpora/coding-loop/corpus.jsonl \
  --max-new-tokens 512 \
  --scenario steady-decode \
  --execute-remote \
  --focused-output target/bench-runs/<pack>/focused-runtime-report.json
skippy-bench local-split-chain-binary \
  --model-path <quantized.gguf> \
  --model-id <model-id> \
  --splits 16,32 \
  --layer-end 48 \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --activation-wire-dtype q8 \
  --prompt "Write a small Rust function that parses a semver string." \
  --output target/bench-runs/<pack>/local-split-chain.json
skippy-bench chat-corpus \
  --base-url http://127.0.0.1:9337/v1 \
  --model <model-id> \
  --prompt-corpus target/bench-corpora/coding-loop/corpus.jsonl \
  --max-tokens 512 \
  --stream \
  --include-usage true \
  --enable-thinking false \
  --output target/bench-runs/<pack>/chat-corpus.json
skippy-bench chat-corpus \
  --base-url http://127.0.0.1:9337/v1 \
  --model <model-id> \
  --prompt-corpus target/bench-corpora/long-context/corpus.jsonl \
  --max-tokens 512 \
  --stream \
  --include-usage true \
  --enable-thinking false \
  --output target/bench-runs/<pack>/long-context-chat-corpus.json
scripts/qa-agent-tool-call-reliability.py --base-url http://127.0.0.1:9337/v1 --models <model>
scripts/qa-kv-tool-loop-stability.py --base-url http://127.0.0.1:9337/v1 --models <model>
skippy-model-package quant-pack certify <quant-pack-run>/ \
  --skippy-bench-report target/bench-runs/<pack>/focused-runtime-report.json \
  --skippy-bench-report target/bench-runs/<pack>/chat-corpus.json \
  --skippy-bench-report target/bench-runs/<pack>/long-context-chat-corpus.json \
  --skippy-bench-report target/bench-corpora/long/prompt-lengths-summary.json \
  --quality-evidence target/agent-tool-call-reliability/results.jsonl \
  --quality-evidence target/kv-tool-loop-stability/<pack>/summary.json \
  --require-skippy-bench \
  --require-quality-evidence \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --activation-wire-dtype f16 \
  --out <quant-pack-run>/certification.json
```

`skippy-bench` should stay the owner of runtime evidence. It already has the
shapes the quant-pack pipeline needs: `focused-runtime` for staged latency and
throughput, coding-loop `chat-corpus` for OpenAI-compatible agent-loop
behavior, long-context `chat-corpus` for product-path long-context stress,
`token-lengths` for real tokenizer/context audits, plus `coding-loop` and
`long-context` generated corpora. `skippy-model-package quant-pack certify`
should consume those reports, hash them, summarize pass/fail status, and bind
them to the exact quant layout, package manifest, context size, GPU-layer
policy, KV cache dtype policy, and activation wire dtype. It should not grow a
second benchmark runner.

That makes the end-to-end split of ownership:

1. `skippy-model-package quant-plan` proposes Skippy-shaped tensor layouts.
2. `skippy-model-package quant-pack build-all` builds candidate GGUF/package
   artifacts and performs cheap local decode-first triage.
3. `skippy-model-package quant-pack evidence-plan-all` selects the best early
   candidates and emits the exact `skippy-bench` runbook, using candidate
   stage hints from the quant plan as the default split topology.
4. Optional Studio-local split-chain evidence proves the candidate's local
   stage slicing, predicted token path, and activation transfer byte counts
   before remote hardware is reachable.
5. `skippy-bench` runs the expensive hardware/product-path lanes and writes the
   focused-runtime, coding-loop chat, long-context chat, and token-length
   reports.
6. `skippy-model-package quant-pack certify` binds those reports and agent QA
   artifacts back to the exact quantized package.
7. The generated runbook reruns `skippy-model-package quant-pack rank` so
   certified evidence is reflected in per-candidate and sweep-level
   `rank-after-evidence.json` reports before the operator chooses the pack to
   publish or route.

When certification is run with `--require-skippy-bench`, the gate should require
the full initial evidence set: focused runtime, coding-loop chat, long-context
chat, and token lengths. This prevents a candidate from being promoted on a
latency-only result while context fit, product-path chat behavior, or
long-context behavior is still unproven. Required `skippy-bench` reports should
also identify the candidate by model id or quantized model path; otherwise
certification cannot prove that the benchmark evidence belongs to the package
being promoted.

Each certification run should write machine-readable artifacts and record:

- model/package id;
- source revision and checksums;
- quant layout hash;
- split points;
- activation wire dtype;
- backend/device/runtime identity;
- prompt and context shape classes;
- success/failure verdicts;
- evidence directory or report refs.

## Planner Integration

The topology planner should continue to map:

```text
model layers -> cached layer slices -> execution stages -> node placement
```

Agent-pack metadata adds scoring inputs:

- preferred split boundaries;
- forbidden or unproven boundaries;
- per-layer decode latency;
- prefill and suffix-prefill guardrail latency;
- per-layer or per-stage memory estimates;
- activation transfer bytes;
- certified activation wire dtype;
- quality/certification status;
- cache policy notes.

The planner should select a pack and split plan by request shape:

```text
score =
  agent_quality_weight
- decode_latency_penalty
- transfer_penalty
- memory_pressure_penalty
- cache_instability_penalty
- uncertified_boundary_penalty
```

For the first implementation, this can be a deterministic ranking over
candidate plans. It does not need a learned optimizer.

## Artifact Lifecycle

Agent-pack artifacts should move through explicit states:

| State | Meaning |
| --- | --- |
| `experimental` | Generated locally; no shared certification claim. |
| `candidate` | Package validates and has partial benchmark evidence. |
| `certified_agent` | Passed package, correctness, agent, cache, and performance gates for declared workload shapes. |
| `deprecated` | Superseded by a newer pack or failed after runtime/model-family changes. |

Published package READMEs should summarize:

- source model and revision;
- quant layout;
- intended workload profile;
- preferred split shapes;
- tested backends/devices;
- certification status;
- report locations;
- known limits.

## Compatibility

Agent packs must preserve the existing compatibility boundaries:

- package metadata additions are optional under schema version `1`;
- tensor ownership, layer indexing, artifact path semantics, and ABI
  requirements are unchanged unless the package spec is explicitly versioned;
- mesh gossip does not need to carry quant layouts;
- older nodes may ignore agent-pack metadata or reject unknown package
  requirements clearly;
- new planner behavior must remain additive and local unless mesh protocol
  fields are explicitly changed under normal compatibility rules.

If future work advertises agent-pack availability through gossip, the fields
must be optional and ignored by older peers.

## Implementation Plan

1. Define the additive quant metadata shape for `model-package.json` or a
   companion `agent-pack.json`.
2. Add `quant-plan` to generate deterministic layer/tensor-band candidate
   layouts for a source model and target Skippy profile.
3. Add `quantize` to apply one candidate layout and write a quantized GGUF plus
   reproducibility metadata.
4. Add `quant-pack build` to chain plan, quantize, package writing, preflight,
   and optional decode profiling for a selected candidate.
5. Extend package preflight to report quant layout identity, stage memory
   summaries, and protected tensor groups.
6. Add native profiler lanes that record decode-first latency by
   backend/device/runtime, with prefill and cache-replay guardrails.
7. Add `certify` to run package, correctness, agent, cache, performance, and
   routing gates against a candidate pack.
8. Add `quant-pack build-all` to sweep candidate layouts and produce a local
   rank report.
9. Add `quant-pack rank` to compare local candidate build evidence before
   expensive quality certification.
10. Add `rank` to compare certified candidate evidence and select the best pack for a
   request shape.
11. Teach topology planning to consume local/package profile hints when scoring
   split plans.
12. Publish one candidate Qwen Coder pack with evidence before generalizing to
   other families.

## First Model Candidates

Start with coding-heavy families that already matter for agent use:

- Qwen Coder family;
- DeepSeek Coder or DeepSeek-derived coding models where package generation is
  practical;
- Llama/Qwen instruct-code variants as comparison baselines.

Do not promote a family only because the base model is popular. Promotion should
depend on Skippy family support, package correctness, quant sensitivity, and
agent certification evidence.

## Open Questions

- Should agent-pack metadata live inside `model-package.json`, a companion
  `agent-pack.json`, or both?
- Should published latency profiles be trusted hints only, or should local
  runtimes persist their own replacement profiles automatically?
- Which agent benchmark should become the first promotion gate for patch
  quality: synthetic patch fixtures, SWE-bench-style trajectories, or real
  project edit loops?
- Should quant-layout generation live in `skippy-model-package`, an `xtask`,
  or a separate research tool?
- How should a mesh choose between a certified agent pack and a higher-quality
  unsplit local model when both are available?
