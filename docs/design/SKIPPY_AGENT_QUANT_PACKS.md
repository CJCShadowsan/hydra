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
- `300657676c6c3372a6d02bbebae02878d6367891cb9017d9aa5ca9a69d2bf118`
  `decode-profile.json`
- `886927bdebc99b833b484bf340cabdc34837f75d8e1441ace49caa2e1cf86728`
  `evidence/local-split-chain.json`
- `c54db85cb7a5d41f8832f49428bf546c10976cf80787fa678d7dbfbfe4b9dadd`
  `evidence/rank-after-local-split.json`

A follow-on Studio-local proxy sweep added comparable measured runs:
`baseline-source-quant`, which packages the source Q4_K_M layout unchanged,
`stage-balanced-proxy`, which lowers layers `19..24` in the largest byte stage
to Q3_K_M, and `stage-balanced-ffn-proxy`, which lowers only FFN tensors in
that same stage band while keeping attention at the source quant. Each run has
the same small local decode profile shape
(`existing_kv_tokens=128`, `warmup_samples=1`, `samples=3`) and the same
local split-chain lane (`splits=10,19`, `ctx_size=1024`, f16 activation wire).
The rank report lives at
`/Volumes/External/skippy-quant-packs/qwen25-coder-7b-proxy/sweep/rank-after-proxy-candidates.json`
with SHA-256
`e8c73be2da8513ba78d2f4d69424634f0ba30888965c4f3ae5b5af2cf1e4786c`.

| Rank | Candidate | Decode mean ms | Package bytes | Slowest stage bytes | Stage imbalance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `baseline-source-quant` | `14.028069` | `4,872,975,232` | `1,800,450,848` | `1.391920` |
| 2 | `stage-balanced-ffn-proxy` | `14.498056` | `4,702,706,560` | `1,779,022,592` | `1.375354` |
| 3 | `stage-balanced-proxy` | `14.615736` | `4,682,263,424` | `1,779,022,592` | `1.375354` |
| 4 | `ffn-compressed-attention-protected` | `15.891708` | `4,209,404,800` | `1,630,182,176` | `1.634234` |

This is the most important proxy result so far: the unchanged packaged source
quant still wins on local decode. `stage-balanced-proxy` improves byte pressure
and stage-balance a little, and `ffn-compressed-attention-protected` saves much
more space, but neither beats the baseline decode profile. The FFN-only repair
does improve over the broad stage-balanced candidate, which is evidence that
protecting attention matters for this proxy, but it also still fails to beat
the baseline. These are valid candidate-pack evidence runs, because they prove
the builder, package, preflight, split-chain, rank, and audit hash flow. They
are not winners.

The next local proxy candidate should therefore be latency-aware rather than
byte-only. The next useful repairs should split FFN compression into smaller
per-layer or per-tensor groups, for example only `ffn_gate`/`ffn_up`, only
layers with Q6 source FFN tensors, or a one-layer-at-a-time sensitivity pass.
Only candidates that improve or hold decode latency while reducing memory
pressure should graduate to the expensive agent-quality and lab/HF evidence
lanes.

Additional proxy artifact hashes:

- `0dc4883de30d028bcc29947fd5ca44f3b3359049337b58db9bcd3258364301fd`
  `baseline-source-quant/baseline-source-quant.gguf`
- `110003c7d08671efebf432df20d6748e7d406da372e1995ae16f2db94dbb96ff`
  `baseline-source-quant/decode-profile.json`
- `39e56ca6a3e25b8d8cc291fb98c9fe781f568e3f7f11f77cf58c4d3dd936bcae`
  `baseline-source-quant/evidence/local-split-chain.json`
- `680bb6a8125d38a01db402aed7ae94c6085f5a79efd97e0bf250331e7140b9b6`
  `stage-balanced-proxy/stage-balanced-proxy.gguf`
- `f42287f22175ed5340142b8c9c961919571cd33b80581178896d7970ee601681`
  `stage-balanced-proxy/decode-profile.json`
- `e3345ae08b85ff1c328ec4d9697e06d97c176de1b1f070bddde9bbfebff9946c`
  `stage-balanced-proxy/evidence/local-split-chain.json`
- `1193892ea3135f8b7bc9b02bd0730eec2c3b3c7cb9fda8db7d1fe5d5d9edaf89`
  `stage-balanced-ffn-proxy/stage-balanced-ffn-proxy.gguf`
- `b1186071523765ddefb2fe7761bac223bf04d253a4a4cefc9a9c9a068e358753`
  `stage-balanced-ffn-proxy/decode-profile.json`
- `324871217f334c98a9cfcf53c003298117f84d6756f7c4d0ad528e2f0644fba0`
  `stage-balanced-ffn-proxy/evidence/local-split-chain.json`

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
