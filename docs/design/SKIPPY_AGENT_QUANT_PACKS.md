# Skippy Agent Quant Packs

Status: design draft

Skippy Agent Quant Packs are layer packages whose quantization layout,
stage-planning hints, latency profiles, and certification evidence are optimized
for coding-agent workloads.

The goal is not to make one smaller GGUF. The goal is to produce a package that
serves well through Skippy's staged architecture while preserving the behaviors
that coding agents depend on: tool-call validity, patch quality, long-context
recall, and stable repeated-prefix execution.

## Problem

Whole-model quantization treats the model as one artifact with one quality and
latency tradeoff. Skippy execution does not work that way.

For staged serving, the planner needs to understand:

- which contiguous layer ranges fit each node;
- how expensive each layer range is during prefill and decode;
- which tensors or layer bands are sensitive to lower precision;
- whether a split plan leaves enough KV/cache headroom for agent loops;
- whether activation transfer cost overwhelms the compute saved by splitting.

Coding-agent workloads make this sharper. They often combine large stable
prefixes, repo context, tool definitions, short decode bursts, JSON/function
arguments, and many repeated turns. A quant that looks good on generic
throughput can still be a poor agent model if it breaks structured outputs or
small patch details.

## Goals

- Define a package-level design for stage-aware mixed quantization.
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
cargo test -p skippy-runtime --lib
cargo test -p skippy-server --lib
cargo test -p mesh-llm-host-runtime --lib inference::skippy
scripts/qa-agent-tool-call-reliability.py --base-url http://127.0.0.1:9337/v1 --models <model>
scripts/qa-kv-tool-loop-stability.py --base-url http://127.0.0.1:9337/v1 --models <model>
```

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

1. Define the additive manifest metadata shape and report format.
2. Add a native layer profiler that records decode-first latency by
   backend/device/runtime, with prefill and cache-replay guardrails.
3. Add a quant-experiment generator for layer/tensor-band candidates.
4. Extend package preflight to report quant layout identity and stage memory
   summaries.
5. Add an agent-pack certification wrapper that runs package, correctness,
   agent, cache, and performance gates.
6. Teach topology planning to consume local/package profile hints when scoring
   split plans.
7. Publish one candidate Qwen Coder pack with evidence before generalizing to
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
