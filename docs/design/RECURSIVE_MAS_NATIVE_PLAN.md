# Native RecursiveMAS Integration Plan

This plan covers a mesh-native implementation of RecursiveMAS-style latent
multi-agent serving for mesh-llm.

Paper: <https://arxiv.org/abs/2604.25917>
Reference implementation: <https://github.com/RecursiveMAS/RecursiveMAS>

## Goal

Host RecursiveMAS systems in mesh-llm without Python in the serving path and
without text or JSON intermediate thought exchange. The public API remains
OpenAI-compatible text in/text out, but the internal agent-to-agent interface
uses native tensor/hidden-state frames.

The first implementation should prove the latency shape we care about:

- intermediate agents run latent rollout and return hidden states;
- only the final solver/summarizer performs serial token decode;
- mixture-style workers can fan out in parallel and return latent frames;
- role handoffs are coarse request-level binary tensor transfers, not per-token
  network turns and not text transcripts.

## Non-Goals

- No Python runtime sidecar as the proof of implementation.
- No JSON/text/base64 hidden-state payloads.
- No quantized first pass. The first certification uses full-precision native
  model conversions and full-precision RecursiveLink weights.
- No first-pass layer-split MAS. Start by hosting whole role models on mesh
  nodes, then combine with skippy layer splitting once role-level latent MAS is
  correct.

## What The Released MAS Models Are

The RecursiveMAS release is not a normal fine-tune where the role LLM weights
carry the main behavior change. The paper states that all base LLM parameters
are frozen and only the inner/outer RecursiveLink modules are trained.

The released Hugging Face repos contain full model weights plus separate
RecursiveLink artifacts:

- role model repos include `model.safetensors`, tokenizer files,
  `adapter_config.json`, `innerlink_config.json`, and task-specific inner
  adapters such as `adapter(math).pt` and `adapter(code).pt`;
- outerlink repos include task-specific outer adapters such as
  `Planner-Critic-Outerlink(math).pt`;
- configs declare role names, hidden widths, source/target roles, and adapter
  types.

Sequential-light math is the first target:

| Role | Released Model | Architecture | Hidden Width |
|---|---|---:|---:|
| Planner | `RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B` | Qwen3 | 2048 |
| Critic | `RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B` | Llama | 2048 |
| Solver | `RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B` | Qwen2 | 1536 |

Sequential-light outer links:

| Edge | Shape |
|---|---:|
| Planner -> Critic | 2048 -> 2048 |
| Critic -> Solver | 2048 -> 1536 |
| Solver -> Planner | 1536 -> 2048 |

The solver model artifact matches the declared base model metadata, and the
paper's training description says base LLMs remain frozen. Therefore the
correct native strategy is to host the same base/role weights converted to
full-precision native artifacts, then host the trained RecursiveLink modules as
separate native adapter artifacts.

## Hosting Strategy

### Whole-Role First

Each mesh node advertises one or more MAS roles backed by a complete model:

```text
request
  -> planner role: prompt tokens -> latent rollout -> planner latent frame
  -> critic role: prompt + planner latent -> latent rollout -> critic latent frame
  -> solver role: prompt + critic latent -> final token decode
```

For mixture:

```text
request
  -> math/code/science roles in parallel -> latent frames
  -> summarizer role consumes all latent frames -> final token decode
```

This keeps the first implementation focused on RecursiveMAS semantics. Skippy
layer splitting can later be used inside any individual role model.

### Same Models, Not Downsampled Quants

Use the released MAS role weights, or byte-equivalent base weights where proven,
converted to native full-precision GGUF:

- prefer BF16/F16;
- keep adapters in F32 or F16 initially;
- do not use Q4/Q5/Q8 until HF-vs-native hidden-state parity is measured.

RecursiveMAS depends on hidden-state geometry. Low-bit quantization is a later
performance knob, not a bootstrap shortcut.

## Native Tensor Contract

Add a reusable binary tensor frame contract by extending/reusing
`skippy-protocol` activation-frame machinery rather than creating a new text
payload format.

Required fields:

- frame kind: role latent, external embedding input, final decode conditioning;
- dtype: F32, F16, later Q8 if certified;
- layout: token-major;
- token or latent-step count;
- hidden width;
- source model id and role id;
- adapter id, task id, and recursion round;
- request/session id;
- raw tensor bytes.

The wire contract must be length-prefixed, bounded, and validated before
allocation. It should reuse the existing activation wire dtype encode/decode
helpers where possible.

## Runtime Work

### 1. Native Hidden-State ABI

Add llama-stage/skippy ABI hooks for full-model latent MAS execution:

- prefill prompt tokens and return last-layer hidden states;
- run autoregressive latent rollout where the next step is an embedding
  produced by the inner RecursiveLink instead of a sampled token;
- accept external embedding frames as prompt slots;
- decode final text from prompt prefix + latent frame + prompt suffix.

Rust wrappers belong in `skippy-runtime`; C ABI changes belong in the durable
llama-stage patch queue and must bump mirrored ABI version constants.

### 2. Native RecursiveLink Execution

Implement the released RecursiveLink modules natively:

- inner link: layer norm, linear, GELU, linear, residual, layer norm;
- outer link: source layer norm, two-layer projection, residual projection,
  target layer norm.

The runtime artifact should not be `.pt`. Add an offline converter that reads
the released PyTorch `.pt` files and writes a stable native adapter artifact
such as safetensors or GGUF-style tensors. Python is acceptable for this
one-time conversion tool; it is not acceptable in the serving path.

First implementation can use native Rust CPU tensor execution for adapter
correctness because adapter compute is small relative to role model forward
passes. Backend/GPU execution should follow once parity is locked.

### 3. MAS Package Manifest

Add a package format for a complete MAS system:

- MAS id and style: sequential, mixture, distillation, deliberation;
- task variant: math, code, default;
- role model refs and expected model metadata;
- role inner adapter refs;
- outer edge adapter refs;
- hidden widths and dtype requirements;
- default latent length;
- default recursion rounds;
- tokenizer/chat-template identities;
- minimum runtime feature flags.

The package resolver should materialize role models and adapter artifacts
separately. Identical base model weights should be deduped by content identity
where possible.

## Mesh Integration

### Capability Advertisement

Add optional additive gossip fields for MAS support:

- supported MAS package ids;
- role ids hosted by the node;
- hidden width per role;
- supported latent dtypes;
- supported adapter artifact ids;
- tokenizer/chat-template identity;
- runtime feature flags for hidden-state output and external embedding input.

Older nodes must ignore these fields. A node that lacks latent MAS support
must not be selected for a latent role.

### Orchestration

Add an explicit latent MAS runtime path in host-runtime:

- resolve requested MAS model id to package;
- place roles on local or remote nodes;
- open binary tensor streams to role nodes;
- execute the role graph;
- return only the final OpenAI-compatible answer.

Initial user-facing model id should be explicit, for example:

```text
recursive-mas/sequential-light-math
```

Do not route normal `mesh` MoA traffic into latent MAS automatically until
quality, routing, and failure behavior are certified.

### Failure Behavior

For the first version:

- if a required role is missing, return a clear unavailable-model error;
- if hidden widths or adapter ids do not match, fail closed;
- if a role times out, fail the MAS request rather than silently falling back to
  text MAS;
- text fallback can be added later behind an explicit compatibility flag.

## Implementation Phases

### Phase 0: Artifact Inspection And Parity Corpus

- Add a small manifest/checker for RecursiveMAS HF repos.
- Record exact model ids, file hashes, hidden widths, adapter shapes, and task
  variants.
- Build a frozen parity corpus of short prompts for sequential-light math.
- Capture HF reference outputs:
  - prompt token ids;
  - hidden rollout frames;
  - inner adapter outputs;
  - outer adapter outputs;
  - final deterministic decode.

### Phase 1: Adapter Conversion And Native Adapter Tests

- Convert `.pt` RecursiveLink weights into native adapter artifacts.
- Add Rust loader and shape validation.
- Add adapter math tests against golden tensors from the HF reference.
- Support F32 and F16 adapter artifacts.

### Phase 2: Native Hidden-State Hooks

- Add llama-stage ABI hooks for hidden-state output and external embedding
  input.
- Expose safe Rust wrappers in `skippy-runtime`.
- Add single-model tests that compare native hidden frames against HF reference
  at F16/BF16 precision.

### Phase 3: Single-Node Sequential-Light

- Run planner, critic, and solver in one process or one machine.
- Execute recursive latent handoff entirely with native frames.
- Decode only in the solver.
- Compare final outputs and timing against HF reference and a text-recursive
  baseline.

### Phase 4: Mesh-Distributed Sequential-Light

- Host planner, critic, and solver roles on separate mesh nodes.
- Transfer latent frames over binary QUIC streams.
- Validate request/session isolation, timeout behavior, and role placement.
- Measure:
  - wall time;
  - TTFT;
  - final decode tokens;
  - bytes transferred between roles;
  - role forward time;
  - adapter time.

### Phase 5: Mixture-Style Fanout

- Add mixture role graph support.
- Run math/code/science latent workers in parallel.
- Let only summarizer decode final text.
- Compare against current mesh MoA text workers plus reducer.

### Phase 6: Quantization And Split-Serving Experiments

Only after full-precision parity:

- test model quantization by family and role;
- test adapter F16 vs F32;
- test skippy layer splitting inside a single role;
- promote a quantized package only if final quality and hidden-frame drift stay
  within documented thresholds.

## Validation Gates

Minimum gates before calling this implemented:

- native adapter output parity against HF reference;
- native hidden-state output parity for each role model;
- single-node sequential-light final-answer parity on the frozen corpus;
- two-node or three-node mesh run with binary latent transfer;
- no text/JSON/base64 intermediate latent payloads;
- OpenAI public response compatibility remains unchanged;
- mixed-version mesh compatibility: older nodes ignore latent MAS gossip.

## Main Risks

- Hidden-state drift from native conversion or backend differences breaks the
  released RecursiveLink adapters.
- Newer released families such as `qwen3_5_text`, Gemma3 conditional, and
  BioMistral may require additional native model support before they can be
  hosted with the same confidence as sequential-light.
- Adapter execution on CPU may be acceptable for first correctness but could
  become visible at high batch sizes or larger mixture systems.
- Recursive rounds still introduce role-level barriers. The latency win is from
  removing intermediate text decode and enabling mixture fanout, not from making
  sequential dependency disappear.

## Recommended First Milestone

Deliver `recursive-mas/sequential-light-math` as a full-precision native MAS
package:

- same released/base model weights converted to F16/BF16 GGUF;
- native inner/outer RecursiveLink artifacts;
- binary latent frame transport;
- solver-only final decode;
- single-node and three-node validation reports.

This is the smallest implementation that proves the actual mesh value rather
than only proving that the RecursiveMAS Python example can be rearranged.
