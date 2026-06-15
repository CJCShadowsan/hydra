# GLM Native MTP Through Skippy

## Objective

Preserve the proven GLM-4.7 Flash native-MTP `n=1` speedup through Skippy before
attempting wider speculative windows or GLM-5.1.

This is intentionally narrower than speculative pipeline decoding. Native MTP is
the proposer; Skippy still owns staged target verification, rollback, cache
state, and user-visible serving.

## Current Evidence

GLM-4.7 Flash native MTP works in the patched llama.cpp serving path when
`--spec-type draft-mtp --spec-draft-n-max 1` is used.

Latest b2d6 rebench, run on 2026-06-15 with
`jamesdumay/GLM-4.7-Flash-MTP-GGUF`:

| Run | Requests | Avg decode tok/s | Speedup | Acceptance |
|---|---:|---:|---:|---:|
| baseline | 8 | 43.66 | 1.00x | n/a |
| MTP `n=1` | 8 | 60.09 | 1.38x | 67.2% |
| MTP `n=2` | 8 | 48.80 | 1.12x | 38.6% |
| MTP `n=3` | 8 | 34.16 | 0.78x | 21.7% |

Earlier 2026-06-11 evidence showed the same shape:

| Run | Requests | Avg decode tok/s | Speedup | Acceptance |
|---|---:|---:|---:|---:|
| baseline | 8 | 41.33 | 1.00x | n/a |
| MTP `n=1` | 8 | 60.84 | 1.47x | 67.2% |
| MTP `n=2` | 8 | 44.07 | 1.07x | 38.6% |
| MTP `n=3` | 8 | 36.06 | 0.87x | 21.7% |

The drafted/accepted token counts are identical across both SPEED-Bench runs.
The smaller b2d6 headline speedup comes from a faster non-MTP baseline, not from
a lower-quality MTP proposer.

Experiment artifacts:

- `lab-experiments/jianyang/phase-4/iteration-2.md`
- `lab-experiments/jianyang/benchmarks/speed-bench/results/20260615T080125Z-b2d6-mtp`

## Decision

Skippy should target native MTP `n=1` first.

`n=2` and `n=3` should stay disabled by default for GLM-4.7 Flash. `n=2` is too
thin and workload-sensitive to justify as a default. `n=3` is a clear
regression because acceptance collapses faster than verification amortization
improves.

## PR Scope

The first implementation PR should:

- keep GLM MTP as a proposer sidecar, not a trunk layer;
- keep Skippy trunk stage ranges bounded to transformer layers only;
- wire one native-MTP proposed token into Skippy target verification;
- preserve byte-identical output versus non-MTP baseline under greedy sampling;
- record drafted, accepted, rejected, acceptance-rate, proposer latency,
  verifier latency, and visible tok/s metrics; and
- expose `n=1` as the only supported GLM native-MTP setting.

## Non-Goals

This first PR should not attempt:

- GLM-5.1 native MTP serving;
- multi-token speculative windows beyond `n=1`;
- external draft-model speculation;
- speculative pipeline decoding;
- multi-host performance claims before single-node parity is proven; or
- turning MTP blocks into ordinary Skippy trunk stages.

## Acceptance Gates

Before promotion from draft, the implementation must show:

1. GLM-4.7 Flash baseline and Skippy native-MTP output are byte-identical for a
   fixed greedy prompt.
2. Accepted draft tokens are nonzero.
3. Skippy native-MTP `n=1` beats Skippy baseline on the same SPEED-Bench corpus.
4. `n=2` and `n=3` remain disabled or explicitly experimental.
5. Benchmark artifacts are recorded in `lab-experiments`.

## Validation Plan

Minimum local validation:

```bash
scripts/prepare-llama.sh pinned
just build
cargo test -p skippy-ffi --lib
cargo test -p skippy-runtime --lib
```

Then run GLM-4.7 Flash native-MTP checks:

```bash
# Native llama.cpp reference
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-native-mtp" \
LLAMA_DIR=/path/to/patched/llama.cpp \
/Users/jdumay/code/lab-experiments/jianyang/benchmarks/speed-bench/run-glm47-mtp-speed-bench.sh
```

The Skippy implementation should add an equivalent focused runner for the
Skippy native-MTP path and record its results beside the native llama.cpp
reference.

## Open Implementation Questions

- Which Skippy runtime owns the MTP proposer context: final stage, coordinator,
  or a dedicated sidecar?
- What is the smallest ABI surface needed to pass the target hidden state into
  the proposer without exposing wider SPD state?
- Can the `n=1` path reuse existing staged verification trim/checkpoint APIs, or
  does it need a simpler single-token rollback path?
- Which metric names should become stable telemetry versus benchmark-only
  fields?
