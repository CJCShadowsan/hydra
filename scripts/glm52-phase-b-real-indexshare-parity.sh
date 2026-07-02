#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/release/skippy-bench}"
LAYER_START="${LAYER_START:-2}"
LAYER_END="${LAYER_END:-6}"
CTX_SIZE="${CTX_SIZE:-128}"
TOKENS="${TOKENS:-1}"
POSITION_START="${POSITION_START:-16}"
ITERATIONS="${ITERATIONS:-1}"
WARMUP="${WARMUP:-0}"
N_BATCH="${N_BATCH:-16}"
N_UBATCH="${N_UBATCH:-16}"
KV_WARMUP_TOKENS="${KV_WARMUP_TOKENS:-16}"
KV_WARMUP_CHUNK_TOKENS="${KV_WARMUP_CHUNK_TOKENS:-16}"
SYNTHETIC_KV_WARMUP="${SYNTHETIC_KV_WARMUP:-0}"
REUSE_KV_WARMUP_CHECKPOINT="${REUSE_KV_WARMUP_CHECKPOINT:-0}"
REUSE_KV_WARMUP_STREAM="${REUSE_KV_WARMUP_STREAM:-0}"
COMPACT_FLASH_ATTN="${COMPACT_FLASH_ATTN:-0}"
ALLOW_COMPACT_FLASH_AUTO="${ALLOW_COMPACT_FLASH_AUTO:-0}"
COMPACT_FLASH_NO_MASK="${COMPACT_FLASH_NO_MASK:-0}"
REQUIRE_COMPACT_FLASH_PROOF="${REQUIRE_COMPACT_FLASH_PROOF:-0}"
DIRECT_SPARSE_ATTN="${DIRECT_SPARSE_ATTN:-0}"
DIRECT_SPARSE_PREFILL="${DIRECT_SPARSE_PREFILL:-0}"
NATIVE_DEFAULT_DIRECT_SPARSE_PREFILL="${NATIVE_DEFAULT_DIRECT_SPARSE_PREFILL:-0}"
NATIVE_DEFAULT_DIRECT_SPARSE_ATTN="${NATIVE_DEFAULT_DIRECT_SPARSE_ATTN:-0}"
ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL="${ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL:-0}"
DIRECT_SPARSE_PREFILL_MAX_TOKENS="${DIRECT_SPARSE_PREFILL_MAX_TOKENS:-}"
DENSE_SPARSE_MASK_MAX_BYTES="${DENSE_SPARSE_MASK_MAX_BYTES:-}"
REQUIRE_DIRECT_SPARSE_DECODE_PROOF="${REQUIRE_DIRECT_SPARSE_DECODE_PROOF:-0}"
REQUIRE_DIRECT_SPARSE_PREFILL_PROOF="${REQUIRE_DIRECT_SPARSE_PREFILL_PROOF:-0}"
SPARSE_ATTN_THREADS="${SPARSE_ATTN_THREADS:-}"
SPARSE_ATTN_GROUP_HEADS="${SPARSE_ATTN_GROUP_HEADS:-}"
METAL_DISPATCH_LOG="${METAL_DISPATCH_LOG:-0}"
METAL_TOPK_MOE_ROUTE_FUSION="${METAL_TOPK_MOE_ROUTE_FUSION:-0}"
TRACE_ROUTE_TENSORS="${TRACE_ROUTE_TENSORS:-0}"
TRACE_ROUTE_TENSOR_FILTER="${TRACE_ROUTE_TENSOR_FILTER:-}"
OUT_DIR="${OUT_DIR:-/tmp}"
REPORT="${REPORT:-}"
LOG="${LOG:-}"

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-b-real-indexshare-parity.sh [options]

Runs a real GLM-5.2 Phase-B Full/Shared IndexShare parity smoke against a local
layer package. This is a local llama.cpp/skippy-bench proof, not a Skippy split
or lab topology launch.

Options:
  --stage-model PATH      Layer package path.
  --model-id ID           Model id recorded in the report.
  --skippy-bench PATH     skippy-bench binary. Default: target/release/skippy-bench
  --layer-start N         Full producer layer. Default: 2
  --layer-end N           Exclusive end layer. Default: 6
  --ctx-size N            Context size. Default: 128
  --tokens N              Tokens. Default: 1
  --position-start N      Decode position start. Default: 16
  --iterations N          Iterations. Default: 1
  --warmup N              Warmup iterations. Default: 0
  --n-batch N             Batch size. Default: 16
  --n-ubatch N            Microbatch size. Default: 16
  --kv-warmup-tokens N    Populate KV prefix before the measured decode. Default: 16
  --kv-warmup-chunk-tokens N
                           KV warmup chunk size. Default: 16
  --synthetic-kv-warmup   Import synthetic zero KV pages for warmup.
  --reuse-kv-warmup-checkpoint
                           Reuse a checkpointed KV warmup prefix.
  --reuse-kv-warmup-stream
                           Reuse one runtime stream for KV warmup and measurement.
  --compact-flash-attn     Enable compact top-k K/V + flash-attention candidate path.
  --allow-compact-flash-auto
                           Allow llama.cpp's native compact flash policy to select the compact path without forcing it.
  --compact-flash-no-mask  Skip compact top-k mask gather for the compact flash candidate path.
  --require-compact-flash-proof
                           Fail unless compact flash eliminated the old sparse path.
  --direct-sparse-attn     Enable llama.cpp direct sparse-attention candidate path.
  --native-default-direct-sparse-attn
                           Do not pass --direct-sparse-attn to skippy-bench;
                           prove llama.cpp's native default selects direct sparse.
  --require-direct-sparse-decode-proof
                           Fail unless decode direct sparse avoids sparse-mask nodes and dispatches DSA sparse attention.
  --direct-sparse-prefill  Enable direct sparse-attention for prefill-shaped candidate runs.
  --native-default-direct-sparse-prefill
                           Do not pass the direct sparse prefill env toggle;
                           prove llama.cpp's native default selects direct sparse prefill.
  --enable-unproven-large-direct-sparse-prefill
                           Set the large direct-sparse prefill experiment opt-in.
  --direct-sparse-prefill-max-tokens N
                           Set the native short-prefill token cap.
  --dense-sparse-mask-max-bytes N
                           Set the dense sparse-mask byte guard threshold.
  --require-direct-sparse-prefill-proof
                           Fail unless direct sparse prefill avoids sparse-mask nodes and dispatches DSA sparse attention.
  --sparse-attn-threads N  Set SKIPPY_GLM_DSA_SPARSE_ATTN_THREADS for candidate runs.
  --sparse-attn-group-heads N
                           Set SKIPPY_GLM_DSA_SPARSE_ATTN_DECODE_GROUP_HEADS for candidate runs.
  --metal-dispatch-log     Capture Metal dispatch records.
  --metal-topk-moe-route-fusion
                           Enable native Metal top-k MoE route fusion during this run.
  --no-metal-topk-moe-route-fusion
                           Disable native Metal top-k MoE route fusion. Default for Phase B.
  --trace-route-tensors    Capture native route tensor digests and compare baseline/candidate traces.
  --trace-route-tensor-filter FILTER
                           Comma-separated tensor-name substrings to trace.
  --out-dir PATH          Default report/log directory. Default: /tmp
  --report PATH           Explicit report path.
  --log PATH              Explicit log path.
  -h, --help              Show this help.

Environment overrides mirror the upper-case option names.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-model)
      STAGE_MODEL="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --skippy-bench)
      SKIPPY_BENCH_BIN="$2"
      shift 2
      ;;
    --layer-start)
      LAYER_START="$2"
      shift 2
      ;;
    --layer-end)
      LAYER_END="$2"
      shift 2
      ;;
    --ctx-size)
      CTX_SIZE="$2"
      shift 2
      ;;
    --tokens)
      TOKENS="$2"
      shift 2
      ;;
    --position-start)
      POSITION_START="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --n-batch)
      N_BATCH="$2"
      shift 2
      ;;
    --n-ubatch)
      N_UBATCH="$2"
      shift 2
      ;;
    --kv-warmup-tokens)
      KV_WARMUP_TOKENS="$2"
      shift 2
      ;;
    --kv-warmup-chunk-tokens)
      KV_WARMUP_CHUNK_TOKENS="$2"
      shift 2
      ;;
    --synthetic-kv-warmup)
      SYNTHETIC_KV_WARMUP=1
      shift
      ;;
    --reuse-kv-warmup-checkpoint)
      REUSE_KV_WARMUP_CHECKPOINT=1
      shift
      ;;
    --reuse-kv-warmup-stream)
      REUSE_KV_WARMUP_STREAM=1
      shift
      ;;
    --compact-flash-attn)
      COMPACT_FLASH_ATTN=1
      shift
      ;;
    --allow-compact-flash-auto)
      ALLOW_COMPACT_FLASH_AUTO=1
      shift
      ;;
    --compact-flash-no-mask)
      COMPACT_FLASH_NO_MASK=1
      COMPACT_FLASH_ATTN=1
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --require-compact-flash-proof)
      REQUIRE_COMPACT_FLASH_PROOF=1
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --direct-sparse-attn)
      DIRECT_SPARSE_ATTN=1
      shift
      ;;
    --native-default-direct-sparse-attn)
      NATIVE_DEFAULT_DIRECT_SPARSE_ATTN=1
      DIRECT_SPARSE_ATTN=1
      shift
      ;;
    --direct-sparse-prefill)
      DIRECT_SPARSE_PREFILL=1
      DIRECT_SPARSE_ATTN=1
      shift
      ;;
    --native-default-direct-sparse-prefill)
      NATIVE_DEFAULT_DIRECT_SPARSE_PREFILL=1
      DIRECT_SPARSE_PREFILL=1
      DIRECT_SPARSE_ATTN=1
      shift
      ;;
    --enable-unproven-large-direct-sparse-prefill)
      ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL=1
      DIRECT_SPARSE_PREFILL=1
      DIRECT_SPARSE_ATTN=1
      shift
      ;;
    --direct-sparse-prefill-max-tokens)
      DIRECT_SPARSE_PREFILL_MAX_TOKENS="$2"
      shift 2
      ;;
    --dense-sparse-mask-max-bytes)
      DENSE_SPARSE_MASK_MAX_BYTES="$2"
      shift 2
      ;;
    --require-direct-sparse-decode-proof)
      REQUIRE_DIRECT_SPARSE_DECODE_PROOF=1
      DIRECT_SPARSE_ATTN=1
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --require-direct-sparse-prefill-proof)
      REQUIRE_DIRECT_SPARSE_PREFILL_PROOF=1
      DIRECT_SPARSE_PREFILL=1
      DIRECT_SPARSE_ATTN=1
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --sparse-attn-threads)
      SPARSE_ATTN_THREADS="$2"
      shift 2
      ;;
    --sparse-attn-group-heads)
      SPARSE_ATTN_GROUP_HEADS="$2"
      shift 2
      ;;
    --metal-dispatch-log)
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --metal-topk-moe-route-fusion)
      METAL_TOPK_MOE_ROUTE_FUSION=1
      shift
      ;;
    --no-metal-topk-moe-route-fusion)
      METAL_TOPK_MOE_ROUTE_FUSION=0
      shift
      ;;
    --trace-route-tensors)
      TRACE_ROUTE_TENSORS=1
      shift
      ;;
    --trace-route-tensor-filter)
      TRACE_ROUTE_TENSORS=1
      TRACE_ROUTE_TENSOR_FILTER="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    --log)
      LOG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$REQUIRE_COMPACT_FLASH_PROOF" == "1" && "$ALLOW_COMPACT_FLASH_AUTO" != "1" ]]; then
  COMPACT_FLASH_ATTN=1
fi

if [[ ! -x "$SKIPPY_BENCH_BIN" ]]; then
  echo "skippy-bench binary not executable: $SKIPPY_BENCH_BIN" >&2
  exit 1
fi

if [[ ! -d "$STAGE_MODEL" ]]; then
  echo "stage model package not found: $STAGE_MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
REPORT="${REPORT:-$OUT_DIR/glm52-real-indexshare-parity-l${LAYER_START}-${STAMP}.json}"
LOG="${LOG:-$OUT_DIR/glm52-real-indexshare-parity-l${LAYER_START}-${STAMP}.log}"

BENCH_ARGS=(
  glm-dsa-layer-microbench
  --stage-model "$STAGE_MODEL"
  --model-id "$MODEL_ID"
  --layer-start "$LAYER_START"
  --layer-end "$LAYER_END"
  --ctx-size "$CTX_SIZE"
  --tokens "$TOKENS"
  --position-start "$POSITION_START"
  --iterations "$ITERATIONS"
  --warmup "$WARMUP"
  --n-batch "$N_BATCH"
  --n-ubatch "$N_UBATCH"
  --direct-sparse-prefill "$([[ "$DIRECT_SPARSE_PREFILL" == "1" ]] && echo true || echo false)"
  --metal-topk-moe-route-fusion "$([[ "$METAL_TOPK_MOE_ROUTE_FUSION" == "1" ]] && echo true || echo false)"
  --op-timing true
  --compare-native-indexshare-producer-consumer
  --require-native-indexshare-proof
  --output "$REPORT"
)

if [[ "$NATIVE_DEFAULT_DIRECT_SPARSE_ATTN" != "1" ]]; then
  BENCH_ARGS+=(--direct-sparse-attn "$([[ "$DIRECT_SPARSE_ATTN" == "1" ]] && echo true || echo false)")
fi
if [[ "$NATIVE_DEFAULT_DIRECT_SPARSE_PREFILL" == "1" ]]; then
  BENCH_ARGS+=(--native-default-direct-sparse-prefill)
fi
if [[ "$ENABLE_UNPROVEN_LARGE_DIRECT_SPARSE_PREFILL" == "1" ]]; then
  BENCH_ARGS+=(--enable-unproven-large-direct-sparse-prefill)
fi
if [[ -n "$DIRECT_SPARSE_PREFILL_MAX_TOKENS" ]]; then
  BENCH_ARGS+=(--direct-sparse-prefill-max-tokens "$DIRECT_SPARSE_PREFILL_MAX_TOKENS")
fi
if [[ -n "$DENSE_SPARSE_MASK_MAX_BYTES" ]]; then
  BENCH_ARGS+=(--dense-sparse-mask-max-bytes "$DENSE_SPARSE_MASK_MAX_BYTES")
fi

if [[ "$KV_WARMUP_TOKENS" != "0" ]]; then
  BENCH_ARGS+=(--kv-warmup-tokens "$KV_WARMUP_TOKENS")
  if [[ -n "$KV_WARMUP_CHUNK_TOKENS" ]]; then
    BENCH_ARGS+=(--kv-warmup-chunk-tokens "$KV_WARMUP_CHUNK_TOKENS")
  fi
fi
if [[ "$SYNTHETIC_KV_WARMUP" == "1" ]]; then
  BENCH_ARGS+=(--synthetic-kv-warmup)
fi
if [[ "$REUSE_KV_WARMUP_CHECKPOINT" == "1" ]]; then
  BENCH_ARGS+=(--reuse-kv-warmup-checkpoint)
fi
if [[ "$REUSE_KV_WARMUP_STREAM" == "1" ]]; then
  BENCH_ARGS+=(--reuse-kv-warmup-stream)
fi
if [[ "$COMPACT_FLASH_ATTN" == "1" ]]; then
  BENCH_ARGS+=(--compact-flash-attn true)
fi
if [[ "$REQUIRE_DIRECT_SPARSE_DECODE_PROOF" == "1" ]]; then
  BENCH_ARGS+=(--require-direct-sparse-decode-proof)
fi
if [[ "$TRACE_ROUTE_TENSORS" == "1" ]]; then
  BENCH_ARGS+=(--trace-route-tensors true)
  if [[ -n "$TRACE_ROUTE_TENSOR_FILTER" ]]; then
    BENCH_ARGS+=(--trace-route-tensor-filter "$TRACE_ROUTE_TENSOR_FILTER")
  fi
fi
if [[ "$REQUIRE_DIRECT_SPARSE_PREFILL_PROOF" == "1" ]]; then
  BENCH_ARGS+=(--require-direct-sparse-prefill-proof)
fi
if [[ -n "$SPARSE_ATTN_THREADS" ]]; then
  BENCH_ARGS+=(--sparse-attn-threads "$SPARSE_ATTN_THREADS")
fi
if [[ -n "$SPARSE_ATTN_GROUP_HEADS" ]]; then
  BENCH_ARGS+=(--sparse-attn-group-heads "$SPARSE_ATTN_GROUP_HEADS")
fi
if [[ "$ALLOW_COMPACT_FLASH_AUTO" == "1" ]]; then
  BENCH_ARGS+=(--allow-compact-flash-auto)
fi
if [[ "$REQUIRE_COMPACT_FLASH_PROOF" == "1" ]]; then
  BENCH_ARGS+=(--require-compact-flash-proof)
fi
if [[ "$METAL_DISPATCH_LOG" == "1" ]]; then
  BENCH_ARGS+=(--metal-dispatch-log true)
fi

if [[ "$COMPACT_FLASH_NO_MASK" == "1" ]]; then
  export SKIPPY_GLM_DSA_ENABLE_COMPACT_FLASH_NO_MASK=1
fi
export GLM52_EXPECT_COMPACT_FLASH_NO_MASK="$COMPACT_FLASH_NO_MASK"

"$SKIPPY_BENCH_BIN" "${BENCH_ARGS[@]}" \
  >"$LOG" 2>&1

python3 - "$REPORT" <<'PY'
import json
import os
import sys

report_path = sys.argv[1]
with open(report_path) as f:
    report = json.load(f)

comparison = report.get("comparison") or {}
parity = comparison.get("parity") or {}
sensitivity = comparison.get("sideband_sensitivity") or {}
guard = report.get("native_indexshare_guard") or {}
baseline = comparison.get("baseline") or {}
candidate = comparison.get("candidate") or {}
candidate_trace = candidate.get("indexshare_trace_summary") or {}
candidate_ops = candidate.get("op_timing_summary") or {}
candidate_indexshare = candidate.get("indexshare_timing_summary") or {}
baseline_timing = baseline.get("timing_summary") or {}
candidate_timing = candidate.get("timing_summary") or {}
compact_guard = report.get("compact_flash_guard")
direct_decode_guard = report.get("direct_sparse_decode_guard")
expect_compact_no_mask = os.environ.get("GLM52_EXPECT_COMPACT_FLASH_NO_MASK") == "1"

def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_dicts(value)

def candidate_metal_records():
    summary = candidate.get("metal_dispatch_summary") or {}
    for key in ("dispatch_shapes", "records"):
        records = summary.get(key) or []
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                yield record
    for record in walk_dicts(candidate):
        if record.get("op") in {"get_rows", "flash_attn_ext"}:
            yield record

failures = []
if not parity.get("passed"):
    failures.append(f"parity failed: {parity}")
if not sensitivity.get("passed"):
    failures.append(f"sideband sensitivity proof failed: {sensitivity}")
if sensitivity.get("poisoned_hidden_mismatches", 0) < 1 and sensitivity.get("poisoned_hidden_max_abs_diff", 0) == 0:
    failures.append(f"poisoned sideband did not change hidden output: {sensitivity}")
if parity.get("hidden_mismatches") not in (0, None):
    failures.append(f"hidden mismatches present: {parity}")
if parity.get("sideband_mismatched_bytes") not in (0, None):
    failures.append(f"sideband mismatch present: {parity}")
if not guard.get("passed"):
    failures.append(f"native IndexShare guard failed: {guard}")
if guard.get("shared_exec_records", 0) < 1:
    failures.append(f"missing Shared execution records: {guard}")
if guard.get("shared_exec_missing_input_top_k", 0) != 0:
    failures.append(f"Shared layer missed top-k sideband: {guard}")
if guard.get("top_k_from_indexer", 0) < 1:
    failures.append(f"Full producer did not produce top-k from indexer: {guard}")
if candidate_trace.get("shared_exec_records", 0) < 1:
    failures.append(f"candidate missing Shared execution trace: {candidate_trace}")
if candidate_trace.get("shared_exec_missing_input_top_k", 0) != 0:
    failures.append(f"candidate Shared layer missed top-k sideband: {candidate_trace}")
if candidate_ops.get("indexer_topk", {}).get("nodes", 0) != 0:
    failures.append(f"candidate still ran indexer_topk nodes: {candidate_ops}")
if candidate_ops.get("indexer", {}).get("nodes", 0) != 0:
    failures.append(f"candidate still ran indexer nodes: {candidate_ops}")
if candidate_ops.get("top_k", {}).get("nodes", 0) != 0:
    failures.append(f"candidate still ran top_k nodes: {candidate_ops}")
if candidate_indexshare.get("producer_groups", 0) != 0:
    failures.append(f"candidate unexpectedly includes producer groups: {candidate_indexshare}")
if compact_guard is not None and not compact_guard.get("passed"):
    failures.append(f"compact flash proof failed: {compact_guard}")
if compact_guard is not None and candidate_ops.get("sparse_mask", {}).get("nodes", 0) != 0:
    failures.append(f"compact flash candidate still ran sparse-mask nodes: {candidate_ops}")
if direct_decode_guard is not None and not direct_decode_guard.get("passed"):
    failures.append(f"direct sparse decode proof failed: {direct_decode_guard}")
if expect_compact_no_mask:
    candidate_records = list(candidate_metal_records())
    mask_gathers = [
        record for record in candidate_records
        if "dsa_compact_mask_topk_rows" in str(record.get("tensor", ""))
    ]
    masked_flash = [
        record for record in candidate_records
        if record.get("op") == "flash_attn_ext" and record.get("mask_type") not in (None, "none")
    ]
    if mask_gathers:
        failures.append(f"compact no-mask path still gathered mask rows: {mask_gathers[:3]}")
    if masked_flash:
        failures.append(f"compact no-mask path still used masked flash attention: {masked_flash[:3]}")

if failures:
    print("GLM-5.2 Phase-B real-artifact parity FAILED", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    sys.exit(1)

baseline_ms = baseline_timing.get("mean_ms")
candidate_ms = candidate_timing.get("mean_ms")
speedup = baseline_ms / candidate_ms if baseline_ms and candidate_ms else None

print("GLM-5.2 Phase-B real-artifact parity passed")
print(f"report={report_path}")
print(f"full_layers={guard.get('full_layers')}")
print(f"shared_layers={guard.get('shared_layers')}")
print(f"shared_exec_with_input_top_k={guard.get('shared_exec_with_input_top_k')}")
print(f"sideband_sensitivity={sensitivity.get('passed')}")
print(f"sideband_poison_changed_i32={((sensitivity.get('poison') or {}).get('changed_i32_count'))}")
print(f"candidate_indexer_topk_nodes={candidate_ops.get('indexer_topk', {}).get('nodes')}")
print(f"candidate_sparse_mask_nodes={candidate_ops.get('sparse_mask', {}).get('nodes')}")
if compact_guard is not None:
    print(f"compact_flash_guard={compact_guard.get('passed')}")
    print(f"compact_flash_failure_summary={compact_guard.get('failure_summary')}")
if direct_decode_guard is not None:
    print(f"direct_sparse_decode_guard={direct_decode_guard.get('passed')}")
    print(f"direct_sparse_decode_failure_summary={direct_decode_guard.get('failure_summary')}")
if expect_compact_no_mask:
    print("compact_flash_no_mask_guard=True")
if speedup:
    print(f"baseline_mean_ms={baseline_ms:.6f}")
    print(f"candidate_mean_ms={candidate_ms:.6f}")
    print(f"diagnostic_ratio={speedup:.6f}x")
PY

printf 'log=%s\n' "$LOG"
