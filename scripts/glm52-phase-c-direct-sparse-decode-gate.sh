#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-c-direct-sparse-decode-gate}"
QUICK=0

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-c-direct-sparse-decode-gate.sh [options]

Runs the strict Phase-C decode gate for native GLM-5.2 direct sparse attention.

This gate assumes Phase A and Phase B are already closed. It proves decode only:

  - direct sparse decode decisions are selected;
  - sparse-mask timing nodes are absent in the candidate;
  - dense sparse-mask Metal dispatches are absent in the candidate;
  - native DSA_SPARSE_ATTN timing and Metal dispatch evidence is present;
  - parity still holds against the dense/direct producer baseline;
  - Shared consumers still reuse Full top-k sideband without recomputing top-k.

This is not a prefill policy gate, not an MTP gate, and not a Skippy split run.

Options:
  --stage-model PATH      GLM-5.2 layer package path.
  --model-id ID           Model id recorded in reports.
  --skippy-bench PATH     skippy-bench binary.
  --out-dir PATH          Artifact directory.
  --quick                 Run one reduced middle-span smoke case.
  -h, --help              Show this help.

Environment overrides mirror upper-case option names.
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
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --quick)
      QUICK=1
      shift
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

if [[ ! -x "$SKIPPY_BENCH_BIN" ]]; then
  echo "skippy-bench binary not executable: $SKIPPY_BENCH_BIN" >&2
  exit 1
fi
if [[ ! -d "$STAGE_MODEL" ]]; then
  echo "stage model package not found: $STAGE_MODEL" >&2
  exit 1
fi
if [[ ! -x "$ROOT/scripts/glm52-phase-b-real-indexshare-parity.sh" ]]; then
  echo "required Phase-B parity wrapper not executable" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

run_decode_case() {
  local name="$1"
  shift
  local case_dir="$OUT_DIR/$name"
  mkdir -p "$case_dir"
  REPORT="$case_dir/report.json" LOG="$case_dir/run.log" \
    DIRECT_SPARSE_ATTN=1 \
    DIRECT_SPARSE_PREFILL=0 \
    COMPACT_FLASH_ATTN=0 \
    ALLOW_COMPACT_FLASH_AUTO=0 \
    METAL_DISPATCH_LOG=1 \
    METAL_TOPK_MOE_ROUTE_FUSION=0 \
    "$ROOT/scripts/glm52-phase-b-real-indexshare-parity.sh" \
      --stage-model "$STAGE_MODEL" \
      --model-id "$MODEL_ID" \
      --skippy-bench "$SKIPPY_BENCH_BIN" \
      --direct-sparse-attn \
      --require-direct-sparse-decode-proof \
      --metal-dispatch-log \
      --no-metal-topk-moe-route-fusion \
      --out-dir "$case_dir" \
      "$@" \
      >"$case_dir/stdout.txt" \
      2>"$case_dir/stderr.txt"
}

if [[ "$QUICK" == "1" ]]; then
  run_decode_case decode-middle-quick \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 128 \
    --tokens 1 \
    --position-start 16 \
    --kv-warmup-tokens 16 \
    --kv-warmup-chunk-tokens 16 \
    --n-batch 16 \
    --n-ubatch 16
else
  run_decode_case decode-early \
    --layer-start 6 \
    --layer-end 10 \
    --ctx-size 128 \
    --tokens 1 \
    --position-start 32 \
    --kv-warmup-tokens 32 \
    --kv-warmup-chunk-tokens 16 \
    --n-batch 16 \
    --n-ubatch 16
  run_decode_case decode-middle \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 256 \
    --tokens 1 \
    --position-start 64 \
    --kv-warmup-tokens 64 \
    --kv-warmup-chunk-tokens 32 \
    --n-batch 32 \
    --n-ubatch 32
  run_decode_case decode-late \
    --layer-start 74 \
    --layer-end 78 \
    --ctx-size 512 \
    --tokens 1 \
    --position-start 128 \
    --kv-warmup-tokens 128 \
    --kv-warmup-chunk-tokens 64 \
    --n-batch 64 \
    --n-ubatch 64
fi

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
failures = []
rows = []

def load(path):
    return json.loads(path.read_text())

for case_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
    report_path = case_dir / "report.json"
    if not report_path.exists():
        failures.append(f"{case_dir.name}: missing report")
        continue
    report = load(report_path)
    comparison = report.get("comparison") or {}
    parity = comparison.get("parity") or {}
    candidate = comparison.get("candidate") or {}
    baseline = comparison.get("baseline") or {}
    guard = report.get("direct_sparse_decode_guard") or {}
    native_guard = report.get("native_indexshare_guard") or {}
    candidate_ops = candidate.get("op_timing_summary") or {}
    baseline_ops = baseline.get("op_timing_summary") or {}
    candidate_dispatch = candidate.get("metal_dispatch_summary") or {}
    baseline_timing = (baseline.get("timing_summary") or {}).get("mean_ms")
    candidate_timing = (candidate.get("timing_summary") or {}).get("mean_ms")
    row = {
        "label": case_dir.name,
        "report": str(report_path),
        "tokens": report.get("tokens"),
        "position_start": report.get("position_start"),
        "kv_warmup_tokens": report.get("kv_warmup_tokens"),
        "parity_passed": bool(parity.get("passed")),
        "hidden_mismatches": parity.get("hidden_mismatches"),
        "sideband_mismatched_bytes": parity.get("sideband_mismatched_bytes"),
        "native_indexshare_guard_passed": bool(native_guard.get("passed")),
        "direct_sparse_decode_guard_passed": bool(guard.get("passed")),
        "direct_sparse_failure_summary": guard.get("failure_summary"),
        "candidate_sparse_mask_nodes": (candidate_ops.get("sparse_mask") or {}).get("nodes"),
        "candidate_dsa_sparse_attn_nodes": (candidate_ops.get("dsa_sparse_attn") or {}).get("nodes"),
        "candidate_dsa_sparse_attn_dispatches": candidate_dispatch.get("dsa_sparse_attn_records"),
        "candidate_dense_sparse_mask_dispatches": guard.get("dense_sparse_mask_dispatches"),
        "candidate_indexer_topk_nodes": (candidate_ops.get("indexer_topk") or {}).get("nodes"),
        "candidate_indexer_nodes": (candidate_ops.get("indexer") or {}).get("nodes"),
        "candidate_top_k_nodes": (candidate_ops.get("top_k") or {}).get("nodes"),
        "baseline_sparse_mask_nodes": (baseline_ops.get("sparse_mask") or {}).get("nodes"),
        "baseline_dsa_sparse_attn_nodes": (baseline_ops.get("dsa_sparse_attn") or {}).get("nodes"),
        "baseline_mean_ms": baseline_timing,
        "candidate_mean_ms": candidate_timing,
    }
    if baseline_timing and candidate_timing:
        row["diagnostic_ratio"] = baseline_timing / candidate_timing
    rows.append(row)

    if not row["parity_passed"]:
        failures.append(f"{case_dir.name}: parity failed")
    if row["hidden_mismatches"] not in (0, None):
        failures.append(f"{case_dir.name}: hidden mismatches {row['hidden_mismatches']}")
    if row["sideband_mismatched_bytes"] not in (0, None):
        failures.append(f"{case_dir.name}: sideband mismatch {row['sideband_mismatched_bytes']}")
    if not row["native_indexshare_guard_passed"]:
        failures.append(f"{case_dir.name}: native IndexShare guard failed")
    if not row["direct_sparse_decode_guard_passed"]:
        failures.append(f"{case_dir.name}: direct sparse decode guard failed: {row['direct_sparse_failure_summary']}")
    if row["candidate_sparse_mask_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: sparse-mask nodes still present")
    if row["candidate_dense_sparse_mask_dispatches"] not in (0, None):
        failures.append(f"{case_dir.name}: dense sparse-mask dispatch still present")
    if not row["candidate_dsa_sparse_attn_nodes"]:
        failures.append(f"{case_dir.name}: missing DSA sparse attention timing nodes")
    if not row["candidate_dsa_sparse_attn_dispatches"]:
        failures.append(f"{case_dir.name}: missing DSA sparse attention Metal dispatches")
    if row["candidate_indexer_topk_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed indexer_topk")
    if row["candidate_indexer_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed indexer")
    if row["candidate_top_k_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed top_k")

summary = {
    "passed": not failures,
    "phase": "C",
    "scope": "native GLM-5.2 direct sparse decode only; compact flash, route fusion, prefill, MTP, and split work disabled",
    "rows": rows,
    "failures": failures,
}
summary_path = out_dir / "phase-c-direct-sparse-decode-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

if failures:
    print("GLM-5.2 Phase-C direct sparse decode gate FAILED", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    print(f"summary={summary_path}", file=sys.stderr)
    raise SystemExit(1)

print("GLM-5.2 Phase-C direct sparse decode gate passed")
print(f"summary={summary_path}")
for row in rows:
    ratio = row.get("diagnostic_ratio")
    ratio_text = f" ratio={ratio:.3f}x" if ratio else ""
    print(
        f"{row['label']}: pos={row['position_start']} kv_warmup={row['kv_warmup_tokens']} "
        f"sparse_mask={row['candidate_sparse_mask_nodes']} "
        f"dsa_nodes={row['candidate_dsa_sparse_attn_nodes']} "
        f"dsa_dispatches={row['candidate_dsa_sparse_attn_dispatches']}{ratio_text}"
    )
PY
