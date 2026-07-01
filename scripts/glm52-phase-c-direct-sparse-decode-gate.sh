#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-c-direct-sparse-decode-gate}"
ITERATIONS="${ITERATIONS:-3}"
WARMUP="${WARMUP:-1}"
QUICK=0
LONG_KV=0
LONG_KV_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-c-direct-sparse-decode-gate.sh [options]

Runs the strict Phase-C decode gate for native GLM-5.2 sparse decode policy.

This gate assumes Phase A and Phase B are already closed. It proves decode only:

  - direct sparse decode decisions are selected for the proven small-top-k shape;
  - compact K/V gather + flash attention is selected for the large-top-k shape;
  - sparse-mask timing nodes are absent in the candidate;
  - dense sparse-mask Metal dispatches are absent in the candidate;
  - native DSA sparse-attention or compact-flash execution evidence is present;
  - parity still holds against the dense/direct producer baseline;
  - Shared consumers still reuse Full top-k sideband without recomputing top-k.

This is not a prefill policy gate, not an MTP gate, and not a Skippy split run.

Options:
  --stage-model PATH      GLM-5.2 layer package path.
  --model-id ID           Model id recorded in reports.
  --skippy-bench PATH     skippy-bench binary.
  --out-dir PATH          Artifact directory.
  --iterations N          Measured iterations per case. Default: 3
  --warmup N              Warmup iterations per case. Default: 1
  --quick                 Run one reduced middle-span smoke case.
  --long-kv               Run larger-KV decode cases after the normal sweep.
  --long-kv-only          Run only the larger-KV decode cases.
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
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    --long-kv)
      LONG_KV=1
      shift
      ;;
    --long-kv-only)
      LONG_KV=1
      LONG_KV_ONLY=1
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
  local proof_mode="$2"
  shift 2
  local require_args=()
  local allow_compact_flash_auto=0
  if [[ "$proof_mode" == "direct" ]]; then
    require_args+=(--require-direct-sparse-decode-proof)
  elif [[ "$proof_mode" == "compact" ]]; then
    allow_compact_flash_auto=1
    require_args+=(--allow-compact-flash-auto --require-compact-flash-proof)
  else
    echo "unknown proof mode for $name: $proof_mode" >&2
    exit 2
  fi
  local case_dir="$OUT_DIR/$name"
  mkdir -p "$case_dir"
  REPORT="$case_dir/report.json" LOG="$case_dir/run.log" \
    DIRECT_SPARSE_ATTN=1 \
    NATIVE_DEFAULT_DIRECT_SPARSE_ATTN=1 \
    DIRECT_SPARSE_PREFILL=0 \
    COMPACT_FLASH_ATTN=0 \
    ALLOW_COMPACT_FLASH_AUTO="$allow_compact_flash_auto" \
    METAL_DISPATCH_LOG=1 \
    METAL_TOPK_MOE_ROUTE_FUSION=0 \
    "$ROOT/scripts/glm52-phase-b-real-indexshare-parity.sh" \
      --stage-model "$STAGE_MODEL" \
      --model-id "$MODEL_ID" \
      --skippy-bench "$SKIPPY_BENCH_BIN" \
      --native-default-direct-sparse-attn \
      --metal-dispatch-log \
      --no-metal-topk-moe-route-fusion \
      --out-dir "$case_dir" \
      --iterations "$ITERATIONS" \
      --warmup "$WARMUP" \
      "${require_args[@]}" \
      "$@" \
      >"$case_dir/stdout.txt" \
      2>"$case_dir/stderr.txt"
}

if [[ "$QUICK" == "1" ]]; then
  run_decode_case decode-middle-quick direct \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 128 \
    --tokens 1 \
    --position-start 16 \
    --kv-warmup-tokens 16 \
    --kv-warmup-chunk-tokens 16 \
    --n-batch 16 \
    --n-ubatch 16
elif [[ "$LONG_KV_ONLY" != "1" ]]; then
  run_decode_case decode-early direct \
    --layer-start 6 \
    --layer-end 10 \
    --ctx-size 128 \
    --tokens 1 \
    --position-start 32 \
    --kv-warmup-tokens 32 \
    --kv-warmup-chunk-tokens 32 \
    --n-batch 32 \
    --n-ubatch 32
  run_decode_case decode-middle direct \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 256 \
    --tokens 1 \
    --position-start 64 \
    --kv-warmup-tokens 64 \
    --kv-warmup-chunk-tokens 64 \
    --n-batch 64 \
    --n-ubatch 64
  run_decode_case decode-late compact \
    --layer-start 74 \
    --layer-end 78 \
    --ctx-size 512 \
    --tokens 1 \
    --position-start 128 \
    --kv-warmup-tokens 128 \
    --kv-warmup-chunk-tokens 128 \
    --n-batch 128 \
    --n-ubatch 128
fi

if [[ "$LONG_KV" == "1" ]]; then
  run_decode_case decode-middle-kv512 direct \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 1024 \
    --tokens 1 \
    --position-start 512 \
    --kv-warmup-tokens 512 \
    --kv-warmup-chunk-tokens 128 \
    --n-batch 128 \
    --n-ubatch 128
  run_decode_case decode-late-kv1024 compact \
    --layer-start 74 \
    --layer-end 78 \
    --ctx-size 2048 \
    --tokens 1 \
    --position-start 1024 \
    --kv-warmup-tokens 1024 \
    --kv-warmup-chunk-tokens 128 \
    --n-batch 128 \
    --n-ubatch 128
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
    compact_guard = report.get("compact_flash_guard") or {}
    native_guard = report.get("native_indexshare_guard") or {}
    candidate_ops = candidate.get("op_timing_summary") or {}
    baseline_ops = baseline.get("op_timing_summary") or {}
    candidate_dispatch = candidate.get("metal_dispatch_summary") or {}
    candidate_dispatch_records = candidate.get("metal_dispatch_records") or []
    dense_sparse_mask_dispatches = guard.get("dense_sparse_mask_dispatches")
    if dense_sparse_mask_dispatches is None:
        dense_sparse_mask_dispatches = sum(
            1
            for record in candidate_dispatch_records
            if isinstance(record, dict) and record.get("op") == "dsa_sparse_mask"
        )
    baseline_timing = (baseline.get("timing_summary") or {}).get("mean_ms")
    candidate_timing = (candidate.get("timing_summary") or {}).get("mean_ms")
    proof_kind = "compact_flash" if compact_guard else "direct_sparse"
    row = {
        "label": case_dir.name,
        "report": str(report_path),
        "proof_kind": proof_kind,
        "tokens": report.get("tokens"),
        "position_start": report.get("position_start"),
        "kv_warmup_tokens": report.get("kv_warmup_tokens"),
        "parity_passed": bool(parity.get("passed")),
        "hidden_mismatches": parity.get("hidden_mismatches"),
        "sideband_mismatched_bytes": parity.get("sideband_mismatched_bytes"),
        "native_indexshare_guard_passed": bool(native_guard.get("passed")),
        "direct_sparse_decode_guard_passed": bool(guard.get("passed")),
        "direct_sparse_failure_summary": guard.get("failure_summary"),
        "compact_flash_guard_passed": bool(compact_guard.get("passed")),
        "compact_flash_failure_summary": compact_guard.get("failure_summary"),
        "compact_flash_selector_reason": compact_guard.get("policy_selector_reason"),
        "candidate_sparse_mask_nodes": (candidate_ops.get("sparse_mask") or {}).get("nodes"),
        "candidate_dsa_sparse_attn_nodes": (candidate_ops.get("dsa_sparse_attn") or {}).get("nodes"),
        "candidate_dsa_sparse_attn_dispatches": candidate_dispatch.get("dsa_sparse_attn_records"),
        "candidate_dense_sparse_mask_dispatches": dense_sparse_mask_dispatches,
        "candidate_flash_attn_ext_records": compact_guard.get("flash_attn_ext_records"),
        "candidate_compact_get_rows_records": compact_guard.get("compact_get_rows_records"),
        "candidate_dsa_compact_get_rows_fused_records": compact_guard.get("dsa_compact_get_rows_fused_records"),
        "candidate_compact_get_rows_nodes": (candidate_ops.get("compact_get_rows") or {}).get("nodes"),
        "candidate_compact_get_rows_us": (candidate_ops.get("compact_get_rows") or {}).get("elapsed_us"),
        "candidate_compact_get_rows_share_of_total": candidate_ops.get("compact_get_rows_share_of_total"),
        "candidate_compact_mask_omission_records": compact_guard.get("execution_mask_omission_records"),
        "candidate_omitted_mla_kq_mask_records": compact_guard.get("omitted_mla_kq_mask_records"),
        "candidate_materialized_mla_kq_mask_records": compact_guard.get("materialized_mla_kq_mask_records"),
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
    if proof_kind == "direct_sparse" and not row["direct_sparse_decode_guard_passed"]:
        failures.append(f"{case_dir.name}: direct sparse decode guard failed: {row['direct_sparse_failure_summary']}")
    if proof_kind == "compact_flash" and not row["compact_flash_guard_passed"]:
        failures.append(f"{case_dir.name}: compact flash guard failed: {row['compact_flash_failure_summary']}")
    if row["candidate_sparse_mask_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: sparse-mask nodes still present")
    if row["candidate_dense_sparse_mask_dispatches"] not in (0, None):
        failures.append(f"{case_dir.name}: dense sparse-mask dispatch still present")
    if proof_kind == "direct_sparse" and not row["candidate_dsa_sparse_attn_nodes"]:
        failures.append(f"{case_dir.name}: missing DSA sparse attention timing nodes")
    if proof_kind == "direct_sparse" and not row["candidate_dsa_sparse_attn_dispatches"]:
        failures.append(f"{case_dir.name}: missing DSA sparse attention Metal dispatches")
    if proof_kind == "compact_flash" and not row["candidate_flash_attn_ext_records"]:
        failures.append(f"{case_dir.name}: missing compact flash attention dispatch evidence")
    if proof_kind == "compact_flash" and not row["candidate_compact_mask_omission_records"]:
        failures.append(f"{case_dir.name}: missing compact MLA KQ mask-omission evidence")
    if proof_kind == "compact_flash" and row["candidate_materialized_mla_kq_mask_records"] not in (0, None):
        failures.append(f"{case_dir.name}: compact flash still materialized MLA KQ mask")
    if proof_kind == "compact_flash" and not (
        row["candidate_compact_get_rows_records"]
        or row["candidate_dsa_compact_get_rows_fused_records"]
    ):
        failures.append(f"{case_dir.name}: missing compact K/V gather evidence")
    if (
        proof_kind == "compact_flash"
        and row["candidate_compact_get_rows_records"]
        and not row["candidate_compact_get_rows_nodes"]
    ):
        failures.append(f"{case_dir.name}: missing compact K/V gather timing nodes")
    if row["candidate_indexer_topk_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed indexer_topk")
    if row["candidate_indexer_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed indexer")
    if row["candidate_top_k_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed top_k")

summary = {
    "passed": not failures,
    "phase": "C",
    "scope": "native GLM-5.2 sparse decode policy; direct sparse for small top-k, compact flash for large top-k; dense sparse masks, route fusion, prefill, MTP, and split work disabled",
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
        f"proof={row['proof_kind']} "
        f"sparse_mask={row['candidate_sparse_mask_nodes']} "
        f"dsa_nodes={row['candidate_dsa_sparse_attn_nodes']} "
        f"dsa_dispatches={row['candidate_dsa_sparse_attn_dispatches']} "
        f"flash_dispatches={row['candidate_flash_attn_ext_records']} "
        f"compact_get_rows_nodes={row['candidate_compact_get_rows_nodes']} "
        f"compact_get_rows_us={row['candidate_compact_get_rows_us']}{ratio_text}"
    )
PY
