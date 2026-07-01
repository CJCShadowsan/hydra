#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-d-policy-gate}"
QUICK=0

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-d-policy-gate.sh [options]

Runs the Phase-D native GLM-5.2 DSA policy gate.

This gate assumes Phase A/B/C are already closed. It proves policy selection
across local llama.cpp layer-slice shapes:

  - decode direct sparse
  - decode compact flash for large top-k
  - short prefill direct sparse
  - long prefill correctness-preserving dense fallback
  - verification-like multi-token suffix direct sparse

Direct/compact cases also prove no sparse-mask materialization, native backend
dispatch evidence, and no Shared-layer top-k recompute. The long-prefill case
currently proves the safe policy fallback because direct sparse prefill is only
correctness-proven for small token counts.

This is not a Skippy split run and not a native MTP implementation.

Options:
  --stage-model PATH      GLM-5.2 layer package path.
  --model-id ID           Model id recorded in reports.
  --skippy-bench PATH     skippy-bench binary.
  --out-dir PATH          Artifact directory.
  --quick                 Run a smaller matrix without compact/verification cases.
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

write_expectation() {
  local case_dir="$1"
  local expected_route="$2"
  local expected_phase="$3"
  local expected_reason="$4"
  local expected_large_prefill="$5"
  python3 - "$case_dir/expected.json" "$expected_route" "$expected_phase" "$expected_reason" "$expected_large_prefill" <<'PY'
import json
import sys

path, route, phase, reason, large_prefill = sys.argv[1:6]
payload = {
    "expected_route": route,
    "expected_phase": phase,
    "expected_reason": reason,
    "expected_large_prefill": large_prefill == "1",
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

run_case() {
  local name="$1"
  local proof="$2"
  local expected_route="$3"
  local expected_phase="$4"
  local expected_reason="$5"
  local expected_large_prefill="$6"
  shift 6

  local case_dir="$OUT_DIR/$name"
  mkdir -p "$case_dir"
  write_expectation "$case_dir" "$expected_route" "$expected_phase" "$expected_reason" "$expected_large_prefill"

  local require_args=()
  local allow_compact_flash_auto=0
  local direct_sparse_prefill=0
  if [[ "$proof" == "decode" ]]; then
    require_args+=(--require-direct-sparse-decode-proof)
  elif [[ "$proof" == "compact" ]]; then
    allow_compact_flash_auto=1
    require_args+=(--allow-compact-flash-auto --require-compact-flash-proof)
  elif [[ "$proof" == "prefill" ]]; then
    direct_sparse_prefill=1
    require_args+=(--direct-sparse-prefill --require-direct-sparse-prefill-proof)
  elif [[ "$proof" == "dense-prefill" ]]; then
    direct_sparse_prefill=1
    require_args+=(--direct-sparse-prefill)
  else
    echo "unknown proof mode for $name: $proof" >&2
    exit 2
  fi

  REPORT="$case_dir/report.json" LOG="$case_dir/run.log" \
    DIRECT_SPARSE_ATTN=1 \
    DIRECT_SPARSE_PREFILL="$direct_sparse_prefill" \
    NATIVE_DEFAULT_DIRECT_SPARSE_ATTN=1 \
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
      --report "$case_dir/report.json" \
      --log "$case_dir/run.log" \
      "${require_args[@]}" \
      "$@" \
      >"$case_dir/stdout.txt" \
      2>"$case_dir/stderr.txt"
}

run_case decode-direct decode direct_sparse decode decode 0 \
  --layer-start 30 \
  --layer-end 34 \
  --ctx-size 256 \
  --tokens 1 \
  --position-start 64 \
  --kv-warmup-tokens 64 \
  --kv-warmup-chunk-tokens 64 \
  --n-batch 64 \
  --n-ubatch 64

run_case prefill-short prefill direct_sparse prefill short_prefill 0 \
  --layer-start 30 \
  --layer-end 34 \
  --ctx-size 128 \
  --tokens 8 \
  --position-start 0 \
  --kv-warmup-tokens 0 \
  --kv-warmup-chunk-tokens 8 \
  --n-batch 8 \
  --n-ubatch 8

run_case prefill-long-safe-fallback dense-prefill dense_mask other direct_sparse_disabled 0 \
  --layer-start 30 \
  --layer-end 34 \
  --ctx-size 512 \
  --tokens 64 \
  --position-start 0 \
  --kv-warmup-tokens 0 \
  --kv-warmup-chunk-tokens 64 \
  --n-batch 64 \
  --n-ubatch 64 \
  --direct-sparse-prefill-max-tokens 8 \
  --dense-sparse-mask-max-bytes 1

if [[ "$QUICK" != "1" ]]; then
  run_case decode-compact-large-topk compact compact_flash decode decode_compact_large_top_k 0 \
    --layer-start 74 \
    --layer-end 78 \
    --ctx-size 512 \
    --tokens 1 \
    --position-start 128 \
    --kv-warmup-tokens 128 \
    --kv-warmup-chunk-tokens 128 \
    --n-batch 128 \
    --n-ubatch 128

  run_case verification-like-suffix prefill direct_sparse prefill short_prefill 0 \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 512 \
    --tokens 4 \
    --position-start 128 \
    --kv-warmup-tokens 128 \
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

def count_dispatch(records, op):
    return sum(1 for record in records if isinstance(record, dict) and record.get("op") == op)

def selected_direct_record(candidate):
    records = candidate.get("direct_sparse_execution_decision_records") or []
    if records:
        return records[-1]
    records = candidate.get("direct_sparse_decision_records") or []
    return records[-1] if records else {}

def compact_reason(report):
    guard = report.get("compact_flash_guard") or {}
    return guard.get("policy_selector_reason")

for case_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
    report_path = case_dir / "report.json"
    expected_path = case_dir / "expected.json"
    if not report_path.exists():
        failures.append(f"{case_dir.name}: missing report")
        continue
    if not expected_path.exists():
        failures.append(f"{case_dir.name}: missing expected policy metadata")
        continue

    report = load(report_path)
    expected = load(expected_path)
    comparison = report.get("comparison") or {}
    parity = comparison.get("parity") or {}
    candidate = comparison.get("candidate") or {}
    baseline = comparison.get("baseline") or {}
    native_guard = report.get("native_indexshare_guard") or {}
    prefill_guard = report.get("direct_sparse_prefill_guard") or {}
    decode_guard = report.get("direct_sparse_decode_guard") or {}
    compact_guard = report.get("compact_flash_guard") or {}
    candidate_ops = candidate.get("op_timing_summary") or {}
    candidate_dispatch = candidate.get("metal_dispatch_summary") or {}
    dispatch_records = candidate.get("metal_dispatch_records") or []
    direct_record = selected_direct_record(candidate)
    baseline_timing = (baseline.get("timing_summary") or {}).get("mean_ms")
    candidate_timing = (candidate.get("timing_summary") or {}).get("mean_ms")

    route = expected["expected_route"]
    if compact_guard:
        observed_route = "compact_flash"
        observed_phase = "decode"
        observed_reason = compact_reason(report)
    elif direct_record:
        observed_route = "direct_sparse" if direct_record.get("use_direct") else "dense_mask"
        observed_phase = direct_record.get("phase")
        observed_reason = direct_record.get("selector_reason")
    else:
        observed_route = "unknown"
        observed_phase = None
        observed_reason = None

    dense_sparse_mask_dispatches = count_dispatch(dispatch_records, "dsa_sparse_mask")
    row = {
        "label": case_dir.name,
        "report": str(report_path),
        "expected_route": route,
        "observed_route": observed_route,
        "expected_phase": expected["expected_phase"],
        "observed_phase": observed_phase,
        "expected_reason": expected["expected_reason"],
        "observed_reason": observed_reason,
        "expected_large_prefill": expected["expected_large_prefill"],
        "observed_large_prefill": bool(direct_record.get("large_prefill_shape")),
        "tokens": report.get("tokens"),
        "position_start": report.get("position_start"),
        "kv_warmup_tokens": report.get("kv_warmup_tokens"),
        "parity_passed": bool(parity.get("passed")),
        "hidden_mismatches": parity.get("hidden_mismatches"),
        "sideband_mismatched_bytes": parity.get("sideband_mismatched_bytes"),
        "native_indexshare_guard_passed": bool(native_guard.get("passed")),
        "direct_sparse_decode_guard_passed": bool(decode_guard.get("passed")),
        "direct_sparse_prefill_guard_passed": bool(prefill_guard.get("passed")),
        "compact_flash_guard_passed": bool(compact_guard.get("passed")),
        "candidate_sparse_mask_nodes": (candidate_ops.get("sparse_mask") or {}).get("nodes"),
        "candidate_dsa_sparse_attn_nodes": (candidate_ops.get("dsa_sparse_attn") or {}).get("nodes"),
        "candidate_dsa_sparse_attn_dispatches": candidate_dispatch.get("dsa_sparse_attn_records"),
        "candidate_flash_attn_ext_records": candidate_dispatch.get("flash_attn_ext_records"),
        "candidate_compact_mask_omission_records": compact_guard.get("execution_mask_omission_records"),
        "candidate_omitted_mla_kq_mask_records": compact_guard.get("omitted_mla_kq_mask_records"),
        "candidate_materialized_mla_kq_mask_records": compact_guard.get("materialized_mla_kq_mask_records"),
        "candidate_dense_sparse_mask_dispatches": dense_sparse_mask_dispatches,
        "candidate_indexer_topk_nodes": (candidate_ops.get("indexer_topk") or {}).get("nodes"),
        "candidate_indexer_nodes": (candidate_ops.get("indexer") or {}).get("nodes"),
        "candidate_top_k_nodes": (candidate_ops.get("top_k") or {}).get("nodes"),
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
    if observed_route != route:
        failures.append(f"{case_dir.name}: route {observed_route!r} != expected {route!r}")
    if observed_phase != expected["expected_phase"]:
        failures.append(f"{case_dir.name}: phase {observed_phase!r} != expected {expected['expected_phase']!r}")
    if observed_reason != expected["expected_reason"]:
        failures.append(f"{case_dir.name}: reason {observed_reason!r} != expected {expected['expected_reason']!r}")
    if row["observed_large_prefill"] != row["expected_large_prefill"]:
        failures.append(f"{case_dir.name}: large_prefill {row['observed_large_prefill']} != expected {row['expected_large_prefill']}")
    if route != "dense_mask" and row["candidate_sparse_mask_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: sparse-mask nodes still present")
    if route != "dense_mask" and row["candidate_dense_sparse_mask_dispatches"] not in (0, None):
        failures.append(f"{case_dir.name}: dense sparse-mask dispatch still present")
    if route == "dense_mask" and not row["candidate_sparse_mask_nodes"]:
        failures.append(f"{case_dir.name}: expected dense sparse-mask fallback evidence")
    if row["candidate_indexer_topk_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed indexer_topk")
    if row["candidate_indexer_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed indexer")
    if row["candidate_top_k_nodes"] not in (0, None):
        failures.append(f"{case_dir.name}: candidate recomputed top_k")
    if route == "direct_sparse" and not row["candidate_dsa_sparse_attn_nodes"]:
        failures.append(f"{case_dir.name}: missing DSA sparse-attention timing nodes")
    if route == "direct_sparse" and not row["candidate_dsa_sparse_attn_dispatches"]:
        failures.append(f"{case_dir.name}: missing DSA sparse-attention dispatch evidence")
    if route == "compact_flash" and not row["compact_flash_guard_passed"]:
        failures.append(f"{case_dir.name}: compact flash proof failed: {compact_guard.get('failure_summary')}")
    if route == "compact_flash" and not row["candidate_flash_attn_ext_records"]:
        failures.append(f"{case_dir.name}: missing compact flash dispatch evidence")
    if route == "compact_flash" and not row["candidate_compact_mask_omission_records"]:
        failures.append(f"{case_dir.name}: missing compact MLA KQ mask-omission evidence")
    if route == "compact_flash" and row["candidate_materialized_mla_kq_mask_records"] not in (0, None):
        failures.append(f"{case_dir.name}: compact flash still materialized MLA KQ mask")

summary = {
    "passed": not failures,
    "phase": "D",
    "scope": "native GLM-5.2 DSA policy matrix for decode, compact decode, short prefill, long-prefill safe fallback, and verification-like suffix shapes",
    "rows": rows,
    "failures": failures,
}
summary_path = out_dir / "phase-d-policy-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

if failures:
    print("GLM-5.2 Phase-D policy gate FAILED", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    print(f"summary={summary_path}", file=sys.stderr)
    raise SystemExit(1)

print("GLM-5.2 Phase-D policy gate passed")
print(f"summary={summary_path}")
for row in rows:
    ratio = row.get("diagnostic_ratio")
    ratio_text = f" ratio={ratio:.3f}x" if ratio else ""
    print(
        f"{row['label']}: route={row['observed_route']} phase={row['observed_phase']} "
        f"reason={row['observed_reason']} sparse_mask={row['candidate_sparse_mask_nodes']} "
        f"dsa_dispatches={row['candidate_dsa_sparse_attn_dispatches']} "
        f"flash_dispatches={row['candidate_flash_attn_ext_records']}{ratio_text}"
    )
PY
