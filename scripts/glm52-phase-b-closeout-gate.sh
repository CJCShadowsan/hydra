#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_MODEL_PACKAGE_BIN="${SKIPPY_MODEL_PACKAGE_BIN:-$ROOT/target/debug/skippy-model-package}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-b-closeout-gate}"
RUN_TINY="${RUN_TINY:-1}"
QUICK=0

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-b-closeout-gate.sh [options]

Runs the strict Phase-B closeout gate for GLM-5.2 native IndexShare:

  1. Tiny fresh fixture proof with Full -> Shared parity and Shared-only failure.
  2. Real GLM-5.2 representative layer matrix.
  3. Real warmed decode cases with KV prefix and chunked warmup.
  4. Real prefill-shaped multi-token case.

This gate deliberately disables direct sparse attention and compact flash. It
closes the Full/Shared IndexShare foundation only; it is not a Phase-C sparse
attention performance run.

Options:
  --stage-model PATH      GLM-5.2 layer package path.
  --model-id ID           Model id recorded in reports.
  --skippy-model-package PATH
                           skippy-model-package binary.
  --skippy-bench PATH     skippy-bench binary.
  --out-dir PATH          Artifact directory.
  --skip-tiny             Skip the tiny fresh fixture proof.
  --quick                 Run a reduced smoke version of the gate.
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
    --skippy-model-package)
      SKIPPY_MODEL_PACKAGE_BIN="$2"
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
    --skip-tiny)
      RUN_TINY=0
      shift
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

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "required executable not found: $path" >&2
    exit 1
  fi
}

if [[ ! -d "$STAGE_MODEL" ]]; then
  echo "stage model package not found: $STAGE_MODEL" >&2
  exit 1
fi
require_executable "$SKIPPY_MODEL_PACKAGE_BIN"
require_executable "$SKIPPY_BENCH_BIN"
require_executable "$ROOT/scripts/glm-dsa-phase-b-indexshare-parity.sh"
require_executable "$ROOT/scripts/glm52-phase-ab-real-indexshare-matrix.sh"
require_executable "$ROOT/scripts/glm52-phase-b-real-indexshare-parity.sh"

mkdir -p "$OUT_DIR"

run_phase_b_parity_case() {
  local name="$1"
  shift
  local case_dir="$OUT_DIR/$name"
  mkdir -p "$case_dir"
  REPORT="$case_dir/report.json" LOG="$case_dir/run.log" \
    DIRECT_SPARSE_ATTN=0 \
    DIRECT_SPARSE_PREFILL=0 \
    COMPACT_FLASH_ATTN=0 \
    ALLOW_COMPACT_FLASH_AUTO=0 \
    "$ROOT/scripts/glm52-phase-b-real-indexshare-parity.sh" \
      --stage-model "$STAGE_MODEL" \
      --model-id "$MODEL_ID" \
      --skippy-bench "$SKIPPY_BENCH_BIN" \
      --out-dir "$case_dir" \
      "$@" \
      >"$case_dir/stdout.txt" \
      2>"$case_dir/stderr.txt"
}

TINY_REPORT=""
if [[ "$RUN_TINY" == "1" ]]; then
  tiny_dir="$OUT_DIR/tiny-fresh"
  tiny_roles="full,full,shared"
  mkdir -p "$tiny_dir"
  WORK_DIR="$tiny_dir/work" \
    REPORT="$tiny_dir/report.json" \
    SKIPPY_MODEL_PACKAGE_BIN="$SKIPPY_MODEL_PACKAGE_BIN" \
    SKIPPY_BENCH_BIN="$SKIPPY_BENCH_BIN" \
    INDEXER_TYPES="$tiny_roles" \
    "$ROOT/scripts/glm-dsa-phase-b-indexshare-parity.sh" \
      --skip-build \
      >"$tiny_dir/stdout.txt" \
      2>"$tiny_dir/stderr.txt"
  TINY_REPORT="$tiny_dir/report.json"
fi

matrix_dir="$OUT_DIR/real-matrix"
matrix_spans="2:6:initial,6:10:early,30:34:middle,74:78:late"
if [[ "$QUICK" == "1" ]]; then
  matrix_spans="6:8:early-quick"
fi
OUT_DIR="$matrix_dir" \
  STAGE_MODEL="$STAGE_MODEL" \
  MODEL_ID="$MODEL_ID" \
  SKIPPY_MODEL_PACKAGE_BIN="$SKIPPY_MODEL_PACKAGE_BIN" \
  SKIPPY_BENCH_BIN="$SKIPPY_BENCH_BIN" \
  SPANS="$matrix_spans" \
  CTX_SIZE=128 \
  TOKENS=1 \
  POSITION_START=0 \
  N_BATCH=1 \
  N_UBATCH=1 \
  "$ROOT/scripts/glm52-phase-ab-real-indexshare-matrix.sh" \
    >"$OUT_DIR/real-matrix.stdout" \
    2>"$OUT_DIR/real-matrix.stderr"

if [[ "$QUICK" == "1" ]]; then
  run_phase_b_parity_case warmed-decode-quick \
    --layer-start 6 \
    --layer-end 8 \
    --ctx-size 64 \
    --tokens 1 \
    --position-start 8 \
    --kv-warmup-tokens 8 \
    --kv-warmup-chunk-tokens 8 \
    --n-batch 8 \
    --n-ubatch 8
  run_phase_b_parity_case prefill-shaped-quick \
    --layer-start 6 \
    --layer-end 8 \
    --ctx-size 64 \
    --tokens 2 \
    --position-start 0 \
    --n-batch 2 \
    --n-ubatch 2
else
  run_phase_b_parity_case warmed-decode-middle \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 256 \
    --tokens 1 \
    --position-start 64 \
    --kv-warmup-tokens 64 \
    --kv-warmup-chunk-tokens 32 \
    --n-batch 32 \
    --n-ubatch 32
  run_phase_b_parity_case warmed-decode-late \
    --layer-start 74 \
    --layer-end 78 \
    --ctx-size 512 \
    --tokens 1 \
    --position-start 128 \
    --kv-warmup-tokens 128 \
    --kv-warmup-chunk-tokens 64 \
    --n-batch 64 \
    --n-ubatch 64
  run_phase_b_parity_case prefill-shaped-middle \
    --layer-start 30 \
    --layer-end 34 \
    --ctx-size 128 \
    --tokens 16 \
    --position-start 0 \
    --n-batch 16 \
    --n-ubatch 16
fi

python3 - "$OUT_DIR" "$TINY_REPORT" "$matrix_dir/matrix.json" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
tiny_report = pathlib.Path(sys.argv[2]) if sys.argv[2] else None
matrix_path = pathlib.Path(sys.argv[3])

failures = []

def load(path):
    return json.loads(path.read_text())

def validate_report(label, path, require_prefill_shape=False, require_warmup=False):
    report = load(path)
    comparison = report.get("comparison") or {}
    parity = comparison.get("parity") or {}
    guard = report.get("native_indexshare_guard") or {}
    candidate = comparison.get("candidate") or {}
    candidate_ops = candidate.get("op_timing_summary") or {}
    candidate_indexshare = candidate.get("indexshare_timing_summary") or {}
    row = {
        "label": label,
        "report": str(path),
        "tokens": report.get("tokens"),
        "position_start": report.get("position_start"),
        "kv_warmup_tokens": report.get("kv_warmup_tokens"),
        "kv_warmup_chunk_tokens": report.get("kv_warmup_chunk_tokens"),
        "parity_passed": bool(parity.get("passed")),
        "hidden_mismatches": parity.get("hidden_mismatches"),
        "sideband_mismatched_bytes": parity.get("sideband_mismatched_bytes"),
        "guard_passed": bool(guard.get("passed")),
        "full_layers": guard.get("full_layers"),
        "shared_layers": guard.get("shared_layers"),
        "shared_exec_with_input_top_k": guard.get("shared_exec_with_input_top_k"),
        "shared_exec_missing_input_top_k": guard.get("shared_exec_missing_input_top_k"),
        "candidate_indexer_topk_nodes": (candidate_ops.get("indexer_topk") or {}).get("nodes"),
        "candidate_indexer_nodes": (candidate_ops.get("indexer") or {}).get("nodes"),
        "candidate_top_k_nodes": (candidate_ops.get("top_k") or {}).get("nodes"),
        "candidate_producer_groups": candidate_indexshare.get("producer_groups"),
    }
    if not row["parity_passed"]:
        failures.append(f"{label}: parity failed")
    if row["hidden_mismatches"] not in (0, None):
        failures.append(f"{label}: hidden mismatches {row['hidden_mismatches']}")
    if row["sideband_mismatched_bytes"] not in (0, None):
        failures.append(f"{label}: sideband mismatch {row['sideband_mismatched_bytes']}")
    if not row["guard_passed"]:
        failures.append(f"{label}: native IndexShare guard failed")
    if row["shared_exec_missing_input_top_k"] not in (0, None):
        failures.append(f"{label}: shared consumer missed top-k")
    if row["candidate_indexer_topk_nodes"] not in (0, None):
        failures.append(f"{label}: candidate recomputed indexer_topk")
    if row["candidate_indexer_nodes"] not in (0, None):
        failures.append(f"{label}: candidate recomputed indexer")
    if row["candidate_top_k_nodes"] not in (0, None):
        failures.append(f"{label}: candidate recomputed top_k")
    if row["candidate_producer_groups"] not in (0, None):
        failures.append(f"{label}: candidate unexpectedly ran producer groups")
    if require_prefill_shape and (row["tokens"] or 0) <= 1:
        failures.append(f"{label}: expected multi-token prefill-shaped run")
    if require_warmup and (row["kv_warmup_tokens"] or 0) <= 0:
        failures.append(f"{label}: expected warmed KV-prefix run")
    return row

matrix = load(matrix_path)
matrix_rows = matrix.get("rows") or []
if not matrix.get("passed"):
    failures.append(f"real matrix failed: {matrix.get('failures')}")
if matrix.get("negative_missing_top_k", {}).get("status") not in ("passed", "skipped"):
    failures.append("real matrix Shared-only negative case did not pass")

case_rows = []
for case_dir in sorted(out_dir.glob("warmed-decode-*")):
    case_rows.append(validate_report(case_dir.name, case_dir / "report.json", require_warmup=True))
for case_dir in sorted(out_dir.glob("prefill-shaped-*")):
    case_rows.append(validate_report(case_dir.name, case_dir / "report.json", require_prefill_shape=True))

summary = {
    "passed": not failures,
    "phase": "B",
    "scope": "native GLM-5.2 Full/Shared IndexShare only; sparse/compact paths disabled",
    "tiny_report": str(tiny_report) if tiny_report else None,
    "matrix": {
        "path": str(matrix_path),
        "rows": matrix_rows,
        "negative_missing_top_k": matrix.get("negative_missing_top_k"),
    },
    "cases": case_rows,
    "failures": failures,
}
summary_path = out_dir / "phase-b-closeout-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

if failures:
    print("GLM-5.2 Phase-B closeout FAILED", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    print(f"summary={summary_path}", file=sys.stderr)
    raise SystemExit(1)

print("GLM-5.2 Phase-B closeout passed")
print(f"summary={summary_path}")
print(f"matrix={matrix_path}")
for row in matrix_rows:
    ratio = row.get("diagnostic_ratio")
    ratio_text = f" ratio={ratio:.3f}x" if ratio else ""
    print(
        f"matrix:{row.get('label')} full={row.get('full_layers')} "
        f"shared={row.get('shared_layers')} topk_recompute={row.get('candidate_indexer_topk_nodes')}{ratio_text}"
    )
for row in case_rows:
    print(
        f"case:{row['label']} tokens={row['tokens']} position={row['position_start']} "
        f"kv_warmup={row['kv_warmup_tokens']} topk_recompute={row['candidate_indexer_topk_nodes']}"
    )
PY
