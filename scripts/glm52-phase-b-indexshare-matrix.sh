#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp}"
GROUP_SPECS="${GROUP_SPECS:-early:2:6 middle:38:42 late:74:78}"
CTX_SIZE="${CTX_SIZE:-128}"
TOKENS="${TOKENS:-1}"
POSITION_START="${POSITION_START:-16}"
ITERATIONS="${ITERATIONS:-1}"
WARMUP="${WARMUP:-0}"
N_BATCH="${N_BATCH:-16}"
N_UBATCH="${N_UBATCH:-16}"
KV_WARMUP_TOKENS="${KV_WARMUP_TOKENS:-16}"
KV_WARMUP_CHUNK_TOKENS="${KV_WARMUP_CHUNK_TOKENS:-16}"
REPORT="${REPORT:-}"

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-b-indexshare-matrix.sh [options]

Runs the GLM-5.2 Phase-B Full/Shared IndexShare proof across representative
local layer groups. This is local llama.cpp/skippy-bench validation only; it
does not start split serving, lab nodes, mesh networking, or topology code.

Options:
  --stage-model PATH       Layer package path.
  --model-id ID            Model id recorded in reports.
  --skippy-bench PATH      skippy-bench binary. Default: target/debug/skippy-bench
  --groups SPEC            Space-separated name:start:end groups.
                           Default: "early:2:6 middle:38:42 late:74:78"
  --ctx-size N             Context size. Default: 128
  --tokens N               Tokens. Default: 1
  --position-start N       Decode position start. Default: 16
  --iterations N           Iterations. Default: 1
  --warmup N               Warmup iterations. Default: 0
  --n-batch N              Batch size. Default: 16
  --n-ubatch N             Microbatch size. Default: 16
  --kv-warmup-tokens N     KV prefix tokens. Default: 16
  --kv-warmup-chunk-tokens N
                            KV warmup chunk size. Default: 16
  --out-dir PATH           Report/log directory. Default: /tmp
  --report PATH            Matrix JSON report path.
  -h, --help               Show this help.

Environment overrides mirror the upper-case option names. Use GROUP_SPECS for
the group list; GROUPS is a special bash variable and is intentionally not used.
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
    --groups)
      GROUP_SPECS="$2"
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
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
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
RUN_DIR="$OUT_DIR/glm52-phase-b-indexshare-matrix-$STAMP"
mkdir -p "$RUN_DIR"
REPORT="${REPORT:-$RUN_DIR/matrix.json}"
SUMMARY_INPUT="$RUN_DIR/cases.tsv"
: >"$SUMMARY_INPUT"

run_case() {
  local name="$1"
  local layer_start="$2"
  local layer_end="$3"
  local report="$RUN_DIR/${name}-l${layer_start}-${layer_end}.json"
  local log="$RUN_DIR/${name}-l${layer_start}-${layer_end}.log"

  printf '== Phase B matrix case %s %s..%s ==\n' "$name" "$layer_start" "$layer_end"
  SKIPPY_BENCH_BIN="$SKIPPY_BENCH_BIN" \
  STAGE_MODEL="$STAGE_MODEL" \
  MODEL_ID="$MODEL_ID" \
  CTX_SIZE="$CTX_SIZE" \
  TOKENS="$TOKENS" \
  POSITION_START="$POSITION_START" \
  ITERATIONS="$ITERATIONS" \
  WARMUP="$WARMUP" \
  N_BATCH="$N_BATCH" \
  N_UBATCH="$N_UBATCH" \
  KV_WARMUP_TOKENS="$KV_WARMUP_TOKENS" \
  KV_WARMUP_CHUNK_TOKENS="$KV_WARMUP_CHUNK_TOKENS" \
    "$ROOT/scripts/glm52-phase-b-real-indexshare-parity.sh" \
      --layer-start "$layer_start" \
      --layer-end "$layer_end" \
      --report "$report" \
      --log "$log"

  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$layer_start" "$layer_end" "$report" "$log" >>"$SUMMARY_INPUT"
}

for spec in $GROUP_SPECS; do
  IFS=: read -r name layer_start layer_end extra <<<"$spec"
  if [[ -z "${name:-}" || -z "${layer_start:-}" || -z "${layer_end:-}" || -n "${extra:-}" ]]; then
    echo "invalid group spec: $spec; expected name:start:end" >&2
    exit 2
  fi
  run_case "$name" "$layer_start" "$layer_end"
done

python3 - "$SUMMARY_INPUT" "$REPORT" <<'PY'
import json
import pathlib
import statistics
import sys

cases_tsv = pathlib.Path(sys.argv[1])
report_path = pathlib.Path(sys.argv[2])

cases = []
failures = []
for line in cases_tsv.read_text().splitlines():
    name, layer_start, layer_end, report, log = line.split("\t")
    report_path_case = pathlib.Path(report)
    with report_path_case.open() as f:
        data = json.load(f)
    comparison = data.get("comparison") or {}
    parity = comparison.get("parity") or {}
    sensitivity = comparison.get("sideband_sensitivity") or {}
    guard = data.get("native_indexshare_guard") or {}
    baseline = comparison.get("baseline") or {}
    candidate = comparison.get("candidate") or {}
    candidate_ops = candidate.get("op_timing_summary") or {}
    baseline_ms = (baseline.get("timing_summary") or {}).get("mean_ms")
    candidate_ms = (candidate.get("timing_summary") or {}).get("mean_ms")
    ratio = baseline_ms / candidate_ms if baseline_ms and candidate_ms else None
    case = {
        "name": name,
        "layer_start": int(layer_start),
        "layer_end": int(layer_end),
        "report": report,
        "log": log,
        "parity_passed": bool(parity.get("passed")),
        "sideband_sensitivity_passed": bool(sensitivity.get("passed")),
        "sideband_poison_changed_i32": ((sensitivity.get("poison") or {}).get("changed_i32_count")),
        "poisoned_hidden_mismatches": sensitivity.get("poisoned_hidden_mismatches"),
        "shared_exec_with_input_top_k": guard.get("shared_exec_with_input_top_k"),
        "shared_exec_missing_input_top_k": guard.get("shared_exec_missing_input_top_k"),
        "candidate_indexer_nodes": (candidate_ops.get("indexer") or {}).get("nodes"),
        "candidate_top_k_nodes": (candidate_ops.get("top_k") or {}).get("nodes"),
        "candidate_sparse_mask_nodes": (candidate_ops.get("sparse_mask") or {}).get("nodes"),
        "baseline_mean_ms": baseline_ms,
        "candidate_mean_ms": candidate_ms,
        "diagnostic_ratio": ratio,
    }
    if not case["parity_passed"]:
        failures.append(f"{name}: parity failed")
    if not case["sideband_sensitivity_passed"]:
        failures.append(f"{name}: sideband sensitivity failed")
    if case["shared_exec_missing_input_top_k"] not in (0, None):
        failures.append(f"{name}: Shared execution missed top-k")
    if case["candidate_indexer_nodes"] not in (0, None):
        failures.append(f"{name}: candidate ran indexer nodes")
    if case["candidate_top_k_nodes"] not in (0, None):
        failures.append(f"{name}: candidate ran top-k nodes")
    cases.append(case)

ratios = [case["diagnostic_ratio"] for case in cases if case["diagnostic_ratio"]]
summary = {
    "passed": not failures,
    "case_count": len(cases),
    "failure_summary": "none" if not failures else ",".join(failures),
    "min_diagnostic_ratio": min(ratios) if ratios else None,
    "mean_diagnostic_ratio": statistics.mean(ratios) if ratios else None,
    "max_diagnostic_ratio": max(ratios) if ratios else None,
    "cases": cases,
}
report_path.write_text(json.dumps(summary, indent=2) + "\n")

print("GLM-5.2 Phase-B IndexShare matrix " + ("passed" if summary["passed"] else "FAILED"))
print(f"report={report_path}")
print(f"case_count={summary['case_count']}")
print(f"min_diagnostic_ratio={summary['min_diagnostic_ratio']}")
print(f"mean_diagnostic_ratio={summary['mean_diagnostic_ratio']}")
print(f"max_diagnostic_ratio={summary['max_diagnostic_ratio']}")
for case in cases:
    print(
        f"{case['name']}={case['layer_start']}..{case['layer_end']} "
        f"parity={case['parity_passed']} sensitivity={case['sideband_sensitivity_passed']} "
        f"ratio={case['diagnostic_ratio']} sparse_mask_nodes={case['candidate_sparse_mask_nodes']}"
    )
if failures:
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    sys.exit(1)
PY
