#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-d-compact-sweep}"
WINDOWS="${WINDOWS:-30:34,50:54,74:78}"
POSITIONS="${POSITIONS:-4096,8192,32768,65536}"
ITERATIONS="${ITERATIONS:-2}"
WARMUP="${WARMUP:-1}"
CTX_SIZE="${CTX_SIZE:-131072}"
TOKENS="${TOKENS:-1}"
N_BATCH="${N_BATCH:-512}"
N_UBATCH="${N_UBATCH:-512}"
SYNTHETIC_KV=1
MIN_DELTA_PERCENT=""
QUICK=0

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-d-compact-sweep.sh [options]

Runs a Phase-D GLM-5.2 compact-flash policy sweep. Each case compares dense
fallback against compact selected-KV flash attention for one layer window and
decode position, then writes per-case JSON plus aggregate results.

The default mode uses synthetic native KV warmup. That keeps long-context policy
checks cheap and relies on the GLM-DSA native KV-page path.

Options:
  --stage-model PATH           GLM-5.2 layer package path.
  --skippy-bench PATH          skippy-bench binary. Default: target/debug/skippy-bench
  --out-dir PATH               Artifact directory.
  --windows CSV                Window CSV, e.g. 30:34,50:54,74:78.
  --positions CSV              Decode position CSV. Default: 4096,8192,32768,65536.
  --iterations N               Measured iterations per case. Default: 2.
  --warmup N                   Warmup iterations per case. Default: 1.
  --ctx-size N                 Context size. Default: 131072.
  --tokens N                   Decode tokens. Default: 1.
  --n-batch N                  llama n_batch. Default: 512.
  --n-ubatch N                 llama n_ubatch. Default: 512.
  --real-kv-warmup             Use real KV warmup instead of synthetic native KV import.
  --min-delta-percent N        Fail if any successful case is below this compact-vs-dense delta.
  --quick                      Alias for --windows 50:54 --positions 4096.
  -h, --help                   Show this help.

Environment overrides mirror upper-case option names.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-model)
      STAGE_MODEL="${2:?missing --stage-model value}"
      shift 2
      ;;
    --skippy-bench)
      SKIPPY_BENCH_BIN="${2:?missing --skippy-bench value}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing --out-dir value}"
      shift 2
      ;;
    --windows)
      WINDOWS="${2:?missing --windows value}"
      shift 2
      ;;
    --positions)
      POSITIONS="${2:?missing --positions value}"
      shift 2
      ;;
    --iterations)
      ITERATIONS="${2:?missing --iterations value}"
      shift 2
      ;;
    --warmup)
      WARMUP="${2:?missing --warmup value}"
      shift 2
      ;;
    --ctx-size)
      CTX_SIZE="${2:?missing --ctx-size value}"
      shift 2
      ;;
    --tokens)
      TOKENS="${2:?missing --tokens value}"
      shift 2
      ;;
    --n-batch)
      N_BATCH="${2:?missing --n-batch value}"
      shift 2
      ;;
    --n-ubatch)
      N_UBATCH="${2:?missing --n-ubatch value}"
      shift 2
      ;;
    --real-kv-warmup)
      SYNTHETIC_KV=0
      shift
      ;;
    --min-delta-percent)
      MIN_DELTA_PERCENT="${2:?missing --min-delta-percent value}"
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

if [[ "$QUICK" == "1" ]]; then
  WINDOWS="50:54"
  POSITIONS="4096"
fi

if [[ ! -x "$SKIPPY_BENCH_BIN" ]]; then
  echo "skippy-bench binary not executable: $SKIPPY_BENCH_BIN" >&2
  exit 1
fi
if [[ ! -d "$STAGE_MODEL" ]]; then
  echo "stage model package not found: $STAGE_MODEL" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
: >"$OUT_DIR/results.tsv"

IFS=',' read -r -a WINDOW_LIST <<<"$WINDOWS"
IFS=',' read -r -a POSITION_LIST <<<"$POSITIONS"

case_paths=()

for window in "${WINDOW_LIST[@]}"; do
  IFS=':' read -r layer_start layer_end <<<"$window"
  if [[ -z "${layer_start:-}" || -z "${layer_end:-}" ]]; then
    echo "invalid window: $window" >&2
    exit 2
  fi
  for position in "${POSITION_LIST[@]}"; do
    case_dir="$OUT_DIR/window${layer_start}-${layer_end}-pos${position}"
    mkdir -p "$case_dir"
    case_paths+=("$case_dir/compare.json")
    cmd=(
      "$SKIPPY_BENCH_BIN" glm-dsa-layer-microbench
      --stage-model "$STAGE_MODEL"
      --layer-start "$layer_start"
      --layer-end "$layer_end"
      --ctx-size "$CTX_SIZE"
      --tokens "$TOKENS"
      --position-start "$position"
      --kv-warmup-tokens "$position"
      --iterations "$ITERATIONS"
      --warmup "$WARMUP"
      --n-batch "$N_BATCH"
      --n-ubatch "$N_UBATCH"
      --direct-sparse-attn true
      --compact-flash-attn true
      --allow-compact-flash-auto
      --direct-sparse-prefill true
      --fused-sparse-mask true
      --metal-topk-moe-route-fusion true
      --op-timing true
      --metal-dispatch-log true
      --compare-dense-fallback
      --require-compact-flash-proof
      --output "$case_dir/compare.json"
    )
    if [[ "$SYNTHETIC_KV" == "1" ]]; then
      cmd+=(--synthetic-kv-warmup)
    fi

    printf '%q ' "${cmd[@]}" >"$case_dir/command.sh"
    printf '\n' >>"$case_dir/command.sh"
    echo "window $layer_start..$layer_end pos=$position -> $case_dir"
    set +e
    env -u LLAMA_GLM_DSA_COMPACT_FLASH_NWG -u LLAMA_GLM_DSA_SELECTED_ROW_FLASH_NWG LLAMA_STAGE_BACKEND=metal \
      "${cmd[@]}" >"$case_dir/run.log" 2>&1
    rc=$?
    set -e
    echo "$rc" >"$case_dir/exit.code"
    if [[ "$rc" != "0" ]]; then
      echo -e "${layer_start}-${layer_end}\t${position}\tFAIL\t${rc}" | tee -a "$OUT_DIR/results.tsv"
      tail -80 "$case_dir/run.log" >&2
      exit "$rc"
    fi
    jq -r --arg window "${layer_start}-${layer_end}" --arg position "$position" '
      [
        $window,
        $position,
        .comparison.baseline.timing_summary.mean_ms,
        .comparison.candidate.timing_summary.mean_ms,
        ((.comparison.baseline.timing_summary.mean_ms - .comparison.candidate.timing_summary.mean_ms) / .comparison.baseline.timing_summary.mean_ms * 100),
        .comparison.parity.passed,
        .comparison.parity.hidden_mismatches,
        .comparison.candidate.timing_summary.coefficient_of_variation,
        .metal_dispatch_summary.get_rows_typed_records,
        (.metal_dispatch_summary.dsa_top1_attn_records // 0),
        .metal_dispatch_summary.dsa_sparse_attn_records,
        (.compact_flash_guard.all_kv_flash_records // 0),
        .compact_flash_policy_summary.execution_use_compact
      ] | @tsv
    ' "$case_dir/compare.json" | tee -a "$OUT_DIR/results.tsv"
  done
done

python3 - "$OUT_DIR" "$MIN_DELTA_PERCENT" "${case_paths[@]}" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
min_delta_arg = sys.argv[2]
min_delta = float(min_delta_arg) if min_delta_arg else None
paths = [pathlib.Path(p) for p in sys.argv[3:]]

rows = []
failures = []
for path in paths:
    report = json.loads(path.read_text())
    baseline = report["comparison"]["baseline"]["timing_summary"]["mean_ms"]
    candidate = report["comparison"]["candidate"]["timing_summary"]["mean_ms"]
    delta = (baseline - candidate) / baseline * 100.0

    baseline_ops = report["comparison"]["baseline"].get("op_timing_summary") or {}
    candidate_ops = report["comparison"]["candidate"].get("op_timing_summary") or {}

    def elapsed_us(summary, name):
        return (summary.get(name) or {}).get("elapsed_us")

    def nodes(summary, name):
        return (summary.get(name) or {}).get("nodes")

    def op_delta_us(name):
        before = elapsed_us(baseline_ops, name)
        after = elapsed_us(candidate_ops, name)
        if before is None or after is None:
            return None
        return before - after

    op_delta_names = [
        "indexer_topk",
        "sparse_mask",
        "dsa_sparse_attn",
        "compact_get_rows",
        "mla_attention",
        "routed_moe",
        "shared_expert",
    ]
    op_delta_map = {name: op_delta_us(name) for name in op_delta_names}
    known_deltas = {name: value for name, value in op_delta_map.items() if value is not None}
    top_savings = sorted(
        (
            {"op": name, "delta_us": value}
            for name, value in known_deltas.items()
            if value > 0
        ),
        key=lambda item: item["delta_us"],
        reverse=True,
    )[:3]
    top_regressions = sorted(
        (
            {"op": name, "delta_us": value}
            for name, value in known_deltas.items()
            if value < 0
        ),
        key=lambda item: item["delta_us"],
    )[:3]

    row = {
        "window": f"{report['layer_start']}..{report['layer_end']}",
        "position_start": report["position_start"],
        "path": str(path),
        "dense_mean_ms": baseline,
        "compact_mean_ms": candidate,
        "delta_percent": delta,
        "parity_passed": bool(report["comparison"]["parity"]["passed"]),
        "hidden_mismatches": report["comparison"]["parity"].get("hidden_mismatches"),
        "candidate_cv": report["comparison"]["candidate"]["timing_summary"].get("coefficient_of_variation"),
        "typed_get_rows_records": report.get("metal_dispatch_summary", {}).get("get_rows_typed_records"),
        "dsa_top1_attn_records": report.get("metal_dispatch_summary", {}).get("dsa_top1_attn_records", 0),
        "dsa_compact_get_rows_fused_records": report.get("metal_dispatch_summary", {}).get("dsa_compact_get_rows_fused_records", 0),
        "all_kv_flash_records": report.get("compact_flash_guard", {}).get("all_kv_flash_records", 0),
        "dsa_sparse_attn_records": report.get("metal_dispatch_summary", {}).get("dsa_sparse_attn_records"),
        "compact_execution_records": report.get("compact_flash_policy_summary", {}).get("execution_use_compact"),
        "baseline_op_total_us": baseline_ops.get("total_us"),
        "candidate_op_total_us": candidate_ops.get("total_us"),
        "op_total_delta_us": (
            baseline_ops.get("total_us") - candidate_ops.get("total_us")
            if baseline_ops.get("total_us") is not None and candidate_ops.get("total_us") is not None
            else None
        ),
        "baseline_indexer_topk_us": elapsed_us(baseline_ops, "indexer_topk"),
        "candidate_indexer_topk_us": elapsed_us(candidate_ops, "indexer_topk"),
        "indexer_topk_delta_us": op_delta_map["indexer_topk"],
        "baseline_sparse_mask_nodes": nodes(baseline_ops, "sparse_mask"),
        "candidate_sparse_mask_nodes": nodes(candidate_ops, "sparse_mask"),
        "sparse_mask_delta_us": op_delta_map["sparse_mask"],
        "baseline_compact_get_rows_nodes": nodes(baseline_ops, "compact_get_rows"),
        "candidate_compact_get_rows_nodes": nodes(candidate_ops, "compact_get_rows"),
        "compact_get_rows_delta_us": op_delta_map["compact_get_rows"],
        "baseline_mla_attention_us": elapsed_us(baseline_ops, "mla_attention"),
        "candidate_mla_attention_us": elapsed_us(candidate_ops, "mla_attention"),
        "mla_attention_delta_us": op_delta_map["mla_attention"],
        "baseline_routed_moe_us": elapsed_us(baseline_ops, "routed_moe"),
        "candidate_routed_moe_us": elapsed_us(candidate_ops, "routed_moe"),
        "routed_moe_delta_us": op_delta_map["routed_moe"],
        "baseline_shared_expert_us": elapsed_us(baseline_ops, "shared_expert"),
        "candidate_shared_expert_us": elapsed_us(candidate_ops, "shared_expert"),
        "shared_expert_delta_us": op_delta_map["shared_expert"],
        "top_op_savings": top_savings,
        "top_op_regressions": top_regressions,
    }
    rows.append(row)
    if not row["parity_passed"]:
        failures.append(f"{row['window']} pos={row['position_start']}: parity failed")
    if row["hidden_mismatches"] not in (0, None):
        failures.append(f"{row['window']} pos={row['position_start']}: hidden mismatches {row['hidden_mismatches']}")
    if (
        row["typed_get_rows_records"] in (0, None)
        and row["dsa_compact_get_rows_fused_records"] in (0, None)
        and row["dsa_top1_attn_records"] in (0, None)
        and row["all_kv_flash_records"] in (0, None)
    ):
        failures.append(f"{row['window']} pos={row['position_start']}: missing compact get-rows, top-1, or all-KV flash evidence")
    if row["dsa_sparse_attn_records"] not in (0, None):
        failures.append(f"{row['window']} pos={row['position_start']}: dense sparse-attention dispatch present")
    if row["compact_execution_records"] in (0, None):
        failures.append(f"{row['window']} pos={row['position_start']}: missing compact execution proof")
    if min_delta is not None and delta < min_delta:
        failures.append(
            f"{row['window']} pos={row['position_start']}: delta {delta:.3f}% < required {min_delta:.3f}%"
        )

summary = {
    "passed": not failures,
    "scope": "GLM-5.2 Phase-D compact selected-KV flash-vs-dense policy sweep",
    "min_delta_percent": min_delta,
    "rows": rows,
    "failures": failures,
}
if rows:
    summary["best_delta_percent"] = max(row["delta_percent"] for row in rows)
    summary["worst_delta_percent"] = min(row["delta_percent"] for row in rows)
    summary["mean_delta_percent"] = sum(row["delta_percent"] for row in rows) / len(rows)

summary_path = out_dir / "compact-sweep-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

if failures:
    print("GLM-5.2 compact sweep FAILED", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    print(f"summary={summary_path}", file=sys.stderr)
    raise SystemExit(1)

print("GLM-5.2 compact sweep passed")
print(f"summary={summary_path}")
for row in rows:
    top_regression = row["top_op_regressions"][0] if row["top_op_regressions"] else None
    top_saving = row["top_op_savings"][0] if row["top_op_savings"] else None
    top_regression_text = (
        f" top_regression={top_regression['op']}:{top_regression['delta_us']}us"
        if top_regression
        else ""
    )
    top_saving_text = (
        f" top_saving={top_saving['op']}:{top_saving['delta_us']}us"
        if top_saving
        else ""
    )
    print(
        f"{row['window']} pos={row['position_start']}: "
        f"dense={row['dense_mean_ms']:.3f}ms compact={row['compact_mean_ms']:.3f}ms "
        f"delta={row['delta_percent']:.2f}% cv={row['candidate_cv']:.3f}"
        f"{top_saving_text}{top_regression_text}"
    )
PY
