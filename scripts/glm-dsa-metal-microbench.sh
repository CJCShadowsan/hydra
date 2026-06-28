#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-$HOME/.cache/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/target/skippy-bench/glm-dsa-metal-microbench/$(date -u +%Y%m%dT%H%M%SZ)}"
LAYER_START="${LAYER_START:-30}"
LAYER_END="${LAYER_END:-31}"
CTX_SIZE="${CTX_SIZE:-4096}"
ACTIVATION_WIDTH="${ACTIVATION_WIDTH:-6144}"
ITERATIONS="${ITERATIONS:-1}"
WARMUP="${WARMUP:-0}"
TOKENS="${TOKENS:-}"
LAYERS="${LAYERS:-}"
FORCE_REBUILD=1
BUILD_ONLY=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-metal-microbench.sh [options]

Runs the local GLM-DSA one-layer Metal microbench cases used to validate the
direct sparse attention admission path. This script deliberately does not touch
lab topology, split placement, remote hosts, or networking.

By default it forces a static-metal llama rebuild and relinks skippy-bench. That
is intentional: patched native archives can otherwise look fresh while Rust
still links an older static-metal build.

Options:
  --stage-model PATH       GLM 5.2 layer package path.
  --model-id ID            Model id recorded in reports.
  --output-dir PATH        Directory for JSON/stdout/summary artifacts.
  --layer-start N          First layer to run. Default: 30.
  --layer-end N            Exclusive layer end. Default: 31.
  --ctx-size N             Context size. Default: 4096.
  --activation-width N     Synthetic activation width. Default: 6144.
  --iterations N           Measured iterations per case. Default: 1.
  --samples N              Alias for --iterations.
  --warmup N               Warmup iterations per case. Default: 0.
  --tokens LIST            Comma-separated token sweep, e.g. 1,8,16,32,33,64.
  --layers LIST            Comma-separated single-layer starts, e.g. 30,45,60.
  --no-force-rebuild       Skip forced static-metal rebuild/relink.
  --build-only             Rebuild/relink only; do not run cases.
  --dry-run                Print commands without executing.
  -h, --help               Show this help.

Environment overrides mirror option names:
  STAGE_MODEL, MODEL_ID, OUTPUT_DIR, LAYER_START, LAYER_END, CTX_SIZE,
  ACTIVATION_WIDTH, ITERATIONS, WARMUP, TOKENS, LAYERS.
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
    --output-dir)
      OUTPUT_DIR="$2"
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
    --activation-width)
      ACTIVATION_WIDTH="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --samples)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --tokens)
      TOKENS="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --no-force-rebuild)
      FORCE_REBUILD=0
      shift
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "required path not found: $path" >&2
    exit 1
  fi
}

split_csv() {
  local value="$1"
  tr ',' '\n' <<<"$value" | while IFS= read -r item; do
    [[ -n "$item" ]] || continue
    printf '%s\n' "$item"
  done
}

prepare_static_metal_bench() {
  if [[ "$FORCE_REBUILD" == "1" ]]; then
    run_cmd rm -rf "$ROOT/.deps/llama-build/build-stage-abi-static-metal"
  fi
  run_cmd env LLAMA_STAGE_BACKEND=metal "$ROOT/scripts/build-llama.sh"
  run_cmd cargo clean -p skippy-ffi -p skippy-runtime -p skippy-bench
  run_cmd cargo build -p skippy-bench
}

run_case() {
  local name="$1"
  local layer_start="$2"
  local layer_end="$3"
  local tokens="$4"
  shift 4

  local report="$OUTPUT_DIR/${name}.json"
  local stdout="$OUTPUT_DIR/${name}.stdout"
  local -a cmd=(
    "$ROOT/target/debug/skippy-bench"
    glm-dsa-layer-microbench
    --stage-model "$STAGE_MODEL"
    --model-id "$MODEL_ID"
    --layer-start "$layer_start"
    --layer-end "$layer_end"
    --tokens "$tokens"
    --ctx-size "$CTX_SIZE"
    --activation-width "$ACTIVATION_WIDTH"
    --iterations "$ITERATIONS"
    --warmup "$WARMUP"
    --compare-dense-fallback
    --output "$report"
    --op-timing true
    "$@"
  )

  printf '+'
  printf ' %q' "${cmd[@]}"
  printf ' >%q 2>&1\n' "$stdout"
  if [[ "$DRY_RUN" == "0" ]]; then
    if "${cmd[@]}" >"$stdout" 2>&1; then
      printf 'case=%s tokens=%s exit=0 report=%s\n' "$name" "$tokens" "$report"
    else
      local rc=$?
      printf 'case=%s tokens=%s exit=%s report=%s\n' "$name" "$tokens" "$rc" "$report"
      return "$rc"
    fi
  fi
}

write_summary() {
  local summary="$OUTPUT_DIR/summary.txt"
  python3 - "$OUTPUT_DIR" >"$summary" <<'PY'
import json
import pathlib
import re
import statistics
import sys

base = pathlib.Path(sys.argv[1])

def case_sort_key(path):
    name = path.stem
    layer = re.search(r"l(\d+)", name)
    token = re.search(r"t(\d+)|-(\d+)$", name)
    variant = 0 if name.startswith("default") else 1
    token_value = 0
    if token:
        token_value = int(next(group for group in token.groups() if group is not None))
    layer_value = int(layer.group(1)) if layer else 0
    return layer_value, token_value, variant, name

cases = sorted(base.glob("*.json"), key=case_sort_key)
paired = {}

def sample_stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }

def format_float(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"

def format_int(value):
    if value is None:
        return "n/a"
    return str(int(value))

def median_field(records, field):
    return sample_stats(record.get(field) for record in records)["median"]

def dispatch_counts(records):
    counts = {}
    for record in records:
        op = record.get("op", "unknown")
        kernel = record.get("kernel")
        key = f"{op}/{kernel}" if kernel else op
        counts[key] = counts.get(key, 0) + 1
    return counts

def format_dispatch_counts(records):
    counts = dispatch_counts(records)
    if not counts:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))

def sparse_attn_dispatch_shape(record):
    return (
        f"batch={record.get('batch')} "
        f"heads={record.get('heads')} "
        f"stream={record.get('stream')} "
        f"kv={record.get('kv')} "
        f"top_k={record.get('top_k')} "
        f"grid={record.get('grid_x')}x{record.get('grid_y')}x{record.get('grid_z')} "
        f"threads_x={record.get('threads_x')}"
    )

def sparse_attn_dispatch_shapes(records):
    counts = {}
    for record in records:
        if record.get("op") != "dsa_sparse_attn":
            continue
        shape = sparse_attn_dispatch_shape(record)
        counts[shape] = counts.get(shape, 0) + 1
    return counts

def format_shape_counts(counts):
    if not counts:
        return "none"
    return "; ".join(f"{shape} count={count}" for shape, count in sorted(counts.items()))

print(f"output_dir={base}")
for path in cases:
    name = path.stem
    print()
    print(name)
    report = json.loads(path.read_text())
    comparison = report["comparison"]
    parity = comparison["parity"]
    candidate = comparison["candidate"]
    op_timing_records = candidate.get("op_timing_records", [])
    timing = op_timing_records[0] if op_timing_records else {}
    dispatch_records = candidate.get("metal_dispatch_records", [])
    decision_summary = candidate.get("direct_sparse_decision_summary", {})
    elapsed_stats = sample_stats(
        timing.get("elapsed_ms") for timing in candidate.get("timings", [])
    )
    native_stats = sample_stats(record.get("total_us") for record in op_timing_records)
    op_stats = {
        "indexer_topk": median_field(op_timing_records, "indexer_topk_us"),
        "sparse_mask": median_field(op_timing_records, "sparse_mask_us"),
        "dsa_sparse_attn": median_field(op_timing_records, "dsa_sparse_attn_us"),
        "routed_moe": median_field(op_timing_records, "routed_moe_us"),
        "shared_expert": median_field(op_timing_records, "shared_expert_us"),
    }
    pair = re.match(r"(default|optin)-l(\d+)-t(\d+)$", name)
    if pair:
        variant, layer, tokens = pair.groups()
        paired.setdefault((int(layer), int(tokens)), {})[variant] = {
            "elapsed_ms": elapsed_stats,
            "total_us": native_stats,
            "op_stats": op_stats,
            "use_direct": decision_summary.get("use_direct", 0),
            "fallback": decision_summary.get("fallback", 0),
            "dsa_sparse_attn_nodes": timing.get("dsa_sparse_attn_nodes"),
            "sparse_mask_nodes": timing.get("sparse_mask_nodes"),
        }
    print(f"  parity={parity['passed']} hidden_mismatches={parity['hidden_mismatches']} sideband_mismatches={parity['sideband_mismatched_bytes']}")
    print(f"  dsa_sparse_attn_nodes={timing.get('dsa_sparse_attn_nodes')} sparse_mask_nodes={timing.get('sparse_mask_nodes')}")
    print(
        "  elapsed_ms="
        f"count={elapsed_stats['count']} "
        f"min={format_float(elapsed_stats['min'])} "
        f"median={format_float(elapsed_stats['median'])} "
        f"max={format_float(elapsed_stats['max'])}"
    )
    print(
        "  native_total_us="
        f"count={native_stats['count']} "
        f"min={format_int(native_stats['min'])} "
        f"median={format_float(native_stats['median'])} "
        f"max={format_int(native_stats['max'])}"
    )
    print(
        "  native_op_median_us="
        f"indexer_topk={format_float(op_stats['indexer_topk'])} "
        f"sparse_mask={format_float(op_stats['sparse_mask'])} "
        f"dsa_sparse_attn={format_float(op_stats['dsa_sparse_attn'])} "
        f"routed_moe={format_float(op_stats['routed_moe'])} "
        f"shared_expert={format_float(op_stats['shared_expert'])}"
    )
    print(f"  metal_dispatch={format_dispatch_counts(dispatch_records)}")
    sparse_attn_shapes = sparse_attn_dispatch_shapes(dispatch_records)
    if sparse_attn_shapes:
        print(f"  dsa_sparse_attn_dispatch_shapes={format_shape_counts(sparse_attn_shapes)}")
    print(
        "  decisions="
        f"{decision_summary.get('records', 0)} "
        f"use_direct={decision_summary.get('use_direct', 0)} "
        f"fallback={decision_summary.get('fallback', 0)} "
        f"decode_shape={decision_summary.get('decode_shape', 0)} "
        f"prefill_shape={decision_summary.get('prefill_shape', 0)}"
    )
    decisions = candidate.get("direct_sparse_decision_records", [])
    if decisions:
        last = decisions[-1]
        print(
            "  last_decision="
            f"tokens={last['ubatch_tokens']} sparse_batch={last['sparse_batch']} "
            f"prefill_enabled={last['prefill_enabled']} prefill_shape={last['prefill_shape']} "
            f"decode_shape={last['decode_shape']} use_direct={last['use_direct']}"
        )

if paired:
    print()
    print("pairwise_optin_vs_default")
    print("layer tokens samples default_ms_median optin_ms_median elapsed_ratio default_native_us_median optin_native_us_median native_ratio indexer_topk_ratio sparse_mask_ratio dsa_sparse_attn_ratio routed_moe_ratio shared_expert_ratio optin_direct optin_fallback")
    for (layer, tokens), variants in sorted(paired.items()):
        default = variants.get("default")
        optin = variants.get("optin")
        if not default or not optin:
            continue
        default_elapsed = default["elapsed_ms"]["median"]
        optin_elapsed = optin["elapsed_ms"]["median"]
        default_native = default["total_us"]["median"]
        optin_native = optin["total_us"]["median"]
        samples = min(default["elapsed_ms"]["count"], optin["elapsed_ms"]["count"])
        elapsed_ratio = optin_elapsed / default_elapsed if default_elapsed not in (None, 0) and optin_elapsed is not None else None
        native_ratio = optin_native / default_native if default_native not in (None, 0) and optin_native is not None else None
        op_ratios = {}
        for op_name in ("indexer_topk", "sparse_mask", "dsa_sparse_attn", "routed_moe", "shared_expert"):
            default_op = default["op_stats"][op_name]
            optin_op = optin["op_stats"][op_name]
            op_ratios[op_name] = optin_op / default_op if default_op not in (None, 0) and optin_op is not None else None
        print(
            f"{layer} {tokens} "
            f"{samples} "
            f"{format_float(default_elapsed)} {format_float(optin_elapsed)} {format_float(elapsed_ratio)} "
            f"{format_float(default_native)} {format_float(optin_native)} {format_float(native_ratio)} "
            f"{format_float(op_ratios['indexer_topk'])} "
            f"{format_float(op_ratios['sparse_mask'])} "
            f"{format_float(op_ratios['dsa_sparse_attn'])} "
            f"{format_float(op_ratios['routed_moe'])} "
            f"{format_float(op_ratios['shared_expert'])} "
            f"{optin['use_direct']} {optin['fallback']}"
        )
PY
  printf 'summary=%s\n' "$summary"
  if [[ "$DRY_RUN" == "0" ]]; then
    cat "$summary"
  fi
}

cd "$ROOT"
require_path "$STAGE_MODEL"
mkdir -p "$OUTPUT_DIR"

prepare_static_metal_bench
if [[ "$BUILD_ONLY" == "1" ]]; then
  exit 0
fi

run_default_cases() {
  run_case default-1 "$LAYER_START" "$LAYER_END" 1
  run_case default-33 "$LAYER_START" "$LAYER_END" 33
  run_case optin-prefill-32 "$LAYER_START" "$LAYER_END" 32 --direct-sparse-prefill true
  run_case optin-prefill-33 "$LAYER_START" "$LAYER_END" 33 --direct-sparse-prefill true
}

run_sweep_cases() {
  local layers="$LAYERS"
  if [[ -z "$layers" ]]; then
    layers="$LAYER_START"
  fi
  local layer_start
  local tokens
  while IFS= read -r layer_start; do
    local layer_end=$((layer_start + 1))
    while IFS= read -r tokens; do
      run_case "default-l${layer_start}-t${tokens}" "$layer_start" "$layer_end" "$tokens"
      run_case "optin-l${layer_start}-t${tokens}" "$layer_start" "$layer_end" "$tokens" --direct-sparse-prefill true
    done < <(split_csv "$TOKENS")
  done < <(split_csv "$layers")
}

if [[ -n "$TOKENS" ]]; then
  run_sweep_cases
else
  run_default_cases
fi
write_summary
