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
INDEXER_MODE="${INDEXER_MODE:-parallel}"
SPARSE_ATTN_THREADS="${SPARSE_ATTN_THREADS:-256}"
SPARSE_ATTN_CACHE_TOPK="${SPARSE_ATTN_CACHE_TOPK:-off}"
SPARSE_ATTN_DECODE_GROUP_HEADS="${SPARSE_ATTN_DECODE_GROUP_HEADS:-1}"
MUL_MM_ID_MIN_TOKENS="${MUL_MM_ID_MIN_TOKENS:-32}"
SYNTHETIC_TOP_K_SIDEBAND="${SYNTHETIC_TOP_K_SIDEBAND:-off}"
SYNTHETIC_TOP_K_WIDTH="${SYNTHETIC_TOP_K_WIDTH:-256}"
REAL_TOP_K_SOURCE_LAYER_START="${REAL_TOP_K_SOURCE_LAYER_START:-}"
REAL_TOP_K_CHAIN_SOURCES="${REAL_TOP_K_CHAIN_SOURCES:-}"
REAL_TOP_K_MAX_SOURCE_BYTES="${REAL_TOP_K_MAX_SOURCE_BYTES:-}"
REAL_TOP_K_CACHE_DIR="${REAL_TOP_K_CACHE_DIR:-}"
WARM_REAL_TOP_K_CACHE="${WARM_REAL_TOP_K_CACHE:-off}"
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
  --indexer-mode MODE      Lightning indexer mode: serial, parallel, or both.
                           Default: parallel, matching skippy-bench's current
                           glm-dsa-layer-microbench default.
  --sparse-attn-threads LIST
                           Comma-separated Metal dsa_sparse_attn threadgroup
                           widths to sweep. Allowed: 32,64,128,256. Default: 256.
  --sparse-attn-cache-topk MODE
                           Cache top-k indices in dsa_sparse_attn threadgroup
                           memory: off, on, or both. Default: off.
  --sparse-attn-decode-group-heads LIST
                           Comma-separated decode head group sizes to sweep.
                           Allowed: 1,2,4. Default: 1.
  --mul-mm-id-min-tokens LIST
                           Comma-separated Metal mul_mat_id token thresholds
                           for trying the matrix-matrix ID kernel. Default: 32.
  --synthetic-top-k-sideband MODE
                           Append synthetic GLM-DSA top-k sideband to input
                           activations: off or on. Default: off.
  --synthetic-top-k-width N
                           Synthetic top-k sideband width. Default: 256.
  --real-top-k-source-layer-start N
                           Generate input with real GLM-DSA top-k sideband by
                           running local source range N..<target layer>.
                           Cannot be combined with --synthetic-top-k-sideband on.
  --real-top-k-chain-sources LIST
                           Comma-separated source starts used to build real
                           top-k inputs for deeper consumer source spans, e.g.
                           30 lets layer 75 run as 30..60 -> 60..75 -> 75..76.
  --real-top-k-max-source-bytes N|off
                           Refuse to generate a real top-k source span when
                           selected layer artifacts exceed N bytes. Default is
                           skippy-bench's built-in 120 GiB guard. Use off only
                           for deliberate stress testing.
  --real-top-k-cache-dir PATH
                           Reuse generated real top-k activation frames from
                           PATH. Defaults to <output-dir>/real-top-k-cache when
                           --real-top-k-source-layer-start is set.
  --warm-real-top-k-cache MODE
                           Pre-generate real top-k cache frames before measured
                           cases and require cache hits during measurement:
                           off or on. Default: off.
  --no-force-rebuild       Skip forced static-metal rebuild/relink.
  --build-only             Rebuild/relink only; do not run cases.
  --dry-run                Print commands without executing.
  -h, --help               Show this help.

Environment overrides mirror option names:
  STAGE_MODEL, MODEL_ID, OUTPUT_DIR, LAYER_START, LAYER_END, CTX_SIZE,
  ACTIVATION_WIDTH, ITERATIONS, WARMUP, TOKENS, LAYERS, INDEXER_MODE,
  SPARSE_ATTN_THREADS, SPARSE_ATTN_CACHE_TOPK, SPARSE_ATTN_DECODE_GROUP_HEADS,
  MUL_MM_ID_MIN_TOKENS, SYNTHETIC_TOP_K_SIDEBAND, SYNTHETIC_TOP_K_WIDTH,
  REAL_TOP_K_SOURCE_LAYER_START, REAL_TOP_K_CHAIN_SOURCES,
  REAL_TOP_K_MAX_SOURCE_BYTES, REAL_TOP_K_CACHE_DIR, WARM_REAL_TOP_K_CACHE.
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
    --indexer-mode)
      INDEXER_MODE="$2"
      shift 2
      ;;
    --sparse-attn-threads)
      SPARSE_ATTN_THREADS="$2"
      shift 2
      ;;
    --sparse-attn-cache-topk)
      SPARSE_ATTN_CACHE_TOPK="$2"
      shift 2
      ;;
    --sparse-attn-decode-group-heads)
      SPARSE_ATTN_DECODE_GROUP_HEADS="$2"
      shift 2
      ;;
    --mul-mm-id-min-tokens)
      MUL_MM_ID_MIN_TOKENS="$2"
      shift 2
      ;;
    --synthetic-top-k-sideband)
      SYNTHETIC_TOP_K_SIDEBAND="$2"
      shift 2
      ;;
    --synthetic-top-k-width)
      SYNTHETIC_TOP_K_WIDTH="$2"
      shift 2
      ;;
    --real-top-k-source-layer-start)
      REAL_TOP_K_SOURCE_LAYER_START="$2"
      shift 2
      ;;
    --real-top-k-chain-sources)
      REAL_TOP_K_CHAIN_SOURCES="$2"
      shift 2
      ;;
    --real-top-k-max-source-bytes)
      REAL_TOP_K_MAX_SOURCE_BYTES="$2"
      shift 2
      ;;
    --real-top-k-cache-dir)
      REAL_TOP_K_CACHE_DIR="$2"
      shift 2
      ;;
    --warm-real-top-k-cache)
      WARM_REAL_TOP_K_CACHE="$2"
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

validate_indexer_mode() {
  case "$INDEXER_MODE" in
    serial|parallel|both)
      ;;
    *)
      echo "invalid --indexer-mode: $INDEXER_MODE (expected serial, parallel, or both)" >&2
      exit 2
      ;;
  esac
}

validate_sparse_attn_threads() {
  local threads
  while IFS= read -r threads; do
    case "$threads" in
      32|64|128|256)
        ;;
      *)
        echo "invalid --sparse-attn-threads entry: $threads (expected 32, 64, 128, or 256)" >&2
        exit 2
        ;;
    esac
  done < <(split_csv "$SPARSE_ATTN_THREADS")
}

validate_sparse_attn_cache_topk() {
  case "$SPARSE_ATTN_CACHE_TOPK" in
    off|on|both)
      ;;
    *)
      echo "invalid --sparse-attn-cache-topk: $SPARSE_ATTN_CACHE_TOPK (expected off, on, or both)" >&2
      exit 2
      ;;
  esac
}

validate_sparse_attn_decode_group_heads() {
  local value
  while IFS= read -r value; do
    case "$value" in
      1|2|4)
        ;;
      *)
        echo "invalid --sparse-attn-decode-group-heads value: $value (expected 1,2,4)" >&2
        exit 2
        ;;
    esac
  done < <(split_csv "$SPARSE_ATTN_DECODE_GROUP_HEADS")
}

validate_mul_mm_id_min_tokens() {
  local value
  while IFS= read -r value; do
    case "$value" in
      ''|*[!0-9]*)
        echo "invalid --mul-mm-id-min-tokens value: $value (expected positive integer)" >&2
        exit 2
        ;;
      *)
        if [[ "$value" -lt 1 ]]; then
          echo "invalid --mul-mm-id-min-tokens value: $value (expected positive integer)" >&2
          exit 2
        fi
        ;;
    esac
  done < <(split_csv "$MUL_MM_ID_MIN_TOKENS")
}

validate_synthetic_top_k_sideband() {
  case "$SYNTHETIC_TOP_K_SIDEBAND" in
    off|on)
      ;;
    *)
      echo "invalid --synthetic-top-k-sideband: $SYNTHETIC_TOP_K_SIDEBAND (expected off or on)" >&2
      exit 2
      ;;
  esac
  case "$SYNTHETIC_TOP_K_WIDTH" in
    ''|*[!0-9]*)
      echo "invalid --synthetic-top-k-width: $SYNTHETIC_TOP_K_WIDTH (expected positive integer)" >&2
      exit 2
      ;;
    *)
      if [[ "$SYNTHETIC_TOP_K_WIDTH" -lt 1 ]]; then
        echo "invalid --synthetic-top-k-width: $SYNTHETIC_TOP_K_WIDTH (expected positive integer)" >&2
        exit 2
      fi
      ;;
  esac
}

validate_real_top_k_source_layer_start() {
  case "$REAL_TOP_K_SOURCE_LAYER_START" in
    ''|off|0)
      if [[ -n "$REAL_TOP_K_CHAIN_SOURCES" ]]; then
        echo "--real-top-k-chain-sources requires --real-top-k-source-layer-start" >&2
        exit 2
      fi
      return
      ;;
    *[!0-9]*)
      echo "invalid --real-top-k-source-layer-start: $REAL_TOP_K_SOURCE_LAYER_START (expected positive integer or off)" >&2
      exit 2
      ;;
  esac
  if [[ "$SYNTHETIC_TOP_K_SIDEBAND" == "on" ]]; then
    echo "--real-top-k-source-layer-start cannot be combined with --synthetic-top-k-sideband on" >&2
    exit 2
  fi
  if [[ -z "$REAL_TOP_K_CACHE_DIR" ]]; then
    REAL_TOP_K_CACHE_DIR="$OUTPUT_DIR/real-top-k-cache"
  fi
}

validate_real_top_k_chain_sources() {
  [[ -n "$REAL_TOP_K_CHAIN_SOURCES" ]] || return 0
  local source
  while IFS= read -r source; do
    case "$source" in
      ''|*[!0-9]*)
        echo "invalid --real-top-k-chain-sources entry: $source (expected positive integer)" >&2
        exit 2
        ;;
      *)
        if [[ "$source" -lt 1 ]]; then
          echo "invalid --real-top-k-chain-sources entry: $source (expected positive integer)" >&2
          exit 2
        fi
        ;;
    esac
  done < <(split_csv "$REAL_TOP_K_CHAIN_SOURCES")
}

validate_real_top_k_max_source_bytes() {
  [[ -n "$REAL_TOP_K_MAX_SOURCE_BYTES" ]] || return 0
  case "$REAL_TOP_K_MAX_SOURCE_BYTES" in
    off|OFF|Off|0)
      ;;
    *[!0-9]*)
      echo "invalid --real-top-k-max-source-bytes: $REAL_TOP_K_MAX_SOURCE_BYTES (expected positive integer, 0, or off)" >&2
      exit 2
      ;;
    *)
      if [[ "$REAL_TOP_K_MAX_SOURCE_BYTES" -lt 1 ]]; then
        echo "invalid --real-top-k-max-source-bytes: $REAL_TOP_K_MAX_SOURCE_BYTES (expected positive integer, 0, or off)" >&2
        exit 2
      fi
      ;;
  esac
}

validate_warm_real_top_k_cache() {
  case "$WARM_REAL_TOP_K_CACHE" in
    off|on)
      ;;
    *)
      echo "invalid --warm-real-top-k-cache: $WARM_REAL_TOP_K_CACHE (expected off or on)" >&2
      exit 2
      ;;
  esac
  if [[ "$WARM_REAL_TOP_K_CACHE" == "on" ]]; then
    case "$REAL_TOP_K_SOURCE_LAYER_START" in
      ''|off|0)
        echo "--warm-real-top-k-cache on requires --real-top-k-source-layer-start" >&2
        exit 2
        ;;
    esac
  fi
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
  local -a env_args=()
  local -a bench_args=()
  local arg
  for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
      env_args+=("$arg")
    else
      bench_args+=("$arg")
    fi
  done
  local -a cmd=(
    env
    "${env_args[@]}"
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
    "${bench_args[@]}"
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

run_cache_warm_case() {
  local layer_start="$1"
  local layer_end="$2"
  local tokens="$3"

  local warm_dir="$OUTPUT_DIR/cache-warm"
  local report="$warm_dir/warm-realtopk${REAL_TOP_K_SOURCE_LAYER_START}-l${layer_start}-t${tokens}.json"
  local stdout="$warm_dir/warm-realtopk${REAL_TOP_K_SOURCE_LAYER_START}-l${layer_start}-t${tokens}.stdout"
  mkdir -p "$warm_dir"
  local -a cmd=(
    env
    "SKIPPY_GLM_DSA_SPARSE_ATTN_THREADS=256"
    "SKIPPY_GLM_DSA_SPARSE_ATTN_CACHE_TOPK=0"
    "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_SIDEBAND=0"
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_SOURCE_LAYER_START=$REAL_TOP_K_SOURCE_LAYER_START"
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CHAIN_SOURCES=$REAL_TOP_K_CHAIN_SOURCES"
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_MAX_SOURCE_BYTES=$REAL_TOP_K_MAX_SOURCE_BYTES"
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CACHE_DIR=$REAL_TOP_K_CACHE_DIR"
    "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_REQUIRE_CACHE=0"
    "$ROOT/target/debug/skippy-bench"
    glm-dsa-layer-microbench
    --stage-model "$STAGE_MODEL"
    --model-id "$MODEL_ID"
    --layer-start "$layer_start"
    --layer-end "$layer_end"
    --tokens "$tokens"
    --ctx-size "$CTX_SIZE"
    --activation-width "$ACTIVATION_WIDTH"
    --iterations 1
    --warmup 0
    --compare-dense-fallback
    --output "$report"
    --op-timing false
    --parallel-lightning-indexer false
  )

  printf '+'
  printf ' %q' "${cmd[@]}"
  printf ' >%q 2>&1\n' "$stdout"
  if [[ "$DRY_RUN" == "0" ]]; then
    if "${cmd[@]}" >"$stdout" 2>&1; then
      printf 'cache_warm layer=%s tokens=%s exit=0 report=%s\n' "$layer_start" "$tokens" "$report"
    else
      local rc=$?
      printf 'cache_warm layer=%s tokens=%s exit=%s report=%s\n' "$layer_start" "$tokens" "$rc" "$report"
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
    threads = re.search(r"th(\d+)", name)
    mm_id = re.search(r"mmid(\d+)", name)
    variant = 0 if name.startswith("default") else 1
    token_value = 0
    if token:
        token_value = int(next(group for group in token.groups() if group is not None))
    layer_value = int(layer.group(1)) if layer else 0
    threads_value = int(threads.group(1)) if threads else 0
    mm_id_value = int(mm_id.group(1)) if mm_id else 32
    return layer_value, token_value, threads_value, mm_id_value, variant, name

cases = sorted(base.glob("*.json"), key=case_sort_key)
paired = {}
mode_cases = {}

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

def route_fusion_reason_counts(records):
    counts = {}
    for record in records:
        op = record.get("op", "unknown")
        if not op.startswith("topk_moe_route_"):
            continue
        reason = record.get("reason") or record.get("kernel")
        if reason is None and op == "topk_moe_route_fused":
            reason = "fused"
        if reason is None:
            reason = "unknown"
        key = f"{op}/{reason}"
        counts[key] = counts.get(key, 0) + 1
    return counts

def format_route_fusion_reasons(records):
    counts = route_fusion_reason_counts(records)
    if not counts:
        return "none"
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))

def timing_summary_stats(summary):
    if not summary:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "count": summary.get("samples", 0),
        "min": summary.get("min_ms"),
        "median": summary.get("p50_ms"),
        "max": summary.get("max_ms"),
    }

def format_hot_tensors(records, limit=8):
    if not records:
        return "none"
    by_record = {}
    for record in records:
        by_record.setdefault(record.get("record_index", 0), []).append(record)
    latest_index = max(by_record)
    latest = sorted(by_record[latest_index], key=lambda record: record.get("rank", 0))
    formatted = []
    for record in latest[:limit]:
        formatted.append(
            f"{record.get('rank')}:{record.get('kind')}/{record.get('op')}"
            f"/{record.get('name')}={record.get('elapsed_us')}us"
        )
    return "; ".join(formatted)

def format_input_source(source):
    if not source:
        return "unknown"
    kind = source.get("kind", "unknown")
    if kind != "real_top_k":
        top_k_sideband = source.get("top_k_sideband")
        return kind if top_k_sideband is None else f"{kind}/topk{top_k_sideband}"
    return (
        f"real_top_k source={source.get('layer_start')}..{source.get('layer_end')} "
        f"sideband_bytes={source.get('sideband_bytes')} "
        f"cache_hit={source.get('cache_hit')} "
        f"cache_path={source.get('cache_path', 'none')}"
    )

def sparse_attn_dispatch_shape(record):
    return (
        f"batch={record.get('batch')} "
        f"heads={record.get('heads')} "
        f"stream={record.get('stream')} "
        f"kv={record.get('kv')} "
        f"top_k={record.get('top_k')} "
        f"grid={record.get('grid_x')}x{record.get('grid_y')}x{record.get('grid_z')} "
        f"threads={record.get('threads_x')}x{record.get('threads_y') or 1}"
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

def case_mode_from_name(name):
    if "-serial-" in name:
        return "serial"
    return "parallel"

def case_threads_from_name(name):
    match = re.search(r"-th(\d+)(?:-|$)", name)
    return int(match.group(1)) if match else 256

def case_cache_topk_from_name(name):
    if "-cachetopk-" in name:
        return "on"
    if "-nocachetopk-" in name:
        return "off"
    return "off"

def case_decode_group_from_name(name):
    match = re.search(r"-hgrp(\d+)(?:-|$)", name)
    return int(match.group(1)) if match else 1

def case_mul_mm_id_min_tokens_from_name(name):
    match = re.search(r"-mmid(\d+)(?:-|$)", name)
    return int(match.group(1)) if match else 32

def case_topk_input_from_name(name):
    if "-syntopk" in name:
        return "synthetic"
    if "-realtopk" in name:
        return "real"
    return "none"

print(f"output_dir={base}")
for path in cases:
    name = path.stem
    print()
    print(name)
    report = json.loads(path.read_text())
    input_source = report.get("input_source", {})
    comparison = report.get("comparison")
    parity = comparison.get("parity") if comparison else None
    candidate = comparison["candidate"] if comparison else report
    optimized_probe = report.get("optimized_dispatch_probe")
    profile_integrity = report.get("profile_integrity", {})
    op_timing_records = candidate.get("op_timing_records", [])
    timing = op_timing_records[0] if op_timing_records else {}
    dispatch_records = candidate.get("metal_dispatch_records", [])
    hot_tensor_records = candidate.get("hot_tensor_records", [])
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
        "routed_moe_route": median_field(op_timing_records, "routed_moe_route_us"),
        "routed_moe_gate_up": median_field(op_timing_records, "routed_moe_gate_up_us"),
        "routed_moe_gate": median_field(op_timing_records, "routed_moe_gate_us"),
        "routed_moe_up": median_field(op_timing_records, "routed_moe_up_us"),
        "routed_moe_act": median_field(op_timing_records, "routed_moe_act_us"),
        "routed_moe_down": median_field(op_timing_records, "routed_moe_down_us"),
        "routed_moe_weighted": median_field(op_timing_records, "routed_moe_weighted_us"),
        "routed_moe_aggregate": median_field(op_timing_records, "routed_moe_aggregate_us"),
        "shared_expert": median_field(op_timing_records, "shared_expert_us"),
    }
    pair = re.match(r"(default|optin)(?:-(serial|parallel))?(?:-th(\d+))?(?:-(cachetopk|nocachetopk))?(?:-hgrp(\d+))?(?:-mmid(\d+))?(?:-(syntopk\d+|realtopk\d+(?:-chain[0-9_]+)?))?-l(\d+)-t(\d+)$", name)
    if pair:
        variant, mode, threads, cache_topk, decode_group, mm_id_min_tokens, topk_input, layer, tokens = pair.groups()
        mode = mode or case_mode_from_name(name)
        threads = int(threads) if threads else case_threads_from_name(name)
        cache_topk = "on" if cache_topk == "cachetopk" else case_cache_topk_from_name(name)
        decode_group = int(decode_group) if decode_group else case_decode_group_from_name(name)
        mm_id_min_tokens = int(mm_id_min_tokens) if mm_id_min_tokens else case_mul_mm_id_min_tokens_from_name(name)
        topk_input = case_topk_input_from_name(name)
        case_summary = {
            "elapsed_ms": elapsed_stats,
            "total_us": native_stats,
            "op_stats": op_stats,
            "use_direct": decision_summary.get("use_direct", 0),
            "fallback": decision_summary.get("fallback", 0),
            "dsa_sparse_attn_nodes": timing.get("dsa_sparse_attn_nodes"),
            "sparse_mask_nodes": timing.get("sparse_mask_nodes"),
        }
        paired.setdefault((mode, threads, cache_topk, decode_group, mm_id_min_tokens, topk_input, int(layer), int(tokens)), {})[variant] = case_summary
        mode_cases[(variant, int(layer), int(tokens), mode, threads, cache_topk, decode_group, mm_id_min_tokens, topk_input)] = case_summary
    print(f"  input_source={format_input_source(input_source)}")
    if parity:
        print(f"  parity={parity['passed']} hidden_mismatches={parity['hidden_mismatches']} sideband_mismatches={parity['sideband_mismatched_bytes']}")
    else:
        print("  parity=n/a")
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
    if op_stats["routed_moe_route"] is not None:
        print(
            "  routed_moe_median_us="
            f"route={format_float(op_stats['routed_moe_route'])} "
            f"gate_up={format_float(op_stats['routed_moe_gate_up'])} "
            f"gate={format_float(op_stats['routed_moe_gate'])} "
            f"up={format_float(op_stats['routed_moe_up'])} "
            f"act={format_float(op_stats['routed_moe_act'])} "
            f"down={format_float(op_stats['routed_moe_down'])} "
            f"weighted={format_float(op_stats['routed_moe_weighted'])} "
            f"aggregate={format_float(op_stats['routed_moe_aggregate'])}"
        )
    print(f"  metal_dispatch={format_dispatch_counts(dispatch_records)}")
    print(f"  route_fusion_reasons={format_route_fusion_reasons(dispatch_records)}")
    if profile_integrity:
        print(
            "  profile_integrity="
            f"route_fusion_active={profile_integrity.get('route_fusion_active')} "
            f"optimized_route_fusion_active={profile_integrity.get('optimized_probe_route_fusion_active')} "
            f"diagnostic_may_disable_route_fusion={profile_integrity.get('diagnostic_timing_may_disable_route_fusion')} "
            f"diagnostic_ms={format_float(profile_integrity.get('diagnostic_mean_ms'))} "
            f"optimized_ms={format_float(profile_integrity.get('optimized_probe_mean_ms'))} "
            f"diagnostic_slowdown={format_float(profile_integrity.get('diagnostic_slowdown_vs_optimized_probe'))}"
        )
    if optimized_probe:
        optimized_dispatch = optimized_probe.get("metal_dispatch_records", [])
        optimized_stats = timing_summary_stats(optimized_probe.get("timing_summary", {}))
        print(
            "  optimized_probe_elapsed_ms="
            f"count={optimized_stats['count']} "
            f"min={format_float(optimized_stats['min'])} "
            f"median={format_float(optimized_stats['median'])} "
            f"max={format_float(optimized_stats['max'])}"
        )
        print(f"  optimized_probe_route_fusion_reasons={format_route_fusion_reasons(optimized_dispatch)}")
    print(f"  hot_tensors={format_hot_tensors(hot_tensor_records)}")
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
    print("mode sparse_attn_threads cache_topk decode_group_heads mul_mm_id_min_tokens top_k_input layer tokens samples default_ms_median optin_ms_median elapsed_ratio default_native_us_median optin_native_us_median native_ratio indexer_topk_ratio sparse_mask_ratio dsa_sparse_attn_ratio routed_moe_ratio shared_expert_ratio optin_direct optin_fallback")
    for (mode, threads, cache_topk, decode_group, mm_id_min_tokens, topk_input, layer, tokens), variants in sorted(paired.items()):
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
            f"{mode} {threads} {cache_topk} {decode_group} {mm_id_min_tokens} "
            f"{topk_input} {layer} {tokens} "
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

if mode_cases:
    print()
    print("parallel_vs_serial")
    print("variant sparse_attn_threads cache_topk decode_group_heads mul_mm_id_min_tokens top_k_input layer tokens samples serial_ms_median parallel_ms_median elapsed_ratio serial_native_us_median parallel_native_us_median native_ratio indexer_topk_ratio")
    keys = sorted({(variant, layer, tokens, threads, cache_topk, decode_group, mm_id_min_tokens, topk_input) for (variant, layer, tokens, _mode, threads, cache_topk, decode_group, mm_id_min_tokens, topk_input) in mode_cases})
    for variant, layer, tokens, threads, cache_topk, decode_group, mm_id_min_tokens, topk_input in keys:
        serial = mode_cases.get((variant, layer, tokens, "serial", threads, cache_topk, decode_group, mm_id_min_tokens, topk_input))
        parallel = mode_cases.get((variant, layer, tokens, "parallel", threads, cache_topk, decode_group, mm_id_min_tokens, topk_input))
        if not serial or not parallel:
            continue
        serial_elapsed = serial["elapsed_ms"]["median"]
        parallel_elapsed = parallel["elapsed_ms"]["median"]
        serial_native = serial["total_us"]["median"]
        parallel_native = parallel["total_us"]["median"]
        serial_indexer = serial["op_stats"]["indexer_topk"]
        parallel_indexer = parallel["op_stats"]["indexer_topk"]
        samples = min(serial["elapsed_ms"]["count"], parallel["elapsed_ms"]["count"])
        elapsed_ratio = parallel_elapsed / serial_elapsed if serial_elapsed not in (None, 0) and parallel_elapsed is not None else None
        native_ratio = parallel_native / serial_native if serial_native not in (None, 0) and parallel_native is not None else None
        indexer_ratio = parallel_indexer / serial_indexer if serial_indexer not in (None, 0) and parallel_indexer is not None else None
        print(
            f"{variant} {threads} {cache_topk} {decode_group} {mm_id_min_tokens} "
            f"{topk_input} {layer} {tokens} "
            f"{samples} "
            f"{format_float(serial_elapsed)} {format_float(parallel_elapsed)} {format_float(elapsed_ratio)} "
            f"{format_float(serial_native)} {format_float(parallel_native)} {format_float(native_ratio)} "
            f"{format_float(indexer_ratio)}"
        )
PY
  printf 'summary=%s\n' "$summary"
  if [[ "$DRY_RUN" == "0" ]]; then
    cat "$summary"
  fi
}

cd "$ROOT"
require_path "$STAGE_MODEL"
validate_indexer_mode
validate_sparse_attn_threads
validate_sparse_attn_cache_topk
validate_sparse_attn_decode_group_heads
validate_mul_mm_id_min_tokens
validate_synthetic_top_k_sideband
validate_real_top_k_source_layer_start
validate_real_top_k_chain_sources
validate_real_top_k_max_source_bytes
validate_warm_real_top_k_cache
mkdir -p "$OUTPUT_DIR"

prepare_static_metal_bench
if [[ "$BUILD_ONLY" == "1" ]]; then
  exit 0
fi

run_default_cases() {
  local mode
  for mode in $(indexer_modes); do
    local threads
    for threads in $(sparse_attn_thread_counts); do
      local cache_topk
      for cache_topk in $(sparse_attn_cache_topk_modes); do
        local decode_group_heads
        for decode_group_heads in $(sparse_attn_decode_group_head_counts); do
          local mm_id_min_tokens
          for mm_id_min_tokens in $(mul_mm_id_min_token_counts); do
            run_case "$(case_name "default-1" "$mode" "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens")" "$LAYER_START" "$LAYER_END" 1 $(case_env "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens") $(indexer_args "$mode")
            run_case "$(case_name "default-33" "$mode" "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens")" "$LAYER_START" "$LAYER_END" 33 $(case_env "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens") $(indexer_args "$mode")
            run_case "$(case_name "optin-prefill-32" "$mode" "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens")" "$LAYER_START" "$LAYER_END" 32 $(case_env "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens") --direct-sparse-prefill true $(indexer_args "$mode")
            run_case "$(case_name "optin-prefill-33" "$mode" "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens")" "$LAYER_START" "$LAYER_END" 33 $(case_env "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens") --direct-sparse-prefill true $(indexer_args "$mode")
          done
        done
      done
    done
  done
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
      local mode
      for mode in $(indexer_modes); do
        local threads
        for threads in $(sparse_attn_thread_counts); do
          local cache_topk
          for cache_topk in $(sparse_attn_cache_topk_modes); do
            local decode_group_heads
            for decode_group_heads in $(sparse_attn_decode_group_head_counts); do
              local mm_id_min_tokens
              for mm_id_min_tokens in $(mul_mm_id_min_token_counts); do
                run_case "$(case_name "default-l${layer_start}-t${tokens}" "$mode" "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens")" "$layer_start" "$layer_end" "$tokens" $(case_env "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens") $(indexer_args "$mode")
                run_case "$(case_name "optin-l${layer_start}-t${tokens}" "$mode" "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens")" "$layer_start" "$layer_end" "$tokens" $(case_env "$threads" "$cache_topk" "$decode_group_heads" "$mm_id_min_tokens") --direct-sparse-prefill true $(indexer_args "$mode")
              done
            done
          done
        done
      done
    done < <(split_csv "$TOKENS")
  done < <(split_csv "$layers")
}

warm_real_top_k_cache() {
  [[ "$WARM_REAL_TOP_K_CACHE" == "on" ]] || return 0
  local layers="$LAYERS"
  if [[ -z "$layers" ]]; then
    layers="$LAYER_START"
  fi
  local layer_start
  local tokens
  while IFS= read -r layer_start; do
    local layer_end=$((layer_start + 1))
    while IFS= read -r tokens; do
      run_cache_warm_case "$layer_start" "$layer_end" "$tokens"
    done < <(split_csv "$TOKENS")
  done < <(split_csv "$layers")
}

case_env() {
  local threads="$1"
  local cache_topk="$2"
  local decode_group_heads="$3"
  local mm_id_min_tokens="$4"
  printf '%s\n' "SKIPPY_GLM_DSA_SPARSE_ATTN_THREADS=$threads"
  printf '%s\n' "SKIPPY_GLM_DSA_SPARSE_ATTN_DECODE_GROUP_HEADS=$decode_group_heads"
  printf '%s\n' "SKIPPY_GLM_DSA_MUL_MM_ID_MIN_TOKENS=$mm_id_min_tokens"
  if [[ "$cache_topk" == "on" ]]; then
    printf '%s\n' "SKIPPY_GLM_DSA_SPARSE_ATTN_CACHE_TOPK=1"
  else
    printf '%s\n' "SKIPPY_GLM_DSA_SPARSE_ATTN_CACHE_TOPK=0"
  fi
  if [[ "$SYNTHETIC_TOP_K_SIDEBAND" == "on" ]]; then
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_SIDEBAND=1"
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_WIDTH=$SYNTHETIC_TOP_K_WIDTH"
  else
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_SYNTHETIC_TOP_K_SIDEBAND=0"
  fi
  if [[ -n "$REAL_TOP_K_SOURCE_LAYER_START" && "$REAL_TOP_K_SOURCE_LAYER_START" != "off" && "$REAL_TOP_K_SOURCE_LAYER_START" != "0" ]]; then
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_SOURCE_LAYER_START=$REAL_TOP_K_SOURCE_LAYER_START"
    if [[ -n "$REAL_TOP_K_CHAIN_SOURCES" ]]; then
      printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CHAIN_SOURCES=$REAL_TOP_K_CHAIN_SOURCES"
    fi
    if [[ -n "$REAL_TOP_K_MAX_SOURCE_BYTES" ]]; then
      printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_MAX_SOURCE_BYTES=$REAL_TOP_K_MAX_SOURCE_BYTES"
    fi
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CACHE_DIR=$REAL_TOP_K_CACHE_DIR"
    if [[ "$WARM_REAL_TOP_K_CACHE" == "on" ]]; then
      printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_REQUIRE_CACHE=1"
    else
      printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_REQUIRE_CACHE=0"
    fi
  else
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_SOURCE_LAYER_START=off"
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CACHE_DIR=off"
    printf '%s\n' "SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_REQUIRE_CACHE=0"
  fi
}

sparse_attn_thread_counts() {
  split_csv "$SPARSE_ATTN_THREADS"
}

sparse_attn_cache_topk_modes() {
  case "$SPARSE_ATTN_CACHE_TOPK" in
    both)
      printf 'off\non\n'
      ;;
    *)
      printf '%s\n' "$SPARSE_ATTN_CACHE_TOPK"
      ;;
  esac
}

sparse_attn_decode_group_head_counts() {
  split_csv "$SPARSE_ATTN_DECODE_GROUP_HEADS"
}

mul_mm_id_min_token_counts() {
  split_csv "$MUL_MM_ID_MIN_TOKENS"
}

indexer_modes() {
  case "$INDEXER_MODE" in
    both)
      printf 'serial\nparallel\n'
      ;;
    *)
      printf '%s\n' "$INDEXER_MODE"
      ;;
  esac
}

case_name() {
  local base_name="$1"
  local mode="$2"
  local threads="$3"
  local cache_topk="$4"
  local decode_group_heads="$5"
  local mm_id_min_tokens="$6"
  local threads_suffix=""
  local cache_suffix=""
  local decode_group_suffix=""
  local mm_id_suffix=""
  local sideband_suffix=""
  if [[ "$SPARSE_ATTN_THREADS" != "256" || "$threads" != "256" ]]; then
    threads_suffix="-th${threads}"
  fi
  if [[ "$SPARSE_ATTN_CACHE_TOPK" == "both" ]]; then
    if [[ "$cache_topk" == "on" ]]; then
      cache_suffix="-cachetopk"
    else
      cache_suffix="-nocachetopk"
    fi
  elif [[ "$cache_topk" == "on" ]]; then
    cache_suffix="-cachetopk"
  fi
  if [[ "$SPARSE_ATTN_DECODE_GROUP_HEADS" != "1" || "$decode_group_heads" != "1" ]]; then
    decode_group_suffix="-hgrp${decode_group_heads}"
  fi
  if [[ "$MUL_MM_ID_MIN_TOKENS" != "32" || "$mm_id_min_tokens" != "32" ]]; then
    mm_id_suffix="-mmid${mm_id_min_tokens}"
  fi
  if [[ "$SYNTHETIC_TOP_K_SIDEBAND" == "on" ]]; then
    sideband_suffix="-syntopk${SYNTHETIC_TOP_K_WIDTH}"
  elif [[ -n "$REAL_TOP_K_SOURCE_LAYER_START" && "$REAL_TOP_K_SOURCE_LAYER_START" != "off" && "$REAL_TOP_K_SOURCE_LAYER_START" != "0" ]]; then
    sideband_suffix="-realtopk${REAL_TOP_K_SOURCE_LAYER_START}"
    if [[ -n "$REAL_TOP_K_CHAIN_SOURCES" ]]; then
      local chain_suffix="${REAL_TOP_K_CHAIN_SOURCES//,/_}"
      sideband_suffix="${sideband_suffix}-chain${chain_suffix}"
    fi
  fi
  if [[ "$INDEXER_MODE" == "parallel" && "$mode" == "parallel" && -z "$threads_suffix" && -z "$cache_suffix" && -z "$decode_group_suffix" && -z "$mm_id_suffix" && -z "$sideband_suffix" ]]; then
    printf '%s\n' "$base_name"
  else
    case "$base_name" in
      default-l*|optin-l*)
        printf '%s-%s%s%s%s%s%s-l%s\n' "${base_name%%-l*}" "$mode" "$threads_suffix" "$cache_suffix" "$decode_group_suffix" "$mm_id_suffix" "$sideband_suffix" "${base_name#*-l}"
        ;;
      *)
        printf '%s-%s%s%s%s%s%s\n' "$base_name" "$mode" "$threads_suffix" "$cache_suffix" "$decode_group_suffix" "$mm_id_suffix" "$sideband_suffix"
        ;;
    esac
  fi
}

indexer_args() {
  local mode="$1"
  case "$mode" in
    serial)
      printf '%s\n%s\n' --parallel-lightning-indexer false
      ;;
    parallel)
      printf '%s\n%s\n' --parallel-lightning-indexer true
      ;;
  esac
}

warm_real_top_k_cache

if [[ -n "$TOKENS" ]]; then
  run_sweep_cases
else
  run_default_cases
fi
write_summary
