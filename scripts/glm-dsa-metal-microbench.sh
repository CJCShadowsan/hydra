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
  --warmup N               Warmup iterations per case. Default: 0.
  --no-force-rebuild       Skip forced static-metal rebuild/relink.
  --build-only             Rebuild/relink only; do not run cases.
  --dry-run                Print commands without executing.
  -h, --help               Show this help.

Environment overrides mirror option names:
  STAGE_MODEL, MODEL_ID, OUTPUT_DIR, LAYER_START, LAYER_END, CTX_SIZE,
  ACTIVATION_WIDTH, ITERATIONS, WARMUP.
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
    --warmup)
      WARMUP="$2"
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
  local tokens="$2"
  shift 2

  local report="$OUTPUT_DIR/${name}.json"
  local stdout="$OUTPUT_DIR/${name}.stdout"
  local -a cmd=(
    "$ROOT/target/debug/skippy-bench"
    glm-dsa-layer-microbench
    --stage-model "$STAGE_MODEL"
    --model-id "$MODEL_ID"
    --layer-start "$LAYER_START"
    --layer-end "$LAYER_END"
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
import sys

base = pathlib.Path(sys.argv[1])
cases = [
    "default-1",
    "default-33",
    "optin-prefill-32",
    "optin-prefill-33",
]

print(f"output_dir={base}")
for name in cases:
    path = base / f"{name}.json"
    print()
    print(name)
    if not path.exists():
        print("  missing")
        continue
    report = json.loads(path.read_text())
    comparison = report["comparison"]
    parity = comparison["parity"]
    candidate = comparison["candidate"]
    timing = candidate["op_timing_records"][0]
    decisions = candidate.get("direct_sparse_decision_records", [])
    use_direct = sum(1 for record in decisions if record.get("use_direct"))
    fallback = len(decisions) - use_direct
    print(f"  parity={parity['passed']} hidden_mismatches={parity['hidden_mismatches']} sideband_mismatches={parity['sideband_mismatched_bytes']}")
    print(f"  dsa_sparse_attn_nodes={timing.get('dsa_sparse_attn_nodes')} sparse_mask_nodes={timing.get('sparse_mask_nodes')}")
    print(f"  decisions={len(decisions)} use_direct={use_direct} fallback={fallback}")
    if decisions:
        last = decisions[-1]
        print(
            "  last_decision="
            f"tokens={last['ubatch_tokens']} sparse_batch={last['sparse_batch']} "
            f"prefill_enabled={last['prefill_enabled']} prefill_shape={last['prefill_shape']} "
            f"decode_shape={last['decode_shape']} use_direct={last['use_direct']}"
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

run_case default-1 1
run_case default-33 33
run_case optin-prefill-32 32 --direct-sparse-prefill true
run_case optin-prefill-33 33 --direct-sparse-prefill true
write_summary
