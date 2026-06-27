#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="pair"
DRY_RUN=0
SYNC_REMOTE=0
BUILD_LOCAL=0
BUILD_REMOTE=0
RESET_STAGE_CACHE=0

RUN_PREFIX="${RUN_PREFIX:-glm52-li}"
PROMPT="${PROMPT:-What is the capital of France?}"
PROMPT_CORPUS="${PROMPT_CORPUS:-}"
PROMPT_LIMIT="${PROMPT_LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-128}"

CTX_SIZE="${CTX_SIZE:-131072}"
SPLITS="${SPLITS:-25}"
LAYER_END="${LAYER_END:-79}"
ACTIVATION_WIDTH="${ACTIVATION_WIDTH:-6144}"
ACTIVATION_WIRE_DTYPE="${ACTIVATION_WIRE_DTYPE:-f16}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
STAGE_MAX_INFLIGHT="${STAGE_MAX_INFLIGHT:-1}"
STAGE_TELEMETRY_LEVEL="${STAGE_TELEMETRY_LEVEL:-debug}"
STAGE_TELEMETRY_QUEUE_CAPACITY="${STAGE_TELEMETRY_QUEUE_CAPACITY:-32768}"
STAGE_DISABLE_MMAP_BUFFER="${STAGE_DISABLE_MMAP_BUFFER:-1}"

MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-GGUF:Q2_K-MTP-Q8}"
STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
STAGE_LOAD_MODE="${STAGE_LOAD_MODE:-layer-package}"

HOSTS="${HOSTS:-localhost,micstudio}"
ENDPOINT_HOST_MAP="${ENDPOINT_HOST_MAP:-localhost=192.168.0.5,micstudio=192.168.0.10}"
REMOTE_ROOT="${REMOTE_ROOT:-/Users/lab/models/skippy-runtime-bench}"
REMOTE_ROOT_MAP="${REMOTE_ROOT_MAP:-localhost=/Volumes/External/skippy-runtime-bench,micstudio=/Users/lab/models/skippy-runtime-bench}"
REMOTE_SHARED_ROOT_MAP="${REMOTE_SHARED_ROOT_MAP:-localhost=/Volumes/External/skippy-runtime-bench,micstudio=/Volumes/External/models/skippy-runtime-bench}"
WORK_DIR="${WORK_DIR:-/Volumes/External/skippy-runtime-bench}"
FIRST_STAGE_PORT="${FIRST_STAGE_PORT:-19240}"
METRICS_HTTP_ADDR="${METRICS_HTTP_ADDR:-127.0.0.1:18080}"
METRICS_OTLP_GRPC_ADDR="${METRICS_OTLP_GRPC_ADDR:-192.168.0.5:14317}"
METRICS_OTLP_GRPC_URL="${METRICS_OTLP_GRPC_URL:-http://${METRICS_OTLP_GRPC_ADDR}}"

STAGE_SERVER_BIN="${STAGE_SERVER_BIN:-target/release/skippy-server}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-target/release/skippy-bench}"
METRICS_SERVER_BIN="${METRICS_SERVER_BIN:-target/release/metrics-server}"
REMOTE_SOURCE_DIR="${REMOTE_SOURCE_DIR:-/Users/lab/src/mesh-llm-codex}"
REMOTE_BUILD_HOST="${REMOTE_BUILD_HOST:-micstudio}"

usage() {
  cat <<'EOF'
Usage: scripts/glm52-lightning-indexer-lab-bench.sh [options]

Runs the repeatable GLM 5.2 two-node lab benchmark used to compare the serial
and parallel Metal GLM-DSA Lightning Indexer paths.

Default topology:
  stage 0: studio54/localhost, layers 0..25, 192.168.0.5:19240
  stage 1: micstudio,         layers 25..79, 192.168.0.10:19241

Options:
  --mode serial|parallel|pair   Run serial only, parallel only, or serial then parallel. Default: pair.
  --prompt TEXT                 Prompt for the one-prompt smoke run.
  --prompt-corpus PATH          skippy-bench JSONL corpus path.
  --prompt-limit N              Limit prompt corpus rows.
  --max-new-tokens N            Generated token budget per prompt. Default: 8.
  --run-prefix NAME             Prefix for generated run ids. Default: glm52-li.
  --work-dir PATH               Local benchmark work dir. Default: /Volumes/External/skippy-runtime-bench.
  --stage-model PATH            Local GLM 5.2 layer package path.
  --sync-remote                 Sync this source tree to micstudio before running.
  --build-local                 Build local release skippy-bench/skippy-server/metrics-server first.
  --build-remote                Build remote release skippy-server first.
  --reset-stage-cache           Remove cached materialized stage GGUFs for the selected topology first.
  --dry-run                     Print commands without executing.
  -h, --help                    Show this help.

Useful environment overrides:
  HOSTS, ENDPOINT_HOST_MAP, SPLITS, CTX_SIZE, STAGE_MAX_INFLIGHT,
  STAGE_DISABLE_MMAP_BUFFER, STAGE_SERVER_BIN, SKIPPY_BENCH_BIN,
  METRICS_SERVER_BIN, REMOTE_SOURCE_DIR.

Examples:
  scripts/glm52-lightning-indexer-lab-bench.sh --mode serial
  scripts/glm52-lightning-indexer-lab-bench.sh --mode parallel
  scripts/glm52-lightning-indexer-lab-bench.sh --mode pair --max-new-tokens 16
  PROMPT_CORPUS=crates/skippy-bench/corpora/glm_dsa_long_context_coding_prompts.jsonl \
    scripts/glm52-lightning-indexer-lab-bench.sh --mode pair --prompt-limit 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --prompt-corpus)
      PROMPT_CORPUS="$2"
      shift 2
      ;;
    --prompt-limit)
      PROMPT_LIMIT="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --stage-model)
      STAGE_MODEL="$2"
      shift 2
      ;;
    --sync-remote)
      SYNC_REMOTE=1
      shift
      ;;
    --build-local)
      BUILD_LOCAL=1
      shift
      ;;
    --build-remote)
      BUILD_REMOTE=1
      shift
      ;;
    --reset-stage-cache)
      RESET_STAGE_CACHE=1
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

case "$MODE" in
  serial|parallel|pair) ;;
  *)
    echo "invalid --mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "required path not found: $path" >&2
    exit 1
  fi
}

safe_cache_component() {
  LC_ALL=C sed -E 's/[^A-Za-z0-9_.-]/_/g' <<<"$1"
}

remote_shared_roots() {
  local value="$REMOTE_SHARED_ROOT_MAP"
  local entry
  tr ',' '\n' <<<"$value" | while IFS= read -r entry; do
    [[ -n "$entry" ]] || continue
    printf '%s\n' "${entry#*=}"
  done
}

reset_stage_cache() {
  local variant="$1"
  local model_key
  local topology_key
  model_key="$(safe_cache_component "$MODEL_ID")"
  topology_key="$(safe_cache_component "glm52-dsa-li-${variant}")"

  local -a roots=("$WORK_DIR")
  while IFS= read -r root; do
    roots+=("$root")
  done < <(remote_shared_roots)

  local root
  local seen_roots=$'\n'
  for root in "${roots[@]}"; do
    [[ -n "$root" && "$root" == /* ]] || continue
    if [[ "$seen_roots" == *$'\n'"$root"$'\n'* ]]; then
      continue
    fi
    seen_roots+="$root"$'\n'
    run_cmd rm -rf "$root/model-cache/$model_key/$topology_key"
  done
}

build_local() {
  run_cmd just with-lld cargo build --release --locked \
    -p skippy-bench -p skippy-server -p metrics-server
}

sync_remote() {
  run_cmd "$ROOT/scripts/sync-lab-source.sh" "$REMOTE_BUILD_HOST" "$REMOTE_SOURCE_DIR"
}

build_remote() {
  local remote_script="/tmp/codex-build-glm52-li-skippy-server.sh"
  if [[ "$DRY_RUN" == "1" ]]; then
    cat <<EOF
+ ssh $REMOTE_BUILD_HOST 'cat > $remote_script && chmod +x $remote_script' <<'REMOTE'
#!/usr/bin/env zsh
set -euo pipefail
cd "$REMOTE_SOURCE_DIR"
export PATH="/opt/homebrew/bin:\$PATH"
export RUSTC_WRAPPER=/opt/homebrew/bin/sccache
export LLAMA_STAGE_BACKEND=metal
export SKIPPY_FORCE_LLAMA_BUILD=1
scripts/prepare-llama.sh pinned
scripts/build-llama.sh
just with-lld cargo build --release --locked -p skippy-server
REMOTE
+ ssh $REMOTE_BUILD_HOST '/bin/zsh -ilc zsh $remote_script'
EOF
    return
  fi
  ssh "$REMOTE_BUILD_HOST" "cat > $remote_script && chmod +x $remote_script" <<EOF
#!/usr/bin/env zsh
set -euo pipefail
cd "$REMOTE_SOURCE_DIR"
export PATH="/opt/homebrew/bin:\$PATH"
export RUSTC_WRAPPER=/opt/homebrew/bin/sccache
export LLAMA_STAGE_BACKEND=metal
export SKIPPY_FORCE_LLAMA_BUILD=1
scripts/prepare-llama.sh pinned
scripts/build-llama.sh
just with-lld cargo build --release --locked -p skippy-server
EOF
  run_cmd ssh -tt "$REMOTE_BUILD_HOST" "/bin/zsh -ilc 'zsh $remote_script'"
}

run_one() {
  local variant="$1"
  local run_id="${RUN_PREFIX}-${variant}-$(date +%Y%m%d-%H%M%S)"
  local topology_id="glm52-dsa-li-${variant}"
  local output="${WORK_DIR}/${run_id}/driver-result.json"
  local -a env_prefix=(SKIPPY_BINARY_WARM_PRECONNECT=1)

  if [[ "$variant" == "parallel" ]]; then
    env_prefix+=(LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER=1)
  else
    env_prefix+=(LLAMA_GLM_DSA_PARALLEL_LIGHTNING_INDEXER=0)
  fi

  if [[ "$RESET_STAGE_CACHE" == "1" ]]; then
    reset_stage_cache "$variant"
  fi

  local -a cmd=(
    "$SKIPPY_BENCH_BIN" run
    --metrics-server-bin "$METRICS_SERVER_BIN"
    --stage-server-bin "$STAGE_SERVER_BIN"
    --hosts "$HOSTS"
    --run-id "$run_id"
    --topology-id "$topology_id"
    --model-id "$MODEL_ID"
    --stage-model "$STAGE_MODEL"
    --stage-load-mode "$STAGE_LOAD_MODE"
    --splits "$SPLITS"
    --layer-end "$LAYER_END"
    --ctx-size "$CTX_SIZE"
    --activation-width "$ACTIVATION_WIDTH"
    --activation-wire-dtype "$ACTIVATION_WIRE_DTYPE"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --prefill-chunk-size "$PREFILL_CHUNK_SIZE"
    --work-dir "$WORK_DIR"
    --remote-root "$REMOTE_ROOT"
    --remote-root-map "$REMOTE_ROOT_MAP"
    --remote-shared-root-map "$REMOTE_SHARED_ROOT_MAP"
    --endpoint-host-map "$ENDPOINT_HOST_MAP"
    --first-stage-port "$FIRST_STAGE_PORT"
    --metrics-http-addr "$METRICS_HTTP_ADDR"
    --metrics-otlp-grpc-addr "$METRICS_OTLP_GRPC_ADDR"
    --metrics-otlp-grpc-url "$METRICS_OTLP_GRPC_URL"
    --execute-remote
    --allow-unbalanced-stages
    --rsync-model-artifacts
    --stage-connectivity-probe
    --stage-connectivity-diagnostics
    --keep-remote-on-failure
    --startup-timeout-secs 1800
    --stage-max-inflight "$STAGE_MAX_INFLIGHT"
    --stage-telemetry-queue-capacity "$STAGE_TELEMETRY_QUEUE_CAPACITY"
    --stage-telemetry-level "$STAGE_TELEMETRY_LEVEL"
    --glm-dsa-op-timing
    --glm-dsa-direct-sparse-attn
    --output "$output"
  )
  if [[ "$STAGE_DISABLE_MMAP_BUFFER" == "1" ]]; then
    cmd+=(--stage-disable-mmap-buffer)
  fi

  if [[ -n "$PROMPT_CORPUS" ]]; then
    cmd+=(--prompt-corpus "$PROMPT_CORPUS")
  else
    cmd+=(--prompt "$PROMPT")
  fi
  if [[ -n "$PROMPT_LIMIT" ]]; then
    cmd+=(--prompt-limit "$PROMPT_LIMIT")
  fi

  printf '== GLM 5.2 Lightning Indexer %s run ==\n' "$variant"
  printf 'run_id=%s\n' "$run_id"
  printf 'work_dir=%s\n' "$WORK_DIR"
  printf '+'
  printf ' %q' "${env_prefix[@]}" "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    env "${env_prefix[@]}" "${cmd[@]}"
    write_op_reports "$WORK_DIR/$run_id"
  fi
}

write_op_reports() {
  local run_dir="$1"
  local report_dir="$run_dir/glm-dsa-op-reports"
  mkdir -p "$report_dir"
  local found=0
  local reported=0
  while IFS= read -r -d '' log_path; do
    found=1
    if ! grep -q 'skippy: glm_dsa_op_timing ' "$log_path"; then
      printf 'skipping log without GLM-DSA timing: %s\n' "$log_path" >&2
      continue
    fi
    reported=1
    local base
    base="$(basename "$log_path" .log)"
    run_cmd "$SKIPPY_BENCH_BIN" glm-dsa-op-report \
      --log "$log_path" \
      --output "$report_dir/${base}.json"
  done < <(find "$run_dir" -type f -name '*.log' -print0 2>/dev/null)
  if [[ "$found" == "1" ]]; then
    if [[ "$reported" == "1" ]]; then
      printf 'wrote GLM-DSA op reports under %s\n' "$report_dir"
    else
      printf 'no logs with GLM-DSA timing found under %s\n' "$run_dir" >&2
    fi
  else
    printf 'no stage logs found for GLM-DSA op reports under %s\n' "$run_dir" >&2
  fi
}

main() {
  cd "$ROOT"
  if [[ "$BUILD_LOCAL" == "1" ]]; then
    build_local
  fi
  if [[ "$SYNC_REMOTE" == "1" ]]; then
    sync_remote
  fi
  if [[ "$BUILD_REMOTE" == "1" ]]; then
    build_remote
  fi

  require_file "$SKIPPY_BENCH_BIN"
  require_file "$STAGE_SERVER_BIN"
  require_file "$METRICS_SERVER_BIN"
  require_file "$STAGE_MODEL/model-package.json"
  mkdir -p "$WORK_DIR"

  case "$MODE" in
    serial)
      run_one serial
      ;;
    parallel)
      run_one parallel
      ;;
    pair)
      run_one serial
      run_one parallel
      ;;
  esac
}

main
