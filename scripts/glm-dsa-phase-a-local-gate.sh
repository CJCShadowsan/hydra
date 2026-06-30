#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD="${BUILD:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_BACKEND_OPS="${RUN_BACKEND_OPS:-1}"
BACKENDS="${BACKENDS:-CPU,MTL0}"
JOBS="${JOBS:-8}"
BF16_DRY_RUN="${BF16_DRY_RUN:-auto}"
BF16_SHARD="${BF16_SHARD:-1}"
BF16_SOURCE="${BF16_SOURCE:-}"
BF16_TARGET="${BF16_TARGET:-}"
SKIPPY_QUANTIZE_BIN="${SKIPPY_QUANTIZE_BIN:-$ROOT/target/debug/skippy-quantize}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"

LLAMA_BENCH_BUILD_DIR="${LLAMA_BENCH_BUILD_DIR:-$ROOT/.deps/llama-build/build-glm52-deps-metal}"
BACKEND_OPS_BUILD_DIR="${BACKEND_OPS_BUILD_DIR:-$ROOT/.deps/llama-build/build-metal-tests}"

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-phase-a-local-gate.sh [options]

Runs the local, non-destructive Phase A GLM-DSA gate:
  - script syntax and Python syntax checks
  - native llama-only GLM-DSA contract smoke
  - CPU/Metal backend-op parity for GLM-DSA required ops
  - optional BF16 shard dry-run verification only; never repairs shards

This script deliberately does not start Skippy split serving, lab nodes, mesh
networking, topology code, stage-boundary ABI work, or layer placement work.

Options:
  --skip-build              Do not build local binaries before running checks.
  --skip-smoke              Skip native GLM-DSA contract smoke.
  --skip-backend-ops        Skip backend-op parity tests.
  --backends LIST           Comma-separated backend list. Default: CPU,MTL0.
  --jobs N                  Build parallelism. Default: 8.
  --bf16-dry-run MODE       auto, off, or required. Default: auto.
  --bf16-source PATH        Source GLM-5.2 SafeTensors checkpoint for dry-run.
  --bf16-target PATH        BF16 GGUF repo root for dry-run.
  --bf16-shard N            BF16 shard number for dry-run. Default: 1.
  --skippy-quantize-bin PATH
                            skippy-quantize binary. Default: target/debug/skippy-quantize.
  --skippy-bench-bin PATH   skippy-bench binary. Default: target/debug/skippy-bench.
  -h, --help                Show this help.

Environment overrides mirror option names:
  BUILD, RUN_SMOKE, RUN_BACKEND_OPS, BACKENDS, JOBS, BF16_DRY_RUN,
  BF16_SOURCE, BF16_TARGET, BF16_SHARD, SKIPPY_QUANTIZE_BIN,
  SKIPPY_BENCH_BIN.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      BUILD=0
      shift
      ;;
    --skip-smoke)
      RUN_SMOKE=0
      shift
      ;;
    --skip-backend-ops)
      RUN_BACKEND_OPS=0
      shift
      ;;
    --backends)
      BACKENDS="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --bf16-dry-run)
      BF16_DRY_RUN="$2"
      shift 2
      ;;
    --bf16-source)
      BF16_SOURCE="$2"
      shift 2
      ;;
    --bf16-target)
      BF16_TARGET="$2"
      shift 2
      ;;
    --bf16-shard)
      BF16_SHARD="$2"
      shift 2
      ;;
    --skippy-quantize-bin)
      SKIPPY_QUANTIZE_BIN="$2"
      shift 2
      ;;
    --skippy-bench-bin)
      SKIPPY_BENCH_BIN="$2"
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

phase() {
  printf '\n== %s ==\n' "$1"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "required path not found: $path" >&2
    exit 1
  fi
}

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "required executable not found: $path" >&2
    exit 1
  fi
}

detect_bf16_source() {
  local candidates=(
    "/Volumes/External/models/huggingface/hub/models--zai-org--GLM-5.2/snapshots/53783022a4d492a25927417d22698a9535b743a4"
    "/Users/lab/models/huggingface/hub/models--zai-org--GLM-5.2/snapshots/53783022a4d492a25927417d22698a9535b743a4"
    "/Volumes/models/huggingface/hub/models--zai-org--GLM-5.2/snapshots/53783022a4d492a25927417d22698a9535b743a4"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

detect_bf16_target() {
  local candidates=(
    "/Users/lab/glm52-work/bf16-gguf"
    "/Volumes/External/models/glm52-work/bf16-gguf"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate/BF16" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

syntax_checks() {
  phase "syntax"
  python3 -m py_compile \
    "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
    "$ROOT/scripts/glm-dsa-inventory-verifier.py" \
    "$ROOT/scripts/glm-dsa-tiny-contract-fixture.py"

  bash -n \
    "$ROOT/scripts/glm-dsa-indexshare-local-smoke.sh" \
    "$ROOT/scripts/glm-dsa-bf16-rebuild-window.sh" \
    "$ROOT/scripts/glm-dsa-bf16-rebuild-window-test.sh" \
    "$ROOT/scripts/glm-dsa-phase-a-local-gate.sh"
}

run_rebuild_helper_fixture() {
  phase "BF16 rebuild helper fixture"
  bash "$ROOT/scripts/glm-dsa-bf16-rebuild-window-test.sh"
}

build_local_tools() {
  if [[ "$BUILD" != "1" ]]; then
    return
  fi

  phase "build Rust tools"
  (cd "$ROOT" && just skippy-quantize-build)
  (cd "$ROOT" && just with-lld cargo build -p skippy-bench)

  phase "build llama.cpp GLM-DSA test binaries"
  cmake --build "$LLAMA_BENCH_BUILD_DIR" --target llama-bench -j "$JOBS"
  cmake --build "$BACKEND_OPS_BUILD_DIR" --target test-backend-ops -j "$JOBS"
}

run_contract_smoke() {
  if [[ "$RUN_SMOKE" != "1" ]]; then
    return
  fi

  phase "native llama-only GLM-DSA contract smoke"
  require_executable "$SKIPPY_QUANTIZE_BIN"
  require_executable "$SKIPPY_BENCH_BIN"
  require_executable "$LLAMA_BENCH_BUILD_DIR/bin/llama-bench"
  REBUILD_FIXTURE=1 \
    SKIPPY_QUANTIZE_BIN="$SKIPPY_QUANTIZE_BIN" \
    SKIPPY_BENCH_BIN="$SKIPPY_BENCH_BIN" \
    "$ROOT/scripts/glm-dsa-indexshare-local-smoke.sh"
}

run_backend_ops() {
  if [[ "$RUN_BACKEND_OPS" != "1" ]]; then
    return
  fi

  phase "GLM-DSA backend op parity"
  local test_bin="$BACKEND_OPS_BUILD_DIR/bin/test-backend-ops"
  require_executable "$test_bin"

  local backends=()
  IFS=',' read -r -a backends <<<"$BACKENDS"
  local backend
  local op
  for backend in "${backends[@]}"; do
    for op in LIGHTNING_INDEXER DSA_SPARSE_MASK DSA_SPARSE_ATTN; do
      "$test_bin" test -o "$op" -b "$backend" -j 1
    done
  done
}

run_bf16_dry_run() {
  case "$BF16_DRY_RUN" in
    off)
      return
      ;;
    auto|required)
      ;;
    *)
      echo "--bf16-dry-run must be auto, off, or required, got: $BF16_DRY_RUN" >&2
      exit 2
      ;;
  esac

  if [[ -z "$BF16_SOURCE" ]]; then
    BF16_SOURCE="$(detect_bf16_source || true)"
  fi
  if [[ -z "$BF16_TARGET" ]]; then
    BF16_TARGET="$(detect_bf16_target || true)"
  fi

  if [[ -z "$BF16_SOURCE" || -z "$BF16_TARGET" ]]; then
    if [[ "$BF16_DRY_RUN" == "required" ]]; then
      echo "BF16 dry-run required but source or target path was not found" >&2
      exit 1
    fi
    phase "BF16 dry-run skipped"
    echo "source: ${BF16_SOURCE:-not found}"
    echo "target: ${BF16_TARGET:-not found}"
    echo "This is expected on hosts without the micstudio BF16 artifact mounted."
    return
  fi

  phase "BF16 shard dry-run"
  require_executable "$SKIPPY_QUANTIZE_BIN"
  local bf16_log="${TMPDIR:-/tmp}/glm-dsa-bf16-dry-run.$$.log"
  set +e
  "$ROOT/scripts/glm-dsa-bf16-rebuild-window.sh" \
    --shard "$BF16_SHARD" \
    --source "$BF16_SOURCE" \
    --target "$BF16_TARGET" \
    --skippy-quantize-bin "$SKIPPY_QUANTIZE_BIN" \
    2>&1 | tee "$bf16_log"
  local bf16_status=${PIPESTATUS[0]}
  set -e
  if [[ "$bf16_status" != "0" ]]; then
    return "$bf16_status"
  fi
  if grep -Eq 'current shard verifier status: [1-9][0-9]*|current shard is missing|GGUF inventory failed' "$bf16_log"; then
    echo "BF16 dry-run found an artifact that does not satisfy the GLM-DSA Phase A contract" >&2
    return 1
  fi
}

syntax_checks
run_rebuild_helper_fixture
build_local_tools
run_contract_smoke
run_backend_ops
run_bf16_dry_run

phase "Phase A local gate complete"
