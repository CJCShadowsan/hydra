#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-/tmp/glm-dsa-contract-smoke/out.gguf}"
SOURCE_DIR="${SOURCE_DIR:-/tmp/glm-dsa-contract-smoke-source}"
WORK_DIR="${WORK_DIR:-/tmp/glm-dsa-indexshare-local-smoke}"
DEFAULT_LLAMA_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$ROOT/.deps/llama-build/build-stage-abi-static-metal}"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-$DEFAULT_LLAMA_BUILD_DIR/bin/llama-bench}"
SKIPPY_QUANTIZE_BIN="${SKIPPY_QUANTIZE_BIN:-$ROOT/target/debug/skippy-quantize}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-indexshare-local-smoke.sh [options]

Runs a local llama.cpp-only GLM-DSA IndexShare smoke. This deliberately does
not start Skippy split serving, lab nodes, mesh networking, or topology code.

The smoke proves:
  - a valid GLM-DSA GGUF loads and generates through native llama.cpp
  - native logs contain Full producer -> Shared consumer IndexShare flow
  - missing IndexShare metadata fails llama.cpp load
  - Shared layers with indexer tensors fail llama.cpp load
  - contradictory IndexShare role/frequency metadata fails llama.cpp load
  - MTP/NextN layers without complete indexer tensors fail llama.cpp load
  - missing split KV-B tensors fail llama.cpp load
  - stale unsplit attn_kv_b tensors fail llama.cpp load
  - invalid NextN and MoE hparams fail llama.cpp load

Options:
  --model PATH                Valid tiny GLM-DSA GGUF. Default: /tmp/glm-dsa-contract-smoke/out.gguf
  --source-dir PATH           Tiny SafeTensors fixture source dir used when --model is missing.
  --work-dir PATH             Output directory for logs/reports. Default: /tmp/glm-dsa-indexshare-local-smoke
  --llama-bench-bin PATH      llama-bench binary path.
  --skippy-quantize-bin PATH  skippy-quantize binary path.
  --skippy-bench-bin PATH     skippy-bench binary path.
  -h, --help                  Show this help.

Environment overrides mirror option names:
  MODEL, SOURCE_DIR, WORK_DIR, LLAMA_BENCH_BIN, SKIPPY_QUANTIZE_BIN, SKIPPY_BENCH_BIN.
  Set REBUILD_FIXTURE=1 to regenerate the tiny SafeTensors source and GGUF.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --llama-bench-bin)
      LLAMA_BENCH_BIN="$2"
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

run_llama() {
  local name="$1"
  local model="$2"
  local stdout="$WORK_DIR/${name}.stdout"
  local stderr="$WORK_DIR/${name}.stderr"

  set +e
  LLAMA_LOG_LEVEL=debug \
  LLAMA_GLM_DSA_INDEXSHARE_EXEC_LOG=1 \
    "$LLAMA_BENCH_BIN" \
      -v \
      -m "$model" \
      -p 1 \
      -n 1 \
      -ngl 0 \
      -r 1 \
      --no-warmup \
      >"$stdout" 2>"$stderr"
  local exit_code=$?
  set -e
  printf '%s\n' "$exit_code"
}

build_fixture_model() {
  require_executable "$SKIPPY_QUANTIZE_BIN"
  require_executable "$ROOT/scripts/glm-dsa-tiny-contract-fixture.py"

  rm -rf "$SOURCE_DIR"
  mkdir -p "$(dirname "$MODEL")"
  python3 "$ROOT/scripts/glm-dsa-tiny-contract-fixture.py" "$SOURCE_DIR" >"$WORK_DIR/fixture.stdout"

  "$SKIPPY_QUANTIZE_BIN" convert \
    --backend native-rust \
    --stream-buffer-bytes 8192 \
    --output-type bf16 \
    --expected-splits 1 \
    -o "$MODEL" \
    "$SOURCE_DIR" \
    >"$WORK_DIR/convert.stdout" \
    2>"$WORK_DIR/convert.stderr"
}

assert_log_contains() {
  local path="$1"
  local needle="$2"
  if ! grep -Fq "$needle" "$path"; then
    echo "expected log to contain: $needle" >&2
    echo "log: $path" >&2
    exit 1
  fi
}

require_executable "$LLAMA_BENCH_BIN"
require_executable "$SKIPPY_BENCH_BIN"
require_executable "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py"
require_executable "$ROOT/scripts/glm-dsa-inventory-verifier.py"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

if [[ "${REBUILD_FIXTURE:-0}" == "1" || ! -e "$MODEL" || ! -e "$SOURCE_DIR/config.json" ]]; then
  build_fixture_model
fi

require_file "$MODEL"

"$ROOT/scripts/glm-dsa-inventory-verifier.py" \
  --checkpoint "$SOURCE_DIR" \
  --gguf "$MODEL" \
  --expected-target-layers 3 \
  --expected-nextn-layers 1 \
  --json \
  >"$WORK_DIR/inventory.json"

missing_model="$WORK_DIR/missing-indexshare.gguf"
shared_bad_model="$WORK_DIR/shared-with-indexer.gguf"
odd_rope_model="$WORK_DIR/odd-rope.gguf"
short_indexer_model="$WORK_DIR/short-indexer-head.gguf"
frequency_conflict_model="$WORK_DIR/indexshare-frequency-conflict.gguf"
missing_mtp_indexer_model="$WORK_DIR/missing-mtp-indexer.gguf"
bad_nextn_model="$WORK_DIR/bad-nextn-count.gguf"
bad_experts_model="$WORK_DIR/bad-expert-counts.gguf"
missing_split_model="$WORK_DIR/missing-split-kv-b.gguf"
stale_unsplit_model="$WORK_DIR/stale-unsplit-kv-b.gguf"

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$missing_model" \
  --drop-indexshare-metadata

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$shared_bad_model" \
  --set-indexer-types full,shared,shared \
  --set-u32 glm-dsa.attention.indexer.top_k_frequency=3

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$odd_rope_model" \
  --set-u32 glm-dsa.rope.dimension_count=3

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$short_indexer_model" \
  --set-u32 glm-dsa.attention.indexer.key_length=2

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$frequency_conflict_model" \
  --set-u32 glm-dsa.attention.indexer.top_k_frequency=1

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$missing_mtp_indexer_model" \
  --rename-tensor blk.3.indexer.k_norm.weight=blk.3.indexer.k_norm_missing.weight \
  --rename-tensor blk.3.indexer.k_norm.bias=blk.3.indexer.k_norm_missing.bias \
  --rename-tensor blk.3.indexer.proj.weight=blk.3.indexer.proj_missing.weight \
  --rename-tensor blk.3.indexer.attn_k.weight=blk.3.indexer.attn_k_missing.weight \
  --rename-tensor blk.3.indexer.attn_q_b.weight=blk.3.indexer.attn_q_b_missing.weight

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$bad_nextn_model" \
  --set-u32 glm-dsa.nextn_predict_layers=4

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$bad_experts_model" \
  --set-u32 glm-dsa.expert_used_count=3

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$missing_split_model" \
  --rename-tensor blk.0.attn_v_b.weight=blk.0.attn_v_b_missing.weight

python3 "$ROOT/scripts/glm-dsa-gguf-contract-mutator.py" \
  "$MODEL" \
  "$stale_unsplit_model" \
  --rename-tensor blk.0.attn_v_b.weight=blk.0.attn_kv_b.weight \
  --set-tensor-shape blk.0.attn_kv_b.weight=2,5

good_status="$(run_llama good "$MODEL")"
if [[ "$good_status" != "0" ]]; then
  echo "valid GLM-DSA fixture failed native llama load; status=$good_status" >&2
  echo "stderr: $WORK_DIR/good.stderr" >&2
  exit 1
fi

"$SKIPPY_BENCH_BIN" \
  glm-dsa-op-report \
  --log "$WORK_DIR/good.stderr" \
  --require-indexshare-producer-consumer \
  --output "$WORK_DIR/indexshare-report.json" \
  >"$WORK_DIR/indexshare-report.stdout"

missing_status="$(run_llama missing "$missing_model")"
if [[ "$missing_status" == "0" ]]; then
  echo "missing IndexShare metadata fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/missing.stderr" \
  "GLM_DSA IndexShare metadata requires attention.indexer.types or attention.indexer.top_k_frequency"

shared_bad_status="$(run_llama shared_bad "$shared_bad_model")"
if [[ "$shared_bad_status" == "0" ]]; then
  echo "Shared-with-indexer fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/shared_bad.stderr" \
  "GLM_DSA IndexShare metadata declares Shared layer 2 with indexer tensors"

odd_rope_status="$(run_llama odd_rope "$odd_rope_model")"
if [[ "$odd_rope_status" == "0" ]]; then
  echo "odd RoPE fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/odd_rope.stderr" \
  "GLM_DSA rope.dimension_count must be positive and even"

short_indexer_status="$(run_llama short_indexer "$short_indexer_model")"
if [[ "$short_indexer_status" == "0" ]]; then
  echo "short indexer-head fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/short_indexer.stderr" \
  "GLM_DSA attention.indexer.key_length must be greater than rope.dimension_count"

frequency_conflict_status="$(run_llama frequency_conflict "$frequency_conflict_model")"
if [[ "$frequency_conflict_status" == "0" ]]; then
  echo "frequency-conflict fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/frequency_conflict.stderr" \
  "GLM_DSA attention.indexer.types conflicts with top_k_frequency at layer 1"

missing_mtp_indexer_status="$(run_llama missing_mtp_indexer "$missing_mtp_indexer_model")"
if [[ "$missing_mtp_indexer_status" == "0" ]]; then
  echo "missing MTP indexer fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/missing_mtp_indexer.stderr" \
  "GLM_DSA MTP/NextN layer 3 requires complete indexer tensors"

bad_nextn_status="$(run_llama bad_nextn "$bad_nextn_model")"
if [[ "$bad_nextn_status" == "0" ]]; then
  echo "bad NextN-count fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/bad_nextn.stderr" \
  "GLM_DSA nextn_predict_layers must be less than block_count"

bad_experts_status="$(run_llama bad_experts "$bad_experts_model")"
if [[ "$bad_experts_status" == "0" ]]; then
  echo "bad expert-count fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/bad_experts.stderr" \
  "expert_used_count must be less than or equal to expert_count"

missing_split_status="$(run_llama missing_split "$missing_split_model")"
if [[ "$missing_split_status" == "0" ]]; then
  echo "missing split KV-B fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/missing_split.stderr" \
  "GLM_DSA sparse attention requires split attn_k_b and attn_v_b tensors"

stale_unsplit_status="$(run_llama stale_unsplit "$stale_unsplit_model")"
if [[ "$stale_unsplit_status" == "0" ]]; then
  echo "stale unsplit KV-B fixture unexpectedly loaded" >&2
  exit 1
fi
assert_log_contains \
  "$WORK_DIR/stale_unsplit.stderr" \
  "GLM_DSA sparse attention does not support unsplit attn_kv_b tensors"

cat <<EOF
GLM-DSA native IndexShare smoke passed
  model: $MODEL
  source: $SOURCE_DIR
  inventory: $WORK_DIR/inventory.json
  report: $WORK_DIR/indexshare-report.json
  logs: $WORK_DIR
EOF
