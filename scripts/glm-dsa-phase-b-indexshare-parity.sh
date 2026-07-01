#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORK_DIR="${WORK_DIR:-/tmp/glm-dsa-phase-b-indexshare-parity}"
SOURCE_DIR="${SOURCE_DIR:-}"
MODEL="${MODEL:-}"
PACKAGE_DIR="${PACKAGE_DIR:-}"
REPORT="${REPORT:-}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-DSA-Phase-B-Tiny:BF16}"
SKIPPY_QUANTIZE_BIN="${SKIPPY_QUANTIZE_BIN:-$ROOT/target/debug/skippy-quantize}"
SKIPPY_MODEL_PACKAGE_BIN="${SKIPPY_MODEL_PACKAGE_BIN:-$ROOT/target/debug/skippy-model-package}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
LLAMA_STAGE_BACKEND="${LLAMA_STAGE_BACKEND:-metal}"
INDEXER_TYPES="${INDEXER_TYPES:-full,full,shared}"
INDEX_TOPK_FREQ="${INDEX_TOPK_FREQ:-2}"
INDEX_SKIP_TOPK_OFFSET="${INDEX_SKIP_TOPK_OFFSET:-2}"
LAYER_START="${LAYER_START:-1}"
LAYER_END="${LAYER_END:-}"
TOKENS="${TOKENS:-1}"
N_BATCH="${N_BATCH:-$TOKENS}"
N_UBATCH="${N_UBATCH:-$TOKENS}"

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-phase-b-indexshare-parity.sh [options]

Builds a fresh tiny GLM-DSA GGUF, writes a disposable layer package, and runs a
single-process native Phase B proof over a selected layer span:

  layer 1: Full producer creates real top-k
  layer 2+: Shared consumers reuse that top-k

The script deliberately does not use the stale GLM-5.2 derived layer package,
does not start lab nodes, and does not touch Skippy split topology.

Options:
  --work-dir PATH              Working directory. Default: /tmp/glm-dsa-phase-b-indexshare-parity
  --source-dir PATH            Tiny SafeTensors source directory.
  --model PATH                 Tiny converted GGUF path.
  --package-dir PATH           Disposable layer package output path.
  --report PATH                Microbench JSON report path.
  --indexer-types LIST         Comma-separated roles. Default: full,full,shared.
  --index-topk-freq N          Fixture index_topk_freq. Default: 2.
  --index-skip-topk-offset N   Fixture index_skip_topk_offset. Default: 2.
  --layer-start N              Microbench layer start. Default: 1.
  --layer-end N                Microbench layer end. Default: role count.
  --tokens N                   Tokens per measured run. Default: 1.
  --n-batch N                  Batch size. Default: TOKENS.
  --n-ubatch N                 Microbatch size. Default: TOKENS.
  --skip-build                 Do not build local Rust tools first.
  -h, --help                   Show this help.

Environment overrides mirror option names:
  WORK_DIR, SOURCE_DIR, MODEL, PACKAGE_DIR, REPORT, MODEL_ID,
  SKIPPY_QUANTIZE_BIN, SKIPPY_MODEL_PACKAGE_BIN, SKIPPY_BENCH_BIN,
  LLAMA_STAGE_BACKEND, LLAMA_STAGE_BUILD_DIR, INDEXER_TYPES, INDEX_TOPK_FREQ,
  INDEX_SKIP_TOPK_OFFSET, LAYER_START, LAYER_END, TOKENS, N_BATCH, N_UBATCH.
EOF
}

SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --package-dir)
      PACKAGE_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    --indexer-types)
      INDEXER_TYPES="$2"
      shift 2
      ;;
    --index-topk-freq)
      INDEX_TOPK_FREQ="$2"
      shift 2
      ;;
    --index-skip-topk-offset)
      INDEX_SKIP_TOPK_OFFSET="$2"
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
    --tokens)
      TOKENS="$2"
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
    --skip-build)
      SKIP_BUILD=1
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

SOURCE_DIR="${SOURCE_DIR:-$WORK_DIR/source}"
MODEL="${MODEL:-$WORK_DIR/out.gguf}"
PACKAGE_DIR="${PACKAGE_DIR:-$WORK_DIR/package}"
REPORT="${REPORT:-$WORK_DIR/phase-b-indexshare-parity.json}"
LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$(LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" "$ROOT/scripts/build-llama.sh" --print-build-dir)}"
ROLE_COUNT="$(python3 - "$INDEXER_TYPES" <<'PY'
import sys
roles = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(len(roles))
PY
)"
LAYER_END="${LAYER_END:-$ROLE_COUNT}"
FIRST_SHARED_LAYER="$(python3 - "$INDEXER_TYPES" "$LAYER_START" "$LAYER_END" <<'PY'
import sys
roles = [item.strip().lower() for item in sys.argv[1].split(",") if item.strip()]
layer_start = int(sys.argv[2])
layer_end = int(sys.argv[3])
for layer, role in enumerate(roles):
    if layer_start <= layer < layer_end and role == "shared":
        print(layer)
        break
PY
)"

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "required executable not found: $path" >&2
    exit 1
  fi
}

mkdir -p "$WORK_DIR"

if [[ "$SKIP_BUILD" == "0" ]]; then
  (cd "$ROOT" && just skippy-quantize-build)
  (cd "$ROOT" && env LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" scripts/build-llama.sh)
  (cd "$ROOT" && cargo clean -p skippy-ffi -p skippy-runtime -p skippy-model-package -p skippy-bench)
  (cd "$ROOT" && env LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" just with-lld cargo build -p skippy-model-package -p skippy-bench)
fi

require_executable "$SKIPPY_QUANTIZE_BIN"
require_executable "$SKIPPY_MODEL_PACKAGE_BIN"
require_executable "$SKIPPY_BENCH_BIN"
require_executable "$ROOT/scripts/glm-dsa-tiny-contract-fixture.py"

rm -rf "$SOURCE_DIR" "$PACKAGE_DIR"
rm -f "$MODEL" "$REPORT"
mkdir -p "$(dirname "$MODEL")" "$(dirname "$REPORT")"

python3 "$ROOT/scripts/glm-dsa-tiny-contract-fixture.py" \
  --indexer-types "$INDEXER_TYPES" \
  --index-topk-freq "$INDEX_TOPK_FREQ" \
  --index-skip-topk-offset "$INDEX_SKIP_TOPK_OFFSET" \
  "$SOURCE_DIR" \
  >"$WORK_DIR/fixture.stdout"

"$SKIPPY_QUANTIZE_BIN" convert \
  --backend native-rust \
  --stream-buffer-bytes 8192 \
  --output-type bf16 \
  --expected-splits 1 \
  -o "$MODEL" \
  "$SOURCE_DIR" \
  >"$WORK_DIR/convert.stdout" \
  2>"$WORK_DIR/convert.stderr"

env LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
  "$SKIPPY_MODEL_PACKAGE_BIN" write-package \
  "$MODEL" \
  --out-dir "$PACKAGE_DIR" \
  --model-id "$MODEL_ID" \
  --source-repo meshllm/GLM-DSA-Phase-B-Tiny \
  --source-revision local-fresh \
  --source-file "$(basename "$MODEL")" \
  >"$WORK_DIR/write-package.stdout" \
  2>"$WORK_DIR/write-package.stderr"

env LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
  "$SKIPPY_MODEL_PACKAGE_BIN" preflight "$PACKAGE_DIR" --verify-sha256 \
  >"$WORK_DIR/preflight.stdout" \
  2>"$WORK_DIR/preflight.stderr"

env LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
  "$SKIPPY_MODEL_PACKAGE_BIN" glm-dsa-contract "$PACKAGE_DIR" \
  >"$WORK_DIR/glm-dsa-contract.stdout" \
  2>"$WORK_DIR/glm-dsa-contract.stderr"

env \
  LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" \
  LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
  SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_CACHE_DIR="$WORK_DIR/real-top-k-cache" \
  SKIPPY_BENCH_GLM_DSA_REAL_TOP_K_REQUIRE_CACHE=0 \
  "$SKIPPY_BENCH_BIN" glm-dsa-layer-microbench \
    --stage-model "$PACKAGE_DIR" \
    --model-id "$MODEL_ID" \
    --layer-start "$LAYER_START" \
    --layer-end "$LAYER_END" \
    --tokens "$TOKENS" \
    --ctx-size 128 \
    --activation-width 4 \
    --iterations 1 \
    --warmup 0 \
    --n-gpu-layers 0 \
    --n-batch "$N_BATCH" \
    --n-ubatch "$N_UBATCH" \
    --direct-sparse-attn false \
    --op-timing true \
    --compare-native-indexshare-producer-consumer \
    --require-native-indexshare-proof \
    --allow-concurrent \
    --output "$REPORT" \
    >"$WORK_DIR/microbench.stdout" \
    2>"$WORK_DIR/microbench.stderr"

if [[ -n "$FIRST_SHARED_LAYER" ]]; then
  missing_topk_report="$WORK_DIR/shared-missing-topk-negative.json"
  missing_topk_stdout="$WORK_DIR/shared-missing-topk-negative.stdout"
  missing_topk_stderr="$WORK_DIR/shared-missing-topk-negative.stderr"
  missing_topk_end=$((FIRST_SHARED_LAYER + 1))
  set +e
  env \
    LLAMA_STAGE_BACKEND="$LLAMA_STAGE_BACKEND" \
    LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    "$SKIPPY_BENCH_BIN" glm-dsa-layer-microbench \
      --stage-model "$PACKAGE_DIR" \
      --model-id "$MODEL_ID" \
      --layer-start "$FIRST_SHARED_LAYER" \
      --layer-end "$missing_topk_end" \
      --tokens "$TOKENS" \
      --ctx-size 128 \
      --activation-width 4 \
      --iterations 1 \
      --warmup 0 \
      --n-gpu-layers 0 \
      --n-batch "$N_BATCH" \
      --n-ubatch "$N_UBATCH" \
      --direct-sparse-attn false \
      --op-timing true \
      --allow-concurrent \
      --output "$missing_topk_report" \
      >"$missing_topk_stdout" \
      2>"$missing_topk_stderr"
  missing_topk_status=$?
  set -e
  if [[ "$missing_topk_status" == "0" ]]; then
    echo "Shared-only missing-top-k negative case unexpectedly passed" >&2
    exit 1
  fi
  if ! grep -Eq \
    "GLM-DSA consumer slices require top-k sideband input|GLM_DSA split starts inside an IndexShare consumer group without top-k sideband input" \
    "$missing_topk_stderr"; then
    echo "Shared-only missing-top-k negative case failed with unexpected error" >&2
    echo "stderr: $missing_topk_stderr" >&2
    exit 1
  fi
fi

python3 - "$REPORT" "$INDEXER_TYPES" "$LAYER_START" "$LAYER_END" "$TOKENS" <<'PY'
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
indexer_types = [item.strip().lower() for item in sys.argv[2].split(",") if item.strip()]
layer_start = int(sys.argv[3])
layer_end = int(sys.argv[4])
tokens = int(sys.argv[5])
report = json.loads(report_path.read_text())
comparison = report.get("comparison") or {}
parity = comparison.get("parity") or {}
guard = report.get("native_indexshare_guard") or {}
trace = (comparison.get("baseline") or {}).get("indexshare_trace_summary") or {}
candidate_trace = (comparison.get("candidate") or {}).get("indexshare_trace_summary") or {}
expected_full = [
    layer
    for layer, role in enumerate(indexer_types)
    if layer_start <= layer < layer_end and role == "full"
]
expected_shared = [
    layer
    for layer, role in enumerate(indexer_types)
    if layer_start <= layer < layer_end and role == "shared"
]

failures = []
if not parity.get("passed"):
    failures.append(f"parity failed: {parity}")
if not guard.get("passed"):
    failures.append(f"native indexshare guard failed: {guard}")
if not expected_full:
    failures.append(f"selected span {layer_start}..{layer_end} has no expected Full producer")
if not expected_shared:
    failures.append(f"selected span {layer_start}..{layer_end} has no expected Shared consumer")
if sorted(trace.get("full_layers") or []) != expected_full:
    failures.append(f"unexpected Full layers: got {trace.get('full_layers')} expected {expected_full}")
if sorted(trace.get("shared_layers") or []) != expected_shared:
    failures.append(f"unexpected Shared layers: got {trace.get('shared_layers')} expected {expected_shared}")
if trace.get("full_exec_records", 0) < 1:
    failures.append(f"missing Full producer trace: {trace}")
if trace.get("shared_exec_records", 0) < len(expected_shared):
    failures.append(f"missing Shared consumer trace: {trace}")
if trace.get("shared_exec_missing_input_top_k", 0) != 0:
    failures.append(f"Shared consumer missed top-k input: {trace}")
if trace.get("shared_exec_with_input_top_k", 0) < trace.get("shared_exec_records", 0):
    failures.append(f"some Shared records did not receive input top-k: {trace}")
if trace.get("consume_records", 0) < len(expected_shared):
    failures.append(f"missing consume trace: {trace}")
if candidate_trace.get("full_exec_records", 0) != 0:
    failures.append(f"candidate should be Shared-only but produced top-k: {candidate_trace}")
if sorted(candidate_trace.get("shared_layers") or []) != expected_shared:
    failures.append(
        f"unexpected candidate Shared layers: got {candidate_trace.get('shared_layers')} expected {expected_shared}"
    )
if candidate_trace.get("shared_exec_records", 0) < len(expected_shared):
    failures.append(f"candidate missing Shared consumer trace: {candidate_trace}")
if candidate_trace.get("shared_exec_missing_input_top_k", 0) != 0:
    failures.append(f"candidate Shared consumer missed top-k input: {candidate_trace}")
if candidate_trace.get("shared_exec_with_input_top_k", 0) < candidate_trace.get("shared_exec_records", 0):
    failures.append(f"some candidate Shared records did not receive input top-k: {candidate_trace}")
if candidate_trace.get("consume_records", 0) < len(expected_shared):
    failures.append(f"candidate missing consume trace: {candidate_trace}")

if failures:
    for failure in failures:
        print(failure, file=sys.stderr)
    raise SystemExit(1)

print("Phase B tiny IndexShare parity passed")
print(f"  report: {report_path}")
print(f"  tokens: {tokens}")
print(f"  parity max_abs_diff: {parity.get('hidden_max_abs_diff')}")
print(f"  parity max_rel_diff: {parity.get('hidden_max_rel_diff')}")
print(f"  full layers: {trace.get('full_layers')}")
print(f"  shared layers: {trace.get('shared_layers')}")
print(f"  candidate shared layers: {candidate_trace.get('shared_layers')}")
print(f"  candidate shared exec with top-k: {candidate_trace.get('shared_exec_with_input_top_k')}")
print(f"  consume width: {trace.get('min_consume_width')}..{trace.get('max_consume_width')}")
print("  negative missing-top-k case: passed")
PY
