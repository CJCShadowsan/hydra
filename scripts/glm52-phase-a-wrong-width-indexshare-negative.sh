#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-a-wrong-width-indexshare-negative}"
EXPECTED_ERROR="GLM-DSA top-k sideband width does not match expected IndexShare width"

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-a-wrong-width-indexshare-negative.sh [options]

Proves the native GLM-5.2 runtime contract rejects a Shared GLM-DSA layer slice
when the top-k sideband payload is token-major i32 but has the wrong per-token
IndexShare width.

This is a local llama/skippy runtime-contract negative test. It does not start a
mesh, split deployment, scheduler, or two-machine lab run.

Options:
  --stage-model PATH      GLM-5.2 layer package path.
  --model-id ID           Model id recorded in reports.
  --skippy-bench PATH     skippy-bench binary.
  --out-dir PATH          Artifact directory.
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
    --skippy-bench)
      SKIPPY_BENCH_BIN="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
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

set +e
SKIPPY_BENCH_GLM_DSA_MALFORMED_TOP_K_BYTES=8 \
"$SKIPPY_BENCH_BIN" glm-dsa-layer-microbench \
  --stage-model "$STAGE_MODEL" \
  --model-id "$MODEL_ID" \
  --layer-start 3 \
  --layer-end 4 \
  --ctx-size 32 \
  --tokens 1 \
  --position-start 0 \
  --iterations 1 \
  --warmup 0 \
  --n-batch 1 \
  --n-ubatch 1 \
  --direct-sparse-attn false \
  --direct-sparse-prefill false \
  --op-timing false \
  --output "$OUT_DIR/report.json" \
  >"$OUT_DIR/stdout.txt" \
  2>"$OUT_DIR/stderr.txt"
rc=$?
set -e

if [[ "$rc" == "0" ]]; then
  echo "wrong-width IndexShare negative unexpectedly succeeded" >&2
  echo "stdout=$OUT_DIR/stdout.txt" >&2
  echo "stderr=$OUT_DIR/stderr.txt" >&2
  exit 1
fi

if ! grep -Fq "$EXPECTED_ERROR" "$OUT_DIR/stderr.txt"; then
  echo "wrong-width IndexShare negative failed for the wrong reason" >&2
  echo "expected: $EXPECTED_ERROR" >&2
  echo "stderr=$OUT_DIR/stderr.txt" >&2
  tail -80 "$OUT_DIR/stderr.txt" >&2 || true
  exit 1
fi

python3 - "$OUT_DIR/summary.json" "$OUT_DIR" "$rc" "$EXPECTED_ERROR" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
rc = int(sys.argv[3])
expected_error = sys.argv[4]
summary = {
    "passed": True,
    "phase": "A",
    "scope": "native GLM-5.2 runtime contract negative: Shared layer rejects wrong-width IndexShare/top-k sideband",
    "layer_start": 3,
    "layer_end": 4,
    "return_code": rc,
    "expected_error": expected_error,
    "malformed_top_k_bytes": 8,
    "actual_i32_per_token": 2,
    "stdout": str(out_dir / "stdout.txt"),
    "stderr": str(out_dir / "stderr.txt"),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
PY

echo "GLM-5.2 Phase-A wrong-width IndexShare negative passed"
echo "summary=$OUT_DIR/summary.json"
