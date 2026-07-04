#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-c-sparse-attn-closeout-gate}"
ITERATIONS="${ITERATIONS:-1}"
WARMUP="${WARMUP:-0}"
QUICK=0

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-c-sparse-attn-closeout-gate.sh [options]

Runs the Phase-C sparse attention closeout wrapper for native GLM-5.2
llama/skippy layer-slice execution.

The wrapper combines:
  - true dense fallback parity vs direct sparse decode
  - direct sparse decode rows with Full/Shared IndexShare sideband reuse
  - compact flash long-KV decode rows
  - high-position Shared-consumer compact flash with explicit 2048-wide
    IndexShare/top-k sideband
  - prefill policy rows for short direct sparse, guarded large direct sparse,
    and safe dense fallback

This is not a Skippy split, topology, MTP, mesh scheduling, or layer placement
gate.

Options:
  --stage-model PATH      GLM-5.2 layer package path.
  --model-id ID           Model id recorded in reports.
  --skippy-bench PATH     skippy-bench binary.
  --out-dir PATH          Artifact directory.
  --iterations N          Measured iterations per case. Default: 1
  --warmup N              Warmup iterations per case. Default: 0
  --quick                 Run reduced decode matrix; prefill still uses quick policy matrix.
  -h, --help              Show this help.
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
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
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

if [[ ! -x "$SKIPPY_BENCH_BIN" ]]; then
  echo "skippy-bench binary not executable: $SKIPPY_BENCH_BIN" >&2
  exit 1
fi
if [[ ! -d "$STAGE_MODEL" ]]; then
  echo "stage model package not found: $STAGE_MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

decode_args=()
if [[ "$QUICK" == "1" ]]; then
  decode_args+=(--quick)
else
  decode_args+=(--long-kv)
fi

"$ROOT/scripts/glm52-phase-c-direct-sparse-decode-gate.sh" \
  "${decode_args[@]}" \
  --stage-model "$STAGE_MODEL" \
  --model-id "$MODEL_ID" \
  --skippy-bench "$SKIPPY_BENCH_BIN" \
  --out-dir "$OUT_DIR/decode-compact" \
  --iterations "$ITERATIONS" \
  --warmup "$WARMUP" \
  >"$OUT_DIR/decode-compact.stdout.txt" \
  2>"$OUT_DIR/decode-compact.stderr.txt"

"$ROOT/scripts/glm52-phase-d-policy-gate.sh" \
  --quick \
  --stage-model "$STAGE_MODEL" \
  --model-id "$MODEL_ID" \
  --skippy-bench "$SKIPPY_BENCH_BIN" \
  --out-dir "$OUT_DIR/prefill-policy" \
  --iterations "$ITERATIONS" \
  --warmup "$WARMUP" \
  >"$OUT_DIR/prefill-policy.stdout.txt" \
  2>"$OUT_DIR/prefill-policy.stderr.txt"

python3 - "$OUT_DIR" "$QUICK" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
quick = sys.argv[2] == "1"
decode_summary_path = out_dir / "decode-compact" / "phase-c-direct-sparse-decode-summary.json"
prefill_summary_path = out_dir / "prefill-policy" / "phase-d-policy-summary.json"
failures = []

def load(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        failures.append(f"missing summary: {path}")
        return {}

decode = load(decode_summary_path)
prefill = load(prefill_summary_path)

if decode and not decode.get("passed"):
    failures.extend(f"decode/compact: {failure}" for failure in decode.get("failures", []))
if prefill and not prefill.get("passed"):
    failures.extend(f"prefill: {failure}" for failure in prefill.get("failures", []))

decode_rows = decode.get("rows") or []
prefill_rows = prefill.get("rows") or []

if not any(row.get("proof_kind") == "dense_parity" for row in decode_rows):
    failures.append("missing dense parity row")
if not any(row.get("proof_kind") == "direct_sparse" for row in decode_rows):
    failures.append("missing direct sparse decode row")
if not quick and not any(row.get("proof_kind") == "compact_flash" for row in decode_rows):
    failures.append("missing compact flash decode row")
if not quick and not any(row.get("proof_kind") == "compact_flash_synthetic_consumer" for row in decode_rows):
    failures.append("missing high-position Shared-consumer compact flash row")
if not any(row.get("label") == "prefill-short" for row in prefill_rows):
    failures.append("missing short prefill row")
if not any(row.get("label") == "prefill-long-direct-sparse" for row in prefill_rows):
    failures.append("missing guarded large direct sparse prefill row")
if not any(row.get("label") == "prefill-long-safe-fallback" for row in prefill_rows):
    failures.append("missing safe dense fallback prefill row")

summary = {
    "passed": not failures,
    "phase": "C",
    "scope": "native GLM-5.2 sparse attention closeout; no split topology, MTP, mesh scheduling, or layer placement",
    "quick": quick,
    "decode_compact_summary": str(decode_summary_path),
    "prefill_policy_summary": str(prefill_summary_path),
    "decode_compact_rows": decode_rows,
    "prefill_policy_rows": prefill_rows,
    "failures": failures,
}
summary_path = out_dir / "phase-c-sparse-attn-closeout-summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

if failures:
    print("GLM-5.2 Phase-C sparse attention closeout FAILED", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    print(f"summary={summary_path}", file=sys.stderr)
    raise SystemExit(1)

print("GLM-5.2 Phase-C sparse attention closeout passed")
print(f"summary={summary_path}")
print(f"decode_compact_rows={len(decode_rows)} prefill_policy_rows={len(prefill_rows)}")
PY
