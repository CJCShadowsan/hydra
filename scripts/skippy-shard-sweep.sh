#!/usr/bin/env bash
# Run a Shard proof command across draft-window K and pipelined-depth values,
# then collect each run's summary.json into one JSONL/JSON evidence table.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/skippy-shard-sweep.sh --kind mesh|hf -- <proof-runner-args...>

Examples:
  scripts/skippy-shard-sweep.sh --kind mesh -- \
    ./target/release/mesh-llm hf://meshllm/Qwen3-8B-Q4_K_M-layers /path/draft.gguf

  scripts/skippy-shard-sweep.sh --kind hf -- \
    hf://meshllm/skippy-shard-qwen25-3b-q4-k-m-layers-proof-20260621 /path/draft.gguf

Environment:
  MESH_SHARD_SWEEP_KS              Space-separated K values. Default: 2 4 6
  MESH_SHARD_SWEEP_DEPTHS          Space-separated pipelined depths. Default: 2 4 6
  MESH_SHARD_SWEEP_DELAYS_MS       Space-separated synthetic delay values, or "inherit".
                                   Default: inherit
  MESH_SHARD_SWEEP_OUT_DIR         Default: /tmp/mesh-shard-sweep.<pid>
  MESH_SHARD_SWEEP_BASE_MODES      Modes for each run. Default: target sync-draft pipelined-draft.
                                   sync-draft is the depth-1 baseline; pipelined-draft uses
                                   speculative.mode = "shard-pipeline" and requires depth > 1.
  MESH_SHARD_SWEEP_CONTINUE_ON_FAIL Default: 1
  MESH_SHARD_SWEEP_MIN_PIPE_VS_SYNC Default: inherited by proof runner

The wrapper sets the appropriate proof-runner environment for each kind:
  mesh: MESH_SHARD_PROOF_DRAFT_MAX_TOKENS, MESH_SHARD_PROOF_PIPELINED_DEPTH,
        MESH_SHARD_PROOF_WORK_DIR, MESH_SHARD_PROOF_MODES,
        MESH_SHARD_PROOF_MIN_PIPELINED_VS_SYNC_SPEEDUP,
        MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS when delays are swept
  hf:   MESH_SHARD_HF_DRAFT_MAX_TOKENS, MESH_SHARD_HF_PIPELINED_DEPTH,
        MESH_SHARD_HF_PROOF_DIR, MESH_SHARD_HF_MODES,
        MESH_SHARD_HF_TASK_ID, MESH_SHARD_HF_MIN_PIPELINED_VS_SYNC_SPEEDUP,
        MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS when delays are swept

Each run still owns its normal cleanup semantics. The aggregate files are:
  <out-dir>/sweep.jsonl
  <out-dir>/sweep.json
EOF
}

kind=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind)
            kind="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "unknown argument before --: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$kind" != "mesh" && "$kind" != "hf" ]]; then
    echo "--kind must be mesh or hf" >&2
    usage >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo "missing proof-runner arguments after --" >&2
    usage >&2
    exit 2
fi

KS="${MESH_SHARD_SWEEP_KS:-2 4 6}"
DEPTHS="${MESH_SHARD_SWEEP_DEPTHS:-2 4 6}"
DELAYS_MS="${MESH_SHARD_SWEEP_DELAYS_MS:-inherit}"
OUT_DIR="${MESH_SHARD_SWEEP_OUT_DIR:-$(mktemp -d "/tmp/mesh-shard-sweep.XXXXXX")}"
BASE_MODES="${MESH_SHARD_SWEEP_BASE_MODES:-target sync-draft pipelined-draft}"
CONTINUE_ON_FAIL="${MESH_SHARD_SWEEP_CONTINUE_ON_FAIL:-1}"
MIN_PIPE_VS_SYNC="${MESH_SHARD_SWEEP_MIN_PIPE_VS_SYNC:-}"
mkdir -p "$OUT_DIR"

runner_script="scripts/skippy-shard-${kind}-proof.sh"
if [[ "$kind" == "hf" ]]; then
    runner_script="scripts/skippy-shard-hf-wan-proof.sh"
elif [[ "$kind" == "mesh" ]]; then
    runner_script="scripts/skippy-shard-mesh-proof.sh"
fi
if [[ ! -x "$runner_script" ]]; then
    echo "proof runner is not executable: $runner_script" >&2
    exit 2
fi

jsonl="${OUT_DIR}/sweep.jsonl"
: >"$jsonl"

run_one() {
    local k="$1"
    local depth="$2"
    local delay_ms="$3"
    shift 3
    local run_id="k${k}-d${depth}"
    if [[ "$delay_ms" != "inherit" ]]; then
        run_id="${run_id}-lat${delay_ms}ms"
    fi
    local run_dir="${OUT_DIR}/${run_id}"
    local delay_env=()
    if [[ "$delay_ms" != "inherit" ]]; then
        delay_env=("MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS=${delay_ms}")
    fi
    mkdir -p "$run_dir"
    echo "=== Shard sweep ${run_id} (${kind}) ==="

    local status=0
    if [[ "$kind" == "hf" ]]; then
        if [[ -n "$MIN_PIPE_VS_SYNC" ]]; then
            env \
                "${delay_env[@]}" \
                MESH_SHARD_HF_DRAFT_MAX_TOKENS="$k" \
                MESH_SHARD_HF_PIPELINED_DEPTH="$depth" \
                MESH_SHARD_HF_PROOF_DIR="$run_dir" \
                MESH_SHARD_HF_TASK_ID="shard-sweep-${run_id}-$(date -u +%Y%m%dT%H%M%SZ)" \
                MESH_SHARD_HF_MODES="$BASE_MODES" \
                MESH_SHARD_HF_MIN_PIPELINED_VS_SYNC_SPEEDUP="$MIN_PIPE_VS_SYNC" \
                "$runner_script" "$@" || status=$?
        else
            env \
                "${delay_env[@]}" \
                MESH_SHARD_HF_DRAFT_MAX_TOKENS="$k" \
                MESH_SHARD_HF_PIPELINED_DEPTH="$depth" \
                MESH_SHARD_HF_PROOF_DIR="$run_dir" \
                MESH_SHARD_HF_TASK_ID="shard-sweep-${run_id}-$(date -u +%Y%m%dT%H%M%SZ)" \
                MESH_SHARD_HF_MODES="$BASE_MODES" \
                "$runner_script" "$@" || status=$?
        fi
    else
        if [[ -n "$MIN_PIPE_VS_SYNC" ]]; then
            env \
                "${delay_env[@]}" \
                MESH_SHARD_PROOF_DRAFT_MAX_TOKENS="$k" \
                MESH_SHARD_PROOF_PIPELINED_DEPTH="$depth" \
                MESH_SHARD_PROOF_WORK_DIR="$run_dir" \
                MESH_SHARD_PROOF_MODES="$BASE_MODES" \
                MESH_SHARD_PROOF_MIN_PIPELINED_VS_SYNC_SPEEDUP="$MIN_PIPE_VS_SYNC" \
                "$runner_script" "$@" || status=$?
        else
            env \
                "${delay_env[@]}" \
                MESH_SHARD_PROOF_DRAFT_MAX_TOKENS="$k" \
                MESH_SHARD_PROOF_PIPELINED_DEPTH="$depth" \
                MESH_SHARD_PROOF_WORK_DIR="$run_dir" \
                MESH_SHARD_PROOF_MODES="$BASE_MODES" \
                "$runner_script" "$@" || status=$?
        fi
    fi

    python3 - "$jsonl" "$kind" "$k" "$depth" "$delay_ms" "$status" "$run_dir" <<'PY'
import json
import pathlib
import sys

jsonl, kind, k, depth, delay_ms, status, run_dir = sys.argv[1:]
run_path = pathlib.Path(run_dir)
summary_path = run_path / "results" / "summary.json"
metadata_path = run_path / "results" / "metadata.json"
if not metadata_path.exists():
    metadata_path = run_path / "metadata.json"
payload = {
    "kind": kind,
    "k": int(k),
    "depth": int(depth),
    "delay_ms": None if delay_ms == "inherit" else int(delay_ms),
    "exit_status": int(status),
    "run_dir": str(run_path),
    "summary_path": str(summary_path),
    "metadata": json.loads(metadata_path.read_text()) if metadata_path.exists() else None,
    "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
}
with open(jsonl, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
PY

    if [[ "$status" -ne 0 && "$CONTINUE_ON_FAIL" != "1" ]]; then
        return "$status"
    fi
    return 0
}

for k in $KS; do
    for depth in $DEPTHS; do
        for delay_ms in $DELAYS_MS; do
            run_one "$k" "$depth" "$delay_ms" "$@"
        done
    done
done

python3 - "$jsonl" "${OUT_DIR}/sweep.json" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(rows, indent=2, sort_keys=True))
print(sys.argv[2])
PY

echo "sweep evidence: $OUT_DIR"
