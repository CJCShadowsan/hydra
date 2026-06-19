#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

PATCH_REPO="${PATCH_REPO:-meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8}"
PATCH_REVISION="${PATCH_REVISION:-main}"
PATCH_PATH_IN_REPO="${PATCH_PATH_IN_REPO:?set PATCH_PATH_IN_REPO to the uploaded patch path}"
MESH_REF="${MESH_REF:-codex/skippy-spd-proof}"
WORK_DIR="${WORK_DIR:-/workspace/spd-qualification}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/workspace/spd-bootstrap}"
OUTPUT_REPO="${OUTPUT_REPO:-meshllm/skippy-spd-qwen3-coder-480b-a35b-ud-q4-k-xl-s8}"
JOB_TIMEOUT="${JOB_TIMEOUT:-4.5h}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-512}"
HELDOUT_PROMPTS="${HELDOUT_PROMPTS:-64}"
VERIFY_STEPS="${VERIFY_STEPS:-4}"
STREAM_LIVE_TAP_STAGES="${STREAM_LIVE_TAP_STAGES:-false}"
export PATCH_REPO PATCH_REVISION PATCH_PATH_IN_REPO MESH_REF WORK_DIR BOOTSTRAP_DIR OUTPUT_REPO JOB_TIMEOUT
export TRAIN_PROMPTS HELDOUT_PROMPTS VERIFY_STEPS STREAM_LIVE_TAP_STAGES

apt-get update
apt-get install -y --no-install-recommends ca-certificates git python3-pip
python3 -m pip install -U pip huggingface_hub

mkdir -p "$BOOTSTRAP_DIR" "$WORK_DIR"
git clone --depth 1 --branch "$MESH_REF" \
  https://github.com/Mesh-LLM/mesh-llm.git "$BOOTSTRAP_DIR/mesh-llm"

python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

patch_path = hf_hub_download(
    repo_id=os.environ["PATCH_REPO"],
    repo_type="model",
    revision=os.environ.get("PATCH_REVISION", "main"),
    filename=os.environ["PATCH_PATH_IN_REPO"],
)
target = Path(os.environ["BOOTSTRAP_DIR"]) / "mesh-llm.patch"
target.write_bytes(Path(patch_path).read_bytes())
print(target)
PY

cd "$BOOTSTRAP_DIR/mesh-llm"
export MESH_LLM_PATCH_PATH="$BOOTSTRAP_DIR/mesh-llm.patch"
git apply "$MESH_LLM_PATCH_PATH"
git status --short

EXTRA_PLANNER_ARGS=()
if [[ "$STREAM_LIVE_TAP_STAGES" == "true" ]]; then
  EXTRA_PLANNER_ARGS+=(--stream-live-tap-stages)
fi

python3 evals/spd/plan_hf_spd_qualification.py \
  --base-model Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --package-ref meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers \
  --qualification-mode native-package-fresh \
  --num-stages 8 \
  --stage-layer-boundaries 8,16,24,32,40,48,55,62 \
  --num-spec-layers 4 \
  --draft-top-k 4 \
  --draft-vocab-size 32000 \
  --vocab-size 151936 \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-prompts "$TRAIN_PROMPTS" \
  --heldout-prompts "$HELDOUT_PROMPTS" \
  --max-prompt-tokens 480 \
  --verify-steps "$VERIFY_STEPS" \
  --ctx-size 1024 \
  --physical-node-count 4 \
  --logical-stage-ms 40 \
  --hop-ms 0.2,1,5,10 \
  --flavor rtx-pro-6000x4 \
  --timeout "$JOB_TIMEOUT" \
  --max-cost-usd 50 \
  --mesh-llm-ref "$MESH_REF" \
  --output-repo "$OUTPUT_REPO" \
  --work-dir "$WORK_DIR" \
  --out "$WORK_DIR/native-package-fresh-plan.json" \
  --json \
  "${EXTRA_PLANNER_ARGS[@]}"

python3 evals/spd/run_hf_spd_qualification_plan.py \
  --plan "$WORK_DIR/native-package-fresh-plan.json" \
  --script-out "$WORK_DIR/native-package-fresh-plan.sh"
