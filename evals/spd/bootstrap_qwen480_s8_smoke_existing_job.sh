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
ARTIFACT_REPO="${ARTIFACT_REPO:-$OUTPUT_REPO}"
ARTIFACT_REVISION="${ARTIFACT_REVISION:-main}"
ARTIFACT_RUN_PATH="${ARTIFACT_RUN_PATH:-runs/native-package-fresh}"
JOB_TIMEOUT="${JOB_TIMEOUT:-1.5h}"
HELDOUT_PROMPTS="${HELDOUT_PROMPTS:-8}"
SMOKE_STAGE_BACKEND_DEVICES="${SMOKE_STAGE_BACKEND_DEVICES:-CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3}"
export PATCH_REPO PATCH_REVISION PATCH_PATH_IN_REPO MESH_REF WORK_DIR BOOTSTRAP_DIR OUTPUT_REPO
export ARTIFACT_REPO ARTIFACT_REVISION ARTIFACT_RUN_PATH JOB_TIMEOUT HELDOUT_PROMPTS
export SMOKE_STAGE_BACKEND_DEVICES

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
  --train-prompts 32 \
  --heldout-prompts "$HELDOUT_PROMPTS" \
  --max-prompt-tokens 480 \
  --verify-steps 1 \
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
  --smoke-stage-backend-devices "$SMOKE_STAGE_BACKEND_DEVICES" \
  --out "$WORK_DIR/native-package-fresh-plan.json" \
  --json

python3 evals/spd/run_hf_spd_qualification_plan.py \
  --plan "$WORK_DIR/native-package-fresh-plan.json" \
  --groups setup,write_physical_stage_ms,build_prompts \
  --script-out "$WORK_DIR/native-package-fresh-setup-plan.sh"

python3 - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

work_dir = Path(os.environ["WORK_DIR"])
artifact_dir = work_dir / "artifact"
download_dir = work_dir / "existing-artifact"
run_path = os.environ["ARTIFACT_RUN_PATH"].strip("/")
snapshot_download(
    repo_id=os.environ["ARTIFACT_REPO"],
    repo_type="model",
    revision=os.environ.get("ARTIFACT_REVISION", "main"),
    allow_patterns=[run_path + "/*"],
    local_dir=download_dir,
)
source = download_dir / run_path
if not source.is_dir():
    raise SystemExit(f"artifact run path not found after download: {source}")
artifact_dir.mkdir(parents=True, exist_ok=True)
for item in source.iterdir():
    destination = artifact_dir / item.name
    if item.is_dir():
        shutil.copytree(item, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(item, destination)
required = [
    "skippy-spd-head.json",
    "spd-head.safetensors",
    "spd-serving-fixture.safetensors",
]
missing = [name for name in required if not (artifact_dir / name).is_file()]
if missing:
    raise SystemExit(f"hydrated artifact missing required files: {missing}")
print(f"hydrated {source} into {artifact_dir}")
PY

cd "$WORK_DIR/mesh-llm"
if [[ ! -x target/release/skippy-bench || ! -x target/release/skippy-server ]]; then
  echo "missing release skippy binaries under $PWD/target/release" >&2
  exit 1
fi
if [[ ! -f physical-stage-ms.txt ]]; then
  echo "missing physical-stage-ms.txt in $PWD" >&2
  exit 1
fi

python3 evals/spd/run_hf_spd_qualification_plan.py \
  --plan "$WORK_DIR/native-package-fresh-plan.json" \
  --groups package_smoke,latency_simulation,upload \
  --script-out "$WORK_DIR/native-package-fresh-smoke-plan.sh"
