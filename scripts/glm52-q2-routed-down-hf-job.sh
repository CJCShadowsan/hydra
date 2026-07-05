#!/usr/bin/env bash
set -euo pipefail

# HF Jobs entrypoint for the GLM-5.2 q2_K routed-down layer-package
# experiment. It expects the BF16 GGUF repo to be mounted read-only and a
# writable bucket/work volume for the package plus quant scratch.
#
# Cheap preflight example:
#
#   hf jobs run \
#     --namespace meshllm \
#     --flavor cpu-upgrade \
#     --timeout 2h \
#     --secrets HF_TOKEN \
#     --volume hf://models/meshllm/GLM-5.2-MTP-BF16-GGUF:/mnt/bf16:ro \
#     --volume hf://buckets/meshllm/glm52-q2-routed-down:/mnt/work \
#     --env DRY_RUN=1 \
#     --env TARGET_PACKAGE_REPO=meshllm/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-layers \
#     --detach \
#     -- \
#     rust:1.88-bookworm \
#     bash -lc 'curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/feat/jianyang-glm-52/scripts/glm52-q2-routed-down-hf-job.sh | bash'
#
# Real run: use the same command without DRY_RUN=1 and with a long timeout.

MESH_LLM_REPO="${MESH_LLM_REPO:-https://github.com/Mesh-LLM/mesh-llm.git}"
MESH_LLM_REF="${MESH_LLM_REF:-feat/jianyang-glm-52}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/bf16}"
SOURCE_PREFIX="${SOURCE_PREFIX:-BF16}"
WORK_ROOT="${WORK_ROOT:-/mnt/work}"
DRY_RUN="${DRY_RUN:-0}"

TARGET_PACKAGE_REPO="${TARGET_PACKAGE_REPO:-meshllm/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-layers}"
PACKAGE_SOURCE_REPO="${PACKAGE_SOURCE_REPO:-meshllm/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-GGUF}"
PACKAGE_MODEL_ID="${PACKAGE_MODEL_ID:-meshllm/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-GGUF:Q2_K-RoutedDown-MTP-Q8}"
PACKAGE_SOURCE_REVISION="${PACKAGE_SOURCE_REVISION:-local}"

BUILD_ROOT="${BUILD_ROOT:-/tmp/mesh-llm-build}"
BUILD_DIR="${BUILD_DIR:-$BUILD_ROOT/source}"
CARGO_HOME="${CARGO_HOME:-$BUILD_ROOT/cargo-home}"
RUSTUP_HOME="${RUSTUP_HOME:-$BUILD_ROOT/rustup-home}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD_ROOT/cargo-target}"
PACKAGE_DIR="${PACKAGE_DIR:-$WORK_ROOT/package/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-layers}"
TARGET_ROOT="${TARGET_ROOT:-$WORK_ROOT/quant-scratch}"
MANIFEST="${MANIFEST:-$WORK_ROOT/manifests/glm52-q2-routed-down-quant.json}"
JOB_STATUS_FILE="${JOB_STATUS_FILE:-$WORK_ROOT/status/glm52-q2-routed-down-status.json}"
RECORD_DIR="${RECORD_DIR:-$WORK_ROOT/records}"
SPOOL_DIR="${SPOOL_DIR:-$WORK_ROOT/spool}"
NTHREADS="${NTHREADS:-}"

export CARGO_HOME RUSTUP_HOME CARGO_TARGET_DIR

log_step() {
  echo
  echo "=== $* ==="
}

log_storage() {
  local label="$1"
  echo "Storage snapshot ($label):"
  df -h /tmp "$WORK_ROOT" "$SOURCE_ROOT" 2>/dev/null || true
  du -sh "$WORK_ROOT" 2>/dev/null || true
}

need_hf_token_for_real_run() {
  if [[ "$DRY_RUN" != "1" && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for the real run so the package can be uploaded." >&2
    exit 1
  fi
}

validate_mounts() {
  if [[ ! -d "$SOURCE_ROOT/$SOURCE_PREFIX" ]]; then
    echo "missing BF16 source prefix: $SOURCE_ROOT/$SOURCE_PREFIX" >&2
    echo "mount meshllm/GLM-5.2-MTP-BF16-GGUF at SOURCE_ROOT before running" >&2
    exit 1
  fi
  mkdir -p "$WORK_ROOT" "$PACKAGE_DIR" "$TARGET_ROOT" "$(dirname "$MANIFEST")" \
    "$(dirname "$JOB_STATUS_FILE")" "$RECORD_DIR" "$SPOOL_DIR" "$BUILD_ROOT"
}

install_system_deps() {
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      build-essential cmake git curl ca-certificates pkg-config libssl-dev \
      lld ninja-build python3 python3-pip python3-venv >/dev/null
    apt-get clean
    rm -rf /var/lib/apt/lists/*
  fi
}

checkout_mesh_llm() {
  rm -rf "$BUILD_DIR"
  git clone --filter=blob:none "$MESH_LLM_REPO" "$BUILD_DIR"
  cd "$BUILD_DIR"
  git fetch --depth 1 origin "$MESH_LLM_REF"
  git checkout --detach FETCH_HEAD
  git rev-parse HEAD
}

build_tools() {
  cd "$BUILD_DIR"
  scripts/prepare-llama.sh pinned
  LLAMA_STAGE_BACKEND=cpu LLAMA_STAGE_LINK_MODE=static scripts/build-llama.sh
  LLAMA_STAGE_BACKEND=cpu LLAMA_STAGE_LINK_MODE=static \
    cargo build --release --locked -p skippy-quantize -p skippy-model-package
}

run_package_workflow() {
  local launcher="$BUILD_DIR/scripts/glm52-q2-routed-down-layer-package.sh"
  local common_env=(
    "SKIPPY_QUANTIZE_BIN=$CARGO_TARGET_DIR/release/skippy-quantize"
    "SKIPPY_MODEL_PACKAGE_BIN=$CARGO_TARGET_DIR/release/skippy-model-package"
    "SOURCE_ROOT=$SOURCE_ROOT"
    "SOURCE_PREFIX=$SOURCE_PREFIX"
    "TARGET_ROOT=$TARGET_ROOT"
    "MANIFEST=$MANIFEST"
    "PACKAGE_DIR=$PACKAGE_DIR"
    "PACKAGE_MODEL_ID=$PACKAGE_MODEL_ID"
    "PACKAGE_SOURCE_REPO=$PACKAGE_SOURCE_REPO"
    "PACKAGE_SOURCE_REVISION=$PACKAGE_SOURCE_REVISION"
    "WORK_DIR=$WORK_ROOT/native-work"
    "SPOOL_DIR=$SPOOL_DIR"
    "RECORD_DIR=$RECORD_DIR"
    "JSON_EVENT_FILE=$JOB_STATUS_FILE"
  )
  if [[ -n "$NTHREADS" ]]; then
    common_env+=("NTHREADS=$NTHREADS")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    env "${common_env[@]}" DRY_RUN=1 "$launcher"
  else
    env "${common_env[@]}" "$launcher"
  fi
}

upload_package() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; package upload skipped."
    return 0
  fi
  python3 -m pip install -q --break-system-packages huggingface_hub
  TARGET_PACKAGE_REPO="$TARGET_PACKAGE_REPO" PACKAGE_DIR="$PACKAGE_DIR" python3 <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = os.environ["TARGET_PACKAGE_REPO"]
package_dir = Path(os.environ["PACKAGE_DIR"])
manifest = package_dir / "model-package.json"
if not manifest.is_file():
    raise SystemExit(f"missing package manifest: {manifest}")

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(package_dir),
    commit_message="Add GLM-5.2 q2_K routed-down layer package",
)
print(f"uploaded https://huggingface.co/{repo_id}")
PY
}

main() {
  echo "GLM-5.2 q2_K routed-down HF job"
  echo "  mesh ref:        $MESH_LLM_REF"
  echo "  source:          $SOURCE_ROOT/$SOURCE_PREFIX"
  echo "  work root:       $WORK_ROOT"
  echo "  package dir:     $PACKAGE_DIR"
  echo "  target package:  $TARGET_PACKAGE_REPO"
  echo "  dry run:         $DRY_RUN"
  need_hf_token_for_real_run
  validate_mounts
  log_storage "start"
  log_step "Installing system deps"
  install_system_deps
  log_step "Cloning mesh-llm"
  checkout_mesh_llm
  log_step "Building skippy tools"
  build_tools
  log_storage "after build"
  log_step "Running routed-down package workflow"
  run_package_workflow
  log_storage "after package workflow"
  log_step "Uploading package"
  upload_package
  log_step "Done"
}

main "$@"
