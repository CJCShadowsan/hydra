#!/usr/bin/env bash
# Build a current-worktree Linux CUDA mesh-llm worker artifact on Hugging Face
# Jobs, upload it to an HF dataset repo, and write the env consumed by
# skippy-shard-hf-wan-proof.sh.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/skippy-shard-hf-build-artifact.sh

Builds the current dirty worktree on a short-lived Hugging Face GPU job and
uploads a worker tarball suitable for:

  scripts/skippy-shard-hf-wan-proof.sh <target-model-ref> <draft-gguf>

Environment:
  MESH_SHARD_HF_BUILD_ARTIFACT_REPO  HF dataset repo. Default: meshllm/mesh-llm-shard-proof-artifacts
  MESH_SHARD_HF_BUILD_NAMESPACE      HF namespace. Default: meshllm
  MESH_SHARD_HF_BUILD_FLAVOR         HF flavor. Default: t4-small
  MESH_SHARD_HF_BUILD_IMAGE          Docker image. Default: nvidia/cuda:12.4.1-devel-ubuntu22.04
  MESH_SHARD_HF_BUILD_TIMEOUT        HF job timeout. Default: 2h
  MESH_SHARD_HF_BUILD_CUDA_ARCH      llama.cpp CUDA arch list. Default: 75
  MESH_SHARD_HF_BUILD_TASK_ID        Stable task/artifact id. Default: shard-build-<utc>-<gitsha>
  MESH_SHARD_HF_BUILD_DIR            Scratch dir. Default: /tmp/mesh-shard-hf-build-<task-id>
  MESH_SHARD_HF_BUILD_LEDGER         Job ledger. Default: /tmp/mesh-llm-hf-jobs-<task-id>.jsonl
  MESH_SHARD_HF_BUILD_WAIT_SECS      Wait for uploaded artifact. Default: 7200
  MESH_SHARD_HF_BUILD_DRY_RUN        If 1, create source tar and print paths without launching.

Requires HF_TOKEN in the environment and the `hf` CLI authenticated for uploads.
Only the job created by this script is recorded in the scratch ledger.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

now_utc_compact() { date -u +%Y%m%dT%H%M%SZ; }
now_utc_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

git_short_sha() {
    git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null || printf 'nogit'
}

ARTIFACT_REPO="${MESH_SHARD_HF_BUILD_ARTIFACT_REPO:-meshllm/mesh-llm-shard-proof-artifacts}"
NAMESPACE="${MESH_SHARD_HF_BUILD_NAMESPACE:-meshllm}"
FLAVOR="${MESH_SHARD_HF_BUILD_FLAVOR:-t4-small}"
IMAGE="${MESH_SHARD_HF_BUILD_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
TIMEOUT="${MESH_SHARD_HF_BUILD_TIMEOUT:-2h}"
CUDA_ARCH="${MESH_SHARD_HF_BUILD_CUDA_ARCH:-75}"
TASK_ID="${MESH_SHARD_HF_BUILD_TASK_ID:-shard-build-$(now_utc_compact)-$(git_short_sha)}"
WORK_DIR="${MESH_SHARD_HF_BUILD_DIR:-/tmp/mesh-shard-hf-build-${TASK_ID}}"
LEDGER="${MESH_SHARD_HF_BUILD_LEDGER:-/tmp/mesh-llm-hf-jobs-${TASK_ID}.jsonl}"
WAIT_SECS="${MESH_SHARD_HF_BUILD_WAIT_SECS:-7200}"
DRY_RUN="${MESH_SHARD_HF_BUILD_DRY_RUN:-0}"

SOURCE_ROOT="mesh-llm-${TASK_ID}"
SOURCE_TAR="${WORK_DIR}/source/${SOURCE_ROOT}.tar.gz"
SOURCE_PATH="artifacts/${TASK_ID}/source/${SOURCE_ROOT}.tar.gz"
BUILD_NAME="mesh-llm-linux-cuda-sm${CUDA_ARCH}-${TASK_ID}.tar.gz"
BUILD_PATH="artifacts/${TASK_ID}/build/${BUILD_NAME}"
BUILD_SHA_PATH="${BUILD_PATH}.sha256"
META_PATH="artifacts/${TASK_ID}/build/metadata.json"
ENV_FILE="${WORK_DIR}/artifact.env"

require_command() {
    local command="$1"
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "missing required command: $command" >&2
        exit 2
    fi
}

sha256_file() {
    local file="$1"
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print tolower($1)}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print tolower($1)}'
    else
        echo "shasum or sha256sum is required" >&2
        exit 2
    fi
}

write_ledger_row() {
    local event="$1"
    local job_id="${2:-}"
    local status="${3:-}"
    python3 - "$LEDGER" "$event" "$job_id" "$status" "$NAMESPACE" "$FLAVOR" "$TASK_ID" "$ARTIFACT_REPO" "$SOURCE_PATH" "$BUILD_PATH" <<'PY'
import datetime
import json
import sys

ledger, event, job_id, status, namespace, flavor, task_id, repo, source_path, build_path = sys.argv[1:]
row = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "event": event,
    "job_id": job_id or None,
    "namespace": namespace,
    "flavor": flavor,
    "purpose": "shard-build-artifact",
    "task_id": task_id,
    "artifact_repo": repo,
    "source_path": source_path,
    "build_path": build_path,
    "cleanup_status": status or None,
}
with open(ledger, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

create_source_tar() {
    local stage="${WORK_DIR}/source-stage"
    local file_list="${WORK_DIR}/source-files.nul"
    rm -rf "$stage"
    mkdir -p "$stage/$SOURCE_ROOT" "$(dirname "$SOURCE_TAR")"

    (
        cd "$REPO_ROOT"
        git ls-files -co --exclude-standard -z >"$file_list"
        rsync -a --from0 --files-from="$file_list" ./ "$stage/$SOURCE_ROOT/"
    )

    tar -C "$stage" -czf "$SOURCE_TAR" "$SOURCE_ROOT"
}

upload_file() {
    local local_path="$1"
    local repo_path="$2"
    hf upload "$ARTIFACT_REPO" "$local_path" "$repo_path" \
        --repo-type dataset \
        --token "$HF_TOKEN" \
        >"${WORK_DIR}/$(basename "$repo_path").upload.log"
}

worker_command() {
    cat <<'EOF'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.cargo/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"

mkdir -p /work/download /work/src /work/out /work/cache/hf-home/hub /work/cache/xdg-cache
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends \
  ca-certificates curl git build-essential cmake ninja-build pkg-config \
  protobuf-compiler lld clang libssl-dev python3 python3-pip xz-utils rsync \
  >/dev/null

python3 -m pip install --no-cache-dir -q --upgrade 'huggingface_hub[cli]'
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable >/tmp/rustup.log
  . /root/.cargo/env
fi

hf download "$ARTIFACT_REPO" "$SOURCE_PATH" \
  --repo-type dataset \
  --local-dir /work/download \
  --token "$HF_TOKEN" >/tmp/hf-source-download.log
SOURCE_FILE="/work/download/$SOURCE_PATH"
printf '%s  %s\n' "$SOURCE_SHA" "$SOURCE_FILE" | sha256sum -c -
tar -xzf "$SOURCE_FILE" -C /work/src
cd "/work/src/$SOURCE_ROOT"

export CARGO_INCREMENTAL=0
export MESH_LLM_DYNAMIC_NATIVE_RUNTIME=0
export MESH_LLM_BUILD_PROFILE=release
scripts/build-linux.sh --skip-ui --backend cuda --cuda-arch "$CUDA_ARCH"

strip target/release/mesh-llm || true
tar -C target/release -czf "/work/out/$BUILD_NAME" mesh-llm
BUILD_SHA="$(sha256sum "/work/out/$BUILD_NAME" | awk '{print tolower($1)}')"
printf '%s  %s\n' "$BUILD_SHA" "$BUILD_NAME" >"/work/out/$BUILD_NAME.sha256"

{
  echo "{"
  echo "  \"task_id\": \"${TASK_ID}\","
  echo "  \"artifact_repo\": \"${ARTIFACT_REPO}\","
  echo "  \"source_path\": \"${SOURCE_PATH}\","
  echo "  \"source_sha256\": \"${SOURCE_SHA}\","
  echo "  \"artifact_path\": \"${BUILD_PATH}\","
  echo "  \"artifact_sha256\": \"${BUILD_SHA}\","
  echo "  \"cuda_arch\": \"${CUDA_ARCH}\","
  echo "  \"flavor\": \"${FLAVOR}\","
  echo "  \"image\": \"${IMAGE}\","
  echo "  \"built_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"mesh_llm_version\": \"$(./target/release/mesh-llm --version 2>/dev/null | sed 's/"/\\"/g')\""
  echo "}"
} > /work/out/metadata.json

hf upload "$ARTIFACT_REPO" "/work/out/$BUILD_NAME" "$BUILD_PATH" \
  --repo-type dataset --token "$HF_TOKEN" >/tmp/hf-build-upload.log
hf upload "$ARTIFACT_REPO" "/work/out/$BUILD_NAME.sha256" "$BUILD_SHA_PATH" \
  --repo-type dataset --token "$HF_TOKEN" >/tmp/hf-sha-upload.log
hf upload "$ARTIFACT_REPO" /work/out/metadata.json "$META_PATH" \
  --repo-type dataset --token "$HF_TOKEN" >/tmp/hf-metadata-upload.log

echo "MESH_SHARD_HF_ARTIFACT_REPO=$ARTIFACT_REPO"
echo "MESH_SHARD_HF_ARTIFACT_PATH=$BUILD_PATH"
echo "MESH_SHARD_HF_ARTIFACT_SHA=$BUILD_SHA"
EOF
}

launch_build_job() {
    local command output job_id
    command="$(worker_command)"
    output="$(
        hf jobs run \
            --namespace "$NAMESPACE" \
            --flavor "$FLAVOR" \
            --timeout "$TIMEOUT" \
            --secrets HF_TOKEN \
            --env MESH_LLM_CREATED_BY=codex \
            --env MESH_LLM_TASK_ID="$TASK_ID" \
            --env MESH_LLM_PURPOSE="shard-build-artifact" \
            --env PYTHONUNBUFFERED=1 \
            --env ARTIFACT_REPO="$ARTIFACT_REPO" \
            --env SOURCE_PATH="$SOURCE_PATH" \
            --env SOURCE_SHA="$SOURCE_SHA" \
            --env SOURCE_ROOT="$SOURCE_ROOT" \
            --env BUILD_NAME="$BUILD_NAME" \
            --env BUILD_PATH="$BUILD_PATH" \
            --env BUILD_SHA_PATH="$BUILD_SHA_PATH" \
            --env META_PATH="$META_PATH" \
            --env CUDA_ARCH="$CUDA_ARCH" \
            --env TASK_ID="$TASK_ID" \
            --env FLAVOR="$FLAVOR" \
            --env IMAGE="$IMAGE" \
            --detach \
            "$IMAGE" -- bash -lc "$command" 2>&1
    )"
    printf '%s\n' "$output" >"${WORK_DIR}/build-job-launch.txt"
    job_id="$(
        printf '%s\n' "$output" |
            python3 -c 'import re,sys; text=sys.stdin.read(); match=re.search(r"ID:\s*([A-Za-z0-9]+)", text); print(match.group(1) if match else "")'
    )"
    if [[ -z "$job_id" ]]; then
        echo "could not parse HF build job id" >&2
        printf '%s\n' "$output" >&2
        return 1
    fi
    printf '%s\n' "$job_id" >"${WORK_DIR}/build-job-id"
    write_ledger_row "build_job_started" "$job_id" "running"
    printf '%s\n' "$job_id"
}

download_artifact_env() {
    local download_dir="${WORK_DIR}/download"
    local sha_file="${download_dir}/${BUILD_SHA_PATH}"
    mkdir -p "$download_dir"
    hf download "$ARTIFACT_REPO" "$BUILD_SHA_PATH" \
        --repo-type dataset \
        --local-dir "$download_dir" \
        --token "$HF_TOKEN" >/dev/null
    [[ -s "$sha_file" ]] || return 1
    local build_sha
    build_sha="$(awk '{print tolower($1)}' "$sha_file")"
    [[ "$build_sha" =~ ^[0-9a-f]{64}$ ]] || return 1
    cat >"$ENV_FILE" <<EOF
export MESH_SHARD_HF_ARTIFACT_REPO=${ARTIFACT_REPO}
export MESH_SHARD_HF_ARTIFACT_PATH=${BUILD_PATH}
export MESH_SHARD_HF_ARTIFACT_SHA=${build_sha}
EOF
}

wait_for_artifact() {
    local job_id="$1"
    local elapsed=0
    while ((elapsed < WAIT_SECS)); do
        if download_artifact_env 2>"${WORK_DIR}/artifact-download.err"; then
            write_ledger_row "build_artifact_uploaded" "$job_id" "finished"
            return 0
        fi
        if ((elapsed % 60 == 0)); then
            hf jobs inspect "$job_id" --namespace "$NAMESPACE" \
                >"${WORK_DIR}/build-job-inspect.txt" 2>&1 || true
            if grep -Eqi 'failed|cancelled|canceled|error|timeout' "${WORK_DIR}/build-job-inspect.txt"; then
                echo "HF build job appears to have failed: $job_id" >&2
                sed -n '1,160p' "${WORK_DIR}/build-job-inspect.txt" >&2 || true
                write_ledger_row "build_job_failed" "$job_id" "failed"
                return 1
            fi
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "timed out waiting for build artifact after ${WAIT_SECS}s: $job_id" >&2
    write_ledger_row "build_artifact_wait_timeout" "$job_id" "unknown"
    return 124
}

require_command git
require_command rsync
require_command tar
require_command hf
require_command python3

if [[ "$DRY_RUN" != "1" && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required for source/artifact upload and the HF build job secret" >&2
    exit 2
fi

mkdir -p "$WORK_DIR" "$(dirname "$LEDGER")"

create_source_tar
SOURCE_SHA="$(sha256_file "$SOURCE_TAR")"

cat >"${WORK_DIR}/source-metadata.txt" <<EOF
task_id=${TASK_ID}
artifact_repo=${ARTIFACT_REPO}
source_path=${SOURCE_PATH}
source_sha256=${SOURCE_SHA}
build_path=${BUILD_PATH}
cuda_arch=${CUDA_ARCH}
flavor=${FLAVOR}
image=${IMAGE}
created_at=$(now_utc_iso)
EOF

echo "source_tar=${SOURCE_TAR}"
echo "source_sha=${SOURCE_SHA}"
echo "artifact_repo=${ARTIFACT_REPO}"
echo "artifact_path=${BUILD_PATH}"
echo "ledger=${LEDGER}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry run: not uploading source or launching HF job"
    exit 0
fi

upload_file "$SOURCE_TAR" "$SOURCE_PATH"
write_ledger_row "source_uploaded" "" "not_applicable"

JOB_ID="$(launch_build_job)"
echo "build_job_id=${JOB_ID}"
echo "inspect: hf jobs inspect ${JOB_ID} --namespace ${NAMESPACE}"

wait_for_artifact "$JOB_ID"

echo "artifact_env=${ENV_FILE}"
cat "$ENV_FILE"
