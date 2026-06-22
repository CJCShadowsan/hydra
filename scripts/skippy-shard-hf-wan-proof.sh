#!/usr/bin/env bash
# Run a repeatable Shard-style WAN proof with a local coordinator and one
# Hugging Face GPU worker. The worker is short-lived, owned by this script, and
# cancelled after each mode.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/skippy-shard-hf-wan-proof.sh [target-model-ref] [draft-gguf]

Required environment:
  MESH_SHARD_HF_ARTIFACT_REPO   HF dataset/model repo containing a Linux mesh-llm tarball
  MESH_SHARD_HF_ARTIFACT_PATH   Path inside the repo to the tarball
  MESH_SHARD_HF_ARTIFACT_SHA    Expected sha256 of the tarball

Common environment:
  MESH_SHARD_HF_MESH_LLM        Local mesh-llm binary. Default: ./target/release/mesh-llm
  MESH_SHARD_HF_TARGET_MODEL    Target model ref if not passed as arg 1
  MESH_SHARD_HF_DRAFT_GGUF      Draft GGUF if not passed as arg 2
  MESH_SHARD_HF_MODES           Default: target sync-draft pipelined-draft
                                Add "tree" to validate Shard tree speculation.
  MESH_SHARD_HF_NAMESPACE       HF namespace. Default: meshllm
  MESH_SHARD_HF_WORKER_FLAVOR   HF worker flavor. Default: t4-small
  MESH_SHARD_HF_TASK_ID         Default: shard-hf-<utc timestamp>
  MESH_SHARD_HF_PROOF_DIR       Default: /tmp/mesh-shard-hf-<task-id>
  MESH_SHARD_HF_LEDGER          Default: /tmp/mesh-llm-hf-jobs-<task-id>.jsonl
  MESH_SHARD_HF_CACHE_ROOT      Default: /tmp/mesh-shard-hf-cache
  MESH_SHARD_HF_NATIVE_RUNTIME_CACHE Default: <cache-root>/native-runtime
  MESH_SHARD_HF_CTX_SIZE        Default: 512
  MESH_SHARD_HF_UBATCH          Default: 16
  MESH_SHARD_HF_MAX_TOKENS      Default: 24
  MESH_SHARD_HF_DRAFT_MAX_TOKENS Default: 4
  MESH_SHARD_HF_PIPELINED_DEPTH Default: 6
  MESH_SHARD_HF_MIN_ACCEPT_RATE Default: 0.05
  MESH_SHARD_HF_MIN_PIPELINED_SPEEDUP Default: 1.05
  MESH_SHARD_HF_MIN_PIPELINED_VS_SYNC_SPEEDUP Default: 1.00
  MESH_SHARD_HF_REQUIRE_SHARD_GATES Default: 1
  MESH_SHARD_HF_REQUIRE_REFERENCE Default: 0
  MESH_SHARD_HF_REQUIRE_ADVERSARIAL Default: 0
  MESH_SHARD_HF_REFERENCE_BASE_URL Optional OpenAI-compatible full-target endpoint
  MESH_SHARD_HF_REFERENCE_MODEL    Model id for the reference endpoint. Default: target model ref
  MESH_SHARD_HF_REFERENCE_API_KEY  Reference endpoint bearer token. Default: mesh
  MESH_SHARD_HF_REFERENCE_RESULTS_JSON Optional precomputed reference results JSON
  MESH_SHARD_HF_REFERENCE_TARGET_ID Canonical target id expected in reference metadata.
                                   Default: target model ref
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS Optional synthetic per-stage downstream delay
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS Optional synthetic per-stage downstream jitter
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS Optional synthetic downstream bandwidth cap
  SKIPPY_SPEC_DRAFT_FAULT_EVERY Optional validation hook: force every Nth draft token off-greedy
  SKIPPY_SPEC_DRAFT_FAULT_OFFSET Optional validation hook offset. Default: 0
  SKIPPY_SPEC_DRAFT_FAULT_RANK Optional validation hook alternative rank. Default: 2
  SKIPPY_SPEC_RETURN_DELAY_EVERY Optional validation hook: delay every Nth verify return
  SKIPPY_SPEC_RETURN_DELAY_OFFSET Optional validation hook offset. Default: 0
  SKIPPY_SPEC_RETURN_DELAY_MS Optional validation hook delay in milliseconds
  SKIPPY_SPEC_RETURN_RECONNECT_EVERY Optional validation hook: force direct-return writer reconnect after every N replies
  MESH_LLM_SPLIT_FORCE_BOUNDARIES Default: 18

The local coordinator uses the draft GGUF. The HF worker runs only target split
stages from the uploaded Linux artifact. HF_TOKEN and the mesh join token are
passed as HF secrets; raw tokens are not written to the repo.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

now_utc_compact() { date -u +%Y%m%dT%H%M%SZ; }
now_utc_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_LLM="${MESH_SHARD_HF_MESH_LLM:-./target/release/mesh-llm}"
TARGET_MODEL="${1:-${MESH_SHARD_HF_TARGET_MODEL:-}}"
DRAFT_GGUF="${2:-${MESH_SHARD_HF_DRAFT_GGUF:-}}"
MODES="${MESH_SHARD_HF_MODES:-target sync-draft pipelined-draft}"
NAMESPACE="${MESH_SHARD_HF_NAMESPACE:-meshllm}"
WORKER_FLAVOR="${MESH_SHARD_HF_WORKER_FLAVOR:-t4-small}"
WORKER_IMAGE="${MESH_SHARD_HF_WORKER_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
TASK_ID="${MESH_SHARD_HF_TASK_ID:-shard-hf-$(now_utc_compact)}"
PROOF_DIR="${MESH_SHARD_HF_PROOF_DIR:-/tmp/mesh-shard-hf-${TASK_ID}}"
LEDGER="${MESH_SHARD_HF_LEDGER:-/tmp/mesh-llm-hf-jobs-${TASK_ID}.jsonl}"
ARTIFACT_REPO="${MESH_SHARD_HF_ARTIFACT_REPO:-}"
ARTIFACT_PATH="${MESH_SHARD_HF_ARTIFACT_PATH:-}"
ARTIFACT_SHA="${MESH_SHARD_HF_ARTIFACT_SHA:-}"

CTX_SIZE="${MESH_SHARD_HF_CTX_SIZE:-512}"
UBATCH="${MESH_SHARD_HF_UBATCH:-16}"
MAX_TOKENS="${MESH_SHARD_HF_MAX_TOKENS:-24}"
DRAFT_MAX_TOKENS="${MESH_SHARD_HF_DRAFT_MAX_TOKENS:-4}"
PIPELINED_DEPTH="${MESH_SHARD_HF_PIPELINED_DEPTH:-6}"
MIN_ACCEPT_RATE="${MESH_SHARD_HF_MIN_ACCEPT_RATE:-0.05}"
MIN_PIPELINED_SPEEDUP="${MESH_SHARD_HF_MIN_PIPELINED_SPEEDUP:-1.05}"
MIN_PIPELINED_VS_SYNC_SPEEDUP="${MESH_SHARD_HF_MIN_PIPELINED_VS_SYNC_SPEEDUP:-1.00}"
REQUIRE_SHARD_GATES="${MESH_SHARD_HF_REQUIRE_SHARD_GATES:-1}"
REQUIRE_CANONICAL_REFERENCE="${MESH_SHARD_HF_REQUIRE_REFERENCE:-0}"
REQUIRE_ADVERSARIAL="${MESH_SHARD_HF_REQUIRE_ADVERSARIAL:-0}"
FORCED_BOUNDARIES="${MESH_LLM_SPLIT_FORCE_BOUNDARIES:-18}"
REFERENCE_BASE_URL="${MESH_SHARD_HF_REFERENCE_BASE_URL:-}"
REFERENCE_MODEL="${MESH_SHARD_HF_REFERENCE_MODEL:-$TARGET_MODEL}"
REFERENCE_RESULTS_JSON="${MESH_SHARD_HF_REFERENCE_RESULTS_JSON:-}"
REFERENCE_API_KEY="${MESH_SHARD_HF_REFERENCE_API_KEY:-mesh}"
REFERENCE_TARGET_ID="${MESH_SHARD_HF_REFERENCE_TARGET_ID:-$TARGET_MODEL}"
LOCAL_MAX_VRAM_GB="${MESH_SHARD_HF_LOCAL_MAX_VRAM_GB:-8}"
WORKER_MAX_VRAM_GB="${MESH_SHARD_HF_WORKER_MAX_VRAM_GB:-16}"
STAGE_LOAD_TIMEOUT="${MESH_SHARD_HF_STAGE_LOAD_TIMEOUT:-900}"
MAX_WAIT="${MESH_SHARD_HF_MAX_WAIT:-900}"
HF_LOG_TIMEOUT="${MESH_SHARD_HF_LOG_TIMEOUT:-45}"

CACHE_ROOT="${MESH_SHARD_HF_CACHE_ROOT:-/tmp/mesh-shard-hf-cache}"
HF_HOME_DIR="${MESH_SHARD_HF_HOME:-${CACHE_ROOT}/hf-home}"
HF_HUB_CACHE_DIR="${MESH_SHARD_HF_HUB_CACHE:-${HF_HOME_DIR}/hub}"
XDG_CACHE_HOME_DIR="${MESH_SHARD_HF_XDG_CACHE_HOME:-${CACHE_ROOT}/xdg-cache}"
NATIVE_RUNTIME_CACHE_DIR="${MESH_SHARD_HF_NATIVE_RUNTIME_CACHE:-${CACHE_ROOT}/native-runtime}"
PROMPTS_JSONL="${MESH_SHARD_HF_PROMPTS_JSONL:-${PROOF_DIR}/prompts.jsonl}"

SEED_API_PORT_BASE="${MESH_SHARD_HF_SEED_API_PORT:-9737}"
SEED_CONSOLE_PORT_BASE="${MESH_SHARD_HF_SEED_CONSOLE_PORT:-3531}"
SEED_BIND_PORT_BASE="${MESH_SHARD_HF_SEED_BIND_PORT:-55647}"
WORKER_API_PORT_BASE="${MESH_SHARD_HF_WORKER_API_PORT:-9847}"
WORKER_CONSOLE_PORT_BASE="${MESH_SHARD_HF_WORKER_CONSOLE:-3545}"
WORKER_BIND_PORT_BASE="${MESH_SHARD_HF_WORKER_BIND_PORT:-55648}"
MODE_PORT_STRIDE="${MESH_SHARD_HF_MODE_PORT_STRIDE:-50}"

require_value() {
    local name="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        echo "missing required value: $name" >&2
        usage >&2
        exit 2
    fi
}

require_value "target model" "$TARGET_MODEL"
require_value "draft GGUF" "$DRAFT_GGUF"
require_value "MESH_SHARD_HF_ARTIFACT_REPO" "$ARTIFACT_REPO"
require_value "MESH_SHARD_HF_ARTIFACT_PATH" "$ARTIFACT_PATH"
require_value "MESH_SHARD_HF_ARTIFACT_SHA" "$ARTIFACT_SHA"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "mesh-llm binary is not executable: $MESH_LLM" >&2
    exit 2
fi
if [[ ! -f "$DRAFT_GGUF" ]]; then
    echo "draft GGUF does not exist: $DRAFT_GGUF" >&2
    exit 2
fi
if [[ -n "$REFERENCE_RESULTS_JSON" && ! -f "$REFERENCE_RESULTS_JSON" ]]; then
    echo "reference results JSON does not exist: $REFERENCE_RESULTS_JSON" >&2
    exit 2
fi
if [[ "$REQUIRE_CANONICAL_REFERENCE" == "1" && -z "$REFERENCE_RESULTS_JSON" && -z "$REFERENCE_BASE_URL" ]]; then
    echo "MESH_SHARD_HF_REQUIRE_REFERENCE=1 requires MESH_SHARD_HF_REFERENCE_RESULTS_JSON or MESH_SHARD_HF_REFERENCE_BASE_URL" >&2
    exit 2
fi

mkdir -p \
    "$PROOF_DIR" \
    "$PROOF_DIR/configs" \
    "$PROOF_DIR/process" \
    "$PROOF_DIR/results" \
    "$PROOF_DIR/secrets" \
    "$HF_HOME_DIR" \
    "$HF_HUB_CACHE_DIR" \
    "$XDG_CACHE_HOME_DIR" \
    "$NATIVE_RUNTIME_CACHE_DIR"

if [[ ! -f "$PROMPTS_JSONL" ]]; then
    cat >"$PROMPTS_JSONL" <<'EOF'
{"id":"exact-1","prompt":"Return exactly: cache locality matters"}
{"id":"exact-2","prompt":"Return exactly: speculative decoding is deterministic"}
{"id":"exact-3","prompt":"Repeat exactly, with no extra words: direct return pipelines stale windows across wide area links"}
EOF
fi

redact_sensitive_log() {
    sed -E \
        -e 's/"token":"[^"]+"/"token":"<redacted>"/g' \
        -e 's/(MESH_JOIN_TOKEN=).*/\1<redacted>/g' \
        -e 's/(--join )[A-Za-z0-9._~+\/=:-]+/\1<redacted>/g'
}

run_with_watchdog() {
    local seconds="$1"
    local output_file="$2"
    shift 2
    "$@" >"$output_file" 2>&1 &
    local pid="$!"
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        if ((elapsed >= seconds)); then
            {
                echo
                echo "timed out after ${seconds}s: $*"
            } >>"$output_file"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            return 124
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    wait "$pid"
}

local_binary_has_static_skippy() {
    command -v nm >/dev/null 2>&1 || return 1
    nm "$MESH_LLM" 2>/dev/null | grep -Eq '(^|[[:space:]_])skippy_(abi_version|context_create|model_open)($|[[:space:]])'
}

prepare_local_native_runtime() {
    local install_json="${PROOF_DIR}/process/local-native-runtime-install.json"
    local install_err="${PROOF_DIR}/process/local-native-runtime-install.err"
    local list_json="${PROOF_DIR}/process/local-native-runtime-list.json"

    if local_binary_has_static_skippy; then
        echo "local native runtime: static Skippy symbols found in ${MESH_LLM}" >&2
        return 0
    fi

    echo "local native runtime: no static Skippy symbols found; installing into ${NATIVE_RUNTIME_CACHE_DIR}" >&2
    if "$MESH_LLM" runtime install \
        --cache-dir "$NATIVE_RUNTIME_CACHE_DIR" \
        --json >"$install_json" 2>"$install_err"; then
        "$MESH_LLM" runtime list \
            --cache-dir "$NATIVE_RUNTIME_CACHE_DIR" \
            --installed \
            --json >"$list_json" 2>/dev/null || true
        echo "local native runtime: installed runtime for dynamic local coordinator" >&2
        return 0
    fi

    echo "failed to prepare local Skippy native runtime before launching HF worker" >&2
    echo "binary: ${MESH_LLM}" >&2
    echo "install stderr:" >&2
    sed -n '1,80p' "$install_err" >&2 || true
    echo "install json:" >&2
    sed -n '1,120p' "$install_json" >&2 || true
    echo "build a static local coordinator with: MESH_LLM_DYNAMIC_NATIVE_RUNTIME=0 just release-build" >&2
    return 1
}

status_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/status" 2>/dev/null || true
}

runtime_stages_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/runtime/stages" 2>/dev/null || true
}

query_json_field() {
    local field="$1"
    python3 -c '
import json
import sys

field = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
print(data.get(field) or "")
' "$field"
}

peer_count() {
    python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
print(len(data.get("peers") or []))
'
}

node_id_matches() {
    local actual="${1:-}"
    local expected="${2:-}"
    [[ -n "$actual" && -n "$expected" ]] || return 1
    [[ "$actual" == "$expected" || "$actual" == "$expected"* || "$expected" == "$actual"* ]]
}

write_base_config() {
    local path="$1"
    cat >"$path" <<EOF
version = 1

[defaults.model_fit]
ctx_size = ${CTX_SIZE}
batch = 256
ubatch = ${UBATCH}
flash_attention = "disabled"
cache_type_k = "f16"
cache_type_v = "f16"

[defaults.hardware]
gpu_layers = -1

[defaults.skippy]
activation_wire_dtype = "f16"
prefill_chunking = "fixed"
prefill_chunk_size = 128

[defaults.request_defaults]
max_tokens = ${MAX_TOKENS}
temperature = 0.0
EOF
}

write_seed_config() {
    local mode="$1"
    local path="$2"
    write_base_config "$path"
    case "$mode" in
        target)
            ;;
        sync-draft)
            cat >>"$path" <<EOF

[defaults.speculative]
mode = "draft"
draft_model_path = "${DRAFT_GGUF}"
draft_selection_policy = "manual"
draft_max_tokens = ${DRAFT_MAX_TOKENS}
draft_gpu_layers = -1
pipelined_depth = 1
pairing_fault = "fail_closed"
EOF
            ;;
        pipelined-draft)
            cat >>"$path" <<EOF

[defaults.speculative]
mode = "shard-pipeline"
draft_model_path = "${DRAFT_GGUF}"
draft_selection_policy = "manual"
draft_max_tokens = ${DRAFT_MAX_TOKENS}
draft_gpu_layers = -1
pipelined_depth = ${PIPELINED_DEPTH}
pairing_fault = "fail_closed"
EOF
            ;;
        tree)
            cat >>"$path" <<EOF

[defaults.speculative]
mode = "tree"
draft_model_path = "${DRAFT_GGUF}"
draft_selection_policy = "manual"
draft_max_tokens = ${DRAFT_MAX_TOKENS}
draft_gpu_layers = -1
pairing_fault = "fail_closed"
EOF
            ;;
        *)
            echo "unsupported HF proof mode: $mode" >&2
            exit 2
            ;;
    esac
}

split_coordinator_from_log() {
    local log_file="$1"
    sed -n 's/.*Split runtime coordinator is \([0-9a-f]*\);.*/\1/p' "$log_file" 2>/dev/null | tail -1
}

forced_ready_model_id() {
    local seed_node_id="$1"
    local models_json="$2"
    local stages_json="$3"
    python3 - "$seed_node_id" "$FORCED_BOUNDARIES" "$models_json" "$stages_json" <<'PY'
import json
import sys

seed_node_id, forced_raw, models_json, stages_json = sys.argv[1:]
try:
    boundaries = [int(value.strip()) for value in forced_raw.split(",") if value.strip()]
except ValueError:
    boundaries = []
try:
    models = json.loads(models_json) if models_json else {}
except Exception:
    models = {}
try:
    data = json.loads(stages_json) if stages_json else {}
except Exception:
    data = {}

model_ids = [row.get("id") for row in (models.get("data") or []) if row.get("id")]
topologies = data.get("topologies") or []
statuses = data.get("stages") or data.get("statuses") or []

def node_id_matches(actual, expected):
    if not actual or not expected:
        return False
    actual = str(actual)
    expected = str(expected)
    return actual == expected or actual.startswith(expected) or expected.startswith(actual)

def expected_ranges(stages):
    if not boundaries:
        return len(stages) >= 2
    if len(stages) != len(boundaries) + 1:
        return False
    ordered = sorted(stages, key=lambda stage: stage.get("stage_index", 0))
    starts = [0, *boundaries]
    for index, stage in enumerate(ordered):
        if stage.get("layer_start") != starts[index]:
            return False
        if index < len(boundaries):
            if stage.get("layer_end") != boundaries[index]:
                return False
        elif not isinstance(stage.get("layer_end"), int) or stage.get("layer_end") <= starts[index]:
            return False
    return True

def downstream_ready(topology, stages):
    for downstream in sorted(stages, key=lambda stage: stage.get("stage_index", 0))[1:]:
        if not any(
            status.get("topology_id") == topology.get("topology_id")
            and status.get("run_id") == topology.get("run_id")
            and status.get("stage_id") == downstream.get("stage_id")
            and status.get("state") == "ready"
            and status.get("layer_start") == downstream.get("layer_start")
            and status.get("layer_end") == downstream.get("layer_end")
            for status in statuses
        ):
            return False
    return True

def base_model_id(model_id):
    # /v1/models advertises the routable stage-0 shard as "<model>:layer-000",
    # while the runtime topology records the bare "<model>" id. Compare on the
    # base id so split readiness is not missed because of the routing suffix.
    if not model_id:
        return model_id
    return str(model_id).rsplit(":layer-", 1)[0]

for model_id in model_ids:
    for topology in topologies:
        if base_model_id(topology.get("model_id")) != base_model_id(model_id):
            continue
        stages = sorted(topology.get("stages") or [], key=lambda stage: stage.get("stage_index", 0))
        if (
            stages
            and node_id_matches(stages[0].get("node_id"), seed_node_id)
            and expected_ranges(stages)
            and downstream_ready(topology, stages)
        ):
            print(model_id)
            raise SystemExit(0)
PY
}

write_topology_snapshot() {
    local mode="$1"
    local model_id="$2"
    local output_json="$3"
    local stages_json
    stages_json="$(runtime_stages_json "$SEED_CONSOLE_PORT")"
    python3 - "$mode" "$model_id" "$output_json" "$stages_json" <<'PY'
import json
import sys

mode, model_id, output_path, stages_json = sys.argv[1:]
try:
    data = json.loads(stages_json) if stages_json else {}
except Exception:
    data = {}
topologies = data.get("topologies") or []
statuses = data.get("stages") or data.get("statuses") or []
matching_topologies = [top for top in topologies if top.get("model_id") == model_id]
matching_statuses = [stage for stage in statuses if stage.get("model_id") == model_id]
selected_topologies = matching_topologies or topologies
selected_statuses = matching_statuses or statuses
nodes = sorted({
    stage.get("node_id")
    for topology in selected_topologies
    for stage in topology.get("stages") or []
    if stage.get("node_id")
} | {
    stage.get("node_id")
    for stage in selected_statuses
    if stage.get("node_id")
})
topology_stage_count = max((len(top.get("stages") or []) for top in selected_topologies), default=0)
payload = {
    "mode": mode,
    "model_id": model_id,
    "topology_stage_count": topology_stage_count,
    "runtime_stage_count": len(selected_statuses),
    "active_stage_count": max(topology_stage_count, len(selected_statuses)),
    "node_count": len(nodes),
    "node_ids": nodes,
    "topologies": selected_topologies,
    "stages": selected_statuses,
    "raw": data,
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
print(output_path)
PY
}

start_seed() {
    local mode="$1"
    local config="$2"
    local log_file="$3"
    local api_port="$4"
    local console_port="$5"
    local bind_port="$6"
    local home="${PROOF_DIR}/process/${mode}-seed/home"
    local runtime="${PROOF_DIR}/process/${mode}-seed/runtime"
    mkdir -p "$home" "$runtime"
    local -a args=(
        --log-format json
        --debug
        serve
        --model "$TARGET_MODEL"
        --split
        --config "$config"
        --port "$api_port"
        --console "$console_port"
        --bind-port "$bind_port"
        --headless
        --llama-flavor metal
        --max-vram "$LOCAL_MAX_VRAM_GB"
        --mesh-name "shard-wan-${mode}-${TASK_ID}"
        --name "${mode}-seed"
    )
    if [[ "$mode" == "target" ]]; then
        args+=(--no-draft)
    fi
    env \
        HOME="$home" \
        HF_HOME="$HF_HOME_DIR" \
        HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR" \
        HF_HUB_CACHE="$HF_HUB_CACHE_DIR" \
        XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR" \
        MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$NATIVE_RUNTIME_CACHE_DIR" \
        MESH_LLM_RUNTIME_ROOT="$runtime" \
        MESH_LLM_EPHEMERAL_KEY=1 \
        MESH_LLM_SPLIT_PREFERRED_STAGE0=local \
        MESH_LLM_SPLIT_MIN_PARTICIPANTS=2 \
        MESH_LLM_SPLIT_FORCE_BOUNDARIES="$FORCED_BOUNDARIES" \
        MESH_LLM_STAGE_LOAD_TIMEOUT_SECS="$STAGE_LOAD_TIMEOUT" \
        MESH_LLM_STAGE_TRANSPORT_DEBUG=1 \
        MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS:-}" \
        MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS:-}" \
        MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS:-}" \
        MESH_LLM_ALLOW_SLOW_DIRECT_STAGE_PATHS=1 \
        MESH_LLM_DYNAMIC_NATIVE_RUNTIME=0 \
        SKIPPY_SPEC_DRAFT_FAULT_EVERY="${SKIPPY_SPEC_DRAFT_FAULT_EVERY:-}" \
        SKIPPY_SPEC_DRAFT_FAULT_OFFSET="${SKIPPY_SPEC_DRAFT_FAULT_OFFSET:-}" \
        SKIPPY_SPEC_DRAFT_FAULT_RANK="${SKIPPY_SPEC_DRAFT_FAULT_RANK:-}" \
        SKIPPY_SPEC_RETURN_DELAY_EVERY="${SKIPPY_SPEC_RETURN_DELAY_EVERY:-}" \
        SKIPPY_SPEC_RETURN_DELAY_OFFSET="${SKIPPY_SPEC_RETURN_DELAY_OFFSET:-}" \
        SKIPPY_SPEC_RETURN_DELAY_MS="${SKIPPY_SPEC_RETURN_DELAY_MS:-}" \
        SKIPPY_SPEC_RETURN_RECONNECT_EVERY="${SKIPPY_SPEC_RETURN_RECONNECT_EVERY:-}" \
        SKIPPY_TELEMETRY_STDERR=1 \
        "$MESH_LLM" "${args[@]}" >"$log_file" 2>&1 &
    printf '%s\n' "$!"
}

wait_for_token() {
    local pid="$1"
    local console_port="$2"
    local log_file="$3"
    for _ in $(seq 1 "$MAX_WAIT"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "seed exited before invite token" >&2
            tail -160 "$log_file" | redact_sensitive_log >&2 || true
            return 1
        fi
        local token
        token="$(status_json "$console_port" | query_json_field token)"
        if [[ -n "$token" ]]; then
            printf '%s\n' "$token"
            return 0
        fi
        sleep 1
    done
    echo "timed out waiting for invite token" >&2
    tail -160 "$log_file" | redact_sensitive_log >&2 || true
    return 1
}

worker_command() {
    cat <<'EOF'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /work/bin /work/download /work/config /work/home /work/runtime /work/cache/hf-home/hub /work/cache/xdg-cache
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends ca-certificates curl jq python3 python3-pip xz-utils >/dev/null
python3 -m pip install --no-cache-dir -q --upgrade 'huggingface_hub[cli]'
hf download "$ARTIFACT_REPO" "$ARTIFACT_PATH" --repo-type dataset --local-dir /work/download --token "$HF_TOKEN" >/tmp/hf-download.log
ARTIFACT_FILE="/work/download/$ARTIFACT_PATH"
printf '%s  %s\n' "$ARTIFACT_SHA" "$ARTIFACT_FILE" | sha256sum -c -
tar -xzf "$ARTIFACT_FILE" -C /work/bin
chmod +x /work/bin/mesh-llm
cat > /work/config/worker.toml <<EOF_CONF
version = 1

[defaults.model_fit]
ctx_size = ${CTX_SIZE}
batch = 256
ubatch = ${UBATCH}
flash_attention = "disabled"
cache_type_k = "f16"
cache_type_v = "f16"

[defaults.hardware]
gpu_layers = -1

[defaults.skippy]
activation_wire_dtype = "f16"
prefill_chunking = "fixed"
prefill_chunk_size = 128

[defaults.request_defaults]
max_tokens = ${MAX_TOKENS}
temperature = 0.0
EOF_CONF
/work/bin/mesh-llm --version
exec env HOME=/work/home \
  HF_HOME=/work/cache/hf-home \
  HUGGINGFACE_HUB_CACHE=/work/cache/hf-home/hub \
  HF_HUB_CACHE=/work/cache/hf-home/hub \
  XDG_CACHE_HOME=/work/cache/xdg-cache \
  MESH_LLM_RUNTIME_ROOT=/work/runtime \
  MESH_LLM_EPHEMERAL_KEY=1 \
  MESH_LLM_SPLIT_PREFERRED_STAGE0="$SEED_NODE_ID" \
  MESH_LLM_SPLIT_MIN_PARTICIPANTS=2 \
  MESH_LLM_SPLIT_FORCE_BOUNDARIES="$FORCED_BOUNDARIES" \
  MESH_LLM_STAGE_LOAD_TIMEOUT_SECS=900 \
  MESH_LLM_STAGE_TRANSPORT_DEBUG=1 \
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS:-}" \
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS:-}" \
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS:-}" \
  MESH_LLM_ALLOW_SLOW_DIRECT_STAGE_PATHS=1 \
  SKIPPY_SPEC_DRAFT_FAULT_EVERY="${SKIPPY_SPEC_DRAFT_FAULT_EVERY:-}" \
  SKIPPY_SPEC_DRAFT_FAULT_OFFSET="${SKIPPY_SPEC_DRAFT_FAULT_OFFSET:-}" \
  SKIPPY_SPEC_DRAFT_FAULT_RANK="${SKIPPY_SPEC_DRAFT_FAULT_RANK:-}" \
  SKIPPY_SPEC_RETURN_DELAY_EVERY="${SKIPPY_SPEC_RETURN_DELAY_EVERY:-}" \
  SKIPPY_SPEC_RETURN_DELAY_OFFSET="${SKIPPY_SPEC_RETURN_DELAY_OFFSET:-}" \
  SKIPPY_SPEC_RETURN_DELAY_MS="${SKIPPY_SPEC_RETURN_DELAY_MS:-}" \
  SKIPPY_SPEC_RETURN_RECONNECT_EVERY="${SKIPPY_SPEC_RETURN_RECONNECT_EVERY:-}" \
  SKIPPY_TELEMETRY_STDERR=1 \
  /work/bin/mesh-llm --log-format json --debug serve \
    --model "$TARGET_MODEL" --split --config /work/config/worker.toml \
    --port "$WORKER_PORT" --console "$WORKER_CONSOLE" --bind-port "$WORKER_BIND_PORT" \
    --listen-all --headless --llama-flavor cuda --max-vram "$WORKER_MAX_VRAM_GB" \
    --mesh-name "$MESH_NAME" --name "$WORKER_NAME" --join "$MESH_JOIN_TOKEN" --no-draft
EOF
}

launch_worker() {
    local mode="$1"
    local token="$2"
    local mesh_name="$3"
    local worker_port="$4"
    local worker_console="$5"
    local worker_bind="$6"
    local seed_node_id="$7"
    local secrets_file="${PROOF_DIR}/secrets/${mode}-worker.env"
    printf 'MESH_JOIN_TOKEN=%s\n' "$token" >"$secrets_file"
    chmod 600 "$secrets_file"
    local command output job_id
    command="$(worker_command)"
    output="$(
        hf jobs run \
            --namespace "$NAMESPACE" \
            --flavor "$WORKER_FLAVOR" \
            --timeout 1h \
            --secrets HF_TOKEN \
            --secrets-file "$secrets_file" \
            --env MESH_LLM_CREATED_BY=codex \
            --env MESH_LLM_TASK_ID="$TASK_ID" \
            --env MESH_LLM_PURPOSE="shard-wan-${mode}" \
            --env PYTHONUNBUFFERED=1 \
            --env ARTIFACT_REPO="$ARTIFACT_REPO" \
            --env ARTIFACT_PATH="$ARTIFACT_PATH" \
            --env ARTIFACT_SHA="$ARTIFACT_SHA" \
            --env TARGET_MODEL="$TARGET_MODEL" \
            --env MESH_NAME="$mesh_name" \
            --env SEED_NODE_ID="$seed_node_id" \
            --env FORCED_BOUNDARIES="$FORCED_BOUNDARIES" \
            --env CTX_SIZE="$CTX_SIZE" \
            --env UBATCH="$UBATCH" \
            --env MAX_TOKENS="$MAX_TOKENS" \
            --env WORKER_NAME="${mode}-hf-worker" \
            --env WORKER_PORT="$worker_port" \
            --env WORKER_CONSOLE="$worker_console" \
            --env WORKER_BIND_PORT="$worker_bind" \
            --env WORKER_MAX_VRAM_GB="$WORKER_MAX_VRAM_GB" \
            --env MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS:-}" \
            --env MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS:-}" \
            --env MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS="${MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS:-}" \
            --env SKIPPY_SPEC_DRAFT_FAULT_EVERY="${SKIPPY_SPEC_DRAFT_FAULT_EVERY:-}" \
            --env SKIPPY_SPEC_DRAFT_FAULT_OFFSET="${SKIPPY_SPEC_DRAFT_FAULT_OFFSET:-}" \
            --env SKIPPY_SPEC_DRAFT_FAULT_RANK="${SKIPPY_SPEC_DRAFT_FAULT_RANK:-}" \
            --env SKIPPY_SPEC_RETURN_DELAY_EVERY="${SKIPPY_SPEC_RETURN_DELAY_EVERY:-}" \
            --env SKIPPY_SPEC_RETURN_DELAY_OFFSET="${SKIPPY_SPEC_RETURN_DELAY_OFFSET:-}" \
            --env SKIPPY_SPEC_RETURN_DELAY_MS="${SKIPPY_SPEC_RETURN_DELAY_MS:-}" \
            --env SKIPPY_SPEC_RETURN_RECONNECT_EVERY="${SKIPPY_SPEC_RETURN_RECONNECT_EVERY:-}" \
            --detach \
            "$WORKER_IMAGE" -- bash -lc "$command" 2>&1
    )"
    printf '%s\n' "$output" >"${PROOF_DIR}/process/${mode}-hf-worker-launch.txt"
    job_id="$(
        printf '%s\n' "$output" |
            python3 -c 'import re,sys; text=sys.stdin.read(); match=re.search(r"ID:\s*([A-Za-z0-9]+)", text); print(match.group(1) if match else "")'
    )"
    if [[ -z "$job_id" ]]; then
        echo "could not parse HF job id for $mode" >&2
        printf '%s\n' "$output" | redact_sensitive_log >&2
        return 1
    fi
    printf '%s\n' "$job_id" >"${PROOF_DIR}/process/${mode}-hf-worker-job-id"
    python3 - "$LEDGER" "$job_id" "$mode" "$PROOF_DIR" "$ARTIFACT_SHA" "$NAMESPACE" "$WORKER_FLAVOR" "$TASK_ID" <<'PY'
import datetime
import json
import sys

ledger, job_id, mode, proof_dir, artifact_sha, namespace, flavor, task_id = sys.argv[1:]
row = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "job_id": job_id,
    "namespace": namespace,
    "flavor": flavor,
    "purpose": f"shard-wan-{mode}",
    "task_id": task_id,
    "proof_dir": proof_dir,
    "artifact_sha256": artifact_sha,
    "cleanup_status": "running",
}
with open(ledger, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
    printf '%s\n' "$job_id"
}

cancel_worker() {
    local job_id="${1:-}"
    local mode="${2:-unknown}"
    local reason="${3:-cleanup}"
    [[ -n "$job_id" ]] || return 0
    hf jobs cancel "$job_id" --namespace "$NAMESPACE" >/dev/null 2>&1 || true
    python3 - "$LEDGER" "$job_id" "$mode" "$reason" "$NAMESPACE" "$TASK_ID" <<'PY'
import datetime
import json
import sys

ledger, job_id, mode, reason, namespace, task_id = sys.argv[1:]
row = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "job_id": job_id,
    "namespace": namespace,
    "event": "worker_cancelled",
    "purpose": f"shard-wan-{mode}",
    "task_id": task_id,
    "cleanup_status": "cancelled",
    "reason": reason,
}
with open(ledger, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

kill_seed() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
}

wait_for_model() {
    local mode="$1"
    local seed_pid="$2"
    local seed_node_id="$3"
    local seed_log="$4"
    local job_id="$5"
    for i in $(seq 1 "$MAX_WAIT"); do
        if ! kill -0 "$seed_pid" 2>/dev/null; then
            echo "seed exited unexpectedly for $mode" >&2
            tail -180 "$seed_log" | redact_sensitive_log >&2 || true
            return 1
        fi
        local coordinator_id
        coordinator_id="$(split_coordinator_from_log "$seed_log")"
        if [[ -n "$coordinator_id" ]] && ! node_id_matches "$coordinator_id" "$seed_node_id"; then
            echo "seed lost split coordinator election: seed=$seed_node_id coordinator=$coordinator_id" >&2
            return 1
        fi
        local peers
        peers="$(status_json "$SEED_CONSOLE_PORT" | peer_count)"
        if [[ "$peers" -ge 1 ]]; then
            local models_json stages_json model_id
            models_json="$(curl -fsS --max-time 5 "http://127.0.0.1:${SEED_API_PORT}/v1/models" 2>/dev/null || true)"
            stages_json="$(runtime_stages_json "$SEED_CONSOLE_PORT")"
            model_id="$(forced_ready_model_id "$seed_node_id" "$models_json" "$stages_json")"
            if [[ -n "$model_id" ]]; then
                echo "mode ${mode} ready after ${i}s: model=${model_id} forced=${FORCED_BOUNDARIES} active_stages=2" >&2
                printf '%s\n' "$model_id"
                return 0
            fi
        fi
        if ((i % 30 == 0)); then
            echo "waiting ${mode}: ${i}s peers=${peers} job=${job_id}" >&2
            run_with_watchdog \
                "$HF_LOG_TIMEOUT" \
                "${PROOF_DIR}/process/${mode}-hf-inspect-${i}.json" \
                hf jobs inspect "$job_id" --namespace "$NAMESPACE" || true
        fi
        sleep 1
    done
    echo "timed out waiting for $mode split model" >&2
    run_with_watchdog "$HF_LOG_TIMEOUT" "${PROOF_DIR}/process/${mode}-hf-timeout-inspect.json" \
        hf jobs inspect "$job_id" --namespace "$NAMESPACE" || true
    run_with_watchdog "$HF_LOG_TIMEOUT" "${PROOF_DIR}/process/${mode}-hf-timeout-log-tail.txt" \
        hf jobs logs "$job_id" --namespace "$NAMESPACE" --tail 160 || true
    tail -180 "$seed_log" | redact_sensitive_log >&2 || true
    return 1
}

run_requests() {
    local mode="$1"
    local model_id="$2"
    local output_json="$3"
    local seed_log="$4"
    python3 - "$mode" "$model_id" "$PROMPTS_JSONL" "$output_json" "$SEED_API_PORT" "$seed_log" "$MAX_TOKENS" <<'PY'
import json
import os
import sys
import time
import urllib.request

mode, model_id, prompts_path, output_path, port, seed_log_path, max_tokens = sys.argv[1:]
with open(prompts_path, "r", encoding="utf-8") as handle:
    prompts = [json.loads(line) for line in handle if line.strip()]

def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def telemetry_since(path, offset):
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            raw = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    events = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        if not text.startswith("{"):
            starts = [pos for pos in (line.find('{"attributes"'), line.find('{"event"')) if pos >= 0]
            if not starts:
                continue
            text = line[min(starts):]
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events

def spec_metrics(event):
    attrs = event.get("attributes") or {}
    prefixes = (
        "llama_stage.spec.",
        "llama_stage.decode_",
        "llama_stage.prompt_",
        "llama_stage.completion_",
        "skippy.",
    )
    return {key: value for key, value in attrs.items() if any(key.startswith(prefix) for prefix in prefixes)}

def verify_window_metrics(event):
    attrs = event.get("attributes") or {}
    keys = (
        "llama_stage.decode_step",
        "llama_stage.spec.proposed",
        "llama_stage.spec.verify_inputs",
        "llama_stage.spec.pipelined_window_index",
        "llama_stage.spec.message_decode_step",
        "llama_stage.spec.pipelined_fifo_order_ok",
        "llama_stage.spec.accepted",
        "llama_stage.spec.rejected",
        "llama_stage.stage0_compute_ms",
        "llama_stage.forward_write_ms",
        "llama_stage.downstream_wait_ms",
    )
    return {key: attrs[key] for key in keys if key in attrs}

def completion_token_ids(events):
    token_ids = []
    for event in events:
        attrs = event.get("attributes") or {}
        token = attrs.get("llama_stage.predicted_token")
        if isinstance(token, int):
            token_ids.append(token)
    return token_ids

results = []
for prompt in prompts:
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Answer deterministically and briefly."},
            {"role": "user", "content": prompt["prompt"]},
        ],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer mesh"},
        method="POST",
    )
    offset = log_size(seed_log_path)
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - started
    time.sleep(0.2)
    telemetry = telemetry_since(seed_log_path, offset)
    decodes = [event for event in telemetry if event.get("event") == "stage.openai_decode"]
    windows = [event for event in telemetry if event.get("event") == "stage.openai_decode_verify_window"]
    token_events = [event for event in telemetry if event.get("event") == "stage.openai_decode_token"]
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    results.append({
        "mode": mode,
        "prompt_id": prompt["id"],
        "prompt": prompt["prompt"],
        "content": payload["choices"][0]["message"]["content"],
        "completion_token_ids": completion_token_ids(token_events),
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_s": completion_tokens / elapsed if completion_tokens else None,
        "usage": usage,
        "spec_metrics": spec_metrics(decodes[-1]) if decodes else None,
        "spec_verify_windows": [verify_window_metrics(event) for event in windows],
        "telemetry_decode_event_count": len(decodes),
        "telemetry_verify_window_count": len(windows),
        "telemetry_decode_token_event_count": len(token_events),
    })
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump({"mode": mode, "model_id": model_id, "results": results}, handle, indent=2, sort_keys=True)
print(output_path)
PY
}

run_reference_requests() {
    local output_json="$1"
    MESH_SHARD_REFERENCE_API_KEY="$REFERENCE_API_KEY" python3 - \
        "$REFERENCE_BASE_URL" \
        "$REFERENCE_MODEL" \
        "$PROMPTS_JSONL" \
        "$output_json" \
        "$TARGET_MODEL" \
        "$REFERENCE_TARGET_ID" \
        "$CTX_SIZE" \
        "$UBATCH" \
        "$MAX_TOKENS" <<'PY'
import json
import os
import sys
import time
import urllib.request

(
    base_url,
    model_id,
    prompts_path,
    output_path,
    target_model,
    reference_target_id,
    ctx_size,
    ubatch,
    max_tokens,
) = sys.argv[1:]
api_key = os.environ.get("MESH_SHARD_REFERENCE_API_KEY", "mesh")
with open(prompts_path, "r", encoding="utf-8") as handle:
    prompts = [json.loads(line) for line in handle if line.strip()]

base = base_url.rstrip("/")
if base.endswith("/chat/completions"):
    url = base
elif base.endswith("/v1"):
    url = f"{base}/chat/completions"
else:
    url = f"{base}/v1/chat/completions"

results = []
for prompt in prompts:
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Answer deterministically and briefly."},
            {"role": "user", "content": prompt["prompt"]},
        ],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - started
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    results.append({
        "mode": "reference",
        "prompt_id": prompt["id"],
        "prompt": prompt["prompt"],
        "content": payload["choices"][0]["message"]["content"],
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_s": completion_tokens / elapsed if completion_tokens else None,
        "usage": usage,
    })
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "mode": "reference",
            "model_id": model_id,
            "base_url": base_url,
            "target_identity": {
                "target_id": reference_target_id,
                "requested_target_model": target_model,
                "served_model_id": model_id,
            },
            "request_defaults": {
                "ctx_size": int(ctx_size),
                "ubatch": int(ubatch),
                "max_tokens": int(max_tokens),
                "temperature": 0.0,
                "system_prompt": "Answer deterministically and briefly.",
                "stream": False,
            },
            "prompts": [{"id": prompt["id"], "prompt": prompt["prompt"]} for prompt in prompts],
            "results": results,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
print(output_path)
PY
}

prepare_reference_results() {
    local output_json="${PROOF_DIR}/results/reference.json"
    if [[ -n "$REFERENCE_RESULTS_JSON" ]]; then
        cp "$REFERENCE_RESULTS_JSON" "$output_json"
        echo "reference: copied ${REFERENCE_RESULTS_JSON} -> ${output_json}"
    elif [[ -n "$REFERENCE_BASE_URL" ]]; then
        run_reference_requests "$output_json"
    else
        return 0
    fi
    local -a validate_args=(
        "$SCRIPT_DIR/skippy-shard-reference-validate.py"
        --reference "$output_json"
        --prompts "$PROMPTS_JSONL"
        --target-id "$REFERENCE_TARGET_ID"
        --max-tokens "$MAX_TOKENS"
    )
    if [[ "$REQUIRE_CANONICAL_REFERENCE" == "1" ]]; then
        validate_args+=(--require-metadata)
    fi
    python3 "${validate_args[@]}"
}

summarize_results() {
    python3 - \
        "$PROOF_DIR" \
        "$PROOF_DIR/results" \
        "$DRAFT_MAX_TOKENS" \
        "$PIPELINED_DEPTH" \
        "$MIN_ACCEPT_RATE" \
        "$MIN_PIPELINED_SPEEDUP" \
        "$MIN_PIPELINED_VS_SYNC_SPEEDUP" \
        "$REQUIRE_SHARD_GATES" \
        "$REQUIRE_CANONICAL_REFERENCE" \
        "$REQUIRE_ADVERSARIAL" \
        $MODES <<'PY'
import json
import os
import pathlib
import re
import sys

proof_dir = pathlib.Path(sys.argv[1])
result_dir = pathlib.Path(sys.argv[2])
draft_max_tokens = int(sys.argv[3])
pipelined_depth = int(sys.argv[4])
min_accept_rate = float(sys.argv[5])
min_pipelined_speedup = float(sys.argv[6])
min_pipelined_vs_sync_speedup = float(sys.argv[7])
require_shard_gates = sys.argv[8] == "1"
require_canonical_reference = sys.argv[9] == "1"
require_adversarial = sys.argv[10] == "1"
modes = sys.argv[11:]
baseline = json.loads((result_dir / "target.json").read_text())
baseline_rows = baseline["results"]
baseline_by_prompt = {row["prompt_id"]: row for row in baseline_rows}
baseline_elapsed = sum(row["elapsed_s"] for row in baseline_rows)
baseline_tokens = sum(row.get("completion_tokens") or 0 for row in baseline_rows)
baseline_tps = baseline_tokens / baseline_elapsed if baseline_elapsed and baseline_tokens else None
def mode_perf(mode):
    path = result_dir / f"{mode}.json"
    if not path.exists():
        return (None, None, None)
    rows = json.loads(path.read_text()).get("results") or []
    elapsed = sum(row["elapsed_s"] for row in rows)
    tokens = sum(row.get("completion_tokens") or 0 for row in rows)
    tps = tokens / elapsed if elapsed and tokens else None
    return (elapsed, tokens, tps)
sync_elapsed, sync_tokens, sync_tps = mode_perf("sync-draft")
reference_path = result_dir / "reference.json"
reference_payload = json.loads(reference_path.read_text()) if reference_path.exists() else None
reference_rows = (reference_payload or {}).get("results") or []
reference_by_prompt = {row["prompt_id"]: row for row in reference_rows if row.get("prompt_id")}

def positive_env_int(name):
    try:
        return int(os.environ.get(name, "0") or "0") > 0
    except ValueError:
        return False

return_delay_requested = (
    positive_env_int("SKIPPY_SPEC_RETURN_DELAY_EVERY")
    and positive_env_int("SKIPPY_SPEC_RETURN_DELAY_MS")
)
return_reconnect_requested = positive_env_int("SKIPPY_SPEC_RETURN_RECONNECT_EVERY")

metric_keys = [
    "llama_stage.spec.windows",
    "llama_stage.spec.proposed",
    "llama_stage.spec.accepted",
    "llama_stage.spec.rejected",
    "llama_stage.spec.full_accept_windows",
    "llama_stage.spec.rejected_windows",
    "llama_stage.spec.early_reject_windows",
    "llama_stage.spec.tail_reject_windows",
    "llama_stage.spec.primary_verify_requests",
    "llama_stage.spec.primary_verify_tokens",
    "llama_stage.spec.primary_verify_elapsed_ms",
    "llama_stage.spec.primary_verify_stage0_compute_ms",
    "llama_stage.spec.primary_verify_forward_write_ms",
    "llama_stage.spec.primary_verify_downstream_wait_ms",
    "llama_stage.spec.draft_propose_ms",
    "llama_stage.spec.draft_reset_ms",
    "llama_stage.spec.recovery_ms",
    "llama_stage.spec.recovery_restore_local_ms",
    "llama_stage.spec.recovery_restore_downstream_write_ms",
    "llama_stage.spec.recovery_restore_downstream_wait_ms",
    "llama_stage.spec.pipelined_sent_windows",
    "llama_stage.spec.pipelined_committed_windows",
    "llama_stage.spec.pipelined_stale_windows",
    "llama_stage.spec.pipelined_async_draft_windows",
    "llama_stage.spec.pipelined_stale_draft_windows",
    "llama_stage.spec.pipelined_async_draft_wait_ms",
    "llama_stage.spec.pipelined_fifo_return_windows",
    "llama_stage.spec.pipelined_fifo_return_violations",
    "llama_stage.spec.pipelined_identity_violations",
    "llama_stage.spec.tree_windows",
    "llama_stage.spec.tree_nodes",
    "llama_stage.spec.tree_gather_ms",
    "skippy.verify_span_session_auto_align_count",
    "skippy.verify_span_session_auto_align_ms",
    "skippy.verify_span_session_auto_align_trimmed_tokens",
]
max_metric_keys = [
    "llama_stage.spec.pipelined_max_inflight_windows",
]

def number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except Exception:
        return None

def file_contains(path, needle):
    try:
        return needle in path.read_text(errors="ignore")
    except OSError:
        return False

def file_text(path):
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""

def observed_direct_rtts(path):
    text = file_text(path)
    values = [int(value) for value in re.findall(r"RTT: ([0-9]+)ms \(direct\)", text)]
    values.extend(int(value) for value in re.findall(r"rtt_ms=Some\(([0-9]+)\)", text))
    return values

def aggregate(rows):
    totals = {key: 0 for key in metric_keys}
    maxima = {key: None for key in max_metric_keys}
    seen = set()
    enabled = False
    decode_events = 0
    verify_events = 0
    committed_proposed = 0
    committed_accepted = 0
    for row in rows:
        metrics = row.get("spec_metrics") or {}
        enabled = enabled or metrics.get("llama_stage.spec.enabled") is True
        decode_events += row.get("telemetry_decode_event_count") or 0
        verify_windows = row.get("spec_verify_windows") or []
        verify_events += row.get("telemetry_verify_window_count") or len(verify_windows)
        for window in verify_windows:
            proposed = number(window.get("llama_stage.spec.proposed")) or 0
            accepted = number(window.get("llama_stage.spec.accepted")) or 0
            committed_proposed += proposed
            committed_accepted += accepted
        for key in metric_keys:
            value = number(metrics.get(key))
            if value is not None:
                totals[key] += value
                seen.add(key)
        for key in max_metric_keys:
            value = number(metrics.get(key))
            if value is not None:
                maxima[key] = value if maxima[key] is None else max(maxima[key], value)
    out = {key: totals[key] for key in metric_keys if key in seen}
    out.update({key: value for key, value in maxima.items() if value is not None})
    proposed = out.get("llama_stage.spec.proposed", 0)
    accepted = out.get("llama_stage.spec.accepted", 0)
    out["llama_stage.spec.accept_rate"] = accepted / proposed if proposed else None
    out["llama_stage.spec.committed_proposed"] = committed_proposed
    out["llama_stage.spec.committed_accepted"] = committed_accepted
    out["llama_stage.spec.committed_accept_rate"] = (
        committed_accepted / committed_proposed if committed_proposed else None
    )
    out["llama_stage.spec.enabled"] = enabled
    out["telemetry_decode_event_count"] = decode_events
    out["telemetry_verify_window_count"] = verify_events
    verify_elapsed = out.get("llama_stage.spec.primary_verify_elapsed_ms")
    downstream_wait = out.get("llama_stage.spec.primary_verify_downstream_wait_ms")
    stage0_compute = out.get("llama_stage.spec.primary_verify_stage0_compute_ms")
    out["llama_stage.spec.primary_verify_downstream_wait_share"] = (
        downstream_wait / verify_elapsed if downstream_wait is not None and verify_elapsed else None
    )
    out["llama_stage.spec.primary_verify_downstream_wait_vs_stage0_compute"] = (
        downstream_wait / stage0_compute if downstream_wait is not None and stage0_compute else None
    )
    return out

def verify_chunk_shape(rows):
    checked = 0
    max_proposed = 0
    violations = []
    for row in rows:
        for window in row.get("spec_verify_windows") or []:
            proposed = window.get("llama_stage.spec.proposed")
            verify_inputs = window.get("llama_stage.spec.verify_inputs")
            if proposed is None and verify_inputs is None:
                continue
            checked += 1
            if not isinstance(proposed, int) or proposed <= 0:
                violations.append({"prompt_id": row.get("prompt_id"), "reason": "invalid_proposed", "window": window})
                continue
            max_proposed = max(max_proposed, proposed)
            if proposed > draft_max_tokens:
                violations.append({"prompt_id": row.get("prompt_id"), "reason": "proposed_gt_k", "window": window})
            if verify_inputs != proposed + 1:
                violations.append({"prompt_id": row.get("prompt_id"), "reason": "verify_inputs_not_proposed_plus_one", "window": window})
    return {
        "checked_windows": checked,
        "max_proposed": max_proposed,
        "expected_k": draft_max_tokens,
        "observed_full_k_window": max_proposed == draft_max_tokens,
        "violations": violations,
        "ok": checked > 0 and not violations and max_proposed == draft_max_tokens,
    }

def first_content_mismatch(rows):
    for row in rows:
        prompt_id = row.get("prompt_id")
        expected = baseline_by_prompt.get(prompt_id, {}).get("content")
        actual = row.get("content")
        if expected == actual:
            continue
        if not isinstance(expected, str) or not isinstance(actual, str):
            return {
                "prompt_id": prompt_id,
                "expected_type": type(expected).__name__,
                "actual_type": type(actual).__name__,
            }
        index = next(
            (
                idx
                for idx, (left, right) in enumerate(zip(expected, actual))
                if left != right
            ),
            min(len(expected), len(actual)),
        )
        start = max(0, index - 80)
        end = index + 160
        return {
            "prompt_id": prompt_id,
            "first_diff_char": index,
            "expected_len": len(expected),
            "actual_len": len(actual),
            "expected_excerpt": expected[start:end],
            "actual_excerpt": actual[start:end],
        }
    return None

def token_ids_complete(row):
    token_ids = row.get("completion_token_ids")
    completion_tokens = row.get("completion_tokens")
    if not isinstance(token_ids, list) or not isinstance(completion_tokens, int):
        return False
    # stage.openai_decode_token includes the terminal EOS token when the native
    # backend emits it; OpenAI usage.completion_tokens excludes that EOS.
    return len(token_ids) in {completion_tokens, completion_tokens + 1}

def first_token_mismatch(rows, expected_by_prompt):
    for row in rows:
        prompt_id = row.get("prompt_id")
        expected = expected_by_prompt.get(prompt_id, {}).get("completion_token_ids")
        actual = row.get("completion_token_ids")
        if expected == actual:
            continue
        return {
            "prompt_id": prompt_id,
            "expected_len": len(expected) if isinstance(expected, list) else None,
            "actual_len": len(actual) if isinstance(actual, list) else None,
            "expected_prefix": expected[:32] if isinstance(expected, list) else None,
            "actual_prefix": actual[:32] if isinstance(actual, list) else None,
        }
    return None

summary = []
failed = False
gate_failures = []
for mode in modes:
    path = result_dir / f"{mode}.json"
    if not path.exists():
        continue
    payload = json.loads(path.read_text())
    rows = payload["results"]
    matches = [
        row["content"] == baseline_by_prompt.get(row["prompt_id"], {}).get("content")
        for row in rows
    ]
    token_matches = [
        row.get("completion_token_ids")
        == baseline_by_prompt.get(row["prompt_id"], {}).get("completion_token_ids")
        for row in rows
    ]
    reference_matches = None
    reference_token_matches = None
    if reference_by_prompt:
        reference_matches = [
            row["content"] == reference_by_prompt.get(row["prompt_id"], {}).get("content")
            for row in rows
        ]
        if any("completion_token_ids" in row for row in reference_rows):
            reference_token_matches = [
                row.get("completion_token_ids")
                == reference_by_prompt.get(row["prompt_id"], {}).get("completion_token_ids")
                for row in rows
            ]
    if mode != "target" and (not all(matches) or (require_shard_gates and not all(token_matches))):
        failed = True
    elapsed = sum(row["elapsed_s"] for row in rows)
    tokens = sum(row.get("completion_tokens") or 0 for row in rows)
    tps = tokens / elapsed if elapsed and tokens else None
    topology_path = result_dir / f"{mode}-topology.json"
    topology = json.loads(topology_path.read_text()) if topology_path.exists() else {}
    ready_log = proof_dir / "process" / f"{mode}-hf-ready-log-tail.txt"
    final_log = proof_dir / "process" / f"{mode}-hf-final-log-tail.txt"
    seed_log = proof_dir / f"{mode}-seed.log"
    direct_stage_path_observed = file_contains(seed_log, "path_kind=Direct")
    direct_prediction_return_observed = (
        file_contains(ready_log, "direct prediction return using upstream-opened sink")
        or file_contains(final_log, "direct prediction return using upstream-opened sink")
    )
    direct_return_delay_observed = (
        file_contains(ready_log, "skippy direct return validation delay:")
        or file_contains(final_log, "skippy direct return validation delay:")
    )
    direct_return_reconnect_observed = (
        file_contains(ready_log, "direct prediction return writer reconnected:")
        or file_contains(final_log, "direct prediction return writer reconnected:")
    )
    rtts = observed_direct_rtts(seed_log)
    spec_summary = aggregate(rows)
    chunk_shape = (
        verify_chunk_shape(rows)
        if mode in {"sync-draft", "pipelined-draft"}
        else None
    )
    first_mismatch = first_content_mismatch(rows)
    first_token_id_mismatch = first_token_mismatch(rows, baseline_by_prompt)
    token_ids_observed = all(token_ids_complete(row) for row in rows)
    proof_gates = {
        "output_matches_target": all(matches),
    }
    if require_shard_gates:
        proof_gates["completion_token_ids_observed"] = token_ids_observed
        proof_gates["completion_token_ids_match_target"] = all(token_matches)
    if require_canonical_reference:
        proof_gates["canonical_reference_checked"] = reference_matches is not None
    if reference_matches is not None:
        proof_gates["matches_canonical_reference"] = all(reference_matches)
    if reference_token_matches is not None:
        proof_gates["completion_token_ids_match_canonical_reference"] = all(
            reference_token_matches
        )

    if require_shard_gates:
        proof_gates["direct_stage_path_observed"] = direct_stage_path_observed
        if mode != "target":
            proof_gates["direct_prediction_return_observed"] = direct_prediction_return_observed
            if return_delay_requested:
                proof_gates["direct_return_delay_observed"] = direct_return_delay_observed
            if return_reconnect_requested:
                proof_gates["direct_return_reconnect_observed"] = direct_return_reconnect_observed

    if require_shard_gates and mode in {"sync-draft", "pipelined-draft"}:
        accept_rate = spec_summary.get("llama_stage.spec.accept_rate")
        committed_accept_rate = (
            spec_summary.get("llama_stage.spec.committed_accept_rate") or accept_rate
        )
        proposed = spec_summary.get("llama_stage.spec.proposed") or 0
        accepted = spec_summary.get("llama_stage.spec.accepted") or 0
        committed_proposed = spec_summary.get("llama_stage.spec.committed_proposed") or 0
        committed_accepted = spec_summary.get("llama_stage.spec.committed_accepted") or 0
        proof_gates["speculation_enabled"] = spec_summary.get("llama_stage.spec.enabled") is True
        proof_gates["speculation_engaged"] = (
            (committed_proposed or proposed) > 0
            and (committed_accepted or accepted) > 0
            and committed_accept_rate is not None
            and committed_accept_rate >= min_accept_rate
        )
        proof_gates["verify_chunk_shape_ok"] = bool(chunk_shape and chunk_shape.get("ok"))
        rejected_windows = spec_summary.get("llama_stage.spec.rejected_windows") or 0
        stale_windows = spec_summary.get("llama_stage.spec.pipelined_stale_windows") or 0
        recovery_ms = spec_summary.get("llama_stage.spec.recovery_ms") or 0
        draft_reset_ms = spec_summary.get("llama_stage.spec.draft_reset_ms") or 0
        auto_align_count = spec_summary.get("skippy.verify_span_session_auto_align_count") or 0
        auto_align_trimmed = (
            spec_summary.get("skippy.verify_span_session_auto_align_trimmed_tokens") or 0
        )
        proof_gates["post_reject_draft_recovery_observed"] = (
            rejected_windows > 0 and draft_reset_ms > 0
            if require_adversarial
            else rejected_windows == 0 or draft_reset_ms > 0
        )
        if mode == "pipelined-draft":
            sent_windows = spec_summary.get("llama_stage.spec.pipelined_sent_windows") or 0
            committed_windows = spec_summary.get("llama_stage.spec.pipelined_committed_windows") or 0
            max_inflight_windows = spec_summary.get("llama_stage.spec.pipelined_max_inflight_windows") or 0
            fifo_windows = spec_summary.get("llama_stage.spec.pipelined_fifo_return_windows") or 0
            fifo_violations = (
                spec_summary.get("llama_stage.spec.pipelined_fifo_return_violations") or 0
            )
            identity_violations = spec_summary.get(
                "llama_stage.spec.pipelined_identity_violations"
            )
            proof_gates["pipelined_depth_engaged"] = (
                pipelined_depth > 1
                and sent_windows > 0
                and fifo_windows > 0
                and max_inflight_windows > 1
            )
            proof_gates["pipelined_fifo_return_accounted"] = (
                sent_windows > 0
                and fifo_windows == sent_windows
                and fifo_violations == 0
            )
            proof_gates["pipelined_identity_accounted"] = identity_violations is not None
            proof_gates["pipelined_identity_match"] = identity_violations == 0
            proof_gates["pipelined_commit_stale_accounted"] = (
                sent_windows > 0
                and committed_windows + stale_windows == sent_windows
            )
            proof_gates["pipelined_stale_kv_recovery_observed"] = (
                stale_windows > 0
                and (recovery_ms > 0 or auto_align_count > 0 or auto_align_trimmed > 0)
                if require_adversarial
                else (
                    stale_windows == 0
                    or recovery_ms > 0
                    or auto_align_count > 0
                    or auto_align_trimmed > 0
                )
            )
            proof_gates["pipelined_speedup_ok"] = (
                tps is not None
                and baseline_tps is not None
                and (tps / baseline_tps) >= min_pipelined_speedup
            )
            proof_gates["pipelined_speedup_vs_sync_ok"] = (
                tps is not None
                and sync_tps is not None
                and (tps / sync_tps) >= min_pipelined_vs_sync_speedup
            )
    elif require_shard_gates and mode == "tree":
        tree_windows = spec_summary.get("llama_stage.spec.tree_windows") or 0
        tree_nodes = spec_summary.get("llama_stage.spec.tree_nodes") or 0
        proof_gates["tree_speculation_enabled"] = (
            spec_summary.get("llama_stage.spec.enabled") is True
        )
        proof_gates["tree_speculation_engaged"] = tree_windows > 0 and tree_nodes > tree_windows
    for gate, ok in proof_gates.items():
        if not ok:
            gate_failures.append({"mode": mode, "gate": gate})
    summary.append({
        "mode": mode,
        "request_count": len(rows),
        "content_matches_target": all(matches),
        "completion_token_ids_observed": token_ids_observed,
        "completion_token_ids_match_target": all(token_matches),
        "canonical_reference_checked": reference_matches is not None,
        "content_matches_canonical_reference": (
            all(reference_matches) if reference_matches is not None else None
        ),
        "completion_token_ids_match_canonical_reference": (
            all(reference_token_matches) if reference_token_matches is not None else None
        ),
        "elapsed_s_total": elapsed,
        "completion_tokens_total": tokens,
        "tokens_per_s": tps,
        "tokens_per_s_ratio_vs_target": tps / baseline_tps if tps and baseline_tps else None,
        "tokens_per_s_ratio_vs_sync_draft": tps / sync_tps if tps and sync_tps else None,
        "elapsed_ratio_vs_target": elapsed / baseline_elapsed if baseline_elapsed else None,
        "elapsed_ratio_vs_sync_draft": elapsed / sync_elapsed if sync_elapsed else None,
        "active_stage_count": topology.get("active_stage_count"),
        "topology_stage_count": topology.get("topology_stage_count"),
        "runtime_stage_count": topology.get("runtime_stage_count"),
        "topology_node_count": topology.get("node_count"),
        "direct_stage_path_observed": direct_stage_path_observed,
        "direct_prediction_return_observed": direct_prediction_return_observed,
        "direct_return_delay_requested": return_delay_requested,
        "direct_return_delay_observed": direct_return_delay_observed,
        "direct_return_reconnect_requested": return_reconnect_requested,
        "direct_return_reconnect_observed": direct_return_reconnect_observed,
        "first_content_mismatch": first_mismatch,
        "first_token_id_mismatch": first_token_id_mismatch,
        "observed_direct_rtt_ms": {
            "count": len(rtts),
            "min": min(rtts) if rtts else None,
            "max": max(rtts) if rtts else None,
            "last": rtts[-1] if rtts else None,
        },
        "verify_chunk_shape": chunk_shape,
        "proof_gates": proof_gates,
        "spec_summary": spec_summary,
    })

(result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
if failed or gate_failures:
    if gate_failures:
        print("proof gate failures: " + json.dumps(gate_failures, sort_keys=True), file=sys.stderr)
    raise SystemExit("Shard WAN proof gates failed")
PY
}

CURRENT_SEED_PID=""
CURRENT_JOB_ID=""
CURRENT_MODE=""

cleanup_current() {
    if [[ -n "${CURRENT_JOB_ID:-}" ]]; then
        cancel_worker "$CURRENT_JOB_ID" "$CURRENT_MODE" "trap_cleanup"
    fi
    if [[ -n "${CURRENT_SEED_PID:-}" ]]; then
        kill_seed "$CURRENT_SEED_PID"
    fi
}
trap cleanup_current EXIT

cat >"${PROOF_DIR}/metadata.json" <<EOF
{
  "artifact_path": "${ARTIFACT_PATH}",
  "artifact_repo": "${ARTIFACT_REPO}",
  "artifact_sha256": "${ARTIFACT_SHA}",
  "ctx_size": ${CTX_SIZE},
  "draft_gguf": "${DRAFT_GGUF}",
  "draft_max_tokens": ${DRAFT_MAX_TOKENS},
  "forced_boundaries": "${FORCED_BOUNDARIES}",
  "local_native_runtime_cache": "${NATIVE_RUNTIME_CACHE_DIR}",
  "local_mesh_llm_version": "$("$MESH_LLM" --version 2>/dev/null || true)",
  "max_tokens": ${MAX_TOKENS},
  "min_accept_rate": ${MIN_ACCEPT_RATE},
  "min_pipelined_speedup": ${MIN_PIPELINED_SPEEDUP},
  "min_pipelined_vs_sync_speedup": ${MIN_PIPELINED_VS_SYNC_SPEEDUP},
  "modes": "$(printf '%s' "$MODES")",
  "namespace": "${NAMESPACE}",
  "pipelined_depth": ${PIPELINED_DEPTH},
  "require_adversarial": $([[ "$REQUIRE_ADVERSARIAL" == "1" ]] && echo true || echo false),
  "require_canonical_reference": $([[ "$REQUIRE_CANONICAL_REFERENCE" == "1" ]] && echo true || echo false),
  "require_shard_gates": $([[ "$REQUIRE_SHARD_GATES" == "1" ]] && echo true || echo false),
  "reference_base_url": "${REFERENCE_BASE_URL}",
  "reference_model": "${REFERENCE_MODEL}",
  "reference_results_json": "${REFERENCE_RESULTS_JSON}",
  "reference_target_id": "${REFERENCE_TARGET_ID}",
  "skippy_spec_draft_fault_every": "${SKIPPY_SPEC_DRAFT_FAULT_EVERY:-}",
  "skippy_spec_draft_fault_offset": "${SKIPPY_SPEC_DRAFT_FAULT_OFFSET:-}",
  "skippy_spec_draft_fault_rank": "${SKIPPY_SPEC_DRAFT_FAULT_RANK:-}",
  "skippy_spec_return_reconnect_every": "${SKIPPY_SPEC_RETURN_RECONNECT_EVERY:-}",
  "stage_downstream_wire_delay_ms": "${MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS:-}",
  "stage_downstream_wire_jitter_ms": "${MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS:-}",
  "stage_downstream_wire_mbps": "${MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS:-}",
  "target_model": "${TARGET_MODEL}",
  "task_id": "${TASK_ID}",
  "worker_flavor": "${WORKER_FLAVOR}",
  "worker_image": "${WORKER_IMAGE}"
}
EOF

echo "proof_dir=${PROOF_DIR}"
echo "ledger=${LEDGER}"
echo "artifact_sha=${ARTIFACT_SHA}"
echo "target=${TARGET_MODEL} draft=${DRAFT_GGUF} forced=${FORCED_BOUNDARIES} modes=${MODES}"
if [[ -n "$REFERENCE_RESULTS_JSON" || -n "$REFERENCE_BASE_URL" ]]; then
    prepare_reference_results
fi
prepare_local_native_runtime

mode_index=0
for mode in $MODES; do
    CURRENT_MODE="$mode"
    CURRENT_JOB_ID=""
    CURRENT_SEED_PID=""
    SEED_API_PORT=$((SEED_API_PORT_BASE + mode_index * MODE_PORT_STRIDE))
    SEED_CONSOLE_PORT=$((SEED_CONSOLE_PORT_BASE + mode_index * MODE_PORT_STRIDE))
    SEED_BIND_PORT=$((SEED_BIND_PORT_BASE + mode_index * MODE_PORT_STRIDE))
    WORKER_PORT=$((WORKER_API_PORT_BASE + mode_index * MODE_PORT_STRIDE))
    WORKER_CONSOLE=$((WORKER_CONSOLE_PORT_BASE + mode_index * MODE_PORT_STRIDE))
    WORKER_BIND_PORT=$((WORKER_BIND_PORT_BASE + mode_index * MODE_PORT_STRIDE))
    mode_index=$((mode_index + 1))

    config="${PROOF_DIR}/configs/${mode}-seed.toml"
    write_seed_config "$mode" "$config"
    "$MESH_LLM" config validate --config-path "$config" --json >"${PROOF_DIR}/configs/${mode}-validate.json"

    seed_log="${PROOF_DIR}/${mode}-seed.log"
    CURRENT_SEED_PID="$(start_seed "$mode" "$config" "$seed_log" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT")"
    echo "${mode}: seed pid ${CURRENT_SEED_PID} ports api=${SEED_API_PORT} console=${SEED_CONSOLE_PORT} bind=${SEED_BIND_PORT}"
    token="$(wait_for_token "$CURRENT_SEED_PID" "$SEED_CONSOLE_PORT" "$seed_log")"
    seed_node_id="$(status_json "$SEED_CONSOLE_PORT" | query_json_field node_id)"
    if [[ -z "$seed_node_id" ]]; then
        echo "${mode}: missing seed node id" >&2
        exit 1
    fi
    mesh_name="shard-wan-${mode}-${TASK_ID}"
    CURRENT_JOB_ID="$(launch_worker "$mode" "$token" "$mesh_name" "$WORKER_PORT" "$WORKER_CONSOLE" "$WORKER_BIND_PORT" "$seed_node_id")"
    echo "${mode}: hf job ${CURRENT_JOB_ID} launched"

    model_id="$(wait_for_model "$mode" "$CURRENT_SEED_PID" "$seed_node_id" "$seed_log" "$CURRENT_JOB_ID")"
    write_topology_snapshot "$mode" "$model_id" "${PROOF_DIR}/results/${mode}-topology.json" >/dev/null
    run_with_watchdog "$HF_LOG_TIMEOUT" "${PROOF_DIR}/process/${mode}-hf-ready-log-tail.txt" \
        hf jobs logs "$CURRENT_JOB_ID" --namespace "$NAMESPACE" --tail 220 || true
    run_requests "$mode" "$model_id" "${PROOF_DIR}/results/${mode}.json" "$seed_log"
    run_with_watchdog "$HF_LOG_TIMEOUT" "${PROOF_DIR}/process/${mode}-hf-final-log-tail.txt" \
        hf jobs logs "$CURRENT_JOB_ID" --namespace "$NAMESPACE" --tail 260 || true
    run_with_watchdog "$HF_LOG_TIMEOUT" "${PROOF_DIR}/process/${mode}-hf-final-inspect.json" \
        hf jobs inspect "$CURRENT_JOB_ID" --namespace "$NAMESPACE" || true
    cancel_worker "$CURRENT_JOB_ID" "$mode" "mode_complete"
    CURRENT_JOB_ID=""
    kill_seed "$CURRENT_SEED_PID"
    CURRENT_SEED_PID=""
    echo "${mode}: complete"
done

summarize_results
trap - EXIT
echo "wrote summary ${PROOF_DIR}/results/summary.json"
