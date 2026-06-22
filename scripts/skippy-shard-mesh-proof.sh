#!/usr/bin/env bash
# Prove Shard-style speculative serving over mesh split stages on a persistent
# GPU host. Run target-only first, then sync draft, pipelined draft, and tree
# with the same target, draft, prompts, split topology, and optional synthetic
# downstream wire latency.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/skippy-shard-mesh-proof.sh <mesh-llm-binary> <target-model-ref> [draft-gguf]

Environment:
  MESH_SHARD_PROOF_MODES                 Space-separated modes. Default: target sync-draft pipelined-draft tree
  MESH_SHARD_PROOF_WORK_DIR              Evidence directory. Default: /tmp/mesh-shard-proof.<pid>
  MESH_SHARD_PROOF_PROCESS_ROOT          Short per-node HOME/runtime root. Default: /tmp/mesh-shard-process.*
  MESH_SHARD_PROOF_CACHE_ROOT            Shared cache root. Default: <work-dir>/cache
  MESH_SHARD_PROOF_HF_HOME               Default: <cache-root>/hf-home
  MESH_SHARD_PROOF_HUGGINGFACE_HUB_CACHE Default: <hf-home>/hub
  MESH_SHARD_PROOF_XDG_CACHE_HOME        Default: <cache-root>/xdg-cache
  MESH_SHARD_PROOF_PROMPTS_JSONL         Optional JSONL with {"id","prompt"} rows
  MESH_SHARD_PROOF_CTX_SIZE              Default: 512
  MESH_SHARD_PROOF_UBATCH                Default: 16
  MESH_SHARD_PROOF_MAX_TOKENS            Default: 32
  MESH_SHARD_PROOF_DRAFT_MAX_TOKENS      Default: 6
  MESH_SHARD_PROOF_PIPELINED_DEPTH       Default: 6
  MESH_SHARD_PROOF_MIN_ACCEPT_RATE       Default: 0.05
  MESH_SHARD_PROOF_MIN_PIPELINED_SPEEDUP Default: 1.05
  MESH_SHARD_PROOF_MIN_PIPELINED_VS_SYNC_SPEEDUP Default: 1.00
  MESH_SHARD_PROOF_REQUIRE_SHARD_GATES   Fail if Shard mechanics are missing. Default: 1
  MESH_SHARD_PROOF_REQUIRE_REFERENCE     Fail unless canonical reference is checked. Default: 0
  MESH_SHARD_PROOF_REQUIRE_ADVERSARIAL   Require rejection/stale-window paths. Default: 0
  MESH_SHARD_PROOF_WORKER_COUNT          Number of local mesh workers. Default: 1
  MESH_SHARD_PROOF_MIN_ACTIVE_STAGES     Required active split stages. Default: worker-count + 1
  MESH_LLM_SPLIT_MIN_PARTICIPANTS        Set automatically: required split participants for first topology
  MESH_LLM_SPLIT_FORCE_BOUNDARIES        Optional validation-only forced layer boundaries
  MESH_SHARD_PROOF_MAX_VRAM_GB           Default: 6
  MESH_SHARD_PROOF_SEED_MAX_VRAM_GB      Default: <max-vram-gb>
  MESH_SHARD_PROOF_WORKER_MAX_VRAM_GB    Default: <max-vram-gb>
  MESH_SHARD_PROOF_LLAMA_FLAVOR          Default: cuda
  MESH_SHARD_PROOF_CACHE_TYPE_K          Default: f16
  MESH_SHARD_PROOF_CACHE_TYPE_V          Default: f16
  MESH_SHARD_PROOF_STAGE_LOAD_TIMEOUT    Default: 900
  MESH_SHARD_PROOF_DEBUG                 Enable mesh-llm --debug for telemetry. Default: 1
  MESH_SHARD_PROOF_TELEMETRY_STDERR      Emit Skippy telemetry JSON to node logs. Default: 1
  MESH_SHARD_PROOF_REFERENCE_BASE_URL     Optional OpenAI-compatible full-target endpoint
  MESH_SHARD_PROOF_REFERENCE_MODEL        Model id for reference endpoint. Default: target model ref
  MESH_SHARD_PROOF_REFERENCE_API_KEY      Reference endpoint bearer token. Default: mesh
  MESH_SHARD_PROOF_REFERENCE_RESULTS_JSON Optional precomputed reference results JSON
  MESH_SHARD_PROOF_REFERENCE_TARGET_ID    Canonical target id expected in reference metadata.
                                          Default: target model ref
  MESH_LLM_SPLIT_PREFERRED_STAGE0         Set automatically: seed node coordinates split stages
  MESH_LLM_STAGE_TRANSPORT_DEBUG          Set automatically: log split-stage bridge/open diagnostics
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS Optional synthetic per-stage downstream delay
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS Optional synthetic per-stage downstream jitter
  MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS     Optional synthetic downstream bandwidth cap
  MESH_LLM_ALLOW_SLOW_DIRECT_STAGE_PATHS  Optional validation hook: admit high-RTT direct split peers
  SKIPPY_SPEC_DRAFT_FAULT_EVERY           Optional validation hook: force every Nth draft token off-greedy
  SKIPPY_SPEC_DRAFT_FAULT_OFFSET          Optional validation hook offset. Default: 0
  SKIPPY_SPEC_DRAFT_FAULT_RANK            Optional validation hook alternative rank. Default: 2
  SKIPPY_SPEC_RETURN_DELAY_EVERY          Optional validation hook: delay every Nth verify return
  SKIPPY_SPEC_RETURN_DELAY_OFFSET         Optional validation hook offset. Default: 0
  SKIPPY_SPEC_RETURN_DELAY_MS             Optional validation hook delay in milliseconds
  SKIPPY_SPEC_RETURN_RECONNECT_EVERY      Optional validation hook: force direct-return writer reconnect after every N replies

The script launches one seed plus N local mesh-llm serving workers on the
current host. Use it inside a persistent SSH GPU instance such as Vast/RunPod,
then inspect the written logs and results before moving to HF or larger
GLM-class runs.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_LLM="${1:?missing mesh-llm binary; see --help}"
TARGET_MODEL="${2:?missing target model ref/path; see --help}"
MODES="${MESH_SHARD_PROOF_MODES:-target sync-draft pipelined-draft tree}"
DRAFT_GGUF="${3:-}"
REFERENCE_BASE_URL="${MESH_SHARD_PROOF_REFERENCE_BASE_URL:-}"
REFERENCE_MODEL="${MESH_SHARD_PROOF_REFERENCE_MODEL:-$TARGET_MODEL}"
REFERENCE_API_KEY="${MESH_SHARD_PROOF_REFERENCE_API_KEY:-mesh}"
REFERENCE_RESULTS_JSON="${MESH_SHARD_PROOF_REFERENCE_RESULTS_JSON:-}"
REFERENCE_TARGET_ID="${MESH_SHARD_PROOF_REFERENCE_TARGET_ID:-$TARGET_MODEL}"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "mesh-llm binary is not executable: $MESH_LLM" >&2
    exit 1
fi
mode_needs_draft() {
    for mode in $MODES; do
        if [[ "$mode" != "target" ]]; then
            return 0
        fi
    done
    return 1
}

if mode_needs_draft && [[ -z "$DRAFT_GGUF" ]]; then
    echo "draft GGUF is required unless MESH_SHARD_PROOF_MODES only contains target" >&2
    exit 1
fi
if [[ -n "$DRAFT_GGUF" && ! -f "$DRAFT_GGUF" ]]; then
    echo "draft GGUF does not exist: $DRAFT_GGUF" >&2
    exit 1
fi
if [[ -n "$REFERENCE_RESULTS_JSON" && ! -f "$REFERENCE_RESULTS_JSON" ]]; then
    echo "reference results JSON does not exist: $REFERENCE_RESULTS_JSON" >&2
    exit 1
fi

WORK_DIR="${MESH_SHARD_PROOF_WORK_DIR:-$(mktemp -d "/tmp/mesh-shard-proof.XXXXXX")}"
PROCESS_ROOT="${MESH_SHARD_PROOF_PROCESS_ROOT:-$(mktemp -d "/tmp/mesh-shard-process.XXXXXX")}"
CACHE_ROOT="${MESH_SHARD_PROOF_CACHE_ROOT:-${WORK_DIR}/cache}"
HF_HOME_DIR="${MESH_SHARD_PROOF_HF_HOME:-${CACHE_ROOT}/hf-home}"
HF_HUB_CACHE_DIR="${MESH_SHARD_PROOF_HUGGINGFACE_HUB_CACHE:-${HF_HOME_DIR}/hub}"
XDG_CACHE_HOME_DIR="${MESH_SHARD_PROOF_XDG_CACHE_HOME:-${CACHE_ROOT}/xdg-cache}"
CONFIG_DIR="${WORK_DIR}/configs"
RESULT_DIR="${WORK_DIR}/results"
PROMPTS_JSONL="${MESH_SHARD_PROOF_PROMPTS_JSONL:-${WORK_DIR}/prompts.jsonl}"
CTX_SIZE="${MESH_SHARD_PROOF_CTX_SIZE:-512}"
UBATCH="${MESH_SHARD_PROOF_UBATCH:-16}"
MAX_TOKENS="${MESH_SHARD_PROOF_MAX_TOKENS:-32}"
DRAFT_MAX_TOKENS="${MESH_SHARD_PROOF_DRAFT_MAX_TOKENS:-6}"
PIPELINED_DEPTH="${MESH_SHARD_PROOF_PIPELINED_DEPTH:-6}"
MIN_ACCEPT_RATE="${MESH_SHARD_PROOF_MIN_ACCEPT_RATE:-0.05}"
MIN_PIPELINED_SPEEDUP="${MESH_SHARD_PROOF_MIN_PIPELINED_SPEEDUP:-1.05}"
MIN_PIPELINED_VS_SYNC_SPEEDUP="${MESH_SHARD_PROOF_MIN_PIPELINED_VS_SYNC_SPEEDUP:-1.00}"
REQUIRE_SHARD_GATES="${MESH_SHARD_PROOF_REQUIRE_SHARD_GATES:-1}"
REQUIRE_CANONICAL_REFERENCE="${MESH_SHARD_PROOF_REQUIRE_REFERENCE:-0}"
REQUIRE_ADVERSARIAL="${MESH_SHARD_PROOF_REQUIRE_ADVERSARIAL:-0}"
WORKER_COUNT="${MESH_SHARD_PROOF_WORKER_COUNT:-1}"
MIN_ACTIVE_STAGES="${MESH_SHARD_PROOF_MIN_ACTIVE_STAGES:-$((WORKER_COUNT + 1))}"
MAX_VRAM_GB="${MESH_SHARD_PROOF_MAX_VRAM_GB:-6}"
SEED_MAX_VRAM_GB="${MESH_SHARD_PROOF_SEED_MAX_VRAM_GB:-$MAX_VRAM_GB}"
WORKER_MAX_VRAM_GB="${MESH_SHARD_PROOF_WORKER_MAX_VRAM_GB:-$MAX_VRAM_GB}"
LLAMA_FLAVOR="${MESH_SHARD_PROOF_LLAMA_FLAVOR:-cuda}"
CACHE_TYPE_K="${MESH_SHARD_PROOF_CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${MESH_SHARD_PROOF_CACHE_TYPE_V:-f16}"
STAGE_LOAD_TIMEOUT="${MESH_SHARD_PROOF_STAGE_LOAD_TIMEOUT:-900}"
PROOF_DEBUG="${MESH_SHARD_PROOF_DEBUG:-1}"
TELEMETRY_STDERR="${MESH_SHARD_PROOF_TELEMETRY_STDERR:-1}"
MESH_NAME="shard-proof-$(date -u +%Y%m%dT%H%M%SZ)-$$"

SEED_API_PORT_BASE="${MESH_SHARD_PROOF_SEED_API_PORT:-9337}"
SEED_CONSOLE_PORT_BASE="${MESH_SHARD_PROOF_SEED_CONSOLE_PORT:-3131}"
SEED_BIND_PORT_BASE="${MESH_SHARD_PROOF_SEED_BIND_PORT:-53647}"
WORKER_API_PORT_BASE="${MESH_SHARD_PROOF_WORKER_API_PORT:-9447}"
WORKER_CONSOLE_PORT_BASE="${MESH_SHARD_PROOF_WORKER_CONSOLE_PORT:-3145}"
WORKER_BIND_PORT_BASE="${MESH_SHARD_PROOF_WORKER_BIND_PORT:-53648}"
MODE_PORT_STRIDE="${MESH_SHARD_PROOF_MODE_PORT_STRIDE:-50}"
SEED_API_PORT="$SEED_API_PORT_BASE"
SEED_CONSOLE_PORT="$SEED_CONSOLE_PORT_BASE"
SEED_BIND_PORT="$SEED_BIND_PORT_BASE"
WORKER_API_PORT="$WORKER_API_PORT_BASE"
WORKER_CONSOLE_PORT="$WORKER_CONSOLE_PORT_BASE"
WORKER_BIND_PORT="$WORKER_BIND_PORT_BASE"
MAX_WAIT="${MESH_SHARD_PROOF_MAX_WAIT:-900}"

if ! [[ "$WORKER_COUNT" =~ ^[0-9]+$ ]] || [[ "$WORKER_COUNT" -lt 1 ]]; then
    echo "MESH_SHARD_PROOF_WORKER_COUNT must be a positive integer" >&2
    exit 2
fi
if ! [[ "$MIN_ACTIVE_STAGES" =~ ^[0-9]+$ ]] || [[ "$MIN_ACTIVE_STAGES" -lt 2 ]]; then
    echo "MESH_SHARD_PROOF_MIN_ACTIVE_STAGES must be an integer >= 2" >&2
    exit 2
fi
if [[ "$REQUIRE_CANONICAL_REFERENCE" == "1" && -z "$REFERENCE_RESULTS_JSON" && -z "$REFERENCE_BASE_URL" ]]; then
    echo "MESH_SHARD_PROOF_REQUIRE_REFERENCE=1 requires MESH_SHARD_PROOF_REFERENCE_RESULTS_JSON or MESH_SHARD_PROOF_REFERENCE_BASE_URL" >&2
    exit 2
fi

mkdir -p "$PROCESS_ROOT" "$CONFIG_DIR" "$RESULT_DIR" "$HF_HOME_DIR" "$HF_HUB_CACHE_DIR" "$XDG_CACHE_HOME_DIR"

if [[ ! -f "$PROMPTS_JSONL" ]]; then
    cat >"$PROMPTS_JSONL" <<'EOF'
{"id":"exact-1","prompt":"Return exactly: cache locality matters"}
{"id":"exact-2","prompt":"Return exactly: speculative decoding is deterministic"}
{"id":"exact-3","prompt":"Repeat exactly, with no extra words: direct return pipelines stale windows across wide area links"}
EOF
fi

echo "=== Skippy Shard Mesh Proof ==="
echo "  mesh-llm:       $MESH_LLM"
echo "  target:         $TARGET_MODEL"
echo "  draft:          ${DRAFT_GGUF:-not required for target-only}"
echo "  modes:          $MODES"
echo "  work dir:       $WORK_DIR"
echo "  process root:   $PROCESS_ROOT"
echo "  hf hub cache:   $HF_HUB_CACHE_DIR"
echo "  ctx size:       $CTX_SIZE"
  echo "  max tokens:     $MAX_TOKENS"
  echo "  worker count:   $WORKER_COUNT"
  echo "  min stages:     $MIN_ACTIVE_STAGES"
  echo "  min accept:     $MIN_ACCEPT_RATE"
  echo "  min pipe gain:  $MIN_PIPELINED_SPEEDUP"
  echo "  min pipe/sync:  $MIN_PIPELINED_VS_SYNC_SPEEDUP"
  echo "  shard gates:    require=${REQUIRE_SHARD_GATES} reference=${REQUIRE_CANONICAL_REFERENCE} adversarial=${REQUIRE_ADVERSARIAL}"
  echo "  seed vram gb:   $SEED_MAX_VRAM_GB"
  echo "  worker vram gb: $WORKER_MAX_VRAM_GB"
  echo "  stage0 pref:    seed node"
  echo "  force bounds:   ${MESH_LLM_SPLIT_FORCE_BOUNDARIES:-unset}"
  echo "  stage delay ms: ${MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS:-0}"
  echo "  stage jitter ms: ${MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS:-0}"
  echo "  stage mbps:     ${MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS:-unset}"
  echo "  allow slow WAN: ${MESH_LLM_ALLOW_SLOW_DIRECT_STAGE_PATHS:-0}"
  echo "  debug telemetry: mesh-debug=${PROOF_DEBUG} stderr=${TELEMETRY_STDERR}"
  echo "  draft fault:    every=${SKIPPY_SPEC_DRAFT_FAULT_EVERY:-0} offset=${SKIPPY_SPEC_DRAFT_FAULT_OFFSET:-0} rank=${SKIPPY_SPEC_DRAFT_FAULT_RANK:-2}"
  echo "  return delay:   every=${SKIPPY_SPEC_RETURN_DELAY_EVERY:-0} offset=${SKIPPY_SPEC_RETURN_DELAY_OFFSET:-0} ms=${SKIPPY_SPEC_RETURN_DELAY_MS:-0}"
  echo "  return reconnect every: ${SKIPPY_SPEC_RETURN_RECONNECT_EVERY:-0}"
  echo "  reference:      ${REFERENCE_RESULTS_JSON:-${REFERENCE_BASE_URL:-unchecked}}"
python3 - \
    "$RESULT_DIR/metadata.json" \
    "$TARGET_MODEL" \
    "$DRAFT_GGUF" \
    "$MODES" \
    "$CTX_SIZE" \
    "$MAX_TOKENS" \
    "$DRAFT_MAX_TOKENS" \
    "$PIPELINED_DEPTH" \
    "$MIN_ACCEPT_RATE" \
    "$MIN_PIPELINED_SPEEDUP" \
    "$MIN_PIPELINED_VS_SYNC_SPEEDUP" \
    "$REQUIRE_SHARD_GATES" \
    "$REQUIRE_CANONICAL_REFERENCE" \
    "$REQUIRE_ADVERSARIAL" \
    "$CACHE_TYPE_K" \
    "$CACHE_TYPE_V" \
    "$WORKER_COUNT" \
    "$MIN_ACTIVE_STAGES" \
    "$SEED_MAX_VRAM_GB" \
    "$WORKER_MAX_VRAM_GB" \
    "seed" \
    "$PROCESS_ROOT" \
    "$HF_HOME_DIR" \
    "$HF_HUB_CACHE_DIR" \
    "$XDG_CACHE_HOME_DIR" \
    "${MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS:-}" \
    "${MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS:-}" \
    "${MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS:-}" \
    "$PROOF_DEBUG" \
    "$TELEMETRY_STDERR" \
    "$REFERENCE_BASE_URL" \
    "$REFERENCE_MODEL" \
    "$REFERENCE_RESULTS_JSON" \
    "$REFERENCE_TARGET_ID" \
    "${SKIPPY_SPEC_DRAFT_FAULT_EVERY:-}" \
    "${SKIPPY_SPEC_DRAFT_FAULT_OFFSET:-}" \
    "${SKIPPY_SPEC_DRAFT_FAULT_RANK:-}" \
    "${SKIPPY_SPEC_RETURN_RECONNECT_EVERY:-}" <<'PY'
import json
import sys

(
    output_path,
    target_model,
    draft_gguf,
    modes,
    ctx_size,
    max_tokens,
    draft_max_tokens,
    pipelined_depth,
    min_accept_rate,
    min_pipelined_speedup,
    min_pipelined_vs_sync_speedup,
    require_shard_gates,
    require_canonical_reference,
    require_adversarial,
    cache_type_k,
    cache_type_v,
    worker_count,
    min_active_stages,
    seed_max_vram_gb,
    worker_max_vram_gb,
    preferred_stage0,
    process_root,
    hf_home,
    hf_hub_cache,
    xdg_cache_home,
    wire_delay_ms,
    wire_jitter_ms,
    wire_mbps,
    proof_debug,
    telemetry_stderr,
    reference_base_url,
    reference_model,
    reference_results_json,
    reference_target_id,
    draft_fault_every,
    draft_fault_offset,
    draft_fault_rank,
    return_reconnect_every,
) = sys.argv[1:]
metadata = {
    "target_model": target_model,
    "draft_gguf": draft_gguf or None,
    "modes": modes.split(),
    "ctx_size": int(ctx_size),
    "max_tokens": int(max_tokens),
    "draft_max_tokens": int(draft_max_tokens),
    "pipelined_depth": int(pipelined_depth),
    "min_accept_rate": float(min_accept_rate),
    "min_pipelined_speedup": float(min_pipelined_speedup),
    "min_pipelined_vs_sync_speedup": float(min_pipelined_vs_sync_speedup),
    "require_shard_gates": require_shard_gates == "1",
    "require_canonical_reference": require_canonical_reference == "1",
    "require_adversarial": require_adversarial == "1",
    "cache_type_k": cache_type_k,
    "cache_type_v": cache_type_v,
    "worker_count": int(worker_count),
    "min_active_stages": int(min_active_stages),
    "seed_max_vram_gb": seed_max_vram_gb,
    "worker_max_vram_gb": worker_max_vram_gb,
    "preferred_stage0": preferred_stage0,
    "process_root": process_root,
    "hf_home": hf_home,
    "hf_hub_cache": hf_hub_cache,
    "xdg_cache_home": xdg_cache_home,
    "stage_downstream_wire_delay_ms": wire_delay_ms or None,
    "stage_downstream_wire_jitter_ms": wire_jitter_ms or None,
    "stage_downstream_wire_mbps": wire_mbps or None,
    "proof_debug": proof_debug == "1",
    "reference_base_url": reference_base_url or None,
    "reference_model": reference_model,
    "reference_results_json": reference_results_json or None,
    "reference_target_id": reference_target_id,
    "telemetry_stderr": telemetry_stderr == "1",
    "draft_fault_every": draft_fault_every or None,
    "draft_fault_offset": draft_fault_offset or None,
    "draft_fault_rank": draft_fault_rank or None,
    "return_reconnect_every": return_reconnect_every or None,
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
PY

descendant_pids() {
    local pid="$1"
    local children
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    for child in $children; do
        descendant_pids "$child"
        printf '%s\n' "$child"
    done
}

kill_tree() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    local children
    children="$(descendant_pids "$pid" | sort -u || true)"
    kill "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill 2>/dev/null || true
    fi
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill -9 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

SEED_PID=""
WORKER_PIDS=()
WORKER_LOGS=()
CURRENT_MODE=""

cleanup_mode() {
    for worker_pid in "${WORKER_PIDS[@]}"; do
        kill_tree "$worker_pid"
    done
    kill_tree "$SEED_PID"
    SEED_PID=""
    if [[ -n "$CURRENT_MODE" ]]; then
        echo "--- ${CURRENT_MODE} seed log tail ---"
        tail -120 "${WORK_DIR}/${CURRENT_MODE}-seed.log" 2>/dev/null | redact_sensitive_log || true
        for worker_log in "${WORKER_LOGS[@]}"; do
            echo "--- ${CURRENT_MODE} $(basename "$worker_log") tail ---"
            tail -120 "$worker_log" 2>/dev/null | redact_sensitive_log || true
        done
        echo "--- ${CURRENT_MODE} native log tails ---"
        {
            find "${PROCESS_ROOT}/${CURRENT_MODE}-seed" "${PROCESS_ROOT}/${CURRENT_MODE}-worker"* \
                -path '*/logs/skippy-native.log' -type f -print 2>/dev/null || true
        } | while IFS= read -r native_log; do
                echo "--- ${native_log} ---"
                tail -80 "$native_log" | redact_sensitive_log || true
            done
        echo "--- end ${CURRENT_MODE} logs ---"
    fi
    WORKER_PIDS=()
    WORKER_LOGS=()
}

redact_sensitive_log() {
    sed -E \
        -e 's/"token":"[^"]+"/"token":"<redacted>"/g' \
        -e 's/(--join )[A-Za-z0-9._~+\/=:-]+/\1<redacted>/g'
}

cleanup_all() {
    cleanup_mode
    echo "evidence: $WORK_DIR"
}
trap cleanup_all EXIT

write_base_config() {
    local path="$1"
    cat >"$path" <<EOF
version = 1

[defaults.model_fit]
ctx_size = ${CTX_SIZE}
batch = 256
ubatch = ${UBATCH}
flash_attention = "disabled"
cache_type_k = "${CACHE_TYPE_K}"
cache_type_v = "${CACHE_TYPE_V}"

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

write_mode_config() {
    local mode="$1"
    local path="$2"
    write_base_config "$path"
    case "$mode" in
        target)
            ;;
        sync-draft)
            require_draft_for_mode "$mode"
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
            require_draft_for_mode "$mode"
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
            require_draft_for_mode "$mode"
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
            echo "unknown proof mode: $mode" >&2
            exit 2
            ;;
    esac
}

require_draft_for_mode() {
    local mode="$1"
    if [[ -z "$DRAFT_GGUF" ]]; then
        echo "mode ${mode} requires a draft GGUF" >&2
        exit 2
    fi
}

status_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/status" 2>/dev/null || true
}

runtime_stages_json() {
    local console_port="$1"
    curl -fsS --max-time 5 "http://127.0.0.1:${console_port}/api/runtime/stages" 2>/dev/null || true
}

model_start_error_line() {
    local log_file="$1"
    if [[ ! -f "$log_file" ]]; then
        return 0
    fi
    grep -E "Failed to start model|skippy incompatible speculative draft pairing" "$log_file" \
        | tail -1 \
        || true
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

first_model_id() {
    python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    data = {}
models = data.get("data") or []
print((models[0] or {}).get("id", "") if models else "")
'
}

topology_stage_count() {
    local model_id="$1"
    local stages_json="$2"
    local seed_node_id="${3:-}"
    python3 - "$model_id" "$stages_json" "$seed_node_id" "${MESH_LLM_SPLIT_FORCE_BOUNDARIES:-}" <<'PY'
import json
import sys

model_id = sys.argv[1]
stages_json = sys.argv[2]
seed_node_id = sys.argv[3]
forced_raw = sys.argv[4].strip()
try:
    data = json.loads(stages_json) if stages_json else {}
except Exception:
    data = {}
topologies = data.get("topologies") or []
statuses = data.get("stages") or data.get("statuses") or []
matching_topologies = [top for top in topologies if top.get("model_id") == model_id]
matching_statuses = [stage for stage in statuses if stage.get("model_id") == model_id]

def node_id_matches(actual, expected):
    if not actual or not expected:
        return False
    actual = str(actual)
    expected = str(expected)
    return actual == expected or actual.startswith(expected) or expected.startswith(actual)

def forced_boundaries():
    if not forced_raw:
        return []
    return [int(value.strip()) for value in forced_raw.split(",") if value.strip()]

def ranges_match_forced(stages, boundaries):
    if not boundaries:
        return True
    if len(stages) != len(boundaries) + 1:
        return False
    ordered = sorted(stages, key=lambda stage: stage.get("stage_index", 0))
    expected_starts = [0, *boundaries]
    for index, stage in enumerate(ordered):
        if stage.get("layer_start") != expected_starts[index]:
            return False
        if index < len(boundaries):
            if stage.get("layer_end") != boundaries[index]:
                return False
        elif not isinstance(stage.get("layer_end"), int) or stage.get("layer_end") <= expected_starts[index]:
            return False
    return True

def downstream_statuses_ready(topology, stages):
    for stage in sorted(stages, key=lambda stage: stage.get("stage_index", 0))[1:]:
        if not any(
            status.get("topology_id") == topology.get("topology_id")
            and status.get("run_id") == topology.get("run_id")
            and status.get("stage_id") == stage.get("stage_id")
            and status.get("state") == "ready"
            and status.get("layer_start") == stage.get("layer_start")
            and status.get("layer_end") == stage.get("layer_end")
            for status in matching_statuses
        ):
            return False
    return True

boundaries = forced_boundaries()
if boundaries:
    for topology in matching_topologies:
        stages = topology.get("stages") or []
        ordered = sorted(stages, key=lambda stage: stage.get("stage_index", 0))
        if (
            ordered
            and node_id_matches(ordered[0].get("node_id"), seed_node_id)
            and ranges_match_forced(ordered, boundaries)
            and downstream_statuses_ready(topology, ordered)
        ):
            print(len(ordered))
            break
    else:
        print(0)
else:
    selected_topologies = matching_topologies or topologies
    selected_statuses = matching_statuses or statuses
    topology_count = max((len(top.get("stages") or []) for top in selected_topologies), default=0)
    status_count = len(selected_statuses)
    print(max(topology_count, status_count) if topology_count or status_count else 0)
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
topology_stage_count = max(
    (len(topology.get("stages") or []) for topology in selected_topologies),
    default=0,
)
runtime_stage_count = len(selected_statuses)
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
payload = {
    "mode": mode,
    "model_id": model_id,
    "active_stage_count": max(topology_stage_count, runtime_stage_count),
    "topology_stage_count": topology_stage_count,
    "runtime_stage_count": runtime_stage_count,
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

split_coordinator_from_log() {
    local log_file="$1"
    sed -n 's/.*Split runtime coordinator is \([0-9a-f]*\);.*/\1/p' "$log_file" 2>/dev/null | tail -1
}

start_node() {
    local mode="$1"
    local label="$2"
    local join_token="$3"
    local api_port="$4"
    local console_port="$5"
    local bind_port="$6"
    local config_path="$7"
    local log_file="$8"
    local max_vram_gb="$9"
    local preferred_stage0="${10:-}"
    local home="${PROCESS_ROOT}/${mode}-${label}/home"
    local runtime="${PROCESS_ROOT}/${mode}-${label}/runtime"
    mkdir -p "$home" "$runtime"

    local -a args=(
        --log-format json
    )
    if [[ "$PROOF_DEBUG" != "0" ]]; then
        args+=(--debug)
    fi
    args+=(
        serve
        --model "$TARGET_MODEL"
        --split
        --config "$config_path"
        --port "$api_port"
        --console "$console_port"
        --bind-port "$bind_port"
        --headless
        --llama-flavor "$LLAMA_FLAVOR"
        --max-vram "$max_vram_gb"
        --mesh-name "$MESH_NAME"
        --name "${mode}-${label}"
    )
    if [[ "$mode" == "target" && "$label" == "seed" ]]; then
        args+=(--no-draft)
    fi
    if [[ -n "$join_token" ]]; then
        args+=(--join "$join_token")
    fi

    local -a env_cmd=(env)
    local -a env_vars=(
        "HOME=$home"
        "HF_HOME=$HF_HOME_DIR"
        "HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE_DIR"
        "HF_HUB_CACHE=$HF_HUB_CACHE_DIR"
        "XDG_CACHE_HOME=$XDG_CACHE_HOME_DIR"
        "MESH_LLM_RUNTIME_ROOT=$runtime"
        "MESH_LLM_EPHEMERAL_KEY=1"
        "MESH_LLM_SPLIT_PREFERRED_STAGE0=$preferred_stage0"
        "MESH_LLM_SPLIT_MIN_PARTICIPANTS=${MESH_LLM_SPLIT_MIN_PARTICIPANTS:-$MIN_ACTIVE_STAGES}"
        "MESH_LLM_SPLIT_FORCE_BOUNDARIES=${MESH_LLM_SPLIT_FORCE_BOUNDARIES:-}"
        "MESH_LLM_STAGE_LOAD_TIMEOUT_SECS=$STAGE_LOAD_TIMEOUT"
        "MESH_LLM_STAGE_TRANSPORT_DEBUG=1"
        "MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS=${MESH_LLM_STAGE_DOWNSTREAM_WIRE_DELAY_MS:-}"
        "MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS=${MESH_LLM_STAGE_DOWNSTREAM_WIRE_JITTER_MS:-}"
        "MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS=${MESH_LLM_STAGE_DOWNSTREAM_WIRE_MBPS:-}"
        "MESH_LLM_ALLOW_SLOW_DIRECT_STAGE_PATHS=${MESH_LLM_ALLOW_SLOW_DIRECT_STAGE_PATHS:-}"
        "SKIPPY_SPEC_DRAFT_FAULT_EVERY=${SKIPPY_SPEC_DRAFT_FAULT_EVERY:-}"
        "SKIPPY_SPEC_DRAFT_FAULT_OFFSET=${SKIPPY_SPEC_DRAFT_FAULT_OFFSET:-}"
        "SKIPPY_SPEC_DRAFT_FAULT_RANK=${SKIPPY_SPEC_DRAFT_FAULT_RANK:-}"
        "SKIPPY_SPEC_RETURN_DELAY_EVERY=${SKIPPY_SPEC_RETURN_DELAY_EVERY:-}"
        "SKIPPY_SPEC_RETURN_DELAY_OFFSET=${SKIPPY_SPEC_RETURN_DELAY_OFFSET:-}"
        "SKIPPY_SPEC_RETURN_DELAY_MS=${SKIPPY_SPEC_RETURN_DELAY_MS:-}"
        "SKIPPY_SPEC_RETURN_RECONNECT_EVERY=${SKIPPY_SPEC_RETURN_RECONNECT_EVERY:-}"
    )
    if [[ "$TELEMETRY_STDERR" != "0" ]]; then
        env_vars+=("SKIPPY_TELEMETRY_STDERR=1")
    else
        env_cmd+=("-u" "SKIPPY_TELEMETRY_STDERR")
    fi
    "${env_cmd[@]}" "${env_vars[@]}" "$MESH_LLM" "${args[@]}" >"$log_file" 2>&1 &
    printf '%s\n' "$!"
}

wait_for_token() {
    local pid="$1"
    local console_port="$2"
    local log_file="$3"
    local token=""
    for _ in $(seq 1 "$MAX_WAIT"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "node exited before invite token: $log_file" >&2
            tail -160 "$log_file" >&2 || true
            exit 1
        fi
        token="$(status_json "$console_port" | query_json_field token)"
        if [[ -n "$token" ]]; then
            printf '%s\n' "$token"
            return 0
        fi
        sleep 1
    done
    echo "timed out waiting for invite token" >&2
    tail -160 "$log_file" >&2 || true
    exit 1
}

wait_for_model() {
    local seed_pid="$1"
    local mode="$2"
    local seed_node_id="$3"
    local seed_log="$4"
    shift 4
    local worker_pids=("$@")
    local model_id=""
    for i in $(seq 1 "$MAX_WAIT"); do
        if ! kill -0 "$seed_pid" 2>/dev/null; then
            echo "seed exited unexpectedly for $mode" >&2
            return 1
        fi
        local start_error
        start_error="$(model_start_error_line "$seed_log")"
        if [[ -n "$start_error" ]]; then
            echo "model failed to start for ${mode}: ${start_error}" >&2
            tail -160 "$seed_log" >&2 || true
            return 1
        fi
        for worker_pid in "${worker_pids[@]}"; do
            if ! kill -0 "$worker_pid" 2>/dev/null; then
                echo "worker exited unexpectedly for $mode" >&2
                return 1
            fi
        done
        local coordinator_id
        coordinator_id="$(split_coordinator_from_log "$seed_log")"
        if [[ -n "$coordinator_id" && "$coordinator_id" != "$seed_node_id" ]]; then
            echo "seed lost split coordinator election for ${mode}: seed=${seed_node_id} coordinator=${coordinator_id}" >&2
            return 1
        fi
        local seed_peers
        seed_peers="$(status_json "$SEED_CONSOLE_PORT" | peer_count)"
        if [[ "$seed_peers" -ge "$WORKER_COUNT" ]]; then
            local models_json
            models_json="$(curl -fsS --max-time 5 "http://127.0.0.1:${SEED_API_PORT}/v1/models" 2>/dev/null || true)"
            model_id="$(printf '%s\n' "$models_json" | first_model_id)"
            if [[ -n "$model_id" ]]; then
                local active_stages
                active_stages="$(
                    topology_stage_count \
                        "$model_id" \
                        "$(runtime_stages_json "$SEED_CONSOLE_PORT")" \
                        "$seed_node_id"
                )"
                if [[ "$active_stages" -ge "$MIN_ACTIVE_STAGES" ]]; then
                    echo "mode ${mode} ready after ${i}s: model=${model_id} active_stages=${active_stages}" >&2
                    printf '%s\n' "$model_id"
                    return 0
                fi
            fi
        fi
        sleep 1
    done
    echo "timed out waiting for split model in mode $mode with >=${MIN_ACTIVE_STAGES} active stages" >&2
    return 1
}

run_requests() {
    local mode="$1"
    local model_id="$2"
    local output_json="$3"
    local seed_log="$4"
    python3 - "$mode" "$model_id" "$PROMPTS_JSONL" "$output_json" "$SEED_API_PORT" "$seed_log" <<'PY'
import json
import os
import sys
import time
import urllib.request

mode, model_id, prompts_path, output_path, port, seed_log_path = sys.argv[1:]
results = []
with open(prompts_path, "r", encoding="utf-8") as handle:
    prompts = [json.loads(line) for line in handle if line.strip()]

def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def telemetry_objects_since(path, offset):
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            raw = handle.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    objects = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        text = stripped
        if not text.startswith("{"):
            starts = [pos for pos in (line.find('{"attributes"'), line.find('{"event"')) if pos >= 0]
            if not starts:
                continue
            text = line[min(starts):]
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("attributes"), dict):
            objects.append(event)
    return objects

def spec_metrics(event):
    attrs = event.get("attributes") or {}
    wanted_prefixes = (
        "llama_stage.spec.",
        "llama_stage.decode_",
        "llama_stage.prompt_",
        "llama_stage.completion_",
        "skippy.",
    )
    return {
        key: value
        for key, value in attrs.items()
        if any(key.startswith(prefix) for prefix in wanted_prefixes)
    }

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

for item in prompts:
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Answer deterministically and briefly."},
            {"role": "user", "content": item["prompt"]},
        ],
        "temperature": 0,
        "max_tokens": int(__import__("os").environ.get("MESH_SHARD_PROOF_MAX_TOKENS", "32")),
        "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer mesh"},
        method="POST",
    )
    log_offset = log_size(seed_log_path)
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - started
    time.sleep(0.1)
    telemetry = telemetry_objects_since(seed_log_path, log_offset)
    decode_events = [event for event in telemetry if event.get("event") == "stage.openai_decode"]
    verify_events = [
        event for event in telemetry if event.get("event") == "stage.openai_decode_verify_window"
    ]
    token_events = [
        event for event in telemetry if event.get("event") == "stage.openai_decode_token"
    ]
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    results.append({
        "mode": mode,
        "prompt_id": item.get("id"),
        "prompt": item["prompt"],
        "content": payload["choices"][0]["message"]["content"],
        "completion_token_ids": completion_token_ids(token_events),
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_s": completion_tokens / elapsed if completion_tokens else None,
        "usage": usage,
        "spec_metrics": spec_metrics(decode_events[-1]) if decode_events else None,
        "spec_verify_windows": [verify_window_metrics(event) for event in verify_events],
        "telemetry_decode_event_count": len(decode_events),
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
for item in prompts:
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Answer deterministically and briefly."},
            {"role": "user", "content": item["prompt"]},
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
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - started
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    results.append({
        "mode": "reference",
        "prompt_id": item.get("id"),
        "prompt": item["prompt"],
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
            "prompts": [{"id": item.get("id"), "prompt": item["prompt"]} for item in prompts],
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
    local output_json="${RESULT_DIR}/reference.json"
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

compare_results() {
    python3 - \
        "$WORK_DIR" \
        "$RESULT_DIR" \
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
baseline_path = result_dir / "target.json"
if not baseline_path.exists():
    raise SystemExit("missing target baseline result")
baseline = json.loads(baseline_path.read_text())
baseline_by_prompt = {row["prompt_id"]: row for row in baseline["results"]}
baseline_elapsed = sum(row["elapsed_s"] for row in baseline["results"])
baseline_tokens = sum(row.get("completion_tokens") or 0 for row in baseline["results"])
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
SUM_METRIC_KEYS = [
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
MAX_METRIC_KEYS = [
    "llama_stage.spec.pipelined_max_inflight_windows",
]

def metric_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def aggregate_spec_metrics(rows):
    totals = {key: 0 for key in SUM_METRIC_KEYS}
    maxima = {key: None for key in MAX_METRIC_KEYS}
    seen = set()
    decode_event_count = 0
    verify_window_count = 0
    committed_proposed = 0
    committed_accepted = 0
    spec_enabled = False
    for row in rows:
        metrics = row.get("spec_metrics") or {}
        decode_event_count += row.get("telemetry_decode_event_count") or 0
        verify_windows = row.get("spec_verify_windows") or []
        verify_window_count += len(verify_windows)
        for window in verify_windows:
            proposed = metric_number(window.get("llama_stage.spec.proposed")) or 0
            accepted = metric_number(window.get("llama_stage.spec.accepted")) or 0
            committed_proposed += proposed
            committed_accepted += accepted
        spec_enabled = spec_enabled or metrics.get("llama_stage.spec.enabled") is True
        for key in SUM_METRIC_KEYS:
            number = metric_number(metrics.get(key))
            if number is not None:
                totals[key] += number
                seen.add(key)
        for key in MAX_METRIC_KEYS:
            number = metric_number(metrics.get(key))
            if number is not None:
                maxima[key] = number if maxima[key] is None else max(maxima[key], number)
    aggregated = {key: totals[key] for key in SUM_METRIC_KEYS if key in seen}
    aggregated.update({key: value for key, value in maxima.items() if value is not None})
    proposed = aggregated.get("llama_stage.spec.proposed", 0)
    accepted = aggregated.get("llama_stage.spec.accepted", 0)
    aggregated["llama_stage.spec.accept_rate"] = accepted / proposed if proposed else None
    aggregated["llama_stage.spec.committed_proposed"] = committed_proposed
    aggregated["llama_stage.spec.committed_accepted"] = committed_accepted
    aggregated["llama_stage.spec.committed_accept_rate"] = (
        committed_accepted / committed_proposed if committed_proposed else None
    )
    aggregated["telemetry_decode_event_count"] = decode_event_count
    aggregated["telemetry_verify_window_count"] = verify_window_count
    aggregated["llama_stage.spec.enabled"] = spec_enabled
    verify_elapsed = aggregated.get("llama_stage.spec.primary_verify_elapsed_ms")
    downstream_wait = aggregated.get("llama_stage.spec.primary_verify_downstream_wait_ms")
    stage0_compute = aggregated.get("llama_stage.spec.primary_verify_stage0_compute_ms")
    aggregated["llama_stage.spec.primary_verify_downstream_wait_share"] = (
        downstream_wait / verify_elapsed if downstream_wait is not None and verify_elapsed else None
    )
    aggregated["llama_stage.spec.primary_verify_downstream_wait_vs_stage0_compute"] = (
        downstream_wait / stage0_compute if downstream_wait is not None and stage0_compute else None
    )
    return aggregated

def file_text(path):
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""

def mode_log_paths(mode):
    paths = list(proof_dir.glob(f"{mode}-*.log"))
    paths.extend((proof_dir / "process").glob(f"{mode}-*log*.txt"))
    paths.extend((proof_dir / "process").glob(f"{mode}-*.log"))
    return sorted({path for path in paths if path.exists()})

def files_contain(paths, needle):
    return any(needle in file_text(path) for path in paths)

def observed_direct_rtts(paths):
    values = []
    for path in paths:
        text = file_text(path)
        values.extend(int(value) for value in re.findall(r"RTT: ([0-9]+)ms \(direct\)", text))
        values.extend(int(value) for value in re.findall(r"rtt_ms=Some\(([0-9]+)\)", text))
    return values

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
                violations.append({
                    "prompt_id": row.get("prompt_id"),
                    "reason": "invalid_proposed",
                    "window": window,
                })
                continue
            max_proposed = max(max_proposed, proposed)
            if proposed > draft_max_tokens:
                violations.append({
                    "prompt_id": row.get("prompt_id"),
                    "reason": "proposed_gt_k",
                    "window": window,
                })
            if verify_inputs != proposed + 1:
                violations.append({
                    "prompt_id": row.get("prompt_id"),
                    "reason": "verify_inputs_not_proposed_plus_one",
                    "window": window,
                })
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
    matches = []
    token_matches = []
    reference_matches = None
    reference_token_matches = None
    if reference_by_prompt:
        reference_matches = []
        if any("completion_token_ids" in row for row in reference_rows):
            reference_token_matches = []
    for row in rows:
        expected = baseline_by_prompt.get(row["prompt_id"], {}).get("content")
        equal = row["content"] == expected
        matches.append(equal)
        if mode != "target" and not equal:
            failed = True
        expected_tokens = baseline_by_prompt.get(row["prompt_id"], {}).get("completion_token_ids")
        token_equal = row.get("completion_token_ids") == expected_tokens
        token_matches.append(token_equal)
        if require_shard_gates and mode != "target" and not token_equal:
            failed = True
        if reference_matches is not None:
            reference_expected = reference_by_prompt.get(row["prompt_id"], {}).get("content")
            reference_equal = row["content"] == reference_expected
            reference_matches.append(reference_equal)
            if not reference_equal:
                failed = True
        if reference_token_matches is not None:
            reference_expected_tokens = reference_by_prompt.get(row["prompt_id"], {}).get(
                "completion_token_ids"
            )
            reference_token_equal = row.get("completion_token_ids") == reference_expected_tokens
            reference_token_matches.append(reference_token_equal)
            if not reference_token_equal:
                failed = True
    elapsed = sum(row["elapsed_s"] for row in rows)
    tokens = sum(row.get("completion_tokens") or 0 for row in rows)
    tps = tokens / elapsed if elapsed and tokens else None
    spec_summary = aggregate_spec_metrics(rows)
    topology_path = result_dir / f"{mode}-topology.json"
    topology = json.loads(topology_path.read_text()) if topology_path.exists() else {}
    logs = mode_log_paths(mode)
    direct_stage_path_observed = files_contain(logs, "path_kind=Direct")
    direct_prediction_return_observed = files_contain(
        logs,
        "direct prediction return using upstream-opened sink",
    )
    direct_return_delay_observed = files_contain(
        logs,
        "skippy direct return validation delay:",
    )
    direct_return_reconnect_observed = files_contain(
        logs,
        "direct prediction return writer reconnected:",
    )
    rtts = observed_direct_rtts(logs)
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
        if mode in {"sync-draft", "pipelined-draft"}:
            accept_rate = spec_summary.get("llama_stage.spec.accept_rate")
            committed_accept_rate = (
                spec_summary.get("llama_stage.spec.committed_accept_rate") or accept_rate
            )
            proposed = spec_summary.get("llama_stage.spec.proposed") or 0
            accepted = spec_summary.get("llama_stage.spec.accepted") or 0
            committed_proposed = spec_summary.get("llama_stage.spec.committed_proposed") or 0
            committed_accepted = spec_summary.get("llama_stage.spec.committed_accepted") or 0
            rejected_windows = spec_summary.get("llama_stage.spec.rejected_windows") or 0
            stale_windows = spec_summary.get("llama_stage.spec.pipelined_stale_windows") or 0
            recovery_ms = spec_summary.get("llama_stage.spec.recovery_ms") or 0
            draft_reset_ms = spec_summary.get("llama_stage.spec.draft_reset_ms") or 0
            auto_align_count = spec_summary.get("skippy.verify_span_session_auto_align_count") or 0
            auto_align_trimmed = (
                spec_summary.get("skippy.verify_span_session_auto_align_trimmed_tokens") or 0
            )
            proof_gates["speculation_enabled"] = (
                spec_summary.get("llama_stage.spec.enabled") is True
            )
            proof_gates["speculation_engaged"] = (
                (committed_proposed or proposed) > 0
                and (committed_accepted or accepted) > 0
                and committed_accept_rate is not None
                and committed_accept_rate >= min_accept_rate
            )
            proof_gates["verify_chunk_shape_ok"] = bool(chunk_shape and chunk_shape.get("ok"))
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
        elif mode == "tree":
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
        "elapsed_ratio_vs_target": elapsed / baseline_elapsed if baseline_elapsed else None,
        "tokens_per_s_ratio_vs_target": tps / baseline_tps if tps and baseline_tps else None,
        "elapsed_ratio_vs_sync_draft": elapsed / sync_elapsed if sync_elapsed else None,
        "tokens_per_s_ratio_vs_sync_draft": tps / sync_tps if tps and sync_tps else None,
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
    raise SystemExit("Shard mesh proof gates failed")
PY
}

prepare_reference_results

mode_index=0
for mode in $MODES; do
    cleanup_mode
    CURRENT_MODE="$mode"
    port_offset=$((mode_index * MODE_PORT_STRIDE))
    SEED_API_PORT=$((SEED_API_PORT_BASE + port_offset))
    SEED_CONSOLE_PORT=$((SEED_CONSOLE_PORT_BASE + port_offset))
    SEED_BIND_PORT=$((SEED_BIND_PORT_BASE + port_offset))
    mode_index=$((mode_index + 1))
    seed_config="${CONFIG_DIR}/${mode}-seed.toml"
    write_mode_config "$mode" "$seed_config"
    "$MESH_LLM" config validate --config-path "$seed_config" --json >/dev/null

    seed_log="${WORK_DIR}/${mode}-seed.log"
    SEED_PID="$(start_node "$mode" seed "" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT" "$seed_config" "$seed_log" "$SEED_MAX_VRAM_GB" local)"
    token="$(wait_for_token "$SEED_PID" "$SEED_CONSOLE_PORT" "$seed_log")"
    seed_node_id="$(status_json "$SEED_CONSOLE_PORT" | query_json_field node_id)"
    if [[ -z "$seed_node_id" ]]; then
        echo "seed node id is unavailable after invite token" >&2
        exit 1
    fi
    WORKER_PIDS=()
    WORKER_LOGS=()
    for worker_index in $(seq 1 "$WORKER_COUNT"); do
        worker_label="worker${worker_index}"
        worker_config="${CONFIG_DIR}/${mode}-${worker_label}.toml"
        worker_log="${WORK_DIR}/${mode}-${worker_label}.log"
        worker_port_offset=$((port_offset + worker_index - 1))
        WORKER_API_PORT=$((WORKER_API_PORT_BASE + worker_port_offset))
        WORKER_CONSOLE_PORT=$((WORKER_CONSOLE_PORT_BASE + worker_port_offset))
        WORKER_BIND_PORT=$((WORKER_BIND_PORT_BASE + worker_port_offset))
        write_base_config "$worker_config"
        "$MESH_LLM" config validate --config-path "$worker_config" --json >/dev/null
        worker_pid="$(start_node "$mode" "$worker_label" "$token" "$WORKER_API_PORT" "$WORKER_CONSOLE_PORT" "$WORKER_BIND_PORT" "$worker_config" "$worker_log" "$WORKER_MAX_VRAM_GB" "$seed_node_id")"
        WORKER_PIDS+=("$worker_pid")
        WORKER_LOGS+=("$worker_log")
    done

    model_id="$(wait_for_model "$SEED_PID" "$mode" "$seed_node_id" "$seed_log" "${WORKER_PIDS[@]}")"
    write_topology_snapshot "$mode" "$model_id" "${RESULT_DIR}/${mode}-topology.json" >/dev/null
    run_requests "$mode" "$model_id" "${RESULT_DIR}/${mode}.json" "$seed_log"
done

CURRENT_MODE=""
cleanup_mode
compare_results

echo "wrote summary: ${RESULT_DIR}/summary.json"
