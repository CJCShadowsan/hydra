#!/usr/bin/env bash
# Capture canonical greedy OpenAI-compatible responses from a full, non-split
# target model. The output JSON is accepted by the Shard proof runners as
# MESH_SHARD_*_REFERENCE_RESULTS_JSON.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/skippy-shard-reference-capture.sh <mesh-llm-binary> <target-model-ref-or-gguf> [output-json]

Environment:
  MESH_SHARD_REFERENCE_WORK_DIR       Default: /tmp/mesh-shard-reference.<pid>
  MESH_SHARD_REFERENCE_PROMPTS_JSONL  Optional JSONL with {"id","prompt"} rows
  MESH_SHARD_REFERENCE_TARGET_ID      Canonical target id to store in the reference JSON.
                                      Default: target-model-ref-or-gguf argument
  MESH_SHARD_REFERENCE_CTX_SIZE       Default: 512
  MESH_SHARD_REFERENCE_UBATCH         Default: 16
  MESH_SHARD_REFERENCE_MAX_TOKENS     Default: 24
  MESH_SHARD_REFERENCE_PORT           Default: 9937
  MESH_SHARD_REFERENCE_CONSOLE        Default: 3731
  MESH_SHARD_REFERENCE_BIND_PORT      Default: 57647
  MESH_SHARD_REFERENCE_LLAMA_FLAVOR   Default: metal
  MESH_SHARD_REFERENCE_MAX_VRAM_GB    Default: 12
  MESH_SHARD_REFERENCE_WAIT_SECS      Default: 900

The script starts one local full-target mesh-llm server without --split, sends
deterministic chat requests, writes reference JSON, and stops the process.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MESH_LLM="${1:?missing mesh-llm binary; see --help}"
TARGET_MODEL="${2:?missing target model ref/path; see --help}"
WORK_DIR="${MESH_SHARD_REFERENCE_WORK_DIR:-$(mktemp -d "/tmp/mesh-shard-reference.XXXXXX")}"
OUTPUT_JSON="${3:-${WORK_DIR}/reference.json}"
PROMPTS_JSONL="${MESH_SHARD_REFERENCE_PROMPTS_JSONL:-${WORK_DIR}/prompts.jsonl}"
REFERENCE_TARGET_ID="${MESH_SHARD_REFERENCE_TARGET_ID:-$TARGET_MODEL}"
CTX_SIZE="${MESH_SHARD_REFERENCE_CTX_SIZE:-512}"
UBATCH="${MESH_SHARD_REFERENCE_UBATCH:-16}"
MAX_TOKENS="${MESH_SHARD_REFERENCE_MAX_TOKENS:-24}"
API_PORT="${MESH_SHARD_REFERENCE_PORT:-9937}"
CONSOLE_PORT="${MESH_SHARD_REFERENCE_CONSOLE:-3731}"
BIND_PORT="${MESH_SHARD_REFERENCE_BIND_PORT:-57647}"
LLAMA_FLAVOR="${MESH_SHARD_REFERENCE_LLAMA_FLAVOR:-metal}"
MAX_VRAM_GB="${MESH_SHARD_REFERENCE_MAX_VRAM_GB:-12}"
WAIT_SECS="${MESH_SHARD_REFERENCE_WAIT_SECS:-900}"
MODEL_MODE="model"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "mesh-llm binary is not executable: $MESH_LLM" >&2
    exit 2
fi

if [[ -f "$TARGET_MODEL" ]]; then
    MODEL_MODE="gguf"
fi

mkdir -p "$WORK_DIR" "$(dirname "$OUTPUT_JSON")"

if [[ ! -f "$PROMPTS_JSONL" ]]; then
    cat >"$PROMPTS_JSONL" <<'EOF'
{"id":"exact-1","prompt":"Return exactly: cache locality matters"}
{"id":"exact-2","prompt":"Return exactly: speculative decoding is deterministic"}
{"id":"exact-3","prompt":"Repeat exactly, with no extra words: direct return pipelines stale windows across wide area links"}
EOF
fi

CONFIG_PATH="${WORK_DIR}/reference.toml"
LOG_PATH="${WORK_DIR}/reference.log"
HOME_DIR="${WORK_DIR}/home"
RUNTIME_DIR="${WORK_DIR}/runtime"
mkdir -p "$HOME_DIR" "$RUNTIME_DIR"

cat >"$CONFIG_PATH" <<EOF
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

[defaults.request_defaults]
max_tokens = ${MAX_TOKENS}
temperature = 0.0
EOF

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

PID=""
cleanup() {
    kill_tree "$PID"
}
trap cleanup EXIT

args=(
    --log-format json
    --debug
    serve
    --config "$CONFIG_PATH"
    --port "$API_PORT"
    --console "$CONSOLE_PORT"
    --bind-port "$BIND_PORT"
    --headless
    --llama-flavor "$LLAMA_FLAVOR"
    --max-vram "$MAX_VRAM_GB"
    --mesh-name "shard-reference-$$"
    --name "reference-full"
)
if [[ "$MODEL_MODE" == "gguf" ]]; then
    args+=(--gguf "$TARGET_MODEL")
else
    args+=(--model "$TARGET_MODEL")
fi

env \
    HOME="$HOME_DIR" \
    MESH_LLM_RUNTIME_ROOT="$RUNTIME_DIR" \
    MESH_LLM_EPHEMERAL_KEY=1 \
    MESH_LLM_DYNAMIC_NATIVE_RUNTIME=0 \
    SKIPPY_TELEMETRY_STDERR=1 \
    "$MESH_LLM" "${args[@]}" >"$LOG_PATH" 2>&1 &
PID="$!"

python3 - \
    "$API_PORT" \
    "$PROMPTS_JSONL" \
    "$OUTPUT_JSON" \
    "$TARGET_MODEL" \
    "$REFERENCE_TARGET_ID" \
    "$CTX_SIZE" \
    "$UBATCH" \
    "$MAX_TOKENS" \
    "$WAIT_SECS" \
    "$LOG_PATH" <<'PY'
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

(
    port,
    prompts_path,
    output_path,
    target_model,
    reference_target_id,
    ctx_size,
    ubatch,
    max_tokens,
    wait_secs,
    log_path,
) = sys.argv[1:]
base = f"http://127.0.0.1:{port}"
deadline = time.time() + int(wait_secs)
last_error = None
models = []
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data") or []
        if models:
            break
    except Exception as exc:  # noqa: BLE001 - surfaced below with log path
        last_error = exc
    time.sleep(1)
else:
    raise SystemExit(f"timed out waiting for /v1/models; last_error={last_error}; log={log_path}")

model_id = models[0].get("id")
if not model_id:
    raise SystemExit(f"/v1/models returned no usable model id; log={log_path}")

def log_size(path):
    try:
        return pathlib.Path(path).stat().st_size
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

def completion_token_ids(events):
    token_ids = []
    for event in events:
        attrs = event.get("attributes") or {}
        token = attrs.get("llama_stage.predicted_token")
        if isinstance(token, int):
            token_ids.append(token)
    return token_ids

with open(prompts_path, "r", encoding="utf-8") as handle:
    prompts = [json.loads(line) for line in handle if line.strip()]

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
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer mesh"},
        method="POST",
    )
    offset = log_size(log_path)
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.time() - started
    time.sleep(0.1)
    telemetry = telemetry_since(log_path, offset)
    token_events = [
        event for event in telemetry if event.get("event") == "stage.openai_decode_token"
    ]
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    results.append({
        "mode": "reference",
        "prompt_id": item.get("id"),
        "prompt": item["prompt"],
        "content": payload["choices"][0]["message"]["content"],
        "completion_token_ids": completion_token_ids(token_events),
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tokens_per_s": completion_tokens / elapsed if completion_tokens else None,
        "usage": usage,
        "telemetry_decode_token_event_count": len(token_events),
    })

output = {
    "mode": "reference",
    "model_id": model_id,
    "base_url": base,
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
}
path = pathlib.Path(output_path)
path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
print(path)
PY

echo "reference: $OUTPUT_JSON"
echo "log: $LOG_PATH"
