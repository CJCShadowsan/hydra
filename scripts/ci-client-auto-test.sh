#!/usr/bin/env bash
# ci-client-auto-test.sh — verify `mesh-llm client --auto` boots its API
# AND actually joins a mesh without depending on public relays.
#
# Two invariants:
#   1. While joining a direct local peer, the management API on :3132 must come
#      up. A broken implementation can wedge in join/bootstrap and never bind
#      the console port.
#   2. The node must actually join the local mesh — i.e. /api/status reports
#      mesh_id set AND peers non-empty. This catches regressions where gossip
#      or the join handshake silently break and the node sits idle.
#
# Usage: scripts/ci-client-auto-test.sh <mesh-llm-binary>
#
# Exits 0 only if both invariants hold; 1 otherwise.

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONSOLE_PORT="${MESH_LLM_CLIENT_AUTO_CONSOLE_PORT:-3132}" # avoid clashing with other CI steps
API_PORT="${MESH_LLM_CLIENT_AUTO_API_PORT:-9338}"
SEED_CONSOLE_PORT="${MESH_LLM_CLIENT_AUTO_SEED_CONSOLE_PORT:-3133}"
SEED_API_PORT="${MESH_LLM_CLIENT_AUTO_SEED_API_PORT:-9339}"
SEED_BIND_PORT="${MESH_LLM_CLIENT_AUTO_SEED_BIND_PORT:-53338}"
MAX_WAIT="${MESH_LLM_CLIENT_AUTO_MAX_WAIT:-120}" # seconds for local direct join
JOIN_WAIT="${MESH_LLM_CLIENT_AUTO_JOIN_WAIT:-60}"
LOG=/tmp/mesh-llm-client-auto.log
SEED_LOG=/tmp/mesh-llm-client-auto-seed.log

echo "=== CI Client-Auto Test ==="
echo "  mesh-llm:       $MESH_LLM"
echo "  client console: $CONSOLE_PORT"
echo "  client api:     $API_PORT"
echo "  seed console:   $SEED_CONSOLE_PORT"
echo "  seed api:       $SEED_API_PORT"
echo "  seed bind:      $SEED_BIND_PORT"
echo "  max wait:       ${MAX_WAIT}s"
echo "  join wait:      ${JOIN_WAIT}s"
echo "  os:             $(uname -s)"

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

RUNTIME_CACHE="$("$REPO_ROOT/scripts/ci-install-native-runtime.sh" "$MESH_LLM" "$REPO_ROOT/target/client-auto-native-runtime" cpu)"
export MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$RUNTIME_CACHE"

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
MESH_PID=""
cleanup() {
    echo "Shutting down mesh-llm processes..."
    kill_tree "$MESH_PID"
    kill_tree "$SEED_PID"
    echo "--- Seed log (last 80 lines) ---"
    tail -80 "$SEED_LOG" 2>/dev/null || true
    echo "--- Client log (last 80 lines) ---"
    tail -80 "$LOG" 2>/dev/null || true
    echo "--- End logs ---"
    echo "Cleanup done."
}
trap cleanup EXIT

echo "Starting local seed mesh without iroh relays..."
"$MESH_LLM" \
    --log-format json \
    --mesh-discovery-mode mdns \
    client \
    --auto \
    --port "$SEED_API_PORT" \
    --console "$SEED_CONSOLE_PORT" \
    --bind-ip 127.0.0.1 \
    --bind-port "$SEED_BIND_PORT" \
    --headless \
    > "$SEED_LOG" 2>&1 &
SEED_PID=$!
echo "  Seed PID: $SEED_PID"

TOKEN=""
echo "Waiting for seed invite token on port ${SEED_CONSOLE_PORT} (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$SEED_PID" 2>/dev/null; then
        echo "❌ seed mesh exited unexpectedly"
        tail -80 "$SEED_LOG" 2>/dev/null || true
        exit 1
    fi

    STATUS=$(curl -sf --max-time 2 "http://localhost:${SEED_CONSOLE_PORT}/api/status" 2>/dev/null || echo "")
    TOKEN="$(
        printf '%s' "$STATUS" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("token", ""))
except Exception:
    print("")' 2>/dev/null || echo ""
    )"

    if [ -n "$TOKEN" ]; then
        echo "✅ Seed mesh is ready after ${i}s"
        break
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting for seed... (${i}s)"
        tail -3 "$SEED_LOG" 2>/dev/null | sed 's/^/    /' || true
    fi
    sleep 1
done

if [ -z "$TOKEN" ]; then
    echo "❌ Timed out waiting for seed invite token"
    exit 1
fi

echo "Starting mesh-llm client --auto with direct local join token..."
"$MESH_LLM" \
    --log-format json \
    --mesh-discovery-mode mdns \
    client \
    --auto \
    --join "$TOKEN" \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    --headless \
    > "$LOG" 2>&1 &
MESH_PID=$!
echo "  Client PID: $MESH_PID"

# Wait for the console API to become reachable.
# This is the core assertion: the management API MUST come up while the node is
# joining a mesh peer.
echo "Waiting for console API on port ${CONSOLE_PORT} (up to ${MAX_WAIT}s)..."
API_UP=false
for i in $(seq 1 "$MAX_WAIT"); do
    # Check process is still alive
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "⚠️  mesh-llm exited before the console API became reachable"
        echo "--- Log tail ---"
        tail -40 "$LOG" 2>/dev/null || true
        echo "❌ mesh-llm exited unexpectedly"
        exit 1
    fi

    # Try to hit the console API
    if curl -sf --max-time 2 "http://localhost:${CONSOLE_PORT}/api/status" > /dev/null 2>&1; then
        echo "✅ Console API is up after ${i}s"
        API_UP=true
        break
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
        # Show last few log lines for debugging
        tail -3 "$LOG" 2>/dev/null | sed 's/^/    /' || true
    fi
    sleep 1
done

if [ "$API_UP" = true ]; then
    # Bonus: verify the status endpoint returns valid JSON
    echo "Verifying /api/status response..."
    STATUS=$(curl -sf --max-time 5 "http://localhost:${CONSOLE_PORT}/api/status" 2>&1 || echo "")
    if [ -n "$STATUS" ]; then
        # Check it's valid JSON with expected fields
        if echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'version' in d or 'peers' in d or 'models' in d" 2>/dev/null; then
            echo "✅ /api/status returns valid JSON"
        else
            echo "⚠️  /api/status returned something but couldn't validate: $STATUS"
        fi
    fi

    # Verify /v1/models is reachable (through the proxy port)
    echo "Checking /v1/models on port ${API_PORT}..."
    if curl -sf --max-time 5 "http://localhost:${API_PORT}/v1/models" > /dev/null 2>&1; then
        echo "✅ /v1/models is reachable"
    else
        echo "⚠️  /v1/models not reachable (may be expected with no live peers)"
    fi

    # Required "actually joined a mesh" signal.
    #
    # Predicate: mesh_id set AND peers non-empty.
    #
    # We deliberately do NOT accept `first_joined_mesh_ts` as evidence of a
    # successful join. Standalone fallback also sets that timestamp with zero
    # peers. The honest signal is at least one real peer.
    #
    echo "Polling /api/status for at least one peer (mesh_id set AND peers non-empty, up to ${JOIN_WAIT}s)..."
    JOINED=false
    for j in $(seq 1 "$JOIN_WAIT"); do
        STATUS=$(curl -sf --max-time 5 "http://localhost:${CONSOLE_PORT}/api/status" 2>/dev/null || echo "")
        if [ -n "$STATUS" ] && echo "$STATUS" | python3 -c '
import json, sys
d = json.load(sys.stdin)
mesh_id = d.get("mesh_id")
peers = d.get("peers") or []
if mesh_id and peers:
    sys.exit(0)
sys.exit(1)
' 2>/dev/null; then
            MESH_ID=$(echo "$STATUS" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("mesh_id",""))')
            PEER_COUNT=$(echo "$STATUS" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("peers") or []))')
            echo "✅ Joined mesh after ${j}s — mesh_id=${MESH_ID} peers=${PEER_COUNT}"
            JOINED=true
            break
        fi
        if [ $((j % 10)) -eq 0 ]; then
            echo "  Still waiting for at least one peer... (${j}s)"
        fi
        sleep 1
    done

    if [ "$JOINED" = false ]; then
        echo ""
        echo "❌ Console API came up but the node never saw any peer within ${JOIN_WAIT}s."
        echo "   Required signal: /api/status with mesh_id set AND peers non-empty."
        echo "   (first_joined_mesh_ts alone is NOT accepted — standalone fallback sets it too.)"
        echo "   Last /api/status body:"
        echo "$STATUS" | sed 's/^/     /'
        exit 1
    fi

    echo ""
    echo "=== Client-auto test passed ==="
    exit 0
fi

echo ""
if grep -qE "Joining:|Joined mesh" "$LOG" 2>/dev/null; then
    echo "❌ Console API never became reachable within ${MAX_WAIT}s"
    echo "   Node started joining the local mesh but the API never came up."
else
    echo "❌ Console API never became reachable within ${MAX_WAIT}s"
    echo "   Unknown state — check logs above."
fi
exit 1
