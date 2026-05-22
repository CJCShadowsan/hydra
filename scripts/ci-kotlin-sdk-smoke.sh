#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

scripts/check-sdk-contract.sh

scripts/prepare-llama.sh "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"
LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-kotlin-cpu" \
LLAMA_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-kotlin-cpu" \
    scripts/build-llama.sh

LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-ci-kotlin-cpu" \
    cargo build -p mesh-llm-ffi --no-default-features --features host,embedded-runtime

scripts/ci-sdk-fixture.sh "$1" "$2" "$3" -- \
    bash -lc '
        set -euo pipefail
        if [ -x /usr/libexec/java_home ]; then
            JAVA_HOME="$(/usr/libexec/java_home -v 21 2>/dev/null || printf "%s" "${JAVA_HOME:-}")"
            export JAVA_HOME
        fi
        if [ -n "${JAVA_HOME:-}" ]; then
            export ORG_GRADLE_JAVA_HOME="${ORG_GRADLE_JAVA_HOME:-$JAVA_HOME}"
            export GRADLE_OPTS="${GRADLE_OPTS:-} -Dorg.gradle.java.installations.auto-detect=false -Dorg.gradle.java.installations.paths=$ORG_GRADLE_JAVA_HOME"
        fi
        for ext in dylib so; do
            lib='"$REPO_ROOT"'/target/debug/libmeshllm_ffi.$ext
            deps_lib='"$REPO_ROOT"'/target/debug/deps/libmeshllm_ffi.$ext
            if [ ! -f "$lib" ] && [ -f "$deps_lib" ]; then
                ln -sf deps/libmeshllm_ffi.$ext "$lib"
            fi
            if [ -f "$lib" ]; then
                ln -sf libmeshllm_ffi.$ext '"$REPO_ROOT"'/target/debug/libuniffi_mesh_ffi.$ext
            fi
        done
        export JAVA_TOOL_OPTIONS="-Djna.library.path='"$REPO_ROOT"'/target/debug"
        cd '"$REPO_ROOT"'/sdk/kotlin/example/example-jvm
        ./gradlew --no-daemon run --args="$MESH_SDK_INVITE_TOKEN"
    '
