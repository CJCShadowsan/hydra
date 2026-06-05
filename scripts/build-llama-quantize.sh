#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-${LLAMA_BACKEND:-}}}" ]]; then
  case "$(uname -s)" in
    Darwin)
      export LLAMA_STAGE_BACKEND=metal
      ;;
    *)
      export LLAMA_STAGE_BACKEND=cpu
      ;;
  esac
fi

"$ROOT/scripts/build-llama.sh"

LLAMA_BUILD_DIR="$("$ROOT/scripts/build-llama.sh" --print-build-dir)"
cmake --build "$LLAMA_BUILD_DIR" --config "${CMAKE_BUILD_TYPE:-Release}" --target llama-quantize

if [[ -x "$LLAMA_BUILD_DIR/bin/llama-quantize" ]]; then
  printf '%s\n' "$LLAMA_BUILD_DIR/bin/llama-quantize"
else
  found="$(find "$LLAMA_BUILD_DIR" -type f -perm +111 -name llama-quantize -print -quit)"
  if [[ -z "$found" ]]; then
    echo "llama-quantize target built but binary was not found under $LLAMA_BUILD_DIR" >&2
    exit 1
  fi
  printf '%s\n' "$found"
fi
