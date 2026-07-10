#!/usr/bin/env bash
set -euo pipefail

output="${1:-/Users/lab/models/skippy-runtime-bench/glm52-multi-session-$(date -u +%Y%m%dT%H%M%SZ).json}"
root="${ROOT:-/Users/lab/src/mesh-llm-codex}"
base_url="${BASE_URL:-http://127.0.0.1:9337/v1}"
model="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
corpus="${PROMPT_CORPUS:-crates/skippy-bench/corpora/speculative_coding_prompts.jsonl}"
prompt_limit="${PROMPT_LIMIT:-8}"
max_tokens="${MAX_TOKENS:-64}"
concurrency="${CONCURRENCY_DEPTH:-8}"
session_prefix="${SESSION_PREFIX:-glm52-multi-session}"

host_name="$(hostname -s | tr '[:upper:]' '[:lower:]')"
if [[ "$host_name" != micstudio* ]]; then
  echo "benchmark must run on micstudio, not $host_name" >&2
  exit 1
fi

cd "$root"
mkdir -p "$(dirname "$output")"

target/release/skippy-bench chat-corpus \
  --base-url "$base_url" \
  --model "$model" \
  --prompt-corpus "$corpus" \
  --prompt-limit "$prompt_limit" \
  --max-tokens "$max_tokens" \
  --concurrency-depth "$concurrency" \
  --stream \
  --include-usage true \
  --request-timeout-secs 1800 \
  --session-prefix "$session_prefix" \
  --temperature 0 \
  --top-p 1 \
  --top-k 1 \
  --seed 42 \
  --enable-thinking false \
  --output "$output"

jq '{
  request_count,
  concurrency_depth,
  completion_tokens: .summary.completion_tokens,
  total_wall_ms: .summary.total_wall_ms,
  completion_tok_s: .summary.completion_tok_s,
  ttft_ms_p50: .summary.ttft_ms_p50,
  ttft_ms_p95: .summary.ttft_ms_p95,
  errors: .summary.errors,
  result_completion_tokens: [.results[].completion_tokens],
  result_elapsed_ms: [.results[].elapsed_ms],
  result_ttft_ms: [.results[].ttft_ms]
}' "$output"

printf 'output=%s\n' "$output"
