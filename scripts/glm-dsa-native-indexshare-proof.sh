#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-native-indexshare-proof.sh --stage-model PATH --output-dir DIR [options]

Runs the GLM-DSA native Full->Shared IndexShare producer/consumer proof using
skippy-bench. The baseline runs the full span; the candidate runs the target
Shared span from a real top-k sideband produced by the source layer. By default
the benchmark also poisons the sideband and requires the poisoned run to fail
parity, proving the consumer actually used the sideband.

Options:
  --stage-model PATH       Layer package directory containing model-package.json.
  --output-dir DIR         Directory for compare.json.
  --layer-start N          Source Full producer layer. Default: 6.
  --layer-end N            Exclusive end layer. Default: layer-start + 2.
  --iterations N           Measured iterations. Default: 3.
  --warmup N               Warmup iterations. Default: 1.
  --ctx-size N             Context size. Default: 131072.
  --tokens N               Decode tokens. Default: 1.
  --position-start N       Decode position. Default: 4096.
  --kv-warmup-tokens N     KV warmup tokens. Default: 4096.
  --n-batch N              llama n_batch. Default: 512.
  --n-ubatch N             llama n_ubatch. Default: 512.
  --bench-bin PATH         skippy-bench binary. Default: target/debug/skippy-bench.
  --skip-poison            Skip poisoned-sideband sensitivity rerun.
  --dry-run                Print command without running it.
  -h, --help               Show this help.
EOF
}

stage_model=""
output_dir=""
layer_start=6
layer_end=""
iterations=3
warmup=1
ctx_size=131072
tokens=1
position_start=4096
kv_warmup_tokens=4096
n_batch=512
n_ubatch=512
bench_bin="target/debug/skippy-bench"
skip_poison=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-model)
      stage_model="${2:?missing --stage-model value}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?missing --output-dir value}"
      shift 2
      ;;
    --layer-start)
      layer_start="${2:?missing --layer-start value}"
      shift 2
      ;;
    --layer-end)
      layer_end="${2:?missing --layer-end value}"
      shift 2
      ;;
    --iterations)
      iterations="${2:?missing --iterations value}"
      shift 2
      ;;
    --warmup)
      warmup="${2:?missing --warmup value}"
      shift 2
      ;;
    --ctx-size)
      ctx_size="${2:?missing --ctx-size value}"
      shift 2
      ;;
    --tokens)
      tokens="${2:?missing --tokens value}"
      shift 2
      ;;
    --position-start)
      position_start="${2:?missing --position-start value}"
      shift 2
      ;;
    --kv-warmup-tokens)
      kv_warmup_tokens="${2:?missing --kv-warmup-tokens value}"
      shift 2
      ;;
    --n-batch)
      n_batch="${2:?missing --n-batch value}"
      shift 2
      ;;
    --n-ubatch)
      n_ubatch="${2:?missing --n-ubatch value}"
      shift 2
      ;;
    --bench-bin)
      bench_bin="${2:?missing --bench-bin value}"
      shift 2
      ;;
    --skip-poison)
      skip_poison=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$stage_model" || -z "$output_dir" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$layer_end" ]]; then
  layer_end="$((layer_start + 2))"
fi

if [[ "$layer_end" -le "$((layer_start + 1))" ]]; then
  echo "--layer-end must be at least layer-start + 2 for a producer/consumer proof" >&2
  exit 2
fi

if [[ ! -f "$stage_model/model-package.json" ]]; then
  echo "missing model-package.json under stage model: $stage_model" >&2
  exit 1
fi

if [[ ! -x "$bench_bin" ]]; then
  echo "skippy-bench binary is not executable: $bench_bin" >&2
  exit 1
fi

mkdir -p "$output_dir"

build_dir="${LLAMA_STAGE_BUILD_DIR:-}"
if [[ -z "$build_dir" ]]; then
  build_dir="$(LLAMA_STAGE_BACKEND=metal scripts/build-llama.sh --print-build-dir)"
fi

cmd=(
  "$bench_bin" glm-dsa-layer-microbench
  --stage-model "$stage_model"
  --layer-start "$layer_start"
  --layer-end "$layer_end"
  --ctx-size "$ctx_size"
  --tokens "$tokens"
  --position-start "$position_start"
  --kv-warmup-tokens "$kv_warmup_tokens"
  --iterations "$iterations"
  --warmup "$warmup"
  --n-batch "$n_batch"
  --n-ubatch "$n_ubatch"
  --direct-sparse-attn true
  --compact-flash-attn true
  --allow-compact-flash-auto
  --direct-sparse-prefill true
  --fused-sparse-mask true
  --metal-topk-moe-route-fusion true
  --op-timing true
  --metal-dispatch-log true
  --compare-native-indexshare-producer-consumer
  --require-native-indexshare-proof
  --output "$output_dir/compare.json"
)

if [[ "$skip_poison" == 1 ]]; then
  cmd+=(--skip-native-indexshare-poison)
fi

printf '%q ' "${cmd[@]}" >"$output_dir/command.sh"
printf '\n' >>"$output_dir/command.sh"

if [[ "$dry_run" == 1 ]]; then
  cat "$output_dir/command.sh"
  exit 0
fi

LLAMA_STAGE_BACKEND=metal LLAMA_STAGE_BUILD_DIR="$build_dir" "${cmd[@]}"

if command -v jq >/dev/null 2>&1; then
  jq -r '
    .comparison as $c
    | "baseline_ms=\($c.baseline.timing_summary.mean_ms) candidate_ms=\($c.candidate.timing_summary.mean_ms) win_pct=\((($c.baseline.timing_summary.mean_ms - $c.candidate.timing_summary.mean_ms) / $c.baseline.timing_summary.mean_ms * 100)) parity=\($c.parity.passed) poisoned_parity=\($c.poisoned_parity.passed) sensitivity=\($c.sideband_sensitivity.passed) hidden_mismatches=\($c.parity.hidden_mismatches)"
  ' "$output_dir/compare.json"
fi

echo "$output_dir"
