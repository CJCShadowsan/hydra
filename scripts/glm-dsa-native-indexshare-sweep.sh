#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-native-indexshare-sweep.sh --stage-model PATH --output-dir DIR [options]

Runs GLM-DSA native Full->Shared IndexShare proof windows and writes an
aggregate summary.json. Each window runs the existing
glm-dsa-native-indexshare-proof.sh helper and stores its own compare.json,
command.sh, wrapper.log, and exit.code.

Options:
  --stage-model PATH       Layer package directory containing model-package.json.
  --output-dir DIR         Directory for per-window artifacts and summary.json.
  --windows CSV            Explicit Full producer layers, e.g. 6,10,14.
  --max-windows N          Limit discovered windows after package discovery.
  --window-size N          Exclusive span size. Default: 4 for GLM-DSA top_k_frequency=4.
  --iterations N           Measured iterations per window. Default: 1.
  --warmup N               Warmup iterations per window. Default: 0.
  --ctx-size N             Context size. Default: 131072.
  --tokens N               Decode tokens. Default: 1.
  --position-start N       Decode position. Default: 4096.
  --kv-warmup-tokens N     KV warmup tokens. Default: 4096.
  --n-batch N              llama n_batch. Default: 512.
  --n-ubatch N             llama n_ubatch. Default: 512.
  --bench-bin PATH         skippy-bench binary. Default: target/debug/skippy-bench.
  --skip-poison            Skip poisoned-sideband sensitivity rerun.
  --dry-run                Print discovered commands without running them.
  -h, --help               Show this help.
EOF
}

stage_model=""
output_dir=""
windows_csv=""
max_windows=""
window_size=4
iterations=1
warmup=0
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
    --windows)
      windows_csv="${2:?missing --windows value}"
      shift 2
      ;;
    --max-windows)
      max_windows="${2:?missing --max-windows value}"
      shift 2
      ;;
    --window-size)
      window_size="${2:?missing --window-size value}"
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

if [[ ! -f "$stage_model/model-package.json" ]]; then
  echo "missing model-package.json under stage model: $stage_model" >&2
  exit 1
fi

if [[ ! -x "$bench_bin" ]]; then
  echo "skippy-bench binary is not executable: $bench_bin" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

if [[ ! -x scripts/glm-dsa-native-indexshare-proof.sh ]]; then
  echo "missing executable scripts/glm-dsa-native-indexshare-proof.sh" >&2
  exit 1
fi

max_layer="$(jq -r '[.layers[].layer_index // empty] | max' "$stage_model/model-package.json")"
last_exclusive="$((max_layer + 1))"

windows=()
if [[ -n "$windows_csv" ]]; then
  IFS=',' read -r -a windows <<<"$windows_csv"
else
  while IFS= read -r layer; do
    if [[ "$((layer + window_size))" -le "$last_exclusive" ]]; then
      windows+=("$layer")
    fi
  done < <(
    jq -r '
      .layers[]
      | select(.layer_index != null)
      | select((.tensor_count // 0) > 18)
      | .layer_index
    ' "$stage_model/model-package.json"
  )
fi

if [[ -n "$max_windows" ]]; then
  windows=("${windows[@]:0:max_windows}")
fi

if [[ ${#windows[@]} -eq 0 ]]; then
  echo "no GLM-DSA Full producer windows selected" >&2
  exit 1
fi

mkdir -p "$output_dir"
printf '%s\n' "${windows[@]}" >"$output_dir/windows.txt"

for layer_start in "${windows[@]}"; do
  layer_end="$((layer_start + window_size))"
  window_dir="$output_dir/window-${layer_start}-${layer_end}"
  mkdir -p "$window_dir"
  cmd=(
    scripts/glm-dsa-native-indexshare-proof.sh
    --stage-model "$stage_model"
    --output-dir "$window_dir"
    --layer-start "$layer_start"
    --layer-end "$layer_end"
    --iterations "$iterations"
    --warmup "$warmup"
    --ctx-size "$ctx_size"
    --tokens "$tokens"
    --position-start "$position_start"
    --kv-warmup-tokens "$kv_warmup_tokens"
    --n-batch "$n_batch"
    --n-ubatch "$n_ubatch"
    --bench-bin "$bench_bin"
  )
  if [[ "$skip_poison" == 1 ]]; then
    cmd+=(--skip-poison)
  fi
  printf '%q ' "${cmd[@]}" >"$window_dir/sweep-command.sh"
  printf '\n' >>"$window_dir/sweep-command.sh"
  echo "window $layer_start..$layer_end -> $window_dir"
  if [[ "$dry_run" == 0 ]]; then
    set +e
    "${cmd[@]}" >"$window_dir/wrapper.log" 2>&1
    status=$?
    set -e
    echo "$status" >"$window_dir/exit.code"
  fi
done

if [[ "$dry_run" == 1 ]]; then
  echo "dry run complete: $output_dir"
  exit 0
fi

tmp_rows="$output_dir/.rows.jsonl"
: >"$tmp_rows"
for layer_start in "${windows[@]}"; do
  layer_end="$((layer_start + window_size))"
  window_dir="$output_dir/window-${layer_start}-${layer_end}"
  exit_code="missing"
  if [[ -f "$window_dir/exit.code" ]]; then
    exit_code="$(cat "$window_dir/exit.code")"
  fi
  if [[ -f "$window_dir/compare.json" ]]; then
    jq -c \
      --argjson layer_start "$layer_start" \
      --argjson layer_end "$layer_end" \
      --arg window_dir "$window_dir" \
      --arg exit_code "$exit_code" '
      .comparison as $c
      | .native_indexshare_guard as $g
      | .comparison.baseline.indexshare_timing_summary as $b
      | .comparison.candidate.indexshare_timing_summary as $cand
      | {
          layer_start: $layer_start,
          layer_end: $layer_end,
          window_dir: $window_dir,
          exit_code: $exit_code,
          report_present: true,
          baseline_ms: $c.baseline.timing_summary.mean_ms,
          candidate_ms: $c.candidate.timing_summary.mean_ms,
          win_pct: ((($c.baseline.timing_summary.mean_ms - $c.candidate.timing_summary.mean_ms) / $c.baseline.timing_summary.mean_ms) * 100.0),
          parity_passed: $c.parity.passed,
          hidden_mismatches: $c.parity.hidden_mismatches,
          poisoned_parity_passed: ($c.poisoned_parity.passed // null),
          poisoned_hidden_mismatches: ($c.poisoned_parity.hidden_mismatches // null),
          sideband_sensitivity_passed: ($c.sideband_sensitivity.passed // null),
          guard_passed: $g.passed,
          shared_exec_missing_input_top_k: $g.shared_exec_missing_input_top_k,
          full_layers: $g.full_layers,
          shared_layers: $g.shared_layers,
          consume_records: $g.consume_records,
          baseline_indexer_topk_us: $b.indexer_topk_us,
          candidate_indexer_topk_us: $cand.indexer_topk_us,
          avoided_indexer_topk_us: ($b.indexer_topk_us - $cand.indexer_topk_us),
          baseline_consumer_us: $b.consumer_total_us,
          candidate_consumer_us: $cand.consumer_total_us
        }
    ' "$window_dir/compare.json" >>"$tmp_rows"
  else
    jq -n -c \
      --argjson layer_start "$layer_start" \
      --argjson layer_end "$layer_end" \
      --arg window_dir "$window_dir" \
      --arg exit_code "$exit_code" '
      {
        layer_start: $layer_start,
        layer_end: $layer_end,
        window_dir: $window_dir,
        exit_code: $exit_code,
        report_present: false
      }
    ' >>"$tmp_rows"
  fi
done

jq -s '
  def mean($xs): if ($xs | length) == 0 then null else (($xs | add) / ($xs | length)) end;
  . as $windows
  | ($windows | map(select(.report_present == true))) as $reported
  | {
      window_count: ($windows | length),
      report_count: ($reported | length),
      failure_count: ($windows | map(select(.report_present != true or .exit_code != "0")) | length),
      parity_passed: (all($reported[]; .parity_passed == true)),
      guard_passed: (all($reported[]; .guard_passed == true)),
      sideband_sensitivity_passed: (
        if ($reported | map(select(.sideband_sensitivity_passed != null)) | length) == 0
        then null
        else all($reported[]; (.sideband_sensitivity_passed // true) == true)
        end
      ),
      hidden_mismatches: ($reported | map(.hidden_mismatches // 0) | add),
      shared_exec_missing_input_top_k: ($reported | map(.shared_exec_missing_input_top_k // 0) | add),
      total_baseline_ms: ($reported | map(.baseline_ms // 0) | add),
      total_candidate_ms: ($reported | map(.candidate_ms // 0) | add),
      total_win_pct: (
        if ($reported | length) == 0 then null
        else ((($reported | map(.baseline_ms // 0) | add) - ($reported | map(.candidate_ms // 0) | add)) / ($reported | map(.baseline_ms // 0) | add) * 100.0)
        end
      ),
      total_avoided_indexer_topk_us: ($reported | map(.avoided_indexer_topk_us // 0) | add),
      mean_win_pct: mean($reported | map(.win_pct)),
      windows: ($windows | sort_by(.layer_start))
    }
' "$tmp_rows" >"$output_dir/summary.json"
rm -f "$tmp_rows"

jq -r '
  "windows=\(.window_count) reports=\(.report_count) failures=\(.failure_count) parity=\(.parity_passed) guard=\(.guard_passed) sensitivity=\(.sideband_sensitivity_passed) missing_topk=\(.shared_exec_missing_input_top_k) total_win_pct=\(.total_win_pct) avoided_indexer_topk_us=\(.total_avoided_indexer_topk_us)"
' "$output_dir/summary.json"

echo "$output_dir"
