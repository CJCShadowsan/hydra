#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE="${SOURCE:-}"
TARGET="${TARGET:-}"
TARGET_PREFIX="${TARGET_PREFIX:-BF16}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-GLM-5.2-BF16}"
EXPECTED_SPLITS="${EXPECTED_SPLITS:-306}"
WORK_ROOT="${WORK_ROOT:-}"
REPORT_FILE="${REPORT_FILE:-}"
SKIPPY_QUANTIZE_BIN="${SKIPPY_QUANTIZE_BIN:-$ROOT/target/release/skippy-quantize}"
INVENTORY_VERIFIER="${INVENTORY_VERIFIER:-$ROOT/scripts/glm-dsa-inventory-verifier.py}"
REBUILD_HELPER="${REBUILD_HELPER:-$ROOT/scripts/glm-dsa-bf16-rebuild-window.sh}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-}"
ONE_SHARD_LOCK_DIR="${ONE_SHARD_LOCK_DIR:-}"
SHARDS_CSV=""
LIMIT=""
MAX_EXECUTE_SHARDS="${MAX_EXECUTE_SHARDS:-4}"
EXECUTE=0
CONFIRM_REPAIR=0
DELETE_BACKUP_AFTER_VERIFY="${DELETE_BACKUP_AFTER_VERIFY:-0}"
BATCH_LOCK_DIR="${BATCH_LOCK_DIR:-}"
BATCH_LOCK_HELD=0

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-bf16-repair-stale-shards.sh [options]

Derives stale GLM-5.2 BF16 GGUF shards from the header-only inventory verifier,
then repairs the selected shards by invoking the one-shard rebuild helper
sequentially. Default mode is dry-run only and does not modify model artifacts.

Options:
  --source PATH                     GLM-5.2 SafeTensors checkpoint.
  --target PATH                     BF16 GGUF repo root containing BF16/.
  --target-prefix NAME              Default: BF16.
  --output-basename NAME            Default: GLM-5.2-BF16.
  --expected-splits N               Default: 306.
  --work-root PATH                  Root for reports, locks, manifests, spool.
  --report-file PATH                Where to write the stale-shard report JSON.
  --skippy-quantize-bin PATH        Default: target/release/skippy-quantize.
  --inventory-verifier PATH         Default: scripts/glm-dsa-inventory-verifier.py.
  --rebuild-helper PATH             Default: scripts/glm-dsa-bf16-rebuild-window.sh.
  --shards CSV                      Repair only these stale shard numbers.
  --limit N                         Repair only the first N selected stale shards.
  --max-execute-shards N            Execute safety cap. Default: 4.
  --min-free-bytes BYTES            Pass through to one-shard repair helper.
  --one-shard-lock-dir PATH         Active one-shard repair lock to check.
  --execute                         Actually repair selected shards.
  --confirm-repair-stale-shards     Required with --execute.
  --delete-backup-after-verify      Pass through to one-shard repair helper.
  -h, --help                        Show this help.

Execute mode refuses broad repair: it requires --shards or --limit, plus the
confirmation flag, and selected shard count must not exceed --max-execute-shards.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --target-prefix)
      TARGET_PREFIX="$2"
      shift 2
      ;;
    --output-basename)
      OUTPUT_BASENAME="$2"
      shift 2
      ;;
    --expected-splits)
      EXPECTED_SPLITS="$2"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="$2"
      shift 2
      ;;
    --report-file)
      REPORT_FILE="$2"
      shift 2
      ;;
    --skippy-quantize-bin)
      SKIPPY_QUANTIZE_BIN="$2"
      shift 2
      ;;
    --inventory-verifier)
      INVENTORY_VERIFIER="$2"
      shift 2
      ;;
    --rebuild-helper)
      REBUILD_HELPER="$2"
      shift 2
      ;;
    --shards)
      SHARDS_CSV="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --max-execute-shards)
      MAX_EXECUTE_SHARDS="$2"
      shift 2
      ;;
    --min-free-bytes)
      MIN_FREE_BYTES="$2"
      shift 2
      ;;
    --one-shard-lock-dir)
      ONE_SHARD_LOCK_DIR="$2"
      shift 2
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    --confirm-repair-stale-shards)
      CONFIRM_REPAIR=1
      shift
      ;;
    --delete-backup-after-verify)
      DELETE_BACKUP_AFTER_VERIFY=1
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

detect_source() {
  for candidate in \
    "/Users/lab/models/huggingface/hub/models--zai-org--GLM-5.2/snapshots/53783022a4d492a25927417d22698a9535b743a4" \
    "/Volumes/External/models/huggingface/hub/models--zai-org--GLM-5.2/snapshots/53783022a4d492a25927417d22698a9535b743a4"
  do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

detect_target() {
  for candidate in \
    "/Users/lab/glm52-work/bf16-gguf" \
    "/Volumes/External/models/glm52-work/bf16-gguf"
  do
    if [[ -d "$candidate/$TARGET_PREFIX" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "required executable not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "required directory not found: $path" >&2
    exit 1
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer, got: $value" >&2
    exit 1
  fi
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 1
  fi
}

join_csv() {
  local first=1
  local value
  for value in "$@"; do
    if [[ "$first" == "1" ]]; then
      printf '%s' "$value"
      first=0
    else
      printf ',%s' "$value"
    fi
  done
  printf '\n'
}

acquire_batch_lock() {
  mkdir -p "$(dirname "$BATCH_LOCK_DIR")"
  if ! mkdir "$BATCH_LOCK_DIR" 2>/dev/null; then
    echo "batch repair lock is already held: $BATCH_LOCK_DIR" >&2
    exit 1
  fi
  BATCH_LOCK_HELD=1
  printf '%s\n' "$$" > "$BATCH_LOCK_DIR/pid"
  echo "acquired batch repair lock: $BATCH_LOCK_DIR"
}

release_batch_lock() {
  if [[ "$BATCH_LOCK_HELD" == "1" ]]; then
    rm -f "$BATCH_LOCK_DIR/pid"
    rmdir "$BATCH_LOCK_DIR" 2>/dev/null || true
    BATCH_LOCK_HELD=0
  fi
}

refuse_active_one_shard_repair() {
  if [[ -d "$ONE_SHARD_LOCK_DIR" ]]; then
    echo "one-shard repair lock is active: $ONE_SHARD_LOCK_DIR" >&2
    if [[ -f "$ONE_SHARD_LOCK_DIR/pid" ]]; then
      echo "one-shard repair pid: $(cat "$ONE_SHARD_LOCK_DIR/pid")" >&2
    fi
    echo "refusing to scan or batch-repair while the BF16 artifact may be temporarily missing a shard" >&2
    exit 1
  fi
}

if [[ -z "$SOURCE" ]]; then
  SOURCE="$(detect_source || true)"
fi
if [[ -z "$TARGET" ]]; then
  TARGET="$(detect_target || true)"
fi
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="$(dirname "$TARGET")"
fi
if [[ -z "$REPORT_FILE" ]]; then
  REPORT_FILE="$WORK_ROOT/status/glm-dsa-stale-shards.json"
fi
if [[ -z "$BATCH_LOCK_DIR" ]]; then
  BATCH_LOCK_DIR="$WORK_ROOT/locks/glm-dsa-bf16-repair-stale-shards.lock"
fi
if [[ -z "$ONE_SHARD_LOCK_DIR" ]]; then
  ONE_SHARD_LOCK_DIR="$WORK_ROOT/locks/glm-dsa-bf16-rebuild-window.lock"
fi

require_dir "$SOURCE"
require_dir "$TARGET/$TARGET_PREFIX"
require_executable "$INVENTORY_VERIFIER"
require_executable "$REBUILD_HELPER"
require_positive_int "expected-splits" "$EXPECTED_SPLITS"
require_positive_int "max-execute-shards" "$MAX_EXECUTE_SHARDS"
if [[ -n "$LIMIT" ]]; then
  require_positive_int "limit" "$LIMIT"
fi
if [[ -n "$MIN_FREE_BYTES" ]]; then
  require_nonnegative_int "min-free-bytes" "$MIN_FREE_BYTES"
fi
if [[ "$EXECUTE" == "1" ]]; then
  require_executable "$SKIPPY_QUANTIZE_BIN"
fi

refuse_active_one_shard_repair

mkdir -p "$(dirname "$REPORT_FILE")"

"$INVENTORY_VERIFIER" \
  --checkpoint "$SOURCE" \
  --gguf "$TARGET/$TARGET_PREFIX" \
  --stale-shard-report \
  --json > "$REPORT_FILE"

selection_file="$(mktemp "${TMPDIR:-/tmp}/glm-dsa-stale-selection.XXXXXX")"
summary_file="$(mktemp "${TMPDIR:-/tmp}/glm-dsa-stale-summary.XXXXXX")"
cleanup_selection() {
  rm -f "$selection_file" "$summary_file"
}
trap cleanup_selection EXIT

set +e
python3 - "$REPORT_FILE" "$OUTPUT_BASENAME" "$SHARDS_CSV" "$LIMIT" >"$selection_file" 2>"$summary_file" <<'PY'
import json
import re
import sys
from pathlib import Path

report_path, output_basename, shards_csv, limit_text = sys.argv[1:5]
report = json.loads(Path(report_path).read_text())["stale_shard_report"]
stale = []
for item in report["stale_files"]:
    name = Path(item["file"]).name
    match = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
    if not match:
        raise SystemExit(f"cannot parse shard number from stale file: {name}")
    stale.append(int(match.group(1)))

requested = []
if shards_csv:
    for raw in shards_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if not raw.isdigit() or int(raw) < 1:
            raise SystemExit(f"invalid shard number in --shards: {raw}")
        requested.append(int(raw))
    missing = sorted(set(requested) - set(stale))
    if missing:
        raise SystemExit(f"requested shard(s) are not stale: {missing}")
    selected = requested
else:
    selected = stale

if limit_text:
    selected = selected[: int(limit_text)]

print(
    json.dumps(
        {
            "stale_file_count": report["stale_file_count"],
            "stale_layer_count": report["stale_layer_count"],
            "missing_mtp_split_tensors": report["missing_mtp_split_tensors"],
            "contains_mtp_unsplit_kv_b": report["contains_mtp_unsplit_kv_b"],
            "selected_count": len(selected),
        },
        sort_keys=True,
    ),
    file=sys.stderr,
)
for shard in selected:
    print(shard)
PY
selection_status=$?
set -e
if [[ "$selection_status" != "0" ]]; then
  cat "$summary_file" >&2
  exit "$selection_status"
fi

selected=()
while IFS= read -r shard; do
  if [[ -n "$shard" ]]; then
    selected+=("$shard")
  fi
done < "$selection_file"

selected_count="${#selected[@]}"

echo "GLM-5.2 BF16 stale shard repair"
echo "  mode: $([[ "$EXECUTE" == "1" ]] && echo execute || echo dry-run)"
echo "  source: $SOURCE"
echo "  target: $TARGET/$TARGET_PREFIX"
echo "  report: $REPORT_FILE"
echo "  summary: $(cat "$summary_file")"
echo "  selected_count: $selected_count"
if [[ "$selected_count" -gt 0 ]]; then
  echo "  selected_shards: $(join_csv "${selected[@]}")"
fi

if [[ "$selected_count" == "0" ]]; then
  echo "no stale shards selected"
  exit 0
fi

if [[ "$EXECUTE" != "1" ]]; then
  cat <<EOF

Dry-run only. To repair selected shards:

  $0 --execute --confirm-repair-stale-shards --limit ${LIMIT:-1} \\
    --source "$SOURCE" \\
    --target "$TARGET" \\
    --skippy-quantize-bin "$SKIPPY_QUANTIZE_BIN"

Use --shards CSV for an explicit shard set. Execute mode also enforces
--max-execute-shards (currently $MAX_EXECUTE_SHARDS).
EOF
  exit 0
fi

if [[ "$CONFIRM_REPAIR" != "1" ]]; then
  echo "--execute requires --confirm-repair-stale-shards" >&2
  exit 2
fi
if [[ -z "$SHARDS_CSV" && -z "$LIMIT" ]]; then
  echo "--execute requires --shards or --limit to scope the repair batch" >&2
  exit 2
fi
if [[ "$selected_count" -gt "$MAX_EXECUTE_SHARDS" ]]; then
  echo "selected shard count $selected_count exceeds --max-execute-shards $MAX_EXECUTE_SHARDS" >&2
  exit 2
fi

acquire_batch_lock
trap 'release_batch_lock; cleanup_selection' EXIT

for shard in "${selected[@]}"; do
  args=(
    --shard "$shard"
    --source "$SOURCE"
    --target "$TARGET"
    --target-prefix "$TARGET_PREFIX"
    --output-basename "$OUTPUT_BASENAME"
    --expected-splits "$EXPECTED_SPLITS"
    --work-root "$WORK_ROOT"
    --skippy-quantize-bin "$SKIPPY_QUANTIZE_BIN"
    --inventory-verifier "$INVENTORY_VERIFIER"
    --execute
    --confirm-replace-stale-shard
  )
  if [[ -n "$MIN_FREE_BYTES" ]]; then
    args+=(--min-free-bytes "$MIN_FREE_BYTES")
  fi
  if [[ "$DELETE_BACKUP_AFTER_VERIFY" == "1" ]]; then
    args+=(--delete-backup-after-verify)
  fi

  echo "repairing stale shard $shard"
  "$REBUILD_HELPER" "${args[@]}"
done

"$INVENTORY_VERIFIER" \
  --checkpoint "$SOURCE" \
  --gguf "$TARGET/$TARGET_PREFIX" \
  --stale-shard-report \
  --json > "$REPORT_FILE"

python3 - "$REPORT_FILE" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())["stale_shard_report"]
print(f"updated_stale_file_count={report['stale_file_count']}")
print(f"updated_stale_layer_count={report['stale_layer_count']}")
print(f"updated_missing_mtp_split_tensors={report['missing_mtp_split_tensors']}")
PY
