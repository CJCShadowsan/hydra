#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE="${SOURCE:-}"
TARGET="${TARGET:-}"
TARGET_PREFIX="${TARGET_PREFIX:-BF16}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-GLM-5.2-BF16}"
EXPECTED_SPLITS="${EXPECTED_SPLITS:-306}"
WINDOW_SIZE="${WINDOW_SIZE:-1}"
MAX_MEMORY="${MAX_MEMORY:-32G}"
SPLIT_MAX_SIZE="${SPLIT_MAX_SIZE:-50G}"
STREAM_BUFFER_BYTES="${STREAM_BUFFER_BYTES:-8388608}"
WORK_ROOT="${WORK_ROOT:-}"
MANIFEST="${MANIFEST:-}"
SPOOL_DIR="${SPOOL_DIR:-}"
RECORD_DIR="${RECORD_DIR:-}"
STATUS_FILE="${STATUS_FILE:-}"
SKIPPY_QUANTIZE_BIN="${SKIPPY_QUANTIZE_BIN:-$ROOT/target/release/skippy-quantize}"
INVENTORY_VERIFIER="${INVENTORY_VERIFIER:-$ROOT/scripts/glm-dsa-inventory-verifier.py}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-}"
LOCK_DIR="${LOCK_DIR:-}"
SHARD=""
EXECUTE=0
CONFIRM_REPLACE=0
DELETE_BACKUP_AFTER_VERIFY="${DELETE_BACKUP_AFTER_VERIFY:-0}"
LOCK_HELD=0

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-bf16-rebuild-window.sh --shard N [options]

Safely rebuilds one GLM-5.2 BF16 GGUF shard using skippy-quantize's manifest
window flow. Default mode is dry-run only. Execute mode moves the stale shard to
a backup path before rebuilding, verifies the replacement, and restores the
backup if conversion or verification fails.

Options:
  --shard N                         1-based shard number to rebuild.
  --source PATH                     GLM-5.2 SafeTensors checkpoint.
  --target PATH                     BF16 GGUF repo root containing BF16/.
  --target-prefix NAME              Default: BF16.
  --output-basename NAME            Default: GLM-5.2-BF16.
  --expected-splits N               Default: 306.
  --work-root PATH                  Root for manifests/spool/records/status.
  --skippy-quantize-bin PATH        Default: target/release/skippy-quantize.
  --inventory-verifier PATH         Default: scripts/glm-dsa-inventory-verifier.py.
  --min-free-bytes BYTES            Override execute-mode free-space requirement.
  --lock-dir PATH                   Override execute-mode lock directory.
  --execute                         Move the stale shard aside and rebuild.
  --confirm-replace-stale-shard     Required with --execute.
  --confirm-delete-stale-shard      Deprecated alias for --confirm-replace-stale-shard.
  --delete-backup-after-verify      Delete the stale backup after replacement verifies.
  -h, --help                        Show this help.

Environment overrides mirror option names where practical:
  SOURCE, TARGET, WORK_ROOT, SKIPPY_QUANTIZE_BIN, INVENTORY_VERIFIER, etc.

This script does not rebuild multiple shards in one invocation. It is intended
for low-disk in-place repair where side-by-side BF16 output will not fit. Execute
mode still requires enough free space for one replacement shard while the stale
shard backup is retained.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard)
      SHARD="$2"
      shift 2
      ;;
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
    --skippy-quantize-bin)
      SKIPPY_QUANTIZE_BIN="$2"
      shift 2
      ;;
    --inventory-verifier)
      INVENTORY_VERIFIER="$2"
      shift 2
      ;;
    --min-free-bytes)
      MIN_FREE_BYTES="$2"
      shift 2
      ;;
    --lock-dir)
      LOCK_DIR="$2"
      shift 2
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    --confirm-replace-stale-shard)
      CONFIRM_REPLACE=1
      shift
      ;;
    --confirm-delete-stale-shard)
      CONFIRM_REPLACE=1
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

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
    echo "$name must be a positive integer, got: $value" >&2
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

file_size_bytes() {
  local path="$1"
  stat -f '%z' "$path" 2>/dev/null || stat -c '%s' "$path"
}

df_probe_path() {
  local path="$1"
  while [[ ! -e "$path" && "$path" != "/" ]]; do
    path="$(dirname "$path")"
  done
  printf '%s\n' "$path"
}

df_key() {
  local path
  path="$(df_probe_path "$1")"
  df -Pk "$path" | awk 'NR == 2 { print $1 ":" $6 }'
}

available_bytes() {
  local path
  path="$(df_probe_path "$1")"
  local available_kib
  available_kib="$(df -Pk "$path" | awk 'NR == 2 { print $4 }')"
  printf '%s\n' "$((available_kib * 1024))"
}

format_bytes() {
  awk -v bytes="$1" 'BEGIN {
    gib = bytes / 1024 / 1024 / 1024;
    if (gib >= 1) {
      printf "%.2f GiB", gib;
    } else {
      printf "%.2f MiB", bytes / 1024 / 1024;
    }
  }'
}

default_space_requirement() {
  local shard_bytes="$1"
  local one_gib=$((1024 * 1024 * 1024))
  if [[ "$shard_bytes" -gt 0 ]]; then
    printf '%s\n' "$((shard_bytes * 2 + one_gib))"
  else
    printf '%s\n' "$((8 * one_gib))"
  fi
}

single_fs_space_requirement() {
  local shard_bytes="$1"
  if [[ -n "$MIN_FREE_BYTES" ]]; then
    printf '%s\n' "$MIN_FREE_BYTES"
  else
    default_space_requirement "$shard_bytes"
  fi
}

split_fs_space_requirement() {
  local shard_bytes="$1"
  local one_gib=$((1024 * 1024 * 1024))
  if [[ -n "$MIN_FREE_BYTES" ]]; then
    printf '%s\n' "$MIN_FREE_BYTES"
  elif [[ "$shard_bytes" -gt 0 ]]; then
    printf '%s\n' "$((shard_bytes + one_gib))"
  else
    printf '%s\n' "$((4 * one_gib))"
  fi
}

run_space_preflight() {
  local shard_bytes=0
  if [[ -e "$target_shard" ]]; then
    shard_bytes="$(file_size_bytes "$target_shard")"
  fi

  local target_path="$TARGET/$TARGET_PREFIX"
  local target_available
  target_available="$(available_bytes "$target_path")"

  local spool_available
  spool_available="$(available_bytes "$SPOOL_DIR")"

  local target_key
  target_key="$(df_key "$target_path")"
  local spool_key
  spool_key="$(df_key "$SPOOL_DIR")"

  echo "space preflight:"
  echo "  current shard size: $(format_bytes "$shard_bytes") ($shard_bytes bytes)"
  echo "  target filesystem: $target_key"
  echo "  spool filesystem: $spool_key"

  if [[ "$target_key" == "$spool_key" ]]; then
    local required
    required="$(single_fs_space_requirement "$shard_bytes")"
    echo "  available free space: $(format_bytes "$target_available") ($target_available bytes)"
    echo "  required free space: $(format_bytes "$required") ($required bytes)"
    if [[ "$EXECUTE" == "1" && "$target_available" -lt "$required" ]]; then
      echo "not enough free space for execute-mode repair" >&2
      exit 1
    fi
  else
    local required_each
    required_each="$(split_fs_space_requirement "$shard_bytes")"
    echo "  target free space: $(format_bytes "$target_available") ($target_available bytes)"
    echo "  spool free space: $(format_bytes "$spool_available") ($spool_available bytes)"
    echo "  required free space on each filesystem: $(format_bytes "$required_each") ($required_each bytes)"
    if [[ "$EXECUTE" == "1" && ( "$target_available" -lt "$required_each" || "$spool_available" -lt "$required_each" ) ]]; then
      echo "not enough free space for execute-mode repair" >&2
      exit 1
    fi
  fi
}

acquire_lock() {
  mkdir -p "$(dirname "$LOCK_DIR")"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "repair lock is already held: $LOCK_DIR" >&2
    exit 1
  fi
  LOCK_HELD=1
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
  echo "acquired repair lock: $LOCK_DIR"
}

release_lock() {
  if [[ "$LOCK_HELD" == "1" ]]; then
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    LOCK_HELD=0
  fi
}

if [[ -z "$SHARD" ]]; then
  echo "--shard is required" >&2
  usage >&2
  exit 2
fi

require_positive_int "shard" "$SHARD"
require_positive_int "expected-splits" "$EXPECTED_SPLITS"
if [[ -n "$MIN_FREE_BYTES" ]]; then
  require_nonnegative_int "min-free-bytes" "$MIN_FREE_BYTES"
fi

if [[ "$SHARD" -gt "$EXPECTED_SPLITS" ]]; then
  echo "shard $SHARD is greater than expected split count $EXPECTED_SPLITS" >&2
  exit 1
fi

if [[ -z "$SOURCE" ]]; then
  SOURCE="$(detect_source || true)"
fi
if [[ -z "$TARGET" ]]; then
  TARGET="$(detect_target || true)"
fi
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="$(dirname "$TARGET")"
fi
if [[ -z "$MANIFEST" ]]; then
  MANIFEST="$WORK_ROOT/manifests/glm52-bf16-rebuild-window.json"
fi
if [[ -z "$SPOOL_DIR" ]]; then
  SPOOL_DIR="$WORK_ROOT/spool/bf16-rebuild-window"
fi
if [[ -z "$RECORD_DIR" ]]; then
  RECORD_DIR="$WORK_ROOT/records/bf16-rebuild-window"
fi
if [[ -z "$STATUS_FILE" ]]; then
  STATUS_FILE="$WORK_ROOT/status/bf16-rebuild-window.json"
fi
if [[ -z "$LOCK_DIR" ]]; then
  LOCK_DIR="$WORK_ROOT/locks/glm-dsa-bf16-rebuild-window.lock"
fi

require_dir "$SOURCE"
require_dir "$TARGET/$TARGET_PREFIX"
require_executable "$SKIPPY_QUANTIZE_BIN"
require_executable "$INVENTORY_VERIFIER"

split_id="$(printf '%05d' "$SHARD")"
split_total="$(printf '%05d' "$EXPECTED_SPLITS")"
target_shard="$TARGET/$TARGET_PREFIX/$OUTPUT_BASENAME-$split_id-of-$split_total.gguf"

echo "GLM-5.2 BF16 rebuild window"
echo "  mode: $([[ "$EXECUTE" == "1" ]] && echo execute || echo dry-run)"
echo "  shard: $SHARD/$EXPECTED_SPLITS"
echo "  source: $SOURCE"
echo "  target shard: $target_shard"
echo "  manifest: $MANIFEST"
echo "  spool: $SPOOL_DIR"
echo "  records: $RECORD_DIR"
echo "  status: $STATUS_FILE"
echo "  lock: $LOCK_DIR"

if [[ -e "$target_shard" ]]; then
  echo "current shard header check:"
  set +e
  "$INVENTORY_VERIFIER" \
    --checkpoint "$SOURCE" \
    --gguf "$target_shard" \
    --partial-gguf \
    --json
  verifier_status=$?
  set -e
  echo "current shard verifier status: $verifier_status"
else
  echo "current shard is missing"
fi

run_space_preflight

if [[ "$EXECUTE" != "1" ]]; then
  cat <<EOF

Dry-run only. To rebuild this shard in place:

  $0 --shard $SHARD --execute --confirm-replace-stale-shard \\
    --source "$SOURCE" \\
    --target "$TARGET" \\
    --skippy-quantize-bin "$SKIPPY_QUANTIZE_BIN"

This will move the current target shard to a backup path, rebuild the shard,
verify the replacement, and restore the backup if the rebuild fails.
EOF
  exit 0
fi

if [[ "$CONFIRM_REPLACE" != "1" ]]; then
  echo "--execute requires --confirm-replace-stale-shard" >&2
  exit 2
fi

acquire_lock
trap release_lock EXIT

for ((i = 1; i < SHARD; i++)); do
  prior="$TARGET/$TARGET_PREFIX/$OUTPUT_BASENAME-$(printf '%05d' "$i")-of-$split_total.gguf"
  if [[ ! -e "$prior" ]]; then
    echo "cannot rebuild shard $SHARD: earlier shard $i is already missing, so run-convert-window would choose it first" >&2
    exit 1
  fi
done

mkdir -p "$TARGET" "$SPOOL_DIR" "$RECORD_DIR" "$(dirname "$MANIFEST")" "$(dirname "$STATUS_FILE")"

"$SKIPPY_QUANTIZE_BIN" init-convert \
  --source "$SOURCE" \
  --target "$TARGET" \
  --target-prefix "$TARGET_PREFIX" \
  --output-basename "$OUTPUT_BASENAME" \
  --output-type bf16 \
  --expected-splits "$EXPECTED_SPLITS" \
  --window-size "$WINDOW_SIZE" \
  --manifest "$MANIFEST"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="$WORK_ROOT/stale-shard-backups/$TARGET_PREFIX"
backup_shard=""

restore_stale_backup() {
  local status=$?
  if [[ -n "$backup_shard" && -e "$backup_shard" ]]; then
    echo "rebuild failed; restoring stale shard backup: $backup_shard" >&2
    mkdir -p "$backup_dir"
    if [[ -e "$target_shard" ]]; then
      local failed_shard="$backup_dir/$(basename "$target_shard").failed.$timestamp.$$"
      mv "$target_shard" "$failed_shard" || true
      echo "kept failed replacement candidate: $failed_shard" >&2
    fi
    mv "$backup_shard" "$target_shard"
  fi
  exit "$status"
}

trap restore_stale_backup ERR

if [[ -e "$target_shard" ]]; then
  mkdir -p "$backup_dir"
  backup_shard="$backup_dir/$(basename "$target_shard").stale.$timestamp.$$"
  echo "moving stale shard to backup: $backup_shard"
  mv "$target_shard" "$backup_shard"
fi

"$SKIPPY_QUANTIZE_BIN" run-convert-window \
  --manifest "$MANIFEST" \
  --max-memory "$MAX_MEMORY" \
  --split-max-size "$SPLIT_MAX_SIZE" \
  --stream-buffer-bytes "$STREAM_BUFFER_BYTES" \
  --spool-dir "$SPOOL_DIR" \
  --record-dir "$RECORD_DIR" \
  --json-event-file "$STATUS_FILE" \
  --json-event-interval-seconds 30 \
  --json-event-window 8

if [[ ! -e "$target_shard" ]]; then
  echo "expected rebuilt shard was not published: $target_shard" >&2
  exit 1
fi

"$INVENTORY_VERIFIER" \
  --checkpoint "$SOURCE" \
  --gguf "$target_shard" \
  --partial-gguf \
  --json

trap - ERR

if [[ -n "$backup_shard" && -e "$backup_shard" ]]; then
  if [[ "$DELETE_BACKUP_AFTER_VERIFY" == "1" ]]; then
    rm -f "$backup_shard"
    echo "deleted stale backup after successful verification"
  else
    echo "kept stale backup after successful verification: $backup_shard"
  fi
fi

echo "rebuilt and verified shard: $target_shard"
