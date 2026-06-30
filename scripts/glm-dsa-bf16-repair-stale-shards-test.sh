#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER="$ROOT/scripts/glm-dsa-bf16-repair-stale-shards.sh"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/glm-dsa-bf16-repair-test.XXXXXX")"

cleanup() {
  rm -rf "$TMP_ROOT"
}

trap cleanup EXIT

fail() {
  echo "error: $*" >&2
  exit 1
}

assert_contains() {
  local path="$1"
  local expected="$2"
  grep -Fq -- "$expected" "$path" || fail "$path did not contain: $expected"
}

assert_file_content() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || fail "expected file missing: $path"
  local content
  content="$(cat "$path")"
  [[ "$content" == "$expected" ]] || fail "unexpected content in $path: $content"
}

expect_status() {
  local expected="$1"
  local log="$2"
  shift 2
  set +e
  "$@" >"$log" 2>&1
  local actual=$?
  set -e
  [[ "$actual" == "$expected" ]] || {
    cat "$log" >&2
    fail "expected status $expected, got $actual"
  }
}

write_stub_tools() {
  local tools_dir="$1"
  STUB_VERIFIER="$tools_dir/stub-inventory-verifier"
  STUB_REBUILD="$tools_dir/stub-rebuild-helper"
  STUB_SKIPPY="$tools_dir/stub-skippy-quantize"

  cat >"$STUB_VERIFIER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

gguf=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf)
      gguf="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

python3 - "$gguf" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
stale_files = []
for path in sorted(root.glob("*.gguf")):
    if path.read_text().strip() != "stale":
        continue
    match = re.search(r"-(\d{5})-of-\d{5}\.gguf$", path.name)
    shard = int(match.group(1))
    stale_files.append(
        {
            "file": str(path),
            "unsplit_kv_b_count": 1,
            "layers": [shard],
            "examples": [f"blk.{shard}.attn_kv_b.weight"],
        }
    )

print(
    json.dumps(
        {
            "checkpoint": {"target_layers": 78, "nextn_layers": 1},
            "stale_shard_report": {
                "gguf_files": len(list(root.glob("*.gguf"))),
                "gguf_tensors": 0,
                "metadata_missing": [],
                "block_count": 79,
                "expected_block_count": 79,
                "stale_file_count": len(stale_files),
                "stale_layer_count": len(stale_files),
                "stale_layers": [item["layers"][0] for item in stale_files],
                "stale_files": stale_files,
                "missing_mtp_split_tensors": [],
                "contains_mtp_unsplit_kv_b": False,
            },
        }
    )
)
PY
EOF

  cat >"$STUB_REBUILD" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

shard=""
target=""
target_prefix="BF16"
output_basename="Test-BF16"
expected_splits="5"
execute=0
confirm=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard)
      shard="$2"; shift 2 ;;
    --target)
      target="$2"; shift 2 ;;
    --target-prefix)
      target_prefix="$2"; shift 2 ;;
    --output-basename)
      output_basename="$2"; shift 2 ;;
    --expected-splits)
      expected_splits="$2"; shift 2 ;;
    --execute)
      execute=1; shift ;;
    --confirm-replace-stale-shard)
      confirm=1; shift ;;
    *)
      shift ;;
  esac
done

if [[ "$execute" != "1" || "$confirm" != "1" ]]; then
  echo "stub rebuild requires execute confirmation" >&2
  exit 2
fi

split_id="$(printf '%05d' "$shard")"
split_total="$(printf '%05d' "$expected_splits")"
path="$target/$target_prefix/$output_basename-$split_id-of-$split_total.gguf"
printf 'repaired' >"$path"
echo "stub repaired shard $shard"
EOF

  cat >"$STUB_SKIPPY" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

  chmod +x "$STUB_VERIFIER" "$STUB_REBUILD" "$STUB_SKIPPY"
}

make_fixture() {
  FIXTURE_ROOT="$TMP_ROOT/$1"
  SOURCE_DIR="$FIXTURE_ROOT/source"
  TARGET_DIR="$FIXTURE_ROOT/target"
  WORK_DIR="$FIXTURE_ROOT/work"
  mkdir -p "$SOURCE_DIR" "$TARGET_DIR/BF16" "$WORK_DIR"
  local shard
  for shard in 1 2 3 4 5; do
    local split_id
    split_id="$(printf '%05d' "$shard")"
    if [[ "$shard" == "2" || "$shard" == "4" || "$shard" == "5" ]]; then
      printf 'stale' >"$TARGET_DIR/BF16/Test-BF16-$split_id-of-00005.gguf"
    else
      printf 'clean' >"$TARGET_DIR/BF16/Test-BF16-$split_id-of-00005.gguf"
    fi
  done
}

run_driver() {
  "$DRIVER" \
    --source "$SOURCE_DIR" \
    --target "$TARGET_DIR" \
    --target-prefix BF16 \
    --output-basename Test-BF16 \
    --expected-splits 5 \
    --work-root "$WORK_DIR" \
    --inventory-verifier "$STUB_VERIFIER" \
    --rebuild-helper "$STUB_REBUILD" \
    --skippy-quantize-bin "$STUB_SKIPPY" \
    "$@"
}

tools_dir="$TMP_ROOT/tools"
mkdir -p "$tools_dir"
write_stub_tools "$tools_dir"

make_fixture dry_run
expect_status 0 "$FIXTURE_ROOT/dry-run.log" run_driver
assert_contains "$FIXTURE_ROOT/dry-run.log" "mode: dry-run"
assert_contains "$FIXTURE_ROOT/dry-run.log" "selected_shards: 2,4,5"
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00002-of-00005.gguf" stale

make_fixture active_one_shard_lock
mkdir -p "$WORK_DIR/locks/glm-dsa-bf16-rebuild-window.lock"
printf '12345\n' >"$WORK_DIR/locks/glm-dsa-bf16-rebuild-window.lock/pid"
expect_status 1 "$FIXTURE_ROOT/active-one-shard-lock.log" run_driver
assert_contains "$FIXTURE_ROOT/active-one-shard-lock.log" "one-shard repair lock is active"
assert_contains "$FIXTURE_ROOT/active-one-shard-lock.log" "one-shard repair pid: 12345"
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00002-of-00005.gguf" stale

make_fixture execute_unscoped
expect_status 2 "$FIXTURE_ROOT/execute-unscoped.log" \
  run_driver --execute --confirm-repair-stale-shards
assert_contains "$FIXTURE_ROOT/execute-unscoped.log" "--execute requires --shards or --limit"
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00002-of-00005.gguf" stale

make_fixture execute_limited
expect_status 0 "$FIXTURE_ROOT/execute-limited.log" \
  run_driver --execute --confirm-repair-stale-shards --limit 2
assert_contains "$FIXTURE_ROOT/execute-limited.log" "updated_stale_file_count=1"
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00002-of-00005.gguf" repaired
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00004-of-00005.gguf" repaired
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00005-of-00005.gguf" stale

make_fixture execute_explicit
expect_status 0 "$FIXTURE_ROOT/execute-explicit.log" \
  run_driver --execute --confirm-repair-stale-shards --shards 5
assert_contains "$FIXTURE_ROOT/execute-explicit.log" "updated_stale_file_count=2"
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00002-of-00005.gguf" stale
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00004-of-00005.gguf" stale
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00005-of-00005.gguf" repaired

make_fixture execute_requested_clean
expect_status 1 "$FIXTURE_ROOT/execute-requested-clean.log" run_driver --shards 1
assert_contains "$FIXTURE_ROOT/execute-requested-clean.log" "requested shard(s) are not stale"
assert_file_content "$TARGET_DIR/BF16/Test-BF16-00001-of-00005.gguf" clean

echo "GLM-DSA BF16 stale shard repair fixture tests passed"
