#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="$ROOT/scripts/glm-dsa-bf16-rebuild-window.sh"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/glm-dsa-bf16-rebuild-test.XXXXXX")"

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
  grep -Fq "$expected" "$path" || fail "$path did not contain: $expected"
}

assert_file_content() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || fail "expected file missing: $path"
  local content
  content="$(cat "$path")"
  [[ "$content" == "$expected" ]] || fail "unexpected content in $path: $content"
}

assert_no_lock() {
  local lock_dir="$1"
  [[ ! -e "$lock_dir" ]] || fail "lock was not released: $lock_dir"
}

write_stub_tools() {
  local tools_dir="$1"
  STUB_SKIPPY="$tools_dir/stub-skippy-quantize"
  STUB_VERIFIER="$tools_dir/stub-inventory-verifier"

  cat >"$STUB_SKIPPY" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

command="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$command" in
  init-convert)
    manifest=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --manifest)
          manifest="$2"
          shift 2
          ;;
        *)
          shift
          ;;
      esac
    done
    mkdir -p "$(dirname "$manifest")"
    printf '{"stub":true}\n' >"$manifest"
    ;;
  run-convert-window)
    mkdir -p "$(dirname "$STUB_TARGET_SHARD")"
    case "${STUB_MODE:-success}" in
      success)
        printf 'replacement-good' >"$STUB_TARGET_SHARD"
        ;;
      fail-convert)
        printf 'replacement-partial' >"$STUB_TARGET_SHARD"
        exit 7
        ;;
      verify-fail)
        printf 'replacement-bad' >"$STUB_TARGET_SHARD"
        ;;
      no-publish)
        ;;
      *)
        echo "unknown STUB_MODE: ${STUB_MODE:-}" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "unknown stub command: $command" >&2
    exit 2
    ;;
esac
EOF

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

case "$(cat "$gguf" 2>/dev/null || true)" in
  stale)
    echo '{"stub":"stale"}'
    exit 1
    ;;
  replacement-good)
    echo '{"stub":"ok"}'
    ;;
  replacement-bad)
    echo '{"stub":"bad"}' >&2
    exit 9
    ;;
  replacement-partial)
    echo '{"stub":"partial"}' >&2
    exit 8
    ;;
  *)
    echo '{"stub":"unknown"}' >&2
    exit 10
    ;;
esac
EOF

  chmod +x "$STUB_SKIPPY" "$STUB_VERIFIER"
}

make_fixture() {
  local name="$1"
  FIXTURE_ROOT="$TMP_ROOT/$name"
  SOURCE_DIR="$FIXTURE_ROOT/source"
  TARGET_DIR="$FIXTURE_ROOT/target"
  WORK_DIR="$FIXTURE_ROOT/work"
  mkdir -p "$SOURCE_DIR" "$TARGET_DIR/BF16" "$WORK_DIR"
  TARGET_SHARD="$TARGET_DIR/BF16/Test-BF16-00001-of-00003.gguf"
  LOCK_DIR="$WORK_DIR/locks/glm-dsa-bf16-rebuild-window.lock"
  printf 'stale' >"$TARGET_SHARD"
}

run_helper() {
  STUB_TARGET_SHARD="$TARGET_SHARD" \
    "$HELPER" \
      --shard 1 \
      --source "$SOURCE_DIR" \
      --target "$TARGET_DIR" \
      --work-root "$WORK_DIR" \
      --expected-splits 3 \
      --output-basename Test-BF16 \
      --skippy-quantize-bin "$STUB_SKIPPY" \
      --inventory-verifier "$STUB_VERIFIER" \
      --min-free-bytes 0 \
      "$@"
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

tools_dir="$TMP_ROOT/tools"
mkdir -p "$tools_dir"
write_stub_tools "$tools_dir"

make_fixture dry_run
expect_status 0 "$FIXTURE_ROOT/dry-run.log" run_helper
assert_contains "$FIXTURE_ROOT/dry-run.log" "Dry-run only"
assert_file_content "$TARGET_SHARD" stale
[[ ! -d "$WORK_DIR/stale-shard-backups" ]] || fail "dry-run created a backup"

make_fixture execute_success
STUB_MODE=success expect_status 0 "$FIXTURE_ROOT/success.log" \
  run_helper --execute --confirm-replace-stale-shard
assert_contains "$FIXTURE_ROOT/success.log" "rebuilt and verified shard"
assert_file_content "$TARGET_SHARD" replacement-good
backup_file="$(find "$WORK_DIR/stale-shard-backups" -type f -name '*.stale.*' -print -quit)"
[[ -n "$backup_file" ]] || fail "success path did not keep stale backup"
assert_file_content "$backup_file" stale
assert_no_lock "$LOCK_DIR"

make_fixture convert_failure
STUB_MODE=fail-convert expect_status 7 "$FIXTURE_ROOT/convert-failure.log" \
  run_helper --execute --confirm-replace-stale-shard
assert_contains "$FIXTURE_ROOT/convert-failure.log" "rebuild failed; restoring stale shard backup"
assert_file_content "$TARGET_SHARD" stale
failed_file="$(find "$WORK_DIR/stale-shard-backups" -type f -name '*.failed.*' -print -quit)"
[[ -n "$failed_file" ]] || fail "conversion failure did not keep failed candidate"
assert_file_content "$failed_file" replacement-partial
assert_no_lock "$LOCK_DIR"

make_fixture verify_failure
STUB_MODE=verify-fail expect_status 9 "$FIXTURE_ROOT/verify-failure.log" \
  run_helper --execute --confirm-replace-stale-shard
assert_contains "$FIXTURE_ROOT/verify-failure.log" "rebuild failed; restoring stale shard backup"
assert_file_content "$TARGET_SHARD" stale
failed_file="$(find "$WORK_DIR/stale-shard-backups" -type f -name '*.failed.*' -print -quit)"
[[ -n "$failed_file" ]] || fail "verification failure did not keep failed candidate"
assert_file_content "$failed_file" replacement-bad
assert_no_lock "$LOCK_DIR"

make_fixture lock_refusal
mkdir -p "$LOCK_DIR"
STUB_MODE=success expect_status 1 "$FIXTURE_ROOT/lock-refusal.log" \
  run_helper --execute --confirm-replace-stale-shard
assert_contains "$FIXTURE_ROOT/lock-refusal.log" "repair lock is already held"
assert_file_content "$TARGET_SHARD" stale
[[ -d "$LOCK_DIR" ]] || fail "pre-existing lock should not be removed"

echo "GLM-DSA BF16 rebuild helper fixture tests passed"
