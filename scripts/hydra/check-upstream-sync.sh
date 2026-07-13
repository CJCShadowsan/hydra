#!/usr/bin/env bash
set -euo pipefail

upstream_remote="${UPSTREAM_REMOTE:-upstream}"
upstream_ref="${UPSTREAM_REF:-${upstream_remote}/main}"
base_ref="${BASE_REF:-HEAD}"

if ! git remote get-url "${upstream_remote}" >/dev/null 2>&1; then
  echo "upstream remote '${upstream_remote}' is not configured"
  echo "expected: git remote add upstream https://github.com/Mesh-LLM/mesh-llm.git"
  exit 1
fi

git fetch "${upstream_remote}" main --quiet

upstream_sha="$(git rev-parse "${upstream_ref}")"
base_sha="$(git rev-parse "${base_ref}")"
merge_base="$(git merge-base "${base_ref}" "${upstream_ref}")"

echo "Hydra upstream drift report"
echo "base_ref=${base_ref}"
echo "base_sha=${base_sha}"
echo "upstream_ref=${upstream_ref}"
echo "upstream_sha=${upstream_sha}"
echo "merge_base=${merge_base}"
echo

echo "Hydra-owned files changed since merge base:"
git diff --name-only "${merge_base}..${base_ref}" -- \
  crates/hydra \
  docs/hydra \
  scripts/hydra \
  .github/workflows/hydra-upstream-drift.yml \
  || true
echo

echo "upstream changes in Hydra integration files:"
git diff --name-only "${merge_base}..${upstream_ref}" -- \
  Cargo.toml \
  crates/mesh-llm/Cargo.toml \
  crates/mesh-llm/src/commands/mod.rs \
  crates/mesh-llm-cli/src/parser.rs \
  crates/mesh-llm-config/src/model.rs \
  crates/mesh-llm-config/src/lib.rs \
  crates/mesh-llm-host-runtime/Cargo.toml \
  crates/mesh-llm-host-runtime/src/api/mod.rs \
  crates/mesh-llm-host-runtime/src/api/routes/mod.rs \
  crates/mesh-llm-host-runtime/src/api/status.rs \
  crates/mesh-llm-host-runtime/src/api/state.rs \
  crates/mesh-llm-host-runtime/src/mesh/mod.rs \
  crates/mesh-llm-host-runtime/src/network/openai/transport.rs \
  crates/mesh-llm-host-runtime/src/runtime_data/api_views.rs \
  crates/mesh-llm-host-runtime/src/runtime_data/mod.rs \
  || true
