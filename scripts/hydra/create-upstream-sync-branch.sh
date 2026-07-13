#!/usr/bin/env bash
set -euo pipefail

upstream_remote="${UPSTREAM_REMOTE:-upstream}"
upstream_main_branch="${UPSTREAM_MAIN_BRANCH:-upstream-main}"
fork_branch="${FORK_BRANCH:-fork/main}"
date_suffix="${SYNC_DATE:-$(date +%F)}"
sync_branch="${SYNC_BRANCH:-sync/upstream-${date_suffix}}"

if ! git remote get-url "${upstream_remote}" >/dev/null 2>&1; then
  echo "upstream remote '${upstream_remote}' is not configured"
  echo "expected: git remote add upstream https://github.com/Mesh-LLM/mesh-llm.git"
  exit 1
fi

git config rerere.enabled true
git fetch "${upstream_remote}" main

if git show-ref --verify --quiet "refs/heads/${upstream_main_branch}"; then
  git switch "${upstream_main_branch}"
else
  git switch --create "${upstream_main_branch}" "${upstream_remote}/main"
fi
git merge --ff-only "${upstream_remote}/main"

git switch "${fork_branch}"
git switch --create "${sync_branch}"
git merge --no-ff "${upstream_main_branch}"

echo "created ${sync_branch}; resolve conflicts, run Hydra CI, and open a PR into ${fork_branch}"
