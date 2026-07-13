# Upstream Sync Workflow

This repository is a long-lived fork of `Mesh-LLM/mesh-llm`. Keep upstream
history merge-based so deployed fork branches remain stable and auditable.

## Remotes

```bash
git remote add upstream https://github.com/Mesh-LLM/mesh-llm.git
git fetch upstream
```

Expected branch roles:

- `upstream-main`: exact local mirror of `upstream/main`.
- `fork/main`: shipping fork branch.
- `sync/upstream-YYYY-MM-DD`: temporary branch for each upstream merge.

## Weekly Sync

```bash
git fetch upstream
git switch upstream-main
git merge --ff-only upstream/main
git switch fork/main
scripts/hydra/create-upstream-sync-branch.sh
```

Then resolve conflicts, run CI, and open a PR from the generated
`sync/upstream-YYYY-MM-DD` branch into `fork/main`.

Do not rebase shared fork history. Use merge commits for upstream syncs so
deployed SHAs remain stable and conflict resolution can be repeated with
`git rerere`.

## Drift Report

Run:

```bash
scripts/hydra/check-upstream-sync.sh
```

The report prints the current upstream SHA, merge base, Hydra file changes,
and upstream changes in files that are likely to conflict with this fork.
