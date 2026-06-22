---
name: hf-mesh-jobs
description: Use when launching, monitoring, validating, or cleaning up Hugging Face Jobs or HF-hosted instances for mesh-llm distributed inference experiments, especially Skippy/Shard-style target-draft speculative runs that need modest GPU workers, catalog-backed model pairs, private mesh joins, or strict ownership limits in the meshllm org.
---

# HF Mesh Jobs

## Overview

Use this skill for spend-bearing Hugging Face compute that participates in a
mesh-llm test mesh. It is for validation jobs and short-lived experiments, not
for shared service administration.

## Operating Rules

- Run HF jobs under `--namespace meshllm` only when the work explicitly needs
  org-owned workers or artifacts.
- Use modest hardware by default: `t4-small` first, then `l4x1` if the model or
  build needs more VRAM/RAM. Do not use A100, H100/H200, multi-GPU, or large
  CPU flavors unless the user explicitly approves that cost.
- Only stop, cancel, delete, or restart jobs/instances created by the current
  agent task. Never operate on shared org jobs just because they are visible.
- Record every created job id in a scratch ledger such as
  `/tmp/mesh-llm-hf-jobs-<task>.jsonl` with command, namespace, flavor, start
  time, purpose, and cleanup status.
- Add ownership markers when the HF surface supports labels or environment
  variables, for example `MESH_LLM_TASK_ID`, `MESH_LLM_CREATED_BY=codex`, and a
  short purpose string.
- Pass secrets with HF secret mechanisms such as `--secrets HF_TOKEN` or a
  temporary secrets file outside the repo. Do not print raw HF tokens, mesh join
  tokens, or private service URLs in logs, committed files, or final summaries.
- Keep scratch scripts, logs, downloaded models, invite tokens, and job ledgers
  outside the repo unless the user asks for a documented artifact.

## Model Choice

Prefer catalog-backed target/draft pairs so the draft pairing is grounded in
mesh-llm's published model metadata.

Start small:

- Target: `Qwen2.5-3B-Instruct-Q4_K_M`
- Draft: `Qwen2.5-0.5B-Instruct-Q4_K_M`

Ratchet up only after small-model correctness and distributed behavior are
proven. A reasonable next catalog pair is:

- Target: `Qwen3-4B-Q4_K_M`
- Draft: `Qwen3-0.6B-Q4_K_M`

Use non-catalog models only when the experiment explicitly needs them; record why
the catalog pair was insufficient.

## Workflow

1. Inspect the mesh catalog or local catalog JSON for the target and draft refs,
   sizes, revisions, and whether a layer package exists.
2. Pick the cheapest topology that can prove the claim: usually one local M4
   coordinator plus one HF worker, or two HF workers when local networking is
   not representative enough.
3. Build or package the exact current mesh-llm code the HF worker will run. Do
   not claim an HF proof if the worker is running a stale released binary.
4. Start the coordinator and capture the private join token only into a scratch
   secret/env file outside the repo.
5. Launch HF worker jobs with a short timeout, modest hardware, `--detach`, and
   secrets passed through HF secret handling.
6. Confirm the mesh is healthy before benchmarking: peers joined, split stages
   visible, `/api/status` is coherent, and `/v1/models` shows the expected
   route/model.
7. Run deterministic OpenAI requests with `temperature = 0` first. For Shard or
   Skippy speculation, compare target-only greedy output with speculative greedy
   output token-for-token before treating performance as meaningful.
8. Capture evidence in `/tmp`: job ids, status snippets, request JSON, response
   text, token ids when available, metrics, and cleanup results.
9. Cancel/stop only the jobs recorded in the ledger for this task.

## Launch Shape

Adapt this shape instead of ad hoc shell history. Keep exact tokens out of the
command transcript.

```bash
hf jobs run \
  --namespace meshllm \
  --flavor t4-small \
  --timeout 1h \
  --secrets HF_TOKEN \
  --secrets-file /tmp/mesh-llm-hf-secrets.env \
  --env MESH_LLM_CREATED_BY=codex \
  --env MESH_LLM_TASK_ID=<task-id> \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  <image-or-command>
```

Monitor only owned jobs:

```bash
hf jobs inspect <job-id> --namespace meshllm
hf jobs logs <job-id> --namespace meshllm --tail 120
```

Cancel only owned jobs recorded in the ledger:

```bash
hf jobs cancel <job-id> --namespace meshllm
```

## Evidence Gate

For mesh-llm speculative validation, do not report success until the evidence
shows:

- exact target/draft model refs and quant files;
- topology and forced split shape;
- worker job ids and hardware flavors;
- current code revision or uploaded source artifact used by the workers;
- target-only greedy output and speculative greedy output match;
- rejection cases are exercised for speculative tree validation;
- latency/tok/s metrics come from the same prompt/model/topology as the
  correctness run;
- cleanup status for every owned HF job.
