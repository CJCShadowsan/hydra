---
name: hf-gguf-quant-jobs
description: Use when creating, monitoring, validating, or documenting low-memory Hugging Face Jobs that quantize split BF16/FP16 GGUF model repos into quantized GGUF repos, especially Skippy/layer-aware quants that use mounted model repos, tensor-type files, resumable shard windows, incremental Hub uploads, or patched llama.cpp quantization branches.
---

# HF GGUF Quant Jobs

Use this skill to turn an existing split BF16/FP16 GGUF model repo into a
quantized GGUF model repo on Hugging Face Jobs without requiring the job host to
hold the full model in memory or on local disk at once.

The proven Jianyang pattern is: mount the source GGUF repo, quantize split
windows with `llama-quantize --keep-split`, upload completed output shards to a
target model repo, delete local staged files immediately, and resume from the
already uploaded target shards after cancellation or failure.

## Preconditions

- Use a split BF16/FP16 GGUF repo as the source when possible. Do not re-read
  SafeTensors for requants if a BF16 GGUF artifact already exists.
- Verify the source repo is complete before spending on quantization. Count all
  expected split shards and refuse to run if any are missing.
- Use a tensor-type file for any custom recipe. Treat MTP tensors, output
  tensors, precision-sensitive tensors, and latency-sensitive layer ranges as
  explicit recipe inputs.
- Run jobs under the intended HF org and pass `HF_TOKEN` as a secret, not a
  printed environment variable.
- Prefer mounted Hub repos over full `hf download` when the job only needs to
  stream or stage one shard/window at a time.

## Workflow

1. Identify the source BF16/FP16 GGUF repo, target quant repo, output prefix,
   output basename, source prefix, quant type, llama.cpp branch, and
   tensor-type file.
2. Preflight both Hub and mounted source paths. Log `gguf_count`,
   `expected_total`, `missing_count`, and `first_missing`; stop if incomplete.
3. Write or upload a `quant-plan.json` with source repo/revision, target repo,
   branch, quant type, shard count, output prefix, tensor policy, and resume
   settings.
4. Launch the job with `QUANT_WINDOW_SIZE=1` for the first full model run unless
   a smaller fixture proves a larger window is safe on the chosen hardware.
5. For each split window, stage only the required input shard, run
   `llama-quantize --keep-split`, upload finished shards, then delete local
   staged input and output files.
6. Monitor for progress markers. A healthy job should repeatedly emit staged
   source drops, `llama-quantize` tensor progress, upload completion, and local
   output cache drops.
7. Validate the target repo after completion by counting GGUF shards, checking
   the first and last shard names, and confirming `quant-plan.json` plus the
   tensor-type file are present.
8. Record the artifact in the experiment card and create an iteration card for
   the run, including job id, command, environment, repo SHA, shard count, and
   follow-up decisions.

## Launch Template

Adapt this shape rather than rebuilding the job from scratch:

```bash
hf jobs uv run \
  --namespace meshllm \
  --flavor cpu-upgrade \
  --timeout 3d \
  --secrets HF_TOKEN \
  --volume hf://models/<source-repo>:/mnt/source-gguf \
  --volume hf://models/<target-repo>:/mnt/target-quant \
  --env WORK_DIR=/tmp/<job-name>-work \
  --env OUTPUT_DIR=/tmp/<job-name>-output \
  --env PRIVATE=1 \
  --env RESUME_EXISTING=1 \
  --env QUANT_WINDOW_SIZE=1 \
  --env STAGE_COPY_CHUNK_BYTES=16777216 \
  --env LLAMA_QUANTIZE_WATCHDOG_SECONDS=120 \
  --env PYTHONUNBUFFERED=1 \
  --detach \
  /path/to/quant_wrapper.py \
  -- \
  --source-repo <source-repo> \
  --source-path /mnt/source-gguf \
  --source-prefix <source-prefix> \
  --target-repo <target-repo> \
  --target-prefix <target-prefix> \
  --tensor-type-file /mnt/target-quant/<tensor-type-file> \
  --llama-branch <llama-branch> \
  --output-basename <output-basename> \
  --execute
```

Current Jianyang reference wrapper:

```text
<lab-experiments>/jianyang/hf-jobs/glm51_quant_gguf.py
```

Use it as the behavior reference for other models, but parameterize model names,
prefixes, recipes, and branch names. Do not bake GLM-specific assumptions into a
new reusable wrapper unless the model architecture truly requires them.

## Monitoring

Check status and logs:

```bash
hf jobs inspect <job-id> --namespace meshllm
hf jobs logs <job-id> --namespace meshllm --tail 120
```

Useful healthy markers:

- `source_repo_preflight ... complete=true`
- `mounted_source_preflight ... complete=true`
- `first_missing=[]`
- `quant_window ... --first-split N --last-split N`
- `quant_upload_done path=...`
- `quant_output_cache_drop ... dropped=true`
- `quant_windows_complete=true`
- `quant_manifest_upload_done repo=...`

Concerning markers:

- repeated watchdog lines with no shard, tensor, upload, or cache-drop progress;
- cgroup memory pinned near the hardware limit;
- the same split window restarting repeatedly without new uploaded target files;
- fallback quant warnings for tensors that the recipe expected to preserve.

If a job stalls, cancel it before changing code or hardware. With
`RESUME_EXISTING=1`, the next run should skip already uploaded shards and resume
at the first missing output shard.

## Validation

After completion, verify the target repo with an authenticated Hub API or CLI
check. Record at least:

- target repo and commit SHA;
- privacy setting;
- total file count;
- GGUF shard count;
- first and last shard names;
- manifest/plan presence;
- tensor-type file presence.

For local smoke tests, use a small split GGUF source first and verify:

- `llama-quantize --dry-run --keep-split` can read the output;
- `llama-gguf-split --merge --dry-run` can see the split set;
- max RSS stays bounded compared with full-model size;
- output is byte-identical when comparing a streaming/chunked path to the
  upstream equivalent on the same fixture.

## Documentation Contract

For Jianyang-style experiments, update both records:

- the main experiment card with the promoted artifact;
- a phase iteration card with the job id, exact command, environment,
  verification output, decision, and follow-ups.

Keep post-experiment upstream notes separate from the run decision. The job can
be successful while the converter or quantizer patches still need extraction
into clean upstream PRs.
