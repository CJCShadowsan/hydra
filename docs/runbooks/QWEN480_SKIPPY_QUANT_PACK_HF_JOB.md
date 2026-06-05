# Qwen 480B Skippy Quant-Pack HF Job

This runbook builds the first Qwen Coder quant-pack candidate remotely instead
of loading the 480B source model on Studio.

## Inputs

- Source repo: `unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF`
- Source revision: `b86deeefd82f1a3374c5536dfc1dd0ce27ac092d`
- Source include: `UD-Q4_K_XL/*.gguf`
- Candidate: `ffn-compressed-attention-protected`
- Stage count: `4`
- Context shape: `8192`, `cache_type_k=f16`, `cache_type_v=f16`,
  `activation_wire_dtype=f16`
- Job image: `ghcr.io/mesh-llm/skippy-quant-pack-job:sha-0b433a3`
- Published image digest:
  `sha256:7dd65ff4c1abf851116c5ac8b788123c5e350445cc17cf24c048f8fb0459ac69`
- Output repo: `alexz-oai/qwen480-skippy-pack`

Generated local handoff artifacts live under:

```text
/Volumes/External/skippy-quant-packs/qwen3-coder-480b/hf-jobs/
```

The important files are:

- `qwen480-source-plan.json`
- `qwen480-quant-pack-workload.sh`
- `qwen480-hf-jobs-submit.json`

## Validate The Submit Payload

Before building the image or submitting remote compute, validate the generated
payload:

```bash
skippy-model-package quant-pack hf-jobs-validate \
  /Volumes/External/skippy-quant-packs/qwen3-coder-480b/hf-jobs/qwen480-hf-jobs-submit.json \
  --expected-image ghcr.io/mesh-llm/skippy-quant-pack-job:sha-0b433a3 \
  --expected-upload-repo alexz-oai/qwen480-skippy-pack
```

The validator checks the HF Jobs `run` envelope, known flavor, timeout,
detached execution, `HF_TOKEN` secret, source download, `quant-pack build-all`,
idempotent output repo creation, and upload command.
The validation report also writes an equivalent Hugging Face CLI command at
`hf_jobs_cli.shell`.

## Build And Push The Job Image

Preferred path: run the `docker` GitHub Actions workflow from this branch. It
builds and pushes:

```text
ghcr.io/mesh-llm/skippy-quant-pack-job:sha-0b433a3
```

The mutable `:cpu` tag was also pushed by workflow run
`27044374997`, but the Qwen480 submit payload should use the commit-specific
tag above.

From a shell with GitHub CLI access:

```bash
gh workflow run docker.yml --ref design/skippy-agent-quant-packs -f target=quant-pack-job
```

Local fallback, when Docker is running:

```bash
just docker-build-quant-pack-job ghcr.io/mesh-llm/skippy-quant-pack-job:sha-0b433a3
just docker-push-quant-pack-job ghcr.io/mesh-llm/skippy-quant-pack-job:sha-0b433a3
```

The image contains `skippy-model-package`, `llama-quantize`, and `hf`. Its
default command checks that all three tools are present.

## Submit The Job

Submit the generated `qwen480-hf-jobs-submit.json` payload through Hugging Face
Jobs with `HF_TOKEN` supplied as a secret. The payload is intentionally
reviewable JSON and is not submitted by `skippy-model-package`.

The payload runs:

1. `hf download` for the pinned source revision.
2. source shard discovery under `UD-Q4_K_XL/*.gguf`.
3. `skippy-model-package quant-pack build-all`.
4. `hf upload` of the generated quant-pack directory to
   `alexz-oai/qwen480-skippy-pack`.

Do not run the workload script on Studio for this model size.

## Expected Outputs

The remote job should publish candidate artifacts for
`ffn-compressed-attention-protected` to the output repo. After the job completes,
download or reference those artifacts for:

```bash
skippy-model-package preflight <package-dir> --stages 4
skippy-model-package quant-pack evidence-plan <candidate-run> \
  --hosts <stage0>,<stage1>,<stage2>,<stage3> \
  --splits <boundary0>,<boundary1>,<boundary2> \
  --ctx-size 8192 \
  --n-gpu-layers -1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --activation-wire-dtype f16
```

Certification is not complete until focused-runtime, coding-loop chat,
long-context chat, token-length, agent tool-call, and KV tool-loop evidence are
bound back to the exact generated artifacts.
