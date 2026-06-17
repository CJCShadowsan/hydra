#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.0.0",
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.30.0",
#   "numpy",
#   "pyarrow",
#   "setproctitle",
#   "torch>=2.8.0",
#   "tqdm",
#   "transformers>=5.6.0",
# ]
# ///
"""Train and evaluate an SPD speculation head on Hugging Face Jobs or locally.

This is intentionally a proof runner, not serving code. It produces a real
`speculation_head_final.pt` from the reference SPD implementation, evaluates it,
and uploads the checkpoint + eval summaries to a private HF model repo by
default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from topology_plan import write_topology_plan


REFERENCE_REPO = "https://github.com/yuyijiong/speculative_pipeline_decoding.git"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real SPD head proof or smoke job")
    parser.add_argument("--work-dir", default="/tmp/skippy-spd-qwen06-proof")
    parser.add_argument("--reference-repo", default=REFERENCE_REPO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--train-rows", type=int, default=1024)
    parser.add_argument("--eval-rows-per-set", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument(
        "--topology-policy",
        choices=("fixed", "generic-plan"),
        default="fixed",
        help=(
            "Topology handling. 'fixed' preserves the reference trainer contract. "
            "'generic-plan' writes randomized contiguous-layer tap plans and exits; "
            "it is the scaffold for topology-independent sidecar training."
        ),
    )
    parser.add_argument(
        "--topology-plan-out",
        default="",
        help=(
            "Output JSON path for --topology-policy generic-plan. Defaults under "
            "the current artifact directory."
        ),
    )
    parser.add_argument(
        "--topology-plan-samples",
        type=int,
        default=32,
        help="Number of randomized contiguous split layouts to write in generic-plan mode.",
    )
    parser.add_argument(
        "--topology-min-stages",
        type=int,
        default=2,
        help="Minimum stage count to sample in generic-plan mode.",
    )
    parser.add_argument(
        "--topology-max-stages",
        type=int,
        default=6,
        help="Maximum stage count to sample in generic-plan mode.",
    )
    parser.add_argument(
        "--topology-seed",
        type=int,
        default=47,
        help="Random seed for generic-plan topology sampling.",
    )
    parser.add_argument(
        "--topology-tap-dropout",
        type=float,
        default=0.25,
        help="Recorded tap-dropout probability for future generic topology training.",
    )
    parser.add_argument(
        "--topology-num-hidden-layers",
        type=int,
        default=0,
        help="Override target layer count for generic-plan mode when config loading is unavailable.",
    )
    parser.add_argument(
        "--stage-layer-boundaries",
        default="",
        help=(
            "Comma-separated target layer end indices for non-uniform topologies, "
            "for example 15,31,47 for GLM 4.7 Flash."
        ),
    )
    parser.add_argument(
        "--shallow-hidden-layer-indices",
        default="",
        help=(
            "Explicit semicolon-separated HF hidden-state tap rows for [g_n..g_1]. "
            "Overrides --stage-layer-boundaries when set."
        ),
    )
    parser.add_argument("--num-spec-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for the speculation head trainer.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--draft-top-k", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Device for local/reference execution. HF GPU jobs can leave this as auto.",
    )
    parser.add_argument(
        "--draft-vocab-json",
        default="draft_vocab/ultrachat_qwen3_0.6b_top_32k.json",
        help=(
            "Draft vocab JSON path. Relative paths are resolved inside the reference repo; "
            "absolute paths are passed through. Empty disables reduced draft vocab."
        ),
    )
    parser.add_argument(
        "--build-draft-vocab-size",
        type=int,
        default=0,
        help="Build a tokenizer-specific draft vocab from the loaded train rows before training.",
    )
    parser.add_argument(
        "--draft-vocab-out",
        default="",
        help="Output JSON for --build-draft-vocab-size. Defaults under the artifact data dir.",
    )
    parser.add_argument(
        "--upload-repo",
        default="auto",
        help="HF model repo for artifacts. Use 'auto' for <user>/skippy-spd-qwen06-proof.",
    )
    parser.add_argument("--public", action="store_true", help="Create upload repo as public")
    parser.add_argument(
        "--spec-head-path",
        default="",
        help="Existing speculation_head checkpoint to evaluate instead of training.",
    )
    parser.add_argument(
        "--spec-head-repo",
        default="",
        help="HF model repo containing an existing speculation_head checkpoint.",
    )
    parser.add_argument(
        "--spec-head-file",
        default="",
        help="Filename inside --spec-head-repo for an existing speculation_head checkpoint.",
    )
    parser.add_argument(
        "--manifest-base-model-path",
        default="",
        help="Override the base_model_path written to the Skippy SPD manifest.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def clone_reference(repo_url: str, dest: Path) -> None:
    if dest.exists():
        print(f"reference repo already exists: {dest}", flush=True)
        return
    run(["git", "clone", "--depth", "1", repo_url, str(dest)])


def patch_reference_for_proof(reference_dir: Path) -> None:
    write_qwen3_nonthink_template(reference_dir / "qwen3-nonthink-template")
    write_glm4_moe_lite_template(reference_dir / "glm4-moe-lite-template")
    replace_once(
        reference_dir / "train.py",
        '        report_to="wandb",\n',
        "        report_to=[],\n",
    )
    patch_reference_for_transformers(reference_dir)
    patch_reference_for_glm_training_smoke(reference_dir)


def write_qwen3_nonthink_template(path: Path) -> None:
    path.write_text(
        """{%- for message in messages %}
{%- if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
{% generation %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>{% endgeneration %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}
""",
        encoding="utf-8",
    )


def write_glm4_moe_lite_template(path: Path) -> None:
    path.write_text(
        """[gMASK]<sop>
{%- macro visible_text(content) -%}
    {%- if content is string -%}
        {{- content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}
                {{- item.text }}
            {%- elif item is string -%}
                {{- item }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- content }}
    {%- endif -%}
{%- endmacro -%}
{% for m in messages %}
{%- if m.role == 'system' -%}
<|system|>{{ visible_text(m.content) }}
{%- elif m.role == 'user' -%}
<|user|>{{ visible_text(m.content) }}
{%- elif m.role == 'assistant' -%}
{% generation %}<|assistant|>{{ visible_text(m.content) }}{% endgeneration %}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|assistant|>{{- '</think>' if (enable_thinking is defined and not enable_thinking) else '<think>' -}}
{%- endif -%}
""",
        encoding="utf-8",
    )


def patch_reference_for_transformers(reference_dir: Path) -> None:
    patch_reference_linear_cache_import(reference_dir / "pipeline_linear_cache.py")
    replace_once(
        reference_dir / "pipeline_model.py",
        '            "cache_position": cache_position,\n',
        "",
    )


def patch_reference_for_glm_training_smoke(reference_dir: Path) -> None:
    patch_pipeline_model_for_glm(reference_dir / "pipeline_model.py")
    patch_train_for_glm_template(reference_dir / "train.py")


def patch_train_for_glm_template(path: Path) -> None:
    replace_once(
        path,
        '''    if model_type is not None and not _model_type_looks_like_qwen(model_type):
        log.info("Skip Qwen file chat template (model_type=%s).", model_type)
        return
    if template_dir is None:
        template_dir = "."
''',
        '''    if template_dir is None:
        template_dir = "."
    if model_type is not None and str(model_type).lower() == "glm4_moe_lite":
        path = os.path.join(template_dir, "glm4-moe-lite-template")
        if not os.path.isfile(path):
            log.warning("GLM chat template not found: %s (use tokenizer default).", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
        log.info("Loaded GLM chat template from %s (model_type=%s).", path, model_type)
        return
    if model_type is not None and not _model_type_looks_like_qwen(model_type):
        log.info("Skip Qwen file chat template (model_type=%s).", model_type)
        return
''',
    )


def patch_pipeline_model_for_glm(path: Path) -> None:
    replace_once(
        path,
        'supported = {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama"}',
        'supported = {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama", "glm4_moe_lite"}',
    )
    replace_once(
        path,
        '''        if self.num_layers % self.num_stages != 0:
            raise ValueError(
                f"num_layers ({self.num_layers}) must be divisible by num_stages ({self.num_stages})"
            )
        self.layers_per_stage = self.num_layers // self.num_stages
''',
        '''        if self.num_layers % self.num_stages != 0:
            if shallow_hidden_layer_indices is None:
                raise ValueError(
                    f"num_layers ({self.num_layers}) must be divisible by num_stages ({self.num_stages}) "
                    "unless shallow_hidden_layer_indices supplies an explicit non-uniform topology"
                )
            self.layers_per_stage = max(1, (self.num_layers + self.num_stages - 1) // self.num_stages)
        else:
            self.layers_per_stage = self.num_layers // self.num_stages
''',
    )
    replace_once(
        path,
        "        self.shallow_hidden_layer_indices = self._normalize_stage_feature_indices(shallow_hidden_layer_indices)\n",
        """        self.shallow_hidden_layer_indices = self._normalize_stage_feature_indices(shallow_hidden_layer_indices)
        self.stage_layer_ranges = self._derive_stage_layer_ranges()
""",
    )
    replace_once(
        path,
        '''    def _snap_indices_needed(self) -> Set[int]:
        want: Set[int] = set()
        for row in self.shallow_hidden_layer_indices:
            for idx in row:
                want.add(int(idx))
        return want
''',
        '''    def _snap_indices_needed(self) -> Set[int]:
        want: Set[int] = set()
        for row in self.shallow_hidden_layer_indices:
            for idx in row:
                want.add(int(idx))
        return want

    def _derive_stage_layer_ranges(self) -> List[Tuple[int, int]]:
        rows = self.shallow_hidden_layer_indices
        deepest_row = tuple(sorted({int(x) for x in rows[0] if int(x) > 0})) if rows else ()
        if (
            len(deepest_row) >= self.num_stages
            and deepest_row[-1] == self.num_layers
            and all(a < b for a, b in zip(deepest_row, deepest_row[1:]))
        ):
            ranges: List[Tuple[int, int]] = []
            start = 0
            for end in deepest_row[: self.num_stages]:
                ranges.append((start, end))
                start = end
            return ranges

        ranges = []
        for stage_idx in range(self.num_stages):
            start = min(stage_idx * self.layers_per_stage, self.num_layers)
            end = min((stage_idx + 1) * self.layers_per_stage, self.num_layers)
            ranges.append((start, end))
        return ranges
''',
    )
    replace_once(
        path,
        '''                start_layer = stage_idx * lps
                end_layer = (stage_idx + 1) * lps
''',
        '''                start_layer, end_layer = self.stage_layer_ranges[stage_idx]
''',
    )
    replace_once(
        path,
        '''                start_layer = stage_idx * lps
                end_layer = (stage_idx + 1) * lps
''',
        '''                start_layer, end_layer = self.stage_layer_ranges[stage_idx]
''',
    )


def patch_reference_linear_cache_import(path: Path) -> None:
    replace_once(
        path,
        '''from transformers.cache_utils import (
    LinearAttentionAndFullAttentionLayer,
    LinearAttentionCacheLayerMixin,
    LinearAttentionLayer,
)
''',
        '''from transformers.cache_utils import LinearAttentionCacheLayerMixin, LinearAttentionLayer

try:
    from transformers.cache_utils import LinearAttentionAndFullAttentionLayer
except ImportError:
    class LinearAttentionAndFullAttentionLayer(LinearAttentionLayer):
        """Compatibility placeholder for Transformers 4.x without hybrid linear caches."""

        pass

''',
    )


def patch_reference_for_device(reference_dir: Path, device: str) -> None:
    if device == "auto":
        return
    patch_train_for_device(reference_dir / "train.py")
    patch_eval_for_device(reference_dir / "eval.py")
    patch_pipeline_inference_for_device(reference_dir / "pipeline_inference.py")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"expected text not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_train_for_device(path: Path) -> None:
    replace_once(
        path,
        '    device_map: Any = {"": local_rank} if torch.cuda.is_available() else "cpu"\n',
        '''    spd_device = os.environ.get("SPD_DEVICE", "auto").lower()
    if spd_device == "mps":
        device_map: Any = {"": "mps"}
    elif spd_device == "cpu":
        device_map = "cpu"
    elif spd_device == "cuda":
        device_map = {"": local_rank}
    else:
        device_map = {"": local_rank} if torch.cuda.is_available() else "cpu"

''',
    )
    replace_once(
        path,
        "        torch_dtype=torch.bfloat16,\n",
        '''        torch_dtype=(
            torch.float32 if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else torch.bfloat16
        ),
''',
    )
    replace_once(
        path,
        "        bf16=True,\n        fp16=False,\n",
        '''        bf16=torch.cuda.is_available(),
        fp16=False,
''',
    )


def patch_eval_for_device(path: Path) -> None:
    for _ in range(2):
        replace_once(
            path,
            '''        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2",
''',
            '''        dtype=torch.float16 if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else torch.bfloat16,
        device_map={"": "mps"} if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else ({"": 0} if torch.cuda.is_available() else None),
        attn_implementation=os.environ.get("SPD_ATTN_IMPLEMENTATION", "sdpa"),
''',
        )
    replace_once(
        path,
        '    map_loc = "cuda"\n',
        '    map_loc = "mps" if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else ("cuda" if torch.cuda.is_available() else "cpu")\n',
    )


def patch_pipeline_inference_for_device(path: Path) -> None:
    replace_once(
        path,
        '''        dtype=dtype,
        device_map={"":0},
        trust_remote_code=True,
''',
        '''        dtype=torch.float16 if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else dtype,
        device_map={"": "mps"} if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else {"": 0},
        trust_remote_code=True,
''',
    )
    replace_once(
        path,
        '    map_loc = "cuda" if torch.cuda.is_available() else "cpu"\n',
        '    map_loc = "mps" if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else ("cuda" if torch.cuda.is_available() else "cpu")\n',
    )


def load_training_rows(dataset_name: str, split: str, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    wanted = max(1, int(limit))
    split_expr = f"{split}[:{wanted}]"
    print(f"loading training data: {dataset_name} {split_expr}", flush=True)
    try:
        ds = load_dataset(dataset_name, split=split_expr)
    except Exception:
        fallback = f"train[:{wanted}]"
        print(f"failed to load split {split_expr!r}; trying {fallback!r}", flush=True)
        ds = load_dataset(dataset_name, split=fallback)

    rows: list[dict[str, Any]] = []
    for row in ds:
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            continue
        normalized = []
        for message in messages:
            role = message.get("role") or message.get("from")
            content = message.get("content") or message.get("value")
            if role is None or content is None:
                normalized = []
                break
            if role == "human":
                role = "user"
            if role == "gpt":
                role = "assistant"
            normalized.append({"role": str(role), "content": str(content)})
        if normalized:
            rows.append({"messages": normalized})
    if not rows:
        raise RuntimeError(f"no usable messages/conversations rows found in {dataset_name}")
    return rows[:wanted]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}", flush=True)


def create_mini_eval_data(reference_dir: Path, out_dir: Path, rows_per_set: int) -> None:
    source_root = reference_dir / "eval_data"
    for name in ("mt_bench", "humaneval", "gsm8k"):
        source = source_root / name / "question.jsonl"
        dest_dir = out_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "question.jsonl"
        copied = 0
        with source.open("r", encoding="utf-8") as src, dest.open("w", encoding="utf-8") as dst:
            for line in src:
                if copied >= rows_per_set:
                    break
                if line.strip():
                    dst.write(line)
                    copied += 1
        print(f"mini eval {name}: {copied} rows -> {dest}", flush=True)


def parse_stage_layer_boundaries(value: str) -> list[int] | None:
    value = (value or "").strip()
    if not value:
        return None
    boundaries = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not boundaries:
        raise RuntimeError("--stage-layer-boundaries must not be empty when set")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise RuntimeError(f"--stage-layer-boundaries must be strictly increasing: {boundaries}")
    return boundaries


def derive_hidden_tap_indices(boundaries: list[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    for depth in range(len(boundaries), 0, -1):
        rows.append([0, *boundaries[:depth]])
    return rows


def hidden_tap_rows_arg(args: argparse.Namespace) -> str:
    explicit = args.shallow_hidden_layer_indices.strip()
    if explicit:
        return explicit
    boundaries = parse_stage_layer_boundaries(args.stage_layer_boundaries)
    if boundaries is None:
        return ""
    if len(boundaries) != args.num_stages:
        raise RuntimeError(
            f"--stage-layer-boundaries has {len(boundaries)} entries but --num-stages is {args.num_stages}"
        )
    return ";".join(",".join(str(index) for index in row) for row in derive_hidden_tap_indices(boundaries))


def resolve_draft_vocab_path(reference_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return reference_dir / path


def build_draft_vocab_json(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    output_path: Path,
    vocab_size: int,
) -> Path:
    from collections import Counter

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    counts: Counter[int] = Counter()
    for row in rows:
        text = row_text_for_vocab(tokenizer, row.get("messages") or [])
        if not text:
            continue
        counts.update(int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False))
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ):
        if token_id is not None:
            counts[int(token_id)] += 1
    if not counts:
        raise RuntimeError("could not build draft vocab: no tokens counted")
    token_ids = [
        token_id
        for token_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            : max(1, int(vocab_size))
        ]
    ]
    token_ids = sorted(set(token_ids))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "draft_vocab_size": len(token_ids),
        "token_ids": token_ids,
        "metadata": {
            "base_model_path": args.model_name,
            "source": "hf_train_eval_qwen06.py --build-draft-vocab-size",
            "train_rows": len(rows),
            "requested_vocab_size": int(vocab_size),
        },
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote draft vocab ({len(token_ids)} ids) -> {output_path}", flush=True)
    return output_path


def row_text_for_vocab(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if not messages:
        return ""
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if isinstance(rendered, str):
            return rendered
    except Exception:
        pass
    return "\n".join(str(message.get("content", "")) for message in messages)


def train_head(args: argparse.Namespace, reference_dir: Path, train_jsonl: Path, train_dir: Path) -> Path:
    cmd = [
        sys.executable,
        "train.py",
        "--model_name",
        args.model_name,
        "--data_path",
        str(train_jsonl),
        "--num_stages",
        str(args.num_stages),
        "--num_spec_layers",
        str(args.num_spec_layers),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--lr",
        str(args.learning_rate),
        "--max_length",
        str(args.max_length),
        "--max_length_overflow",
        "truncate",
        "--min_length",
        "10",
        "--num_proc",
        "2",
        "--attn_implementation",
        args.attn_implementation,
        "--output_dir",
        str(train_dir),
    ]
    hidden_rows = hidden_tap_rows_arg(args)
    if hidden_rows:
        cmd.extend(["--shallow_hidden_layer_indices", hidden_rows])
    if args.draft_vocab_json:
        cmd.extend(["--draft_vocab_json", str(resolve_draft_vocab_path(reference_dir, args.draft_vocab_json))])
    started = time.perf_counter()
    run(cmd, cwd=reference_dir, env=reference_env(args))
    elapsed = time.perf_counter() - started
    ckpt = train_dir / "speculation_head_final.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"training did not produce {ckpt}")
    print(f"training complete in {elapsed / 60.0:.2f} min: {ckpt}", flush=True)
    write_skippy_spd_manifest(args, ckpt, train_dir / "skippy-spd-head.json")
    return ckpt


def prepare_existing_spec_head(args: argparse.Namespace, train_dir: Path) -> Path | None:
    local_path = args.spec_head_path.strip()
    repo = args.spec_head_repo.strip()
    filename = args.spec_head_file.strip()
    if not local_path and not repo and not filename:
        return None
    if local_path and (repo or filename):
        raise RuntimeError("--spec-head-path cannot be combined with --spec-head-repo/file")
    if bool(repo) != bool(filename):
        raise RuntimeError("--spec-head-repo and --spec-head-file must be set together")

    if repo:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="model"))
    else:
        source = Path(local_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"speculation head checkpoint not found: {source}")

    train_dir.mkdir(parents=True, exist_ok=True)
    dest = train_dir / "speculation_head_final.pt"
    if source.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
    write_skippy_spd_manifest(args, dest, train_dir / "skippy-spd-head.json")
    print(f"prepared existing speculation head -> {dest}", flush=True)
    return dest


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_manifest_model_type(config: dict[str, Any], model_name: str) -> str | None:
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type:
        return model_type

    try:
        from transformers import AutoConfig

        base_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not resolve base model_type for manifest: {exc}", flush=True)
        return None

    model_type = getattr(base_config, "model_type", None)
    return model_type if isinstance(model_type, str) and model_type else None


def write_skippy_spd_manifest(args: argparse.Namespace, ckpt: Path, manifest_path: Path) -> None:
    import torch

    try:
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt, map_location="cpu")
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict):
        raise RuntimeError(f"{ckpt} does not contain a config dict")

    draft_token_ids = config.get("draft_token_ids")
    stage_layer_boundaries = config.get("stage_layer_boundaries")
    if stage_layer_boundaries is None:
        stage_layer_boundaries = parse_stage_layer_boundaries(args.stage_layer_boundaries)
    manifest_base_model_path = args.manifest_base_model_path.strip()
    if not manifest_base_model_path:
        manifest_base_model_path = config.get("base_model_path") or args.model_name

    manifest = {
        "schema": "skippy-spd-head/v1",
        "checkpoint": {
            "path": ckpt.name,
            "sha256": file_sha256(ckpt),
            "bytes": ckpt.stat().st_size,
        },
        "source": {
            "format": "torch-speculation-head-v10",
            "reference_repo": args.reference_repo,
            "base_model_path": manifest_base_model_path,
            "model_type": resolve_manifest_model_type(config, args.model_name),
            "checkpoint_version": int(config.get("version", 0)),
        },
        "topology": {
            "hidden_size": int(config["hidden_size"]),
            "vocab_size": int(config["vocab_size"]),
            "draft_vocab_size": int(config.get("draft_vocab_size", config["vocab_size"])),
            "num_stages": int(config["num_stages"]),
            "stage_layer_boundaries": stage_layer_boundaries,
            "num_spec_layers": int(config["num_spec_layers"]),
            "trained_with_use_deepest": bool(config.get("trained_with_use_deepest", False)),
            "shallow_hidden_layer_indices": config["shallow_hidden_layer_indices"],
            "spec_init_from_base_layers": config.get("spec_init_from_base_layers"),
            "draft_token_ids": draft_token_ids,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote Skippy SPD manifest -> {manifest_path}", flush=True)


def evaluate_head(
    args: argparse.Namespace,
    reference_dir: Path,
    ckpt: Path,
    eval_data: Path,
    eval_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "eval.py",
        "--spec_head_ckpt",
        str(ckpt),
        "--base_model_path",
        args.model_name,
        "--data_dir",
        str(eval_data),
        "--output_dir",
        str(eval_dir),
        "--gpus",
        "0",
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        "0.0",
        "--draft_top_k",
        str(args.draft_top_k),
        "--no-baseline",
    ]
    started = time.perf_counter()
    run(cmd, cwd=reference_dir, env=reference_env(args))
    elapsed = time.perf_counter() - started
    print(f"eval complete in {elapsed / 60.0:.2f} min", flush=True)


def print_eval_summary(eval_dir: Path) -> None:
    summary_dir = eval_dir / "summary"
    if not summary_dir.is_dir():
        print(f"no eval summary dir found: {summary_dir}", flush=True)
        return
    for path in sorted(summary_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        print(f"summary: {path}", flush=True)
        overall = obj.get("overall") or {}
        if not overall and obj.get("results"):
            overall = obj["results"][0].get("overall", {})
        interesting = {
            key: overall.get(key)
            for key in (
                "acceptance_rate",
                "equivalent_accept_length",
                "theoretical_speedup",
                "new_tokens",
                "decode_loop_steps",
            )
            if key in overall
        }
        print(json.dumps(interesting or overall, indent=2, sort_keys=True), flush=True)


def resolve_upload_repo(value: str) -> str | None:
    value = (value or "").strip()
    if not value or value.lower() in {"none", "off", "false", "disabled"}:
        return None
    if value != "auto":
        return value
    from huggingface_hub import HfApi

    who = HfApi().whoami()
    name = who.get("name")
    if not name:
        raise RuntimeError("could not resolve HF username for --upload-repo auto")
    return f"{name}/skippy-spd-qwen06-proof"


def upload_artifacts(upload_repo: str | None, artifact_dir: Path, *, public: bool) -> None:
    if upload_repo is None:
        print(f"no upload repo configured; artifacts remain at {artifact_dir}", flush=True)
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(upload_repo, repo_type="model", private=not public, exist_ok=True)
    api.upload_folder(
        repo_id=upload_repo,
        repo_type="model",
        folder_path=str(artifact_dir),
        path_in_repo=f"runs/{artifact_dir.name}",
    )
    print(f"uploaded artifacts to hf://models/{upload_repo}/runs/{artifact_dir.name}", flush=True)


def reference_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.device != "auto":
        env["SPD_DEVICE"] = args.device
    env["SPD_ATTN_IMPLEMENTATION"] = args.attn_implementation
    return env


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    reference_dir = work_dir / "speculative_pipeline_decoding"
    artifact_dir = work_dir / "artifacts" / time.strftime("%Y%m%d-%H%M%S")
    data_dir = work_dir / "data"
    train_dir = artifact_dir / "train"
    eval_dir = artifact_dir / "eval"
    mini_eval_dir = data_dir / "mini_eval_data"
    train_jsonl = data_dir / "train_conversations.jsonl"

    work_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.topology_policy == "generic-plan":
        write_topology_plan(args, artifact_dir)
        print(
            "generic-plan mode stops before training because the donor SPD head "
            "still owns fixed per-stage projection tensors.",
            flush=True,
        )
        return

    clone_reference(args.reference_repo, reference_dir)
    patch_reference_for_proof(reference_dir)
    patch_reference_for_device(reference_dir, args.device)

    existing_ckpt = prepare_existing_spec_head(args, train_dir)
    if existing_ckpt is not None:
        ckpt = existing_ckpt
    elif not args.skip_train:
        rows = load_training_rows(args.dataset, args.dataset_split, args.train_rows)
        write_jsonl(train_jsonl, rows)
        if args.build_draft_vocab_size > 0:
            draft_vocab_out = (
                Path(args.draft_vocab_out).expanduser().resolve()
                if args.draft_vocab_out
                else data_dir / f"draft_vocab_top_{args.build_draft_vocab_size}.json"
            )
            args.draft_vocab_json = str(
                build_draft_vocab_json(
                    args=args,
                    rows=rows,
                    output_path=draft_vocab_out,
                    vocab_size=args.build_draft_vocab_size,
                )
            )
        ckpt = train_head(args, reference_dir, train_jsonl, train_dir)
    else:
        ckpt = train_dir / "speculation_head_final.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"--skip-train requires existing checkpoint at {ckpt}")

    if not args.skip_eval:
        if mini_eval_dir.exists():
            shutil.rmtree(mini_eval_dir)
        create_mini_eval_data(reference_dir, mini_eval_dir, args.eval_rows_per_set)
        evaluate_head(args, reference_dir, ckpt, mini_eval_dir, eval_dir)
        print_eval_summary(eval_dir)

    upload_repo = resolve_upload_repo(args.upload_repo)
    upload_artifacts(upload_repo, artifact_dir, public=args.public)


if __name__ == "__main__":
    main()
