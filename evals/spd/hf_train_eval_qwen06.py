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
"""Train and evaluate a small SPD speculation head on Hugging Face Jobs.

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


REFERENCE_REPO = "https://github.com/yuyijiong/speculative_pipeline_decoding.git"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real Qwen3-0.6B SPD head proof job")
    parser.add_argument("--work-dir", default="/tmp/skippy-spd-qwen06-proof")
    parser.add_argument("--reference-repo", default=REFERENCE_REPO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--train-rows", type=int, default=1024)
    parser.add_argument("--eval-rows-per-set", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument(
        "--stage-layer-boundaries",
        default="",
        help=(
            "Comma-separated target layer end indices for the trained logical "
            "pipeline topology, for example 8,16,24,32."
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
        "--model-torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help=(
            "Base-model dtype for the patched reference trainer. 'auto' preserves "
            "the current proof defaults: float32 on MPS, bfloat16 otherwise."
        ),
    )
    parser.add_argument(
        "--draft-vocab-json",
        default="draft_vocab/ultrachat_qwen3_0.6b_top_32k.json",
        help="Path inside the reference repo; empty disables reduced draft vocab.",
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
    parser.add_argument(
        "--dry-run-topology",
        action="store_true",
        help=(
            "Validate and print the Skippy SPD topology plan as JSON, then exit "
            "before cloning, downloading, training, evaluating, or uploading."
        ),
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
    replace_once(
        reference_dir / "train.py",
        '        report_to="wandb",\n',
        "        report_to=[],\n",
    )
    patch_reference_for_transformers(reference_dir)


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


def patch_reference_for_transformers(reference_dir: Path) -> None:
    patch_reference_linear_cache_import(reference_dir / "pipeline_linear_cache.py")
    replace_once(
        reference_dir / "pipeline_model.py",
        '            "cache_position": cache_position,\n',
        "",
    )
    patch_reference_for_stage_boundaries(reference_dir)


def patch_reference_for_stage_boundaries(reference_dir: Path) -> None:
    patch_pipeline_model_for_stage_boundaries(reference_dir / "pipeline_model.py")
    patch_pipeline_inference_for_stage_boundaries(reference_dir / "pipeline_inference.py")
    patch_train_checkpoint_for_stage_boundaries(reference_dir / "train.py")


def patch_pipeline_model_for_stage_boundaries(path: Path) -> None:
    replace_once(
        path,
        """        shallow_hidden_layer_indices: Optional[Sequence[Sequence[int]]] = None,
        trained_with_use_deepest: bool = False,
    ):
""",
        """        shallow_hidden_layer_indices: Optional[Sequence[Sequence[int]]] = None,
        trained_with_use_deepest: bool = False,
        stage_layer_boundaries: Optional[Sequence[int]] = None,
    ):
""",
    )
    replace_once(
        path,
        """        self.shallow_hidden_layer_indices = self._normalize_stage_feature_indices(shallow_hidden_layer_indices)

        v_full = int(dec_cfg.vocab_size)
""",
        """        self.shallow_hidden_layer_indices = self._normalize_stage_feature_indices(shallow_hidden_layer_indices)
        self.stage_layer_boundaries = self._normalize_stage_layer_boundaries(stage_layer_boundaries)

        v_full = int(dec_cfg.vocab_size)
""",
    )
    replace_once(
        path,
        """    def _default_stage_feature_indices(self) -> List[Tuple[int, ...]]:
""",
        """    def _normalize_stage_layer_boundaries(
        self,
        stage_layer_boundaries: Optional[Sequence[int]],
    ) -> List[int]:
        if stage_layer_boundaries is None:
            first_row = self.shallow_hidden_layer_indices[0] if self.shallow_hidden_layer_indices else ()
            if (
                len(first_row) == self.num_stages + 1
                and int(first_row[0]) == 0
                and int(first_row[-1]) == self.num_layers
            ):
                stage_layer_boundaries = [int(x) for x in first_row[1:]]
            else:
                stage_layer_boundaries = [
                    (idx + 1) * self.layers_per_stage for idx in range(self.num_stages)
                ]

        boundaries = [int(x) for x in stage_layer_boundaries]
        if len(boundaries) != self.num_stages:
            raise ValueError(
                f"stage_layer_boundaries must have length num_stages={self.num_stages}, got {len(boundaries)}"
            )
        if boundaries[-1] != self.num_layers:
            raise ValueError(
                f"stage_layer_boundaries must end at num_layers={self.num_layers}, got {boundaries}"
            )
        prev = 0
        for boundary in boundaries:
            if boundary <= prev:
                raise ValueError(f"stage_layer_boundaries must be strictly increasing: {boundaries}")
            prev = boundary
        return boundaries

    def _stage_layer_range(self, stage_idx: int) -> Tuple[int, int]:
        idx = int(stage_idx)
        if idx < 0 or idx >= self.num_stages:
            raise IndexError(f"stage_idx out of range: {idx}")
        start = 0 if idx == 0 else int(self.stage_layer_boundaries[idx - 1])
        end = int(self.stage_layer_boundaries[idx])
        return start, end

    def _default_stage_feature_indices(self) -> List[Tuple[int, ...]]:
""",
    )
    for _ in range(2):
        replace_once(
            path,
            """                    start_layer = stage_idx * lps
                    end_layer = (stage_idx + 1) * lps
""",
            """                    start_layer, end_layer = self._stage_layer_range(stage_idx)
""",
        )


def patch_pipeline_inference_for_stage_boundaries(path: Path) -> None:
    replace_once(
        path,
        """    shallow_hidden_layer_indices = (
        [[int(y) for y in x] for x in raw_shallow] if raw_shallow is not None else None
    )
    kw: dict[str, Any] = {
""",
        """    shallow_hidden_layer_indices = (
        [[int(y) for y in x] for x in raw_shallow] if raw_shallow is not None else None
    )
    raw_boundaries = cfg.get("stage_layer_boundaries")
    if raw_boundaries is None and shallow_hidden_layer_indices:
        first_row = shallow_hidden_layer_indices[0]
        if len(first_row) == int(cfg["num_stages"]) + 1 and first_row[0] == 0:
            raw_boundaries = first_row[1:]
    stage_layer_boundaries = (
        [int(x) for x in raw_boundaries] if raw_boundaries is not None else None
    )
    kw: dict[str, Any] = {
""",
    )
    replace_once(
        path,
        """        "shallow_hidden_layer_indices": shallow_hidden_layer_indices,
    }
""",
        """        "shallow_hidden_layer_indices": shallow_hidden_layer_indices,
        "stage_layer_boundaries": stage_layer_boundaries,
    }
""",
    )


def patch_train_checkpoint_for_stage_boundaries(path: Path) -> None:
    replace_once(
        path,
        """                    "num_stages": pm.num_stages,
                    "num_spec_layers": pm.num_spec_layers,
""",
        """                    "num_stages": pm.num_stages,
                    "stage_layer_boundaries": list(getattr(pm, "stage_layer_boundaries", [])),
                    "num_spec_layers": pm.num_spec_layers,
""",
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
            {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
                os.environ.get("SPD_TORCH_DTYPE", "auto").lower()
            )
            or (torch.float32 if os.environ.get("SPD_DEVICE", "auto").lower() == "mps" else torch.bfloat16)
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
    if any(boundary <= 0 for boundary in boundaries):
        raise RuntimeError(f"--stage-layer-boundaries must be positive: {boundaries}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise RuntimeError(f"--stage-layer-boundaries must be strictly increasing: {boundaries}")
    return boundaries


def parse_hidden_tap_rows(value: str) -> list[list[int]]:
    value = (value or "").strip()
    if not value:
        return []
    rows: list[list[int]] = []
    for row_text in value.split(";"):
        row_text = row_text.strip()
        if not row_text:
            continue
        row = [int(part.strip()) for part in row_text.split(",") if part.strip()]
        if not row:
            raise RuntimeError("--shallow-hidden-layer-indices contains an empty row")
        if any(index < 0 for index in row):
            raise RuntimeError(f"hidden-state tap indices must be non-negative: {row}")
        if row[0] != 0:
            raise RuntimeError(f"hidden-state tap rows must start with embedding row 0: {row}")
        if any(left >= right for left, right in zip(row, row[1:])):
            raise RuntimeError(f"hidden-state tap rows must be strictly increasing: {row}")
        rows.append(row)
    if not rows:
        raise RuntimeError("--shallow-hidden-layer-indices must not be empty when set")
    return rows


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
            f"--stage-layer-boundaries has {len(boundaries)} entries but "
            f"--num-stages is {args.num_stages}"
        )
    return ";".join(",".join(str(index) for index in row) for row in derive_hidden_tap_indices(boundaries))


def topology_dry_run(args: argparse.Namespace) -> None:
    if args.num_stages <= 0:
        raise RuntimeError(f"--num-stages must be positive: {args.num_stages}")
    if args.num_spec_layers <= 0:
        raise RuntimeError(f"--num-spec-layers must be positive: {args.num_spec_layers}")

    boundaries = parse_stage_layer_boundaries(args.stage_layer_boundaries)
    explicit_rows = parse_hidden_tap_rows(args.shallow_hidden_layer_indices)
    if explicit_rows:
        tap_rows = explicit_rows
        hidden_rows_arg = ";".join(",".join(str(index) for index in row) for row in tap_rows)
        derived_from_stage_layer_boundaries = False
    else:
        hidden_rows_arg = hidden_tap_rows_arg(args)
        tap_rows = parse_hidden_tap_rows(hidden_rows_arg) if hidden_rows_arg else []
        derived_from_stage_layer_boundaries = bool(boundaries)

    physical_split_boundaries = boundaries[:-1] if boundaries else []
    layer_end = boundaries[-1] if boundaries else None
    required_hf_hidden_state_indices = sorted({index for row in tap_rows for index in row})
    spd_tap_return_hf_indices = [index for index in required_hf_hidden_state_indices if index != 0]
    manifest_base_model_path = args.manifest_base_model_path.strip() or args.model_name
    plan = {
        "model_name": args.model_name,
        "manifest_base_model_path": manifest_base_model_path,
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "train_rows": args.train_rows,
        "eval_rows_per_set": args.eval_rows_per_set,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "model_torch_dtype": args.model_torch_dtype,
        "attn_implementation": args.attn_implementation,
        "num_stages": args.num_stages,
        "stage_layer_boundaries": boundaries,
        "physical_split_boundaries": physical_split_boundaries,
        "layer_end": layer_end,
        "num_spec_layers": args.num_spec_layers,
        "draft_top_k": args.draft_top_k,
        "draft_vocab_json": args.draft_vocab_json,
        "shallow_hidden_layer_indices": hidden_rows_arg,
        "shallow_hidden_layer_index_rows": tap_rows,
        "derived_from_stage_layer_boundaries": derived_from_stage_layer_boundaries,
        "required_hf_hidden_state_indices": required_hf_hidden_state_indices,
        "spd_tap_return_hf_indices": spd_tap_return_hf_indices,
        "dry_run": {
            "clones_reference_repo": False,
            "downloads_model": False,
            "trains": False,
            "evaluates": False,
            "uploads": False,
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)


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
        cmd.extend(["--draft_vocab_json", str(reference_dir / args.draft_vocab_json)])
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
    rotary_metadata = resolve_rotary_metadata(manifest_base_model_path, config)

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
            "model_type": config.get("model_type"),
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
            **rotary_metadata,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote Skippy SPD manifest -> {manifest_path}", flush=True)


def resolve_rotary_metadata(base_model_path: str, checkpoint_config: dict[str, Any]) -> dict[str, int]:
    rope_theta = checkpoint_config.get("rope_theta")
    rotary_dim = checkpoint_config.get("rotary_dim")
    if rope_theta is not None or rotary_dim is not None:
        if rope_theta is None or rotary_dim is None:
            raise RuntimeError("checkpoint rotary metadata must include both rope_theta and rotary_dim")
        return validated_rotary_metadata(rope_theta, rotary_dim)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    dec_cfg = getattr(cfg, "text_config", None) or cfg
    head_dim = int(getattr(dec_cfg, "head_dim"))
    rope_parameters = getattr(dec_cfg, "rope_parameters", None) or {}
    rope_theta = getattr(dec_cfg, "rope_theta", None) or rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise RuntimeError(f"could not resolve rope_theta for {base_model_path}")
    partial_rotary_factor = getattr(dec_cfg, "partial_rotary_factor", None)
    if partial_rotary_factor is None:
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
    if partial_rotary_factor is None:
        rotary_dim = head_dim
    else:
        rotary_dim = int(round(head_dim * float(partial_rotary_factor)))
    return validated_rotary_metadata(rope_theta, rotary_dim)


def validated_rotary_metadata(rope_theta: Any, rotary_dim: Any) -> dict[str, int]:
    rope_theta_int = int(rope_theta)
    rotary_dim_int = int(rotary_dim)
    if rope_theta_int <= 0 or rotary_dim_int <= 0:
        raise RuntimeError("rope_theta and rotary_dim must be positive")
    return {"rope_theta": rope_theta_int, "rotary_dim": rotary_dim_int}


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
    if not value:
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
    env["SPD_TORCH_DTYPE"] = args.model_torch_dtype
    env["SPD_ATTN_IMPLEMENTATION"] = args.attn_implementation
    return env


def main() -> None:
    args = parse_args()
    if args.dry_run_topology:
        topology_dry_run(args)
        return

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
    clone_reference(args.reference_repo, reference_dir)
    patch_reference_for_proof(reference_dir)
    patch_reference_for_device(reference_dir, args.device)

    existing_ckpt = prepare_existing_spec_head(args, train_dir)
    if existing_ckpt is not None:
        ckpt = existing_ckpt
    elif not args.skip_train:
        rows = load_training_rows(args.dataset, args.dataset_split, args.train_rows)
        write_jsonl(train_jsonl, rows)
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
