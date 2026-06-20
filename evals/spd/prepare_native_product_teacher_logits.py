#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
# ]
# ///
"""Convert native Skippy product verifier logits into teacher safetensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prepare_product_activation_corpus import CORPUS_SCHEMA, read_json, read_rows


NATIVE_ROW_SCHEMA = "skippy-spd-native-teacher-row/v1"
NATIVE_MANIFEST_SCHEMA = "skippy-spd-native-teacher-logits/v1"
OUT_SCHEMA = "skippy-spd-product-activation-teacher-logits/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert native Q4 product-verifier logits emitted by "
            "skippy-bench spd-live-tap-parity --product-native-teacher-logits "
            "into the teacher-logit safetensors consumed by "
            "train_product_activation_head.py."
        )
    )
    parser.add_argument("--corpus-dir", required=True, help="Input product corpus directory")
    parser.add_argument("--out", required=True, help="Output teacher safetensors path")
    parser.add_argument(
        "--save-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--summary-json", help="Optional JSON summary output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    corpus_manifest = read_json(corpus_dir / "manifest.json")
    if corpus_manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unsupported corpus schema: {corpus_manifest.get('schema')!r}")
    native_manifest = read_json(corpus_dir / "native_teacher_manifest.json")
    if native_manifest.get("schema") != NATIVE_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported native teacher schema: {native_manifest.get('schema')!r}"
        )
    product_rows = read_rows(corpus_dir / "rows.jsonl")
    native_rows = read_native_rows(corpus_dir / "native_teacher_rows.jsonl")
    if len(product_rows) != len(native_rows):
        raise ValueError(
            f"product row count {len(product_rows)} does not match native teacher row "
            f"count {len(native_rows)}"
        )
    draft_token_ids = [
        int(token)
        for token in corpus_manifest.get("topology", {}).get("draft_token_ids", [])
    ]
    logit_width = int(native_manifest["logit_width"])
    if len(draft_token_ids) != logit_width:
        raise ValueError(
            f"draft_token_ids length {len(draft_token_ids)} does not match native "
            f"logit_width {logit_width}"
        )
    tensors, summary = build_tensors(
        corpus_dir=corpus_dir,
        product_rows=product_rows,
        native_rows=native_rows,
        draft_token_ids=draft_token_ids,
        logit_width=logit_width,
        save_dtype=args.save_dtype,
        base_model_path=resolve_base_model_path(corpus_manifest),
    )
    save_teacher_tensors(Path(args.out), tensors, summary)
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def read_native_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("schema") != NATIVE_ROW_SCHEMA:
                raise ValueError(
                    f"{path}:{line_number}: unsupported row schema {row.get('schema')!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no native teacher rows")
    return rows


def resolve_base_model_path(corpus_manifest: dict[str, Any]) -> str:
    manifest_path = corpus_manifest.get("manifest_path")
    if isinstance(manifest_path, str) and Path(manifest_path).is_file():
        manifest = read_json(Path(manifest_path))
        base_model_path = manifest.get("source", {}).get("base_model_path")
        if base_model_path:
            return str(base_model_path)
    return ""


def build_tensors(
    *,
    corpus_dir: Path,
    product_rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
    draft_token_ids: list[int],
    logit_width: int,
    save_dtype: str,
    base_model_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    validate_alignment(product_rows, native_rows)
    logits = read_logits(
        corpus_dir / "native_teacher_logits.f32",
        len(native_rows),
        logit_width,
    )
    save_torch_dtype = torch_dtype(save_dtype)
    sample_indices = []
    target_positions = []
    logit_positions = []
    query_row_indices = []
    query_positions = []
    target_logit_indices = []
    label_in_scope = []
    argmax_token_ids = []
    argmax_indices = []
    topk_token_ids = []
    topk_indices = []
    topk_logits = []
    for row in native_rows:
        sample_indices.append(int(row["sample_index"]))
        target_position = int(row["target_position"])
        target_positions.append(target_position)
        logit_positions.append(target_position - 1)
        query_row_indices.append(int(row["query_row_index"]))
        query_positions.append(int(row["query_position"]))
        target_logit_index = int(row["target_logit_index"])
        target_logit_indices.append(target_logit_index)
        label_in_scope.append(1 if bool(row["label_in_logit_scope"]) else 0)
        argmax_indices.append(int(row["teacher_argmax_index"]))
        argmax_token_ids.append(int(row["teacher_argmax_token_id"]))
        top_k = row["teacher_top_k"]
        topk_indices.append([int(value) for value in top_k["indices"]])
        topk_token_ids.append([int(value) for value in top_k["token_ids"]])
        topk_logits.append([float(value) for value in top_k["logits"]])
    tensors = {
        "teacher_logits": logits.to(dtype=save_torch_dtype).contiguous(),
        "teacher_argmax_token_ids": torch.tensor(argmax_token_ids, dtype=torch.long),
        "teacher_argmax_indices": torch.tensor(argmax_indices, dtype=torch.long),
        "target_logit_indices": torch.tensor(target_logit_indices, dtype=torch.long),
        "teacher_label_in_logit_scope": torch.tensor(label_in_scope, dtype=torch.long),
        "teacher_topk_token_ids": torch.tensor(topk_token_ids, dtype=torch.long),
        "teacher_topk_indices": torch.tensor(topk_indices, dtype=torch.long),
        "teacher_topk_logits": torch.tensor(topk_logits, dtype=save_torch_dtype),
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "target_positions": torch.tensor(target_positions, dtype=torch.long),
        "logit_positions": torch.tensor(logit_positions, dtype=torch.long),
        "query_row_indices": torch.tensor(query_row_indices, dtype=torch.long),
        "query_positions": torch.tensor(query_positions, dtype=torch.long),
        "teacher_logit_token_ids": torch.tensor(draft_token_ids, dtype=torch.long),
    }
    labels_in_scope = int(tensors["teacher_label_in_logit_scope"].sum().item())
    summary = {
        "schema": OUT_SCHEMA,
        "source_schema": NATIVE_MANIFEST_SCHEMA,
        "base_model_path": base_model_path,
        "sample_count": len(native_rows),
        "start_sample": 0,
        "logit_scope": "draft",
        "logit_width": logit_width,
        "top_k": int(tensors["teacher_topk_indices"].shape[1]),
        "save_dtype": save_dtype,
        "target_logits_available": True,
        "teacher_source": "native_skippy_product_verifier_current_logits",
        "native_product_teacher_logits": True,
        "paper_kl_training_ready": True,
        "labels_in_logit_scope": labels_in_scope,
        "labels_missing_from_logit_scope": len(native_rows) - labels_in_scope,
        "full_vocab_target_in_draft_scope": labels_in_scope,
        "full_vocab_target_missing_from_draft_scope": len(native_rows) - labels_in_scope,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    return tensors, summary


def validate_alignment(
    product_rows: list[dict[str, Any]],
    native_rows: list[dict[str, Any]],
) -> None:
    for index, (product, native) in enumerate(zip(product_rows, native_rows, strict=True)):
        for key in ["sample_index", "prompt_index", "step_index", "target_position"]:
            if int(product[key]) != int(native[key]):
                raise ValueError(f"row {index} has mismatched {key}")
        if int(product["target_token"]) != int(native["target_token"]):
            raise ValueError(f"row {index} has mismatched target_token")
        if int(product["query_row_index"]) != int(native["query_row_index"]):
            raise ValueError(f"row {index} has mismatched query_row_index")


def read_logits(path: Path, sample_count: int, logit_width: int) -> Any:
    import torch

    if sys.byteorder != "little":
        raise RuntimeError("native logits are little-endian; this converter expects little-endian")
    expected_floats = sample_count * logit_width
    expected_bytes = expected_floats * 4
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path} has {len(raw)} bytes, expected {expected_bytes} "
            f"for [{sample_count}, {logit_width}] f32"
        )
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(
        sample_count,
        logit_width,
    )


def torch_dtype(value: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def save_teacher_tensors(
    out: Path,
    tensors: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    from safetensors.torch import save_file

    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": OUT_SCHEMA,
        "base_model_path": str(summary["base_model_path"]),
        "sample_count": str(summary["sample_count"]),
        "logit_scope": str(summary["logit_scope"]),
        "logit_width": str(summary["logit_width"]),
        "save_dtype": str(summary["save_dtype"]),
        "teacher_source": str(summary["teacher_source"]),
        "native_product_teacher_logits": "true",
        "paper_kl_training_ready": "true",
    }
    save_file(tensors, out, metadata=metadata)
    summary["out"] = str(out)
    summary["bytes"] = out.stat().st_size


if __name__ == "__main__":
    main()
