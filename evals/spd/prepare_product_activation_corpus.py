#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
# ]
# ///
"""Convert live Skippy SPD product activations into a tensor corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CORPUS_SCHEMA = "skippy-spd-product-activation-corpus/v1"
ROW_SCHEMA = "skippy-spd-product-activation-row/v1"
OUT_SCHEMA = "skippy-spd-product-activation-safetensors/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert rows.f32 + rows.jsonl emitted by "
            "skippy-bench spd-live-tap-parity --product-corpus-dir into safetensors."
        )
    )
    parser.add_argument("--corpus-dir", required=True, help="Input product corpus directory")
    parser.add_argument("--out", required=True, help="Output safetensors path")
    parser.add_argument(
        "--manifest",
        help="Optional SPD manifest JSON used for draft-token label mapping. "
        "Defaults to topology embedded in the corpus manifest.",
    )
    parser.add_argument(
        "--require-labels-in-draft-vocab",
        action="store_true",
        help="Fail if any target greedy token is not present in draft_token_ids.",
    )
    parser.add_argument("--summary-json", help="Optional JSON summary output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    manifest = read_json(corpus_dir / "manifest.json")
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unsupported corpus schema: {manifest.get('schema')!r}")
    rows = read_rows(corpus_dir / "rows.jsonl")
    row_count = int(manifest["row_count"])
    hidden_size = int(manifest["hidden_size"])
    cur_in = read_cur_in(corpus_dir / "rows.f32", len(rows), row_count, hidden_size)
    raw_tap_concat, raw_tap_offsets, raw_tap_widths = read_raw_tap_concat(
        corpus_dir,
        manifest,
        len(rows),
    )
    final_norm_weight = read_f32_vector(
        corpus_dir / "final_norm_weight.f32",
        hidden_size,
        "final_norm_weight",
    )
    draft_token_ids = load_draft_token_ids(args.manifest, manifest)
    tensors, summary = build_tensors(
        rows=rows,
        cur_in=cur_in,
        raw_tap_concat=raw_tap_concat,
        raw_tap_offsets=raw_tap_offsets,
        raw_tap_widths=raw_tap_widths,
        final_norm_weight=final_norm_weight,
        row_count=row_count,
        hidden_size=hidden_size,
        draft_token_ids=draft_token_ids,
        require_labels_in_draft_vocab=bool(args.require_labels_in_draft_vocab),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    target_logits_available = bool(manifest.get("target_logits_available"))
    paper_kl_training_ready = bool(manifest.get("paper_kl_training_ready"))
    native_product_teacher_logits = bool(manifest.get("native_teacher_logits"))
    metadata = {
        "schema": OUT_SCHEMA,
        "source_schema": CORPUS_SCHEMA,
        "source_corpus_dir": str(corpus_dir),
        "sample_count": str(len(rows)),
        "row_count": str(row_count),
        "hidden_size": str(hidden_size),
        "cur_in_convention": str(manifest.get("cur_in_convention", "")),
        "raw_tap_convention": str(manifest.get("raw_tap_convention", "")),
        "label_kind": str(manifest.get("label_kind", "")),
        "target_logits_available": str(target_logits_available).lower(),
        "paper_kl_training_ready": str(paper_kl_training_ready).lower(),
        "native_product_teacher_logits": str(native_product_teacher_logits).lower(),
        "raw_tap_concat_available": str(raw_tap_concat is not None).lower(),
    }
    save_file(tensors, out_path, metadata=metadata)
    summary["target_logits_available"] = target_logits_available
    summary["paper_kl_training_ready"] = paper_kl_training_ready
    summary["native_product_teacher_logits"] = native_product_teacher_logits
    summary.update(
        {
            "schema": OUT_SCHEMA,
            "out": str(out_path),
            "bytes": out_path.stat().st_size,
            "source_corpus_dir": str(corpus_dir),
        }
    )
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("schema") != ROW_SCHEMA:
                raise ValueError(
                    f"{path}:{line_number}: unsupported row schema {row.get('schema')!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contained no product activation rows")
    return rows


def read_cur_in(path: Path, sample_count: int, row_count: int, hidden_size: int) -> Any:
    import torch

    if sys.byteorder != "little":
        raise RuntimeError("rows.f32 is little-endian; this converter expects a little-endian host")
    expected_floats = sample_count * row_count * hidden_size
    expected_bytes = expected_floats * 4
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path} has {len(raw)} bytes, expected {expected_bytes} "
            f"for [{sample_count}, {row_count}, {hidden_size}] f32"
        )
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(
        sample_count,
        row_count,
        hidden_size,
    )


def read_raw_tap_concat(
    corpus_dir: Path,
    manifest: dict[str, Any],
    sample_count: int,
) -> tuple[Any | None, list[int], list[int]]:
    raw_tensor = manifest.get("raw_row_tensor")
    if raw_tensor is None:
        return None, [], []
    if not isinstance(raw_tensor, dict):
        raise ValueError("manifest raw_row_tensor must be an object")
    path = corpus_dir / str(raw_tensor.get("path", "raw_rows.f32"))
    offsets = [int(value) for value in raw_tensor.get("row_offsets", [])]
    widths = [int(value) for value in raw_tensor.get("row_widths", [])]
    if len(offsets) != len(widths) + 1:
        raise ValueError("raw_row_tensor row_offsets must have row_widths + 1 entries")
    if not offsets or offsets[0] != 0:
        raise ValueError("raw_row_tensor row_offsets must start at 0")
    raw_width = int(offsets[-1])
    if raw_width <= 0:
        raise ValueError("raw_row_tensor raw width must be positive")
    expected_bytes = sample_count * raw_width * 4
    raw = path.read_bytes()
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path} has {len(raw)} bytes, expected {expected_bytes} "
            f"for [{sample_count}, {raw_width}] f32"
        )
    import torch

    tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(
        sample_count,
        raw_width,
    )
    return tensor, offsets, widths


def read_f32_vector(path: Path, length: int, name: str) -> Any:
    import torch

    raw = path.read_bytes()
    expected_bytes = length * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"{name} has {len(raw)} bytes, expected {expected_bytes}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def load_draft_token_ids(manifest_path: str | None, corpus_manifest: dict[str, Any]) -> list[int]:
    if manifest_path:
        manifest = read_json(Path(manifest_path))
        return [int(token) for token in manifest.get("topology", {}).get("draft_token_ids", [])]
    return [
        int(token)
        for token in corpus_manifest.get("topology", {}).get("draft_token_ids", [])
    ]


def build_tensors(
    *,
    rows: list[dict[str, Any]],
    cur_in: Any,
    raw_tap_concat: Any | None,
    raw_tap_offsets: list[int],
    raw_tap_widths: list[int],
    final_norm_weight: Any,
    row_count: int,
    hidden_size: int,
    draft_token_ids: list[int],
    require_labels_in_draft_vocab: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    draft_index_by_token = {
        int(token): int(index) for index, token in enumerate(draft_token_ids)
    }
    max_top_k = max(len(row.get("proposal_top_k", {}).get("token_ids", [])) for row in rows)
    position_ids = []
    row_positions = []
    current_token_ids = []
    query_row_indices = []
    query_positions = []
    target_positions = []
    target_token_ids = []
    committed_token_ids = []
    baseline_greedy_token_ids = []
    accepted = []
    context_lengths = []
    label_draft_indices = []
    label_in_draft_vocab = []
    proposal_token_ids = []
    proposal_draft_indices = []
    proposal_logits = []
    proposal_mask = []
    context_tokens_available = all("context_tokens" in row for row in rows)
    max_context_length = (
        max(len(row["context_tokens"]) for row in rows)
        if context_tokens_available
        else 0
    )
    context_token_ids = []
    context_token_lengths = []
    for sample_index, row in enumerate(rows):
        assert_sample_index(row, sample_index)
        positions = [int(value) for value in row["position_ids"]]
        if len(positions) != row_count:
            raise ValueError(
                f"sample {sample_index} has {len(positions)} positions, expected {row_count}"
            )
        query_row_index = int(row.get("query_row_index", row_count - 1))
        if query_row_index < 0 or query_row_index >= row_count:
            raise ValueError(
                f"sample {sample_index} query_row_index={query_row_index} "
                f"is outside row_count={row_count}"
            )
        position_ids.append(positions)
        row_positions.append([int(value) for value in row["row_positions"]])
        query_row_indices.append(query_row_index)
        query_positions.append(int(row.get("query_position", positions[query_row_index])))
        target_positions.append(
            int(row.get("target_position", row["context_token_count_before"]))
        )
        current_token_ids.append(int(row["current_token"]))
        target_token = int(row["target_token"])
        target_token_ids.append(target_token)
        committed_token_ids.append(int(row["committed_token"]))
        baseline_greedy_token_ids.append(int(row["baseline_greedy_token"]))
        accepted.append(1 if bool(row["accepted"]) else 0)
        context_lengths.append(int(row["context_token_count_before"]))
        draft_index = draft_index_by_token.get(target_token, -1)
        label_draft_indices.append(draft_index)
        label_in_draft_vocab.append(1 if draft_index >= 0 else 0)
        topk = row.get("proposal_top_k", {})
        token_ids = [int(value) for value in topk.get("token_ids", [])]
        draft_indices = [int(value) for value in topk.get("draft_indices", [])]
        logits = [float(value) for value in topk.get("logits", [])]
        proposal_token_ids.append(pad_ints(token_ids, max_top_k, -1))
        proposal_draft_indices.append(pad_ints(draft_indices, max_top_k, -1))
        proposal_logits.append(pad_floats(logits, max_top_k, 0.0))
        proposal_mask.append(pad_ints([1] * len(token_ids), max_top_k, 0))
        if context_tokens_available:
            tokens = [int(value) for value in row["context_tokens"]]
            context_token_ids.append(pad_ints(tokens, max_context_length, -1))
            context_token_lengths.append(len(tokens))
    missing_labels = sum(1 for value in label_draft_indices if value < 0)
    if require_labels_in_draft_vocab and missing_labels:
        raise ValueError(f"{missing_labels} target labels are missing from draft_token_ids")
    row_stage_ids = [int(value) for value in rows[0]["row_stage_ids"]]
    row_hf_indices_flat, row_hf_indices_offsets = flatten_row_hf_indices(
        rows[0]["row_hf_indices"]
    )
    tensors = {
        "cur_in": cur_in.contiguous(),
        "final_norm_weight": final_norm_weight.contiguous(),
        "position_ids": torch.tensor(position_ids, dtype=torch.long),
        "row_positions": torch.tensor(row_positions, dtype=torch.long),
        "row_i_stages": torch.tensor(row_stage_ids, dtype=torch.long),
        "row_hf_indices_flat": torch.tensor(row_hf_indices_flat, dtype=torch.long),
        "row_hf_indices_offsets": torch.tensor(row_hf_indices_offsets, dtype=torch.long),
        "query_row_indices": torch.tensor(query_row_indices, dtype=torch.long),
        "query_positions": torch.tensor(query_positions, dtype=torch.long),
        "target_positions": torch.tensor(target_positions, dtype=torch.long),
        "current_token_ids": torch.tensor(current_token_ids, dtype=torch.long),
        "target_token_ids": torch.tensor(target_token_ids, dtype=torch.long),
        "label_draft_indices": torch.tensor(label_draft_indices, dtype=torch.long),
        "label_in_draft_vocab": torch.tensor(label_in_draft_vocab, dtype=torch.long),
        "committed_token_ids": torch.tensor(committed_token_ids, dtype=torch.long),
        "baseline_greedy_token_ids": torch.tensor(
            baseline_greedy_token_ids,
            dtype=torch.long,
        ),
        "accepted": torch.tensor(accepted, dtype=torch.long),
        "context_lengths": torch.tensor(context_lengths, dtype=torch.long),
        "proposal_topk_token_ids": torch.tensor(proposal_token_ids, dtype=torch.long),
        "proposal_topk_draft_indices": torch.tensor(
            proposal_draft_indices,
            dtype=torch.long,
        ),
        "proposal_topk_logits": torch.tensor(proposal_logits, dtype=torch.float32),
        "proposal_topk_mask": torch.tensor(proposal_mask, dtype=torch.long),
    }
    if context_tokens_available:
        tensors.update(
            {
                "context_token_ids": torch.tensor(context_token_ids, dtype=torch.long),
                "context_token_lengths": torch.tensor(context_token_lengths, dtype=torch.long),
            }
        )
    if raw_tap_concat is not None:
        if len(raw_tap_widths) != row_count:
            raise ValueError(
                f"raw_tap_widths has {len(raw_tap_widths)} rows, expected {row_count}"
            )
        tensors.update(
            {
                "raw_tap_concat": raw_tap_concat.contiguous(),
                "raw_tap_offsets": torch.tensor(raw_tap_offsets, dtype=torch.long),
                "raw_tap_widths": torch.tensor(raw_tap_widths, dtype=torch.long),
            }
        )
    summary = {
        "sample_count": len(rows),
        "row_count": row_count,
        "hidden_size": hidden_size,
        "raw_tap_concat_available": raw_tap_concat is not None,
        "raw_tap_width": raw_tap_offsets[-1] if raw_tap_offsets else None,
        "draft_vocab_size": len(draft_token_ids),
        "labels_in_draft_vocab": len(rows) - missing_labels,
        "labels_missing_from_draft_vocab": missing_labels,
        "accepted_count": sum(accepted),
        "acceptance_rate": sum(accepted) / len(rows),
        "context_tokens_available": context_tokens_available,
        "max_context_length": max_context_length if context_tokens_available else None,
        "target_logits_available": False,
        "paper_kl_training_ready": False,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    return tensors, summary


def assert_sample_index(row: dict[str, Any], expected: int) -> None:
    actual = int(row["sample_index"])
    if actual != expected:
        raise ValueError(f"sample index gap: got {actual}, expected {expected}")


def flatten_row_hf_indices(rows: list[list[int]]) -> tuple[list[int], list[int]]:
    flat: list[int] = []
    offsets = [0]
    for row in rows:
        flat.extend(int(value) for value in row)
        offsets.append(len(flat))
    return flat, offsets


def pad_ints(values: list[int], length: int, fill: int) -> list[int]:
    return values + [fill] * (length - len(values))


def pad_floats(values: list[float], length: int, fill: float) -> list[float]:
    return values + [fill] * (length - len(values))


if __name__ == "__main__":
    main()
