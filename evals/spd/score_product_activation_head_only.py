#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Score an SPD head from native product rows without loading base weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from train_product_activation_head import (
    PRODUCT_SCHEMA,
    TEACHER_SCHEMA,
    batch_indices,
    batch_product_cur_in,
    patch_reference_checkout,
    read_safetensors_metadata,
    resolve_device,
    resolve_model_dtype,
    validate_input_mode,
    validate_metadata,
    validate_product_convention,
    validate_sample_alignment,
    validate_tensor_shapes,
)
from train_product_activation_head_only import (
    HeadOnlyPipeline,
    build_rotary_embedding,
    forward_head_logits,
    normalize_dense_sidecar_config,
    product_row_hf_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a freshly trained SPD speculation head on product-captured raw "
            "tap rows and native verifier logits. This path loads AutoConfig only, "
            "not the base model weights."
        )
    )
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--product-corpus", required=True)
    parser.add_argument("--teacher-logits", required=True)
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--model-torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--summary-json")
    parser.add_argument(
        "--trust-remote-code",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    patch_reference_checkout(Path(args.reference_dir))
    sys.path.insert(0, str(Path(args.reference_dir)))

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from pipeline_model import (  # type: ignore[import-not-found]
        SpeculationHeadTransformer,
        _decoder_relevant_config,
        _get_apply_rotary_pos_emb,
    )

    checkpoint, state_dict, checkpoint_config = load_checkpoint(Path(args.checkpoint), torch)
    product_tensors = load_file(args.product_corpus)
    teacher_tensors = load_file(args.teacher_logits)
    product_metadata = read_safetensors_metadata(Path(args.product_corpus))
    teacher_metadata = read_safetensors_metadata(Path(args.teacher_logits))
    validate_metadata(product_metadata, PRODUCT_SCHEMA, args.product_corpus)
    validate_metadata(teacher_metadata, TEACHER_SCHEMA, args.teacher_logits)
    validate_product_convention(product_metadata, args.product_corpus)
    validate_tensor_shapes(product_tensors, teacher_tensors)
    validate_sample_alignment(product_tensors, teacher_tensors)
    validate_input_mode("fresh", "raw", product_metadata, args.product_corpus)

    product_tensors, teacher_tensors, row_selection = select_rows(
        product_tensors,
        teacher_tensors,
        row_start=int(args.row_start),
        row_limit=int(args.row_limit),
    )
    device = resolve_device(args.device)
    model_dtype = resolve_model_dtype(args.model_torch_dtype, device)
    base_model_path = args.base_model_path or str(checkpoint_config.get("base_model_path") or "")
    if not base_model_path:
        raise ValueError("--base-model-path is required when checkpoint config lacks base_model_path")
    hf_config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    dec_cfg = _decoder_relevant_config(hf_config)
    normalize_dense_sidecar_config(dec_cfg, args.attn_implementation)
    shallow_rows = product_row_hf_indices(product_tensors)
    draft_token_ids = teacher_tensors["teacher_logit_token_ids"].cpu().tolist()
    hidden_size = int(product_tensors["final_norm_weight"].shape[0])
    if int(getattr(dec_cfg, "hidden_size")) != hidden_size:
        raise ValueError(
            f"AutoConfig hidden_size={getattr(dec_cfg, 'hidden_size')} does not match "
            f"product final_norm_weight length {hidden_size}"
        )
    rotary = build_rotary_embedding(dec_cfg, device)
    head = SpeculationHeadTransformer(
        dec_cfg,
        model_dtype,
        device,
        base_rotary_emb=rotary,
        apply_rotary_fn=_get_apply_rotary_pos_emb(hf_config),
        stage_feature_hf_indices=shallow_rows,
        num_spec_layers=int(checkpoint_config["num_spec_layers"]),
        init_weights_from_base_layer_indices=None,
        base_decoder_layers=None,
        draft_vocab_size=len(draft_token_ids),
    ).to(device)
    head.load_state_dict(state_dict, strict=True)
    head.eval()

    metrics = score_rows(
        args=args,
        head=head,
        product_tensors=product_tensors,
        teacher_tensors=teacher_tensors,
        device=device,
        model_dtype=model_dtype,
        torch=torch,
    )
    summary = {
        "schema": "skippy-spd-head-only-score-summary/v1",
        "base_model_load": "skipped",
        "checkpoint": str(args.checkpoint),
        "product_corpus": str(args.product_corpus),
        "teacher_logits": str(args.teacher_logits),
        "base_model_path": base_model_path,
        "input_mode": "raw",
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "row_start": row_selection["row_start"],
        "row_limit": row_selection["row_limit"],
        "row_end_exclusive": row_selection["row_end_exclusive"],
        "teacher_source": teacher_metadata.get("teacher_source"),
        "native_product_teacher_logits": teacher_metadata.get("native_product_teacher_logits"),
        "draft_vocab_size": len(draft_token_ids),
        **metrics,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    _ = checkpoint


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.row_start < 0:
        raise ValueError("--row-start must be non-negative")
    if args.row_limit < 0:
        raise ValueError("--row-limit must be non-negative")


def load_checkpoint(path: Path, torch_module: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch_module.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} did not contain a dict checkpoint")
    state_dict = checkpoint.get("state_dict")
    config = checkpoint.get("config")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path} missing checkpoint state_dict")
    if not isinstance(config, dict):
        raise ValueError(f"{path} missing checkpoint config")
    return checkpoint, state_dict, config


def select_rows(
    product_tensors: dict[str, Any],
    teacher_tensors: dict[str, Any],
    *,
    row_start: int,
    row_limit: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int | None]]:
    sample_count = int(product_tensors["cur_in"].shape[0])
    if row_start > sample_count:
        raise ValueError(f"--row-start {row_start} exceeds sample count {sample_count}")
    row_end = sample_count if row_limit == 0 else min(sample_count, row_start + row_limit)
    row_slice = slice(row_start, row_end)
    return (
        slice_sample_tensors(product_tensors, row_slice, sample_count),
        slice_sample_tensors(teacher_tensors, row_slice, sample_count),
        {
            "row_start": row_start,
            "row_limit": row_limit or None,
            "row_end_exclusive": row_end,
        },
    )


def slice_sample_tensors(tensors: dict[str, Any], row_slice: slice, sample_count: int) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for name, tensor in tensors.items():
        if hasattr(tensor, "shape") and len(tensor.shape) > 0 and int(tensor.shape[0]) == sample_count:
            selected[name] = tensor[row_slice]
        else:
            selected[name] = tensor
    return selected


def score_rows(
    *,
    args: argparse.Namespace,
    head: Any,
    product_tensors: dict[str, Any],
    teacher_tensors: dict[str, Any],
    device: Any,
    model_dtype: Any,
    torch: Any,
) -> dict[str, Any]:
    projected_cur_in = product_tensors["cur_in"].to(device=device, dtype=model_dtype)
    raw_tap_concat = product_tensors["raw_tap_concat"].to(device=device, dtype=model_dtype)
    raw_tap_offsets = product_tensors["raw_tap_offsets"].cpu().tolist()
    raw_row_stage_ids = product_tensors["row_i_stages"].cpu().tolist()
    position_ids = product_tensors["position_ids"].to(device=device)
    final_norm_weight = product_tensors["final_norm_weight"].to(device=device, dtype=model_dtype)
    teacher_logits = teacher_tensors["teacher_logits"].to(device=device, dtype=torch.float32)
    teacher_argmax = teacher_logits.argmax(dim=-1)
    label_draft_indices = product_tensors["label_draft_indices"].to(device=device, dtype=torch.long)
    sample_count = int(projected_cur_in.shape[0])
    top_k = min(int(args.top_k), int(teacher_logits.shape[1]))
    pipeline_view = HeadOnlyPipeline(head)
    teacher_top1 = 0
    teacher_topk = 0
    hard_label_top1 = 0
    hard_label_topk = 0
    hard_label_count = 0
    with torch.no_grad():
        for batch_idx in batch_indices(sample_count, int(args.batch_size)):
            batch_cur_in = batch_product_cur_in(
                pipeline_view,
                "raw",
                projected_cur_in,
                raw_tap_concat,
                raw_tap_offsets,
                raw_row_stage_ids,
                batch_idx,
            )
            logits = forward_head_logits(
                head,
                batch_cur_in,
                position_ids[batch_idx],
                final_norm_weight,
            )
            pred = logits.argmax(dim=-1)
            batch_teacher_argmax = teacher_argmax[batch_idx]
            teacher_top1 += int((pred == batch_teacher_argmax).sum().item())
            topk_indices = logits.topk(top_k, dim=-1).indices
            teacher_topk += int((topk_indices == batch_teacher_argmax.unsqueeze(-1)).any(dim=-1).sum().item())
            batch_labels = label_draft_indices[batch_idx]
            in_scope = batch_labels >= 0
            if bool(in_scope.any()):
                hard_label_count += int(in_scope.sum().item())
                hard_label_top1 += int((pred[in_scope] == batch_labels[in_scope]).sum().item())
                hard_label_topk += int(
                    (topk_indices[in_scope] == batch_labels[in_scope].unsqueeze(-1)).any(dim=-1).sum().item()
                )
    return {
        "sample_count": sample_count,
        "teacher_top1_matches": teacher_top1,
        "teacher_top1_rate": safe_rate(teacher_top1, sample_count),
        "teacher_topk_matches": teacher_topk,
        "teacher_topk_rate": safe_rate(teacher_topk, sample_count),
        "hard_label_in_scope": hard_label_count,
        "hard_label_top1_matches": hard_label_top1,
        "hard_label_top1_rate": safe_rate(hard_label_top1, hard_label_count),
        "hard_label_topk_matches": hard_label_topk,
        "hard_label_topk_rate": safe_rate(hard_label_topk, hard_label_count),
        "serving_target_in_draft_scope": hard_label_count,
        "serving_target_top1_matches": hard_label_top1,
        "serving_target_top1_rate": safe_rate(hard_label_top1, hard_label_count),
        "serving_target_topk_matches": hard_label_topk,
        "serving_target_topk_rate": safe_rate(hard_label_topk, hard_label_count),
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    main()
