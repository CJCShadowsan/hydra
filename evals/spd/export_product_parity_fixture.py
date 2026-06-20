#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Export a Rust parity fixture from native product SPD rows.

This is the native-package-fresh companion to export_parity_fixture.py. It does
not load base-model weights. It loads a trained head-only checkpoint plus a
product activation corpus, runs the Python head on one selected native row, and
writes the existing `skippy-spd-parity-fixture/v1` format consumed by
`skippy-bench spd-fixture-parity`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from train_product_activation_head import (
    PRODUCT_SCHEMA,
    TEACHER_SCHEMA,
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
    normalize_dense_sidecar_config,
    product_row_hf_indices,
    qwen_rms_norm,
)
from score_product_activation_head_only import load_checkpoint, select_rows


FIXTURE_SCHEMA = "skippy-spd-parity-fixture/v1"
TAP_INPUT_SCHEMA = "skippy-spd-tap-input-fixture/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a true Rust/Python SPD parity fixture from native product "
            "activation rows and a head-only checkpoint."
        )
    )
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--product-corpus", required=True)
    parser.add_argument("--teacher-logits", required=True)
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--model-torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    patch_reference_checkout(Path(args.reference_dir))
    sys.path.insert(0, str(Path(args.reference_dir)))

    import torch
    from safetensors.torch import load_file, save_file
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
        row_start=int(args.row_index),
        row_limit=1,
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
    draft_token_ids = [int(value) for value in teacher_tensors["teacher_logit_token_ids"].cpu().tolist()]
    validate_draft_token_ids(checkpoint_config, draft_token_ids)
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

    fixture_tensors, fixture_summary = build_fixture_tensors(
        args=args,
        head=head,
        product_tensors=product_tensors,
        teacher_tensors=teacher_tensors,
        draft_token_ids=draft_token_ids,
        device=device,
        model_dtype=model_dtype,
        torch=torch,
    )
    ensure_fixture_tensors_finite(fixture_tensors, torch)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": FIXTURE_SCHEMA,
        "source_schema": PRODUCT_SCHEMA,
        "source_product_corpus": str(args.product_corpus),
        "teacher_logits": str(args.teacher_logits),
        "base_model_path": base_model_path,
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "checkpoint_version": str(int(checkpoint_config.get("version", 0))),
        "num_stages": str(int(checkpoint_config["num_stages"])),
        "num_spec_layers": str(int(checkpoint_config["num_spec_layers"])),
        "row_index": str(int(args.row_index)),
        "row_start": str(row_selection["row_start"]),
        "row_end_exclusive": str(row_selection["row_end_exclusive"]),
        "tap_input_schema": TAP_INPUT_SCHEMA,
        "use_deepest": str(bool(checkpoint_config.get("trained_with_use_deepest", False))).lower(),
        "product_parity_fixture": "true",
        "cached_reference": "false",
    }
    save_file(fixture_tensors, out_path, metadata=metadata)
    summary = {
        "schema": FIXTURE_SCHEMA,
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
        "sha256": file_sha256(out_path),
        "checkpoint": str(args.checkpoint),
        "product_corpus": str(args.product_corpus),
        "teacher_logits": str(args.teacher_logits),
        "base_model_path": base_model_path,
        "row_index": int(args.row_index),
        "top_k": int(args.top_k),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in fixture_tensors.items()},
        **fixture_summary,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    _ = checkpoint


def validate_args(args: argparse.Namespace) -> None:
    if args.row_index < 0:
        raise ValueError("--row-index must be non-negative")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")


def validate_draft_token_ids(checkpoint_config: dict[str, Any], teacher_token_ids: list[int]) -> None:
    checkpoint_token_ids = checkpoint_config.get("draft_token_ids")
    if checkpoint_token_ids is None:
        raise ValueError("checkpoint config is missing draft_token_ids")
    checkpoint_token_ids = [int(value) for value in checkpoint_token_ids]
    if checkpoint_token_ids != teacher_token_ids:
        first_mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(checkpoint_token_ids, teacher_token_ids))
                if left != right
            ),
            None,
        )
        raise ValueError(
            "checkpoint draft_token_ids do not match teacher_logit_token_ids "
            f"(checkpoint len={len(checkpoint_token_ids)}, teacher len={len(teacher_token_ids)}, "
            f"first_mismatch={first_mismatch})"
        )


def build_fixture_tensors(
    *,
    args: argparse.Namespace,
    head: Any,
    product_tensors: dict[str, Any],
    teacher_tensors: dict[str, Any],
    draft_token_ids: list[int],
    device: Any,
    model_dtype: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projected_cur_in = product_tensors["cur_in"].to(device=device, dtype=model_dtype)
    raw_tap_concat = product_tensors["raw_tap_concat"].to(device=device, dtype=model_dtype)
    raw_tap_offsets = [int(value) for value in product_tensors["raw_tap_offsets"].cpu().tolist()]
    raw_row_stage_ids = [int(value) for value in product_tensors["row_i_stages"].cpu().tolist()]
    position_ids = product_tensors["position_ids"].to(device=device)
    final_norm_weight = product_tensors["final_norm_weight"].to(device=device, dtype=model_dtype)

    pipeline_view = HeadOnlyPipeline(head)
    batch_idx = torch.arange(0, 1, dtype=torch.long, device=device)
    cur_in = batch_product_cur_in(
        pipeline_view,
        "raw",
        projected_cur_in,
        raw_tap_concat,
        raw_tap_offsets,
        raw_row_stage_ids,
        batch_idx,
    )
    proc, layer_inputs, layer_queries = forward_with_trace(
        head=head,
        cur_in=cur_in,
        position_ids=position_ids,
        torch=torch,
    )
    final_hidden = qwen_rms_norm(proc, final_norm_weight)
    logits = head.lm_head(final_hidden[:, -1:, :]).float()
    top_k = min(int(args.top_k), int(logits.shape[-1]))
    values, draft_indices = torch.topk(logits[0, 0], k=top_k)
    draft_token_tensor = torch.tensor(draft_token_ids, device=device, dtype=torch.long)
    token_ids = draft_token_tensor[draft_indices]

    label_draft_index = int(product_tensors["label_draft_indices"][0].item())
    target_token = int(product_tensors["target_token_ids"][0].item())
    predicted_token = int(token_ids[0].item())
    teacher_logits = teacher_tensors["teacher_logits"].to(device=device, dtype=torch.float32)
    teacher_argmax = int(teacher_logits[0].argmax().item())
    teacher_argmax_token = int(draft_token_tensor[teacher_argmax].item())

    tensors: dict[str, Any] = {
        "cur_in": cur_in.detach().cpu().float().contiguous(),
        "final_norm_weight": final_norm_weight.detach().cpu().float().contiguous(),
        "position_ids": position_ids[0].detach().cpu().long().contiguous(),
        "prompt_input_ids": torch.zeros((1, 1), dtype=torch.long),
        "row_i_stages": product_tensors["row_i_stages"].detach().cpu().long().contiguous(),
        "row_positions": product_tensors["row_positions"][0].detach().cpu().long().contiguous(),
        "python_logits": logits.detach().cpu().float().contiguous(),
        "python_spec_query": proc.detach().cpu().float().contiguous(),
        "python_final_hidden": final_hidden.detach().cpu().float().contiguous(),
        "python_topk_draft_indices": draft_indices.detach().cpu().long().contiguous(),
        "python_topk_logits": values.detach().cpu().float().contiguous(),
        "python_topk_token_ids": token_ids.detach().cpu().long().contiguous(),
    }
    for layer_index, value in enumerate(layer_inputs):
        tensors[f"python_layer_{layer_index}_full_in"] = value.float().contiguous()
    for layer_index, value in enumerate(layer_queries):
        tensors[f"python_layer_{layer_index}_query"] = value.float().contiguous()
    row_taps = split_raw_tap_rows(raw_tap_concat[0], raw_tap_offsets, torch)
    row_hf_indices = product_row_hf_indices(product_tensors)
    for row_index, (tap_row, hf_indices) in enumerate(zip(row_taps, row_hf_indices)):
        tensors[f"tap_row_{row_index}_concat"] = tap_row.detach().cpu().float().contiguous()
        tensors[f"tap_row_{row_index}_hf_indices"] = torch.tensor(hf_indices, dtype=torch.long)

    summary = {
        "cur_in_shape": list(tensors["cur_in"].shape),
        "draft_vocab_size": len(draft_token_ids),
        "top_token_ids": [int(value) for value in token_ids.detach().cpu().tolist()],
        "top_draft_indices": [int(value) for value in draft_indices.detach().cpu().tolist()],
        "target_token": target_token,
        "target_in_draft_scope": label_draft_index >= 0,
        "target_draft_index": label_draft_index if label_draft_index >= 0 else None,
        "predicted_token": predicted_token,
        "predicted_target_match": predicted_token == target_token,
        "teacher_argmax_draft_index": teacher_argmax,
        "teacher_argmax_token": teacher_argmax_token,
        "predicted_teacher_match": int(draft_indices[0].item()) == teacher_argmax,
    }
    return tensors, summary


def forward_with_trace(
    *,
    head: Any,
    cur_in: Any,
    position_ids: Any,
    torch: Any,
) -> tuple[Any, list[Any], list[Any]]:
    layer_inputs: list[Any] = []
    layer_queries: list[Any] = []
    pre_handles = [
        layer.register_forward_pre_hook(
            lambda _module, inputs: layer_inputs.append(inputs[0].detach().cpu().contiguous())
        )
        for layer in head.spec_layers
    ]
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: layer_queries.append(
                output[:, -1:, :].detach().cpu().contiguous()
            )
        )
        for layer in head.spec_layers
    ]
    try:
        with torch.no_grad():
            proc = head.forward_inference_g1_only_with_rotary(
                cur_in,
                position_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
        for handle in pre_handles:
            handle.remove()
    return proc, layer_inputs, layer_queries


def split_raw_tap_rows(raw_row: Any, offsets: list[int], torch: Any) -> list[Any]:
    if len(offsets) < 2:
        raise ValueError("raw_tap_offsets must contain at least two offsets")
    rows = []
    for row_index in range(len(offsets) - 1):
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        if end < start:
            raise ValueError("raw_tap_offsets must be non-decreasing")
        rows.append(raw_row[start:end].detach().clone())
    _ = torch
    return rows


def ensure_fixture_tensors_finite(tensors: dict[str, Any], torch: Any) -> None:
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            continue
        finite = torch.isfinite(tensor.float())
        if bool(finite.all()):
            continue
        finite_count = int(finite.sum().item())
        raise RuntimeError(
            f"fixture tensor {name!r} contains non-finite values: "
            f"{finite_count}/{tensor.numel()} finite"
        )


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
