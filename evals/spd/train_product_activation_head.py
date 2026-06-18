#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Fine-tune an SPD head on captured product activations and teacher logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hf_train_eval_qwen06 import patch_reference_for_transformers


PRODUCT_SCHEMA = "skippy-spd-product-activation-safetensors/v1"
TEACHER_SCHEMA = "skippy-spd-product-activation-teacher-logits/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing SPD speculation head on product-captured cur_in "
            "rows with precomputed HF teacher logits."
        )
    )
    parser.add_argument("--reference-dir", required=True, help="Reference SPD repo checkout")
    parser.add_argument("--checkpoint", required=True, help="Input speculation_head_final.pt")
    parser.add_argument("--product-corpus", required=True, help="Product corpus safetensors")
    parser.add_argument("--teacher-logits", required=True, help="Teacher logits safetensors")
    parser.add_argument("--out-checkpoint", required=True, help="Output speculation head checkpoint")
    parser.add_argument("--base-model-path", help="Override checkpoint config base_model_path")
    parser.add_argument("--summary-json", help="Optional JSON summary output path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1.0,
        help="Weight for KL loss against teacher logits.",
    )
    parser.add_argument(
        "--hard-label-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for cross-entropy against product native target labels. "
            "Samples whose target token is outside the draft vocab are ignored."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--model-torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--trust-remote-code",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.kl_weight < 0.0:
        raise ValueError("--kl-weight must be non-negative")
    if args.hard_label_weight < 0.0:
        raise ValueError("--hard-label-weight must be non-negative")
    if args.kl_weight == 0.0 and args.hard_label_weight == 0.0:
        raise ValueError("at least one of --kl-weight or --hard-label-weight must be positive")
    patch_reference_checkout(Path(args.reference_dir))
    sys.path.insert(0, str(Path(args.reference_dir)))

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from pipeline_inference import (  # type: ignore[import-not-found]
        _infer_pipeline_kind,
        _read_spec_config,
        build_pipeline_from_spec_ckpt,
    )

    product_tensors = load_file(args.product_corpus)
    teacher_tensors = load_file(args.teacher_logits)
    product_metadata = read_safetensors_metadata(Path(args.product_corpus))
    teacher_metadata = read_safetensors_metadata(Path(args.teacher_logits))
    validate_metadata(product_metadata, PRODUCT_SCHEMA, args.product_corpus)
    validate_metadata(teacher_metadata, TEACHER_SCHEMA, args.teacher_logits)
    validate_product_convention(product_metadata, args.product_corpus)
    validate_tensor_shapes(product_tensors, teacher_tensors)
    validate_sample_alignment(product_tensors, teacher_tensors)

    device = resolve_device(args.device)
    model_dtype = resolve_model_dtype(args.model_torch_dtype, device)
    checkpoint_path = Path(args.checkpoint)
    spec_cfg = _read_spec_config(str(checkpoint_path))
    _infer_pipeline_kind(spec_cfg)
    base_model_path = args.base_model_path or str(spec_cfg["base_model_path"])
    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs).to(device)
    base_model.eval()
    pipeline = build_pipeline_from_spec_ckpt(
        base_model,
        str(checkpoint_path),
        spec_cfg,
        map_location="cpu",
    ).to(device)
    pipeline.base_model.requires_grad_(False)
    pipeline.speculation_module.train()

    cur_in = product_tensors["cur_in"].to(device=device, dtype=model_dtype)
    position_ids = product_tensors["position_ids"].to(device=device)
    teacher_logits = teacher_tensors["teacher_logits"].to(device=device, dtype=torch.float32)
    teacher_argmax = teacher_logits.argmax(dim=-1)
    label_draft_indices = product_tensors.get("label_draft_indices")
    if args.hard_label_weight > 0.0 and label_draft_indices is None:
        raise ValueError(
            "--hard-label-weight requires label_draft_indices in the product corpus"
        )
    if label_draft_indices is not None:
        label_draft_indices = label_draft_indices.to(device=device, dtype=torch.long)
    optimizer = torch.optim.AdamW(
        pipeline.speculation_module.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    sample_count = int(cur_in.shape[0])
    losses: list[float] = []
    accs: list[float] = []
    hard_label_accs: list[float] = []
    steps = 0
    for _epoch in range(int(args.epochs)):
        for batch_idx in batch_indices(sample_count, int(args.batch_size)):
            batch_cur_in = cur_in[batch_idx]
            batch_position_ids = position_ids[batch_idx]
            batch_teacher = teacher_logits[batch_idx]
            proc = pipeline.speculation_module.forward_inference_g1_only_with_rotary(
                batch_cur_in,
                batch_position_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
            )
            final_hidden = pipeline.final_norm(proc)
            student_logits = pipeline.speculation_module.lm_head(final_hidden[:, -1:, :])
            student_logits = student_logits.squeeze(1).float()
            if student_logits.shape != batch_teacher.shape:
                raise ValueError(
                    f"student logits shape {tuple(student_logits.shape)} does not match "
                    f"teacher logits shape {tuple(batch_teacher.shape)}"
                )
            loss_parts = []
            if args.kl_weight > 0.0:
                teacher_probs = F.softmax(batch_teacher / float(args.temperature), dim=-1)
                student_log_probs = F.log_softmax(
                    student_logits / float(args.temperature),
                    dim=-1,
                )
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                kl_loss = kl_loss * (float(args.temperature) ** 2)
                loss_parts.append(float(args.kl_weight) * kl_loss)
            if args.hard_label_weight > 0.0:
                assert label_draft_indices is not None
                batch_labels = label_draft_indices[batch_idx]
                in_scope = batch_labels >= 0
                if bool(in_scope.any()):
                    hard_label_loss = F.cross_entropy(
                        student_logits[in_scope],
                        batch_labels[in_scope],
                    )
                    loss_parts.append(float(args.hard_label_weight) * hard_label_loss)
            if not loss_parts:
                continue
            loss = sum(loss_parts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = student_logits.argmax(dim=-1)
                acc = (pred == teacher_argmax[batch_idx]).float().mean()
                if label_draft_indices is not None:
                    batch_labels = label_draft_indices[batch_idx]
                    in_scope = batch_labels >= 0
                    if bool(in_scope.any()):
                        hard_acc = (pred[in_scope] == batch_labels[in_scope]).float().mean()
                        hard_label_accs.append(float(hard_acc.detach().cpu()))
            losses.append(float(loss.detach().cpu()))
            accs.append(float(acc.detach().cpu()))
            steps += 1
            if args.max_steps > 0 and steps >= int(args.max_steps):
                break
        if args.max_steps > 0 and steps >= int(args.max_steps):
            break

    out_checkpoint = Path(args.out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_speculation_head(str(out_checkpoint))
    summary = {
        "schema": "skippy-spd-product-activation-train-summary/v1",
        "input_checkpoint": str(checkpoint_path),
        "out_checkpoint": str(out_checkpoint),
        "product_corpus": str(args.product_corpus),
        "teacher_logits": str(args.teacher_logits),
        "cur_in_convention": product_metadata.get("cur_in_convention"),
        "teacher_source": teacher_metadata.get("teacher_source"),
        "native_product_teacher_logits": teacher_metadata.get("native_product_teacher_logits"),
        "base_model_path": base_model_path,
        "device": str(device),
        "model_torch_dtype": str(model_dtype).replace("torch.", ""),
        "sample_count": sample_count,
        "batch_size": int(args.batch_size),
        "epochs_requested": int(args.epochs),
        "steps_completed": steps,
        "learning_rate": float(args.learning_rate),
        "temperature": float(args.temperature),
        "kl_weight": float(args.kl_weight),
        "hard_label_weight": float(args.hard_label_weight),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "initial_argmax_acc": accs[0] if accs else None,
        "final_argmax_acc": accs[-1] if accs else None,
        "initial_hard_label_acc": hard_label_accs[0] if hard_label_accs else None,
        "final_hard_label_acc": hard_label_accs[-1] if hard_label_accs else None,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def patch_reference_checkout(reference_dir: Path) -> None:
    patch_reference_for_transformers(reference_dir)


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    import json as json_module

    with path.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json_module.loads(handle.read(header_len))
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


def validate_metadata(metadata: dict[str, str], schema: str, path: str) -> None:
    if metadata.get("schema") != schema:
        raise ValueError(f"{path} has schema {metadata.get('schema')!r}, expected {schema!r}")


def validate_product_convention(metadata: dict[str, str], path: str) -> None:
    convention = metadata.get("cur_in_convention")
    if convention != "terminal_final_normed_cur_in":
        raise ValueError(
            f"{path} has cur_in_convention {convention!r}, expected "
            "'terminal_final_normed_cur_in'"
        )


def validate_tensor_shapes(product: dict[str, Any], teacher: dict[str, Any]) -> None:
    required_product = {"cur_in", "position_ids"}
    required_teacher = {"teacher_logits"}
    missing_product = sorted(required_product.difference(product))
    missing_teacher = sorted(required_teacher.difference(teacher))
    if missing_product:
        raise ValueError(f"product corpus missing tensors: {missing_product}")
    if missing_teacher:
        raise ValueError(f"teacher logits missing tensors: {missing_teacher}")
    if int(product["cur_in"].shape[0]) != int(teacher["teacher_logits"].shape[0]):
        raise ValueError(
            "product corpus sample count does not match teacher logits sample count"
        )
    if int(product["cur_in"].shape[0]) == 0:
        raise ValueError("product corpus is empty")


def validate_sample_alignment(product: dict[str, Any], teacher: dict[str, Any]) -> None:
    for product_name, teacher_name in [
        ("query_positions", "query_positions"),
        ("target_positions", "target_positions"),
        ("query_row_indices", "query_row_indices"),
    ]:
        if product_name not in product or teacher_name not in teacher:
            continue
        product_tensor = product[product_name].cpu()
        teacher_tensor = teacher[teacher_name].cpu()
        if product_tensor.shape != teacher_tensor.shape:
            raise ValueError(
                f"{product_name} shape {tuple(product_tensor.shape)} does not match "
                f"{teacher_name} shape {tuple(teacher_tensor.shape)}"
            )
        if bool((product_tensor != teacher_tensor).any()):
            raise ValueError(f"{product_name} does not match teacher {teacher_name}")


def resolve_device(value: str) -> Any:
    import torch

    if value == "cuda":
        return torch.device("cuda")
    if value == "mps":
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_dtype(value: str, device: Any) -> Any:
    import torch

    if value != "auto":
        return torch_dtype(value)
    if device.type in {"cuda", "mps"}:
        return torch.bfloat16
    return torch.float32


def torch_dtype(value: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def batch_indices(sample_count: int, batch_size: int) -> list[slice]:
    return [
        slice(start, min(start + batch_size, sample_count))
        for start in range(0, sample_count, batch_size)
    ]


if __name__ == "__main__":
    main()
