#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.0.0",
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Export a real SPD Python forward fixture for Rust parity work."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional


FIXTURE_SCHEMA = "skippy-spd-parity-fixture/v1"
TAP_INPUT_SCHEMA = "skippy-spd-tap-input-fixture/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real hidden-state SPD inputs and Python top-k outputs"
    )
    parser.add_argument("--reference-dir", required=True, help="Reference SPD repo checkout")
    parser.add_argument("--checkpoint", required=True, help="Input speculation_head_final.pt")
    parser.add_argument("--base-model-path", required=True, help="HF id or local base model path")
    parser.add_argument("--out", required=True, help="Output safetensors fixture path")
    parser.add_argument("--prompt", default="What is 24*36? Answer briefly.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--newest-pos",
        type=int,
        default=-1,
        help="Prompt position used as the newest speculative row; -1 means last prompt token.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = Path(args.reference_dir)
    patch_reference_checkout(reference_dir)
    sys.path.insert(0, str(reference_dir))

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipeline_inference import (  # type: ignore[import-not-found]
        _infer_pipeline_kind,
        _read_spec_config,
        build_pipeline_from_spec_ckpt,
    )

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    spec_cfg = _read_spec_config(str(checkpoint_path))
    _infer_pipeline_kind(spec_cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if device.type == "cuda":
        model_kwargs["device_map"] = {"": 0}
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)
    if device.type != "cuda":
        base_model = base_model.to(device)
    base_model.eval()

    map_location = "cuda" if device.type == "cuda" else "cpu"
    pipeline = build_pipeline_from_spec_ckpt(
        base_model,
        str(checkpoint_path),
        spec_cfg,
        map_location=map_location,
    )
    pipeline.eval()

    fixture = build_fixture(
        pipeline=pipeline,
        tokenizer=tokenizer,
        prompt=args.prompt,
        top_k=args.top_k,
        newest_pos_arg=args.newest_pos,
    )
    ensure_fixture_tensors_finite(fixture.tensors)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": FIXTURE_SCHEMA,
        "base_model_path": args.base_model_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_version": str(int(spec_cfg.get("version", 0))),
        "num_stages": str(int(pipeline.num_stages)),
        "num_spec_layers": str(int(pipeline.num_spec_layers)),
        "prompt_json": json.dumps(args.prompt),
        "row_kinds_json": json.dumps(fixture.row_kinds),
        "tap_input_schema": TAP_INPUT_SCHEMA,
        "tap_row_count": str(fixture.tap_row_count),
        "use_deepest": str(bool(getattr(pipeline, "trained_with_use_deepest", False))).lower(),
        "cached_reference": str(fixture.cached_prefill_row_count > 0).lower(),
        "cached_prefill_rows": str(fixture.cached_prefill_row_count),
    }
    if fixture.cached_crop_position is not None:
        metadata["cached_crop_position"] = str(fixture.cached_crop_position)
    save_file(fixture.tensors, out_path, metadata=metadata)
    print(
        json.dumps(
            {
                "fixture": str(out_path),
                "schema": FIXTURE_SCHEMA,
                "bytes": out_path.stat().st_size,
                "sha256": file_sha256(out_path),
                "cur_in_shape": list(fixture.tensors["cur_in"].shape),
                "python_logits_shape": list(fixture.tensors["python_logits"].shape),
                "cached_prefill_rows": fixture.cached_prefill_row_count,
                "top_token_ids": fixture.tensors["python_topk_token_ids"].tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def patch_reference_checkout(reference_dir: Path) -> None:
    from hf_train_eval_qwen06 import patch_reference_for_transformers

    patch_reference_for_transformers(reference_dir)


class Fixture:
    def __init__(
        self,
        tensors: dict[str, Any],
        row_kinds: list[str],
        tap_row_count: int,
        cached_prefill_row_count: int,
        cached_crop_position: Optional[int],
    ) -> None:
        self.tensors = tensors
        self.row_kinds = row_kinds
        self.tap_row_count = tap_row_count
        self.cached_prefill_row_count = cached_prefill_row_count
        self.cached_crop_position = cached_crop_position


def build_fixture(
    *,
    pipeline: Any,
    tokenizer: Any,
    prompt: str,
    top_k: int,
    newest_pos_arg: int,
) -> Fixture:
    import torch
    from transformers.cache_utils import DynamicCache

    device = next(pipeline.base_model.parameters()).device
    batch = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = batch["input_ids"].to(device)
    with torch.no_grad():
        outputs = pipeline.base_model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        completed_snaps = pipeline._extract_position_snapshots_from_hidden_states(  # noqa: SLF001
            outputs.hidden_states
        )
        n = int(pipeline.num_stages)
        newest_pos = resolve_newest_pos(input_ids.shape[1], n, newest_pos_arg)
        oldest_needed = newest_pos - n + 1
        use_deepest = bool(getattr(pipeline, "trained_with_use_deepest", False))
        rows = []
        row_positions: list[int] = []
        row_i_stages: list[int] = []
        row_kinds: list[str] = []
        tap_inputs: list[Any] = []
        tap_hf_indices: list[list[int]] = []

        for pos in range(oldest_needed, newest_pos + 1):
            i_nominal_pipe = newest_pos - pos
            if i_nominal_pipe == 0:
                token = input_ids[:, pos : pos + 1]
                tap_input = pipeline.embed_tokens(token)
                row = pipeline._build_inference_g0_row_from_hs(tap_input)  # noqa: SLF001
                i_stages = 0
                row_kind = "g0"
                hf_indices = [0]
            else:
                snap = completed_snaps[pos]
                i_stages = pipeline._choose_inference_i_stages_for_snap(  # noqa: SLF001
                    snap,
                    i_nominal_pipe,
                    use_deepest,
                    search_hi=None,
                )
                block = int(pipeline.num_stages) - int(i_stages)
                hf_indices = [
                    int(idx) for idx in pipeline.shallow_hidden_layer_indices[block]
                ]
                tap_input = torch.cat([snap[int(idx)] for idx in hf_indices], dim=-1)
                row = pipeline._build_inference_row_from_snap(snap, i_stages)  # noqa: SLF001
                row_kind = f"g{i_stages}"
            rows.append(row)
            row_positions.append(pos)
            row_i_stages.append(int(i_stages))
            row_kinds.append(row_kind)
            tap_inputs.append(tap_input)
            tap_hf_indices.append(hf_indices)

        cur_in = torch.cat(rows, dim=1)
        position_ids = torch.tensor([row_positions], device=device, dtype=torch.long)
        layer_inputs: list[Any] = []
        layer_queries: list[Any] = []
        pre_handles = [
            layer.register_forward_pre_hook(
                lambda _module, inputs: layer_inputs.append(
                    inputs[0].detach().cpu().contiguous()
                )
            )
            for layer in pipeline.speculation_module.spec_layers
        ]
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, output: layer_queries.append(
                    output[:, -1:, :].detach().cpu().contiguous()
                )
            )
            for layer in pipeline.speculation_module.spec_layers
        ]
        try:
            proc = pipeline.speculation_module.forward_inference_g1_only_with_rotary(
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
        final_hidden = pipeline.final_norm(proc)
        logits = pipeline.speculation_module.lm_head(final_hidden[:, -1:, :]).float()
        values, draft_indices = torch.topk(logits[0, 0], k=int(top_k))
        if getattr(pipeline, "_use_draft_vocab", False):
            draft_token_ids = pipeline._draft_token_ids.to(device).long()  # noqa: SLF001
            token_ids = draft_token_ids[draft_indices]
        else:
            token_ids = draft_indices

        cached_prefill_row_count = 0
        cached_crop_position: Optional[int] = None
        prefill_cur_in = None
        prefill_position_ids = None
        cached_proc = None
        cached_final_hidden = None
        cached_logits = None
        cached_values = None
        cached_draft_indices = None
        cached_token_ids = None
        prefill_cache_len = max(0, oldest_needed)
        if prefill_cache_len > 0:
            spec_past_kv = DynamicCache()
            prefill_rows = [
                pipeline._build_inference_row_from_snap(completed_snaps[pos], n)  # noqa: SLF001
                for pos in range(prefill_cache_len)
            ]
            prefill_cur_in = torch.cat(prefill_rows, dim=1)
            prefill_position_ids = torch.arange(
                prefill_cache_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            pipeline.speculation_module.forward_inference_g1_only_with_rotary(
                prefill_cur_in,
                prefill_position_ids,
                attention_mask=None,
                past_key_values=spec_past_kv,
                use_cache=True,
            )
            cached_crop_position = min(row_positions)
            pipeline._rollback_spec_kv_cache(  # noqa: SLF001
                spec_past_kv,
                cached_crop_position,
            )
            cached_proc = pipeline.speculation_module.forward_inference_g1_only_with_rotary(
                cur_in,
                position_ids,
                attention_mask=None,
                past_key_values=spec_past_kv,
                use_cache=True,
            )
            cached_final_hidden = pipeline.final_norm(cached_proc)
            cached_logits = pipeline.speculation_module.lm_head(
                cached_final_hidden[:, -1:, :]
            ).float()
            cached_values, cached_draft_indices = torch.topk(
                cached_logits[0, 0],
                k=int(top_k),
            )
            if getattr(pipeline, "_use_draft_vocab", False):
                cached_token_ids = draft_token_ids[cached_draft_indices]
            else:
                cached_token_ids = cached_draft_indices
            cached_prefill_row_count = prefill_cache_len

    tensors = {
        "cur_in": cur_in.detach().cpu().contiguous(),
        "final_norm_weight": pipeline.final_norm.weight.detach().cpu().contiguous(),
        "position_ids": position_ids.cpu().contiguous(),
        "prompt_input_ids": input_ids.cpu().contiguous(),
        "row_i_stages": torch.tensor(row_i_stages, dtype=torch.long),
        "row_positions": torch.tensor(row_positions, dtype=torch.long),
        "python_logits": logits.detach().cpu().contiguous(),
        "python_spec_query": proc.detach().cpu().contiguous(),
        "python_final_hidden": final_hidden.detach().cpu().contiguous(),
        "python_topk_draft_indices": draft_indices.detach().cpu().long().contiguous(),
        "python_topk_logits": values.detach().cpu().float().contiguous(),
        "python_topk_token_ids": token_ids.detach().cpu().long().contiguous(),
    }
    if cached_prefill_row_count > 0:
        assert prefill_cur_in is not None
        assert prefill_position_ids is not None
        assert cached_proc is not None
        assert cached_final_hidden is not None
        assert cached_logits is not None
        assert cached_values is not None
        assert cached_draft_indices is not None
        assert cached_token_ids is not None
        tensors.update(
            {
                "cached_prefill_cur_in": prefill_cur_in.detach().cpu().contiguous(),
                "cached_prefill_position_ids": prefill_position_ids.cpu().contiguous(),
                "python_cached_logits": cached_logits.detach().cpu().contiguous(),
                "python_cached_spec_query": cached_proc.detach().cpu().contiguous(),
                "python_cached_final_hidden": cached_final_hidden.detach()
                .cpu()
                .contiguous(),
                "python_cached_topk_draft_indices": cached_draft_indices.detach()
                .cpu()
                .long()
                .contiguous(),
                "python_cached_topk_logits": cached_values.detach()
                .cpu()
                .float()
                .contiguous(),
                "python_cached_topk_token_ids": cached_token_ids.detach()
                .cpu()
                .long()
                .contiguous(),
            }
        )
    for idx, value in enumerate(layer_queries):
        tensors[f"python_layer_{idx}_query"] = value
    for idx, value in enumerate(layer_inputs):
        tensors[f"python_layer_{idx}_full_in"] = value
    for idx, value in enumerate(tap_inputs):
        tensors[f"tap_row_{idx}_concat"] = value.detach().cpu().contiguous()
    for idx, indices in enumerate(tap_hf_indices):
        tensors[f"tap_row_{idx}_hf_indices"] = torch.tensor(indices, dtype=torch.long)
    return Fixture(
        tensors=tensors,
        row_kinds=row_kinds,
        tap_row_count=len(tap_inputs),
        cached_prefill_row_count=cached_prefill_row_count,
        cached_crop_position=cached_crop_position,
    )


def ensure_fixture_tensors_finite(tensors: dict[str, Any]) -> None:
    import torch

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


def resolve_newest_pos(seq_len: int, num_stages: int, newest_pos_arg: int) -> int:
    newest_pos = seq_len - 1 if newest_pos_arg < 0 else int(newest_pos_arg)
    if newest_pos < num_stages - 1:
        raise ValueError(
            f"newest_pos={newest_pos} is too early for {num_stages} SPD rows"
        )
    if newest_pos >= seq_len:
        raise ValueError(f"newest_pos={newest_pos} is outside prompt length {seq_len}")
    return newest_pos


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


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
