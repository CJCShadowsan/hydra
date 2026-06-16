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
from typing import Any


FIXTURE_SCHEMA = "skippy-spd-parity-fixture/v1"


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
        "use_deepest": str(bool(getattr(pipeline, "trained_with_use_deepest", False))).lower(),
    }
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
    def __init__(self, tensors: dict[str, Any], row_kinds: list[str]) -> None:
        self.tensors = tensors
        self.row_kinds = row_kinds


def build_fixture(
    *,
    pipeline: Any,
    tokenizer: Any,
    prompt: str,
    top_k: int,
    newest_pos_arg: int,
) -> Fixture:
    import torch

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

        for pos in range(oldest_needed, newest_pos + 1):
            i_nominal_pipe = newest_pos - pos
            if i_nominal_pipe == 0:
                token = input_ids[:, pos : pos + 1]
                row = pipeline._build_inference_g0_row_from_hs(pipeline.embed_tokens(token))  # noqa: SLF001
                i_stages = 0
                row_kind = "g0"
            else:
                snap = completed_snaps[pos]
                i_stages = pipeline._choose_inference_i_stages_for_snap(  # noqa: SLF001
                    snap,
                    i_nominal_pipe,
                    use_deepest,
                    search_hi=None,
                )
                row = pipeline._build_inference_row_from_snap(snap, i_stages)  # noqa: SLF001
                row_kind = f"g{i_stages}"
            rows.append(row)
            row_positions.append(pos)
            row_i_stages.append(int(i_stages))
            row_kinds.append(row_kind)

        cur_in = torch.cat(rows, dim=1)
        position_ids = torch.tensor([row_positions], device=device, dtype=torch.long)
        proc = pipeline.speculation_module.forward_inference_g1_only_with_rotary(
            cur_in,
            position_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
        )
        proc = pipeline.final_norm(proc)
        logits = pipeline.speculation_module.lm_head(proc[:, -1:, :]).float()
        values, draft_indices = torch.topk(logits[0, 0], k=int(top_k))
        if getattr(pipeline, "_use_draft_vocab", False):
            draft_token_ids = pipeline._draft_token_ids.to(device).long()  # noqa: SLF001
            token_ids = draft_token_ids[draft_indices]
        else:
            token_ids = draft_indices

    tensors = {
        "cur_in": cur_in.detach().cpu().contiguous(),
        "position_ids": position_ids.cpu().contiguous(),
        "prompt_input_ids": input_ids.cpu().contiguous(),
        "row_i_stages": torch.tensor(row_i_stages, dtype=torch.long),
        "row_positions": torch.tensor(row_positions, dtype=torch.long),
        "python_logits": logits.detach().cpu().contiguous(),
        "python_topk_draft_indices": draft_indices.detach().cpu().long().contiguous(),
        "python_topk_logits": values.detach().cpu().float().contiguous(),
        "python_topk_token_ids": token_ids.detach().cpu().long().contiguous(),
    }
    return Fixture(tensors=tensors, row_kinds=row_kinds)


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
