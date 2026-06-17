#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
# ]
# ///
"""Export a reference SPD PyTorch checkpoint into a Skippy serving artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SERVING_FORMAT = "safetensors-spd-head-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an SPD .pt checkpoint to a Rust-readable safetensors artifact"
    )
    parser.add_argument("--checkpoint", required=True, help="Input speculation_head_final.pt")
    parser.add_argument("--manifest", required=True, help="Input skippy-spd-head.json")
    parser.add_argument(
        "--manifest-out",
        default="",
        help="Manifest to write. Defaults to updating --manifest in place.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to the manifest output directory.",
    )
    parser.add_argument("--out-name", default="spd-head.safetensors")
    parser.add_argument(
        "--dtype",
        choices=("keep", "bfloat16", "float16", "float32"),
        default="keep",
        help="Tensor dtype for the serving artifact.",
    )
    parser.add_argument(
        "--base-model-path",
        default="",
        help="Optional manifest base_model_path override, for portable public manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    manifest_out = Path(args.manifest_out) if args.manifest_out else manifest_path
    out_dir = Path(args.out_dir) if args.out_dir else manifest_out.parent
    output_path = out_dir / args.out_name

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors, checkpoint_config = load_tensors(checkpoint_path, args.dtype)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    dtype_label = common_dtype_label(tensors)
    metadata = {
        "format": SERVING_FORMAT,
        "source_checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "source_format": manifest["source"]["format"],
        "tensor_count": str(len(tensors)),
        "dtype": dtype_label,
    }
    base_model_path = args.base_model_path.strip()
    if base_model_path:
        manifest["source"]["base_model_path"] = base_model_path
    metadata["base_model_path"] = manifest["source"]["base_model_path"]

    save_file(tensors, output_path, metadata=metadata)

    manifest["serving_checkpoint"] = {
        "path": manifest_relative_path(output_path, manifest_out.parent),
        "sha256": file_sha256(output_path),
        "bytes": output_path.stat().st_size,
        "format": SERVING_FORMAT,
        "tensor_count": len(tensors),
        "dtype": dtype_label,
    }
    if checkpoint_config.get("version") is not None:
        manifest["source"]["checkpoint_version"] = int(checkpoint_config["version"])

    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "serving_checkpoint": str(output_path),
                "manifest": str(manifest_out),
                "format": SERVING_FORMAT,
                "tensor_count": len(tensors),
                "dtype": dtype_label,
                "bytes": output_path.stat().st_size,
                "sha256": manifest["serving_checkpoint"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_tensors(checkpoint_path: Path, dtype: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{checkpoint_path} must contain a dict checkpoint")

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"{checkpoint_path} does not contain a state dict")

    target_dtype = {
        "keep": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    tensors: dict[str, Any] = {}
    for name in sorted(state_dict):
        value = state_dict[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        if target_dtype is not None:
            tensor = tensor.to(target_dtype)
        tensors[name] = tensor

    if not tensors:
        raise RuntimeError(f"{checkpoint_path} did not contain any tensors")
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    return tensors, config


def common_dtype_label(tensors: dict[str, Any]) -> str:
    import torch

    labels = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
    }
    seen = {labels.get(tensor.dtype, str(tensor.dtype)) for tensor in tensors.values()}
    if len(seen) == 1:
        return next(iter(seen))
    return "mixed"


def manifest_relative_path(path: Path, manifest_dir: Path) -> str:
    resolved_path = path.resolve()
    resolved_manifest_dir = manifest_dir.resolve()
    try:
        relative = resolved_path.relative_to(resolved_manifest_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"{path} must be inside the manifest directory {manifest_dir}"
        ) from exc
    if any(part in ("", ".", "..") for part in relative.parts):
        raise RuntimeError(f"unsafe manifest-relative path: {relative}")
    return relative.as_posix()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
