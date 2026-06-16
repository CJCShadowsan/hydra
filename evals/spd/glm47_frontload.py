#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Frontload GLM 4.7 SPD integration checks.

This utility is intentionally lightweight: it inspects a local GLM checkpoint,
derives SPD topology metadata for non-uniform Skippy stage boundaries, and can
write tiny manifest-compatible smoke artifacts. The smoke artifacts validate the
Skippy SPD manifest and serving-checkpoint shape contract; they are not trained
weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REFERENCE_REPO = "https://github.com/yuyijiong/speculative_pipeline_decoding.git"
DEFAULT_GLM47_FLASH_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/"
    / "snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
)
SMOKE_CHECKPOINT_FORMAT = "torch-speculation-head-v10"
SERVING_FORMAT = "safetensors-spd-head-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and frontload GLM 4.7 SPD metadata")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_GLM47_FLASH_SNAPSHOT),
        help="Local GLM checkpoint directory containing config.json.",
    )
    parser.add_argument("--work-dir", default="/tmp/skippy-spd-glm47-frontload")
    parser.add_argument("--reference-repo", default=REFERENCE_REPO)
    parser.add_argument(
        "--patch-reference",
        action="store_true",
        help="Clone and patch the SPD reference repo's model-type allowlist for GLM.",
    )
    parser.add_argument(
        "--write-smoke-artifacts",
        action="store_true",
        help="Write tiny manifest-compatible SPD smoke artifacts.",
    )
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument(
        "--stage-layer-boundaries",
        default="",
        help="Comma-separated target layer end indices, e.g. 15,31,47.",
    )
    parser.add_argument("--num-spec-layers", type=int, default=1)
    parser.add_argument("--draft-vocab-size", type=int, default=8)
    parser.add_argument("--out-name", default="glm47-spd-frontload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    out_dir = work_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    inspection = inspect_checkpoint(model_path)
    boundaries = resolve_stage_boundaries(
        args.stage_layer_boundaries,
        num_stages=args.num_stages,
        num_layers=inspection["num_hidden_layers"],
    )
    hidden_indices = derive_hidden_tap_indices(boundaries)
    topology = {
        "hidden_size": inspection["hidden_size"],
        "vocab_size": inspection["vocab_size"],
        "draft_vocab_size": args.draft_vocab_size,
        "num_stages": len(boundaries),
        "stage_layer_boundaries": boundaries,
        "num_spec_layers": args.num_spec_layers,
        "trained_with_use_deepest": False,
        "shallow_hidden_layer_indices": hidden_indices,
        "spec_init_from_base_layers": None,
        "draft_token_ids": list(range(args.draft_vocab_size)),
    }

    report = {
        "model_path": str(model_path),
        "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": inspection,
        "topology": topology,
        "notes": [
            "Smoke artifacts are shape-contract fixtures, not trained SPD weights.",
            "Stage boundaries are target layer end indices.",
            "Hidden-state indices follow Hugging Face convention: 0 is embeddings, k is output after layer k-1.",
        ],
    }
    report_path = out_dir / "glm47-spd-frontload.json"
    write_json(report_path, report)

    if args.patch_reference:
        reference_dir = work_dir / "speculative_pipeline_decoding"
        clone_reference(args.reference_repo, reference_dir)
        patch_reference_for_glm(reference_dir)

    if args.write_smoke_artifacts:
        write_smoke_artifacts(
            out_dir=out_dir,
            model_path=model_path,
            inspection=inspection,
            topology=topology,
            reference_repo=args.reference_repo,
        )

    print(json.dumps(report, indent=2, sort_keys=True))


def inspect_checkpoint(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"GLM config.json not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_map = read_weight_map(model_path)
    auxiliary_tensors = sorted(
        name
        for name in weight_map
        if any(part in name.lower() for part in ("eh_proj", "enorm", "hnorm", "nextn", "mtp"))
    )
    return {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "hidden_size": positive_int(config, "hidden_size"),
        "vocab_size": positive_int(config, "vocab_size"),
        "num_hidden_layers": positive_int(config, "num_hidden_layers"),
        "num_nextn_predict_layers": int(config.get("num_nextn_predict_layers") or 0),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "rope_theta": config.get("rope_theta"),
        "tokenizer": inspect_tokenizer(model_path),
        "weight_shards": len({shard for shard in weight_map.values()}),
        "tensor_count": len(weight_map),
        "auxiliary_tensors": auxiliary_tensors,
    }


def read_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"invalid weight_map in {index_path}")
    return {str(k): str(v) for k, v in weight_map.items()}


def inspect_tokenizer(model_path: Path) -> dict[str, Any]:
    tokenizer_config = model_path / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        return {}
    config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    return {
        "tokenizer_class": config.get("tokenizer_class"),
        "model_max_length": config.get("model_max_length"),
        "has_chat_template": bool(config.get("chat_template") or (model_path / "chat_template.jinja").is_file()),
    }


def positive_int(config: dict[str, Any], key: str) -> int:
    value = int(config.get(key) or 0)
    if value <= 0:
        raise RuntimeError(f"config field {key!r} must be a positive integer")
    return value


def resolve_stage_boundaries(value: str, *, num_stages: int, num_layers: int) -> list[int]:
    if value.strip():
        boundaries = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif num_layers == 47 and num_stages == 3:
        boundaries = [15, 31, 47]
    else:
        boundaries = [
            round(num_layers * (stage + 1) / num_stages) for stage in range(num_stages)
        ]
    if not boundaries:
        raise RuntimeError("stage_layer_boundaries must not be empty")
    if boundaries[-1] != num_layers:
        raise RuntimeError(
            f"last stage boundary must equal num_hidden_layers={num_layers}, got {boundaries[-1]}"
        )
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise RuntimeError(f"stage boundaries must be strictly increasing: {boundaries}")
    return boundaries


def derive_hidden_tap_indices(boundaries: list[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    for depth in range(len(boundaries), 0, -1):
        rows.append([0, *boundaries[:depth]])
    return rows


def clone_reference(repo_url: str, dest: Path) -> None:
    if dest.exists():
        print(f"reference repo already exists: {dest}", file=sys.stderr)
        return
    run(["git", "clone", "--depth", "1", repo_url, str(dest)])


def patch_reference_for_glm(reference_dir: Path) -> None:
    pipeline_model = reference_dir / "pipeline_model.py"
    replace_once(
        pipeline_model,
        'supported = {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama"}',
        'supported = {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama", "glm4_moe_lite"}',
    )


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"expected text not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_smoke_artifacts(
    *,
    out_dir: Path,
    model_path: Path,
    inspection: dict[str, Any],
    topology: dict[str, Any],
    reference_repo: str,
) -> None:
    checkpoint_path = out_dir / "speculation_head_final.pt"
    checkpoint_payload = {
        "note": "GLM SPD frontload placeholder; not a torch training checkpoint.",
        "config": checkpoint_config(model_path, inspection, topology),
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n")
    serving_path = out_dir / "spd-head.safetensors"
    write_smoke_safetensors(serving_path, topology)
    manifest = {
        "schema": "skippy-spd-head/v1",
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "serving_checkpoint": {
            "path": serving_path.name,
            "sha256": file_sha256(serving_path),
            "bytes": serving_path.stat().st_size,
            "format": SERVING_FORMAT,
            "tensor_count": smoke_tensor_count(topology),
            "dtype": "F32",
        },
        "source": {
            "format": SMOKE_CHECKPOINT_FORMAT,
            "reference_repo": reference_repo,
            "base_model_path": str(model_path),
            "model_type": inspection["model_type"],
            "checkpoint_version": 10,
        },
        "topology": topology,
    }
    write_json(out_dir / "skippy-spd-head.json", manifest)


def checkpoint_config(
    model_path: Path,
    inspection: dict[str, Any],
    topology: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 10,
        "base_model_path": str(model_path),
        "model_type": inspection["model_type"],
        **topology,
    }


def write_smoke_safetensors(path: Path, topology: dict[str, Any]) -> None:
    hidden = int(topology["hidden_size"])
    tensors: list[tuple[str, str, list[int]]] = []
    for stage, indices in enumerate(topology["shallow_hidden_layer_indices"]):
        tensors.append((f"stage_projs.{stage}.weight", "F32", [hidden, hidden * len(indices)]))
    tensors.append(("g0_proj.weight", "F32", [hidden, hidden]))
    tensors.append(("lm_head.weight", "F32", [int(topology["draft_vocab_size"]), hidden]))
    for layer in range(int(topology["num_spec_layers"])):
        tensors.append((f"spec_layers.{layer}.input_layernorm.weight", "F32", [hidden]))
        tensors.append((f"spec_layers.{layer}.post_attention_layernorm.weight", "F32", [hidden]))

    header_entries: dict[str, Any] = {
        "__metadata__": {
            "format": SERVING_FORMAT,
            "purpose": "glm47-spd-frontload-smoke",
        }
    }
    data_len = 0
    for name, dtype, shape in tensors:
        byte_len = tensor_byte_len(dtype, shape)
        header_entries[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [data_len, data_len + byte_len],
        }
        data_len += byte_len
    header = json.dumps(header_entries, sort_keys=True, separators=(",", ":")).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.truncate(8 + len(header) + data_len)


def smoke_tensor_count(topology: dict[str, Any]) -> int:
    return len(topology["shallow_hidden_layer_indices"]) + 2 + 2 * int(topology["num_spec_layers"])


def tensor_byte_len(dtype: str, shape: list[int]) -> int:
    sizes = {"F32": 4}
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    return elements * sizes[dtype]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
