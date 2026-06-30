#!/usr/bin/env python3
"""Verify GLM-DSA checkpoint and GGUF tensor inventory without reading payloads."""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO


GGUF_MAGIC = b"GGUF"
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

SCALAR_WIDTHS = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}

ATTENTION_SUFFIXES = [
    "input_layernorm.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
]

MOE_ROUTER_SUFFIXES = [
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
]

SHARED_EXPERT_SUFFIX_VARIANTS = [
    ["mlp.shared_expert.gate_proj.weight", "mlp.shared_experts.gate_proj.weight"],
    ["mlp.shared_expert.down_proj.weight", "mlp.shared_experts.down_proj.weight"],
    ["mlp.shared_expert.up_proj.weight", "mlp.shared_experts.up_proj.weight"],
]

INDEXER_SUFFIXES = [
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.wq_b.weight",
]

MTP_SUFFIXES = [
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "shared_head.norm.weight",
]

GGUF_ATTENTION_NAMES = [
    "attn_norm.weight",
    "attn_q_a_norm.weight",
    "attn_kv_a_norm.weight",
    "attn_q_a.weight",
    "attn_q_b.weight",
    "attn_kv_a_mqa.weight",
    "attn_k_b.weight",
    "attn_v_b.weight",
    "attn_output.weight",
    "ffn_norm.weight",
]

GGUF_INDEXER_NAMES = [
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.proj.weight",
    "indexer.attn_k.weight",
    "indexer.attn_q_b.weight",
]

GGUF_MTP_NAMES = [
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
]

LAYER_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class Inventory:
    metadata: dict[str, object] = field(default_factory=dict)
    tensors: set[str] = field(default_factory=set)
    files: int = 0


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of file")
    return data


def read_u32(file: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(file, 4))[0]


def read_u64(file: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(file, 8))[0]


def read_string(file: BinaryIO) -> str:
    length = read_u64(file)
    return read_exact(file, length).decode("utf-8")


def read_scalar(file: BinaryIO, value_type: int) -> object:
    if value_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<?", read_exact(file, 1))[0])
    if value_type == GGUF_TYPE_UINT32:
        return read_u32(file)
    if value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", read_exact(file, 4))[0]
    if value_type == GGUF_TYPE_UINT64:
        return read_u64(file)
    if value_type == GGUF_TYPE_STRING:
        return read_string(file)
    width = SCALAR_WIDTHS.get(value_type)
    if width is None:
        fail(f"unsupported GGUF metadata scalar type {value_type}")
    file.seek(width, 1)
    return None


def read_array(file: BinaryIO) -> object:
    element_type = read_u32(file)
    length = read_u64(file)
    if element_type == GGUF_TYPE_STRING:
        return [read_string(file) for _ in range(length)]
    width = SCALAR_WIDTHS.get(element_type)
    if width is None:
        fail(f"unsupported GGUF metadata array element type {element_type}")
    file.seek(width * length, 1)
    return None


def read_metadata_value(file: BinaryIO, value_type: int) -> object:
    if value_type == GGUF_TYPE_ARRAY:
        return read_array(file)
    return read_scalar(file, value_type)


def parse_gguf_header(path: Path) -> Inventory:
    with path.open("rb") as file:
        if read_exact(file, 4) != GGUF_MAGIC:
            fail(f"{path} is not a GGUF file")
        _version = read_u32(file)
        tensor_count = read_u64(file)
        metadata_count = read_u64(file)
        inventory = Inventory(files=1)
        for _ in range(metadata_count):
            key = read_string(file)
            value_type = read_u32(file)
            value = read_metadata_value(file, value_type)
            if value is not None:
                inventory.metadata[key] = value
        for _ in range(tensor_count):
            name = read_string(file)
            dim_count = read_u32(file)
            file.seek(8 * dim_count, 1)
            file.seek(4, 1)
            file.seek(8, 1)
            inventory.tensors.add(name)
    return inventory


def merge_gguf_inventory(paths: list[Path]) -> Inventory:
    merged = Inventory()
    for path in paths:
        item = parse_gguf_header(path)
        merged.files += 1
        merged.tensors.update(item.tensors)
        for key, value in item.metadata.items():
            merged.metadata.setdefault(key, value)
    return merged


def gguf_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    paths = sorted(root.rglob("*.gguf"))
    if not paths:
        fail(f"no GGUF files found under {root}")
    return paths


def tensor_layer(name: str) -> int | None:
    match = LAYER_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def read_safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        header_len = read_u64(file)
        return json.loads(read_exact(file, header_len))


def read_checkpoint_tensors(root: Path) -> set[str]:
    tensors: set[str] = set()
    for path in sorted(root.glob("*.safetensors")):
        header = read_safetensors_header(path)
        tensors.update(key for key in header if key != "__metadata__")
    if not tensors:
        fail(f"no safetensors found under {root}")
    return tensors


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def source_name(layer: int, suffix: str) -> str:
    return f"model.layers.{layer}.{suffix}"


def has_any_source_name(tensors: set[str], layer: int, suffixes: list[str]) -> bool:
    return any(source_name(layer, suffix) in tensors for suffix in suffixes)


def gguf_name(layer: int, suffix: str) -> str:
    return f"blk.{layer}.{suffix}"


def first_present_u32(mapping: dict[str, object], keys: list[str]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int):
            return value
    return None


def frequency_layer_is_full(layer: int, offset: int, frequency: int) -> bool:
    return layer < offset or (layer >= offset and ((layer - offset + 1) % frequency) == 0)


def validate_indexshare_frequency_contract(
    roles: object,
    frequency: int | None,
    offset: int | None,
    errors: list[str],
    *,
    label: str,
) -> None:
    if frequency is None:
        return
    require(frequency > 0, f"{label} IndexShare frequency must be positive", errors)
    require(offset is not None, f"{label} IndexShare skip-top-k offset missing while frequency is present", errors)
    if frequency <= 0 or offset is None or not isinstance(roles, list):
        return
    for layer, role in enumerate(roles):
        role_full = role == "full"
        freq_full = frequency_layer_is_full(layer, offset, frequency)
        require(
            role_full == freq_full,
            f"{label} indexer_types conflicts with frequency metadata at layer {layer}",
            errors,
        )


def verify_source_checkpoint(root: Path, expected_target_layers: int, expected_nextn_layers: int) -> dict[str, object]:
    config = json.loads(root.joinpath("config.json").read_text())
    tensors = read_checkpoint_tensors(root)
    errors: list[str] = []

    target_layers = config.get("num_hidden_layers")
    nextn_layers = config.get("num_nextn_predict_layers", 0)
    mtp_layer = target_layers
    roles = config.get("indexer_types")
    frequency = first_present_u32(config, ["index_topk_freq", "indexer_top_k_freq"])
    offset = first_present_u32(config, ["index_skip_topk_offset", "indexer_skip_top_k_offset"])

    require(config.get("model_type") == "glm_moe_dsa", "config model_type is not glm_moe_dsa", errors)
    require(
        target_layers == expected_target_layers,
        f"expected {expected_target_layers} target layers, got {target_layers}",
        errors,
    )
    require(
        nextn_layers == expected_nextn_layers,
        f"expected {expected_nextn_layers} nextn layer(s), got {nextn_layers}",
        errors,
    )
    require(isinstance(roles, list), "indexer_types missing from config", errors)
    if isinstance(roles, list):
        require(len(roles) == target_layers, f"indexer_types length {len(roles)} != target layers {target_layers}", errors)
        require(set(roles) <= {"full", "shared"}, "indexer_types contains values other than full/shared", errors)
    validate_indexshare_frequency_contract(roles, frequency, offset, errors, label="source")

    require(not any(name.startswith("model.layers.79.") for name in tensors), "source contains unexpected model.layers.79 tensors", errors)

    for suffix in ATTENTION_SUFFIXES:
        require(source_name(mtp_layer, suffix) in tensors, f"MTP layer missing attention tensor {suffix}", errors)
    for suffix in MOE_ROUTER_SUFFIXES:
        require(source_name(mtp_layer, suffix) in tensors, f"MTP layer missing MoE tensor {suffix}", errors)
    require(
        any(name.startswith(f"model.layers.{mtp_layer}.mlp.experts.") for name in tensors),
        "MTP layer missing routed expert tensors",
        errors,
    )
    for variants in SHARED_EXPERT_SUFFIX_VARIANTS:
        require(has_any_source_name(tensors, mtp_layer, variants), f"MTP layer missing shared expert tensor variant {variants}", errors)
    for suffix in INDEXER_SUFFIXES:
        require(source_name(mtp_layer, suffix) in tensors, f"MTP layer missing indexer tensor {suffix}", errors)
    for suffix in MTP_SUFFIXES:
        require(source_name(mtp_layer, suffix) in tensors, f"MTP layer missing native MTP tensor {suffix}", errors)

    if isinstance(roles, list):
        for layer, role in enumerate(roles):
            has_indexer = all(source_name(layer, suffix) in tensors for suffix in INDEXER_SUFFIXES)
            require(has_indexer == (role == "full"), f"layer {layer} role {role} indexer tensor presence is {has_indexer}", errors)

    if errors:
        fail("source checkpoint inventory failed:\n  - " + "\n  - ".join(errors))

    return {
        "target_layers": target_layers,
        "nextn_layers": nextn_layers,
        "source_tensors": len(tensors),
        "full_roles": roles.count("full") if isinstance(roles, list) else None,
        "shared_roles": roles.count("shared") if isinstance(roles, list) else None,
    }


def verify_gguf_inventory(
    root: Path,
    expected_target_layers: int,
    expected_nextn_layers: int,
    *,
    partial: bool = False,
) -> dict[str, object]:
    inventory = merge_gguf_inventory(gguf_paths(root))
    errors: list[str] = []
    tensors = inventory.tensors
    metadata = inventory.metadata

    block_count = metadata.get("glm-dsa.block_count")
    nextn_layers = metadata.get("glm-dsa.nextn_predict_layers", 0)
    roles = metadata.get("glm-dsa.attention.indexer.types")
    frequency = metadata.get("glm-dsa.attention.indexer.top_k_frequency")
    offset = metadata.get("glm-dsa.attention.indexer.skip_top_k_offset")
    expected_block_count = expected_target_layers + expected_nextn_layers
    mtp_layer = expected_target_layers

    require(metadata.get("general.architecture") == "glm-dsa", "GGUF architecture is not glm-dsa", errors)
    require(
        block_count == expected_block_count,
        f"expected GGUF block_count {expected_block_count}, got {block_count}",
        errors,
    )
    require(
        nextn_layers == expected_nextn_layers,
        f"expected GGUF nextn_predict_layers {expected_nextn_layers}, got {nextn_layers}",
        errors,
    )
    require(isinstance(roles, list), "GGUF indexer.types metadata missing", errors)
    if isinstance(roles, list):
        require(
            len(roles) == expected_target_layers,
            f"GGUF indexer.types length {len(roles)} != target layers {expected_target_layers}",
            errors,
        )
        require(set(roles) <= {"full", "shared"}, "GGUF indexer.types contains values other than full/shared", errors)
    validate_indexshare_frequency_contract(roles, frequency, offset, errors, label="GGUF")

    require(
        not any(name.startswith(f"blk.{expected_block_count}.") for name in tensors),
        f"GGUF contains unexpected blk.{expected_block_count} tensors",
        errors,
    )
    unsplit_kv_b = sorted(name for name in tensors if name.endswith(".attn_kv_b.weight"))
    require(
        not unsplit_kv_b,
        f"GGUF contains unsplit GLM-DSA attn_kv_b tensors, first examples: {unsplit_kv_b[:5]}",
        errors,
    )

    if not partial and isinstance(mtp_layer, int):
        for suffix in GGUF_ATTENTION_NAMES:
            require(gguf_name(mtp_layer, suffix) in tensors, f"MTP layer missing GGUF attention tensor {suffix}", errors)
        for suffix in GGUF_INDEXER_NAMES:
            require(gguf_name(mtp_layer, suffix) in tensors, f"MTP layer missing GGUF indexer tensor {suffix}", errors)
        for suffix in GGUF_MTP_NAMES:
            require(gguf_name(mtp_layer, suffix) in tensors, f"MTP layer missing GGUF native MTP tensor {suffix}", errors)

    if errors:
        fail("GGUF inventory failed:\n  - " + "\n  - ".join(errors))

    return {
        "gguf_files": inventory.files,
        "gguf_tensors": len(tensors),
        "block_count": block_count,
        "nextn_layers": nextn_layers,
        "full_roles": roles.count("full") if isinstance(roles, list) else None,
        "shared_roles": roles.count("shared") if isinstance(roles, list) else None,
        "partial": partial,
        "contains_unsplit_kv_b": bool(unsplit_kv_b),
    }


def stale_shard_report(root: Path, expected_target_layers: int, expected_nextn_layers: int) -> dict[str, object]:
    paths = gguf_paths(root)
    stale_files: list[dict[str, object]] = []
    stale_layers: set[int] = set()
    merged = Inventory()

    for path in paths:
        inventory = parse_gguf_header(path)
        merged.files += 1
        merged.tensors.update(inventory.tensors)
        for key, value in inventory.metadata.items():
            merged.metadata.setdefault(key, value)

        unsplit_kv_b = sorted(name for name in inventory.tensors if name.endswith(".attn_kv_b.weight"))
        if not unsplit_kv_b:
            continue

        layers = sorted({layer for name in unsplit_kv_b if (layer := tensor_layer(name)) is not None})
        stale_layers.update(layers)
        stale_files.append(
            {
                "file": str(path),
                "unsplit_kv_b_count": len(unsplit_kv_b),
                "layers": layers,
                "examples": unsplit_kv_b[:5],
            }
        )

    expected_block_count = expected_target_layers + expected_nextn_layers
    mtp_layer = expected_target_layers
    required_mtp_split = [
        gguf_name(mtp_layer, "attn_k_b.weight"),
        gguf_name(mtp_layer, "attn_v_b.weight"),
    ]
    missing_mtp_split = [name for name in required_mtp_split if name not in merged.tensors]
    missing_metadata = [
        key
        for key in [
            "general.architecture",
            "glm-dsa.block_count",
            "glm-dsa.nextn_predict_layers",
            "glm-dsa.attention.indexer.types",
        ]
        if key not in merged.metadata
    ]

    return {
        "gguf_files": len(paths),
        "gguf_tensors": len(merged.tensors),
        "metadata_missing": missing_metadata,
        "block_count": merged.metadata.get("glm-dsa.block_count"),
        "expected_block_count": expected_block_count,
        "stale_file_count": len(stale_files),
        "stale_layer_count": len(stale_layers),
        "stale_layers": sorted(stale_layers),
        "stale_files": stale_files,
        "missing_mtp_split_tensors": missing_mtp_split,
        "contains_mtp_unsplit_kv_b": gguf_name(mtp_layer, "attn_kv_b.weight") in merged.tensors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="GLM-5.2 SafeTensors checkpoint directory")
    parser.add_argument("--gguf", type=Path, help="BF16 GGUF file or directory to verify")
    parser.add_argument("--expected-target-layers", type=int, default=78)
    parser.add_argument("--expected-nextn-layers", type=int, default=1)
    parser.add_argument(
        "--partial-gguf",
        action="store_true",
        help="Validate GGUF metadata and present tensor names without requiring every block tensor.",
    )
    parser.add_argument(
        "--stale-shard-report",
        action="store_true",
        help="Print a non-failing read-only report of GGUF shards that still contain stale unsplit GLM-DSA tensors.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {
        "checkpoint": verify_source_checkpoint(
            args.checkpoint,
            args.expected_target_layers,
            args.expected_nextn_layers,
        )
    }
    if args.stale_shard_report:
        if not args.gguf:
            fail("--stale-shard-report requires --gguf")
        summary["stale_shard_report"] = stale_shard_report(
            args.gguf,
            args.expected_target_layers,
            args.expected_nextn_layers,
        )
    elif args.gguf:
        summary["gguf"] = verify_gguf_inventory(
            args.gguf,
            args.expected_target_layers,
            args.expected_nextn_layers,
            partial=args.partial_gguf,
        )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("GLM-DSA inventory verification passed")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
