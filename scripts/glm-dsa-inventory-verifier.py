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
class TensorInfo:
    shape: tuple[int, ...]
    ggml_type: int
    file: str


@dataclass
class SourceTensorInfo:
    shape: tuple[int, ...]
    dtype: str
    file: str


@dataclass
class Inventory:
    metadata: dict[str, object] = field(default_factory=dict)
    tensors: set[str] = field(default_factory=set)
    tensor_info: dict[str, TensorInfo] = field(default_factory=dict)
    tensor_info_offset: int = 0
    files: int = 0


@dataclass
class SourceInventory:
    tensors: set[str] = field(default_factory=set)
    tensor_info: dict[str, SourceTensorInfo] = field(default_factory=dict)
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


def skip_string(file: BinaryIO) -> None:
    length = read_u64(file)
    file.seek(length, 1)


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


def skip_scalar(file: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        skip_string(file)
        return
    width = SCALAR_WIDTHS.get(value_type)
    if width is None:
        fail(f"unsupported GGUF metadata scalar type {value_type}")
    file.seek(width, 1)


def skip_array(file: BinaryIO) -> None:
    element_type = read_u32(file)
    length = read_u64(file)
    if element_type == GGUF_TYPE_STRING:
        for _ in range(length):
            skip_string(file)
        return
    width = SCALAR_WIDTHS.get(element_type)
    if width is None:
        fail(f"unsupported GGUF metadata array element type {element_type}")
    file.seek(width * length, 1)


def skip_metadata_value(file: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_TYPE_ARRAY:
        skip_array(file)
        return
    skip_scalar(file, value_type)


def parse_gguf_header(
    path: Path,
    *,
    collect_metadata: bool = True,
    tensor_info_offset_hint: int | None = None,
) -> Inventory:
    with path.open("rb") as file:
        if read_exact(file, 4) != GGUF_MAGIC:
            fail(f"{path} is not a GGUF file")
        _version = read_u32(file)
        tensor_count = read_u64(file)
        metadata_count = read_u64(file)
        inventory = Inventory(files=1)
        if tensor_info_offset_hint is not None and not collect_metadata:
            file.seek(tensor_info_offset_hint)
        else:
            for _ in range(metadata_count):
                if collect_metadata:
                    key = read_string(file)
                else:
                    skip_string(file)
                    key = ""
                value_type = read_u32(file)
                if collect_metadata:
                    value = read_metadata_value(file, value_type)
                else:
                    skip_metadata_value(file, value_type)
                    value = None
                if collect_metadata and value is not None:
                    inventory.metadata[key] = value
        inventory.tensor_info_offset = file.tell()
        for _ in range(tensor_count):
            name = read_string(file)
            dim_count = read_u32(file)
            shape = tuple(read_u64(file) for _ in range(dim_count))
            ggml_type = read_u32(file)
            file.seek(8, 1)
            inventory.tensors.add(name)
            inventory.tensor_info[name] = TensorInfo(shape=shape, ggml_type=ggml_type, file=str(path))
    return inventory


def merge_gguf_inventory(paths: list[Path]) -> Inventory:
    merged = Inventory()
    tensor_info_offset_hint: int | None = None
    for index, path in enumerate(paths):
        item = parse_gguf_header(
            path,
            collect_metadata=index == 0,
            tensor_info_offset_hint=tensor_info_offset_hint,
        )
        if index == 0:
            tensor_info_offset_hint = item.tensor_info_offset
            merged.tensor_info_offset = item.tensor_info_offset
        merged.files += 1
        merged.tensors.update(item.tensors)
        merged.tensor_info.update(item.tensor_info)
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
    return read_checkpoint_inventory(root).tensors


def read_checkpoint_inventory(root: Path) -> SourceInventory:
    inventory = SourceInventory()
    for path in sorted(root.glob("*.safetensors")):
        header = read_safetensors_header(path)
        inventory.files += 1
        for key, value in header.items():
            if key == "__metadata__":
                continue
            shape = value.get("shape")
            dtype = value.get("dtype")
            if not isinstance(shape, list) or not isinstance(dtype, str):
                fail(f"{path} has invalid safetensors header for {key}")
            inventory.tensors.add(key)
            inventory.tensor_info[key] = SourceTensorInfo(
                shape=tuple(int(dim) for dim in shape),
                dtype=dtype,
                file=str(path),
            )
    if not inventory.tensors:
        fail(f"no safetensors found under {root}")
    return inventory


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def require_tensor(tensors: set[str], name: str, errors: list[str]) -> None:
    require(name in tensors, f"missing GGUF tensor {name}", errors)


def require_source_tensor_shape(
    tensor_info: dict[str, SourceTensorInfo],
    name: str,
    shape: tuple[int, ...],
    errors: list[str],
    *,
    dtype: str | None = None,
) -> None:
    info = tensor_info.get(name)
    if info is None:
        errors.append(f"missing source tensor {name}")
        return
    require(info.shape == shape, f"source tensor {name} shape {info.shape} != {shape}", errors)
    if dtype is not None:
        require(info.dtype == dtype, f"source tensor {name} dtype {info.dtype} != {dtype}", errors)


def require_gguf_tensor_shape(
    tensor_info: dict[str, TensorInfo],
    name: str,
    shape: tuple[int, ...],
    errors: list[str],
) -> None:
    info = tensor_info.get(name)
    if info is None:
        errors.append(f"missing GGUF tensor {name}")
        return
    require(info.shape == shape, f"GGUF tensor {name} shape {info.shape} != {shape}", errors)


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


def required_int(mapping: dict[str, object], key: str, errors: list[str]) -> int:
    value = mapping.get(key)
    require(isinstance(value, int), f"config {key} must be present and integer", errors)
    return value if isinstance(value, int) else 0


def frequency_layer_is_full(layer: int, offset: int, frequency: int) -> bool:
    return layer < offset or (layer >= offset and ((layer - offset + 1) % frequency) == 0)


def glm52_expected_dims(config: dict[str, object], errors: list[str]) -> dict[str, int]:
    dims = {
        "hidden": required_int(config, "hidden_size", errors),
        "vocab": required_int(config, "vocab_size", errors),
        "heads": required_int(config, "num_attention_heads", errors),
        "q_lora": required_int(config, "q_lora_rank", errors),
        "kv_lora": required_int(config, "kv_lora_rank", errors),
        "qk_nope": required_int(config, "qk_nope_head_dim", errors),
        "qk_rope": required_int(config, "qk_rope_head_dim", errors),
        "v_head": required_int(config, "v_head_dim", errors),
        "dense_ff": required_int(config, "intermediate_size", errors),
        "moe_ff": required_int(config, "moe_intermediate_size", errors),
        "experts": required_int(config, "n_routed_experts", errors),
        "shared_experts": required_int(config, "n_shared_experts", errors),
        "experts_used": required_int(config, "num_experts_per_tok", errors),
        "index_heads": required_int(config, "index_n_heads", errors),
        "index_head": required_int(config, "index_head_dim", errors),
        "index_topk": required_int(config, "index_topk", errors),
        "dense_lead": required_int(config, "first_k_dense_replace", errors),
    }
    dims["qk_head"] = dims["qk_nope"] + dims["qk_rope"]
    dims["q_width"] = dims["heads"] * dims["qk_head"]
    dims["v_width"] = dims["heads"] * dims["v_head"]
    dims["kv_b_width"] = dims["heads"] * (dims["qk_nope"] + dims["v_head"])
    dims["index_width"] = dims["index_heads"] * dims["index_head"]
    return dims


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


def expected_glm52_full_layers() -> list[int]:
    return [0, 1, 2] + list(range(6, 78, 4))


def expected_glm52_roles() -> list[str]:
    full_layers = set(expected_glm52_full_layers())
    return ["full" if layer in full_layers else "shared" for layer in range(78)]


def validate_strict_glm52_indexshare_contract(
    roles: object,
    frequency: int | None,
    offset: int | None,
    errors: list[str],
    *,
    label: str,
) -> None:
    require(frequency == 4, f"{label} GLM-5.2 IndexShare frequency {frequency!r} != 4", errors)
    require(offset == 3, f"{label} GLM-5.2 IndexShare skip-top-k offset {offset!r} != 3", errors)
    if not isinstance(roles, list):
        return

    expected_roles = expected_glm52_roles()
    expected_full = expected_glm52_full_layers()
    actual_full = [layer for layer, role in enumerate(roles) if role == "full"]
    actual_shared = [layer for layer, role in enumerate(roles) if role == "shared"]
    expected_shared = [layer for layer, role in enumerate(expected_roles) if role == "shared"]

    require(roles == expected_roles, f"{label} GLM-5.2 IndexShare roles do not match the expected Full/Shared schedule", errors)
    require(actual_full == expected_full, f"{label} GLM-5.2 Full layers {actual_full} != {expected_full}", errors)
    require(actual_shared == expected_shared, f"{label} GLM-5.2 Shared layers {actual_shared} != {expected_shared}", errors)
    require(len(actual_full) == 21, f"{label} GLM-5.2 Full layer count {len(actual_full)} != 21", errors)
    require(len(actual_shared) == 57, f"{label} GLM-5.2 Shared layer count {len(actual_shared)} != 57", errors)


def strict_glm52_indexshare_errors(roles: object, frequency: int | None, offset: int | None) -> list[str]:
    errors: list[str] = []
    validate_strict_glm52_indexshare_contract(roles, frequency, offset, errors, label="self-test")
    return errors


def require_self_test_failure(name: str, roles: object, frequency: int | None, offset: int | None) -> dict[str, object]:
    errors = strict_glm52_indexshare_errors(roles, frequency, offset)
    if not errors:
        fail(f"self-test {name} unexpectedly passed")
    return {"name": name, "passed": True, "errors": errors}


def run_self_tests() -> dict[str, object]:
    expected_roles = expected_glm52_roles()
    passing_errors = strict_glm52_indexshare_errors(expected_roles, 4, 3)
    if passing_errors:
        fail("self-test valid GLM-5.2 role schedule failed:\n  - " + "\n  - ".join(passing_errors))

    swapped_roles = expected_roles.copy()
    swapped_roles[3] = "full"
    swapped_roles[6] = "shared"

    short_roles = expected_roles[:-1]

    cases = [
        require_self_test_failure("wrong_frequency", expected_roles, 3, 3),
        require_self_test_failure("wrong_offset", expected_roles, 4, 0),
        require_self_test_failure("swapped_full_shared_roles", swapped_roles, 4, 3),
        require_self_test_failure("truncated_role_schedule", short_roles, 4, 3),
    ]
    return {
        "valid_schedule_passed": True,
        "expected_full_layers": expected_glm52_full_layers(),
        "expected_full_count": expected_roles.count("full"),
        "expected_shared_count": expected_roles.count("shared"),
        "negative_cases": cases,
    }


def source_contract_summary(
    dims: dict[str, int],
    roles: object,
    frequency: int | None,
    offset: int | None,
    target_layers: object,
    nextn_layers: object,
) -> dict[str, object]:
    return {
        "target_layers": target_layers,
        "nextn_layers": nextn_layers,
        "block_count": int(target_layers) + int(nextn_layers) if isinstance(target_layers, int) and isinstance(nextn_layers, int) else None,
        "dense_lead_layers": dims["dense_lead"],
        "hidden_size": dims["hidden"],
        "attention_head_count": dims["heads"],
        "attention_key_length": dims["kv_lora"] + dims["qk_rope"],
        "attention_key_length_mla": dims["qk_head"],
        "attention_value_length": dims["v_head"],
        "attention_value_length_mla": dims["v_head"],
        "q_lora_rank": dims["q_lora"],
        "kv_lora_rank": dims["kv_lora"],
        "rope_dimension_count": dims["qk_rope"],
        "expert_count": dims["experts"],
        "expert_used_count": dims["experts_used"],
        "expert_shared_count": dims["shared_experts"],
        "feed_forward_length": dims["dense_ff"],
        "expert_feed_forward_length": dims["moe_ff"],
        "indexer_head_count": dims["index_heads"],
        "indexer_key_length": dims["index_head"],
        "indexer_top_k": dims["index_topk"],
        "indexer_top_k_frequency": frequency,
        "indexer_skip_top_k_offset": offset,
        "indexer_types": roles if isinstance(roles, list) else None,
    }


def gguf_contract_summary(
    dims: dict[str, int],
    roles: object,
    frequency: object,
    offset: object,
    block_count: object,
    nextn_layers: object,
) -> dict[str, object]:
    target_layers = block_count - nextn_layers if isinstance(block_count, int) and isinstance(nextn_layers, int) else None
    return {
        "target_layers": target_layers,
        "nextn_layers": nextn_layers,
        "block_count": block_count,
        "dense_lead_layers": dims["dense_lead"],
        "hidden_size": dims["hidden"],
        "attention_head_count": dims["heads"],
        "attention_key_length": dims["key_length"],
        "attention_key_length_mla": dims["key_mla"],
        "attention_value_length": dims["value_length"],
        "attention_value_length_mla": dims["value_mla"],
        "q_lora_rank": dims["q_lora"],
        "kv_lora_rank": dims["kv_lora"],
        "rope_dimension_count": dims["rope"],
        "expert_count": dims["experts"],
        "expert_used_count": dims["experts_used"],
        "expert_shared_count": dims["shared_experts"],
        "feed_forward_length": dims["dense_ff"],
        "expert_feed_forward_length": dims["moe_ff"],
        "indexer_head_count": dims["index_heads"],
        "indexer_key_length": dims["index_head"],
        "indexer_top_k": dims["index_topk"],
        "indexer_top_k_frequency": frequency,
        "indexer_skip_top_k_offset": offset,
        "indexer_types": roles if isinstance(roles, list) else None,
    }


def compare_contract_summaries(source: dict[str, object], gguf: dict[str, object]) -> dict[str, object]:
    keys = sorted(source.keys())
    mismatches = [
        {
            "field": key,
            "source": source.get(key),
            "gguf": gguf.get(key),
        }
        for key in keys
        if source.get(key) != gguf.get(key)
    ]
    if mismatches:
        details = "\n  - ".join(
            f"{item['field']}: source={item['source']!r} gguf={item['gguf']!r}" for item in mismatches
        )
        fail("source-to-GGUF GLM-DSA contract comparison failed:\n  - " + details)
    return {
        "matched": True,
        "checked_fields": len(keys),
        "fields": keys,
    }


def validate_source_attention_shapes(
    tensor_info: dict[str, SourceTensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
    dtype: str | None,
) -> None:
    hidden = dims["hidden"]
    require_source_tensor_shape(tensor_info, source_name(layer, "input_layernorm.weight"), (hidden,), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "post_attention_layernorm.weight"), (hidden,), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "self_attn.q_a_layernorm.weight"), (dims["q_lora"],), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "self_attn.kv_a_layernorm.weight"), (dims["kv_lora"],), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "self_attn.q_a_proj.weight"), (dims["q_lora"], hidden), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "self_attn.q_b_proj.weight"), (dims["q_width"], dims["q_lora"]), errors, dtype=dtype)
    require_source_tensor_shape(
        tensor_info,
        source_name(layer, "self_attn.kv_a_proj_with_mqa.weight"),
        (dims["kv_lora"] + dims["qk_rope"], hidden),
        errors,
        dtype=dtype,
    )
    require_source_tensor_shape(
        tensor_info,
        source_name(layer, "self_attn.kv_b_proj.weight"),
        (dims["kv_b_width"], dims["kv_lora"]),
        errors,
        dtype=dtype,
    )
    require_source_tensor_shape(tensor_info, source_name(layer, "self_attn.o_proj.weight"), (hidden, dims["v_width"]), errors, dtype=dtype)


def validate_source_indexer_shapes(
    tensor_info: dict[str, SourceTensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
    dtype: str | None,
) -> None:
    require_source_tensor_shape(tensor_info, source_name(layer, "self_attn.indexer.k_norm.weight"), (dims["index_head"],), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "self_attn.indexer.k_norm.bias"), (dims["index_head"],), errors, dtype=dtype)
    require_source_tensor_shape(
        tensor_info,
        source_name(layer, "self_attn.indexer.weights_proj.weight"),
        (dims["index_heads"], dims["hidden"]),
        errors,
        dtype=dtype,
    )
    require_source_tensor_shape(
        tensor_info,
        source_name(layer, "self_attn.indexer.wk.weight"),
        (dims["index_head"], dims["hidden"]),
        errors,
        dtype=dtype,
    )
    require_source_tensor_shape(
        tensor_info,
        source_name(layer, "self_attn.indexer.wq_b.weight"),
        (dims["index_width"], dims["q_lora"]),
        errors,
        dtype=dtype,
    )


def validate_source_dense_mlp_shapes(
    tensor_info: dict[str, SourceTensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
    dtype: str | None,
) -> None:
    require_source_tensor_shape(tensor_info, source_name(layer, "mlp.gate_proj.weight"), (dims["dense_ff"], dims["hidden"]), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "mlp.up_proj.weight"), (dims["dense_ff"], dims["hidden"]), errors, dtype=dtype)
    require_source_tensor_shape(tensor_info, source_name(layer, "mlp.down_proj.weight"), (dims["hidden"], dims["dense_ff"]), errors, dtype=dtype)


def validate_source_sparse_mlp_shapes(
    tensor_info: dict[str, SourceTensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
    dtype: str | None,
) -> None:
    require_source_tensor_shape(tensor_info, source_name(layer, "mlp.gate.weight"), (dims["experts"], dims["hidden"]), errors, dtype=dtype)
    require_source_tensor_shape(
        tensor_info,
        source_name(layer, "mlp.gate.e_score_correction_bias"),
        (dims["experts"],),
        errors,
        dtype="F32",
    )
    for expert in range(dims["experts"]):
        prefix = f"mlp.experts.{expert}"
        require_source_tensor_shape(
            tensor_info,
            source_name(layer, f"{prefix}.gate_proj.weight"),
            (dims["moe_ff"], dims["hidden"]),
            errors,
            dtype=dtype,
        )
        require_source_tensor_shape(
            tensor_info,
            source_name(layer, f"{prefix}.up_proj.weight"),
            (dims["moe_ff"], dims["hidden"]),
            errors,
            dtype=dtype,
        )
        require_source_tensor_shape(
            tensor_info,
            source_name(layer, f"{prefix}.down_proj.weight"),
            (dims["hidden"], dims["moe_ff"]),
            errors,
            dtype=dtype,
        )
    for suffix in ["gate_proj.weight", "up_proj.weight"]:
        require_source_tensor_shape(
            tensor_info,
            source_name(layer, f"mlp.shared_experts.{suffix}"),
            (dims["moe_ff"] * dims["shared_experts"], dims["hidden"]),
            errors,
            dtype=dtype,
        )
    require_source_tensor_shape(
        tensor_info,
        source_name(layer, "mlp.shared_experts.down_proj.weight"),
        (dims["hidden"], dims["moe_ff"] * dims["shared_experts"]),
        errors,
        dtype=dtype,
    )


def validate_source_config_contract(
    config: dict[str, object],
    target_layers: object,
    nextn_layers: object,
    errors: list[str],
    *,
    strict_glm52: bool,
) -> dict[str, int]:
    dims = glm52_expected_dims(config, errors)
    if not strict_glm52:
        return dims
    expected_values = {
        "hidden": 6144,
        "vocab": 154880,
        "heads": 64,
        "q_lora": 2048,
        "kv_lora": 512,
        "qk_nope": 192,
        "qk_rope": 64,
        "v_head": 256,
        "dense_ff": 12288,
        "moe_ff": 2048,
        "experts": 256,
        "shared_experts": 1,
        "experts_used": 8,
        "index_heads": 32,
        "index_head": 128,
        "index_topk": 2048,
        "dense_lead": 3,
    }
    for key, expected in expected_values.items():
        require(dims[key] == expected, f"config-derived {key} {dims[key]} != {expected}", errors)
    require(target_layers == 78, f"GLM-5.2 target layer count should be 78, got {target_layers}", errors)
    require(nextn_layers == 1, f"GLM-5.2 native MTP/NextN layer count should be 1, got {nextn_layers}", errors)
    require(config.get("indexer_rope_interleave") is True, "config indexer_rope_interleave must be true", errors)
    require(config.get("rope_interleave") is True, "config rope_interleave must be true", errors)
    require(config.get("norm_topk_prob") is True, "config norm_topk_prob must be true", errors)
    require(config.get("scoring_func") == "sigmoid", "config scoring_func must be sigmoid", errors)
    return dims


def glm52_expected_gguf_dims(metadata: dict[str, object], errors: list[str], *, strict_glm52: bool) -> dict[str, int]:
    def required_meta_int(key: str) -> int:
        value = metadata.get(key)
        require(isinstance(value, int), f"missing or invalid GGUF metadata {key}", errors)
        return value if isinstance(value, int) else 0

    dims = {
        "hidden": required_meta_int("glm-dsa.embedding_length"),
        "heads": required_meta_int("glm-dsa.attention.head_count"),
        "key_length": required_meta_int("glm-dsa.attention.key_length"),
        "value_length": required_meta_int("glm-dsa.attention.value_length"),
        "key_mla": required_meta_int("glm-dsa.attention.key_length_mla"),
        "value_mla": required_meta_int("glm-dsa.attention.value_length_mla"),
        "q_lora": required_meta_int("glm-dsa.attention.q_lora_rank"),
        "kv_lora": required_meta_int("glm-dsa.attention.kv_lora_rank"),
        "rope": required_meta_int("glm-dsa.rope.dimension_count"),
        "dense_ff": required_meta_int("glm-dsa.feed_forward_length"),
        "moe_ff": required_meta_int("glm-dsa.expert_feed_forward_length"),
        "experts": required_meta_int("glm-dsa.expert_count"),
        "shared_experts": required_meta_int("glm-dsa.expert_shared_count"),
        "experts_used": required_meta_int("glm-dsa.expert_used_count"),
        "dense_lead": required_meta_int("glm-dsa.leading_dense_block_count"),
        "index_heads": required_meta_int("glm-dsa.attention.indexer.head_count"),
        "index_head": required_meta_int("glm-dsa.attention.indexer.key_length"),
        "index_topk": required_meta_int("glm-dsa.attention.indexer.top_k"),
    }
    dims["q_width"] = dims["heads"] * dims["key_mla"]
    dims["v_width"] = dims["heads"] * dims["value_length"]
    dims["key_nope"] = dims["key_mla"] - dims["rope"]
    dims["index_width"] = dims["index_heads"] * dims["index_head"]
    if strict_glm52:
        expected_values = {
            "hidden": 6144,
            "heads": 64,
            "key_length": 576,
            "value_length": 256,
            "key_mla": 256,
            "value_mla": 256,
            "q_lora": 2048,
            "kv_lora": 512,
            "rope": 64,
            "dense_ff": 12288,
            "moe_ff": 2048,
            "experts": 256,
            "shared_experts": 1,
            "experts_used": 8,
            "dense_lead": 3,
            "index_heads": 32,
            "index_head": 128,
            "index_topk": 2048,
        }
        for key, expected in expected_values.items():
            require(dims[key] == expected, f"GGUF-derived {key} {dims[key]} != {expected}", errors)
        require(dims["key_nope"] == 192, f"GGUF-derived key_nope {dims['key_nope']} != 192", errors)
    return dims


def validate_gguf_attention_shapes(
    tensor_info: dict[str, TensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
) -> None:
    hidden = dims["hidden"]
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "attn_norm.weight"), (hidden,), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "ffn_norm.weight"), (hidden,), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "attn_q_a_norm.weight"), (dims["q_lora"],), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "attn_kv_a_norm.weight"), (dims["kv_lora"],), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "attn_q_a.weight"), (hidden, dims["q_lora"]), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "attn_q_b.weight"), (dims["q_lora"], dims["q_width"]), errors)
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "attn_kv_a_mqa.weight"),
        (hidden, dims["key_length"]),
        errors,
    )
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "attn_k_b.weight"),
        (dims["key_nope"], dims["kv_lora"], dims["heads"]),
        errors,
    )
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "attn_v_b.weight"),
        (dims["kv_lora"], dims["value_mla"], dims["heads"]),
        errors,
    )
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "attn_output.weight"), (dims["v_width"], hidden), errors)


def validate_gguf_indexer_shapes(
    tensor_info: dict[str, TensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
) -> None:
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "indexer.k_norm.weight"), (dims["index_head"],), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "indexer.k_norm.bias"), (dims["index_head"],), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "indexer.proj.weight"), (dims["hidden"], dims["index_heads"]), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "indexer.attn_k.weight"), (dims["hidden"], dims["index_head"]), errors)
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "indexer.attn_q_b.weight"),
        (dims["q_lora"], dims["index_width"]),
        errors,
    )


def validate_gguf_dense_mlp_shapes(
    tensor_info: dict[str, TensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
) -> None:
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "ffn_gate.weight"), (dims["hidden"], dims["dense_ff"]), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "ffn_up.weight"), (dims["hidden"], dims["dense_ff"]), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "ffn_down.weight"), (dims["dense_ff"], dims["hidden"]), errors)


def validate_gguf_sparse_mlp_shapes(
    tensor_info: dict[str, TensorInfo],
    layer: int,
    dims: dict[str, int],
    errors: list[str],
) -> None:
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "ffn_gate_inp.weight"), (dims["hidden"], dims["experts"]), errors)
    require_gguf_tensor_shape(tensor_info, gguf_name(layer, "exp_probs_b.bias"), (dims["experts"],), errors)
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "ffn_gate_exps.weight"),
        (dims["hidden"], dims["moe_ff"], dims["experts"]),
        errors,
    )
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "ffn_up_exps.weight"),
        (dims["hidden"], dims["moe_ff"], dims["experts"]),
        errors,
    )
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "ffn_down_exps.weight"),
        (dims["moe_ff"], dims["hidden"], dims["experts"]),
        errors,
    )
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "ffn_gate_shexp.weight"),
        (dims["hidden"], dims["moe_ff"] * dims["shared_experts"]),
        errors,
    )
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "ffn_up_shexp.weight"),
        (dims["hidden"], dims["moe_ff"] * dims["shared_experts"]),
        errors,
    )
    require_gguf_tensor_shape(
        tensor_info,
        gguf_name(layer, "ffn_down_shexp.weight"),
        (dims["moe_ff"] * dims["shared_experts"], dims["hidden"]),
        errors,
    )


def verify_source_checkpoint(root: Path, expected_target_layers: int, expected_nextn_layers: int) -> dict[str, object]:
    config = json.loads(root.joinpath("config.json").read_text())
    inventory = read_checkpoint_inventory(root)
    tensors = inventory.tensors
    tensor_info = inventory.tensor_info
    errors: list[str] = []
    strict_glm52 = expected_target_layers == 78 and expected_nextn_layers == 1
    source_weight_dtype = "BF16" if strict_glm52 else None

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
    dims = validate_source_config_contract(config, target_layers, nextn_layers, errors, strict_glm52=strict_glm52)
    require(isinstance(roles, list), "indexer_types missing from config", errors)
    if isinstance(roles, list):
        require(len(roles) == target_layers, f"indexer_types length {len(roles)} != target layers {target_layers}", errors)
        require(set(roles) <= {"full", "shared"}, "indexer_types contains values other than full/shared", errors)
    validate_indexshare_frequency_contract(roles, frequency, offset, errors, label="source")

    mlp_layer_types = config.get("mlp_layer_types")
    if strict_glm52:
        require(isinstance(mlp_layer_types, list), "mlp_layer_types missing from config", errors)
    if isinstance(mlp_layer_types, list):
        require(len(mlp_layer_types) == target_layers, f"mlp_layer_types length {len(mlp_layer_types)} != target layers {target_layers}", errors)
        for layer, layer_type in enumerate(mlp_layer_types):
            expected = "dense" if layer < dims["dense_lead"] else "sparse"
            require(layer_type == expected, f"mlp_layer_types[{layer}] {layer_type!r} != {expected!r}", errors)
    if strict_glm52:
        validate_strict_glm52_indexshare_contract(roles, frequency, offset, errors, label="source")

    expected_block_count = expected_target_layers + expected_nextn_layers
    require(
        not any(name.startswith(f"model.layers.{expected_block_count}.") for name in tensors),
        f"source contains unexpected model.layers.{expected_block_count} tensors",
        errors,
    )

    if isinstance(roles, list):
        for layer, role in enumerate(roles):
            validate_source_attention_shapes(tensor_info, layer, dims, errors, source_weight_dtype)
            if layer < dims["dense_lead"]:
                validate_source_dense_mlp_shapes(tensor_info, layer, dims, errors, source_weight_dtype)
            else:
                validate_source_sparse_mlp_shapes(tensor_info, layer, dims, errors, source_weight_dtype)
            has_indexer = all(source_name(layer, suffix) in tensors for suffix in INDEXER_SUFFIXES)
            require(has_indexer == (role == "full"), f"layer {layer} role {role} indexer tensor presence is {has_indexer}", errors)
            if role == "full":
                validate_source_indexer_shapes(tensor_info, layer, dims, errors, source_weight_dtype)

    if isinstance(mtp_layer, int):
        validate_source_attention_shapes(tensor_info, mtp_layer, dims, errors, source_weight_dtype)
        validate_source_sparse_mlp_shapes(tensor_info, mtp_layer, dims, errors, source_weight_dtype)
        validate_source_indexer_shapes(tensor_info, mtp_layer, dims, errors, source_weight_dtype)
        require_source_tensor_shape(tensor_info, source_name(mtp_layer, "eh_proj.weight"), (dims["hidden"], 2 * dims["hidden"]), errors, dtype=source_weight_dtype)
        require_source_tensor_shape(tensor_info, source_name(mtp_layer, "enorm.weight"), (dims["hidden"],), errors, dtype=source_weight_dtype)
        require_source_tensor_shape(tensor_info, source_name(mtp_layer, "hnorm.weight"), (dims["hidden"],), errors, dtype=source_weight_dtype)
        require_source_tensor_shape(tensor_info, source_name(mtp_layer, "shared_head.norm.weight"), (dims["hidden"],), errors, dtype=source_weight_dtype)

    if errors:
        fail("source checkpoint inventory failed:\n  - " + "\n  - ".join(errors))

    return {
        "checkpoint_files": inventory.files,
        "target_layers": target_layers,
        "nextn_layers": nextn_layers,
        "source_tensors": len(tensors),
        "full_roles": roles.count("full") if isinstance(roles, list) else None,
        "shared_roles": roles.count("shared") if isinstance(roles, list) else None,
        "dense_lead_layers": dims["dense_lead"],
        "hidden_size": dims["hidden"],
        "attention_q_width": dims["q_width"],
        "attention_v_width": dims["v_width"],
        "indexer_top_k": dims["index_topk"],
        "contract": source_contract_summary(dims, roles, frequency, offset, target_layers, nextn_layers),
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
    tensor_info = inventory.tensor_info
    metadata = inventory.metadata
    strict_glm52 = expected_target_layers == 78 and expected_nextn_layers == 1

    block_count = metadata.get("glm-dsa.block_count")
    nextn_layers = metadata.get("glm-dsa.nextn_predict_layers", 0)
    roles = metadata.get("glm-dsa.attention.indexer.types")
    frequency = metadata.get("glm-dsa.attention.indexer.top_k_frequency")
    offset = metadata.get("glm-dsa.attention.indexer.skip_top_k_offset")
    expected_block_count = expected_target_layers + expected_nextn_layers
    mtp_layer = expected_target_layers

    require(metadata.get("general.architecture") == "glm-dsa", "GGUF architecture is not glm-dsa", errors)
    if strict_glm52:
        require(metadata.get("glm-dsa.leading_dense_block_count") == 3, "expected leading_dense_block_count 3", errors)
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
    dims = glm52_expected_gguf_dims(metadata, errors, strict_glm52=strict_glm52)
    require(isinstance(roles, list), "GGUF indexer.types metadata missing", errors)
    if isinstance(roles, list):
        require(
            len(roles) == expected_target_layers,
            f"GGUF indexer.types length {len(roles)} != target layers {expected_target_layers}",
            errors,
        )
        require(set(roles) <= {"full", "shared"}, "GGUF indexer.types contains values other than full/shared", errors)
    validate_indexshare_frequency_contract(roles, frequency, offset, errors, label="GGUF")
    if strict_glm52:
        validate_strict_glm52_indexshare_contract(roles, frequency, offset, errors, label="GGUF")

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

    if not partial and isinstance(roles, list):
        for layer, role in enumerate(roles):
            validate_gguf_attention_shapes(tensor_info, layer, dims, errors)
            if layer < dims["dense_lead"]:
                validate_gguf_dense_mlp_shapes(tensor_info, layer, dims, errors)
            else:
                validate_gguf_sparse_mlp_shapes(tensor_info, layer, dims, errors)
            present_indexers = [gguf_name(layer, suffix) for suffix in GGUF_INDEXER_NAMES if gguf_name(layer, suffix) in tensors]
            if role == "full":
                validate_gguf_indexer_shapes(tensor_info, layer, dims, errors)
            elif role == "shared":
                require(
                    not present_indexers,
                    f"Shared layer {layer} has GGUF indexer tensors: {present_indexers[:5]}",
                    errors,
                )
            else:
                require(False, f"layer {layer} has unsupported role {role!r}", errors)

    if not partial:
        validate_gguf_attention_shapes(tensor_info, mtp_layer, dims, errors)
        validate_gguf_sparse_mlp_shapes(tensor_info, mtp_layer, dims, errors)
        validate_gguf_indexer_shapes(tensor_info, mtp_layer, dims, errors)
        require_gguf_tensor_shape(tensor_info, gguf_name(mtp_layer, "nextn.eh_proj.weight"), (2 * dims["hidden"], dims["hidden"]), errors)
        require_gguf_tensor_shape(tensor_info, gguf_name(mtp_layer, "nextn.enorm.weight"), (dims["hidden"],), errors)
        require_gguf_tensor_shape(tensor_info, gguf_name(mtp_layer, "nextn.hnorm.weight"), (dims["hidden"],), errors)
        require_gguf_tensor_shape(tensor_info, gguf_name(mtp_layer, "nextn.shared_head_norm.weight"), (dims["hidden"],), errors)

    bf16_count = sum(1 for info in tensor_info.values() if info.ggml_type == 30)
    f32_count = sum(1 for info in tensor_info.values() if info.ggml_type == 0)

    if errors:
        fail("GGUF inventory failed:\n  - " + "\n  - ".join(errors))

    return {
        "gguf_files": inventory.files,
        "gguf_tensors": len(tensors),
        "gguf_bf16_tensors": bf16_count,
        "gguf_f32_tensors": f32_count,
        "block_count": block_count,
        "nextn_layers": nextn_layers,
        "role_source": "metadata_types" if isinstance(roles, list) else "none",
        "full_roles": roles.count("full") if isinstance(roles, list) else None,
        "shared_roles": roles.count("shared") if isinstance(roles, list) else None,
        "dense_lead_layers": dims["dense_lead"],
        "hidden_size": dims["hidden"],
        "attention_q_width": dims["q_width"],
        "attention_v_width": dims["v_width"],
        "indexer_top_k": dims["index_topk"],
        "first_full_layers": [idx for idx, role in enumerate(roles) if role == "full"][:10] if isinstance(roles, list) else [],
        "first_shared_layers": [idx for idx, role in enumerate(roles) if role == "shared"][:10] if isinstance(roles, list) else [],
        "partial": partial,
        "contains_unsplit_kv_b": bool(unsplit_kv_b),
        "contract": gguf_contract_summary(dims, roles, frequency, offset, block_count, nextn_layers),
    }


def stale_shard_report(root: Path, expected_target_layers: int, expected_nextn_layers: int) -> dict[str, object]:
    paths = gguf_paths(root)
    stale_files: list[dict[str, object]] = []
    stale_layers: set[int] = set()
    merged = Inventory()

    tensor_info_offset_hint: int | None = None
    for index, path in enumerate(paths):
        inventory = parse_gguf_header(
            path,
            collect_metadata=index == 0,
            tensor_info_offset_hint=tensor_info_offset_hint,
        )
        if index == 0:
            tensor_info_offset_hint = inventory.tensor_info_offset
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
    parser.add_argument("--checkpoint", type=Path, help="GLM-5.2 SafeTensors checkpoint directory")
    parser.add_argument("--gguf", type=Path, help="BF16 GGUF file or directory to verify")
    parser.add_argument("--expected-target-layers", type=int, default=78)
    parser.add_argument("--expected-nextn-layers", type=int, default=1)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in negative tests for the strict GLM-5.2 contract verifier.",
    )
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
    if args.self_test:
        summary = {"self_test": run_self_tests()}
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print("GLM-DSA inventory verifier self-test passed")
            print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.checkpoint is None:
        fail("--checkpoint is required unless --self-test is set")

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
        if args.expected_target_layers == 78 and args.expected_nextn_layers == 1:
            summary["source_to_gguf"] = compare_contract_summaries(
                summary["checkpoint"]["contract"],
                summary["gguf"]["contract"],
            )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("GLM-DSA inventory verification passed")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
