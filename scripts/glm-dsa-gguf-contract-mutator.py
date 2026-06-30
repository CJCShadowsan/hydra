#!/usr/bin/env python3
"""Create tiny GLM-DSA GGUF contract variants for llama.cpp load smokes."""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

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

INDEXSHARE_KEYS = {
    "glm-dsa.attention.indexer.types",
    "glm-dsa.attention.indexer.top_k_frequency",
    "glm-dsa.attention.indexer.skip_top_k_offset",
}

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


@dataclass
class Field:
    key: str
    value_type: int
    raw: bytes


@dataclass
class TensorInfo:
    name: str
    raw_after_name: bytes


@dataclass
class ParsedGguf:
    version: int
    tensor_count: int
    fields: list[Field]
    tensor_infos: list[TensorInfo]
    data: bytes
    alignment: int


def read_u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def read_u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = read_u64(data, offset)
    end = offset + length
    return data[offset:end].decode("utf-8"), end


def write_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def skip_value(data: bytes, offset: int, value_type: int) -> int:
    if value_type in SCALAR_WIDTHS:
        return offset + SCALAR_WIDTHS[value_type]
    if value_type == GGUF_TYPE_STRING:
        length, value_offset = read_u64(data, offset)
        return value_offset + length
    if value_type == GGUF_TYPE_ARRAY:
        element_type, offset = read_u32(data, offset)
        length, offset = read_u64(data, offset)
        if element_type == GGUF_TYPE_STRING:
            for _ in range(length):
                _, offset = read_string(data, offset)
            return offset
        if element_type not in SCALAR_WIDTHS:
            raise ValueError(f"unsupported GGUF array element type {element_type}")
        return offset + SCALAR_WIDTHS[element_type] * length
    raise ValueError(f"unsupported GGUF value type {value_type}")


def parse_gguf(path: Path) -> ParsedGguf:
    data = path.read_bytes()
    offset = 0
    if data[:4] != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF file")
    offset += 4
    version, offset = read_u32(data, offset)
    tensor_count, offset = read_u64(data, offset)
    kv_count, offset = read_u64(data, offset)

    fields = []
    alignment = GGUF_DEFAULT_ALIGNMENT
    for _ in range(kv_count):
        key, offset = read_string(data, offset)
        value_type, offset = read_u32(data, offset)
        value_start = offset
        offset = skip_value(data, offset, value_type)
        raw = data[value_start:offset]
        fields.append(Field(key, value_type, raw))
        if key == "general.alignment" and value_type == GGUF_TYPE_UINT32:
            alignment = struct.unpack_from("<I", raw, 0)[0]

    tensor_infos = []
    for _ in range(tensor_count):
        name, offset = read_string(data, offset)
        raw_start = offset
        dim_count, offset = read_u32(data, offset)
        offset += 8 * dim_count
        _, offset = read_u32(data, offset)
        _, offset = read_u64(data, offset)
        tensor_infos.append(TensorInfo(name, data[raw_start:offset]))
    data_start = align(offset, alignment)
    return ParsedGguf(
        version=version,
        tensor_count=tensor_count,
        fields=fields,
        tensor_infos=tensor_infos,
        data=data[data_start:],
        alignment=alignment,
    )


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def array_string_raw(values: list[str]) -> bytes:
    out = bytearray()
    out += struct.pack("<I", GGUF_TYPE_STRING)
    out += struct.pack("<Q", len(values))
    for value in values:
        out += write_string(value)
    return bytes(out)


def u32_raw(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"uint32 value out of range: {value}")
    return struct.pack("<I", value)


def parse_u32_assignment(value: str) -> tuple[str, int]:
    key, separator, raw = value.partition("=")
    if not separator or not key:
        raise ValueError(f"expected KEY=VALUE uint32 assignment, got: {value}")
    return key, int(raw, 0)


def parse_rename_assignment(value: str) -> tuple[str, str]:
    old, separator, new = value.partition("=")
    if not separator or not old or not new:
        raise ValueError(f"expected OLD=NEW tensor rename, got: {value}")
    return old, new


def parse_shape_assignment(value: str) -> tuple[str, list[int]]:
    name, separator, raw_shape = value.partition("=")
    if not separator or not name or not raw_shape:
        raise ValueError(f"expected NAME=DIM[,DIM...] tensor shape assignment, got: {value}")
    shape = [int(raw_dim, 0) for raw_dim in raw_shape.split(",")]
    if not shape or len(shape) > 4 or any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid tensor shape for {name}: {raw_shape}")
    return name, shape


def tensor_info_shape_and_tail(info: TensorInfo) -> tuple[list[int], bytes]:
    dim_count, offset = read_u32(info.raw_after_name, 0)
    dims = []
    for _ in range(dim_count):
        dim, offset = read_u64(info.raw_after_name, offset)
        dims.append(dim)
    return dims, info.raw_after_name[offset:]


def tensor_info_with_shape(info: TensorInfo, shape: list[int]) -> TensorInfo:
    _, tail = tensor_info_shape_and_tail(info)
    raw = bytearray()
    raw += struct.pack("<I", len(shape))
    for dim in shape:
        raw += struct.pack("<Q", dim)
    raw += tail
    return TensorInfo(info.name, bytes(raw))


def mutate_fields(args: argparse.Namespace, fields: list[Field]) -> list[Field]:
    u32_overrides = dict(parse_u32_assignment(item) for item in args.set_u32)
    mutated = []
    for field in fields:
        if args.drop_indexshare_metadata and field.key in INDEXSHARE_KEYS:
            continue
        if args.set_indexer_types and field.key == "glm-dsa.attention.indexer.types":
            continue
        if field.key in u32_overrides:
            continue
        mutated.append(field)

    if args.set_indexer_types:
        mutated.append(
            Field(
                "glm-dsa.attention.indexer.types",
                GGUF_TYPE_ARRAY,
                array_string_raw(args.set_indexer_types.split(",")),
            )
        )
    for key, value in u32_overrides.items():
        mutated.append(Field(key, GGUF_TYPE_UINT32, u32_raw(value)))
    return mutated


def mutate_tensor_infos(args: argparse.Namespace, tensor_infos: list[TensorInfo]) -> list[TensorInfo]:
    renames = dict(parse_rename_assignment(item) for item in args.rename_tensor)
    shape_overrides = dict(parse_shape_assignment(item) for item in args.set_tensor_shape)
    mutated = []
    seen_old_names = set()
    seen_shape_names = set()
    for info in tensor_infos:
        if info.name in renames:
            seen_old_names.add(info.name)
        info = TensorInfo(renames.get(info.name, info.name), info.raw_after_name)
        if info.name in shape_overrides:
            seen_shape_names.add(info.name)
            info = tensor_info_with_shape(info, shape_overrides[info.name])
        mutated.append(info)
    missing = sorted(set(renames) - seen_old_names)
    if missing:
        raise ValueError(f"tensor rename source not found: {missing}")
    missing_shapes = sorted(set(shape_overrides) - seen_shape_names)
    if missing_shapes:
        raise ValueError(f"tensor shape target not found: {missing_shapes}")
    return mutated


def write_gguf(path: Path, parsed: ParsedGguf, fields: list[Field]) -> None:
    out = bytearray()
    out += GGUF_MAGIC
    out += struct.pack("<I", parsed.version)
    out += struct.pack("<Q", parsed.tensor_count)
    out += struct.pack("<Q", len(fields))
    for field in fields:
        out += write_string(field.key)
        out += struct.pack("<I", field.value_type)
        out += field.raw
    for info in parsed.tensor_infos:
        out += write_string(info.name)
        out += info.raw_after_name
    out += b"\0" * (align(len(out), parsed.alignment) - len(out))
    out += parsed.data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--drop-indexshare-metadata", action="store_true")
    parser.add_argument("--set-indexer-types")
    parser.add_argument(
        "--set-u32",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set or add a uint32 metadata value. May be repeated.",
    )
    parser.add_argument(
        "--rename-tensor",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rename one tensor metadata entry. May be repeated.",
    )
    parser.add_argument(
        "--set-tensor-shape",
        action="append",
        default=[],
        metavar="NAME=DIM[,DIM...]",
        help="Set one tensor metadata shape. Applied after renames. May be repeated.",
    )
    args = parser.parse_args()
    if (
        not args.drop_indexshare_metadata
        and not args.set_indexer_types
        and not args.set_u32
        and not args.rename_tensor
        and not args.set_tensor_shape
    ):
        parser.error(
            "provide --drop-indexshare-metadata, --set-indexer-types, --set-u32, "
            "--rename-tensor, or --set-tensor-shape"
        )
    return args


def main() -> None:
    args = parse_args()
    parsed = parse_gguf(args.input)
    fields = mutate_fields(args, parsed.fields)
    parsed.tensor_infos = mutate_tensor_infos(args, parsed.tensor_infos)
    write_gguf(args.output, parsed, fields)


if __name__ == "__main__":
    main()
