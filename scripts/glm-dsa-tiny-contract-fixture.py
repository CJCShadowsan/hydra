#!/usr/bin/env python3
"""Generate a tiny GLM-DSA SafeTensors checkpoint for native llama.cpp smokes."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path


VOCAB_SIZE = 8
HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 8
DEFAULT_TARGET_LAYERS = 3
ATTENTION_HEADS = 1
QK_NOPE_HEAD_DIM = 3
QK_ROPE_HEAD_DIM = 2
V_HEAD_DIM = 2
Q_LORA_RANK = 2
KV_LORA_RANK = 2
INDEX_HEADS = 1
INDEX_HEAD_DIM = 64
ROUTED_EXPERTS = 2
EXPERTS_PER_TOKEN = 1
SHARED_EXPERTS = 1
MOE_INTERMEDIATE_SIZE = 2


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: list[int]
    data: bytes


def f32_bytes(count: int, value: float) -> bytes:
    return struct.pack("<" + "f" * count, *([value] * count))


def bf16_bytes(count: int, value: int = 0) -> bytes:
    return struct.pack("<" + "H" * count, *([value] * count))


def element_count(shape: list[int]) -> int:
    out = 1
    for item in shape:
        out *= item
    return out


def add_f32(tensors: list[Tensor], name: str, shape: list[int], value: float = 0.01) -> None:
    tensors.append(Tensor(name, "F32", shape, f32_bytes(element_count(shape), value)))


def add_bf16(tensors: list[Tensor], name: str, shape: list[int]) -> None:
    tensors.append(Tensor(name, "BF16", shape, bf16_bytes(element_count(shape))))


def layer_name(layer: int, suffix: str) -> str:
    return f"model.layers.{layer}.{suffix}"


def add_attention(tensors: list[Tensor], layer: int) -> None:
    add_f32(tensors, layer_name(layer, "input_layernorm.weight"), [HIDDEN_SIZE], 1.0)
    add_f32(tensors, layer_name(layer, "self_attn.q_a_layernorm.weight"), [Q_LORA_RANK], 1.0)
    add_f32(tensors, layer_name(layer, "self_attn.kv_a_layernorm.weight"), [KV_LORA_RANK], 1.0)
    add_f32(tensors, layer_name(layer, "self_attn.q_a_proj.weight"), [Q_LORA_RANK, HIDDEN_SIZE])
    add_f32(
        tensors,
        layer_name(layer, "self_attn.q_b_proj.weight"),
        [ATTENTION_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), Q_LORA_RANK],
    )
    add_f32(
        tensors,
        layer_name(layer, "self_attn.kv_a_proj_with_mqa.weight"),
        [KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_SIZE],
    )
    add_bf16(
        tensors,
        layer_name(layer, "self_attn.kv_b_proj.weight"),
        [ATTENTION_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK],
    )
    add_f32(
        tensors,
        layer_name(layer, "self_attn.o_proj.weight"),
        [HIDDEN_SIZE, ATTENTION_HEADS * V_HEAD_DIM],
    )
    add_f32(tensors, layer_name(layer, "post_attention_layernorm.weight"), [HIDDEN_SIZE], 1.0)


def add_dense_ffn(tensors: list[Tensor], layer: int) -> None:
    add_f32(tensors, layer_name(layer, "mlp.gate_proj.weight"), [INTERMEDIATE_SIZE, HIDDEN_SIZE])
    add_f32(tensors, layer_name(layer, "mlp.down_proj.weight"), [HIDDEN_SIZE, INTERMEDIATE_SIZE])
    add_f32(tensors, layer_name(layer, "mlp.up_proj.weight"), [INTERMEDIATE_SIZE, HIDDEN_SIZE])


def add_moe(tensors: list[Tensor], layer: int) -> None:
    add_f32(tensors, layer_name(layer, "mlp.gate.weight"), [ROUTED_EXPERTS, HIDDEN_SIZE])
    add_f32(tensors, layer_name(layer, "mlp.gate.e_score_correction_bias"), [ROUTED_EXPERTS])
    add_f32(
        tensors,
        layer_name(layer, "mlp.shared_experts.gate_proj.weight"),
        [MOE_INTERMEDIATE_SIZE * SHARED_EXPERTS, HIDDEN_SIZE],
    )
    add_f32(
        tensors,
        layer_name(layer, "mlp.shared_experts.down_proj.weight"),
        [HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE * SHARED_EXPERTS],
    )
    add_f32(
        tensors,
        layer_name(layer, "mlp.shared_experts.up_proj.weight"),
        [MOE_INTERMEDIATE_SIZE * SHARED_EXPERTS, HIDDEN_SIZE],
    )
    for expert in range(ROUTED_EXPERTS):
        add_f32(tensors, layer_name(layer, f"mlp.experts.{expert}.gate_proj.weight"), [MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE])
        add_f32(tensors, layer_name(layer, f"mlp.experts.{expert}.down_proj.weight"), [HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE])
        add_f32(tensors, layer_name(layer, f"mlp.experts.{expert}.up_proj.weight"), [MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE])


def add_indexer(tensors: list[Tensor], layer: int) -> None:
    add_f32(tensors, layer_name(layer, "self_attn.indexer.k_norm.weight"), [INDEX_HEAD_DIM], 1.0)
    add_f32(tensors, layer_name(layer, "self_attn.indexer.k_norm.bias"), [INDEX_HEAD_DIM])
    add_f32(tensors, layer_name(layer, "self_attn.indexer.weights_proj.weight"), [INDEX_HEADS, HIDDEN_SIZE])
    add_f32(tensors, layer_name(layer, "self_attn.indexer.wk.weight"), [INDEX_HEAD_DIM, HIDDEN_SIZE])
    add_f32(tensors, layer_name(layer, "self_attn.indexer.wq_b.weight"), [INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK])


def add_mtp(tensors: list[Tensor], layer: int) -> None:
    add_f32(tensors, layer_name(layer, "eh_proj.weight"), [HIDDEN_SIZE, HIDDEN_SIZE * 2])
    add_f32(tensors, layer_name(layer, "enorm.weight"), [HIDDEN_SIZE], 1.0)
    add_f32(tensors, layer_name(layer, "hnorm.weight"), [HIDDEN_SIZE], 1.0)
    add_f32(tensors, layer_name(layer, "shared_head.norm.weight"), [HIDDEN_SIZE], 1.0)


def build_tensors(indexer_types: list[str]) -> list[Tensor]:
    target_layers = len(indexer_types)
    mtp_layer = target_layers
    tensors: list[Tensor] = []
    add_f32(tensors, "model.embed_tokens.weight", [VOCAB_SIZE, HIDDEN_SIZE])
    add_f32(tensors, "model.norm.weight", [HIDDEN_SIZE], 1.0)
    for layer in range(target_layers):
        add_attention(tensors, layer)
    add_attention(tensors, mtp_layer)
    add_dense_ffn(tensors, 0)
    for layer in range(1, target_layers):
        add_moe(tensors, layer)
    add_moe(tensors, mtp_layer)
    for layer, role in enumerate(indexer_types):
        if role == "full":
            add_indexer(tensors, layer)
    add_indexer(tensors, mtp_layer)
    add_mtp(tensors, mtp_layer)
    return tensors


def write_config(root: Path, indexer_types: list[str], index_topk_freq: int, index_skip_topk_offset: int) -> None:
    target_layers = len(indexer_types)
    config = {
        "model_type": "glm_moe_dsa",
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": 128,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": target_layers,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": ATTENTION_HEADS,
        "num_key_value_heads": 1,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "q_lora_rank": Q_LORA_RANK,
        "kv_lora_rank": KV_LORA_RANK,
        "index_n_heads": INDEX_HEADS,
        "index_head_dim": INDEX_HEAD_DIM,
        "index_topk": 2,
        "index_topk_freq": index_topk_freq,
        "index_skip_topk_offset": index_skip_topk_offset,
        "indexer_types": indexer_types,
        "n_routed_experts": ROUTED_EXPERTS,
        "num_experts_per_tok": EXPERTS_PER_TOKEN,
        "n_shared_experts": SHARED_EXPERTS,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "rms_norm_eps": 1e-5,
    }
    root.joinpath("config.json").write_text(json.dumps(config, indent=2) + "\n")


def write_safetensors(path: Path, tensors: list[Tensor]) -> None:
    offset = 0
    entries = {}
    payload = bytearray()
    for tensor in sorted(tensors, key=lambda item: item.name):
        end = offset + len(tensor.data)
        entries[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offset, end],
        }
        payload += tensor.data
        offset = end

    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def write_tokenizer(root: Path) -> None:
    tokenizer = {
        "model": {
            "type": "BPE",
            "vocab": {
                "a": 0,
                "b": 1,
                "[gMASK]": 2,
                "<|user|>": 3,
                "<|observation|>": 4,
                "<|endoftext|>": 5,
                "<|assistant|>": 6,
                "<|system|>": 7,
            },
            "merges": ["a b"],
        },
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {"id": 2, "content": "[gMASK]", "special": True},
            {"id": 3, "content": "<|user|>", "special": True},
            {"id": 4, "content": "<|observation|>", "special": True},
            {"id": 5, "content": "<|endoftext|>", "special": True},
            {"id": 6, "content": "<|assistant|>", "special": True},
            {"id": 7, "content": "<|system|>", "special": True},
        ],
    }
    root.joinpath("tokenizer.json").write_text(json.dumps(tokenizer, separators=(",", ":")))
    root.joinpath("tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<|assistant|>",
                "pad_token": "<|endoftext|>",
                "mask_token": "[gMASK]",
                "add_bos_token": False,
            },
            separators=(",", ":"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indexer-types",
        default="full,shared,full",
        help=(
            "Comma-separated target-layer IndexShare roles. "
            f"Default has {DEFAULT_TARGET_LAYERS} layers: full,shared,full."
        ),
    )
    parser.add_argument("--index-topk-freq", type=int, default=2)
    parser.add_argument("--index-skip-topk-offset", type=int, default=1)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indexer_types = [item.strip() for item in args.indexer_types.split(",") if item.strip()]
    if len(indexer_types) < 1:
        raise SystemExit("--indexer-types must contain at least one role")
    invalid_roles = sorted(set(indexer_types) - {"full", "shared"})
    if invalid_roles:
        raise SystemExit(f"--indexer-types contains invalid roles: {invalid_roles}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_config(args.output_dir, indexer_types, args.index_topk_freq, args.index_skip_topk_offset)
    write_tokenizer(args.output_dir)
    write_safetensors(args.output_dir / "model.safetensors", build_tensors(indexer_types))
    print(f"wrote tiny GLM-DSA contract fixture to {args.output_dir}")


if __name__ == "__main__":
    main()
