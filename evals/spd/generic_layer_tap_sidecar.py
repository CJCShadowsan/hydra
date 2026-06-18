#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.0.0",
#   "datasets>=3.0.0",
#   "numpy",
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Train/export/evaluate a topology-independent SPD layer-tap sidecar.

This is the generic GLM 4.7 path. Unlike the donor SPD head, this sidecar does
not own per-stage projection tensors. It consumes a set of logical hidden-state
taps plus tap features, then predicts the next N draft tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from topology_plan import build_topology_plan


SERVING_FORMAT = "safetensors-spd-head-v1"
SOURCE_FORMAT = "generic-layer-tap-sidecar-v1"
HEAD_KIND = "generic-layer-tap-v1"


@dataclass
class TapExample:
    hidden: torch.Tensor
    features: torch.Tensor
    labels: list[int]
    topology_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a generic SPD layer-tap sidecar")
    parser.add_argument("--model-name", default="GLM-4.7-Flash-shape-only")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument("--work-dir", default="/tmp/skippy-spd-generic-layer-tap")
    parser.add_argument("--topology-plan", default="")
    parser.add_argument("--topology-plan-samples", type=int, default=32)
    parser.add_argument("--topology-min-stages", type=int, default=2)
    parser.add_argument("--topology-max-stages", type=int, default=6)
    parser.add_argument("--topology-seed", type=int, default=47)
    parser.add_argument("--topology-tap-dropout", type=float, default=0.25)
    parser.add_argument("--topology-num-hidden-layers", type=int, default=47)
    parser.add_argument(
        "--fixed-layer-taps",
        default="",
        help=(
            "Comma-separated logical hidden-state indices for a fixed layer-tap "
            "control run, for example 0,12,24,35,47."
        ),
    )
    parser.add_argument("--num-spec-layers", type=int, default=1)
    parser.add_argument("--draft-top-k", type=int, default=1)
    parser.add_argument("--draft-vocab-size", type=int, default=4096)
    parser.add_argument("--train-rows", type=int, default=128)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--positions-per-row", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--examples-cache-in", default="")
    parser.add_argument("--examples-cache-out", default="")
    parser.add_argument("--encoder", choices=("mean", "mean_mlp", "attention"), default="mean")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--eval-top-k", default="1,2,4,8,16,32")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--model-device-map",
        choices=("single", "auto", "cpu"),
        default="single",
        help="Device map for loading the target model during hidden-state extraction.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--export-dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--smoke-synthetic", action="store_true")
    parser.add_argument("--synthetic-hidden-size", type=int, default=64)
    parser.add_argument("--synthetic-vocab-size", type=int, default=512)
    parser.add_argument("--synthetic-train-examples", type=int, default=64)
    parser.add_argument("--synthetic-eval-examples", type=int, default=32)
    parser.add_argument("--manifest-base-model-path", default="")
    return parser.parse_args()


class GenericLayerTapSidecar(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        draft_vocab_size: int,
        num_spec_layers: int,
        *,
        encoder_kind: str = "mean",
        attention_heads: int = 4,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.encoder_kind = encoder_kind
        self.tap_proj = nn.Linear(hidden_size, hidden_size)
        self.depth_proj = nn.Linear(2, hidden_size)
        self.tap_norm = nn.LayerNorm(hidden_size)
        if encoder_kind == "attention":
            self.attn_query = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.tap_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=attention_heads,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_size)
        elif encoder_kind == "mean_mlp":
            mlp_hidden_size = max(1, int(hidden_size * mlp_ratio))
            self.pool_mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, hidden_size),
            )
        elif encoder_kind != "mean":
            raise ValueError(f"unsupported encoder_kind: {encoder_kind}")
        self.output_norm = nn.LayerNorm(hidden_size)
        self.draft_heads = nn.ModuleList(
            [nn.Linear(hidden_size, draft_vocab_size) for _ in range(num_spec_layers)]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        encoded = torch.tanh(self.tap_norm(self.tap_proj(hidden) + self.depth_proj(features)))
        if self.encoder_kind == "attention":
            query = self.attn_query.expand(encoded.shape[0], -1, -1)
            attended, _ = self.tap_attention(
                query,
                encoded,
                encoded,
                key_padding_mask=~mask,
                need_weights=False,
            )
            pooled = self.output_norm(self.attn_norm(attended[:, 0, :]))
        else:
            masked = encoded * mask.unsqueeze(-1).to(encoded.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
            pooled = masked.sum(dim=1) / denom
            if self.encoder_kind == "mean_mlp":
                pooled = pooled + self.pool_mlp(pooled)
            pooled = self.output_norm(pooled)
        return [head(pooled) for head in self.draft_heads]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    artifact_dir = Path(args.work_dir).expanduser().resolve() / "artifacts" / timestamp()
    train_dir = artifact_dir / "train"
    eval_dir = artifact_dir / "eval" / "summary"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    plan = load_or_build_topology_plan(args)
    cache_in = Path(args.examples_cache_in).expanduser() if args.examples_cache_in else None
    if cache_in is not None:
        model_meta, draft_token_ids, train_examples, eval_examples, plan = load_examples_cache(cache_in)
    elif args.smoke_synthetic:
        model_meta, draft_token_ids, train_examples, eval_examples = synthetic_examples(args, plan)
    else:
        model_meta, draft_token_ids, train_examples, eval_examples = real_glm_examples(args, plan)
    if args.examples_cache_out and cache_in is None:
        write_examples_cache(
            Path(args.examples_cache_out).expanduser(),
            model_meta,
            draft_token_ids,
            train_examples,
            eval_examples,
            plan,
            args,
        )
    train_examples = align_example_width(train_examples, int(args.num_spec_layers), "train")
    eval_examples = align_example_width(eval_examples, int(args.num_spec_layers), "eval")

    device = resolve_device(args.device)
    sidecar = GenericLayerTapSidecar(
        hidden_size=int(model_meta["hidden_size"]),
        draft_vocab_size=len(draft_token_ids),
        num_spec_layers=int(args.num_spec_layers),
        encoder_kind=args.encoder,
        attention_heads=int(args.attention_heads),
        mlp_ratio=float(args.mlp_ratio),
    ).to(device)
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=float(args.learning_rate))
    label_map = {token_id: index for index, token_id in enumerate(draft_token_ids)}

    train_started = time.perf_counter()
    train_loss = train_sidecar(
        sidecar,
        optimizer,
        train_examples,
        label_map,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        device=device,
    )
    train_elapsed = time.perf_counter() - train_started

    eval_summary = evaluate_sidecar(
        sidecar,
        eval_examples,
        label_map,
        batch_size=int(args.batch_size),
        device=device,
        top_k_values=parse_top_k_values(args.eval_top_k),
    )
    eval_summary["train_loss"] = train_loss
    eval_summary["train_wall_seconds"] = train_elapsed
    eval_summary["total_wall_seconds"] = time.perf_counter() - started
    eval_summary["encoder"] = {
        "kind": args.encoder,
        "attention_heads": int(args.attention_heads) if args.encoder == "attention" else None,
        "mlp_ratio": float(args.mlp_ratio) if args.encoder == "mean_mlp" else None,
    }
    eval_summary["topology_policy"] = plan["policy"]
    eval_summary["topology_eval"] = summarize_examples_by_topology(eval_examples)
    eval_summary["model"] = model_meta

    checkpoint_path = train_dir / "generic-layer-tap-sidecar.pt"
    checkpoint = {
        "format": SOURCE_FORMAT,
        "version": 1,
        "config": {
            "head_kind": HEAD_KIND,
            "hidden_size": int(model_meta["hidden_size"]),
            "vocab_size": int(model_meta["vocab_size"]),
            "draft_vocab_size": len(draft_token_ids),
            "num_spec_layers": int(args.num_spec_layers),
            "encoder_kind": args.encoder,
            "attention_heads": int(args.attention_heads),
            "mlp_ratio": float(args.mlp_ratio) if args.encoder == "mean_mlp" else None,
            "max_taps": max_taps(plan),
            "tap_feature_size": 2,
            "draft_token_ids": draft_token_ids,
            "topology_plan": plan,
        },
        "state_dict": {name: value.detach().cpu() for name, value in sidecar.state_dict().items()},
        "eval_summary": eval_summary,
    }
    torch.save(checkpoint, checkpoint_path)

    serving_path = train_dir / "spd-head.safetensors"
    save_serving_safetensors(sidecar, serving_path, args.export_dtype)
    manifest_path = train_dir / "skippy-spd-head.json"
    write_manifest(
        args=args,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        serving_path=serving_path,
        model_meta=model_meta,
        draft_token_ids=draft_token_ids,
        plan=plan,
    )
    summary_path = eval_dir / "generic_layer_tap_eval_summary.json"
    summary_path.write_text(json.dumps(eval_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "manifest": str(manifest_path),
                "serving_checkpoint": str(serving_path),
                "summary": str(summary_path),
                "acceptance_rate": eval_summary["acceptance_rate"],
                "equivalent_accept_length": eval_summary["equivalent_accept_length"],
                "proposal_latency_ms": eval_summary["proposal_latency_ms"],
                "eval_wall_seconds": eval_summary["eval_wall_seconds"],
                "top_k": eval_summary["top_k_diagnostics"]["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_or_build_topology_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.topology_plan:
        return json.loads(Path(args.topology_plan).expanduser().read_text(encoding="utf-8"))
    return build_topology_plan(args)


def write_examples_cache(
    path: Path,
    model_meta: dict[str, Any],
    draft_token_ids: list[int],
    train_examples: list[TapExample],
    eval_examples: list[TapExample],
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "skippy-spd-layer-tap-examples/v1",
        "created_at": timestamp(),
        "model": model_meta,
        "draft_token_ids": draft_token_ids,
        "topology_plan": plan,
        "metadata": {
            "num_spec_layers": int(args.num_spec_layers),
            "train_rows": int(args.train_rows),
            "eval_rows": int(args.eval_rows),
            "positions_per_row": int(args.positions_per_row),
            "max_length": int(args.max_length),
        },
        "train_examples": [example_to_record(example) for example in train_examples],
        "eval_examples": [example_to_record(example) for example in eval_examples],
    }
    torch.save(payload, path)
    print(f"wrote layer-tap examples cache -> {path}", flush=True)


def load_examples_cache(
    path: Path,
) -> tuple[dict[str, Any], list[int], list[TapExample], list[TapExample], dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != "skippy-spd-layer-tap-examples/v1":
        raise RuntimeError(f"unsupported examples cache schema in {path}")
    train_examples = [record_to_example(record) for record in payload["train_examples"]]
    eval_examples = [record_to_example(record) for record in payload["eval_examples"]]
    print(
        f"loaded layer-tap examples cache -> {path} "
        f"(train={len(train_examples)}, eval={len(eval_examples)})",
        flush=True,
    )
    return (
        dict(payload["model"]),
        [int(token_id) for token_id in payload["draft_token_ids"]],
        train_examples,
        eval_examples,
        dict(payload["topology_plan"]),
    )


def example_to_record(example: TapExample) -> dict[str, Any]:
    return {
        "hidden": example.hidden.detach().cpu().to(torch.float16),
        "features": example.features.detach().cpu().to(torch.float16),
        "labels": [int(label) for label in example.labels],
        "topology_key": example.topology_key,
    }


def record_to_example(record: dict[str, Any]) -> TapExample:
    return TapExample(
        hidden=record["hidden"].float(),
        features=record["features"].float(),
        labels=[int(label) for label in record["labels"]],
        topology_key=str(record["topology_key"]),
    )


def align_example_width(
    examples: list[TapExample],
    num_spec_layers: int,
    label: str,
) -> list[TapExample]:
    widths = [len(example.labels) for example in examples]
    too_narrow = [width for width in widths if width < num_spec_layers]
    if too_narrow:
        raise RuntimeError(
            f"{label} examples cannot satisfy --num-spec-layers={num_spec_layers}; "
            f"found label widths including {too_narrow[:5]}"
        )
    if any(width > num_spec_layers for width in widths):
        print(
            f"truncating {label} examples to --num-spec-layers={num_spec_layers}",
            flush=True,
        )
        return [
            TapExample(
                hidden=example.hidden,
                features=example.features,
                labels=example.labels[:num_spec_layers],
                topology_key=example.topology_key,
            )
            for example in examples
        ]
    return examples


def synthetic_examples(
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], list[int], list[TapExample], list[TapExample]]:
    rng = random.Random(int(args.topology_seed))
    hidden_size = int(args.synthetic_hidden_size)
    vocab_size = int(args.synthetic_vocab_size)
    draft_token_ids = list(range(min(int(args.draft_vocab_size), vocab_size)))
    model_meta = {
        "model_name": args.model_name,
        "model_type": "synthetic",
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_hidden_layers": int(plan["model"]["num_hidden_layers"]),
    }
    train = [
        synthetic_example(hidden_size, draft_token_ids, int(args.num_spec_layers), plan, rng)
        for _ in range(int(args.synthetic_train_examples))
    ]
    evals = [
        synthetic_example(hidden_size, draft_token_ids, int(args.num_spec_layers), plan, rng)
        for _ in range(int(args.synthetic_eval_examples))
    ]
    return model_meta, draft_token_ids, train, evals


def synthetic_example(
    hidden_size: int,
    draft_token_ids: list[int],
    num_spec_layers: int,
    plan: dict[str, Any],
    rng: random.Random,
) -> TapExample:
    indices = sample_tap_indices(plan, rng)
    hidden = torch.randn(len(indices), hidden_size)
    features = tap_features(indices, int(plan["model"]["num_hidden_layers"]))
    labels = [rng.choice(draft_token_ids) for _ in range(num_spec_layers)]
    return TapExample(
        hidden=hidden,
        features=features,
        labels=labels,
        topology_key=topology_key(indices),
    )


def real_glm_examples(
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], list[int], list[TapExample], list[TapExample]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(args.device)
    dtype = torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    rows = load_training_rows(args.dataset, args.dataset_split, int(args.train_rows + args.eval_rows))
    texts = [row_text_for_vocab(tokenizer, row.get("messages") or []) for row in rows]
    draft_token_ids = build_draft_vocab(tokenizer, texts[: int(args.train_rows)], int(args.draft_vocab_size))
    model = load_target_model(args, dtype, device)
    model.eval()
    model_meta = {
        "model_name": args.model_name,
        "model_type": getattr(model.config, "model_type", None),
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
    }
    train_texts = texts[: int(args.train_rows)]
    eval_texts = texts[int(args.train_rows) : int(args.train_rows + args.eval_rows)]
    rng = random.Random(int(args.topology_seed))
    model_device = next(model.parameters()).device
    train = collect_examples_from_texts(args, tokenizer, model, train_texts, plan, rng, model_device)
    evals = collect_examples_from_texts(args, tokenizer, model, eval_texts, plan, rng, model_device)
    return model_meta, draft_token_ids, train, evals


def load_target_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
        "local_files_only": bool(args.local_files_only),
    }
    if args.model_device_map == "auto":
        kwargs["device_map"] = "auto"
    elif args.model_device_map == "cpu":
        kwargs["device_map"] = {"": "cpu"}
    elif device.type != "cpu":
        kwargs["device_map"] = {"": device.type}
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    if args.model_device_map == "single" and device.type == "cpu":
        model = model.to(device)
    return model


def collect_examples_from_texts(
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    texts: list[str],
    plan: dict[str, Any],
    rng: random.Random,
    device: torch.device,
) -> list[TapExample]:
    examples: list[TapExample] = []
    num_layers = int(plan["model"]["num_hidden_layers"])
    batch_size = max(1, int(args.extract_batch_size))
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=int(args.max_length),
            add_special_tokens=True,
            padding=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = [state.detach().cpu().float() for state in output.hidden_states]
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        for row in range(input_ids_cpu.shape[0]):
            examples.extend(
                collect_examples_from_encoded_row(
                    args,
                    hidden_states,
                    input_ids_cpu,
                    attention_mask_cpu,
                    row,
                    plan,
                    rng,
                    num_layers,
                )
            )
        print(
            f"collected {len(examples)} examples from {min(start + batch_size, len(texts))}/{len(texts)} texts",
            flush=True,
        )
    if not examples:
        raise RuntimeError("no real GLM examples collected")
    return examples


def collect_examples_from_encoded_row(
    args: argparse.Namespace,
    hidden_states: list[torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    row: int,
    plan: dict[str, Any],
    rng: random.Random,
    num_layers: int,
) -> list[TapExample]:
    real_positions = attention_mask[row].nonzero(as_tuple=False).flatten().tolist()
    token_ids = [int(input_ids[row, col]) for col in real_positions]
    if len(token_ids) <= int(args.num_spec_layers) + 1:
        return []
    max_pos = len(token_ids) - int(args.num_spec_layers) - 1
    positions = sorted(rng.sample(range(max_pos), min(int(args.positions_per_row), max_pos)))
    examples: list[TapExample] = []
    for pos in positions:
        indices = sample_tap_indices(plan, rng)
        tensor_pos = int(real_positions[pos])
        hidden = torch.stack([hidden_states[index][row, tensor_pos] for index in indices], dim=0)
        features = tap_features(indices, num_layers)
        labels = [token_ids[pos + offset] for offset in range(1, int(args.num_spec_layers) + 1)]
        examples.append(
            TapExample(
                hidden=hidden,
                features=features,
                labels=labels,
                topology_key=topology_key(indices),
            )
        )
    return examples


def sample_tap_indices(plan: dict[str, Any], rng: random.Random) -> list[int]:
    layout = rng.choice(plan["layouts"])
    row = list(layout["shallow_hidden_layer_indices"][0])
    dropout = float(layout.get("tap_dropout", {}).get("probability", 0.0))
    required = set(int(index) for index in layout.get("tap_dropout", {}).get("required_indices", []))
    kept = [index for index in row if index in required or rng.random() >= dropout]
    if not kept:
        kept = [row[0], row[-1]]
    return sorted(set(int(index) for index in kept))


def topology_key(indices: list[int]) -> str:
    return ",".join(str(index) for index in indices)


def tap_features(indices: list[int], num_layers: int) -> torch.Tensor:
    rows = []
    for index in indices:
        rows.append([float(index) / float(max(1, num_layers)), 1.0 if index == 0 else 0.0])
    return torch.tensor(rows, dtype=torch.float32)


def train_sidecar(
    sidecar: GenericLayerTapSidecar,
    optimizer: torch.optim.Optimizer,
    examples: list[TapExample],
    label_map: dict[int, int],
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> float:
    sidecar.train()
    losses: list[float] = []
    rng = random.Random(17)
    for _ in range(max(1, epochs)):
        rng.shuffle(examples)
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            hidden, features, mask, labels = collate_batch(batch, label_map, device)
            logits = sidecar(hidden, features, mask)
            loss = batch_loss(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def evaluate_sidecar(
    sidecar: GenericLayerTapSidecar,
    examples: list[TapExample],
    label_map: dict[int, int],
    *,
    batch_size: int,
    device: torch.device,
    top_k_values: list[int],
) -> dict[str, Any]:
    sidecar.eval()
    accepted = 0
    proposal_slots = 0
    covered_labels = 0
    total_labels = 0
    proposal_time = 0.0
    top_k_stats = new_top_k_stats(top_k_values, len(examples[0].labels))
    by_topology = {
        example.topology_key: {"examples": 0, "accepted": 0, "slots": 0}
        for example in examples
    }
    eval_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            hidden, features, mask, labels = collate_batch(batch, label_map, device)
            proposal_started = time.perf_counter()
            logits = sidecar(hidden, features, mask)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            proposal_time += time.perf_counter() - proposal_started
            update_top_k_stats(logits, labels, top_k_stats)
            predictions = torch.stack([head.argmax(dim=-1) for head in logits], dim=1)
            for row_idx in range(predictions.shape[0]):
                topology = batch[row_idx].topology_key
                by_topology[topology]["examples"] += 1
                for spec_idx in range(predictions.shape[1]):
                    label = int(labels[row_idx, spec_idx].detach().cpu())
                    total_labels += 1
                    if label < 0:
                        break
                    covered_labels += 1
                    proposal_slots += 1
                    by_topology[topology]["slots"] += 1
                    if int(predictions[row_idx, spec_idx].detach().cpu()) != label:
                        break
                    accepted += 1
                    by_topology[topology]["accepted"] += 1
    eval_wall = time.perf_counter() - eval_started
    denominator = max(1, proposal_slots)
    topology_rows = []
    for key, row in sorted(by_topology.items()):
        slots = max(1, row["slots"])
        topology_rows.append(
            {
                "tap_indices": key,
                "examples": row["examples"],
                "accepted_draft_tokens": row["accepted"],
                "proposal_slots": row["slots"],
                "acceptance_rate": row["accepted"] / slots,
            }
        )
    return {
        "examples": len(examples),
        "accepted_draft_tokens": accepted,
        "proposal_slots": proposal_slots,
        "acceptance_rate": accepted / denominator,
        "equivalent_accept_length": accepted / max(1, len(examples)),
        "draft_vocab_label_coverage": covered_labels / max(1, total_labels),
        "covered_labels": covered_labels,
        "total_labels": total_labels,
        "proposal_latency_ms": (proposal_time / max(1, len(examples))) * 1000.0,
        "proposal_wall_seconds": proposal_time,
        "eval_wall_seconds": eval_wall,
        "top_k_diagnostics": summarize_top_k_stats(top_k_stats),
        "by_topology": topology_rows,
    }


def parse_top_k_values(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--eval-top-k values must be positive")
        values.append(value)
    return sorted(set(values)) or [1]


def new_top_k_stats(top_k_values: list[int], num_spec_layers: int) -> dict[str, Any]:
    return {
        "k_values": list(top_k_values),
        "overall": {k: {"hits": 0, "labels": 0} for k in top_k_values},
        "by_spec_layer": [
            {k: {"hits": 0, "labels": 0} for k in top_k_values}
            for _ in range(num_spec_layers)
        ],
    }


def update_top_k_stats(
    logits: list[torch.Tensor],
    labels: torch.Tensor,
    stats: dict[str, Any],
) -> None:
    for spec_idx, head_logits in enumerate(logits):
        target = labels[:, spec_idx]
        valid = target.ge(0)
        if not valid.any():
            continue
        valid_targets = target[valid].detach().cpu()
        valid_logits = head_logits[valid].detach()
        max_k = min(max(stats["k_values"]), valid_logits.shape[-1])
        top_indices = valid_logits.topk(k=max_k, dim=-1).indices.detach().cpu()
        for k in stats["k_values"]:
            clipped_k = min(k, top_indices.shape[-1])
            hits = int(top_indices[:, :clipped_k].eq(valid_targets.unsqueeze(1)).any(dim=1).sum())
            labels_count = int(valid_targets.numel())
            stats["overall"][k]["hits"] += hits
            stats["overall"][k]["labels"] += labels_count
            stats["by_spec_layer"][spec_idx][k]["hits"] += hits
            stats["by_spec_layer"][spec_idx][k]["labels"] += labels_count


def summarize_top_k_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": top_k_rows(stats["overall"], stats["k_values"]),
        "by_spec_layer": [
            {
                "spec_layer": spec_idx,
                "rows": top_k_rows(layer_stats, stats["k_values"]),
            }
            for spec_idx, layer_stats in enumerate(stats["by_spec_layer"])
        ],
    }


def top_k_rows(rows: dict[int, dict[str, int]], k_values: list[int]) -> list[dict[str, Any]]:
    result = []
    for k in k_values:
        labels = rows[k]["labels"]
        hits = rows[k]["hits"]
        result.append(
            {
                "k": k,
                "hits": hits,
                "labels": labels,
                "hit_rate": hits / max(1, labels),
            }
        )
    return result


def summarize_examples_by_topology(examples: list[TapExample]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.topology_key] = counts.get(example.topology_key, 0) + 1
    return [{"tap_indices": key, "examples": value} for key, value in sorted(counts.items())]


def collate_batch(
    batch: list[TapExample],
    label_map: dict[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_taps = max(example.hidden.shape[0] for example in batch)
    hidden_size = batch[0].hidden.shape[1]
    num_spec_layers = len(batch[0].labels)
    hidden = torch.zeros(len(batch), max_taps, hidden_size, dtype=torch.float32)
    features = torch.zeros(len(batch), max_taps, 2, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_taps, dtype=torch.bool)
    labels = torch.full((len(batch), num_spec_layers), -100, dtype=torch.long)
    for row, example in enumerate(batch):
        taps = example.hidden.shape[0]
        hidden[row, :taps] = example.hidden
        features[row, :taps] = example.features
        mask[row, :taps] = True
        for spec_idx, token_id in enumerate(example.labels):
            labels[row, spec_idx] = label_map.get(int(token_id), -100)
    return hidden.to(device), features.to(device), mask.to(device), labels.to(device)


def batch_loss(logits: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for spec_idx, head_logits in enumerate(logits):
        target = labels[:, spec_idx]
        valid = target.ge(0)
        if valid.any():
            losses.append(nn.functional.cross_entropy(head_logits[valid], target[valid]))
    if not losses:
        return logits[0].sum() * 0.0
    return torch.stack(losses).mean()


def save_serving_safetensors(
    sidecar: GenericLayerTapSidecar,
    output_path: Path,
    dtype_name: str,
) -> None:
    from safetensors.torch import save_file

    dtype = torch_dtype(dtype_name)
    tensors = {
        name: tensor.detach().cpu().to(dtype)
        for name, tensor in sorted(sidecar.state_dict().items())
    }
    save_file(
        tensors,
        output_path,
        metadata={
            "format": SERVING_FORMAT,
            "source_format": SOURCE_FORMAT,
            "head_kind": HEAD_KIND,
            "dtype": safetensors_dtype_label(dtype),
            "tensor_count": str(len(tensors)),
        },
    )


def write_manifest(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    checkpoint_path: Path,
    serving_path: Path,
    model_meta: dict[str, Any],
    draft_token_ids: list[int],
    plan: dict[str, Any],
) -> None:
    checkpoint_rel = checkpoint_path.name
    serving_rel = serving_path.name
    max_stages = int(plan["policy"]["max_stages"])
    manifest = {
        "schema": "skippy-spd-head/v1",
        "checkpoint": {
            "path": checkpoint_rel,
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "serving_checkpoint": {
            "path": serving_rel,
            "sha256": file_sha256(serving_path),
            "bytes": serving_path.stat().st_size,
            "format": SERVING_FORMAT,
            "tensor_count": len(dict(torch.load(checkpoint_path, map_location="cpu")["state_dict"])),
            "dtype": safetensors_dtype_label(torch_dtype(args.export_dtype)),
        },
        "source": {
            "format": SOURCE_FORMAT,
            "reference_repo": None,
            "base_model_path": args.manifest_base_model_path.strip() or args.model_name,
            "model_type": model_meta.get("model_type"),
            "checkpoint_version": 1,
        },
        "topology": {
            "hidden_size": int(model_meta["hidden_size"]),
            "vocab_size": int(model_meta["vocab_size"]),
            "draft_vocab_size": len(draft_token_ids),
            "head_kind": HEAD_KIND,
            "encoder_kind": args.encoder,
            "attention_heads": int(args.attention_heads) if args.encoder == "attention" else None,
            "mlp_ratio": float(args.mlp_ratio) if args.encoder == "mean_mlp" else None,
            "num_stages": max_stages,
            "stage_layer_boundaries": None,
            "num_spec_layers": int(args.num_spec_layers),
            "max_taps": max_taps(plan),
            "tap_feature_size": 2,
            "trained_with_use_deepest": False,
            "shallow_hidden_layer_indices": representative_tap_rows(plan),
            "spec_init_from_base_layers": None,
            "draft_token_ids": draft_token_ids,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def representative_tap_rows(plan: dict[str, Any]) -> list[list[int]]:
    rows: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for layout in plan["layouts"]:
        row = tuple(int(index) for index in layout["shallow_hidden_layer_indices"][0])
        if row in seen:
            continue
        seen.add(row)
        rows.append(list(row))
        if len(rows) >= 16:
            break
    return rows


def max_taps(plan: dict[str, Any]) -> int:
    return max(
        len(layout["shallow_hidden_layer_indices"][0])
        for layout in plan["layouts"]
    )


def load_training_rows(dataset_name: str, split: str, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=f"{split}[:{max(1, limit)}]")
    rows = []
    for row in ds:
        messages = row.get("messages") or row.get("conversations")
        if messages:
            rows.append({"messages": normalize_messages(messages)})
    return rows


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value")
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role is not None and content is not None:
            normalized.append({"role": str(role), "content": str(content)})
    return normalized


def row_text_for_vocab(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if isinstance(rendered, str):
            return rendered
    except Exception:
        pass
    return "\n".join(message.get("content", "") for message in messages)


def build_draft_vocab(tokenizer: Any, texts: list[str], draft_vocab_size: int) -> list[int]:
    from collections import Counter

    counts: Counter[int] = Counter()
    for text in texts:
        counts.update(int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False))
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ):
        if token_id is not None:
            counts[int(token_id)] += 1
    if not counts:
        raise RuntimeError("could not build draft vocab")
    ids = [
        token_id
        for token_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            : max(1, draft_vocab_size)
        ]
    ]
    return sorted(set(ids))


def resolve_device(value: str) -> torch.device:
    if value == "cuda" or (value == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if value == "mps" or (value == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def torch_dtype(value: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def safetensors_dtype_label(dtype: torch.dtype) -> str:
    return {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
    }[dtype]


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


if __name__ == "__main__":
    main()
