#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.18.0",
#   "transformers>=4.40.0",
# ]
# ///
"""Build tokenized HF prompt shards for SPD product-corpus capture.

The output JSONL files are accepted by:

    skippy-bench spd-live-tap-parity --prompt-token-file ...

Rows are rendered with the target tokenizer chat template and
``enable_thinking=False`` so native Q4 tap/logit capture can train and evaluate
the same prompt surface as the Skippy OpenAI smokes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize train/held-out prompts from a Hugging Face dataset."
    )
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-split", default="train_sft")
    parser.add_argument(
        "--dataset-config",
        default="",
        help=(
            "Optional Hugging Face dataset config name. For comma-separated "
            "--dataset values, provide one config for all datasets or the same "
            "number of comma-separated configs; use empty entries for datasets "
            "without an explicit config."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--train-prompts", type=int, default=512)
    parser.add_argument("--heldout-prompts", type=int, default=128)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=480)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--conversation-field",
        default="messages",
        help="Dataset field containing chat messages.",
    )
    parser.add_argument(
        "--user-turn-limit",
        type=int,
        default=1,
        help="Maximum user turns to keep from each row; use 0 for all user turns.",
    )
    parser.add_argument(
        "--exclude-prompt-token-file",
        action="append",
        default=[],
        type=Path,
        help=(
            "JSONL prompt-token file whose dataset/source_index/question_id rows "
            "must be excluded from both train and held-out outputs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: list[dict[str, Any]] = []
    for spec in dataset_specs(args):
        load_kwargs = {"split": spec.split}
        if spec.config:
            load_kwargs["name"] = spec.config
        source = load_dataset(spec.name, **load_kwargs)
        if args.max_source_rows > 0:
            source = source.select(range(min(args.max_source_rows, len(source))))
        rows.extend(build_rows(args, source, tokenizer, spec))
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    excluded_keys = load_excluded_prompt_keys(args.exclude_prompt_token_file)
    usable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        if prompt_key(row) in excluded_keys:
            excluded.append(row)
            continue
        if args.max_prompt_tokens > 0 and row["prompt_token_count"] > args.max_prompt_tokens:
            skipped.append(row)
            continue
        usable.append(row)

    train_rows = usable[: args.train_prompts]
    heldout_rows = usable[args.train_prompts : args.train_prompts + args.heldout_prompts]
    write_outputs(args, train_rows, heldout_rows, skipped, excluded)


def validate_args(args: argparse.Namespace) -> None:
    if args.train_prompts < 0 or args.heldout_prompts < 0:
        raise SystemExit("--train-prompts and --heldout-prompts must be non-negative")
    if args.train_prompts == 0 and args.heldout_prompts == 0:
        raise SystemExit("nothing to write: both prompt counts are zero")
    if args.max_source_rows < 0:
        raise SystemExit("--max-source-rows must be non-negative")
    if args.user_turn_limit < 0:
        raise SystemExit("--user-turn-limit must be non-negative")
    dataset_count = len(split_csv_arg(args.dataset))
    split_count = len(split_csv_arg(args.dataset_split))
    config_count = len(split_csv_arg(args.dataset_config, keep_empty=True))
    if dataset_count == 0 or split_count == 0:
        raise SystemExit("--dataset and --dataset-split must not be empty")
    if split_count not in {1, dataset_count}:
        raise SystemExit(
            "--dataset-split must contain one value or the same number of values as --dataset"
        )
    if config_count not in {0, 1, dataset_count}:
        raise SystemExit(
            "--dataset-config must be empty, contain one value, or contain the same "
            "number of values as --dataset"
        )


def split_csv_arg(value: str, *, keep_empty: bool = False) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    if keep_empty:
        return parts if any(parts) else []
    return [part for part in parts if part]


class DatasetSpec:
    def __init__(self, name: str, split: str, config: str = "") -> None:
        self.name = name
        self.split = split
        self.config = config


def dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    datasets = split_csv_arg(args.dataset)
    splits = split_csv_arg(args.dataset_split)
    configs = split_csv_arg(args.dataset_config, keep_empty=True)
    if len(splits) == 1:
        splits *= len(datasets)
    if not configs:
        configs = [""] * len(datasets)
    elif len(configs) == 1:
        configs *= len(datasets)
    return [
        DatasetSpec(name=dataset_name, split=dataset_split, config=dataset_config)
        for dataset_name, dataset_split, dataset_config in zip(
            datasets, splits, configs, strict=True
        )
    ]


def build_rows(
    args: argparse.Namespace,
    source: Any,
    tokenizer: Any,
    spec: DatasetSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, obj in enumerate(source):
        messages = extract_user_messages(
            obj,
            conversation_field=args.conversation_field,
            user_turn_limit=args.user_turn_limit,
        )
        if not messages:
            continue
        tokens = render_prompt_tokens(tokenizer, messages)
        rows.append(
            {
                "dataset": spec.name,
                "dataset_split": spec.split,
                "dataset_config": spec.config,
                "source_index": index,
                "question_id": obj.get("id") or obj.get("question_id"),
                "category": obj.get("category"),
                "prompt": first_user_prompt(messages),
                "messages": messages,
                "prompt_token_ids": tokens,
                "prompt_token_count": len(tokens),
            }
        )
    return rows


def extract_user_messages(
    obj: dict[str, Any], *, conversation_field: str, user_turn_limit: int
) -> list[dict[str, str]]:
    raw = obj.get(conversation_field)
    if raw is None:
        raw = obj.get("conversations")
    if raw is None:
        prompt = obj.get("prompt") or obj.get("question") or obj.get("text")
        return [{"role": "user", "content": str(prompt)}] if prompt else []
    if not isinstance(raw, list):
        return [{"role": "user", "content": str(raw)}] if str(raw).strip() else []

    messages: list[dict[str, str]] = []
    user_turns = 0
    for item in raw:
        role, content = message_role_and_content(item)
        if role != "user" or not content.strip():
            continue
        messages.append({"role": "user", "content": content})
        user_turns += 1
        if user_turn_limit > 0 and user_turns >= user_turn_limit:
            break
    return messages


def message_role_and_content(item: Any) -> tuple[str, str]:
    if isinstance(item, dict):
        role = item.get("role") or item.get("from") or item.get("speaker")
        content = item.get("content") or item.get("value") or item.get("text") or ""
        return normalize_role(role), str(content)
    return "user", str(item)


def normalize_role(role: Any) -> str:
    value = str(role or "user").lower()
    if value in {"human", "user"}:
        return "user"
    if value in {"assistant", "gpt", "bot"}:
        return "assistant"
    return value


def first_user_prompt(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    return ""


def render_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(ids, dict):
        ids = ids.get("input_ids")
    elif hasattr(ids, "get"):
        ids = ids.get("input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        raise TypeError(f"unexpected chat-template token output: {type(ids).__name__}")
    return [int(token) for token in ids]


def load_excluded_prompt_keys(paths: list[Path]) -> set[tuple[str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}:{line_number}: exclude rows must be JSON objects"
                    )
                keys.add(prompt_key(obj))
    return keys


def prompt_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset")),
        str(row.get("dataset_split")),
        str(row.get("dataset_config", "")),
        str(row.get("source_index")),
        str(row.get("question_id")),
    )


def write_outputs(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tokens = args.out_dir / "train-prompt-token-ids.jsonl"
    heldout_tokens = args.out_dir / "heldout-prompt-token-ids.jsonl"
    train_prompts = args.out_dir / "train-prompts.jsonl"
    heldout_prompts = args.out_dir / "heldout-prompts.jsonl"
    summary = args.out_dir / "summary.json"

    write_jsonl(train_tokens, [token_row(row) for row in train_rows])
    write_jsonl(heldout_tokens, [token_row(row) for row in heldout_rows])
    write_jsonl(train_prompts, [prompt_row(row) for row in train_rows])
    write_jsonl(heldout_prompts, [prompt_row(row) for row in heldout_rows])
    summary_obj = {
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_specs": [
            {
                "dataset": spec.name,
                "dataset_split": spec.split,
                "dataset_config": spec.config,
            }
            for spec in dataset_specs(args)
        ],
        "model_name": args.model_name,
        "train_prompt_token_file": str(train_tokens),
        "heldout_prompt_token_file": str(heldout_tokens),
        "train_prompt_count": len(train_rows),
        "heldout_prompt_count": len(heldout_rows),
        "skipped_prompt_count": len(skipped),
        "excluded_prompt_count": len(excluded),
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_source_rows": args.max_source_rows,
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "conversation_field": args.conversation_field,
        "user_turn_limit": args.user_turn_limit,
        "exclude_prompt_token_files": [str(path) for path in args.exclude_prompt_token_file],
        "train_token_stats": token_stats(train_rows),
        "heldout_token_stats": token_stats(heldout_rows),
        "skipped_token_stats": token_stats(skipped),
        "excluded_token_stats": token_stats(excluded),
    }
    summary.write_text(json.dumps(summary_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_obj, indent=2, ensure_ascii=False))


def token_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_token_ids": row["prompt_token_ids"],
        "dataset": row["dataset"],
        "dataset_split": row["dataset_split"],
        "dataset_config": row.get("dataset_config", ""),
        "source_index": row["source_index"],
        "question_id": row["question_id"],
        "category": row["category"],
        "prompt_token_count": row["prompt_token_count"],
    }


def prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "messages": row["messages"],
        "dataset": row["dataset"],
        "dataset_split": row["dataset_split"],
        "dataset_config": row.get("dataset_config", ""),
        "source_index": row["source_index"],
        "question_id": row["question_id"],
        "category": row["category"],
        "prompt_token_count": row["prompt_token_count"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = sorted(int(row["prompt_token_count"]) for row in rows)
    if not counts:
        return {"count": 0}
    return {
        "count": len(counts),
        "min": counts[0],
        "p50": median(counts),
        "max": counts[-1],
        "mean": sum(counts) / len(counts),
    }


if __name__ == "__main__":
    main()
