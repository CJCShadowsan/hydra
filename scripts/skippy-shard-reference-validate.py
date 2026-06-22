#!/usr/bin/env python3
"""Validate that a Shard proof reference matches the planned proof shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected top-level JSON object")
    return payload


def load_prompts(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row.get("id")
            prompt = row.get("prompt")
            if not isinstance(prompt_id, str) or not isinstance(prompt, str):
                raise SystemExit(f"{path}:{line_number}: expected string id and prompt")
            prompts.append({"id": prompt_id, "prompt": prompt})
    if not prompts:
        raise SystemExit(f"{path}: prompt set is empty")
    return prompts


def reference_prompts(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("reference results are empty or missing")
    prompts: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"reference results[{index}] is not an object")
        prompt_id = row.get("prompt_id")
        prompt = row.get("prompt")
        if not isinstance(prompt_id, str) or not isinstance(prompt, str):
            raise SystemExit(
                f"reference results[{index}] is missing string prompt_id/prompt"
            )
        prompts.append({"id": prompt_id, "prompt": prompt})
    return prompts


def ensure_exact_prompts(
    expected: list[dict[str, str]], actual: list[dict[str, str]]
) -> None:
    if len(expected) != len(actual):
        raise SystemExit(
            "reference prompt count mismatch: "
            f"expected {len(expected)} rows, got {len(actual)}"
        )
    for index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        if expected_row != actual_row:
            raise SystemExit(
                "reference prompt mismatch at row "
                f"{index}: expected {expected_row!r}, got {actual_row!r}"
            )


def metadata_value(payload: dict[str, Any], *path: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def reference_target_id(payload: dict[str, Any]) -> Any:
    return (
        metadata_value(payload, "target_identity", "target_id")
        or payload.get("target_id")
        or payload.get("target_model")
    )


def request_default(payload: dict[str, Any], key: str) -> Any:
    request_defaults = payload.get("request_defaults")
    if isinstance(request_defaults, dict) and key in request_defaults:
        return request_defaults[key]
    return payload.get(key)


def ensure_metadata(
    payload: dict[str, Any],
    expected_target_id: str,
    expected_max_tokens: int,
    require_metadata: bool,
) -> None:
    actual_target_id = reference_target_id(payload)
    actual_max_tokens = request_default(payload, "max_tokens")
    actual_temperature = request_default(payload, "temperature")

    if actual_target_id is None:
        if require_metadata:
            raise SystemExit(
                "reference JSON has no target_identity.target_id; recapture it "
                "or set the matching MESH_SHARD_*_REFERENCE_TARGET_ID when "
                "capturing and proving"
            )
    elif str(actual_target_id) != expected_target_id:
        raise SystemExit(
            "reference target id mismatch: "
            f"expected {expected_target_id!r}, got {actual_target_id!r}"
        )

    if actual_max_tokens is None:
        if require_metadata:
            raise SystemExit("reference JSON has no request_defaults.max_tokens")
    elif int(actual_max_tokens) != expected_max_tokens:
        raise SystemExit(
            "reference max_tokens mismatch: "
            f"expected {expected_max_tokens}, got {actual_max_tokens}"
        )

    if actual_temperature is None:
        if require_metadata:
            raise SystemExit("reference JSON has no request_defaults.temperature")
    elif float(actual_temperature) != 0.0:
        raise SystemExit(
            "reference temperature mismatch: expected greedy temperature 0.0, "
            f"got {actual_temperature}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--require-metadata", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.reference)
    ensure_exact_prompts(load_prompts(args.prompts), reference_prompts(payload))
    ensure_metadata(
        payload,
        expected_target_id=args.target_id,
        expected_max_tokens=args.max_tokens,
        require_metadata=args.require_metadata,
    )
    print(
        "reference validated: "
        f"{args.reference} target_id={args.target_id} max_tokens={args.max_tokens}"
    )


if __name__ == "__main__":
    main()
