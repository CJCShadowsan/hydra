#!/usr/bin/env python3
"""Compare reference SPD proposal traces with product spd-openai-smoke reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact-prompt reference SPD proposals with Skippy product proposals."
    )
    parser.add_argument("--reference-raw", required=True, type=Path)
    parser.add_argument("--product-report", required=True, type=Path)
    parser.add_argument(
        "--prompt-substring",
        default="",
        help="Select the reference/product prompt containing this text.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        help="Select a specific reference per-sample index instead of prompt text.",
    )
    parser.add_argument(
        "--product-prompt-index",
        type=int,
        default=0,
        help="Product prompt_index to compare. Defaults to 0 for single-prompt reports.",
    )
    parser.add_argument("--max-generated", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_reference_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_reference(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.reference_index is not None:
        for row in rows:
            if int(row.get("index", -1)) == args.reference_index:
                return row
        raise SystemExit(f"reference index not found: {args.reference_index}")
    if not args.prompt_substring:
        if len(rows) == 1:
            return rows[0]
        raise SystemExit("--prompt-substring or --reference-index is required for multi-row raw traces")
    matches = [
        row
        for row in rows
        if args.prompt_substring in json.dumps(row, ensure_ascii=False)
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one reference prompt match for {args.prompt_substring!r}, got {len(matches)}"
        )
    return matches[0]


def select_product_case(report: dict[str, Any], name: str, prompt_index: int) -> dict[str, Any]:
    matches = [
        case
        for case in report.get("cases", [])
        if case.get("name") == name and int(case.get("prompt_index", -1)) == prompt_index
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one product {name!r} case for prompt_index={prompt_index}, got {len(matches)}"
        )
    return matches[0]


def generated_tokens_from_case(case: dict[str, Any]) -> list[int]:
    return [int(event["predicted_token"]) for event in case.get("token_events", [])]


def reference_proposals(reference: dict[str, Any], max_generated: int) -> list[dict[str, Any]]:
    proposals = reference.get("proposal_trace", {}).get("draft_proposals")
    if not isinstance(proposals, list):
        raise SystemExit("reference row does not contain proposal_trace.draft_proposals")
    if max_generated <= 0:
        return proposals
    return [
        item
        for item in proposals
        if int(item.get("target_gen_idx", max_generated + 1)) < max_generated
    ]


def product_probes(case: dict[str, Any], max_generated: int) -> list[dict[str, Any]]:
    probes = [
        probe
        for probe in case.get("inline_probes", [])
        if probe.get("proposed_token") is not None
    ]
    if max_generated <= 0:
        return probes
    return [probe for probe in probes if int(probe.get("step", max_generated + 1)) < max_generated]


def compare_proposals(
    reference_items: list[dict[str, Any]],
    product_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    product_by_step = {int(item["step"]): item for item in product_items}
    rows = []
    for ref in reference_items:
        step = int(ref.get("target_gen_idx", -1))
        product = product_by_step.get(step)
        if product is None:
            rows.append(
                {
                    "target_gen_idx": step,
                    "reference": ref,
                    "product": None,
                    "proposal_token_match": False,
                    "target_token_match": False,
                    "accepted_match": False,
                }
            )
            continue
        proposal_token_match = int(ref["proposal_token"]) == int(product["proposed_token"])
        target_token_match = int(ref["target_token"]) == int(product["target_token"])
        accepted_match = bool(ref["accepted"]) == bool(product["accepted"])
        rows.append(
            {
                "target_gen_idx": step,
                "target_position": ref.get("target_position"),
                "reference_proposal_token": int(ref["proposal_token"]),
                "product_proposed_token": int(product["proposed_token"]),
                "reference_target_token": int(ref["target_token"]),
                "product_target_token": int(product["target_token"]),
                "reference_accepted": bool(ref["accepted"]),
                "product_accepted": bool(product["accepted"]),
                "product_phase": product.get("phase"),
                "product_tap_source": product.get("tap_source"),
                "product_row_positions": product.get("proposal_row_positions"),
                "product_row_i_stages": product.get("proposal_row_i_stages"),
                "product_row_evicted_prefix_position": product.get(
                    "proposal_row_evicted_prefix_position"
                ),
                "proposal_token_match": proposal_token_match,
                "target_token_match": target_token_match,
                "accepted_match": accepted_match,
            }
        )
    return rows


def count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def main() -> None:
    args = parse_args()
    reference_rows = load_reference_rows(args.reference_raw)
    reference = select_reference(args, reference_rows)
    product_report = load_json(args.product_report)
    baseline_case = select_product_case(product_report, "baseline", args.product_prompt_index)
    spd_case = select_product_case(product_report, "spd", args.product_prompt_index)

    max_generated = args.max_generated
    reference_generated = [int(token) for token in reference.get("generated_token_ids", [])]
    product_generated = generated_tokens_from_case(spd_case)
    if max_generated > 0:
        reference_generated = reference_generated[:max_generated]
        product_generated = product_generated[:max_generated]

    comparisons = compare_proposals(
        reference_proposals(reference, max_generated),
        product_probes(spd_case, max_generated),
    )
    proposal_count = len(comparisons)
    proposal_token_matches = count_true(comparisons, "proposal_token_match")
    target_token_matches = count_true(comparisons, "target_token_match")
    accepted_matches = count_true(comparisons, "accepted_match")

    result = {
        "mode": "reference-product-spd-comparison",
        "reference_raw": str(args.reference_raw),
        "product_report": str(args.product_report),
        "reference_index": reference.get("index"),
        "reference_dataset": reference.get("dataset"),
        "reference_question_id": reference.get("question_id"),
        "product_prompt_index": args.product_prompt_index,
        "product_prompt": spd_case.get("prompt"),
        "target_token_match": reference_generated == product_generated,
        "reference_generated_tokens": reference_generated,
        "product_generated_tokens": product_generated,
        "baseline_generated_tokens": generated_tokens_from_case(baseline_case)[: len(product_generated)],
        "proposal_count": proposal_count,
        "proposal_token_matches": proposal_token_matches,
        "target_token_matches": target_token_matches,
        "accepted_matches": accepted_matches,
        "proposal_token_parity": proposal_count > 0 and proposal_token_matches == proposal_count,
        "target_token_parity": proposal_count > 0 and target_token_matches == proposal_count,
        "accepted_decision_parity": proposal_count > 0 and accepted_matches == proposal_count,
        "comparisons": comparisons,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote comparison -> {args.output}")
    print(text)


if __name__ == "__main__":
    main()
