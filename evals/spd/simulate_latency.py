#!/usr/bin/env python3
"""Simulate split-stage latency from real SPD eval traces.

This is not a serving benchmark. It consumes the per-sample JSONL emitted by the
reference SPD eval and applies an explicit latency model to its real
``new_tokens`` and ``decode_loop_steps`` counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceTotals:
    samples: int
    stages: int
    tokens: int
    decode_steps: int
    accepted_flags: int
    acceptance_flags: int

    @property
    def pipeline_cycles(self) -> float:
        return self.decode_steps / max(self.stages, 1)

    @property
    def aggregate_acceptance_rate(self) -> float:
        if self.decode_steps == 0:
            return 0.0
        return self.tokens / self.decode_steps

    @property
    def equivalent_accept_length(self) -> float:
        return self.stages * self.aggregate_acceptance_rate

    @property
    def paper_theoretical_gain_pct(self) -> float:
        if self.pipeline_cycles <= 0.0:
            return 0.0
        return ((self.tokens / self.pipeline_cycles) - 1.0) * 100.0


@dataclass(frozen=True)
class LatencyScenario:
    stage_ms: tuple[float, ...]
    hop_ms: float

    @property
    def stages(self) -> int:
        return len(self.stage_ms)

    @property
    def serial_step_ms(self) -> float:
        return sum(self.stage_ms) + self.hop_ms * max(self.stages - 1, 0)

    @property
    def pipeline_slot_ms(self) -> float:
        if not self.stage_ms:
            return 0.0
        slots = []
        for index, stage_ms in enumerate(self.stage_ms):
            outgoing_hop = self.hop_ms if index + 1 < self.stages else 0.0
            slots.append(stage_ms + outgoing_hop)
        return max(slots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="SPD eval raw per-sample JSONL")
    parser.add_argument(
        "--stage-ms",
        default="4,4",
        help="Comma-separated per-stage compute latency in ms, e.g. 4,4 or 3,5,6",
    )
    parser.add_argument(
        "--hop-ms",
        default="0,1,5,10,25",
        help="Comma-separated inter-stage activation hop latency scenarios in ms",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the default human table",
    )
    return parser.parse_args()


def parse_float_list(value: str, label: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise SystemExit(f"invalid {label}: {value!r}") from error
    if not parsed:
        raise SystemExit(f"{label} must contain at least one number")
    if any(item < 0.0 for item in parsed):
        raise SystemExit(f"{label} values must be non-negative")
    return parsed


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON") from error
    if not rows:
        raise SystemExit(f"{path} contains no rows")
    return rows


def trace_totals(rows: list[dict[str, Any]]) -> TraceTotals:
    stages = {int(row.get("num_stages") or 0) for row in rows}
    stages.discard(0)
    if len(stages) != 1:
        raise SystemExit(f"expected one num_stages value, got {sorted(stages)}")
    return TraceTotals(
        samples=len(rows),
        stages=stages.pop(),
        tokens=sum(int(row.get("new_tokens") or 0) for row in rows),
        decode_steps=sum(int(row.get("decode_loop_steps") or 0) for row in rows),
        accepted_flags=sum(int(row.get("n_accepted") or 0) for row in rows),
        acceptance_flags=sum(int(row.get("n_acceptance_flags") or 0) for row in rows),
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def simulate(rows: list[dict[str, Any]], totals: TraceTotals, scenario: LatencyScenario) -> dict[str, Any]:
    if scenario.stages != totals.stages:
        raise SystemExit(
            f"--stage-ms has {scenario.stages} stages but trace has {totals.stages} stages"
        )

    serial_ms_by_sample = [
        int(row.get("new_tokens") or 0) * scenario.serial_step_ms for row in rows
    ]
    spd_ms_by_sample = [
        (int(row.get("decode_loop_steps") or 0) / totals.stages) * scenario.pipeline_slot_ms
        for row in rows
    ]
    serial_total_ms = sum(serial_ms_by_sample)
    spd_total_ms = sum(spd_ms_by_sample)
    paper_like_no_spd_ms = totals.tokens * scenario.pipeline_slot_ms

    return {
        "stage_ms": list(scenario.stage_ms),
        "hop_ms": scenario.hop_ms,
        "serial_step_ms": scenario.serial_step_ms,
        "pipeline_slot_ms": scenario.pipeline_slot_ms,
        "paper_like_no_spd_ms": paper_like_no_spd_ms,
        "serial_split_no_spd_ms": serial_total_ms,
        "spd_pipeline_ms": spd_total_ms,
        "spd_vs_paper_like_no_spd": safe_ratio(paper_like_no_spd_ms, spd_total_ms),
        "spd_vs_serial_split_no_spd": safe_ratio(serial_total_ms, spd_total_ms),
        "serial_split_tok_s": safe_tok_s(totals.tokens, serial_total_ms),
        "spd_pipeline_tok_s": safe_tok_s(totals.tokens, spd_total_ms),
        "serial_request_p50_ms": percentile(serial_ms_by_sample, 0.5),
        "serial_request_p95_ms": percentile(serial_ms_by_sample, 0.95),
        "spd_request_p50_ms": percentile(spd_ms_by_sample, 0.5),
        "spd_request_p95_ms": percentile(spd_ms_by_sample, 0.95),
        "request_latency_p50_ratio": safe_ratio(
            percentile(serial_ms_by_sample, 0.5),
            percentile(spd_ms_by_sample, 0.5),
        ),
        "request_latency_p95_ratio": safe_ratio(
            percentile(serial_ms_by_sample, 0.95),
            percentile(spd_ms_by_sample, 0.95),
        ),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def safe_tok_s(tokens: int, total_ms: float) -> float:
    if total_ms <= 0.0:
        return 0.0
    return tokens / (total_ms / 1000.0)


def emit_table(totals: TraceTotals, results: list[dict[str, Any]]) -> None:
    print("Trace")
    print(f"  samples: {totals.samples}")
    print(f"  stages: {totals.stages}")
    print(f"  generated tokens: {totals.tokens}")
    print(f"  decode loop steps: {totals.decode_steps}")
    print(f"  accepted draft flags: {totals.accepted_flags}/{totals.acceptance_flags}")
    print(f"  aggregate acceptance: {totals.aggregate_acceptance_rate:.4f}")
    print(f"  equivalent accept length: {totals.equivalent_accept_length:.4f}")
    print(f"  paper theoretical gain: {totals.paper_theoretical_gain_pct:.2f}%")
    print()
    print("Latency scenarios")
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "hop_ms",
            "slot_ms",
            "serial_tok_s",
            "spd_tok_s",
            "spd_vs_serial",
            "paper_like_gain",
            "p50_serial_ms",
            "p50_spd_ms",
            "p95_serial_ms",
            "p95_spd_ms",
        ]
    )
    for result in results:
        writer.writerow(
            [
                f"{result['hop_ms']:.3f}",
                f"{result['pipeline_slot_ms']:.3f}",
                f"{result['serial_split_tok_s']:.2f}",
                f"{result['spd_pipeline_tok_s']:.2f}",
                f"{result['spd_vs_serial_split_no_spd']:.3f}",
                f"{result['spd_vs_paper_like_no_spd']:.3f}",
                f"{result['serial_request_p50_ms']:.2f}",
                f"{result['spd_request_p50_ms']:.2f}",
                f"{result['serial_request_p95_ms']:.2f}",
                f"{result['spd_request_p95_ms']:.2f}",
            ]
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.raw)
    totals = trace_totals(rows)
    stage_ms = parse_float_list(args.stage_ms, "--stage-ms")
    hop_values = parse_float_list(args.hop_ms, "--hop-ms")
    scenarios = [LatencyScenario(stage_ms=stage_ms, hop_ms=hop_ms) for hop_ms in hop_values]
    results = [simulate(rows, totals, scenario) for scenario in scenarios]

    payload = {
        "raw": str(args.raw),
        "totals": {
            "samples": totals.samples,
            "stages": totals.stages,
            "tokens": totals.tokens,
            "decode_steps": totals.decode_steps,
            "accepted_flags": totals.accepted_flags,
            "acceptance_flags": totals.acceptance_flags,
            "aggregate_acceptance_rate": totals.aggregate_acceptance_rate,
            "equivalent_accept_length": totals.equivalent_accept_length,
            "paper_theoretical_gain_pct": totals.paper_theoretical_gain_pct,
        },
        "assumptions": {
            "serial_split_no_spd": (
                "one generated token traverses every stage and inter-stage hop before the "
                "next target token is known"
            ),
            "spd_pipeline": (
                "real reference decode_loop_steps are converted to pipeline cycles by "
                "dividing by num_stages; each cycle costs the slowest stage slot"
            ),
            "stage_slot": "stage compute plus outgoing hop, except the final stage has no outgoing hop",
        },
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        emit_table(totals, results)


if __name__ == "__main__":
    main()
