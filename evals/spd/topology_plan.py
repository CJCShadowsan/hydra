"""Generic SPD topology plan helpers.

The current donor SPD head is fixed-stage. These helpers describe logical
hidden-state evidence separately from the donor architecture so the generic
sidecar path can evolve without pretending fixed `stage_projs` are topology
independent.
"""

from __future__ import annotations

import json
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Any


TOPOLOGY_PLAN_SCHEMA = "skippy-spd-topology-plan/v1"


def write_topology_plan(args: Namespace, artifact_dir: Path) -> Path:
    plan = build_topology_plan(args)
    if args.topology_plan_out:
        path = Path(args.topology_plan_out).expanduser().resolve()
    else:
        path = artifact_dir / "topology" / "skippy-spd-topology-plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote generic SPD topology plan -> {path}", flush=True)
    return path


def build_topology_plan(args: Namespace) -> dict[str, Any]:
    model = resolve_model_config_summary(args)
    num_layers = int(model["num_hidden_layers"])
    min_stages = validate_positive_int("topology_min_stages", args.topology_min_stages)
    max_stages = validate_positive_int("topology_max_stages", args.topology_max_stages)
    if min_stages > max_stages:
        raise RuntimeError(
            f"--topology-min-stages {min_stages} must be <= --topology-max-stages {max_stages}"
        )
    if max_stages > num_layers:
        raise RuntimeError(
            f"--topology-max-stages {max_stages} cannot exceed num_hidden_layers {num_layers}"
        )
    samples = validate_positive_int("topology_plan_samples", args.topology_plan_samples)
    if not 0.0 <= args.topology_tap_dropout < 1.0:
        raise RuntimeError("--topology-tap-dropout must be in [0.0, 1.0)")

    layouts = sample_topology_layouts(
        num_layers=num_layers,
        min_stages=min_stages,
        max_stages=max_stages,
        samples=samples,
        seed=int(args.topology_seed),
        tap_dropout=float(args.topology_tap_dropout),
    )

    return {
        "schema": TOPOLOGY_PLAN_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "hidden_state_convention": {
            "index_0": "token embedding before layer 0",
            "index_k": "hidden state after target layer k-1",
            "final_index": num_layers,
        },
        "policy": {
            "name": "generic-contiguous-layer-plan",
            "seed": int(args.topology_seed),
            "samples": samples,
            "min_stages": min_stages,
            "max_stages": max_stages,
            "tap_dropout": float(args.topology_tap_dropout),
            "num_spec_layers": int(args.num_spec_layers),
            "draft_top_k": int(args.draft_top_k),
        },
        "layouts": layouts,
        "notes": [
            "This plan is topology data for the generic sidecar path; it is not a trained head.",
            "The current reference SPD architecture still has per-stage projection weights.",
            "Generic training must consume logical layer-indexed taps plus masks rather than fixed stage IDs.",
        ],
    }


def resolve_model_config_summary(args: Namespace) -> dict[str, Any]:
    model_name = args.model_name
    config_path = Path(model_name).expanduser() / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            if args.topology_num_hidden_layers <= 0:
                raise RuntimeError(
                    "could not load model config for topology planning; pass "
                    "--topology-num-hidden-layers to generate a model-shape-only plan"
                ) from exc
            config = {}
        else:
            config = hf_config.to_dict()

    num_layers = int(config.get("num_hidden_layers") or args.topology_num_hidden_layers or 0)
    validate_positive_int("num_hidden_layers", num_layers)
    return {
        "model_name": model_name,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "num_hidden_layers": num_layers,
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
    }


def sample_topology_layouts(
    *,
    num_layers: int,
    min_stages: int,
    max_stages: int,
    samples: int,
    seed: int,
    tap_dropout: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    layouts: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    for num_stages in range(min_stages, max_stages + 1):
        boundaries = tuple(balanced_stage_boundaries(num_layers, num_stages))
        seen.add(boundaries)
        layouts.append(topology_layout_record("balanced", boundaries, tap_dropout))

    attempts = 0
    while len(layouts) < samples and attempts < samples * 64:
        attempts += 1
        num_stages = rng.randint(min_stages, max_stages)
        boundaries = tuple(random_stage_boundaries(num_layers, num_stages, rng))
        if boundaries in seen:
            continue
        seen.add(boundaries)
        layouts.append(topology_layout_record("random", boundaries, tap_dropout))

    if len(layouts) < samples:
        raise RuntimeError(
            f"could only generate {len(layouts)} unique topology layouts after {attempts} attempts"
        )
    return layouts[:samples]


def topology_layout_record(
    source: str,
    boundaries: tuple[int, ...],
    tap_dropout: float,
) -> dict[str, Any]:
    rows = derive_hidden_tap_indices(list(boundaries))
    return {
        "source": source,
        "num_stages": len(boundaries),
        "stage_layer_boundaries": list(boundaries),
        "shallow_hidden_layer_indices": rows,
        "logical_hidden_taps": [logical_hidden_taps(row) for row in rows],
        "tap_dropout": {
            "probability": float(tap_dropout),
            "required_indices": [0, boundaries[-1]],
            "dropout_applies_to": "intermediate logical layer taps",
        },
    }


def balanced_stage_boundaries(num_layers: int, num_stages: int) -> list[int]:
    validate_positive_int("num_layers", num_layers)
    validate_positive_int("num_stages", num_stages)
    if num_stages > num_layers:
        raise RuntimeError(f"num_stages {num_stages} cannot exceed num_layers {num_layers}")
    boundaries = [round(num_layers * (stage + 1) / num_stages) for stage in range(num_stages)]
    if boundaries[-1] != num_layers:
        boundaries[-1] = num_layers
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise RuntimeError(f"balanced topology is not strictly increasing: {boundaries}")
    return boundaries


def random_stage_boundaries(num_layers: int, num_stages: int, rng: random.Random) -> list[int]:
    validate_positive_int("num_layers", num_layers)
    validate_positive_int("num_stages", num_stages)
    if num_stages > num_layers:
        raise RuntimeError(f"num_stages {num_stages} cannot exceed num_layers {num_layers}")
    if num_stages == 1:
        return [num_layers]
    cuts = sorted(rng.sample(range(1, num_layers), num_stages - 1))
    return [*cuts, num_layers]


def derive_hidden_tap_indices(boundaries: list[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    for depth in range(len(boundaries), 0, -1):
        rows.append([0, *boundaries[:depth]])
    return rows


def logical_hidden_taps(indices: list[int]) -> list[dict[str, Any]]:
    if not indices:
        raise RuntimeError("hidden tap row must not be empty")
    max_index = max(indices)
    taps: list[dict[str, Any]] = []
    for index in indices:
        normalized_depth = 0.0 if max_index == 0 else round(float(index) / float(max_index), 6)
        taps.append(
            {
                "layer_index": int(index),
                "kind": "embedding" if index == 0 else "layer_output",
                "normalized_depth": normalized_depth,
            }
        )
    return taps


def validate_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero, got {value}")
    return value
