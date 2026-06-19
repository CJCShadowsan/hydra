#!/usr/bin/env python3
"""Dry-run a Hugging Face SPD sidecar qualification job.

The planner is intentionally side-effect free: it resolves topology, hardware,
cost, and command shape, then prints a reviewable plan. It does not submit an
HF Job and does not upload artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"
DEFAULT_DOCKER_IMAGE = "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel"


@dataclass(frozen=True)
class HardwarePlan:
    flavor: str
    pretty_name: str
    cpu: str | None
    ram: str | None
    accelerator: dict[str, Any] | None
    unit_cost_usd: float
    unit_label: str
    timeout_seconds: int
    max_cost_usd: float
    max_cost_limit_usd: float
    within_budget: bool
    auto_selected_hardware: bool
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--package-ref", default="meshllm/Qwen3-8B-Q4_K_M-layers")
    parser.add_argument("--package-revision", default="main")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--layer-count", type=int, default=0)
    parser.add_argument("--activation-width", type=int, default=0)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help=(
            "Target model vocabulary size. Required for native-package-fresh "
            "unless package metadata provides vocab_size."
        ),
    )
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument(
        "--stage-layer-boundaries",
        default="23,36",
        help=(
            "Comma-separated logical stage end indices. Leave empty to derive "
            "near-even boundaries from --layer-count or package metadata."
        ),
    )
    parser.add_argument("--num-spec-layers", type=int, default=4)
    parser.add_argument("--draft-top-k", type=int, default=4)
    parser.add_argument("--draft-vocab-size", type=int, default=32000)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--train-prompts", type=int, default=4096)
    parser.add_argument("--heldout-prompts", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=480)
    parser.add_argument("--verify-steps", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hard-label-weight", type=float, default=0.1)
    parser.add_argument("--warm-start-repo", default="meshllm/skippy-spd-qwen3-8b-s2-23")
    parser.add_argument("--warm-start-path", default="runs/20260618-122936/train")
    parser.add_argument(
        "--qualification-mode",
        choices=("raw-q4-adapt", "reference-train", "native-package-fresh"),
        default="raw-q4-adapt",
    )
    parser.add_argument("--output-repo", default="")
    parser.add_argument("--hf-namespace", default="meshllm")
    parser.add_argument("--mesh-llm-ref", default="")
    parser.add_argument("--work-dir", default="/workspace/spd-qualification")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--flavor", default="auto")
    parser.add_argument("--min-vram-gb", type=float, default=80.0)
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--max-cost-usd", type=float, default=50.0)
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument(
        "--physical-groups",
        default="",
        help=(
            "Optional physical fit of logical stage indices, e.g. '0-3,4-6,7-9'. "
            "Default is one physical bucket per logical stage."
        ),
    )
    parser.add_argument(
        "--physical-node-count",
        type=int,
        default=0,
        help="Build near-even contiguous physical groups when --physical-groups is omitted.",
    )
    parser.add_argument(
        "--logical-stage-ms",
        default="40",
        help=(
            "Comma-separated logical stage costs. One value is repeated for every "
            "logical stage. Used only for generated latency-simulation commands."
        ),
    )
    parser.add_argument(
        "--physical-stage-ms",
        default="",
        help="Override fitted physical stage costs for latency-simulation commands.",
    )
    parser.add_argument("--hop-ms", default="0.2,1,5,10")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    package_metadata = fetch_package_metadata(args.package_ref, args.package_revision, args.hf_endpoint)
    layer_count = resolve_positive_int(args.layer_count, package_metadata.get("layer_count"), "layer_count")
    activation_width = resolve_positive_int(
        args.activation_width,
        package_metadata.get("activation_width"),
        "activation_width",
    )
    vocab_size = resolve_vocab_size(args, package_metadata)
    boundaries = resolve_boundaries(args, layer_count)
    tap_rows = derive_hidden_tap_indices(boundaries)
    physical_groups = resolve_physical_groups(args, len(boundaries))
    physical_stage_ms = resolve_physical_stage_ms(args, physical_groups)
    hardware = plan_hardware(args)
    output_repo = args.output_repo or default_output_repo(args, boundaries)
    plan = build_plan(
        args=args,
        package_metadata=package_metadata,
        layer_count=layer_count,
        activation_width=activation_width,
        vocab_size=vocab_size,
        boundaries=boundaries,
        tap_rows=tap_rows,
        physical_groups=physical_groups,
        physical_stage_ms=physical_stage_ms,
        hardware=hardware,
        output_repo=output_repo,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        emit_human_summary(plan)


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "num_stages",
        "num_spec_layers",
        "draft_top_k",
        "draft_vocab_size",
        "train_prompts",
        "heldout_prompts",
        "verify_steps",
        "ctx_size",
        "epochs",
        "batch_size",
    )
    for field in positive_fields:
        if int(getattr(args, field)) <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")
    if args.max_cost_usd <= 0:
        raise SystemExit("--max-cost-usd must be positive")
    if args.min_vram_gb <= 0:
        raise SystemExit("--min-vram-gb must be positive")
    if args.vocab_size < 0:
        raise SystemExit("--vocab-size must be non-negative")
    if args.qualification_mode == "raw-q4-adapt" and not args.warm_start_repo.strip():
        raise SystemExit("--warm-start-repo is required for raw-q4-adapt")


def fetch_package_metadata(package_ref: str, revision: str, endpoint: str) -> dict[str, Any]:
    if not package_ref:
        return {}
    url = f"{endpoint.rstrip('/')}/{package_ref}/resolve/{revision}/model-package.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403, 404}:
            return {"metadata_error": f"{error.code} fetching {url}"}
        raise
    except Exception as error:
        return {"metadata_error": str(error)}


def resolve_positive_int(cli_value: int, metadata_value: Any, label: str) -> int:
    if cli_value > 0:
        return int(cli_value)
    if metadata_value is not None:
        value = int(metadata_value)
        if value > 0:
            return value
    raise SystemExit(f"could not resolve {label}; pass --{label.replace('_', '-')}")


def resolve_vocab_size(args: argparse.Namespace, package_metadata: dict[str, Any]) -> int:
    if args.vocab_size > 0:
        return int(args.vocab_size)
    for key in ("vocab_size", "vocabulary_size"):
        metadata_value = package_metadata.get(key)
        if metadata_value is not None and int(metadata_value) > 0:
            return int(metadata_value)
    if args.qualification_mode == "native-package-fresh":
        raise SystemExit(
            "could not resolve vocab_size for native-package-fresh; pass --vocab-size"
        )
    return 0


def resolve_boundaries(args: argparse.Namespace, layer_count: int) -> list[int]:
    raw = args.stage_layer_boundaries.strip()
    if raw:
        boundaries = parse_int_list(raw, "--stage-layer-boundaries")
    else:
        boundaries = derive_even_boundaries(layer_count, args.num_stages)
    if len(boundaries) != args.num_stages:
        raise SystemExit(
            f"--stage-layer-boundaries has {len(boundaries)} entries but "
            f"--num-stages is {args.num_stages}"
        )
    if boundaries[-1] != layer_count:
        raise SystemExit(
            f"last stage boundary must equal layer count {layer_count}, got {boundaries[-1]}"
        )
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise SystemExit(f"stage boundaries must be strictly increasing: {boundaries}")
    return boundaries


def parse_int_list(value: str, label: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"invalid {label}: {value!r}") from error
    if not parsed:
        raise SystemExit(f"{label} must not be empty")
    if any(item <= 0 for item in parsed):
        raise SystemExit(f"{label} values must be positive: {parsed}")
    return parsed


def derive_even_boundaries(layer_count: int, num_stages: int) -> list[int]:
    boundaries: list[int] = []
    previous = 0
    for index in range(1, num_stages + 1):
        boundary = round(index * layer_count / num_stages)
        boundary = max(boundary, previous + 1)
        boundary = min(boundary, layer_count)
        boundaries.append(boundary)
        previous = boundary
    if boundaries[-1] != layer_count:
        boundaries[-1] = layer_count
    return boundaries


def derive_hidden_tap_indices(boundaries: list[int]) -> list[list[int]]:
    rows: list[list[int]] = []
    for depth in range(len(boundaries), 0, -1):
        rows.append([0, *boundaries[:depth]])
    return rows


def resolve_physical_groups(args: argparse.Namespace, logical_stages: int) -> list[list[int]]:
    if args.physical_groups.strip():
        groups = parse_physical_groups(args.physical_groups, logical_stages)
    elif args.physical_node_count > 0:
        groups = derive_physical_groups(logical_stages, args.physical_node_count)
    else:
        groups = [[index] for index in range(logical_stages)]
    seen = sorted(index for group in groups for index in group)
    expected = list(range(logical_stages))
    if seen != expected:
        raise SystemExit(f"physical groups must cover logical stages {expected}, got {seen}")
    return groups


def parse_physical_groups(value: str, logical_stages: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw_group in value.split(","):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        if "-" in raw_group:
            left, right = raw_group.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise SystemExit(f"invalid physical group range: {raw_group}")
            group = list(range(start, end + 1))
        else:
            group = [int(raw_group)]
        if any(index < 0 or index >= logical_stages for index in group):
            raise SystemExit(f"physical group out of range for {logical_stages} stages: {group}")
        groups.append(group)
    if not groups:
        raise SystemExit("--physical-groups was empty")
    return groups


def derive_physical_groups(logical_stages: int, physical_nodes: int) -> list[list[int]]:
    if physical_nodes <= 0:
        raise SystemExit("--physical-node-count must be positive")
    if physical_nodes > logical_stages:
        raise SystemExit("--physical-node-count cannot exceed logical stage count")
    groups: list[list[int]] = []
    start = 0
    for node_index in range(physical_nodes):
        remaining_stages = logical_stages - start
        remaining_nodes = physical_nodes - node_index
        width = math.ceil(remaining_stages / remaining_nodes)
        groups.append(list(range(start, start + width)))
        start += width
    return groups


def resolve_physical_stage_ms(args: argparse.Namespace, physical_groups: list[list[int]]) -> list[float]:
    if args.physical_stage_ms.strip():
        values = parse_float_list(args.physical_stage_ms, "--physical-stage-ms")
        if len(values) != len(physical_groups):
            raise SystemExit(
                f"--physical-stage-ms has {len(values)} values but physical fit has "
                f"{len(physical_groups)} groups"
            )
        return values
    logical_values = parse_float_list(args.logical_stage_ms, "--logical-stage-ms")
    logical_count = max(index for group in physical_groups for index in group) + 1
    if len(logical_values) == 1:
        logical_values = logical_values * logical_count
    if len(logical_values) != logical_count:
        raise SystemExit(
            f"--logical-stage-ms has {len(logical_values)} values but logical fit has {logical_count} stages"
        )
    return [sum(logical_values[index] for index in group) for group in physical_groups]


def parse_float_list(value: str, label: str) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"invalid {label}: {value!r}") from error
    if not parsed:
        raise SystemExit(f"{label} must not be empty")
    if any(item < 0.0 for item in parsed):
        raise SystemExit(f"{label} values must be non-negative: {parsed}")
    return parsed


def plan_hardware(args: argparse.Namespace) -> HardwarePlan:
    hardware = fetch_hardware(args.hf_endpoint)
    selected, reason = select_hardware(hardware, args.flavor, args.min_vram_gb)
    timeout_seconds = parse_duration_seconds(args.timeout)
    unit_cost = resolved_unit_cost_usd(selected)
    unit_label = selected.get("unitLabel") or selected.get("unit_label") or "minute"
    max_cost = estimate_cost_usd(unit_cost, unit_label, timeout_seconds)
    within_budget = max_cost <= args.max_cost_usd
    if not within_budget:
        raise SystemExit(
            f"planned job max cost ${max_cost:.2f} exceeds --max-cost-usd ${args.max_cost_usd:.2f}"
        )
    return HardwarePlan(
        flavor=str(selected["name"]),
        pretty_name=str(selected.get("prettyName") or selected["name"]),
        cpu=selected.get("cpu"),
        ram=selected.get("ram"),
        accelerator=selected.get("accelerator"),
        unit_cost_usd=unit_cost,
        unit_label=str(unit_label),
        timeout_seconds=timeout_seconds,
        max_cost_usd=max_cost,
        max_cost_limit_usd=args.max_cost_usd,
        within_budget=within_budget,
        auto_selected_hardware=args.flavor == "auto",
        selection_reason=reason,
    )


def fetch_hardware(endpoint: str) -> list[dict[str, Any]]:
    url = f"{endpoint.rstrip('/')}/api/jobs/hardware"
    with urllib.request.urlopen(url, timeout=30) as response:
        obj = json.load(response)
    if not isinstance(obj, list):
        raise SystemExit(f"unexpected HF hardware response from {url}")
    return [item for item in obj if isinstance(item, dict)]


def select_hardware(
    hardware: list[dict[str, Any]],
    requested_flavor: str,
    min_vram_gb: float,
) -> tuple[dict[str, Any], str]:
    gpu_flavors = [item for item in hardware if item.get("accelerator")]
    if requested_flavor != "auto":
        for item in hardware:
            if item.get("name") == requested_flavor:
                return item, "requested explicitly"
        raise SystemExit(f"unknown Hugging Face Jobs flavor: {requested_flavor}")
    candidates = [
        item
        for item in gpu_flavors
        if accelerator_vram_gb(item.get("accelerator")) >= min_vram_gb
    ]
    if not candidates:
        raise SystemExit(f"no GPU HF Jobs flavor has at least {min_vram_gb:.1f} GB VRAM")
    candidates.sort(key=lambda item: resolved_unit_cost_usd(item))
    selected = candidates[0]
    return (
        selected,
        f"auto-selected cheapest GPU flavor with at least {min_vram_gb:.1f} GB VRAM",
    )


def accelerator_vram_gb(accelerator: Any) -> float:
    if not isinstance(accelerator, dict):
        return 0.0
    return parse_size_gb(str(accelerator.get("vram") or "0 GB"))


def parse_size_gb(value: str) -> float:
    parts = value.split()
    if not parts:
        return 0.0
    amount = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else "gb"
    multipliers = {
        "gb": 1.0,
        "gib": 1.073741824,
        "mb": 1.0 / 1024.0,
        "mib": 1.0 / 1024.0,
        "tb": 1024.0,
        "tib": 1099.511627776,
    }
    return amount * multipliers.get(unit, 1.0)


def resolved_unit_cost_usd(item: dict[str, Any]) -> float:
    if item.get("unitCostUSD") is not None:
        return float(item["unitCostUSD"])
    if item.get("unitCostMicroUSD") is not None:
        return float(item["unitCostMicroUSD"]) / 1_000_000.0
    raise SystemExit(f"HF hardware flavor {item.get('name')} has no unit cost")


def parse_duration_seconds(value: str) -> int:
    value = value.strip().lower()
    if not value:
        raise SystemExit("duration must not be empty")
    units = {"s": 1, "m": 60, "h": 3600, "d": 86_400}
    suffix = value[-1]
    if suffix in units:
        amount = float(value[:-1])
        return int(amount * units[suffix])
    return int(float(value))


def estimate_cost_usd(unit_cost_usd: float, unit_label: str, timeout_seconds: int) -> float:
    unit = unit_label.lower()
    if unit == "second":
        return unit_cost_usd * timeout_seconds
    if unit == "minute":
        return unit_cost_usd * timeout_seconds / 60.0
    if unit == "hour":
        return unit_cost_usd * timeout_seconds / 3600.0
    if unit == "day":
        return unit_cost_usd * timeout_seconds / 86_400.0
    raise SystemExit(f"unsupported HF pricing unit: {unit_label}")


def default_output_repo(args: argparse.Namespace, boundaries: list[int]) -> str:
    model_stem = args.base_model.split("/")[-1].lower().replace(".", "").replace("_", "-")
    quant_stem = args.package_ref.split("/")[-1].replace("-layers", "").lower()
    topology = "s" + str(len(boundaries)) + "-" + "-".join(str(item) for item in boundaries[:-1])
    return f"{args.hf_namespace}/skippy-spd-{model_stem}-{quant_stem}-{topology}-product"


def current_git_ref() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        branch = result.stdout.strip()
        return branch if branch else "main"
    except Exception:
        return "main"


def build_plan(
    *,
    args: argparse.Namespace,
    package_metadata: dict[str, Any],
    layer_count: int,
    activation_width: int,
    vocab_size: int,
    boundaries: list[int],
    tap_rows: list[list[int]],
    physical_groups: list[list[int]],
    physical_stage_ms: list[float],
    hardware: HardwarePlan,
    output_repo: str,
) -> dict[str, Any]:
    mesh_ref = args.mesh_llm_ref.strip() or current_git_ref()
    layer_end = boundaries[-1]
    splits = boundaries[:-1]
    required_taps = sorted({tap for row in tap_rows for tap in row})
    non_embedding_taps = [tap for tap in required_taps if tap != 0]
    package_model_id = package_metadata.get("model_id")
    model_id = args.model_id or str(package_model_id or args.package_ref)
    work_dir = args.work_dir.rstrip("/")
    commands = build_commands(
        args,
        work_dir,
        boundaries,
        splits,
        layer_end,
        activation_width,
        vocab_size,
        output_repo,
        mesh_ref,
    )
    return {
        "schema": "skippy-spd-hf-qualification-plan/v1",
        "dry_run": True,
        "confirm_required": True,
        "qualification_mode": args.qualification_mode,
        "model": {
            "base_model": args.base_model,
            "package_ref": args.package_ref,
            "package_revision": args.package_revision,
            "model_id": model_id,
            "layer_count": layer_count,
            "activation_width": activation_width,
            "vocab_size": vocab_size if vocab_size > 0 else None,
            "package_metadata_error": package_metadata.get("metadata_error"),
        },
        "topology": {
            "num_stages": len(boundaries),
            "stage_layer_boundaries": boundaries,
            "physical_split_boundaries": splits,
            "layer_end": layer_end,
            "num_spec_layers": args.num_spec_layers,
            "draft_top_k": args.draft_top_k,
            "draft_vocab_size": args.draft_vocab_size,
            "vocab_size": vocab_size if vocab_size > 0 else None,
            "shallow_hidden_layer_indices": ";".join(
                ",".join(str(index) for index in row) for row in tap_rows
            ),
            "shallow_hidden_layer_index_rows": tap_rows,
            "required_hf_hidden_state_indices": required_taps,
            "spd_tap_return_hf_indices": non_embedding_taps,
        },
        "physical_fit": {
            "logical_stage_groups": physical_groups,
            "physical_stage_count": len(physical_groups),
            "physical_stage_ms_for_latency_sim": physical_stage_ms,
            "hop_ms_scenarios": parse_float_list(args.hop_ms, "--hop-ms"),
            "note": (
                "logical groups are contiguous stage-index buckets; internal taps "
                "must still be returned when logical stages are colocated"
            ),
        },
        "data": {
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "train_prompts": args.train_prompts,
            "heldout_prompts": args.heldout_prompts,
            "max_prompt_tokens": args.max_prompt_tokens,
            "verify_steps": args.verify_steps,
            "ctx_size": args.ctx_size,
        },
        "training": {
            "warm_start_repo": args.warm_start_repo if args.qualification_mode == "raw-q4-adapt" else None,
            "warm_start_path": args.warm_start_path if args.qualification_mode == "raw-q4-adapt" else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "kl_weight": args.kl_weight,
            "hard_label_weight": args.hard_label_weight,
            "input_mode": training_input_mode(args.qualification_mode),
            "torch_dtype": "bfloat16",
            "base_model_load": training_base_model_load(args.qualification_mode),
        },
        "hf_job": {
            "namespace": args.hf_namespace,
            "output_repo": output_repo,
            "mesh_llm_ref": mesh_ref,
            "docker_image": args.docker_image,
            "hardware": asdict(hardware),
            "spec_preview": {
                "dockerImage": args.docker_image,
                "command": ["bash", "-lc", "bash run-spd-qualification.sh"],
                "arguments": [],
                "environment": job_environment(args, output_repo, mesh_ref),
                "secrets": {"HF_TOKEN": "<redacted; required only on confirmed submit>"},
                "flavor": hardware.flavor,
                "timeoutSeconds": hardware.timeout_seconds,
                "volumes": [],
            },
        },
        "artifact_distribution": {
            "layer_package": (
                "Mesh/Skippy should keep resolving and downloading the base layer "
                "package per physical stage node; the HF qualification job downloads "
                "the package snapshot only because it is a single-machine pre-LAN gate."
            ),
            "spd_bundle": (
                "The SPD predictor bundle is coordinator-owned by default. Worker "
                "nodes do not need sidecar weights; they need the derived "
                "spd_tap_return_hf_indices allowlist and must return the requested taps."
            ),
            "colocated_logical_stages": (
                "When Mesh fits multiple logical SPD stages onto one physical node, "
                "that node still has to expose internal logical-boundary taps."
            ),
        },
        "commands": commands,
        "acceptance_gate": {
            "must_match_content": True,
            "tap_failures_must_equal": 0,
            "rust_fixture_parity": rust_fixture_parity_gate(args.qualification_mode),
            "broad_heldout_saved_round_trips": "must exceed unsaved round trips with margin",
            "latency_simulation": (
                "run simulate_latency.py on the generated OpenAI smoke report using "
                "physical_stage_ms_for_latency_sim and hop_ms_scenarios"
            ),
            "not_a_speed_claim": True,
        },
    }


def job_environment(args: argparse.Namespace, output_repo: str, mesh_ref: str) -> dict[str, str]:
    return {
        "BASE_MODEL": args.base_model,
        "PACKAGE_REF": args.package_ref,
        "PACKAGE_REVISION": args.package_revision,
        "OUTPUT_REPO": output_repo,
        "MESH_LLM_REF": mesh_ref,
        "DATASET": args.dataset,
        "DATASET_SPLIT": args.dataset_split,
        "QUALIFICATION_MODE": args.qualification_mode,
    }


def training_input_mode(qualification_mode: str) -> str:
    if qualification_mode in {"raw-q4-adapt", "native-package-fresh"}:
        return "raw"
    return "reference"


def training_base_model_load(qualification_mode: str) -> str:
    if qualification_mode == "native-package-fresh":
        return "skipped_autoconfig_only"
    if qualification_mode == "raw-q4-adapt":
        return "full_base_model_loaded_by_current_adaptation_script"
    return "full_base_model_loaded_by_reference_trainer"


def rust_fixture_parity_gate(qualification_mode: str) -> str:
    if qualification_mode == "native-package-fresh":
        return (
            "skipped for the first native-package-fresh lane; it exports a "
            "serving fixture only, until native parity fixture export exists"
        )
    return "must pass if export runs"


def build_commands(
    args: argparse.Namespace,
    work_dir: str,
    boundaries: list[int],
    splits: list[int],
    layer_end: int,
    activation_width: int,
    vocab_size: int,
    output_repo: str,
    mesh_ref: str,
) -> dict[str, Any]:
    split_arg = ",".join(str(item) for item in splits)
    stage_ms = "$(cat physical-stage-ms.txt)"
    warm_start_dir = f"{work_dir}/warm-start"
    prompt_dir = f"{work_dir}/prompts"
    train_corpus_dir = f"{work_dir}/product-train-corpus"
    heldout_corpus_dir = f"{work_dir}/product-heldout-corpus"
    artifact_dir = f"{work_dir}/artifact"
    reference_dir = f"{work_dir}/speculative_pipeline_decoding"
    package_dir = f"{work_dir}/package"
    setup = [
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "apt-get update",
        (
            "apt-get install -y --no-install-recommends "
            "ca-certificates curl git build-essential pkg-config cmake ninja-build "
            "clang lld protobuf-compiler libssl-dev python3-pip"
        ),
        "python3 -m pip install -U pip",
        (
            "if ! command -v cargo >/dev/null 2>&1; then "
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --profile minimal; "
            "fi"
        ),
        "export PATH=\"$HOME/.cargo/bin:$PATH\"",
        "rustup default stable",
        "if ! command -v just >/dev/null 2>&1; then cargo install just --locked; fi",
        f"mkdir -p {work_dir} {prompt_dir} {artifact_dir}",
        f"git clone --depth 1 --branch {shell_quote(mesh_ref)} https://github.com/Mesh-LLM/mesh-llm.git {work_dir}/mesh-llm",
        f"cd {work_dir}/mesh-llm",
        "if [[ -n \"${MESH_LLM_PATCH_PATH:-}\" ]]; then git apply \"$MESH_LLM_PATCH_PATH\"; fi",
        "git status --short",
        "CUDA_ARCH=\"$(scripts/detect-cuda-arch.sh)\"",
        "CUDA_ARCH_SUFFIX=\"$(printf '%s' \"$CUDA_ARCH\" | tr ';, /:' '_____' | tr -cd 'A-Za-z0-9_.-')\"",
        "export LLAMA_STAGE_BACKEND=cuda",
        "export LLAMA_STAGE_CUDA_ARCHITECTURES=\"$CUDA_ARCH\"",
        "export LLAMA_STAGE_BUILD_DIR=\"$PWD/.deps/llama-build/build-stage-abi-cuda-sm$CUDA_ARCH_SUFFIX\"",
        "MESH_LLM_SKIP_UI=1 MESH_LLM_BUILD_PROFILE=release just build-runtime backend=cuda cuda_arch=\"$CUDA_ARCH\"",
        (
            "just with-lld env LLAMA_STAGE_BACKEND=cuda "
            "LLAMA_STAGE_BUILD_DIR=\"$LLAMA_STAGE_BUILD_DIR\" "
            "cargo build --release --locked -p skippy-bench -p skippy-server"
        ),
        f"git clone --depth 1 https://github.com/yuyijiong/speculative_pipeline_decoding.git {reference_dir}",
    ]
    prompt_command = (
        "python3 evals/spd/build_hf_prompt_tokens.py "
        f"--dataset {shell_quote(args.dataset)} "
        f"--dataset-split {shell_quote(args.dataset_split)} "
        f"--model-name {shell_quote(args.base_model)} "
        f"--out-dir {prompt_dir} "
        f"--train-prompts {args.train_prompts} "
        f"--heldout-prompts {args.heldout_prompts} "
        f"--max-prompt-tokens {args.max_prompt_tokens} "
        "--shuffle --seed 23"
    )
    if args.qualification_mode == "reference-train":
        return build_reference_train_commands(
            args=args,
            setup=setup,
            prompt_command=prompt_command,
            work_dir=work_dir,
            package_dir=package_dir,
            prompt_dir=prompt_dir,
            split_arg=split_arg,
            layer_end=layer_end,
            activation_width=activation_width,
            artifact_dir=artifact_dir,
            stage_ms=stage_ms,
            output_repo=output_repo,
        )

    if args.qualification_mode == "native-package-fresh":
        return build_native_package_fresh_commands(
            args=args,
            setup=setup,
            prompt_command=prompt_command,
            work_dir=work_dir,
            package_dir=package_dir,
            prompt_dir=prompt_dir,
            train_corpus_dir=train_corpus_dir,
            heldout_corpus_dir=heldout_corpus_dir,
            artifact_dir=artifact_dir,
            reference_dir=reference_dir,
            split_arg=split_arg,
            boundaries=boundaries,
            layer_end=layer_end,
            activation_width=activation_width,
            vocab_size=vocab_size,
            stage_ms=stage_ms,
            output_repo=output_repo,
        )

    if args.qualification_mode == "raw-q4-adapt":
        setup.extend(
            [
                "python3 -m pip install -U pip huggingface_hub safetensors transformers datasets accelerate torch",
                download_warm_start_command(args, warm_start_dir),
                download_package_command(args, package_dir),
            ]
        )
    capture_common = (
        "target/release/skippy-bench spd-live-tap-parity "
        f"--manifest {warm_start_dir}/skippy-spd-head.json "
        f"--fixture {warm_start_dir}/spd-parity-fixture.safetensors "
        f"--model-path {package_dir} "
        f"--splits {split_arg} --layer-end {layer_end} "
        f"--ctx-size {args.ctx_size} --n-gpu-layers=-1 "
        f"--top-k {args.draft_top_k} --verify-steps {args.verify_steps} "
        "--product-native-teacher-logits"
    )
    capture_train = (
        f"{capture_common} --prompt-token-file {prompt_dir}/train-prompt-token-ids.jsonl "
        f"--product-corpus-dir {train_corpus_dir} --output {work_dir}/train-capture.json"
    )
    capture_heldout = (
        f"{capture_common} --prompt-token-file {prompt_dir}/heldout-prompt-token-ids.jsonl "
        f"--product-corpus-dir {heldout_corpus_dir} --output {work_dir}/heldout-capture.json"
    )
    convert = [
        "python3 evals/spd/prepare_product_activation_corpus.py "
        f"--corpus-dir {train_corpus_dir} --out {work_dir}/train-corpus.safetensors "
        f"--summary-json {work_dir}/train-corpus-summary.json",
        "python3 evals/spd/prepare_native_product_teacher_logits.py "
        f"--corpus-dir {train_corpus_dir} --out {work_dir}/train-teacher.safetensors "
        f"--summary-json {work_dir}/train-teacher-summary.json",
        "python3 evals/spd/prepare_product_activation_corpus.py "
        f"--corpus-dir {heldout_corpus_dir} --out {work_dir}/heldout-corpus.safetensors "
        f"--summary-json {work_dir}/heldout-corpus-summary.json",
        "python3 evals/spd/prepare_native_product_teacher_logits.py "
        f"--corpus-dir {heldout_corpus_dir} --out {work_dir}/heldout-teacher.safetensors "
        f"--summary-json {work_dir}/heldout-teacher-summary.json",
    ]
    train = (
        "PYTHONPATH=evals/spd python3 evals/spd/train_product_activation_head.py "
        f"--reference-dir {reference_dir} "
        f"--checkpoint {warm_start_dir}/speculation_head_final.pt "
        f"--product-corpus {work_dir}/train-corpus.safetensors "
        f"--teacher-logits {work_dir}/train-teacher.safetensors "
        f"--out-checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--base-model-path {shell_quote(args.base_model)} "
        f"--summary-json {artifact_dir}/train-summary.json "
        "--init-mode checkpoint --input-mode raw "
        f"--epochs {args.epochs} --batch-size {args.batch_size} "
        f"--learning-rate {args.learning_rate} --weight-decay {args.weight_decay} "
        f"--kl-weight {args.kl_weight} --hard-label-weight {args.hard_label_weight} "
        "--device cuda --model-torch-dtype bfloat16"
    )
    score = (
        "PYTHONPATH=evals/spd python3 evals/spd/score_product_activation_head.py "
        f"--reference-dir {reference_dir} "
        f"--checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--product-corpus {work_dir}/heldout-corpus.safetensors "
        f"--teacher-logits {work_dir}/heldout-teacher.safetensors "
        f"--base-model-path {shell_quote(args.base_model)} --input-mode raw "
        f"--batch-size {args.batch_size} --top-k {args.draft_top_k} "
        "--device cuda --model-torch-dtype bfloat16 "
        f"--summary-json {artifact_dir}/score-heldout-summary.json"
    )
    export = [
        "python3 evals/spd/export_spd_head.py "
        f"--checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--manifest {warm_start_dir}/skippy-spd-head.json "
        f"--manifest-out {artifact_dir}/skippy-spd-head.json "
        f"--out-dir {artifact_dir} --base-model-path {shell_quote(args.base_model)}",
        "python3 evals/spd/export_parity_fixture.py "
        f"--reference-dir {reference_dir} "
        f"--checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--base-model-path {shell_quote(args.base_model)} "
        f"--out {artifact_dir}/spd-parity-fixture.safetensors "
        f"--top-k {max(args.draft_top_k, 8)} --device cuda",
        "target/release/skippy-bench spd-fixture-parity "
        f"--manifest {artifact_dir}/skippy-spd-head.json "
        f"--fixture {artifact_dir}/spd-parity-fixture.safetensors "
        f"--top-k {max(args.draft_top_k, 8)}",
    ]
    smoke = (
        "target/release/skippy-bench spd-openai-smoke "
        "--stage-server-bin target/release/skippy-server "
        f"--manifest {artifact_dir}/skippy-spd-head.json "
        f"--fixture {artifact_dir}/spd-parity-fixture.safetensors "
        f"--model-path {package_dir} --model-id {shell_quote(args.package_ref)} "
        f"--splits {split_arg} --layer-end {layer_end} "
        f"--ctx-size {args.ctx_size} --n-gpu-layers=-1 "
        f"--activation-width {activation_width} --max-tokens 4 "
        f"--prompt-file {prompt_dir}/heldout-prompts.jsonl --prompt-limit {args.heldout_prompts} "
        "--repeat-count 1 --run-baseline true --run-spd true "
        "--spd-rolling-executor --max-inflight 4 "
        f"--output {artifact_dir}/openai-heldout-rolling.json"
    )
    latency = (
        "python3 evals/spd/simulate_latency.py "
        f"--openai-report {artifact_dir}/openai-heldout-rolling.json "
        f"--stage-ms {stage_ms} --hop-ms {shell_quote(args.hop_ms)} "
        f"--json > {artifact_dir}/latency-simulation.json"
    )
    upload = (
        "python3 - <<'PY'\n"
        "from huggingface_hub import HfApi\n"
        f"api = HfApi(); api.create_repo({output_repo!r}, repo_type='model', private=True, exist_ok=True)\n"
        f"api.upload_folder(repo_id={output_repo!r}, repo_type='model', folder_path={artifact_dir!r}, path_in_repo='runs/product-q4')\n"
        "PY"
    )
    return {
        "setup": setup,
        "build_prompts": [prompt_command],
        "capture": [capture_train, capture_heldout],
        "convert": convert,
        "train": [train],
        "score": [score],
        "export_and_parity": export,
        "package_smoke": [smoke],
        "latency_simulation": [latency],
        "upload": [upload],
        "write_physical_stage_ms": [
            "printf '%s\\n' "
            + shell_quote(",".join(f"{value:g}" for value in resolve_physical_stage_ms(args, resolve_physical_groups(args, len(boundaries)))))
            + " > physical-stage-ms.txt"
        ],
    }


def build_native_package_fresh_commands(
    *,
    args: argparse.Namespace,
    setup: list[str],
    prompt_command: str,
    work_dir: str,
    package_dir: str,
    prompt_dir: str,
    train_corpus_dir: str,
    heldout_corpus_dir: str,
    artifact_dir: str,
    reference_dir: str,
    split_arg: str,
    boundaries: list[int],
    layer_end: int,
    activation_width: int,
    vocab_size: int,
    stage_ms: str,
    output_repo: str,
) -> dict[str, Any]:
    if vocab_size <= 0:
        raise SystemExit("--vocab-size is required for native-package-fresh")
    setup = [
        *setup,
        "python3 -m pip install -U pip huggingface_hub safetensors transformers datasets accelerate torch",
        download_package_command(args, package_dir),
    ]
    boundaries_arg = ",".join(str(item) for item in boundaries)
    capture_common = (
        "target/release/skippy-bench spd-product-corpus-capture "
        f"--model-path {package_dir} "
        f"--splits {split_arg} --layer-end {layer_end} "
        f"--hidden-size {activation_width} --vocab-size {vocab_size} "
        f"--draft-vocab-size {args.draft_vocab_size} "
        f"--num-spec-layers {args.num_spec_layers} "
        f"--ctx-size {args.ctx_size} --n-gpu-layers=-1 "
        f"--top-k {args.draft_top_k} --verify-steps {args.verify_steps} "
        "--product-native-teacher-logits"
    )
    capture_train = (
        f"{capture_common} --prompt-token-file {prompt_dir}/train-prompt-token-ids.jsonl "
        f"--product-corpus-dir {train_corpus_dir} --output {work_dir}/train-capture.json"
    )
    capture_heldout = (
        f"{capture_common} --prompt-token-file {prompt_dir}/heldout-prompt-token-ids.jsonl "
        f"--product-corpus-dir {heldout_corpus_dir} --output {work_dir}/heldout-capture.json"
    )
    convert = [
        "python3 evals/spd/prepare_product_activation_corpus.py "
        f"--corpus-dir {train_corpus_dir} --out {work_dir}/train-corpus.safetensors "
        f"--summary-json {work_dir}/train-corpus-summary.json",
        "python3 evals/spd/prepare_native_product_teacher_logits.py "
        f"--corpus-dir {train_corpus_dir} --out {work_dir}/train-teacher.safetensors "
        f"--summary-json {work_dir}/train-teacher-summary.json",
        "python3 evals/spd/prepare_product_activation_corpus.py "
        f"--corpus-dir {heldout_corpus_dir} --out {work_dir}/heldout-corpus.safetensors "
        f"--summary-json {work_dir}/heldout-corpus-summary.json",
        "python3 evals/spd/prepare_native_product_teacher_logits.py "
        f"--corpus-dir {heldout_corpus_dir} --out {work_dir}/heldout-teacher.safetensors "
        f"--summary-json {work_dir}/heldout-teacher-summary.json",
    ]
    train = (
        "PYTHONPATH=evals/spd python3 evals/spd/train_product_activation_head_only.py "
        f"--reference-dir {reference_dir} "
        f"--product-corpus {work_dir}/train-corpus.safetensors "
        f"--teacher-logits {work_dir}/train-teacher.safetensors "
        f"--out-checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--manifest-out {artifact_dir}/skippy-spd-head.json "
        f"--base-model-path {shell_quote(args.base_model)} "
        f"--stage-layer-boundaries {boundaries_arg} "
        f"--num-spec-layers {args.num_spec_layers} "
        f"--summary-json {artifact_dir}/train-summary.json "
        f"--epochs {args.epochs} --batch-size {args.batch_size} "
        f"--learning-rate {args.learning_rate} --weight-decay {args.weight_decay} "
        f"--kl-weight {args.kl_weight} --hard-label-weight {args.hard_label_weight} "
        "--device cuda --model-torch-dtype bfloat16"
    )
    score = (
        "PYTHONPATH=evals/spd python3 evals/spd/score_product_activation_head_only.py "
        f"--reference-dir {reference_dir} "
        f"--checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--product-corpus {work_dir}/heldout-corpus.safetensors "
        f"--teacher-logits {work_dir}/heldout-teacher.safetensors "
        f"--base-model-path {shell_quote(args.base_model)} "
        f"--batch-size {args.batch_size} --top-k {args.draft_top_k} "
        "--device cuda --model-torch-dtype bfloat16 "
        f"--summary-json {artifact_dir}/score-heldout-summary.json"
    )
    export = [
        "python3 evals/spd/export_spd_head.py "
        f"--checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--manifest {artifact_dir}/skippy-spd-head.json "
        f"--manifest-out {artifact_dir}/skippy-spd-head.json "
        f"--out-dir {artifact_dir} --base-model-path {shell_quote(args.base_model)}",
        "python3 evals/spd/export_product_serving_fixture.py "
        f"--product-corpus {work_dir}/heldout-corpus.safetensors "
        f"--out {artifact_dir}/spd-serving-fixture.safetensors "
        f"--summary-json {artifact_dir}/serving-fixture-summary.json",
    ]
    smoke = (
        "target/release/skippy-bench spd-openai-smoke "
        "--stage-server-bin target/release/skippy-server "
        f"--manifest {artifact_dir}/skippy-spd-head.json "
        f"--fixture {artifact_dir}/spd-serving-fixture.safetensors "
        f"--model-path {package_dir} --model-id {shell_quote(args.package_ref)} "
        f"--splits {split_arg} --layer-end {layer_end} "
        f"--ctx-size {args.ctx_size} --n-gpu-layers=-1 "
        f"--activation-width {activation_width} --max-tokens 4 "
        f"--prompt-file {prompt_dir}/heldout-prompts.jsonl --prompt-limit {args.heldout_prompts} "
        "--repeat-count 1 --run-baseline true --run-spd true "
        "--spd-rolling-executor --max-inflight 4 "
        f"--output {artifact_dir}/openai-heldout-rolling.json"
    )
    latency = (
        "python3 evals/spd/simulate_latency.py "
        f"--openai-report {artifact_dir}/openai-heldout-rolling.json "
        f"--stage-ms {stage_ms} --hop-ms {shell_quote(args.hop_ms)} "
        f"--json > {artifact_dir}/latency-simulation.json"
    )
    upload = (
        "python3 - <<'PY'\n"
        "from huggingface_hub import HfApi\n"
        f"api = HfApi(); api.create_repo({output_repo!r}, repo_type='model', private=True, exist_ok=True)\n"
        f"api.upload_folder(repo_id={output_repo!r}, repo_type='model', folder_path={artifact_dir!r}, path_in_repo='runs/native-package-fresh')\n"
        "PY"
    )
    return {
        "setup": setup,
        "build_prompts": [prompt_command],
        "capture": [capture_train, capture_heldout],
        "convert": convert,
        "train": [train],
        "score": [score],
        "export_serving_bundle": export,
        "rust_fixture_parity": [
            "echo native-package-fresh exports a serving fixture only; Rust fixture parity is skipped until native fixture export exists"
        ],
        "package_smoke": [smoke],
        "latency_simulation": [latency],
        "upload": [upload],
        "write_physical_stage_ms": [
            "printf '%s\\n' "
            + shell_quote(",".join(f"{value:g}" for value in resolve_physical_stage_ms(args, resolve_physical_groups(args, len(boundaries)))))
            + " > physical-stage-ms.txt"
        ],
    }


def build_reference_train_commands(
    *,
    args: argparse.Namespace,
    setup: list[str],
    prompt_command: str,
    work_dir: str,
    package_dir: str,
    prompt_dir: str,
    split_arg: str,
    layer_end: int,
    activation_width: int,
    artifact_dir: str,
    stage_ms: str,
    output_repo: str,
) -> dict[str, Any]:
    setup = [
        *setup,
        "python3 -m pip install -U pip huggingface_hub safetensors transformers datasets accelerate torch",
        download_package_command(args, package_dir),
    ]
    reference_work = f"{work_dir}/reference-train"
    boundaries_arg = args.stage_layer_boundaries.strip() or ",".join(
        str(item) for item in derive_even_boundaries(layer_end, args.num_stages)
    )
    train = (
        "python3 evals/spd/hf_train_eval_qwen06.py "
        f"--work-dir {reference_work} "
        f"--model-name {shell_quote(args.base_model)} "
        f"--manifest-base-model-path {shell_quote(args.base_model)} "
        f"--dataset {shell_quote(args.dataset)} "
        f"--dataset-split {shell_quote(args.dataset_split)} "
        f"--train-rows {args.train_prompts} "
        f"--eval-rows-per-set {max(1, min(args.heldout_prompts, 96) // 3)} "
        f"--num-stages {args.num_stages} "
        f"--stage-layer-boundaries {boundaries_arg} "
        f"--num-spec-layers {args.num_spec_layers} "
        f"--max-length {args.ctx_size} "
        "--max-new-tokens 64 "
        f"--draft-top-k {args.draft_top_k} "
        "--device cuda --model-torch-dtype bfloat16 "
        "--upload-repo ''"
    )
    locate_artifact = (
        f"ARTIFACT_DIR=$(ls -td {reference_work}/artifacts/*/train | head -1); "
        f"mkdir -p {artifact_dir}; "
        f"cp \"$ARTIFACT_DIR\"/speculation_head_final.pt \"$ARTIFACT_DIR\"/skippy-spd-head.json {artifact_dir}/"
    )
    export = [
        locate_artifact,
        "python3 evals/spd/export_spd_head.py "
        f"--checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--manifest {artifact_dir}/skippy-spd-head.json "
        f"--out-dir {artifact_dir} --base-model-path {shell_quote(args.base_model)}",
        "python3 evals/spd/export_parity_fixture.py "
        f"--reference-dir {reference_work}/speculative_pipeline_decoding "
        f"--checkpoint {artifact_dir}/speculation_head_final.pt "
        f"--base-model-path {shell_quote(args.base_model)} "
        f"--out {artifact_dir}/spd-parity-fixture.safetensors "
        f"--top-k {max(args.draft_top_k, 8)} --device cuda",
        "target/release/skippy-bench spd-fixture-parity "
        f"--manifest {artifact_dir}/skippy-spd-head.json "
        f"--fixture {artifact_dir}/spd-parity-fixture.safetensors "
        f"--top-k {max(args.draft_top_k, 8)}",
    ]
    smoke = (
        "target/release/skippy-bench spd-openai-smoke "
        "--stage-server-bin target/release/skippy-server "
        f"--manifest {artifact_dir}/skippy-spd-head.json "
        f"--fixture {artifact_dir}/spd-parity-fixture.safetensors "
        f"--model-path {package_dir} --model-id {shell_quote(args.package_ref)} "
        f"--splits {split_arg} --layer-end {layer_end} "
        f"--ctx-size {args.ctx_size} --n-gpu-layers=-1 "
        f"--activation-width {activation_width} --max-tokens 4 "
        f"--prompt-file {prompt_dir}/heldout-prompts.jsonl --prompt-limit {args.heldout_prompts} "
        "--repeat-count 1 --run-baseline true --run-spd true "
        "--spd-rolling-executor --max-inflight 4 "
        f"--output {artifact_dir}/openai-heldout-rolling.json"
    )
    latency = (
        "python3 evals/spd/simulate_latency.py "
        f"--openai-report {artifact_dir}/openai-heldout-rolling.json "
        f"--stage-ms {stage_ms} --hop-ms {shell_quote(args.hop_ms)} "
        f"--json > {artifact_dir}/latency-simulation.json"
    )
    return {
        "setup": setup,
        "build_prompts": [prompt_command],
        "reference_train": [train],
        "export_and_parity": export,
        "package_smoke": [smoke],
        "latency_simulation": [latency],
        "upload": [
            "python3 - <<'PY'\n"
            "from huggingface_hub import HfApi\n"
            f"api = HfApi(); api.create_repo({output_repo!r}, repo_type='model', private=True, exist_ok=True)\n"
            f"api.upload_folder(repo_id={output_repo!r}, repo_type='model', folder_path={artifact_dir!r}, path_in_repo='runs/reference-train')\n"
            "PY"
        ],
        "write_physical_stage_ms": [
            "printf '%s\\n' "
            + shell_quote(",".join(f"{value:g}" for value in resolve_physical_stage_ms(args, resolve_physical_groups(args, args.num_stages))))
            + " > physical-stage-ms.txt"
        ],
    }


def download_warm_start_command(args: argparse.Namespace, out_dir: str) -> str:
    repo = args.warm_start_repo
    path = args.warm_start_path.strip("/")
    return (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "from huggingface_hub import hf_hub_download\n"
        f"repo = {repo!r}; path = {path!r}; out = Path({out_dir!r}); out.mkdir(parents=True, exist_ok=True)\n"
        "for name in ('speculation_head_final.pt', 'skippy-spd-head.json', 'spd-parity-fixture.safetensors'):\n"
        "    src = hf_hub_download(repo_id=repo, repo_type='model', filename=f'{path}/{name}')\n"
        "    target = out / name\n"
        "    target.write_bytes(Path(src).read_bytes())\n"
        "PY"
    )


def download_package_command(args: argparse.Namespace, out_dir: str) -> str:
    return (
        "python3 - <<'PY'\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download(repo_id={args.package_ref!r}, repo_type='model', revision={args.package_revision!r}, local_dir={out_dir!r})\n"
        "PY"
    )


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def emit_human_summary(plan: dict[str, Any]) -> None:
    model = plan["model"]
    topology = plan["topology"]
    fit = plan["physical_fit"]
    hardware = plan["hf_job"]["hardware"]
    print("SPD HF qualification dry run")
    print(f"  model: {model['base_model']}")
    print(f"  package: {model['package_ref']}@{model['package_revision']}")
    print(f"  topology: S{topology['num_stages']} boundaries {topology['stage_layer_boundaries']}")
    print(f"  required taps: {topology['required_hf_hidden_state_indices']}")
    print(f"  physical fit: {fit['logical_stage_groups']} -> stage-ms {fit['physical_stage_ms_for_latency_sim']}")
    print(f"  data: {plan['data']['train_prompts']} train / {plan['data']['heldout_prompts']} heldout prompts")
    print(
        "  hardware: "
        f"{hardware['flavor']} ({hardware['pretty_name']}), "
        f"timeout {hardware['timeout_seconds']}s, max ${hardware['max_cost_usd']:.2f}"
    )
    print(f"  output repo: {plan['hf_job']['output_repo']}")
    print("  dry run: no HF Job submitted; confirmation is still required")
    print()
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
