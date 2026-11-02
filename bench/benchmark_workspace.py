"""Build and attest the three-pillar benchmark input workspace.

This module never imports an adapter and never calculates radiomic features.
It serially prepares the two synthetic pillars and the fixed IBSI 2 Phase 3
cohort, validates every committed representation, and writes one immutable
workspace manifest used by later timing runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from bench.adapters.registry import get_adapter
from bench.benchmark_contract import (
    BenchmarkContract,
    default_screen_contract_path,
    load_benchmark_contract,
    one_library_thread_screen_contract_path,
)
from bench.benchmark_ledger import sha256_file
from bench.dataset_manifest import DatasetValidationError, atomic_write_json
from bench.ibsi2_phase3_dataset import (
    prepare_ibsi2_phase3_dataset,
    validate_ibsi2_phase3_dataset,
)
from bench.pillar1_dataset import build_pillar1_dataset, validate_pillar1_dataset
from bench.pillar2_dataset import build_pillar2_dataset, validate_pillar2_dataset


WORKSPACE_ID = "pictologics_three_pillar_benchmark_inputs"
ADAPTERS = ("pictologics", "pyradiomics", "mirp", "medimage", "zrad")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DatasetValidationError(
            f"benchmark input must be inside the workspace root: {path}"
        ) from exc


def _dataset_record(
    root: Path,
    dataset_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": _relative(root, dataset_dir),
        "dataset": manifest["dataset"],
        "dataset_kind": manifest["dataset_kind"],
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "case_count": len(manifest["cases"]),
        "file_count": len(manifest["files"]),
        "validation": dict(validation),
    }


def _task_inventory(
    datasets: Mapping[str, Mapping[str, Any]],
    contract: BenchmarkContract,
) -> dict[str, Any]:
    repeats = int(contract.payload["timing"]["fresh_process_repeats"])
    measured_calls = int(contract.payload["timing"]["measured_calls_per_process"])
    warmups = int(contract.payload["timing"]["untimed_warmup_calls_per_process"])
    families = list(contract.payload["family_order"])
    scheduled_per_case = len(ADAPTERS) * len(families) * repeats
    supported_families = {
        adapter: [
            family
            for family in families
            if contract.expected_feature_count(adapter, family) > 0
        ]
        for adapter in ADAPTERS
    }
    eligible_per_case = (
        sum(len(values) for values in supported_families.values()) * repeats
    )
    rows: dict[str, Any] = {}
    total_cases = 0
    for pillar, dataset in datasets.items():
        cases = int(dataset["case_count"])
        total_cases += cases
        scheduled = cases * scheduled_per_case
        eligible = cases * eligible_per_case
        rows[pillar] = {
            "case_count": cases,
            "scheduled_task_records": scheduled,
            "eligible_calculation_tasks": eligible,
            "preempted_unsupported_tasks": scheduled - eligible,
            "discarded_warmup_calls_if_all_eligible_complete": eligible * warmups,
            "measured_calls_if_all_eligible_complete": eligible * measured_calls,
        }
    scheduled = total_cases * scheduled_per_case
    eligible = total_cases * eligible_per_case
    return {
        "family_count": len(families),
        "adapter_count": len(ADAPTERS),
        "fresh_process_repeats": repeats,
        "measured_calls_per_process": measured_calls,
        "untimed_warmup_calls_per_process": warmups,
        "supported_families_by_adapter": supported_families,
        "pillars": rows,
        "totals": {
            "case_count": total_cases,
            "scheduled_task_records": scheduled,
            "eligible_calculation_tasks": eligible,
            "preempted_unsupported_tasks": scheduled - eligible,
            "discarded_warmup_calls_if_all_eligible_complete": eligible * warmups,
            "measured_calls_if_all_eligible_complete": eligible * measured_calls,
        },
    }


def _contract_record(contract: BenchmarkContract) -> dict[str, Any]:
    return {
        "contract_id": contract.contract_id,
        "path": contract.path.relative_to(
            Path(__file__).resolve().parents[1]
        ).as_posix(),
        "sha256": contract.sha256,
    }


def write_workspace_manifest(
    root: Path,
    *,
    endpoint_contract_path: Path | None = None,
    deep: bool = True,
) -> dict[str, Any]:
    """Deep-validate all pillars and atomically bind the release workspace."""

    root = root.expanduser().resolve()
    pillar1 = root / "pillar1"
    pillar2 = root / "pillar2_a1"
    pillar3 = root / "ibsi2_phase3"
    contract = load_benchmark_contract(endpoint_contract_path)
    screen_contract = load_benchmark_contract(default_screen_contract_path())
    validations = {
        "pillar1_morphology": validate_pillar1_dataset(pillar1, deep=deep),
        "pillar2_whole_anatomy": validate_pillar2_dataset(pillar2),
        "pillar3_ibsi2_phase3": validate_ibsi2_phase3_dataset(pillar3),
    }
    datasets = {
        "pillar1_morphology": _dataset_record(
            root, pillar1, validations["pillar1_morphology"]
        ),
        "pillar2_whole_anatomy": _dataset_record(
            root, pillar2, validations["pillar2_whole_anatomy"]
        ),
        "pillar3_ibsi2_phase3": _dataset_record(
            root, pillar3, validations["pillar3_ibsi2_phase3"]
        ),
    }
    repository = Path(__file__).resolve().parents[1]
    workspace_sources = [
        repository / "bench" / "benchmark_workspace.py",
        repository / "scripts" / "prepare_workspace.py",
        repository / "scripts" / "launch_benchmark.py",
        repository / "poetry.lock",
    ]
    record = {
        "schema_version": 1,
        "workspace_id": WORKSPACE_ID,
        "status": "inputs_validated_calculations_not_started",
        "benchmark_timing_executed": False,
        "adapter_order": list(ADAPTERS),
        "endpoint_contract": _contract_record(contract),
        "screen_contract": _contract_record(screen_contract),
        # The same screen with one maths-library thread, as a sensitivity
        # check on the native-thread rule; it schedules the same tasks.
        "one_library_thread_screen_contract": _contract_record(
            load_benchmark_contract(one_library_thread_screen_contract_path())
        ),
        "workspace_sources": [
            {
                "path": path.relative_to(repository).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in workspace_sources
        ],
        "datasets": datasets,
        "task_inventory": _task_inventory(datasets, contract),
        "screen_task_inventory": _task_inventory(datasets, screen_contract),
        # The screen runs one warmup and one timed call under a shorter
        # per-call limit to map timeouts before the main run.
        "screen_launch_policy": {"per_call_timeout_seconds": 600.0},
        "launch_policy": {
            "aggregation": "3d_merge",
            "family_tasks_are_independent": True,
            "runtime_normalization": "none",
            "warmup_is_untimed": True,
            "within_process_primary_statistic": "median",
            "between_process_primary_statistic": "median",
            "per_call_timeout_seconds": 1800.0,
            "checkpoint_interval_tasks": 25,
            "progress_interval_seconds": 30.0,
            "speed_guardrail_baseline": "pictologics",
            "speed_ratio_rule": "disabled",
            "stop_truncation": (
                "call_timeouts_memory_stops_and_second_startup_timeouts_"
                "skip_this_and_larger_sizes"
            ),
            "memory_preflight": "total_memory_static_plus_largest_input",
            "unsupported_is_not_zero": True,
            "censored_is_not_zero": True,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "workspace_manifest.json", record)
    return record


def prepare_workspace(
    root: Path,
    *,
    ibsi2_source: Path,
    pillar1_profile: Path,
    pillar2_profile: Path,
    endpoint_contract_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Serially prepare all inputs; no adapter process is invoked."""

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    build_pillar1_dataset(root / "pillar1", profile_path=pillar1_profile, resume=resume)
    build_pillar2_dataset(
        root / "pillar2_a1", profile_path=pillar2_profile, resume=resume
    )
    prepare_ibsi2_phase3_dataset(
        ibsi2_source,
        root / "ibsi2_phase3",
        expected_subjects=51,
        resume=resume,
    )
    return write_workspace_manifest(
        root, endpoint_contract_path=endpoint_contract_path, deep=True
    )


def validate_workspace(root: Path, *, deep: bool = True) -> dict[str, Any]:
    """Recreate the manifest and require its calculation-free status."""

    record = write_workspace_manifest(root, deep=deep)
    for adapter in ADAPTERS:
        get_adapter(adapter)
    return record


__all__ = [
    "ADAPTERS",
    "WORKSPACE_ID",
    "prepare_workspace",
    "validate_workspace",
    "write_workspace_manifest",
]
