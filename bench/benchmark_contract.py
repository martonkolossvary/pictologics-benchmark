from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bench.benchmark_ledger import sha256_file
from bench.ibsi_families import GROUP_ORDER


MAIN_TIMING_POLICY = {
    "contract_version": 1,
    "scope": "prepared_family_inputs_to_radiomic_calculations",
    "file_io_included": False,
    "mask_preparation_included": False,
    "resegmentation_included": False,
    "discretization_included": False,
    "result_normalization_or_serialization_included": False,
    "matrix_mesh_neighborhood_construction_included": True,
    "untimed_warmup_calls_per_process": 1,
    "measured_calls_per_process": 3,
    "fresh_process_repeats": 3,
    "within_process_primary_statistic": "median",
    "between_process_primary_statistic": "median",
    "retain_all_raw_samples": True,
    "runtime_normalization": "none",
    # Every adapter uses its native threading.
    "thread_policy": "adapter_native_threads",
}
# The feasibility screen keeps the same calculation boundary with one warmup,
# one timed call and one process repeat.
SCREEN_TIMING_POLICY = {
    **MAIN_TIMING_POLICY,
    "purpose": "feasibility_screen",
    "measured_calls_per_process": 1,
    "fresh_process_repeats": 1,
}
# A separately labelled screen limits the maths libraries to one thread as a
# sensitivity check.
REVIEWED_TIMING_POLICIES = (
    MAIN_TIMING_POLICY,
    SCREEN_TIMING_POLICY,
    {**SCREEN_TIMING_POLICY, "thread_policy": "one_library_thread"},
)


@dataclass(frozen=True)
class BenchmarkContract:
    contract_id: str
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    def expected_feature_count(self, adapter: str, family: str) -> int:
        inventories = self.payload["adapter_inventories"]
        try:
            value = inventories[adapter]["family_output_counts"][family]
        except KeyError as exc:
            raise ValueError(
                f"endpoint contract has no feature count for {adapter}/{family}"
            ) from exc
        return int(value)


def default_contract_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "benchmark" / "main.json"


def default_screen_contract_path() -> Path:
    return default_contract_path().with_name("screen.json")


def one_library_thread_screen_contract_path() -> Path:
    return default_contract_path().with_name("screen_one_library_thread.json")


def load_benchmark_contract(path: str | Path | None = None) -> BenchmarkContract:
    contract_path = Path(path or default_contract_path()).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"benchmark endpoint contract not found: {contract_path}"
        )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("benchmark endpoint contract must use schema version 1")
    if payload.get("aggregation") != "3d_merge":
        raise ValueError("benchmark endpoint contract must require 3d_merge")
    # The benchmark times groups of families in one adapter call.
    family_order = payload.get("family_order")
    if family_order != list(GROUP_ORDER):
        raise ValueError("benchmark endpoint contract family order is not canonical")
    timing = payload.get("timing")
    if not isinstance(timing, dict) or timing not in REVIEWED_TIMING_POLICIES:
        raise ValueError(
            "benchmark endpoint timing contract is not the reviewed policy"
        )
    inventories = payload.get("adapter_inventories")
    if not isinstance(inventories, dict) or not inventories:
        raise ValueError("benchmark endpoint contract has no adapter inventories")
    for adapter, inventory in inventories.items():
        counts = inventory.get("family_output_counts")
        if not isinstance(counts, dict) or set(counts) != set(family_order):
            raise ValueError(f"invalid family count inventory for {adapter}")
        normalized = [int(counts[family]) for family in family_order]
        if any(value < 0 for value in normalized):
            raise ValueError(f"negative family feature count for {adapter}")
        if sum(normalized) != int(inventory.get("native_output_count", -1)):
            raise ValueError(f"native feature count does not sum for {adapter}")
    evidence = payload.get("evidence", {}).get("feature_surface_contract", {})
    evidence_path = Path(__file__).resolve().parents[1] / str(evidence.get("path", ""))
    if not evidence_path.is_file() or sha256_file(evidence_path) != evidence.get(
        "sha256"
    ):
        raise ValueError(
            "benchmark endpoint feature-surface contract is missing or changed"
        )
    contract_id = str(payload.get("contract_id") or "").strip()
    if not contract_id:
        raise ValueError("benchmark endpoint contract_id is missing")
    return BenchmarkContract(
        contract_id=contract_id,
        path=contract_path,
        sha256=sha256_file(contract_path),
        payload=payload,
    )
