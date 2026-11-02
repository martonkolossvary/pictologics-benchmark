from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


RUN_SPEC_SCHEMA_VERSION = 1

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_MEASURED = "measured"
STATUS_FAILED = "failed"
STATUS_TIMED_OUT = "timed_out_censored"
STATUS_INTERRUPTED = "interrupted"
STATUS_UNSUPPORTED = "unsupported"
STATUS_SKIPPED = "skipped_policy"
STATUS_SKIPPED_MEMORY = "skipped_memory_preflight"
STATUS_MEMORY_EXCEEDED = "memory_exceeded_censored"

TERMINAL_STATUSES = frozenset(
    {
        STATUS_MEASURED,
        STATUS_FAILED,
        STATUS_TIMED_OUT,
        STATUS_UNSUPPORTED,
        STATUS_SKIPPED,
        STATUS_SKIPPED_MEMORY,
        STATUS_MEMORY_EXCEEDED,
    }
)
RECOVERABLE_STATUSES = frozenset({STATUS_PENDING, STATUS_INTERRUPTED})
ALL_STATUSES = frozenset(
    {
        STATUS_PENDING,
        STATUS_RUNNING,
        STATUS_MEASURED,
        STATUS_FAILED,
        STATUS_TIMED_OUT,
        STATUS_INTERRUPTED,
        STATUS_UNSUPPORTED,
        STATUS_SKIPPED,
        STATUS_SKIPPED_MEMORY,
        STATUS_MEMORY_EXCEEDED,
    }
)


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for all identities."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _ordered_mapping_items(value: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple((str(key), value[key]) for key in sorted(value))


@dataclass(frozen=True)
class TaskSpec:
    ordinal: int
    case_id: str
    dataset: str
    modality: Optional[str]
    size: int
    variant: int
    mask_id: str
    mask_label: str
    image_path: str
    mask_path: str
    image_sha256: str
    source_image_path: str
    source_image_sha256: str
    mask_sha256: str
    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    image_voxels: int
    mask_voxels: Optional[int]
    mask_fraction: Optional[float]
    complexity: int
    subject_id: str
    input_contract: str
    representation_id: str
    representation_derivation_sha256: Optional[str]
    configured_levels: Optional[int]
    occupied_levels: Optional[int]
    adapter: str
    category: str
    family: str
    repeat: int
    discretization: str
    bins: int
    bin_width: float
    intensity_min: Optional[float]
    intensity_max: Optional[float]
    timing_iterations: int
    no_warmup: bool
    dropped: bool
    endpoint_contract_id: Optional[str] = None
    endpoint_contract_sha256: Optional[str] = None
    expected_feature_count: Optional[int] = None
    input_uncompressed_bytes: Optional[int] = None

    @property
    def workload_key(self) -> str:
        return f"family:{self.family}"

    @property
    def guardrail_group(self) -> str:
        """Return the scientifically comparable stratum for speed truncation."""

        modality = str(self.modality or "").strip().lower()
        if modality == "synthetic":
            return (
                f"subject:{self.subject_id}|mask:{self.mask_id}|"
                f"representation:{self.representation_id}"
            )
        if modality:
            # A fixed real-world cohort is not a monotone scaling ladder:
            # larger arrays from different patients can have smaller ROIs,
            # bounding boxes, or grey-level workloads.  Keep every case in its
            # own scope so a slow/timeout result never censors another patient.
            return f"case:{self.case_id}|modality:{modality}"
        return f"case:{self.case_id}|dataset:{self.dataset}"

    @property
    def task_id(self) -> str:
        identity = {
            "case_id": self.case_id,
            "modality": self.modality,
            "image_sha256": self.image_sha256,
            "source_image_sha256": self.source_image_sha256,
            "mask_sha256": self.mask_sha256,
            "input_contract": self.input_contract,
            "representation_id": self.representation_id,
            "representation_derivation_sha256": self.representation_derivation_sha256,
            "configured_levels": self.configured_levels,
            "occupied_levels": self.occupied_levels,
            "adapter": self.adapter,
            "workload": self.workload_key,
            "repeat": self.repeat,
            "discretization": self.discretization,
            "bins": self.bins,
            "bin_width": self.bin_width,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "timing_iterations": self.timing_iterations,
            "no_warmup": self.no_warmup,
            "endpoint_contract_id": self.endpoint_contract_id,
            "endpoint_contract_sha256": self.endpoint_contract_sha256,
            "expected_feature_count": self.expected_feature_count,
            "input_uncompressed_bytes": self.input_uncompressed_bytes,
        }
        return fingerprint(identity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "ordinal": self.ordinal,
            "case_id": self.case_id,
            "dataset": self.dataset,
            "modality": self.modality,
            "size": self.size,
            "variant": self.variant,
            "mask_id": self.mask_id,
            "mask_label": self.mask_label,
            "image_path": self.image_path,
            "source_image_path": self.source_image_path,
            "mask_path": self.mask_path,
            "image_sha256": self.image_sha256,
            "source_image_sha256": self.source_image_sha256,
            "mask_sha256": self.mask_sha256,
            "shape": list(self.shape),
            "spacing": list(self.spacing),
            "image_voxels": self.image_voxels,
            "mask_voxels": self.mask_voxels,
            "mask_fraction": self.mask_fraction,
            "complexity": self.complexity,
            "subject_id": self.subject_id,
            "input_contract": self.input_contract,
            "representation_id": self.representation_id,
            "representation_derivation_sha256": self.representation_derivation_sha256,
            "configured_levels": self.configured_levels,
            "occupied_levels": self.occupied_levels,
            "adapter": self.adapter,
            "category": self.category,
            "family": self.family,
            "workload_key": self.workload_key,
            "guardrail_group": self.guardrail_group,
            "repeat": self.repeat,
            "discretization": self.discretization,
            "bins": self.bins,
            "bin_width": self.bin_width,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "timing_iterations": self.timing_iterations,
            "no_warmup": self.no_warmup,
            "dropped": self.dropped,
            "endpoint_contract_id": self.endpoint_contract_id,
            "endpoint_contract_sha256": self.endpoint_contract_sha256,
            "expected_feature_count": self.expected_feature_count,
            "input_uncompressed_bytes": self.input_uncompressed_bytes,
        }


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset: str
    dataset_kind: str
    dataset_manifest_schema_version: int
    dataset_dir: str
    manifest_sha256: str
    dataset_hashes_verified: bool
    dataset_values_inspected: bool
    selected_case_ids: Tuple[str, ...]
    adapters: Tuple[str, ...]
    workloads: Tuple[str, ...]
    repeats: int
    aggregation: str
    input_contract: str
    discretization: str
    bins: int
    bin_width: float
    intensity_min: Optional[float]
    intensity_max: Optional[float]
    timing_iterations: int
    capture_values: bool
    drop_first_repeat: bool
    no_warmup: bool
    timeout_seconds: Optional[float]
    keep_going: bool
    task_plan_sha256: str
    runtime_profiles_sha256: Optional[str]
    benchmark_sources_sha256: str
    benchmark_machine: Tuple[Tuple[str, Any], ...]
    adapter_environments: Tuple[Tuple[str, Any], ...]
    thread_policy: Tuple[Tuple[str, Any], ...]
    initialization_policy: Tuple[Tuple[str, Any], ...]
    guardrail: Tuple[Tuple[str, Any], ...]
    endpoint_contract_id: Optional[str] = None
    endpoint_contract_path: Optional[str] = None
    endpoint_contract_sha256: Optional[str] = None
    memory_preflight: Tuple[Tuple[str, Any], ...] = ()
    schema_version: int = RUN_SPEC_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        benchmark_machine: Mapping[str, Any],
        adapter_environments: Mapping[str, Any],
        thread_policy: Mapping[str, Any],
        initialization_policy: Mapping[str, Any],
        guardrail: Mapping[str, Any],
        memory_preflight: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "RunSpec":
        return cls(
            benchmark_machine=_ordered_mapping_items(benchmark_machine),
            adapter_environments=_ordered_mapping_items(adapter_environments),
            thread_policy=_ordered_mapping_items(thread_policy),
            initialization_policy=_ordered_mapping_items(initialization_policy),
            guardrail=_ordered_mapping_items(guardrail),
            memory_preflight=_ordered_mapping_items(memory_preflight or {}),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "dataset_kind": self.dataset_kind,
            "dataset_manifest_schema_version": self.dataset_manifest_schema_version,
            "dataset_dir": self.dataset_dir,
            "manifest_sha256": self.manifest_sha256,
            "dataset_hashes_verified": self.dataset_hashes_verified,
            "dataset_values_inspected": self.dataset_values_inspected,
            "selected_case_ids": list(self.selected_case_ids),
            "adapters": list(self.adapters),
            "workloads": list(self.workloads),
            "repeats": self.repeats,
            "aggregation": self.aggregation,
            "input_contract": self.input_contract,
            "discretization": self.discretization,
            "bins": self.bins,
            "bin_width": self.bin_width,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "timing_iterations": self.timing_iterations,
            "capture_values": self.capture_values,
            "drop_first_repeat": self.drop_first_repeat,
            "no_warmup": self.no_warmup,
            "timeout_seconds": self.timeout_seconds,
            "keep_going": self.keep_going,
            "task_plan_sha256": self.task_plan_sha256,
            "runtime_profiles_sha256": self.runtime_profiles_sha256,
            "benchmark_sources_sha256": self.benchmark_sources_sha256,
            "benchmark_machine": dict(self.benchmark_machine),
            "adapter_environments": dict(self.adapter_environments),
            "thread_policy": dict(self.thread_policy),
            "initialization_policy": dict(self.initialization_policy),
            "guardrail": dict(self.guardrail),
            "memory_preflight": dict(self.memory_preflight),
            "endpoint_contract_id": self.endpoint_contract_id,
            "endpoint_contract_path": self.endpoint_contract_path,
            "endpoint_contract_sha256": self.endpoint_contract_sha256,
        }

    @property
    def run_fingerprint(self) -> str:
        return fingerprint(run_spec_identity(self.to_dict()))


def run_spec_identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the immutable protocol identity, excluding append-only horizon fields."""

    identity = dict(value)
    identity.pop("repeats", None)
    identity.pop("task_plan_sha256", None)
    return identity


def task_plan_fingerprint(tasks: Iterable[TaskSpec]) -> str:
    return fingerprint([task.to_dict() for task in tasks])
