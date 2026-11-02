"""Resumable IBSI compliance execution, kept separate from timing benchmarks."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from bench import env as benchmark_env
from bench.adapters.protocol import (
    ADAPTER_PROTOCOL_VERSION,
    REQUIRED_AGGREGATION,
    resolve_aggregation,
    supports_aggregation,
)
from bench.adapters.registry import get_adapter
from bench.benchmark_ledger import (
    RunIntegrityError,
    RunLock,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from bench.benchmark_models import fingerprint
from bench.compliance.evaluate import evaluate_adapter_payload
from bench.compliance.ibsi2_protocol import (
    IBSI2_MANUAL_VERSION,
    IBSI2_PHASE2_BOUNDARY_POLICY,
    IBSI2_PROTOCOL_REVIEW,
    PHASE1_FILTER_SPECS_BY_ID,
    validate_phase1_filter_config as validate_exact_phase1_filter_config,
    validate_phase2_filter_config as validate_exact_phase2_filter_config,
)
from bench.compliance.ibsi2_native_backends import normalize_parameters
from bench.compliance.models import ComparisonRecord
from bench.compliance.references import (
    IBSI_DATA_COMMIT,
    IBSI_DATA_REPOSITORY,
    IBSI1_DIGITAL_PHANTOM_IMAGE_SHA256,
    IBSI1_DIGITAL_PHANTOM_MASK_SHA256,
    IBSI2_PHASE1_SOURCE_IMAGE_SHA256,
    IBSI2_PHASE1_SOURCE_MASK_SHA256,
    IBSI2_PHASE1_TEST_IDS,
    IBSI2_PHASE2_FILTER_IDS,
    IBSI2_PHASE2_SOURCE_IMAGE_SHA256,
    IBSI2_PHASE2_SOURCE_MASK_SHA256,
    load_reference_csv,
    select_ibsi1_digital_phantom_profile,
    validate_reference_table_manifest,
)
from bench.compliance.report import comparison_to_csv, generate_compliance_report
from bench.ibsi_families import GROUP_ORDER, MORPHOLOGY_SPATIAL_CODES, WORKLOAD_GROUPS
from bench.run import AdapterInterrupted, run_adapter_process


COMPLIANCE_RUN_SCHEMA_VERSION = 1
IBSI2_CANDIDATE_SCHEMA_VERSION = 1


def _validate_official_ibsi1_digital_phantom(image: Path, mask: Path) -> None:
    """Fail closed unless both inputs are the pinned official IBSI 1 phantom."""

    image_digest = sha256_file(image)
    mask_digest = sha256_file(mask)
    if image_digest != IBSI1_DIGITAL_PHANTOM_IMAGE_SHA256:
        raise ValueError(
            "IBSI 1 digital phantom image SHA-256 mismatch: "
            f"{image_digest} != {IBSI1_DIGITAL_PHANTOM_IMAGE_SHA256}"
        )
    if mask_digest != IBSI1_DIGITAL_PHANTOM_MASK_SHA256:
        raise ValueError(
            "IBSI 1 digital phantom mask SHA-256 mismatch: "
            f"{mask_digest} != {IBSI1_DIGITAL_PHANTOM_MASK_SHA256}"
        )


class IBSI2CandidateEntries(list[dict[str, Any]]):
    """Validated response-map entries plus their exhaustive native-support grid."""

    def __init__(
        self,
        entries: Iterable[dict[str, Any]],
        *,
        adapters: Sequence[str],
        support_declarations: Mapping[tuple[str, str], Mapping[str, Any]],
        support_class: str = "native",
    ) -> None:
        super().__init__(entries)
        self.adapters = tuple(adapters)
        self.support_declarations = {
            key: dict(value) for key, value in support_declarations.items()
        }
        self.support_class = support_class


# Exact native support, or the separate class that runs a filter with the
# package's own boundary or truncation when its API cannot set them.
IBSI2_SUPPORT_CLASSES = ("native", "native_default")


def _manifest_support_class(manifest: Mapping[str, Any]) -> str:
    """Return the one support class of a candidate manifest and all its entries."""

    support_class = manifest.get("support_class", "native")
    if support_class not in IBSI2_SUPPORT_CLASSES:
        raise ValueError(f"Unknown IBSI 2 support class: {support_class!r}")
    entries = manifest.get("entries")
    for entry in entries if isinstance(entries, list) else ():
        if isinstance(entry, Mapping) and (
            entry.get("support_class", "native") != support_class
        ):
            raise ValueError(
                "An IBSI 2 candidate manifest never mixes native and native-default maps"
            )
    return support_class


def ibsi2_execution_contracts(
    entries: Iterable[Mapping[str, Any]],
    *,
    id_field: str,
) -> list[dict[str, Any]]:
    """Return portable structured filter/boundary provenance for result reports."""

    if id_field not in {"test_id", "filter_id"}:
        raise ValueError("id_field must be test_id or filter_id")
    output: list[dict[str, Any]] = []
    for entry in entries:
        parameters = entry.get("executed_parameters")
        boundary = entry.get("boundary_execution")
        if not isinstance(parameters, Mapping) or not isinstance(boundary, Mapping):
            raise ValueError("Validated IBSI 2 entry lacks execution provenance")
        identifier = str(entry[id_field])
        filter_name = str(parameters.get("filter", ""))
        official_dimensionality = parameters.get("dimensionality")
        if id_field == "filter_id":
            workflow_dimensionality = 2 if identifier.upper().endswith(".A") else 3
        else:
            workflow_dimensionality = official_dimensionality
        kernel_dimensionality = (
            None
            if filter_name == "none"
            else 2
            if filter_name == "gabor"
            else official_dimensionality
        )
        output.append(
            {
                "adapter": str(entry["adapter"]),
                id_field: identifier,
                "filter": filter_name,
                "workflow_dimensionality": workflow_dimensionality,
                "kernel_dimensionality": kernel_dimensionality,
                "orthogonal_plane_averaging": bool(
                    parameters.get("average_over_planes", False)
                ),
                "executed_parameters": dict(parameters),
                "boundary_execution": dict(boundary),
                "native_capability": (
                    dict(entry["native_capability"])
                    if isinstance(entry.get("native_capability"), Mapping)
                    else None
                ),
            }
        )
    return sorted(output, key=lambda row: (row["adapter"], row[id_field]))


def _validate_native_support_grid(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    id_field: str,
    expected_ids: Sequence[str],
) -> tuple[tuple[str, ...], dict[tuple[str, str], dict[str, Any]]]:
    """Require one reviewed native-support decision for every adapter/test pair."""

    raw_adapters = manifest.get("adapters")
    if (
        not isinstance(raw_adapters, list)
        or not raw_adapters
        or any(
            not isinstance(adapter, str) or not adapter.strip()
            for adapter in raw_adapters
        )
    ):
        raise ValueError(
            f"IBSI 2 {phase} candidate manifest requires a non-empty adapters list"
        )
    adapters = tuple(adapter.strip() for adapter in raw_adapters)
    if len(adapters) != len(set(adapters)):
        raise ValueError(f"IBSI 2 {phase} candidate manifest adapters must be unique")
    for adapter in adapters:
        get_adapter(adapter)

    declarations = manifest.get("support_declarations")
    if not isinstance(declarations, list):
        raise ValueError(
            f"IBSI 2 {phase} candidate manifest requires support_declarations"
        )
    expected = {(adapter, test_id) for adapter in adapters for test_id in expected_ids}
    output: dict[tuple[str, str], dict[str, Any]] = {}
    required = {"adapter", id_field, "native_supported", "reason", "evidence"}
    expected_id_set = set(expected_ids)
    for declaration in declarations:
        if not isinstance(declaration, Mapping) or not required.issubset(declaration):
            raise ValueError(
                f"IBSI 2 {phase} support declaration lacks required fields: "
                f"{declaration!r}"
            )
        adapter = str(declaration["adapter"]).strip()
        identifier = str(declaration[id_field]).strip()
        identifier = (
            identifier.casefold() if id_field == "test_id" else identifier.upper()
        )
        if adapter not in adapters or identifier not in expected_id_set:
            raise ValueError(
                f"IBSI 2 {phase} support declaration has an unknown adapter/{id_field}: "
                f"{adapter!r}, {identifier!r}"
            )
        if type(declaration["native_supported"]) is not bool:
            raise ValueError(
                f"IBSI 2 {phase} support declaration native_supported must be boolean"
            )
        key = (adapter, identifier)
        if key in output:
            raise ValueError(f"Duplicate IBSI 2 {phase} support declaration: {key}")
        reason = _require_concrete_provenance(
            declaration["reason"], field="reason", context=f"Support declaration {key}"
        )
        evidence = _require_concrete_provenance(
            declaration["evidence"],
            field="evidence",
            context=f"Support declaration {key}",
        )
        output[key] = {
            "adapter": adapter,
            id_field: identifier,
            "native_supported": declaration["native_supported"],
            "reason": reason,
            "evidence": evidence,
        }
    missing = sorted(expected.difference(output))
    extra = sorted(set(output).difference(expected))
    if missing or extra:
        raise ValueError(
            f"IBSI 2 {phase} support declarations must be the exact "
            f"adapter x {len(expected_ids)} grid; missing={missing}, extra={extra}"
        )
    return adapters, output


def _require_requested_adapters(
    requested: Sequence[str],
    declared: Sequence[str],
    *,
    phase: str,
) -> None:
    if tuple(requested) != tuple(declared):
        raise ValueError(
            f"IBSI 2 {phase} requested adapters must exactly match the candidate "
            f"manifest adapters in the same order: {list(declared)!r}"
        )


def _runtime_profile(adapter: str):
    """Return the reviewed runtime profile matching one built-in adapter."""

    capabilities = get_adapter(adapter)
    profiles = benchmark_env.load_runtime_profiles()
    profile = profiles.get(adapter)
    if profile is None:
        raise RuntimeError(f"Missing runtime profile for compliance adapter: {adapter}")
    if profile.distribution.casefold() != capabilities.distribution.casefold():
        raise RunIntegrityError(
            f"Runtime profile distribution mismatch for {adapter}: "
            f"{profile.distribution!r} != {capabilities.distribution!r}"
        )
    return profile


def configured_adapter_profiles(adapters: Sequence[str]) -> dict[str, dict[str, str]]:
    """Return the exact reviewed distribution/version expected for report inputs."""

    output: dict[str, dict[str, str]] = {}
    for adapter in dict.fromkeys(adapters):
        profile = _runtime_profile(adapter)
        output[adapter] = {
            "distribution": profile.distribution,
            "version": profile.version,
            "distribution_metadata_version": (
                profile.metadata_version or profile.version
            ),
            "python": profile.python,
        }
    return output


def _require_concrete_provenance(value: Any, *, field: str, context: str) -> str:
    token = str(value).strip()
    folded = token.casefold()
    if (
        not token
        or folded in {"unknown", "none", "n/a", "na", "replace-me", "placeholder"}
        or token.startswith("<")
        or token.endswith(">")
    ):
        raise ValueError(f"{context} requires a concrete {field}")
    return token


def _json_safe_payload(value: Any) -> Any:
    """Preserve non-finite observations as explicit JSON strings for row-level audit."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_payload(item) for item in value]
    return value


def _verified_environment_snapshots(adapters: Sequence[str]) -> dict[str, Any]:
    """Verify exact pins/imports/dependencies before creating immutable run state."""

    profiles = benchmark_env.load_runtime_profiles()
    missing_profiles = sorted(set(adapters).difference(profiles))
    if missing_profiles:
        raise RuntimeError(
            "Missing runtime profiles for compliance adapters: "
            + ", ".join(missing_profiles)
        )
    return {
        adapter: benchmark_env.verify_profile(profiles[adapter], smoke=True)
        for adapter in adapters
    }


def _controller_source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    return fingerprint(sources)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RunIntegrityError(f"Expected a JSON object: {path}")
    return value


def _verified_relative_file(
    *,
    root: Path,
    relative_path: Any,
    expected_sha256: Any,
    context: str,
) -> Path:
    """Resolve a manifest file without escape and bind it to a SHA-256 value."""

    relative = str(relative_path).strip()
    digest = str(expected_sha256).strip().casefold()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{context} requires a 64-character SHA-256 digest")
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"{context} path must remain inside the manifest directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != digest:
        raise RunIntegrityError(f"{context} checksum mismatch")
    return path


def _validated_generator_provenance(
    entry: Mapping[str, Any],
    *,
    root: Path,
    context: str,
) -> dict[str, Path]:
    """Validate code/config/environment evidence attached to one response map."""

    for field in (
        "generator_source_revision",
        "generator_entrypoint",
        "generator_command",
        "filter_config_revision",
    ):
        _require_concrete_provenance(entry.get(field), field=field, context=context)
    return {
        "generator_source": _verified_relative_file(
            root=root,
            relative_path=entry.get("generator_source_path"),
            expected_sha256=entry.get("generator_source_sha256"),
            context=f"{context} generator source",
        ),
        "filter_config": _verified_relative_file(
            root=root,
            relative_path=entry.get("filter_config_path"),
            expected_sha256=entry.get("filter_config_sha256"),
            context=f"{context} filter configuration",
        ),
        "environment_lock": _verified_relative_file(
            root=root,
            relative_path=entry.get("environment_lock_path"),
            expected_sha256=entry.get("environment_lock_sha256"),
            context=f"{context} environment lock",
        ),
    }


def _validate_protocol_review(
    manifest: Mapping[str, Any], *, phase: str
) -> dict[str, str]:
    review = manifest.get("protocol_review")
    if not isinstance(review, Mapping) or review.get("status") != "reviewed":
        raise ValueError(
            f"IBSI 2 {phase} candidate manifest requires a completed protocol_review"
        )
    output: dict[str, str] = {"status": "reviewed"}
    for field in ("reviewed_against", "reviewed_by", "reviewed_at"):
        output[field] = _require_concrete_provenance(
            review.get(field), field=field, context=f"IBSI 2 {phase} protocol review"
        )
    if output["reviewed_against"] != IBSI2_PROTOCOL_REVIEW:
        raise ValueError(
            f"IBSI 2 {phase} protocol review must be bound to {IBSI2_PROTOCOL_REVIEW!r}"
        )
    return output


def validate_ibsi2_phase2_filter_config(path: Path, *, filter_id: str) -> None:
    """Require a reviewable filter definition tied to the advertised test ID."""

    config = _read_json(path)
    parameters = config.get("parameters")
    if (
        config.get("schema_version") != 1
        or config.get("specification") != "IBSI 2"
        or config.get("phase") != "phase2"
        or str(config.get("filter_id", "")).upper() != filter_id
        or str(config.get("reference_manual_version", "")) != IBSI2_MANUAL_VERSION
        or config.get("boundary_policy") != IBSI2_PHASE2_BOUNDARY_POLICY
        or not isinstance(parameters, Mapping)
        or not parameters
    ):
        raise ValueError(
            f"IBSI 2 Phase 2 {filter_id} filter configuration is incomplete or mismatched"
        )
    validate_exact_phase2_filter_config(config, filter_id=filter_id)


def validate_ibsi2_phase1_candidate_configs(
    filter_path: Path,
    preprocessing_path: Path,
    *,
    test_id: str,
    source_image_sha256: str,
) -> None:
    """Bind Phase 1 configuration content to its test and official phantom."""

    filter_config = _read_json(filter_path)
    preprocessing_config = _read_json(preprocessing_path)
    common = {
        "schema_version": 1,
        "specification": "IBSI 2",
        "phase": "phase1",
        "reference_manual_version": IBSI2_MANUAL_VERSION,
        "test_id": test_id,
    }
    if any(filter_config.get(key) != value for key, value in common.items()):
        raise ValueError(f"IBSI 2 Phase 1 {test_id} filter configuration is mismatched")
    if (
        filter_config.get("source_image_sha256") != source_image_sha256
        or not isinstance(filter_config.get("parameters"), Mapping)
        or not filter_config["parameters"]
    ):
        raise ValueError(
            f"IBSI 2 Phase 1 {test_id} filter configuration lacks source/parameter evidence"
        )
    specification = PHASE1_FILTER_SPECS_BY_ID[test_id]
    if source_image_sha256 != specification.source_image_sha256:
        raise ValueError(
            f"IBSI 2 Phase 1 {test_id} uses the wrong official source phantom"
        )
    validate_exact_phase1_filter_config(filter_config, test_id=test_id)
    if any(preprocessing_config.get(key) != value for key, value in common.items()):
        raise ValueError(
            f"IBSI 2 Phase 1 {test_id} preprocessing configuration is mismatched"
        )
    if (
        not isinstance(preprocessing_config.get("settings"), Mapping)
        or not preprocessing_config["settings"]
    ):
        raise ValueError(
            f"IBSI 2 Phase 1 {test_id} preprocessing configuration lacks settings"
        )
    if preprocessing_config != specification.preprocessing_config():
        raise ValueError(
            f"IBSI 2 Phase 1 {test_id} preprocessing differs from the reviewed protocol"
        )


def _validate_environment_lock(
    path: Path,
    *,
    adapter: str,
    profile: Any,
) -> None:
    """Bind a candidate lock to the exact locally verified adapter environment."""

    expected = benchmark_env.env_dir_for_profile(profile) / "environment.json"
    if not expected.is_file():
        raise FileNotFoundError(expected)
    if sha256_file(path) != sha256_file(expected):
        raise RunIntegrityError(
            f"Candidate {adapter} environment lock differs from the verified runtime"
        )


def validate_ibsi2_phase2_preprocessing_config(path: Path, *, filter_id: str) -> None:
    """Validate the exact protocol-level A/B preprocessing contract."""

    config = _read_json(path)
    dimension = filter_id.rsplit(".", 1)[-1]
    expected_common = {
        "schema_version": 1,
        "specification": "IBSI 2",
        "phase": "phase2",
        "reference_manual_version": IBSI2_MANUAL_VERSION,
        "configuration_dimension": dimension,
        "crop": False,
        "intensity_resegmentation_range_hu": [-1000, 400],
        "intensity_resegmentation_source": "unfiltered_image",
        "response_map_discretization": "none",
        "statistics_roi": "complete_3d",
    }
    for key, expected in expected_common.items():
        if config.get(key) != expected:
            raise ValueError(
                f"IBSI 2 Phase 2 {filter_id} preprocessing mismatch for {key}: "
                f"{config.get(key)!r} != {expected!r}"
            )
    if dimension == "A":
        if config.get("resampling") != {"enabled": False}:
            raise ValueError(
                f"IBSI 2 Phase 2 {filter_id} A preprocessing must disable resampling"
            )
        return
    expected_resampling = {
        "enabled": True,
        "spacing_mm": [1.0, 1.0, 1.0],
        "image_interpolation": "tricubic",
        "mask_interpolation": "trilinear",
        "mask_threshold": 0.5,
        "intensity_rounding": "nearest_integer",
    }
    if config.get("resampling") != expected_resampling:
        raise ValueError(
            f"IBSI 2 Phase 2 {filter_id} B preprocessing does not match the reviewed protocol"
        )


def _validate_ibsi2_execution_contract(
    entry: Mapping[str, Any],
    *,
    adapter: str,
    filter_config_path: Path,
    context: str,
    support_class: str = "native",
) -> None:
    """Bind recorded executed parameters to the reviewed package-neutral config.

    A native-default entry must name the boundary or truncation controls that
    its package could not set, and the boundary that the package used.
    """

    filter_config = _read_json(filter_config_path)
    raw_parameters = filter_config.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise ValueError(f"{context} filter configuration lacks parameters")
    expected_parameters = normalize_parameters(raw_parameters, adapter=adapter)
    native_default = entry.get("native_default")
    if support_class == "native_default":
        if not isinstance(native_default, Mapping) or not native_default.get(
            "uncontrolled"
        ):
            raise ValueError(
                f"{context} native-default run lacks its uncontrolled controls"
            )
        expected_parameters = {**expected_parameters, "native_default": True}
    elif native_default is not None:
        raise ValueError(f"{context} exact native entry has a native-default record")
    executed_parameters = entry.get("executed_parameters")
    if (
        not isinstance(executed_parameters, Mapping)
        or dict(executed_parameters) != expected_parameters
    ):
        raise ValueError(
            f"{context} executed parameters do not match the reviewed native contract"
        )
    execution = entry.get("boundary_execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"{context} lacks boundary execution provenance")
    expected_policy = (
        "not_applicable"
        if expected_parameters["filter"] == "none"
        else (
            "protocol_explicit"
            if "boundary" in raw_parameters
            else IBSI2_PHASE2_BOUNDARY_POLICY
        )
    )
    if execution.get("policy") != expected_policy:
        raise ValueError(
            f"{context} boundary policy differs from the reviewed protocol"
        )
    expected_selected = expected_parameters.get("boundary")
    if execution.get("selected") != expected_selected:
        raise ValueError(f"{context} selected boundary is inconsistent")
    expected_effective = (
        None
        if expected_parameters["filter"] == "none"
        else native_default["effective_boundary"]
        if support_class == "native_default"
        else expected_parameters["boundary"]
    )
    if execution.get("effective") != expected_effective:
        raise ValueError(f"{context} effective boundary is inconsistent")
    _require_concrete_provenance(
        execution.get("implementation"),
        field="boundary implementation",
        context=context,
    )
    capability = entry.get("native_capability")
    if adapter != "pictologics" or expected_parameters["filter"] == "none":
        if capability is not None:
            raise ValueError(f"{context} has an unexpected native capability record")
        return
    if not isinstance(capability, Mapping):
        raise ValueError(f"{context} lacks Pictologics capability provenance")
    # Pictologics 1.0.0 writes capability schema 1.1.0, which adds the
    # Gaussian filter; the fields of the other filters are those of 1.0.0.
    if (
        capability.get("schema_version") != "1.1.0"
        or capability.get("filter") != expected_parameters["filter"]
    ):
        raise ValueError(f"{context} Pictologics capability identity is inconsistent")
    expected_kernel_dimensionality = (
        2 if expected_parameters["filter"] == "gabor" else 3
    )
    expected_plane_execution = expected_parameters["filter"] == "gabor"
    if (
        capability.get("input_dimensionality") != [3]
        or capability.get("kernel_dimensionality") != expected_kernel_dimensionality
        or capability.get("slice_plane_execution") is not expected_plane_execution
        or capability.get("orthogonal_plane_averaging") is not expected_plane_execution
        or capability.get("structure_tensor_steering") is not False
    ):
        raise ValueError(
            f"{context} Pictologics dimensionality/steering capabilities are inconsistent"
        )
    if (
        expected_parameters["filter"] == "gabor"
        and capability.get("anisotropic_spacing") != "supported"
    ):
        raise ValueError(
            f"{context} Pictologics Gabor spacing capability is inconsistent"
        )
    requested_pooling = expected_parameters.get("pooling")
    if requested_pooling is not None:
        rotation_pooling = capability.get("rotation_pooling")
        if (
            not isinstance(rotation_pooling, list)
            or requested_pooling not in rotation_pooling
        ):
            raise ValueError(
                f"{context} requested pooling is absent from Pictologics capabilities"
            )
    if expected_parameters["filter"] in {"riesz_log", "riesz_simoncelli"}:
        _require_concrete_provenance(
            capability.get("supported_riesz_orders"),
            field="supported_riesz_orders",
            context=context,
        )
    supported_boundaries = capability.get("supported_boundaries")
    if (
        not isinstance(supported_boundaries, list)
        or str(expected_selected).upper() not in supported_boundaries
    ):
        raise ValueError(
            f"{context} selected boundary is absent from Pictologics capabilities"
        )
    capability_mode = capability.get("effective_boundary")
    expected_implementation = (
        "native_periodic_fft"
        if capability_mode == "as_specified_via_padding"
        and expected_selected == "periodic"
        else capability_mode
    )
    if execution.get("implementation") != expected_implementation:
        raise ValueError(
            f"{context} boundary implementation conflicts with Pictologics capabilities"
        )


def _prepare_state(
    output_dir: Path,
    immutable_spec: Mapping[str, Any],
    task_ids: Sequence[str],
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = dict(immutable_spec)
    spec["schema_version"] = COMPLIANCE_RUN_SCHEMA_VERSION
    spec["controller_source_sha256"] = _controller_source_sha256()
    spec["fingerprint"] = fingerprint(spec)
    manifest_path = output_dir / "run_manifest.json"
    state_path = output_dir / "run_state.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"Compliance run already exists at {output_dir}; use --resume or a new output directory"
            )
        existing = _read_json(manifest_path)
        existing_without_fingerprint = dict(existing)
        stored_fingerprint = existing_without_fingerprint.pop("fingerprint", None)
        if (
            not stored_fingerprint
            or fingerprint(existing_without_fingerprint) != stored_fingerprint
            or stored_fingerprint != spec["fingerprint"]
        ):
            raise RunIntegrityError(
                "Compliance run inputs/source changed; refusing unsafe resume"
            )
        if not state_path.is_file():
            raise RunIntegrityError(
                "Compliance run manifest exists but run state is missing"
            )
        state = _read_json(state_path)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        orphaned = sorted(
            path.name
            for path in output_dir.iterdir()
            if path.name != ".compliance.lock"
        )
        if orphaned:
            raise FileExistsError(
                "Compliance output directory contains artifacts but no run manifest: "
                + ", ".join(orphaned)
            )
        atomic_write_json(manifest_path, spec)
        state = {
            "schema_version": COMPLIANCE_RUN_SCHEMA_VERSION,
            "tasks": {
                task_id: {
                    "status": "pending",
                    "attempt": 0,
                    "payload_path": None,
                    "payload_sha256": None,
                    "error": None,
                }
                for task_id in task_ids
            },
            "updated_at": time.time(),
        }
        atomic_write_json(state_path, state)
    if state.get("schema_version") != COMPLIANCE_RUN_SCHEMA_VERSION:
        raise RunIntegrityError("Compliance state schema version is invalid")
    tasks = state.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(task_ids):
        raise RunIntegrityError(
            "Compliance state task set does not match immutable run manifest"
        )
    allowed_statuses = {"pending", "running", "interrupted", "completed", "error"}
    reset_interrupted = False
    for task in tasks.values():
        if (
            not isinstance(task, dict)
            or task.get("status") not in allowed_statuses
            or not isinstance(task.get("attempt"), int)
            or task["attempt"] < 0
        ):
            raise RunIntegrityError("Compliance state contains a malformed task record")
        if task.get("status") in {"running", "interrupted"}:
            task["status"] = "pending"
            reset_interrupted = True
    if reset_interrupted:
        _commit_state(output_dir, state)
    return spec, state


def _commit_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    atomic_write_json(output_dir / "run_state.json", state)


def _completed_payload(
    output_dir: Path, task: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    if task.get("status") != "completed":
        return None
    relative = task.get("payload_path")
    expected_hash = task.get("payload_sha256")
    if not relative or not expected_hash:
        raise RunIntegrityError("Completed compliance task lacks payload provenance")
    path = output_dir / str(relative)
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RunIntegrityError(
            f"Completed compliance payload is missing or changed: {path}"
        )
    payload = _read_json(path)
    if payload.get("schema_version") != ADAPTER_PROTOCOL_VERSION:
        raise RunIntegrityError(f"Adapter payload protocol mismatch: {path}")
    return payload


def _validate_compliance_payload(
    payload: Mapping[str, Any],
    *,
    adapter: str,
    family: str,
    discretization: str,
    aggregation: str,
    bins: int,
    bin_width: float,
    intensity_min: Optional[float],
    intensity_max: Optional[float],
) -> None:
    """Validate identity, version, selection, and preprocessing before commit."""

    if aggregation != REQUIRED_AGGREGATION:
        raise RunIntegrityError(
            f"Compliance runs require {REQUIRED_AGGREGATION}, got {aggregation}"
        )

    if payload.get("schema_version") != ADAPTER_PROTOCOL_VERSION:
        raise RunIntegrityError(
            f"{adapter} compliance payload uses an unexpected protocol version"
        )
    if str(payload.get("adapter") or "") != adapter:
        raise RunIntegrityError(f"Compliance payload identity mismatch for {adapter}")

    profile = _runtime_profile(adapter)
    software = payload.get("software")
    if not isinstance(software, Mapping):
        raise RunIntegrityError(
            f"{adapter} compliance payload lacks software provenance"
        )
    if (
        str(software.get("distribution") or "").casefold()
        != profile.distribution.casefold()
    ):
        raise RunIntegrityError(
            f"{adapter} compliance payload distribution is incorrect"
        )
    expected_metadata_version = profile.metadata_version or profile.version
    if str(software.get("version") or "") != expected_metadata_version:
        raise RunIntegrityError(
            f"{adapter} compliance payload metadata version is not the reviewed "
            f"{expected_metadata_version} for release {profile.version}"
        )

    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise RunIntegrityError(
            f"{adapter} compliance payload lacks selection provenance"
        )
    requested = selection.get("requested_families")
    unsupported = selection.get("unsupported_families")
    if (
        requested != get_adapter(adapter).supported_members(family)
        or not isinstance(unsupported, list)
        or family in unsupported
    ):
        raise RunIntegrityError(
            f"{adapter} compliance payload does not attest the scheduled family {family}"
        )
    if not str(selection.get("mode") or "").strip():
        raise RunIntegrityError(f"{adapter} compliance payload lacks a selection mode")

    features_container = payload.get("features")
    values_container = payload.get("values")
    if not isinstance(features_container, Mapping) or not isinstance(
        values_container, Mapping
    ):
        raise RunIntegrityError(
            f"{adapter} compliance payload requires features and values containers"
        )
    feature_names = features_container.get("all")
    raw_values = values_container.get("all")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(name, str) or not name.strip() for name in feature_names)
        or len(feature_names) != len(set(feature_names))
    ):
        raise RunIntegrityError(
            f"{adapter} compliance payload requires a non-empty unique feature surface"
        )
    if not isinstance(raw_values, Mapping) or any(
        not isinstance(name, str) or name not in feature_names for name in raw_values
    ):
        raise RunIntegrityError(
            f"{adapter} compliance payload values do not match its feature surface"
        )

    metadata = payload.get("metadata")
    preprocessing = (
        metadata.get("preprocessing") if isinstance(metadata, Mapping) else None
    )
    aggregation_metadata = (
        metadata.get("aggregation") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(preprocessing, Mapping) or not isinstance(
        aggregation_metadata, Mapping
    ):
        raise RunIntegrityError(
            f"{adapter} compliance payload lacks preprocessing/aggregation provenance"
        )
    if str(preprocessing.get("discretization") or "") != discretization:
        raise RunIntegrityError(f"{adapter} compliance payload discretization mismatch")
    try:
        observed_bins = int(preprocessing.get("bins", -1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunIntegrityError(
            f"{adapter} compliance payload bin count is invalid"
        ) from exc
    if observed_bins != int(bins):
        raise RunIntegrityError(f"{adapter} compliance payload bin-count mismatch")
    try:
        observed_width = float(preprocessing.get("bin_width"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunIntegrityError(
            f"{adapter} compliance payload bin width is invalid"
        ) from exc
    if not math.isclose(observed_width, float(bin_width), rel_tol=0.0, abs_tol=1e-12):
        raise RunIntegrityError(f"{adapter} compliance payload bin-width mismatch")
    expected_range = (
        None
        if intensity_min is None or intensity_max is None
        else [float(intensity_min), float(intensity_max)]
    )
    if preprocessing.get("intensity_range") != expected_range:
        raise RunIntegrityError(
            f"{adapter} compliance payload intensity-range mismatch"
        )
    if str(aggregation_metadata.get("requested") or "") != aggregation:
        raise RunIntegrityError(
            f"{adapter} compliance payload requested aggregation mismatch"
        )
    expected_effective_aggregation = resolve_aggregation(
        adapter,
        aggregation,
        [family],
    )
    if (
        str(aggregation_metadata.get("effective_directional") or "")
        != expected_effective_aggregation
    ):
        raise RunIntegrityError(f"{adapter} compliance payload aggregation mismatch")


def _execute_task(
    *,
    output_dir: Path,
    state: dict[str, Any],
    task_id: str,
    adapter: str,
    image: Path,
    mask: Path,
    family: str,
    discretization: str,
    aggregation: str,
    bins: int = 32,
    bin_width: float = 32.0,
    intensity_min: Optional[float] = None,
    intensity_max: Optional[float] = None,
    timeout: Optional[float],
    stop_requested: Optional[Callable[[], bool]],
    timed: bool = False,
) -> Optional[dict[str, Any]]:
    task = state["tasks"][task_id]
    completed = _completed_payload(output_dir, task)
    if completed is not None:
        _validate_compliance_payload(
            completed,
            adapter=adapter,
            family=family,
            discretization=discretization,
            aggregation=aggregation,
            bins=bins,
            bin_width=bin_width,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
        )
        return completed
    task.update(
        status="running",
        attempt=int(task.get("attempt", 0)) + 1,
        error=None,
    )
    _commit_state(output_dir, state)
    try:
        payload, _ = run_adapter_process(
            adapter,
            image=str(image),
            mask=str(mask),
            discretization=discretization,
            aggregation=aggregation,
            bins=bins,
            bin_width=bin_width,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            families=get_adapter(adapter).supported_members(family),
            include_values=True,
            timed=timed,
            no_warmup=True,
            timeout=timeout,
            stop_requested=stop_requested,
            iterations=1,
        )
        payload = _json_safe_payload(payload)
        if not isinstance(payload, Mapping):
            raise RunIntegrityError(
                f"{adapter} returned a non-object compliance payload"
            )
        _validate_compliance_payload(
            payload,
            adapter=adapter,
            family=family,
            discretization=discretization,
            aggregation=aggregation,
            bins=bins,
            bin_width=bin_width,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
        )
        relative = Path("raw") / f"{task_id}.json"
        payload_hash = atomic_write_json(output_dir / relative, payload)
        task.update(
            status="completed",
            payload_path=str(relative),
            payload_sha256=payload_hash,
            error=None,
        )
        _commit_state(output_dir, state)
        return payload
    except (KeyboardInterrupt, AdapterInterrupted):
        task.update(status="interrupted", error="interrupted")
        _commit_state(output_dir, state)
        raise
    except Exception as exc:
        task.update(status="error", error=f"{type(exc).__name__}: {exc}")
        _commit_state(output_dir, state)
        return None


def _merge_family_payloads(
    adapter: str, payloads: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    names: list[str] = []
    values: dict[str, Any] = {}
    software: Optional[Mapping[str, Any]] = None
    family_metadata: list[Mapping[str, Any]] = []
    for payload in payloads:
        current_software = payload.get("software", {})
        if software is None:
            software = current_software if isinstance(current_software, Mapping) else {}
        elif current_software != software:
            raise RunIntegrityError(
                f"{adapter} family payloads report inconsistent software versions"
            )
        feature_container = payload.get("features", {})
        value_container = payload.get("values", {})
        if isinstance(feature_container, Mapping):
            names.extend(str(name) for name in feature_container.get("all", []))
        if isinstance(value_container, Mapping) and isinstance(
            value_container.get("all", {}), Mapping
        ):
            for name, value in value_container["all"].items():
                if name in values and values[name] != value:
                    raise RunIntegrityError(
                        f"{adapter} returned conflicting values for {name}"
                    )
                values[str(name)] = value
        metadata = payload.get("metadata", {})
        if isinstance(metadata, Mapping):
            family_metadata.append(metadata)
    return {
        "schema_version": ADAPTER_PROTOCOL_VERSION,
        "adapter": adapter,
        "software": dict(software or {}),
        "selection": {"mode": "compliance_family_merge"},
        "features": {"all": list(dict.fromkeys(names))},
        "values": {"all": values},
        "metadata": {"family_payload_metadata": family_metadata},
    }


def _record_workload(record: ComparisonRecord) -> str:
    """Return the timed group whose call produced one IBSI reference row."""

    if record.family == "morphology":
        return (
            "group_spatial_morphology"
            if record.ibsi_code in MORPHOLOGY_SPATIAL_CODES
            else "group_morphology"
        )
    return next(
        (
            group
            for group, members in WORKLOAD_GROUPS.items()
            if record.family in members
        ),
        record.family,
    )


def _mark_execution_errors(
    records: Sequence[ComparisonRecord],
    failed: Mapping[tuple[str, str], str],
) -> list[ComparisonRecord]:
    output = []
    for record in records:
        error = failed.get((record.adapter, record.family)) or failed.get(
            (record.adapter, _record_workload(record))
        )
        if error:
            output.append(
                replace(
                    record,
                    attempted=True,
                    finite=False,
                    evaluated=False,
                    passed=None,
                    status="error",
                    detail=error,
                )
            )
        else:
            output.append(record)
    return output


def run_ibsi1_digital_phantom(
    *,
    image: Path,
    mask: Path,
    references_csv: Path,
    reference_manifest: Path,
    output_dir: Path,
    adapters: Sequence[str],
    resume: bool = False,
    timeout: Optional[float] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    render_report: bool = True,
) -> list[ComparisonRecord]:
    """Run the official digital phantom once per adapter/family, with safe resume."""

    image = Path(image).expanduser().resolve()
    mask = Path(mask).expanduser().resolve()
    references_csv = Path(references_csv).expanduser().resolve()
    reference_manifest = Path(reference_manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for path in (image, mask, references_csv, reference_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate_official_ibsi1_digital_phantom(image, mask)
    _validate_nifti_pair(
        image,
        mask,
        context="IBSI 1 digital phantom",
        require_positive_integer_image=True,
    )
    validate_reference_table_manifest(
        references_csv,
        reference_manifest,
        expected_specification="IBSI 1",
    )
    adapters = tuple(dict.fromkeys(adapters))
    if not adapters:
        raise ValueError("At least one adapter is required")
    adapter_profiles = configured_adapter_profiles(adapters)

    task_specs: list[tuple[str, str, str, str]] = []
    aggregation_unsupported: dict[str, list[str]] = {}
    for adapter in adapters:
        capabilities = get_adapter(adapter)
        aggregation_unsupported[adapter] = []
        # One task per timed group, so accuracy uses the benchmark's calls.
        for family in GROUP_ORDER:
            if not capabilities.supports(family):
                continue
            if not supports_aggregation(
                adapter, REQUIRED_AGGREGATION, capabilities.supported_members(family)
            ):
                aggregation_unsupported[adapter].append(family)
                continue
            if capabilities.supports(family):
                task_specs.append(
                    (
                        f"{adapter}__{family}",
                        adapter,
                        family,
                        REQUIRED_AGGREGATION,
                    )
                )
    immutable = {
        "kind": "ibsi1_digital_phantom",
        # Values come from the same adapter code path that the benchmark times,
        # run once without a warmup.
        "adapter_code_path": "timed_calculation_one_call",
        "inputs": {
            "image": str(image),
            "image_sha256": sha256_file(image),
            "mask": str(mask),
            "mask_sha256": sha256_file(mask),
            "references": str(references_csv),
            "references_sha256": sha256_file(references_csv),
            "reference_manifest": str(reference_manifest),
            "reference_manifest_sha256": sha256_file(reference_manifest),
        },
        "adapters": list(adapters),
        "configured_adapter_profiles": adapter_profiles,
        "verified_adapter_environments": _verified_environment_snapshots(adapters),
        "required_directional_aggregation": REQUIRED_AGGREGATION,
        "aggregation_unsupported_families": aggregation_unsupported,
        "discretization": "identity",
        "tasks": [task_id for task_id, _, _, _ in task_specs],
    }
    references = load_reference_csv(references_csv)
    all_records: list[ComparisonRecord] = []
    audits: dict[str, Any] = {}
    failed: dict[tuple[str, str], str] = {}

    with RunLock(output / ".compliance.lock"):
        _, state = _prepare_state(
            output,
            immutable,
            [task_id for task_id, _, _, _ in task_specs],
            resume=resume,
        )
        payloads_by_adapter: dict[str, list[Mapping[str, Any]]] = {
            adapter: [] for adapter in adapters
        }
        for task_id, adapter, family, aggregation in task_specs:
            payload = _execute_task(
                output_dir=output,
                state=state,
                task_id=task_id,
                adapter=adapter,
                image=image,
                mask=mask,
                family=family,
                discretization="identity",
                aggregation=aggregation,
                timeout=timeout,
                stop_requested=stop_requested,
                timed=True,
            )
            if payload is None:
                failed[(adapter, family)] = str(
                    state["tasks"][task_id].get("error") or "adapter error"
                )
            else:
                payloads_by_adapter[adapter].append(payload)

        for adapter in adapters:
            profile = select_ibsi1_digital_phantom_profile(references)
            merged = _merge_family_payloads(adapter, payloads_by_adapter[adapter])
            records, audit = evaluate_adapter_payload(
                adapter=adapter,
                payload=merged,
                references=profile,
                release_version=adapter_profiles[adapter]["version"],
            )
            all_records.extend(records)
            audits[adapter] = audit
        all_records = _mark_execution_errors(all_records, failed)
        atomic_write_text(output / "comparisons.csv", comparison_to_csv(all_records))
        atomic_write_json(output / "mapping_audit.json", audits)
        atomic_write_json(
            output / "result_manifest.json",
            {
                "schema_version": COMPLIANCE_RUN_SCHEMA_VERSION,
                "kind": "ibsi1_digital_phantom",
                "record_count": len(all_records),
                "required_directional_aggregation": REQUIRED_AGGREGATION,
                "aggregation_unsupported_families": aggregation_unsupported,
                "failed_family_tasks": [
                    {"adapter": adapter, "family": family, "error": error}
                    for (adapter, family), error in sorted(failed.items())
                ],
                "comparison_table_sha256": sha256_file(output / "comparisons.csv"),
                "complete": not failed,
            },
        )

    if render_report:
        generate_compliance_report(
            all_records,
            output / "report",
            source_metadata={
                "run_kind": "ibsi1_digital_phantom",
                "run_manifest_name": "run_manifest.json",
                "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
                "complete": not failed,
            },
        )
    return all_records


def load_ibsi2_phase2_candidate_manifest(path: Path) -> IBSI2CandidateEntries:
    """Validate provenance-attested, package-specific filtered response maps."""

    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != IBSI2_CANDIDATE_SCHEMA_VERSION
        or manifest.get("kind") != "ibsi2_phase2_response_maps"
    ):
        raise ValueError("Invalid IBSI 2 Phase 2 candidate manifest schema/kind")
    support_class = _manifest_support_class(manifest)
    root = manifest_path.parent
    _validate_protocol_review(manifest, phase="Phase 2")
    adapters, support_declarations = _validate_native_support_grid(
        manifest,
        phase="Phase 2",
        id_field="filter_id",
        expected_ids=IBSI2_PHASE2_FILTER_IDS,
    )
    source = manifest.get("source_data")
    if not isinstance(source, Mapping) or (
        source.get("repository") != IBSI_DATA_REPOSITORY
        or source.get("commit") != IBSI_DATA_COMMIT
        or source.get("image_sha256") != IBSI2_PHASE2_SOURCE_IMAGE_SHA256
        or source.get("mask_sha256") != IBSI2_PHASE2_SOURCE_MASK_SHA256
    ):
        raise ValueError(
            "IBSI 2 Phase 2 candidates must be bound to the pinned official PAT1 CT and mask"
        )
    source_image = _verified_relative_file(
        root=root,
        relative_path=source.get("image_path"),
        expected_sha256=source.get("image_sha256"),
        context="IBSI 2 Phase 2 source image",
    )
    source_mask = _verified_relative_file(
        root=root,
        relative_path=source.get("mask_path"),
        expected_sha256=source.get("mask_sha256"),
        context="IBSI 2 Phase 2 source mask",
    )
    _validate_nifti_pair(
        source_image,
        source_mask,
        context="IBSI 2 Phase 2 source data",
        require_binary_mask=True,
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Candidate manifest entries must be a list")
    seen: set[tuple[str, str]] = set()
    validated: list[dict[str, Any]] = []
    required = {
        "adapter",
        "filter_id",
        "image_path",
        "mask_path",
        "image_sha256",
        "mask_sha256",
        "filter_input_path",
        "filter_input_sha256",
        "generator_distribution",
        "generator_version",
        "generator_source_revision",
        "generator_entrypoint",
        "generator_command",
        "generator_source_path",
        "generator_source_sha256",
        "executed_parameters",
        "boundary_execution",
        "native_capability",
        "filter_config_revision",
        "filter_config_path",
        "filter_config_sha256",
        "preprocessing_config_path",
        "preprocessing_config_sha256",
        "environment_lock_path",
        "environment_lock_sha256",
    }
    for entry in entries:
        if not isinstance(entry, Mapping) or not required.issubset(entry):
            raise ValueError(
                f"Candidate manifest entry lacks required provenance: {entry!r}"
            )
        adapter = str(entry["adapter"])
        filter_id = str(entry["filter_id"]).upper()
        if adapter not in adapters:
            raise ValueError(
                f"Candidate response map uses an adapter absent from the manifest: {adapter}"
            )
        capabilities = get_adapter(adapter)
        profile = _runtime_profile(adapter)
        if filter_id not in IBSI2_PHASE2_FILTER_IDS:
            raise ValueError(f"Unknown IBSI 2 Phase 2 filter ID: {filter_id}")
        key = (adapter, filter_id)
        if key in seen:
            raise ValueError(f"Duplicate candidate response map: {key}")
        if not support_declarations[key]["native_supported"]:
            raise ValueError(
                f"Candidate response map {key} is forbidden because native_supported is false"
            )
        seen.add(key)
        image = _verified_relative_file(
            root=root,
            relative_path=entry["image_path"],
            expected_sha256=entry["image_sha256"],
            context=f"Candidate {key} response map",
        )
        mask = _verified_relative_file(
            root=root,
            relative_path=entry["mask_path"],
            expected_sha256=entry["mask_sha256"],
            context=f"Candidate {key} response-map mask",
        )
        filter_input = _verified_relative_file(
            root=root,
            relative_path=entry["filter_input_path"],
            expected_sha256=entry["filter_input_sha256"],
            context=f"Candidate {key} controlled filter input",
        )
        if (
            str(entry["generator_distribution"]).casefold()
            != capabilities.distribution.casefold()
        ):
            raise ValueError(
                f"Candidate {key} is labelled as {adapter} but was generated by "
                f"{entry['generator_distribution']!r}"
            )
        generator_version = _require_concrete_provenance(
            entry["generator_version"],
            field="generator_version",
            context=f"Candidate {key}",
        )
        if generator_version != profile.version:
            raise ValueError(
                f"Candidate {key} was generated by version {generator_version}; "
                f"the reviewed adapter profile requires {profile.version}"
            )
        provenance_files = _validated_generator_provenance(
            entry,
            root=root,
            context=f"Candidate {key}",
        )
        _validate_environment_lock(
            provenance_files["environment_lock"],
            adapter=adapter,
            profile=profile,
        )
        preprocessing_config = _verified_relative_file(
            root=root,
            relative_path=entry["preprocessing_config_path"],
            expected_sha256=entry["preprocessing_config_sha256"],
            context=f"Candidate {key} preprocessing configuration",
        )
        validate_ibsi2_phase2_filter_config(
            provenance_files["filter_config"],
            filter_id=filter_id,
        )
        _validate_ibsi2_execution_contract(
            entry,
            adapter=adapter,
            filter_config_path=provenance_files["filter_config"],
            context=f"Candidate {key}",
            support_class=support_class,
        )
        validate_ibsi2_phase2_preprocessing_config(
            preprocessing_config,
            filter_id=filter_id,
        )
        _validate_nifti_pair(
            image,
            mask,
            context=f"IBSI 2 Phase 2 {key}",
            require_binary_mask=True,
        )
        _validate_nifti_pair(
            filter_input,
            mask,
            context=f"IBSI 2 Phase 2 {key} controlled preprocessing",
            require_binary_mask=True,
        )
        _validate_same_nifti_geometry(
            image,
            filter_input,
            context=f"IBSI 2 Phase 2 {key} response-map geometry",
        )
        _validate_phase2_grid(
            image=filter_input,
            mask=mask,
            source_image=source_image,
            dimension=filter_id.rsplit(".", 1)[-1],
            context=f"IBSI 2 Phase 2 {key}",
        )
        validated.append(
            {
                **dict(entry),
                "adapter": adapter,
                "filter_id": filter_id,
                "image": image,
                "mask": mask,
                "source_image": source_image,
                "source_mask": source_mask,
                "filter_input": filter_input,
                "generator_source": provenance_files["generator_source"],
                "filter_config": provenance_files["filter_config"],
                "preprocessing_config": preprocessing_config,
                "environment_lock": provenance_files["environment_lock"],
            }
        )
    supported_keys = {
        key
        for key, declaration in support_declarations.items()
        if declaration["native_supported"]
    }
    missing_supported = sorted(supported_keys.difference(seen))
    if missing_supported:
        raise ValueError(
            "Every IBSI 2 Phase 2 native-supported filter requires a candidate "
            f"response map entry; missing={missing_supported}"
        )
    _require_one_generator_version_per_adapter(validated, kind="IBSI 2 Phase 2")
    return IBSI2CandidateEntries(
        validated,
        adapters=adapters,
        support_declarations=support_declarations,
        support_class=support_class,
    )


def load_ibsi2_phase1_candidate_manifest(path: Path) -> IBSI2CandidateEntries:
    """Validate package/version/config provenance for Phase 1 response maps."""

    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != IBSI2_CANDIDATE_SCHEMA_VERSION
        or manifest.get("kind") != "ibsi2_phase1_response_maps"
    ):
        raise ValueError("Invalid IBSI 2 Phase 1 candidate manifest schema/kind")
    support_class = _manifest_support_class(manifest)
    root = manifest_path.parent
    _validate_protocol_review(manifest, phase="Phase 1")
    adapters, support_declarations = _validate_native_support_grid(
        manifest,
        phase="Phase 1",
        id_field="test_id",
        expected_ids=IBSI2_PHASE1_TEST_IDS,
    )
    source_identity = manifest.get("source_data")
    if not isinstance(source_identity, Mapping) or (
        source_identity.get("repository") != IBSI_DATA_REPOSITORY
        or source_identity.get("commit") != IBSI_DATA_COMMIT
    ):
        raise ValueError(
            "IBSI 2 Phase 1 candidates must identify the pinned official phantom repository"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Candidate manifest entries must be a list")
    required = {
        "adapter",
        "test_id",
        "response_map_path",
        "response_map_sha256",
        "source_image_path",
        "source_image_sha256",
        "generator_distribution",
        "generator_version",
        "generator_source_revision",
        "generator_entrypoint",
        "generator_command",
        "generator_source_path",
        "generator_source_sha256",
        "executed_parameters",
        "boundary_execution",
        "native_capability",
        "filter_config_revision",
        "filter_config_path",
        "filter_config_sha256",
        "preprocessing_config_path",
        "preprocessing_config_sha256",
        "environment_lock_path",
        "environment_lock_sha256",
    }
    seen: set[tuple[str, str]] = set()
    validated: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not required.issubset(entry):
            raise ValueError(
                f"Candidate manifest entry lacks required provenance: {entry!r}"
            )
        adapter = str(entry["adapter"])
        test_id = str(entry["test_id"]).casefold()
        if adapter not in adapters:
            raise ValueError(
                f"Candidate response map uses an adapter absent from the manifest: {adapter}"
            )
        capabilities = get_adapter(adapter)
        profile = _runtime_profile(adapter)
        if test_id not in IBSI2_PHASE1_TEST_IDS:
            raise ValueError(f"Unknown IBSI 2 Phase 1 test ID: {test_id}")
        key = (adapter, test_id)
        if key in seen:
            raise ValueError(f"Duplicate candidate response map: {key}")
        if not support_declarations[key]["native_supported"]:
            raise ValueError(
                f"Candidate response map {key} is forbidden because native_supported is false"
            )
        seen.add(key)
        response_map = _verified_relative_file(
            root=root,
            relative_path=entry["response_map_path"],
            expected_sha256=entry["response_map_sha256"],
            context=f"Candidate {key} response map",
        )
        source_image = _verified_relative_file(
            root=root,
            relative_path=entry["source_image_path"],
            expected_sha256=entry["source_image_sha256"],
            context=f"Candidate {key} source image",
        )
        if (
            str(entry["source_image_sha256"]).casefold()
            not in IBSI2_PHASE1_SOURCE_IMAGE_SHA256
        ):
            raise ValueError(
                f"Candidate {key} source image is not one of the pinned official phantoms"
            )
        source_mask = None
        mask_path_present = bool(str(entry.get("source_mask_path", "")).strip())
        mask_hash_present = bool(str(entry.get("source_mask_sha256", "")).strip())
        if mask_path_present != mask_hash_present:
            raise ValueError(
                f"Candidate {key} must provide both source-mask path and hash, or neither"
            )
        if mask_path_present:
            if (
                str(entry["source_mask_sha256"]).casefold()
                != IBSI2_PHASE1_SOURCE_MASK_SHA256
            ):
                raise ValueError(
                    f"Candidate {key} source mask is not the pinned official phantom mask"
                )
            source_mask = _verified_relative_file(
                root=root,
                relative_path=entry["source_mask_path"],
                expected_sha256=entry["source_mask_sha256"],
                context=f"Candidate {key} source mask",
            )
            _validate_nifti_pair(
                source_image,
                source_mask,
                context=f"IBSI 2 Phase 1 {key} source data",
                require_binary_mask=True,
            )
        if (
            str(entry["generator_distribution"]).casefold()
            != capabilities.distribution.casefold()
        ):
            raise ValueError(
                f"Candidate {key} is labelled as {adapter} but was generated by "
                f"{entry['generator_distribution']!r}"
            )
        generator_version = _require_concrete_provenance(
            entry["generator_version"],
            field="generator_version",
            context=f"Candidate {key}",
        )
        if generator_version != profile.version:
            raise ValueError(
                f"Candidate {key} was generated by version {generator_version}; "
                f"the reviewed adapter profile requires {profile.version}"
            )
        provenance_files = _validated_generator_provenance(
            entry,
            root=root,
            context=f"Candidate {key}",
        )
        _validate_environment_lock(
            provenance_files["environment_lock"],
            adapter=adapter,
            profile=profile,
        )
        preprocessing_config = _verified_relative_file(
            root=root,
            relative_path=entry["preprocessing_config_path"],
            expected_sha256=entry["preprocessing_config_sha256"],
            context=f"Candidate {key} preprocessing configuration",
        )
        validate_ibsi2_phase1_candidate_configs(
            provenance_files["filter_config"],
            preprocessing_config,
            test_id=test_id,
            source_image_sha256=str(entry["source_image_sha256"]).casefold(),
        )
        _validate_ibsi2_execution_contract(
            entry,
            adapter=adapter,
            filter_config_path=provenance_files["filter_config"],
            context=f"Candidate {key}",
            support_class=support_class,
        )
        if not response_map.name.casefold().endswith((".nii", ".nii.gz")):
            raise ValueError(f"Candidate response map is not NIfTI: {response_map}")
        _validate_nifti_response_map(
            response_map,
            context=f"IBSI 2 Phase 1 {key}",
        )
        _validate_same_nifti_geometry(
            response_map,
            source_image,
            context=f"IBSI 2 Phase 1 {key} response-map geometry",
        )
        validated.append(
            {
                **dict(entry),
                "adapter": adapter,
                "test_id": test_id,
                "response_map": response_map,
                "source_image": source_image,
                "source_mask": source_mask,
                "generator_source": provenance_files["generator_source"],
                "filter_config": provenance_files["filter_config"],
                "preprocessing_config": preprocessing_config,
                "environment_lock": provenance_files["environment_lock"],
            }
        )
    supported_keys = {
        key
        for key, declaration in support_declarations.items()
        if declaration["native_supported"]
    }
    missing_supported = sorted(supported_keys.difference(seen))
    if missing_supported:
        raise ValueError(
            "Every IBSI 2 Phase 1 native-supported test requires a candidate "
            f"response map entry; missing={missing_supported}"
        )
    _require_one_generator_version_per_adapter(validated, kind="IBSI 2 Phase 1")
    return IBSI2CandidateEntries(
        validated,
        adapters=adapters,
        support_declarations=support_declarations,
        support_class=support_class,
    )


def _validate_nifti_response_map(path: Path, *, context: str) -> None:
    """Require a readable 3D response map with valid physical geometry.

    Candidate voxel finiteness remains a row-level comparison outcome, so a
    non-finite standardized map is reported as a failed calculation rather
    than aborting the complete multi-adapter evaluation.
    """

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "nibabel and numpy are required to validate compliance inputs"
        ) from exc
    try:
        response = nib.load(str(path))
        data = np.asanyarray(response.dataobj)
    except Exception as exc:
        raise ValueError(f"{context}: response map is not a readable NIfTI") from exc
    if data.ndim != 3 or any(int(value) <= 0 for value in data.shape):
        raise ValueError(f"{context}: response map must be a non-empty 3D array")
    if not np.isfinite(response.affine).all():
        raise ValueError(f"{context}: response-map affine must be finite")
    spacing = response.header.get_zooms()[:3]
    if len(spacing) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0.0 for value in spacing
    ):
        raise ValueError(f"{context}: response-map spacing must be finite and positive")


def _validate_nifti_pair(
    image: Path,
    mask: Path,
    *,
    context: str,
    require_positive_integer_image: bool = False,
    require_binary_mask: bool = False,
) -> None:
    """Validate 3D geometry before any isolated extraction process is scheduled."""

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "nibabel and numpy are required to validate compliance inputs"
        ) from exc
    image_header = nib.load(str(image))
    mask_header = nib.load(str(mask))
    if image_header.shape != mask_header.shape or len(image_header.shape) != 3:
        raise ValueError(f"{context}: image/mask shape mismatch or input is not 3D")
    if not np.allclose(image_header.affine, mask_header.affine, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context}: image/mask affine mismatch")
    spacing = image_header.header.get_zooms()[:3]
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in spacing):
        raise ValueError(f"{context}: image spacing must be finite and positive")
    mask_values = np.asanyarray(mask_header.dataobj)
    if not np.isfinite(mask_values).all():
        raise ValueError(f"{context}: mask contains non-finite values")
    if require_binary_mask and not np.isin(mask_values, (0, 1)).all():
        raise ValueError(f"{context}: mask must be binary with values 0 and 1")
    roi = mask_values > 0
    if not np.any(roi):
        raise ValueError(f"{context}: mask has no positive voxels")
    roi_values = np.asanyarray(image_header.dataobj)[roi]
    if not np.isfinite(roi_values).all():
        raise ValueError(f"{context}: image contains non-finite ROI values")
    if require_positive_integer_image and (
        np.any(roi_values < 1.0)
        or not np.allclose(roi_values, np.rint(roi_values), rtol=0.0, atol=1e-8)
    ):
        raise ValueError(
            f"{context}: identity discretization requires positive-integer ROI grey levels"
        )


def _validate_same_nifti_geometry(
    candidate: Path,
    expected: Path,
    *,
    context: str,
) -> None:
    """Require identical array shape and physical geometry."""

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "nibabel and numpy are required to validate compliance geometry"
        ) from exc
    candidate_image = nib.load(str(candidate))
    expected_image = nib.load(str(expected))
    if candidate_image.shape != expected_image.shape:
        raise ValueError(
            f"{context}: shape mismatch {candidate_image.shape} != {expected_image.shape}"
        )
    if not np.allclose(
        candidate_image.affine,
        expected_image.affine,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{context}: affine differs from the required grid")


def _validate_phase2_grid(
    *,
    image: Path,
    mask: Path,
    source_image: Path,
    dimension: str,
    context: str,
) -> None:
    """Enforce the deterministic A/B geometry and resegmented-mask guardrails."""

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "nibabel and numpy are required to validate IBSI 2 preprocessing"
        ) from exc
    image_nifti = nib.load(str(image))
    mask_nifti = nib.load(str(mask))
    if dimension == "A":
        _validate_same_nifti_geometry(
            image,
            source_image,
            context=f"{context} A grid",
        )
        return
    if dimension != "B":
        raise ValueError(f"{context}: unknown Phase 2 dimension {dimension!r}")
    if tuple(int(value) for value in image_nifti.shape) != (200, 197, 180):
        raise ValueError(
            f"{context}: B grid must have shape (200, 197, 180), got "
            f"{image_nifti.shape}"
        )
    spacing = tuple(float(value) for value in image_nifti.header.get_zooms()[:3])
    if not np.allclose(spacing, (1.0, 1.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError(f"{context}: B grid spacing must be exactly 1 mm isotropic")
    mask_values = np.asanyarray(mask_nifti.dataobj)
    if int(np.count_nonzero(mask_values)) != 357_802:
        raise ValueError(f"{context}: B resegmented mask must contain 357802 voxels")
    image_values = np.asanyarray(image_nifti.dataobj)
    if not np.isfinite(image_values).all() or not np.allclose(
        image_values,
        np.rint(image_values),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            f"{context}: B preprocessed image must contain rounded finite intensities"
        )


def _require_one_generator_version_per_adapter(
    entries: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> None:
    versions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for entry in entries:
        versions[str(entry["adapter"])].add(
            (
                str(entry["generator_distribution"]),
                str(entry["generator_version"]),
            )
        )
    mixed = {
        adapter: sorted(values)
        for adapter, values in versions.items()
        if len(values) != 1
    }
    if mixed:
        raise ValueError(
            f"{kind} candidate manifest mixes generator versions for one adapter: "
            + json.dumps(mixed, sort_keys=True)
        )


def run_ibsi2_phase2_from_response_maps(
    *,
    candidate_manifest: Path,
    references_csv: Path,
    reference_manifest: Path,
    output_dir: Path,
    adapters: Sequence[str],
    resume: bool = False,
    timeout: Optional[float] = None,
    render_report: bool = True,
) -> list[ComparisonRecord]:
    """Calculate the 18 Phase 2 statistics from audited package response maps."""

    entries = load_ibsi2_phase2_candidate_manifest(candidate_manifest)
    references_csv = Path(references_csv).expanduser().resolve()
    reference_manifest = Path(reference_manifest).expanduser().resolve()
    validate_reference_table_manifest(
        references_csv,
        reference_manifest,
        expected_specification="IBSI 2",
        expected_phase="phase2",
    )
    references = [
        record
        for record in load_reference_csv(references_csv)
        if record.specification == "IBSI 2" and record.phase == "phase2"
    ]
    adapters = tuple(dict.fromkeys(adapters))
    if not adapters:
        raise ValueError("At least one adapter is required")
    _require_requested_adapters(adapters, entries.adapters, phase="Phase 2")
    adapter_profiles = configured_adapter_profiles(adapters)
    support_declarations = entries.support_declarations
    execution_contracts = ibsi2_execution_contracts(
        entries,
        id_field="filter_id",
    )
    entries_by_key = {
        (entry["adapter"], entry["filter_id"]): entry for entry in entries
    }
    task_specs = [
        (
            f"{adapter}__{filter_id.replace('.', '_')}",
            adapter,
            filter_id,
            entries_by_key[(adapter, filter_id)],
        )
        for adapter in adapters
        for filter_id in IBSI2_PHASE2_FILTER_IDS
        if (adapter, filter_id) in entries_by_key
    ]
    output = Path(output_dir).expanduser().resolve()
    immutable = {
        "kind": "ibsi2_phase2_from_response_maps",
        "support_class": entries.support_class,
        "candidate_manifest": str(Path(candidate_manifest).expanduser().resolve()),
        "candidate_manifest_sha256": sha256_file(
            Path(candidate_manifest).expanduser().resolve()
        ),
        "references": str(references_csv),
        "references_sha256": sha256_file(references_csv),
        "reference_manifest": str(reference_manifest),
        "reference_manifest_sha256": sha256_file(reference_manifest),
        "adapters": list(adapters),
        "support_declarations": [
            support_declarations[(adapter, filter_id)]
            for adapter in adapters
            for filter_id in IBSI2_PHASE2_FILTER_IDS
        ],
        "configured_adapter_profiles": adapter_profiles,
        "execution_contracts": execution_contracts,
        "verified_adapter_environments": _verified_environment_snapshots(adapters),
        "tasks": [task_id for task_id, _, _, _ in task_specs],
    }
    records: list[ComparisonRecord] = []
    audits: dict[str, Any] = {}
    failed_configs: dict[tuple[str, str], str] = {}
    missing_supported_filters = [
        {"adapter": adapter, "filter_id": filter_id}
        for adapter in adapters
        for filter_id in IBSI2_PHASE2_FILTER_IDS
        if support_declarations[(adapter, filter_id)]["native_supported"]
        and (adapter, filter_id) not in entries_by_key
    ]

    with RunLock(output / ".compliance.lock"):
        _, state = _prepare_state(output, immutable, immutable["tasks"], resume=resume)
        payload_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for task_id, adapter, filter_id, entry in task_specs:
            payload = _execute_task(
                output_dir=output,
                state=state,
                task_id=task_id,
                adapter=adapter,
                image=entry["image"],
                mask=entry["mask"],
                family="intensity",
                # IBSI 2 Phase 2 compares first-order statistics of continuous
                # response-map intensities.  No grey-level discretisation is
                # applied; A/B describe filter preprocessing, not aggregation.
                discretization="raw",
                aggregation=REQUIRED_AGGREGATION,
                timeout=timeout,
                stop_requested=None,
            )
            if payload is None:
                failed_configs[(adapter, filter_id)] = str(
                    state["tasks"][task_id].get("error") or "adapter error"
                )
            else:
                payload_by_key[(adapter, filter_id)] = payload

        empty_payload = {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "software": {"version": "unknown"},
            "features": {"all": []},
            "values": {"all": {}},
        }
        for adapter in adapters:
            audits[adapter] = {}
            for filter_id in IBSI2_PHASE2_FILTER_IDS:
                config_refs = [
                    record for record in references if record.configuration == filter_id
                ]
                payload = payload_by_key.get((adapter, filter_id), empty_payload)
                config_records, audit = evaluate_adapter_payload(
                    adapter=adapter,
                    payload=payload,
                    references=config_refs,
                    release_version=adapter_profiles[adapter]["version"],
                )
                declaration = support_declarations[(adapter, filter_id)]
                support_detail = (
                    f"reviewed native-supported filter declaration: {declaration['reason']}; "
                    f"evidence: {declaration['evidence']}"
                )
                if not declaration["native_supported"]:
                    config_records = [
                        replace(
                            record,
                            status="native_unsupported",
                            detail=(
                                f"reviewed native-support declaration: {declaration['reason']}; "
                                f"evidence: {declaration['evidence']}"
                            ),
                        )
                        for record in config_records
                    ]
                elif (adapter, filter_id) in failed_configs:
                    config_records = [
                        replace(
                            record,
                            attempted=True,
                            status="error",
                            detail=(
                                f"{support_detail}; execution error: "
                                f"{failed_configs[(adapter, filter_id)]}"
                            ),
                        )
                        for record in config_records
                    ]
                else:
                    config_records = [
                        replace(
                            record,
                            detail=(
                                support_detail
                                + (f"; {record.detail}" if record.detail else "")
                            ),
                        )
                        for record in config_records
                    ]
                records.extend(config_records)
                audits[adapter][filter_id] = audit

        atomic_write_text(output / "comparisons.csv", comparison_to_csv(records))
        atomic_write_json(output / "mapping_audit.json", audits)
        atomic_write_json(
            output / "result_manifest.json",
            {
                "schema_version": COMPLIANCE_RUN_SCHEMA_VERSION,
                "kind": "ibsi2_phase2_from_response_maps",
                "support_class": entries.support_class,
                "record_count": len(records),
                "candidate_configurations": sum(
                    entry["adapter"] in adapters for entry in entries
                ),
                "support_declaration_grid_complete": True,
                "native_supported_filter_configurations": {
                    adapter: sum(
                        support_declarations[(adapter, filter_id)]["native_supported"]
                        for filter_id in IBSI2_PHASE2_FILTER_IDS
                    )
                    for adapter in adapters
                },
                "native_filter_denominator": len(IBSI2_PHASE2_FILTER_IDS),
                "missing_supported_candidate_configurations": missing_supported_filters,
                "failed_configurations": [
                    {"adapter": adapter, "filter_id": filter_id, "error": error}
                    for (adapter, filter_id), error in sorted(failed_configs.items())
                ],
                "comparison_table_sha256": sha256_file(output / "comparisons.csv"),
                "configured_adapter_profiles": adapter_profiles,
                "execution_contracts": execution_contracts,
                "supplied_inputs_processed_without_error": not failed_configs,
                "all_native_supported_configurations_supplied": not missing_supported_filters,
                "publication_complete": not failed_configs
                and not missing_supported_filters,
                "defined_checks_per_adapter": len(references),
                "standardized_checks_per_adapter": sum(
                    record.standardized for record in references
                ),
                "reference_manifest_sha256": sha256_file(reference_manifest),
                "candidate_manifest_sha256": sha256_file(
                    Path(candidate_manifest).expanduser().resolve()
                ),
            },
        )
    if render_report:
        generate_compliance_report(
            records,
            output / "report",
            source_metadata={
                "run_kind": "ibsi2_phase2_from_response_maps",
                "candidate_manifest_name": Path(candidate_manifest).name,
                "configured_adapter_profiles": adapter_profiles,
                "execution_contracts": execution_contracts,
                "publication_complete": not failed_configs
                and not missing_supported_filters,
                "support_declaration_grid_complete": True,
                "native_supported_filter_configurations": {
                    adapter: sum(
                        support_declarations[(adapter, filter_id)]["native_supported"]
                        for filter_id in IBSI2_PHASE2_FILTER_IDS
                    )
                    for adapter in adapters
                },
                "native_filter_denominator": len(IBSI2_PHASE2_FILTER_IDS),
                "missing_supported_candidate_configurations": missing_supported_filters,
                "failed_configurations": [
                    {"adapter": adapter, "filter_id": filter_id, "error": error}
                    for (adapter, filter_id), error in sorted(failed_configs.items())
                ],
                "reference_manifest_sha256": sha256_file(reference_manifest),
                "candidate_manifest_sha256": sha256_file(
                    Path(candidate_manifest).expanduser().resolve()
                ),
            },
        )
    return records
