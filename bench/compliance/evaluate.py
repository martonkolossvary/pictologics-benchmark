"""Strict joins between adapter observations and configuration-specific references."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from bench.benchmark_ledger import RunIntegrityError, sha256_file
from bench.compliance.models import ComparisonRecord, ReferenceRecord
from bench.compliance.references import (
    IBSI2_ANALYSIS_COMMIT,
    IBSI2_ANALYSIS_REPOSITORY,
    IBSI2_REFERENCE_COMMIT,
    IBSI2_REFERENCE_README_SHA256,
    IBSI2_REFERENCE_REPOSITORY,
    IBSI2_PHASE1_NONSTANDARDIZED_IDS,
    IBSI2_PHASE1_COMPARISON_RULE,
    IBSI2_PHASE1_COMPARISON_SOURCE,
    IBSI2_PHASE1_COMPARISON_SOURCE_SHA256,
    IBSI2_PHASE1_REFERENCE_SHA256,
    IBSI2_PHASE1_TEST_IDS,
    IBSI_CODE_TO_SEMANTIC_KEY,
)
from bench.compliance.tolerance import compare_absolute, compare_ibsi1
from bench.ibsi_mapping import (
    PYRADIOMICS_EXACT_ALIAS_EVIDENCE,
    classify_feature,
    documented_semantic_aliases,
)


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _payload_surface(
    adapter: str,
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, list[str]],
    dict[str, list[tuple[str, Optional[float]]]],
    dict[str, str],
    list[str],
    list[str],
    list[dict[str, str]],
    set[str],
]:
    """Return direct-native and documented semantic-alias support surfaces."""

    features_container = payload.get("features", {})
    values_container = payload.get("values", {})
    feature_names = (
        features_container.get("all", [])
        if isinstance(features_container, Mapping)
        else []
    )
    raw_values = (
        values_container.get("all", {}) if isinstance(values_container, Mapping) else {}
    )
    if not isinstance(feature_names, Sequence) or isinstance(
        feature_names, (str, bytes)
    ):
        raise ValueError("Adapter payload features.all must be a sequence")
    if not isinstance(raw_values, Mapping):
        raise ValueError("Adapter payload values.all must be an object")

    support: dict[str, list[str]] = defaultdict(list)
    observations: dict[str, list[tuple[str, Optional[float]]]] = defaultdict(list)
    external_codes: dict[str, str] = {}
    unmapped: list[str] = []
    excluded: list[str] = []
    exact_aliases: list[dict[str, str]] = []
    mapped_native_names: set[str] = set()
    all_names = list(
        dict.fromkeys(
            [str(name) for name in feature_names] + [str(name) for name in raw_values]
        )
    )
    for name in all_names:
        code, status = classify_feature(adapter, name)
        if status == "mapped" and code:
            semantic_key = IBSI_CODE_TO_SEMANTIC_KEY.get(code)
            if not semantic_key:
                unmapped.append(name)
                continue
            previous_code = external_codes.setdefault(semantic_key, code)
            if previous_code != code:
                raise ValueError(
                    f"Native features for {semantic_key} resolve through conflicting "
                    f"external identifiers: {previous_code}, {code}"
                )
            support[semantic_key].append(name)
            mapped_native_names.add(name)
            if name in raw_values:
                observations[semantic_key].append(
                    (name, _finite_float(raw_values[name]))
                )
        elif status == "excluded":
            excluded.append(name)
        else:
            unmapped.append(name)

        for alias_code, relation in documented_semantic_aliases(adapter, name).items():
            semantic_key = IBSI_CODE_TO_SEMANTIC_KEY.get(alias_code)
            if not semantic_key:
                raise ValueError(
                    f"Documented semantic alias uses an unknown code: {alias_code}"
                )
            previous_code = external_codes.setdefault(semantic_key, alias_code)
            if previous_code != alias_code:
                raise ValueError(
                    f"Alias for {semantic_key} conflicts with external identifier "
                    f"{previous_code}: {alias_code}"
                )
            alias_label = f"{name} [documented exact alias]"
            support[semantic_key].append(alias_label)
            mapped_native_names.add(name)
            if name in raw_values:
                observations[semantic_key].append(
                    (alias_label, _finite_float(raw_values[name]))
                )
            exact_aliases.append(
                {
                    "source_native_feature": name,
                    "source_ibsi_code": code or "",
                    "alias_ibsi_code": alias_code,
                    "relation": relation,
                    "evidence": PYRADIOMICS_EXACT_ALIAS_EVIDENCE,
                }
            )
    return (
        dict(support),
        dict(observations),
        external_codes,
        unmapped,
        excluded,
        exact_aliases,
        mapped_native_names,
    )


def _unique_observation(
    entries: Sequence[tuple[str, Optional[float]]],
) -> tuple[str, Optional[float], str]:
    if not entries:
        return "missing", None, ""
    finite_entries = [(name, value) for name, value in entries if value is not None]
    if not finite_entries:
        return "nonfinite", None, ", ".join(name for name, _ in entries)
    distinct = {float(value) for _, value in finite_entries if value is not None}
    names = ", ".join(name for name, _ in finite_entries)
    if len(distinct) > 1:
        rendered = "; ".join(f"{name}={value!r}" for name, value in finite_entries)
        return "ambiguous", None, rendered
    return "finite", next(iter(distinct)), names


def evaluate_adapter_payload(
    *,
    adapter: str,
    payload: Mapping[str, Any],
    references: Iterable[ReferenceRecord],
    release_version: Optional[str] = None,
) -> tuple[list[ComparisonRecord], dict[str, Any]]:
    """Evaluate one isolated adapter payload without hiding unsupported/missing rows."""

    references = list(references)
    (
        support,
        observations,
        external_codes,
        unmapped,
        excluded,
        exact_aliases,
        mapped_native_names,
    ) = _payload_surface(adapter, payload)
    software = payload.get("software", {})
    reported_version = (
        str(software.get("version", "unknown"))
        if isinstance(software, Mapping)
        else "unknown"
    )
    # Report the reviewed release pin.  This is normally identical to package
    # metadata, but PyRadiomics' official 3.1.0 tag still exposes stale
    # ``3.0.1a1`` distribution metadata.  Keep both values in the audit rather
    # than mislabelling the release used for the comparison.
    version = str(release_version or reported_version)
    output: list[ComparisonRecord] = []
    for reference in references:
        semantic_key = reference.semantic_key
        native_names = support.get(semantic_key, []) if semantic_key else []
        observed_supported = bool(native_names)
        observation_status, value, detail = _unique_observation(
            observations.get(semantic_key, [])
        )
        # Presence on the direct output surface, or an audited exact identity
        # backed by that surface, proves that the package invoked the source
        # calculation even when serialising its numeric value failed.
        attempted = observed_supported
        finite = observation_status == "finite"
        referencable = bool(
            reference.standardized
            and reference.reference_value is not None
            and reference.tolerance is not None
        )
        evaluated = False
        passed: Optional[bool] = None
        status = "unsupported"
        result = None

        if not reference.standardized:
            status = "not_standardized"
        elif not semantic_key:
            status = "unmapped_reference"
        elif not observed_supported:
            status = "unsupported"
        elif observation_status == "missing":
            status = "missing"
        elif observation_status == "nonfinite":
            status = "nonfinite"
        elif observation_status == "ambiguous":
            status = "ambiguous"
        elif not referencable:
            status = "reference_unavailable"
        else:
            assert value is not None
            assert reference.reference_value is not None
            assert reference.tolerance is not None
            if reference.specification == "IBSI 1":
                result = compare_ibsi1(
                    reference.reference_value, value, reference.tolerance
                )
            elif reference.specification == "IBSI 2":
                result = compare_absolute(
                    reference.reference_value, value, reference.tolerance
                )
            else:
                raise ValueError(
                    f"Unsupported specification: {reference.specification}"
                )
            evaluated = True
            passed = result.passed
            status = "pass" if result.passed else "fail"

        output.append(
            ComparisonRecord(
                specification=reference.specification,
                phase=reference.phase,
                adapter=adapter,
                software_version=version,
                configuration=reference.configuration,
                profile=reference.profile,
                aggregation=reference.aggregation,
                family=reference.family,
                feature_name=reference.feature_name,
                feature_tag=reference.feature_tag,
                semantic_key=reference.semantic_key,
                ibsi_code=reference.ibsi_code,
                standardized=reference.standardized,
                observed_supported=observed_supported,
                mapped=bool(semantic_key and observed_supported),
                attempted=attempted,
                finite=finite,
                referencable=referencable,
                evaluated=evaluated,
                passed=passed,
                status=status,
                native_feature_names=", ".join(native_names),
                value=value,
                reference_value=reference.reference_value,
                tolerance=reference.tolerance,
                raw_abs_error=result.raw_abs_error if result else None,
                comparison_error=result.comparison_error if result else None,
                error_tolerance_ratio=result.error_tolerance_ratio if result else None,
                comparison_policy=result.policy if result else "",
                detail=detail,
                strict_passed=result.strict_passed if result else None,
            )
        )

    audit = {
        "adapter": adapter,
        "software_version": version,
        "distribution_metadata_version": reported_version,
        "native_feature_names": len(mapped_native_names),
        "semantic_mapping_assignments": sum(len(names) for names in support.values()),
        "unique_mapped_semantic_features": len(support),
        "documented_exact_semantic_aliases": exact_aliases,
        "semantic_to_external_code": dict(sorted(external_codes.items())),
        "unmapped_native_features": sorted(unmapped),
        "excluded_native_features": sorted(excluded),
        "mapping_collisions": {
            external_codes[semantic_key]: names
            for semantic_key, names in support.items()
            if len(names) > 1
        },
    }
    return output, audit


def compare_response_maps(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Apply the official IBSI 2 Phase 1 all-voxel one-percent rule.

    The final publication and official reference-data README define tolerance
    from the reference-map range.  An older analysis script used a joint range;
    that value is retained only as a diagnostic and cannot relax the decision.
    """

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "nibabel and numpy are required for response-map evaluation"
        ) from exc

    reference_image = nib.load(str(reference_path))
    candidate_image = nib.load(str(candidate_path))
    reference = np.asarray(reference_image.dataobj, dtype=np.float64)
    candidate = np.asarray(candidate_image.dataobj, dtype=np.float64)
    if reference.shape != candidate.shape:
        return {
            "status": "geometry_mismatch",
            "passed": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "detail": "array shapes differ",
        }
    if not np.allclose(
        reference_image.affine, candidate_image.affine, rtol=0.0, atol=1e-6
    ):
        return {
            "status": "geometry_mismatch",
            "passed": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "detail": "NIfTI affines differ by more than 1e-6",
        }
    if not np.isfinite(reference).all():
        raise ValueError(
            f"Reference response map contains non-finite voxels: {reference_path}"
        )
    if reference.size == 0:
        raise ValueError(f"Reference response map is empty: {reference_path}")
    finite_candidate = np.isfinite(candidate)
    reference_range = float(reference.max() - reference.min())
    finite_candidate_values = candidate[finite_candidate]
    candidate_range = (
        float(finite_candidate_values.max() - finite_candidate_values.min())
        if finite_candidate_values.size
        else None
    )
    joint_minimum = float(reference.min())
    joint_maximum = float(reference.max())
    if finite_candidate_values.size:
        joint_minimum = min(joint_minimum, float(finite_candidate_values.min()))
        joint_maximum = max(joint_maximum, float(finite_candidate_values.max()))
    joint_diagnostic_range = joint_maximum - joint_minimum
    comparison_range = reference_range
    tolerance = 0.01 * reference_range
    errors = np.full(reference.shape, np.inf, dtype=np.float64)
    errors[finite_candidate] = np.abs(
        candidate[finite_candidate] - reference[finite_candidate]
    )
    failing = errors > tolerance
    failing_count = int(np.count_nonzero(failing))
    voxel_count = int(reference.size)
    nonfinite_count = int(np.count_nonzero(~finite_candidate))
    max_error = None if nonfinite_count else float(np.max(errors))
    passed = failing_count == 0
    return {
        "status": "pass"
        if passed
        else "nonfinite_candidate"
        if nonfinite_count
        else "fail",
        "passed": passed,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_range": reference_range,
        "candidate_finite_range": candidate_range,
        "comparison_range": comparison_range,
        "joint_intensity_range_diagnostic": joint_diagnostic_range,
        "voxel_tolerance": tolerance,
        "max_abs_error": max_error,
        "failing_voxel_count": failing_count,
        "voxel_count": voxel_count,
        "failing_voxel_fraction": failing_count / voxel_count if voxel_count else None,
        "nonfinite_candidate_voxels": nonfinite_count,
        "comparison_policy": IBSI2_PHASE1_COMPARISON_RULE,
    }


def evaluate_ibsi2_phase1_candidates(
    *,
    adapter: str,
    reference_manifest: Mapping[str, Any],
    reference_root: Path,
    candidate_paths: Mapping[str, Sequence[Path]],
    candidate_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Evaluate an explicit, provenance-validated candidate path mapping."""

    reference_paths = _validated_phase1_reference_paths(
        reference_manifest, Path(reference_root)
    )
    rows: list[dict[str, Any]] = []
    metadata_by_test = candidate_metadata or {}
    for test_id in IBSI2_PHASE1_TEST_IDS:
        candidates = list(candidate_paths.get(test_id, []))
        candidate_info = metadata_by_test.get(test_id, {})
        standardized = test_id not in IBSI2_PHASE1_NONSTANDARDIZED_IDS
        base = {
            "specification": "IBSI 2",
            "phase": "phase1",
            "adapter": adapter,
            "test_id": test_id,
            "defined": True,
            "standardized": standardized,
            "candidate_supplied": bool(candidates),
            # Machine field for summary tables.  Publication reports must use
            # candidate_supplied because an externally supplied map does not by
            # itself establish native adapter support.
            "supported": bool(candidates),
            "referencable": test_id in reference_paths,
            "evaluated": False,
            "passed": None,
            "candidate_path": (
                str(
                    candidate_info.get("response_map_path") or Path(candidates[0]).name
                ).strip()
                if len(candidates) == 1
                else ""
            ),
            "generator_distribution": str(
                candidate_info.get("generator_distribution", "")
            ),
            "generator_version": str(candidate_info.get("generator_version", "")),
            "filter_config_revision": str(
                candidate_info.get("filter_config_revision", "")
            ),
            "response_map_sha256": str(candidate_info.get("response_map_sha256", "")),
        }
        if not candidates:
            base.update(
                status="candidate_not_supplied", detail="no candidate response map"
            )
        elif len(candidates) > 1:
            base.update(status="ambiguous", detail="multiple candidate response maps")
        elif not standardized:
            base.update(
                status="not_standardized", detail="test has no consensus reference map"
            )
        elif test_id not in reference_paths:
            base.update(
                status="reference_unavailable",
                detail="validated reference manifest lacks test",
            )
        else:
            result = compare_response_maps(reference_paths[test_id], candidates[0])
            base.update(result)
            base["evaluated"] = True
        rows.append(base)
    return rows


def _validated_phase1_reference_paths(
    manifest: Mapping[str, Any],
    reference_root: Path,
) -> dict[str, Path]:
    """Resolve and checksum every standardized Phase 1 response map."""

    if (
        manifest.get("schema_version") != 1
        or manifest.get("specification") != "IBSI 2"
        or manifest.get("phase") != "phase1"
    ):
        raise RunIntegrityError("Invalid IBSI 2 Phase 1 reference manifest identity")
    if manifest.get("defined_tests") != len(IBSI2_PHASE1_TEST_IDS):
        raise RunIntegrityError(
            "IBSI 2 Phase 1 manifest has the wrong defined-test count"
        )
    expected_nonstandardized = list(IBSI2_PHASE1_NONSTANDARDIZED_IDS)
    if (
        manifest.get("standardized_reference_tests")
        != len(IBSI2_PHASE1_TEST_IDS) - len(IBSI2_PHASE1_NONSTANDARDIZED_IDS)
        or manifest.get("nonstandardized_tests") != expected_nonstandardized
        or manifest.get("comparison_rule") != IBSI2_PHASE1_COMPARISON_RULE
    ):
        raise RunIntegrityError(
            "IBSI 2 Phase 1 manifest has invalid standardized-test metadata"
        )
    source = manifest.get("source")
    if not isinstance(source, Mapping) or (
        source.get("reference_repository") != IBSI2_REFERENCE_REPOSITORY
        or source.get("reference_commit") != IBSI2_REFERENCE_COMMIT
        or source.get("reference_readme_sha256") != IBSI2_REFERENCE_README_SHA256
        or source.get("hash_verified") is not True
        or source.get("analysis_repository") != IBSI2_ANALYSIS_REPOSITORY
        or source.get("analysis_commit") != IBSI2_ANALYSIS_COMMIT
        or source.get("comparison_source") != IBSI2_PHASE1_COMPARISON_SOURCE
        or source.get("comparison_source_sha256")
        != IBSI2_PHASE1_COMPARISON_SOURCE_SHA256
    ):
        raise RunIntegrityError(
            "IBSI 2 Phase 1 manifest lacks the reviewed comparison-source provenance"
        )
    maps = manifest.get("maps")
    if not isinstance(maps, list):
        raise RunIntegrityError("IBSI 2 Phase 1 reference manifest maps must be a list")
    root = Path(reference_root).expanduser().resolve()
    resolved: dict[str, Path] = {}
    for entry in maps:
        if not isinstance(entry, Mapping):
            raise RunIntegrityError("IBSI 2 Phase 1 map record must be an object")
        test_id = str(entry.get("test_id", "")).casefold()
        relative = str(entry.get("path", ""))
        expected_hash = str(entry.get("sha256", ""))
        if test_id not in set(IBSI2_PHASE1_TEST_IDS).difference(
            IBSI2_PHASE1_NONSTANDARDIZED_IDS
        ):
            raise RunIntegrityError(
                f"Unexpected standardized Phase 1 test ID: {test_id}"
            )
        if test_id in resolved:
            raise RunIntegrityError(f"Duplicate Phase 1 reference map: {test_id}")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise RunIntegrityError(
                "Phase 1 reference paths must remain inside reference_root"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        reviewed_hash = IBSI2_PHASE1_REFERENCE_SHA256.get(test_id, "")
        if expected_hash != reviewed_hash:
            raise RunIntegrityError(
                f"Phase 1 reference manifest is not pinned to the reviewed hash: {test_id}"
            )
        if not expected_hash or sha256_file(path) != expected_hash:
            raise RunIntegrityError(f"Phase 1 reference checksum mismatch: {test_id}")
        resolved[test_id] = path
    expected_ids = set(IBSI2_PHASE1_TEST_IDS).difference(
        IBSI2_PHASE1_NONSTANDARDIZED_IDS
    )
    if set(resolved) != expected_ids:
        raise RunIntegrityError(
            "Phase 1 reference manifest must contain the exact 33 standardized maps"
        )
    return resolved
