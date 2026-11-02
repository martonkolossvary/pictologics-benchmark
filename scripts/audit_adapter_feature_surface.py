#!/usr/bin/env python3
"""Smoke-audit every adapter family against the frozen benchmark inputs.

It runs each adapter's timed code path once, without a warmup and without a
ledger, so no benchmark observation is created.  It verifies input routing,
finite native output, and retention of every native feature name used by the
reviewed IBSI 1 compliance mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench import env as benchmark_env
from bench.adapters.registry import get_adapter
from bench.benchmark_ledger import atomic_write_json, atomic_write_text
from bench.benchmark_representations import (
    HARMONIZED_INPUT_CONTRACT,
    select_representation,
)
from bench.dataset_manifest import sha256_file
from bench.ibsi_families import (
    GROUP_ORDER,
    WORKLOAD_GROUPS,
)
from bench.ibsi_mapping import classify_feature
from bench.run import run_adapter_process
from scripts.launch_benchmark import (
    _host_profile_preflight,
    _load_host_profile,
)


ADAPTERS = ("pictologics", "pyradiomics", "mirp", "medimage", "zrad")
DEFAULT_EXPECTED_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "reproducibility"
    / "contracts"
    / "adapter_feature_surface.csv"
)


def _load_expected(path: Path) -> dict[tuple[str, str], dict[str, bool]]:
    expected: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            adapter = str(row.get("adapter") or "").strip().lower()
            family = str(row.get("family") or "").strip().lower()
            native_names = str(row.get("native_feature_names") or "").strip()
            if adapter in ADAPTERS and family and native_names:
                # One semantic row may be supported by multiple independently
                # returned source values (for example MIRP 2.7 energy with and
                # without its offset).  The report's public delimiter is
                # comma-space; treating the entire cell as one name would hide
                # a real loss from the executable feature surface.
                for native_name in native_names.split(", "):
                    suffix = " [documented exact alias]"
                    documented_alias = native_name.endswith(suffix)
                    if documented_alias:
                        native_name = native_name[: -len(suffix)]
                    expected[(adapter, family)][native_name] = documented_alias
    return expected


def _normalize_case(dataset_dir: Path, raw: dict[str, Any]) -> dict[str, Any]:
    case = dict(raw)
    case["image_abs"] = str((dataset_dir / str(case["image_path"])).resolve())
    case["mask_abs"] = str((dataset_dir / str(case["mask_path"])).resolve())
    case["discrete_image_abs"] = str(
        (dataset_dir / str(case["discrete_image_path"])).resolve()
    )
    case["ivh_image_abs"] = str((dataset_dir / str(case["ivh_image_path"])).resolve())
    return case


def audit(
    *,
    dataset_dir: Path,
    case_id: str,
    expected_contract: Path,
    output_dir: Path,
    timeout: float,
    host_profile_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_case = next(
        (case for case in manifest["cases"] if case.get("case_id") == case_id),
        None,
    )
    if raw_case is None:
        raise ValueError(f"case not found in manifest: {case_id}")
    case = _normalize_case(dataset_dir, raw_case)
    expected = _load_expected(expected_contract)
    profiles = benchmark_env.load_runtime_profiles()
    configured_profiles = {
        adapter: {
            "distribution": profiles[adapter].distribution,
            "release_version": profiles[adapter].version,
            "distribution_metadata_version": (
                profiles[adapter].metadata_version or profiles[adapter].version
            ),
            "python": profiles[adapter].python,
            "upstream": profiles[adapter].upstream,
            "verified_latest_stable": profiles[adapter].verified_latest_stable,
            "source_commit": profiles[adapter].source_commit or None,
        }
        for adapter in ADAPTERS
    }
    host_profile = (
        _load_host_profile(host_profile_path) if host_profile_path is not None else None
    )
    host_preflight = _host_profile_preflight(host_profile)

    # The surface is checked through the timed group calls of the contract.
    family_order = GROUP_ORDER

    def expected_for(adapter: str, workload: str) -> dict[str, bool]:
        merged: dict[str, bool] = {}
        for member in WORKLOAD_GROUPS.get(workload, (workload,)):
            merged.update(expected.get((adapter, member), {}))
        return merged

    records: list[dict[str, Any]] = []
    versions: dict[str, str] = {}
    all_native_names: dict[str, set[str]] = defaultdict(set)
    for adapter in ADAPTERS:
        capabilities = get_adapter(adapter)
        for family in family_order:
            if not capabilities.supports(family):
                records.append(
                    {
                        "adapter": adapter,
                        "family": family,
                        "status": "unsupported_declared",
                        "representation_id": None,
                        "discretization": None,
                        "native_output_count": 0,
                        "finite_output_count": 0,
                        "mapped_ibsi_code_count": 0,
                        "excluded_output_count": 0,
                        "unmapped_output_count": 0,
                        "expected_compliance_native_name_count": 0,
                        "missing_compliance_native_names": [],
                        "native_feature_names": [],
                        "finite_native_feature_names": [],
                        "ibsi_feature_classifications": [],
                    }
                )
                continue

            representation = select_representation(
                case,
                family,
                input_contract=HARMONIZED_INPUT_CONTRACT,
                internal_discretization="fbn",
                internal_bins=32,
                internal_bin_width=32.0,
                internal_intensity_min=None,
                internal_intensity_max=None,
            )
            payload, _ = run_adapter_process(
                adapter,
                image=representation.image_path,
                mask=str(case["mask_abs"]),
                image_sha256=representation.image_sha256,
                source_image_sha256=str(case["image_sha256"]),
                mask_sha256=str(case["mask_sha256"]),
                input_contract=HARMONIZED_INPUT_CONTRACT,
                input_representation_id=representation.representation_id,
                representation_derivation_sha256=representation.derivation_sha256,
                configured_levels=representation.configured_levels,
                occupied_levels=representation.occupied_levels,
                modality=str(case.get("modality") or "") or None,
                discretization=representation.discretization,
                bins=representation.bins,
                bin_width=representation.bin_width,
                intensity_min=representation.intensity_min,
                intensity_max=representation.intensity_max,
                families=capabilities.supported_members(family),
                iterations=1,
                no_warmup=True,
                include_values=True,
                timed=True,
                timeout=timeout,
            )
            versions[adapter] = str(payload["software"]["version"])
            names = [str(name) for name in payload["features"]["all"]]
            values = payload.get("values", {}).get("all", {})
            finite_names = {
                name
                for name in names
                if isinstance(values.get(name), (int, float))
                and not isinstance(values.get(name), bool)
                and math.isfinite(float(values[name]))
            }
            all_native_names[adapter].update(names)
            classifications = [classify_feature(adapter, name) for name in names]
            mapped_codes = {
                code for code, status in classifications if code and status == "mapped"
            }
            excluded = sum(status == "excluded" for _, status in classifications)
            unmapped = sum(status == "unmapped" for _, status in classifications)
            expected_names = expected_for(adapter, family)
            records.append(
                {
                    "adapter": adapter,
                    "family": family,
                    "status": "pending_cross_family_alias_check",
                    "representation_id": representation.representation_id,
                    "discretization": representation.discretization,
                    "native_output_count": len(names),
                    "finite_output_count": len(finite_names),
                    "mapped_ibsi_code_count": len(mapped_codes),
                    "excluded_output_count": excluded,
                    "unmapped_output_count": unmapped,
                    "expected_compliance_native_name_count": len(expected_names),
                    "missing_compliance_native_names": [],
                    "native_feature_names": sorted(names),
                    "finite_native_feature_names": sorted(finite_names),
                    "ibsi_feature_classifications": [
                        {
                            "native_feature_name": name,
                            "ibsi_code": code,
                            "status": status,
                        }
                        for name, (code, status) in sorted(
                            zip(names, classifications), key=lambda item: item[0]
                        )
                    ],
                    "_finite_complete": len(finite_names) == len(names),
                }
            )

    failures: list[str] = []
    if host_preflight and host_preflight["status"] != "pass":
        failures.append("host_profile")
    for adapter in ADAPTERS:
        expected_version = configured_profiles[adapter]["distribution_metadata_version"]
        if versions.get(adapter) != expected_version:
            failures.append(f"{adapter}/software_version")
    for record in records:
        if record["status"] == "unsupported_declared":
            continue
        adapter = str(record["adapter"])
        family = str(record["family"])
        family_names = set(record["native_feature_names"])
        finite_complete = bool(record.pop("_finite_complete"))
        missing = []
        for name, documented_alias in expected_for(adapter, family).items():
            available = all_native_names[adapter] if documented_alias else family_names
            if name not in available:
                missing.append(
                    name + (" [documented exact alias]" if documented_alias else "")
                )
        record["missing_compliance_native_names"] = sorted(missing)
        record["status"] = "pass" if finite_complete and not missing else "fail"
        if record["status"] == "fail":
            failures.append(f"{adapter}/{family}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "purpose": "adapter_feature_surface_smoke_audit_timed_path_no_ledger",
        "benchmark_observations_created": False,
        "input_contract": HARMONIZED_INPUT_CONTRACT,
        "dataset_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "case_id": case_id,
        },
        "expected_feature_contract": {
            "path": str(expected_contract.resolve()),
            "sha256": sha256_file(expected_contract),
        },
        "configured_adapter_profiles": configured_profiles,
        "host_profile": host_preflight,
        # For the official PyRadiomics 3.1.0 tag the reported version is the
        # stale upstream distribution metadata 3.0.1a1; the configured release
        # version stays explicit above.
        "software_reported_versions": versions,
        "failure_workloads": failures,
        "records": records,
    }
    atomic_write_json(output_dir / "adapter_feature_surface_audit.json", result)

    fields = [
        "adapter",
        "family",
        "status",
        "representation_id",
        "discretization",
        "native_output_count",
        "finite_output_count",
        "mapped_ibsi_code_count",
        "excluded_output_count",
        "unmapped_output_count",
        "expected_compliance_native_name_count",
        "missing_compliance_native_names",
        "native_feature_names",
        "finite_native_feature_names",
        "ibsi_feature_classifications",
    ]
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        row = dict(record)
        row["missing_compliance_native_names"] = "|".join(
            record["missing_compliance_native_names"]
        )
        row["native_feature_names"] = "|".join(record["native_feature_names"])
        row["finite_native_feature_names"] = "|".join(
            record["finite_native_feature_names"]
        )
        row["ibsi_feature_classifications"] = json.dumps(
            record["ibsi_feature_classifications"],
            sort_keys=True,
            separators=(",", ":"),
        )
        writer.writerow(row)
    atomic_write_text(
        output_dir / "adapter_feature_surface_audit.csv", output.getvalue()
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="p1_reference_noise_m1_n032")
    parser.add_argument(
        "--expected-contract",
        type=Path,
        default=DEFAULT_EXPECTED_CONTRACT,
        help="Checked-in frozen native feature-name contract",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--host-profile", type=Path, default=None)
    args = parser.parse_args()
    result = audit(
        dataset_dir=args.dataset_dir.resolve(),
        case_id=args.case_id,
        expected_contract=args.expected_contract.resolve(),
        output_dir=args.output_dir.resolve(),
        timeout=args.timeout,
        host_profile_path=(args.host_profile.resolve() if args.host_profile else None),
    )
    print(
        json.dumps(
            {"status": result["status"], "failures": result["failure_workloads"]}
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
