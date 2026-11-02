from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.adapters.protocol import ADAPTER_PROTOCOL_VERSION, resolve_aggregation
from bench.compliance.evaluate import (
    _validated_phase1_reference_paths,
    compare_response_maps,
    evaluate_adapter_payload,
)
from bench.compliance.models import ComparisonRecord, ReferenceRecord
from bench.compliance.ibsi2_protocol import (
    IBSI2_PROTOCOL_REVIEW,
    PHASE1_FILTER_SPECS_BY_ID,
    PHASE2_FILTER_SPECS_BY_ID,
)
from bench.compliance.references import (
    IBSI_DATA_COMMIT,
    IBSI_DATA_REPOSITORY,
    IBSI1_DIGITAL_PHANTOM_IMAGE_SHA256,
    IBSI1_DIGITAL_PHANTOM_MASK_SHA256,
    IBSI1_BENCHMARK_INSTANCES_BY_FAMILY,
    IBSI1_TABLE2_DEFINITION_COUNT,
    IBSI1_TABLE2_FAMILY_DEFINITIONS,
    IBSI_CODE_TO_SEMANTIC_KEY,
    IBSI2_ANALYSIS_COMMIT,
    IBSI2_ANALYSIS_REPOSITORY,
    IBSI2_REFERENCE_COMMIT,
    IBSI2_REFERENCE_README_SHA256,
    IBSI2_REFERENCE_REPOSITORY,
    IBSI2_PHASE1_COMPARISON_RULE,
    IBSI2_PHASE1_COMPARISON_SOURCE,
    IBSI2_PHASE1_COMPARISON_SOURCE_SHA256,
    IBSI2_PHASE1_NONSTANDARDIZED_IDS,
    IBSI2_PHASE1_REFERENCE_SHA256,
    IBSI2_PHASE1_TEST_IDS,
    IBSI2_PHASE1_STANDARDIZED_IDS,
    IBSI2_PHASE2_DEFINED_CHECKS,
    IBSI2_PHASE2_DEFINED_FILTER_IDS,
    IBSI2_PHASE2_NONSTANDARDIZED_CHECKS,
    IBSI2_PHASE2_NONSTANDARDIZED_FILTER_IDS,
    IBSI2_PHASE2_NONSTANDARDIZED_PUBLISHED_PAIR,
    IBSI2_PHASE2_PUBLISHED_FILTER_IDS,
    IBSI2_PHASE2_PUBLISHED_REFERENCE_ROWS,
    IBSI2_PHASE2_STANDARDIZED_CHECKS,
    IBSI2_PHASE2_TAGS,
    ReferenceValidationError,
    import_ibsi1_workbook,
    import_ibsi2_phase2_csv,
    ibsi1_table2_definition_key,
    load_reference_csv,
    semantic_key_from_tag,
    select_ibsi1_configuration_profile,
    select_ibsi1_digital_phantom_profile,
    validate_reference_table_manifest,
    validate_ibsi2_phase1_bundle,
)
from bench.benchmark_ledger import (
    RunIntegrityError,
    atomic_write_json,
    sha256_file,
)
from bench.compliance.report import _overall_summary
from bench.compliance.run import (
    IBSI2_CANDIDATE_SCHEMA_VERSION,
    _json_safe_payload,
    _prepare_state,
    _validate_official_ibsi1_digital_phantom,
    _validate_compliance_payload,
    configured_adapter_profiles,
    load_ibsi2_phase1_candidate_manifest,
    load_ibsi2_phase2_candidate_manifest,
)
from bench.compliance.tolerance import compare_absolute, compare_ibsi1
from bench.ibsi_families import CODE_TO_FAMILY


PICTOLOGICS_VERSION = configured_adapter_profiles(["pictologics"])["pictologics"][
    "version"
]


def reference_record(
    *,
    code: str = "Q4LE",
    family: str = "intensity",
    tag: str = "stat_mean",
    aggregation: str = "not_applicable",
    standardized: bool = True,
    specification: str = "IBSI 1",
) -> ReferenceRecord:
    return ReferenceRecord(
        specification=specification,
        phase="phase1",
        dataset="digital phantom",
        configuration="digital_phantom",
        profile="digital_phantom_3d_merged",
        in_profile=True,
        aggregation=aggregation,
        family=family,
        feature_name="Mean",
        feature_tag=tag,
        semantic_key="intensity_statistics.mean",
        ibsi_code=code,
        consensus="standardized" if standardized else "not standardized",
        reference_value=10.0 if standardized else None,
        tolerance=1.0 if standardized else None,
        standardized=standardized,
        source_sheet="digital phantom",
        source_row=2,
    )


class TolerancePolicyTests(unittest.TestCase):
    def test_ibsi1_truncates_difference_without_rounding_measurement_first(
        self,
    ) -> None:
        result = compare_ibsi1(556.0, 551.1, 4.0)
        self.assertAlmostEqual(result.raw_abs_error, 4.9)
        self.assertEqual(result.comparison_error, 4.0)
        self.assertTrue(result.passed)
        # A non-zero reference keeps the workbook's three-digit precision.
        self.assertTrue(result.strict_passed)

    def test_ibsi1_zero_reference_reproduces_workbook_fallback(self) -> None:
        result = compare_ibsi1(0.0, 0.9, 0.0)
        self.assertEqual(result.raw_abs_error, 0.9)
        self.assertEqual(result.comparison_error, 0.0)
        self.assertTrue(result.passed)
        self.assertFalse(result.strict_passed)

    def test_ibsi2_uses_raw_absolute_error(self) -> None:
        result = compare_absolute(10.0, 10.11, 0.1)
        self.assertAlmostEqual(result.comparison_error, 0.11)
        self.assertFalse(result.passed)
        self.assertFalse(result.strict_passed)


class AggregationContractTests(unittest.TestCase):
    def test_native_directional_aggregation_is_adapter_specific(self) -> None:
        self.assertEqual(
            resolve_aggregation("pyradiomics", "native", ["glcm"]), "3d_average"
        )
        self.assertEqual(resolve_aggregation("mirp", "native", ["glcm"]), "3d_merge")

    def test_pyradiomics_declares_explicit_3d_matrix_merge(self) -> None:
        self.assertEqual(
            resolve_aggregation("pyradiomics", "3d_merge", ["glcm", "glrlm"]),
            "3d_merge",
        )
        # Aggregation does not affect a scalar-only request.
        self.assertEqual(
            resolve_aggregation("pyradiomics", "3d_merge", ["intensity"]),
            "3d_merge",
        )


class ReferenceImporterTests(unittest.TestCase):
    def test_small_workbook_preserves_all_sheets_and_configuration_keys(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workbook = Workbook()
            workbook.remove(workbook.active)
            sheets = [
                "digital phantom",
                "config A",
                "config B",
                "config C",
                "config D",
                "config E",
            ]
            header = [
                "dataset",
                "family",
                "feature",
                "consensus",
                "reference value",
                "tolerance",
                "your result",
                "difference",
                "check",
                "tag",
            ]
            for sheet_name in sheets:
                sheet = workbook.create_sheet(sheet_name)
                sheet.append(header)
                dataset = (
                    "digital phantom"
                    if sheet_name == "digital phantom"
                    else "configuration " + sheet_name[-1]
                )
                sheet.append(
                    [
                        dataset,
                        "Statistics",
                        "Mean",
                        "≥ 10",
                        10.0,
                        1.0,
                        None,
                        None,
                        None,
                        "stat_mean",
                    ]
                )
                if sheet_name != "digital phantom":
                    sheet.append(
                        [
                            dataset,
                            "Diagnostics-initial image",
                            "Image dimension x",
                            "≥ 10",
                            100.0,
                            0.0,
                            None,
                            None,
                            None,
                            "img_dim_x_init_img",
                        ]
                    )
            source = root / "references.xlsx"
            workbook.save(source)
            expected_sheets = {
                name: (1, 1) if name == "digital phantom" else (2, 2) for name in sheets
            }
            with (
                mock.patch(
                    "bench.compliance.references.IBSI1_WORKBOOK_SHEETS", expected_sheets
                ),
                mock.patch("bench.compliance.references.IBSI1_PROFILE_ROWS", 1),
                mock.patch("bench.compliance.references.IBSI1_PROFILE_STANDARDIZED", 1),
            ):
                manifest = import_ibsi1_workbook(
                    source, root / "out", require_known_hash=False
                )
            self.assertEqual(manifest["total_rows"], 11)
            self.assertEqual(manifest["diagnostic_rows"], 5)
            self.assertEqual(manifest["canonical_profile"]["standardized_rows"], 1)
            self.assertEqual(manifest["source"]["path_at_import"], "references.xlsx")
            table = root / "out" / "ibsi1_references.csv"
            records = load_reference_csv(table)
            self.assertEqual(
                sum(record.family == "diagnostics" for record in records), 5
            )
            self.assertTrue(
                all(
                    record.ibsi_code.startswith("diagnostic:")
                    for record in records
                    if record.family == "diagnostics"
                )
            )
            manifest_path = root / "out" / "ibsi1_reference_manifest.json"
            validated = validate_reference_table_manifest(
                table,
                manifest_path,
                expected_specification="IBSI 1",
                require_reviewed_source=False,
            )
            self.assertEqual(validated["reference_table"]["sha256"], sha256_file(table))
            table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ReferenceValidationError):
                validate_reference_table_manifest(
                    table,
                    manifest_path,
                    expected_specification="IBSI 1",
                    require_reviewed_source=False,
                )

    def test_phase2_import_expands_published_grid_to_full_protocol_surface(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "phase2.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter=";", lineterminator="\n")
                writer.writerow(
                    [
                        "filter_id",
                        "feature",
                        "feature_tag",
                        "consensus_value",
                        "tolerance",
                    ]
                )
                for filter_id in IBSI2_PHASE2_PUBLISHED_FILTER_IDS:
                    for feature_tag in IBSI2_PHASE2_TAGS:
                        missing = (
                            filter_id,
                            feature_tag,
                        ) == IBSI2_PHASE2_NONSTANDARDIZED_PUBLISHED_PAIR
                        writer.writerow(
                            [
                                filter_id,
                                feature_tag,
                                feature_tag,
                                "" if missing else "1.0",
                                "" if missing else "0.1",
                            ]
                        )
            manifest = import_ibsi2_phase2_csv(
                source,
                root / "out",
                require_reviewed_derived_hash=False,
            )
            self.assertEqual(
                manifest["published_reference_rows"],
                IBSI2_PHASE2_PUBLISHED_REFERENCE_ROWS,
            )
            self.assertEqual(manifest["rows"], IBSI2_PHASE2_DEFINED_CHECKS)
            self.assertEqual(
                manifest["standardized_rows"], IBSI2_PHASE2_STANDARDIZED_CHECKS
            )
            self.assertEqual(
                manifest["not_standardized_rows"],
                IBSI2_PHASE2_NONSTANDARDIZED_CHECKS,
            )
            records = load_reference_csv(root / "out" / "ibsi2_phase2_references.csv")
            self.assertEqual(len(records), IBSI2_PHASE2_DEFINED_CHECKS)
            self.assertEqual(
                {record.configuration for record in records},
                set(IBSI2_PHASE2_DEFINED_FILTER_IDS),
            )
            self.assertTrue(
                all(record.aggregation == "not_applicable" for record in records)
            )
            nonstandardized = {
                (record.configuration, record.feature_tag)
                for record in records
                if not record.standardized
            }
            expected_nonstandardized = {
                (filter_id, feature_tag)
                for filter_id in IBSI2_PHASE2_NONSTANDARDIZED_FILTER_IDS
                for feature_tag in IBSI2_PHASE2_TAGS
            }
            expected_nonstandardized.add(IBSI2_PHASE2_NONSTANDARDIZED_PUBLISHED_PAIR)
            self.assertEqual(nonstandardized, expected_nonstandardized)

    def test_phase1_standardized_set_includes_10_b_1_and_excludes_9_b_2(self) -> None:
        self.assertEqual(
            IBSI2_PHASE1_NONSTANDARDIZED_IDS,
            ("9.b.2", "10.a", "10.b.2"),
        )
        self.assertIn("10.b.1", IBSI2_PHASE1_STANDARDIZED_IDS)
        self.assertNotIn("9.b.2", IBSI2_PHASE1_STANDARDIZED_IDS)
        self.assertEqual(len(IBSI2_PHASE1_STANDARDIZED_IDS), 33)

    def test_wrong_phase1_composition_is_rejected_before_image_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_ids = [
                test_id
                for test_id in IBSI2_PHASE1_STANDARDIZED_IDS
                if test_id != "10.b.1"
            ]
            bad_ids.append("9.b.2")
            for test_id in bad_ids:
                (root / f"{test_id.replace('.', '_')}-ValidCRM.nii").write_bytes(b"")
            with self.assertRaises(ReferenceValidationError) as context:
                validate_ibsi2_phase1_bundle(root)
            message = str(context.exception)
            self.assertIn("9.b.2", message)
            self.assertIn("10.b.1", message)

    def test_required_3d_merged_profile_contains_169_standardized_instances(
        self,
    ) -> None:
        nonstandardized = {"ZH1A", "IQYR", "SWZ1", "BRI8", "9CMM"}
        records = []
        for index, (code, family) in enumerate(CODE_TO_FAMILY.items(), start=2):
            current_aggregation = (
                "3d_merge"
                if family in {"glcm", "glrlm"}
                else "3d"
                if family in {"glszm", "gldzm", "ngtdm", "ngldm"}
                else "not_applicable"
            )
            standardized = code not in nonstandardized
            records.append(
                ReferenceRecord(
                    specification="IBSI 1",
                    phase="phase1",
                    dataset="digital phantom",
                    configuration="digital_phantom",
                    profile="",
                    in_profile=False,
                    aggregation=current_aggregation,
                    family=family,
                    feature_name=code,
                    feature_tag=f"tag_{code}",
                    semantic_key=IBSI_CODE_TO_SEMANTIC_KEY[code],
                    ibsi_code=code,
                    consensus="standardized" if standardized else "not standardized",
                    reference_value=1.0 if standardized else None,
                    tolerance=0.1 if standardized else None,
                    standardized=standardized,
                    source_sheet="digital phantom",
                    source_row=index,
                )
            )
        selected = select_ibsi1_digital_phantom_profile(records)
        self.assertEqual(len(selected), 174)
        self.assertEqual(sum(record.standardized for record in selected), 169)
        with self.assertRaisesRegex(ValueError, "requires 3d_merge"):
            select_ibsi1_digital_phantom_profile(
                records, directional_aggregation="3d_average"
            )

    def test_table2_denominator_collapses_only_two_parameterized_ivh_pairs(
        self,
    ) -> None:
        self.assertEqual(IBSI1_TABLE2_DEFINITION_COUNT, 172)
        self.assertEqual(sum(IBSI1_TABLE2_FAMILY_DEFINITIONS.values()), 172)
        self.assertEqual(sum(IBSI1_BENCHMARK_INSTANCES_BY_FAMILY.values()), 174)
        self.assertEqual(IBSI1_TABLE2_FAMILY_DEFINITIONS["ivh"], 5)
        self.assertEqual(IBSI1_BENCHMARK_INSTANCES_BY_FAMILY["ivh"], 7)
        self.assertEqual(
            ibsi1_table2_definition_key("ivh.volume_at_intensity_fraction_10"),
            ibsi1_table2_definition_key("ivh.volume_at_intensity_fraction_90"),
        )

    def test_configuration_profiles_keep_official_a_and_c_denominators(self) -> None:
        nonstandardized = {"ZH1A", "IQYR", "SWZ1", "BRI8", "9CMM"}
        all_records = []
        for configuration in ("A", "C"):
            allowed = set(CODE_TO_FAMILY.values())
            for index, (code, family) in enumerate(CODE_TO_FAMILY.items(), start=2):
                if family not in allowed:
                    continue
                if family in {"glcm", "glrlm"}:
                    aggregation = "2d_average" if configuration == "A" else "3d_merge"
                elif family in {"glszm", "gldzm", "ngtdm", "ngldm"}:
                    aggregation = "2d" if configuration == "A" else "3d"
                else:
                    aggregation = "not_applicable"
                standardized = code not in nonstandardized
                all_records.append(
                    ReferenceRecord(
                        specification="IBSI 1",
                        phase="phase2",
                        dataset=f"configuration {configuration}",
                        configuration=configuration,
                        profile="",
                        in_profile=False,
                        aggregation=aggregation,
                        family=family,
                        feature_name=code,
                        feature_tag=f"tag_{code}",
                        semantic_key=f"{family}.{code.casefold()}",
                        ibsi_code=code,
                        consensus="standardized"
                        if standardized
                        else "not standardized",
                        reference_value=1.0 if standardized else None,
                        tolerance=0.1 if standardized else None,
                        standardized=standardized,
                        source_sheet=f"config {configuration}",
                        source_row=index,
                    )
                )
        profile_a = select_ibsi1_configuration_profile(
            all_records, configuration="A", directional_aggregation="2d_average"
        )
        profile_c = select_ibsi1_configuration_profile(
            all_records, configuration="C", directional_aggregation="3d_merge"
        )
        self.assertEqual(
            (len(profile_a), sum(r.standardized for r in profile_a)), (174, 169)
        )
        self.assertEqual(
            (len(profile_c), sum(r.standardized for r in profile_c)), (174, 169)
        )

    def test_semantic_keys_disambiguate_reused_external_identifiers(self) -> None:
        self.assertEqual(
            semantic_key_from_tag("ih_p10", "histogram"),
            "intensity_histogram.percentile_10",
        )
        self.assertEqual(
            semantic_key_from_tag("ih_p90", "histogram"),
            "intensity_histogram.percentile_90",
        )

    def test_phase1_manifest_paths_are_bound_to_all_33_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = []
            reviewed_hashes = {}
            for test_id in IBSI2_PHASE1_STANDARDIZED_IDS:
                path = root / f"{test_id}.nii"
                path.write_bytes(test_id.encode("ascii"))
                reviewed_hashes[test_id] = sha256_file(path)
                maps.append(
                    {
                        "test_id": test_id,
                        "path": path.name,
                        "sha256": sha256_file(path),
                    }
                )
            manifest = {
                "schema_version": 1,
                "specification": "IBSI 2",
                "phase": "phase1",
                "defined_tests": 36,
                "standardized_reference_tests": 33,
                "nonstandardized_tests": list(IBSI2_PHASE1_NONSTANDARDIZED_IDS),
                "comparison_rule": IBSI2_PHASE1_COMPARISON_RULE,
                "source": {
                    "reference_repository": IBSI2_REFERENCE_REPOSITORY,
                    "reference_commit": IBSI2_REFERENCE_COMMIT,
                    "reference_readme_sha256": IBSI2_REFERENCE_README_SHA256,
                    "hash_verified": True,
                    "analysis_repository": IBSI2_ANALYSIS_REPOSITORY,
                    "analysis_commit": IBSI2_ANALYSIS_COMMIT,
                    "comparison_source": IBSI2_PHASE1_COMPARISON_SOURCE,
                    "comparison_source_sha256": (IBSI2_PHASE1_COMPARISON_SOURCE_SHA256),
                },
                "maps": maps,
            }
            with mock.patch.dict(
                IBSI2_PHASE1_REFERENCE_SHA256,
                reviewed_hashes,
                clear=True,
            ):
                self.assertEqual(
                    len(_validated_phase1_reference_paths(manifest, root)), 33
                )
                (root / maps[0]["path"]).write_bytes(b"changed")
                with self.assertRaises(RunIntegrityError):
                    _validated_phase1_reference_paths(manifest, root)


class StrictEvaluationTests(unittest.TestCase):
    def test_reviewed_release_version_preserves_stale_distribution_metadata(
        self,
    ) -> None:
        payload = {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "software": {"version": "3.0.1a1"},
            "features": {"all": ["stat_mean"]},
            "values": {"all": {"stat_mean": 10.0}},
        }
        records, audit = evaluate_adapter_payload(
            adapter="pyradiomics",
            payload=payload,
            references=[reference_record()],
            release_version="3.1.0",
        )
        self.assertEqual(records[0].software_version, "3.1.0")
        self.assertEqual(audit["software_version"], "3.1.0")
        self.assertEqual(audit["distribution_metadata_version"], "3.0.1a1")

    def test_semantic_join_does_not_depend_on_reference_external_id(self) -> None:
        payload = {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "software": {"version": "2.6.0"},
            "features": {"all": ["stat_mean"]},
            "values": {"all": {"stat_mean": 10.0}},
        }
        reference = reference_record(code="version_scoped_external_id")
        records, _ = evaluate_adapter_payload(
            adapter="mirp", payload=payload, references=[reference]
        )
        self.assertTrue(records[0].observed_supported)
        self.assertTrue(records[0].evaluated)
        self.assertTrue(records[0].passed)

    def test_distinct_native_values_for_one_code_are_ambiguous(self) -> None:
        payload = {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "software": {"version": "2.6.0"},
            "features": {"all": ["stat_mean", "stat_mean_duplicate"]},
            "values": {"all": {"stat_mean": 10.0, "stat_mean_duplicate": 11.0}},
        }
        records, audit = evaluate_adapter_payload(
            adapter="mirp", payload=payload, references=[reference_record()]
        )
        self.assertEqual(records[0].status, "ambiguous")
        self.assertFalse(records[0].evaluated)
        self.assertIn("Q4LE", audit["mapping_collisions"])

    def test_not_standardized_is_never_promoted_to_pass(self) -> None:
        payload = {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "software": {"version": "2.6.0"},
            "features": {"all": ["stat_mean"]},
            "values": {"all": {"stat_mean": 10.0}},
        }
        records, _ = evaluate_adapter_payload(
            adapter="mirp",
            payload=payload,
            references=[reference_record(standardized=False)],
        )
        self.assertEqual(records[0].status, "not_standardized")
        self.assertIsNone(records[0].passed)

    def test_listed_feature_without_numeric_value_was_still_attempted(self) -> None:
        payload = {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "software": {"version": "2.6.0"},
            "features": {"all": ["stat_mean"]},
            "values": {"all": {}},
        }
        records, _ = evaluate_adapter_payload(
            adapter="mirp", payload=payload, references=[reference_record()]
        )
        self.assertTrue(records[0].observed_supported)
        self.assertTrue(records[0].attempted)
        self.assertEqual(records[0].status, "missing")

    def test_phase1_candidate_outlier_cannot_widen_reference_range_tolerance(
        self,
    ) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel and numpy are required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.nii.gz"
            candidate_path = root / "candidate.nii.gz"
            reference = np.asarray([0.0, 100.0], dtype=np.float32).reshape(2, 1, 1)
            candidate = np.asarray([-1.01, 100.0], dtype=np.float32).reshape(2, 1, 1)
            nib.save(nib.Nifti1Image(reference, np.eye(4)), reference_path)
            nib.save(nib.Nifti1Image(candidate, np.eye(4)), candidate_path)

            result = compare_response_maps(reference_path, candidate_path)

            self.assertAlmostEqual(result["reference_range"], 100.0)
            self.assertAlmostEqual(result["comparison_range"], 100.0)
            self.assertAlmostEqual(
                result["joint_intensity_range_diagnostic"], 101.01, places=4
            )
            self.assertAlmostEqual(result["voxel_tolerance"], 1.0)
            self.assertFalse(result["passed"])
            self.assertEqual(result["failing_voxel_count"], 1)

    def test_phase1_identical_constant_maps_pass_at_zero_tolerance(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel and numpy are required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_path = root / "reference.nii.gz"
            candidate_path = root / "candidate.nii.gz"
            data = np.full((2, 2, 2), 5.0, dtype=np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), reference_path)
            nib.save(nib.Nifti1Image(data, np.eye(4)), candidate_path)

            result = compare_response_maps(reference_path, candidate_path)

            self.assertEqual(result["comparison_range"], 0.0)
            self.assertEqual(result["joint_intensity_range_diagnostic"], 0.0)
            self.assertEqual(result["voxel_tolerance"], 0.0)
            self.assertTrue(result["passed"])


class ComplianceIntegrityTests(unittest.TestCase):
    def test_reference_rows_map_to_the_group_call_that_made_them(self) -> None:
        from dataclasses import replace

        from bench.compliance import run as compliance_run

        base = ComparisonRecord(
            specification="IBSI 1",
            phase="phase1",
            adapter="mirp",
            software_version="1",
            configuration="digital_phantom",
            profile="p",
            aggregation="not_applicable",
            family="morphology",
            feature_name="x",
            feature_tag="x",
            semantic_key="morphology.x",
            ibsi_code="RNU0",
            standardized=True,
            observed_supported=True,
            mapped=True,
            attempted=True,
            finite=True,
            referencable=True,
            evaluated=True,
            passed=True,
            status="pass",
        )
        cases = {
            ("morphology", "RNU0"): "group_morphology",
            ("morphology", "N365"): "group_spatial_morphology",
            ("glcm", "8S9J"): "group_texture",
            ("local_intensity", "0F91"): "group_local_intensity",
            ("intensity", "Q4LE"): "group_intensity",
        }
        for (family, code), group in cases.items():
            record = replace(base, family=family, ibsi_code=code)
            self.assertEqual(compliance_run._record_workload(record), group)
        failed = compliance_run._mark_execution_errors(
            [replace(base, ibsi_code="N365"), base],
            {("mirp", "group_spatial_morphology"): "stopped"},
        )
        self.assertEqual([record.status for record in failed], ["error", "pass"])

    def test_candidate_manifest_never_mixes_support_classes(self) -> None:
        from bench.compliance import run as compliance_run

        with self.assertRaisesRegex(ValueError, "never mixes"):
            compliance_run._manifest_support_class(
                {"entries": [{"support_class": "native_default"}]}
            )
        self.assertEqual(
            compliance_run._manifest_support_class(
                {
                    "support_class": "native_default",
                    "entries": [{"support_class": "native_default"}],
                }
            ),
            "native_default",
        )
        self.assertEqual(
            compliance_run._manifest_support_class({"entries": []}), "native"
        )

    def test_compliance_task_can_request_the_timed_adapter_path(self) -> None:
        from bench.compliance import run as compliance_run

        with tempfile.TemporaryDirectory() as temporary:
            state = {"tasks": {"task": {"status": "pending"}}}
            with mock.patch.object(
                compliance_run,
                "run_adapter_process",
                side_effect=RuntimeError("stop after the call"),
            ) as process:
                result = compliance_run._execute_task(
                    output_dir=Path(temporary),
                    state=state,
                    task_id="task",
                    adapter="pictologics",
                    image=Path("image.nii.gz"),
                    mask=Path("mask.nii.gz"),
                    family="glcm",
                    discretization="identity",
                    aggregation="3d_merge",
                    timeout=None,
                    stop_requested=None,
                    timed=True,
                )
        self.assertIsNone(result)
        self.assertTrue(process.call_args.kwargs["timed"])
        self.assertTrue(process.call_args.kwargs["no_warmup"])
        self.assertEqual(process.call_args.kwargs["iterations"], 1)

    def test_ibsi1_runner_accepts_only_the_pinned_official_phantom(self) -> None:
        root = Path(__file__).resolve().parents[1]
        image = root / "data/ibsi1/digital_phantom/image/phantom.nii.gz"
        mask = root / "data/ibsi1/digital_phantom/mask/mask.nii.gz"
        if not image.is_file() or not mask.is_file():
            self.skipTest(
                "external IBSI inputs are data-free by design; run the input bootstrap"
            )

        self.assertEqual(sha256_file(image), IBSI1_DIGITAL_PHANTOM_IMAGE_SHA256)
        self.assertEqual(sha256_file(mask), IBSI1_DIGITAL_PHANTOM_MASK_SHA256)
        _validate_official_ibsi1_digital_phantom(image, mask)

        with tempfile.TemporaryDirectory() as temporary:
            wrong_image = Path(temporary) / "phantom.nii.gz"
            wrong_image.write_bytes(b"not the official phantom")
            with self.assertRaisesRegex(ValueError, "image SHA-256 mismatch"):
                _validate_official_ibsi1_digital_phantom(wrong_image, mask)

    def protocol_payload(self) -> dict:
        return {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "adapter": "pictologics",
            "software": {
                "distribution": "pictologics",
                "version": PICTOLOGICS_VERSION,
            },
            "selection": {
                "requested_families": ["intensity"],
                "unsupported_families": [],
                "mode": "native",
            },
            "features": {"all": ["stat_mean"]},
            "values": {"all": {"stat_mean": 1.0}},
            "metadata": {
                "preprocessing": {
                    "discretization": "identity",
                    "bins": 32,
                    "bin_width": 32.0,
                    "intensity_range": None,
                },
                "aggregation": {
                    "requested": "3d_merge",
                    "effective_directional": "3d_merge",
                },
            },
        }

    def _write_phase1_candidate_manifest(
        self,
        root: Path,
        *,
        generator_version: str,
        response_bytes: bytes | None = None,
    ) -> tuple[Path, str]:
        project_root = Path(__file__).resolve().parents[1]
        specification = PHASE1_FILTER_SPECS_BY_ID["1.a.1"]
        official_source = (
            project_root
            / "data"
            / "ibsi2"
            / "source"
            / (specification.source_image_relative_path)
        )
        if not official_source.is_file():
            self.skipTest(
                "external IBSI inputs are data-free by design; run the input bootstrap"
            )
        response_map = root / "1.a.1.nii.gz"
        source_image = root / "checkerboard.nii.gz"
        filter_config = root / "filter-1.a.1.json"
        preprocessing_config = root / "preprocessing-1.a.1.json"
        environment_lock = root / "environment.json"
        generator_source = root / "generator.py"
        shutil.copy2(official_source, source_image)
        if response_bytes is None:
            shutil.copy2(source_image, response_map)
        else:
            response_map.write_bytes(response_bytes)
        source_digest = sha256_file(source_image)
        filter_config.write_text(
            json.dumps(specification.filter_config()) + "\n",
            encoding="utf-8",
        )
        preprocessing_config.write_text(
            json.dumps(specification.preprocessing_config()) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(
            project_root / ".venvs" / "adapters" / "pictologics" / "environment.json",
            environment_lock,
        )
        generator_source.write_text(
            '"""Unit-test response-map generator evidence."""\n',
            encoding="utf-8",
        )
        generator_source_digest = sha256_file(generator_source)
        manifest = {
            "schema_version": IBSI2_CANDIDATE_SCHEMA_VERSION,
            "kind": "ibsi2_phase1_response_maps",
            "adapters": ["pictologics"],
            "support_declarations": [
                {
                    "adapter": "pictologics",
                    "test_id": test_id,
                    "native_supported": test_id == "1.a.1",
                    "reason": (
                        "unit-test supported filter"
                        if test_id == "1.a.1"
                        else "unit-test reviewed unsupported filter"
                    ),
                    "evidence": "unit-test adapter capability audit",
                }
                for test_id in IBSI2_PHASE1_TEST_IDS
            ],
            "protocol_review": {
                "status": "reviewed",
                "reviewed_against": IBSI2_PROTOCOL_REVIEW,
                "reviewed_by": "unit-test reviewer",
                "reviewed_at": "2026-07-21",
            },
            "source_data": {
                "repository": IBSI_DATA_REPOSITORY,
                "commit": IBSI_DATA_COMMIT,
            },
            "entries": [
                {
                    "adapter": "pictologics",
                    "test_id": "1.a.1",
                    "response_map_path": response_map.name,
                    "response_map_sha256": sha256_file(response_map),
                    "source_image_path": source_image.name,
                    "source_image_sha256": source_digest,
                    "generator_distribution": "pictologics",
                    "generator_version": generator_version,
                    "generator_source_revision": f"sha256:{generator_source_digest}",
                    "generator_entrypoint": "bench.adapters.pictologics_adapter",
                    "generator_command": "python -m bench.adapters.pictologics_adapter",
                    "generator_source_path": generator_source.name,
                    "generator_source_sha256": generator_source_digest,
                    "executed_parameters": {
                        "filter": "mean",
                        "dimensionality": 3,
                        "boundary": "zero",
                        "support": 15,
                    },
                    "boundary_execution": {
                        "policy": "protocol_explicit",
                        "selected": "zero",
                        "effective": "zero",
                        "implementation": "as_specified",
                    },
                    "native_capability": {
                        "schema_version": "1.1.0",
                        "filter": "mean",
                        "supported_boundaries": [
                            "ZERO",
                            "NEAREST",
                            "PERIODIC",
                            "MIRROR",
                        ],
                        "effective_boundary": "as_specified",
                        "input_dimensionality": [3],
                        "kernel_dimensionality": 3,
                        "slice_plane_execution": False,
                        "orthogonal_plane_averaging": False,
                        "structure_tensor_steering": False,
                        "anisotropic_spacing": "not_applicable",
                        "rotation_pooling": [],
                    },
                    "filter_config_revision": "ibsi2-v9-1.a.1",
                    "filter_config_path": filter_config.name,
                    "filter_config_sha256": sha256_file(filter_config),
                    "preprocessing_config_path": preprocessing_config.name,
                    "preprocessing_config_sha256": sha256_file(preprocessing_config),
                    "environment_lock_path": environment_lock.name,
                    "environment_lock_sha256": sha256_file(environment_lock),
                }
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, source_digest

    def _write_phase2_candidate_manifest(
        self,
        root: Path,
        *,
        candidate_mask_value: int = 1,
    ) -> tuple[Path, str, str]:
        project_root = Path(__file__).resolve().parents[1]
        specification = PHASE2_FILTER_SPECS_BY_ID["1.A"]
        official_source_image = (
            project_root
            / "data"
            / "ibsi2"
            / "source"
            / "phase2"
            / "image"
            / "phantom.nii.gz"
        )
        official_source_mask = (
            project_root
            / "data"
            / "ibsi2"
            / "source"
            / "phase2"
            / "mask"
            / "mask.nii.gz"
        )
        if not official_source_image.is_file() or not official_source_mask.is_file():
            self.skipTest(
                "external IBSI inputs are data-free by design; run the input bootstrap"
            )
        source_image = root / "PAT1.nii.gz"
        source_mask = root / "PAT1-mask.nii.gz"
        response_map = root / "1.A.nii.gz"
        response_mask = root / "1.A-mask.nii.gz"
        filter_input = root / "filter-input-A.nii.gz"
        shutil.copy2(official_source_image, source_image)
        shutil.copy2(official_source_mask, source_mask)
        shutil.copy2(source_image, filter_input)
        shutil.copy2(filter_input, response_map)
        if candidate_mask_value == 1:
            shutil.copy2(source_mask, response_mask)
        else:
            try:
                import nibabel as nib
                import numpy as np
            except ImportError:
                self.skipTest("nibabel and numpy are required")
            mask_nifti = nib.load(str(source_mask))
            mask_data = np.asanyarray(mask_nifti.dataobj).astype(np.uint8)
            nib.save(
                nib.Nifti1Image(mask_data * candidate_mask_value, mask_nifti.affine),
                response_mask,
            )
        filter_config = root / "filter-1.A.json"
        preprocessing_config = root / "preprocessing-A.json"
        environment_lock = root / "environment.json"
        generator_source = root / "generator.py"
        filter_config.write_text(
            json.dumps(specification.filter_config()) + "\n",
            encoding="utf-8",
        )
        preprocessing_config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "specification": "IBSI 2",
                    "phase": "phase2",
                    "reference_manual_version": "9",
                    "configuration_dimension": "A",
                    "crop": False,
                    "intensity_resegmentation_range_hu": [-1000, 400],
                    "intensity_resegmentation_source": "unfiltered_image",
                    "response_map_discretization": "none",
                    "statistics_roi": "complete_3d",
                    "resampling": {"enabled": False},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.copy2(
            project_root / ".venvs" / "adapters" / "pictologics" / "environment.json",
            environment_lock,
        )
        generator_source.write_text(
            '"""Unit-test response-map generator evidence."""\n',
            encoding="utf-8",
        )
        generator_source_digest = sha256_file(generator_source)
        source_image_digest = sha256_file(source_image)
        source_mask_digest = sha256_file(source_mask)
        manifest = {
            "schema_version": IBSI2_CANDIDATE_SCHEMA_VERSION,
            "kind": "ibsi2_phase2_response_maps",
            "adapters": ["pictologics"],
            "support_declarations": [
                {
                    "adapter": "pictologics",
                    "filter_id": filter_id,
                    "native_supported": filter_id == "1.A",
                    "reason": (
                        "unit-test supported filter"
                        if filter_id == "1.A"
                        else "unit-test reviewed unsupported filter"
                    ),
                    "evidence": "unit-test adapter capability audit",
                }
                for filter_id in IBSI2_PHASE2_DEFINED_FILTER_IDS
            ],
            "protocol_review": {
                "status": "reviewed",
                "reviewed_against": IBSI2_PROTOCOL_REVIEW,
                "reviewed_by": "unit-test reviewer",
                "reviewed_at": "2026-07-21",
            },
            "source_data": {
                "repository": IBSI_DATA_REPOSITORY,
                "commit": IBSI_DATA_COMMIT,
                "image_path": source_image.name,
                "image_sha256": source_image_digest,
                "mask_path": source_mask.name,
                "mask_sha256": source_mask_digest,
            },
            "entries": [
                {
                    "adapter": "pictologics",
                    "filter_id": "1.A",
                    "image_path": response_map.name,
                    "image_sha256": sha256_file(response_map),
                    "mask_path": response_mask.name,
                    "mask_sha256": sha256_file(response_mask),
                    "filter_input_path": filter_input.name,
                    "filter_input_sha256": sha256_file(filter_input),
                    "generator_distribution": "pictologics",
                    "generator_version": PICTOLOGICS_VERSION,
                    "generator_source_revision": f"sha256:{generator_source_digest}",
                    "generator_entrypoint": "bench.adapters.pictologics_adapter",
                    "generator_command": "python -m bench.adapters.pictologics_adapter",
                    "generator_source_path": generator_source.name,
                    "generator_source_sha256": generator_source_digest,
                    "executed_parameters": {
                        "filter": "none",
                        "dimensionality": 2,
                    },
                    "boundary_execution": {
                        "policy": "not_applicable",
                        "selected": None,
                        "effective": None,
                        "implementation": "not_applicable",
                    },
                    "native_capability": None,
                    "filter_config_revision": "ibsi2-v9-1.A",
                    "filter_config_path": filter_config.name,
                    "filter_config_sha256": sha256_file(filter_config),
                    "preprocessing_config_path": preprocessing_config.name,
                    "preprocessing_config_sha256": sha256_file(preprocessing_config),
                    "environment_lock_path": environment_lock.name,
                    "environment_lock_sha256": sha256_file(environment_lock),
                }
            ],
        }
        manifest_path = root / "phase2-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, source_image_digest, source_mask_digest

    def test_compliance_payload_is_bound_to_pin_and_requested_preprocessing(
        self,
    ) -> None:
        payload = self.protocol_payload()
        _validate_compliance_payload(
            payload,
            adapter="pictologics",
            family="intensity",
            discretization="identity",
            aggregation="3d_merge",
            bins=32,
            bin_width=32.0,
            intensity_min=None,
            intensity_max=None,
        )
        payload["software"]["version"] = "0.4.0"
        with self.assertRaises(RunIntegrityError):
            _validate_compliance_payload(
                payload,
                adapter="pictologics",
                family="intensity",
                discretization="identity",
                aggregation="3d_merge",
                bins=32,
                bin_width=32.0,
                intensity_min=None,
                intensity_max=None,
            )

    def test_compliance_rejects_native_aggregation_substitution(self) -> None:
        payload = self.protocol_payload()
        payload["metadata"]["aggregation"] = {
            "requested": "native",
            "effective_directional": "3d_merge",
        }
        with self.assertRaisesRegex(RunIntegrityError, "require 3d_merge"):
            _validate_compliance_payload(
                payload,
                adapter="pictologics",
                family="intensity",
                discretization="identity",
                aggregation="native",
                bins=32,
                bin_width=32.0,
                intensity_min=None,
                intensity_max=None,
            )

    def test_nonfinite_values_are_persisted_as_explicit_json_tokens(self) -> None:
        payload = self.protocol_payload()
        payload["values"]["all"]["stat_mean"] = math.nan
        safe = _json_safe_payload(payload)
        self.assertEqual(safe["values"]["all"]["stat_mean"], "NaN")
        json.dumps(safe, allow_nan=False)

    def test_resume_resets_interrupted_state_and_rejects_drift_or_orphans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            spec = {"kind": "unit-compliance", "input_sha256": "a" * 64}
            _, state = _prepare_state(output, spec, ["task-a"], resume=False)
            state["tasks"]["task-a"]["status"] = "running"
            atomic_write_json(output / "run_state.json", state)
            _, resumed = _prepare_state(output, spec, ["task-a"], resume=True)
            self.assertEqual(resumed["tasks"]["task-a"]["status"], "pending")
            persisted = json.loads(
                (output / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["tasks"]["task-a"]["status"], "pending")
            with self.assertRaises(RunIntegrityError):
                _prepare_state(
                    output,
                    {"kind": "unit-compliance", "input_sha256": "b" * 64},
                    ["task-a"],
                    resume=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "orphan"
            output.mkdir()
            (output / "stale.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_state(output, {"kind": "unit"}, ["task-a"], resume=False)

    def test_phase1_candidates_must_match_the_reviewed_adapter_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, source_digest = self._write_phase1_candidate_manifest(
                root,
                generator_version="0.4.0",
                response_bytes=b"candidate",
            )
            with (
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE1_SOURCE_IMAGE_SHA256",
                    frozenset({source_digest}),
                ),
                self.assertRaisesRegex(ValueError, "reviewed adapter profile"),
            ):
                load_ibsi2_phase1_candidate_manifest(manifest_path)
            self.assertEqual(
                configured_adapter_profiles(["pictologics"])["pictologics"]["version"],
                "1.0.1",
            )

    def test_phase1_candidate_manifest_accepts_exact_reviewed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_phase1_candidate_manifest(
                Path(temporary),
                generator_version=PICTOLOGICS_VERSION,
            )
            entries = load_ibsi2_phase1_candidate_manifest(manifest_path)
            self.assertEqual(entries.adapters, ("pictologics",))
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["test_id"], "1.a.1")
            self.assertEqual(
                entries[0]["source_image_sha256"],
                PHASE1_FILTER_SPECS_BY_ID["1.a.1"].source_image_sha256,
            )

    def test_phase1_candidate_rejects_stale_protocol_review_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_phase1_candidate_manifest(
                Path(temporary),
                generator_version=PICTOLOGICS_VERSION,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol_review"]["reviewed_against"] = (
                "IBSI 2 reference manual v5"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be bound"):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase1_candidate_rejects_capability_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._write_phase1_candidate_manifest(
                Path(temporary),
                generator_version=PICTOLOGICS_VERSION,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["native_capability"]["supported_boundaries"] = [
                "MIRROR"
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "absent from Pictologics"):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase1_candidate_rejects_reviewed_parameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._write_phase1_candidate_manifest(
                root,
                generator_version=PICTOLOGICS_VERSION,
            )
            filter_path = root / "filter-1.a.1.json"
            config = json.loads(filter_path.read_text(encoding="utf-8"))
            config["parameters"]["support"] = 13
            filter_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["filter_config_sha256"] = sha256_file(filter_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "differs from v9"):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase1_candidate_rejects_wrong_official_phantom_for_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._write_phase1_candidate_manifest(
                root,
                generator_version=PICTOLOGICS_VERSION,
            )
            project_root = Path(__file__).resolve().parents[1]
            impulse = PHASE1_FILTER_SPECS_BY_ID["2.a"]
            source_path = root / "checkerboard.nii.gz"
            shutil.copy2(
                project_root
                / "data"
                / "ibsi2"
                / "source"
                / (impulse.source_image_relative_path),
                source_path,
            )
            wrong_digest = sha256_file(source_path)
            filter_path = root / "filter-1.a.1.json"
            config = json.loads(filter_path.read_text(encoding="utf-8"))
            config["source_phantom"] = impulse.phantom
            config["source_image_sha256"] = wrong_digest
            filter_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["source_image_sha256"] = wrong_digest
            manifest["entries"][0]["filter_config_sha256"] = sha256_file(filter_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "wrong official source phantom"):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase1_candidate_rejects_generator_source_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._write_phase1_candidate_manifest(
                root,
                generator_version=PICTOLOGICS_VERSION,
            )
            generator_source = root / "generator.py"
            generator_source.write_text(
                generator_source.read_text(encoding="utf-8") + "TAMPERED = True\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RunIntegrityError,
                "generator source checksum mismatch",
            ):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase1_candidate_must_be_a_readable_3d_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, source_digest = self._write_phase1_candidate_manifest(
                root,
                generator_version=PICTOLOGICS_VERSION,
                response_bytes=b"not-a-nifti",
            )
            with (
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE1_SOURCE_IMAGE_SHA256",
                    frozenset({source_digest}),
                ),
                self.assertRaisesRegex(ValueError, "readable NIfTI"),
            ):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase1_candidate_rejects_an_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 99,
                        "kind": "ibsi2_phase1_response_maps",
                        "entries": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema/kind"):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase2_candidate_manifest_binds_official_source_and_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, image_digest, mask_digest = (
                self._write_phase2_candidate_manifest(root)
            )
            with (
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_IMAGE_SHA256",
                    image_digest,
                ),
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_MASK_SHA256",
                    mask_digest,
                ),
            ):
                entries = load_ibsi2_phase2_candidate_manifest(manifest_path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries.adapters, ("pictologics",))
            self.assertEqual(len(entries.support_declarations), 22)
            self.assertEqual(entries[0]["filter_id"], "1.A")
            self.assertEqual(
                entries[0]["preprocessing_config"].name, "preprocessing-A.json"
            )
            self.assertEqual(entries[0]["filter_input"].name, "filter-input-A.nii.gz")

    def test_phase2_candidate_manifest_rejects_reviewed_parameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _, _ = self._write_phase2_candidate_manifest(root)
            filter_path = root / "filter-1.A.json"
            config = json.loads(filter_path.read_text(encoding="utf-8"))
            config["parameters"]["boundary"] = "zero"
            filter_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["filter_config_sha256"] = sha256_file(filter_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "differs from v9"):
                load_ibsi2_phase2_candidate_manifest(manifest_path)

    def test_phase2_candidate_manifest_rejects_environment_lock_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _, _ = self._write_phase2_candidate_manifest(root)
            environment_path = root / "environment.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment["unit_test_tamper"] = True
            environment_path.write_text(json.dumps(environment), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["environment_lock_sha256"] = sha256_file(
                environment_path
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                RunIntegrityError,
                "differs from the verified runtime",
            ):
                load_ibsi2_phase2_candidate_manifest(manifest_path)

    def test_phase2_candidate_manifest_rejects_filter_input_geometry_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            try:
                import nibabel as nib
                import numpy as np
            except ImportError:
                self.skipTest("nibabel and numpy are required")

            root = Path(temporary)
            manifest_path, _, _ = self._write_phase2_candidate_manifest(root)
            for name in ("filter-input-A.nii.gz", "1.A.nii.gz", "1.A-mask.nii.gz"):
                path = root / name
                nifti = nib.load(str(path))
                cropped = np.asanyarray(nifti.dataobj)[:-1, :, :]
                nib.save(nib.Nifti1Image(cropped, nifti.affine), path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest["entries"][0]
            entry["filter_input_sha256"] = sha256_file(root / "filter-input-A.nii.gz")
            entry["image_sha256"] = sha256_file(root / "1.A.nii.gz")
            entry["mask_sha256"] = sha256_file(root / "1.A-mask.nii.gz")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "A grid: shape mismatch"):
                load_ibsi2_phase2_candidate_manifest(manifest_path)

    def test_phase2_candidate_manifest_rejects_nonbinary_response_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, image_digest, mask_digest = (
                self._write_phase2_candidate_manifest(
                    root,
                    candidate_mask_value=2,
                )
            )
            with (
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_IMAGE_SHA256",
                    image_digest,
                ),
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_MASK_SHA256",
                    mask_digest,
                ),
                self.assertRaisesRegex(ValueError, "mask must be binary"),
            ):
                load_ibsi2_phase2_candidate_manifest(manifest_path)

    def test_phase2_candidate_manifest_rejects_wrong_preprocessing_content(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, image_digest, mask_digest = (
                self._write_phase2_candidate_manifest(root)
            )
            preprocessing_path = root / "preprocessing-A.json"
            config = json.loads(preprocessing_path.read_text(encoding="utf-8"))
            config["crop"] = True
            preprocessing_path.write_text(json.dumps(config), encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["preprocessing_config_sha256"] = sha256_file(
                preprocessing_path
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_IMAGE_SHA256",
                    image_digest,
                ),
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_MASK_SHA256",
                    mask_digest,
                ),
                self.assertRaisesRegex(ValueError, "preprocessing mismatch for crop"),
            ):
                load_ibsi2_phase2_candidate_manifest(manifest_path)

    def test_phase1_candidate_manifest_requires_exact_support_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = self._write_phase1_candidate_manifest(
                root,
                generator_version=PICTOLOGICS_VERSION,
                response_bytes=b"candidate",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["support_declarations"].pop()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact adapter x 36 grid"):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase1_candidate_entry_cannot_be_declared_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, source_digest = self._write_phase1_candidate_manifest(
                root,
                generator_version=PICTOLOGICS_VERSION,
                response_bytes=b"candidate",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["support_declarations"][0]["native_supported"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE1_SOURCE_IMAGE_SHA256",
                    frozenset({source_digest}),
                ),
                self.assertRaisesRegex(ValueError, "native_supported is false"),
            ):
                load_ibsi2_phase1_candidate_manifest(manifest_path)

    def test_phase2_native_supported_filter_requires_candidate_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, image_digest, mask_digest = (
                self._write_phase2_candidate_manifest(root)
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            declaration = next(
                item
                for item in manifest["support_declarations"]
                if item["filter_id"] == "2.A"
            )
            declaration["native_supported"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with (
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_IMAGE_SHA256",
                    image_digest,
                ),
                mock.patch(
                    "bench.compliance.run.IBSI2_PHASE2_SOURCE_MASK_SHA256",
                    mask_digest,
                ),
                self.assertRaisesRegex(ValueError, "native-supported filter requires"),
            ):
                load_ibsi2_phase2_candidate_manifest(manifest_path)

    def test_execution_error_is_unevaluated_not_unsupported_in_outcome_table(
        self,
    ) -> None:
        record = ComparisonRecord(
            specification="IBSI 1",
            phase="phase1",
            adapter="pictologics",
            software_version="0.5.1",
            configuration="digital_phantom",
            profile="digital_phantom_3d_merge",
            aggregation="not_applicable",
            family="intensity",
            feature_name="Mean",
            feature_tag="stat_mean",
            semantic_key="intensity_statistics.mean",
            ibsi_code="Q4LE",
            standardized=True,
            observed_supported=False,
            mapped=False,
            attempted=True,
            finite=False,
            referencable=True,
            evaluated=False,
            passed=None,
            status="error",
        )
        summary = _overall_summary([record])[0]
        self.assertEqual(summary["unsupported"], 0)
        self.assertEqual(summary["defined_standardized_checks"], 1)

    def test_compliance_summary_never_merges_software_versions(self) -> None:
        first = ComparisonRecord(
            specification="IBSI 1",
            phase="phase1",
            adapter="pictologics",
            software_version="0.5.1",
            configuration="digital_phantom",
            profile="digital_phantom_3d_merge",
            aggregation="not_applicable",
            family="intensity",
            feature_name="Mean",
            feature_tag="stat_mean",
            semantic_key="intensity_statistics.mean",
            ibsi_code="Q4LE",
            standardized=True,
            observed_supported=True,
            mapped=True,
            attempted=True,
            finite=True,
            referencable=True,
            evaluated=True,
            passed=True,
            status="pass",
        )
        second = ComparisonRecord(
            **{
                **first.to_dict(),
                "software_version": "0.6.0",
                "passed": False,
                "status": "fail",
            }
        )

        summary = _overall_summary([first, second])

        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {row["software_version"] for row in summary},
            {"0.5.1", "0.6.0"},
        )


class ComplianceReportTests(unittest.TestCase):
    def test_report_writes_named_denominators_and_accessible_formats(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is required")
        from bench.compliance.report import generate_compliance_report

        rows = []
        for adapter, passed in (("pictologics", True), ("pyradiomics", False)):
            for index, code in enumerate(("Q4LE", "ECT3")):
                rows.append(
                    ComparisonRecord(
                        specification="IBSI 1",
                        phase="phase1",
                        adapter=adapter,
                        software_version="test",
                        configuration="digital_phantom",
                        profile="digital_phantom_3d_merged",
                        aggregation="not_applicable",
                        family="intensity",
                        feature_name=code,
                        feature_tag=f"tag_{code}",
                        semantic_key=f"intensity_statistics.{code.casefold()}",
                        ibsi_code=code,
                        standardized=True,
                        observed_supported=True,
                        mapped=True,
                        attempted=True,
                        finite=True,
                        referencable=True,
                        evaluated=True,
                        passed=passed if index == 0 else True,
                        status="pass" if passed or index == 1 else "fail",
                        value=10.0,
                        reference_value=10.0,
                        tolerance=1.0,
                        raw_abs_error=0.0 if passed or index == 1 else 2.0,
                        comparison_error=0.0 if passed or index == 1 else 2.0,
                        error_tolerance_ratio=0.0 if passed or index == 1 else 2.0,
                        comparison_policy="test",
                    )
                )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = generate_compliance_report(rows, output)
            self.assertTrue((output / "compliance_summary.csv").is_file())
            with (output / "compliance_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 2)
            for summary_row in summary_rows:
                self.assertEqual(summary_row["defined_features"], "172")
                self.assertEqual(summary_row["defined_benchmark_instances"], "174")
                self.assertEqual(summary_row["finite_benchmark_instances"], "2")
                self.assertEqual(summary_row["standardized_benchmark_instances"], "2")
            markdown = (output / "compliance_summary.md").read_text(encoding="utf-8")
            for denominator in ("172", "174", "169"):
                self.assertIn(denominator, markdown)
            for suffix in ("pdf", "svg", "png", "json"):
                self.assertTrue(
                    (output / f"figure_conformance_outcomes.{suffix}").is_file()
                )
                self.assertTrue(
                    (output / f"figure_conformance_by_family.{suffix}").is_file()
                )
            self.assertIn("denominator_policy", manifest)
            self.assertIn("workbook_instance_coverage", manifest["denominator_policy"])
            self.assertTrue(
                all(figure.get("alt_text") for figure in manifest["figures"])
            )
            for figure in manifest["figures"]:
                narrative = f"{figure['caption']} {figure['alt_text']}"
                for denominator in ("172", "174", "169"):
                    self.assertIn(denominator, narrative)

    def test_phase1_report_keeps_36_and_33_denominators(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is required")
        from bench.compliance.report import generate_ibsi2_phase1_report

        standardized = set(IBSI2_PHASE1_STANDARDIZED_IDS)
        rows = []
        for test_id in IBSI2_PHASE1_TEST_IDS:
            is_standardized = test_id in standardized
            rows.append(
                {
                    "adapter": "pictologics",
                    "test_id": test_id,
                    "standardized": is_standardized,
                    "supported": True,
                    "referencable": is_standardized,
                    "evaluated": is_standardized,
                    "passed": True if is_standardized else None,
                    "status": "pass" if is_standardized else "not_standardized",
                    "generator_distribution": "pictologics",
                    "generator_version": "0.5.0",
                    "voxel_tolerance": 1.0 if is_standardized else None,
                    "max_abs_error": 0.1 if is_standardized else None,
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = generate_ibsi2_phase1_report(rows, output)
            self.assertEqual(manifest["defined_tests"], 36)
            self.assertEqual(manifest["standardized_reference_tests"], 33)
            self.assertIn("accessibility", manifest)
            for figure in manifest["figures"]:
                self.assertTrue(figure["alt_text"])
                for filename in figure["files"]:
                    self.assertTrue((output / filename).is_file())


if __name__ == "__main__":
    unittest.main()
