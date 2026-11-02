from __future__ import annotations

import unittest

from bench.adapters.protocol import ADAPTER_PROTOCOL_VERSION
from bench.compliance.evaluate import evaluate_adapter_payload
from bench.compliance.models import ReferenceRecord
from bench.ibsi_mapping import (
    map_pyradiomics,
    pyradiomics_feature_selection,
    pyradiomics_semantic_aliases,
)


_FEATURES = {
    "TF7R": (
        "glcm",
        "Difference average",
        "cm_diff_avg_3D_comb",
        "glcm.diff_avg",
        "3d_merge",
        1.38,
    ),
    "8S9J": (
        "glcm",
        "Dissimilarity",
        "cm_dissimilarity_3D_comb",
        "glcm.dissimilarity",
        "3d_merge",
        1.38,
    ),
    "DG8W": (
        "glcm",
        "Cluster tendency",
        "cm_clust_tend_3D_comb",
        "glcm.clust_tend",
        "3d_merge",
        7.41,
    ),
    "OEEB": (
        "glcm",
        "Sum variance",
        "cm_sum_var_3D_comb",
        "glcm.sum_var",
        "3d_merge",
        7.41,
    ),
    "BJ5W": (
        "histogram",
        "Uniformity",
        "ih_uniformity",
        "intensity_histogram.uniformity",
        "not_applicable",
        0.512,
    ),
    "5SPA": (
        "ngldm",
        "Normalised grey level non-uniformity",
        "ngl_glnu_norm_3D",
        "ngldm.glnu_norm",
        "3d",
        0.512,
    ),
}


def _reference(code: str) -> ReferenceRecord:
    family, name, tag, semantic_key, aggregation, value = _FEATURES[code]
    return ReferenceRecord(
        specification="IBSI 1",
        phase="phase1",
        dataset="digital phantom",
        configuration="digital_phantom",
        profile="digital_phantom_3d_merged",
        in_profile=True,
        aggregation=aggregation,
        family=family,
        feature_name=name,
        feature_tag=tag,
        semantic_key=semantic_key,
        ibsi_code=code,
        consensus=">= 10",
        reference_value=value,
        tolerance=0.0,
        standardized=True,
        source_sheet="digital phantom",
        source_row=1,
    )


class PyRadiomicsSemanticAliasTests(unittest.TestCase):
    def test_active_sources_keep_their_direct_native_mapping(self) -> None:
        expected = {
            "original_glcm_DifferenceAverage": "TF7R",
            "original_glcm_ClusterTendency": "DG8W",
            "original_firstorder_Uniformity": "BJ5W",
        }

        self.assertEqual(
            {native: map_pyradiomics(native) for native in expected},
            expected,
        )

    def test_non_emitting_or_nonexistent_names_are_not_direct_mappings(self) -> None:
        names = (
            "original_glcm_Dissimilarity",
            "original_glcm_SumVariance",
            "original_gldm_GrayLevelNonUniformityNormalized",
            "original_gldm_DependencePercentage",
            "original_firstorder_MedianAbsoluteDeviation",
        )

        for native_name in names:
            with self.subTest(native_name=native_name):
                self.assertIsNone(map_pyradiomics(native_name))

    def test_alias_codes_select_the_active_upstream_source(self) -> None:
        expected = {
            "8S9J": {"glcm": ["DifferenceAverage"]},
            "OEEB": {"glcm": ["ClusterTendency"]},
            "5SPA": {"firstorder": ["Uniformity"]},
        }

        for code, selection in expected.items():
            with self.subTest(code=code):
                self.assertEqual(pyradiomics_feature_selection([code]), selection)

    def test_public_alias_catalog_exposes_only_reviewed_exact_identities(self) -> None:
        expected = {
            "original_glcm_DifferenceAverage": "8S9J",
            "original_glcm_ClusterTendency": "OEEB",
            "original_firstorder_Uniformity": "5SPA",
        }

        for source, code in expected.items():
            with self.subTest(source=source):
                aliases = pyradiomics_semantic_aliases(source)
                self.assertEqual(set(aliases), {code})
                self.assertTrue(aliases[code])

        self.assertEqual(
            pyradiomics_semantic_aliases("original_gldm_DependencePercentage"),
            {},
        )

    def test_direct_and_alias_codes_deduplicate_the_active_sources(self) -> None:
        self.assertEqual(
            pyradiomics_feature_selection(
                ["TF7R", "8S9J", "DG8W", "OEEB", "BJ5W", "5SPA"]
            ),
            {
                "firstorder": ["Uniformity"],
                "glcm": ["ClusterTendency", "DifferenceAverage"],
            },
        )

    def test_evaluator_joins_each_active_value_to_direct_and_alias_semantics(
        self,
    ) -> None:
        payload = {
            "schema_version": ADAPTER_PROTOCOL_VERSION,
            "software": {"version": "3.1.0"},
            "features": {
                "all": [
                    "original_glcm_DifferenceAverage",
                    "original_glcm_ClusterTendency",
                    "original_firstorder_Uniformity",
                ]
            },
            "values": {
                "all": {
                    "original_glcm_DifferenceAverage": 1.3795066413662238,
                    "original_glcm_ClusterTendency": 7.4121967817548775,
                    "original_firstorder_Uniformity": 0.5124178232286339,
                }
            },
        }
        codes = ("TF7R", "8S9J", "DG8W", "OEEB", "BJ5W", "5SPA")

        records, _ = evaluate_adapter_payload(
            adapter="pyradiomics",
            payload=payload,
            references=[_reference(code) for code in codes],
            release_version="3.1.0",
        )

        by_code = {record.ibsi_code: record for record in records}
        self.assertEqual(set(by_code), set(codes))
        for code in codes:
            with self.subTest(code=code):
                self.assertTrue(by_code[code].observed_supported)
                self.assertTrue(by_code[code].finite)
                self.assertTrue(by_code[code].evaluated)
                self.assertTrue(by_code[code].passed)

        pairs = (
            ("TF7R", "8S9J"),
            ("DG8W", "OEEB"),
            ("BJ5W", "5SPA"),
        )
        for direct_code, alias_code in pairs:
            with self.subTest(alias_code=alias_code):
                source = by_code[direct_code].native_feature_names
                alias_label = by_code[alias_code].native_feature_names
                self.assertEqual(alias_label, f"{source} [documented exact alias]")


if __name__ == "__main__":
    unittest.main()
