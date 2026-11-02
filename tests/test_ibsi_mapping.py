from __future__ import annotations

import unittest

from bench.ibsi_families import CODE_TO_FAMILY
from bench.ibsi_mapping import (
    _code_from_name_in_family,
    classify_feature,
    map_mirp,
    map_pyradiomics,
    pyradiomics_feature_selection,
)


class CorrectedFamilyAssignmentTests(unittest.TestCase):
    def test_texture_codes_map_to_their_ibsi_families(self) -> None:
        expected = {
            "ACUI": "glcm",
            "65HE": "ngtdm",
            "8CE5": "glrlm",
            "BYLV": "glszm",
            "OVBL": "glrlm",
            "XMSY": "glszm",
            "Y1RO": "glszm",
            "P30P": "glszm",
            "S1RA": "gldzm",
            "7HP3": "gldzm",
            "VIWW": "gldzm",
            "5SPA": "ngldm",
            "1PFV": "ngldm",
        }
        self.assertEqual(
            {code: CODE_TO_FAMILY.get(code) for code in expected}, expected
        )


class StrictSemanticMappingTests(unittest.TestCase):
    def test_duplicate_names_resolve_only_inside_the_requested_family(self) -> None:
        self.assertEqual(_code_from_name_in_family("Contrast", "glcm"), "ACUI")
        self.assertEqual(_code_from_name_in_family("Contrast", "ngtdm"), "65HE")
        self.assertEqual(
            _code_from_name_in_family("Grey level variance", "glrlm"), "8CE5"
        )
        self.assertEqual(
            _code_from_name_in_family("Grey level variance", "glszm"), "BYLV"
        )

    def test_no_cross_family_fallback(self) -> None:
        self.assertIsNone(_code_from_name_in_family("Contrast", "glrlm"))
        self.assertIsNone(
            _code_from_name_in_family("Grey level variance", "morphology")
        )

    def test_mirp_uses_longest_prefix_before_generic_prefix(self) -> None:
        cases = {
            "cm_inv_diff_fbn_n32": "IB1Z",
            "cm_inv_diff_norm_fbn_n32": "NDRX",
            "cm_inv_diff_mom_fbn_n32": "WF0Z",
            "cm_inv_diff_mom_norm_fbn_n32": "1QCO",
            "ih_max_grad_fbn_n32": "12CE",
            "ih_max_grad_g_fbn_n32": "8E6O",
            "ih_min_grad_fbn_n32": "VQB3",
            "ih_min_grad_g_fbn_n32": "RHQZ",
            "rlm_glnu_fbn_n32": "R5YN",
            "rlm_glnu_norm_fbn_n32": "OVBL",
            "szm_zsnu_fbn_n32": "4JP3",
            "szm_zsnu_norm_fbn_n32": "VB3A",
            "dzm_zdnu_fbn_n32": "V294",
            "dzm_zdnu_norm_fbn_n32": "IATH",
            "ngl_dcnu_fbn_n32": "Z87G",
            "ngl_dcnu_norm_fbn_n32": "OKJI",
        }
        self.assertEqual({feature: map_mirp(feature) for feature in cases}, cases)

    def test_mirp_offset_variants_are_not_ibsi_features(self) -> None:
        self.assertEqual(map_mirp("stat_energy"), "N8CA")
        for name in (
            "stat_energy_offset",
            "stat_rms_offset",
            "stat_total_energy_offset",
        ):
            self.assertIsNone(map_mirp(name))

    def test_pyradiomics_sum_squares_is_ibsi_joint_variance(self) -> None:
        native_name = "original_glcm_SumSquares"
        self.assertEqual(map_pyradiomics(native_name), "UR99")
        self.assertEqual(
            classify_feature("pyradiomics", native_name), ("UR99", "mapped")
        )
        self.assertEqual(
            pyradiomics_feature_selection(["UR99"]),
            {"glcm": ["SumSquares"]},
        )

    def test_pyradiomics_explicit_deprecated_shape_features_are_mapped(self) -> None:
        expected = {
            "original_shape_Compactness1": "SKGS",
            "original_shape_Compactness2": "BQWJ",
            "original_shape_SphericalDisproportion": "KRCK",
        }
        self.assertEqual(
            {native: map_pyradiomics(native) for native in expected},
            expected,
        )
        self.assertEqual(
            pyradiomics_feature_selection(expected.values()),
            {
                "shape": [
                    "Compactness1",
                    "Compactness2",
                    "SphericalDisproportion",
                ]
            },
        )

    def test_zrad_native_ibsi_morphology_outputs_are_not_suppressed(self) -> None:
        expected = {
            "morph_asphericity": "25C7",
            "morph_av": "2PR5",
            "morph_area_mesh": "C0JK",
            "morph_comp_1": "SKGS",
            "morph_comp_2": "BQWJ",
            "morph_diam": "L0JK",
            "morph_integ_int": "99N0",
            "morph_sphericity": "QCFX",
            "morph_sph_dispr": "KRCK",
            "morph_vol_approx": "YEKZ",
            "morph_vol_dens_aabb": "PBX1",
            "morph_area_dens_aabb": "R59B",
            "morph_vol_dens_aee": "6BDE",
            "morph_area_dens_aee": "RDD2",
            "morph_vol_dens_conv_hull": "R3ER",
            "morph_area_dens_conv_hull": "7T7F",
        }
        self.assertEqual(
            {native: classify_feature("zrad", native) for native in expected},
            {native: (code, "mapped") for native, code in expected.items()},
        )


if __name__ == "__main__":
    unittest.main()
