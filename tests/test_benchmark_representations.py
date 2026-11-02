from __future__ import annotations

import unittest

from bench.benchmark_representations import (
    HARMONIZED_INPUT_CONTRACT,
    INTERNAL_INPUT_CONTRACT,
    select_representation,
)


class BenchmarkRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "case_id": "case-1",
            "modality": "ct",
            "image_abs": "/data/raw.nii.gz",
            "image_sha256": "a" * 64,
            "discrete_image_abs": "/data/fbn32.nii.gz",
            "discrete_image_sha256": "b" * 64,
            "ivh_image_abs": "/data/ivh_fbs1.nii.gz",
            "ivh_image_sha256": "d" * 64,
            "raw_representation": "original_hu_int16",
            "texture_representation": {
                "id": "mask_specific_ibsi_fbn32",
                "configured_levels": 32,
                "occupied_levels": 29,
                "roi_min_hu": -84.0,
                "roi_max_hu": 150.0,
                "derivation_sha256": "c" * 64,
            },
            "ivh_representation": {
                "id": "mask_specific_ibsi_fbs1_ivh_indices",
                "configured_levels": 235,
                "occupied_levels": 151,
                "bin_width": 1.0,
                "anchor_hu": -84.0,
                "derivation_sha256": "e" * 64,
            },
        }

    def select(self, family: str, contract: str = HARMONIZED_INPUT_CONTRACT):
        return select_representation(
            self.case,
            family,
            input_contract=contract,
            internal_discretization="fbn",
            internal_bins=16,
            internal_bin_width=25.0,
            internal_intensity_min=-100.0,
            internal_intensity_max=200.0,
        )

    def test_raw_families_use_original_image_without_discretisation(self) -> None:
        for family in ("morphology", "local_intensity", "intensity"):
            with self.subTest(family=family):
                selected = self.select(family)
                self.assertEqual(selected.image_path, "/data/raw.nii.gz")
                self.assertEqual(selected.image_sha256, "a" * 64)
                self.assertEqual(selected.representation_id, "original_hu_int16")
                self.assertEqual(selected.discretization, "raw")
                self.assertIsNone(selected.configured_levels)
                self.assertIsNone(selected.occupied_levels)

    def test_histogram_and_texture_use_the_bound_mask_specific_grid(self) -> None:
        families = (
            "histogram",
            "glcm",
            "glrlm",
            "glszm",
            "gldzm",
            "ngtdm",
            "ngldm",
        )
        for family in families:
            with self.subTest(family=family):
                selected = self.select(family)
                self.assertEqual(selected.image_path, "/data/fbn32.nii.gz")
                self.assertEqual(selected.image_sha256, "b" * 64)
                self.assertEqual(
                    selected.representation_id,
                    "mask_specific_ibsi_fbn32",
                )
                self.assertEqual(selected.discretization, "identity")
                self.assertEqual(selected.bins, 32)
                self.assertEqual(selected.bin_width, 1.0)
                self.assertEqual(selected.configured_levels, 32)
                self.assertEqual(selected.occupied_levels, 29)
                self.assertEqual(selected.derivation_sha256, "c" * 64)

    def test_ivh_uses_bound_fbs1_indices_not_fbn32(self) -> None:
        selected = self.select("ivh")
        self.assertEqual(selected.image_path, "/data/ivh_fbs1.nii.gz")
        self.assertEqual(selected.image_sha256, "d" * 64)
        self.assertEqual(
            selected.representation_id,
            "mask_specific_ibsi_fbs1_ivh_indices",
        )
        self.assertEqual(selected.discretization, "identity")
        self.assertEqual(selected.bin_width, 1.0)
        self.assertEqual(selected.configured_levels, 235)
        self.assertEqual(selected.occupied_levels, 151)
        self.assertEqual(selected.derivation_sha256, "e" * 64)

    def test_harmonized_ivh_accepts_a_bound_noncalibrated_grid(self) -> None:
        self.case["modality"] = "mri"
        self.case["ivh_image_abs"] = "/data/ivh_fbn1000.nii.gz"
        self.case["ivh_representation"] = {
            "id": "mask_specific_ibsi_fbn1000_ivh_indices",
            "configured_levels": 1000,
            "occupied_levels": 672,
            "derivation_sha256": "f" * 64,
        }
        selected = self.select("ivh")
        self.assertEqual(selected.image_path, "/data/ivh_fbn1000.nii.gz")
        self.assertEqual(selected.discretization, "identity")
        self.assertEqual(selected.configured_levels, 1000)
        self.assertEqual(selected.occupied_levels, 672)

    def test_invalid_or_unbound_discrete_metadata_fails_closed(self) -> None:
        self.case["texture_representation"]["occupied_levels"] = 33
        with self.assertRaisesRegex(ValueError, "configured/occupied"):
            self.select("glcm")

        self.case["texture_representation"]["occupied_levels"] = 29
        self.case["discrete_image_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "bound discrete image"):
            self.select("glcm")

    def test_internal_contract_keeps_the_single_image_policy(self) -> None:
        selected = self.select("glcm", INTERNAL_INPUT_CONTRACT)
        self.assertEqual(selected.image_path, "/data/raw.nii.gz")
        self.assertEqual(selected.discretization, "fbn")
        self.assertEqual(selected.bins, 16)
        self.assertEqual(selected.bin_width, 25.0)
        self.assertEqual(selected.intensity_min, -100.0)
        self.assertEqual(selected.intensity_max, 200.0)


if __name__ == "__main__":
    unittest.main()
