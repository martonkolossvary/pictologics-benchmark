from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_adapter_feature_surface import _load_expected, _normalize_case


class AdapterFeatureSurfaceAuditTests(unittest.TestCase):
    def test_case_normalization_binds_all_three_image_representations(self) -> None:
        case = _normalize_case(
            Path("/dataset"),
            {
                "image_path": "raw/image.nii.gz",
                "mask_path": "masks/mask.nii.gz",
                "discrete_image_path": "discrete/fbn32.nii.gz",
                "ivh_image_path": "discrete/ivh_fbs1.nii.gz",
            },
        )
        for key, expected in (
            ("image_abs", "/dataset/raw/image.nii.gz"),
            ("mask_abs", "/dataset/masks/mask.nii.gz"),
            ("discrete_image_abs", "/dataset/discrete/fbn32.nii.gz"),
            ("ivh_image_abs", "/dataset/discrete/ivh_fbs1.nii.gz"),
        ):
            self.assertEqual(Path(case[key]), Path(expected).resolve())

    def test_multiple_native_sources_and_documented_aliases_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparisons.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("adapter", "family", "native_feature_names"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "adapter": "mirp",
                        "family": "intensity",
                        "native_feature_names": "stat_energy, stat_energy_offset",
                    }
                )
                writer.writerow(
                    {
                        "adapter": "pyradiomics",
                        "family": "ngldm",
                        "native_feature_names": (
                            "original_firstorder_Uniformity [documented exact alias]"
                        ),
                    }
                )

            observed = _load_expected(path)

        self.assertEqual(
            observed[("mirp", "intensity")],
            {"stat_energy": False, "stat_energy_offset": False},
        )
        self.assertEqual(
            observed[("pyradiomics", "ngldm")],
            {"original_firstorder_Uniformity": True},
        )


if __name__ == "__main__":
    unittest.main()
