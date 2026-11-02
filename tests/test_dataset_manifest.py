from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal controllers
    nib = None
    np = None

from bench.dataset_manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetValidationError,
    nifti_case_metadata,
    sha256_file,
    validate_manifest,
)


@unittest.skipIf(nib is None or np is None, "nibabel and numpy are required")
class DatasetManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        self.image_path = self.root / "images" / "case_image.nii.gz"
        self.mask_path = self.root / "masks" / "case_mask.nii.gz"
        affine = np.diag([1.0, 2.0, 3.0, 1.0])
        image = np.arange(120, dtype=np.float32).reshape((4, 5, 6))
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1:3, 1:4, 2:5] = 1
        nib.save(nib.Nifti1Image(image, affine), self.image_path)
        nib.save(nib.Nifti1Image(mask, affine), self.mask_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self) -> dict:
        geometry = nifti_case_metadata(self.image_path, self.mask_path)
        image_relative = "images/case_image.nii.gz"
        mask_relative = "masks/case_mask.nii.gz"
        image_hash = sha256_file(self.image_path)
        mask_hash = sha256_file(self.mask_path)
        return {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "dataset": "unit_test",
            "dataset_kind": "real_world",
            "files": [
                {
                    "path": image_relative,
                    "sha256": image_hash,
                    "bytes": self.image_path.stat().st_size,
                    "kind": "image",
                },
                {
                    "path": mask_relative,
                    "sha256": mask_hash,
                    "bytes": self.mask_path.stat().st_size,
                    "kind": "mask",
                },
            ],
            "cases": [
                {
                    "case_id": "case",
                    "modality": "ct",
                    "image_path": image_relative,
                    "mask_path": mask_relative,
                    "image_sha256": image_hash,
                    "mask_sha256": mask_hash,
                    **geometry,
                }
            ],
        }

    def test_valid_pair_uses_actual_voxel_complexity(self) -> None:
        summary = validate_manifest(self.root, self.manifest())
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["total_image_voxels"], 120)

    def test_checksum_tampering_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(DatasetValidationError, "checksum mismatch"):
            validate_manifest(self.root, manifest)

    def test_path_escape_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["files"][0]["path"] = "../outside.nii.gz"
        with self.assertRaisesRegex(DatasetValidationError, "escapes"):
            validate_manifest(self.root, manifest, verify_hashes=False)

    def test_affine_mismatch_is_rejected(self) -> None:
        shifted = np.diag([1.0, 2.0, 3.0, 1.0])
        shifted[0, 3] = 10.0
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1:3, 1:4, 2:5] = 1
        nib.save(nib.Nifti1Image(mask, shifted), self.mask_path)
        with self.assertRaisesRegex(DatasetValidationError, "affine mismatch"):
            nifti_case_metadata(self.image_path, self.mask_path)

    def test_header_spacing_mismatch_is_rejected_even_with_matching_affine(
        self,
    ) -> None:
        mask = nib.load(str(self.mask_path))
        mask.update_header()
        mask.header.set_zooms((1.0, 2.0, 4.0))
        mask.to_filename(str(self.mask_path))

        with self.assertRaisesRegex(
            DatasetValidationError,
            "header spacing mismatch",
        ):
            nifti_case_metadata(self.image_path, self.mask_path)

    def test_conflicting_active_qform_and_sform_are_rejected(self) -> None:
        selected = np.diag([1.0, 2.0, 3.0, 1.0])
        conflicting = selected.copy()
        conflicting[0, 3] = 12.0
        image_data = np.arange(120, dtype=np.float32).reshape((4, 5, 6))
        mask_data = np.zeros((4, 5, 6), dtype=np.uint8)
        mask_data[1:3, 1:4, 2:5] = 1
        for path, data in (
            (self.image_path, image_data),
            (self.mask_path, mask_data),
        ):
            nifti = nib.Nifti1Image(data, selected)
            nifti.set_sform(selected, code=1)
            nifti.set_qform(conflicting, code=1)
            nib.save(nifti, path)

        with self.assertRaisesRegex(
            DatasetValidationError,
            "active qform/sform mismatch",
        ):
            nifti_case_metadata(self.image_path, self.mask_path)

    def test_noncanonical_positive_mask_label_is_rejected(self) -> None:
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1:3, 1:4, 2:5] = 255
        nib.save(nib.Nifti1Image(mask, np.diag([1.0, 2.0, 3.0, 1.0])), self.mask_path)
        with self.assertRaisesRegex(DatasetValidationError, r"binary \{0, 1\}"):
            nifti_case_metadata(self.image_path, self.mask_path)

    def test_nonfinite_image_outside_roi_is_rejected(self) -> None:
        image = np.arange(120, dtype=np.float32).reshape((4, 5, 6))
        image[0, 0, 0] = np.nan
        nib.save(
            nib.Nifti1Image(image, np.diag([1.0, 2.0, 3.0, 1.0])),
            self.image_path,
        )
        with self.assertRaisesRegex(DatasetValidationError, "full-volume finiteness"):
            nifti_case_metadata(self.image_path, self.mask_path)

    def test_manifest_json_rejects_nonfinite_values(self) -> None:
        manifest = self.manifest()
        manifest["cases"][0]["mask_fraction"] = float("nan")
        with self.assertRaises(ValueError):
            json.dumps(manifest, allow_nan=False)

    def test_declared_orientation_must_match_observed_affine(self) -> None:
        manifest = self.manifest()
        manifest["cases"][0]["orientation"] = ["L", "A", "S"]
        with self.assertRaisesRegex(DatasetValidationError, "orientation mismatch"):
            validate_manifest(self.root, manifest)

    def test_synthetic_modality_must_match_dataset_kind(self) -> None:
        manifest = self.manifest()
        manifest["dataset_kind"] = "synthetic"
        manifest["cases"][0]["size"] = 6
        with self.assertRaisesRegex(
            DatasetValidationError,
            "must declare modality 'synthetic'",
        ):
            validate_manifest(self.root, manifest)

    def test_synthetic_size_must_match_observed_cubic_edge(self) -> None:
        affine = np.eye(4)
        image = np.arange(64, dtype=np.float32).reshape((4, 4, 4))
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        nib.save(nib.Nifti1Image(image, affine), self.image_path)
        nib.save(nib.Nifti1Image(mask, affine), self.mask_path)

        manifest = self.manifest()
        manifest["dataset_kind"] = "synthetic"
        manifest["cases"][0]["modality"] = "synthetic"
        manifest["cases"][0]["size"] = 5
        with self.assertRaisesRegex(
            DatasetValidationError,
            "size must equal every cubic image edge",
        ):
            validate_manifest(self.root, manifest)

    def test_named_provenance_record_requires_sha256_binding(self) -> None:
        preparation = self.root / "preparation.json"
        preparation.write_text("{}\n", encoding="utf-8")
        manifest = self.manifest()
        manifest["provenance"] = {"preparation_record": preparation.name}
        with self.assertRaisesRegex(
            DatasetValidationError,
            "preparation_record_sha256",
        ):
            validate_manifest(self.root, manifest)


if __name__ == "__main__":
    unittest.main()
