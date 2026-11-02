from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from bench.ibsi2_phase3_dataset import (
    prepare_ibsi2_phase3_dataset,
    validate_ibsi2_phase3_dataset,
)
from bench.dataset_manifest import sha256_file
from bench.pillar1_dataset import (
    build_pillar1_dataset,
    validate_pillar1_dataset,
)
from bench.pillar2_dataset import build_pillar2_dataset, validate_pillar2_dataset
from bench.synthetic_representations import (
    compile_mask_specific_fbn,
    compile_mask_specific_fbs,
)


def _small_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "pillar1_test",
                "dataset_kind": "synthetic",
                "description": "test",
                "generator": "synthetic_volumetric_phantom",
                "generator_version": 1,
                "coordinate_system": "RAS+",
                "axis_order": "xyz",
                "fov_mm": 256.0,
                "sizes": [32],
                "hu_profiles": ["reference"],
                "mask_ids": ["M1", "M2", "M3", "M4"],
                "seed": 20260809,
                "slab_depth": 7,
                "raw_representation": {
                    "id": "original_hu_int16",
                    "dtype": "int16",
                    "units": "HU",
                    "routing": ["morphology"],
                },
                "texture_representation": {
                    "id": "mask_specific_ibsi_fbn32",
                    "method": "IBSI_fixed_bin_number",
                    "configured_levels": 32,
                    "background_value": 0,
                    "roi_value_range": [1, 32],
                    "routing": ["glcm"],
                },
                "ivh_representation": {
                    "id": "mask_specific_ibsi_fbs1_ivh_indices",
                    "method": "IBSI_fixed_bin_size",
                    "bin_width": 1.0,
                    "anchor": "mask_specific_roi_minimum",
                    "background_value": 0,
                    "routing": ["ivh"],
                },
                "storage": {
                    "format": "NIfTI-1",
                    "compression": "gzip",
                    "extension": ".nii.gz",
                },
                "case_count": 4,
                "raw_image_count": 1,
                "mask_count": 4,
                "discrete_image_count": 4,
                "ivh_image_count": 4,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_pillar1_builder_commits_raw_mask_specific_fbn_and_resumes(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.json"
    destination = tmp_path / "pillar1"
    _small_profile(profile)

    first = build_pillar1_dataset(destination, profile_path=profile, resume=True)
    summary = validate_pillar1_dataset(destination, deep=True)
    second = build_pillar1_dataset(destination, profile_path=profile, resume=True)

    assert len(first["cases"]) == 4
    assert summary["deep_fbn_cases_validated"] == 4
    assert summary["ivh_image_count"] == 4
    assert summary["ready_for_adapter_input"] is True
    assert first == second
    case = next(case for case in first["cases"] if case["mask_id"] == "M3")
    raw = np.asanyarray(nib.load(str(destination / case["image_path"])).dataobj)
    mask = np.asanyarray(nib.load(str(destination / case["mask_path"])).dataobj)
    discrete = np.asanyarray(
        nib.load(str(destination / case["discrete_image_path"])).dataobj
    )
    ivh = np.asanyarray(nib.load(str(destination / case["ivh_image_path"])).dataobj)
    expected = compile_mask_specific_fbn(raw, mask, levels=32)
    np.testing.assert_array_equal(discrete, expected.array)
    np.testing.assert_array_equal(ivh, compile_mask_specific_fbs(raw, mask).array)
    assert raw.dtype == np.dtype(np.int16)
    assert discrete.dtype == np.dtype(np.uint8)
    assert np.all(discrete[mask == 0] == 0)

    independent = tmp_path / "pillar1_independent"
    rebuilt = build_pillar1_dataset(independent, profile_path=profile, resume=True)
    first_hashes = {entry["path"]: entry["sha256"] for entry in first["files"]}
    rebuilt_hashes = {entry["path"]: entry["sha256"] for entry in rebuilt["files"]}
    assert rebuilt_hashes == first_hashes


def test_pillar2_builder_is_restart_safe_and_keeps_the_face_guard(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "pillar2_profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "pictologics_pillar2_a1",
                "dataset_kind": "synthetic",
                "description": "test",
                "generator": "synthetic_volumetric_phantom",
                "generator_version": 1,
                "coordinate_system": "RAS+",
                "axis_order": "xyz",
                "fov_mm": 256.0,
                "sizes": [32],
                "hu_profile": "reference",
                "mask_id": "A1",
                "seed": 20260809,
                "slab_depth": 8,
                "texture_levels": 32,
                "ivh_bin_width": 1.0,
                "case_count": 1,
                "artifact_count": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "pillar2"
    first = build_pillar2_dataset(destination, profile_path=profile, resume=True)
    second = build_pillar2_dataset(destination, profile_path=profile, resume=True)
    summary = validate_pillar2_dataset(destination)

    assert first == second
    assert summary["a1_cases_validated"] == 1
    assert summary["ready_for_adapter_input"] is True
    state = json.loads((destination / "generation_state.json").read_text())
    assert state["status"] == "complete"
    case = first["cases"][0]
    mask = np.asanyarray(nib.load(str(destination / case["mask_path"])).dataobj)
    coordinates = np.argwhere(mask == 1)
    margins = np.concatenate(
        (coordinates.min(0), np.asarray(mask.shape) - 1 - coordinates.max(0))
    )
    assert int(np.min(margins)) >= 1


def _write_phase3_fixture(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    files = []
    for index, modality in enumerate(("ct", "mri", "pet")):
        image = np.arange(120, dtype=np.float32).reshape(4, 5, 6) + index
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1:3, 1:4, 1:5] = 1
        image_nii = nib.Nifti1Image(image, affine)
        mask_nii = nib.Nifti1Image(mask, affine)
        image_nii.set_qform(affine, code=1)
        image_nii.set_sform(affine, code=1)
        mask_nii.set_qform(affine, code=1)
        mask_nii.set_sform(affine, code=1)
        image_path = root / "images" / f"STS_001_{modality}_image.nii.gz"
        mask_path = root / "masks" / f"STS_001_{modality}_mask.nii.gz"
        nib.save(image_nii, image_path)
        nib.save(mask_nii, mask_path)
        for path in (image_path, mask_path):
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    (root / "manifest.json").write_text(
        json.dumps({"dataset": "fixture", "files": files}) + "\n",
        encoding="utf-8",
    )


def test_phase3_import_is_byte_identical_complete_and_workspace_local(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "workspace" / "ibsi2_phase3"
    _write_phase3_fixture(source)

    manifest = prepare_ibsi2_phase3_dataset(
        source,
        destination,
        expected_subjects=1,
        resume=True,
    )
    summary = validate_ibsi2_phase3_dataset(destination)

    assert len(manifest["cases"]) == 3
    assert len(manifest["files"]) == 12
    assert summary["subject_count"] == 1
    assert summary["byte_identical_source_copy"] is True
    assert summary["derived_fbn32_cases_validated"] == 3
    assert summary["derived_fbs1_ivh_cases_validated"] == 1
    assert summary["derived_fbn1000_ivh_cases_validated"] == 2
    assert all(not Path(entry["path"]).is_absolute() for entry in manifest["files"])
    for case in manifest["cases"]:
        assert case["image_sha256"] == case["source_image_sha256"]
        assert case["mask_sha256"] == case["source_mask_sha256"]
    assert (destination / "source_manifest.json").read_bytes() == (
        source / "manifest.json"
    ).read_bytes()
