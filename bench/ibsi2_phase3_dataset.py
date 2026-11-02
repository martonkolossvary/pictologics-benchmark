"""Prepare the IBSI 2 Phase 3 STS cohort as a workspace-local dataset.

The importer copies every discovered CT/MRI/PET NIfTI and binary mask
byte-for-byte, derives metadata from the copied headers and voxel arrays, and
binds both source and destination SHA-256 values.  It never rescales, resamples,
resegments, renames subjects, or creates surrogate size labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np

from bench.dataset_manifest import (
    DatasetValidationError,
    atomic_copy,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_nifti,
    load_and_validate_manifest,
    nifti_case_metadata,
    sha256_file,
)
from bench.synthetic_representations import (
    compile_mask_specific_fbn,
    compile_mask_specific_fbs,
    texture_workload,
)


DATASET_ID = "ibsi2_phase3_sts"
MODALITIES = ("ct", "mri", "pet")
CASE_PATTERN = re.compile(
    r"^(?P<subject>STS_\d{3})_(?P<modality>ct|mri|pet)_image\.nii\.gz$"
)
OFFICIAL_DATASET_PAGE = "https://theibsi.github.io/datasets/"
OFFICIAL_IBSI2_PAGE = "https://theibsi.github.io/ibsi2/"
TCIA_DOI = "https://doi.org/10.7937/TCIA.2019.b7o0uq20"


def discover_pairs(dataset_root: Path) -> list[dict[str, Any]]:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("dataset must contain images/ and masks/ directories")

    pairs: list[dict[str, Any]] = []
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        match = CASE_PATTERN.fullmatch(image_path.name)
        if match is None:
            continue
        subject = match.group("subject")
        modality = match.group("modality")
        mask_path = masks_dir / f"{subject}_{modality}_mask.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"missing mask for {image_path.name}: {mask_path.name}"
            )
        pairs.append(
            {
                "case_id": f"{subject}_{modality}",
                "subject": subject,
                "modality": modality,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    if not pairs:
        raise FileNotFoundError("no STS image-mask pairs were discovered")
    return pairs


def _preparer_source_records() -> list[dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "bench" / "dataset_manifest.py",
        root / "bench" / "synthetic_representations.py",
        root / "poetry.lock",
    )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _readme_text() -> str:
    return f"""# IBSI 2 Phase 3 soft-tissue sarcoma cohort

This directory is a workspace-local preparation of the fixed
51-subject CT/MRI/PET IBSI 2 Phase 3 cohort. It contains 153 three-dimensional
image–mask pairs. Every original image and mask is copied byte-for-byte; no
source artifact was resampled, cropped, resegmented, or scaled, and no
surrogate image-size labels are used.

Each case also has a derived mask-specific IBSI FBN32 image. As in Pillar 1,
its ROI values are 1–32 and its background is 0. This file is used only for
histogram and texture families; the original image remains the input for raw
intensity families and morphology.

CT cases additionally contain a mask-specific FBS1 IVH bin-index image. MRI
and PET use a separately stored mask-specific FBN1000 IVH grid, following the
IBSI arbitrary-unit recommendation. Thus every adapter receives identical,
already-discretised IVH values and no adapter-side binning is timed.

The copied files are bound to their source bytes in `preparation.json` and to
their benchmark cases in `manifest.json`. Later performance analyses must use
the observed image, mask, modality, spacing, and ROI geometry as covariates;
this cohort is not a synthetic scaling ladder.

Redistribution is enabled for this project on the data owner's confirmation.
Public bundles must retain the bound source hashes and this attribution:

- IBSI datasets: {OFFICIAL_DATASET_PAGE}
- IBSI 2: {OFFICIAL_IBSI2_PAGE}
- TCIA source DOI: {TCIA_DOI}

Prepare or resume with:

```bash
poetry run bench prepare-ibsi2-phase3 --source-dir /path/to/ibsi2_validation \\
  --dest-dir data/benchmark/ibsi2_phase3 --resume
```
"""


def _copy_record(
    source_root: Path,
    destination_root: Path,
    source: Path,
    relative: str,
    *,
    kind: str,
    role: str,
    resume: bool,
) -> dict[str, Any]:
    destination = destination_root / relative
    atomic_copy(source, destination, overwrite=False if resume else False)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise DatasetValidationError(f"copy checksum mismatch: {relative}")
    nifti = nib.load(str(destination))
    return {
        "source_path": source.relative_to(source_root).as_posix(),
        "prepared_path": relative,
        "path": relative,
        "kind": kind,
        "role": role,
        "sha256": destination_hash,
        "source_sha256": source_hash,
        "bytes": destination.stat().st_size,
        "dtype": np.dtype(nifti.get_data_dtype()).name,
        "shape": [int(value) for value in nifti.shape],
    }


def _derivation_sha256(image_sha256: str, mask_sha256: str) -> str:
    payload = json.dumps(
        {
            "method": "IBSI_fixed_bin_number",
            "configured_levels": 32,
            "image_sha256": image_sha256,
            "mask_sha256": mask_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_ibsi2_phase3_dataset(
    source_dir: Path,
    destination: Path,
    *,
    expected_subjects: int = 51,
    resume: bool = True,
) -> dict[str, Any]:
    """Copy, attest, and index the complete IBSI 2 Phase 3 cohort."""

    source_dir = source_dir.expanduser().resolve()
    destination = destination.expanduser().resolve()
    pairs = discover_pairs(source_dir)
    subjects = sorted({str(pair["subject"]) for pair in pairs})
    if len(subjects) != expected_subjects:
        raise DatasetValidationError(
            f"IBSI 2 Phase 3 source has {len(subjects)} subjects; "
            f"expected {expected_subjects}"
        )
    by_subject = Counter(str(pair["subject"]) for pair in pairs)
    incomplete = {key: value for key, value in by_subject.items() if value != 3}
    if incomplete or len(pairs) != expected_subjects * len(MODALITIES):
        raise DatasetValidationError(
            f"IBSI 2 Phase 3 source is not a complete CT/MRI/PET block: {incomplete}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = source_dir / "manifest.json"
    source_manifest_hash = (
        sha256_file(source_manifest) if source_manifest.is_file() else None
    )
    if not source_manifest.is_file():
        raise DatasetValidationError("IBSI 2 Phase 3 source manifest is missing")
    try:
        source_manifest_value = json.loads(source_manifest.read_text(encoding="utf-8"))
        source_entries = {
            str(entry["path"]): str(entry["sha256"]).lower()
            for entry in source_manifest_value["files"]
        }
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            "IBSI 2 Phase 3 source manifest has no usable file inventory"
        ) from exc
    expected_source_paths = {
        Path(pair[key]).relative_to(source_dir).as_posix()
        for pair in pairs
        for key in ("image_path", "mask_path")
    }
    if set(source_entries) != expected_source_paths:
        raise DatasetValidationError(
            "IBSI 2 Phase 3 source manifest inventory does not match discovered files"
        )
    for relative in sorted(expected_source_paths):
        observed_hash = sha256_file(source_dir / relative)
        if observed_hash.lower() != source_entries[relative]:
            raise DatasetValidationError(
                f"IBSI 2 Phase 3 source manifest checksum mismatch: {relative}"
            )
    preparer_sources = _preparer_source_records()
    preparer_source_tree_sha256 = _canonical_sha256(preparer_sources)
    parameters = {
        "dataset": DATASET_ID,
        "dataset_kind": "real_world",
        "num_subjects": int(expected_subjects),
        "modalities": list(MODALITIES),
        "copy_policy": "byte_identical_no_transform",
        "derived_texture_representation": "mask_specific_ibsi_fbn32",
        "derived_ct_ivh_representation": "mask_specific_ibsi_fbs1_ivh_indices",
        "derived_noncalibrated_ivh_representation": (
            "mask_specific_ibsi_fbn1000_ivh_indices"
        ),
        "source_manifest_sha256": source_manifest_hash,
        "source_manifest_inventory_verified": True,
        "source_manifest_file_count": len(source_entries),
        "preparer_source_tree_sha256": preparer_source_tree_sha256,
        "official_dataset_page": OFFICIAL_DATASET_PAGE,
        "official_ibsi2_page": OFFICIAL_IBSI2_PAGE,
        "tcia_doi": TCIA_DOI,
    }
    state_path = destination / "preparation_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("parameters") != parameters:
            raise DatasetValidationError(
                "existing IBSI 2 Phase 3 preparation uses different parameters"
            )
        if not resume:
            raise FileExistsError(
                f"IBSI 2 Phase 3 destination already has state: {destination}"
            )
    else:
        state = {
            "schema_version": 1,
            "status": "in_progress",
            "parameters": parameters,
            "prepared_files": {},
        }
        atomic_write_json(state_path, state)
    if source_manifest.is_file():
        atomic_copy(
            source_manifest,
            destination / "source_manifest.json",
            overwrite=False,
        )

    cases: list[dict[str, Any]] = []
    for pair in pairs:
        subject = str(pair["subject"])
        modality = str(pair["modality"])
        image_relative = f"images/{subject}_{modality}_image.nii.gz"
        mask_relative = f"masks/{subject}_{modality}_mask.nii.gz"
        image_record = _copy_record(
            source_dir,
            destination,
            Path(pair["image_path"]),
            image_relative,
            kind="image",
            role="original_source_image",
            resume=resume,
        )
        state["prepared_files"][image_relative] = image_record
        atomic_write_json(state_path, state)
        mask_record = _copy_record(
            source_dir,
            destination,
            Path(pair["mask_path"]),
            mask_relative,
            kind="mask",
            role="binary_roi",
            resume=resume,
        )
        state["prepared_files"][mask_relative] = mask_record
        atomic_write_json(state_path, state)
        image_nifti = nib.load(str(destination / image_relative))
        mask_nifti = nib.load(str(destination / mask_relative))
        image_values = np.asanyarray(image_nifti.dataobj)
        mask_values = np.asanyarray(mask_nifti.dataobj)
        representation = compile_mask_specific_fbn(
            image_values,
            mask_values,
            levels=32,
        )
        discrete_relative = f"discrete/fbn32/{subject}_{modality}_fbn32.nii.gz"
        discrete_path = destination / discrete_relative
        atomic_write_nifti(discrete_path, representation.array, image_nifti.affine)
        derivation_sha256 = _derivation_sha256(
            image_record["sha256"], mask_record["sha256"]
        )
        discrete_record = {
            "source_path": (
                f"derived:IBSI_FBN32({image_record['source_path']},"
                f"{mask_record['source_path']})"
            ),
            "prepared_path": discrete_relative,
            "path": discrete_relative,
            "kind": "image",
            "role": "mask_specific_ibsi_fbn32",
            "sha256": sha256_file(discrete_path),
            "source_sha256": derivation_sha256,
            "derivation_sha256": derivation_sha256,
            "bytes": discrete_path.stat().st_size,
            "dtype": np.dtype(representation.array.dtype).name,
            "shape": [int(value) for value in representation.array.shape],
        }
        state["prepared_files"][discrete_relative] = discrete_record
        atomic_write_json(state_path, state)
        ivh_record = None
        ivh_representation = None
        if modality == "ct":
            ivh_representation = compile_mask_specific_fbs(
                image_values,
                mask_values,
                bin_width=1.0,
            )
            ivh_relative = f"discrete/ivh_fbs1/{subject}_{modality}_ivh_fbs1.nii.gz"
            ivh_path = destination / ivh_relative
            atomic_write_nifti(
                ivh_path,
                ivh_representation.array,
                image_nifti.affine,
            )
            ivh_derivation_sha256 = _canonical_sha256(
                {
                    "method": "IBSI_fixed_bin_size",
                    "bin_width": 1.0,
                    "anchor": ivh_representation.anchor,
                    "image_sha256": image_record["sha256"],
                    "mask_sha256": mask_record["sha256"],
                }
            )
            ivh_record = {
                "source_path": (
                    f"derived:IBSI_FBS1_IVH({image_record['source_path']},"
                    f"{mask_record['source_path']})"
                ),
                "prepared_path": ivh_relative,
                "path": ivh_relative,
                "kind": "image",
                "role": "mask_specific_ibsi_fbs1_ivh_indices",
                "sha256": sha256_file(ivh_path),
                "source_sha256": ivh_derivation_sha256,
                "derivation_sha256": ivh_derivation_sha256,
                "bytes": ivh_path.stat().st_size,
                "dtype": np.dtype(ivh_representation.array.dtype).name,
                "shape": [int(value) for value in ivh_representation.array.shape],
            }
            state["prepared_files"][ivh_relative] = ivh_record
            atomic_write_json(state_path, state)
            ivh_metadata = {
                "id": "mask_specific_ibsi_fbs1_ivh_indices",
                "method": "IBSI_fixed_bin_size",
                "bin_width": 1.0,
                "anchor": ivh_representation.anchor,
                "configured_levels": ivh_representation.configured_levels,
                "occupied_levels": ivh_representation.occupied_levels,
                "roi_min": ivh_representation.roi_min,
                "roi_max": ivh_representation.roi_max,
                "background_value": 0,
                "derivation_sha256": ivh_record["derivation_sha256"],
            }
        else:
            ivh_representation = compile_mask_specific_fbn(
                image_values,
                mask_values,
                levels=1000,
            )
            ivh_relative = (
                f"discrete/ivh_fbn1000/{subject}_{modality}_ivh_fbn1000.nii.gz"
            )
            ivh_path = destination / ivh_relative
            atomic_write_nifti(
                ivh_path,
                ivh_representation.array,
                image_nifti.affine,
            )
            ivh_derivation_sha256 = _canonical_sha256(
                {
                    "method": "IBSI_fixed_bin_number",
                    "configured_levels": 1000,
                    "image_sha256": image_record["sha256"],
                    "mask_sha256": mask_record["sha256"],
                }
            )
            ivh_record = {
                "source_path": (
                    f"derived:IBSI_FBN1000_IVH({image_record['source_path']},"
                    f"{mask_record['source_path']})"
                ),
                "prepared_path": ivh_relative,
                "path": ivh_relative,
                "kind": "image",
                "role": "mask_specific_ibsi_fbn1000_ivh_indices",
                "sha256": sha256_file(ivh_path),
                "source_sha256": ivh_derivation_sha256,
                "derivation_sha256": ivh_derivation_sha256,
                "bytes": ivh_path.stat().st_size,
                "dtype": np.dtype(ivh_representation.array.dtype).name,
                "shape": [int(value) for value in ivh_representation.array.shape],
            }
            state["prepared_files"][ivh_relative] = ivh_record
            atomic_write_json(state_path, state)
            ivh_metadata = {
                "id": "mask_specific_ibsi_fbn1000_ivh_indices",
                "method": "IBSI_fixed_bin_number",
                "configured_levels": 1000,
                "occupied_levels": ivh_representation.occupied_levels,
                "roi_min": ivh_representation.roi_min,
                "roi_max": ivh_representation.roi_max,
                "background_value": 0,
                "derivation_sha256": ivh_record["derivation_sha256"],
            }
        metadata = nifti_case_metadata(
            destination / image_relative,
            destination / mask_relative,
            inspect_values=True,
            inspect_image_values=True,
        )
        cases.append(
            {
                "case_id": f"{subject}_{modality}",
                "subject_id": subject,
                "modality": modality,
                "image_path": image_relative,
                "image_sha256": image_record["sha256"],
                "source_image_sha256": image_record["source_sha256"],
                "mask_path": mask_relative,
                "mask_sha256": mask_record["sha256"],
                "source_mask_sha256": mask_record["source_sha256"],
                "discrete_image_path": discrete_relative,
                "discrete_image_sha256": discrete_record["sha256"],
                "ivh_image_path": ivh_record["path"],
                "ivh_image_sha256": ivh_record["sha256"],
                "ivh_representation": ivh_metadata,
                "texture_workload": texture_workload(representation.array, mask_values),
                "texture_representation": {
                    "id": "mask_specific_ibsi_fbn32",
                    "configured_levels": 32,
                    "occupied_levels": representation.occupied_levels,
                    "roi_min": representation.roi_min,
                    "roi_max": representation.roi_max,
                    "background_value": 0,
                    "derivation_sha256": derivation_sha256,
                },
                **metadata,
            }
        )

    prepared_records = [
        state["prepared_files"][key] for key in sorted(state["prepared_files"])
    ]
    files = [
        {
            "path": record["path"],
            "kind": record["kind"],
            "role": record["role"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
            "dtype": record["dtype"],
            "shape": record["shape"],
        }
        for record in prepared_records
    ]
    preparation = {
        "schema_version": 1,
        "status": "complete",
        "parameters": parameters,
        "preparer_source_files": preparer_sources,
        "prepared_files": [
            {
                "source_path": record["source_path"],
                "prepared_path": record["prepared_path"],
                "kind": record["kind"],
                "role": record["role"],
                "sha256": record["sha256"],
                "source_sha256": record["source_sha256"],
                "bytes": record["bytes"],
                "dtype": record["dtype"],
                "shape": record["shape"],
            }
            for record in prepared_records
        ],
    }
    atomic_write_json(destination / "preparation.json", preparation)
    manifest = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "dataset_kind": "real_world",
        "description": "IBSI 2 Phase 3 51-subject CT/MRI/PET STS cohort",
        "requested_subjects": int(expected_subjects),
        "modalities": list(MODALITIES),
        "transform_policy": "none_byte_identical_copy",
        "redistribution": {
            "status": "project_owner_confirmed_allowed",
            "required_practice": "retain upstream attribution, source links, and bound checksums",
            "legal_opinion": False,
        },
        "representation_contract": {
            "raw": "original_source_image",
            "texture": {
                "id": "mask_specific_ibsi_fbn32",
                "configured_levels": 32,
                "background_value": 0,
                "roi_value_range": [1, 32],
            },
            "ivh_ct": {
                "id": "mask_specific_ibsi_fbs1_ivh_indices",
                "method": "IBSI_fixed_bin_size",
                "bin_width": 1.0,
                "background_value": 0,
            },
            "ivh_mri_pet": {
                "id": "mask_specific_ibsi_fbn1000_ivh_indices",
                "method": "IBSI_fixed_bin_number",
                "configured_levels": 1000,
                "background_value": 0,
            },
        },
        "source_links": {
            "ibsi_datasets": OFFICIAL_DATASET_PAGE,
            "ibsi2": OFFICIAL_IBSI2_PAGE,
            "tcia_doi": TCIA_DOI,
        },
        "provenance": {
            "preparation_record": "preparation.json",
            "preparation_record_sha256": sha256_file(destination / "preparation.json"),
            "source_manifest_path": (
                "source_manifest.json" if source_manifest.is_file() else None
            ),
            "source_manifest_sha256": source_manifest_hash,
            "source_manifest_inventory_verified": True,
            "source_manifest_file_count": len(source_entries),
            "preparer_source_tree_sha256": preparer_source_tree_sha256,
        },
        "files": files,
        "cases": sorted(cases, key=lambda case: case["case_id"]),
    }
    atomic_write_json(destination / "manifest.json", manifest)
    atomic_write_csv(
        destination / "cases.csv",
        [
            {
                "case_id": case["case_id"],
                "subject": case["subject_id"],
                "modality": case["modality"],
                "shape_xyz": "x".join(str(value) for value in case["shape"]),
                "spacing_mm_xyz": "x".join(f"{value:.8g}" for value in case["spacing"]),
                "image_voxels": case["image_voxels"],
                "mask_voxels": case["mask_voxels"],
                "mask_fraction_percent": 100.0 * case["mask_fraction"],
                "image_dtype": next(
                    entry["dtype"]
                    for entry in files
                    if entry["path"] == case["image_path"]
                ),
                "occupied_fbn_levels": case["texture_representation"][
                    "occupied_levels"
                ],
                "ivh_representation_id": case["ivh_representation"]["id"],
                "occupied_ivh_levels": case["ivh_representation"]["occupied_levels"],
            }
            for case in sorted(cases, key=lambda case: case["case_id"])
        ],
    )
    atomic_write_bytes(destination / "README.md", _readme_text().encode("utf-8"))
    state["status"] = "complete"
    state["manifest_sha256"] = sha256_file(destination / "manifest.json")
    atomic_write_json(state_path, state)
    return manifest


def validate_ibsi2_phase3_dataset(
    dataset_dir: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest, core = load_and_validate_manifest(
        dataset_dir,
        verify_hashes=verify_hashes,
        inspect_geometry=True,
        inspect_values=True,
    )
    if manifest.get("dataset") != DATASET_ID:
        raise DatasetValidationError("unexpected IBSI 2 Phase 3 dataset identifier")
    if (
        manifest.get("provenance", {}).get("source_manifest_inventory_verified")
        is not True
    ):
        raise DatasetValidationError("IBSI 2 Phase 3 source inventory is not attested")
    cases = manifest["cases"]
    subjects = {case["subject_id"] for case in cases}
    expected_subjects = int(manifest["requested_subjects"])
    if len(subjects) != expected_subjects or len(cases) != expected_subjects * 3:
        raise DatasetValidationError(
            "IBSI 2 Phase 3 subject/modality grid is incomplete"
        )
    blocks = Counter(case["subject_id"] for case in cases)
    if any(value != 3 for value in blocks.values()):
        raise DatasetValidationError("IBSI 2 Phase 3 subject block is incomplete")
    if {case["modality"] for case in cases} != set(MODALITIES):
        raise DatasetValidationError("IBSI 2 Phase 3 modality set is incomplete")
    for case in cases:
        if case["image_sha256"] != case["source_image_sha256"]:
            raise DatasetValidationError(
                f"image changed during copy: {case['case_id']}"
            )
        if case["mask_sha256"] != case["source_mask_sha256"]:
            raise DatasetValidationError(f"mask changed during copy: {case['case_id']}")
        image_nifti = nib.load(str(dataset_dir / case["image_path"]))
        mask_nifti = nib.load(str(dataset_dir / case["mask_path"]))
        discrete_nifti = nib.load(str(dataset_dir / case["discrete_image_path"]))
        image = np.asanyarray(image_nifti.dataobj)
        mask = np.asanyarray(mask_nifti.dataobj)
        discrete = np.asanyarray(discrete_nifti.dataobj)
        expected = compile_mask_specific_fbn(image, mask, levels=32)
        if discrete.dtype != np.dtype(np.uint8) or not np.array_equal(
            discrete, expected.array
        ):
            raise DatasetValidationError(
                f"derived FBN32 representation mismatch: {case['case_id']}"
            )
        if (
            case["texture_representation"]["occupied_levels"]
            != expected.occupied_levels
        ):
            raise DatasetValidationError(
                f"derived FBN32 provenance mismatch: {case['case_id']}"
            )
        if case["modality"] == "ct":
            expected_ivh = compile_mask_specific_fbs(image, mask, bin_width=1.0)
            ivh_nifti = nib.load(str(dataset_dir / case["ivh_image_path"]))
            observed_ivh = np.asanyarray(ivh_nifti.dataobj)
            if not np.array_equal(observed_ivh, expected_ivh.array):
                raise DatasetValidationError(
                    f"derived FBS1 IVH representation mismatch: {case['case_id']}"
                )
            if case["ivh_representation"]["configured_levels"] != (
                expected_ivh.configured_levels
            ):
                raise DatasetValidationError(
                    f"derived FBS1 IVH provenance mismatch: {case['case_id']}"
                )
        else:
            expected_ivh = compile_mask_specific_fbn(image, mask, levels=1000)
            ivh_nifti = nib.load(str(dataset_dir / case["ivh_image_path"]))
            observed_ivh = np.asanyarray(ivh_nifti.dataobj)
            if not np.array_equal(observed_ivh, expected_ivh.array):
                raise DatasetValidationError(
                    f"derived FBN1000 IVH representation mismatch: {case['case_id']}"
                )
            if case["ivh_representation"]["configured_levels"] != 1000:
                raise DatasetValidationError(
                    f"derived FBN1000 IVH provenance mismatch: {case['case_id']}"
                )
    return {
        **core,
        "subject_count": len(subjects),
        "modality_count": len(MODALITIES),
        "complete_subject_blocks": len(subjects),
        "byte_identical_source_copy": True,
        "derived_fbn32_cases_validated": len(cases),
        "derived_fbs1_ivh_cases_validated": sum(
            case["modality"] == "ct" for case in cases
        ),
        "derived_fbn1000_ivh_cases_validated": sum(
            case["modality"] in {"mri", "pet"} for case in cases
        ),
        "ready_for_fixed_cohort_analysis": True,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--expected-subjects", type=int, default=51)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    destination = Path(arguments.dest_dir)
    if not arguments.validate_only:
        prepare_ibsi2_phase3_dataset(
            Path(arguments.source_dir),
            destination,
            expected_subjects=arguments.expected_subjects,
            resume=arguments.resume,
        )
    summary = validate_ibsi2_phase3_dataset(destination)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DATASET_ID",
    "prepare_ibsi2_phase3_dataset",
    "validate_ibsi2_phase3_dataset",
]
