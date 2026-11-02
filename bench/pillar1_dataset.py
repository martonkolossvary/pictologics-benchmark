"""Public, restart-safe builder for the frozen Pillar 1 benchmark dataset.

The builder writes three hash-bound image representations for every logical
image--mask case:

* the original generated ``int16`` CT image in HU;
* a mask-specific, one-based IBSI fixed-bin-number image for histogram and
  texture calculations (zero is reserved outside the ROI).
* a mask-specific, one-based IBSI fixed-bin-size image for IVH calculation.

No radiomics implementation is imported or executed here.  All adapters must
consume the committed NIfTI bytes described by ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
from scipy import ndimage

from bench.dataset_manifest import (
    DatasetValidationError,
    atomic_copy,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_nifti,
    load_and_validate_manifest,
    sha256_file,
)
from bench.synthetic_volume import HU_PROFILES, VolumeConfig
from bench.synthetic_representations import (
    compile_mask_specific_fbn,
    compile_mask_specific_fbs,
    texture_workload,
)
from bench.synthetic_volume import (
    generate_synthetic_volume,
)


PILLAR1_SCHEMA_VERSION = 1
PILLAR1_CASE_SCHEMA_VERSION = 1
PILLAR1_MASK_IDS = ("M1", "M2", "M3", "M4")
DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1] / "configs" / "benchmark" / "pillar1.json"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_tree_sha256() -> tuple[str, list[dict[str, str]]]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "bench" / "synthetic_representations.py",
        root / "bench" / "synthetic_volume.py",
    )
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return _canonical_sha256(records), records


def load_pillar1_profile(path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"unable to read Pillar 1 profile: {path}"
        ) from exc
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise DatasetValidationError("unsupported Pillar 1 profile schema")
    required_exact = {
        "dataset_kind": "synthetic",
        "coordinate_system": "RAS+",
        "axis_order": "xyz",
        "fov_mm": 256.0,
        "mask_ids": list(PILLAR1_MASK_IDS),
    }
    for key, expected in required_exact.items():
        if profile.get(key) != expected:
            raise DatasetValidationError(
                f"Pillar 1 profile {key} must equal {expected!r}"
            )
    sizes = profile.get("sizes")
    profiles = profile.get("hu_profiles")
    if (
        not isinstance(sizes, list)
        or not sizes
        or len(set(sizes)) != len(sizes)
        or any(not isinstance(value, int) or value < 32 for value in sizes)
    ):
        raise DatasetValidationError("Pillar 1 sizes must be unique integers >= 32")
    if (
        not isinstance(profiles, list)
        or not profiles
        or len(set(profiles)) != len(profiles)
        or any(value not in HU_PROFILES for value in profiles)
    ):
        raise DatasetValidationError("Pillar 1 hu_profiles are invalid")
    texture = profile.get("texture_representation")
    if not isinstance(texture, dict) or texture.get("configured_levels") != 32:
        raise DatasetValidationError(
            "Pillar 1 requires the frozen FBN32 representation"
        )
    expected_cases = len(sizes) * len(profiles) * len(PILLAR1_MASK_IDS)
    expected_counts = {
        "case_count": expected_cases,
        "raw_image_count": len(sizes) * len(profiles),
        "mask_count": len(sizes) * len(PILLAR1_MASK_IDS),
        "discrete_image_count": expected_cases,
    }
    if "ivh_representation" in profile:
        ivh = profile.get("ivh_representation")
        if not isinstance(ivh, dict) or float(ivh.get("bin_width", 0.0)) != 1.0:
            raise DatasetValidationError("Pillar 1 IVH representation must use FBS1")
        expected_counts["ivh_image_count"] = expected_cases
    for key, expected in expected_counts.items():
        if profile.get(key) != expected:
            raise DatasetValidationError(
                f"Pillar 1 profile {key}={profile.get(key)!r}; expected {expected}"
            )
    return profile


def _artifact_record(
    root: Path,
    path: Path,
    *,
    kind: str,
    role: str,
    data: np.ndarray,
    size: int,
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "role": role,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": np.dtype(data.dtype).name,
        "shape": [int(value) for value in data.shape],
        "size": int(size),
        "modality": "synthetic",
    }


def _commit_artifact(
    root: Path,
    relative: str,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    kind: str,
    role: str,
    size: int,
    state: dict[str, Any],
    state_path: Path,
    resume: bool,
) -> dict[str, Any]:
    destination = root / relative
    existing = state["artifacts"].get(relative)
    if resume and isinstance(existing, dict) and destination.is_file():
        if (
            existing.get("bytes") == destination.stat().st_size
            and existing.get("sha256") == sha256_file(destination)
            and existing.get("dtype") == np.dtype(data.dtype).name
            and existing.get("shape") == [int(value) for value in data.shape]
        ):
            return dict(existing)
    atomic_write_nifti(destination, data, affine)
    record = _artifact_record(
        root,
        destination,
        kind=kind,
        role=role,
        data=data,
        size=size,
    )
    state["artifacts"][relative] = record
    atomic_write_json(state_path, state)
    return record


def _case_id(profile_id: str, mask_id: str, size: int) -> str:
    return f"p1_{profile_id}_{mask_id.lower()}_n{size:03d}"


def _cavities_background_6(mask: np.ndarray) -> int:
    coordinates = np.argwhere(mask)
    lower = np.maximum(coordinates.min(axis=0) - 1, 0)
    upper = np.minimum(coordinates.max(axis=0) + 2, np.asarray(mask.shape))
    region = mask[
        tuple(slice(int(left), int(right)) for left, right in zip(lower, upper))
    ]
    padded = np.pad(region, 1, constant_values=False)
    labels, count = ndimage.label(
        ~padded,
        structure=ndimage.generate_binary_structure(3, 1),
    )
    exterior = int(labels[0, 0, 0])
    return int(sum(identifier != exterior for identifier in range(1, int(count) + 1)))


def _measure_mask(
    mask: np.ndarray,
    *,
    body_voxels: int,
    spacing_mm: float,
) -> dict[str, Any]:
    roi = np.asarray(mask) != 0
    coordinates = np.argwhere(roi)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    slices = tuple(
        slice(int(left), int(right) + 1) for left, right in zip(lower, upper)
    )
    cropped = roi[slices]
    components = int(
        ndimage.label(cropped, structure=np.ones((3, 3, 3), dtype=np.uint8))[1]
    )
    count = int(coordinates.shape[0])
    z_slices = int(upper[2] - lower[2] + 1)
    return {
        "voxels": count,
        "image_fraction": float(count / roi.size),
        "body_fraction": float(count / body_voxels),
        "bbox_shape_xyz": tuple(int(value) for value in upper - lower + 1),
        "occupied_z_slices": z_slices,
        "occupied_z_mm": float(z_slices * spacing_mm),
        "components_26": components,
        "cavities_background_6": _cavities_background_6(roi),
    }


def _readme_text(profile: Mapping[str, Any]) -> str:
    return f"""# Pillar 1 public benchmark dataset

This directory was generated from the frozen `{profile["dataset"]}` profile.
It contains {profile["case_count"]} three-dimensional image–mask cases:
{len(profile["hu_profiles"])} CT-inspired HU profiles ×
{len(profile["mask_ids"])} morphology masks × {len(profile["sizes"])} cubic grids.

For every case, `manifest.json` binds:

- the original `int16` HU image used for morphology, local intensity,
  intensity statistics, and calibrated-CT IVH;
- the binary mask;
- a mask-specific IBSI FBN32 image used for the intensity histogram and six
  texture families. Its ROI values are 1–32 and its outside-mask value is 0.
- a mask-specific FBS1 bin-index image used for IVH. Its affine mapping back
  to HU is recorded per case; adapters consume the committed indices directly.

The FBN32 files are inputs, not adapter outputs. Benchmark adapters must not
re-bin them. File loading and hash validation occur outside timed feature
extraction. `generation.json` and `generation_state.json` support audit and
restart, while `cases.csv` is a reader-friendly index. No timing results are
stored here.

Regenerate with:

```bash
poetry run bench generate-pillar1 --profile configs/benchmark/pillar1.json \\
  --dest-dir data/benchmark/pillar1 --resume
```
"""


def build_pillar1_dataset(
    destination: Path,
    *,
    profile_path: Path = DEFAULT_PROFILE,
    resume: bool = True,
) -> dict[str, Any]:
    """Build or resume the complete frozen Pillar 1 dataset."""

    destination = destination.expanduser().resolve()
    profile_path = profile_path.expanduser().resolve()
    profile = load_pillar1_profile(profile_path)
    profile_sha256 = sha256_file(profile_path)
    source_tree_sha256, source_files = _source_tree_sha256()
    parameters = {
        "dataset": profile["dataset"],
        "dataset_kind": "synthetic",
        "profile_sha256": profile_sha256,
        "source_tree_sha256": source_tree_sha256,
        "generator": profile["generator"],
        "generator_version": profile["generator_version"],
        "seed": int(profile["seed"]),
        "sizes": list(profile["sizes"]),
        "variants": list(profile["hu_profiles"]),
        "mask_ids": list(profile["mask_ids"]),
        "fov_mm": float(profile["fov_mm"]),
        "slab_depth": int(profile["slab_depth"]),
        "raw_representation": dict(profile["raw_representation"]),
        "texture_representation": dict(profile["texture_representation"]),
        "ivh_representation": dict(profile.get("ivh_representation", {})),
    }
    parameters_sha256 = _canonical_sha256(parameters)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "generation_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("parameters_sha256") != parameters_sha256:
            raise DatasetValidationError(
                "existing Pillar 1 generation state belongs to a different profile"
            )
        if not resume:
            raise FileExistsError(
                f"Pillar 1 destination already has generation state: {destination}"
            )
    else:
        state = {
            "schema_version": PILLAR1_SCHEMA_VERSION,
            "status": "in_progress",
            "parameters": parameters,
            "parameters_sha256": parameters_sha256,
            "artifacts": {},
            "cases": {},
        }
        atomic_write_json(state_path, state)
    atomic_copy(profile_path, destination / "profile.json", overwrite=False)

    levels = int(profile["texture_representation"]["configured_levels"])
    for size in profile["sizes"]:
        base_config = VolumeConfig(
            edge=int(size),
            fov_mm=float(profile["fov_mm"]),
            seed=int(profile["seed"]),
            slab_depth=int(profile["slab_depth"]),
            hu_profile="reference",
        )
        reference = generate_synthetic_volume(base_config)
        body_voxels = int(np.count_nonzero(reference.masks["A1"]))
        reference_metrics = {
            mask_id: _measure_mask(
                reference.masks[mask_id],
                body_voxels=body_voxels,
                spacing_mm=float(reference.spacing_mm[0]),
            )
            for mask_id in PILLAR1_MASK_IDS
        }
        mask_records: dict[str, dict[str, Any]] = {}
        for mask_id in PILLAR1_MASK_IDS:
            relative = f"masks/n{size:03d}_{mask_id.lower()}_mask.nii.gz"
            mask_records[mask_id] = _commit_artifact(
                destination,
                relative,
                np.asarray(reference.masks[mask_id], dtype=np.uint8),
                reference.affine,
                kind="mask",
                role="binary_roi",
                size=int(size),
                state=state,
                state_path=state_path,
                resume=resume,
            )

        for profile_id in profile["hu_profiles"]:
            if profile_id == "reference":
                bundle = reference
            else:
                bundle = generate_synthetic_volume(
                    VolumeConfig(
                        edge=int(size),
                        fov_mm=float(profile["fov_mm"]),
                        seed=int(profile["seed"]),
                        slab_depth=int(profile["slab_depth"]),
                        hu_profile=str(profile_id),
                    )
                )
                for mask_id in PILLAR1_MASK_IDS:
                    if not np.array_equal(
                        bundle.masks[mask_id], reference.masks[mask_id]
                    ):
                        raise RuntimeError(
                            f"geometry changed with HU profile: n={size}, mask={mask_id}"
                        )
            raw_relative = f"raw/{profile_id}/n{size:03d}_image.nii.gz"
            raw_record = _commit_artifact(
                destination,
                raw_relative,
                np.asarray(bundle.image, dtype=np.int16),
                bundle.affine,
                kind="image",
                role="original_hu_int16",
                size=int(size),
                state=state,
                state_path=state_path,
                resume=resume,
            )
            for mask_id in PILLAR1_MASK_IDS:
                mask = np.asarray(reference.masks[mask_id], dtype=np.uint8)
                representation = compile_mask_specific_fbn(
                    bundle.image,
                    mask,
                    levels=levels,
                )
                discrete_relative = (
                    f"discrete/fbn32/{profile_id}/"
                    f"n{size:03d}_{mask_id.lower()}_fbn32.nii.gz"
                )
                discrete_record = _commit_artifact(
                    destination,
                    discrete_relative,
                    representation.array,
                    bundle.affine,
                    kind="image",
                    role="mask_specific_ibsi_fbn32",
                    size=int(size),
                    state=state,
                    state_path=state_path,
                    resume=resume,
                )
                ivh_representation = None
                ivh_record = None
                if profile.get("ivh_representation"):
                    ivh_representation = compile_mask_specific_fbs(
                        bundle.image,
                        mask,
                        bin_width=float(profile["ivh_representation"]["bin_width"]),
                    )
                    ivh_relative = (
                        f"discrete/ivh_fbs1/{profile_id}/"
                        f"n{size:03d}_{mask_id.lower()}_ivh_fbs1.nii.gz"
                    )
                    ivh_record = _commit_artifact(
                        destination,
                        ivh_relative,
                        ivh_representation.array,
                        bundle.affine,
                        kind="image",
                        role="mask_specific_ibsi_fbs1_ivh_indices",
                        size=int(size),
                        state=state,
                        state_path=state_path,
                        resume=resume,
                    )
                mask_metric = reference_metrics[mask_id]
                identifier = _case_id(str(profile_id), mask_id, int(size))
                state["cases"][identifier] = {
                    "case_schema_version": PILLAR1_CASE_SCHEMA_VERSION,
                    "case_id": identifier,
                    "modality": "synthetic",
                    "subject_id": str(profile_id),
                    "hu_profile": str(profile_id),
                    "mask_id": mask_id,
                    "size": int(size),
                    "image_path": raw_record["path"],
                    "image_sha256": raw_record["sha256"],
                    "mask_path": mask_records[mask_id]["path"],
                    "mask_sha256": mask_records[mask_id]["sha256"],
                    "discrete_image_path": discrete_record["path"],
                    "discrete_image_sha256": discrete_record["sha256"],
                    **(
                        {
                            "ivh_image_path": ivh_record["path"],
                            "ivh_image_sha256": ivh_record["sha256"],
                        }
                        if ivh_record is not None
                        else {}
                    ),
                    "shape": [int(size)] * 3,
                    "spacing": [float(value) for value in bundle.spacing_mm],
                    "orientation": ["R", "A", "S"],
                    "affine": [
                        [float(value) for value in row]
                        for row in bundle.affine.tolist()
                    ],
                    "image_voxels": int(size**3),
                    "complexity": int(size**3),
                    "mask_voxels": int(mask_metric["voxels"]),
                    "mask_fraction": float(mask_metric["image_fraction"]),
                    "mask_body_fraction": float(mask_metric["body_fraction"]),
                    "bbox_shape_xyz": list(mask_metric["bbox_shape_xyz"]),
                    "occupied_z_slices": int(mask_metric["occupied_z_slices"]),
                    "occupied_z_mm": float(mask_metric["occupied_z_mm"]),
                    "components_26": int(mask_metric["components_26"]),
                    "cavities_background_6": int(mask_metric["cavities_background_6"]),
                    "raw_representation": profile["raw_representation"]["id"],
                    "texture_workload": texture_workload(representation.array, mask),
                    "texture_representation": {
                        "id": profile["texture_representation"]["id"],
                        "configured_levels": representation.configured_levels,
                        "occupied_levels": representation.occupied_levels,
                        "roi_min_hu": representation.roi_min,
                        "roi_max_hu": representation.roi_max,
                        "background_value": representation.background_value,
                    },
                    **(
                        {
                            "ivh_representation": {
                                "id": profile["ivh_representation"]["id"],
                                "method": "IBSI_fixed_bin_size",
                                "bin_width": ivh_representation.bin_width,
                                "anchor_hu": ivh_representation.anchor,
                                "configured_levels": ivh_representation.configured_levels,
                                "occupied_levels": ivh_representation.occupied_levels,
                                "roi_min_hu": ivh_representation.roi_min,
                                "roi_max_hu": ivh_representation.roi_max,
                                "background_value": ivh_representation.background_value,
                                "derivation_sha256": _canonical_sha256(
                                    {
                                        "method": "IBSI_fixed_bin_size",
                                        "bin_width": ivh_representation.bin_width,
                                        "anchor_hu": ivh_representation.anchor,
                                        "image_sha256": raw_record["sha256"],
                                        "mask_sha256": mask_records[mask_id]["sha256"],
                                    }
                                ),
                            }
                        }
                        if ivh_representation is not None
                        else {}
                    ),
                }
                atomic_write_json(state_path, state)

    cases = [state["cases"][key] for key in sorted(state["cases"])]
    artifacts = [state["artifacts"][key] for key in sorted(state["artifacts"])]
    if len(cases) != profile["case_count"]:
        raise RuntimeError(
            f"generated {len(cases)} cases; expected {profile['case_count']}"
        )
    expected_artifacts = (
        profile["raw_image_count"]
        + profile["mask_count"]
        + profile["discrete_image_count"]
        + int(profile.get("ivh_image_count", 0))
    )
    if len(artifacts) != expected_artifacts:
        raise RuntimeError(
            f"generated {len(artifacts)} artifacts; expected {expected_artifacts}"
        )

    generation = {
        "schema_version": 1,
        "status": "complete",
        "parameters": parameters,
        "parameters_sha256": parameters_sha256,
        "source_files": source_files,
        "artifact_count": len(artifacts),
        "case_count": len(cases),
        "artifact_inventory_sha256": _canonical_sha256(artifacts),
        "case_inventory_sha256": _canonical_sha256(cases),
    }
    atomic_write_json(destination / "generation.json", generation)
    manifest = {
        "schema_version": 1,
        "dataset": profile["dataset"],
        "dataset_kind": "synthetic",
        "description": profile["description"],
        "coordinate_system": profile["coordinate_system"],
        "axis_order": profile["axis_order"],
        "sizes": list(profile["sizes"]),
        "variants": list(profile["hu_profiles"]),
        "mask_ids": list(profile["mask_ids"]),
        "representation_contract": {
            "raw": profile["raw_representation"],
            "texture": profile["texture_representation"],
            **(
                {"ivh": profile["ivh_representation"]}
                if profile.get("ivh_representation")
                else {}
            ),
        },
        "provenance": {
            "generation_record": "generation.json",
            "generation_record_sha256": sha256_file(destination / "generation.json"),
            "parameters_sha256": parameters_sha256,
            "profile_path": "profile.json",
            "profile_sha256": sha256_file(destination / "profile.json"),
            "source_tree_sha256": source_tree_sha256,
        },
        "files": artifacts,
        "cases": cases,
    }
    atomic_write_json(destination / "manifest.json", manifest)
    csv_rows = [
        {
            "case_id": case["case_id"],
            "profile": case["hu_profile"],
            "mask": case["mask_id"],
            "shape_xyz": "x".join(str(value) for value in case["shape"]),
            "spacing_mm": case["spacing"][0],
            "image_voxels": case["image_voxels"],
            "mask_voxels": case["mask_voxels"],
            "mask_fraction_percent": 100.0 * case["mask_fraction"],
            "z_slices": case["occupied_z_slices"],
            "occupied_fbn_levels": case["texture_representation"]["occupied_levels"],
            "raw_path": case["image_path"],
            "mask_path": case["mask_path"],
            "fbn32_path": case["discrete_image_path"],
            "ivh_fbs1_path": case.get("ivh_image_path"),
        }
        for case in cases
    ]
    atomic_write_csv(destination / "cases.csv", csv_rows)
    atomic_write_bytes(destination / "README.md", _readme_text(profile).encode("utf-8"))
    state["status"] = "complete"
    state["manifest_sha256"] = sha256_file(destination / "manifest.json")
    atomic_write_json(state_path, state)
    return manifest


def validate_pillar1_dataset(
    dataset_dir: Path,
    *,
    verify_hashes: bool = True,
    deep: bool = True,
) -> dict[str, Any]:
    """Validate inventory, geometry, and (optionally) every FBN voxel."""

    dataset_dir = dataset_dir.expanduser().resolve()
    manifest, core = load_and_validate_manifest(
        dataset_dir,
        verify_hashes=verify_hashes,
        inspect_geometry=True,
        inspect_values=True,
    )
    profile = load_pillar1_profile(dataset_dir / "profile.json")
    if manifest.get("dataset") != profile.get("dataset"):
        raise DatasetValidationError("Pillar 1 profile/manifest dataset mismatch")
    expected = {
        (profile_id, mask_id, int(size))
        for profile_id in profile["hu_profiles"]
        for mask_id in profile["mask_ids"]
        for size in profile["sizes"]
    }
    observed = {
        (case.get("hu_profile"), case.get("mask_id"), case.get("size"))
        for case in manifest["cases"]
    }
    if observed != expected:
        raise DatasetValidationError("Pillar 1 case cross-product is incomplete")

    files_by_path = {entry["path"]: entry for entry in manifest["files"]}
    role_counts: defaultdict[str, int] = defaultdict(int)
    for entry in manifest["files"]:
        role_counts[str(entry.get("role"))] += 1
    expected_roles = {
        "original_hu_int16": int(profile["raw_image_count"]),
        "binary_roi": int(profile["mask_count"]),
        "mask_specific_ibsi_fbn32": int(profile["discrete_image_count"]),
    }
    if profile.get("ivh_representation"):
        expected_roles["mask_specific_ibsi_fbs1_ivh_indices"] = int(
            profile["ivh_image_count"]
        )
    if dict(role_counts) != expected_roles:
        raise DatasetValidationError(
            f"Pillar 1 artifact roles mismatch: {dict(role_counts)}"
        )

    checked_cases = 0
    if deep:
        grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for case in manifest["cases"]:
            grouped[str(case["image_path"])].append(case)
        for image_relative, cases in sorted(grouped.items()):
            image_nifti = nib.load(str(dataset_dir / image_relative))
            raw = np.asanyarray(image_nifti.dataobj)
            if raw.dtype != np.dtype(np.int16):
                raise DatasetValidationError(
                    f"raw image is not int16: {image_relative}"
                )
            for case in cases:
                mask_nifti = nib.load(str(dataset_dir / case["mask_path"]))
                discrete_nifti = nib.load(
                    str(dataset_dir / case["discrete_image_path"])
                )
                mask = np.asanyarray(mask_nifti.dataobj)
                discrete = np.asanyarray(discrete_nifti.dataobj)
                if discrete.dtype != np.dtype(np.uint8):
                    raise DatasetValidationError(
                        f"FBN image is not uint8: {case['discrete_image_path']}"
                    )
                if raw.shape != mask.shape or raw.shape != discrete.shape:
                    raise DatasetValidationError(
                        f"representation shape mismatch: {case['case_id']}"
                    )
                if not np.allclose(
                    image_nifti.affine, mask_nifti.affine
                ) or not np.allclose(image_nifti.affine, discrete_nifti.affine):
                    raise DatasetValidationError(
                        f"representation affine mismatch: {case['case_id']}"
                    )
                roi = mask != 0
                if np.any(discrete[~roi] != 0):
                    raise DatasetValidationError(
                        f"FBN background is nonzero: {case['case_id']}"
                    )
                if int(np.min(discrete[roi])) < 1 or int(np.max(discrete[roi])) > 32:
                    raise DatasetValidationError(
                        f"FBN ROI is outside 1..32: {case['case_id']}"
                    )
                recomputed = compile_mask_specific_fbn(raw, mask, levels=32)
                if not np.array_equal(discrete, recomputed.array):
                    raise DatasetValidationError(
                        f"FBN bytes do not match the raw image and mask: {case['case_id']}"
                    )
                declared = case["texture_representation"]
                if (
                    declared["occupied_levels"] != recomputed.occupied_levels
                    or not math.isclose(declared["roi_min_hu"], recomputed.roi_min)
                    or not math.isclose(declared["roi_max_hu"], recomputed.roi_max)
                ):
                    raise DatasetValidationError(
                        f"FBN provenance mismatch: {case['case_id']}"
                    )
                entry = files_by_path[case["discrete_image_path"]]
                if case["discrete_image_sha256"] != entry["sha256"]:
                    raise DatasetValidationError(
                        f"FBN hash metadata mismatch: {case['case_id']}"
                    )
                if profile.get("ivh_representation"):
                    ivh_nifti = nib.load(str(dataset_dir / case["ivh_image_path"]))
                    ivh_array = np.asanyarray(ivh_nifti.dataobj)
                    expected_ivh = compile_mask_specific_fbs(
                        raw,
                        mask,
                        bin_width=float(profile["ivh_representation"]["bin_width"]),
                    )
                    if not np.array_equal(ivh_array, expected_ivh.array):
                        raise DatasetValidationError(
                            f"FBS1 IVH bytes do not match raw/mask: {case['case_id']}"
                        )
                    if not np.allclose(image_nifti.affine, ivh_nifti.affine):
                        raise DatasetValidationError(
                            f"IVH representation affine mismatch: {case['case_id']}"
                        )
                    declared_ivh = case["ivh_representation"]
                    if (
                        declared_ivh["configured_levels"]
                        != expected_ivh.configured_levels
                        or declared_ivh["occupied_levels"]
                        != expected_ivh.occupied_levels
                        or not math.isclose(
                            declared_ivh["anchor_hu"], expected_ivh.anchor
                        )
                    ):
                        raise DatasetValidationError(
                            f"FBS1 IVH provenance mismatch: {case['case_id']}"
                        )
                checked_cases += 1

    return {
        **core,
        "profile_sha256": sha256_file(dataset_dir / "profile.json"),
        "raw_image_count": expected_roles["original_hu_int16"],
        "mask_count": expected_roles["binary_roi"],
        "discrete_image_count": expected_roles["mask_specific_ibsi_fbn32"],
        "ivh_image_count": expected_roles.get("mask_specific_ibsi_fbs1_ivh_indices", 0),
        "case_cross_product_complete": True,
        "deep_fbn_cases_validated": checked_cases,
        "ready_for_adapter_input": True,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--shallow-validation", action="store_true")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    destination = Path(arguments.dest_dir)
    if not arguments.validate_only:
        build_pillar1_dataset(
            destination,
            profile_path=Path(arguments.profile),
            resume=arguments.resume,
        )
    summary = validate_pillar1_dataset(
        destination,
        deep=not arguments.shallow_validation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PROFILE",
    "PILLAR1_MASK_IDS",
    "build_pillar1_dataset",
    "load_pillar1_profile",
    "validate_pillar1_dataset",
]
