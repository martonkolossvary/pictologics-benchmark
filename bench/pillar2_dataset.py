"""Generate and validate the dense whole-anatomy (A1) scaling pillar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np

from bench.dataset_manifest import (
    DatasetValidationError,
    atomic_copy,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_nifti,
    load_and_validate_manifest,
    nifti_case_metadata,
    sha256_file,
)
from bench.pillar1_dataset import _canonical_sha256, _measure_mask
from bench.synthetic_volume import VolumeConfig
from bench.synthetic_representations import (
    compile_mask_specific_fbn,
    compile_mask_specific_fbs,
    texture_workload,
)
from bench.synthetic_volume import generate_synthetic_volume


DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1] / "configs" / "benchmark" / "pillar2.json"
)


def _load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("mask_id") != "A1":
        raise DatasetValidationError("unsupported Pillar 2 A1 profile")
    sizes = profile.get("sizes")
    if not isinstance(sizes, list) or len(sizes) != len(set(sizes)):
        raise DatasetValidationError("Pillar 2 sizes must be a unique list")
    if profile.get("case_count") != len(sizes) or profile.get(
        "artifact_count"
    ) != 4 * len(sizes):
        raise DatasetValidationError("Pillar 2 profile inventory is inconsistent")
    return profile


def _write_artifact(
    root: Path,
    relative: str,
    array: np.ndarray,
    affine: np.ndarray,
    *,
    role: str,
    kind: str,
    size: int,
) -> dict[str, Any]:
    path = root / relative
    atomic_write_nifti(path, array, affine)
    return {
        "path": relative,
        "kind": kind,
        "role": role,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": np.dtype(array.dtype).name,
        "shape": [int(value) for value in array.shape],
        "size": int(size),
        "modality": "synthetic",
    }


def build_pillar2_dataset(
    destination: Path,
    *,
    profile_path: Path = DEFAULT_PROFILE,
    resume: bool = True,
) -> dict[str, Any]:
    destination = destination.expanduser().resolve()
    profile_path = profile_path.expanduser().resolve()
    profile = _load_profile(profile_path)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_copy(profile_path, destination / "profile.json", overwrite=False)
    # The source hash makes a resume fail when the generator code changed.
    source_files = [
        {"path": f"bench/{name}", "sha256": sha256_file(Path(__file__).parent / name)}
        for name in (
            "pillar2_dataset.py",
            "pillar1_dataset.py",
            "synthetic_representations.py",
            "synthetic_volume.py",
        )
    ]
    source_tree_sha256 = _canonical_sha256(source_files)
    parameters = {
        "dataset": profile["dataset"],
        "dataset_kind": "synthetic",
        "profile_sha256": sha256_file(profile_path),
        "source_tree_sha256": source_tree_sha256,
        "sizes": list(profile["sizes"]),
        "variants": [str(profile["hu_profile"])],
        "spacing": "derived_from_fixed_256mm_fov",
    }
    state_path = destination / "generation_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("parameters") != parameters:
            raise DatasetValidationError(
                "existing Pillar 2 generation uses different parameters"
            )
        if not resume:
            raise FileExistsError(
                f"Pillar 2 destination already has state: {destination}"
            )
    else:
        state = {
            "schema_version": 1,
            "status": "in_progress",
            "parameters": parameters,
            "artifacts": {},
            "cases": {},
        }
        atomic_write_json(state_path, state)

    artifacts_by_path = dict(state.get("artifacts") or {})
    cases_by_id = dict(state.get("cases") or {})
    completed_sizes = {
        int(case["size"])
        for case in cases_by_id.values()
        if isinstance(case, dict) and case.get("size") is not None
    }
    for relative, record in artifacts_by_path.items():
        artifact_path = destination / relative
        if not artifact_path.is_file() or sha256_file(artifact_path) != record.get(
            "sha256"
        ):
            raise DatasetValidationError(
                f"recorded Pillar 2 artifact is missing or changed: {relative}"
            )

    for size in profile["sizes"]:
        if int(size) in completed_sizes:
            continue
        bundle = generate_synthetic_volume(
            VolumeConfig(
                edge=int(size),
                fov_mm=float(profile["fov_mm"]),
                seed=int(profile["seed"]),
                slab_depth=int(profile["slab_depth"]),
                hu_profile=str(profile["hu_profile"]),
            )
        )
        raw = np.asarray(bundle.image, dtype=np.int16)
        mask = np.asarray(bundle.masks["A1"], dtype=np.uint8)
        fbn = compile_mask_specific_fbn(
            raw, mask, levels=int(profile["texture_levels"])
        )
        fbs = compile_mask_specific_fbs(
            raw, mask, bin_width=float(profile["ivh_bin_width"])
        )
        paths = {
            "raw": f"raw/n{size:03d}_image.nii.gz",
            "mask": f"masks/n{size:03d}_a1_mask.nii.gz",
            "fbn": f"discrete/fbn32/n{size:03d}_a1_fbn32.nii.gz",
            "ivh": f"discrete/ivh_fbs1/n{size:03d}_a1_ivh_fbs1.nii.gz",
        }
        records = {
            "raw": _write_artifact(
                destination,
                paths["raw"],
                raw,
                bundle.affine,
                role="original_hu_int16",
                kind="image",
                size=int(size),
            ),
            "mask": _write_artifact(
                destination,
                paths["mask"],
                mask,
                bundle.affine,
                role="binary_whole_anatomy_roi",
                kind="mask",
                size=int(size),
            ),
            "fbn": _write_artifact(
                destination,
                paths["fbn"],
                fbn.array,
                bundle.affine,
                role="mask_specific_ibsi_fbn32",
                kind="image",
                size=int(size),
            ),
            "ivh": _write_artifact(
                destination,
                paths["ivh"],
                fbs.array,
                bundle.affine,
                role="mask_specific_ibsi_fbs1_ivh_indices",
                kind="image",
                size=int(size),
            ),
        }
        artifacts_by_path.update(
            {record["path"]: record for record in records.values()}
        )
        metric = _measure_mask(
            mask,
            body_voxels=int(np.count_nonzero(mask)),
            spacing_mm=float(bundle.spacing_mm[0]),
        )
        derivation_base = {
            "image_sha256": records["raw"]["sha256"],
            "mask_sha256": records["mask"]["sha256"],
        }
        metadata = nifti_case_metadata(
            destination / paths["raw"], destination / paths["mask"]
        )
        case = {
            "case_id": f"p2_a1_n{size:03d}",
            "subject_id": "reference",
            "modality": "synthetic",
            "size": int(size),
            "variant": 1,
            "mask_id": "A1",
            "mask_label": "dense whole anatomy including bone and pathology",
            "image_path": paths["raw"],
            "image_sha256": records["raw"]["sha256"],
            "mask_path": paths["mask"],
            "mask_sha256": records["mask"]["sha256"],
            "discrete_image_path": paths["fbn"],
            "discrete_image_sha256": records["fbn"]["sha256"],
            "ivh_image_path": paths["ivh"],
            "ivh_image_sha256": records["ivh"]["sha256"],
            "raw_representation": "original_hu_int16",
            "texture_workload": texture_workload(fbn.array, mask),
            "texture_representation": {
                "id": "mask_specific_ibsi_fbn32",
                "configured_levels": fbn.configured_levels,
                "occupied_levels": fbn.occupied_levels,
                "roi_min_hu": fbn.roi_min,
                "roi_max_hu": fbn.roi_max,
                "background_value": 0,
                "derivation_sha256": _canonical_sha256(
                    {
                        **derivation_base,
                        "method": "IBSI_fixed_bin_number",
                        "levels": 32,
                    }
                ),
            },
            "ivh_representation": {
                "id": "mask_specific_ibsi_fbs1_ivh_indices",
                "configured_levels": fbs.configured_levels,
                "occupied_levels": fbs.occupied_levels,
                "bin_width": fbs.bin_width,
                "anchor_hu": fbs.anchor,
                "roi_min_hu": fbs.roi_min,
                "roi_max_hu": fbs.roi_max,
                "background_value": 0,
                "derivation_sha256": _canonical_sha256(
                    {
                        **derivation_base,
                        "method": "IBSI_fixed_bin_size",
                        "bin_width": 1.0,
                        "anchor_hu": fbs.anchor,
                    }
                ),
            },
            "mask_body_fraction": 1.0,
            "bbox_shape_xyz": list(metric["bbox_shape_xyz"]),
            "occupied_z_slices": metric["occupied_z_slices"],
            "occupied_z_mm": metric["occupied_z_mm"],
            "components_26": metric["components_26"],
            "cavities_background_6": metric["cavities_background_6"],
            "complexity": int(size) ** 3,
            **metadata,
        }
        cases_by_id[case["case_id"]] = case
        state["artifacts"] = artifacts_by_path
        state["cases"] = cases_by_id
        atomic_write_json(state_path, state)

    artifacts = sorted(artifacts_by_path.values(), key=lambda value: value["path"])
    cases = sorted(cases_by_id.values(), key=lambda value: value["case_id"])
    if len(cases) != int(profile["case_count"]) or len(artifacts) != int(
        profile["artifact_count"]
    ):
        raise DatasetValidationError("Pillar 2 generated inventory is incomplete")
    generation = {
        "schema_version": 1,
        "status": "complete",
        "parameters": parameters,
        "parameters_sha256": _canonical_sha256(parameters),
        "source_files": source_files,
        "artifact_count": len(artifacts),
        "case_count": len(cases),
    }
    atomic_write_json(destination / "generation.json", generation)
    manifest = {
        "schema_version": 1,
        "dataset": profile["dataset"],
        "dataset_kind": "synthetic",
        "description": profile["description"],
        "coordinate_system": "RAS+",
        "axis_order": "xyz",
        "sizes": list(profile["sizes"]),
        "variants": [str(profile["hu_profile"])],
        "mask_ids": ["A1"],
        "provenance": {
            "generation_record": "generation.json",
            "generation_record_sha256": sha256_file(destination / "generation.json"),
            "generation_state": "generation_state.json",
            "parameters_sha256": generation["parameters_sha256"],
        },
        "files": artifacts,
        "cases": cases,
    }
    atomic_write_json(destination / "manifest.json", manifest)
    atomic_write_csv(
        destination / "cases.csv",
        [
            {
                "case_id": case["case_id"],
                "shape_xyz": "x".join(str(value) for value in case["shape"]),
                "spacing_mm": case["spacing"][0],
                "image_voxels": case["image_voxels"],
                "mask_voxels": case["mask_voxels"],
                "mask_fraction_percent": 100.0 * case["mask_fraction"],
                "occupied_z_slices": case["occupied_z_slices"],
            }
            for case in cases
        ],
    )
    state["status"] = "complete"
    state["manifest_sha256"] = sha256_file(destination / "manifest.json")
    atomic_write_json(state_path, state)
    return manifest


def validate_pillar2_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest, summary = load_and_validate_manifest(
        dataset_dir, verify_hashes=True, inspect_geometry=True, inspect_values=True
    )
    if manifest.get("dataset") != "pictologics_pillar2_a1":
        raise DatasetValidationError("unexpected Pillar 2 dataset")
    for case in manifest["cases"]:
        raw = np.asanyarray(nib.load(str(dataset_dir / case["image_path"])).dataobj)
        mask = np.asanyarray(nib.load(str(dataset_dir / case["mask_path"])).dataobj)
        fbn = np.asanyarray(
            nib.load(str(dataset_dir / case["discrete_image_path"])).dataobj
        )
        ivh = np.asanyarray(nib.load(str(dataset_dir / case["ivh_image_path"])).dataobj)
        if not np.array_equal(
            fbn, compile_mask_specific_fbn(raw, mask, levels=32).array
        ):
            raise DatasetValidationError(f"Pillar 2 FBN mismatch: {case['case_id']}")
        if not np.array_equal(
            ivh, compile_mask_specific_fbs(raw, mask, bin_width=1.0).array
        ):
            raise DatasetValidationError(f"Pillar 2 IVH mismatch: {case['case_id']}")
        coordinates = np.argwhere(mask != 0)
        margins = np.concatenate(
            (coordinates.min(0), np.asarray(mask.shape) - 1 - coordinates.max(0))
        )
        if int(np.min(margins)) < 1:
            raise DatasetValidationError(
                f"Pillar 2 face guard failed: {case['case_id']}"
            )
    return {
        **summary,
        "a1_cases_validated": len(manifest["cases"]),
        "ready_for_adapter_input": True,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-dir", required=True)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    destination = Path(args.dest_dir)
    if not args.validate_only:
        build_pillar2_dataset(
            destination,
            profile_path=Path(args.profile),
            resume=args.resume,
        )
    print(json.dumps(validate_pillar2_dataset(destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
