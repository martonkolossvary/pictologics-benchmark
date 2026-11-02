from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_MANIFEST_SCHEMA_VERSION = 1
DATASET_KINDS = frozenset({"synthetic", "real_world", "reference"})
MODALITIES = frozenset({"synthetic", "ct", "mri", "pet", "other"})


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot safely be used by the benchmark runner."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace *path* only after the complete payload is durable on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    atomic_write_bytes(path, (payload + "\n").encode("utf-8"))


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    row_list = [dict(row) for row in rows]
    if not row_list:
        atomic_write_bytes(path, b"")
        return

    fieldnames = list(row_list[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in row_list:
                normalized = {
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
                writer.writerow(normalized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_copy(source: Path, destination: Path, *, overwrite: bool) -> None:
    """Copy a dataset file without ever exposing a partial destination file."""

    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(
                f"destination differs from source: {destination}; pass --overwrite to replace it"
            )
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_nifti(path: Path, data: Any, affine: Any) -> None:
    """Durably replace a NIfTI-1 artifact with matched active transforms."""

    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise RuntimeError(
            "nibabel and numpy are required to write NIfTI data"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = np.asarray(affine, dtype=np.float64)
    image = nib.Nifti1Image(np.asarray(data), transform)
    image.header.set_xyzt_units("mm")
    image.set_qform(transform, code=1)
    image.set_sform(transform, code=1)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".nii.gz", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        nib.save(image, str(temporary))
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def contained_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DatasetValidationError(f"dataset path must be relative: {relative!r}")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DatasetValidationError(
            f"dataset path escapes its root: {relative!r}"
        ) from exc
    return candidate


def _validate_active_nifti_transforms(image: Any, description: str) -> None:
    """Reject ambiguous active qform/sform transforms.

    Backends do not all resolve conflicting active NIfTI transforms the same
    way. One inactive transform is unambiguous, but two active transforms must
    describe the same voxel-to-world geometry.
    """

    try:
        import numpy as np
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency error is installation-specific
        raise RuntimeError("numpy is required to inspect benchmark datasets") from exc

    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    active_qform = qform if int(qform_code) > 0 else None
    active_sform = sform if int(sform_code) > 0 else None
    for name, transform in (("qform", active_qform), ("sform", active_sform)):
        if transform is not None and not np.all(np.isfinite(transform)):
            raise DatasetValidationError(
                f"{description} active {name} contains non-finite values"
            )
    if (
        active_qform is not None
        and active_sform is not None
        and not np.allclose(active_qform, active_sform, rtol=1e-5, atol=1e-5)
    ):
        raise DatasetValidationError(f"{description} active qform/sform mismatch")


def nifti_case_metadata(
    image_path: Path,
    mask_path: Path,
    *,
    inspect_values: bool = True,
    inspect_image_values: bool = True,
) -> dict[str, Any]:
    """Validate a NIfTI pair and return geometry/ROI metadata for its manifest."""

    try:
        import nibabel as nib
        import numpy as np
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency error is installation-specific
        raise RuntimeError(
            "nibabel and numpy are required to inspect benchmark datasets"
        ) from exc

    try:
        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))
    except Exception as exc:
        raise DatasetValidationError(
            f"unable to load NIfTI pair {image_path.name}/{mask_path.name}: {exc}"
        ) from exc

    image_shape = tuple(int(value) for value in image.shape)
    mask_shape = tuple(int(value) for value in mask.shape)
    if len(image_shape) != 3 or len(mask_shape) != 3:
        raise DatasetValidationError(
            f"benchmark inputs must be 3D: image={image_shape}, mask={mask_shape}"
        )
    if image_shape != mask_shape:
        raise DatasetValidationError(
            f"image/mask shape mismatch: image={image_shape}, mask={mask_shape}"
        )
    _validate_active_nifti_transforms(image, "image")
    _validate_active_nifti_transforms(mask, "mask")
    if not np.all(np.isfinite(image.affine)) or not np.all(np.isfinite(mask.affine)):
        raise DatasetValidationError("image/mask selected affine must be finite")
    if not np.allclose(image.affine, mask.affine, rtol=1e-5, atol=1e-5):
        raise DatasetValidationError("image/mask affine mismatch")
    try:
        orientation = tuple(nib.aff2axcodes(image.affine))
    except Exception as exc:
        raise DatasetValidationError(
            "unable to derive image orientation from affine"
        ) from exc
    if any(value not in {"L", "R", "P", "A", "I", "S"} for value in orientation):
        raise DatasetValidationError(
            "unable to derive three valid image orientation axis codes"
        )

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise DatasetValidationError(f"invalid image spacing: {spacing}")
    mask_spacing = tuple(float(value) for value in mask.header.get_zooms()[:3])
    if any(not math.isfinite(value) or value <= 0 for value in mask_spacing):
        raise DatasetValidationError(f"invalid mask spacing: {mask_spacing}")
    if not np.allclose(spacing, mask_spacing, rtol=1e-5, atol=1e-5):
        raise DatasetValidationError(
            f"image/mask header spacing mismatch: image={spacing}, mask={mask_spacing}"
        )

    image_voxels = int(np.prod(image_shape, dtype=np.int64))
    mask_voxels = None
    mask_fraction = None
    if inspect_values:
        mask_data = np.asanyarray(mask.dataobj)
        if not np.all(np.isfinite(mask_data)):
            raise DatasetValidationError("mask contains non-finite values")
        if not np.all((mask_data == 0) | (mask_data == 1)):
            raise DatasetValidationError(
                "benchmark masks must be canonical binary {0, 1}; "
                "positive-label and multi-label masks must be normalized during preparation"
            )
        roi = mask_data == 1
        mask_voxels = int(np.count_nonzero(roi))
        if mask_voxels == 0:
            raise DatasetValidationError("mask has no positive voxels")

        if inspect_image_values:
            image_data = np.asanyarray(image.dataobj)
            if not np.all(np.isfinite(image_data)):
                raise DatasetValidationError(
                    "image contains non-finite values; full-volume finiteness is required "
                    "because local-intensity calculations inspect ROI neighbourhoods"
                )
        mask_fraction = float(mask_voxels) / float(image_voxels)

    return {
        "shape": list(image_shape),
        "spacing": list(spacing),
        "orientation": list(orientation),
        "affine": [[float(value) for value in row] for row in image.affine.tolist()],
        "image_voxels": image_voxels,
        "mask_voxels": mask_voxels,
        "mask_fraction": mask_fraction,
        # The --sizes filter selects on this edge length; complexity is image_voxels.
        "size": max(image_shape),
        "complexity": image_voxels,
    }


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{context} must be an object")
    return value


def _require_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise DatasetValidationError(f"{context} must be a list")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_bound_json_record(
    dataset_dir: Path,
    provenance: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    description: str,
    verify_hashes: bool,
) -> Mapping[str, Any] | None:
    """Validate a provenance JSON record whenever a manifest names one.

    A hand-authored manifest may omit provenance records. The dataset generators
    name their record, so they must also bind the exact bytes into the manifest
    with SHA-256.
    """

    raw_relative = provenance.get(path_key)
    if raw_relative is None:
        if hash_key in provenance:
            raise DatasetValidationError(
                f"{description} SHA-256 is present without {path_key}"
            )
        return None

    relative = str(raw_relative).strip()
    if not relative:
        raise DatasetValidationError(f"{description} path must not be empty")
    expected_hash = provenance.get(hash_key)
    if not _is_sha256(expected_hash):
        raise DatasetValidationError(
            f"manifest provenance requires a valid {hash_key} for {description}"
        )
    path = contained_path(dataset_dir, relative)
    if not path.is_file():
        raise DatasetValidationError(f"{description} is missing: {relative}")
    if verify_hashes and sha256_file(path).lower() != str(expected_hash).lower():
        raise DatasetValidationError(f"{description} checksum mismatch: {relative}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"invalid {description}: {relative}") from exc
    return _require_mapping(record, description)


def _validate_provenance(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    *,
    dataset_id: str,
    dataset_kind: str,
    verify_hashes: bool,
) -> None:
    raw_provenance = manifest.get("provenance")
    if raw_provenance is None:
        return
    provenance = _require_mapping(raw_provenance, "manifest provenance")

    generation = _validate_bound_json_record(
        dataset_dir,
        provenance,
        path_key="generation_record",
        hash_key="generation_record_sha256",
        description="generation record",
        verify_hashes=verify_hashes,
    )
    preparation = _validate_bound_json_record(
        dataset_dir,
        provenance,
        path_key="preparation_record",
        hash_key="preparation_record_sha256",
        description="preparation record",
        verify_hashes=verify_hashes,
    )
    if generation is not None and preparation is not None:
        raise DatasetValidationError(
            "manifest provenance cannot name both generation and preparation records"
        )

    if generation is not None:
        if dataset_kind != "synthetic":
            raise DatasetValidationError(
                "generation_record is only valid for synthetic datasets"
            )
        if generation.get("schema_version") != 1:
            raise DatasetValidationError("unsupported generation record schema")
        parameters = _require_mapping(
            generation.get("parameters"), "generation record parameters"
        )
        parameters_hash = generation.get("parameters_sha256")
        if not _is_sha256(parameters_hash) or (
            _canonical_json_sha256(parameters).lower() != str(parameters_hash).lower()
        ):
            raise DatasetValidationError(
                "generation record parameters checksum mismatch"
            )
        if provenance.get("parameters_sha256") != parameters_hash:
            raise DatasetValidationError(
                "manifest/generation record parameters checksum mismatch"
            )
        if parameters.get("dataset") != dataset_id:
            raise DatasetValidationError("manifest/generation record dataset mismatch")
        if parameters.get("dataset_kind") != dataset_kind:
            raise DatasetValidationError(
                "manifest/generation record dataset_kind mismatch"
            )
        if generation.get("status") != "complete":
            raise DatasetValidationError("generation record is not complete")
        manifest_files = _require_sequence(manifest.get("files"), "manifest files")
        manifest_cases = _require_sequence(manifest.get("cases"), "manifest cases")
        if generation.get("artifact_count") != len(manifest_files):
            raise DatasetValidationError(
                "manifest/generation record artifact count mismatch"
            )
        if generation.get("case_count") != len(manifest_cases):
            raise DatasetValidationError(
                "manifest/generation record case count mismatch"
            )
        for key in ("seed", "sizes", "variants", "spacing"):
            if key in manifest and manifest.get(key) != parameters.get(key):
                raise DatasetValidationError(
                    f"manifest/generation record {key} mismatch"
                )

    if preparation is not None:
        if dataset_kind != "real_world":
            raise DatasetValidationError(
                "preparation_record is only valid for real-world datasets"
            )
        if preparation.get("schema_version") != 1:
            raise DatasetValidationError("unsupported preparation record schema")
        parameters = _require_mapping(
            preparation.get("parameters"), "preparation record parameters"
        )
        if preparation.get("status") != "complete":
            raise DatasetValidationError("preparation record is not complete")
        if parameters.get("dataset") != dataset_id:
            raise DatasetValidationError("manifest/preparation record dataset mismatch")
        if "requested_subjects" in manifest and (
            manifest.get("requested_subjects") != parameters.get("num_subjects")
        ):
            raise DatasetValidationError(
                "manifest/preparation record requested subject count mismatch"
            )
        if "modalities" in manifest and (
            manifest.get("modalities") != parameters.get("modalities")
        ):
            raise DatasetValidationError(
                "manifest/preparation record modalities mismatch"
            )
        prepared_files = _require_sequence(
            preparation.get("prepared_files"), "preparation record prepared_files"
        )
        manifest_files = _require_sequence(manifest.get("files"), "manifest files")
        prepared_core: list[tuple[Any, Any, Any, Any]] = []
        for index, raw_entry in enumerate(prepared_files):
            entry = _require_mapping(
                raw_entry, f"preparation record prepared_files[{index}]"
            )
            prepared_path = entry.get("prepared_path")
            kind = entry.get("kind")
            sha256 = entry.get("sha256")
            byte_count = entry.get("bytes")
            if (
                not isinstance(prepared_path, str)
                or not prepared_path
                or kind not in {"image", "mask"}
                or not _is_sha256(sha256)
                or not isinstance(byte_count, int)
                or byte_count < 1
            ):
                raise DatasetValidationError(
                    f"invalid preparation record prepared_files[{index}]"
                )
            prepared_core.append((prepared_path, kind, str(sha256).lower(), byte_count))
        manifest_core: list[tuple[Any, Any, Any, Any]] = []
        for index, raw_entry in enumerate(manifest_files):
            entry = _require_mapping(raw_entry, f"files[{index}]")
            relative = entry.get("path")
            kind = entry.get("kind")
            sha256 = entry.get("sha256")
            byte_count = entry.get("bytes")
            if (
                not isinstance(relative, str)
                or not relative
                or kind not in {"image", "mask"}
                or not _is_sha256(sha256)
                or not isinstance(byte_count, int)
                or byte_count < 1
            ):
                raise DatasetValidationError(f"invalid manifest files[{index}]")
            manifest_core.append((relative, kind, str(sha256).lower(), byte_count))
        if sorted(prepared_core) != sorted(manifest_core):
            raise DatasetValidationError(
                "manifest/preparation record prepared file inventory mismatch"
            )


def validate_manifest(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    *,
    verify_hashes: bool = True,
    inspect_geometry: bool = True,
    inspect_values: bool = True,
) -> dict[str, Any]:
    """Validate the portable dataset contract used by every benchmark mode."""

    dataset_dir = dataset_dir.resolve()
    schema_version = manifest.get("schema_version")
    if schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
        raise DatasetValidationError(
            f"unsupported dataset manifest schema {schema_version!r}; "
            f"expected {DATASET_MANIFEST_SCHEMA_VERSION}"
        )
    dataset_id = str(manifest.get("dataset") or "").strip()
    if not dataset_id:
        raise DatasetValidationError("manifest requires a non-empty dataset identifier")
    dataset_kind = str(manifest.get("dataset_kind") or "").strip().lower()
    if dataset_kind not in DATASET_KINDS:
        raise DatasetValidationError(
            f"dataset_kind must be one of: {', '.join(sorted(DATASET_KINDS))}"
        )

    _validate_provenance(
        dataset_dir,
        manifest,
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        verify_hashes=verify_hashes,
    )

    files = _require_sequence(manifest.get("files"), "manifest files")
    cases = _require_sequence(manifest.get("cases"), "manifest cases")
    if not cases:
        raise DatasetValidationError("manifest cases must not be empty")

    files_by_path: dict[str, Mapping[str, Any]] = {}
    for index, raw_file in enumerate(files):
        entry = _require_mapping(raw_file, f"files[{index}]")
        relative = str(entry.get("path") or "")
        if relative in files_by_path:
            raise DatasetValidationError(f"duplicate file manifest entry: {relative}")
        kind = str(entry.get("kind") or "")
        if kind not in {"image", "mask"}:
            raise DatasetValidationError(
                f"invalid dataset file kind for {relative}: {kind!r}"
            )
        files_by_path[relative] = entry
        path = contained_path(dataset_dir, relative)
        if not path.is_file():
            raise DatasetValidationError(f"dataset file is missing: {relative}")
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes != path.stat().st_size:
            raise DatasetValidationError(f"dataset file size mismatch: {relative}")
        expected_hash = entry.get("sha256")
        if not _is_sha256(expected_hash):
            raise DatasetValidationError(f"invalid SHA-256 in manifest: {relative}")
        if verify_hashes and sha256_file(path).lower() != str(expected_hash).lower():
            raise DatasetValidationError(f"dataset checksum mismatch: {relative}")

    seen_cases: set[str] = set()
    inspected_image_values: set[Path] = set()
    total_voxels = 0
    for index, raw_case in enumerate(cases):
        case = _require_mapping(raw_case, f"cases[{index}]")
        case_id = str(case.get("case_id") or "").strip()
        if not case_id:
            raise DatasetValidationError(f"cases[{index}] requires case_id")
        if case_id in seen_cases:
            raise DatasetValidationError(f"duplicate case_id: {case_id}")
        seen_cases.add(case_id)

        modality = str(case.get("modality") or "").strip().lower()
        if modality not in MODALITIES:
            raise DatasetValidationError(
                f"case {case_id} has unsupported modality {modality!r}"
            )
        if dataset_kind == "synthetic" and modality != "synthetic":
            raise DatasetValidationError(
                f"synthetic case {case_id} must declare modality 'synthetic'"
            )
        if dataset_kind == "real_world" and modality == "synthetic":
            raise DatasetValidationError(
                f"real-world case {case_id} cannot declare modality 'synthetic'"
            )
        image_relative = str(case.get("image_path") or "")
        mask_relative = str(case.get("mask_path") or "")
        if image_relative not in files_by_path or mask_relative not in files_by_path:
            raise DatasetValidationError(
                f"case {case_id} references a file absent from manifest files"
            )
        if files_by_path[image_relative].get("kind") != "image":
            raise DatasetValidationError(
                f"case {case_id} image_path is not an image entry"
            )
        if files_by_path[mask_relative].get("kind") != "mask":
            raise DatasetValidationError(
                f"case {case_id} mask_path is not a mask entry"
            )
        if case.get("image_sha256") != files_by_path[image_relative].get("sha256"):
            raise DatasetValidationError(
                f"case {case_id} image checksum metadata mismatch"
            )
        if case.get("mask_sha256") != files_by_path[mask_relative].get("sha256"):
            raise DatasetValidationError(
                f"case {case_id} mask checksum metadata mismatch"
            )
        image_path = contained_path(dataset_dir, image_relative)
        mask_path = contained_path(dataset_dir, mask_relative)

        shape = case.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise DatasetValidationError(f"case {case_id} requires a positive 3D shape")
        declared_voxels = case.get("image_voxels")
        declared_complexity = case.get("complexity", declared_voxels)
        if not isinstance(declared_voxels, int) or declared_voxels <= 0:
            raise DatasetValidationError(f"case {case_id} has invalid image_voxels")
        if math.prod(shape) != declared_voxels:
            raise DatasetValidationError(f"case {case_id} shape/image_voxels mismatch")
        if dataset_kind == "synthetic":
            declared_size = case.get("size")
            if not isinstance(declared_size, int) or declared_size <= 0:
                raise DatasetValidationError(
                    f"synthetic case {case_id} requires a positive integer size"
                )
            if shape != [declared_size, declared_size, declared_size]:
                raise DatasetValidationError(
                    f"synthetic case {case_id} size must equal every cubic image edge"
                )
        if declared_complexity != declared_voxels:
            raise DatasetValidationError(
                f"case {case_id} complexity must equal actual image_voxels"
            )
        total_voxels += declared_voxels

        mask_voxels = case.get("mask_voxels")
        if not isinstance(mask_voxels, int) or not 0 < mask_voxels <= declared_voxels:
            raise DatasetValidationError(f"case {case_id} has invalid mask_voxels")
        mask_fraction = case.get("mask_fraction")
        expected_fraction = float(mask_voxels) / float(declared_voxels)
        if (
            not isinstance(mask_fraction, (int, float))
            or not math.isfinite(float(mask_fraction))
            or not math.isclose(
                float(mask_fraction), expected_fraction, rel_tol=1e-10, abs_tol=1e-12
            )
        ):
            raise DatasetValidationError(f"case {case_id} has invalid mask_fraction")
        declared_spacing = case.get("spacing")
        if (
            not isinstance(declared_spacing, list)
            or len(declared_spacing) != 3
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in declared_spacing
            )
        ):
            raise DatasetValidationError(
                f"case {case_id} requires positive finite spacing"
            )
        declared_orientation = case.get("orientation")
        if (
            not isinstance(declared_orientation, list)
            or len(declared_orientation) != 3
            or any(
                not isinstance(value, str)
                or value not in {"L", "R", "P", "A", "I", "S"}
                for value in declared_orientation
            )
        ):
            raise DatasetValidationError(
                f"case {case_id} requires three valid orientation axis codes"
            )
        declared_affine = case.get("affine")
        if (
            not isinstance(declared_affine, list)
            or len(declared_affine) != 4
            or any(
                not isinstance(row, list) or len(row) != 4 for row in declared_affine
            )
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for row in declared_affine
                for value in row
            )
        ):
            raise DatasetValidationError(f"case {case_id} requires a finite 4x4 affine")

        if inspect_geometry:
            observed = nifti_case_metadata(
                image_path,
                mask_path,
                inspect_values=inspect_values,
                inspect_image_values=(
                    inspect_values and image_path not in inspected_image_values
                ),
            )
            if inspect_values:
                inspected_image_values.add(image_path)
            observed_keys = ["shape", "image_voxels"]
            if inspect_values:
                observed_keys.append("mask_voxels")
            for key in observed_keys:
                declared = case.get(key)
                if declared != observed[key]:
                    raise DatasetValidationError(
                        f"case {case_id} {key} mismatch: declared={declared!r}, "
                        f"observed={observed[key]!r}"
                    )
            if declared_orientation != observed["orientation"]:
                raise DatasetValidationError(f"case {case_id} orientation mismatch")
            if any(
                not math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-5)
                for left, right in zip(declared_spacing, observed["spacing"])
            ):
                raise DatasetValidationError(f"case {case_id} spacing mismatch")
            if any(
                not math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-7)
                for declared_row, observed_row in zip(
                    declared_affine, observed["affine"]
                )
                for left, right in zip(declared_row, observed_row)
            ):
                raise DatasetValidationError(f"case {case_id} affine mismatch")

        image_entry = files_by_path[image_relative]
        mask_entry = files_by_path[mask_relative]
        for entry_description, entry in (
            ("image", image_entry),
            ("mask", mask_entry),
        ):
            if "shape" in entry and entry.get("shape") != shape:
                raise DatasetValidationError(
                    f"case {case_id} {entry_description} file shape metadata mismatch"
                )
            if "modality" in entry and (
                str(entry.get("modality") or "").strip().lower() != modality
            ):
                raise DatasetValidationError(
                    f"case {case_id} {entry_description} file modality metadata mismatch"
                )
            if (
                dataset_kind == "synthetic"
                and "size" in entry
                and (entry.get("size") != case.get("size"))
            ):
                raise DatasetValidationError(
                    f"case {case_id} {entry_description} file size metadata mismatch"
                )

    if dataset_kind == "synthetic" and "sizes" in manifest:
        declared_sizes = manifest.get("sizes")
        if (
            not isinstance(declared_sizes, list)
            or not declared_sizes
            or any(not isinstance(value, int) or value <= 0 for value in declared_sizes)
            or len(set(declared_sizes)) != len(declared_sizes)
        ):
            raise DatasetValidationError(
                "synthetic manifest sizes must be unique positive integers"
            )
        observed_sizes = {int(case["size"]) for case in cases}
        if set(declared_sizes) != observed_sizes:
            raise DatasetValidationError(
                "synthetic manifest sizes do not match case sizes"
            )

    return {
        "schema_version": schema_version,
        "dataset": dataset_id,
        "dataset_kind": dataset_kind,
        "case_count": len(cases),
        "file_count": len(files),
        "total_image_voxels": total_voxels,
        "hashes_verified": bool(verify_hashes),
        "geometry_inspected": bool(inspect_geometry),
        "voxel_values_inspected": bool(inspect_geometry and inspect_values),
    }


def load_and_validate_manifest(
    dataset_dir: Path,
    *,
    verify_hashes: bool = True,
    inspect_geometry: bool = True,
    inspect_values: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = dataset_dir.resolve() / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except FileNotFoundError:
        raise DatasetValidationError(f"dataset manifest not found: {manifest_path}")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(
            f"dataset manifest is not valid UTF-8: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"invalid dataset manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetValidationError("dataset manifest must be an object")
    summary = validate_manifest(
        dataset_dir,
        manifest,
        verify_hashes=verify_hashes,
        inspect_geometry=inspect_geometry,
        inspect_values=inspect_values,
    )
    summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest, summary
