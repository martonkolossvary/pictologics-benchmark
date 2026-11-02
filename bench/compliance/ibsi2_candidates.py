"""Generate provenance-complete IBSI 2 response-map candidate bundles.

This module starts from the pinned official IBSI NIfTI files.  Phase 1 sends
each unmodified digital phantom to an installed package-native filter wrapper.
Phase 2 first creates one reviewed A/B preprocessing pair from PAT1, then sends
the identical preprocessed image to every package-native filter wrapper.  This
isolates native filter and first-order feature implementations while keeping
the preprocessing input controlled and explicitly documented.

Generation is resumable.  Every completed or reviewed-unsupported adapter/test
pair is committed to ``generation_state.json`` before the next task starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bench.benchmark_ledger import (
    RunIntegrityError,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from bench.benchmark_models import fingerprint
from bench.compliance.ibsi2_protocol import (
    IBSI2_PROTOCOL_REVIEW,
    PHASE1_FILTER_SPECS,
    PHASE2_FILTER_SPECS,
)
from bench.compliance.references import (
    IBSI_DATA_COMMIT,
    IBSI_DATA_REPOSITORY,
    IBSI2_PHASE1_SOURCE_MASK_SHA256,
    IBSI2_PHASE2_SOURCE_IMAGE_SHA256,
    IBSI2_PHASE2_SOURCE_MASK_SHA256,
)


GENERATION_SCHEMA_VERSION = 1
DEFAULT_ADAPTERS = ("pictologics", "pyradiomics", "mirp", "medimage", "zrad")
REVIEWED_BY = "Codex-assisted Pictologics manuscript protocol audit"
REVIEWED_AT = "2026-07-24"
PHASE2_EXPECTED_B_SHAPE = (200, 197, 180)
PHASE2_EXPECTED_B_MASK_VOXELS = 357_802


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _portable_command(command: str, *, root: Path, bundle: Path) -> str:
    """Replace machine-local roots in a recorded command with stable variables."""

    portable = command.replace(str(bundle.resolve()), "${IBSI2_CANDIDATE_BUNDLE}")
    return portable.replace(str(root.resolve()), "${PICTOLOGICS_PROJECT_ROOT}")


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if sha256_file(source) != expected_sha256:
        raise RunIntegrityError(f"Pinned source checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected_sha256:
        raise RunIntegrityError(f"Copied source checksum mismatch: {destination}")


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256_file(target) == sha256_file(path):
            continue
        shutil.copy2(path, target)


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _adapter_environment(adapter: str, root: Path) -> tuple[Path, dict[str, Any]]:
    environment = root / ".venvs" / "adapters" / adapter / "environment.json"
    if not environment.is_file():
        raise FileNotFoundError(environment)
    metadata = _json_object(environment)
    if metadata.get("profile") != adapter:
        raise RunIntegrityError(f"Adapter environment identity mismatch: {adapter}")
    python = (
        root
        / ".venvs"
        / "adapters"
        / adapter
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if not python.is_file():
        raise FileNotFoundError(python)
    return python, metadata


def _save_pictologics_image(image: Any, path: Path) -> None:
    # Pictologics 1.0 keeps its geometry in LPS+; its own writer turns it back
    # into the RAS+ affine of NIfTI.
    from pictologics import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, path)


def preprocess_phase2(
    *,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the controlled IBSI 2 A/B inputs using reviewed protocol steps."""

    import nibabel as nib
    import numpy as np
    from pictologics.loader import Image, load_image
    from pictologics.pipeline import PipelineState, RadiomicsPipeline, SourceMode

    image_path = Path(image_path).resolve()
    mask_path = Path(mask_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_state() -> tuple[Any, Any]:
        image = load_image(str(image_path))
        raw_mask = load_image(str(mask_path))
        intensity_mask = Image(
            array=raw_mask.array.copy(),
            spacing=raw_mask.spacing,
            origin=raw_mask.origin,
            direction=raw_mask.direction,
            modality=raw_mask.modality,
        )
        state = PipelineState(
            image=image,
            raw_image=image,
            morph_mask=raw_mask,
            intensity_mask=intensity_mask,
            source_mode=SourceMode.FULL_IMAGE,
        )
        return state, RadiomicsPipeline(load_standard=False)

    outputs: dict[str, Any] = {}
    steps_by_dimension = {
        "A": [
            ("resegment", {"range_min": -1000, "range_max": 400}),
        ],
        "B": [
            (
                "resample",
                {
                    "new_spacing": (1.0, 1.0, 1.0),
                    "interpolation": "cubic",
                    "mask_interpolation": "linear",
                    "mask_threshold": 0.5,
                },
            ),
            ("round_intensities", {}),
            ("resegment", {"range_min": -1000, "range_max": 400}),
        ],
    }
    for dimension, steps in steps_by_dimension.items():
        state, pipeline = make_state()
        for step_name, parameters in steps:
            pipeline._execute_preprocessing_step(state, step_name, parameters)
        output_image = output_dir / f"{dimension}-image.nii.gz"
        output_mask = output_dir / f"{dimension}-mask.nii.gz"
        _save_pictologics_image(state.image, output_image)
        _save_pictologics_image(state.intensity_mask, output_mask)

        image_nifti = nib.load(str(output_image))
        mask_nifti = nib.load(str(output_mask))
        image_data = np.asarray(image_nifti.dataobj)
        mask_data = np.asarray(mask_nifti.dataobj)
        if image_data.shape != mask_data.shape:
            raise RuntimeError(f"Phase 2 {dimension} image/mask shape mismatch")
        if not np.allclose(image_nifti.affine, mask_nifti.affine, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"Phase 2 {dimension} image/mask affine mismatch")
        if not np.isfinite(image_data).all() or not np.isfinite(mask_data).all():
            raise RuntimeError(
                f"Phase 2 {dimension} preprocessing produced non-finite data"
            )
        binary_mask = mask_data > 0
        if not np.array_equal(mask_data, binary_mask.astype(mask_data.dtype)):
            raise RuntimeError(f"Phase 2 {dimension} preprocessing mask is not binary")
        if not binary_mask.any():
            raise RuntimeError(f"Phase 2 {dimension} preprocessing mask is empty")
        outputs[dimension] = {
            "image_path": output_image.name,
            "mask_path": output_mask.name,
            "image_sha256": sha256_file(output_image),
            "mask_sha256": sha256_file(output_mask),
            "shape": list(image_data.shape),
            "mask_voxels": int(np.count_nonzero(binary_mask)),
            "spacing_mm": [
                float(value) for value in image_nifti.header.get_zooms()[:3]
            ],
        }

    if tuple(outputs["B"]["shape"]) != PHASE2_EXPECTED_B_SHAPE:
        raise RuntimeError(
            "IBSI 2 B preprocessing geometry regression: "
            f"{outputs['B']['shape']} != {list(PHASE2_EXPECTED_B_SHAPE)}"
        )
    if outputs["B"]["mask_voxels"] != PHASE2_EXPECTED_B_MASK_VOXELS:
        raise RuntimeError(
            "IBSI 2 B preprocessing mask regression: "
            f"{outputs['B']['mask_voxels']} != {PHASE2_EXPECTED_B_MASK_VOXELS}"
        )
    atomic_write_json(output_dir / "preprocessing_metadata.json", outputs)
    return outputs


def _parse_json_lines(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _run_filter(
    *,
    python: Path,
    adapter: str,
    input_path: Path,
    output_path: Path,
    config_path: Path,
    root: Path,
    bundle: Path,
    log_path: Path,
    timeout: float | None,
    native_default: bool = False,
) -> tuple[str, dict[str, Any], str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-m",
        "bench.compliance.ibsi2_native_backends",
        "--adapter",
        adapter,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--parameters-file",
        str(config_path),
    ]
    if native_default:
        command.append("--native-default")
    rendered_command = shlex.join(command)
    portable_command = _portable_command(rendered_command, root=root, bundle=bundle)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    atomic_write_text(
        log_path,
        f"command: {portable_command}\nexit_code: {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n",
    )
    metadata = _parse_json_lines(result.stdout) or _parse_json_lines(result.stderr)
    if result.returncode == 0:
        if not output_path.is_file():
            raise RuntimeError(
                f"Native backend reported success without output: {output_path}"
            )
        return "supported", metadata, portable_command
    if result.returncode == 2:
        if output_path.exists():
            raise RuntimeError(
                f"Unsupported native backend left an ambiguous output: {output_path}"
            )
        return "unsupported", metadata, portable_command
    raise RuntimeError(
        f"Native IBSI 2 backend failed for {adapter} with exit {result.returncode}; "
        f"see {log_path}"
    )


def _prepare_bundle_source(root: Path, bundle: Path) -> None:
    source_root = root / "data" / "ibsi2" / "source"
    _copy_tree(source_root, bundle / "source")
    _copy_tree(root / "data" / "ibsi2" / "licenses", bundle / "licenses")
    _copy_verified(
        source_root / "phase2" / "image" / "phantom.nii.gz",
        bundle / "source" / "phase2" / "image" / "phantom.nii.gz",
        IBSI2_PHASE2_SOURCE_IMAGE_SHA256,
    )
    _copy_verified(
        source_root / "phase2" / "mask" / "mask.nii.gz",
        bundle / "source" / "phase2" / "mask" / "mask.nii.gz",
        IBSI2_PHASE2_SOURCE_MASK_SHA256,
    )
    for spec in PHASE1_FILTER_SPECS:
        _copy_verified(
            source_root / spec.source_image_relative_path,
            bundle / "source" / spec.source_image_relative_path,
            spec.source_image_sha256,
        )
        _copy_verified(
            source_root / spec.source_mask_relative_path,
            bundle / "source" / spec.source_mask_relative_path,
            IBSI2_PHASE1_SOURCE_MASK_SHA256,
        )


def _prepare_environment_locks(
    root: Path,
    bundle: Path,
    adapters: Sequence[str],
) -> dict[str, tuple[Path, dict[str, Any], Path]]:
    output: dict[str, tuple[Path, dict[str, Any], Path]] = {}
    for adapter in adapters:
        python, metadata = _adapter_environment(adapter, root)
        source = root / ".venvs" / "adapters" / adapter / "environment.json"
        destination = bundle / "environments" / f"{adapter}.json"
        _copy_verified(source, destination, sha256_file(source))
        output[adapter] = (python, metadata, destination)
    return output


def _source_fingerprint(root: Path, adapters: Sequence[str]) -> str:
    paths = [
        Path(__file__),
        root / "bench" / "compliance" / "ibsi2_protocol.py",
        root / "bench" / "compliance" / "ibsi2_native_backends.py",
        root / "configs" / "compliance" / "ibsi2_phase2_preprocessing_A.json",
        root / "configs" / "compliance" / "ibsi2_phase2_preprocessing_B.json",
        root / "data" / "ibsi2" / "source" / "manifest.json",
    ]
    paths.extend(
        root / ".venvs" / "adapters" / adapter / "environment.json"
        for adapter in adapters
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return fingerprint(
        {str(path.relative_to(root)): sha256_file(path) for path in paths}
    )


def _load_or_create_state(
    bundle: Path,
    *,
    immutable_fingerprint: str,
    resume: bool,
) -> dict[str, Any]:
    state_path = bundle / "generation_state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"Generation state already exists: {state_path}")
        state = _json_object(state_path)
        if (
            state.get("schema_version") != GENERATION_SCHEMA_VERSION
            or state.get("immutable_fingerprint") != immutable_fingerprint
        ):
            raise RunIntegrityError(
                "IBSI 2 generator source/inputs changed; unsafe resume refused"
            )
        return state
    if bundle.exists() and any(bundle.iterdir()):
        allowed = {"source", "licenses", "environments"}
        unexpected = sorted(
            path.name for path in bundle.iterdir() if path.name not in allowed
        )
        if unexpected:
            raise FileExistsError(
                "IBSI 2 candidate bundle contains artifacts without generation state: "
                + ", ".join(unexpected)
            )
    state = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "immutable_fingerprint": immutable_fingerprint,
        "tasks": {},
        "updated_at": time.time(),
    }
    atomic_write_json(state_path, state)
    return state


def _commit_state(bundle: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    atomic_write_json(bundle / "generation_state.json", state)


def _completed_task(
    state: Mapping[str, Any],
    task_id: str,
    output_path: Path,
) -> dict[str, Any] | None:
    tasks = state.get("tasks")
    record = tasks.get(task_id) if isinstance(tasks, Mapping) else None
    if not isinstance(record, dict) or record.get("status") not in {
        "supported",
        "unsupported",
    }:
        return None
    if record["status"] == "unsupported":
        if output_path.exists():
            raise RunIntegrityError(
                f"Unsupported task has an output map: {output_path}"
            )
        return record
    expected = str(record.get("output_sha256", ""))
    if (
        not output_path.is_file()
        or not expected
        or sha256_file(output_path) != expected
    ):
        raise RunIntegrityError(
            f"Completed response map changed or disappeared: {output_path}"
        )
    return record


# Exact native support, and the separate class that runs a filter with the
# package's own boundary or truncation when its API cannot set them.
SUPPORT_CLASSES = ("native", "native_default")


def _class_folder(phase: str, support_class: str) -> str:
    return phase if support_class == "native" else f"{phase}_native_default"


def _filter_task(
    *,
    state: dict[str, Any],
    bundle: Path,
    root: Path,
    task_id: str,
    output_map: Path,
    log_path: Path,
    exact: Mapping[str, Any] | None,
    python: Path,
    adapter: str,
    input_path: Path,
    config_path: Path,
    timeout: float | None,
) -> dict[str, Any]:
    """Run or resume one filter task; ``exact`` is set for a native-default task.

    A native-default task runs only when the exact task failed on a boundary
    or truncation control alone.
    """

    if exact is not None:
        metadata = exact.get("backend_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if exact.get("status") == "supported":
            reason = "the exact run supports this configuration"
        elif not metadata.get("native_default_possible"):
            reason = "the package lacks the operation itself: " + str(
                metadata.get("reason") or "no native filter"
            )
        else:
            reason = None
        if reason is not None:
            return {
                "status": "unsupported",
                "backend_metadata": {
                    "reason": f"no native-default run; {reason}",
                    "evidence": ["exact task: " + str(exact.get("status"))],
                },
            }
    task = _completed_task(state, task_id, output_map)
    if task is not None:
        return task
    status, metadata, command = _run_filter(
        python=python,
        adapter=adapter,
        input_path=input_path,
        output_path=output_map,
        config_path=config_path,
        root=root,
        bundle=bundle,
        log_path=log_path,
        timeout=timeout,
        native_default=exact is not None,
    )
    task = {
        "status": status,
        "backend_metadata": metadata,
        "command": command,
        "config_sha256": sha256_file(config_path),
        "output_sha256": sha256_file(output_map) if status == "supported" else None,
        "log_sha256": sha256_file(log_path),
    }
    state["tasks"][task_id] = task
    _commit_state(bundle, state)
    return task


def _support_declaration(
    *,
    adapter: str,
    id_field: str,
    identifier: str,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    supported = task.get("status") == "supported"
    metadata = task.get("backend_metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if supported:
        boundary = metadata.get("boundary_execution")
        boundary = boundary if isinstance(boundary, Mapping) else {}
        reason = (
            "Installed package-native backend executed the reviewed IBSI "
            "parameter set under the recorded boundary policy"
        )
        evidence = (
            f"backend={metadata.get('backend', 'bench.compliance.ibsi2_native_backends')}; "
            f"reported_distribution_version={metadata.get('distribution_version', 'recorded-in-lock')}; "
            f"boundary_policy={boundary.get('policy', 'unrecorded')}; "
            f"selected_boundary={boundary.get('selected', 'unrecorded')}; "
            f"effective_boundary={boundary.get('effective', 'unrecorded')}; "
            f"boundary_implementation={boundary.get('implementation', 'unrecorded')}; "
            f"output_sha256={task.get('output_sha256')}; log_sha256={task.get('log_sha256')}"
        )
    else:
        reason = str(
            metadata.get("reason")
            or metadata.get("error")
            or "Installed package cannot express every reviewed IBSI parameter"
        )
        raw_evidence = metadata.get("evidence")
        if isinstance(raw_evidence, Sequence) and not isinstance(
            raw_evidence, (str, bytes)
        ):
            evidence = "; ".join(str(value) for value in raw_evidence)
        else:
            evidence = str(
                raw_evidence
                or f"native backend rejection; log_sha256={task.get('log_sha256')}"
            )
    return {
        "adapter": adapter,
        id_field: identifier,
        "native_supported": supported,
        "reason": reason,
        "evidence": evidence,
        "uncontrolled_parameters": list(
            (metadata.get("native_default") or {}).get("uncontrolled", [])
            if supported
            else metadata.get("uncontrolled", [])
        ),
    }


def _execution_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    parameters = metadata.get("parameters")
    boundary = metadata.get("boundary_execution")
    if not isinstance(parameters, Mapping) or not isinstance(boundary, Mapping):
        raise RunIntegrityError(
            "Supported native IBSI 2 execution lacks normalized parameter or "
            "boundary provenance"
        )
    contract: dict[str, Any] = {
        "executed_parameters": dict(parameters),
        "boundary_execution": dict(boundary),
        "native_capability": None,
    }
    if metadata.get("native_default") is not None:
        contract["native_default"] = dict(metadata["native_default"])
    capability = metadata.get("native_capability")
    if isinstance(capability, Mapping):
        contract["native_capability"] = dict(capability)
    elif (
        metadata.get("adapter") == "pictologics" and parameters.get("filter") != "none"
    ):
        raise RunIntegrityError(
            "Pictologics native execution lacks its versioned capability record"
        )
    return contract


def _ensure_phase2_preprocessing(
    *,
    root: Path,
    bundle: Path,
    pictologics_python: Path,
    timeout: float | None,
) -> tuple[dict[str, Any], str]:
    output_dir = bundle / "preprocessed" / "phase2"
    metadata_path = output_dir / "preprocessing_metadata.json"
    command = [
        str(pictologics_python),
        "-m",
        "bench.compliance.ibsi2_candidates",
        "preprocess-phase2",
        "--image",
        str(bundle / "source" / "phase2" / "image" / "phantom.nii.gz"),
        "--mask",
        str(bundle / "source" / "phase2" / "mask" / "mask.nii.gz"),
        "--output-dir",
        str(output_dir),
    ]
    rendered = shlex.join(command)
    portable_command = _portable_command(rendered, root=root, bundle=bundle)
    if not metadata_path.is_file():
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root)
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        atomic_write_text(
            bundle / "logs" / "phase2_preprocessing.log",
            f"command: {portable_command}\nexit_code: {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}\n",
        )
        if result.returncode != 0:
            raise RuntimeError(
                "IBSI 2 Phase 2 preprocessing failed; see logs/phase2_preprocessing.log"
            )
    metadata = _json_object(metadata_path)
    for dimension in ("A", "B"):
        entry = metadata.get(dimension)
        if not isinstance(entry, Mapping):
            raise RunIntegrityError(
                f"Missing Phase 2 {dimension} preprocessing metadata"
            )
        for kind in ("image", "mask"):
            path = output_dir / f"{dimension}-{kind}.nii.gz"
            if not path.is_file() or sha256_file(path) != entry.get(f"{kind}_sha256"):
                raise RunIntegrityError(
                    f"Phase 2 {dimension} preprocessing artifact changed"
                )
    return metadata, portable_command


def generate_candidate_bundle(
    *,
    output_dir: Path,
    adapters: Sequence[str] = DEFAULT_ADAPTERS,
    phases: Iterable[str] = ("phase1", "phase2"),
    resume: bool = False,
    timeout: float | None = 1800.0,
) -> dict[str, Any]:
    """Generate exhaustive support grids and all native-supported response maps."""

    root = _repo_root()
    bundle = Path(output_dir).expanduser().resolve()
    adapters = tuple(dict.fromkeys(adapters))
    phases = tuple(dict.fromkeys(str(phase).casefold() for phase in phases))
    if not adapters:
        raise ValueError("At least one adapter is required")
    if not phases or any(phase not in {"phase1", "phase2"} for phase in phases):
        raise ValueError("phases must contain phase1 and/or phase2")
    bundle.mkdir(parents=True, exist_ok=True)
    immutable_fingerprint = _source_fingerprint(root, adapters)
    state = _load_or_create_state(
        bundle,
        immutable_fingerprint=immutable_fingerprint,
        resume=resume,
    )
    _prepare_bundle_source(root, bundle)
    environments = _prepare_environment_locks(root, bundle, adapters)

    backend_source = root / "bench" / "compliance" / "ibsi2_native_backends.py"
    protocol_source = root / "bench" / "compliance" / "ibsi2_protocol.py"
    generator_revision = (
        f"backend-sha256:{sha256_file(backend_source)};"
        f"protocol-sha256:{sha256_file(protocol_source)}"
    )
    source_archive = bundle / "generator" / "ibsi2_native_backends.py"
    protocol_archive = bundle / "generator" / "ibsi2_protocol.py"
    controller_archive = bundle / "generator" / "ibsi2_candidates.py"
    _copy_verified(backend_source, source_archive, sha256_file(backend_source))
    _copy_verified(protocol_source, protocol_archive, sha256_file(protocol_source))
    _copy_verified(Path(__file__), controller_archive, sha256_file(Path(__file__)))

    manifests: dict[str, Any] = {}
    if "phase1" in phases:
        entries: dict[str, list[dict[str, Any]]] = {c: [] for c in SUPPORT_CLASSES}
        declarations: dict[str, list[dict[str, Any]]] = {c: [] for c in SUPPORT_CLASSES}
        for spec in PHASE1_FILTER_SPECS:
            filter_config = bundle / "configs" / "phase1" / f"{spec.test_id}.json"
            preprocessing_config = (
                bundle / "configs" / "phase1" / f"{spec.test_id}-preprocessing.json"
            )
            _write_json(filter_config, spec.filter_config())
            _write_json(preprocessing_config, spec.preprocessing_config())
            source_image = bundle / "source" / spec.source_image_relative_path
            source_mask = bundle / "source" / spec.source_mask_relative_path
            for adapter, support_class in (
                (adapter, support_class)
                for adapter in adapters
                for support_class in SUPPORT_CLASSES
            ):
                python, environment, environment_lock = environments[adapter]
                folder = _class_folder("phase1", support_class)
                output_map = (
                    bundle / "maps" / folder / adapter / f"{spec.test_id}.nii.gz"
                )
                task = _filter_task(
                    state=state,
                    bundle=bundle,
                    root=root,
                    task_id=f"{folder}::{adapter}::{spec.test_id}",
                    output_map=output_map,
                    log_path=bundle / "logs" / folder / adapter / f"{spec.test_id}.log",
                    exact=None
                    if support_class == "native"
                    else state["tasks"][f"phase1::{adapter}::{spec.test_id}"],
                    python=python,
                    adapter=adapter,
                    input_path=source_image,
                    config_path=filter_config,
                    timeout=timeout,
                )
                declarations[support_class].append(
                    _support_declaration(
                        adapter=adapter,
                        id_field="test_id",
                        identifier=spec.test_id,
                        task=task,
                    )
                )
                if task["status"] != "supported":
                    continue
                profile = environment["profile_definition"]
                entries[support_class].append(
                    {
                        "support_class": support_class,
                        "adapter": adapter,
                        "test_id": spec.test_id,
                        "response_map_path": _relative(output_map, bundle),
                        "response_map_sha256": task["output_sha256"],
                        "source_image_path": _relative(source_image, bundle),
                        "source_image_sha256": spec.source_image_sha256,
                        "source_mask_path": _relative(source_mask, bundle),
                        "source_mask_sha256": IBSI2_PHASE1_SOURCE_MASK_SHA256,
                        "generator_distribution": profile["distribution"],
                        "generator_version": profile["version"],
                        "generator_distribution_metadata_version": environment.get(
                            "distribution_metadata_version", environment["version"]
                        ),
                        "generator_source_revision": generator_revision,
                        "generator_source_path": _relative(source_archive, bundle),
                        "generator_source_sha256": sha256_file(source_archive),
                        "generator_entrypoint": "bench.compliance.ibsi2_native_backends:main",
                        "generator_command": task["command"],
                        **_execution_contract(task["backend_metadata"]),
                        "filter_config_revision": (
                            f"ibsi2-manual-v9-sha256:{sha256_file(filter_config)}"
                        ),
                        "filter_config_path": _relative(filter_config, bundle),
                        "filter_config_sha256": sha256_file(filter_config),
                        "preprocessing_config_path": _relative(
                            preprocessing_config, bundle
                        ),
                        "preprocessing_config_sha256": sha256_file(
                            preprocessing_config
                        ),
                        "environment_lock_path": _relative(environment_lock, bundle),
                        "environment_lock_sha256": sha256_file(environment_lock),
                    }
                )
        manifest = {
            "schema_version": 1,
            "kind": "ibsi2_phase1_response_maps",
            "source_data": {
                "repository": IBSI_DATA_REPOSITORY,
                "commit": IBSI_DATA_COMMIT,
            },
            "protocol_review": {
                "status": "reviewed",
                "reviewed_against": IBSI2_PROTOCOL_REVIEW,
                "reviewed_by": REVIEWED_BY,
                "reviewed_at": REVIEWED_AT,
                "author_signoff": {
                    "status": "pending",
                    "required_for": "manuscript publication claims",
                },
            },
            "adapters": list(adapters),
        }
        for support_class in SUPPORT_CLASSES:
            folder = _class_folder("phase1", support_class)
            manifest_path = bundle / f"ibsi2_{folder}_candidates.json"
            _write_json(
                manifest_path,
                {
                    **manifest,
                    "support_class": support_class,
                    "support_declarations": declarations[support_class],
                    "entries": entries[support_class],
                },
            )
            manifests[folder] = {
                "path": _relative(manifest_path, bundle),
                "sha256": sha256_file(manifest_path),
                "supported_maps": len(entries[support_class]),
                "support_declarations": len(declarations[support_class]),
            }

    if "phase2" in phases:
        pictologics_python = environments["pictologics"][0]
        preprocessing_metadata, preprocessing_command = _ensure_phase2_preprocessing(
            root=root,
            bundle=bundle,
            pictologics_python=pictologics_python,
            timeout=timeout,
        )
        portable_preprocessing_metadata = {
            dimension: {
                **dict(values),
                "image_path": _relative(
                    bundle / "preprocessed" / "phase2" / f"{dimension}-image.nii.gz",
                    bundle,
                ),
                "mask_path": _relative(
                    bundle / "preprocessed" / "phase2" / f"{dimension}-mask.nii.gz",
                    bundle,
                ),
            }
            for dimension, values in preprocessing_metadata.items()
        }
        for dimension in ("A", "B"):
            source = (
                root
                / "configs"
                / "compliance"
                / (f"ibsi2_phase2_preprocessing_{dimension}.json")
            )
            destination = (
                bundle / "configs" / "phase2" / f"preprocessing_{dimension}.json"
            )
            _copy_verified(source, destination, sha256_file(source))
        entries = {c: [] for c in SUPPORT_CLASSES}
        declarations = {c: [] for c in SUPPORT_CLASSES}
        for spec in PHASE2_FILTER_SPECS:
            filter_config = bundle / "configs" / "phase2" / f"{spec.filter_id}.json"
            _write_json(filter_config, spec.filter_config())
            base_image = (
                bundle / "preprocessed" / "phase2" / f"{spec.dimension}-image.nii.gz"
            )
            base_mask = (
                bundle / "preprocessed" / "phase2" / f"{spec.dimension}-mask.nii.gz"
            )
            preprocessing_config = (
                bundle / "configs" / "phase2" / f"preprocessing_{spec.dimension}.json"
            )
            for adapter, support_class in (
                (adapter, support_class)
                for adapter in adapters
                for support_class in SUPPORT_CLASSES
            ):
                python, environment, environment_lock = environments[adapter]
                folder = _class_folder("phase2", support_class)
                output_map = (
                    bundle / "maps" / folder / adapter / f"{spec.filter_id}.nii.gz"
                )
                task = _filter_task(
                    state=state,
                    bundle=bundle,
                    root=root,
                    task_id=f"{folder}::{adapter}::{spec.filter_id}",
                    output_map=output_map,
                    log_path=bundle
                    / "logs"
                    / folder
                    / adapter
                    / f"{spec.filter_id}.log",
                    exact=None
                    if support_class == "native"
                    else state["tasks"][f"phase2::{adapter}::{spec.filter_id}"],
                    python=python,
                    adapter=adapter,
                    input_path=base_image,
                    config_path=filter_config,
                    timeout=timeout,
                )
                declarations[support_class].append(
                    _support_declaration(
                        adapter=adapter,
                        id_field="filter_id",
                        identifier=spec.filter_id,
                        task=task,
                    )
                )
                if task["status"] != "supported":
                    continue
                profile = environment["profile_definition"]
                entries[support_class].append(
                    {
                        "support_class": support_class,
                        "adapter": adapter,
                        "filter_id": spec.filter_id,
                        "image_path": _relative(output_map, bundle),
                        "mask_path": _relative(base_mask, bundle),
                        "image_sha256": task["output_sha256"],
                        "mask_sha256": sha256_file(base_mask),
                        "filter_input_path": _relative(base_image, bundle),
                        "filter_input_sha256": sha256_file(base_image),
                        "generator_distribution": profile["distribution"],
                        "generator_version": profile["version"],
                        "generator_distribution_metadata_version": environment.get(
                            "distribution_metadata_version", environment["version"]
                        ),
                        "generator_source_revision": generator_revision,
                        "generator_source_path": _relative(source_archive, bundle),
                        "generator_source_sha256": sha256_file(source_archive),
                        "generator_entrypoint": "bench.compliance.ibsi2_native_backends:main",
                        "generator_command": task["command"],
                        **_execution_contract(task["backend_metadata"]),
                        "filter_config_revision": (
                            f"ibsi2-manual-v9-sha256:{sha256_file(filter_config)}"
                        ),
                        "filter_config_path": _relative(filter_config, bundle),
                        "filter_config_sha256": sha256_file(filter_config),
                        "preprocessing_config_path": _relative(
                            preprocessing_config, bundle
                        ),
                        "preprocessing_config_sha256": sha256_file(
                            preprocessing_config
                        ),
                        "environment_lock_path": _relative(environment_lock, bundle),
                        "environment_lock_sha256": sha256_file(environment_lock),
                    }
                )
        manifest = {
            "schema_version": 1,
            "kind": "ibsi2_phase2_response_maps",
            "source_data": {
                "repository": IBSI_DATA_REPOSITORY,
                "commit": IBSI_DATA_COMMIT,
                "image_path": "source/phase2/image/phantom.nii.gz",
                "image_sha256": IBSI2_PHASE2_SOURCE_IMAGE_SHA256,
                "mask_path": "source/phase2/mask/mask.nii.gz",
                "mask_sha256": IBSI2_PHASE2_SOURCE_MASK_SHA256,
            },
            "controlled_preprocessing": {
                "scope": (
                    "common reviewed A/B preprocessing; package-native filters and "
                    "package-native first-order feature extractors are evaluated"
                ),
                "generator_distribution": "pictologics",
                "generator_version": environments["pictologics"][1][
                    "profile_definition"
                ]["version"],
                "generator_command": preprocessing_command,
                "generator_source_path": _relative(controller_archive, bundle),
                "generator_source_sha256": sha256_file(controller_archive),
                "environment_lock_path": _relative(
                    environments["pictologics"][2], bundle
                ),
                "environment_lock_sha256": sha256_file(environments["pictologics"][2]),
                "metadata_path": "preprocessed/phase2/preprocessing_metadata.json",
                "metadata_sha256": sha256_file(
                    bundle / "preprocessed" / "phase2" / "preprocessing_metadata.json"
                ),
                "outputs": portable_preprocessing_metadata,
            },
            "protocol_review": {
                "status": "reviewed",
                "reviewed_against": IBSI2_PROTOCOL_REVIEW,
                "reviewed_by": REVIEWED_BY,
                "reviewed_at": REVIEWED_AT,
                "author_signoff": {
                    "status": "pending",
                    "required_for": "manuscript publication claims",
                },
            },
            "adapters": list(adapters),
        }
        for support_class in SUPPORT_CLASSES:
            folder = _class_folder("phase2", support_class)
            manifest_path = bundle / f"ibsi2_{folder}_candidates.json"
            _write_json(
                manifest_path,
                {
                    **manifest,
                    "support_class": support_class,
                    "support_declarations": declarations[support_class],
                    "entries": entries[support_class],
                },
            )
            manifests[folder] = {
                "path": _relative(manifest_path, bundle),
                "sha256": sha256_file(manifest_path),
                "supported_maps": len(entries[support_class]),
                "support_declarations": len(declarations[support_class]),
            }

    result = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "candidate_bundle": ".",
        "immutable_fingerprint": immutable_fingerprint,
        "adapters": list(adapters),
        "phases": list(phases),
        "manifests": manifests,
    }
    _write_json(bundle / "bundle_manifest.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench.compliance.ibsi2_candidates")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser(
        "generate", help="Generate a resumable candidate bundle"
    )
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--adapters", default=",".join(DEFAULT_ADAPTERS))
    generate.add_argument("--phases", default="phase1,phase2")
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--timeout", type=float, default=1800.0)
    preprocess = subparsers.add_parser(
        "preprocess-phase2", help="Internal exact A/B preprocessing subprocess"
    )
    preprocess.add_argument("--image", required=True)
    preprocess.add_argument("--mask", required=True)
    preprocess.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preprocess-phase2":
        result = preprocess_phase2(
            image_path=Path(args.image),
            mask_path=Path(args.mask),
            output_dir=Path(args.output_dir),
        )
    else:
        if args.timeout is not None and args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        result = generate_candidate_bundle(
            output_dir=Path(args.output_dir),
            adapters=tuple(
                value.strip() for value in args.adapters.split(",") if value.strip()
            ),
            phases=tuple(
                value.strip() for value in args.phases.split(",") if value.strip()
            ),
            resume=args.resume,
            timeout=args.timeout,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["generate_candidate_bundle", "main", "preprocess_phase2"]
