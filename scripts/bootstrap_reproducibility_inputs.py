#!/usr/bin/env python3
"""Fetch every non-Git input at pinned revisions and verify every byte.

The repository intentionally does not contain NIfTI images, clinical data, or
the IBSI workbook.  This standard-library bootstrapper materialises those
inputs from their authoritative sources according to
``reproducibility/inputs/manifest.json``.  It is safe to rerun: valid files are
reused, mismatches fail closed, and replacement requires ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_COMPONENTS = (
    "ibsi1",
    "ibsi2-phase1",
    "ibsi2-phase2",
    "ibsi2-phase3",
    "licenses",
    "workbook",
)


class BootstrapError(RuntimeError):
    pass


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise BootstrapError(
            f"manifest destination escapes output root: {relative}"
        ) from exc
    return target


def _run(command: list[str]) -> str:
    try:
        process = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise BootstrapError(f"command failed: {' '.join(command)}\n{detail}") from exc
    return process.stdout.strip()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"unable to read input manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BootstrapError("unsupported reproducibility input manifest")
    return value


def _verify_repository(path: Path, expected_commit: str) -> Path:
    if not (path / ".git").is_dir():
        raise BootstrapError(f"not a Git checkout: {path}")
    observed = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
    if observed != expected_commit:
        raise BootstrapError(
            f"repository {path} is at {observed}; expected {expected_commit}"
        )
    return path


def _ensure_repository(
    name: str,
    specification: Mapping[str, Any],
    cache: Path,
    overrides: Mapping[str, Path],
) -> Path:
    commit = str(specification["commit"])
    if name in overrides:
        return _verify_repository(overrides[name].expanduser().resolve(), commit)
    target = cache / name
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        # Keep the upstream bytes: Git on Windows converts line endings otherwise.
        _run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--config",
                "core.autocrlf=false",
                str(specification["url"]),
                str(target),
            ]
        )
        _run(["git", "-C", str(target), "checkout", "--detach", commit])
    return _verify_repository(target, commit)


def _atomic_copy(source: Path, destination: Path, expected: str, *, force: bool) -> str:
    if sha256_file(source) != expected:
        raise BootstrapError(f"upstream checksum mismatch: {source}")
    if destination.is_file():
        observed = sha256_file(destination)
        if observed == expected:
            return "reused"
        if not force:
            raise BootstrapError(
                f"existing destination checksum mismatch: {destination}; "
                "inspect it or rerun with --force"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(Path(temporary)) != expected:
            raise BootstrapError(f"temporary copy checksum mismatch: {destination}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "copied"


def _download(url: str, destination: Path, expected: str, *, force: bool) -> str:
    if destination.is_file():
        observed = sha256_file(destination)
        if observed == expected:
            return "reused"
        if not force:
            raise BootstrapError(
                f"existing download checksum mismatch: {destination}; "
                "inspect it or rerun with --force"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(descriptor)
    try:
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                open(temporary, "wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
        except Exception as urllib_error:
            # Some python.org macOS interpreters do not inherit the system CA
            # store until their optional certificate installer has been run.
            # The system curl client uses the native trust store and still
            # performs full TLS verification; never fall back to insecure TLS.
            curl = shutil.which("curl")
            if not curl:
                raise BootstrapError(
                    f"secure download failed and curl is unavailable: {url}"
                ) from urllib_error
            _run(
                [
                    curl,
                    "--location",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "300",
                    "--output",
                    temporary,
                    url,
                ]
            )
        if sha256_file(Path(temporary)) != expected:
            raise BootstrapError(f"download checksum mismatch: {url}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "downloaded"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_or_verify_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    verify_only: bool,
) -> None:
    expected = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if verify_only:
        if not path.is_file() or path.read_bytes() != expected:
            raise BootstrapError(f"generated manifest is missing or changed: {path}")
        return
    _atomic_json(path, value)


def _entries_for(
    manifest: Mapping[str, Any], components: set[str]
) -> list[Mapping[str, Any]]:
    return [
        entry for entry in manifest["files"] if str(entry["component"]) in components
    ]


def _require_destinations(root: Path, entries: Iterable[Mapping[str, Any]]) -> None:
    for entry in entries:
        path = _inside(root, str(entry["destination"]))
        if not path.is_file():
            raise BootstrapError(f"required input is missing: {path}")
        if sha256_file(path) != str(entry["sha256"]):
            raise BootstrapError(f"required input checksum mismatch: {path}")


def _write_generated_manifests(
    root: Path,
    manifest: Mapping[str, Any],
    components: set[str],
    *,
    verify_only: bool,
) -> None:
    all_entries = list(manifest["files"])
    if "ibsi1" in components:
        rows = [entry for entry in all_entries if entry["component"] == "ibsi1"]
        _require_destinations(root, rows)
        files = [
            {
                "path": Path(str(row["destination"]))
                .relative_to("data/ibsi1/digital_phantom")
                .as_posix(),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
            for row in rows
        ]
        _write_or_verify_json(
            root / "data/ibsi1/digital_phantom/manifest.json",
            {
                "schema_version": 1,
                "dataset": "IBSI 1 digital phantom",
                "protocol": {
                    "preprocessing": "none",
                    "grey_levels": "identity",
                    "directional_texture_aggregation": "3d_merge",
                    "nondirectional_texture_aggregation": "3d_single_matrix",
                },
                "source": manifest["repositories"]["ibsi_data_sets"],
                "files": files,
            },
            verify_only=verify_only,
        )

    if {"ibsi2-phase1", "ibsi2-phase2"}.intersection(components):
        phase1 = [
            entry
            for entry in all_entries
            if entry["component"] == "ibsi2-phase1"
            and str(entry["destination"]).startswith("data/ibsi2/source/")
        ]
        phase2 = [
            entry
            for entry in all_entries
            if entry["component"] == "ibsi2-phase2"
            and str(entry["destination"]).startswith("data/ibsi2/source/")
        ]
        if (
            phase1
            and phase2
            and all(
                _inside(root, str(entry["destination"])).is_file()
                for entry in [*phase1, *phase2]
            )
        ):
            _require_destinations(root, [*phase1, *phase2])
            images: dict[str, str] = {}
            masks: list[str] = []
            for entry in phase1:
                destination = Path(str(entry["destination"]))
                if destination.parent.name == "image":
                    images[destination.stem.removesuffix(".nii")] = str(entry["sha256"])
                elif destination.parent.name == "mask":
                    masks.append(str(entry["sha256"]))
            if len(set(masks)) != 1:
                raise BootstrapError("IBSI 2 Phase 1 masks are not byte-identical")
            phase2_by_role = {
                Path(str(entry["destination"])).parent.name: entry for entry in phase2
            }
            _write_or_verify_json(
                root / "data/ibsi2/source/manifest.json",
                {
                    "schema_version": 1,
                    "source": manifest["repositories"]["ibsi_data_sets"],
                    "phase1": {
                        "dataset": "ibsi_2_digital_phantom",
                        "license": "CC BY 4.0",
                        "license_path": "../licenses/digital_phantom_LICENSE.md",
                        "common_mask_sha256": masks[0],
                        "images": dict(sorted(images.items())),
                        "note": "The official orientation phantom has no mask file.",
                    },
                    "phase2": {
                        "dataset": "ibsi_2_ct_radiomics_phantom",
                        "license": "CC BY-NC 3.0",
                        "license_path": "../licenses/ct_phantom_LICENSE.md",
                        "image_path": "phase2/image/phantom.nii.gz",
                        "image_sha256": phase2_by_role["image"]["sha256"],
                        "mask_path": "phase2/mask/mask.nii.gz",
                        "mask_sha256": phase2_by_role["mask"]["sha256"],
                    },
                },
                verify_only=verify_only,
            )

    if "ibsi2-phase3" in components:
        rows = [entry for entry in all_entries if entry["component"] == "ibsi2-phase3"]
        _require_destinations(root, rows)
        files = [
            {
                "path": Path(str(row["destination"]))
                .relative_to("data/ibsi2_validation")
                .as_posix(),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
            for row in sorted(rows, key=lambda item: str(item["destination"]))
        ]
        _write_or_verify_json(
            root / "data/ibsi2_validation/manifest.json",
            {
                "schema_version": 1,
                "dataset": "IBSI validation STS NIfTI cohort",
                "subjects": 51,
                "modalities": ["ct", "mri", "pet"],
                "source": manifest["repositories"]["ibsi_data_sets"],
                "files": files,
            },
            verify_only=verify_only,
        )


def _parse_overrides(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise BootstrapError("--repository-source must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in parsed:
            raise BootstrapError(f"invalid repository override: {value}")
        parsed[name] = Path(raw_path)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root() / "reproducibility/inputs/manifest.json",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=DEFAULT_COMPONENTS,
        help="Materialise only this component; repeat to select several",
    )
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument(
        "--repository-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Use an already checked-out, exact-commit source (air-gapped testing)",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    source_cache = (
        args.source_cache.expanduser().resolve()
        if args.source_cache
        else root / "data/.upstream"
    )
    manifest = _load_manifest(args.manifest.expanduser().resolve())
    components = set(args.component or DEFAULT_COMPONENTS)
    overrides = _parse_overrides(args.repository_source)
    entries = _entries_for(manifest, components)
    summary = {"copied": 0, "downloaded": 0, "reused": 0, "verified": 0}

    if args.verify_only:
        _require_destinations(root, entries)
        selected_downloads = [
            item
            for item in manifest["downloads"]
            if str(item["component"]) in components
        ]
        _require_destinations(root, selected_downloads)
        summary["verified"] = len(entries) + len(selected_downloads)
    else:
        selected_repositories = sorted({str(entry["repository"]) for entry in entries})
        repositories = {
            name: _ensure_repository(
                name,
                manifest["repositories"][name],
                source_cache,
                overrides,
            )
            for name in selected_repositories
        }
        for index, entry in enumerate(entries, 1):
            source = repositories[str(entry["repository"])] / str(entry["source"])
            destination = _inside(root, str(entry["destination"]))
            outcome = _atomic_copy(
                source, destination, str(entry["sha256"]), force=args.force
            )
            summary[outcome] += 1
            if index % 50 == 0 or index == len(entries):
                print(f"inputs {index}/{len(entries)}: {destination}")
        for item in manifest["downloads"]:
            if str(item["component"]) not in components:
                continue
            outcome = _download(
                str(item["url"]),
                _inside(root, str(item["destination"])),
                str(item["sha256"]),
                force=args.force,
            )
            summary[outcome] += 1

    _write_generated_manifests(
        root,
        manifest,
        components,
        verify_only=args.verify_only,
    )
    print(json.dumps({"status": "pass", "components": sorted(components), **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
