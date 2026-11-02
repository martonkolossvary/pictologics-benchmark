#!/usr/bin/env python3
"""Fail if a staged/tracked publication repository contains generated payloads."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


FORBIDDEN_ROOTS = {".venv", ".venvs", "artifacts", "data", "envs", "results"}
FORBIDDEN_SUFFIXES = {
    ".dcm",
    ".docx",
    ".bundle",
    ".db",
    ".db-shm",
    ".db-wal",
    ".jpg",
    ".jpeg",
    ".jsonl",
    ".mat",
    ".mha",
    ".mhd",
    ".nii",
    ".npy",
    ".npz",
    ".nrrd",
    ".pdf",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".whl",
    ".sqlite",
    ".sqlite3",
    ".xls",
    ".xlsx",
    ".zip",
}
FORBIDDEN_NAMES = {".DS_Store", "logs.log"}
REQUIRED = {
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "README.md",
    "poetry.lock",
    "pyproject.toml",
    "reproducibility/inputs/manifest.json",
    "reproducibility/contracts/adapter_feature_surface.csv",
}
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024


def _tracked(root: Path) -> list[str]:
    process = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return sorted(
        value.decode("utf-8") for value in process.stdout.split(b"\0") if value
    )


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    tracked = _tracked(root)
    failures: list[str] = []
    tracked_set = set(tracked)
    for missing in sorted(REQUIRED - tracked_set):
        failures.append(f"required file is not tracked: {missing}")
    for relative in tracked:
        path = Path(relative)
        suffixes = "".join(path.suffixes).casefold()
        if path.parts and path.parts[0] in FORBIDDEN_ROOTS:
            failures.append(f"generated root is tracked: {relative}")
        if path.name in FORBIDDEN_NAMES:
            failures.append(f"local artifact is tracked: {relative}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES or suffixes in {
            ".nii.gz",
            ".tar.gz",
        }:
            failures.append(f"binary/generated payload is tracked: {relative}")
        absolute = root / path
        if absolute.is_file() and absolute.stat().st_size > MAX_TRACKED_FILE_BYTES:
            failures.append(
                f"tracked file exceeds {MAX_TRACKED_FILE_BYTES} bytes: {relative}"
            )
    return {
        "status": "pass" if not failures else "fail",
        "tracked_files": len(tracked),
        "tracked_bytes": sum(
            (root / relative).stat().st_size
            for relative in tracked
            if (root / relative).is_file()
        ),
        "failures": failures,
    }


def main() -> int:
    result = audit(Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
