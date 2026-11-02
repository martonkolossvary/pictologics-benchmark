"""The release lock: every checksum that a reproduction of this release must match.

``reproducibility/release-lock.json`` binds the contracts, profiles, locks,
inputs, the timing and dataset code, and the built workspace manifests.
Refresh it after a deliberate change with ``python -m bench.release_lock --write``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bench.benchmark_ledger import atomic_write_json, sha256_file

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "reproducibility" / "release-lock.json"
WORKSPACE = REPO / "data" / "benchmark"
PILLAR_FOLDERS = ("pillar1", "pillar2_a1", "ibsi2_phase3")
BOUND_PATTERNS = (
    "poetry.lock",
    "pyproject.toml",
    "configs/adapters/*.yaml",
    "configs/benchmark/*.json",
    "configs/benchmark/hosts/*.json",
    "configs/compliance/*.json",
    "reproducibility/toolchain.txt",
    "reproducibility/inputs/manifest.json",
    "reproducibility/contracts/*",
    "reproducibility/environment-locks/*/*.txt",
    "bench/*.py",
    "bench/adapters/*.py",
    "bench/compliance/*.py",
    "scripts/launch_benchmark.py",
    "scripts/prepare_workspace.py",
    "scripts/bootstrap_reproducibility_inputs.py",
)


class ReleaseLockError(RuntimeError):
    pass


def bound_files() -> dict[str, str]:
    paths = sorted({path for pattern in BOUND_PATTERNS for path in REPO.glob(pattern)})
    return {path.relative_to(REPO).as_posix(): sha256_file(path) for path in paths}


def workspace_hashes() -> dict[str, Any] | None:
    manifest = WORKSPACE / "workspace_manifest.json"
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "workspace_manifest_sha256": sha256_file(manifest),
        "dataset_manifest_sha256": {
            folder: sha256_file(WORKSPACE / folder / "manifest.json")
            for folder in PILLAR_FOLDERS
        },
        "scheduled_tasks": {
            pillar: counts["scheduled_task_records"]
            for pillar, counts in record["task_inventory"]["pillars"].items()
        },
    }


def current() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "files": bound_files(),
        "workspace": workspace_hashes(),
    }


def load() -> dict[str, Any]:
    if not LOCK.is_file():
        raise ReleaseLockError(f"release lock is missing: {LOCK}")
    return json.loads(LOCK.read_text(encoding="utf-8"))


def verify(*, workspace: bool = False) -> None:
    """Fail when a bound file, or the built workspace, differs from the release."""

    expected = load()
    observed = bound_files()
    changed = sorted(
        path
        for path in set(expected["files"]) | set(observed)
        if expected["files"].get(path) != observed.get(path)
    )
    if changed:
        raise ReleaseLockError(
            "These files differ from the release; restore them with git checkout:\n  "
            + "\n  ".join(changed)
        )
    if workspace and workspace_hashes() != expected["workspace"]:
        raise ReleaseLockError(
            "The built workspace differs from the release; remove data/benchmark "
            "and run the command again to rebuild it."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Compare with the lock")
    mode.add_argument("--write", action="store_true", help="Rewrite the lock")
    args = parser.parse_args(argv)
    observed = current()
    if args.write:
        if observed["workspace"] is None:
            raise SystemExit("build data/benchmark first: the lock binds its manifests")
        atomic_write_json(LOCK, observed)
        print(f"Wrote {LOCK.relative_to(REPO)} ({len(observed['files'])} files)")
        return 0
    try:
        verify(workspace=observed["workspace"] is not None)
    except ReleaseLockError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("The release lock matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
