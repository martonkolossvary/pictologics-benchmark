#!/usr/bin/env python3
"""Resolve each adapter environment lock for Linux and Windows from the macOS lock.

The macOS Apple Silicon lock is the reference.  Every package that also exists
on the target platform keeps its reference version; packages that only the
target platform needs resolve with the reference lock's cutoff date.  The
seeded ``pip``, ``setuptools`` and ``wheel`` versions resolve together with the
adapter, so their own dependencies are locked too.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

from bench.env import canonical_requirement, load_runtime_profiles, repo_root

REFERENCE_KEY = "Darwin-arm64"
# Linux on ARM is not a target: Z-Rad requires PyQt5, which has no wheel there.
TARGETS = {
    "Linux-x86_64": "x86_64-unknown-linux-gnu",
    "Windows-AMD64": "x86_64-pc-windows-msvc",
}
# The date the reference locks were resolved.
CUTOFF = "2026-08-29T23:59:59Z"
# A later date for one package: its new release, while every other package
# keeps the common cutoff.
PACKAGE_CUTOFFS = {"pictologics": "2026-11-01T23:59:59Z"}
SEEDS = {"pip", "setuptools", "wheel"}


def _lines_of(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _lines(path: Path) -> list[str]:
    return _lines_of(path.read_text(encoding="utf-8"))


def _name(line: str) -> str:
    return canonical_requirement(line).split("==")[0].split(" @ ")[0]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lock_block(entries: dict[str, tuple[str, str, str]]) -> str:
    lines = ["environment_locks:"]
    for key, (path, sha256, freeze_sha256) in entries.items():
        lines += [
            f"  {key}:",
            f'    path: "{path}"',
            f'    sha256: "{sha256}"',
            f'    freeze_sha256: "{freeze_sha256}"',
        ]
    return "\n".join(lines) + "\n"


def _resolve(
    profile, seeds: list[str], constraints: list[str], target: str
) -> tuple[list[str], list[str]]:
    """Resolve one target; drop a reference pin only when it has no target wheel."""

    relaxed: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        requirements = Path(temporary) / "requirements.in"
        requirements.write_text(
            "\n".join([profile.requirement, *profile.extra_requirements, *seeds])
            + "\n",
            encoding="utf-8",
        )
        constraint_file = Path(temporary) / "constraints.txt"
        output = Path(temporary) / "lock.txt"
        while True:
            kept = [line for line in constraints if _name(line) not in relaxed]
            constraint_file.write_text("\n".join(kept) + "\n", encoding="utf-8")
            process = subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    str(requirements),
                    "--constraint",
                    str(constraint_file),
                    "--python-platform",
                    target,
                    "--python-version",
                    profile.python,
                    "--exclude-newer",
                    CUTOFF,
                    *(
                        argument
                        for package, date in PACKAGE_CUTOFFS.items()
                        for argument in ("--exclude-newer-package", f"{package}={date}")
                    ),
                    "--no-header",
                    "--no-annotate",
                    "--quiet",
                    "--output-file",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            if process.returncode == 0:
                return _lines(output), relaxed
            missing = re.search(
                r"Because ([A-Za-z0-9_.-]+)==\S+ has no wheels with a matching platform",
                process.stderr,
            )
            if missing is None or _name(missing.group(1)) in relaxed:
                raise RuntimeError(f"{profile.name} on {target}: {process.stderr}")
            relaxed.append(_name(missing.group(1)))


def compile_locks(write: bool) -> None:
    root = repo_root()
    for name, profile in sorted(load_runtime_profiles().items()):
        reference = next(
            lock
            for lock in profile.environment_locks
            if lock.platform_key == REFERENCE_KEY
        )
        reference_lines = _lines(root / reference.path)
        seeds = [line for line in reference_lines if _name(line) in SEEDS]
        constraints = [
            line
            for line in reference_lines
            if _name(line) not in SEEDS and "==" in line and " @ " not in line
        ]
        entries = {}
        for key in [REFERENCE_KEY, *TARGETS]:
            relative = f"reproducibility/environment-locks/{key.lower()}/{name}.txt"
            path = root / relative
            if key == REFERENCE_KEY:
                text = path.read_text(encoding="utf-8")
            else:
                resolved, relaxed = _resolve(profile, seeds, constraints, TARGETS[key])
                header = [
                    f"# {package}: the reference version has no {key} wheel"
                    for package in relaxed
                ]
                body = sorted(resolved, key=str.lower)
                text = "\n".join([*header, *body]) + "\n"
                if write:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text, encoding="utf-8", newline="\n")
            canonical = sorted(canonical_requirement(line) for line in _lines_of(text))
            entries[key] = (relative, _sha256(text), _sha256("\n".join(canonical)))
            print(f"{name:12s} {key:14s} {len(canonical):4d} packages")
        if write:
            yaml_path = root / "configs" / "adapters" / f"{name}.yaml"
            source = yaml_path.read_text(encoding="utf-8")
            head = source[: source.index("environment_locks:")]
            yaml_path.write_text(
                head + _lock_block(entries), encoding="utf-8", newline="\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write locks and profiles")
    compile_locks(parser.parse_args().write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
