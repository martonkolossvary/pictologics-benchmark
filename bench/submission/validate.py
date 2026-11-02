"""Check a submitted result package offline.

Usage: ``python -I bench/submission/validate.py REPO SUBMISSION_FOLDER``

The checks need only the standard library, so CI runs this file from the main
branch on the files of a pull request.  Hashes prove that a package is
complete and consistent; they do not prove that its timings are genuine.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

MACHINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LIMIT_BYTES = 45 * 1024**2
STATUSES = {
    "measured",
    "unsupported",
    "timed_out_censored",
    "memory_exceeded_censored",
    "skipped_memory_preflight",
    "skipped_policy",
    "failed",
}
FORBIDDEN_KEYS = {"serial", "serial_number", "hardware_uuid", "hostname", "udid"}
# Personal paths.  Wheel build machines leave /Users/runner in package metadata.
LEAKS = (
    re.compile(r"/Users/(?!runner/|\{)[^/\s\"']+"),
    re.compile(r"/home/(?!runner/|\{)[^/\s\"']+"),
    re.compile(r"[A-Za-z]:\\\\?Users\\\\?(?!\{)[^\\\s\"']+"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(path: Path) -> str:
    data = path.read_bytes()
    return (gzip.decompress(data) if path.suffix == ".gz" else data).decode("utf-8")


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return {str(key).casefold() for key in value} | {
            key for entry in value.values() for key in _keys(entry)
        }
    if isinstance(value, list):
        return {key for entry in value for key in _keys(entry)}
    return set()


def validate(repo: Path, folder: Path) -> list[str]:
    problems: list[str] = []
    if not (MACHINE_ID.match(folder.parent.name) and DATE.match(folder.name)):
        return [f"{folder} is not submissions/<machine-id>/<YYYY-MM-DD>"]
    files = sorted(
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file()
    )
    total = sum((folder / name).stat().st_size for name in files)
    if total > LIMIT_BYTES:
        problems.append(f"the package is {total / 1024**2:.1f} MiB; the limit is 45")
    sums = {}
    for line in (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name] = digest
    if set(sums) != set(files) - {"SHA256SUMS"}:
        problems.append("SHA256SUMS does not list exactly the files of the package")
    problems += [
        f"checksum mismatch: {name}"
        for name, digest in sums.items()
        if (folder / name).is_file() and _sha256(folder / name) != digest
    ]
    submission = json.loads((folder / "submission.json").read_text(encoding="utf-8"))
    if submission.get("schema_version") != 1:
        problems.append("submission.json has an unknown schema")
    if submission.get("machine_id") != folder.parent.name:
        problems.append("the machine ID differs from the folder name")
    accepted = json.loads(
        (repo / "submissions" / "accepted_releases.json").read_text(encoding="utf-8")
    )
    release = submission.get("release_lock_sha256")
    if release not in accepted["release_lock_sha256"]:
        problems.append("the package comes from a release that is not accepted")
    lock_path = repo / "reproducibility" / "release-lock.json"
    current = _sha256(lock_path) == release
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    host = json.loads((folder / "host_profile.json").read_text(encoding="utf-8"))
    if _keys(host) & FORBIDDEN_KEYS:
        problems.append("the host profile holds a private field")
    if host.get("expected_hardware") != submission.get("hardware"):
        problems.append("the host profile and the submission disagree on the hardware")
    logical = int(submission.get("hardware", {}).get("cpu_count_logical") or 1)
    for pillar, summary in submission.get("pillars", {}).items():
        rows = [
            json.loads(line)
            for line in _text(folder / pillar / "tasks.jsonl.gz").splitlines()
            if line
        ]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if row["status"] not in STATUSES:
                problems.append(f"{pillar}: unknown status {row['status']!r}")
            if row["status"] != "measured":
                continue
            record = row["record"] or {}
            wall = float(record.get("duration_median_sec") or 0)
            cpu = float(record.get("cpu_time_median_sec") or 0)
            if not row["payload_sha256"] or record.get("task_status") != "measured":
                problems.append(
                    f"{pillar}: measured task {row['task_id']} is incomplete"
                )
            elif wall <= 0 or cpu > wall * (logical + 1):
                problems.append(
                    f"{pillar}: task {row['task_id']} has implausible times"
                )
        if counts != summary.get("status_counts"):
            problems.append(
                f"{pillar}: the status counts disagree with submission.json"
            )
        if current and len(rows) != lock["workspace"]["scheduled_tasks"][pillar]:
            problems.append(f"{pillar}: {len(rows)} tasks; the release schedules more")
        if summary.get("run_status") not in {"completed", "completed_with_failures"}:
            problems.append(f"{pillar}: the run is not complete")
    for name in files:
        for pattern in LEAKS:
            match = pattern.search(_text(folder / name))
            if match:
                problems.append(f"{name}: personal data {match.group(0)!r}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    problems = validate(Path(argv[0]).resolve(), Path(argv[1]).resolve())
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print("The package is complete and consistent.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
