"""Build the compact result package of one finished main run, free of personal data.

The package holds the task records of the three pillars, the run
specifications, the environment freezes, the host profile and the IBSI
conformance tables.  It leaves out the staged input copies, the raw payload
files, the SQLite ledger itself and the run lock.
"""

from __future__ import annotations

import getpass
import gzip
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from bench.benchmark_ledger import RunLock, sha256_file
from bench.env import env_dir_for_profile, load_runtime_profiles, native_platform_key

REPO = Path(__file__).resolve().parents[2]
PILLARS = ("pillar1_morphology", "pillar2_whole_anatomy", "pillar3_ibsi2_phase3")
COMPLIANCE = (
    "ibsi1",
    "ibsi2-phase1",
    "ibsi2-phase2",
    "ibsi2-phase1-native-default",
    "ibsi2-phase2-native-default",
)
PACKAGE_LIMIT_BYTES = 45 * 1024**2
STDERR_KEEP_CHARACTERS = 4096


class PackageError(RuntimeError):
    pass


def _uv_python_dir() -> Path:
    configured = os.environ.get("UV_PYTHON_INSTALL_DIR")
    if configured:
        return Path(configured)
    if platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home())) / "uv" / "python"
    return Path.home() / ".local" / "share" / "uv" / "python"


class Scrubber:
    """Replace this computer's paths, user name and host names with tokens."""

    def __init__(self, result_root: Path):
        tokens = {
            "{RESULT_ROOT}": result_root,
            "{REPO}": REPO,
            "{UV_PYTHON}": _uv_python_dir(),
            "{TMP}": Path(tempfile.gettempdir()),
            "{HOME}": Path.home(),
        }
        forms: dict[str, str] = {}
        for token, path in tokens.items():
            for candidate in (path, path.resolve()):
                for text in (str(candidate), candidate.as_posix()):
                    if len(text) > 3:
                        forms[text] = token
                        forms[text.replace("\\", "\\\\")] = token
        # The longest form first, so a deeper path keeps its own token.
        self.paths = sorted(forms.items(), key=lambda item: len(item[0]), reverse=True)
        names = {getpass.getuser(): "{USER}"}
        for host in (socket.gethostname(), platform.node()):
            if host:
                names[host] = "{HOST}"
                names[host.split(".")[0]] = "{HOST}"
        self.names = {name: token for name, token in names.items() if len(name) > 2}
        self.counts: Counter[str] = Counter()

    def text(self, value: str) -> str:
        for form, token in self.paths:
            if form in value:
                self.counts[token] += value.count(form)
                value = value.replace(form, token)
        for name, token in self.names.items():
            value, replaced = re.subn(
                rf"(?<![\w.-]){re.escape(name)}(?![\w-])", token, value, flags=re.I
            )
            self.counts[token] += replaced
        return value

    def value(self, item: Any, key: str = "") -> Any:
        if isinstance(item, dict):
            return {name: self.value(entry, name) for name, entry in item.items()}
        if isinstance(item, list):
            return [self.value(entry, key) for entry in item]
        if isinstance(item, str):
            if key == "adapter_stderr":
                item = item[-STDERR_KEEP_CHARACTERS:]
            return self.text(item)
        return item

    def leaks(self, text: str) -> list[str]:
        found = [form for form, _ in self.paths if form in text]
        found += [
            name
            for name in self.names
            if re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", text, flags=re.I)
        ]
        return found


def _ledger(run_dir: Path) -> tuple[list[tuple], dict[str, str]]:
    """Copy the ledger with the SQLite backup API, so the WAL is included."""

    lock = RunLock(run_dir / ".benchmark.lock").acquire()
    try:
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "ledger.sqlite3"
            source = sqlite3.connect(
                f"file:{run_dir / 'benchmark.sqlite3'}?mode=ro", uri=True
            )
            target = sqlite3.connect(copy)
            try:
                source.backup(target)
                if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise PackageError(f"the ledger copy of {run_dir} is damaged")
                rows = target.execute(
                    "SELECT task_id, ordinal, status, attempt, payload_sha256, "
                    "record_json FROM tasks ORDER BY ordinal"
                ).fetchall()
                metadata = dict(target.execute("SELECT key, value FROM metadata"))
            finally:
                source.close()
                target.close()
    finally:
        lock.release()
    return rows, metadata


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as stream:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=9
            ) as archive:
                archive.write(data)
    else:
        path.write_bytes(data)


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def build_package(machine_root: Path, result_root: Path) -> Path:
    """Write ``<machine_root>/submissions/<machine-id>/<UTC date>/``; return it."""

    host_profile = json.loads((machine_root / "host_profile.json").read_text("utf-8"))
    machine_id = host_profile["machine_id"]
    date = time.strftime("%Y-%m-%d", time.gmtime())
    output = machine_root / "submissions" / machine_id / date
    if output.exists():
        raise PackageError(f"the package already exists: {output}")
    scrub = Scrubber(result_root)
    pillars: dict[str, Any] = {}
    staging = output.with_name(f".{date}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _write(staging / "host_profile.json", _json(host_profile))
        for pillar in PILLARS:
            run_dir = machine_root / pillar
            rows, metadata = _ledger(run_dir)
            if metadata.get("run_status") not in {
                "completed",
                "completed_with_failures",
            }:
                raise PackageError(f"{pillar} is not complete; finish the run first")
            lines = [
                json.dumps(
                    {
                        "task_id": task_id,
                        "ordinal": ordinal,
                        "status": status,
                        "attempt": attempt,
                        "payload_sha256": payload_sha256,
                        "record": scrub.value(json.loads(record) if record else None),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                for task_id, ordinal, status, attempt, payload_sha256, record in rows
            ]
            _write(
                staging / pillar / "tasks.jsonl.gz", ("\n".join(lines) + "\n").encode()
            )
            run_spec = json.loads((run_dir / "run_spec.json").read_text("utf-8"))
            _write(staging / pillar / "run_spec.json", _json(scrub.value(run_spec)))
            pillars[pillar] = {
                "run_id": metadata.get("run_id") or run_spec.get("run_id"),
                "run_fingerprint": metadata.get("run_fingerprint"),
                "run_status": metadata.get("run_status"),
                "status_counts": dict(Counter(row[2] for row in rows)),
            }
        for name, profile in sorted(load_runtime_profiles().items()):
            environment = env_dir_for_profile(profile) / "environment.json"
            _write(
                staging / "environment" / f"{name}.json",
                _json(scrub.value(json.loads(environment.read_text("utf-8")))),
            )
        for name in COMPLIANCE:
            for table in sorted(
                (machine_root / "compliance" / name / "report").glob("*.csv")
            ):
                _write(
                    staging / "compliance" / name / f"{table.name}.gz",
                    scrub.text(table.read_text("utf-8")).encode(),
                )
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        lock = REPO / "reproducibility" / "release-lock.json"
        submission = {
            "schema_version": 1,
            "machine_id": machine_id,
            "created_utc_date": date,
            "release_lock_sha256": sha256_file(lock),
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
            ).stdout.strip(),
            "platform_key": native_platform_key(),
            "hardware": host_profile["expected_hardware"],
            "pillars": pillars,
            "scrubbed_values": dict(sorted(scrub.counts.items())),
            "files": [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
        }
        _write(staging / "submission.json", _json(submission))
        files.append(staging / "submission.json")
        sums = "".join(
            f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
            for path in sorted(files)
        )
        _write(staging / "SHA256SUMS", sums.encode())
        total = sum(
            path.stat().st_size for path in staging.rglob("*") if path.is_file()
        )
        if total > PACKAGE_LIMIT_BYTES:
            raise PackageError(
                f"the package is {total / 1024**2:.1f} MiB; the limit is 45"
            )
        leaks = []
        for path in staging.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                text = (gzip.decompress(data) if path.suffix == ".gz" else data).decode(
                    "utf-8", "replace"
                )
                leaks += [f"{path.name}: {leak}" for leak in scrub.leaks(text)]
        if leaks:
            raise PackageError(
                "personal data remains:\n  " + "\n  ".join(sorted(set(leaks)))
            )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
