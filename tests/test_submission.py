from __future__ import annotations

import gzip
import hashlib
import json
import socket
import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench.env import native_platform_key
from bench.reproduce.host_profile import write_host_profile
from bench.submission import package, validate

LOCK_FILE = (
    f"reproducibility/environment-locks/{native_platform_key().lower()}/fake.txt"
)
ENVIRONMENT_LOCK = {"path": LOCK_FILE, "sha256": "e" * 64, "freeze_sha256": "d" * 64}


def _ledger(run_dir: Path, records: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    with closing(sqlite3.connect(run_dir / "benchmark.sqlite3")) as database:
        database.execute(
            "CREATE TABLE tasks (task_id TEXT, ordinal INTEGER, status TEXT, "
            "attempt INTEGER, payload_sha256 TEXT, record_json TEXT)"
        )
        database.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        for ordinal, record in enumerate(records):
            database.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"task-{ordinal}",
                    ordinal,
                    "measured",
                    1,
                    "a" * 64,
                    json.dumps(record),
                ),
            )
        database.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("run_status", "completed"), ("run_fingerprint", "f" * 64)],
        )
        database.commit()


class SubmissionPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / "repo"
        (self.repo / "reproducibility").mkdir(parents=True)
        (self.repo / "submissions").mkdir()
        self.write_lock({"configs/benchmark/main.json": "c" * 64, LOCK_FILE: "e" * 64})
        self.result_root = self.root / "results"
        self.machine_root = self.result_root / "machine-a"
        write_host_profile(self.machine_root / "host_profile.json", "machine-a")
        personal = f"{Path.home()}/scratch on {socket.gethostname()}"
        for pillar in package.PILLARS:
            run_dir = self.machine_root / pillar
            _ledger(
                run_dir,
                [
                    {
                        "task_status": "measured",
                        "adapter": "fake",
                        "workload": "group_texture",
                        "endpoint_contract_sha256": "c" * 64,
                        "duration_median_sec": 1.0,
                        "cpu_time_median_sec": 0.5,
                        "image_path": str(run_dir / "inputs" / "image.nii.gz"),
                        "adapter_stderr": personal,
                    }
                ]
                * 2,
            )
            (run_dir / "run_spec.json").write_text(
                json.dumps({"run_id": pillar, "dataset_dir": personal}),
                encoding="utf-8",
            )
        report = self.machine_root / "compliance" / "ibsi1" / "report"
        report.mkdir(parents=True)
        (report / "compliance_summary.csv").write_text("adapter,passed\nmirp,169\n")
        environment = self.root / "env"
        environment.mkdir()
        self.environment_record = environment / "environment.json"
        self.environment_record.write_text(
            json.dumps(
                {
                    "python_executable": f"{Path.home()}/bin/python",
                    "environment_lock": ENVIRONMENT_LOCK,
                    "freeze_sha256": "d" * 64,
                }
            ),
            encoding="utf-8",
        )
        for patcher in (
            mock.patch.object(package, "REPO", self.repo),
            mock.patch.object(
                package,
                "load_runtime_profiles",
                return_value={"fake": SimpleNamespace()},
            ),
            mock.patch.object(package, "env_dir_for_profile", return_value=environment),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_lock(self, files: dict[str, str]) -> None:
        lock = self.repo / "reproducibility" / "release-lock.json"
        lock.write_text(
            json.dumps(
                {
                    "files": files,
                    "workspace": {"scheduled_tasks": dict.fromkeys(package.PILLARS, 2)},
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        (self.repo / "submissions" / "accepted_releases.json").write_text(
            json.dumps({"release_lock_sha256": [digest]}), encoding="utf-8"
        )

    def test_a_package_is_complete_consistent_and_free_of_personal_data(self) -> None:
        folder = package.build_package(self.machine_root, self.result_root)

        self.assertEqual(folder.parent.name, "machine-a")
        self.assertEqual(validate.validate(self.repo, folder), [])
        text = "".join(
            (
                gzip.decompress(path.read_bytes())
                if path.suffix == ".gz"
                else path.read_bytes()
            ).decode()
            for path in folder.rglob("*")
            if path.is_file()
        )
        self.assertNotIn(str(Path.home()), text)
        self.assertNotIn(socket.gethostname(), text)
        self.assertIn("{RESULT_ROOT}", text)
        self.assertFalse((folder / "pillar1_morphology" / "benchmark.sqlite3").exists())

    def test_the_check_finds_a_changed_byte_an_extra_file_and_a_personal_path(
        self,
    ) -> None:
        folder = package.build_package(self.machine_root, self.result_root)
        (folder / "extra.txt").write_text("/Users/someone/notes\n", encoding="utf-8")
        host = folder / "host_profile.json"
        host.write_bytes(host.read_bytes() + b" ")

        problems = "\n".join(validate.validate(self.repo, folder))
        self.assertIn("SHA256SUMS does not list exactly the files", problems)
        self.assertIn("checksum mismatch: host_profile.json", problems)
        self.assertIn("personal data '/Users/someone'", problems)

    def test_the_check_needs_main_run_records_and_locked_environments(self) -> None:
        self.write_lock({"configs/benchmark/main.json": "0" * 64, LOCK_FILE: "e" * 64})
        self.environment_record.write_text(
            json.dumps(
                {"environment_lock": ENVIRONMENT_LOCK, "freeze_sha256": "1" * 64}
            ),
            encoding="utf-8",
        )
        folder = package.build_package(self.machine_root, self.result_root)

        problems = "\n".join(validate.validate(self.repo, folder))
        self.assertIn("pillar1_morphology: 2 tasks are not main-run records", problems)
        self.assertIn("environment/fake.json: it differs from its lock", problems)

    def test_an_unfinished_run_is_never_packaged(self) -> None:
        with closing(
            sqlite3.connect(
                self.machine_root / "pillar2_whole_anatomy" / "benchmark.sqlite3"
            )
        ) as database:
            database.execute(
                "UPDATE metadata SET value = 'interrupted' WHERE key = 'run_status'"
            )
            database.commit()
        with self.assertRaisesRegex(package.PackageError, "not complete"):
            package.build_package(self.machine_root, self.result_root)


if __name__ == "__main__":
    unittest.main()
