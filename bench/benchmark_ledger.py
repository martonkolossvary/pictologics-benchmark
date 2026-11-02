from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from bench.benchmark_models import (
    ALL_STATUSES,
    RECOVERABLE_STATUSES,
    STATUS_INTERRUPTED,
    STATUS_MEASURED,
    STATUS_PENDING,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
    RunSpec,
    TaskSpec,
    canonical_json,
)


class RunSpecMismatch(RuntimeError):
    pass


class RunAlreadyExists(RuntimeError):
    pass


class RunLocked(RuntimeError):
    pass


class RunIntegrityError(RuntimeError):
    pass


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


def atomic_write_bytes(path: Path, data: bytes) -> str:
    """Durably replace *path* and return the SHA-256 of the committed bytes."""
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(data).hexdigest()


def atomic_write_text(path: Path, text: str) -> str:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
    )
    return atomic_write_text(path, data + "\n")


class RunLock:
    """Cross-platform advisory single-writer lock for a benchmark run directory."""

    def __init__(self, path: Path):
        self.path = path
        self._stream = None

    def acquire(self) -> "RunLock":
        if self._stream is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    # Another holder's lock makes even the read fail.
                    stream.seek(0)
                    if not stream.read(1):
                        stream.seek(0)
                        stream.write("\0")
                        stream.flush()
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RunLocked(
                        f"benchmark run is already locked: {self.path}"
                    ) from exc
            else:
                import fcntl

                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RunLocked(
                        f"benchmark run is already locked: {self.path}"
                    ) from exc

            stream.seek(0)
            stream.truncate()
            stream.write(
                canonical_json(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                        "acquired_at": time.time(),
                    }
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
            self._stream = stream
            return self
        except BaseException:
            stream.close()
            raise

    def release(self) -> None:
        if self._stream is None:
            return
        stream = self._stream
        self._stream = None
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def __del__(self) -> None:
        # Planning errors can occur after an early lock acquisition but before
        # the execution context is entered. Never leave that advisory lock held
        # for the lifetime of a long-lived embedding process.
        try:
            self.release()
        except Exception:
            pass


class BenchmarkLedger:
    """Transactional execution state. SQLite, not report files, is authoritative."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def portable_payload_path(self, path: Path) -> str:
        """Encode a run-local payload relative to the relocatable run root."""

        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.path.parent.resolve())
        except ValueError as exc:
            raise RunIntegrityError(
                f"payload is outside the run directory: {resolved}"
            ) from exc
        return relative.as_posix()

    def resolve_payload_path(self, value: str) -> Path:
        """Resolve a ledger payload path relative to the run directory."""

        path = Path(value)
        if path.is_absolute():
            raise RunIntegrityError(f"ledger payload path is not relative: {value}")
        return self.path.parent / path

    def _create_schema(self) -> None:
        status_values = ",".join(f"'{status}'" for status in sorted(ALL_STATUSES))
        self.connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL UNIQUE,
                spec_json TEXT NOT NULL,
                case_id TEXT NOT NULL,
                adapter TEXT NOT NULL,
                workload_key TEXT NOT NULL,
                repeat INTEGER NOT NULL,
                complexity INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ({status_values})),
                attempt INTEGER NOT NULL DEFAULT 0,
                started_at REAL,
                finished_at REAL,
                duration_sec REAL,
                payload_path TEXT,
                payload_sha256 TEXT,
                record_json TEXT,
                error TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_match
            ON tasks(case_id, adapter, workload_key, repeat, status);

            CREATE INDEX IF NOT EXISTS idx_tasks_status
            ON tasks(status, ordinal);

            CREATE TABLE IF NOT EXISTS task_attempts (
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ({status_values})),
                started_at REAL,
                finished_at REAL,
                error TEXT,
                PRIMARY KEY(task_id, attempt)
            );

            CREATE TABLE IF NOT EXISTS guardrail_observations (
                task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
                scope_key TEXT NOT NULL,
                baseline_task_id TEXT REFERENCES tasks(task_id),
                complexity INTEGER NOT NULL,
                ratio REAL NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_guardrail_observations_scope
            ON guardrail_observations(scope_key, ratio);

            CREATE TABLE IF NOT EXISTS guardrail_decisions (
                scope_key TEXT PRIMARY KEY,
                adapter TEXT NOT NULL,
                workload_key TEXT NOT NULL,
                guardrail_group TEXT NOT NULL,
                cutoff_complexity INTEGER NOT NULL,
                reason TEXT NOT NULL,
                evidence_task_id TEXT REFERENCES tasks(task_id),
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                event_type TEXT NOT NULL,
                task_id TEXT,
                details_json TEXT
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "BenchmarkLedger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _metadata(self, key: str) -> Optional[str]:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def set_metadata(self, key: str, value: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def initialize(
        self,
        run_spec: RunSpec,
        tasks: Iterable[TaskSpec],
        *,
        resume: bool,
    ) -> int:
        existing = self._metadata("run_fingerprint")
        expected = run_spec.run_fingerprint
        stored_spec_json = self._metadata("run_spec_json")
        stored_repeats = 0
        if stored_spec_json is not None:
            try:
                stored_repeats = int(json.loads(stored_spec_json).get("repeats", 0))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RunIntegrityError(
                    "stored run specification is malformed"
                ) from exc
        if existing is not None and not resume:
            raise RunAlreadyExists(
                f"run ledger already exists at {self.path}; pass --resume to continue it"
            )
        if existing is not None and existing != expected:
            raise RunSpecMismatch(
                "resume rejected because the immutable run specification changed: "
                f"stored={existing}, requested={expected}"
            )
        if existing is not None and run_spec.repeats < stored_repeats:
            raise RunSpecMismatch(
                "repeat horizon cannot shrink on resume: "
                f"stored={stored_repeats}, requested={run_spec.repeats}"
            )

        task_list = list(tasks)
        now = time.time()
        with self.connection:
            if existing is None:
                values = {
                    "run_fingerprint": expected,
                    "run_spec_json": canonical_json(run_spec.to_dict()),
                    "run_status": "created",
                    "created_at": str(now),
                }
                self.connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES(?, ?)", values.items()
                )

            for task in task_list:
                task_dict = task.to_dict()
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO tasks(
                        task_id, ordinal, spec_json, case_id, adapter, workload_key,
                        repeat, complexity, status
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.task_id,
                        task.ordinal,
                        canonical_json(task_dict),
                        task.case_id,
                        task.adapter,
                        task.workload_key,
                        task.repeat,
                        task.complexity,
                        STATUS_PENDING,
                    ),
                )

            rows = self.connection.execute(
                "SELECT task_id, spec_json FROM tasks ORDER BY ordinal"
            ).fetchall()
            expected_specs = {
                task.task_id: canonical_json(task.to_dict()) for task in task_list
            }
            actual_specs = {str(row["task_id"]): str(row["spec_json"]) for row in rows}
            if actual_specs != expected_specs:
                raise RunSpecMismatch(
                    "stored task plan differs from the requested task plan"
                )

            if existing is not None:
                self.connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'run_spec_json'",
                    (canonical_json(run_spec.to_dict()),),
                )

            self.connection.execute(
                "INSERT INTO events(created_at, event_type, details_json) VALUES(?, ?, ?)",
                (
                    now,
                    (
                        "repeat_horizon_extended"
                        if existing is not None and run_spec.repeats > stored_repeats
                        else "resume"
                        if existing is not None
                        else "created"
                    ),
                    canonical_json(
                        {
                            "run_fingerprint": expected,
                            "tasks": len(task_list),
                            "previous_repeats": stored_repeats or None,
                            "requested_repeats": run_spec.repeats,
                            "task_plan_sha256": run_spec.task_plan_sha256,
                        }
                    ),
                ),
            )
        return len(task_list)

    def recover_running(self) -> int:
        now = time.time()
        with self.connection:
            rows = self.connection.execute(
                "SELECT task_id, attempt FROM tasks WHERE status = ?", (STATUS_RUNNING,)
            ).fetchall()
            for row in rows:
                task_id = str(row["task_id"])
                attempt = int(row["attempt"])
                message = "controller stopped before the running attempt committed"
                self.connection.execute(
                    """
                    UPDATE task_attempts
                    SET status = ?, finished_at = ?, error = ?
                    WHERE task_id = ? AND attempt = ?
                    """,
                    (STATUS_INTERRUPTED, now, message, task_id, attempt),
                )
                self.connection.execute(
                    """
                    UPDATE tasks
                    SET status = ?, finished_at = ?, error = ?
                    WHERE task_id = ?
                    """,
                    (STATUS_INTERRUPTED, now, message, task_id),
                )
                self.connection.execute(
                    "INSERT INTO events(created_at, event_type, task_id, details_json) "
                    "VALUES(?, ?, ?, ?)",
                    (
                        now,
                        "recovered_interrupted",
                        task_id,
                        canonical_json({"attempt": attempt}),
                    ),
                )
        return len(rows)

    def mark_running(self, task_id: str) -> int:
        now = time.time()
        with self.connection:
            row = self.connection.execute(
                "SELECT status, attempt FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if str(row["status"]) not in RECOVERABLE_STATUSES:
                raise RuntimeError(f"task {task_id} is not runnable: {row['status']}")
            attempt = int(row["attempt"]) + 1
            self.connection.execute(
                """
                UPDATE tasks
                SET status = ?, attempt = ?, started_at = ?, finished_at = NULL,
                    duration_sec = NULL, payload_path = NULL, payload_sha256 = NULL,
                    record_json = NULL, error = NULL
                WHERE task_id = ?
                """,
                (STATUS_RUNNING, attempt, now, task_id),
            )
            self.connection.execute(
                "INSERT INTO task_attempts(task_id, attempt, status, started_at) "
                "VALUES(?, ?, ?, ?)",
                (task_id, attempt, STATUS_RUNNING, now),
            )
            self.connection.execute(
                "INSERT INTO events(created_at, event_type, task_id, details_json) "
                "VALUES(?, ?, ?, ?)",
                (now, "task_started", task_id, canonical_json({"attempt": attempt})),
            )
        return attempt

    def mark_terminal(
        self,
        task_id: str,
        status: str,
        record: Mapping[str, Any],
        *,
        duration_sec: Optional[float] = None,
        payload_path: Optional[str] = None,
        payload_sha256: Optional[str] = None,
        error: Optional[str] = None,
        adopted: bool = False,
    ) -> None:
        if status not in TERMINAL_STATUSES and status != STATUS_INTERRUPTED:
            raise ValueError(f"not a terminal attempt status: {status}")
        now = time.time()
        record_json = canonical_json(dict(record))
        with self.connection:
            row = self.connection.execute(
                "SELECT status, attempt FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            current = str(row["status"])
            attempt = int(row["attempt"])
            if current in TERMINAL_STATUSES and not adopted:
                raise RuntimeError(f"task {task_id} is already terminal: {current}")

            if attempt == 0:
                self.connection.execute(
                    "INSERT INTO task_attempts(task_id, attempt, status, started_at, finished_at, error) "
                    "VALUES(?, 0, ?, NULL, ?, ?)",
                    (task_id, status, now, error),
                )
            else:
                self.connection.execute(
                    """
                    UPDATE task_attempts
                    SET status = ?, finished_at = ?, error = ?
                    WHERE task_id = ? AND attempt = ?
                    """,
                    (status, now, error, task_id, attempt),
                )

            self.connection.execute(
                """
                UPDATE tasks
                SET status = ?, finished_at = ?, duration_sec = ?, payload_path = ?,
                    payload_sha256 = ?, record_json = ?, error = ?
                WHERE task_id = ?
                """,
                (
                    status,
                    now,
                    duration_sec,
                    payload_path,
                    payload_sha256,
                    record_json,
                    error,
                    task_id,
                ),
            )
            self.connection.execute(
                "INSERT INTO events(created_at, event_type, task_id, details_json) "
                "VALUES(?, ?, ?, ?)",
                (
                    now,
                    "task_adopted" if adopted else "task_finished",
                    task_id,
                    canonical_json({"attempt": attempt, "status": status}),
                ),
            )

    def task_rows(self) -> List[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM tasks ORDER BY ordinal"
        ).fetchall()

    def task_row(self, task_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    def status(self, task_id: str) -> str:
        return str(self.task_row(task_id)["status"])

    def records(self) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT record_json FROM tasks WHERE record_json IS NOT NULL ORDER BY ordinal"
        ).fetchall()
        return [json.loads(str(row["record_json"])) for row in rows]

    def metadata_value(self, key: str) -> Optional[str]:
        """Return immutable/run-state metadata for read-only report consumers."""

        return self._metadata(key)

    def verified_records_and_payloads(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load report inputs from the ledger and verify every measured payload.

        The SQLite ledger is authoritative.  Derived CSV files and directory
        globs are intentionally ignored so edits, stale files, or partial
        report writes cannot silently change a publication table.
        """

        run_fingerprint = self._metadata("run_fingerprint")
        if not run_fingerprint:
            raise RunIntegrityError("benchmark ledger has no run fingerprint")

        records: List[Dict[str, Any]] = []
        payloads: List[Dict[str, Any]] = []
        for row in self.task_rows():
            task_id = str(row["task_id"])
            status = str(row["status"])
            record_json = row["record_json"]
            if record_json is None:
                if status in RECOVERABLE_STATUSES or status == STATUS_RUNNING:
                    continue
                raise RunIntegrityError(
                    f"terminal task {task_id} has no authoritative record"
                )
            try:
                record = json.loads(str(record_json))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RunIntegrityError(
                    f"task {task_id} has malformed record JSON"
                ) from exc
            if (
                not isinstance(record, dict)
                or record.get("task_id") != task_id
                or record.get("task_status") != status
            ):
                raise RunIntegrityError(
                    f"task {task_id} record identity/status differs from the ledger"
                )
            # The run order and clock times let a later analysis test order
            # and drift effects.
            records.append(
                {
                    **record,
                    "ordinal": int(row["ordinal"]),
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                }
            )

            if status != STATUS_MEASURED:
                continue
            path_value = str(row["payload_path"] or "")
            expected_digest = str(row["payload_sha256"] or "")
            if not path_value or len(expected_digest) != 64:
                raise RunIntegrityError(
                    f"measured task {task_id} has incomplete payload metadata"
                )
            path = self.resolve_payload_path(path_value)
            if not path.is_file():
                raise RunIntegrityError(
                    f"measured task {task_id} payload is missing: {path}"
                )
            if sha256_file(path) != expected_digest:
                raise RunIntegrityError(
                    f"measured task {task_id} payload checksum changed: {path}"
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise RunIntegrityError(
                    f"measured task {task_id} payload is invalid JSON"
                ) from exc
            benchmark = payload.get("benchmark") if isinstance(payload, dict) else None
            if (
                not isinstance(benchmark, dict)
                or benchmark.get("task_id") != task_id
                or benchmark.get("run_fingerprint") != run_fingerprint
                or benchmark.get("status") != STATUS_MEASURED
                or benchmark.get("record") != record
            ):
                raise RunIntegrityError(
                    f"measured task {task_id} payload identity differs from the ledger"
                )
            payload["_verified_payload_path"] = str(path)
            payload["_verified_payload_sha256"] = expected_digest
            payloads.append(payload)
        return records, payloads

    def status_counts(self) -> Dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def measured_match(
        self,
        *,
        case_id: str,
        adapter: str,
        workload_key: str,
        repeat: int,
    ) -> Optional[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT task_id, duration_sec FROM tasks
            WHERE case_id = ? AND adapter = ? AND workload_key = ?
              AND repeat = ? AND status = ?
            """,
            (case_id, adapter, workload_key, repeat, STATUS_MEASURED),
        ).fetchone()

    def record_guardrail_observation(
        self,
        *,
        task_id: str,
        scope_key: str,
        baseline_task_id: str,
        complexity: int,
        ratio: float,
        slow_threshold: float,
    ) -> int:
        now = time.time()
        with self.connection:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO guardrail_observations(
                    task_id, scope_key, baseline_task_id, complexity, ratio, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (task_id, scope_key, baseline_task_id, complexity, ratio, now),
            )
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM guardrail_observations "
                "WHERE scope_key = ? AND ratio >= ?",
                (scope_key, slow_threshold),
            ).fetchone()
        return int(row["count"])

    def activate_guardrail(
        self,
        *,
        scope_key: str,
        adapter: str,
        workload_key: str,
        guardrail_group: str,
        cutoff_complexity: int,
        reason: str,
        evidence_task_id: str,
    ) -> None:
        now = time.time()
        with self.connection:
            existing = self.connection.execute(
                "SELECT cutoff_complexity FROM guardrail_decisions WHERE scope_key = ?",
                (scope_key,),
            ).fetchone()
            if (
                existing is not None
                and int(existing["cutoff_complexity"]) <= cutoff_complexity
            ):
                return
            self.connection.execute(
                """
                INSERT INTO guardrail_decisions(
                    scope_key, adapter, workload_key, guardrail_group,
                    cutoff_complexity, reason, evidence_task_id, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_key) DO UPDATE SET
                    cutoff_complexity = excluded.cutoff_complexity,
                    reason = excluded.reason,
                    evidence_task_id = excluded.evidence_task_id,
                    created_at = excluded.created_at
                """,
                (
                    scope_key,
                    adapter,
                    workload_key,
                    guardrail_group,
                    cutoff_complexity,
                    reason,
                    evidence_task_id,
                    now,
                ),
            )
            self.connection.execute(
                "INSERT INTO events(created_at, event_type, task_id, details_json) "
                "VALUES(?, ?, ?, ?)",
                (
                    now,
                    "guardrail_activated",
                    evidence_task_id,
                    canonical_json(
                        {
                            "scope_key": scope_key,
                            "cutoff_complexity": cutoff_complexity,
                            "reason": reason,
                        }
                    ),
                ),
            )

    def guardrail_decision(self, scope_key: str) -> Optional[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM guardrail_decisions WHERE scope_key = ?", (scope_key,)
        ).fetchone()

    def guardrail_decisions(self) -> List[Dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM guardrail_decisions ORDER BY scope_key"
        ).fetchall()
        return [dict(row) for row in rows]

    def set_run_status(self, status: str) -> None:
        now = time.time()
        with self.connection:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES('run_status', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (status,),
            )
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES('updated_at', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(now),),
            )
            self.connection.execute(
                "INSERT INTO events(created_at, event_type, details_json) VALUES(?, ?, ?)",
                (now, "run_status", canonical_json({"status": status})),
            )

    def record_event(self, event_type: str, details: Mapping[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO events(created_at, event_type, details_json) VALUES(?, ?, ?)",
                (time.time(), event_type, canonical_json(dict(details))),
            )

    def run_status(self) -> Optional[str]:
        return self._metadata("run_status")
