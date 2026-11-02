from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bench.benchmark_ledger import (
    BenchmarkLedger,
    RunLock,
    RunLocked,
    RunIntegrityError,
    RunSpecMismatch,
    atomic_write_json,
)
from bench.benchmark_models import (
    STATUS_INTERRUPTED,
    STATUS_MEASURED,
    RunSpec,
    TaskSpec,
    task_plan_fingerprint,
)


def make_task(*, bins: int = 32) -> TaskSpec:
    return TaskSpec(
        ordinal=1,
        case_id="case-1",
        dataset="fake-v1",
        modality="synthetic",
        size=10,
        variant=1,
        mask_id="M1",
        mask_label="mask",
        image_path="/tmp/image.nii.gz",
        source_image_path="/tmp/image.nii.gz",
        mask_path="/tmp/mask.nii.gz",
        image_sha256="1" * 64,
        source_image_sha256="1" * 64,
        mask_sha256="2" * 64,
        shape=(10, 10, 10),
        spacing=(1.0, 1.0, 1.0),
        image_voxels=1000,
        mask_voxels=500,
        mask_fraction=0.5,
        complexity=1000,
        subject_id="subject-1",
        input_contract="adapter_internal_preprocessing",
        representation_id="single_image_internal_preprocessing",
        representation_derivation_sha256=None,
        configured_levels=None,
        occupied_levels=None,
        adapter="fake",
        category="texture",
        family="glcm",
        repeat=1,
        discretization="fbn",
        bins=bins,
        bin_width=32.0,
        intensity_min=None,
        intensity_max=None,
        timing_iterations=2,
        no_warmup=False,
        dropped=False,
    )


def make_run_spec(task: TaskSpec) -> RunSpec:
    return RunSpec.create(
        run_id="unit-run",
        dataset="fake-v1",
        dataset_kind="synthetic",
        dataset_manifest_schema_version=2,
        dataset_dir="/tmp/fake-v1",
        manifest_sha256="manifest",
        dataset_hashes_verified=True,
        dataset_values_inspected=True,
        selected_case_ids=("case-1",),
        adapters=("fake",),
        workloads=("family:glcm",),
        repeats=1,
        aggregation="3d_merge",
        input_contract="adapter_internal_preprocessing",
        discretization="fbn",
        bins=task.bins,
        bin_width=32.0,
        intensity_min=None,
        intensity_max=None,
        timing_iterations=2,
        capture_values=True,
        drop_first_repeat=False,
        no_warmup=False,
        timeout_seconds=None,
        keep_going=True,
        task_plan_sha256=task_plan_fingerprint([task]),
        runtime_profiles_sha256="profiles",
        benchmark_sources_sha256="sources",
        benchmark_machine={"machine_id": "unit-machine"},
        adapter_environments={"fake": {"version": "1.0"}},
        thread_policy={
            "mode": "single_thread",
            "environment": {"OMP_NUM_THREADS": "1"},
        },
        initialization_policy={"environment": {"PICTOLOGICS_DISABLE_WARMUP": "1"}},
        guardrail={"enabled": True, "skip_ratio": 10.0},
    )


class BenchmarkLedgerTests(unittest.TestCase):
    def test_run_local_payload_paths_are_relocatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "machine" / "pillar"
            payload = run_dir / "records" / "case" / "result.json"
            payload.parent.mkdir(parents=True)
            payload.write_text("{}\n", encoding="utf-8")
            with BenchmarkLedger(run_dir / "benchmark.sqlite3") as ledger:
                encoded = ledger.portable_payload_path(payload)
                self.assertEqual(encoded, "records/case/result.json")
                self.assertEqual(ledger.resolve_payload_path(encoded), payload)

    def test_repeat_horizon_extends_append_only_without_changing_protocol_id(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            first_task = make_task()
            first_spec = make_run_spec(first_task)
            second_task = replace(first_task, ordinal=2, repeat=2)
            extended_spec = replace(
                first_spec,
                repeats=2,
                task_plan_sha256=task_plan_fingerprint([first_task, second_task]),
            )
            self.assertEqual(first_spec.run_fingerprint, extended_spec.run_fingerprint)
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(first_spec, [first_task], resume=False)
                ledger.initialize(extended_spec, [first_task, second_task], resume=True)
                self.assertEqual(len(ledger.task_rows()), 2)
                self.assertEqual(
                    int(ledger.metadata_value("run_spec_json") is not None), 1
                )

    def test_resume_rejects_changed_immutable_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            first_task = make_task(bins=32)
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(make_run_spec(first_task), [first_task], resume=False)

            changed_task = make_task(bins=64)
            with BenchmarkLedger(path) as ledger:
                with self.assertRaises(RunSpecMismatch):
                    ledger.initialize(
                        make_run_spec(changed_task), [changed_task], resume=True
                    )

    def test_running_attempt_is_recovered_as_interrupted_and_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            task = make_task()
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(make_run_spec(task), [task], resume=False)
                ledger.mark_running(task.task_id)

            with BenchmarkLedger(path) as ledger:
                self.assertEqual(ledger.recover_running(), 1)
                self.assertEqual(ledger.status(task.task_id), STATUS_INTERRUPTED)
                self.assertEqual(ledger.mark_running(task.task_id), 2)

    def test_run_lock_rejects_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".benchmark.lock"
            first = RunLock(path).acquire()
            try:
                with self.assertRaises(RunLocked):
                    RunLock(path).acquire()
            finally:
                first.release()

    def test_report_loader_rejects_payload_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger_path = root / "ledger.sqlite3"
            payload_path = root / "payload.json"
            task = make_task()
            run_spec = make_run_spec(task)
            record = {
                "task_id": task.task_id,
                "task_status": STATUS_MEASURED,
            }
            with BenchmarkLedger(ledger_path) as ledger:
                ledger.initialize(run_spec, [task], resume=False)
                ledger.mark_running(task.task_id)
                digest = atomic_write_json(
                    payload_path,
                    {
                        "benchmark": {
                            "task_id": task.task_id,
                            "run_fingerprint": run_spec.run_fingerprint,
                            "status": STATUS_MEASURED,
                            "record": record,
                        }
                    },
                )
                ledger.mark_terminal(
                    task.task_id,
                    STATUS_MEASURED,
                    record,
                    duration_sec=1.0,
                    payload_path=ledger.portable_payload_path(payload_path),
                    payload_sha256=digest,
                )
                records, payloads = ledger.verified_records_and_payloads()
                self.assertEqual(len(records), 1)
                self.assertEqual({key: records[0][key] for key in record}, record)
                self.assertEqual(records[0]["ordinal"], task.ordinal)
                self.assertIsNotNone(records[0]["started_at"])
                self.assertEqual(len(payloads), 1)

            payload_path.write_text("{}\n", encoding="utf-8")
            with BenchmarkLedger(ledger_path) as ledger:
                with self.assertRaisesRegex(
                    RunIntegrityError,
                    "checksum changed",
                ):
                    ledger.verified_records_and_payloads()


if __name__ == "__main__":
    unittest.main()
