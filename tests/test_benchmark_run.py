from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import psutil

from bench import run
from bench.adapters.protocol import ADAPTER_PROTOCOL_VERSION
from bench.adapters.protocol import timing_contract_metadata
from bench.adapters.registry import get_adapter
from bench.benchmark_ledger import (
    BenchmarkLedger,
    RunAlreadyExists,
    RunIntegrityError,
    RunSpecMismatch,
)
from bench.benchmark_models import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_MEASURED,
    STATUS_MEMORY_EXCEEDED,
    STATUS_SKIPPED,
    STATUS_SKIPPED_MEMORY,
    STATUS_TIMED_OUT,
    STATUS_UNSUPPORTED,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(duration: float) -> dict:
    return {
        "duration_sec": duration,
        "duration_mean_sec": duration,
        "duration_median_sec": duration,
        "duration_std_sec": 0.0,
        "duration_max_sec": duration,
        "cpu_time_sec": duration,
        "cpu_time_mean_sec": duration,
        "cpu_time_median_sec": duration,
        "cpu_time_std_sec": 0.0,
        "cpu_time_max_sec": duration,
        "measured_iterations": 1,
        "total_iterations": 1,
        "peak_rss_bytes": 1024,
        "host_peak_rss_bytes": 1024,
        "worker_ready_rss_bytes": 768,
        "calculation_peak_rss_bytes": 1024,
        "incremental_calculation_peak_rss_bytes": 256,
        "host_wall_time_sec": duration,
        "adapter_event_count": 8,
        "adapter_stderr": "",
        "timing_source": "fake",
        "timing_scope": "fake",
        "memory_scope": "fake",
    }


def adapter_payload(
    adapter: str,
    family: str,
    feature_names: list[str] | None = None,
) -> dict:
    """Return a complete adapter payload for runner unit tests."""

    try:
        capabilities = get_adapter(adapter)
        distribution = capabilities.distribution
        selection_mode = capabilities.selection_mode
    except ValueError:
        distribution = f"test-{adapter}"
        selection_mode = "unit-test"
    return {
        "schema_version": ADAPTER_PROTOCOL_VERSION,
        "adapter": adapter,
        "software": {"distribution": distribution, "version": "test-1.0"},
        "selection": {
            "requested_families": [family],
            "unsupported_families": [],
            "mode": selection_mode,
        },
        "features": {"all": feature_names or ["feature"]},
    }


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.report = self.root / "run"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_process_event_stream_delimits_calculation_memory(self) -> None:
        script = "\n".join(
            [
                "import json, time",
                "held = bytearray(8 * 1024 * 1024)",
                "def event(name):",
                "    print('BENCH_EVENT ' + json.dumps({'event': name}), flush=True)",
                "event('worker_ready')",
                "time.sleep(0.05)",
                "event('warmup_complete')",
                "event('calculation_start')",
                "extra = bytearray(8 * 1024 * 1024)",
                "time.sleep(0.10)",
                "event('calculation_complete')",
                "print(json.dumps({'ok': True}), flush=True)",
            ]
        )
        stdout, stderr, observed = run._run_process_command(
            [sys.executable, "-c", script],
            adapter_name="event-fixture",
            sample_interval=0.005,
            timeout=5.0,
        )

        self.assertIn('"ok": true', stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(observed["adapter_event_count"], 4)
        self.assertGreater(observed["worker_ready_rss_bytes"], 0)
        self.assertGreaterEqual(
            observed["calculation_peak_rss_bytes"],
            observed["worker_ready_rss_bytes"],
        )
        self.assertEqual(
            observed["incremental_calculation_peak_rss_bytes"],
            observed["calculation_peak_rss_bytes"] - observed["worker_ready_rss_bytes"],
        )
        self.assertEqual(len(observed["calculation_peak_rss_samples_bytes"]), 1)
        self.assertGreaterEqual(
            observed["calculation_peak_rss_samples_bytes"][0],
            observed["worker_ready_rss_bytes"],
        )

    def test_process_progress_reports_phase_and_measured_iteration(self) -> None:
        script = "\n".join(
            [
                "import json, time",
                "def event(name, **extra):",
                "    print('BENCH_EVENT ' + json.dumps({'event': name, **extra}), flush=True)",
                "event('worker_ready')",
                "event('calculation_start', iteration=1)",
                "time.sleep(0.08)",
                "event('calculation_complete', iteration=1)",
                "print(json.dumps({'ok': True}), flush=True)",
            ]
        )
        snapshots = []
        run._run_process_command(
            [sys.executable, "-c", script],
            adapter_name="progress-fixture",
            sample_interval=0.002,
            timeout=5.0,
            progress_callback=snapshots.append,
            progress_interval=0.01,
        )
        self.assertTrue(snapshots)
        self.assertTrue(
            any(snapshot["phase"] == "calculation" for snapshot in snapshots)
        )
        self.assertTrue(
            any(snapshot["current_iteration"] == 1 for snapshot in snapshots)
        )

    def test_declared_unsupported_precedes_resource_preflight(self) -> None:
        task = mock.Mock(adapter="pyradiomics", family="ivh")
        self.assertEqual(
            run._declared_unsupported_reason(task),
            "pyradiomics does not declare support for: ivh",
        )

    def make_dataset(
        self,
        complexities: list[int],
        *,
        modalities: list[str] | None = None,
        synthetic_series: bool = False,
    ) -> None:
        case_modalities = modalities or (
            ["synthetic"] * len(complexities)
            if synthetic_series
            else ["ct"] * len(complexities)
        )
        cases = []
        files = []
        for index, complexity in enumerate(complexities, 1):
            size = 10 if complexity == 1000 else 20
            case_id = f"case-{index}"
            image = self.dataset / f"{case_id}-image.nii.gz"
            mask = self.dataset / f"{case_id}-mask.nii.gz"
            image_data = np.arange(complexity, dtype=np.float32).reshape(
                (size, size, size)
            )
            mask_data = np.zeros(complexity, dtype=np.uint8)
            mask_data[: max(1, complexity // 2)] = 1
            nib.save(nib.Nifti1Image(image_data, np.eye(4)), image)
            nib.save(
                nib.Nifti1Image(mask_data.reshape((size, size, size)), np.eye(4)),
                mask,
            )
            files.extend(
                [
                    {
                        "path": image.name,
                        "sha256": digest(image),
                        "bytes": image.stat().st_size,
                        "kind": "image",
                    },
                    {
                        "path": mask.name,
                        "sha256": digest(mask),
                        "bytes": mask.stat().st_size,
                        "kind": "mask",
                    },
                ]
            )
            cases.append(
                {
                    "case_id": case_id,
                    "modality": case_modalities[index - 1],
                    "size": size,
                    "variant": index,
                    "subject_id": "shared-series" if synthetic_series else case_id,
                    "mask_id": "M1" if synthetic_series else f"M{index}",
                    "mask_label": f"mask-{index}",
                    "image_path": image.name,
                    "mask_path": mask.name,
                    "image_sha256": digest(image),
                    "mask_sha256": digest(mask),
                    "shape": [size, size, size],
                    "spacing": [1.0, 1.0, 1.0],
                    "orientation": ["R", "A", "S"],
                    "affine": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "image_voxels": complexity,
                    "complexity": complexity,
                    "mask_voxels": max(1, complexity // 2),
                    "mask_fraction": 0.5,
                }
            )
        (self.dataset / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset": "fake-v1",
                    "dataset_kind": "synthetic" if synthetic_series else "real_world",
                    "files": files,
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )

    def args(self, *extra: str) -> list[str]:
        return [
            "--dataset-dir",
            str(self.dataset),
            "--report-dir",
            str(self.report),
            "--run-id",
            "unit-run",
            "--checkpoint-interval",
            "1000",
            "--repeats",
            "1",
            "--endpoint-contract",
            "none",
            *extra,
        ]

    def patches(self, fake):
        return mock.patch.multiple(
            run,
            run_adapter_process=mock.DEFAULT,
            _adapter_environment_snapshots=mock.DEFAULT,
            _machine_info=mock.DEFAULT,
            _git_commit=mock.DEFAULT,
        )

    def test_default_workloads_are_all_canonical_families(self) -> None:
        parsed = run._make_parser().parse_args(["--dataset-dir", "unused"])
        self.assertEqual(run._parse_families(parsed.families), list(run.FAMILY_ORDER))
        self.assertEqual(run._parse_families(""), [])

    def test_task_plan_rotates_adapter_order_across_exact_comparison_blocks(
        self,
    ) -> None:
        case = {
            "case_id": "case",
            "modality": "ct",
            "size": 10,
            "variant": 1,
            "mask_id": "M1",
            "mask_label": "roi",
            "image_abs": "/tmp/image.nii.gz",
            "mask_abs": "/tmp/mask.nii.gz",
            "image_sha256": "a" * 64,
            "mask_sha256": "b" * 64,
            "shape": [10, 10, 10],
            "spacing": [1.0, 1.0, 1.0],
            "image_voxels": 1000,
            "mask_voxels": 500,
            "mask_fraction": 0.5,
            "complexity": 1000,
        }
        tasks = run.build_task_plan(
            cases=[case],
            dataset="unit",
            adapters=["pictologics", "pyradiomics"],
            families=["glcm", "glrlm"],
            repeats=2,
            discretization="fbn",
            bins=32,
            bin_width=32.0,
            intensity_min=None,
            intensity_max=None,
            timing_iterations=2,
            no_warmup=False,
            drop_first_repeat=False,
        )

        self.assertEqual(
            [(task.family, task.repeat, task.adapter) for task in tasks],
            [
                ("glcm", 1, "pictologics"),
                ("glcm", 1, "pyradiomics"),
                ("glrlm", 1, "pyradiomics"),
                ("glrlm", 1, "pictologics"),
                ("glcm", 2, "pyradiomics"),
                ("glcm", 2, "pictologics"),
                ("glrlm", 2, "pictologics"),
                ("glrlm", 2, "pyradiomics"),
            ],
        )
        self.assertEqual([task.ordinal for task in tasks], list(range(1, 9)))

    def test_rotation_balances_every_adapter_across_execution_positions(self) -> None:
        case = {
            "case_id": "case",
            "modality": "ct",
            "size": 2,
            "variant": 1,
            "mask_id": "M1",
            "mask_label": "roi",
            "image_abs": "/tmp/image.nii.gz",
            "mask_abs": "/tmp/mask.nii.gz",
            "image_sha256": "a" * 64,
            "mask_sha256": "b" * 64,
            "shape": [2, 2, 2],
            "spacing": [1.0, 1.0, 1.0],
            "image_voxels": 8,
            "mask_voxels": 4,
            "mask_fraction": 0.5,
            "complexity": 8,
        }
        adapters = ["pictologics", "mirp", "medimage", "pyradiomics", "zrad"]
        tasks = run.build_task_plan(
            cases=[case],
            dataset="unit",
            adapters=adapters,
            families=["glcm"],
            repeats=len(adapters),
            discretization="fbn",
            bins=32,
            bin_width=32.0,
            intensity_min=None,
            intensity_max=None,
            timing_iterations=2,
            no_warmup=False,
            drop_first_repeat=False,
        )
        blocks = [
            tasks[index : index + len(adapters)]
            for index in range(0, len(tasks), len(adapters))
        ]
        for adapter in adapters:
            self.assertEqual(
                sorted(
                    position
                    for block in blocks
                    for position, task in enumerate(block)
                    if task.adapter == adapter
                ),
                list(range(len(adapters))),
            )

    def test_order_balances_positions_and_neighbours_across_cases(self) -> None:
        adapters = ["pictologics", "pyradiomics", "mirp", "medimage", "zrad"]
        cases = [
            {
                "case_id": f"case{index}",
                "modality": "ct",
                "size": 2,
                "variant": 1,
                "mask_id": "M1",
                "mask_label": "roi",
                "image_abs": "/tmp/image.nii.gz",
                "mask_abs": "/tmp/mask.nii.gz",
                "image_sha256": "a" * 64,
                "mask_sha256": "b" * 64,
                "shape": [2, 2, 2],
                "spacing": [1.0, 1.0, 1.0],
                "image_voxels": 8,
                "mask_voxels": 4,
                "mask_fraction": 0.5,
                "complexity": 8,
            }
            for index in range(2 * len(adapters))
        ]
        tasks = run.build_task_plan(
            cases=cases,
            dataset="unit",
            adapters=adapters,
            families=["glcm", "glrlm"],
            repeats=1,
            discretization="fbn",
            bins=32,
            bin_width=32.0,
            intensity_min=None,
            intensity_max=None,
            timing_iterations=2,
            no_warmup=False,
            drop_first_repeat=False,
        )
        for family in ("glcm", "glrlm"):
            blocks = {}
            for task in tasks:
                if task.family == family:
                    blocks.setdefault(task.case_id, []).append(task.adapter)
            positions = Counter(
                (adapter, position)
                for order in blocks.values()
                for position, adapter in enumerate(order)
            )
            neighbours = Counter(
                pair for order in blocks.values() for pair in zip(order, order[1:])
            )
            # Ten cases: each adapter twice in each position, and each ordered
            # pair of adapters twice back to back.
            self.assertEqual(set(positions.values()), {2})
            self.assertEqual(len(positions), len(adapters) ** 2)
            self.assertEqual(set(neighbours.values()), {2})
            self.assertEqual(len(neighbours), len(adapters) * (len(adapters) - 1))

    def test_environment_probe_captures_numpy_build_configuration(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", run._adapter_environment_probe_script()],
            capture_output=True,
            text=True,
            check=True,
        )
        snapshot = json.loads(completed.stdout)

        self.assertEqual(snapshot["python_version"], sys.version.split()[0])
        self.assertTrue(snapshot["numpy_config"]["available"])
        self.assertEqual(snapshot["numpy_config"]["version"], np.__version__)
        self.assertTrue(snapshot["numpy_config"]["show_config"].strip())

    def test_execution_source_scope_binds_transitive_and_lock_dependencies(
        self,
    ) -> None:
        sources = run._benchmark_execution_sources(["pyradiomics"])
        for required in (
            "pyproject.toml",
            "poetry.lock",
            "bench/run.py",
            "bench/ibsi_mapping.py",
            "bench/ibsi_codes.py",
            "bench/adapters/pyradiomics_adapter.py",
        ):
            self.assertIn(required, sources)
        self.assertNotIn("bench/report.py", sources)
        self.assertFalse(any(path.startswith("bench/compliance/") for path in sources))

    def test_nonfinite_bin_width_is_rejected_during_planning(self) -> None:
        case = {
            "case_id": "case",
            "modality": "ct",
            "size": 2,
            "variant": 1,
            "mask_id": "M1",
            "mask_label": "roi",
            "image_abs": "/tmp/image.nii.gz",
            "mask_abs": "/tmp/mask.nii.gz",
            "image_sha256": "a" * 64,
            "mask_sha256": "b" * 64,
            "shape": [2, 2, 2],
            "spacing": [1.0, 1.0, 1.0],
            "image_voxels": 8,
            "mask_voxels": 4,
            "mask_fraction": 0.5,
            "complexity": 8,
        }
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "finite and positive"),
            ):
                run.build_task_plan(
                    cases=[case],
                    dataset="unit",
                    adapters=["pictologics"],
                    families=["glcm"],
                    repeats=1,
                    discretization="fbn",
                    bins=32,
                    bin_width=invalid,
                    intensity_min=None,
                    intensity_max=None,
                    timing_iterations=2,
                    no_warmup=False,
                    drop_first_repeat=False,
                )

    def test_darwin_cpu_probe_uses_chip_without_persisting_hardware_report(
        self,
    ) -> None:
        hardware_report = subprocess.CompletedProcess(
            args=["system_profiler"],
            returncode=0,
            stdout=(
                "Hardware:\n"
                "    Chip: Apple M4 Pro\n"
                "    Serial Number (system): must-not-be-recorded\n"
            ),
            stderr="",
        )
        with (
            mock.patch.object(run.platform, "system", return_value="Darwin"),
            mock.patch.object(run.platform, "processor", return_value="arm"),
            mock.patch.object(run.subprocess, "run", return_value=hardware_report),
        ):
            self.assertEqual(run._auto_cpu_model(), "Apple M4 Pro")

    def test_machine_probe_is_anonymous_and_rejects_bogus_frequency(self) -> None:
        frequency = mock.Mock(current=4.0, max=4.0)
        memory = mock.Mock(total=48 * 1024**3)
        with (
            mock.patch.object(run, "_auto_cpu_model", return_value="Apple M4 Pro"),
            mock.patch.object(
                run.socket, "gethostname", return_value="private-device-id"
            ),
            mock.patch.object(run.psutil, "cpu_freq", return_value=frequency),
            mock.patch.object(run.psutil, "cpu_count", side_effect=[14, None]),
            mock.patch.object(run.os, "cpu_count", return_value=14),
            mock.patch.object(run.psutil, "virtual_memory", return_value=memory),
        ):
            machine = run._machine_info()

        self.assertNotIn("hostname", machine)
        self.assertRegex(machine["machine_id"], r"^anonymous-[0-9a-f]{16}$")
        self.assertNotIn("private-device-id", json.dumps(machine))
        self.assertEqual(machine["machine_label"], "Apple M4 Pro")
        self.assertIsNone(machine["cpu_current_ghz"])
        self.assertIsNone(machine["cpu_max_ghz"])
        self.assertEqual(machine["cpu_count_logical"], 14)

        with self.assertRaisesRegex(ValueError, "cpu_base_ghz"):
            run._machine_info(cpu_base_ghz=0.004)

    def test_frozen_host_profile_is_part_of_benchmark_machine_identity(self) -> None:
        machine = run._machine_info(
            machine_id="test-host",
            host_profile_id="test-host",
            host_profile_sha256="a" * 64,
            host_settings_json='{"power_source":"AC Power"}',
        )
        identity = run._benchmark_machine_identity(machine)
        self.assertEqual(identity["host_profile_id"], "test-host")
        self.assertEqual(identity["host_profile_sha256"], "a" * 64)
        self.assertEqual(identity["host_settings"], {"power_source": "AC Power"})

        with self.assertRaisesRegex(ValueError, "valid JSON"):
            run._machine_info(host_settings_json="not-json")
        with self.assertRaisesRegex(ValueError, "provided together"):
            run._machine_info(host_profile_id="test-host")

    def test_hash_verification_can_only_be_disabled_for_dry_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "restricted to dry-run"):
            run.main(
                [
                    "--dataset-dir",
                    str(self.dataset),
                    "--no-verify-dataset-hashes",
                ]
            )

    def test_case_selection_can_filter_modalities(self) -> None:
        self.make_dataset([1000, 1000, 1000], modalities=["ct", "mri", "pet"])
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        selected = run._validate_and_select_cases(
            self.dataset,
            manifest,
            sizes=None,
            variants=None,
            masks=None,
            modalities="ct,pet",
            verify_hashes=False,
        )
        self.assertEqual(
            {case["modality"] for case in selected},
            {"ct", "pet"},
        )

    def test_payload_contract_rejects_wrong_aggregation(self) -> None:
        task = run.build_task_plan(
            cases=[
                {
                    "case_id": "case",
                    "modality": "ct",
                    "size": 10,
                    "variant": 1,
                    "mask_id": "M1",
                    "mask_label": "roi",
                    "image_abs": "/tmp/image.nii.gz",
                    "mask_abs": "/tmp/mask.nii.gz",
                    "image_sha256": "a" * 64,
                    "mask_sha256": "b" * 64,
                    "shape": [10, 10, 10],
                    "spacing": [1.0, 1.0, 1.0],
                    "image_voxels": 1000,
                    "mask_voxels": 500,
                    "mask_fraction": 0.5,
                    "complexity": 1000,
                }
            ],
            dataset="unit",
            adapters=["pictologics"],
            families=["glcm"],
            repeats=1,
            discretization="fbn",
            bins=32,
            bin_width=32.0,
            intensity_min=None,
            intensity_max=None,
            timing_iterations=2,
            no_warmup=False,
            drop_first_repeat=False,
        )[0]
        self.assertEqual(task.shape, (10, 10, 10))
        self.assertEqual(task.spacing, (1.0, 1.0, 1.0))
        self.assertEqual(task.to_dict()["shape"], [10, 10, 10])
        self.assertEqual(task.to_dict()["spacing"], [1.0, 1.0, 1.0])
        record = run._record_template(task, "unit-run")
        self.assertEqual(record["shape"], [10, 10, 10])
        self.assertEqual(record["spacing"], [1.0, 1.0, 1.0])
        payload = adapter_payload("pictologics", "glcm")
        payload["timing"] = {
            "duration_sec": 1.0,
            "duration_min_sec": 1.0,
            "duration_mean_sec": 1.0,
            "duration_median_sec": 1.0,
            "duration_std_sec": 0.0,
            "duration_max_sec": 1.0,
            "cpu_time_sec": 1.0,
            "cpu_time_min_sec": 1.0,
            "cpu_time_mean_sec": 1.0,
            "cpu_time_median_sec": 1.0,
            "cpu_time_std_sec": 0.0,
            "cpu_time_max_sec": 1.0,
            "measured_iterations": 2,
            "warmup_iterations": 1,
            "total_iterations": 3,
            "duration_samples_sec": [1.0, 1.0],
            "cpu_time_samples_sec": [1.0, 1.0],
            "preparation_samples_sec": [0.0, 0.0],
            "finalization_samples_sec": [0.0, 0.0],
            "warmup_duration_sec": 1.0,
            "warmup_cpu_time_sec": 1.0,
        }
        payload["metadata"] = {
            "input": {
                "image_sha256": task.image_sha256,
                "source_image_sha256": task.source_image_sha256,
                "mask_sha256": task.mask_sha256,
                "modality": task.modality,
                "input_contract": task.input_contract,
                "representation_id": task.representation_id,
                "representation_derivation_sha256": (
                    task.representation_derivation_sha256
                ),
                "configured_levels": task.configured_levels,
                "occupied_levels": task.occupied_levels,
            },
            "preprocessing": {
                "discretization": "fbn",
                "bins": 32,
                "bin_width": 32.0,
                "intensity_range": None,
            },
            "aggregation": {
                "requested": "3d_average",
                "effective_directional": "3d_average",
            },
            "timing_contract": timing_contract_metadata(),
        }
        observed_metrics = metrics(1.0)
        observed_metrics.update(
            {
                "measured_iterations": 2,
                "warmup_iterations": 1,
                "total_iterations": 3,
            }
        )
        with self.assertRaisesRegex(
            run.AdapterProcessError,
            "3D-merged aggregation",
        ):
            run._validate_measured_result(
                payload,
                observed_metrics,
                expected_task=task,
            )

        payload["metadata"]["aggregation"] = {
            "requested": "3d_merge",
            "effective_directional": "3d_merge",
        }
        payload["metadata"]["timing_contract"]["version"] = 999
        with self.assertRaisesRegex(
            run.AdapterProcessError,
            "timing contract",
        ):
            run._validate_measured_result(
                payload,
                observed_metrics,
                expected_task=task,
            )

        payload["metadata"]["timing_contract"]["version"] = run.TIMING_CONTRACT_VERSION
        payload["metadata"]["package_initialization"] = {
            "jit_warmup_performed": True,
            "outside_measured_region": True,
        }
        for invalid in ("1.0", True):
            payload["values"] = {"all": {"feature": invalid}}
            with (
                self.subTest(value=invalid),
                self.assertRaisesRegex(
                    run.AdapterProcessError,
                    "non-finite/non-scalar",
                ),
            ):
                run._validate_measured_result(
                    payload,
                    observed_metrics,
                    expected_task=task,
                )

    def stable_patchers(self, fake):
        def contractual_fake(adapter, **kwargs):
            payload, observed_metrics = fake(adapter, **kwargs)
            iterations = int(kwargs["iterations"])
            warmups = 0 if kwargs.get("no_warmup") else 1
            measured = iterations
            duration = float(observed_metrics["duration_sec"])
            payload["timing"] = {
                "duration_sec": duration,
                "duration_min_sec": duration,
                "duration_mean_sec": duration,
                "duration_median_sec": duration,
                "duration_std_sec": 0.0,
                "duration_max_sec": duration,
                "cpu_time_sec": duration,
                "cpu_time_min_sec": duration,
                "cpu_time_mean_sec": duration,
                "cpu_time_median_sec": duration,
                "cpu_time_std_sec": 0.0,
                "cpu_time_max_sec": duration,
                "measured_iterations": measured,
                "warmup_iterations": warmups,
                "total_iterations": iterations + warmups,
                "duration_samples_sec": [duration] * measured,
                "cpu_time_samples_sec": [duration] * measured,
                "preparation_samples_sec": [0.0] * measured,
                "finalization_samples_sec": [0.0] * measured,
                "warmup_duration_sec": duration if warmups else None,
                "warmup_cpu_time_sec": duration if warmups else None,
                "warmup_preparation_sec": 0.0 if warmups else None,
                "warmup_finalization_sec": 0.0 if warmups else None,
            }
            payload["metadata"] = {
                "input": {
                    "image_sha256": kwargs["image_sha256"],
                    "source_image_sha256": kwargs["source_image_sha256"],
                    "mask_sha256": kwargs["mask_sha256"],
                    "modality": kwargs.get("modality"),
                    "input_contract": kwargs["input_contract"],
                    "representation_id": kwargs["input_representation_id"],
                    "representation_derivation_sha256": kwargs.get(
                        "representation_derivation_sha256"
                    ),
                    "configured_levels": kwargs.get("configured_levels"),
                    "occupied_levels": kwargs.get("occupied_levels"),
                },
                "preprocessing": {
                    "discretization": kwargs["discretization"],
                    "bins": kwargs["bins"],
                    "bin_width": kwargs["bin_width"],
                    "intensity_range": (
                        [kwargs["intensity_min"], kwargs["intensity_max"]]
                        if kwargs.get("intensity_min") is not None
                        else None
                    ),
                },
                "aggregation": {
                    "requested": kwargs["aggregation"],
                    "effective_directional": kwargs["aggregation"],
                },
                "timing_contract": timing_contract_metadata(),
            }
            if adapter == "pictologics":
                payload["metadata"]["package_initialization"] = {
                    "jit_warmup_performed": not bool(kwargs.get("no_warmup")),
                    "outside_measured_region": True,
                }
            if adapter == "medimage":
                payload["metadata"]["preprocessing"]["intensity_type"] = (
                    run.MEDIMAGE_BENCHMARK_INTENSITY_TYPE
                )
            payload["values"] = {
                "all": {
                    feature_name: 1.0 for feature_name in payload["features"]["all"]
                }
            }
            observed_metrics.update(
                {
                    "measured_iterations": measured,
                    "warmup_iterations": warmups,
                    "total_iterations": iterations + warmups,
                    "adapter_event_count": 1 + 2 * measured + warmups,
                }
            )
            return payload, observed_metrics

        @contextmanager
        def process_with_stable_bindings():
            with (
                mock.patch.object(
                    run,
                    "run_adapter_process",
                    side_effect=contractual_fake,
                ) as process,
                mock.patch.object(run, "_verify_execution_bindings"),
                mock.patch.object(
                    run, "_benchmark_sources_sha256", return_value="test-sources"
                ),
                # Keep the memory preflight independent of the test host.
                mock.patch.object(
                    run.psutil,
                    "virtual_memory",
                    return_value=mock.Mock(available=64 * 1024**3, total=128 * 1024**3),
                ),
            ):
                yield process

        process = process_with_stable_bindings()
        environments = mock.patch.object(
            run,
            "_adapter_environment_snapshots",
            return_value={"pictologics": {"version": "1"}, "slow": {"version": "1"}},
        )
        machine = mock.patch.object(
            run,
            "_machine_info",
            return_value={
                "cpu_model": "fake",
                "cpu_count_physical": 1,
                "memory_total_bytes": 1024**3,
            },
        )
        commit = mock.patch.object(run, "_git_commit", return_value="test")
        return process, environments, machine, commit

    def test_family_guardrail_skips_only_strictly_larger_matching_workload(
        self,
    ) -> None:
        self.make_dataset([1000, 1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            family = kwargs["families"][0]
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append((adapter, family, case_id))
            duration = (
                1.0
                if adapter == "pictologics"
                else (20.0 if family == "glrlm" else 2.0)
            )
            return (
                adapter_payload(adapter, family, [f"{family}-feature"]),
                metrics(duration),
            )

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            result = run.main(
                self.args(
                    "--adapters",
                    "pictologics,slow",
                    "--families",
                    "glcm,glrlm",
                    "--guardrail-skip-ratio",
                    "10",
                    "--keep-going",
                )
            )
        self.assertEqual(result, 0)
        # glrlm's first comparison block deliberately runs the candidate before
        # its baseline. Deferred reconciliation must still establish the cutoff.
        self.assertNotIn(("slow", "glrlm", "case-3"), calls)
        # case-1 and case-2 intentionally contain identical image bytes, so the
        # content-addressed input stage reuses case-1's snapshot filename.
        self.assertEqual(
            sum(
                adapter == "slow" and family == "glrlm" for adapter, family, _ in calls
            ),
            2,
        )
        self.assertIn(("slow", "glcm", "case-3"), calls)
        self.assertEqual(process.call_count, 11)

        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            counts = ledger.status_counts()
            self.assertEqual(counts[STATUS_MEASURED], 11)
            self.assertEqual(counts[STATUS_SKIPPED], 1)
            decision = ledger.guardrail_decision(
                "slow\x1ffamily:glrlm\x1fsubject:shared-series|mask:M1|"
                "representation:single_image_internal_preprocessing"
            )
            self.assertEqual(int(decision["cutoff_complexity"]), 1000)

        with (self.report / "summary.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        skipped = [row for row in rows if row["task_status"] == STATUS_SKIPPED]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["case_id"], "case-3")
        self.assertEqual(skipped[0]["success"], "False")

    def test_guardrail_does_not_cross_real_world_cases(self) -> None:
        self.make_dataset([1000, 8000], modalities=["ct", "ct"])
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append((adapter, case_id))
            duration = 1.0 if adapter == "pictologics" else 20.0
            return adapter_payload(adapter, kwargs["families"][0]), metrics(duration)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics,slow",
                        "--families",
                        "glcm",
                        "--guardrail-skip-ratio",
                        "10",
                        "--keep-going",
                    )
                ),
                0,
            )

        self.assertIn(("slow", "case-2"), calls)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertIsNotNone(
                ledger.guardrail_decision(
                    "slow\x1ffamily:glcm\x1fcase:case-1|modality:ct"
                )
            )
            self.assertIsNotNone(
                ledger.guardrail_decision(
                    "slow\x1ffamily:glcm\x1fcase:case-2|modality:ct"
                )
            )

    def test_resume_does_not_rerun_measured_tasks_and_rejects_parameter_drift(
        self,
    ) -> None:
        self.make_dataset([1000])
        modalities = []

        def fake(adapter, **kwargs):
            modalities.append(kwargs["modality"])
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        common = self.args("--adapters", "pictologics", "--families", "glcm")
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)
        self.assertEqual(modalities, ["ct"])

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            process.assert_not_called()

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            with self.assertRaises(RunSpecMismatch):
                run.main([*common, "--resume", "--bins", "64"])

    def test_power_loss_stops_before_the_next_task_and_resume_continues(
        self,
    ) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--families", "glcm")
        battery = ["power source must be 'AC Power'; observed 'Battery Power'"]
        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            with mock.patch.object(run, "_runtime_state_errors", return_value=battery):
                self.assertEqual(run.main(common), run.RUNTIME_STATE_EXIT_CODE)
            process.assert_not_called()
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.run_status(), "interrupted")
            stops = ledger.connection.execute(
                "SELECT details_json FROM events WHERE event_type = 'runtime_state_stop'"
            ).fetchall()
        self.assertEqual(len(stops), 1)

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            with mock.patch.object(run, "_runtime_state_errors", return_value=[]):
                self.assertEqual(run.main([*common, "--resume"]), 0)
            process.assert_called_once()

    def test_resume_rejects_machine_drift(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--families", "glcm")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)

        process = mock.patch.object(run, "run_adapter_process", side_effect=fake)
        environments = mock.patch.object(
            run,
            "_adapter_environment_snapshots",
            return_value={"pictologics": {"version": "1"}},
        )
        machine = mock.patch.object(
            run,
            "_machine_info",
            return_value={
                "machine_id": "different-machine",
                "cpu_model": "fake",
                "cpu_count_physical": 1,
                "memory_total_bytes": 1024**3,
            },
        )
        commit = mock.patch.object(run, "_git_commit", return_value="test")
        with process as process_mock, environments, machine, commit:
            with self.assertRaises(RunSpecMismatch):
                run.main([*common, "--resume"])
            process_mock.assert_not_called()

    def test_resume_rejects_corrupted_committed_payload(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--families", "glcm")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)
        payload = next((self.report / "records").rglob("*.json"))
        payload.write_text("{}\n", encoding="utf-8")

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            with self.assertRaises(RunIntegrityError):
                run.main([*common, "--resume"])
            process.assert_not_called()

    def test_non_resume_run_rejects_orphaned_benchmark_artifacts(self) -> None:
        self.make_dataset([1000])
        orphaned = self.report / "records" / "case-1" / "stale.json"
        orphaned.parent.mkdir(parents=True)
        orphaned.write_text("{}\n", encoding="utf-8")

        def fake(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            with self.assertRaises(RunAlreadyExists):
                run.main(self.args("--adapters", "pictologics", "--families", "glcm"))
            process.assert_not_called()

    def test_fbs_requires_finite_ordered_intensity_bounds(self) -> None:
        self.make_dataset([1000])
        base = self.args("--adapters", "pictologics", "--families", "glcm")
        invalid_ranges = [
            ["--discretization", "fbs"],
            ["--discretization", "fbs", "--intensity-min", "-1024"],
            [
                "--discretization",
                "fbs",
                "--intensity-min",
                "nan",
                "--intensity-max",
                "3096",
            ],
            [
                "--discretization",
                "fbs",
                "--intensity-min",
                "5",
                "--intensity-max",
                "5",
            ],
        ]
        for extra in invalid_ranges:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                run.main([*base, *extra])

    def test_intensity_range_is_forwarded_and_immutable(self) -> None:
        self.make_dataset([1000])
        ranges = []

        def fake(adapter, **kwargs):
            ranges.append((kwargs["intensity_min"], kwargs["intensity_max"]))
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        common = self.args(
            "--adapters",
            "pictologics",
            "--families",
            "glcm",
            "--discretization",
            "fbs",
            "--intensity-min",
            "-1024",
            "--intensity-max",
            "3096",
        )
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)
        self.assertEqual(ranges, [(-1024.0, 3096.0)])
        stored_spec = json.loads(
            (self.report / "run_spec.json").read_text(encoding="utf-8")
        )
        self.assertEqual(stored_spec["intensity_min"], -1024.0)
        self.assertEqual(stored_spec["intensity_max"], 3096.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            with self.assertRaises(RunSpecMismatch):
                run.main(
                    [
                        *common,
                        "--resume",
                        "--intensity-min",
                        "-1000",
                        "--intensity-max",
                        "3000",
                    ]
                )

    def test_adapter_command_receives_intensity_range(self) -> None:
        environment_dir = self.root / "adapter-env"
        python = run._env_python(environment_dir)
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        payload = json.dumps(
            {
                "features": {"all": ["feature"]},
                "timing": {"duration_sec": 1.0},
            }
        )
        host = {
            "host_wall_time_sec": 1.0,
            "host_peak_rss_bytes": 1024,
            "host_cpu_time_sec": 1.0,
        }
        with (
            mock.patch.object(run, "_adapter_env_dir", return_value=environment_dir),
            mock.patch.object(
                run,
                "_run_process_command",
                return_value=(payload, "", host),
            ) as process,
            mock.patch.dict(
                run.os.environ, {"OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
            ),
        ):
            run.run_adapter_process(
                "external",
                image="image.nii.gz",
                mask="mask.nii.gz",
                discretization="fbs",
                intensity_min=-1024.0,
                intensity_max=3096.0,
                image_sha256="a" * 64,
                mask_sha256="b" * 64,
                no_warmup=True,
            )
        command = process.call_args.args[0]
        self.assertEqual(command[command.index("--intensity-min") + 1], "-1024.0")
        self.assertEqual(command[command.index("--intensity-max") + 1], "3096.0")
        self.assertEqual(command[command.index("--image-sha256") + 1], "a" * 64)
        self.assertEqual(command[command.index("--mask-sha256") + 1], "b" * 64)
        self.assertIn("--no-warmup", command)
        environment = process.call_args.kwargs["environment"]
        # A shell thread limit never reaches the adapter.
        self.assertFalse(
            set(run.BENCHMARK_THREAD_ENV_REMOVED).intersection(environment)
        )
        self.assertEqual(
            {key: environment[key] for key in run.BENCHMARK_INITIALIZATION_ENV},
            run.BENCHMARK_INITIALIZATION_ENV,
        )

    def test_one_library_thread_policy_limits_only_maths_libraries(self) -> None:
        policy = run.thread_policy("one_library_thread")
        self.assertEqual(
            policy["set_environment"], {name: "1" for name in run.LIBRARY_THREAD_ENV}
        )
        # Package parallel code keeps its native threads.
        self.assertNotIn("NUMBA_NUM_THREADS", policy["set_environment"])
        self.assertNotIn(
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", policy["set_environment"]
        )
        self.assertEqual(run.thread_policy()["set_environment"], {})
        with self.assertRaisesRegex(ValueError, "unknown thread policy"):
            run.thread_policy("single")

    def test_medimage_command_receives_explicit_intensity_type(self) -> None:
        environment_dir = self.root / "adapter-env"
        python = run._env_python(environment_dir)
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        payload = json.dumps(
            {
                "features": {"all": ["feature"]},
                "timing": {"duration_sec": 1.0},
            }
        )
        host = {
            "host_wall_time_sec": 1.0,
            "host_peak_rss_bytes": 1024,
            "host_cpu_time_sec": 1.0,
        }
        with (
            mock.patch.object(run, "_adapter_env_dir", return_value=environment_dir),
            mock.patch.object(
                run,
                "_run_process_command",
                return_value=(payload, "", host),
            ) as process,
        ):
            run.run_adapter_process(
                "medimage",
                image="image.nii.gz",
                mask="mask.nii.gz",
                no_warmup=True,
            )
        command = process.call_args.args[0]
        self.assertEqual(
            command[command.index("--intensity-type") + 1],
            run.MEDIMAGE_BENCHMARK_INTENSITY_TYPE,
        )

    def test_enabled_guardrail_requires_selected_baseline(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(ValueError, "baseline adapter"):
            run.main(
                self.args(
                    "--adapters",
                    "slow",
                    "--families",
                    "glcm",
                )
            )

    def test_disabled_guardrail_still_requires_selected_baseline(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(ValueError, "baseline adapter"):
            run.main(
                self.args(
                    "--adapters",
                    "pyradiomics",
                    "--families",
                    "glcm",
                    "--disable-speed-truncation",
                )
            )

    def test_selected_baseline_must_support_every_scheduled_family(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(
            ValueError,
            "baseline adapter.*unsupported families",
        ):
            run.main(
                self.args(
                    "--adapters",
                    "pyradiomics",
                    "--guardrail-baseline",
                    "pyradiomics",
                    "--families",
                    "all",
                    "--dry-run",
                )
            )

    def test_an_empty_family_selection_is_rejected(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(ValueError, "select one or more --families"):
            run.main(
                self.args(
                    "--adapters",
                    "pictologics",
                    "--families",
                    "",
                )
            )

    def test_drop_first_repeat_requires_a_retained_repeat(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(ValueError, "at least two repeats"):
            run.main(
                self.args(
                    "--adapters",
                    "pictologics",
                    "--families",
                    "glcm",
                    "--drop-first-repeat",
                )
            )

    def test_timeout_is_censored_and_persistently_skips_larger_task(self) -> None:
        self.make_dataset([1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append((adapter, case_id))
            if adapter == "slow":
                raise run.AdapterTimeout(
                    "slow",
                    5.0,
                    12.0,
                    999999,
                    phase="calculation",
                    partial_duration_samples_sec=(1.25,),
                    partial_cpu_time_samples_sec=(1.0,),
                    details={
                        "stop_phase": "calculation",
                        "stop_call_index": 2,
                        "censor_lower_bound_sec": 5.1,
                        "warmup_duration_sec": 1.5,
                        "task_elapsed_sec": 12.0,
                    },
                )
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            result = run.main(
                self.args(
                    "--adapters",
                    "pictologics,slow",
                    "--families",
                    "glcm",
                    "--timeout",
                    "5",
                    "--keep-going",
                )
            )
        self.assertEqual(result, 0)
        self.assertNotIn(("slow", "case-2"), calls)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            records = ledger.records()
        timed = [
            record for record in records if record["task_status"] == STATUS_TIMED_OUT
        ]
        skipped = [
            record for record in records if record["task_status"] == STATUS_SKIPPED
        ]
        self.assertEqual(len(timed), 1)
        self.assertEqual(len(skipped), 1)
        self.assertIsNone(timed[0]["duration_sec"])
        self.assertEqual(timed[0]["censor_lower_bound_sec"], 5.1)
        self.assertEqual(timed[0]["stop_phase"], "calculation")
        self.assertEqual(timed[0]["stop_call_index"], 2)
        self.assertEqual(timed[0]["warmup_duration_sec"], 1.5)
        self.assertEqual(timed[0]["per_call_timeout_sec"], 5.0)
        self.assertEqual(len(timed[0]["load_average_start"]), 3)
        self.assertEqual(timed[0]["partial_duration_samples_sec"], [1.25])
        self.assertEqual(timed[0]["partial_cpu_time_samples_sec"], [1.0])
        self.assertEqual(timed[0]["partial_completed_iterations"], 1)
        self.assertFalse(timed[0]["success"])

    def test_baseline_timeout_truncates_like_every_adapter(self) -> None:
        self.make_dataset([1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append((adapter, case_id))
            if adapter == "pictologics":
                raise run.AdapterTimeout(adapter, 5.0, 5.1, 999999)
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics,slow",
                        "--families",
                        "glcm",
                        "--timeout",
                        "5",
                        "--keep-going",
                    )
                ),
                0,
            )
        self.assertEqual(
            calls,
            [
                ("pictologics", "case-1"),
                ("slow", "case-1"),
                ("slow", "case-2"),
            ],
        )
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.status_counts()[STATUS_TIMED_OUT], 1)
            self.assertEqual(ledger.status_counts()[STATUS_MEASURED], 2)
            self.assertEqual(ledger.status_counts()[STATUS_SKIPPED], 1)

    def test_group_categories_follow_their_members(self) -> None:
        self.assertEqual(run._category_for_family("group_texture"), "texture")
        self.assertEqual(run._category_for_family("group_intensity"), "intensity")
        self.assertEqual(
            run._category_for_family("group_spatial_morphology"), "morphology"
        )

    def test_group_task_asks_for_every_supported_member_in_one_call(self) -> None:
        self.make_dataset([1000])
        calls = []

        def fake(adapter, **kwargs):
            calls.append((adapter, tuple(kwargs["families"])))
            # Without a contract the native policy sets no thread variable.
            self.assertEqual(kwargs["thread_environment"], {})
            payload = adapter_payload(adapter, kwargs["families"][0])
            payload["selection"]["requested_families"] = list(kwargs["families"])
            if adapter == "pictologics" and "ivh" not in kwargs["families"]:
                # A wrong member list must fail validation.
                payload["selection"]["requested_families"] = ["glcm"]
            return payload, metrics(1.0)

        patchers = self.stable_patchers(fake)
        environments = mock.patch.object(
            run,
            "_adapter_environment_snapshots",
            return_value={
                "pictologics": {"version": "1"},
                "pyradiomics": {"version": "1"},
            },
        )
        with patchers[0], patchers[1], patchers[2], patchers[3], environments:
            run.main(
                self.args(
                    "--adapters",
                    "pictologics,pyradiomics",
                    "--families",
                    "group_texture,group_ivh",
                    "--keep-going",
                )
            )
        texture = ("histogram", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm")
        self.assertIn(("pictologics", texture), calls)
        self.assertIn(("pyradiomics", tuple(f for f in texture if f != "gldzm")), calls)
        self.assertIn(("pictologics", ("ivh",)), calls)
        self.assertNotIn("pyradiomics", [a for a, f in calls if f == ("ivh",)])
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            counts = ledger.status_counts()
        self.assertEqual(counts[STATUS_UNSUPPORTED], 1)
        self.assertEqual(counts[STATUS_FAILED], 1)
        self.assertEqual(counts[STATUS_MEASURED], 2)

    def test_timeout_skips_the_same_size_in_later_repeats(self) -> None:
        self.make_dataset([1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append((adapter, case_id))
            if adapter == "slow" and case_id == "case-1":
                raise run.AdapterTimeout(adapter, 5.0, 5.1, 999999)
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        arguments = self.args(
            "--adapters",
            "pictologics,slow",
            "--families",
            "glcm",
            "--timeout",
            "5",
            "--disable-speed-truncation",
            "--keep-going",
        )
        arguments[arguments.index("--repeats") + 1] = "2"
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(arguments), 0)
        # The ratio rule is off, yet the timeout still truncates.  Repeat 2
        # skips the timed-out size as well as the larger one.
        self.assertEqual(calls.count(("slow", "case-1")), 1)
        self.assertNotIn(("slow", "case-2"), calls)
        self.assertEqual(calls.count(("pictologics", "case-2")), 2)

    def test_one_startup_timeout_skips_only_its_task(self) -> None:
        self.make_dataset([1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append((adapter, case_id))
            if adapter == "slow" and case_id == "case-1":
                raise run.AdapterTimeout(
                    adapter, 5.0, 5.1, 999999, details={"stop_phase": "startup"}
                )
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        arguments = self.args(
            "--adapters",
            "pictologics,slow",
            "--families",
            "glcm",
            "--timeout",
            "5",
            "--disable-speed-truncation",
            "--keep-going",
        )
        arguments[arguments.index("--repeats") + 1] = "2"
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(arguments), 0)
        # Repeat 1: the start-up timeout does not skip the larger case.
        # Repeat 2: the second start-up timeout in the branch skips it.
        self.assertEqual(calls.count(("slow", "case-1")), 2)
        self.assertEqual(calls.count(("slow", "case-2")), 1)

    def test_memory_stop_is_censored_and_truncates(self) -> None:
        self.make_dataset([1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append((adapter, case_id, kwargs.get("memory_limit_bytes")))
            if adapter == "slow":
                raise run.AdapterMemoryExceeded(
                    adapter,
                    100,
                    250,
                    3.0,
                    999999,
                    details={"stop_phase": "warmup", "censor_lower_bound_sec": 2.0},
                )
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        memory = mock.Mock(available=64 * 1024**3, total=128 * 1024**3)
        with (
            patchers[0],
            patchers[1],
            patchers[2],
            patchers[3],
            mock.patch.object(run.psutil, "virtual_memory", return_value=memory),
        ):
            result = run.main(
                self.args(
                    "--adapters",
                    "pictologics,slow",
                    "--families",
                    "glcm",
                    "--keep-going",
                )
            )
        self.assertEqual(result, 0)
        self.assertNotIn(
            "case-2", [case for adapter, case, _ in calls if adapter == "slow"]
        )
        self.assertTrue(all(limit and limit > 0 for _, _, limit in calls))
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            records = ledger.records()
        stopped = [
            record
            for record in records
            if record["task_status"] == STATUS_MEMORY_EXCEEDED
        ]
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]["memory_observed_bytes"], 250)
        self.assertEqual(stopped[0]["stop_phase"], "warmup")
        self.assertIsNone(stopped[0]["censor_lower_bound_sec"])
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.run_status(), "completed_with_failures")
        self.assertEqual(
            [record["task_status"] for record in records].count(STATUS_SKIPPED), 1
        )

    def test_dropped_repeat_cannot_activate_a_guardrail(self) -> None:
        self.make_dataset([1000, 8000])
        candidate_calls = 0

        def fake(adapter, **kwargs):
            nonlocal candidate_calls
            duration = 1.0
            if adapter == "slow":
                candidate_calls += 1
                if candidate_calls == 1:
                    duration = 20.0
            return adapter_payload(adapter, kwargs["families"][0]), metrics(duration)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics,slow",
                        "--families",
                        "glcm",
                        "--repeats",
                        "2",
                        "--drop-first-repeat",
                        "--guardrail-skip-ratio",
                        "10",
                    )
                ),
                0,
            )
        self.assertEqual(candidate_calls, 4)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertIsNone(
                ledger.guardrail_decision(
                    "slow\x1ffamily:glcm\x1fcase:case-1|modality:ct"
                )
            )

    def test_zero_process_tree_rss_is_rejected(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            observed = metrics(1.0)
            observed["peak_rss_bytes"] = 0
            observed["host_peak_rss_bytes"] = 0
            return adapter_payload(adapter, kwargs["families"][0]), observed

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--families",
                        "glcm",
                    )
                ),
                1,
            )
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["task_status"], STATUS_FAILED)
        self.assertIn("peak RSS", record["error"])

    def test_memory_preflight_is_a_distinct_terminal_outcome(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--families",
                        "glcm",
                        "--memory-cap-gib",
                        "0.01",
                    )
                ),
                0,
            )
            process.assert_not_called()
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["task_status"], STATUS_SKIPPED_MEMORY)
        self.assertEqual(record["memory_preflight_decision"], "skip")
        self.assertFalse(record["success"])

    def test_generic_failure_never_activates_speed_truncation(self) -> None:
        self.make_dataset([1000, 8000])
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-image")[0]
            calls.append(case_id)
            if case_id == "case-1":
                raise RuntimeError("ordinary adapter failure")
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--families",
                        "glcm",
                        "--disable-speed-truncation",
                        "--keep-going",
                    )
                ),
                0,
            )
        self.assertEqual(calls, ["case-1", "case-2"])
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            counts = ledger.status_counts()
            self.assertEqual(counts[STATUS_FAILED], 1)
            self.assertEqual(counts[STATUS_MEASURED], 1)
            self.assertIsNone(
                ledger.guardrail_decision(
                    "pictologics\x1ffamily:glcm\x1fcase:case-1|modality:ct"
                )
            )

    def test_interrupted_task_is_retried_on_resume(self) -> None:
        self.make_dataset([1000])

        def interrupted(adapter, **kwargs):
            raise run.AdapterInterrupted(adapter, 0.2, 999999)

        patchers = self.stable_patchers(interrupted)
        common = self.args(
            "--adapters",
            "pictologics",
            "--families",
            "glcm",
            "--disable-speed-truncation",
        )
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 130)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(next(iter(ledger.status_counts())), STATUS_INTERRUPTED)

        def measured(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(measured)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            self.assertEqual(process.call_count, 1)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.status_counts()[STATUS_MEASURED], 1)

    def test_atomic_payload_is_adopted_after_ledger_interruption(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--families", "glcm")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)

        connection = sqlite3.connect(self.report / "benchmark.sqlite3")
        try:
            connection.execute(
                "UPDATE tasks SET status = ?, record_json = NULL WHERE status = ?",
                (STATUS_INTERRUPTED, STATUS_MEASURED),
            )
            connection.execute(
                "UPDATE task_attempts SET status = ?", (STATUS_INTERRUPTED,)
            )
            connection.commit()
        finally:
            connection.close()

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            process.assert_not_called()
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.status_counts()[STATUS_MEASURED], 1)

    def test_atomic_payload_with_conflicting_embedded_record_is_not_adopted(
        self,
    ) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--families", "glcm")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)

        payload_path = next((self.report / "records").rglob("*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["benchmark"]["record"]["case_id"] = "WRONG-CASE"
        payload["benchmark"]["record"]["duration_sec"] = 999.0
        payload_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        connection = sqlite3.connect(self.report / "benchmark.sqlite3")
        try:
            connection.execute(
                "UPDATE tasks SET status = ?, record_json = NULL WHERE status = ?",
                (STATUS_INTERRUPTED, STATUS_MEASURED),
            )
            connection.execute(
                "UPDATE task_attempts SET status = ?",
                (STATUS_INTERRUPTED,),
            )
            connection.commit()
        finally:
            connection.close()

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            self.assertEqual(process.call_count, 1)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["case_id"], "case-1")
        self.assertEqual(record["duration_sec"], 1.0)
        self.assertEqual(record["attempt"], 2)

    def test_run_uses_snapshot_if_source_changes_after_validation(self) -> None:
        self.make_dataset([1000])
        source = self.dataset / "case-1-image.nii.gz"
        expected_hash = digest(source)
        seen_paths = []

        def fake(adapter, **kwargs):
            staged = Path(kwargs["image"])
            seen_paths.append(staged)
            self.assertEqual(digest(staged), expected_hash)
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        def mutate_source(_adapters):
            nib.save(
                nib.Nifti1Image(
                    np.full((10, 10, 10), 7.0, dtype=np.float32),
                    np.eye(4),
                ),
                source,
            )
            return {"pictologics": {"version": "1"}}

        patchers = self.stable_patchers(fake)
        environments = mock.patch.object(
            run,
            "_adapter_environment_snapshots",
            side_effect=mutate_source,
        )
        with patchers[0], environments, patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--families",
                        "glcm",
                    )
                ),
                0,
            )
        self.assertNotEqual(digest(source), expected_hash)
        self.assertEqual(len(seen_paths), 1)
        self.assertTrue(
            seen_paths[0].resolve().is_relative_to((self.report / "inputs").resolve())
        )

    def test_staged_input_mutation_during_adapter_execution_is_rejected(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            staged = Path(kwargs["image"])
            staged.chmod(0o644)
            staged.write_bytes(staged.read_bytes() + b"tampered")
            return adapter_payload(adapter, kwargs["families"][0]), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with (
            patchers[0],
            patchers[1],
            patchers[2],
            patchers[3],
            self.assertRaisesRegex(RunIntegrityError, "staged task image changed"),
        ):
            run.main(
                self.args(
                    "--adapters",
                    "pictologics",
                    "--families",
                    "glcm",
                )
            )
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["task_status"], STATUS_INTERRUPTED)
        self.assertIn("changed during the run", record["error"])

    def test_process_timeout_reaps_fake_process(self) -> None:
        with self.assertRaises(run.AdapterTimeout) as caught:
            run._run_process_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                adapter_name="fake",
                timeout=0.05,
                sample_interval=0.01,
                termination_grace=0.1,
            )
        if os.name != "nt":
            # Windows keeps the PID while the controller holds the process handle.
            self.assertFalse(psutil.pid_exists(caught.exception.pid))
        self.assertEqual(caught.exception.phase, "startup")
        self.assertIsNone(caught.exception.details["censor_lower_bound_sec"])

    def test_process_timeout_retains_completed_call_samples(self) -> None:
        code = (
            "import json,time; p='BENCH_EVENT '; "
            "print(p+json.dumps({'event':'worker_ready'}),flush=True); "
            "print(p+json.dumps({'event':'calculation_start','iteration':1}),flush=True); "
            "print(p+json.dumps({'event':'calculation_complete','iteration':1,"
            "'calculation_sec':0.0125,'cpu_time_sec':0.01}),flush=True); "
            "print(p+json.dumps({'event':'calculation_start','iteration':2}),flush=True); "
            "time.sleep(30)"
        )
        with self.assertRaises(run.AdapterTimeout) as caught:
            run._run_process_command(
                [sys.executable, "-c", code],
                adapter_name="fake",
                # Long enough for a slow Python start-up on a CI runner.
                timeout=2.0,
                sample_interval=0.01,
                termination_grace=0.1,
            )
        self.assertEqual(caught.exception.phase, "calculation")
        self.assertEqual(caught.exception.details["stop_call_index"], 2)
        # The finished call keeps its peak memory in the stop record.
        self.assertEqual(
            len(caught.exception.details["calculation_peak_rss_samples_bytes"]), 1
        )
        self.assertLess(caught.exception.details["censor_lower_bound_sec"], 3.0)
        self.assertEqual(caught.exception.partial_duration_samples_sec, (0.0125,))
        self.assertEqual(caught.exception.partial_cpu_time_samples_sec, (0.01,))

    def test_process_peak_includes_event_memory_samples(self) -> None:
        import threading

        code = (
            "import json; p='BENCH_EVENT '; "
            "print(p+json.dumps({'event':'worker_ready'}),flush=True); "
            "print(p+json.dumps({'event':'calculation_start','iteration':1}),flush=True); "
            "print(p+json.dumps({'event':'calculation_complete','iteration':1,"
            "'calculation_sec':0.001,'cpu_time_sec':0.001}),flush=True)"
        )

        # Event samples (reader thread) see more memory than the poll loop.
        def rss(_process):
            return (
                200
                if threading.current_thread() is not threading.main_thread()
                else 100
            )

        with mock.patch.object(run, "_rss_bytes", side_effect=rss):
            _, _, host = run._run_process_command(
                [sys.executable, "-c", code],
                adapter_name="fake",
                sample_interval=0.01,
            )
        self.assertEqual(host["calculation_peak_rss_bytes"], 200)
        self.assertGreaterEqual(host["host_peak_rss_bytes"], 200)

    def test_event_sample_after_worker_exit_keeps_last_positive_memory(self) -> None:
        import threading

        code = (
            "import json; p='BENCH_EVENT '; "
            "print(p+json.dumps({'event':'worker_ready'}),flush=True); "
            "print(p+json.dumps({'event':'calculation_start','iteration':1}),flush=True)"
        )

        # The reader thread finds the worker gone (0); the poll loop saw 150.
        def rss(_process):
            return (
                0 if threading.current_thread() is not threading.main_thread() else 150
            )

        with mock.patch.object(run, "_rss_bytes", side_effect=rss):
            _, _, host = run._run_process_command(
                [sys.executable, "-c", code],
                adapter_name="fake",
                sample_interval=0.01,
            )
        self.assertEqual(host["worker_ready_rss_bytes"], 150)
        self.assertEqual(host["calculation_peak_rss_bytes"], 150)

    def test_warmup_has_its_own_per_call_limit_and_setup_time(self) -> None:
        code = (
            "import json,time; p='BENCH_EVENT '; "
            "print(p+json.dumps({'event':'inputs_loaded'}),flush=True); "
            "time.sleep(0.05); "
            "print(p+json.dumps({'event':'warmup_start','maxrss_bytes':7}),flush=True); "
            "time.sleep(30)"
        )
        with self.assertRaises(run.AdapterTimeout) as caught:
            run._run_process_command(
                [sys.executable, "-c", code],
                adapter_name="fake",
                timeout=0.3,
                sample_interval=0.01,
                termination_grace=0.1,
            )
        details = caught.exception.details
        self.assertEqual(caught.exception.phase, "warmup")
        self.assertEqual(details["stop_call_index"], 0)
        self.assertGreaterEqual(details["censor_lower_bound_sec"], 0.3)
        self.assertGreater(details["setup_sec"], 0.0)
        self.assertEqual(details["worker_maxrss_last_bytes"], 7)

    def test_memory_limit_stops_the_process_tree(self) -> None:
        with self.assertRaises(run.AdapterMemoryExceeded) as caught:
            run._run_process_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                adapter_name="fake",
                timeout=30.0,
                memory_limit_bytes=1,
                sample_interval=0.01,
                termination_grace=0.1,
            )
        self.assertFalse(psutil.pid_exists(caught.exception.pid))
        self.assertGreater(caught.exception.details["memory_observed_bytes"], 1)

    def test_memory_stop_counts_memory_that_rss_leaves_out(self) -> None:
        with mock.patch.object(run, "tree_memory_in_use", return_value=10**12):
            with self.assertRaises(run.AdapterMemoryExceeded) as caught:
                run._run_process_command(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    adapter_name="fake",
                    timeout=30.0,
                    memory_limit_bytes=10**9,
                    sample_interval=0.01,
                    termination_grace=0.1,
                )
        self.assertEqual(caught.exception.details["memory_observed_bytes"], 10**12)

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_operating_system_kill_of_a_large_process_is_a_memory_stop(self) -> None:
        suicide = "import os, signal, time; time.sleep(0.3); os.kill(os.getpid(), signal.SIGKILL)"
        with mock.patch.object(run, "tree_memory_in_use", return_value=6 * 10**8):
            with self.assertRaises(run.AdapterMemoryExceeded) as caught:
                run._run_process_command(
                    [sys.executable, "-c", suicide],
                    adapter_name="fake",
                    timeout=30.0,
                    memory_limit_bytes=10**9,
                    sample_interval=0.01,
                )
        self.assertEqual(caught.exception.details["stopped_by"], "operating_system")
        with mock.patch.object(run, "tree_memory_in_use", return_value=1000):
            with self.assertRaises(run.AdapterProcessError):
                run._run_process_command(
                    [sys.executable, "-c", suicide],
                    adapter_name="fake",
                    timeout=30.0,
                    memory_limit_bytes=10**9,
                    sample_interval=0.01,
                )

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_terminal_hang_up_is_a_clean_stop_with_output_discarded(self) -> None:
        script = "\n".join(
            [
                "import os, signal, sys",
                "from bench.run import _SignalStop",
                "with _SignalStop() as stop:",
                "    os.kill(os.getpid(), signal.SIGHUP)",
                "    print('after the hang-up', flush=True)",
                "sys.exit(0 if stop.signal_number == signal.SIGHUP else 3)",
            ]
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path(run.__file__).resolve().parents[1],
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_stray_adapters_on_this_run_inputs_are_stopped(self) -> None:
        inputs = self.root / "run" / "inputs"
        sleeper = "import subprocess, sys, time; "
        stray = subprocess.Popen(
            [
                sys.executable,
                "-c",
                sleeper
                + "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                + "time.sleep(60)",
                "bench.adapters.fake_adapter",
                str(inputs / "ab" / "image.nii.gz"),
            ]
        )
        other = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
                "bench.adapters.fake_adapter",
                str(self.root / "other" / "inputs" / "image.nii.gz"),
            ]
        )
        try:
            deadline = time.monotonic() + 10
            while (
                not psutil.Process(stray.pid).children() and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            child = psutil.Process(stray.pid).children()[0]
            # On Windows a virtual-environment launcher adds a parent process.
            self.assertIn(stray.pid, run._stop_stray_adapters(inputs, 1.0))
            self.assertIsNotNone(stray.wait(timeout=5))
            self.assertFalse(child.is_running())
            self.assertIsNone(other.poll())
        finally:
            for process in (stray, other):
                process.kill()
                process.wait()

    def test_runtime_state_is_probed_only_for_a_required_state(self) -> None:
        battery = {
            "power_source": "Battery Power",
            "low_power_mode": 0,
            "sleep_assertion": True,
            "probe_errors": [],
        }
        required = {
            "power_source": "AC Power",
            "low_power_mode": 0,
            "sleep_assertion_during_calculation": True,
        }
        with mock.patch.object(run, "power_state", return_value=battery) as probe:
            self.assertEqual(run._runtime_state_errors({"host_settings": None}), [])
            probe.assert_not_called()
            errors = run._runtime_state_errors(
                {"host_settings": {"required_runtime_state": required}}
            )
        self.assertEqual(len(errors), 1)
        self.assertIn("Battery Power", errors[0])


class QcChecksTests(unittest.TestCase):
    def test_peak_above_free_memory_is_flagged(self) -> None:
        record = {
            "task_status": STATUS_MEASURED,
            "success": True,
            "duration_sec": 1.0,
            "feature_count": 3,
            "peak_rss_bytes": 200,
            "memory_available_bytes": 100,
        }
        result = run.run_qc_checks("run", [record])
        issue_types = {issue["issue_type"] for issue in result["issues"]}
        self.assertIn("peak_memory_above_free_memory", issue_types)


if __name__ == "__main__":
    unittest.main()
