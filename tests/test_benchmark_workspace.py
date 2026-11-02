from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest

from bench.benchmark_contract import (
    default_screen_contract_path,
    load_benchmark_contract,
)
from bench.benchmark_ledger import sha256_file
from bench.benchmark_workspace import _task_inventory
from bench.host_state import ac_low_power_mode
from scripts.launch_benchmark import (
    _known_sync_root,
    _load_host_profile,
    _portable_command,
    _profile_value,
    _run_controller,
    _validate_machine_id,
    build_commands,
)


class BenchmarkWorkspaceTests(unittest.TestCase):
    @staticmethod
    def _workspace() -> dict:
        return {
            "launch_policy": {
                "speed_guardrail_baseline": "pictologics",
                "per_call_timeout_seconds": 1800.0,
                "checkpoint_interval_tasks": 25,
                "progress_interval_seconds": 15.0,
            },
            "screen_launch_policy": {"per_call_timeout_seconds": 600.0},
            "endpoint_contract": {"path": "contract.json"},
            "screen_contract": {"path": "screen.json"},
            "one_library_thread_screen_contract": {"path": "screen_one_thread.json"},
            "adapter_order": ["pictologics", "pyradiomics"],
            "task_inventory": {
                "fresh_process_repeats": 3,
                "measured_calls_per_process": 3,
                "adapter_count": 2,
                "family_count": 2,
            },
            "screen_task_inventory": {
                "fresh_process_repeats": 1,
                "measured_calls_per_process": 1,
                "adapter_count": 2,
                "family_count": 2,
            },
            "datasets": {
                "pillar1_morphology": {"case_count": 2},
                "pillar2_whole_anatomy": {"case_count": 3},
                "pillar3_ibsi2_phase3": {"case_count": 5},
            },
        }

    def test_launcher_scales_project_progress_for_appended_repeats(self) -> None:
        workspace = self._workspace()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data" / "benchmark"
            root.mkdir(parents=True)
            commands = build_commands(
                root,
                workspace,
                result_root=Path(temporary) / "results",
                dry_run=True,
                machine_id="test-host",
                host_profile_id="test-host",
                host_profile_sha256="a" * 64,
                host_settings={"power_source": "AC Power"},
                repeats=5,
            )
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertIn("--extend-repeats", command)
            self.assertEqual(command[command.index("--repeats") + 1], "5")
            self.assertEqual(command[command.index("--project-total-tasks") + 1], "200")
            self.assertIn("--disable-speed-truncation", command)
            self.assertNotIn("--guardrail-skip-ratio", command)
            self.assertEqual(command[command.index("--timeout") + 1], "1800.0")
            self.assertEqual(command[command.index("--machine-id") + 1], "test-host")
            self.assertEqual(
                command[command.index("--host-profile-id") + 1], "test-host"
            )
            self.assertEqual(
                command[command.index("--host-profile-sha256") + 1], "a" * 64
            )
            self.assertEqual(
                json.loads(command[command.index("--host-settings-json") + 1]),
                {"power_source": "AC Power"},
            )
            report_dir = Path(command[command.index("--report-dir") + 1])
            self.assertEqual(report_dir.parent.name, "test-host")
        offsets = [
            int(command[command.index("--project-task-offset") + 1])
            for command in commands
        ]
        self.assertEqual(offsets, [0, 40, 100])

    def test_screen_commands_use_the_screen_contract_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data" / "benchmark"
            root.mkdir(parents=True)
            arguments = dict(
                result_root=Path(temporary) / "results",
                dry_run=True,
                machine_id="test-host",
                screen=True,
            )
            commands = build_commands(root, self._workspace(), **arguments)
            with self.assertRaisesRegex(ValueError, "single repeat"):
                build_commands(root, self._workspace(), repeats=3, **arguments)
        for command in commands:
            self.assertTrue(command[command.index("--run-id") + 1].endswith("_screen"))
            self.assertTrue(
                command[command.index("--endpoint-contract") + 1].endswith(
                    "screen.json"
                )
            )
            self.assertEqual(command[command.index("--repeats") + 1], "1")
            self.assertEqual(command[command.index("--timing-iterations") + 1], "1")
            self.assertEqual(command[command.index("--timeout") + 1], "600.0")
            report_dir = Path(command[command.index("--report-dir") + 1])
            self.assertEqual(report_dir.parent.name, "screen")

    def test_one_library_thread_screen_is_separately_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data" / "benchmark"
            root.mkdir(parents=True)
            arguments = dict(
                result_root=Path(temporary) / "results",
                dry_run=True,
                machine_id="test-host",
            )
            commands = build_commands(
                root,
                self._workspace(),
                screen=True,
                one_library_thread=True,
                **arguments,
            )
            with self.assertRaisesRegex(ValueError, "screen-only"):
                build_commands(
                    root, self._workspace(), one_library_thread=True, **arguments
                )
        for command in commands:
            self.assertTrue(
                command[command.index("--run-id") + 1].endswith(
                    "_screen_one_library_thread"
                )
            )
            self.assertTrue(
                command[command.index("--endpoint-contract") + 1].endswith(
                    "screen_one_thread.json"
                )
            )
            report_dir = Path(command[command.index("--report-dir") + 1])
            self.assertEqual(report_dir.parent.name, "screen_one_library_thread")

    def test_one_library_thread_contract_differs_only_in_thread_policy(self) -> None:
        from bench.benchmark_contract import (
            default_screen_contract_path,
            load_benchmark_contract,
            one_library_thread_screen_contract_path,
        )

        native = dict(load_benchmark_contract(default_screen_contract_path()).payload)
        single = dict(
            load_benchmark_contract(one_library_thread_screen_contract_path()).payload
        )
        self.assertEqual(single["timing"].pop("thread_policy"), "one_library_thread")
        self.assertEqual(
            native["timing"].pop("thread_policy"), "adapter_native_threads"
        )
        for payload in (native, single):
            payload.pop("contract_id")
            payload.pop("status")
        self.assertEqual(native, single)

    def test_group_members_follow_the_order_adapters_report(self) -> None:
        from bench.adapters.registry import ADAPTERS, validate_families
        from bench.ibsi_families import WORKLOAD_GROUPS

        for group, members in WORKLOAD_GROUPS.items():
            for adapter, capabilities in ADAPTERS.items():
                supported, _ = validate_families(adapter, list(reversed(members)))
                with self.subTest(group=group, adapter=adapter):
                    self.assertEqual(supported, capabilities.supported_members(group))

    def test_low_power_probe_reads_old_and_new_pmset_keys(self) -> None:
        def custom(ac_line: str) -> str:
            return (
                "Battery Power:\n powermode            1\n"
                f"AC Power:\n {ac_line}\n sleep                1\n"
            )

        # Only the AC Power block counts; the battery block differs on purpose.
        self.assertEqual(ac_low_power_mode(custom("lowpowermode         0")), 0)
        self.assertEqual(ac_low_power_mode(custom("lowpowermode         1")), 1)
        self.assertEqual(ac_low_power_mode(custom("powermode            0")), 0)
        self.assertEqual(ac_low_power_mode(custom("powermode            1")), 1)
        self.assertEqual(ac_low_power_mode(custom("powermode            2")), 0)
        self.assertIsNone(ac_low_power_mode(custom("sleep                0")))
        self.assertIsNone(ac_low_power_mode("Battery Power:\n powermode 1\n"))

    def test_machine_id_rejects_path_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "machine ID"):
            _validate_machine_id("../windows")

    def test_host_profile_is_checksum_bound_and_rejects_identity_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "host.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile_id": "test-host",
                        "machine_id": "test-host",
                        "machine_label": "Test host",
                        "cpu_model": "Test CPU",
                        "expected_hardware": {},
                        "required_runtime_state": {},
                        "benchmark_settings": {},
                    }
                ),
                encoding="utf-8",
            )
            profile = _load_host_profile(path)
            expected_sha256 = sha256_file(path)

        self.assertEqual(profile["sha256"], expected_sha256)
        self.assertEqual(
            _profile_value(None, profile, "machine_id", flag="--machine-id"),
            "test-host",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _profile_value("different-host", profile, "machine_id", flag="--machine-id")

    def test_cloud_storage_path_is_detected(self) -> None:
        path = Path("/Users/test/Library/CloudStorage/OneDrive-Personal/results")
        self.assertEqual(_known_sync_root(path), "path_marker")

    @unittest.skipIf(os.name == "nt", "POSIX paths")
    def test_portable_command_keeps_repository_symlink_path_lexical(self) -> None:
        repository = Path("/workspace/project")
        result_root = Path("/local/results")
        command = [
            "/workspace/project/.venv/bin/python",
            "--report-dir",
            "/local/results/host/pillar",
        ]
        self.assertEqual(
            _portable_command(command, repository, result_root),
            [
                ".venv/bin/python",
                "--report-dir",
                "{RESULT_ROOT}/host/pillar",
            ],
        )

    def test_task_inventory_distinguishes_scheduled_and_eligible(self) -> None:
        contract = load_benchmark_contract()
        datasets = {
            "pillar1_morphology": {"case_count": 1},
            "pillar2_whole_anatomy": {"case_count": 2},
            "pillar3_ibsi2_phase3": {"case_count": 3},
        }
        result = _task_inventory(datasets, contract)

        self.assertEqual(result["totals"]["case_count"], 6)
        # Five adapters x six groups x three repeats.  PyRadiomics has no
        # local intensity, no IVH and no Moran's I or Geary's C group.
        self.assertEqual(result["totals"]["scheduled_task_records"], 540)
        self.assertEqual(result["totals"]["eligible_calculation_tasks"], 486)
        self.assertEqual(result["totals"]["preempted_unsupported_tasks"], 54)
        self.assertEqual(
            result["totals"]["measured_calls_if_all_eligible_complete"], 1458
        )
        self.assertEqual(len(result["supported_families_by_adapter"]["pyradiomics"]), 3)
        self.assertEqual(len(result["supported_families_by_adapter"]["pictologics"]), 6)

        screen = _task_inventory(
            datasets, load_benchmark_contract(default_screen_contract_path())
        )
        self.assertEqual(screen["totals"]["eligible_calculation_tasks"], 162)
        self.assertEqual(
            screen["totals"]["measured_calls_if_all_eligible_complete"], 162
        )

    def test_launch_loader_rejects_changed_dataset_manifest(self) -> None:
        from scripts.launch_benchmark import _load_workspace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets = {}
            for pillar, directory in (
                ("pillar1_morphology", "pillar1"),
                ("pillar2_whole_anatomy", "pillar2_a1"),
                ("pillar3_ibsi2_phase3", "ibsi2_phase3"),
            ):
                path = root / directory
                path.mkdir()
                manifest = path / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                datasets[pillar] = {
                    "path": directory,
                    "manifest_sha256": sha256_file(manifest),
                }
            workspace = {
                "schema_version": 1,
                "benchmark_timing_executed": False,
                "datasets": datasets,
            }
            (root / "workspace_manifest.json").write_text(
                json.dumps(workspace), encoding="utf-8"
            )
            _load_workspace(root)
            (root / "pillar1" / "manifest.json").write_text(
                '{"changed":true}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "changed after validation"):
                _load_workspace(root)

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_launcher_waits_for_the_controller_after_ctrl_c(self) -> None:
        controller = [
            sys.executable,
            "-c",
            "import sys, time; time.sleep(1.5); sys.exit(130)",
        ]
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT)).start()
        started = time.monotonic()
        self.assertEqual(_run_controller(controller), 130)
        self.assertGreaterEqual(time.monotonic() - started, 1.4)

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_launcher_forwards_termination_to_the_controller(self) -> None:
        controller = [
            sys.executable,
            "-c",
            "import signal, sys, time; "
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(143)); "
            "time.sleep(30)",
        ]
        threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM)).start()
        self.assertEqual(_run_controller(controller), 143)


if __name__ == "__main__":
    unittest.main()
