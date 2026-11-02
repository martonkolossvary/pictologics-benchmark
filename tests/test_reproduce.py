from __future__ import annotations

import argparse
import ast
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import run
from bench.reproduce import __main__ as reproduce
from bench.reproduce import host_profile, stages
from scripts.launch_benchmark import _load_host_profile, build_commands

ROOT = Path(__file__).resolve().parents[1]


class HostProfileTests(unittest.TestCase):
    def test_generated_profile_passes_the_launcher_rules_without_the_hostname(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "host_profile.json"
            host_profile.write_host_profile(path, "linux-test-cpu-ab12")
            loaded = _load_host_profile(path)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(loaded["payload"]["machine_id"], "linux-test-cpu-ab12")
        self.assertNotIn(socket.gethostname(), text)

    def test_a_profile_is_never_rewritten_for_another_machine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "host_profile.json"
            host_profile.write_host_profile(path, "machine-a")
            with self.assertRaisesRegex(ValueError, "belongs to machine"):
                host_profile.write_host_profile(path, "machine-b")

    def test_suggested_ids_are_valid_and_not_the_hostname(self) -> None:
        suggested = host_profile.suggested_machine_id()
        self.assertRegex(suggested, r"^[a-z0-9][a-z0-9-]{0,63}$")
        self.assertNotIn(socket.gethostname().lower(), suggested)


class SettingsAndMarkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = Path(self.temporary.name) / "settings.json"
        patcher = mock.patch.object(reproduce, "SETTINGS", self.settings)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temporary.cleanup)

    def arguments(self, *, quick: bool = False, screens: bool = False):
        return argparse.Namespace(
            yes=True,
            result_root=self.temporary.name,
            machine_id=None,
            quick=quick,
            screens=screens,
        )

    def test_a_second_run_with_another_mode_is_refused(self) -> None:
        reproduce._settings(self.arguments(quick=True))
        self.assertTrue(json.loads(self.settings.read_text())["quick"])
        with self.assertRaisesRegex(stages.StageError, "already runs with screens"):
            reproduce._settings(self.arguments(screens=True))

    def test_markers_are_bound_to_the_release_lock_and_the_mode(self) -> None:
        context = stages.Context(
            result_root=Path(self.temporary.name),
            machine_id="machine",
            quick=False,
            screens=False,
            yes=True,
        )
        markers = stages.Markers(context)
        markers.mark("inputs")
        self.assertTrue(stages.Markers(context).done("inputs"))
        context.quick = True
        self.assertFalse(stages.Markers(context).done("inputs"))

    def test_a_stage_that_does_not_finish_its_work_runs_again(self) -> None:
        calls = []
        drawn = stages.Stage(
            "figures", "Draw", lambda c: calls.append("figures") or False, True
        )
        written = stages.Stage(
            "reports", "Write", lambda c: calls.append("reports"), True
        )
        with mock.patch.object(reproduce, "STAGES", (written, drawn)):
            arguments = ["--yes", "--result-root", self.temporary.name]
            self.assertEqual(reproduce.main(arguments), 0)
            self.assertEqual(reproduce.main(arguments), 0)
        self.assertEqual(calls, ["reports", "figures", "figures"])


class ComplianceStageTests(unittest.TestCase):
    def test_an_ibsi2_failure_is_reported_and_the_run_goes_on(self) -> None:
        def fake_bench(*arguments, step, **options):
            if "generate-ibsi2-candidates" in arguments:
                raise stages.StageError("IBSI 2 candidate maps failed with exit code 1")

        with tempfile.TemporaryDirectory() as temporary:
            context = stages.Context(
                result_root=Path(temporary),
                machine_id="m",
                quick=True,
                screens=False,
                yes=True,
            )
            with mock.patch.object(stages, "bench", side_effect=fake_bench):
                self.assertFalse(stages.compliance(context))
        self.assertIn("IBSI 2 conformance did not finish", context.notes[0])

    def test_an_ibsi1_failure_stops_the_run(self) -> None:
        def fake_bench(*arguments, step, **options):
            raise stages.StageError("IBSI 1 conformance failed with exit code 1")

        with tempfile.TemporaryDirectory() as temporary:
            context = stages.Context(
                result_root=Path(temporary),
                machine_id="m",
                quick=True,
                screens=False,
                yes=True,
            )
            with mock.patch.object(stages, "bench", side_effect=fake_bench):
                with self.assertRaises(stages.StageError):
                    stages.compliance(context)


class ReportAndFigureStageTests(unittest.TestCase):
    def run_failing(self, stage, *, quick: bool) -> tuple[stages.Context, object]:
        failure = stages.StageError("figures failed with exit code 1")
        context = stages.Context(
            result_root=Path("."), machine_id="m", quick=quick, screens=False, yes=True
        )
        with (
            mock.patch.object(stages, "run", side_effect=failure),
            mock.patch.object(stages, "bench", side_effect=failure),
            mock.patch("bench.figures.style.require_font"),
        ):
            return context, stage(context)

    def test_a_full_run_notes_the_failure_and_goes_on_to_its_package(self) -> None:
        for stage in (stages.reports, stages.figures):
            context, finished = self.run_failing(stage, quick=False)
            self.assertFalse(finished)
            self.assertIn("The result package does not need them", context.notes[0])

    def test_a_quick_run_stops_on_the_failure(self) -> None:
        for stage in (stages.reports, stages.figures):
            with self.assertRaises(stages.StageError):
                self.run_failing(stage, quick=True)


class IBSI2MemoryCapTests(unittest.TestCase):
    def candidate_options(self, system: str) -> dict:
        calls = {}

        def fake_bench(*arguments, step, **options):
            calls[arguments[1]] = options

        context = stages.Context(
            result_root=Path("."), machine_id="m", quick=True, screens=False, yes=True
        )
        with (
            mock.patch.object(stages.platform, "system", return_value=system),
            mock.patch.object(stages, "bench", side_effect=fake_bench),
            mock.patch.object(stages, "address_space_cap", return_value="cap"),
            mock.patch.object(Path, "is_dir", return_value=True),
        ):
            stages.compliance(context)
        return calls

    def test_linux_caps_only_the_ibsi2_candidate_maps(self) -> None:
        calls = self.candidate_options("Linux")
        self.assertEqual(calls["generate-ibsi2-candidates"], {"preexec_fn": "cap"})
        self.assertEqual(calls["run-ibsi1"], {})

    def test_other_systems_run_without_a_cap(self) -> None:
        for system in ("Darwin", "Windows"):
            calls = self.candidate_options(system)
            self.assertEqual(calls["generate-ibsi2-candidates"], {})

    @unittest.skipUnless(sys.platform == "linux", "RLIMIT_AS works only on Linux")
    def test_the_cap_turns_a_large_allocation_into_memory_error(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "bytearray(2 * 1024**3)"],
            preexec_fn=stages.address_space_cap(1024**3),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MemoryError", result.stderr)


class WorkspaceStageTests(unittest.TestCase):
    def build_environment(self, system: str) -> dict[str, str] | None:
        context = stages.Context(
            result_root=Path("."), machine_id="m", quick=True, screens=False, yes=True
        )
        with (
            mock.patch.object(stages.platform, "system", return_value=system),
            mock.patch.object(stages, "run") as fake_run,
            mock.patch.object(stages.release_lock, "verify"),
        ):
            stages.workspace(context)
        return fake_run.call_args.kwargs["env"]

    def test_linux_builds_without_numpy_avx512(self) -> None:
        env = self.build_environment("Linux")
        self.assertEqual(env["NPY_DISABLE_CPU_FEATURES"], stages.NUMPY_AVX512_OFF)

    def test_other_systems_build_with_the_normal_environment(self) -> None:
        self.assertIsNone(self.build_environment("Darwin"))
        self.assertIsNone(self.build_environment("Windows"))


class QuickLaunchTests(unittest.TestCase):
    def test_quick_commands_use_the_screen_contract_on_small_cases(self) -> None:
        workspace = {
            "launch_policy": {
                "speed_guardrail_baseline": "pictologics",
                "checkpoint_interval_tasks": 25,
                "per_call_timeout_seconds": 1800,
            },
            "screen_launch_policy": {"per_call_timeout_seconds": 600},
            "adapter_order": ["pictologics"],
            "endpoint_contract": {"path": "configs/benchmark/main.json"},
            "screen_contract": {"path": "configs/benchmark/screen.json"},
            "task_inventory": {
                "fresh_process_repeats": 3,
                "measured_calls_per_process": 3,
                "adapter_count": 1,
                "family_count": 5,
            },
            "screen_task_inventory": {
                "fresh_process_repeats": 1,
                "measured_calls_per_process": 1,
                "adapter_count": 1,
                "family_count": 5,
            },
            "datasets": {pillar: {"case_count": 100} for pillar in stages.PILLARS},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo" / "data" / "benchmark"
            commands = build_commands(
                root,
                workspace,
                result_root=Path(temporary) / "results",
                dry_run=True,
                machine_id="machine",
                quick=True,
            )
        first = commands[0]
        self.assertTrue(
            first[first.index("--endpoint-contract") + 1].endswith("screen.json")
        )
        self.assertIn(
            "/quick/", first[first.index("--report-dir") + 1].replace("\\", "/")
        )
        self.assertEqual(first[first.index("--timeout") + 1], "300.0")
        self.assertEqual(first[first.index("--sizes") + 1], "32,64")
        self.assertEqual(first[first.index("--project-total-tasks") + 1], str(12 * 5))


class IsolationTests(unittest.TestCase):
    def test_timed_code_never_imports_the_runner_or_the_submission_code(self) -> None:
        bound = {*run._EXECUTION_SOURCE_FILES, "bench/benchmark_workspace.py"}
        bound.update(
            f"bench/adapters/{name}_adapter.py"
            for name in ("pictologics", "pyradiomics", "mirp", "medimage", "zrad")
        )
        for relative in sorted(path for path in bound if path.endswith(".py")):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(
                        node.module.startswith(("bench.reproduce", "bench.submission")),
                        relative,
                    )


if __name__ == "__main__":
    unittest.main()
