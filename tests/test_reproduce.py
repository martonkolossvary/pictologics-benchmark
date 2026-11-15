from __future__ import annotations

import argparse
import ast
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench import run
from bench.reproduce import __main__ as reproduce
from bench.reproduce import host_profile, pictologics_only, stages
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

    def arguments(
        self,
        *,
        quick: bool = False,
        screens: bool = False,
        pictologics_only: bool = False,
    ):
        return argparse.Namespace(
            yes=True,
            result_root=self.temporary.name,
            machine_id=None,
            quick=quick,
            screens=screens,
            pictologics_only=pictologics_only,
        )

    def test_a_second_run_with_another_mode_is_refused(self) -> None:
        reproduce._settings(self.arguments(quick=True))
        self.assertTrue(json.loads(self.settings.read_text())["quick"])
        with self.assertRaisesRegex(stages.StageError, "already runs with screens"):
            reproduce._settings(self.arguments(screens=True))

    def test_a_started_full_run_never_switches_to_pictologics_only(self) -> None:
        settings = reproduce._settings(self.arguments())
        self.assertFalse(json.loads(self.settings.read_text())["pictologics_only"])
        machine_root = Path(settings["result_root"]) / settings["machine_id"]
        machine_root.mkdir(parents=True)
        (machine_root / stages.RUN_KIND_FILE).write_text('{"run_kind": "full"}')
        with self.assertRaisesRegex(
            stages.StageError, "already runs with pictologics_only = False"
        ):
            reproduce._settings(self.arguments(pictologics_only=True))

    def test_a_full_run_that_never_started_can_switch(self) -> None:
        # For example, the full run stopped at its compiler check.
        full = self.arguments()
        full.result_root = None
        solo = self.arguments(pictologics_only=True)
        solo.result_root = None
        with mock.patch.object(Path, "home", return_value=Path(self.temporary.name)):
            reproduce._settings(full)
            switched = reproduce._settings(solo)
        self.assertTrue(switched["pictologics_only"])
        self.assertEqual(Path(switched["result_root"]).name, "pictologics-only-results")
        self.assertTrue(json.loads(self.settings.read_text())["pictologics_only"])

    def test_a_pictologics_only_run_suggests_its_own_result_folder(self) -> None:
        arguments = self.arguments(pictologics_only=True)
        arguments.result_root = None
        with mock.patch.object(Path, "home", return_value=Path(self.temporary.name)):
            settings = reproduce._settings(arguments)
        self.assertEqual(Path(settings["result_root"]).name, "pictologics-only-results")

    def test_a_pictologics_only_folder_keeps_its_mode(self) -> None:
        reproduce._settings(self.arguments(pictologics_only=True))
        # The resume command may leave out the option.
        self.assertTrue(reproduce._settings(self.arguments())["pictologics_only"])

    def test_settings_from_before_the_option_belong_to_a_full_run(self) -> None:
        self.settings.write_text(
            json.dumps(
                {
                    "result_root": str(Path(self.temporary.name).resolve()),
                    "machine_id": "machine",
                    "quick": False,
                    "screens": False,
                    "first_start": {},
                }
            ),
            encoding="utf-8",
        )
        self.assertFalse(reproduce._settings(self.arguments())["pictologics_only"])
        # Its run has started: the machine folder holds its host profile.
        machine_root = Path(self.temporary.name) / "machine"
        machine_root.mkdir()
        (machine_root / "host_profile.json").write_text("{}")
        with self.assertRaises(stages.StageError):
            reproduce._settings(self.arguments(pictologics_only=True))

    def test_the_screens_are_refused_with_pictologics_only(self) -> None:
        with self.assertRaises(SystemExit):
            reproduce.main(["--pictologics-only", "--screens", "--status"])

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


def _context(result_root: Path, **options) -> stages.Context:
    return stages.Context(
        result_root=result_root,
        machine_id="machine",
        quick=False,
        screens=False,
        yes=True,
        **options,
    )


class PictologicsOnlyStageTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.full = _context(self.root)
        self.solo = _context(self.root, pictologics_only=True)

    def test_the_layout_is_that_of_the_full_run_with_its_own_stage_records(
        self,
    ) -> None:
        # The same depth as the full run keeps the Windows path budget.
        for context in (self.full, self.solo):
            self.assertEqual(context.run_root, self.root / "machine")
            self.assertEqual(
                context.compliance_root, self.root / "machine" / "compliance"
            )
        self.assertNotEqual(
            stages._release_digest(self.full), stages._release_digest(self.solo)
        )

    def test_a_machine_folder_never_holds_both_kinds_of_run(self) -> None:
        stages.claim_machine_folder(self.solo)
        stages.claim_machine_folder(self.solo)
        with self.assertRaisesRegex(stages.StageError, "holds a pictologics-only run"):
            stages.claim_machine_folder(self.full)
        other = _context(self.root / "other")
        stages.claim_machine_folder(other)
        with self.assertRaisesRegex(stages.StageError, "holds a full run"):
            stages.claim_machine_folder(replace(other, pictologics_only=True))

    def test_the_stage_records_name_the_kind_without_run_kind_json(self) -> None:
        markers = stages.Markers(self.solo)
        markers.mark("inputs")
        self.assertEqual(
            stages.recorded_run_kind(self.solo.machine_root, self.solo),
            "pictologics_only",
        )
        with self.assertRaisesRegex(stages.StageError, "holds a pictologics-only run"):
            stages.claim_machine_folder(self.full)

    def test_a_full_run_from_before_the_option_keeps_its_folder(self) -> None:
        (self.root / "machine" / "pillar1_morphology").mkdir(parents=True)
        stages.claim_machine_folder(self.full)
        with self.assertRaisesRegex(stages.StageError, "holds a full run"):
            stages.claim_machine_folder(self.solo)
        self.assertFalse((self.root / "machine" / stages.RUN_KIND_FILE).exists())

    def test_only_the_screens_and_the_figures_are_left_out(self) -> None:
        skipped = [s.name for s in stages.STAGES if not stages.applies(s, self.solo)]
        self.assertEqual(skipped, ["screens", "figures"])
        skipped = [s.name for s in stages.STAGES if not stages.applies(s, self.full)]
        self.assertEqual(skipped, ["screens"])

    def problems(self, context: stages.Context) -> str:
        with (
            mock.patch.object(stages, "REPO", self.root),
            mock.patch.object(
                stages, "native_platform_key", return_value="Linux-x86_64"
            ),
            mock.patch.object(
                stages, "_compiler_problems", return_value=["install a compiler"]
            ),
            mock.patch.object(stages.shutil, "which", return_value="/usr/bin/git"),
            mock.patch.object(
                stages.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=100 * stages.GIB),
            ),
            mock.patch("scripts.launch_benchmark._known_sync_root", return_value=None),
        ):
            try:
                stages.prerequisites(context)
            except stages.StageError as exc:
                return str(exc)
        return ""

    def test_no_c_compiler_is_needed(self) -> None:
        self.assertIn("install a compiler", self.problems(self.full))
        self.assertEqual(self.problems(self.solo), "")
        # Only a passed check claims the machine folder.
        kind = json.loads((self.root / "machine" / stages.RUN_KIND_FILE).read_text())
        self.assertEqual(kind, {"run_kind": "pictologics_only"})

    def test_only_the_pictologics_environment_is_built(self) -> None:
        profiles = {"pictologics": "picto-profile", "mirp": "mirp-profile"}
        with (
            mock.patch("bench.env.load_runtime_profiles", return_value=profiles),
            mock.patch("bench.env.create_profile") as create,
            mock.patch.object(stages, "bench") as verify,
        ):
            stages.environments(self.solo)
        create.assert_called_once_with("picto-profile")
        self.assertEqual(
            verify.call_args.args, ("env", "verify", "--profiles", "pictologics")
        )

    def test_conformance_runs_for_pictologics_alone(self) -> None:
        calls = []

        def fake_bench(*arguments, step, **options):
            calls.append(arguments)

        with (
            mock.patch.object(stages, "bench", side_effect=fake_bench),
            mock.patch.object(Path, "is_dir", return_value=False),
        ):
            self.assertTrue(stages.compliance(self.solo))
        self.assertEqual(len(calls), 6)
        for arguments in calls:
            self.assertEqual(arguments[-2:], ("--adapters", "pictologics"))
            output = arguments[arguments.index("--output-dir") + 1]
            self.assertTrue(Path(output).is_relative_to(self.solo.compliance_root))

    def test_the_timing_stage_uses_the_pictologics_launcher(self) -> None:
        context = replace(self.solo, host_profile=self.root / "host_profile.json")
        with mock.patch.object(
            stages.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as fake:
            stages.timing_run(context)
        command = fake.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "bench.reproduce.pictologics_only"])
        self.assertIn("--execute", command)


class PictologicsOnlyLaunchTests(unittest.TestCase):
    WORKSPACE = {
        "launch_policy": {
            "speed_guardrail_baseline": "pictologics",
            "checkpoint_interval_tasks": 25,
            "per_call_timeout_seconds": 1800,
        },
        "adapter_order": ["pictologics", "mirp"],
        "endpoint_contract": {"path": "configs/benchmark/main.json"},
        "task_inventory": {
            "fresh_process_repeats": 3,
            "measured_calls_per_process": 3,
            "adapter_count": 2,
            "family_count": 5,
        },
        "datasets": {pillar: {"case_count": 10} for pillar in stages.PILLARS},
    }

    def test_only_the_libraries_run_id_and_counters_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo" / "data" / "benchmark"
            commands = build_commands(
                root,
                self.WORKSPACE,
                result_root=Path(temporary) / "results",
                dry_run=False,
                machine_id="machine",
            )
            solo = pictologics_only.pictologics_commands(commands, self.WORKSPACE)
        for pillar, original, changed in zip(stages.PILLARS, commands, solo):
            self.assertEqual(len(original), len(changed))
            differing = {
                original[index - 1]
                for index, (old, new) in enumerate(zip(original, changed))
                if old != new
            }
            self.assertLessEqual(differing, set(pictologics_only.CHANGED_OPTIONS))

            def value(option: str, command=changed) -> str:
                return command[command.index(option) + 1]

            self.assertEqual(value("--adapters"), "pictologics")
            self.assertEqual(value("--report-dir"), value("--report-dir", original))
            self.assertEqual(value("--run-id"), f"{pillar}_main_pictologics")
            self.assertEqual(
                value("--endpoint-contract"), value("--endpoint-contract", original)
            )
            self.assertEqual(value("--timeout"), value("--timeout", original))
        # Pillars x cases x groups x repeats, now for one library.
        self.assertEqual(solo[0][solo[0].index("--project-total-tasks") + 1], "450")
        self.assertEqual(solo[2][solo[2].index("--project-task-offset") + 1], "300")

    def test_a_workspace_without_pictologics_is_refused(self) -> None:
        workspace = {**self.WORKSPACE, "adapter_order": ["mirp"]}
        with self.assertRaisesRegex(ValueError, "does not schedule Pictologics"):
            pictologics_only.pictologics_commands([], workspace)


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
