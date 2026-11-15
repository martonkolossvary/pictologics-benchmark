"""Run the timing benchmark for Pictologics alone, to compare hardware.

``scripts/launch_benchmark.py`` belongs to the release lock and always times the
five libraries.  This launcher reuses its checks and builds the same controller
commands with it.  It then changes only what a one-library run must change: the
library list, the result folder, the run ID and the progress counters.  The
controller, the contract, the cases, the repeats, the time limit, the memory
budget and the skip rules stay those of the full run, so the Pictologics times
of both runs measure the same tasks.

Usage (the runner calls it): ``python -m bench.reproduce.pictologics_only
--result-root DIR --host-profile FILE [--quick] [--execute --confirm CALCULATE]``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bench.benchmark_ledger import sha256_file
from bench.reproduce.stages import PICTOLOGICS, PICTOLOGICS_FOLDER
from scripts import launch_benchmark as launcher

# The only options in which a Pictologics-only command differs from the
# command of the full run.
CHANGED_OPTIONS = (
    "--adapters",
    "--report-dir",
    "--run-id",
    "--project-total-tasks",
    "--project-task-offset",
)


def run_root(machine_root: Path, *, quick: bool) -> Path:
    root = machine_root / PICTOLOGICS_FOLDER
    return root / "quick" if quick else root


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _set(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def pictologics_commands(
    commands: list[list[str]],
    workspace: dict[str, Any],
    *,
    machine_root: Path,
    quick: bool,
) -> list[list[str]]:
    """Turn the release launcher's commands into Pictologics-only commands."""

    adapters = list(workspace["adapter_order"])
    if PICTOLOGICS not in adapters:
        raise ValueError("the workspace does not schedule Pictologics")
    if len(commands) != len(launcher.PILLARS):
        raise ValueError("expected one controller command for each pillar")
    count = len(adapters)
    folder = run_root(machine_root.expanduser().resolve(), quick=quick)
    result = []
    for (pillar, _), original in zip(launcher.PILLARS, commands):
        if _value(original, "--adapters").split(",") != adapters:
            raise ValueError(f"unexpected library list in the {pillar} command")
        command = list(original)
        _set(command, "--adapters", PICTOLOGICS)
        _set(command, "--report-dir", str(folder / pillar))
        _set(command, "--run-id", f"{_value(original, '--run-id')}_{PICTOLOGICS}")
        # Every case, group and repeat has one task for each library.
        for option in ("--project-total-tasks", "--project-task-offset"):
            tasks = int(_value(original, option))
            if tasks % count:
                raise ValueError(f"{option} {tasks} is not a multiple of {count}")
            _set(command, option, str(tasks // count))
        result.append(command)
    return result


def _path_problem(preflight: dict[str, Any], machine_root: Path, quick: bool) -> str:
    """The release launcher checks the path length of the full run's folder."""

    extra = len(str(run_root(machine_root, quick=quick))) - len(str(machine_root))
    longest = int(preflight["estimated_max_path_chars"]) + extra
    if os.name == "nt" and longest > launcher.WINDOWS_PATH_BUDGET:
        return (
            f"estimated result path length {longest} exceeds the "
            f"{launcher.WINDOWS_PATH_BUDGET}-character native-library safety budget"
        )
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default="data/benchmark")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--host-profile", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args(argv)
    if args.execute and args.confirm != "CALCULATE":
        parser.error("--execute requires --confirm CALCULATE")

    root = Path(args.workspace_root).expanduser().resolve()
    workspace = launcher._load_workspace(root)
    profile = launcher._load_host_profile(args.host_profile)
    payload = profile["payload"]
    machine_id = launcher._validate_machine_id(str(payload["machine_id"]))
    result_root = Path(args.result_root).expanduser().resolve()
    machine_root = result_root / machine_id
    preflight = launcher._result_preflight(
        root,
        workspace,
        result_root=result_root,
        machine_id=machine_id,
        allow_synced_results=False,
        host_profile=profile,
    )
    problem = _path_problem(preflight, machine_root, args.quick)
    if problem:
        preflight["errors"].append(problem)
        preflight["status"] = "fail"
    if args.execute and preflight["status"] != "pass":
        parser.error("execution preflight failed: " + "; ".join(preflight["errors"]))
    commands = launcher.build_commands(
        root,
        workspace,
        result_root=result_root,
        dry_run=False,
        machine_id=machine_id,
        machine_label=payload["machine_label"],
        cpu_model=payload["cpu_model"],
        cpu_base_ghz=payload.get("cpu_base_ghz"),
        host_profile_id=str(payload["profile_id"]),
        host_profile_sha256=str(profile["sha256"]),
        # The controller re-checks the required power state before each task.
        host_settings={
            **payload["benchmark_settings"],
            "required_runtime_state": payload["required_runtime_state"],
        },
        quick=args.quick,
    )
    commands = pictologics_commands(
        commands, workspace, machine_root=machine_root, quick=args.quick
    )
    plan = {
        "workspace_manifest_sha256": sha256_file(root / "workspace_manifest.json"),
        "run_kind": "pictologics_only_quick" if args.quick else "pictologics_only",
        "adapters": [PICTOLOGICS],
        "mode": "execute" if args.execute else "print_only",
        "radiomic_calculation_started": bool(args.execute),
        "machine": {
            "machine_id": machine_id,
            "machine_label": payload["machine_label"],
            "host_profile_id": payload["profile_id"],
            "host_profile_sha256": profile["sha256"],
        },
        "execution_preflight": preflight,
        "commands": commands,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    for command in commands:
        returncode = launcher._run_controller(command)
        if returncode != 0:
            return returncode if returncode > 0 else 128 - returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
