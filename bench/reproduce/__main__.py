"""Run the whole benchmark on this computer: ``reproduce.sh`` or ``reproduce.ps1``.

Run the same command again to resume.  The first run asks for a machine ID and
a result folder and stores them in ``.reproduce/settings.json``.  With
``--pictologics-only``, the run times Pictologics alone on the tasks of the full
run, to compare hardware.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from bench.benchmark_ledger import atomic_write_json
from bench.reproduce.host_profile import suggested_machine_id
from bench.reproduce.stages import (
    PILLARS,
    REPO,
    RESULT_FOLDERS,
    RUN_KIND_FILE,
    STAGES,
    Context,
    Markers,
    StageError,
    applies,
)

SETTINGS = REPO / ".reproduce" / "settings.json"
CHECKLIST = """
Before the timing run:
  1. Connect the computer to mains power.
  2. Close other programs that use the CPU or the memory.
  3. Turn off automatic operating-system updates until the run ends.
  4. Do not move or rename this folder, and do not run git pull in it,
     until the run ends.
  5. To stop, press Ctrl-C once. Run the same command again to resume.
"""


def _ask(question: str, default: str, yes: bool) -> str:
    if yes:
        return default
    answer = input(f"{question} [{default}]: ").strip()
    return answer or default


def _git(*arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments], cwd=REPO, capture_output=True, text=True, check=False
    )
    return process.stdout.strip()


def _default_result_root(pictologics_only: bool) -> Path:
    # Not the folder that "git clone" makes in the home folder.  A
    # Pictologics-only run has its own result folder.
    folder = RESULT_FOLDERS[pictologics_only]
    if platform.system() == "Windows":
        return Path.home() / "AppData" / "Local" / folder
    return Path.home() / folder


def _switchable(stored: dict) -> bool:
    """Whether a full-run clone has not started: no screen, run or package."""

    machine_root = Path(stored["result_root"]) / stored["machine_id"]
    return (
        not stored.get("quick")
        and not stored.get("screens")
        and not any(
            (machine_root / name).exists()
            for name in (RUN_KIND_FILE, "reproduce_stages.json", "host_profile.json")
        )
    )


def _switch_to_pictologics_only(stored: dict) -> dict:
    switched = {**stored, "pictologics_only": True}
    if Path(stored["result_root"]) == _default_result_root(False).resolve():
        switched["result_root"] = str(_default_result_root(True).resolve())
    atomic_write_json(SETTINGS, switched)
    print(f"This clone now runs only Pictologics; results: {switched['result_root']}")
    return switched


def _settings(args: argparse.Namespace) -> dict:
    stored = (
        json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.is_file() else {}
    )
    if stored:
        # Settings from before the Pictologics-only mode belong to a full run.
        stored.setdefault("pictologics_only", False)
    pictologics_only = bool(getattr(args, "pictologics_only", False))
    requested = {
        "result_root": (
            str(Path(args.result_root).expanduser().resolve())
            if args.result_root
            else None
        ),
        "machine_id": args.machine_id,
        "quick": True if args.quick else None,
        "screens": True if args.screens else None,
        "pictologics_only": True if pictologics_only else None,
    }
    if (
        stored
        and requested["pictologics_only"] is not None
        and not stored["pictologics_only"]
        and _switchable(stored)
    ):
        # A clone may become a Pictologics-only clone until its run starts,
        # for example when the full run asked for a missing C compiler.
        stored = _switch_to_pictologics_only(stored)
    for key, value in requested.items():
        if stored and value is not None and stored.get(key) != value:
            raise StageError(
                f"This folder already runs with {key} = {stored.get(key)!r}. "
                "Use another clone of the repository for a different run."
            )
    if stored:
        return stored
    default_root = _default_result_root(pictologics_only)
    settings = {
        "result_root": requested["result_root"]
        or str(
            Path(_ask("Result folder", str(default_root), args.yes))
            .expanduser()
            .resolve()
        ),
        "machine_id": requested["machine_id"]
        or _ask("Machine ID (no personal names)", suggested_machine_id(), args.yes),
        "quick": bool(args.quick),
        "screens": bool(args.screens),
        "pictologics_only": pictologics_only,
        "first_start": {
            "commit": _git("rev-parse", "HEAD"),
            "operating_system": f"{platform.system()} {platform.release()}",
            "repository": str(REPO),
        },
    }
    atomic_write_json(SETTINGS, settings)
    return settings


def _check_unchanged(settings: dict) -> None:
    """Explain a change that would stop the controller from resuming."""

    first = settings["first_start"]
    now = f"{platform.system()} {platform.release()}"
    if first["operating_system"] != now:
        raise StageError(
            f"The operating system changed ({first['operating_system']} -> {now}). "
            "A started run cannot resume after an update; use a new result folder."
        )
    if first["repository"] != str(REPO):
        raise StageError(
            f"The repository moved ({first['repository']} -> {REPO}). Move it back "
            "to resume."
        )
    if _git("rev-parse", "HEAD") != first["commit"]:
        print(
            "Note: the code changed since the first start. The controller refuses "
            "to resume if the timing code changed."
        )


def _status(context: Context, markers: Markers) -> None:
    mode = " (Pictologics only)" if context.pictologics_only else ""
    print(f"Machine {context.machine_id}{mode}, results in {context.machine_root}")
    for stage in STAGES:
        state = (
            "done"
            if markers.done(stage.name)
            else "-"
            if applies(stage, context)
            else "not in this run"
        )
        print(f"  {stage.title:42s} {state}")
    for pillar in PILLARS:
        ledger = context.run_root / pillar / "benchmark.sqlite3"
        if not ledger.is_file():
            continue
        with closing(sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)) as database:
            counts = dict(
                database.execute("SELECT status, count(*) FROM tasks GROUP BY status")
            )
        print(f"  {pillar}: {counts}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reproduce", description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Accept every default")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="A short check of about an hour; not comparable, never submitted",
    )
    parser.add_argument(
        "--screens",
        action="store_true",
        help="Run the two feasibility screens before the main run",
    )
    parser.add_argument(
        "--pictologics-only",
        action="store_true",
        help=(
            "Time Pictologics alone on the tasks of the full run, to compare "
            "hardware; needs no C compiler"
        ),
    )
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--machine-id", default=None)
    parser.add_argument(
        "--host-profile", default=None, help="A frozen host profile to use instead"
    )
    send = parser.add_mutually_exclusive_group()
    send.add_argument(
        "--submit",
        action="store_true",
        default=None,
        help="Send the results by pull request",
    )
    send.add_argument("--no-submit", dest="submit", action="store_false")
    parser.add_argument("--status", action="store_true", help="Show progress only")
    args = parser.parse_args(argv)
    if args.pictologics_only and args.screens:
        parser.error(
            "the feasibility screens compare the libraries; leave out --screens"
        )
    try:
        settings = _settings(args)
        context = Context(
            result_root=Path(settings["result_root"]),
            machine_id=settings["machine_id"],
            quick=settings["quick"],
            screens=settings["screens"] and not settings["quick"],
            yes=args.yes,
            host_profile=Path(args.host_profile).resolve()
            if args.host_profile
            else None,
            submit=args.submit,
            pictologics_only=bool(settings.get("pictologics_only")),
        )
        markers = Markers(context)
        if args.status:
            _status(context, markers)
            return 0
        for stage in STAGES:
            if not applies(stage, context):
                continue
            if stage.recorded and markers.done(stage.name):
                print(f"[done] {stage.title}")
                continue
            if stage.name == "timing" or stage.name == "screens":
                _check_unchanged(settings)
                if not markers.records.get(stage.name):
                    print(CHECKLIST)
                    if (
                        _ask("Type CALCULATE to start", "CALCULATE", args.yes)
                        != "CALCULATE"
                    ):
                        print("Not started.")
                        return 1
            print(f"\n=== {stage.title}", flush=True)
            finished = stage.function(context)
            if stage.recorded and finished is not False:
                markers.mark(stage.name)
    except StageError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.", file=sys.stderr)
        return 130
    print("\nThe benchmark is complete.")
    for note in context.notes:
        print(note)
    print(f"Results: {context.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
