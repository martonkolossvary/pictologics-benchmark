"""The stages of one complete benchmark reproduction, in their run order."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

from bench import release_lock
from bench.benchmark_ledger import atomic_write_json
from bench.env import native_platform_key

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
WORKSPACE = REPO / "data" / "benchmark"
PILLARS = ("pillar1_morphology", "pillar2_whole_anatomy", "pillar3_ibsi2_phase3")
SUPPORTED_PLATFORMS = ("Darwin-arm64", "Linux-x86_64", "Windows-AMD64")
GIB = 1024**3
# On Linux, numpy's AVX-512 math routines round some results differently, and
# the synthetic images then miss the release checksums.  The workspace build
# turns them off; the timed adapter processes keep them.
NUMPY_AVX512_OFF = (
    "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL "
    "AVX512_ICL AVX512_SPR"
)


class StageError(RuntimeError):
    """A stage cannot finish; the message says what the operator must do."""


@dataclass
class Context:
    result_root: Path
    machine_id: str
    quick: bool
    screens: bool
    yes: bool
    host_profile: Path | None = None
    submit: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def machine_root(self) -> Path:
        return self.result_root / self.machine_id

    @property
    def compliance_root(self) -> Path:
        return self.machine_root / "compliance"

    @property
    def run_root(self) -> Path:
        return self.machine_root / "quick" if self.quick else self.machine_root


def run(command: list[str], *, step: str, **options: Any) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    code = subprocess.run(command, cwd=REPO, **options).returncode
    if code != 0:
        raise StageError(f"{step} failed with exit code {code}")


def bench(*arguments: str, step: str, **options: Any) -> None:
    run([PYTHON, "-m", "bench.cli", *arguments], step=step, **options)


def address_space_cap(limit: int) -> Callable[[], None]:
    """Make a child process fail with MemoryError above ``limit`` bytes."""
    import resource

    return lambda: resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _release_digest(context: Context) -> str:
    lock = REPO / "reproducibility" / "release-lock.json"
    content = lock.read_bytes() if lock.is_file() else b"no-release-lock"
    mode = b"quick" if context.quick else b"main"
    return hashlib.sha256(content + mode).hexdigest()


class Markers:
    """Stage completion records of one machine, bound to the release lock."""

    def __init__(self, context: Context):
        self.path = context.machine_root / "reproduce_stages.json"
        self.digest = _release_digest(context)
        self.records = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.is_file()
            else {}
        )

    def done(self, stage: str) -> bool:
        record = self.records.get(stage)
        return bool(record) and record.get("digest") == self.digest

    def mark(self, stage: str) -> None:
        self.records[stage] = {
            "digest": self.digest,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_json(self.path, self.records)


# ---------------------------------------------------------------- stages


def prerequisites(context: Context) -> None:
    from scripts.launch_benchmark import _known_sync_root

    problems = []
    key = native_platform_key()
    if key not in SUPPORTED_PLATFORMS:
        problems.append(
            f"this platform ({key}) is not supported; use macOS on Apple Silicon, "
            "Linux x86_64 or Windows x86_64"
        )
    if not shutil.which("git"):
        problems.append("install git")
    system = platform.system()
    if system == "Darwin":
        xcode = subprocess.run(["xcode-select", "-p"], capture_output=True)
        if xcode.returncode != 0:
            problems.append("install the compiler: xcode-select --install")
    elif system == "Linux" and not (shutil.which("cc") or shutil.which("gcc")):
        problems.append(
            "install a C compiler, for example: sudo apt install build-essential"
        )
    elif system == "Windows":
        vswhere = (
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer")
            / "vswhere.exe"
        )
        found = (
            vswhere.is_file()
            and subprocess.run(
                [
                    str(vswhere),
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        if not found:
            problems.append(
                "install the Microsoft C++ Build Tools (Desktop development with C++)"
            )
        if len(str(REPO)) > 80:
            problems.append(
                f"move this folder to a shorter path (at most 80 characters): {REPO}"
            )
    for label, path, need in (
        ("repository", REPO, 10),
        ("result folder", context.result_root, 3 if context.quick else 8),
    ):
        if _known_sync_root(path):
            problems.append(
                f"the {label} {path} is in a cloud-sync folder; use a local folder"
            )
        existing = path
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free / GIB
        if free < need:
            problems.append(
                f"the {label} needs {need} GiB free disk space; {free:.1f} GiB is free"
            )
    if problems:
        raise StageError("Fix these first:\n  - " + "\n  - ".join(problems))


def release_check(context: Context) -> None:
    try:
        release_lock.verify()
    except release_lock.ReleaseLockError as exc:
        raise StageError(str(exc)) from exc


def inputs(context: Context) -> None:
    script = "scripts/bootstrap_reproducibility_inputs.py"
    run([PYTHON, script], step="input download")
    run([PYTHON, script, "--verify-only"], step="input check")


def references(context: Context) -> None:
    bench(
        "compliance", "import-ibsi1",
        "--workbook", "data/downloads/IBSI-1-submission-table.xlsx",
        "--output-dir", "data/ibsi1/references",
        step="IBSI 1 reference import",
    )  # fmt: skip
    bench(
        "compliance", "validate-ibsi2-phase1",
        "--reference-dir", "data/ibsi2/references/phase1/maps",
        "--manifest-out", "data/ibsi2/references/phase1/reference_manifest.json",
        step="IBSI 2 Phase 1 reference check",
    )  # fmt: skip
    bench(
        "compliance", "import-ibsi2-phase2",
        "--csv", "data/ibsi2/references/phase2/source/reference_values.csv",
        "--output-dir", "data/ibsi2/references/phase2",
        step="IBSI 2 Phase 2 reference import",
    )  # fmt: skip


def environments(context: Context) -> None:
    from bench import env

    for name, profile in env.load_runtime_profiles().items():
        try:
            env.create_profile(profile)
        except Exception as exc:  # noqa: BLE001 - a drifted env is rebuilt
            print(f"Rebuilding the {name} environment: {exc}")
            env.create_profile(profile, force=True)
    bench("env", "verify", step="environment check")


def workspace(context: Context) -> None:
    run(
        [
            PYTHON, "scripts/prepare_workspace.py",
            "--output-root", "data/benchmark",
            "--ibsi2-source", "data/ibsi2_validation",
            "--resume",
        ],
        step="workspace build",
        env=(
            {**os.environ, "NPY_DISABLE_CPU_FEATURES": NUMPY_AVX512_OFF}
            if platform.system() == "Linux"
            else None
        ),
    )  # fmt: skip
    try:
        release_lock.verify(workspace=True)
    except release_lock.ReleaseLockError as exc:
        raise StageError(str(exc)) from exc


def compliance(context: Context) -> bool:
    root = context.compliance_root
    candidates = root / "ibsi2-candidates"
    bench(
        "compliance", "run-ibsi1",
        "--image", "data/ibsi1/digital_phantom/image/phantom.nii.gz",
        "--mask", "data/ibsi1/digital_phantom/mask/mask.nii.gz",
        "--references", "data/ibsi1/references/ibsi1_references.csv",
        "--reference-manifest", "data/ibsi1/references/ibsi1_reference_manifest.json",
        "--output-dir", str(root / "ibsi1"),
        "--resume",
        step="IBSI 1 conformance",
    )  # fmt: skip
    # IBSI 2 needs more memory than the timing run (MEDimage's Gabor filter
    # holds several 2.5 GiB arrays).  A failure here leaves the stage open and
    # lets the timing benchmark go on.  When memory runs out, Linux stops
    # processes without an error, and can stop the whole computer; the cap
    # makes the large allocation fail with MemoryError instead.
    memory_cap = (
        {"preexec_fn": address_space_cap(int(0.75 * psutil.virtual_memory().total))}
        if platform.system() == "Linux"
        else {}
    )
    try:
        bench(
            "compliance", "generate-ibsi2-candidates",
            "--output-dir", str(candidates),
            "--phases", "phase1,phase2",
            "--resume",
            step="IBSI 2 candidate maps",
            **memory_cap,
        )  # fmt: skip
        # Exact native support first; then the separate class that runs a
        # filter with the package's own boundary or truncation.
        for suffix, label in (("", ""), ("_native_default", " (native default)")):
            folder = suffix.replace("_", "-")
            if not (root / f"ibsi2-phase1{folder}" / "report").is_dir():
                bench(
                    "compliance", "evaluate-ibsi2-phase1",
                    "--reference-manifest", "data/ibsi2/references/phase1/reference_manifest.json",
                    "--reference-dir", "data/ibsi2/references/phase1/maps",
                    "--candidate-manifest", str(candidates / f"ibsi2_phase1{suffix}_candidates.json"),
                    "--output-dir", str(root / f"ibsi2-phase1{folder}"),
                    step=f"IBSI 2 Phase 1 conformance{label}",
                )  # fmt: skip
            bench(
                "compliance", "run-ibsi2-phase2",
                "--candidate-manifest", str(candidates / f"ibsi2_phase2{suffix}_candidates.json"),
                "--references", "data/ibsi2/references/phase2/ibsi2_phase2_references.csv",
                "--reference-manifest",
                "data/ibsi2/references/phase2/ibsi2_phase2_reference_manifest.json",
                "--output-dir", str(root / f"ibsi2-phase2{folder}"),
                "--resume",
                step=f"IBSI 2 Phase 2 conformance{label}",
            )  # fmt: skip
    except StageError as exc:
        context.notes.append(
            f"IBSI 2 conformance did not finish ({exc}). The next run tries again."
        )
        return False
    return True


def host_profile(context: Context) -> None:
    from bench.reproduce.host_profile import write_host_profile

    if context.host_profile is None:
        context.host_profile = write_host_profile(
            context.machine_root / "host_profile.json", context.machine_id
        )


def _launch(context: Context, *flags: str, step: str) -> None:
    command = [
        PYTHON, "scripts/launch_benchmark.py",
        "--workspace-root", "data/benchmark",
        "--result-root", str(context.result_root),
        "--host-profile", str(context.host_profile),
        "--execute", "--confirm", "CALCULATE",
        *flags,
    ]  # fmt: skip
    print(f"\n$ {' '.join(command)}", flush=True)
    code = subprocess.run(command, cwd=REPO).returncode
    if code == 0:
        return
    if code == 75:
        raise StageError(
            "The computer left its required power state (mains power, power saving "
            "off). Fix it, then run the same command again to resume."
        )
    if code in (130, 143, 129):
        raise StageError("Stopped. Run the same command again to resume.")
    raise StageError(f"{step} failed with exit code {code}")


def screens(context: Context) -> None:
    _launch(context, "--screen", step="feasibility screen")
    _launch(
        context, "--screen", "--one-library-thread", step="one-library-thread screen"
    )


def timing_run(context: Context) -> None:
    _launch(context, *(["--quick"] if context.quick else []), step="timing run")


def _not_in_package(context: Context, exc: StageError, what: str) -> bool:
    """Note a failed report or figure; only a quick run stops on it.

    The result package holds neither, so a full run still builds and sends it.
    """
    if context.quick:
        raise exc
    context.notes.append(
        f"The {what} failed ({exc}). The result package does not need them; "
        "the next run tries again."
    )
    return False


def reports(context: Context) -> bool:
    try:
        for pillar in PILLARS:
            folder = context.run_root / pillar
            bench(
                "report",
                "--input-dir", str(folder),
                "--output-dir", str(folder / "report"),
                step=f"{pillar} report",
            )  # fmt: skip
    except StageError as exc:
        return _not_in_package(context, exc, "reports")
    return True


def figures(context: Context) -> bool:
    from bench.figures.style import require_font

    try:
        require_font()
    except RuntimeError:
        hint = (
            "sudo apt install ttf-mscorefonts-installer"
            if platform.system() == "Linux"
            else "install the Arial font"
        )
        context.notes.append(f"Figures skipped: Arial is missing ({hint}).")
        return False
    try:
        run(
            [
                PYTHON, "-m", "bench.figures.build",
                "--output-dir", str(context.run_root / "figures"),
                "--compliance-root", str(context.compliance_root),
                "--workspace-root", "data/benchmark",
                "--synthetic-size", "64" if context.quick else "128",
                "--runs", *(str(context.run_root / pillar) for pillar in PILLARS),
            ],
            step="figures",
        )  # fmt: skip
    except StageError as exc:
        return _not_in_package(context, exc, "figures")
    return True


def _latest_package(context: Context) -> Path | None:
    folders = sorted(
        (context.machine_root / "submissions" / context.machine_id).glob("*")
    )
    return folders[-1] if folders else None


def package(context: Context) -> None:
    from bench.submission.package import PackageError, build_package
    from bench.submission.validate import validate

    try:
        folder = _latest_package(context) or build_package(
            context.machine_root, context.result_root
        )
    except PackageError as exc:
        raise StageError(str(exc)) from exc
    problems = validate(REPO, folder)
    if problems:
        raise StageError(
            "The result package fails its check:\n  " + "\n  ".join(problems)
        )
    context.notes.append(f"Result package: {folder}")


def submission(context: Context) -> None:
    from bench.submission.submit import submit

    folder = _latest_package(context)
    if context.submit is None and not context.yes:
        answer = input("Send the results to the benchmark repository now? [y/N]: ")
        context.submit = answer.strip().lower() in {"y", "yes"}
    if not context.submit:
        context.notes.append(
            "To send the results later, run the same command with --submit."
        )
        return
    try:
        context.notes.append(f"Pull request: {submit(folder)}")
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        context.notes.append(str(exc))


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    # A stage that returns False did not finish its work and is not recorded.
    function: Callable[[Context], bool | None]
    # A recorded stage is skipped on the next run; the others always run.
    recorded: bool


STAGES = (
    Stage("prerequisites", "Check this computer", prerequisites, False),
    Stage("release", "Check the release files", release_check, False),
    Stage("inputs", "Download and check the public inputs", inputs, True),
    Stage("references", "Import the IBSI reference values", references, True),
    Stage("environments", "Build the five library environments", environments, False),
    Stage("workspace", "Build and check the benchmark datasets", workspace, True),
    Stage("compliance", "Check IBSI conformance", compliance, True),
    Stage("host_profile", "Record this computer's hardware", host_profile, False),
    Stage("screens", "Run the feasibility screens", screens, True),
    Stage("timing", "Run the timing benchmark", timing_run, True),
    Stage("reports", "Write the reports", reports, True),
    Stage("figures", "Draw the figures", figures, True),
    Stage("package", "Build the result package", package, True),
    Stage("submit", "Send the results", submission, False),
)
