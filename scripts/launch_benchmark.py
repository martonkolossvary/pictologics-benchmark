#!/usr/bin/env python3
"""Print, dry-validate, or explicitly execute the frozen three-pillar runs.

The default action only prints the commands.  Radiomic calculation requires
both ``--execute`` and the literal acknowledgement ``--confirm CALCULATE``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
from typing import Any

import psutil

from bench.benchmark_ledger import atomic_write_json, sha256_file
from bench.host_state import power_state, runtime_state_errors


PILLARS = (
    ("pillar1_morphology", "pillar1"),
    ("pillar2_whole_anatomy", "pillar2_a1"),
    ("pillar3_ibsi2_phase3", "ibsi2_phase3"),
)


MACHINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
WINDOWS_PATH_BUDGET = 240
RESULT_BYTES_PER_ELIGIBLE_TASK = 64 * 1024
RESULT_FIXED_RESERVE_BYTES = 2 * 1024**3
# The quick run checks a machine in about an hour with the screen contract on
# a few small cases.  Its results are not comparable with the main run.
QUICK_CASES = {
    "pillar1_morphology": (["--sizes", "32,64", "--masks", "M1"], 6),
    "pillar2_whole_anatomy": (["--sizes", "32,48,64"], 3),
    "pillar3_ibsi2_phase3": (["--sizes", "136,112,49"], 3),
}
QUICK_PER_CALL_TIMEOUT_SECONDS = 300.0


def _portable_command(
    command: list[str],
    repository: Path,
    result_root: Path | None = None,
) -> list[str]:
    roots = [(repository.resolve(), None)]
    if result_root is not None:
        roots.insert(0, (result_root.resolve(), "{RESULT_ROOT}"))
    portable: list[str] = []
    for value in command:
        path = Path(value)
        if not path.is_absolute():
            portable.append(value)
            continue
        replacement = value
        for root, token in roots:
            try:
                # Preserve a repository-local interpreter path lexically. Its
                # executable may be a symlink to a system Python outside the
                # repository, but the invocation itself remains portable.
                relative = path.relative_to(root)
            except ValueError:
                continue
            replacement = (
                relative.as_posix()
                if token is None
                else f"{token}/{relative.as_posix()}"
            )
            break
        portable.append(replacement)
    return portable


def _default_machine_id() -> str:
    private_identity = socket.gethostname().encode("utf-8")
    return "anonymous-" + hashlib.sha256(private_identity).hexdigest()[:16]


def _validate_machine_id(value: str | None) -> str:
    resolved = str(value or "").strip() or _default_machine_id()
    if not MACHINE_ID_PATTERN.fullmatch(resolved):
        raise ValueError(
            "machine ID must be 1-64 characters using only letters, numbers, "
            "periods, underscores, or hyphens"
        )
    return resolved


def _load_host_profile(path: Path) -> dict[str, Any]:
    """Load a public-safe, checksum-bound benchmark host profile."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported benchmark host profile")
    required_strings = ("profile_id", "machine_id", "machine_label", "cpu_model")
    for field in required_strings:
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"host profile is missing {field}")
    if payload["profile_id"] != payload["machine_id"]:
        raise ValueError("host profile ID and machine ID must match")
    _validate_machine_id(str(payload["machine_id"]))
    for field in ("expected_hardware", "required_runtime_state", "benchmark_settings"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"host profile {field} must be a JSON object")
    forbidden_keys = {"serial", "serial_number", "hardware_uuid", "hostname", "udid"}
    pending: list[Any] = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise ValueError(f"host profile contains private field: {key}")
                pending.append(item)
        elif isinstance(value, list):
            pending.extend(value)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "payload": payload,
    }


def _profile_value(
    supplied: Any,
    profile: dict[str, Any] | None,
    field: str,
    *,
    flag: str,
) -> Any:
    if profile is None:
        return supplied
    frozen = profile["payload"].get(field)
    if supplied is not None and str(supplied) != str(frozen):
        raise ValueError(f"{flag} conflicts with the frozen host profile")
    return frozen


def _host_profile_preflight(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    payload = profile["payload"]
    expected = payload["expected_hardware"]
    logical_cores = psutil.cpu_count(logical=True)
    if logical_cores is None:
        logical_cores = os.cpu_count()
    observed = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": logical_cores,
        "memory_total_bytes": psutil.virtual_memory().total,
    }
    errors: list[str] = []
    for field, frozen in expected.items():
        if observed.get(field) != frozen:
            errors.append(
                f"host hardware mismatch for {field}: expected {frozen!r}, "
                f"observed {observed.get(field)!r}"
            )
    required_state = payload["required_runtime_state"]
    # The controller holds its own sleep guard and checks it before each task.
    runtime_state = power_state()
    errors.extend(
        runtime_state_errors(
            required_state, runtime_state, require_sleep_assertion=False
        )
    )
    return {
        "status": "pass" if not errors else "fail",
        "profile_id": payload["profile_id"],
        "profile_path": profile["path"],
        "profile_sha256": profile["sha256"],
        "expected_hardware": expected,
        "observed_hardware": observed,
        "required_runtime_state": required_state,
        "observed_runtime_state": runtime_state,
        "benchmark_settings": payload["benchmark_settings"],
        "operator_confirmation_required": [
            "background_load",
            "operating_system_updates",
        ],
        "errors": errors,
    }


def _known_sync_root(path: Path) -> str | None:
    """Return evidence that *path* is managed by a desktop sync client."""

    resolved = path.expanduser().resolve()
    folded = str(resolved).casefold().replace("\\", "/")
    markers = (
        "/library/cloudstorage/",
        "/library/mobile documents/",
        "/onedrive",
        "/dropbox/",
        "/google drive/",
        "/google/drivefs/",
    )
    if any(marker in folded for marker in markers):
        return "path_marker"
    for variable in (
        "OneDrive",
        "OneDriveConsumer",
        "OneDriveCommercial",
        "Dropbox",
        "GoogleDrive",
    ):
        raw = str(os.environ.get(variable) or "").strip()
        if not raw:
            continue
        try:
            resolved.relative_to(Path(raw).expanduser().resolve())
        except ValueError:
            continue
        return f"environment:{variable}"
    return None


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise ValueError(f"no existing parent for result root: {path}")
    return candidate


def _staged_input_bytes(root: Path, workspace: dict[str, Any]) -> int:
    total = 0
    for pillar, _ in PILLARS:
        dataset = workspace["datasets"][pillar]
        manifest = json.loads(
            (root / dataset["path"] / "manifest.json").read_text(encoding="utf-8")
        )
        unique_files: dict[tuple[str, str], int] = {}
        for item in manifest.get("files", []):
            checksum = str(item.get("sha256") or "").lower()
            suffix = (
                ".nii.gz"
                if str(item.get("path") or "").lower().endswith(".nii.gz")
                else Path(str(item.get("path") or "")).suffix.lower()
            )
            unique_files[(checksum, suffix)] = int(item.get("bytes") or 0)
        total += sum(unique_files.values())
    return total


def _result_preflight(
    root: Path,
    workspace: dict[str, Any],
    *,
    result_root: Path,
    machine_id: str,
    allow_synced_results: bool,
    repeats: int | None = None,
    host_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check storage isolation without creating a result directory."""

    result_root = result_root.expanduser().resolve()
    machine_root = result_root / machine_id
    existing_parent = _existing_parent(result_root)
    disk = shutil.disk_usage(existing_parent)
    staged_bytes = _staged_input_bytes(root, workspace)
    reviewed_repeats = int(workspace["task_inventory"]["fresh_process_repeats"])
    requested_repeats = reviewed_repeats if repeats is None else int(repeats)
    if requested_repeats < reviewed_repeats:
        raise ValueError(f"repeats cannot be below reviewed minimum {reviewed_repeats}")
    reviewed_eligible_tasks = int(
        workspace["task_inventory"]["totals"]["eligible_calculation_tasks"]
    )
    eligible_tasks = reviewed_eligible_tasks * requested_repeats // reviewed_repeats
    estimated_payload_bytes = eligible_tasks * RESULT_BYTES_PER_ELIGIBLE_TASK
    minimum_free_bytes = (
        staged_bytes + estimated_payload_bytes + RESULT_FIXED_RESERVE_BYTES
    )
    sync_evidence = _known_sync_root(result_root)
    repository_sync_evidence = _known_sync_root(root.resolve().parents[1])
    longest_pillar = max(pillar for pillar, _ in PILLARS)
    representative_record = (
        machine_root
        / longest_pillar
        / "records"
        / "p1_reference_noise_m4_n512"
        / "medimage_group_spatial_morphology_rep3_0123456789ab.json"
    )
    representative_input = (
        machine_root
        / longest_pillar
        / "inputs"
        / "ff"
        / ("f" * 64)
        / "STS_051_pet_ivh_fbn1000.nii.gz"
    )
    estimated_max_path_chars = max(
        len(str(representative_record)), len(str(representative_input))
    )
    errors: list[str] = []
    warnings: list[str] = []
    host_preflight = _host_profile_preflight(host_profile)
    if host_preflight and host_preflight["status"] != "pass":
        errors.extend(host_preflight["errors"])
    if sync_evidence and not allow_synced_results:
        errors.append(
            "result root is managed by a desktop sync client; SQLite WAL files, "
            "locks, and atomic payload commits require a local unsynchronised path"
        )
    if disk.free < minimum_free_bytes:
        errors.append(
            "insufficient free space for staged immutable inputs and a conservative "
            "result-payload reserve"
        )
    if os.name == "nt" and estimated_max_path_chars > WINDOWS_PATH_BUDGET:
        errors.append(
            f"estimated result path length {estimated_max_path_chars} exceeds the "
            f"{WINDOWS_PATH_BUDGET}-character native-library safety budget"
        )
    elif estimated_max_path_chars > WINDOWS_PATH_BUDGET:
        warnings.append(
            f"this result root would exceed the {WINDOWS_PATH_BUDGET}-character "
            "Windows path budget if reused there"
        )
    if repository_sync_evidence:
        warnings.append(
            "the source/input workspace is cloud-synchronised; fully hydrate it and "
            "prefer a short local execution copy, especially on Windows"
        )
    temp_roots = [
        Path(value).resolve()
        for value in {str(Path(os.getenv("TMPDIR") or "/tmp")), "/tmp"}
    ]
    if any(
        machine_root == temporary or temporary in machine_root.parents
        for temporary in temp_roots
    ):
        warnings.append(
            "result root is temporary storage; copy the complete machine directory "
            "to durable storage only after the run is stopped and SQLite is closed"
        )
    return {
        "status": "pass" if not errors else "fail",
        "machine_id": machine_id,
        "workspace_root": str(root),
        "result_root": str(result_root),
        "machine_result_root": str(machine_root),
        "result_root_sync_evidence": sync_evidence,
        "workspace_sync_evidence": repository_sync_evidence,
        "staged_input_bytes": staged_bytes,
        "estimated_payload_bytes": estimated_payload_bytes,
        "fresh_process_repeats": requested_repeats,
        "fixed_reserve_bytes": RESULT_FIXED_RESERVE_BYTES,
        "minimum_free_bytes": minimum_free_bytes,
        "available_bytes": disk.free,
        "estimated_max_path_chars": estimated_max_path_chars,
        "windows_path_budget_chars": WINDOWS_PATH_BUDGET,
        "host_profile": host_preflight,
        "errors": errors,
        "warnings": warnings,
    }


def _load_workspace(root: Path) -> dict[str, Any]:
    path = root / "workspace_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported benchmark workspace manifest")
    if value.get("benchmark_timing_executed") is not False:
        raise ValueError("workspace manifest is not a pristine input-only record")
    repository = root.resolve().parents[1]
    for source in value.get("workspace_sources", []):
        source_path = repository / source["path"]
        if sha256_file(source_path) != source["sha256"]:
            raise ValueError(
                f"workspace preparation/launch source changed: {source['path']}"
            )
    for pillar, _ in PILLARS:
        dataset = value["datasets"][pillar]
        manifest_path = root / dataset["path"] / "manifest.json"
        if sha256_file(manifest_path) != dataset["manifest_sha256"]:
            raise ValueError(f"dataset manifest changed after validation: {pillar}")
    return value


def build_commands(
    root: Path,
    workspace: dict[str, Any],
    *,
    result_root: Path,
    dry_run: bool,
    machine_id: str | None = None,
    machine_label: str | None = None,
    cpu_model: str | None = None,
    cpu_base_ghz: float | None = None,
    host_profile_id: str | None = None,
    host_profile_sha256: str | None = None,
    host_settings: dict[str, Any] | None = None,
    repeats: int | None = None,
    screen: bool = False,
    one_library_thread: bool = False,
    quick: bool = False,
) -> list[list[str]]:
    if one_library_thread and not screen:
        raise ValueError("one library thread is a screen-only sensitivity check")
    if quick and screen:
        raise ValueError("choose either the quick run or a screen")
    policy = workspace["launch_policy"]
    endpoint = workspace[
        "one_library_thread_screen_contract"
        if one_library_thread
        else "screen_contract"
        if screen or quick
        else "endpoint_contract"
    ]
    inventory = workspace[
        "screen_task_inventory" if screen or quick else "task_inventory"
    ]
    per_call_timeout = (
        QUICK_PER_CALL_TIMEOUT_SECONDS
        if quick
        else (workspace["screen_launch_policy"] if screen else policy)[
            "per_call_timeout_seconds"
        ]
    )
    repository = root.resolve().parents[1]
    resolved_machine_id = _validate_machine_id(machine_id)
    machine_result_root = result_root.expanduser().resolve() / resolved_machine_id
    if screen or quick:
        if repeats is not None:
            raise ValueError("the screen and the quick run have a fixed single repeat")
        machine_result_root = machine_result_root / (
            "quick"
            if quick
            else "screen_one_library_thread"
            if one_library_thread
            else "screen"
        )
    reviewed_repeats = int(inventory["fresh_process_repeats"])
    requested_repeats = reviewed_repeats if repeats is None else int(repeats)
    if requested_repeats < reviewed_repeats:
        raise ValueError(f"repeats cannot be below reviewed minimum {reviewed_repeats}")
    run_suffix = (
        "quick"
        if quick
        else "screen_one_library_thread"
        if one_library_thread
        else "screen"
        if screen
        else "main"
    )
    commands: list[list[str]] = []
    pillar_task_counts = {
        pillar: int(
            QUICK_CASES[pillar][1]
            if quick
            else workspace["datasets"][pillar]["case_count"]
        )
        * int(inventory["adapter_count"])
        * int(inventory["family_count"])
        * requested_repeats
        for pillar, _ in PILLARS
    }
    project_total_tasks = sum(pillar_task_counts.values())
    project_task_offset = 0
    for pillar, directory in PILLARS:
        command = [
            sys.executable,
            "-m",
            "bench.cli",
            "run",
            "--dataset-dir",
            str((root / directory).resolve()),
            "--run-id",
            f"{pillar}_{run_suffix}",
            "--report-dir",
            str((machine_result_root / pillar).resolve()),
            "--endpoint-contract",
            str((repository / endpoint["path"]).resolve()),
            "--input-contract",
            "manifest_harmonized",
            "--adapters",
            ",".join(workspace["adapter_order"]),
            "--families",
            "all",
            "--repeats",
            str(requested_repeats),
            "--timing-iterations",
            str(inventory["measured_calls_per_process"]),
            "--timeout",
            str(per_call_timeout),
            "--checkpoint-interval",
            str(policy["checkpoint_interval_tasks"]),
            "--progress-interval",
            str(policy.get("progress_interval_seconds", 30.0)),
            "--project-total-tasks",
            str(project_total_tasks),
            "--project-task-offset",
            str(project_task_offset),
            "--machine-id",
            resolved_machine_id,
            "--guardrail-baseline",
            str(policy["speed_guardrail_baseline"]),
            # Only timeouts and memory stops skip larger sizes, for every adapter.
            "--disable-speed-truncation",
            "--keep-going",
            "--resume",
        ]
        if machine_label:
            command.extend(["--machine-label", str(machine_label)])
        if cpu_model:
            command.extend(["--cpu-model", str(cpu_model)])
        if cpu_base_ghz is not None:
            command.extend(["--cpu-base-ghz", str(float(cpu_base_ghz))])
        if host_profile_id:
            command.extend(["--host-profile-id", host_profile_id])
        if host_profile_sha256:
            command.extend(["--host-profile-sha256", host_profile_sha256])
        if host_settings is not None:
            command.extend(
                [
                    "--host-settings-json",
                    json.dumps(host_settings, sort_keys=True, separators=(",", ":")),
                ]
            )
        if quick:
            command.extend(QUICK_CASES[pillar][0])
        if requested_repeats > reviewed_repeats:
            command.append("--extend-repeats")
        if dry_run:
            command.append("--dry-run")
        commands.append(command)
        project_task_offset += pillar_task_counts[pillar]
    return commands


def _run_controller(command: list[str]) -> int:
    """Run one pillar and let its controller finish a clean stop.

    A terminal Ctrl-C reaches the controller too, which stops its adapter and
    checkpoints the ledger.  ``subprocess.run`` would kill the controller
    0.25 s later, so the launcher waits instead and forwards SIGTERM.
    """

    process = subprocess.Popen(command)
    previous = signal.signal(
        signal.SIGTERM, lambda signum, frame: process.send_signal(signum)
    )
    try:
        try:
            return process.wait()
        except KeyboardInterrupt:
            return process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default="data/benchmark")
    parser.add_argument("--result-root", required=True)
    parser.add_argument(
        "--machine-id",
        default=None,
        help="Stable public-safe ID; defaults to an anonymous hostname hash",
    )
    parser.add_argument("--machine-label", default=None)
    parser.add_argument("--cpu-model", default=None)
    parser.add_argument("--cpu-base-ghz", type=float, default=None)
    parser.add_argument(
        "--host-profile",
        type=Path,
        default=None,
        help="Frozen public-safe host identity and operating-settings profile",
    )
    parser.add_argument(
        "--allow-synced-results",
        action="store_true",
        help="Explicitly override the fail-closed cloud-synchronised result check",
    )
    parser.add_argument(
        "--validate-plans",
        action="store_true",
        help="Invoke each controller with --dry-run; no adapter calculation occurs",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Absolute fresh-process repeat horizon; use 5 to append repeats 4 and 5",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Run the feasibility screen: one warmup, one timed call, one repeat",
    )
    parser.add_argument(
        "--one-library-thread",
        action="store_true",
        help=(
            "With --screen: the separately labelled sensitivity screen with one "
            "maths-library thread (Numba and ITK keep native threads)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short check on a few small cases; not comparable, never submitted",
    )
    args = parser.parse_args()
    if args.quick and args.screen:
        parser.error("choose either --quick or --screen")
    if args.execute and args.confirm != "CALCULATE":
        parser.error("--execute requires --confirm CALCULATE")
    if args.execute and args.validate_plans:
        parser.error("choose either --execute or --validate-plans")

    root = Path(args.workspace_root).expanduser().resolve()
    workspace = _load_workspace(root)
    host_profile = _load_host_profile(args.host_profile) if args.host_profile else None
    machine_id = _validate_machine_id(
        _profile_value(
            args.machine_id,
            host_profile,
            "machine_id",
            flag="--machine-id",
        )
    )
    machine_label = _profile_value(
        args.machine_label,
        host_profile,
        "machine_label",
        flag="--machine-label",
    )
    cpu_model = _profile_value(
        args.cpu_model,
        host_profile,
        "cpu_model",
        flag="--cpu-model",
    )
    cpu_base_ghz = _profile_value(
        args.cpu_base_ghz,
        host_profile,
        "cpu_base_ghz",
        flag="--cpu-base-ghz",
    )
    result_root = Path(args.result_root).expanduser().resolve()
    preflight = _result_preflight(
        root,
        workspace,
        result_root=result_root,
        machine_id=machine_id,
        allow_synced_results=args.allow_synced_results,
        repeats=args.repeats,
        host_profile=host_profile,
    )
    if (args.execute or args.validate_plans) and preflight["status"] != "pass":
        parser.error("execution preflight failed: " + "; ".join(preflight["errors"]))
    commands = build_commands(
        root,
        workspace,
        result_root=result_root,
        dry_run=args.validate_plans,
        machine_id=machine_id,
        machine_label=machine_label,
        cpu_model=cpu_model,
        cpu_base_ghz=cpu_base_ghz,
        host_profile_id=(
            str(host_profile["payload"]["profile_id"]) if host_profile else None
        ),
        host_profile_sha256=(str(host_profile["sha256"]) if host_profile else None),
        # The controller re-checks the required power state before each task.
        host_settings=(
            {
                **host_profile["payload"]["benchmark_settings"],
                "required_runtime_state": host_profile["payload"][
                    "required_runtime_state"
                ],
            }
            if host_profile
            else None
        ),
        repeats=args.repeats,
        screen=args.screen,
        one_library_thread=args.one_library_thread,
        quick=args.quick,
    )
    plan = {
        "workspace_manifest_sha256": sha256_file(root / "workspace_manifest.json"),
        "run_kind": (
            "quick"
            if args.quick
            else "feasibility_screen_one_library_thread"
            if args.one_library_thread
            else "feasibility_screen"
            if args.screen
            else "main"
        ),
        "mode": (
            "execute"
            if args.execute
            else "controller_dry_run"
            if args.validate_plans
            else "print_only"
        ),
        "radiomic_calculation_started": bool(args.execute),
        "machine": {
            "machine_id": machine_id,
            "machine_label": machine_label,
            "host_profile_id": (
                host_profile["payload"]["profile_id"] if host_profile else None
            ),
            "host_profile_sha256": host_profile["sha256"] if host_profile else None,
        },
        "execution_preflight": preflight,
        "commands": commands,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute and not args.validate_plans:
        return 0
    if args.validate_plans:
        validated: list[dict[str, Any]] = []
        repository = root.resolve().parents[1]
        for (pillar, _), command in zip(PILLARS, commands):
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
            fingerprint_line = next(
                (
                    line
                    for line in output.splitlines()
                    if line.startswith("Immutable run fingerprint: ")
                ),
                None,
            )
            validated_line = next(
                (line for line in output.splitlines() if line.startswith("Validated ")),
                None,
            )
            if fingerprint_line is None or validated_line is None:
                raise RuntimeError(
                    f"controller dry-run did not emit its attestation for {pillar}"
                )
            validated.append(
                {
                    "pillar": pillar,
                    "summary": validated_line,
                    "run_fingerprint": fingerprint_line.split(": ", 1)[1],
                    "command": _portable_command(command, repository, result_root),
                }
            )
            print(output)
        record = {
            "schema_version": 1,
            "status": "controller_plans_validated_calculations_not_started",
            "radiomic_calculation_started": False,
            "workspace_manifest_sha256": plan["workspace_manifest_sha256"],
            "machine": plan["machine"],
            "execution_preflight": preflight,
            "plans": validated,
        }
        attestation_dir = root / "host_attestations" / machine_id
        attestation = (
            "controller_dry_run_screen_one_library_thread.json"
            if args.one_library_thread
            else "controller_dry_run_screen.json"
            if args.screen
            else "controller_dry_run.json"
        )
        atomic_write_json(attestation_dir / attestation, record)
        print(json.dumps(record, indent=2))
        return 0
    for command in commands:
        returncode = _run_controller(command)
        if returncode != 0:
            return returncode if returncode > 0 else 128 - returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
