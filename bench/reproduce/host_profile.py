"""Measure this computer and write its benchmark host profile."""

from __future__ import annotations

import json
import os
import platform
import re
import secrets
from pathlib import Path
from typing import Any

import psutil

from bench.benchmark_ledger import atomic_write_json
from bench.host_state import can_prevent_sleep
from bench.run import _auto_cpu_model

REQUIRED_RUNTIME_STATE = {"power_source": "AC Power", "low_power_mode": 0}
BENCHMARK_SETTINGS = {
    "background_load": "operator_quiescent",
    "operating_system_updates": "deferred_for_complete_machine_run",
    "power_source": "AC Power",
    "low_power_mode": "off",
    "sleep_prevention": "controller_sleep_guard",
    "result_storage": "local_unsynchronised_disk",
    "thread_policy": "adapter_native_threads",
}


def cpu_model() -> str:
    return _auto_cpu_model() or platform.machine() or "unknown CPU"


def suggested_machine_id() -> str:
    """Return an ID from the OS and CPU plus a random part, never the hostname."""

    system = {"Darwin": "macos", "Windows": "windows"}.get(
        platform.system(), platform.system().lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", cpu_model().lower()).strip("-")[:24]
    return f"{system}-{slug}-{secrets.token_hex(2)}"


def hardware() -> dict[str, Any]:
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True) or os.cpu_count(),
        "memory_total_bytes": psutil.virtual_memory().total,
    }


def write_host_profile(path: Path, machine_id: str) -> Path:
    """Write the profile once; later runs keep it, so the hardware stays bound."""

    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded.get("machine_id") != machine_id:
            raise ValueError(
                f"{path} belongs to machine {recorded.get('machine_id')!r}"
            )
        return path
    measured = hardware()
    memory_gib = round(measured["memory_total_bytes"] / 1024**3)
    model = cpu_model()
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "profile_id": machine_id,
            "machine_id": machine_id,
            "machine_label": (
                f"{model} ({measured['cpu_count_logical']}-thread CPU, "
                f"{memory_gib} GiB RAM)"
            ),
            "cpu_model": model,
            "expected_hardware": measured,
            "required_runtime_state": {
                **REQUIRED_RUNTIME_STATE,
                # A Linux server without logind cannot block sleep, and does
                # not sleep either.
                "sleep_assertion_during_calculation": can_prevent_sleep(),
            },
            "benchmark_settings": BENCHMARK_SETTINGS,
            "notes": [
                "Measured by reproduce on the first run of this machine.",
                "The profile holds no serial number, hardware UUID, hostname "
                "or account name.",
            ],
        },
    )
    return path
