"""Power and sleep state of the benchmark host, read before timed work."""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def ac_low_power_mode(pmset_custom: str) -> int | None:
    """Return 1 when AC power uses low-power mode, 0 when not, or None.

    Older macOS reports ``lowpowermode 0|1``.  Newer macOS reports
    ``powermode 0|1|2`` (automatic, low power, high power).
    """

    ac_block = re.search(
        r"(?:^|\n)AC Power:\s*(.*?)(?=\n\S[^\n]*:\s*$|\Z)",
        pmset_custom,
        flags=re.DOTALL,
    )
    if not ac_block:
        return None
    setting = re.search(
        r"^\s*(lowpowermode|powermode)\s+(\d+)\s*$", ac_block.group(1), re.MULTILINE
    )
    if not setting:
        return None
    value = int(setting.group(2))
    return value if setting.group(1) == "lowpowermode" else int(value == 1)


def darwin_power_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "power_source": None,
        "low_power_mode": None,
        "sleep_assertion": None,
        "probe_errors": [],
    }
    try:
        battery = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        match = re.search(r"Now drawing from '([^']+)'", battery.stdout or "")
        if battery.returncode == 0 and match:
            state["power_source"] = match.group(1)
        else:
            state["probe_errors"].append("power_source_unavailable")
    except (OSError, subprocess.SubprocessError):
        state["probe_errors"].append("power_source_probe_failed")
    try:
        custom = subprocess.run(
            ["/usr/bin/pmset", "-g", "custom"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        low_power = (
            ac_low_power_mode(custom.stdout or "") if custom.returncode == 0 else None
        )
        if low_power is not None:
            state["low_power_mode"] = low_power
        else:
            state["probe_errors"].append("low_power_mode_unavailable")
    except (OSError, subprocess.SubprocessError):
        state["probe_errors"].append("low_power_mode_probe_failed")
    try:
        assertions = subprocess.run(
            ["/usr/bin/pmset", "-g", "assertions"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        values = [
            int(value)
            for value in re.findall(
                r"^\s*(?:PreventSystemSleep|PreventUserIdleSystemSleep)\s+(\d+)\s*$",
                assertions.stdout or "",
                flags=re.MULTILINE,
            )
        ]
        if assertions.returncode == 0 and values:
            state["sleep_assertion"] = any(value > 0 for value in values)
        else:
            state["probe_errors"].append("sleep_assertion_unavailable")
    except (OSError, subprocess.SubprocessError):
        state["probe_errors"].append("sleep_assertion_probe_failed")
    return state


def _empty_state() -> dict[str, Any]:
    return {
        "power_source": None,
        "low_power_mode": None,
        "sleep_assertion": None,
        "probe_errors": [],
    }


def linux_power_state(
    power_supply: Path = Path("/sys/class/power_supply"),
) -> dict[str, Any]:
    """Read mains power, the power profile and systemd sleep inhibitors."""

    state = _empty_state()
    supplies = sorted(power_supply.glob("*")) if power_supply.is_dir() else []
    types = {
        path.name: (path / "type").read_text(encoding="ascii").strip()
        for path in supplies
        if (path / "type").is_file()
    }
    mains = [name for name, kind in types.items() if kind == "Mains"]
    online = [
        name
        for name in mains
        if (power_supply / name / "online").read_text(encoding="ascii").strip() == "1"
    ]
    if online or "Battery" not in types.values():
        # A desktop without a battery runs on mains power.
        state["power_source"] = "AC Power"
    else:
        state["power_source"] = "Battery Power"
    profile = shutil.which("powerprofilesctl")
    if profile is None:
        state["low_power_mode"] = 0
        state["probe_errors"].append("power_profiles_daemon_unavailable")
    else:
        current = subprocess.run(
            [profile, "get"], capture_output=True, text=True, check=False, timeout=5
        )
        state["low_power_mode"] = int(current.stdout.strip() == "power-saver")
    inhibit = shutil.which("systemd-inhibit")
    if inhibit is None:
        state["probe_errors"].append("systemd_inhibit_unavailable")
    else:
        listing = subprocess.run(
            [inhibit, "--list", "--no-legend", "--no-pager"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        state["sleep_assertion"] = any(
            "sleep" in line and "block" in line for line in listing.stdout.splitlines()
        )
    return state


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


# SetThreadExecutionState flags: keep the system awake for this thread.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_WINDOWS_SLEEP_HELD = False


def hold_windows_sleep_assertion() -> None:
    """Keep Windows awake while this process runs; it ends with the process."""

    global _WINDOWS_SLEEP_HELD
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    _WINDOWS_SLEEP_HELD = bool(
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    )


def windows_power_state() -> dict[str, Any]:
    """Read mains power and battery saver; the sleep hold is this process's own."""

    state = _empty_state()
    status = _SystemPowerStatus()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
        # BatteryFlag 128 means that the computer has no battery.
        if status.ACLineStatus == 1 or status.BatteryFlag & 128:
            state["power_source"] = "AC Power"
        elif status.ACLineStatus == 0:
            state["power_source"] = "Battery Power"
        else:
            state["probe_errors"].append("power_source_unknown")
        state["low_power_mode"] = int(status.SystemStatusFlag == 1)
    else:
        state["probe_errors"].append("power_status_unavailable")
    state["sleep_assertion"] = _WINDOWS_SLEEP_HELD
    return state


def can_prevent_sleep() -> bool:
    """Return whether this computer can hold a sleep-prevention assertion."""

    if platform.system() != "Linux":
        return platform.system() in {"Darwin", "Windows"}
    inhibit = shutil.which("systemd-inhibit")
    if inhibit is None:
        return False
    test = subprocess.run(
        [inhibit, "--what=sleep:idle", "--mode=block", "true"],
        capture_output=True,
        check=False,
        timeout=10,
    )
    return test.returncode == 0


def power_state() -> dict[str, Any]:
    """Return the power state of this host, read with the probe for its system."""

    system = platform.system()
    if system == "Darwin":
        return darwin_power_state()
    if system == "Linux":
        return linux_power_state()
    if system == "Windows":
        return windows_power_state()
    state = _empty_state()
    state["probe_errors"].append(f"no_power_probe_for_{system}")
    return state


@contextmanager
def prevent_sleep(reason: str) -> Iterator[None]:
    """Keep the computer awake until the block ends."""

    guard = None
    if platform.system() == "Darwin":
        guard = subprocess.Popen(
            ["/usr/bin/caffeinate", "-dimsu", "-w", str(os.getpid())]
        )
    elif platform.system() == "Linux" and shutil.which("systemd-inhibit"):
        guard = subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=sleep:idle:handle-lid-switch",
                "--who=pictologics-benchmark",
                f"--why={reason}",
                "--mode=block",
                "sleep",
                "infinity",
            ]
        )
    elif platform.system() == "Windows":
        hold_windows_sleep_assertion()
    try:
        yield
    finally:
        if guard is not None:
            guard.terminate()
            guard.wait(timeout=10)


def runtime_state_errors(
    required: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    require_sleep_assertion: bool,
) -> list[str]:
    """Compare the observed power state with a host profile's required state."""

    errors: list[str] = []
    required_power = required.get("power_source")
    if required_power and state["power_source"] != required_power:
        errors.append(
            f"power source must be {required_power!r}; observed "
            f"{state['power_source']!r}"
        )
    required_low_power = required.get("low_power_mode")
    if required_low_power is not None and state["low_power_mode"] != required_low_power:
        errors.append(
            f"low-power mode must be {required_low_power!r}; observed "
            f"{state['low_power_mode']!r}"
        )
    if (
        require_sleep_assertion
        and required.get("sleep_assertion_during_calculation") is True
        and state["sleep_assertion"] is not True
    ):
        errors.append(
            "a sleep-prevention assertion is required during calculation; "
            "start the benchmark with reproduce.sh or reproduce.ps1"
        )
    return errors
