"""Memory that an adapter process tree holds, on each operating system."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import psutil


class _RusageInfoV0(ctypes.Structure):
    """``struct rusage_info_v0`` from the macOS ``<sys/resource.h>``."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
        )
    ]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None


def memory_in_use(process: psutil.Process) -> int:
    """Return the memory one process holds, compressed and swapped pages included.

    RSS leaves out pages that macOS compresses and pages that any system
    swaps out, so a process can exhaust memory while its RSS stays small.
    """

    if _LIBPROC is not None:
        info = _RusageInfoV0()
        if _LIBPROC.proc_pid_rusage(process.pid, 0, ctypes.byref(info)) == 0:
            return int(info.phys_footprint)
    elif sys.platform.startswith("linux"):
        status = Path(f"/proc/{process.pid}/status").read_text(encoding="ascii")
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        return sum(
            int(fields.get(key, "0 kB").split()[0]) * 1024
            for key in ("VmRSS", "VmSwap")
        )
    elif sys.platform == "win32":
        return int(process.memory_info().private)
    return int(process.memory_info().rss)


def tree_memory_in_use(process: psutil.Process) -> int:
    """Return the memory that a process and all its descendants hold."""

    try:
        members = [process, *process.children(recursive=True)]
    except psutil.Error:
        members = [process]
    total = 0
    for member in members:
        try:
            total += memory_in_use(member)
        except (psutil.Error, OSError):
            continue
    return total


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001


class WindowsJob:
    """A kill-on-close Windows Job Object that holds one adapter process tree.

    Windows closes the job handle when the controller exits for any reason,
    and then ends every process in the job, so no adapter can outlive it.
    """

    def __init__(self, pid: int):
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.OpenProcess.restype = ctypes.c_void_p
        self._kernel32 = kernel32
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError("CreateJobObjectW failed")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            ctypes.c_void_p(self._handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise OSError("SetInformationJobObject failed")
        process = kernel32.OpenProcess(_PROCESS_SET_QUOTA_AND_TERMINATE, False, pid)
        if not process:
            raise OSError("OpenProcess failed")
        try:
            if not kernel32.AssignProcessToJobObject(
                ctypes.c_void_p(self._handle), ctypes.c_void_p(process)
            ):
                raise OSError("AssignProcessToJobObject failed")
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(process))

    def close(self) -> None:
        """Close the job; Windows ends any process still inside it."""

        if self._handle:
            self._kernel32.CloseHandle(ctypes.c_void_p(self._handle))
            self._handle = None
