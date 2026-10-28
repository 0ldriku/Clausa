"""The computer's cores and memory, for the default number of parallel MFA alignments (standard library only).

Stage 2 aligns each recording in its own MFA call; `--mfa-workers N` runs N of
them at once. Outputs do not depend on N: every call has its own MFA root
(pipeline/*/asr_mfa_*.py), and 1, 4 and 8 workers gave byte-identical outputs
on 10 + 10 recordings (docs/development.md, "How a run works"). One worker, a pool
process with its `mfa align`, peaked at about 1.7 GB of RSS.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys

MAX_WORKERS = 8   # the range checked byte for byte; 4 -> 8 workers gained only 56 -> 44 s
WORKER_GB = 2.0   # memory per MFA worker (measured peak 1.6-1.7 GB RSS)
RESERVE_GB = 2.0  # memory left for the stage's own process, the app and the system
RESERVE_TOTAL_GB = 4.0  # the same when only the total memory is known


def _cgroup(name: str) -> str | None:
    """A cgroup v2 limit of this process (a container's --cpus or -m), or None.

    cgroup v1 (older Docker hosts) is not read: there the host's cores and memory count."""
    try:
        with open(f"/sys/fs/cgroup/{name}", encoding="utf-8") as f:
            value = f.read().strip()
    except OSError:
        return None
    return None if not value or value.startswith("max") else value


def physical_cores() -> int:
    """Physical cores this process may use: within its CPU affinity and a container's CPU quota."""
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if sys.platform == "darwin":  # Apple Silicon has no SMT; Intel Macs report their physical cores
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=10)
        return min(logical, int(out.stdout.strip()))
    if sys.platform.startswith("linux"):
        cores, phys = set(), None
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    cores.add((phys, line.split(":")[1].strip()))
        n = min(logical, len(cores) or logical // 2 or 1)
        quota = _cgroup("cpu.max")  # "<quota> <period>" under podman run --cpus
        if quota:
            q, period = quota.split()
            n = min(n, math.ceil(int(q) / int(period)))
        return n
    if os.name == "nt":
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                              "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],
                             capture_output=True, text=True, timeout=30,
                             creationflags=subprocess.CREATE_NO_WINDOW)  # no console window from the app
        return min(logical, int(out.stdout.strip()))
    return logical // 2 or 1


def available_gb() -> tuple[float, bool]:
    """(memory available to new processes in GB, True), or (the total, False) where only that is known."""
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
        avail = info["MemAvailable"]
        limit, used = _cgroup("memory.max"), _cgroup("memory.current")
        if limit:
            avail = min(avail, int(limit) - int(used or 0))
        return avail / 1e9, True
    if sys.platform == "darwin":  # free and inactive pages can be handed out
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        page = int(re.search(r"page size of (\d+) bytes", out)[1])
        pages = sum(int(m[1]) for m in re.finditer(r"Pages (?:free|inactive):\s+(\d+)", out))
        return pages * page / 1e9, True
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return status.ullAvailPhys / 1e9, True
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, False


def default_mfa_workers(n_files: int) -> tuple[int, str]:
    """(MFA alignments to run at once, why): at most MAX_WORKERS and the recordings, one physical core
    left for the stage's own process, the app and the system, and WORKER_GB of available memory each
    beyond the reserve. Measure just before Stage 2. 1 if the cores or memory cannot be read."""
    if n_files <= 1:  # nothing to share out: no probe (on Windows a PowerShell start)
        return 1, f"{n_files} of {n_files} recordings at once"
    try:
        cores = physical_cores()
        mem, available = available_gb()
    except Exception as exc:  # anything unexpected on an unusual system: one at a time, as before
        return 1, f"1 at a time: cores or memory could not be read ({type(exc).__name__}: {exc})"
    reserve = RESERVE_GB if available else RESERVE_TOTAL_GB
    by_memory = math.floor((mem - reserve) / WORKER_GB)
    n = max(1, min(MAX_WORKERS, n_files, cores - 1, by_memory))
    return n, (f"{n} of {n_files} recordings at once ({cores} physical cores, "
               f"{mem:.1f} GB of memory {'available' if available else 'in all'}, at most {MAX_WORKERS})")
