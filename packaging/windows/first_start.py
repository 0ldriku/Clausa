"""First start of the Windows app: everything after the first environment (packaging/build_windows.py).

Clausa.exe unpacks the first environment (qwen3-asr), then opens the First start window
(runner/first_start_window.py), which runs this as its driver with that environment's Python and
the app's process environment (packaging/windows/launcher.cs):

    envs\\qwen3-asr\\python.exe -B -s app\\packaging\\windows\\first_start.py <Clausa folder>

Each run appends its `start` and the events of what it does to logs\\first-start-events.jsonl,
which the window shows (runner/first_start_events.py; docs/development.md, "The app zips"):

1. unpack: the other environments' tars (Windows' tar.exe, the tar piped in, so the bytes read
   are the progress; a tar is deleted once it is unpacked whole, and while it is there its folder
   is made again); in each environment conda-unpack (its own Python runs its
   Scripts\\conda-unpack-script.py, which writes the environment's location into the files that
   held the build machine's) and entry points whose first line is #!python.exe (the Python of the
   environment they are in, wherever that is: the folder can be moved afterwards); the marker
   envs\\<name>\\.unpacked. Then the Mark-of-the-Web, which Explorer's "Extract all" copies from
   a downloaded zip onto every file it extracts, is removed from the files outside the
   environments, so that Clausa.exe starts without SmartScreen's question once it was answered.
2. download: PyTorch's wheels (item gpu-libs; on a PC without an NVIDIA driver the processor
   build) and the pinned speech models (runner/models.py), resumable and sha256-checked. A zip
   built with --include-models has all of them: the step is then skipped.
3. prepare: PyTorch put into the ASR environment from its wheels (install_torch), the models of
   such a zip put in place, the main libraries of each environment loaded once (the three in
   parallel: Windows' virus scanner checks them now and not in the first analysis), MFA started
   once, and which processor speech recognition will use. Then envs\\.first-start-done and `done`.

Running it again skips what is done; an interrupted download continues where it stopped. Exit
codes: 0 done; 2 no connection; 3 disk full; 4 a download did not match its checksum; 5 anything
else (each after an `error` event that says what to do). Its lines go to standard output (the
window writes it into logs\\first-start.log) and, with the details, into that log.
HF_ENDPOINT, CLAUSA_MFA_BASE_URL and the proxy variables reach the downloads as they are.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zlib
from datetime import datetime
from pathlib import Path

ENVS = ("qwen3-asr", "mfa", "ja-electra310")
# An entry point's first line: #!, maybe /usr/bin/env, the Python (quoted or not; its path may hold
# spaces), maybe options.
SHEBANG = re.compile(r'^#!\s*(?:/usr/bin/env\s+)?"?(?P<exe>.*?(?P<name>pythonw?\.exe))"?(?P<args>\s.*)?$', re.I)
# Imported once in step 3: what the stages and the app load.
WARM = {
    "qwen3-asr": "import torch, qwen_asr, spacy, librosa, transformers, spacy_transformers, webview, clr\n"
                 "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')",
    "mfa": "import montreal_forced_aligner, kalpy, _kalpy, pynini, sqlalchemy, librosa",
    "ja-electra310": "import spacy, ginza, torch, transformers, sudachipy",
}
TORCH_MANIFEST = "qwen3-asr-torch.json"  # in envs\, from build_windows.py split_torch
TORCH_LABELS = {"cuda": "GPU libraries (PyTorch with NVIDIA CUDA)", "cpu": "PyTorch for the processor"}
TORCH_PLACING = {"cuda": "Putting the GPU libraries in place", "cpu": "Installing PyTorch for the processor"}
CPU_NOTE = ("No NVIDIA graphics card with a working driver was found: Clausa runs on the processor, which works "
            "but is slower: a few minutes per recording (about 2-6, depending on the processor).")
CPU_TAIL = {"cpu": " (With an NVIDIA graphics card: install its driver, then extract Clausa again from the "
                   "download; it is then several times faster.)",
            "cuda": " (With an NVIDIA graphics card, installing or updating its driver makes it several times "
                    "faster.)"}
PLACING_MODELS = "Putting the speech models in place"
EXIT = {"network": 2, "disk": 3, "checksum": 4}
GB = 1 << 30
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Stop(Exception):
    """First start cannot go on: kind (network, disk, checksum, folder, system or other), the
    plain message (what happened, what to do), the step, the technical detail, and whether Try
    again can help."""

    def __init__(self, kind: str, message: str, step: str | None = None, detail: str | None = None,
                 again: bool = True):
        super().__init__(message)
        self.kind, self.message, self.step, self.detail, self.again = kind, message, step, detail, again


def append_only(path: Path):
    """A binary file object on path that only appends: every write lands at the end, also when
    another handle (another process's) writes to the file meanwhile. On Windows a file opened
    with "a" only seeks to the end before its own writes, and a handle inherited by a child process
    (the First start window gives its driver the log as standard output) does not even do that:
    the child writes at its own position, over what others appended. A handle opened with
    FILE_APPEND_DATA alone appends in the file system itself. Elsewhere "ab" does the same."""
    if os.name != "nt":
        return open(path, "ab", buffering=0)
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    # FILE_APPEND_DATA | SYNCHRONIZE; shared for reading, writing and deleting; OPEN_ALWAYS
    handle = kernel32.CreateFileW(str(path), 0x0004 | 0x00100000, 0x7, None, 4, 0x80, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    return open(msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_APPEND), "ab", buffering=0)


def env_vars(prefix: Path, root: Path | None = None) -> dict[str, str]:
    """A process environment for a command of the environment at prefix: its folders and Windows'
    on PATH, as runner/envs.py sets them for the stages; with root, MFA's folder in the cache, as
    the stages have it (else `mfa version` makes Documents\\MFA)."""
    system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    dirs = [prefix, prefix / "Library" / "mingw-w64" / "bin", prefix / "Library" / "usr" / "bin",
            prefix / "Library" / "bin", prefix / "Scripts", prefix / "bin",
            system / "System32", system, system / "System32" / "Wbem"]
    env = dict(os.environ)
    env.update(PATH=os.pathsep.join(map(str, dirs)), CONDA_PREFIX=str(prefix), PYTHONNOUSERSITE="1",
               PYTHONIOENCODING="utf-8")
    if root is not None:
        env["MFA_ROOT_DIR"] = str(root / "cache" / "mfa")
    for v in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(v, None)
    return env


def relative_entry_points(prefix: Path) -> int:
    """Scripts\\<name>-script.py whose first line names a Python by its full path (this
    environment's, after conda-unpack): #!python.exe (or pythonw.exe) instead. The launcher beside
    it (<name>.exe, setuptools' cli-64.exe, as conda-pack's own conda-unpack.exe uses) looks for a
    relative name in the folder above Scripts, which is the environment; the folder can then move.
    (pip's own launcher .exe files keep conda-unpack's absolute path; nothing Clausa runs uses one:
    the stages run python.exe, and MFA is conda's entry point mfa.exe with mfa-script.py.)"""
    count = 0
    for script in [*prefix.glob("Scripts/*-script.py"), *prefix.glob("Scripts/*-script.pyw")]:
        data = script.read_bytes()
        first, sep, rest = data.partition(b"\n")
        line = first.rstrip(b"\r").decode("utf-8", "replace")
        # conda-unpack writes "#!C:/a b/envs/mfa\python.exe": forward slashes, a space left unquoted.
        m = SHEBANG.match(line)
        if not m or not re.search(r"[\\/]", m["exe"]):
            continue
        script.write_bytes(f"#!{m['name']}{m['args'] or ''}".encode("utf-8") + first[len(first.rstrip(b"\r")):] + sep + rest)
        count += 1
    return count


class FirstStart:
    def __init__(self, root: Path):
        self.root = root
        self.envs = root / "envs"
        self.log_path = root / "logs" / "first-start.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker = self.envs / ".first-start-done"
        self.wheels_dir = root / "cache" / "wheels"
        os.environ["CLAUSA_CACHE"] = str(root / "cache")  # before runner.paths is imported
        if str(root / "app") not in sys.path:
            sys.path.insert(0, str(root / "app"))
        from runner import __version__
        from runner import first_start_events as fse
        from runner import models
        self.fse, self.M, self.version = fse, models, __version__
        events = fse.read_events(fse.events_path(root))
        # the sizes items had in earlier runs (a tar is deleted once it is unpacked); a run other
        # than the one Clausa.exe began for this start (it unpacks qwen3-asr) did part of the work
        self.sizes = {(e.get("step"), e.get("item")): e["total"] for e in events
                      if e.get("event") == "item" and e.get("total") is not None}
        own = os.environ.get("CLAUSA_FIRST_START_RUN")
        self.resumed = any(e.get("event") == "start" and e.get("run") != own for e in events)
        self.downloaded_before = any(e.get("event") == "step" and e.get("step") == "download"
                                     and e.get("status") == "running" for e in events)
        self.w = fse.Writer(root, heartbeat=True)
        self.active: str | None = None
        self.said_cpu = False
        self.verified: set[str] = set()  # wheels whose sha256 this run checked
        self.torch = self.read_torch_manifest()
        self.variant = self.torch_variant() if self.torch else None
        # One handle that only appends, for the log lines and, when the window runs this (its
        # standard output is the log), for standard output and standard error too: see append_only.
        try:
            self.out_is_log = os.path.samestat(os.fstat(sys.stdout.fileno()), os.stat(self.log_path))
        except (OSError, ValueError, AttributeError):
            self.out_is_log = False
        try:
            self.log_file = append_only(self.log_path)
        except OSError:  # (a virus scanner holding the file just now): a plain append, else none
            try:
                self.log_file = open(self.log_path, "ab", buffering=0)
            except OSError:
                self.log_file = None
        if self.out_is_log and self.log_file is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(self.log_file.fileno(), 1)
            os.dup2(self.log_file.fileno(), 2)

    # ---------- output ----------
    def log(self, text: str) -> None:
        if self.log_file is None:
            return
        try:
            sys.stdout.flush()  # what was printed before stays before
        except Exception:  # noqa: a closed or missing standard output does not stop the log
            pass
        try:
            self.log_file.write((text.rstrip("\n") + "\n").encode("utf-8"))
        except OSError:
            pass

    def say(self, text: str = "") -> None:
        """A line for the person: standard output and the log."""
        print(text, flush=True)
        if not self.out_is_log:
            self.log(text)

    def begin(self, step: str) -> None:
        self.active = step
        self.w.step(step, "running")

    # ---------- what there is to do ----------
    def read_torch_manifest(self) -> dict | None:
        path = self.envs / TORCH_MANIFEST
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def torch_done(self) -> bool:
        return self.torch is None or (self.envs / "qwen3-asr" / ".torch-installed").exists()

    def torch_variant(self) -> str:
        """The variant installed (the marker's second line), else: cuda where there is an NVIDIA
        driver (nvidia-smi, as setup_envs.default_torch decides), else cpu; cuda also where only
        the CUDA wheels are at hand (a zip built with --include-models): they run on the processor
        too."""
        marker = self.envs / "qwen3-asr" / ".torch-installed"
        if marker.exists():
            lines = marker.read_text(encoding="utf-8").splitlines()
            if len(lines) > 1 and lines[1] in TORCH_LABELS:
                return lines[1]
        if shutil.which("nvidia-smi"):
            return "cuda"
        have = lambda wheels: all(self.whole(w) for w in wheels)  # noqa: E731
        return "cuda" if have(self.torch["wheels"]) and not have(self.torch.get("cpu_wheels", [])) else "cpu"

    def wheels(self) -> list[dict]:
        return self.torch["wheels"] if self.variant == "cuda" else self.torch["cpu_wheels"]

    def whole(self, w: dict) -> bool:
        whl = self.wheels_dir / w["file"]
        return whl.is_file() and whl.stat().st_size == w["size"]

    def have(self, w: dict) -> int:
        part = self.wheels_dir / (w["file"] + ".incomplete")
        return w["size"] if self.whole(w) else min(part.stat().st_size, w["size"]) if part.is_file() else 0

    def model_state(self, m: dict) -> str:
        """ok; fetch (a file to download); or place: every file is here, but a Hugging Face blob is
        not linked into its snapshot yet (a zip built with --include-models holds the blobs only: a
        zip has no links), or refs/main is not written yet."""
        if not self.M.problems(m):
            return "ok"
        for f, used, stored in self.M.targets(m):
            if not self.complete(f, used) and not (m["source"] == "huggingface" and self.complete(f, stored)):
                return "fetch"
        return "place"

    @staticmethod
    def complete(f: dict, path: Path) -> bool:
        return path.is_file() and path.stat().st_size == f["size"]

    def to_fetch(self) -> tuple[list[dict], list[dict]]:
        """The wheels and the models with something to download."""
        # A whole CUDA wheel is not checked again: every file written from it is (write_from_wheels);
        # a CPU wheel is, since pip installs it as it is.
        wheels = [] if self.torch_done() else [w for w in self.wheels() if not self.whole(w) or (
            self.variant == "cpu" and w["file"] not in self.verified)]
        return wheels, [m for m in self.M.needed() if self.model_state(m) == "fetch"]

    def to_place(self) -> tuple[list[dict], int]:
        """The models to put in place without a download, and the bytes to link or copy."""
        models = [m for m in self.M.needed() if self.model_state(m) == "place"]
        return models, sum(f["size"] for m in models for f, used, _ in self.M.targets(m) if not self.complete(f, used))

    def model_size(self, m: dict) -> int:
        return sum(f["size"] for f in m["files"])

    # ---------- the steps ----------
    def declare(self) -> None:
        """Every step's items as they stand, so that each step's total is known from the start."""
        for name in ENVS:
            tar = self.envs / f"{name}.tar"
            if tar.exists():
                total, status = tar.stat().st_size, "waiting"
            elif (self.envs / name / "python.exe").exists():
                total = self.sizes.get(("unpack", name))
                status = "done" if (self.envs / name / ".unpacked").exists() else "waiting"
            else:
                raise Stop("other", f"Part of the app is missing (neither envs\\{name}.tar nor envs\\{name} is "
                                     "there). Please extract the download again.", "unpack", again=False)
            self.w.item("unpack", name, self.fse.ENV_LABELS[name], total, status)
        wheels, models = self.to_fetch()
        if not self.torch_done():
            self.w.item("download", "gpu-libs", TORCH_LABELS[self.variant], sum(w["size"] for w in self.wheels()),
                        "waiting" if wheels else "done")
        for m in self.M.needed():
            self.w.item("download", m["id"], self.fse.model_label(m), self.model_size(m),
                        "waiting" if m in models else "done")
        if not self.torch_done():
            self.w.item("prepare", "gpu-libs", TORCH_PLACING[self.variant],
                        self.torch["bytes"] if self.variant == "cuda" else None, "waiting")
        placing = self.to_place()[1]
        if placing:
            self.w.item("prepare", "models", PLACING_MODELS, placing, "waiting")
        for name in ENVS:
            self.w.item("prepare", name, self.fse.ENV_LABELS[name], None,
                        "done" if self.marker.exists() else "waiting")
        if self.variant == "cpu":
            self.w.message(CPU_NOTE + CPU_TAIL["cpu"])
            self.said_cpu = True


    def unpack(self) -> None:
        todo = [n for n in ENVS if (self.envs / f"{n}.tar").exists() or not (self.envs / n / ".unpacked").exists()]
        if not todo:
            self.clear_mark_of_the_web()  # cheap; an interrupted run may not have finished it
            self.w.step("unpack", "done", "Already done")
            return
        self.begin("unpack")
        for name in todo:
            label = self.fse.ENV_LABELS[name]
            if (self.envs / f"{name}.tar").exists():
                size = self.unpack_tar(name)
            else:  # unpacked by Clausa.exe: its tar is gone
                size = self.sizes.get(("unpack", name))
                self.w.item("unpack", name, label, size, "running", done=size or 0)
            self.conda_unpack(name)
            self.w.item("unpack", name, label, size, "done")
        cleared = self.clear_mark_of_the_web()
        self.log(f"  Mark-of-the-Web removed from {cleared} files")
        self.w.step("unpack", "done")

    def unpack_tar(self, name: str) -> int:
        """envs\\<name>.tar into envs\\<name>, the tar piped into tar.exe; returns its size."""
        label = self.fse.ENV_LABELS[name]
        tar = self.envs / f"{name}.tar"
        size = tar.stat().st_size
        free, need = shutil.disk_usage(self.envs).free, size + 2 * GB
        if free < need:
            self.log(f"  not enough space for {name}: {free} bytes free, {need} needed")
            raise Stop("disk", f"Unpacking needs about {-(-need // 10**9)} GB free on this disk; there are "
                               f"{free // 10**9} GB. Free some space (you can delete the downloaded zip), then "
                               "try again.", "unpack")
        self.say(f"Unpacking: {label} ({size / 1e9:.1f} GB)...")
        start = time.monotonic()
        target = self.envs / name
        shutil.rmtree(target, ignore_errors=True)  # the tar is still there: what is there is not whole
        if target.exists():
            raise Stop("other", f"{label} could not be unpacked: the unfinished envs\\{name} could not be "
                                "removed (is a file in it open?). Try again.", "unpack")
        self.w.item("unpack", name, label, size, "running", done=0)
        tar_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
        done = 0
        # tar.exe's complaints, if any, into the log through the same appending handle
        err = self.log_file if self.log_file is not None else subprocess.DEVNULL
        proc = subprocess.Popen([str(tar_exe), "-xf", "-", "-C", str(self.envs)], stdin=subprocess.PIPE,
                                stdout=err, stderr=err, creationflags=NO_WINDOW)
        try:
            with tar.open("rb") as src:
                while chunk := src.read(4 << 20):
                    proc.stdin.write(chunk)
                    done += len(chunk)
                    self.w.progress("unpack", name, done)
        except OSError as exc:  # tar.exe ended early (a full disk, say: its exit code says so), or
            self.log(f"  unpacking {name}: {exc!r}")  # the tar could not be read
        finally:
            try:
                proc.stdin.close()  # else tar.exe would wait for more
            except OSError:
                pass
            code = proc.wait()
        if code != 0 or done < size or not (target / "python.exe").exists():
            raise Stop("disk", f"{label} could not be unpacked (is the disk full?). Free some space, then try "
                               "again.", "unpack", detail=f"tar.exe exited with {code} on envs\\{name}.tar")
        self.w.progress("unpack", name, done, force=True)
        tar.unlink()
        self.log(f"  {name} unpacked in {time.monotonic() - start:.0f} s")
        return size

    def conda_unpack(self, name: str) -> None:
        """conda-unpack, then relative entry points; marker .unpacked."""
        prefix = self.envs / name
        if (prefix / ".unpacked").exists():
            return
        script = prefix / "Scripts" / "conda-unpack-script.py"
        if not script.exists():
            raise Stop("other", f"envs\\{name} is incomplete (its Scripts\\conda-unpack-script.py is missing). "
                                 "Please extract the download again.", "unpack", again=False)
        start = time.monotonic()
        result = subprocess.run([str(prefix / "python.exe"), "-B", "-s", str(script)], env=env_vars(prefix),
                                capture_output=True, text=True, errors="replace", creationflags=NO_WINDOW)
        self.log(f"--- conda-unpack {name} (exit {result.returncode}, {time.monotonic() - start:.0f} s)\n"
                 f"{result.stdout}{result.stderr}")
        if result.returncode != 0:
            raise Stop("other", f"{self.fse.ENV_LABELS[name]} could not be set up (conda-unpack failed). Please "
                                "extract the download again.", "unpack", detail=result.stderr.strip()[-300:],
                       again=False)
        changed = relative_entry_points(prefix)
        self.log(f"  {name}: {changed} entry points now start the Python beside them (#!python.exe)")
        # The marker first: once it is there, a rerun never needs the script again.
        (prefix / ".unpacked").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
        for f in (prefix / "Scripts" / "conda-unpack.exe", script):
            f.unlink(missing_ok=True)

    def clear_mark_of_the_web(self) -> int:
        """Remove the Zone.Identifier stream from every file outside the unpacked environments."""
        cleared = 0
        for dirpath, dirnames, names in os.walk(self.root):
            if Path(dirpath) == self.envs:
                dirnames[:] = []  # the environments: unpacked by tar.exe, which does not copy the mark
            for n in names:
                try:
                    os.remove(os.path.join(dirpath, n) + ":Zone.Identifier")
                    cleared += 1
                except OSError:  # no such stream (or a drive without streams)
                    pass
        return cleared

    def download(self) -> None:
        wheels, models = self.to_fetch()
        if not wheels and not models:
            if self.downloaded_before:
                self.w.step("download", "done", "Already done")
            else:  # a zip built with --include-models
                self.w.step("download", "skipped", "The speech models and the GPU libraries came with the download.")
            return
        self.begin("download")
        # Room for all of it: what is left to download, PyTorch's files (written from the wheels,
        # which stay until then) and 1 GB to spare.
        left = sum(w["size"] - self.have(w) for w in wheels)
        left += sum(f["size"] for m in models for f, used, _ in self.M.targets(m)
                    if not (used.is_file() and used.stat().st_size == f["size"]))
        unpacked = 0 if self.torch_done() else self.torch["bytes"] if self.variant == "cuda" else \
            2 * sum(w["size"] for w in self.wheels())
        need, free = left + unpacked + GB, shutil.disk_usage(self.root).free
        if free < need:
            raise Stop("disk", f"There is not enough free disk space for the downloads: they need about "
                               f"{need / 1e9:.1f} GB, and there are {free / 1e9:.1f} GB. Free some space (you can "
                               "delete the downloaded zip), then try again; it continues where it stopped.",
                       "download")
        if wheels:
            self.fetch_wheels()
        if models:
            self.fetch_models(models)
        self.w.step("download", "done")

    def fetch_wheels(self) -> None:
        """The variant's wheels into cache\\wheels\\ (runner.models.fetch_file), as one item: their
        bytes add up."""
        label, wheels = TORCH_LABELS[self.variant], self.wheels()
        total = sum(w["size"] for w in wheels)
        self.log(f"  PyTorch: {self.variant} ({', '.join(w['file'] for w in wheels)})")
        self.say(f"Downloading the {label} ({total / 1e9:.1f} GB, only once)...")
        self.wheels_dir.mkdir(parents=True, exist_ok=True)
        offset = 0
        state = {"declared": False}

        def report(done: int, status: str = "running") -> None:
            at = offset + done
            if status == "running" and state["declared"]:
                self.w.progress("download", "gpu-libs", at)
            elif status in ("running", "checking"):
                self.w.item("download", "gpu-libs", label, total, status, done=at)
                state["declared"] = status == "running"

        for w in wheels:
            try:
                self.M.fetch_file(w["url"], self.wheels_dir / w["file"], w["size"], w["sha256"],
                                  label if w["name"] == "torch" else f"{label}: torchaudio", report=report)
            except self.M.DownloadError as exc:
                raise self.download_stop(exc, f"the {label}") from exc
            self.verified.add(w["file"])
            offset += w["size"]
            state["declared"] = False
        self.w.item("download", "gpu-libs", label, total, "done")

    def fetch_models(self, models: list[dict]) -> None:
        self.say(f"Downloading the speech models ({sum(map(self.model_size, models)) / 1e9:.1f} GB)...")
        state = {"current": None}

        def report(m: dict, done: int, status: str) -> None:
            if status == "running" and state["current"] == m["id"]:
                self.w.progress("download", m["id"], done)
                return
            # a new model, or a change of status (checking: the next "running" declares it again)
            state["current"] = m["id"] if status == "running" else None
            self.w.item("download", m["id"], self.fse.model_label(m), self.model_size(m), status, done=done)

        try:
            self.M.download(models, out=sys.stdout, report=report)
        except self.M.DownloadError as exc:
            raise self.download_stop(exc, "the speech models") from exc
        for m in models:
            self.w.item("download", m["id"], self.fse.model_label(m), self.model_size(m), "done")

    def download_stop(self, exc, what: str) -> Stop:
        """A models.DownloadError as the plain message First start stops with."""
        self.log(f"Download failed: {exc}")
        message = {
            "network": f"{what[0].upper()}{what[1:]} could not be downloaded: no internet connection, or the "
                       "download server cannot be reached. Check the connection, then try again; it continues "
                       "where it stopped.",
            "disk": f"The disk is full: {what} need more free space. Free some space (you can delete the "
                    "downloaded zip), then try again; it continues where it stopped.",
            "checksum": f"A downloaded file of {what} was damaged (it did not match its checksum) and was "
                        "deleted. Try again to download it again.",
        }[exc.kind]
        return Stop(exc.kind, message, "download", detail=str(exc))

    def prepare(self) -> None:
        if self.torch_done() and self.wheels_dir.exists():  # a run ended between the marker and their removal
            self.remove_wheels()
        place = self.to_place()[0]
        if self.marker.exists() and self.torch_done() and not place:
            self.w.step("prepare", "done", "Already done")
            return
        self.begin("prepare")
        if not self.torch_done():
            self.install_torch()
        if place:
            self.place_models(place)
        if not self.marker.exists():
            self.warm_up()
        if self.wheels_dir.exists():  # a virus scanner held one when they were removed: 3.5 GB not to keep
            self.remove_wheels()
        self.w.step("prepare", "done")

    def remove_wheels(self) -> None:
        """cache\\wheels\\ removed, with a few tries: a virus scanner may be reading a wheel just now."""
        for attempt in range(5):
            shutil.rmtree(self.wheels_dir, ignore_errors=True)
            if not self.wheels_dir.exists():
                return
            time.sleep(1)
        self.log(f"  could not remove {self.wheels_dir}")

    def install_torch(self) -> None:
        """Put PyTorch into the ASR environment: the validated CUDA build, or on a PC without an
        NVIDIA driver the same version built for the processor.

        The zip leaves out the files the CUDA wheels hold byte for byte (build_windows.py
        split_torch; envs\\qwen3-asr-torch.json lists the wheels and the files), and step 2
        downloaded the wheels of the variant into cache\\wheels\\. cuda: each listed file is
        written from its wheel, checked against its sha256 and given its time back, so the
        environment is the validated one file for file. cpu: pip installs the CPU wheels offline
        over the CUDA build's remains (its RECORD, .pyc files and launchers, which pip removes
        first). Then the marker .torch-installed (with the variant), and the wheels go.
        """
        label, wheels = TORCH_PLACING[self.variant], self.wheels()
        total = self.torch["bytes"] if self.variant == "cuda" else None
        self.say(f"{label}...")
        self.w.item("prepare", "gpu-libs", label, total, "running", done=0)
        if total:
            free = shutil.disk_usage(self.root).free
            if free < total + GB:
                raise Stop("disk", f"There is not enough free disk space for the GPU libraries: they need about "
                                   f"{(total + GB) / 1e9:.1f} GB, and there are {free / 1e9:.1f} GB. Free some space "
                                   "(you can delete the downloaded zip), then try again.", "prepare")
        start = time.monotonic()
        if self.variant == "cuda":
            self.write_from_wheels(lambda done: self.w.progress("prepare", "gpu-libs", done))
        else:
            self.pip_install(wheels)
        (self.envs / "qwen3-asr" / ".torch-installed").write_text(
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n{self.variant}\n", encoding="utf-8")
        self.log(f"  PyTorch ({self.variant}) in place in {time.monotonic() - start:.0f} s")
        self.remove_wheels()
        self.w.item("prepare", "gpu-libs", label, total, "done")

    def write_from_wheels(self, progress) -> None:
        """The CUDA build: every listed file from its wheel, checked, with its time. A wheel that
        gives a file other than the pinned one (damaged on the disk) is deleted, to be downloaded
        again."""
        import hashlib
        import zipfile
        prefix = self.envs / "qwen3-asr"
        by_wheel: dict[str, list] = {}
        for f in self.torch["files"]:
            by_wheel.setdefault(f["wheel"], []).append(f)
        done = 0
        for wheel, files in by_wheel.items():
            bad = None
            try:
                with zipfile.ZipFile(self.wheels_dir / wheel) as z:
                    for f in files:
                        dest = prefix / f["path"]
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        h = hashlib.sha256()
                        with z.open(f["member"]) as src, dest.open("wb") as out:
                            for chunk in iter(lambda: src.read(8 << 20), b""):
                                h.update(chunk)
                                out.write(chunk)
                                done += len(chunk)
                                progress(done)
                        if h.hexdigest() != f["sha256"]:
                            dest.unlink(missing_ok=True)
                            bad = f["path"]
                            break
                        os.utime(dest, (f["mtime"], f["mtime"]))
            except (zipfile.BadZipFile, KeyError, EOFError, zlib.error) as exc:
                bad = f"{wheel}: {exc}"
            except OSError as exc:  # writing: a full disk, or a file a virus scanner holds just now
                if exc.errno == errno.ENOSPC:
                    raise Stop("disk", "The disk is full: the GPU libraries need more free space. Free some "
                                       "space (you can delete the downloaded zip), then try again.", "prepare",
                               detail=str(exc)) from exc
                raise Stop("other", "The GPU libraries could not be written. Try again; if it happens again, "
                                    "please send the log.", "prepare", detail=str(exc)) from exc
            if bad:
                (self.wheels_dir / wheel).unlink(missing_ok=True)  # closed now: Windows lets it go
                raise Stop("checksum", "A file of the GPU libraries came out different from the tested one, so "
                                       "the downloaded GPU libraries were deleted. Try again to download them "
                                       "again.", "prepare", detail=bad)
        self.log(f"  PyTorch: {len(self.torch['files'])} files ({self.torch['bytes'] / 1e9:.1f} GB) "
                 "written from the wheels")

    def pip_install(self, wheels: list) -> None:
        """The processor build: pip installs the downloaded wheels, offline, without dependencies (the
        environment has them); it removes the CUDA build's remains first."""
        prefix = self.envs / "qwen3-asr"
        result = subprocess.run([str(prefix / "python.exe"), "-m", "pip", "install", "--no-deps", "--no-index",
                                 "--no-cache-dir", "--disable-pip-version-check",
                                 *[str(self.wheels_dir / w["file"]) for w in wheels]],
                                env=env_vars(prefix), capture_output=True, text=True, errors="replace",
                                creationflags=NO_WINDOW)
        self.log(f"--- pip install (exit {result.returncode})\n{result.stdout}{result.stderr}")
        if result.returncode != 0:
            raise Stop("other", "PyTorch for the processor could not be installed. Try again; if it happens "
                                "again, please send the log.", "prepare", detail=result.stderr.strip()[-300:])

    def place_models(self, models: list[dict]) -> None:
        """Models with every file here: each Hugging Face blob linked (else copied) into its
        snapshot, as the download would, and refs/main written. The zip's files are not checked
        again (as the regular install trusts its cache)."""
        files = [(m, f, used, stored) for m in models for f, used, stored in self.M.targets(m)
                 if not self.complete(f, used)]
        total, done = sum(f["size"] for _, f, _, _ in files), 0
        if total:
            self.say("Putting the speech models in place...")
            self.w.item("prepare", "models", PLACING_MODELS, total, "running", done=0)
        for m, f, used, stored in files:
            self.M.place(m, used, stored)
            done += f["size"]
            self.w.progress("prepare", "models", done)
        for m in models:
            self.M.finish_repo(m)
        if total:
            self.w.item("prepare", "models", PLACING_MODELS, total, "done")

    def warm_up(self) -> None:
        self.say("Preparing (about a minute, only once)...")
        start = time.monotonic()
        results: dict[str, subprocess.CompletedProcess] = {}

        def load(name: str) -> None:
            prefix = self.envs / name
            results[name] = subprocess.run([str(prefix / "python.exe"), "-W", "ignore", "-c", WARM[name]],
                                           env=env_vars(prefix, self.root), capture_output=True, text=True,
                                           errors="replace", creationflags=NO_WINDOW)
            if results[name].returncode == 0 and name != "mfa":  # mfa: once `mfa version` ran too
                self.w.item("prepare", name, self.fse.ENV_LABELS[name], None, "done")

        for name in ENVS:
            self.w.item("prepare", name, self.fse.ENV_LABELS[name], None, "running")
        threads = [threading.Thread(target=load, args=(n,)) for n in ENVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        mfa = subprocess.run([str(self.envs / "mfa" / "Scripts" / "mfa.exe"), "version"],
                             env=env_vars(self.envs / "mfa", self.root), capture_output=True, text=True,
                             errors="replace", creationflags=NO_WINDOW)
        for n in ENVS:
            r = results[n]
            self.log(f"--- {n} (exit {r.returncode})\n{r.stdout}{r.stderr}")
        self.log(f"--- mfa version (exit {mfa.returncode})\n{mfa.stdout}{mfa.stderr}")
        failed = [n for n in ENVS if results[n].returncode != 0 or n == "mfa" and mfa.returncode != 0]
        self.log(f"  warm-up: {'FAILED ' + ' '.join(failed) if failed else 'ready'} ({time.monotonic() - start:.0f} s)")
        if failed:
            raise Stop("system", f"Part of the app could not be loaded "
                                 f"({', '.join(self.fse.ENV_LABELS[n].lower() for n in failed)}). Please extract "
                                 "the download again.", "prepare", again=False)
        self.w.item("prepare", "mfa", self.fse.ENV_LABELS["mfa"], None, "done")
        gpu = next((line[5:].strip() for line in results["qwen3-asr"].stdout.splitlines()
                    if line.startswith("GPU: ")), "-")
        if gpu != "-":
            text = f"Speech recognition will run on the graphics card: {gpu} (CUDA)."
        else:
            text = None if self.said_cpu else CPU_NOTE + CPU_TAIL[self.variant or "cuda"]
        if text:
            self.say(text)
            self.w.message(text)
        self.marker.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n{self.root}\n", encoding="utf-8")

    def main(self) -> int:
        self.w.start("windows", self.version, ["unpack", "download", "prepare"], resumed=self.resumed)
        self.log(f"\n{datetime.now():%Y-%m-%d %H:%M:%S} First start (Python part) in {self.root}")
        try:
            self.declare()
            self.unpack()
            self.download()
            self.prepare()
        except (Exception, KeyboardInterrupt) as exc:  # noqa: everything stops with an error the window shows
            if isinstance(exc, KeyboardInterrupt):
                exc = Stop("other", "First start was interrupted. Start Clausa again; it continues where it "
                                    "stopped.", again=False)
            elif not isinstance(exc, Stop):
                self.log(traceback.format_exc())
                exc = Stop("other", "First start stopped because of an unexpected problem. Try again; if it "
                                    "happens again, please send the log.", detail=f"{type(exc).__name__}: {exc}")
            stop = exc
            stop.step = stop.step or self.active
            self.log(f"Stopped: {stop.message}" + (f" ({stop.detail})" if stop.detail else ""))
            self.w.error(stop.kind, stop.message, step=stop.step, detail=stop.detail, again=stop.again)
            print(f"\nStopped: {stop.message}" + (f" ({stop.detail})" if stop.detail else ""), flush=True)
            return EXIT.get(stop.kind, 5)
        self.w.done()
        self.say("Done.")
        return 0


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]).resolve()
    return FirstStart(root).main()


if __name__ == "__main__":
    raise SystemExit(main())
