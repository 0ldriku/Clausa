"""macOS First start, the part after the first environment: called by First start.command.

    python -m runner.first_start_macos --root <Clausa folder>

First start.command checks the Mac, clears the quarantine flag, writes the run's `start` and
unpacks qwen3-asr (whose Python this runs with), then starts the window
(runner/first_start_window.py) and this. It continues the run's events
(runner/first_start_events.py): it unpacks the other environments, downloads the speech models
(runner/models.py), loads the libraries once, writes envs/.first-start-done and sends `done`.
Each step skips what is in place, so after an interruption it continues where it stopped.

Exit codes: 0 done; 2 no connection; 3 disk full; 4 a download did not match its checksum;
5 anything else (each with an `error` event); 1 or a signal: it ended without one (Python
itself failed, or it was stopped). On an error it sends `error` (what happened, what to do) and prints
"Stopped: ..." for Terminal when the window is not open. Its log lines go to
logs/first-start.log; while the window is not open (logs/first-start-window.pid), short lines go
to Terminal too (the window may be still starting, or could not open).
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from runner.first_start_events import ENV_LABELS, Writer, events_path, last_run, model_label, read_events

ENVS = ["qwen3-asr", "mfa", "ja-electra310"]
WARM = {"qwen3-asr": "torch, mlx.core, mlx_qwen3_asr, qwen_asr, spacy, librosa, transformers, spacy_transformers, webview",
        "mfa": "montreal_forced_aligner, kalpy, pynini, _kalpy, sqlalchemy, librosa",
        "ja-electra310": "spacy, ginza, torch, transformers, sudachipy"}
GB = 1 << 30
EXIT = {"network": 2, "disk": 3, "checksum": 4}
CHILDREN: list[subprocess.Popen] = []


class Stop(Exception):
    def __init__(self, kind: str, message: str, step: str | None, detail: str | None = None, again: bool = True):
        super().__init__(message)
        self.kind, self.message, self.step, self.detail, self.again = kind, message, step, detail, again


class FirstStart:
    def __init__(self, root: Path):
        self.root = root
        self.envs = root / "envs"
        self.log_path = root / "logs" / "first-start.log"
        self.marker = self.envs / ".first-start-done"
        self.w = Writer(root, heartbeat=True)
        events = read_events(events_path(root))
        self.run = last_run(events)
        self.resumed = bool(self.run and self.run[0].get("resumed"))
        # the sizes items had in earlier runs: a tar is deleted once it is unpacked
        self.active = None  # the step that is running
        self.sizes = {(e.get("step"), e.get("item")): e["total"] for e in events
                      if e.get("event") == "item" and e.get("total") is not None}

    # ---------- output ----------
    def log(self, text: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")

    def window_open(self) -> bool:
        return (self.root / "logs" / "first-start-window.pid").exists()

    def say(self, text: str) -> None:
        """A line for Terminal while the window is not open; always in the log."""
        self.log(text)
        if not self.window_open():
            print(text, flush=True)

    # ---------- the steps ----------
    def begin(self, step: str) -> None:
        self.active = step
        self.w.step(step, "running")

    def declare(self, models: list[dict]) -> None:
        for name in ENVS:
            tar = self.envs / f"{name}.tar"
            if tar.exists() and name != "qwen3-asr":  # First start.command unpacks qwen3-asr, which this runs with
                total, status = tar.stat().st_size, "waiting"
            elif (self.envs / name / "bin" / "python").exists():
                total, status = self.sizes.get(("unpack", name)), "done"
            else:
                raise Stop("other", f"Part of the app is missing (neither envs/{name}.tar nor envs/{name} is "
                                     "there). Please unzip the download again.", "unpack", again=False)
            self.w.item("unpack", name, ENV_LABELS[name], total, status)
        from runner import models as M
        for m in models:
            total = sum(f["size"] for f in m["files"])
            if not M.problems(m):
                self.w.item("download", m["id"], model_label(m), total, "done")
            else:
                self.w.item("download", m["id"], model_label(m), total, "waiting")
        for name in ENVS:
            self.w.item("prepare", name, ENV_LABELS[name], None, "done" if self.marker.exists() else "waiting")

    def unpack(self) -> None:
        todo = [n for n in ENVS[1:] if (self.envs / f"{n}.tar").exists()]
        if not todo:
            self.w.step("unpack", "done", None if any(self.unpacked_now()) else "Already done")
            return
        self.begin("unpack")
        for name in todo:
            self.unpack_one(name)
        self.w.step("unpack", "done")

    def unpacked_now(self):
        for e in self.run:
            if e.get("event") == "item" and e.get("step") == "unpack" and e.get("status") == "running":
                yield e.get("item")

    def unpack_one(self, name: str) -> None:
        tar = self.envs / f"{name}.tar"
        size = tar.stat().st_size
        free = shutil.disk_usage(self.envs).free
        need = size + 2 * GB  # the unpacked environment, plus 2 GB to spare
        if free < need:
            self.log(f"  not enough space for {name}: {free} bytes free, {need} needed")
            raise Stop("disk", f"Unpacking needs about {-(-need // GB)} GB free on this disk; there are "
                               f"{free // GB} GB. Free some space (you can delete the downloaded zip), then try "
                               "again.", "unpack")
        self.say(f"Unpacking {ENV_LABELS[name].lower()} ({size / 1e9:.1f} GB)...")
        start = time.monotonic()
        work = self.envs / f".unpacking-{name}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        self.w.item("unpack", name, ENV_LABELS[name], size, "running", done=0)
        err = self.log_path.open("ab")
        proc = subprocess.Popen(["/usr/bin/tar", "-xf", "-", "-C", str(work)], stdin=subprocess.PIPE, stderr=err)
        CHILDREN.append(proc)
        done = 0
        try:
            with tar.open("rb") as src:
                while chunk := src.read(4 << 20):
                    proc.stdin.write(chunk)
                    done += len(chunk)
                    self.w.progress("unpack", name, done)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        finally:
            ok = proc.wait() == 0
            CHILDREN.remove(proc)
            err.close()
        if not ok:
            shutil.rmtree(work, ignore_errors=True)
            raise Stop("disk", f"{ENV_LABELS[name]} could not be unpacked (is the disk full?). Free some space, "
                               "then try again.", "unpack", detail=f"tar exited with {proc.returncode} on envs/{name}.tar")
        self.w.progress("unpack", name, done, force=True)
        shutil.rmtree(self.envs / name, ignore_errors=True)
        try:
            (work / name).rename(self.envs / name)
            work.rmdir()
        except OSError as exc:
            raise Stop("other", f"{ENV_LABELS[name]} could not be moved into place in envs/. Try again.",
                       "unpack", detail=str(exc))
        subprocess.run(["/usr/bin/xattr", "-drs", "com.apple.quarantine", str(self.envs / name)],
                       stderr=subprocess.DEVNULL)
        tar.unlink()
        self.w.item("unpack", name, ENV_LABELS[name], size, "done")
        self.log(f"  {name} unpacked in {time.monotonic() - start:.0f} s")

    def download(self, models: list[dict]) -> None:
        from runner import models as M
        todo = [m for m in models if M.problems(m)]
        if not todo:
            if self.resumed:
                self.w.step("download", "done", "Already done")
            else:  # a zip made with --include-models
                self.w.step("download", "skipped", "The speech models came with the download.")
            return
        self.begin("download")
        left = sum(f["size"] for m in todo for f in m["files"])
        self.say(f"Downloading the speech models ({left / 1e9:.1f} GB)...")
        out = open(self.log_path, "a", encoding="utf-8")  # runner.models' own lines
        shown: dict[str, int] = {}  # model -> tenths shown in Terminal

        def report(m: dict, done: int, status: str) -> None:
            total = sum(f["size"] for f in m["files"])
            if not self.window_open():  # Terminal: a line at every tenth, as runner.models prints to a log
                tenth = 10 if status == "done" else min(9, done * 10 // max(total, 1))
                if tenth > shown.get(m["id"], -1):
                    shown[m["id"]] = tenth
                    print(f"  {m['label']}: {done / 1e6:,.0f} of {total / 1e6:,.0f} MB"
                          + (", ok" if status == "done" else ""), flush=True)
            if status == "running" and self.current == m["id"]:
                self.w.progress("download", m["id"], done)
                return
            # a new model, or a change of status (checking: the next "running" declares it again)
            self.current = m["id"] if status == "running" else None
            self.w.item("download", m["id"], model_label(m), total, status, done=done)

        self.current = None
        try:
            M.download(todo, out=out, report=report)
        except M.DownloadError as exc:
            what = {"network": "The speech models could not be downloaded: no internet connection, or the "
                               "download server cannot be reached. Check the connection, then try again; it "
                               "continues where it stopped.",
                    "disk": "The disk is full: the speech models need more free space. Free some space (you can "
                            "delete the downloaded zip), then try again; it continues where it stopped.",
                    "checksum": "A downloaded model file was damaged (it did not match its checksum) and was "
                                "deleted. Try again to download it again."}[exc.kind]
            raise Stop(exc.kind, what, "download", detail=str(exc)) from exc
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise Stop("disk", "The disk is full. Free some space, then try again; it continues where it "
                                   "stopped.", "download", detail=str(exc)) from exc
            raise
        finally:
            out.close()
        for m in models:
            self.w.item("download", m["id"], model_label(m), sum(f["size"] for f in m["files"]), "done")
        self.w.step("download", "done")

    def prepare(self) -> None:
        if self.marker.exists():
            self.w.step("prepare", "done", "Already done")
            return
        self.begin("prepare")
        self.say("Preparing (about a minute, only once)...")
        start = time.monotonic()
        procs = {}
        for name in ENVS:
            prefix = self.envs / name
            env = {"HOME": os.environ.get("HOME", ""), "CONDA_PREFIX": str(prefix),
                   "PATH": f"{prefix}/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
            out = open(self.root / "logs" / f"first-start.log.{name}", "wb")
            procs[name] = (subprocess.Popen([str(prefix / "bin" / "python"), "-W", "ignore", "-c", f"import {WARM[name]}"],
                                            env=env, stdout=out, stderr=subprocess.STDOUT, cwd=str(self.root)), out)
            CHILDREN.append(procs[name][0])
            self.w.item("prepare", name, ENV_LABELS[name], None, "running")
        failed = []
        waiting = dict(procs)
        while waiting:
            for name, (proc, out) in list(waiting.items()):
                if proc.poll() is None:
                    continue
                del waiting[name]
                CHILDREN.remove(proc)
                out.close()
                part = self.root / "logs" / f"first-start.log.{name}"
                text = part.read_text(encoding="utf-8", errors="replace")
                part.unlink()
                if text.strip():
                    self.log(f"--- {name}\n{text}")
                if proc.returncode == 0:
                    self.w.item("prepare", name, ENV_LABELS[name], None, "done")
                else:
                    failed.append(name)
            time.sleep(0.2)
        self.log(f"  warm-up: {'FAILED ' + ' '.join(failed) if failed else 'ready'} ({time.monotonic() - start:.0f} s)")
        if failed:
            raise Stop("system", f"Part of the app could not be loaded ({', '.join(ENV_LABELS[n].lower() for n in failed)}). "
                                 "Please unzip the download again.", "prepare", again=False)
        self.marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        self.w.step("prepare", "done")

    def main(self) -> int:
        from runner import models as M
        models = M.needed()
        try:
            self.declare(models)
            self.unpack()
            self.download(models)
            self.prepare()
        except Exception as exc:  # noqa: everything stops with an error the window shows
            if not isinstance(exc, Stop):
                import traceback
                self.log(traceback.format_exc())
                exc = Stop("other", "First start stopped because of an unexpected problem. Try again; if it "
                                    "happens again, please send the log.", None, detail=f"{type(exc).__name__}: {exc}")
            stop = exc
            stop.step = stop.step or self.active
            self.log(f"Stopped: {stop.message}" + (f" ({stop.detail})" if stop.detail else ""))
            self.w.error(stop.kind, stop.message, step=stop.step, detail=stop.detail, again=stop.again)
            if not self.window_open():
                print(f"\nStopped: {stop.message}" + (f" ({stop.detail})" if stop.detail else ""), flush=True)
            return EXIT.get(stop.kind, 5)
        self.w.done()
        self.say("Done.")
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runner.first_start_macos", description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="the Clausa folder")
    args = ap.parse_args(argv)

    def stop(signum, frame) -> None:  # the window was closed: end the step's processes too
        for proc in list(CHILDREN):
            try:
                proc.terminate()
            except OSError:
                pass
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return FirstStart(args.root.resolve()).main()


if __name__ == "__main__":
    raise SystemExit(main())
