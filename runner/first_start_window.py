"""The First start window: shows First start's progress in the app's design.

    python -m runner.first_start_window --root <Clausa folder> [--driver <command ...>]

It reads <folder>/logs/first-start-events.jsonl (runner/first_start_events.py) and shows
runner/app_ui/first-start.html, which polls the folded state. It never opens the app; it tells
the program that started it by its exit code (docs/development.md, "The app
zips"):

    0   "Open Clausa" was clicked (after done): the starter opens the app
    1   closed before done
    3   closed after done without "Open Clausa"
    75  "Try again" was clicked (watch-only): the starter runs First start again
    10  the window could not open, or failed (any failure before its page loaded gives 10, never
        1): the starter goes on without it

Without --driver it only watches (macOS: First start.command does the work and is the `pid` of
the run's `start`; if that process is gone before done or an error, the page says so). With
--driver (Windows) it runs the command, restarts it on Try again and stops it, with its child
processes, when the window is closed; the command's output goes to logs/first-start.log.
While the window is open, logs/first-start-window.pid holds its process id.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from runner import paths
from runner.first_start_events import Reader, events_path, fold

OPEN, CLOSED, CLOSED_READY, AGAIN, NO_WINDOW = 0, 1, 3, 75, 10


def pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would end the process on Windows
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Api:
    """What the page calls. pywebview calls these from its own threads.

    Everything else is private (a leading underscore): pywebview offers the page every public
    attribute of this object and walks into each one to find methods, so a public reference to the
    window would have it walk the native window's whole object tree (WinForms on Windows: minutes,
    and the page's `loaded` comes only after the walk)."""

    def __init__(self, root: Path, driver: list[str] | None):
        self._root = root
        self._driver = driver
        self._log = root / "logs" / "first-start.log"
        self._reader = Reader(events_path(root))
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stopped: dict | None = None  # the error the window adds when its writer is gone
        self._stopped_run = None  # the run it belongs to
        self._code = CLOSED
        self._window = None

    # ---------- the driver (--driver) ----------
    def _start_driver(self) -> None:
        self._log.parent.mkdir(parents=True, exist_ok=True)
        out = self._log.open("ab")
        kw: dict = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT, "cwd": str(self._root)}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kw["start_new_session"] = True  # its own process group: stop() reaches its children too
        try:
            self._proc = subprocess.Popen(self._driver, **kw)
        except OSError as exc:
            self._stopped = {"event": "error", "kind": "system", "again": True, "step": None,
                            "message": "First start could not be started. Click Try again; if it happens again, "
                                       "please send the log.", "detail": str(exc)}
        finally:
            out.close()

    def _stop_driver(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            proc.wait()

    # ---------- the page's calls ----------
    def _events(self) -> list[dict]:
        self._reader.poll()
        return list(self._reader.events)

    def get_state(self) -> dict:
        with self._lock:
            events = self._events()
            state = fold(events)
            if self._stopped is not None and self._stopped_run is not None and state["run"] != self._stopped_run:
                self._stopped = None  # a new run has begun since (this window watches a new First start)
            gone = False
            if self._driver is not None:
                if state["phase"] == "running" and self._stopped is None and self._proc is not None \
                        and self._proc.poll() is not None:
                    self._stopped = {"event": "error", "kind": "system", "again": True, "step": None,
                                    "message": f"First start stopped unexpectedly (exit {self._proc.returncode}). "
                                               "Click Try again; it continues where it stopped."}
                    self._stopped_run = state["run"]
            elif state["pid"] and state["phase"] != "starting" and not pid_alive(int(state["pid"])):
                # the starter is gone (its Terminal window was closed): nothing will run First start
                # again or open the app when this window closes
                gone = True
                events = self._events()  # its last words may have come in since
                state = fold(events)
                if state["phase"] == "running" and self._stopped is None:
                    again = "start Clausa again" if state["platform"] == "windows" else "double-click First start again"
                    self._stopped = {"event": "error", "kind": "system", "again": False, "step": None,
                                    "message": "First start ended before it was done (was its Terminal window "
                                               f"closed?). To continue, {again}; it continues where it stopped."}
                    self._stopped_run = state["run"]
            if self._stopped is not None and state["phase"] in ("running", "starting"):
                state = fold(events + [dict(self._stopped, step=state["current"], _at=time.monotonic())])
            state["starter_gone"] = gone
            if gone and state["error"]:
                state["error"]["again"] = False
            return state

    def retry(self) -> None:
        with self._lock:
            state = fold(self._events())
            if self._stopped is None and state["phase"] != "error":
                return
            if self._driver is None:
                if state["pid"] and not pid_alive(int(state["pid"])):
                    return  # no one to run it again (the page shows no Try again then)
                self._code = AGAIN
                threading.Thread(target=self._window.destroy, daemon=True).start()
                return
            self._stop_driver()
            self._stopped = None
            self._start_driver()

    def open_app(self) -> None:
        with self._lock:
            if fold(self._events())["phase"] == "done":
                self._code = OPEN
                threading.Thread(target=self._window.destroy, daemon=True).start()

    def _final_code(self) -> int:
        """The exit code once the window has closed: what was clicked, else whether First start was done."""
        with self._lock:
            done = fold(self._events())["phase"] == "done"
        return CLOSED_READY if self._code == CLOSED and done else self._code

    def show_log(self) -> None:
        if not self._log.exists():
            return
        if sys.platform == "darwin":
            subprocess.Popen(["/usr/bin/open", "-t", str(self._log)])
        elif os.name == "nt":
            os.startfile(str(self._log))  # noqa: the default program for .log, Notepad
        else:
            subprocess.Popen(["xdg-open", str(self._log)])


def _test_hook(window, folder: Path) -> None:
    """Tests (CLAUSA_FIRST_START_TEST=<folder>, macOS): a snapshot of the page (WKWebView, so no
    screen recording permission) each time what it shows changes, the states in states.jsonl,
    and a click on the button whose id is written into <folder>/click ("close": the window's own
    close button)."""
    import json
    import AppKit
    from PyObjCTools import AppHelper
    folder.mkdir(parents=True, exist_ok=True)
    states = folder / "states.jsonl"  # numbered on after an earlier window's (Try again)
    sig_js = ("(function(){var s=document.querySelectorAll('.step');return [document.getElementById('fs').className,"
              "document.getElementById('title').textContent].concat([].map.call(s,function(e){return e.className+':'+"
              "[].map.call(e.querySelectorAll('.item'),function(i){return i.className}).join(',')})).concat(['buttons:'+"
              "[].filter.call(document.querySelectorAll('.foot-actions button'),function(b){return !b.hidden})"
              ".map(function(b){return b.id}).join(',')]).join('|')})()")
    last, n = None, len(states.read_text().splitlines()) if states.exists() else 0

    def snap(name: str) -> None:
        done = threading.Event()

        def take() -> None:
            def handler(image, error) -> None:
                if image is not None:
                    rep = AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
                    rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, None).writeToFile_atomically_(
                        str(folder / name), True)
                done.set()
            try:
                view = window.native.webview if hasattr(window.native, "webview") else window.native.contentView()
                view.takeSnapshotWithConfiguration_completionHandler_(None, handler)
            except Exception:
                traceback.print_exc()
                done.set()
        AppHelper.callAfter(take)
        done.wait(10)

    while True:
        time.sleep(0.4)
        try:
            click = folder / "click"
            if click.exists():
                target = click.read_text().strip()
                click.unlink()
                if target == "close":  # as the window's close button
                    threading.Thread(target=window.destroy, daemon=True).start()
                else:
                    window.evaluate_js(f"document.getElementById({json.dumps(target)}).click()")
            sig = window.evaluate_js(sig_js)
        except Exception:
            continue
        if sig and sig != last:
            last, n = sig, n + 1
            time.sleep(0.6)  # the page draws the state
            phase = sig.split("|")[0].replace("fs phase-", "")
            snap(f"{n:02d}-{phase}.png")
            with states.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"n": n, "t": time.time(), "sig": sig}) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runner.first_start_window", description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="the Clausa folder")
    ap.add_argument("--driver", nargs=argparse.REMAINDER, help="the command that runs First start (Windows)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    api = Api(root, args.driver or None)
    pidfile = root / "logs" / "first-start-window.pid"
    shown = threading.Event()  # the page has loaded
    code = None
    try:
        from runner.app import DARK_BG, LIGHT_BG, mac_app_name, system_light, windows_app_id
        if sys.platform == "darwin":
            mac_app_name("Clausa")
        elif os.name == "nt":
            windows_app_id("org.clausa.app")
        import webview
        window = webview.create_window(
            "Clausa: First start", url=str(paths.APP_UI_DIR / "first-start.html"), js_api=api,
            width=760, height=720, min_size=(600, 520), background_color=LIGHT_BG if system_light() else DARK_BG)
        api._window = window

        def loaded() -> None:
            shown.set()
            try:
                pidfile.write_text(str(os.getpid()), encoding="utf-8")
            except OSError:
                pass

        window.events.loaded += loaded
        if os.environ.get("CLAUSA_FIRST_START_TEST") and sys.platform == "darwin":
            threading.Thread(target=_test_hook, args=(window, Path(os.environ["CLAUSA_FIRST_START_TEST"])),
                             daemon=True).start()

        def quit_on_signal(signum, frame) -> None:
            threading.Thread(target=window.destroy, daemon=True).start()

        signal.signal(signal.SIGINT, quit_on_signal)
        signal.signal(signal.SIGTERM, quit_on_signal)
        if api._driver is not None:
            api._start_driver()
        icon = paths.APP_ICON_ICO if os.name == "nt" else paths.APP_ICON
        webview.start(icon=str(icon) if icon.exists() else None, gui="edgechromium" if os.name == "nt" else None)
    except Exception:
        # not shown, or failed while shown: either way the starter goes on without the window
        traceback.print_exc()
        code = NO_WINDOW
    finally:
        api._stop_driver()
        try:
            pidfile.unlink()
        except OSError:
            pass
    if code is None:
        # closed before its page ever loaded (pywebview gave up without an error): the starter goes on
        code = api._final_code() if shown.is_set() else NO_WINDOW
    for stream in (sys.__stdout__, sys.__stderr__):
        if stream is not None:
            stream.flush()
    os._exit(code)  # as the app: a call from the page answered while closing would keep it alive


if __name__ == "__main__":
    raise SystemExit(main())
