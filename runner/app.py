"""The desktop app (./clausa app): a pywebview window around runner/app_ui.

The page (runner/app_ui/app.js) holds no state of its own: it polls
Api.get_state() and calls the other Api methods through window.pywebview.api.

- The window shows projects (runner/projects.py): a named set of recordings in
  one language, kept in its own folder in the projects folder, with a copy of
  each recording, every analysis session and one combined uf_measures.csv. The
  dated run folders the app made before projects are listed read-only as
  earlier analyses.
- A session analyses some recordings of one project (Start: those not done;
  Analyse again: the ones chosen) through runner.run.run_pipeline unchanged, in
  a background thread, one session at a time whichever the project, so their
  results equal those of `./clausa run`. Its folder is the project's
  sessions/<YYYY-MM-DD HH.MM>/: the pipeline's output, the recordings as links
  (recordings/) and log.txt. Afterwards each recording takes its outcome from the
  session (done or failed); a Stop puts the unfinished ones back as they were.
- The runner works stage by stage over all recordings. The window shows its
  stages as five steps (STEPS); a recording's bar is the share of steps done
  for it, from the stage and per-file events (runner.run.Events), and moves
  in between by the time the stage is expected to take (STAGE_TIMES). The
  session's own share is the mean over its recordings.
- Everything the runner prints, and what the app reports itself, goes to the
  log: the window's Log panel (the last LOG_LINES lines), the session's log.txt
  and the app's log file (paths.LOGS_DIR). The log is in English.
- The window says everything in its own language (runner/app_ui/strings.js):
  what the app tells it (a stage, the device, an error, a notice) goes with a
  key, and params where needed, beside the English (msg). The language is
  decided at start (ui_language): CLAUSA_UI_LANGUAGE, else the one chosen in
  Settings (settings.json's ui_language), else the system's. The native parts
  (pywebview's dialogs, the Open dialog's filters) take their words from NATIVE.

Runs in the ASR environment, which has pywebview (cli.py starts it there).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from runner import __version__, paths
from runner.awake import keep_awake
from runner.projects import (AUDIO_DIR, CSV_NAME, PROJECT_FILE, Earlier, Project, ProjectError, Recording, Session,
                             DATED_FOLDER, audio_duration, earlier_analyses, list_projects, now_iso, read_measures,
                             trash)
from runner.run import (AUDIO_EXTS, DECODE_EXTS, GPU_MEMORY, GPU_MEMORY_ERROR, MODES, STAGES, GpuOutOfMemory,
                        Stopped, run_pipeline)

LANGS = {"en": "English", "ja": "Japanese"}
UI_LANGUAGES = ("en", "ja", "zh-CN")  # the window's languages (runner/app_ui/strings.js)
COPYRIGHT = "© 2026 Jialiang Lu and Reiko Sato"  # as in the About, LICENSE.md and the app bundles
LINKS = ("https://github.com/0ldriku/Clausa", "mailto:oldriku@proton.me")  # the About's links (Api.open_url)
# The words of the window's native parts by language, chosen once at start: a language chosen in
# Settings shows here at the next start (pywebview copies its words when the window is made).
# pywebview's own words (its localization keys): the quit confirmation and its buttons (macOS), the
# Save dialog's title (macOS) and its file filter (Windows). Not the macOS app menu: pywebview adds
# the name to About, Hide and Quit ("About Clausa"), which fits English word order alone, so the
# menu bar stays as pywebview has it. The app's own words start with app.: the Open dialog's
# filters, whose names are letters, digits and spaces only (pywebview reads "Name (*.wav;*.mp3)"),
# and the default name of a saved log. A language that lacks a word uses the English one.
NATIVE = {
    "en": {
        "global.quitConfirmation": "An analysis is running. Quit and stop it?",
        "global.quit": "Quit",
        "global.cancel": "Cancel",
        "global.saveFile": "Save file",
        "windows.fileFilter.allFiles": "All files",
        "app.audio_files": "Audio files",
        "app.all_files": "All files",
        "app.log_name": "Clausa log {date}",
    },
    "ja": {
        "global.quitConfirmation": "分析を実行中です。終了して分析を停止しますか？",
        "global.quit": "終了",
        "global.cancel": "キャンセル",
        "global.saveFile": "ファイルを保存",
        "windows.fileFilter.allFiles": "すべてのファイル",
        "app.audio_files": "音声ファイル",
        "app.all_files": "すべてのファイル",
        "app.log_name": "Clausaログ {date}",
    },
    "zh-CN": {
        "global.quitConfirmation": "分析正在进行。要退出并停止分析吗？",
        "global.quit": "退出",
        "global.cancel": "取消",
        "global.saveFile": "存储文件",
        "windows.fileFilter.allFiles": "所有文件",
        "app.audio_files": "音频文件",
        "app.all_files": "所有文件",
        "app.log_name": "Clausa 日志 {date}",
    },
}
TRASH = "Recycle Bin" if os.name == "nt" else "Trash"  # where trash() moves things, as the user knows it
TRASH_KEY = "recycle_bin" if os.name == "nt" else "trash"  # its key in the window (bin.<key>)
DEVICES = {"mlx": "Mac GPU · MLX", "cuda": "NVIDIA GPU · CUDA", "cpu": "CPU"}  # detect_device's answers, in English
LOG_LINES = 2000  # kept in memory for the Log panel

# The runner's stages as the window shows them: five steps.
#   step   1-5; Fillers and Islands share step 4, and decoding (.m4a only) is
#          counted in step 1 but takes no share of its bar
#   label  the stage's one name, in the window and the log (lower case inside a sentence)
#   share  of the step's bar, against the other stages of that step
STEPS = {
    "decode": (1, "Reading audio", 0),
    "asr_stage1": (1, "Speech recognition", 1),
    "mfa_stage2": (2, "Word alignment", 1),
    "clause": (3, "Clauses", 1),
    "gap": (4, "Filled pauses", 1),
    "islands": (4, "Pauses", 1),
    "caf": (5, "Measures", 1),
}
STEP_COUNT = 5
STEP_OF = {label: key for key, (_, label, _) in STEPS.items()}  # a failed_stage as saved, back to its key


@dataclass(frozen=True)
class StageTime:
    """How long a stage takes, in seconds (see STAGE_TIMES)."""
    load: float  # from the stage's start until its first file starts: environment, models
    first: float  # extra time of the first file (warm-up)
    fixed: float  # each file: fixed seconds ...
    rate: float  # ... plus seconds per second of audio
    tail: float  # after the last file, until the stage ends

    def stage(self, durations: list) -> float:
        """The whole stage for recordings of these lengths."""
        files = sum(self.fixed + self.rate * d for d in durations)
        return self.load + (self.first if durations else 0) + files + self.tail


# Expected stage times on the M1 Max (conda route), per language. Fitted to the
# events.jsonl of ALL_001 and CCH16-ST1 run alone (six runs each) and of one
# batch of three recordings per language (19, 96 and 344 s in English; 58, 116
# and 233 s in Japanese). They move the bars inside a stage
# and give the time left; a run rescales them by how fast its finished stages
# were (Run.speed), so another machine corrects them after its first stage.
_DECODE = StageTime(1.0, 0, 0.1, 0.005, 0)
STAGE_TIMES = {
    "en": {"decode": _DECODE,
           "asr_stage1": StageTime(3.0, 5.0, 0.6, 0.042, 0.7),
           "mfa_stage2": StageTime(2.2, 0, 13.5, 0.025, 0.2),
           "clause": StageTime(8.0, 0, 0.8, 0.025, 0.6),
           "gap": StageTime(4.2, 0, 0.05, 0, 0.4),
           "islands": StageTime(1.8, 0, 0, 0, 0),
           "caf": StageTime(1.8, 0, 0.02, 0, 0.1)},
    "ja": {"decode": _DECODE,
           "asr_stage1": StageTime(3.0, 6.0, 2.8, 0.021, 0.7),
           "mfa_stage2": StageTime(2.2, 0, 14.5, 0.035, 0.2),
           "clause": StageTime(8.5, 0, 5.5, 0.008, 0.5),
           "gap": StageTime(4.2, 0, 0.05, 0, 0.4),
           "islands": StageTime(1.8, 0, 0, 0, 0),
           "caf": StageTime(1.9, 0, 0.02, 0, 0.1)},
}
UNKNOWN_DURATION = 120.0  # seconds assumed for a recording whose length could not be read
ESTIMATE_CAP = 0.9  # an estimate fills at most this much of a stage's share before the stage reports the file
SPEED_RANGE = (0.5, 3.0)  # limits of a run's speed factor against STAGE_TIMES
# Stages whose scripts print a line per file (runner.run._FileLines); in the
# others only the stage boundaries move the bars.
PER_FILE = {"asr_stage1", "mfa_stage2", "clause", "gap", "caf"}
WRITES = {"mfa_stage2": "words", "clause": "clauses"}  # the TextGrid each stage writes
RUN_DEFAULTS = {k: p.default for k, p in inspect.signature(run_pipeline).parameters.items()
                if p.default is not inspect.Parameter.empty and k not in ("on_event", "stop")}
MODE = MODES[RUN_DEFAULTS["paper_compat"]]  # the default mode: the app runs with run_pipeline's defaults
PACKAGES = ("mlx", "mlx-qwen3-asr", "qwen-asr", "torch", "transformers", "spacy", "pywebview")
_ERROR_LINE = re.compile(r"ERROR|FAILED|Traceback|Error processing|Exception")


def short_version(version: str = __version__) -> str:
    """'1.1.0' (or '1.1.0.dev0') -> '1.1', as the status bar shows it."""
    m = re.match(r"(\d+)\.(\d+)", version)
    return f"{m[1]}.{m[2]}" if m else version


def home_short(path: Path) -> str:
    """The path with the home folder written as ~ (on Windows, where ~ is not used: the path)."""
    if os.name == "nt":
        return str(path)
    try:
        rel = path.relative_to(Path.home())
    except ValueError:
        return str(path)
    return "~" if rel == Path(".") else f"~{os.sep}{rel}"


def fmt_seconds(s: float) -> str:
    s = round(s)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min" + (f" {s % 60} s" if s % 60 else "")
    return f"{s // 3600} h {s // 60 % 60} min"


def msg(key: str | None, text: str, **params) -> dict:
    """A message for the window: its key in runner/app_ui/strings.js and the values of its
    placeholders (a value may itself be a message), and the same in English (the log, and
    what the page shows without a key)."""
    return {"key": key, "params": params, "text": text}


def as_message(exc: Exception) -> dict:
    """An error as a message: a ProjectError with its key; any other (an OSError) as its English text alone."""
    if isinstance(exc, ProjectError) and exc.key:
        return msg(exc.key, str(exc), **exc.params)
    return msg(None, str(exc))


def not_done(error: dict) -> dict:
    """The answer of a call that could not be done: {"ok": False, "error": the English text,
    "key": ..., "params": ...} for a message."""
    return {"ok": False, "error": error["text"], "key": error["key"], "params": error["params"]}


def set_error(r: Recording, code: str | None = None, text: str | None = None, **params) -> None:
    """Why a recording failed (or, with no code, that it did not): the English text, saved for
    the log and for Clausa 1.0.1, and the code and params by which the window says it."""
    r.error, r.error_code, r.error_params = text, code, params if code else None


def detect_device(started: Callable[[subprocess.Popen], None] | None = None) -> str:
    """What Stage 1 runs on with --device auto: "mlx", "cuda" or "cpu" (DEVICES).

    The rule of pipeline/en/asr_mfa_en.py (load_models): MLX on Apple Silicon
    when mlx-qwen3-asr is installed, else CUDA when torch sees a GPU, else the
    CPU. The app runs in the environment Stage 1 runs in, so it can ask the
    same questions. started gets the child process that asks torch, so that
    closing the app can end it (it takes some seconds).
    """
    spec = importlib.util.spec_from_file_location("_clausa_asr_mlx", paths.PIPELINE_DIR / "shared" / "asr_mlx.py")
    asr_mlx = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # its dataclass looks itself up there
    spec.loader.exec_module(asr_mlx)
    if asr_mlx.available():
        return "mlx"
    try:  # in a child process: importing torch here would cost the app its memory
        with subprocess.Popen([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) as proc:
            if started is not None:
                started(proc)
            try:
                out, _ = proc.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise
        if out.strip().endswith("True"):
            return "cuda"
    except (OSError, subprocess.SubprocessError):
        pass
    return "cpu"


def ui_language_of(tag: str | None) -> str | None:
    """A language tag (ja-JP, zh-Hant-TW, en_GB ...) as one of UI_LANGUAGES: Japanese ja; any
    Chinese zh-CN (Simplified serves readers of Traditional better than English); else en.
    None for no tag."""
    tag = (tag or "").strip().replace("_", "-").lower()
    if not tag:
        return None
    if tag == "ja" or tag.startswith("ja-"):
        return "ja"
    if tag == "zh" or tag.startswith("zh-"):
        return "zh-CN"
    return "en"


def system_language() -> str | None:
    """The user's first preferred language, as a tag: macOS NSLocale.preferredLanguages, Windows
    GetUserPreferredUILanguages (else GetUserDefaultUILanguage), elsewhere LANGUAGE, LC_ALL,
    LC_MESSAGES or LANG. None if it cannot be told."""
    try:
        if sys.platform == "darwin":
            from Foundation import NSLocale
            preferred = NSLocale.preferredLanguages()
            return str(preferred[0]) if preferred else None
        if os.name == "nt":
            return _windows_language()
    except Exception:
        return None
    for name in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(name, "").split(":")[0].split(".")[0]
        if value and value not in ("C", "POSIX"):
            return value
    return None


def _windows_language() -> str | None:
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    count, size = wintypes.ULONG(0), wintypes.ULONG(0)
    MUI_LANGUAGE_NAME = 0x8
    if kernel32.GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, ctypes.byref(count), None, ctypes.byref(size)) \
            and size.value:
        names = ctypes.create_unicode_buffer(size.value)  # "ja-JP\0en-US\0\0": .value is the first
        if kernel32.GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, ctypes.byref(count), names, ctypes.byref(size)) \
                and names.value:
            return names.value
    name = ctypes.create_unicode_buffer(85)  # LOCALE_NAME_MAX_LENGTH
    if kernel32.LCIDToLocaleName(kernel32.GetUserDefaultUILanguage(), name, 85, 0):
        return name.value
    return None


def ui_language(setting: str | None) -> str:
    """The window's language at start: CLAUSA_UI_LANGUAGE (a tests' setting), else the one chosen in
    Settings, else the system's, else English."""
    return (ui_language_of(os.environ.get("CLAUSA_UI_LANGUAGE"))
            or (setting if setting in UI_LANGUAGES else None)
            or ui_language_of(system_language()) or "en")


def native(language: str, key: str, **params) -> str:
    """A word of the native parts (NATIVE) in this language, else in English; {name}s filled from params."""
    text = NATIVE.get(language, {}).get(key) or NATIVE["en"][key]
    return text.format(**params) if params else text


def native_filter(language: str, key: str, patterns: str) -> str:
    """A file filter for pywebview's dialogs, "Name (*.wav;*.mp3)": the name in this language
    where pywebview can read it (letters, digits and spaces), else in English."""
    name = native(language, key)
    if not re.fullmatch(r"[\w ]+", name):
        name = NATIVE["en"][key]
    return f"{name} ({patterns})"


def native_localization(language: str) -> dict:
    """pywebview's words (its localization dict) in this language."""
    return {key: native(language, key) for key in NATIVE["en"] if not key.startswith("app.")}


def mac_app_name(name: str) -> None:
    """macOS: call the app `name` in the Dock and the menu bar instead of Python.

    Run from a plain python, the app has no bundle of its own, so macOS names it
    after the executable. CFBundleName in the main bundle's info dictionary names
    pywebview's menu items (Hide, Quit ...); the name the Dock and the menu bar show
    is the one Launch Services registered at check-in, which only its
    _LSSetApplicationInformationItem changes (Chromium renames its processes
    with it). Both are best effort: if either fails, the app keeps Python's name.
    """
    import ctypes
    try:
        import AppKit
        from Foundation import NSBundle
        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = name
            # what the app menu's standard About panel shows, if it reads them after this
            info["CFBundleShortVersionString"] = info["CFBundleVersion"] = __version__
            info["NSHumanReadableCopyright"] = COPYRIGHT
        AppKit.NSApplication.sharedApplication()  # checks the app in with Launch Services
        services = ctypes.CDLL("/System/Library/Frameworks/CoreServices.framework/CoreServices")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        services._LSGetCurrentApplicationASN.restype = ctypes.c_void_p
        services._LSSetApplicationInformationItem.restype = ctypes.c_int32
        services._LSSetApplicationInformationItem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                              ctypes.c_void_p, ctypes.c_void_p]
        key = ctypes.c_void_p.in_dll(services, "_kLSDisplayNameKey")
        value = cf.CFStringCreateWithCString(None, name.encode("utf-8"), 0x08000100)  # kCFStringEncodingUTF8
        services._LSSetApplicationInformationItem(-2, services._LSGetCurrentApplicationASN(), key, value, None)
    except Exception:
        pass


def windows_app_id(app_id: str) -> None:
    """Windows: the taskbar groups the window as its own app (with the window's icon),
    not with other Python programs. Best effort."""
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id))
    except (OSError, AttributeError):
        pass


APPEARANCES = ("system", "light", "dark")  # Settings › Appearance; system: the system's light or dark
LIGHT_BG, DARK_BG = "#fcfcfc", "#0c0c0c"  # the window before the page paints (app.css: gray-1 of each theme)


def system_light() -> bool:
    """Whether the system shows apps light; False if dark or unknown. macOS: AppleInterfaceStyle
    (Dark when dark), Windows: AppsUseLightTheme."""
    try:
        if sys.platform == "darwin":
            from Foundation import NSUserDefaults
            return NSUserDefaults.standardUserDefaults().stringForKey_("AppleInterfaceStyle") != "Dark"
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
                return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] != 0
    except Exception:
        pass
    return False


def native_appearance(code: str) -> None:
    """The window's frame in the appearance chosen in Settings (system: the system's). Best effort.
    macOS: the application's NSAppearance, which the page's prefers-color-scheme follows too.
    Windows: pywebview gives the title bar the system's theme (BrowserForm.is_dark_theme, asked again
    when the system's changes); here it answers with the chosen one, and open windows take it now."""
    try:
        if sys.platform == "darwin":
            import AppKit
            name = {"light": AppKit.NSAppearanceNameAqua, "dark": AppKit.NSAppearanceNameDarkAqua}.get(code)
            look = AppKit.NSAppearance.appearanceNamed_(name) if name else None

            def apply() -> None:
                AppKit.NSApplication.sharedApplication().setAppearance_(look)
            if threading.current_thread() is threading.main_thread():
                apply()
            else:
                from PyObjCTools import AppHelper
                AppHelper.callAfter(apply)
        elif os.name == "nt":
            from webview.platforms import winforms
            form = winforms.BrowserView.BrowserForm
            system = form.__dict__.get("_system_dark", form.is_dark_theme)
            form._system_dark = system
            form.is_dark_theme = system if code == "system" else (lambda self, dark=code == "dark": dark)
            for browser in list(winforms.BrowserView.instances.values()):
                browser.Invoke(winforms.Func[winforms.Type](browser.update_title_bar_theme))
    except Exception:
        pass


def system_text() -> str:
    if sys.platform == "darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()}{', ' + chip if chip else ''})"
    return platform.platform()


def code_version() -> str | None:
    """git describe of the checkout, if the app runs from one."""
    if not (paths.REPO_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "describe", "--always", "--dirty"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def package_versions() -> str:
    found = []
    for name in PACKAGES:
        try:
            found.append(f"{name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return ", ".join(found)


def open_path(path: Path, reveal: bool = False) -> None:
    """Open a file or folder in its default app; reveal: show it in its folder."""
    if sys.platform == "darwin":
        cmd = ["open", "-R", str(path)] if reveal else ["open", str(path)]
    elif os.name == "nt":
        if not reveal:
            os.startfile(path)  # noqa: S606 (Windows only)
        elif not _windows_select(path):
            subprocess.Popen(f'explorer /select,"{path}"')  # explorer reads its own command line
        return
    else:
        cmd = ["xdg-open", str(path.parent if reveal else path)]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _windows_select(path: Path) -> bool:
    """Windows: an Explorer window of the file's folder with the file selected
    (SHOpenFolderAndSelectItems, which takes any path, spaces and all). False if it failed."""
    import ctypes
    shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32
    shell32.ILCreateFromPathW.restype = ctypes.c_void_p
    shell32.ILCreateFromPathW.argtypes = [ctypes.c_wchar_p]
    shell32.SHOpenFolderAndSelectItems.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
    shell32.ILFree.argtypes = [ctypes.c_void_p]
    ole32.CoInitialize.restype = ctypes.c_long
    hr = ole32.CoInitialize(None)  # pywebview calls the API from threads of its own
    try:
        pidl = shell32.ILCreateFromPathW(str(path))
        if not pidl:
            return False
        try:
            return shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0) == 0
        finally:
            shell32.ILFree(pidl)
    finally:
        if hr >= 0:  # S_OK or S_FALSE; not on a thread COM had set up otherwise (RPC_E_CHANGED_MODE)
            ole32.CoUninitialize()


class Log:
    """The app's log: the Log panel's lines, the run's log.txt and the app's log file.

    Lines go to all three as they come; the app log file is
    <paths.LOGS_DIR>/app-<date>.log. A file that cannot be written is skipped,
    the app carries on.
    """

    def __init__(self, logs_dir: Path, console=None):
        self.lines: deque[dict] = deque(maxlen=LOG_LINES)
        self.run_path: Path | None = None
        self._logs_dir = logs_dir
        self._console = console
        self._lock = threading.RLock()
        self._run_file = None
        self._app_file = None
        self._app_date = None
        # Set while a run is being stopped: the stages' output then (the
        # KeyboardInterrupt tracebacks of the SIGINT that ends them) is info, not error.
        self.stopping = False

    def write(self, text: str, level: str = "info", *, echo: bool = True, show: bool = True) -> None:
        """Log one or more lines. echo: also print them on the console; show: in the Log panel."""
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with self._lock:
            for line in text.split("\n"):
                if show:
                    self.lines.append({"t": round(now.timestamp(), 3), "text": line, "level": level})
                self._to_files(f"{stamp}  {line}\n", now)
                if echo and self._console is not None:
                    try:
                        self._console.write(line + "\n")
                        self._console.flush()
                    except (OSError, ValueError):
                        pass

    def output(self, line: str) -> None:
        """A line the runner or a library printed (already on the console)."""
        line = line.rstrip("\r").rsplit("\r", 1)[-1]  # a progress bar redrawn with \r: its last state
        if line.strip():
            self.write(line, "error" if _ERROR_LINE.search(line) and not self.stopping else "info", echo=False)

    def start_run(self, path: Path, header: list[str]) -> None:
        """Start the run's log.txt: the header without timestamps, then every line."""
        with self._lock:
            self.end_run()
            self.stopping = False
            self.run_path = path
            try:
                self._run_file = path.open("w", encoding="utf-8")
                self._run_file.write("\n".join(header) + "\n\n")
                self._run_file.flush()
            except OSError:
                self._run_file = None
            for line in header:
                self.lines.append({"t": round(time.time(), 3), "text": line, "level": "info"})
                self._to_files(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}"[:-3] + f"  {line}\n", datetime.now(), run=False)

    def end_run(self) -> None:
        with self._lock:
            if self._run_file is not None:
                self._run_file.close()
                self._run_file = None

    def text(self) -> str:
        """The Log panel's lines as text (Copy, Save log as…)."""
        with self._lock:
            return "".join(f"{datetime.fromtimestamp(l['t']):%H:%M:%S}  {l['text']}\n" for l in self.lines)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.lines)

    def close(self) -> None:
        with self._lock:
            self.end_run()
            if self._app_file is not None:
                self._app_file.close()
            self._app_file, self._app_date = None, None

    def _to_files(self, entry: str, now: datetime, run: bool = True) -> None:
        if run and self._run_file is not None:
            try:
                self._run_file.write(entry)
                self._run_file.flush()
            except (OSError, ValueError):
                pass
        try:
            if self._app_date != now.date():
                if self._app_file is not None:
                    self._app_file.close()
                self._logs_dir.mkdir(parents=True, exist_ok=True)
                self._app_file = (self._logs_dir / f"app-{now:%Y-%m-%d}.log").open("a", encoding="utf-8")
                self._app_date = now.date()
            self._app_file.write(entry)
            self._app_file.flush()
        except (OSError, ValueError):
            self._app_file, self._app_date = None, None


class _Tee:
    """sys.stdout or sys.stderr in the app: output still goes to the console, and each line to the Log.

    The runner copies its stages' output to sys.stdout and sys.stderr (runner.run._copy), from
    several threads; each stream keeps its own unfinished line.
    """

    def __init__(self, stream, log: Log):
        self._stream = stream
        self._log = log
        self._pending = ""
        self._lock = threading.Lock()
        self.encoding = getattr(stream, "encoding", None) or "utf-8"
        self.errors = "replace"

    def write(self, s: str) -> int:
        if self._stream is not None:
            try:
                self._stream.write(s)
            except (OSError, ValueError):
                pass
        with self._lock:
            *lines, self._pending = (self._pending + s).split("\n")
        for line in lines:
            self._log.output(line)
        return len(s)

    def flush(self) -> None:
        if self._stream is not None:
            try:
                self._stream.flush()
            except (OSError, ValueError):
                pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if self._stream is None:
            raise OSError("no console")
        return self._stream.fileno()


@dataclass
class Run:
    """A session: one run of the pipeline over some recordings of one project."""
    project: Project
    dir: Path  # the session's folder, sessions/<YYYY-MM-DD HH.MM>
    language: str
    recs: dict  # id -> Recording, in processing order
    session: Session | None = None  # its entry in project.json
    stop: threading.Event = field(default_factory=threading.Event)
    status: str = "running"  # running, stopping, done, stopped or failed
    started: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))
    ended: float | None = None
    order: list = field(default_factory=list)  # file ids in the runner's order
    stages: list = field(default_factory=list)  # keys of the runner stages that run
    stage: str | None = None  # the last stage that started
    finished: set = field(default_factory=set)  # stages completed or skipped
    done: dict = field(default_factory=dict)  # stage -> ids completed or failed in it
    started_file: str | None = None  # a file between its started and completed events
    stage_clock: float = 0.0  # when the running stage started (monotonic)
    file_clock: float = 0.0  # when its current file became current: the stage's start or the last file done
    speed: float = 1.0  # actual / expected time of the finished stages (STAGE_TIMES), within SPEED_RANGE
    actual_s: float = 0.0
    expected_s: float = 0.0
    eta: float | None = None  # seconds left, renewed by Api._estimate


class Api:
    """The window's backend (window.pywebview.api); app.js defines the calls.

    The window shows one project at a time (or one earlier analysis, read-only);
    the sidebar lists them all. pywebview calls each method in a thread of its
    own, and a session has its own thread: all state is guarded by one lock.
    Only the public methods are the API (pywebview exposes no names starting
    with _). Calls that the user can get wrong (a name, the Trash) return
    {"ok": False, "error": message, "key": ..., "params": ...} instead of raising (not_done).
    """

    def __init__(self):
        console = sys.stdout
        self._log = Log(paths.LOGS_DIR, console)
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._log), _Tee(sys.stderr, self._log)
        self._lock = threading.RLock()
        self._adding = threading.Lock()  # one add at a time: ids are checked before the files are copied
        self._run: Run | None = None
        self._thread: threading.Thread | None = None
        self._window = None
        self._connected = False
        self._settings = self._load_settings()
        chosen = self._settings.get("ui_language")
        self._ui_setting = chosen if chosen in UI_LANGUAGES else None  # chosen in Settings; None: the system's
        self._ui_language = ui_language(self._ui_setting)  # the window's
        self._native_language = self._ui_language  # the native parts' (NATIVE), fixed at start
        look = self._settings.get("appearance")
        self._appearance = look if look in APPEARANCES else "system"
        saved = self._settings.get("projects_dir")
        self._root = Path(saved) if isinstance(saved, str) and os.path.isabs(saved) else paths.PROJECTS_DIR
        self._projects: list[Project] = []
        self._earlier: list[Earlier] = []
        self._earlier_recs: dict[str, list[Recording]] = {}
        self._folders_seen: tuple = ()
        self._earlier_seen: tuple = ()
        self._scanned = 0.0
        current = self._settings.get("current")
        self._current: str | None = current if isinstance(current, str) else None
        self._notice = {"id": 0, "text": ""}
        self._device = ""  # a key of DEVICES once known
        self._device_ready = threading.Event()
        self._device_proc: subprocess.Popen | None = None  # the child asking torch for the GPU, while it runs
        self._reload()
        threading.Thread(target=self._find_device, daemon=True).start()

    # ---------- state ----------

    def get_state(self) -> dict:
        with self._lock:
            if self._run is not None and self._run.status == "running":
                self._estimate(self._run)
            if not self._connected:
                self._connected = True
                self._log.write("The window is connected", show=False)
            self._rescan()
            kind, view = self._view()
            if kind == "project":
                project = {"id": view.id, "name": view.name, "language": view.language, "kind": "project",
                           "dir": str(view.folder), "created": view.created,
                           "sessions": sum(s.outcome in ("done", "failed") for s in view.sessions),
                           "last_session": self._last_session(view),
                           "csv": view.csv_path.exists()}
                recs = [self._rec_state(r, view) for r in view.recordings]
            elif kind == "earlier":
                project = {"id": view.id, "name": view.dir.name, "language": view.language or "en", "kind": "earlier",
                           "dir": str(view.dir), "created": "", "sessions": 1,
                           "last_session": self._folder_time(view.dir.name),
                           "csv": (view.dir / "uf" / CSV_NAME).exists()}
                recs = [self._rec_state(r, None, view.dir) for r in self._earlier_rows(view)]
            else:
                project, recs = None, []
            return {
                "app": {"version": short_version(), "version_full": __version__,
                        "device": DEVICES.get(self._device, ""), "device_key": self._device,
                        "sep": os.sep, "trash": TRASH, "trash_key": TRASH_KEY,
                        "ui_language": self._ui_language, "ui_language_setting": self._ui_setting,
                        "appearance": self._appearance},
                "settings": {"projects_dir": str(self._root), "projects_dir_short": home_short(self._root)},
                "projects": [self._project_item(p) for p in self._projects],
                "earlier": [{"id": e.id, "name": e.dir.name, "language": e.language, "count": len(e.files)}
                            for e in self._earlier],
                "project": project,
                "recordings": recs,
                "run": self._run_state(),
                "log": {"path": self._last_log(kind, view),  # the shown project's (also while it runs)
                        "lines": self._log.snapshot()},
                "notice": dict(self._notice),
            }

    def _project_item(self, p: Project) -> dict:
        run = self._run
        running = run is not None and run.status in ("running", "stopping") and run.project is p
        return {"id": p.id, "name": p.name, "language": p.language, "count": len(p.recordings),
                "failed": sum(r.status == "failed" for r in p.recordings),
                "status": "running" if running else "idle",
                "progress": round(self._session_pct(run), 1) if running else 0}

    def _run_state(self) -> dict:
        run = self._run
        if run is None:
            return {"status": "idle", "project_id": None, "project_name": "", "session_dir": None, "csv_path": None,
                    "started": "", "elapsed_s": 0, "eta_s": None, "progress": 0, "stage_index": 0,
                    "stage_count": STEP_COUNT, "stage_label": "", "stage": "", "current_file": None, "current_name": "",
                    "done_count": 0, "failed_count": 0, "total": 0}
        active = run.status in ("running", "stopping")
        elapsed = (run.ended or time.monotonic()) - run.started
        step, label = STEPS[run.stage][:2] if active and run.stage else (0, "")
        current = self._current_file(run) if active else None
        csv_path = run.project.csv_path
        return {
            "status": run.status,
            "project_id": run.project.id,
            "project_name": run.project.name,
            "session_dir": str(run.dir),
            "csv_path": str(csv_path) if csv_path.exists() else None,
            "started": run.started_at,
            "elapsed_s": round(elapsed, 1),
            # 0 once the estimate has run out: the page then says "almost done"
            "eta_s": max(0, round(run.eta)) if run.status == "running" and run.eta is not None else None,
            # the session's share done: every recording in it, a failed one counted as finished
            "progress": round(self._session_pct(run), 1),
            "stage_index": step,
            "stage_count": STEP_COUNT,
            "stage_label": label,
            "stage": run.stage if step else "",
            "current_file": current,
            "current_name": run.recs[current].name if current in run.recs else "",
            "done_count": sum(r.status == "done" for r in run.recs.values()),
            "failed_count": sum(r.status == "failed" for r in run.recs.values()),
            "total": len(run.recs),
        }

    def _rec_state(self, r: Recording, project: Project | None, folder: Path | None = None) -> dict:
        """A row. folder: the run folder of an earlier analysis (project None)."""
        run = self._run
        in_run = run is not None and project is not None and run.project is project and run.recs.get(r.id) is r \
            and run.status in ("running", "stopping")
        session = run.dir if in_run else project.session_dir(r.last_session) if project and r.last_session else folder
        files, textgrids = [], {"words": "no", "clauses": "no"}
        if session is not None and r.status != "new":
            words = session / "asr" / "textgrids_clean" / f"{r.id}.TextGrid"
            clauses = session / "clauses" / f"{r.id}.TextGrid"
            for kind, label, path in (("words", "Words TextGrid", words), ("clauses", "Clauses TextGrid", clauses)):
                exists = path.exists()
                files.append({"label": label, "label_kind": kind, "path": str(path), "exists": exists, "kind": "file"})
                textgrids[kind] = "wip" if kind in r.writing else "yes" if exists else "no"
            if not in_run:
                files.append({"label": f"Session {session.name}", "label_kind": "session", "name": session.name,
                              "path": str(session), "exists": session.is_dir(), "kind": "dir"})
        analysed = None
        if r.status in ("done", "failed") and session is not None:
            analysed = self._session_time(project, session.name)
        return {"id": r.id, "name": r.name, "duration_s": r.duration_s, "status": r.status,
                "progress": round(r.progress, 1) if r.status != "done" else 100.0, "textgrids": textgrids,
                "measures": r.measures if r.status == "done" else None, "files": files, "error": r.error,
                "error_code": r.error_code, "error_params": r.error_params, "failed_stage": r.failed_stage,
                "failed_stage_key": STEP_OF.get(r.failed_stage), "added": r.added, "analysed": analysed}

    # ---------- settings ----------

    def set_appearance(self, code: str) -> dict:
        """Settings › Appearance (APPEARANCES), remembered (settings.json's appearance). The page
        switches at once (theme.js); the window's frame takes it too (native_appearance)."""
        if code not in APPEARANCES:
            return {"ok": False}
        with self._lock:
            self._appearance = self._settings["appearance"] = code
            self._save_settings(tell=True)
        native_appearance(code)
        self._log.write(f"Appearance: {code}", show=False)
        return {"ok": True, "appearance": code}

    def set_ui_language(self, code: str) -> dict:
        """The window's language, chosen in Settings and remembered (settings.json's ui_language).
        The page switches at once; the native parts (NATIVE) keep theirs until the next start. If
        settings.json cannot be written, the language holds until Clausa quits, and a notice says so."""
        if code not in UI_LANGUAGES:
            return {"ok": False}
        with self._lock:
            self._ui_language = self._ui_setting = self._settings["ui_language"] = code
            self._save_settings(tell=True)
        self._log.write(f"Display language: {code}", show=False)
        return {"ok": True, "ui_language": code}

    def open_url(self, url: str) -> bool:
        """Open one of the About's links (LINKS) in the browser or the mail app; nothing else."""
        if url not in LINKS:
            self._log.write(f"Not opened: {url} (not one of the app's links)", "error", show=False)
            return False
        import webbrowser
        try:
            return bool(webbrowser.open(url))
        except Exception as exc:  # webbrowser.Error, OSError
            self._log.write(f"Could not open {url}: {exc}", "error")
            return False

    # ---------- projects ----------

    def select_project(self, project_id: str) -> bool:
        """Show a project, or an earlier analysis (its id starts with earlier:)."""
        with self._lock:
            if not any(x.id == project_id for x in [*self._projects, *self._earlier]):
                return False
            self._current = str(project_id)
            if not self._current.startswith("earlier:"):
                self._settings["current"] = self._current
                self._save_settings()
            return True

    def create_project(self, name: str, language: str) -> dict:
        """A new, empty project in the projects folder; it becomes the one shown."""
        with self._lock:
            try:
                project = Project.create(self._root, name, language)
            except (ProjectError, OSError) as exc:
                return not_done(as_message(exc))
            self._projects.insert(0, project)
            self._current = project.id
            self._settings["current"] = project.id
            self._save_settings()
            self._folders_seen = ()
        self._log.write(f"New project: {project.name} ({LANGS[project.language]}) in {project.folder}")
        return {"ok": True, "id": project.id}

    def rename_project(self, project_id: str, name: str) -> dict:
        """Rename a project and its folder; not while it is being analysed."""
        with self._lock:
            p = self._project(project_id)
            if p is None:
                return not_done(msg("project.gone", "That project is not in the list any more."))
            if self._busy() and self._run.project is p:
                return not_done(msg("project.rename_busy",
                                    "The project is being analysed: rename it when the session has ended."))
            old = p.name
            try:
                p.rename(name)
            except (ProjectError, OSError) as exc:
                return not_done(as_message(exc))
            if self._current == old or self._current == project_id:
                self._current = p.id
                self._settings["current"] = p.id
                self._save_settings()
            self._folders_seen = ()
        self._log.write(f"Project renamed: {old} → {p.name}")
        return {"ok": True, "id": p.id}

    def delete_project(self, project_id: str) -> dict:
        """Move a project's folder to the Trash (the page asked first); not while it is being analysed."""
        with self._lock:
            p = self._project(project_id)
            if p is None:
                return not_done(msg("project.gone", "That project is not in the list any more."))
            if self._busy() and self._run.project is p:
                return not_done(msg("project.delete_busy", "The project is being analysed: stop the session first."))
            try:
                moved = trash(p.folder)
            except OSError as exc:
                self._log.write(f"Could not move {p.folder} to the {TRASH}: {exc}", "error")
                return not_done(msg("project.trash_failed",
                                    f"The project could not be moved to the {TRASH} ({exc}). Its folder is unchanged.",
                                    bin=msg(f"bin.{TRASH_KEY}", TRASH), error=str(exc)))
            self._projects.remove(p)
            if self._current == p.id:
                self._current = self._projects[0].id if self._projects else None
                self._settings["current"] = self._current
                self._save_settings()
            self._folders_seen = ()
        self._log.write(f"Project moved to the {TRASH}: {p.name}" + (f" ({moved})" if moved else ""))
        return {"ok": True}

    def choose_projects_dir(self) -> str | None:
        """Choose the folder that holds the projects; remembered for the next start."""
        import webview
        if self._window is None or self._busy():
            return None
        start = self._root if self._root.is_dir() else Path.home()
        chosen = self._window.create_file_dialog(webview.FileDialog.FOLDER, directory=str(start))
        if not chosen:
            return None
        with self._lock:
            if self._busy():
                return None
            self._root = Path(chosen[0] if isinstance(chosen, (list, tuple)) else chosen)
            self._settings["projects_dir"] = str(self._root)
            self._current = None
            self._settings.pop("current", None)  # a project of the old folder
            self._reload()
            self._save_settings(tell=True)
        self._log.write(f"Projects folder: {self._root}")
        return str(self._root)

    # ---------- recordings ----------

    def add_files(self) -> dict:
        """Choose recordings in the system's file dialog."""
        import webview
        if self._window is None:
            return {"added": 0, "refused": []}
        exts = ";".join(f"*{e}" for e in sorted(AUDIO_EXTS))
        lang = self._native_language
        chosen = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                                 file_types=(native_filter(lang, "app.audio_files", exts),
                                                             native_filter(lang, "app.all_files", "*.*")))
        return self.add_paths(list(chosen)) if chosen else {"added": 0, "refused": []}

    def add_paths(self, items: list) -> dict:
        """Add audio files, or the audio files of a folder, to the project shown.

        Each is copied into the project's audio/ (a clone on APFS) and listed as New.
        A file whose name (without extension) is already in the project is refused:
        the pipeline names its results by it. Returns {"added": n, "refused": [messages]}.
        """
        found, refused = [], []  # refused: messages (msg)
        exts = ", ".join(sorted(AUDIO_EXTS))
        for item in items or []:
            path = Path(str(item)).expanduser()
            if path.is_dir():
                found += sorted(p for p in path.iterdir()
                                if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
            elif path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                found.append(path)
            elif path.is_file():
                refused.append(msg("refused.not_audio", f"{path.name}: not an audio file ({exts})",
                                   file=path.name, exts=exts))
            else:
                refused.append(msg("refused.not_found", f"{path}: not found", path=str(path)))
        found = list(dict.fromkeys(Path(os.path.abspath(p)) for p in found))
        added = []
        with self._adding:
            with self._lock:
                kind, project = self._view()
                if kind != "project":
                    refused = [msg("refused.no_project", "Create or choose a project first, then add recordings to it.")
                               ] if found or refused else []
                    plan = []
                else:
                    plan, taken = project.plan_add(found)
                    refused += [as_message(exc) for exc in taken]
            for src, rec_id, audio in plan:  # outside the lock: a copy can take a while
                try:
                    dst = project.copy_in(src, audio)
                except (ProjectError, OSError) as exc:
                    refused.append(msg("refused.copy_failed",
                                       f"{src.name}: could not be copied into the project ({exc})",
                                       file=src.name, error=as_message(exc)))
                    continue
                added.append(Recording(rec_id, src.name, str(src), now_iso(), duration_s=audio_duration(dst), audio=audio))
            if added:
                with self._lock:
                    # The list may have been read again during the copies (get_state does when
                    # project folders change): add to the project as it is listed now.
                    project = self._project(project.id) or project
                    project.recordings.extend(added)
                    self._save_project(project)
        if added:
            names = ", ".join(r.name for r in added[:5]) + (f" and {len(added) - 5} more" if len(added) > 5 else "")
            self._log.write(f"Added to {project.name}: {names}")
        for m in refused:
            self._log.write(f"Not added: {m['text']}")
        if len(refused) == 1:
            self._note(refused[0])
        elif refused:
            self._note(msg("notice.not_added_many",
                           f"{len(refused)} files were not added: {refused[0]['text']} (and more in the log)",
                           n=len(refused), first=refused[0]))
        return {"added": len(added), "refused": [m["text"] for m in refused]}

    def remove(self, rec_id: str) -> bool:
        """Remove a recording from the project shown: its row, its CSV row and its copy (to the Trash).

        Not while it is part of a session that runs. Its earlier sessions' files stay.
        """
        with self._lock:
            kind, p = self._view()
            r = p.find(str(rec_id)) if kind == "project" else None
            if r is None or (self._busy() and self._run.project is p and r.id in self._run.recs):
                return False
            p.recordings.remove(r)
            self._save_project(p)
            self._write_csv(p)
            copy = p.audio_path(r)
        self._log.write(f"Removed from {p.name}: {r.name}")
        if r.audio and copy.is_file():  # never the audio folder itself
            try:
                trash(copy)
            except OSError as exc:
                self._log.write(f"Its copy stays in the project's audio folder ({exc}): {copy}")
        return True

    # ---------- sessions ----------

    def start(self) -> bool | dict:
        """Analyse every recording of the project shown that is not done (New and Failed ones);
        as _start_session answers."""
        with self._lock:
            kind, p = self._view()
            if kind != "project":
                return False
            recs = [r for r in p.recordings if r.status in ("new", "failed")]
        return self._start_session(p, recs)

    def analyse_again(self, rec_ids: list | None = None) -> bool | dict:
        """Analyse these recordings again (all of the project shown if none are given).

        A new result replaces the row when the session ends; the earlier sessions'
        files stay where they are.
        """
        with self._lock:
            kind, p = self._view()
            if kind != "project":
                return False
            wanted = None if not rec_ids else {str(i) for i in rec_ids}
            recs = [r for r in p.recordings if wanted is None or r.id in wanted]
        return self._start_session(p, recs)

    def _start_session(self, p: Project, recs: list[Recording]) -> bool | dict:
        """Analyse recordings of a project in a new session folder: True once it runs, False if there
        is nothing to start, {"ok": False, ...} (not_done) if its folder could not be made.

        The lock is held only to pick the recordings and register the session, so a
        second Start finds it and returns False (one session at a time, whichever
        project). The links and the log header, which waits until the device is
        known (up to 300 s just after the app starts), are made outside it, so
        get_state keeps answering. A Stop in between ends the session as soon as its
        thread starts.
        """
        with self._lock:
            if self._busy() or not recs:
                return False
            # A recording whose copy is missing, or cannot be read as audio, fails here: the runner
            # works stage by stage, and such a file would stop the session for all the others.
            bad = {}  # id -> (failed_stage, code, error, params)
            for r in recs:
                audio = p.audio_path(r)
                if not audio.is_file():
                    copy = f"{AUDIO_DIR}/{r.audio}"
                    bad[r.id] = (None, "copy_missing", f"The project's copy of this recording is missing ({copy}). "
                                                       "Remove it and add the file again.", {"file": copy})
                elif audio_duration(audio) is None:
                    bad[r.id] = (STEPS["decode"][1], "unreadable",
                                 "This file could not be read as audio (it is empty or damaged), "
                                 "so it was not analysed. Remove it and add a working copy.", {})
            for r in recs:
                if r.id in bad:
                    r.failed_stage, code, text, params = bad[r.id]
                    set_error(r, code, text, **params)
                    # Not analysed now: no session's files are its results any more.
                    r.status, r.measures, r.last_session = "failed", None, None
                    self._log.write(f"{r.name}: {r.error}", "error")
            if bad:
                names = [r.name for r in recs if r.id in bad]
                more = len(names) - 1
                unreadable = all(bad[r.id][0] for r in recs if r.id in bad)  # all there, none readable
                what = "could not be read as audio" if unreadable else "missing or unreadable"
                if more:
                    self._note(msg("notice.unreadable_many" if unreadable else "notice.missing_many",
                                   f"{names[0]} and {more} more: {what}, so they were not analysed",
                                   name=names[0], n=more))
                else:
                    self._note(msg("notice.unreadable_one" if unreadable else "notice.missing_one",
                                   f"{names[0]}: {what}, so it was not analysed", name=names[0]))
                self._write_csv(p)
            recs = [r for r in recs if r.id not in bad]
            if not recs:
                self._save_project(p)
                return False
            try:
                session_dir = p.new_session_dir()
            except OSError as exc:
                self._log.write(f"Could not make a session folder in {p.folder}: {exc}", "error")
                return not_done(msg("session.folder_failed", f"The project's folder could not be used: {exc}",
                                    error=str(exc)))
            run = Run(p, session_dir, p.language, {r.id: r for r in sorted(recs, key=lambda r: r.id)})
            # What the runner will announce in its first event (which replaces this), so that the
            # time left is known from the start: it takes a moment to set up.
            run.order = list(run.recs)
            run.stages = (["decode"] if any(r.audio.lower().endswith(tuple(DECODE_EXTS)) for r in recs) else []) + [
                key for key, _ in STAGES if key != "gap" or paths.FILLER_MODEL.exists()]
            run.done = {key: set() for key in run.stages}
            for r in recs:
                r.before = r.saved_state()
                r.status, r.progress, r.measures, r.failed_stage = "running", 0.0, None, None
                set_error(r)
                r.writing.clear()
            run.session = Session(session_dir.name, now_iso(), list(run.recs))
            p.sessions.append(run.session)
            self._save_project(p)
            previous, self._run = self._run, run
        try:
            p.link_recordings(session_dir, list(run.recs.values()))
        except OSError as exc:
            for folder in (session_dir / "recordings", session_dir):  # only if nothing was written there
                try:
                    folder.rmdir()
                except OSError:
                    pass
            with self._lock:  # back to how it was: the recordings as before, the last run shows again
                for r in recs:
                    self._restore(r)
                if run.session in p.sessions:
                    p.sessions.remove(run.session)
                self._save_project(p)
                self._run = previous
                # The last run has ended (or this Start would have been refused). If its
                # thread found this run registered, it left its log and the flags: end them.
                self._log.end_run()
                self._log.stopping = False
                self._set_confirm_close(False)
            self._log.write(f"Could not prepare the session folder {session_dir}: {exc}", "error")
            return not_done(msg("session.prepare_failed", f"The session folder could not be prepared: {exc}",
                                error=str(exc)))
        self._log.start_run(session_dir / "log.txt", self._header(run))
        self._log.write(f"Session started: {len(recs)} recording{'s' if len(recs) > 1 else ''} of {p.name}, "
                        f"{LANGS[run.language]}, in {session_dir}")
        with self._lock:
            if run.status == "stopping":  # Stop came while the folder was being prepared
                self._log.stopping = True
            self._set_confirm_close(True)  # before the thread, which turns it off when the run ends
            self._thread = threading.Thread(target=self._work, args=(run,), name="clausa-run", daemon=True)
            self._thread.start()
        return True

    def stop(self) -> bool:
        """Stop the session: the running stage ends at once; unfinished recordings go back to how they were."""
        with self._lock:
            run = self._run
            if run is None or run.status != "running":
                return False
            run.status = "stopping"
            self._log.stopping = True
            run.stop.set()
        self._log.write("Stopping…")
        return True

    def _work(self, run: Run) -> None:
        """The session's thread: run_pipeline, then the outcome for each recording."""
        status, error, out_of_memory = "done", None, False
        try:
            # The computer does not go to sleep while the session runs (this thread holds the request).
            with keep_awake() as awake:
                if awake is False:
                    self._log.write("Could not ask the system to stay awake: it may go to sleep during the session")
                run_pipeline(run.language, run.dir / "recordings", run.dir, device="auto",
                             on_event=lambda e: self._on_event(run, e), stop=run.stop)
        except Stopped:
            status = "stopped"
        except Exception as exc:  # a stage failed (CalledProcessError) or the run could not start
            status, error = "failed", str(exc) or type(exc).__name__
            out_of_memory = isinstance(exc, GpuOutOfMemory)
            self._log.write(traceback.format_exc().rstrip(), "error")
        try:
            with self._lock:
                self._finish(run, status, error, out_of_memory)
        except Exception:
            self._log.write(traceback.format_exc().rstrip(), "error")
            with self._lock:
                run.status, run.ended = "failed", run.ended or time.monotonic()
        finally:
            # Only if no newer session has started: a Start can come as soon as _finish has
            # released the lock, and then the log, the stopping flag and the quit
            # confirmation are that session's (its start_run closes this session's log.txt).
            # confirm_close is a plain attribute that pywebview reads on closing, so it
            # can be set under the lock.
            with self._lock:
                if self._run is run:
                    self._log.end_run()
                    self._log.stopping = False
                    self._set_confirm_close(False)

    def _on_event(self, run: Run, e: dict) -> None:
        """An event of runner.run.Events, in the session's thread."""
        kind, status = e["event"], e["status"]
        with self._lock:
            if kind == "run" and status == "started":
                run.order = list(e["files"])
                run.stages = [s["key"] for s in e["stages"]]
                run.done = {key: set() for key in run.stages}
            elif kind == "stage":
                key = e["stage"]
                label = STEPS.get(key, (0, e.get("label", key)))[1]
                if status == "started":
                    run.stage, run.started_file = key, None
                    run.stage_clock = run.file_clock = time.monotonic()
                    self._log.write(f"{label} (step {STEPS[key][0]} of {STEP_COUNT})")
                elif status == "completed":
                    run.finished.add(key)
                    self._stage_ended(run, key, e.get("seconds") or time.monotonic() - run.stage_clock)
                    for r in run.recs.values():
                        r.writing.discard(WRITES.get(key))
                    self._log.write(f"{label} finished in {fmt_seconds(e.get('seconds', 0))}", "ok")
                elif status == "skipped":
                    run.finished.add(key)
                    self._log.write(f"{label}: skipped (no Stage 4 weights)")
            elif kind == "file":
                key, r = e["stage"], run.recs.get(e["file"])
                if r is None:
                    return
                if status == "started":
                    run.started_file = r.id
                    if key in WRITES:
                        r.writing.add(WRITES[key])
                else:
                    if run.started_file == r.id:
                        run.started_file = None
                    r.writing.discard(WRITES.get(key))
                    run.done.setdefault(key, set()).add(r.id)
                    run.file_clock = time.monotonic()
                    if status == "failed":
                        self._file_failed(run, r, key, e.get("error") or "")
            self._advance(run)

    def _file_failed(self, run: Run, r: Recording, stage: str, detail: str) -> None:
        if stage == "gap" and run.language not in MODE.classifier_islands:
            # English measures do not use Stage 4 (runner.run.MODES).
            self._log.write(f"{r.name}: Filled pauses failed ({detail}); "
                            "the English measures do not use that step")
            return
        if r.status != "running":
            return
        r.status, r.failed_stage = "failed", STEPS[stage][1] if stage in STEPS else None
        r.writing.clear()
        if stage == "asr_stage1" and detail == "No words detected":
            set_error(r, "no_speech", "No speech was found in this recording, so it was skipped. "
                                      "Check that the file holds the speaker's speech.")
        elif stage == "asr_stage1" and GPU_MEMORY_ERROR.search(detail):
            set_error(r, "gpu_memory", GPU_MEMORY)
        elif stage == "mfa_stage2" and detail:
            # Also the runner's alignment failures (manifest.json alignment_failed): the file
            # may still get a row in the CSV, on word timings that are not comparable.
            set_error(r, "alignment_failed", f"Word alignment failed for this recording: {detail.rstrip('.')}. "
                                             "Its measures are not shown; the log has the details.",
                      detail=detail.rstrip("."))
        elif stage == "gap":
            set_error(r, "filler_failed",
                      "Filled pauses failed for this recording, so its pauses were measured "
                      "without that step and are not comparable with the others: it is left out of the project's "
                      "CSV (the session folder's own CSV still has that row). The log has the details.")
        else:
            name = STEPS[stage][1] if stage in STEPS else "A stage"
            set_error(r, "stage_failed",
                      f"{name} failed for this recording, so it has no measures. The log has the details.",
                      stage=msg(f"stage.{stage}" if stage in STEPS else "stage.other", name))
        self._log.write(f"{r.name}: {STEPS[stage][1] if stage in STEPS else stage} failed: "
                        f"{detail or 'see the output above'}", "error")

    @staticmethod
    def _shares(run: Run) -> dict:
        """Each running stage's share of a recording's bar (0-1): a fifth per step."""
        shares = {}
        for key in run.stages:
            step, _, share = STEPS[key]
            same = sum(STEPS[k][2] for k in run.stages if STEPS[k][0] == step)
            shares[key] = share / same / STEP_COUNT if same else 0
        return shares

    def _advance(self, run: Run) -> None:
        """After an event: the bars at their true value (never back), then the estimates."""
        shares = self._shares(run)
        for r in run.recs.values():
            if r.status == "running":
                done = sum(s for key, s in shares.items() if key in run.finished or r.id in run.done.get(key, ()))
                r.progress = max(r.progress, min(100.0, 100 * done))
        self._estimate(run)

    def _estimate(self, run: Run) -> None:
        """Move the bars by the time the running stage is expected to take, and renew run.eta.

        In a stage with per-file lines, the file being processed advances by the time
        since it became current against its expected time (STAGE_TIMES, scaled by its
        length and Run.speed); in a stage without them, every recording advances by the
        stage's time. An estimate fills at most ESTIMATE_CAP of the stage's share; the
        next event then sets the true value. Bars never go back. run.eta is the expected
        rest of the running stage plus the stages still to come.
        """
        if not run.stages:  # before the runner's first event
            run.eta = None
            return
        now = time.monotonic()
        times = STAGE_TIMES[run.language]
        shares = self._shares(run)
        active = [run.recs[f] for f in run.order if f in run.recs and run.recs[f].status == "running"]

        def length(r: Recording) -> float:
            return r.duration_s or UNKNOWN_DURATION

        key = run.stage if run.stage is not None and run.stage not in run.finished else None
        partial, left = {}, 0.0
        if key is not None:
            t = times[key]
            pending = [r for r in active if r.id not in run.done.get(key, set())]
            if key in PER_FILE and pending:
                current = run.recs.get(self._current_file(run) or "") or pending[0]
                first = not run.done.get(key)
                expected = run.speed * ((t.load + t.first if first else 0) + t.fixed + t.rate * length(current))
                spent = now - run.file_clock
                partial[current.id] = min(ESTIMATE_CAP, spent / expected) if expected > 0 else ESTIMATE_CAP
                rest = sum(t.fixed + t.rate * length(r) for r in pending if r is not current) + t.tail
                left = max(0.0, expected - spent) + run.speed * rest
            elif key in PER_FILE:  # every file reported, the stage is closing
                left = max(0.0, run.speed * t.tail - (now - run.file_clock))
            else:
                expected = run.speed * t.stage([length(r) for r in active])
                spent = now - run.stage_clock
                for r in active:
                    partial[r.id] = min(ESTIMATE_CAP, spent / expected) if expected > 0 else ESTIMATE_CAP
                left = max(0.0, expected - spent)
        later = [k for k in run.stages if k not in run.finished and k != key]
        run.eta = left + run.speed * sum(times[k].stage([length(r) for r in active]) for k in later)
        for r in active:
            done = sum(s for k, s in shares.items() if k in run.finished or r.id in run.done.get(k, ()))
            estimate = 100 * (done + partial.get(r.id, 0.0) * shares.get(key, 0.0))
            r.progress = max(r.progress, min(100.0, estimate))

    def _stage_ended(self, run: Run, key: str, seconds: float) -> None:
        """Rescale the expected times by how long this run's finished stages took (Run.speed)."""
        recs = [r for r in run.recs.values() if r.status == "running" or r.id in run.done.get(key, ())]
        run.expected_s += STAGE_TIMES[run.language][key].stage([r.duration_s or UNKNOWN_DURATION for r in recs])
        run.actual_s += seconds
        if run.expected_s > 0:
            run.speed = min(SPEED_RANGE[1], max(SPEED_RANGE[0], run.actual_s / run.expected_s))

    @staticmethod
    def _current_file(run: Run) -> str | None:
        """The recording the running stage works on: the one it started, or the next in order."""
        key = run.stage
        if key is None or key in run.finished or key not in PER_FILE:
            return None
        if run.started_file:
            return run.started_file
        done = run.done.get(key, set())
        return next((f for f in run.order if f not in done and f in run.recs
                     and run.recs[f].status == "running"), None)

    def _finish(self, run: Run, status: str, error: str | None, out_of_memory: bool = False) -> None:
        """The session has ended: each recording's outcome, project.json, the project's CSV and the log.
        out_of_memory: it failed for lack of GPU memory (GpuOutOfMemory)."""
        run.ended = time.monotonic()
        elapsed = fmt_seconds(run.ended - run.started)
        p = run.project
        for r in run.recs.values():
            r.writing.clear()
        uf_csv = run.dir / "uf" / CSV_NAME
        back = []
        if status == "stopped":
            back = [r for r in run.recs.values() if r.status == "running"]
            for r in back:
                self._restore(r)
            self._log.write(f"Stopped by the user after {elapsed}; {len(back)} recording"
                            f"{'s' if len(back) != 1 else ''} back to how {'they were' if len(back) != 1 else 'it was'}")
            self._log.write(f"Result: stopped · {elapsed}")
        elif status == "done":
            rows = read_measures(uf_csv)
            manifest = self._read_manifest(run.dir / "manifest.json")
            stage4_failed = (manifest.get("stage4_failed", {}) if run.language in MODE.classifier_islands
                             else {})
            for r in run.recs.values():
                if r.status != "running":
                    continue
                if r.id in manifest.get("alignment_failed", {}):
                    self._file_failed(run, r, "mfa_stage2", manifest["alignment_failed"][r.id])
                elif r.id in stage4_failed:
                    self._file_failed(run, r, "gap", stage4_failed[r.id] or "listed in manifest.json")
                elif r.id in rows:
                    r.status, r.progress, r.measures = "done", 100.0, rows[r.id]
                    r.failed_stage = None
                    set_error(r)
                else:
                    r.status, r.failed_stage = "failed", STEPS["caf"][1]
                    set_error(r, "no_measures",
                              "No measures were produced for this recording. The log has the details.")
                    self._log.write(f"{r.name}: no row in uf_measures.csv", "error")
            done = sum(r.status == "done" for r in run.recs.values())
            failed = len(run.recs) - done
            self._log.write(f"Session finished: {done} of {len(run.recs)} analysed"
                            + (f", {failed} failed" if failed else "") + f" in {elapsed}",
                            "ok" if not failed else "error")
        else:
            label = STEPS[run.stage][1] if run.stage else None
            during = label.lower() if label else "the preparation"
            for r in run.recs.values():
                if r.status == "running":
                    r.status, r.failed_stage = "failed", label
                    if out_of_memory:
                        set_error(r, "gpu_memory", GPU_MEMORY)
                    elif label:
                        set_error(r, "session_error", f"{label} stopped the session with an error, so this "
                                  "recording has no measures. The log has the details.",
                                  stage=msg(f"stage.{run.stage}", label))
                    else:
                        set_error(r, "session_error_prep", "The session stopped with an error during the preparation, "
                                  "so this recording has no measures. The log has the details.")
            self._log.write(f"The session failed during {during}: {error}", "error")
            self._log.write(f"Result: failed during {during} · {elapsed}")
        # Every recording the session finished (done or failed, also before a Stop) now has this
        # session as its last; those a Stop put back keep theirs.
        back_ids = {r.id for r in back}
        for r in run.recs.values():
            if r.id not in back_ids and r.status in ("done", "failed"):
                r.last_session = run.dir.name
                if r.status == "failed":
                    r.measures = None
            r.before = None
        if run.session is not None:
            run.session.ended, run.session.outcome = now_iso(), status
            run.session.done = sum(r.status == "done" and r.last_session == run.dir.name for r in run.recs.values())
            run.session.failed = sum(r.status == "failed" and r.last_session == run.dir.name for r in run.recs.values())
        self._save_project(p)
        rows = self._write_csv(p)
        if status == "done":
            self._log.write(f"Result: {'done' if not run.session.failed else 'done with failures'} · "
                            f"{run.session.done} of {len(run.recs)} analysed · {elapsed} · "
                            f"project measures: {p.csv_path if rows else 'none'} ({rows} row{'s' if rows != 1 else ''})")
        run.status = status

    @staticmethod
    def _read_manifest(path: Path) -> dict:
        """The per-file failures in manifest.json: {key: {file_id: error}} for
        stage4_failed and alignment_failed (lists of {"file_id", "error"})."""
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        found = {}
        for key in ("stage4_failed", "alignment_failed"):
            items = manifest.get(key) if isinstance(manifest, dict) else None
            found[key] = {str(f["file_id"]): str(f.get("error") or "") for f in items or []
                          if isinstance(f, dict) and "file_id" in f}
        return found

    # ---------- opening things ----------

    def open_project_folder(self) -> bool:
        """Open the folder of the project shown (or of the earlier analysis shown)."""
        with self._lock:
            kind, view = self._view()
            folder = view.folder if kind == "project" else view.dir if kind == "earlier" else self._root
            if not folder.is_dir():  # moved or in the Trash: it is not made again
                self._note(msg("notice.folder_not_found", f"The folder was not found: {home_short(folder)}",
                               path=home_short(folder)))
                return False
        open_path(folder)
        return True

    def open_csv(self) -> bool:
        """Open the measures of the project shown (or of the earlier analysis) in the default app."""
        with self._lock:
            kind, view = self._view()
            path = view.csv_path if kind == "project" else view.dir / "uf" / CSV_NAME if kind == "earlier" else None
        if path is None or not path.exists():
            return False
        open_path(path)
        return True

    def open_folder(self, rec_id: str) -> bool:
        """Open the session folder that holds a recording's latest results."""
        with self._lock:
            kind, view = self._view()
            folder = None
            if kind == "project":
                r = view.find(str(rec_id))
                run = self._run
                if r is not None and run is not None and run.project is view and run.recs.get(r.id) is r \
                        and run.status in ("running", "stopping"):
                    folder = run.dir
                elif r is not None and r.last_session:
                    folder = view.session_dir(r.last_session)
            elif kind == "earlier":
                folder = view.dir
            if folder is not None and not folder.is_dir():
                self._note(msg("notice.folder_not_found", f"The folder was not found: {home_short(folder)}",
                               path=home_short(folder)))
        if folder is None or not folder.is_dir():
            return False
        open_path(folder)
        return True

    def reveal(self, path: str) -> bool:
        """Show a result file in its folder."""
        p = Path(str(path))
        if not p.exists():
            return False
        open_path(p, reveal=True)
        return True

    # ---------- the log ----------

    def copy_log(self) -> str:
        return self._log.text()

    def save_log_as(self) -> bool:
        """Save the Log panel's lines wherever the user chooses."""
        import webview
        if self._window is None:
            return False
        with self._lock:
            kind, view = self._view()
            folder = view.folder if kind == "project" else view.dir if kind == "earlier" else self._root
        chosen = self._window.create_file_dialog(
            webview.FileDialog.SAVE, directory=str(folder if folder.is_dir() else Path.home()),
            save_filename=native(self._native_language, "app.log_name",
                                 date=f"{datetime.now():%Y-%m-%d %H.%M}") + ".txt")
        if not chosen:
            return False
        target = Path(chosen[0] if isinstance(chosen, (list, tuple)) else chosen)
        target.write_text(self._log.text(), encoding="utf-8")
        self._log.write(f"Log saved as {target}")
        return True

    def log_js_error(self, message: str, source: str = "", line: int = 0) -> None:
        """An error in the page (reported by the handler that main() adds to it)."""
        self._log.write(f"Window error: {message}" + (f" ({source}:{line})" if source else ""), "error")

    # ---------- helpers ----------

    def _busy(self) -> bool:
        return self._run is not None and self._run.status in ("running", "stopping")

    def _project(self, project_id: str) -> Project | None:
        return next((p for p in self._projects if p.id == str(project_id)), None)

    def _view(self) -> tuple[str | None, object]:
        """What the window shows: ("project", Project), ("earlier", Earlier) or (None, None)."""
        cur = self._current
        if cur and cur.startswith("earlier:"):
            e = next((e for e in self._earlier if e.id == cur), None)
            if e is not None:
                return "earlier", e
        p = self._project(cur) if cur else None
        if p is None and self._projects:
            p = self._projects[0]
            self._current = p.id
        return ("project", p) if p is not None else (None, None)

    def _earlier_rows(self, e: Earlier) -> list[Recording]:
        """An earlier analysis's recordings, read-only, from its folder (kept once read)."""
        if e.id not in self._earlier_recs:
            rows = read_measures(e.dir / "uf" / CSV_NAME)
            links = {}
            try:
                links = {p.stem: p for p in (e.dir / "recordings").iterdir()}
            except OSError:
                pass
            recs = []
            for fid in e.files or list(rows):
                audio = links.get(fid)
                r = Recording(fid, audio.name if audio else fid, str(audio) if audio else "", "",
                              status="done" if fid in rows else "failed",
                              duration_s=audio_duration(audio) if audio else None, audio=audio.name if audio else "")
                r.progress = 100.0 if fid in rows else 0.0
                r.measures = rows.get(fid)
                if fid not in rows:
                    set_error(r, "earlier_no_measures",
                              "This analysis has no measures for this recording. Its log.txt has the details.")
                recs.append(r)
            self._earlier_recs[e.id] = recs
        return self._earlier_recs[e.id]

    def _reload(self) -> None:
        """Read the projects folder and the earlier analyses again.

        A project already in the list is kept as it is (the same object): a session, an
        add or a remove may be changing it, and what it holds is what it saved.
        """
        found, problems = list_projects(self._root)
        held = {p.folder: p for p in self._projects}
        found = [held.get(p.folder, p) for p in found]
        for p in found:
            if p.folder not in held:
                for rec_id, values in p.measures().items():
                    p.find(rec_id).measures = values
        self._projects = found
        for message in problems:
            self._log.write(f"Not a readable project: {message}", "error")
        self._reload_earlier()
        self._folders_seen = self._folders()
        self._scanned = time.monotonic()

    def _reload_earlier(self) -> None:
        self._earlier = earlier_analyses(self._earlier_folders())
        self._earlier_recs = {}
        self._earlier_seen = self._dated_folders()

    def _dated_folders(self) -> tuple:
        """The dated folders in the old results folders (to notice when they change)."""
        names = []
        for folder in self._earlier_folders():
            try:
                names += [str(d) for d in folder.iterdir() if DATED_FOLDER.match(d.name)]
            except OSError:
                pass
        return tuple(sorted(names))

    def _folders(self) -> tuple:
        try:
            return tuple(sorted(p.name for p in self._root.iterdir() if (p / PROJECT_FILE).is_file()))
        except OSError:
            return ()

    def _rescan(self) -> None:
        """Every few seconds: if project folders or earlier analyses came or went (Finder,
        Put Back), read them again."""
        if time.monotonic() - self._scanned < 5:
            return
        self._scanned = time.monotonic()
        if self._folders() != self._folders_seen:
            self._reload()
        elif self._dated_folders() != self._earlier_seen:
            self._reload_earlier()

    def _earlier_folders(self) -> list[Path]:
        """Where the app kept its runs before projects: the folder chosen then (settings.json's
        results_dir) and the default one."""
        folders = []
        saved = self._settings.get("results_dir")
        if isinstance(saved, str) and os.path.isabs(saved):
            folders.append(Path(saved))
        folders.append(paths.RESULTS_DIR)
        return [f for f in dict.fromkeys(folders) if f.resolve() != self._root.resolve()]

    def _note(self, message: dict) -> None:
        """A message (msg) for the status bar: the page shows it for a few seconds."""
        self._notice = {"id": self._notice["id"] + 1, **message}

    @staticmethod
    def _session_pct(run: Run) -> float:
        if not run.recs:
            return 0.0
        return sum(100.0 if r.status in ("done", "failed") else r.progress for r in run.recs.values()) / len(run.recs)

    @staticmethod
    def _restore(r: Recording) -> None:
        """A recording a session did not finish goes back to how it was before the session."""
        before = r.before or {}
        r.status = before.get("status", "new")
        r.last_session, r.error = before.get("last_session"), before.get("error")
        r.error_code, r.error_params = before.get("error_code"), before.get("error_params")
        r.failed_stage, r.measures = before.get("failed_stage"), before.get("measures")
        r.progress = 100.0 if r.status == "done" else 0.0
        r.writing.clear()
        r.before = None

    def _save_project(self, p: Project) -> None:
        try:
            p.save()
        except OSError as exc:
            self._log.write(f"Could not save {p.folder / PROJECT_FILE}: {exc}", "error")
            self._note(msg("notice.save_failed", f"The project could not be saved: {exc}", error=str(exc)))

    def _write_csv(self, p: Project) -> int:
        """Regenerate the project's CSV; return its number of rows.

        A done recording whose session's CSV has other columns than the newest (made by
        another version of Clausa) is left out and marked failed, so that Start analyses it
        again: one CSV never mixes headers.
        """
        try:
            rows, other = p.write_csv()
        except OSError as exc:
            self._log.write(f"Could not write {p.csv_path}: {exc}", "error")
            return 0
        if other:
            for rec_id in other:
                r = p.find(rec_id)
                r.status, r.measures, r.failed_stage = "failed", None, None
                set_error(r, "other_version", "Its measures were made by another version of Clausa (their CSV has "
                                              "other columns), so they are not in the project's CSV. Analyse it again.")
            self._log.write(f"{p.name}: {len(other)} recording{'s' if len(other) != 1 else ''} left out of "
                            f"{p.csv_path.name}, made by another version of Clausa (other columns): "
                            f"{', '.join(other)}. Analyse {'them' if len(other) != 1 else 'it'} again.", "error")
            self._save_project(p)
        return rows

    @staticmethod
    def _folder_time(name: str) -> str | None:
        """'2026-10-23 14.05 (2)' -> '2026-10-23T14:05'."""
        m = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2})\.(\d{2})", name)
        return f"{m[1]}T{m[2]}:{m[3]}" if m else None

    def _session_time(self, p: Project | None, name: str) -> str | None:
        """When a session started (ISO), from project.json, else from its folder's name."""
        if p is not None:
            s = next((s for s in p.sessions if s.name == name), None)
            if s is not None and s.started:
                return s.started
        return self._folder_time(name)

    def _last_session(self, p: Project) -> str | None:
        ended = [s for s in p.sessions if s.outcome in ("done", "failed")]
        return ended[-1].started if ended else None

    def _last_log(self, kind: str | None, view) -> str | None:
        if kind == "project" and view.sessions:
            path = view.session_dir(view.sessions[-1].name) / "log.txt"
            return str(path) if path.exists() else None
        if kind == "earlier" and (view.dir / "log.txt").exists():
            return str(view.dir / "log.txt")
        return None

    def _header(self, run: Run) -> list[str]:
        """The top of log.txt: versions, system, settings, the project and the recordings."""
        self._device_ready.wait(timeout=300)
        code = code_version()
        settings = ", ".join(f"{k} {'auto' if k == 'mfa_workers' and v is None else v}"  # the number: "Aligning ..."
                             for k, v in RUN_DEFAULTS.items() if k not in ("window", "windows_csv"))
        lines = [
            f"Clausa {__version__}" + (f" (code {code})" if code else ""),
            f"Started: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}",
            f"System: {system_text()}, Python {platform.python_version()}",
            f"Device: {DEVICES.get(self._device, 'unknown')} (device auto)",
            "Environments: " + (f"bundled in {paths.ENVS_DIR}" if paths.ENVS_DIR else "conda")
            + f" ({paths.ASR_ENV}, {paths.MFA_ENV}, {paths.JA_ENV})",
            f"Packages ({paths.ASR_ENV}): {package_versions()}",
            f"Project: {run.project.name} ({run.project.folder})",
            f"Language: {LANGS[run.language]} ({run.language})",
            f"Settings: {settings}",
            "Stage 4 weights: " + (str(paths.FILLER_MODEL) if paths.FILLER_MODEL.exists()
                                   else "not found (Stage 4 is skipped)"),
            f"Session folder: {run.dir}",
            f"Recordings ({len(run.recs)}):",
        ]
        for i, r in enumerate(run.recs.values(), 1):
            duration = fmt_seconds(r.duration_s) if r.duration_s is not None else "length unknown"
            lines.append(f"  {i}. {r.id}  ({duration})  {run.project.audio_path(r)}  (added from {r.source})")
        return lines

    def _find_device(self) -> None:
        try:
            self._device = detect_device(started=lambda proc: setattr(self, "_device_proc", proc))
        except Exception as exc:
            self._log.write(f"Could not tell the device: {exc}", "error")
        finally:
            self._device_ready.set()
        self._log.write(f"Clausa {__version__} · {system_text()} · {DEVICES.get(self._device, 'device unknown')}")

    def _load_settings(self) -> dict:
        try:
            data = json.loads(paths.APP_SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_settings(self, tell: bool = False) -> bool:
        """settings.json: the projects folder, the project last shown and the window's language if
        one was chosen (ui_language), and the appearance (appearance). Keys of earlier versions stay (results_dir: where the earlier
        analyses are). False if it could not be written; tell: say so in the status bar too (a
        setting the user chose, which then holds only until Clausa quits)."""
        self._settings["projects_dir"] = str(self._root)
        try:
            tmp = paths.APP_SETTINGS.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(paths.APP_SETTINGS)
            return True
        except OSError as exc:
            self._log.write(f"Could not save the settings in {paths.APP_SETTINGS}: {exc}", "error")
            if tell:
                self._note(msg("notice.settings_not_saved", f"The settings could not be saved: {exc}", error=str(exc)))
            return False

    def _set_confirm_close(self, on: bool) -> None:
        """While a session runs, closing the window asks first (pywebview's quit confirmation)."""
        if self._window is not None:
            self._window.confirm_close = on

    def _dropped(self, event: dict) -> None:
        """pywebview's drop handler: it alone gets the dropped files' full paths."""
        files = (event.get("dataTransfer") or {}).get("files") or []
        found = [f.get("pywebviewFullPath") for f in files if isinstance(f, dict)]
        found = [p for p in found if p]
        if len(found) < len(files):
            self._log.write(f"Not added: {len(files) - len(found)} dropped item(s) without a file path")
        if found:
            self.add_paths(found)
        if self._window is not None:
            self._window.run_js("window.dispatchEvent(new Event('clausa:changed'))")

    def _shutdown(self) -> None:
        """The window has closed: stop an active session and wait until its stage processes are gone."""
        with self._lock:
            run, thread = self._run, self._thread
            if run is not None and run.status == "running":
                run.status = "stopping"
                self._log.stopping = True
                run.stop.set()
                self._log.write("The window was closed: stopping the session")
        if thread is not None:
            thread.join(timeout=60)
        proc = self._device_proc  # closed within seconds of starting: the device question still runs
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        self._log.write("The app has closed", show=False)
        self._log.close()


# Reports errors in the page to the log (Api.log_js_error).
_JS_ERRORS = """
window.addEventListener('error', function (e) {
  var api = window.pywebview && window.pywebview.api;
  if (api && api.log_js_error) api.log_js_error(String(e.message), String(e.filename || ''), e.lineno || 0);
});
window.addEventListener('unhandledrejection', function (e) {
  var api = window.pywebview && window.pywebview.api;
  if (api && api.log_js_error) api.log_js_error('Unhandled rejection: ' + String(e.reason && e.reason.message || e.reason), '', 0);
});
"""


def main() -> int:
    api = Api()  # first: it takes over sys.stdout and sys.stderr, which pywebview's logger keeps
    if sys.platform == "darwin":
        mac_app_name("Clausa")
    elif os.name == "nt":
        windows_app_id("org.clausa.app")
    import webview
    from webview.dom import DOMEventHandler

    # ?ui=, ?theme=: the page's language and appearance from its first paint, before the bridge is
    # ready (app.js, theme.js)
    light = api._appearance == "light" or (api._appearance == "system" and system_light())
    if sys.platform == "darwin":
        native_appearance(api._appearance)
    window = webview.create_window(
        "Clausa", url=f"{paths.APP_UI_DIR / 'index.html'}?ui={api._ui_language}&theme={api._appearance}", js_api=api,
        width=1280, height=800, min_size=(760, 560), background_color=LIGHT_BG if light else DARK_BG,
        localization=native_localization(api._native_language))
    api._window = window

    def loaded() -> None:
        try:
            window.run_js(_JS_ERRORS)
            # With a drop handler registered here, pywebview passes the dropped files' full
            # paths (pywebviewFullPath) to it; the page itself only gets their names.
            window.dom.get_element("#dropzone").events.drop += DOMEventHandler(api._dropped, prevent_default=True)
        except Exception:
            api._log.write(traceback.format_exc().rstrip(), "error")

    window.events.loaded += loaded
    if os.name == "nt":  # the title bar: pywebview's form exists once shown
        window.events.shown += lambda: native_appearance(api._appearance)

    def quit_on_signal(signum, frame) -> None:  # Ctrl+C or a kill: close the window, then stop as on closing
        threading.Thread(target=window.destroy, daemon=True).start()

    signal.signal(signal.SIGINT, quit_on_signal)
    signal.signal(signal.SIGTERM, quit_on_signal)
    code = 0
    try:
        # icon: the Dock icon on macOS (pywebview sets it there too), the window icon on GTK and Qt;
        # on Windows the window's and the taskbar's, which must be an .ico
        icon = paths.APP_ICON_ICO if os.name == "nt" else paths.APP_ICON
        # Windows: WebView2 only; pywebview would otherwise fall back to Internet Explorer's engine,
        # which cannot show the page (the launcher checks for WebView2 first)
        webview.start(icon=str(icon) if icon.exists() else None, gui="edgechromium" if os.name == "nt" else None)
    except Exception:
        api._log.write(traceback.format_exc().rstrip(), "error")
        code = 1
    finally:
        api._shutdown()
    # A call from the page that was being answered as the window closed waits for the page
    # forever, in a thread pywebview does not mark as a daemon, and would keep the process
    # alive. The run has stopped and the logs are closed: end the process here.
    for stream in (sys.__stdout__, sys.__stderr__):
        if stream is not None:
            stream.flush()
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
