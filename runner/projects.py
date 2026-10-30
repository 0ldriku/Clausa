"""Projects for the app: named sets of recordings in one language, analysed over several sessions.

A project is a folder <projects folder>/<name>/:

- project.json: the project (PROJECT_FILE, SCHEMA): name, language, created, the
  recordings in order and the sessions.
- audio/: a copy of each added recording, named <id><ext> (a clone on APFS, so it
  costs no space until one of the two changes). The project stays analysable after
  the originals are moved or deleted.
- sessions/<YYYY-MM-DD HH.MM>/: one analysis session, run_pipeline's output folder
  exactly as `./clausa run` makes it, plus recordings/ (links to the project's audio)
  and log.txt (runner/app.py).
- uf_measures.csv: every done recording's row, in project order, each row
  byte-identical to its row in the uf/uf_measures.csv of the session that last
  analysed it (write_csv). A recording's measures do not depend on the other
  recordings of its session, so rows from different sessions belong together.

A recording's id is its file stem: the pipeline names its outputs by stem, so ids
are unique in a project (compared as the Mac's file system compares names) and a
file whose stem is taken is refused. project.json and the CSV are written
atomically. Nothing here deletes a project or a recording's files: they go to the
Trash (trash()).

The app (runner/app.py) does the analysing; this module only reads and writes the
folders, and lists the earlier analyses, the dated run folders of the app before
projects existed (earlier_analyses()).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

SCHEMA = 1
PROJECT_FILE = "project.json"
CSV_NAME = "uf_measures.csv"
AUDIO_DIR = "audio"
SESSIONS_DIR = "sessions"
LANGS = ("en", "ja")
# A session's folder, and an earlier analysis's: "2026-10-23 14.05", "2026-10-23 14.05 (2)"
DATED_FOLDER = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}\.\d{2}( \(\d+\))?$")
STATUSES = ("new", "done", "failed")  # as saved; "running" exists only in memory, during a session

# Windows: folder names, and paths of at most 259 characters (long paths are off
# unless an administrator turns them on, and the stages' Pythons then fail on
# longer ones). In a session folder the deepest path is
# gap_candidates\per_file\<id>_vad_classifier.csv (44 characters besides the id;
# MFA's working files go to the temp folder on Windows, pipeline/*/mfa_from_pre_mfa_json.py).
# check_name and Project.room keep a margin.
WINDOWS_RESERVED_CHARS = '\\/:*?"<>|'
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                          *(f"LPT{i}" for i in range(1, 10))}
WINDOWS_NAME_MAX = 60
WINDOWS_PATH_MAX = 259
SESSION_PART = len("\\sessions\\2026-10-23 14.05 (99)\\")
PER_ID_DEPTH = 50  # the longest output path in a session besides the recording's id, with a margin


class ProjectError(Exception):
    """A request about a project that cannot be done; the message is for the user, in English
    (the log). key and params say it in the window's language (runner/app_ui/strings.js)."""

    def __init__(self, message: str, key: str | None = None, **params):
        super().__init__(message)
        self.key, self.params = key, params


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def name_key(name: str) -> str:
    """A name as the Mac's file system compares names: one Unicode form, without case."""
    return unicodedata.normalize("NFC", name).casefold()


def check_name(name: str) -> str:
    """The name of a project, cleaned; ProjectError if it cannot be a folder name."""
    name = unicodedata.normalize("NFC", str(name or "")).strip()
    if not name:
        raise ProjectError("Give the project a name.", "project.name_empty")
    if os.name == "nt":
        if any(c in name for c in WINDOWS_RESERVED_CHARS) or any(ord(c) < 32 for c in name):
            chars = " ".join(WINDOWS_RESERVED_CHARS)
            raise ProjectError(f"A project name cannot contain any of {chars} (it becomes the name of its folder).",
                               "project.name_chars_windows", chars=chars)
        if name.endswith(".") or name.split(".")[0].strip().upper() in WINDOWS_RESERVED_NAMES:
            raise ProjectError("Windows does not allow that as a folder name (it ends with a dot, or is "
                               "a name such as CON or NUL).", "project.name_reserved_windows")
        if len(name) > WINDOWS_NAME_MAX:
            raise ProjectError(f"That name is too long: Windows limits the length of paths, so a project "
                               f"name has at most {WINDOWS_NAME_MAX} characters.", "project.name_long_windows",
                               max=WINDOWS_NAME_MAX)
    if any(c in name for c in "/:\\") or any(ord(c) < 32 for c in name):
        raise ProjectError("A project name cannot contain / or : (it becomes the name of its folder).",
                           "project.name_chars")
    if name.startswith("."):
        raise ProjectError("A project name cannot start with a dot (its folder would be hidden).", "project.name_dot")
    if len(name.encode("utf-8")) > 200:
        raise ProjectError("That name is too long for a folder name.", "project.name_long")
    return name


def write_atomic(path: Path, data: bytes) -> None:
    """Write a file whole or not at all: a temporary file beside it, then a rename over it."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def clone_file(src: Path, dst: Path) -> str:
    """Copy a file, as a clone where the file system can (APFS: `cp -c`); 'clone' or 'copy'."""
    if sys.platform == "darwin":
        try:
            if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True, timeout=600).returncode == 0:
                return "clone"
        except (OSError, subprocess.SubprocessError):
            pass
        dst.unlink(missing_ok=True)
    shutil.copy2(src, dst)
    return "copy"


def trash(path: Path) -> Path | None:
    """Move a file or folder to the Trash, so Finder's Put Back works; return where it went.

    macOS: NSFileManager.trashItemAtURL through pyobjc, which pywebview depends on.
    Windows: the Recycle Bin, where Restore puts it back (_recycle); where it went is
    not returned (None). Raises OSError with the system's reason if it cannot:
    nothing is deleted then.
    """
    if os.name == "nt":
        _recycle(path)
        return None
    if sys.platform != "darwin":
        raise OSError("Moving to the Trash is only available on macOS and Windows.")
    from Foundation import NSURL, NSFileManager
    ok, moved, error = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
        NSURL.fileURLWithPath_(str(path)), None, None)
    if not ok:
        raise OSError(str(error.localizedDescription()) if error is not None else "the Trash refused it")
    return Path(str(moved.path())) if moved is not None else None


# SHFileOperationW's errors that say something plain; others are given by number (a file
# open in another program gave 0x7c in a test).
_RECYCLE_ERRORS = {2: "it was not found", 3: "it was not found", 5: "access was denied", 0x78: "access was denied",
                   32: "a file in it is open in another program"}


def _recycle(path: Path) -> None:
    """Windows: move a file or folder to the Recycle Bin, or raise OSError and leave it where it is.

    SHFileOperationW with FOF_ALLOWUNDO and none of its dialogs, only on a fixed
    drive: a USB stick or a network folder has no Recycle Bin, and Windows would
    delete the item outright. FOF_WANTNUKEWARNING keeps one dialog: Windows asks
    before it deletes outright anyway (an item too big for the Recycle Bin), and
    No leaves the item in place.
    """
    import ctypes
    from ctypes import wintypes

    full = os.path.abspath(path)
    if not os.path.lexists(full):
        raise OSError(f"{full} was not found")
    drive = os.path.splitdrive(full)[0]
    if not drive or drive.startswith("\\\\") or ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") != 3:  # DRIVE_FIXED
        raise OSError("it is on a drive without a Recycle Bin (a USB stick or a network folder); "
                      "move or delete it in Explorer yourself")

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    fo_delete = 3
    flags = 0x4 | 0x10 | 0x40 | 0x400 | 0x4000  # SILENT, NOCONFIRMATION, ALLOWUNDO, NOERRORUI, WANTNUKEWARNING
    op = SHFILEOPSTRUCTW(None, fo_delete, full + "\0", None, flags, False, None, None)  # pFrom ends with two NULs
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0:
        raise OSError("Windows could not move it to the Recycle Bin: " + _RECYCLE_ERRORS.get(
            result, f"error {result:#x}; is a file in it open in another program?"))
    if op.fAnyOperationsAborted or os.path.lexists(full):
        raise OSError("it was not moved to the Recycle Bin (cancelled, or a file in it is in use)")


def audio_duration(path: Path) -> float | None:
    """Seconds, from the file header (soundfile; PyAV for .m4a). None if unreadable."""
    try:
        if path.suffix.lower() == ".m4a":
            import av
            with av.open(str(path)) as container:
                if container.duration:
                    return container.duration / av.time_base
                stream = container.streams.audio[0]
                return float(stream.duration * stream.time_base) if stream.duration else None
        import soundfile
        return soundfile.info(str(path)).duration
    except Exception:
        return None


def read_csv_rows(path: Path) -> tuple[bytes, dict[str, bytes]]:
    """A uf_measures.csv as (header line, {id: row line}), each line as its exact bytes.

    The column named file (the last one) holds <id>.TextGrid. Lines are split as
    written (pandas ends each with a newline); each is parsed only to find its id.
    """
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        return b"", {}
    header = next(csv.reader(io.StringIO(lines[0].decode("utf-8"))), [])
    if "file" not in header:
        return lines[0], {}
    col = header.index("file")
    rows = {}
    for line in lines[1:]:
        text = line.decode("utf-8").rstrip("\r\n")
        if not text:
            continue
        values = next(csv.reader(io.StringIO(text)), [])
        name = values[col] if col < len(values) else ""
        rows[name[:-len(".TextGrid")] if name.endswith(".TextGrid") else name] = line
    return lines[0], rows


@dataclass
class Recording:
    """A recording of a project. The fields up to `error_params` are saved in project.json;
    the rest is the app's state while it runs."""
    id: str  # the file stem; the pipeline's file id, so <id>.TextGrid in the CSV
    file: str  # the file's name as added
    source: str  # the path it was added from
    added: str  # ISO time
    status: str = "new"  # new, done or failed (running while a session analyses it)
    last_session: str | None = None  # the session folder that last analysed it (done or failed)
    error: str | None = None  # why it failed, for the user, in English
    failed_stage: str | None = None  # the step it failed in ("Word alignment" ...)
    duration_s: float | None = None
    audio: str = ""  # its copy in audio/
    error_code: str | None = None  # the error as a code (err.<code> in runner/app_ui/strings.js) and its
    error_params: dict | None = None  # params, which the window says in its language; none from Clausa 1.0.1
    # not saved:
    progress: float = 0.0
    writing: set = field(default_factory=set)  # "words", "clauses": TextGrids being written
    measures: dict | None = None
    before: dict | None = None  # its saved state when a session started (a Stop restores it)

    SAVED = ("id", "file", "source", "added", "status", "last_session", "error", "failed_stage", "duration_s", "audio",
             "error_code", "error_params")
    # what a session changes (a Stop puts it back)
    OUTCOME = ("status", "last_session", "error", "failed_stage", "error_code", "error_params")

    @property
    def name(self) -> str:
        return self.file

    def to_json(self) -> dict:
        data = {k: getattr(self, k) for k in self.SAVED}
        if self.status not in STATUSES:  # running: saved as it was before the session
            data.update({k: (self.before or {}).get(k, data[k]) for k in self.OUTCOME})
            if data["status"] not in STATUSES:
                data["status"] = "new"
        return data

    @classmethod
    def from_json(cls, d: dict) -> Recording:
        known = {f.name for f in fields(cls)} & set(cls.SAVED)
        rec = cls(**{k: v for k, v in d.items() if k in known})
        if rec.status not in STATUSES:
            rec.status = "new"
        if not isinstance(rec.error_params, dict):
            rec.error_params = None
        return rec

    def saved_state(self) -> dict:
        return {**{k: getattr(self, k) for k in self.OUTCOME}, "measures": self.measures}


@dataclass
class Session:
    name: str  # its folder in sessions/
    started: str
    recordings: list  # ids, in the runner's order
    ended: str | None = None
    outcome: str | None = None  # done, stopped or failed (the run); None while it runs
    done: int = 0
    failed: int = 0


class Project:
    def __init__(self, folder: Path, name: str, language: str, created: str,
                 recordings: list[Recording] | None = None, sessions: list[Session] | None = None):
        self.folder = folder
        self.name = name
        self.language = language
        self.created = created
        self.recordings = recordings or []
        self.sessions = sessions or []

    # ---------- the folder ----------

    @property
    def id(self) -> str:
        return self.folder.name

    @property
    def csv_path(self) -> Path:
        return self.folder / CSV_NAME

    def audio_path(self, r: Recording) -> Path:
        return self.folder / AUDIO_DIR / r.audio

    def session_dir(self, name: str) -> Path:
        return self.folder / SESSIONS_DIR / name

    @staticmethod
    def room(folder: Path, rec_id: str = "") -> int:
        """Windows: characters to spare on the longest path a session of a project in
        this folder writes (for a recording with this id); negative if too long.
        Elsewhere, paths are not limited this way."""
        if os.name != "nt":
            return WINDOWS_PATH_MAX
        return WINDOWS_PATH_MAX - (len(os.path.abspath(folder)) + SESSION_PART + max(len(rec_id), 16) + PER_ID_DEPTH)

    @classmethod
    def create(cls, root: Path, name: str, language: str) -> Project:
        name = check_name(name)
        if language not in LANGS:
            raise ProjectError(f"Unknown language: {language}", "project.unknown_language", language=str(language))
        if cls.room(root / name) < 0:
            raise ProjectError(f"The projects folder's path is too long for a project of that name: Windows "
                               f"limits paths to 260 characters. Choose a shorter name, or a projects folder "
                               f"with a shorter path in Settings ({root}).", "project.path_long_windows",
                               root=str(root))
        root.mkdir(parents=True, exist_ok=True)
        if any(name_key(p.name) == name_key(name) for p in root.iterdir()):
            raise ProjectError(f"There is already a project or folder called {name} in {root}.", "project.exists_in",
                               name=name, root=str(root))
        folder = root / name
        folder.mkdir()
        (folder / AUDIO_DIR).mkdir()
        (folder / SESSIONS_DIR).mkdir()
        project = cls(folder, name, language, now_iso())
        project.save()
        return project

    @classmethod
    def load(cls, folder: Path) -> Project:
        try:
            data = json.loads((folder / PROJECT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProjectError(f"{folder.name}: {PROJECT_FILE} could not be read ({exc})") from exc
        if not isinstance(data, dict) or data.get("language") not in LANGS:
            raise ProjectError(f"{folder.name}: {PROJECT_FILE} is not a Clausa project")
        if int(data.get("schema", 0)) > SCHEMA:
            raise ProjectError(f"{folder.name}: made by a newer Clausa (project schema {data.get('schema')})")
        recs = [Recording.from_json(r) for r in data.get("recordings", []) if isinstance(r, dict) and r.get("id")]
        sessions = []
        for s in data.get("sessions", []):
            if isinstance(s, dict) and s.get("name"):
                sessions.append(Session(**{k: s.get(k) for k in ("name", "started", "recordings", "ended", "outcome")},
                                        done=int(s.get("done") or 0), failed=int(s.get("failed") or 0)))
        # The folder's name is the project's name (a project renamed in Finder follows its folder).
        return cls(folder, folder.name, data["language"], data.get("created") or "", recs, sessions)

    def save(self) -> None:
        data = {"schema": SCHEMA, "app": "Clausa", "name": self.name, "language": self.language,
                "created": self.created, "recordings": [r.to_json() for r in self.recordings],
                "sessions": [vars(s) for s in self.sessions]}
        write_atomic(self.folder / PROJECT_FILE, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    def rename(self, new_name: str) -> None:
        """Rename the project and its folder (not while it is being analysed: the caller checks)."""
        new_name = check_name(new_name)
        if new_name == self.name:
            return
        target = self.folder.parent / new_name
        clash = [p for p in self.folder.parent.iterdir() if name_key(p.name) == name_key(new_name) and p != self.folder]
        if clash:
            raise ProjectError(f"There is already a project or folder called {clash[0].name}.", "project.exists",
                               name=clash[0].name)
        longest = max([r.id for r in self.recordings], key=len, default="")
        if self.room(target, longest) < 0:
            raise ProjectError("That name is too long for this project: Windows limits paths to 260 characters, "
                               "and its folder holds recordings with long names. Choose a shorter name.",
                               "project.rename_long_windows")
        old_folder, old_name = self.folder, self.name
        os.rename(old_folder, target)  # also a change of case only, on a file system that ignores case
        self.folder, self.name = target, new_name
        try:
            self.save()
        except OSError:
            os.rename(target, old_folder)
            self.folder, self.name = old_folder, old_name
            raise

    def new_session_dir(self) -> Path:
        """sessions/<YYYY-MM-DD HH.MM>, with (2), (3) ... if that folder exists; created."""
        base = self.folder / SESSIONS_DIR
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H.%M")
        n = 1
        while True:
            folder = base / (stamp if n == 1 else f"{stamp} ({n})")
            try:
                folder.mkdir()
                return folder
            except FileExistsError:
                n += 1

    def link_recordings(self, session_dir: Path, recs: list[Recording]) -> None:
        """The session's input: recordings/<audio> linking to the project's copies.

        The links are relative, so a session folder keeps working after the project
        is renamed or moved. On Windows hard links instead (the same file under a second
        name, which keeps working too): Windows lets a user make symbolic links only in
        Developer Mode or when elevated. Where a link cannot be made, a copy.
        """
        folder = session_dir / "recordings"
        folder.mkdir()
        for r in recs:
            target = self.audio_path(r)
            link = folder / r.audio
            try:
                if os.name == "nt":
                    os.link(target, link)
                else:
                    link.symlink_to(os.path.relpath(target, folder))
            except OSError:
                shutil.copy2(target, link)

    # ---------- recordings ----------

    def find(self, rec_id: str) -> Recording | None:
        return next((r for r in self.recordings if r.id == rec_id), None)

    def plan_add(self, files: list[Path]) -> tuple[list[tuple[Path, str, str]], list[ProjectError]]:
        """Which files can be added: ([(file, id, audio name)], [refusals])."""
        taken = {name_key(r.id) for r in self.recordings}
        plan, refused = [], []
        for p in files:
            stem = unicodedata.normalize("NFC", p.stem)
            if name_key(stem) in taken:
                refused.append(ProjectError(f"{p.name}: this project already has a recording named {stem}; "
                                            "rename the file to add it", "refused.taken", file=p.name, id=stem))
                continue
            if self.room(self.folder, stem) < 0:
                refused.append(ProjectError(f"{p.name}: the name is too long for Windows in this project's folder "
                                            "(paths have at most 260 characters); rename the file shorter to add it",
                                            "refused.long_windows", file=p.name))
                continue
            taken.add(name_key(stem))
            plan.append((p, stem, stem + p.suffix.lower()))
        return plan, refused

    def copy_in(self, src: Path, audio: str) -> Path:
        """Copy a recording into audio/ (refusing to overwrite a copy that is there).

        A copy that fails part way (disk full, a read error) is removed again, so the
        file can be added once the cause is fixed."""
        folder = self.folder / AUDIO_DIR
        folder.mkdir(exist_ok=True)
        dst = folder / audio
        if dst.exists():
            raise ProjectError(f"{audio} is already in the project's audio folder", "project.audio_exists", audio=audio)
        try:
            clone_file(src, dst)
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
        return dst

    def write_csv(self) -> tuple[int, list[str]]:
        """Regenerate uf_measures.csv; return (its number of rows, ids left out).

        One row per done recording, in project order, each copied byte for byte from
        the CSV of the session that last analysed it. Headers are never mixed: the
        file takes the header of the newest of those sessions, and a done recording
        whose session's CSV has another header (another version of Clausa) is left
        out and returned, so the caller can ask for it to be analysed again. With no
        row, the file is removed.
        """
        order = {s.name: i for i, s in enumerate(self.sessions)}
        found, cache = [], {}
        for r in self.recordings:
            if r.status != "done" or not r.last_session:
                continue
            path = self.session_dir(r.last_session) / "uf" / CSV_NAME
            if path not in cache:
                try:
                    cache[path] = read_csv_rows(path)
                except OSError:
                    cache[path] = (b"", {})
            h, lines = cache[path]
            if r.id in lines:
                found.append((r, h, lines[r.id]))
        if not found:
            self.csv_path.unlink(missing_ok=True)
            return 0, []
        newest = max(found, key=lambda f: (order.get(f[0].last_session, -1), f[0].last_session))[1]
        rows = [line for r, h, line in found if h == newest]
        write_atomic(self.csv_path, newest + b"".join(rows))
        return len(rows), [r.id for r, h, line in found if h != newest]

    def measures(self) -> dict[str, dict]:
        """{id: {measure: value}} of the done recordings, from their sessions' CSVs."""
        found, cache = {}, {}
        for r in self.recordings:
            if r.status != "done" or not r.last_session:
                continue
            path = self.session_dir(r.last_session) / "uf" / CSV_NAME
            if path not in cache:
                cache[path] = read_measures(path)
            if r.id in cache[path]:
                found[r.id] = cache[path][r.id]
        return found

    def last_activity(self) -> str:
        """The newest of: created, a recording added, a session started (ISO times sort as text)."""
        times = [self.created] + [r.added for r in self.recordings] + [s.started for s in self.sessions]
        return max(t for t in times if t) if any(times) else ""


MEASURES = ("AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD")


def read_measures(uf_csv: Path) -> dict:
    """uf_measures.csv: {id: {measure: value}} (its file column is <id>.TextGrid)."""
    rows = {}
    try:
        with uf_csv.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("file") or ""
                rec_id = name[:-len(".TextGrid")] if name.endswith(".TextGrid") else name
                values = {}
                for k in MEASURES:
                    try:
                        values[k] = float(row[k])
                    except (KeyError, TypeError, ValueError):
                        values[k] = None
                rows[rec_id] = values
    except OSError:
        pass
    return rows


def list_projects(root: Path) -> tuple[list[Project], list[str]]:
    """The projects in a projects folder, newest activity first; and problems, as messages."""
    found, problems = [], []
    try:
        folders = [p for p in root.iterdir() if p.is_dir() and (p / PROJECT_FILE).is_file()]
    except OSError:
        return [], []
    for folder in folders:
        try:
            found.append(Project.load(folder))
        except ProjectError as exc:
            problems.append(str(exc))
    found.sort(key=lambda p: (p.last_activity(), p.name), reverse=True)
    return found, problems


@dataclass
class Earlier:
    """An analysis made before projects existed: a dated run folder of the old results folder."""
    dir: Path
    language: str | None
    files: list  # the run's file ids, in its order

    @property
    def id(self) -> str:
        return "earlier:" + str(self.dir)


def earlier_analyses(folders: list[Path]) -> list[Earlier]:
    """The dated run folders in the old results folders, newest first.

    A run folder of the app before projects: <results>/<YYYY-MM-DD HH.MM>/ holding the
    pipeline's output (uf/uf_measures.csv, manifest.json or file_list.json).
    """
    found, seen = [], set()
    for root in folders:
        try:
            items = list(root.iterdir())
        except OSError:
            continue
        for d in items:
            if not d.is_dir() or not DATED_FOLDER.match(d.name) or d.resolve() in seen:
                continue
            if not any((d / f).exists() for f in ("uf/uf_measures.csv", "manifest.json", "file_list.json")):
                continue
            seen.add(d.resolve())
            language, files = None, []
            for name in ("manifest.json", "file_list.json"):
                try:
                    data = json.loads((d / name).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                language = language or (data.get("language") if data.get("language") in LANGS else None)
                files = files or [str(f) for f in data.get("files", [])]
            if not files and (d / "uf" / CSV_NAME).exists():
                files = list(read_measures(d / "uf" / CSV_NAME))
            if language is None:  # an older run without a manifest: its progress.json names it
                try:
                    language = json.loads((d / "progress.json").read_text(encoding="utf-8")).get("language")
                except (OSError, ValueError):
                    pass
            found.append(Earlier(d, language if language in LANGS else None, files))
    found.sort(key=lambda e: e.dir.name, reverse=True)
    return found
