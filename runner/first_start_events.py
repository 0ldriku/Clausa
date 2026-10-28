"""First start's progress events: writing them, reading them, and folding them into what the
window shows.

The protocol: <Clausa folder>/logs/first-start-events.jsonl, one JSON object per line,
only ever appended to; each run of First start begins with a `start` event, and a reader shows
what follows the last one. Written by First start (macOS: First start.command and the helpers in
runner/first_start_macos.py; Windows: Clausa.exe and packaging/windows/first_start.py) and read
by the window (runner/first_start_window.py). Writers send bytes only; fold() computes the
totals, the speed and the time left. Standard library only.

    python -m runner.first_start_events --root <folder> show   # the folded state, as JSON
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

EVENTS = Path("logs") / "first-start-events.jsonl"
STEPS = {"unpack": "Unpacking the app", "download": "Downloading the speech models",
         "prepare": "Preparing (only once)"}
ENV_LABELS = {"qwen3-asr": "The app and speech recognition", "mfa": "Word alignment",
              "ja-electra310": "Japanese grammar"}
PROGRESS_EVERY = 0.25  # seconds between progress events of one item
ALIVE_EVERY = 4.0  # seconds without an event after which a writer's heartbeat sends `alive`
SPEED_WINDOW = 8.0  # seconds of progress the speed is measured over
QUIET_AFTER = 10.0  # seconds without an event after which the page says it is still working


def events_path(root: Path) -> Path:
    return Path(root) / EVENTS


def model_label(m: dict) -> str:
    """A model's label in plain words first: 'Qwen3-ASR-1.7B (speech recognition)' (the lock's)
    becomes 'Speech recognition (Qwen3-ASR-1.7B)'."""
    label = m.get("label") or m["id"]
    name, _, what = label.partition(" (")
    if not what.endswith(")"):
        return label
    return f"{what[:1].upper()}{what[1:-1]} ({name})"


class Writer:
    """Appends events to the file, one line with one write each. progress() sends at most four
    a second per item; with heartbeat=True a thread sends `alive` when nothing else was sent for
    a few seconds (long silent work: hashing a file, loading libraries)."""

    def __init__(self, root: Path, *, heartbeat: bool = False):
        self.path = events_path(root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last: dict[tuple[str, str], float] = {}
        self._sent = time.monotonic()
        if heartbeat:
            threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self) -> None:
        while True:
            time.sleep(1.0)
            if time.monotonic() - self._sent >= ALIVE_EVERY:
                self.emit("alive")

    def emit(self, event: str, **fields) -> None:
        line = json.dumps({"t": round(time.time(), 3), "event": event, **fields}, ensure_ascii=False) + "\n"
        with self._lock:
            # O_APPEND: each line lands whole at the end, also after another writer's lines.
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
            self._sent = time.monotonic()

    def start(self, platform: str, version: str, steps: list[str], *, pid: int | None = None,
              resumed: bool = False, run: str | None = None) -> None:
        self.emit("start", run=run or f"{int(time.time())}-{os.getpid()}", platform=platform, version=version,
                  steps=[{"id": s, "label": STEPS.get(s, s)} for s in steps],
                  pid=os.getpid() if pid is None else pid, resumed=resumed)

    def step(self, step: str, status: str, note: str | None = None) -> None:
        self.emit("step", step=step, status=status, **({"note": note} if note else {}))

    def item(self, step: str, item: str, label: str, total: int | None, status: str,
             done: int | None = None) -> None:
        if status == "running" and done is None:
            done = 0
        if status == "done" and done is None and total is not None:
            done = total
        self.emit("item", step=step, item=item, label=label, total=total, status=status,
                  **({} if done is None else {"done": int(done)}))

    def progress(self, step: str, item: str, done: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last.get((step, item), 0.0) < PROGRESS_EVERY:
            return
        self._last[(step, item)] = now
        self.emit("progress", step=step, item=item, done=int(done))

    def message(self, text: str, level: str = "info") -> None:
        self.emit("message", text=text, level=level)

    def error(self, kind: str, message: str, step: str | None = None, detail: str | None = None,
              again: bool = True) -> None:
        self.emit("error", step=step, kind=kind, message=message, again=again,
                  **({"detail": detail} if detail else {}))

    def done(self) -> None:
        self.emit("done")


def parse_lines(data: bytes) -> tuple[list[dict], bytes]:
    """The events in data, and the rest after the last line end (a line not complete yet).
    Lines that do not parse are left out; a \\r before the \\n is stripped."""
    *lines, rest = data.split(b"\n")
    events = []
    for raw in lines:
        try:
            event = json.loads(raw.rstrip(b"\r").decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(event, dict) and "event" in event:
            events.append(event)
    return events, rest


def read_events(path: Path) -> list[dict]:
    """All complete events of the file (for tests and the command line)."""
    try:
        return parse_lines(Path(path).read_bytes())[0]
    except OSError:
        return []


class Reader:
    """Reads the file as it grows: at each poll it opens the file, reads on from where it
    stopped and closes it. Each event gets the reader's own monotonic time of arrival (`_at`),
    which fold() measures the speed with. A file that is gone keeps what was read; a shorter
    one (a new file) is read from its start."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.rest = b""
        self.events: list[dict] = []

    def poll(self) -> bool:
        try:
            with open(self.path, "rb") as f:
                if os.fstat(f.fileno()).st_size < self.offset:
                    self.offset, self.rest = 0, b""
                f.seek(self.offset)
                data = f.read()
        except OSError:
            return False
        if not data:
            return False
        self.offset += len(data)
        events, self.rest = parse_lines(self.rest + data)
        now = time.monotonic()
        for e in events:
            e["_at"] = now
        self.events.extend(events)
        return bool(events)


def last_run(events: list[dict]) -> list[dict]:
    """The events from the last `start` on (all of them if there is none)."""
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("event") == "start":
            return events[i:]
    return events


def fold(events: list[dict], now: float | None = None) -> dict:
    """What the window shows of the last run: its steps with their items, done and total bytes,
    speed and time left, the current step, messages, the error if First start stopped, whether
    it is done, and how long nothing but `alive` has come (`quiet`, seconds: the page then says it
    is still working). Times are the events'
    `_at` (the reader's monotonic clock) when present, else their `t`; `now` is on the same clock."""
    events = last_run(events)

    def at(e: dict) -> float:
        return float(e.get("_at", e.get("t") or 0.0))

    if now is None:
        now = time.monotonic() if any("_at" in e for e in events) else time.time()
    state = {"phase": "starting", "run": None, "platform": None, "version": None, "pid": None,
             "resumed": False, "steps": [], "current": None, "messages": [], "error": None,
             "quiet": 0.0}
    steps: dict[str, dict] = {}
    # step -> [(time, bytes that arrived in the step so far)]. Only increases between successive
    # events of one item count, so neither what a resumed item already had nor a file that
    # starts again (done going down) is taken for speed.
    samples: dict[str, list[tuple[float, int]]] = {}
    arrived: dict[str, int] = {}
    last = None
    for e in events:
        kind, t = e.get("event"), at(e)
        if kind != "alive":
            last = t
        if kind == "start":
            state.update(phase="running", run=e.get("run"), platform=e.get("platform"),
                         version=e.get("version"), pid=e.get("pid"), resumed=bool(e.get("resumed")))
            steps = {s["id"]: {"id": s["id"], "label": s.get("label") or STEPS.get(s["id"], s["id"]),
                               "status": "waiting", "note": None, "items": {}}
                     for s in e.get("steps") or [] if isinstance(s, dict) and s.get("id")}
        elif kind == "step" and e.get("step") in steps:
            s = steps[e["step"]]
            s["status"] = e.get("status") or s["status"]
            if e.get("note"):
                s["note"] = e["note"]
            if s["status"] == "running":
                state["current"] = s["id"]
        elif kind in ("item", "progress") and e.get("step") in steps:
            s = steps[e["step"]]
            it = s["items"].setdefault(e.get("item"), {"id": e.get("item"), "label": e.get("item"),
                                                       "total": None, "done": 0, "status": "waiting",
                                                       "first": None})
            before = it["done"] if it["first"] is not None else None
            if kind == "item":
                it["label"] = e.get("label") or it["label"]
                if "total" in e:
                    it["total"] = e["total"]
                it["status"] = e.get("status") or it["status"]
                if e.get("done") is not None:
                    it["done"] = int(e["done"])
                if it["status"] == "done" and it["total"] is not None:
                    it["done"] = it["total"]
            else:
                it["done"] = int(e.get("done") or 0)
                if it["status"] == "waiting":
                    it["status"] = "running"
            if it["status"] in ("running", "checking") and it["first"] is None:
                it["first"] = it["done"]  # what was already there when it started: a resumed download
            elif before is not None and it["done"] > before:
                arrived[s["id"]] = arrived.get(s["id"], 0) + it["done"] - before
            samples.setdefault(s["id"], []).append((t, arrived.get(s["id"], 0)))
        elif kind == "message":
            state["messages"].append({"text": e.get("text", ""), "level": e.get("level", "info")})
        elif kind == "error":
            state["phase"] = "error"
            state["error"] = {"kind": e.get("kind", "other"), "message": e.get("message", ""),
                              "detail": e.get("detail"), "step": e.get("step"),
                              "again": e.get("again", True) is not False}
            if e.get("step") in steps and steps[e["step"]]["status"] == "running":
                steps[e["step"]]["status"] = "failed"
            for it in (i for s in steps.values() for i in s["items"].values()):
                if it["status"] in ("running", "checking"):
                    it["status"] = "stopped"
        elif kind == "done":
            state["phase"] = "done"
            state["current"] = None
    if last is not None and state["phase"] == "running":
        state["quiet"] = max(0.0, now - last)
    out = []
    for s in steps.values():
        items = list(s["items"].values())
        counted = [i for i in items if i["total"] is not None]
        # an item without a total that is still to come leaves the step without one
        open_uncounted = any(i["total"] is None and i["status"] != "done" for i in items)
        total = sum(i["total"] for i in counted) if counted and not open_uncounted else None
        done = sum(max(0, min(i["done"], i["total"])) for i in counted) if total is not None else None
        speed = eta = None
        pts = [p for p in samples.get(s["id"], []) if p[0] >= now - SPEED_WINDOW]
        # at least 3 s of samples: the first second of a download is slow and would promise hours
        if s["status"] == "running" and len(pts) >= 2 and pts[-1][0] - pts[0][0] >= 3.0 and pts[-1][1] > pts[0][1]:
            speed = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
            if total is not None:
                eta = max(0.0, (total - done) / speed)
        out.append({"id": s["id"], "label": s["label"], "status": s["status"], "note": s["note"],
                    "total": total, "done": done, "speed": speed, "eta": eta,
                    "resumed": sum(i["first"] or 0 for i in counted),
                    "items": [{k: i[k] for k in ("id", "label", "status", "done", "total")} for i in items]})
    state["steps"] = out
    return state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runner.first_start_events", description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="the Clausa folder")
    ap.add_argument("what", choices=["show"])
    args = ap.parse_args(argv)
    print(json.dumps(fold(read_events(events_path(args.root))), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
