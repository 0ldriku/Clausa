"""Run the six pipeline stages on one recording or a folder of recordings.

Each stage is a script in pipeline/<lang>/ or pipeline/shared/ run in its conda environment
(in paper mode, --paper-compat, the frozen copy in paper/pipeline/; see MODES):

  (pre)      .m4a input only: decode to WAV (shared/convert_audio.py) -> audio/
  1 asr      Qwen3-ASR transcript with rough word timings  -> asr/pre_mfa_json/
  2 mfa      Montreal Forced Aligner word timings           -> asr/json/, asr/textgrids/, asr/textgrids_clean/
             with window/windows_csv, shared/apply_window.py blanks words outside the
             span -> asr/textgrids_target/ (the clause stage's input)
  3 clause   clause segmentation                            -> clauses/
  4 fillers  filled-pause classifier on silent gaps          -> gap_candidates/, patched_textgrids/
  5 islands  speech MFA found inside pauses, plus the Stage 4 -> islands/
             islands for Japanese (shared/placeholder_islands.py; in
             paper mode the Stage 4 islands only, both languages: MODES)
  6 uf       nine utterance-fluency measures                -> uf/uf_measures.csv

Stage 4 needs the filler classifier weights (models/filler_classifier/, not
distributed). Without them it is skipped (the default English measures do not
use it); in paper mode the measures then equal the paper's Stage 4 ablation.

A recording whose words MFA could not align at all keeps going through the
stages on rough ASR timings; outside paper mode it is reported as failed
(manifest.json alignment_failed, present only when not empty; a WARNING; a
file event; _alignment_check).

A run reports itself in progress.json (per stage, read by the web UI) and
events.jsonl (per stage and per file, for the app; see Events). Each stage's
output is copied through unchanged as it is printed. A run can be stopped from
another thread (run_pipeline's stop).
"""

from __future__ import annotations

import codecs
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from runner import envs, models, paths, resources

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
# A recording MFA aligned no word of (_alignment_check): manifest.json and the file event.
ALIGNMENT_FAILED = "Word alignment failed (MFA aligned no words), so the measures would be wrong. See the log."
# A stage that ran out of GPU memory (torch's error, or cuBLAS or cuDNN failing to allocate): said in
# plain words instead (GPU_MEMORY), by the command line at the end of the run and by the app.
GPU_MEMORY_ERROR = re.compile(r"CUDA out of memory|CUDA error: out of memory|OutOfMemoryError|"
                              r"CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation")
GPU_MEMORY = ("The graphics card ran out of memory during speech recognition. Close other programs that use "
              "the graphics card, then analyse again. Clausa needs an NVIDIA graphics card with at least 8 GB "
              "of memory; on Windows, leave the NVIDIA Control Panel's \"CUDA - Sysmem Fallback Policy\" at "
              "its default (not \"Prefer No Sysmem Fallback\").")
DECODE_EXTS = {".m4a"}  # not readable by libsndfile: decoded to WAV first
STAGES = [("asr_stage1", "ASR"), ("mfa_stage2", "MFA"), ("clause", "Clause"), ("gap", "Fillers"),
          ("islands", "Islands"), ("caf", "UF")]
DECODE = ("decode", "Decode")  # the (pre) step: a stage in events.jsonl, not in progress.json
HEADERS = {"decode": "Decoding audio libsndfile cannot read"}  # printed instead of the label


@dataclass(frozen=True)
class Language:
    code: str
    clause_script: str
    clause_env: str
    uf_script: str
    uf_env: str


LANGUAGES = {
    "en": Language("en", "textgrid_caf_segmenter.py", paths.ASR_ENV, "caf_calculator_vad_classifier.py", paths.ASR_ENV),
    "ja": Language("ja", "ja_clause_segmenter_v102.py", paths.JA_ENV, "caf_calculator_ja_gap_classifier.py", paths.JA_ENV),
}


@dataclass(frozen=True)
class Mode:
    """Everything a run does differently in paper mode (--paper-compat).

    The stages themselves differ by their code: paper mode runs the frozen copy
    of the pipeline in paper/pipeline/ (paper/README.md), the default mode the
    developed pipeline/. The rest is in these fields; nothing else in the
    runner depends on the mode (`./clausa doctor` checks this).
    """
    pipeline_dir: Path                  # the stage scripts, also their import root (PYTHONPATH)
    placeholder_spans: frozenset[str]   # languages whose islands add the speech MFA found in pauses
    classifier_islands: frozenset[str]  # languages whose islands keep the Stage 4 classifier's islands
    alignment_check: bool               # report recordings MFA aligned no word of (_alignment_check)
    mlx: bool                           # Stage 1 may run through MLX (the CLI refuses --device mlx otherwise)
    env_aliases: tuple[tuple[str, str], ...] = ()  # (setting, name the stage scripts read it by) pairs


# The settings paper/pipeline/ (frozen) reads by the tool's earlier names; stage_env passes them on.
PAPER_ENV_ALIASES = (
    ("CLAUSA_MFA_ENV_NAME", "NEWCAF_MFA_ENV_NAME"), ("CLAUSA_MFA_PREFIX", "NEWCAF_MFA_ENV"),
    ("CLAUSA_MFA_EXE", "NEWCAF_MFA_EXE"), ("CLAUSA_MFA_LIB_BIN", "NEWCAF_MFA_LIB_BIN"),
    ("CLAUSA_MFA_RUNTIME_BASE", "NEWCAF_MFA_RUNTIME_BASE"),
    ("CLAUSA_MFA_FULL_DICTIONARY", "CAFASR_MFA_FULL_DICTIONARY"), ("CLAUSA_EN_SPACY_MODEL", "CAFASR_EN_SPACY_MODEL"),
)


# The run's mode, by run_pipeline's paper_compat. English uses the placeholder
# spans instead of the classifier's islands (docs/improvements.md).
MODES = {
    False: Mode(paths.PIPELINE_DIR, placeholder_spans=frozenset({"en", "ja"}),
                classifier_islands=frozenset({"ja"}), alignment_check=True, mlx=True),
    True: Mode(paths.PAPER_PIPELINE_DIR, placeholder_spans=frozenset(),
               classifier_islands=frozenset({"en", "ja"}), alignment_check=False, mlx=False,
               env_aliases=PAPER_ENV_ALIASES),
}


class Stopped(Exception):
    """The run was stopped through run_pipeline's stop event."""


class GpuOutOfMemory(subprocess.CalledProcessError):
    """A stage command failed, and its output says it ran out of GPU memory: str() is GPU_MEMORY."""

    def __str__(self) -> str:
        return GPU_MEMORY


def find_audio(source: Path) -> tuple[Path, list[str]]:
    """Return (folder, file stems) for an audio file or a folder of audio files."""
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"Not an audio file ({', '.join(sorted(AUDIO_EXTS))}): {source}")
        return source.parent, [source.stem]
    if source.is_dir():
        stems = sorted(p.stem for p in source.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
        if not stems:
            raise FileNotFoundError(f"No audio files in {source}")
        return source, stems
    raise FileNotFoundError(source)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Progress:
    """progress.json, read by the web UI while a job runs."""

    def __init__(self, path: Path, **info):
        self.path = path
        self.data = {
            **info,
            "job_status": "pending",
            "current_stage": None,
            "updated_at": _now(),
            "stages": [{"key": k, "label": label, "status": "pending"} for k, label in STAGES],
        }
        self._write()

    def _write(self) -> None:
        self.data["updated_at"] = _now()
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def stage(self, key: str, status: str, error: str | None = None) -> None:
        self.data["current_stage"] = key if status in {"running", "failed"} else None
        if status in {"running", "failed"}:
            self.data["job_status"] = status
        for s in self.data["stages"]:
            if s["key"] == key:
                s["status"] = status
                s["started_at" if status == "running" else "finished_at"] = _now()
                if error:
                    s["error"] = error
        if error:
            self.data["last_error"] = error
        self._write()

    def finish(self, status: str) -> None:
        self.data["job_status"] = status
        self.data["current_stage"] = None
        self._write()


class Events:
    """events.jsonl: the run as one JSON object per line, appended as it happens.

    Rewritten at the start of each run. Every event has "time" (ISO 8601, local
    time with offset), "event" (run, stage or file) and "status":

      run    started   language, device, paper_compat, files (ids), stages (the
                       {key, label} of the stages that will run), output_dir
             completed uf_csv, seconds
             failed    stage (key, or null between stages), error
             stopped   stage (key, or null between stages)
      stage  started, completed (+ seconds), failed (+ error) or stopped, each with
             stage (key), label, index (1..count) and count of the stages that run;
             skipped with stage and label only (Stage 4 without its weights)
      file   started, completed or failed (+ error, the script's message), with
             stage, file (id), i and n: from the stage scripts' per-file lines
             (_FileLines), so only for the stages that print them. Also failed
             with stage mfa_stage2 and error ALIGNMENT_FAILED, after that
             stage's own events for the file, when MFA aligned none of its
             words (_alignment_check; not in paper mode): the run goes on,
             and the file keeps getting events

    on_event, if given, receives each event as it is written, in the thread
    that called run_pipeline.
    """

    def __init__(self, path: Path, on_event: Callable[[dict], None] | None = None):
        self.path = path
        self.on_event = on_event
        path.write_text("", encoding="utf-8")

    def emit(self, event: str, status: str, **fields) -> None:
        record = {"time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                  "event": event, "status": status, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.on_event is not None:
            self.on_event(record)


_STARTED = re.compile(r"\[(\d+)/(\d+)\] (.+)")
_GAP_DONE = re.compile(r"\[(\d+)/(\d+)\] (.+?): (.*)")
_CLAUSE_COUNT = re.compile(r"Processing (\d+) TextGrid files\.\.\.")
_UF_DONE = re.compile(r"Processed: (.+)\.TextGrid")
_UF_ERROR = re.compile(r"Error processing (.+)\.TextGrid: (.*)")


class _FileLines:
    """Turns a stage script's own per-file lines (stdout) into file events.

      ASR, MFA  "[i/N] <id>" as a file starts, then "  Done: ..." or "  FAILED: ..."
                (ASR also "No words detected, skipping."). With --mfa-workers > 1,
                MFA prints both when a file is done, and i counts finished files.
      Clause    "Processing N TextGrid files...", then per file "Processing: <id>.TextGrid"
                and "  Saved: ..." or "  ERROR: ..."
      Fillers   "[ii/N] <id>: ..." when a file is done (": FAILED -> ..." if not):
                completed or failed only
      UF        "Processed: <id>.TextGrid" or "Error processing <id>.TextGrid: ...":
                completed or failed only, n = the run's number of files
      Decode, Islands: no per-file lines.

    Lines about a file that is not in the run are ignored.
    """

    def __init__(self, stage: str, files: list[str], emit: Callable[..., None]):
        self.stage = stage
        self.files = set(files)
        self.emit = emit
        self.i, self.n = 0, len(files)
        self.current: tuple[str, int, int] | None = None  # (file, i, n) started, not yet done

    def feed(self, line: str) -> None:
        if self.stage in ("asr_stage1", "mfa_stage2"):
            m = _STARTED.fullmatch(line)
            if m:
                self._start(m[3], int(m[1]), int(m[2]))
            elif line.startswith("  Done: "):
                self._done("completed")
            elif line.startswith("  FAILED: "):
                self._done("failed", line[len("  FAILED: "):])
            elif line.strip() == "No words detected, skipping.":
                self._done("failed", "No words detected")
        elif self.stage == "clause":
            m = _CLAUSE_COUNT.fullmatch(line)
            if m:
                self.n = int(m[1])
            elif line.startswith("Processing: ") and line.endswith(".TextGrid"):
                self.i += 1
                self._start(line[len("Processing: "):-len(".TextGrid")], self.i, self.n)
            elif line.startswith("  Saved: "):
                self._done("completed")
            elif line.startswith("  ERROR: "):
                self._done("failed", line[len("  ERROR: "):])
        elif self.stage == "gap":
            m = _GAP_DONE.fullmatch(line)
            if m and m[3] in self.files:
                self.current = (m[3], int(m[1]), int(m[2]))
                failed = m[4].startswith("FAILED")
                self._done("failed" if failed else "completed", m[4].split("-> ", 1)[-1] if failed else None)
        elif self.stage == "caf":
            m = _UF_DONE.fullmatch(line) or _UF_ERROR.fullmatch(line)
            if m and m[1] in self.files:
                self.i += 1
                self.current = (m[1], self.i, self.n)
                self._done("completed" if m.re is _UF_DONE else "failed", m[2] if m.re is _UF_ERROR else None)

    def _start(self, file: str, i: int, n: int) -> None:
        self.current = (file, i, n) if file in self.files else None
        if self.current:
            self.emit("file", "started", stage=self.stage, file=file, i=i, n=n)

    def _done(self, status: str, error: str | None = None) -> None:
        if self.current:
            file, i, n = self.current
            self.emit("file", status, stage=self.stage, file=file, i=i, n=n, **({"error": error} if error else {}))
        self.current = None


def _copy(src, dst, on_line: Callable[[str], None] | None = None) -> None:
    """Copy a child's output stream to one of ours as it arrives, unchanged.

    Complete lines also go to on_line (decoded, without the line end). dst
    may be None (no console, e.g. pythonw) or a text stream without .buffer.
    """
    raw = getattr(dst, "buffer", None)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = b""
    while True:
        chunk = src.read(65536)
        if not chunk:
            break
        if raw is not None:
            dst.flush()  # what this process printed itself comes first
            raw.write(chunk)
            raw.flush()
        elif dst is not None:
            dst.write(decoder.decode(chunk))
            dst.flush()
        if on_line is not None:
            *lines, pending = (pending + chunk).split(b"\n")
            for line in lines:
                on_line(line.decode("utf-8", "replace").rstrip("\r"))
    if on_line is not None and pending:
        on_line(pending.decode("utf-8", "replace").rstrip("\r"))


def _end(proc: subprocess.Popen, group: bool) -> None:
    """End a stage command; with group, also every process it started.

    group: the command leads its own process group (POSIX) or tree (Windows).
    POSIX: SIGINT first, so the scripts can remove their temporary folders,
    SIGKILL 5 s later or once the command is gone. Without group, as
    subprocess.run does on an exception: kill the command and wait.
    """
    if not group:
        proc.kill()
        proc.wait()
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        proc.wait()
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    except ProcessLookupError:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _run_command(cmd: list[str], env: dict[str, str], *, on_line: Callable[[str], None] | None = None,
                 stop: threading.Event | None = None) -> None:
    """Run one stage command like subprocess.run(check=True), output copied through.

    The command's stdout and stderr go to this process's sys.stdout and
    sys.stderr unchanged as they arrive (_copy); complete stdout lines also go
    to on_line. With stop, the command gets its own process group, and setting
    stop ends it with everything it started (conda run, MFA, worker processes)
    and raises Stopped.
    """
    group = stop is not None
    if not group:
        extra = {}
    elif os.name == "nt":
        extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        extra = {"start_new_session": True}
    proc = subprocess.Popen(cmd, env=env, cwd=str(paths.REPO_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0, **extra)
    stopped = threading.Event()

    def watch() -> None:
        while proc.poll() is None:
            if stop.wait(0.2):
                stopped.set()
                _end(proc, group)
                return

    tail = []  # the command's last lines, both streams: to tell a stage that ran out of GPU memory

    def remember(line: str) -> None:
        tail.append(line)
        del tail[:-200]

    def stdout_line(line: str) -> None:
        remember(line)
        if on_line is not None:
            on_line(line)

    threads = [threading.Thread(target=_copy, args=(proc.stderr, sys.stderr, remember), daemon=True)]
    if group:
        threads.append(threading.Thread(target=watch, daemon=True))
    for t in threads:
        t.start()
    try:
        _copy(proc.stdout, sys.stdout, stdout_line)
        proc.wait()
    except BaseException:
        _end(proc, group)
        raise
    finally:
        for t in threads:
            t.join(timeout=30)
        proc.stdout.close()
        proc.stderr.close()
    if stopped.is_set():
        raise Stopped("stopped")
    if proc.returncode != 0:
        if any(GPU_MEMORY_ERROR.search(line) for line in tail):
            raise GpuOutOfMemory(proc.returncode, cmd)
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _alignment_check(json_dir: Path, files: list[str]) -> tuple[list[dict], list[dict]]:
    """Recordings whose Stage 2 words did not get MFA timings: (failed, partial),
    each a list of {file_id, rough, words}.

    Stage 2 keeps the rough ASR timing of every word MFA did not align
    (source "rough" in asr/json/<id>.json) and carries on. When MFA aligned
    none of a recording's words (it could not run, or could not align that
    audio), its pauses and speed are measured from rough timings: that
    recording counts as failed. A recording with only some rough words is a
    partial fallback, reported with the count. Recordings without Stage 2
    output (failed before) are not listed.
    """
    failed, partial = [], []
    for file_id in files:
        try:
            words = json.loads((json_dir / f"{file_id}.json").read_text(encoding="utf-8"))["words"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        rough = sum(1 for w in words if w.get("source") == "rough")
        if rough:
            (failed if rough == len(words) else partial).append({"file_id": file_id, "rough": rough,
                                                                 "words": len(words)})
    return failed, partial


def _command(env_name: str, script: Path, *args: str | Path) -> tuple[str, list[str]]:
    """A stage script's command and the environment it runs in."""
    return env_name, envs.python_in_env(env_name, script, *args)


def run_pipeline(
    language: str,
    source: Path,
    output: Path,
    *,
    device: str = "auto",
    mfa_workers: int | None = None,
    threshold: float = 0.50,
    paper_compat: bool = False,
    window: tuple[float, float] | None = None,
    windows_csv: Path | None = None,
    on_event: Callable[[dict], None] | None = None,
    stop: threading.Event | None = None,
) -> Path:
    """Run all stages; return the path of the UF measures CSV.

    paper_compat: run the pipeline used for the article (paper mode, MODES).

    mfa_workers: recordings aligned at once in Stage 2; None: as many as the
    computer's cores and memory allow, measured just before Stage 2
    (resources.default_mfa_workers). The outputs do not depend on it.

    window / windows_csv: analyse only the learner's task speech, one window
    for every file or a CSV of file_id,start,end (shared/apply_window.py).

    on_event: receives every event written to events.jsonl (Events).

    stop: a threading.Event another thread may set to stop the run. It is
    checked before each stage, and a running stage is ended at once with
    every process it started (_end). The stage and the run are then marked
    "stopped" in progress.json and events.jsonl, and Stopped is raised.
    Finished stages keep their output; a stopped run is started again from the
    beginning. Ctrl+C (KeyboardInterrupt) is recorded the same way.
    """
    lang = LANGUAGES[language]
    mode = MODES[paper_compat]
    if not mode.pipeline_dir.is_dir():  # the app zips leave paper/ out
        raise FileNotFoundError(f"{mode.pipeline_dir} not found: paper mode (--paper-compat) is not part of the "
                                "app. Run it from a clone of the repository (docs/reproducing-the-paper.md).")
    audio_dir, files = find_audio(source)
    scripts, shared = mode.pipeline_dir / language, mode.pipeline_dir / "shared"
    output = output.resolve()
    asr_out, clauses_out = output / "asr", output / "clauses"
    gap_out, patched_out = output / "gap_candidates", output / "patched_textgrids"
    islands_out = output / "islands"
    uf_csv = output / "uf" / "uf_measures.csv"
    output.mkdir(parents=True, exist_ok=True)
    # One process environment per conda environment (in bundled mode their PATHs differ).
    stage_envs = {name: envs.stage_env(name, pipeline_dir=mode.pipeline_dir, aliases=mode.env_aliases)
                  for name in dict.fromkeys((paths.ASR_ENV, lang.clause_env, lang.uf_env))}
    input_dir = audio_dir
    decode = any(p.suffix.lower() in DECODE_EXTS and p.stem in files for p in audio_dir.iterdir() if p.is_file())
    if decode:
        # Every stage reads audio with libsndfile; give them WAV copies (others as symlinks).
        audio_dir = output / "audio"

    file_list = output / "file_list.json"
    file_list.write_text(json.dumps({"files": files}, indent=2, ensure_ascii=False), encoding="utf-8")
    run_stage4 = paths.FILLER_MODEL.exists()
    use_spans = language in mode.placeholder_spans
    use_classifier = language in mode.classifier_islands

    windowed = window is not None or windows_csv is not None
    clause_in = asr_out / ("textgrids_target" if windowed else "textgrids_clean")
    window_cmd = []
    if windowed:
        window_cmd = [_command(
            paths.ASR_ENV, shared / "apply_window.py",
            "--in-dir", asr_out / "textgrids_clean", "--out-dir", clause_in,
            "--file-list-json", file_list, "--file-list-key", "files",
            *(["--window", str(window[0]), str(window[1])] if window is not None else []),
            *(["--windows-csv", Path(windows_csv).resolve()] if windows_csv is not None else []))]

    # Each stage: its commands, in order, as (environment, command line).
    commands = {
        "decode": [_command(
            paths.ASR_ENV, shared / "convert_audio.py",
            "--in-dir", input_dir, "--out-dir", audio_dir, "--file", *files)],
        "asr_stage1": [_command(
            paths.ASR_ENV, scripts / "asr_to_pre_mfa_json.py",
            "-i", audio_dir, "-o", asr_out, "--device", device, "--file", *files)],
        "mfa_stage2": [_command(
            paths.ASR_ENV, scripts / "mfa_from_pre_mfa_json.py",
            "-i", asr_out / "pre_mfa_json", "-a", audio_dir, "-o", asr_out,
            "--workers", str(mfa_workers or 1), "--mfa-num-jobs", "1", "--file", *files)],
        "clause": window_cmd + [_command(
            lang.clause_env, scripts / lang.clause_script,
            "-i", clause_in, "-o", clauses_out,
            "--disfluency-model", models.snapshot(paths.DISFLUENCY_MODEL_ID))],
        "gap": [_command(
            paths.ASR_ENV, shared / "postprocess_vad_filler_classifier.py",
            "--file-list-json", file_list, "--file-list-key", "files",
            "--audio-dir", audio_dir, "--asr-json-dir", asr_out / "json",
            "--model-path", paths.FILLER_MODEL, "--out-dir", gap_out,
            "--textgrid-dir", clauses_out, "--out-textgrid-dir", patched_out,
            # The reported run used max_gap 3.0 (stage4_candidates/RUN_SUMMARY.json); the script default is 1.2.
            "--threshold", str(threshold), "--max-gap", "3.0")],
        "islands": [_command(
            paths.ASR_ENV, shared / "placeholder_islands.py",
            "--file-list-json", file_list, "--file-list-key", "files",
            "--asr-json-dir", asr_out / "json", "--stage4-dir", gap_out / "per_file",
            "--out-dir", islands_out, "--placeholder-spans", "on" if use_spans else "off",
            "--classifier-islands", "on" if use_classifier else "off")],
        "caf": [_command(
            lang.uf_env, scripts / lang.uf_script, clauses_out,
            "--candidate-dir", islands_out / "per_file",
            "--file-list-json", file_list, "--file-list-key", "files", "--output", uf_csv)],
    }

    stages = ([DECODE] if decode else []) + STAGES
    running = [key for key, _ in stages if key != "gap" or run_stage4]
    progress = Progress(output / "progress.json", language=language, device=device,
                        input_dir=str(input_dir), output_dir=str(output), files=files)
    events = Events(output / "events.jsonl", on_event)
    events.emit("run", "started", language=language, device=device, paper_compat=paper_compat, files=files,
                stages=[{"key": key, "label": label} for key, label in stages if key in running],
                output_dir=str(output))
    run_started = time.monotonic()
    current = None  # the running stage's key, label, index and count
    alignment_failed = []  # [{file_id, error}] after Stage 2 (_alignment_check); not checked in paper mode
    gpu_memory_failed = []  # files a stage reported as failed for lack of GPU memory (GPU_MEMORY)

    def file_event(*args, **fields) -> None:
        if fields.get("error") and GPU_MEMORY_ERROR.search(fields["error"]):
            gpu_memory_failed.append(fields["file"])
        events.emit(*args, **fields)
    try:
        models.ensure(language)  # the pinned models this language needs, if any is missing
        for key, label in stages:
            if stop is not None and stop.is_set():
                raise Stopped("stopped")
            if key == "gap" and not run_stage4:
                print(f"\n=== {label}: skipped (no filler classifier at {paths.FILLER_MODEL}) ===")
                (gap_out / "per_file").mkdir(parents=True, exist_ok=True)
                progress.stage(key, "skipped")
                events.emit("stage", "skipped", stage=key, label=label)
                continue
            print(f"\n=== {HEADERS.get(key, label)} ===", flush=True)
            if key == "mfa_stage2" and mfa_workers is None:
                mfa_workers, why = resources.default_mfa_workers(len(files))
                if len(files) > 1:
                    print(f"Aligning {why}", flush=True)
                for _, cmd in commands[key]:
                    cmd[cmd.index("--workers") + 1] = str(mfa_workers)
            current = {"stage": key, "label": label, "index": running.index(key) + 1, "count": len(running)}
            progress.stage(key, "running")
            events.emit("stage", "started", **current)
            stage_started = time.monotonic()
            lines = _FileLines(key, files, file_event)
            for env_name, cmd in commands[key]:
                _run_command(cmd, stage_envs[env_name], on_line=lines.feed, stop=stop)
            if key == "mfa_stage2" and mode.alignment_check:
                failed, partial = _alignment_check(asr_out / "json", files)
                for f in partial:
                    print(f"WARNING: {f['file_id']}: MFA left {f['rough']} of {f['words']} words with rough ASR timings")
                for f in failed:
                    print(f"WARNING: {f['file_id']}: MFA aligned none of its {f['words']} words; "
                          "they keep rough ASR timings")
                    events.emit("file", "failed", stage=key, file=f["file_id"], i=files.index(f["file_id"]) + 1,
                                n=len(files), error=ALIGNMENT_FAILED)
                alignment_failed = [{"file_id": f["file_id"], "error": ALIGNMENT_FAILED} for f in failed]
            progress.stage(key, "completed")
            events.emit("stage", "completed", **current, seconds=round(time.monotonic() - stage_started, 1))
            current = None
    except (Stopped, KeyboardInterrupt):
        if current is not None:
            progress.stage(current["stage"], "stopped")
            events.emit("stage", "stopped", **current)
        progress.finish("stopped")
        events.emit("run", "stopped", stage=current and current["stage"])
        raise
    except Exception as exc:
        if gpu_memory_failed and not isinstance(exc, GpuOutOfMemory):  # e.g. Stage 2 had no file left
            print(f"\nERROR: {len(gpu_memory_failed)} of {len(files)} recordings were not analysed: "
                  f"{', '.join(gpu_memory_failed)}. {GPU_MEMORY}")
        if current is not None:
            progress.stage(current["stage"], "failed", error=str(exc))
            events.emit("stage", "failed", **current, error=str(exc))
        progress.finish("failed")
        events.emit("run", "failed", stage=current and current["stage"], error=str(exc))
        raise
    progress.finish("completed")

    # Stage 4 skips a file it cannot read and carries on; list those files so
    # their pauses are not silently scored without filler refinement.
    stage4_failed = []
    if run_stage4 and (gap_out / "RUN_SUMMARY.json").exists():
        stage4_failed = json.loads((gap_out / "RUN_SUMMARY.json").read_text(encoding="utf-8")).get("failed", [])
        for f in stage4_failed:
            print(f"WARNING: Stage 4 failed for {f['file_id']}: {f['error']}")
    if gpu_memory_failed:
        print(f"\nERROR: {len(gpu_memory_failed)} of {len(files)} recordings were not analysed: "
              f"{', '.join(gpu_memory_failed)}. {GPU_MEMORY}")
    # Their measures are in the CSV, computed from rough timings: say so last.
    if alignment_failed:
        print(f"\nWARNING: word alignment failed for {len(alignment_failed)} of {len(files)} recordings; "
              f"their rows in {uf_csv.name} are not reliable:")
        for f in alignment_failed:
            print(f"  {f['file_id']}")

    (output / "manifest.json").write_text(json.dumps({
        "language": language,
        "device": device,
        "input_dir": str(input_dir),
        "files": files,
        "stage4": "run" if run_stage4 else "skipped (no filler classifier)",
        "stage4_failed": stage4_failed,
        **({"alignment_failed": alignment_failed} if alignment_failed else {}),
        "placeholder_spans": "on" if use_spans else "off",
        "classifier_islands": "on" if use_classifier else "off",
        "mfa_workers": mfa_workers,
        "threshold": threshold,
        "paper_compat": paper_compat,
        "window": list(window) if window is not None else None,
        "windows_csv": str(Path(windows_csv).resolve()) if windows_csv is not None else None,
        "outputs": {
            "asr": str(asr_out), "word_textgrids": str(asr_out / "textgrids_clean"),
            "clauses": str(clauses_out), "gap_candidates": str(gap_out), "islands": str(islands_out),
            "patched_textgrids": str(patched_out), "uf_csv": str(uf_csv),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    events.emit("run", "completed", uf_csv=str(uf_csv), seconds=round(time.monotonic() - run_started, 1))
    return uf_csv
