from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, url_for

from runner import envs, paths

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
PIPELINE_STAGE_MAP = {
    "asr_stage1": "asr",
    "mfa_stage2": "mfa",
    "clause": "clause",
    "gap": "gap",
    "islands": "gap",
    "caf": "caf",
}
DISPLAY_STAGE_ORDER = [
    ("asr", "ASR"),
    ("mfa", "MFA"),
    ("clause", "Clause"),
    ("gap", "Gap"),
    ("caf", "UF"),
]

def _safe_upload_name(name: str) -> str:
    """File name for an upload, kept in any script.

    werkzeug's secure_filename drops every non-ASCII character, so a Japanese
    name such as 話者01.wav became 01.wav, or nothing.
    """
    name = unicodedata.normalize("NFC", name.replace("\\", "/").split("/")[-1]).strip()
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("._-")
    return f"{stem}{ext}" if stem else ""


INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clausa</title>
  <style>
    :root {
      --bg: #f4efe5;
      --panel: #fffaf2;
      --ink: #222018;
      --muted: #6a6353;
      --line: #d8cdb7;
      --accent: #145a4a;
      --accent-2: #b45f22;
      --ok: #166534;
      --warn: #92400e;
      --bad: #991b1b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: radial-gradient(circle at top left, #fbf7ef 0%, var(--bg) 48%, #eadfca 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1080px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero, .panel {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 42px rgba(43, 34, 19, 0.08);
    }
    .hero {
      padding: 24px;
      margin-bottom: 22px;
    }
    h1, h2, h3 {
      margin: 0 0 12px;
      line-height: 1.1;
    }
    h1 { font-size: 2.1rem; }
    h2 { font-size: 1.3rem; }
    p { margin: 0 0 10px; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 22px;
    }
    .panel { padding: 22px; }
    form { display: grid; gap: 14px; }
    label {
      display: grid;
      gap: 6px;
      font-size: 0.96rem;
      color: var(--ink);
    }
    input, select, button {
      font: inherit;
    }
    input[type="file"], input[type="number"], select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .inline {
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .btn {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      background: linear-gradient(135deg, var(--accent) 0%, #0f766e 100%);
      color: white;
      cursor: pointer;
      font-weight: 700;
    }
    .btn.secondary {
      background: linear-gradient(135deg, var(--accent-2) 0%, #d97706 100%);
    }
    .note {
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      color: var(--muted);
      font-size: 0.92rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    .badge {
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      background: #ece7dc;
      color: var(--ink);
    }
    .queued { background: #fef3c7; color: var(--warn); }
    .pending { background: #ece7dc; color: var(--ink); }
    .running { background: #dbeafe; color: #1d4ed8; }
    .completed { background: #dcfce7; color: var(--ok); }
    .partial { background: #f3e8ff; color: #7e22ce; }
    .failed { background: #fee2e2; color: var(--bad); }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (max-width: 860px) {
      .grid, .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Clausa</h1>
      <p>Upload recordings to get the nine utterance-fluency measures. Everything runs on this machine.</p>
      <p>Jobs run one at a time.</p>
    </section>
    <div class="grid">
      <section class="panel">
        <h2>Start A Job</h2>
        <form method="post" action="{{ url_for('create_job') }}" enctype="multipart/form-data">
          <label>
            Language
            <select name="language">
              <option value="en">English</option>
              <option value="ja">Japanese</option>
            </select>
          </label>
          <div class="row">
            <label>
              Device
              <select name="device">
                <option value="auto">auto</option>
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
                <option value="mlx">mlx (Apple Silicon)</option>
              </select>
            </label>
            <label>
              File Workers
              <input type="number" name="mfa_workers" min="1" value="1">
            </label>
          </div>
          <label>
            Audio Files
            <input type="file" name="audio_files" multiple required accept=".wav,.mp3,.flac,.ogg,.m4a">
          </label>
          <button class="btn" type="submit">Upload And Queue Job</button>
        </form>
        <div class="note" style="margin-top:14px;">
          The UI writes each run into its own folder under <code>jobs/</code> and keeps the uploaded audio, logs, manifest, and outputs together.
        </div>
      </section>
      <section class="panel">
        <h2>Recent Jobs</h2>
        {% if jobs %}
        <table>
          <thead>
            <tr>
              <th>Job</th>
              <th>Status</th>
              <th>Files</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {% for job in jobs %}
            <tr>
              <td><a href="{{ url_for('job_detail', job_id=job.id) }}">{{ job.id }}</a><br><span class="inline">{{ job.language|upper }} / {{ job.device }}</span></td>
              <td><span class="badge {{ job.status }}">{{ job.status }}</span></td>
              <td>{{ job.file_count }}</td>
              <td>{{ job.created_at }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        {% else %}
        <p>No jobs yet.</p>
        {% endif %}
      </section>
    </div>
  </div>
</body>
</html>
"""

JOB_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clausa job {{ job.id }}</title>
  <style>
    :root {
      --bg: #f4efe5;
      --panel: #fffaf2;
      --ink: #222018;
      --muted: #6a6353;
      --line: #d8cdb7;
      --accent: #145a4a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: radial-gradient(circle at top left, #fbf7ef 0%, var(--bg) 48%, #eadfca 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1080px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 42px rgba(43, 34, 19, 0.08);
      padding: 22px;
      margin-bottom: 18px;
    }
    h1, h2 { margin: 0 0 12px; }
    p { margin: 0 0 10px; color: var(--muted); }
    .meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .meta > div {
      background: white;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }
    .badge {
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
      background: #ece7dc;
      color: var(--ink);
    }
    .queued { background: #fef3c7; color: #92400e; }
    .running { background: #dbeafe; color: #1d4ed8; }
    .completed { background: #dcfce7; color: #166534; }
    .failed { background: #fee2e2; color: #991b1b; }
    .actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    .path-line {
      margin-top: 14px;
      color: var(--muted);
      line-height: 1.45;
      word-break: break-all;
    }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-top: 10px;
    }
    .stage-card {
      background: white;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }
    .btn {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 11px 16px;
      background: linear-gradient(135deg, var(--accent) 0%, #0f766e 100%);
      color: white;
      text-decoration: none;
      font-weight: 700;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #1e1b16;
      color: #efe6d5;
      border-radius: 14px;
      padding: 16px;
      min-height: 320px;
      max-height: 60vh;
      overflow: auto;
      margin: 0;
    }
    code { font-family: Consolas, "Courier New", monospace; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (max-width: 860px) {
      .meta { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Job {{ job.id }}</h1>
      <p>{{ job.language|upper }} pipeline with {{ job.file_count }} file{{ '' if job.file_count == 1 else 's' }}.</p>
      <div class="meta">
        <div><strong>Status</strong><br><span id="status-badge" class="badge {{ job.status }}">{{ job.status }}</span></div>
        <div><strong>Device</strong><br><span id="device">{{ job.device }}</span></div>
        <div><strong>Created</strong><br><span id="created">{{ job.created_at }}</span></div>
        <div><strong>File Workers</strong><br><span id="workers">{{ job.mfa_workers }}</span></div>
        <div><strong>Current Stage</strong><br><span id="current-stage">{{ job.current_stage_label }}</span></div>
      </div>
      <div class="actions">
        <a class="btn" href="{{ url_for('index') }}">Back</a>
        <a class="btn" id="open-output-link" href="{{ url_for('open_output', job_id=job.id) }}">Open Output Folder</a>
      </div>
      <p class="path-line"><strong>Output Folder:</strong> <span id="output-dir">{{ job.output_dir }}</span></p>
      <p id="error-text" style="color:#991b1b;margin-top:14px;">{% if job.error %}{{ job.error }}{% endif %}</p>
    </section>
    <section class="panel">
      <h2>Stage Progress</h2>
      <div id="stage-summary" class="stage-grid"></div>
    </section>
    <section class="panel">
      <h2>File Progress</h2>
      <table>
        <thead>
            <tr>
              <th>File</th>
              <th>ASR</th>
              <th>MFA</th>
              <th>Clause</th>
              <th>Gap</th>
              <th>UF</th>
            </tr>
          </thead>
        <tbody id="file-progress-body"></tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Log</h2>
      <pre id="log">{{ job.log_text }}</pre>
    </section>
  </div>
  <script>
    function renderStages(data) {
      const holder = document.getElementById("stage-summary");
      const stages = (data.stage_summary || []);
      if (!stages.length) {
        holder.innerHTML = "<div class='note'>No stage progress yet.</div>";
        return;
      }
      holder.innerHTML = stages.map(stage =>
        `<div class="stage-card">
          <strong>${stage.label}</strong><br>
          <span class="badge ${stage.status}">${stage.status}</span><br>
          <span class="inline">${stage.done_count}/${stage.total_count} files</span>
        </div>`
      ).join("");
    }

    function renderFiles(data) {
      const body = document.getElementById("file-progress-body");
      const rows = (data.file_progress || []);
      if (!rows.length) {
        body.innerHTML = "<tr><td colspan='6'>No files found for this job.</td></tr>";
        return;
      }
      body.innerHTML = rows.map(row =>
        `<tr>
          <td>${row.name}</td>
          <td><span class="badge ${row.asr}">${row.asr}</span></td>
          <td><span class="badge ${row.mfa}">${row.mfa}</span></td>
          <td><span class="badge ${row.clause}">${row.clause}</span></td>
          <td><span class="badge ${row.gap}">${row.gap}</span></td>
          <td><span class="badge ${row.caf}">${row.caf}</span></td>
        </tr>`
      ).join("");
    }

    async function refreshJob() {
      const res = await fetch("{{ url_for('job_api', job_id=job.id) }}", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      const badge = document.getElementById("status-badge");
      badge.textContent = data.status;
      badge.className = "badge " + data.status;
      document.getElementById("workers").textContent = data.mfa_workers;
      document.getElementById("log").textContent = data.log_text || "";
      document.getElementById("error-text").textContent = data.error || "";
      document.getElementById("output-dir").textContent = data.output_dir || "";
      document.getElementById("current-stage").textContent = data.current_stage_label || "Waiting";
      renderStages(data);
      renderFiles(data);
      if (data.status === "queued" || data.status === "running") {
        window.setTimeout(refreshJob, 2500);
      }
    }
    renderStages({ stage_summary: {{ job.stage_summary|tojson }} });
    renderFiles({ file_progress: {{ job.file_progress|tojson }} });
    window.setTimeout(refreshJob, 1200);
  </script>
</body>
</html>
"""


@dataclass
class Job:
    id: str
    language: str
    device: str
    mfa_workers: int
    created_at: str
    work_dir: Path
    input_dir: Path
    output_dir: Path
    log_path: Path
    uploaded_files: list[str]
    status: str = "queued"
    error: str | None = None
    returncode: int | None = None
    archive_path: Path | None = None
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=400))

    @property
    def file_count(self) -> int:
        return len(self.uploaded_files)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def metadata_path(self) -> Path:
        return self.work_dir / "job.json"

    @property
    def progress_path(self) -> Path:
        return self.output_dir / "progress.json"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "language": self.language,
            "device": self.device,
            "mfa_workers": self.mfa_workers,
            "created_at": self.created_at,
            "status": self.status,
            "error": self.error,
            "file_count": self.file_count,
            "output_dir": str(self.output_dir),
            "log_text": self.read_log(),
            "progress": self.read_progress(),
            "stage_summary": self.stage_summary,
            "file_progress": self.file_progress,
            "current_stage_label": self.current_stage_label,
        }

    def metadata_payload(self) -> dict:
        return {
            "id": self.id,
            "language": self.language,
            "device": self.device,
            "mfa_workers": self.mfa_workers,
            "created_at": self.created_at,
            "work_dir": str(self.work_dir),
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "log_path": str(self.log_path),
            "uploaded_files": self.uploaded_files,
            "status": self.status,
            "error": self.error,
            "returncode": self.returncode,
            "archive_path": str(self.archive_path) if self.archive_path else None,
        }

    def save_metadata(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(self.metadata_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def append_log(self, line: str) -> None:
        self.log_tail.append(line.rstrip("\n"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def read_log(self) -> str:
        if self.log_tail:
            return "\n".join(self.log_tail)
        if self.log_path.exists():
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        return ""

    def read_progress(self) -> dict:
        if self.progress_path.exists():
            try:
                return json.loads(self.progress_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "job_status": self.status,
            "current_stage": None,
            "stages": [
                {"key": key, "label": label, "status": "pending"}
                for key, label in DISPLAY_STAGE_ORDER
            ],
        }

    def _caf_done_files(self) -> set[str]:
        caf_csv = self.output_dir / "uf" / "uf_measures.csv"
        done: set[str] = set()
        if not caf_csv.exists():
            return done
        try:
            with caf_csv.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh):
                    file_name = (row.get("file") or "").strip()
                    if file_name:
                        done.add(Path(file_name).stem)
        except Exception:
            return set()
        return done

    def _cell_status(self, *, done: bool, stage_key: str, progress: dict) -> str:
        current_display_stage = PIPELINE_STAGE_MAP.get(progress.get("current_stage"))
        job_status = progress.get("job_status") or self.status
        if done:
            return "completed"
        for stage in progress.get("stages", []):
            if PIPELINE_STAGE_MAP.get(stage.get("key")) == stage_key and stage.get("status") == "skipped":
                return "skipped"
        if job_status == "failed" and current_display_stage == stage_key:
            return "failed"
        return "pending"

    @property
    def current_stage_label(self) -> str:
        progress = self.read_progress()
        current_display_stage = PIPELINE_STAGE_MAP.get(progress.get("current_stage"))
        if current_display_stage:
            for key, label in DISPLAY_STAGE_ORDER:
                if key == current_display_stage:
                    return label
        if self.status == "completed":
            return "Completed"
        if self.status == "failed":
            return "Failed"
        return "Waiting"

    @property
    def file_progress(self) -> list[dict]:
        progress = self.read_progress()
        caf_done = self._caf_done_files()
        rows: list[dict] = []
        for file_name in self.uploaded_files:
            stem = Path(file_name).stem
            asr_done = (self.output_dir / "asr" / "pre_mfa_json" / f"{stem}.json").exists()
            mfa_done = (
                (self.output_dir / "asr" / "json" / f"{stem}.json").exists()
                and (self.output_dir / "asr" / "textgrids_clean" / f"{stem}.TextGrid").exists()
            )
            clause_done = (self.output_dir / "clauses" / f"{stem}.TextGrid").exists()
            gap_done = (self.output_dir / "gap_candidates" / "per_file" / f"{stem}_vad_classifier.csv").exists()
            caf_done_one = stem in caf_done
            rows.append(
                {
                    "name": file_name,
                    "asr": self._cell_status(done=asr_done, stage_key="asr", progress=progress),
                    "mfa": self._cell_status(done=mfa_done, stage_key="mfa", progress=progress),
                    "clause": self._cell_status(done=clause_done, stage_key="clause", progress=progress),
                    "gap": self._cell_status(done=gap_done, stage_key="gap", progress=progress),
                    "caf": self._cell_status(done=caf_done_one, stage_key="caf", progress=progress),
                }
            )
        return rows

    @property
    def stage_summary(self) -> list[dict]:
        progress = self.read_progress()
        rows = self.file_progress
        total = len(rows)
        current_display_stage = PIPELINE_STAGE_MAP.get(progress.get("current_stage"))
        job_status = progress.get("job_status") or self.status
        summary: list[dict] = []
        for key, label in DISPLAY_STAGE_ORDER:
            done_count = sum(1 for row in rows if row[key] == "completed")
            if total and done_count == total:
                status = "completed"
            elif job_status == "failed" and current_display_stage == key:
                status = "failed"
            elif job_status == "running" and current_display_stage == key:
                status = "running"
            elif done_count > 0:
                status = "partial"
            else:
                status = "pending"
            summary.append(
                {
                    "key": key,
                    "label": label,
                    "status": status,
                    "done_count": done_count,
                    "total_count": total,
                }
            )
        return summary


class JobManager:
    def __init__(self):
        self.jobs_root = paths.JOBS_DIR
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._load_existing_jobs()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _load_existing_jobs(self) -> None:
        for metadata_path in sorted(self.jobs_root.glob("*/job.json")):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                job = Job(
                    id=payload["id"],
                    language=payload["language"],
                    device=payload["device"],
                    mfa_workers=int(payload.get("mfa_workers", 1)),
                    created_at=payload["created_at"],
                    work_dir=Path(payload["work_dir"]),
                    input_dir=Path(payload["input_dir"]),
                    output_dir=Path(payload["output_dir"]),
                    log_path=Path(payload["log_path"]),
                    uploaded_files=list(payload.get("uploaded_files") or []),
                    status=payload.get("status", "queued"),
                    error=payload.get("error"),
                    returncode=payload.get("returncode"),
                    archive_path=Path(payload["archive_path"]) if payload.get("archive_path") else None,
                )
                self._jobs[job.id] = job
            except Exception:
                continue

    def create_job(
        self,
        *,
        language: str,
        device: str,
        mfa_workers: int,
        files,
    ) -> Job:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        work_dir = self.jobs_root / job_id
        input_dir = work_dir / "input" / ("en_audio" if language == "en" else "ja_audio")
        output_dir = work_dir / "output" / language
        log_path = work_dir / "run.log"
        input_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files: list[str] = []
        for storage in files:
            if not storage or not storage.filename:
                continue
            filename = _safe_upload_name(storage.filename)
            if not filename:
                continue
            # Later stages key recordings by stem: a.wav and a.mp3 must not collide.
            stems = {Path(f).stem.casefold() for f in uploaded_files}
            if Path(filename).stem.casefold() in stems:
                stem, ext = Path(filename).stem, Path(filename).suffix
                n = 2
                while f"{stem}_{n}".casefold() in stems:
                    n += 1
                filename = f"{stem}_{n}{ext}"
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_AUDIO_EXTS:
                raise ValueError(f"Unsupported file type: {filename}")
            target = input_dir / filename
            storage.save(target)
            uploaded_files.append(filename)

        if not uploaded_files:
            raise ValueError("No valid audio files were uploaded.")
        job = Job(
            id=job_id,
            language=language,
            device=device,
            mfa_workers=mfa_workers,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            work_dir=work_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            log_path=log_path,
            uploaded_files=uploaded_files,
        )
        job.append_log(f"Queued job {job_id}\n")
        job.append_log(f"Language: {language}\n")
        job.append_log(f"Device: {device}\n")
        job.append_log(f"File workers: {mfa_workers}\n")
        job.append_log(f"Uploaded files: {', '.join(uploaded_files)}\n")
        job.append_log("\n")
        job.save_metadata()

        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent_jobs(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:20]

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            self._run_job(job)
            self._queue.task_done()

    def _run_job(self, job: Job) -> None:
        with self._lock:
            job.status = "running"
            job.error = None
            job.save_metadata()

        cmd = self._build_command(job)
        env = envs.stage_env()
        job.append_log("Running command:\n")
        job.append_log(" ".join(f'"{str(part)}"' if " " in str(part) else str(part) for part in cmd) + "\n\n")

        proc = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(paths.REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            job.append_log(line)
        proc.wait()

        with self._lock:
            job.returncode = proc.returncode
            if proc.returncode == 0:
                job.status = "completed"
                job.append_log("\nJob completed successfully.\n")
            else:
                job.status = "failed"
                job.error = f"Pipeline exited with code {proc.returncode}."
                job.append_log(f"\nERROR: {job.error}\n")
            job.save_metadata()

    def _build_command(self, job: Job) -> list[str]:
        return [
            sys.executable, "-m", "runner", "run", job.language, str(job.input_dir),
            "--output", str(job.output_dir), "--device", job.device,
            "--mfa-workers", str(job.mfa_workers),
        ]

    def build_archive(self, job: Job) -> Path:
        if job.status != "completed":
            raise RuntimeError("Job is not completed.")
        if job.archive_path and job.archive_path.exists():
            return job.archive_path

        archive_base = job.work_dir / f"{job.id}_{job.language}_output"
        if archive_base.with_suffix(".zip").exists():
            archive_base.with_suffix(".zip").unlink()
        archive = shutil.make_archive(str(archive_base), "zip", root_dir=str(job.output_dir.parent), base_dir=job.output_dir.name)
        job.archive_path = Path(archive)
        job.save_metadata()
        return job.archive_path


def create_app() -> Flask:
    app = Flask(__name__)
    manager = JobManager()

    @app.route("/")
    def index():
        return render_template_string(INDEX_TEMPLATE, jobs=manager.recent_jobs())

    @app.post("/jobs")
    def create_job():
        language = request.form.get("language", "en")
        if language not in {"en", "ja"}:
            abort(400, "Invalid language.")
        device = request.form.get("device", "auto")
        if device not in {"auto", "cuda", "cpu", "mlx"}:
            abort(400, "Invalid device.")
        try:
            mfa_workers = max(1, int(request.form.get("mfa_workers", "1")))
        except ValueError:
            abort(400, "File Workers must be an integer.")
        files = request.files.getlist("audio_files")

        try:
            job = manager.create_job(
                language=language,
                device=device,
                mfa_workers=mfa_workers,
                files=files,
            )
        except ValueError as exc:
            abort(400, str(exc))
        return redirect(url_for("job_detail", job_id=job.id))

    @app.get("/jobs/<job_id>")
    def job_detail(job_id: str):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        return render_template_string(JOB_TEMPLATE, job=job)

    @app.get("/api/jobs/<job_id>")
    def job_api(job_id: str):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        return jsonify(job.to_dict())

    @app.get("/download/<job_id>")
    def download_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        archive = manager.build_archive(job)
        return send_file(archive, as_attachment=True, download_name=archive.name)

    @app.get("/open-output/<job_id>")
    def open_output(job_id: str):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        target = job.output_dir if job.output_dir.exists() else job.work_dir
        if hasattr(os, "startfile"):
            os.startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return redirect(url_for("job_detail", job_id=job.id))

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "time": time.time()})

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Clausa's browser interface.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app = create_app()
    print(f"Clausa web interface: http://{args.host}:{args.port}  (jobs in {paths.JOBS_DIR})")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
