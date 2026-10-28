"""The downloaded models, pinned: download, check and place them in the cache.

Every model Clausa downloads is pinned in runner/requirements/models-lock.json:
the four Hugging Face repositories at a commit (Qwen3-ASR, its forced aligner,
the disfluency detector oldriku/Clausa, GiNZA-Electra's transformer) and MFA's English
and Japanese models from the mfa-models GitHub release, with the size and
sha256 of every file. One code path fetches them: `./clausa setup`, `./clausa
models`, the download of missing models at the start of a run (run_pipeline),
and the portable app's First start (`python -m runner.models`). Standard
library only, so it runs in the CLI's Python and in the bundled environment's
alike.

Hugging Face files go into the hub cache layout (blobs/, snapshots/<commit>/,
refs/main) exactly as huggingface_hub stores them. The stages load the models
by repository id with HF_HUB_OFFLINE set, which resolves refs/main, so refs/main
is set to the pinned commit. MFA files go to <MFA_ROOT>/pretrained_models/,
where `mfa align` finds them by name.

Downloads resume (a partial file is kept as <name>.incomplete and continued
with an HTTP range request), every file is checked against the lock before it
is put in place, and nothing asks a question. HF_ENDPOINT (a Hugging Face
mirror) is respected; CLAUSA_MFA_BASE_URL replaces the GitHub release address
of the MFA files (a mirror, or an unreachable address in tests).
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import functools
import hashlib
import http.client
import json
import os
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from runner import paths

LOCK = paths.MODELS_LOCK
HF_ENDPOINT = "https://huggingface.co"
MFA_RELEASES = "https://github.com/MontrealCorpusTools/mfa-models/releases/download"
SPARE = 1 << 30  # free space kept beyond the files still to download
EXIT = {"network": 2, "disk": 3, "checksum": 4}


class DownloadError(Exception):
    """A model could not be downloaded; kind is network, disk or checksum (EXIT)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@functools.lru_cache(maxsize=None)
def lock() -> list[dict]:
    return json.loads(LOCK.read_text(encoding="utf-8"))["models"]


def model(model_id: str) -> dict:
    return next(m for m in lock() if m["id"] == model_id)


def needed(language: str | None = None, only: str | None = None) -> list[dict]:
    """The models a run in this language uses (all with None); only: huggingface or mfa."""
    return [m for m in lock() if (language is None or language in m["languages"])
            and (only is None or m["source"] == only)]


def hf_repo_dir(model_id: str, cache: Path | None = None) -> Path:
    """The model's folder in the Hugging Face hub cache of cache (default: the cache, paths.CACHE_DIR)."""
    return (cache or paths.CACHE_DIR) / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"


def snapshot(model_id: str, cache: Path | None = None) -> Path:
    """The folder a Hugging Face model's pinned files are used from (snapshots/<commit>): what
    the clause stage gets as --disfluency-model for paths.DISFLUENCY_MODEL_ID."""
    return hf_repo_dir(model_id, cache) / "snapshots" / model(model_id)["revision"]


def targets(m: dict, cache: Path | None = None) -> list[tuple[dict, Path, Path]]:
    """(file, where it is used from, where it is stored) for each file of a model.

    Hugging Face: used from snapshots/<commit>/<path>, a link to blobs/<blob>.
    MFA: both are mfa/pretrained_models/<path>.
    """
    out = []
    for f in m["files"]:
        if m["source"] == "huggingface":
            repo = hf_repo_dir(m["id"], cache)
            out.append((f, repo / "snapshots" / m["revision"] / f["path"], repo / "blobs" / f["blob"]))
        else:
            target = (cache or paths.CACHE_DIR) / "mfa" / "pretrained_models" / f["path"]
            out.append((f, target, target))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def problems(m: dict, hashes: bool = False, cache: Path | None = None) -> list[str]:
    """What differs from the lock; empty if the model is in place.

    Hugging Face: refs/main names the pinned commit and its snapshot holds
    every file with the pinned size (with hashes, also the pinned sha256). MFA:
    every file with the pinned size (with hashes, also the sha256).
    """
    found = []
    if m["source"] == "huggingface":
        ref = hf_repo_dir(m["id"], cache) / "refs" / "main"
        current = ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
        if current != m["revision"]:
            found.append(f"refs/main is {current[:12] if current else 'missing'}, "
                         f"not the pinned {m['revision'][:12]}")
    for f, used, _ in targets(m, cache):
        if not used.is_file():
            found.append(f"{f['path']} missing")
        elif used.stat().st_size != f["size"]:
            found.append(f"{f['path']} has another size")
        elif hashes and _sha256(used) != f["sha256"]:
            found.append(f"{f['path']} has another sha256")
    return found


def _ssl_context() -> ssl.SSLContext:
    # A relocated environment's OpenSSL may look for its certificates in the
    # folder it was built in; certifi's bundle travels with the environment.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _url(m: dict, f: dict) -> str:
    if m["source"] == "huggingface":
        endpoint = os.environ.get("HF_ENDPOINT", HF_ENDPOINT).rstrip("/")
        return f"{endpoint}/{m['id']}/resolve/{m['revision']}/{urllib.parse.quote(f['path'])}"
    base = os.environ.get("CLAUSA_MFA_BASE_URL")
    return f["url"].replace(MFA_RELEASES, base.rstrip("/"), 1) if base else f["url"]


class _Progress:
    """One line per model: '  <label>: <done> of <total> MB', redrawn in place on a terminal
    and printed at every tenth of the total elsewhere (a log). report, if given, is called with
    the bytes in place at every change and with "checking" before a file's sha256 is computed
    (First start's window: runner/first_start_macos.py)."""

    def __init__(self, label: str, total: int, done: int, out, report: Callable | None = None):
        self.label, self.total, self.done, self.out = label, total, done, out
        self.report = report or (lambda done, status="running": None)
        self.tty = hasattr(out, "isatty") and out.isatty()
        self.shown, self.last = done, 0.0
        self.report(self.done)
        self.show()

    def add(self, n: int) -> None:
        self.done += n
        self.report(self.done)
        if (time.monotonic() - self.last >= 0.5) if self.tty else (self.done - self.shown >= self.total / 10):
            self.show()

    def show(self, final: bool = False) -> None:
        self.last, self.shown = time.monotonic(), self.done
        line = f"  {self.label}: {self.done / 1e6:,.0f} of {self.total / 1e6:,.0f} MB" + (", ok" if final else "")
        self.out.write(("\r" + line + ("\n" if final else "")) if self.tty else line + "\n")
        self.out.flush()


def _fetch(url: str, partial: Path, f: dict, progress: _Progress) -> None:
    """Download url into partial, continuing a partial file, and check it against the lock."""
    partial.parent.mkdir(parents=True, exist_ok=True)
    have = partial.stat().st_size if partial.exists() else 0
    if have > f["size"]:
        partial.unlink()
        progress.add(-have)
        have = 0
    if have < f["size"]:
        headers = {"User-Agent": "Clausa (+models-lock.json)"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=60, context=_ssl_context())
        except urllib.error.HTTPError as exc:
            if exc.code != 416:
                raise
            # Range not satisfiable: the server's file is shorter than the part already here,
            # so it is not the pinned file. Continuing would fail the same way every time.
            partial.unlink()
            raise DownloadError("checksum", f"{f['path']}: the server's file is shorter than the pinned one "
                                            f"(HTTP 416 at byte {have:,}); the partial download was "
                                            "deleted") from exc
        with response:
            if have and response.status != 206:  # the server sent the whole file: start again
                progress.add(-have)
                have = 0
            with partial.open("ab" if have else "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    progress.add(len(chunk))
    size = partial.stat().st_size
    if size < f["size"]:  # the connection ended early: keep what came, to continue it
        raise DownloadError("network", f"the connection ended before {f['path']} was complete "
                                       f"({size / 1e6:,.0f} of {f['size'] / 1e6:,.0f} MB)")
    if size == f["size"] and size > 1 << 28:
        progress.report(progress.done, "checking")  # a few seconds per GB without progress
    if size != f["size"] or _sha256(partial) != f["sha256"]:
        partial.unlink()
        raise DownloadError("checksum", f"{f['path']} did not match its pinned sha256 after downloading "
                                        f"(from {urllib.parse.urlsplit(url).netloc}); it was deleted")


def place(m: dict, used: Path, stored: Path) -> None:
    """Put a checked file where it is used; for Hugging Face, link the snapshot to the blob."""
    if m["source"] != "huggingface":
        return
    used.parent.mkdir(parents=True, exist_ok=True)
    if used.is_symlink() or used.exists():
        used.unlink()
    if os.name == "nt":
        # A hard link (the same file under a second name: the 8 GB are not stored twice), else a
        # copy. Not a symbolic link: Windows lets a user make one only in Developer Mode or when
        # elevated, so the cache would differ with how setup or First start was started.
        try:
            os.link(stored, used)
        except OSError:
            shutil.copy2(stored, used)
        return
    try:
        used.symlink_to(os.path.relpath(stored, used.parent))
    except OSError:  # a file system without symbolic links
        shutil.copy2(stored, used)


def finish_repo(m: dict, cache: Path | None = None) -> None:
    """Hugging Face: point refs/main at the pinned commit, as huggingface_hub writes it (no line end)."""
    if m["source"] != "huggingface":
        return
    ref = hf_repo_dir(m["id"], cache) / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(m["revision"], encoding="utf-8")
    version = (cache or paths.CACHE_DIR) / "huggingface" / "hub" / "version.txt"
    if not version.exists():
        version.write_text("1", encoding="utf-8")


def _download_model(m: dict, out, report: Callable | None = None) -> None:
    files = targets(m)
    complete = [used.is_file() and used.stat().st_size == f["size"] for f, used, _ in files]
    if all(complete):  # every file is here; only refs/main differs
        finish_repo(m)
        out.write(f"  {m['label']}: set to the pinned version {m['revision'][:12]} (nothing to download)\n")
        out.flush()
        return
    done = sum(f["size"] for (f, _, _), ok in zip(files, complete) if ok)
    for (f, _, stored), ok in zip(files, complete):
        part = stored.with_name(stored.name + ".incomplete")
        if not ok and part.exists():
            done += min(part.stat().st_size, f["size"])
    progress = _Progress(m["label"], sum(f["size"] for f in m["files"]), done, out, report)
    for (f, used, stored), ok in zip(files, complete):
        if ok:
            continue
        if stored.is_file() and stored.stat().st_size == f["size"]:
            if f["size"] > 1 << 28:
                progress.report(progress.done, "checking")
            if _sha256(stored) == f["sha256"]:
                place(m, used, stored)  # downloaded before, not yet linked
                continue
        part = stored.with_name(stored.name + ".incomplete")
        _fetch(_url(m, f), part, f, progress)
        os.replace(part, stored)
        place(m, used, stored)
    finish_repo(m)
    progress.done = progress.total
    progress.report(progress.done, "done")
    progress.show(final=True)


def download(models: list[dict], out=None, report: Callable | None = None) -> None:
    """Download what is missing or different from the lock, model by model.

    Raises DownloadError (network, disk or checksum) with a message in plain
    words; a partial file stays, so the next call continues it. report, if
    given, is called as report(model, bytes in place, status) while a model
    downloads (status "running"; "checking" while a file's sha256 is computed; "done").
    """
    out = out or sys.stdout
    todo = [m for m in models if problems(m)]
    if not todo:
        return
    missing = [(m, f) for m in todo for f, used, _ in targets(m)
               if not (used.is_file() and used.stat().st_size == f["size"])]
    left = sum(f["size"] for _, f in missing)
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.CACHE_DIR).free
    if left and free < left + SPARE:
        raise DownloadError("disk", f"not enough free disk space for the models: they need about "
                                    f"{(left + SPARE) / 1e9:.1f} GB, and there are {free / 1e9:.1f} GB")
    fetch = len({m["id"] for m, _ in missing})
    if fetch:
        out.write(f"Downloading {fetch} model{'s' if fetch > 1 else ''} ({left / 1e9:.1f} GB; how long this "
                  f"takes depends on the internet connection) into {paths.CACHE_DIR}\n")
        out.flush()
    for m in todo:
        with _plain_errors(m["label"], _url(m, m["files"][0]), out):
            _download_model(m, out, None if report is None else
                            lambda done, status="running", m=m: report(m, done, status))


@contextlib.contextmanager
def _plain_errors(label: str, url: str, out):
    """What can go wrong while downloading, as DownloadError (network, disk or checksum)."""
    try:
        yield
    except DownloadError:
        out.write("\n")
        raise
    except urllib.error.HTTPError as exc:
        out.write("\n")
        raise DownloadError("network", f"{urllib.parse.urlsplit(exc.url).netloc} answered {exc.code} "
                                       f"{exc.reason} for {label}") from exc
    except (urllib.error.URLError, socket.timeout, ConnectionError, ssl.SSLError,
            http.client.HTTPException) as exc:
        out.write("\n")
        reason = getattr(exc, "reason", exc)
        raise DownloadError("network", f"could not reach {urllib.parse.urlsplit(url).netloc} for {label}: "
                                       f"{reason}") from exc
    except OSError as exc:
        out.write("\n")
        if exc.errno == errno.ENOSPC:
            raise DownloadError("disk", f"the disk is full (while downloading {label})") from exc
        raise


def fetch_file(url: str, dest: Path, size: int, sha256: str, label: str, out=None,
               report: Callable | None = None) -> None:
    """One pinned file that is not a model (the Windows bundle's PyTorch wheels) into dest.

    As the models: resumed from dest.<name>.incomplete after an interruption, checked
    against its size and sha256 (a file that does not match is deleted), DownloadError
    as download() raises it, report(bytes in place, status) as _download_model calls it.
    A dest that is there and matches is kept. The caller checks the free space.
    """
    out = out or sys.stdout
    if dest.is_file():
        if dest.stat().st_size == size:
            if report is not None and size > 1 << 28:
                report(size, "checking")
            if _sha256(dest) == sha256:
                if report is not None:
                    report(size, "done")
                return
        dest.unlink()
    partial = dest.with_name(dest.name + ".incomplete")
    done = min(partial.stat().st_size, size) if partial.exists() else 0
    progress = _Progress(label, size, done, out, report)
    with _plain_errors(label, url, out):
        _fetch(url, partial, {"path": dest.name, "size": size, "sha256": sha256}, progress)
    os.replace(partial, dest)
    progress.done = size
    progress.report(size, "done")
    progress.show(final=True)


def ensure(language: str | None = None, only: str | None = None) -> None:
    """Download the pinned models a run needs if any is missing or different (fast check)."""
    download(needed(language, only))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runner.models", description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only report whether the pinned models are in place")
    ap.add_argument("--hashes", action="store_true", help="with --check: also compare every file's sha256 (slow)")
    ap.add_argument("--language", choices=["en", "ja"], help="only the models a run in this language uses")
    args = ap.parse_args(argv)
    models = needed(args.language)
    if args.check:
        bad = 0
        for m in models:
            found = problems(m, hashes=args.hashes)
            pin = m.get("revision", m.get("version", ""))[:12]
            print(f"  {'ok' if not found else 'NOT OK':8s} {m['label']}  {m['id']} @ {pin}"
                  + "".join(f"\n             {p}" for p in found[:5]))
            bad += bool(found)
        return 1 if bad else 0
    try:
        download(models)
    except DownloadError as exc:
        hint = {"network": "No internet connection, or the server is down? Run this again when it is back; "
                           "it continues where it stopped.",
                "disk": "Free some space, then run this again; it continues where it stopped.",
                "checksum": "The damaged file was deleted; run this again to download it again."}[exc.kind]
        print(f"\n{hint}\nDownload failed: {exc}", file=sys.stderr)
        return EXIT[exc.kind]
    print("All models are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
