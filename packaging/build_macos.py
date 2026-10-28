#!/usr/bin/env python3
"""Build the portable Mac app: Clausa-<version>-macos-arm64.zip and its .sha256.

Run on an Apple Silicon Mac from a checkout of the commit to release, with any
Python 3.9+ (standard library only) and conda-pack:

    python3 packaging/build_macos.py --out <folder> [--conda-pack <path>]

The zip holds one folder, Clausa/:

    Clausa.app            app bundle; its executable is packaging/macos/launcher.sh
    First start.command   run once after unzipping (packaging/macos/first_start.command)
    Read me first.html    the one-page guide (packaging/macos/read_me_first.html)
    envs/                 qwen3-asr.tar, mfa.tar, ja-electra310.tar: each environment
                          packed with conda-pack, unpacked and conda-unpacked here,
                          libncurses fixed, trimmed, then tarred (First start unpacks it)
    cache/                empty: First start downloads the pinned models into it
                          (runner/models.py); with --include-models it holds them already
    app/                  the code: `git archive` of HEAD (no paper/, docs/, bench/, training/,
                          container/; no Stage 4 weights, which are not distributable)
    logs/                 empty; the launcher, First start and the app write here

Everything that would run on the user's Mac is done here where it can be:
conda-unpack (which re-signs what it changes) and the libncurses fix need
tools a user may not have. What the user's Mac still does is in First start.
The staging folder (--keep-staging) holds the environments unpacked.
See docs/development.md, "The app zips".
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import platform
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.pop("CLAUSA_ENVS_DIR", None)  # find the conda environments, not a bundle's

from runner import envs, models, paths  # noqa: E402

MACOS = REPO / "packaging" / "macos"
ENV_NAMES = (paths.ASR_ENV, paths.MFA_ENV, paths.JA_ENV)
# The code that goes into app/ (git archive pathspecs). Not paper/ (paper mode, --paper-compat) or docs/:
# the app runs the default pipeline only; paper mode stays in the repository.
APP_PATHS = ("clausa", "runner", "pipeline", "models/README.md", "LICENSE.md", "README.md")
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
TOOLS = ("git", "tar", "ditto", "xattr", "codesign", "install_name_tool", "otool", "sips")

T0 = time.monotonic()


def say(msg: str) -> None:
    print(f"[{time.monotonic() - T0:6.0f} s] {msg}", flush=True)


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        sys.exit(f"Failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-3000:]}{result.stderr[-3000:]}")
    return result


def du(path: Path) -> int:
    """Bytes on disk (du), not following symlinks."""
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) * 1024 if out else 0


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def clean_env() -> dict[str, str]:
    """A process environment without conda or Python settings (as on a user's Mac)."""
    return {"HOME": os.environ.get("HOME", ""), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "en_US.UTF-8"}


# ---------------------------------------------------------------- checks


def default_env_prefixes() -> dict[str, Path | None]:
    return {name: envs.env_prefix(name) for name in ENV_NAMES}


def check(args) -> tuple[dict[str, Path], list[str]]:
    """Check every input; print what will be packed. Returns (env prefixes, problems)."""
    problems, notes = [], []
    print(f"Repository: {REPO}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        problems.append(f"needs macOS on Apple Silicon (this is {platform.system()} {platform.machine()})")
    for tool in TOOLS:
        if shutil.which(tool) is None:
            problems.append(f"{tool} not found" + (" (Xcode command line tools: xcode-select --install)"
                                                   if tool in ("install_name_tool", "otool") else ""))
    conda_pack = args.conda_pack or shutil.which("conda-pack")
    if not conda_pack or not Path(conda_pack).exists():
        problems.append("conda-pack not found: pip install conda-pack (in any venv) and pass --conda-pack")
    else:
        args.conda_pack = conda_pack
        print(f"conda-pack: {run([conda_pack, '--version']).stdout.strip()} ({conda_pack})")

    head = run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"]).stdout.strip()
    dirty = run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
    print(f"Commit:     {head}" + (" (the working tree has uncommitted changes; they are NOT in the bundle)" if dirty else ""))
    audio = [f for f in run(["git", "-C", REPO, "ls-files", "--", *APP_PATHS]).stdout.splitlines()
             if Path(f).suffix.lower() in AUDIO_EXTS]
    if audio:
        problems.append(f"audio files in the code: {audio[:5]}")
    print(f"Version:    {args.version}")

    print("\nEnvironments:")
    if args.envs_from:
        prefixes = {name: Path(args.envs_from).expanduser().resolve() / name for name in ENV_NAMES}
    else:
        prefixes = default_env_prefixes()
    found = {}
    for name in ENV_NAMES:
        prefix = prefixes.get(name)
        if prefix is None or not (prefix / "bin" / "python").exists():
            problems.append(f"environment {name} not found ({prefix or 'conda does not know it'})")
            continue
        if not (prefix / "conda-meta").is_dir():
            problems.append(f"{prefix} is not a conda environment (no conda-meta/): conda-pack needs one")
        print(f"  {name:14s} {gb(du(prefix)):>8s}  {prefix}")
        found[name] = prefix
    if paths.MFA_ENV in found and not (found[paths.MFA_ENV] / "bin" / "mfa").exists():
        problems.append(f"no bin/mfa in {found[paths.MFA_ENV]}")
    if paths.ASR_ENV in found and not list(found[paths.ASR_ENV].glob("lib/python3*/site-packages/webview")):
        notes.append(f"pywebview is not installed in {paths.ASR_ENV}: the app window cannot open")

    if args.include_models:
        cache = Path(args.cache_from).expanduser().resolve()
        print(f"\nModels, in the zip (--include-models), from {cache} (checked against the lock, sha256):")
        for m in models.lock():
            differs = models.problems(m, hashes=True, cache=cache)
            # copy_models copies blobs/<blob> as named in the lock: it must be there under that name.
            differs += [f"{f['path']}: blobs/{f['blob']} missing" for f, _, stored in models.targets(m, cache)
                        if not stored.is_file()]
            print(f"  {m['label']:50s} {gb(sum(f['size'] for f in m['files'])):>8s}  {'ok' if not differs else 'NOT OK'}")
            problems += [f"{m['label']} in {cache}: {d}" for d in differs[:3]]
    else:
        total = sum(f["size"] for m in models.lock() for f in m["files"])
        print(f"\nModels: not in the zip; First start downloads them ({gb(total)}, pinned in "
              f"{paths.MODELS_LOCK.relative_to(REPO)})")

    out = Path(args.out).expanduser().resolve()
    if " " in str(out):
        problems.append(f"--out must not contain spaces ({out}): the environments are unpacked there")
    out.mkdir(parents=True, exist_ok=True)
    need = sum(du(p) for p in found.values()) * 2 + 3 * 10**9
    free = shutil.disk_usage(out).free
    print(f"\nOutput:     {out} ({gb(free)} free; about {gb(need)} needed besides the cache clones)")
    if free < need:
        problems.append(f"not enough free space in {out}: {gb(free)}, need about {gb(need)}")

    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(f"PROBLEM: {p}")
    return found, problems


# ---------------------------------------------------------------- Mach-O


def _macho_minos(path: Path) -> tuple[int, int] | None:
    """The minimum macOS version an arm64 Mach-O file asks for (LC_BUILD_VERSION), or None."""
    try:
        with path.open("rb") as f:
            head = f.read(4096)
            if len(head) < 32:
                return None
            if struct.unpack("<I", head[:4])[0] == 0xFEEDFACF:
                offset = 0
            elif struct.unpack(">I", head[:4])[0] == 0xCAFEBABE:  # universal: its arm64 slice
                count = struct.unpack(">I", head[4:8])[0]
                offset = None
                for i in range(min(count, 20)):
                    cputype, _, off, _, _ = struct.unpack(">IIIII", head[8 + 20 * i:28 + 20 * i])
                    if cputype == 0x0100000C:
                        offset = off
                if offset is None:
                    return None
            else:
                return None
            f.seek(offset)
            buf = f.read(65536)
    except OSError:
        return None
    if struct.unpack("<I", buf[:4])[0] != 0xFEEDFACF:
        return None
    ncmds = struct.unpack("<I", buf[16:20])[0]
    pos, version = 32, None
    for _ in range(ncmds):
        if pos + 16 > len(buf):
            break
        cmd, size = struct.unpack("<II", buf[pos:pos + 8])
        if cmd == 0x32:  # LC_BUILD_VERSION: platform, minos
            version = struct.unpack("<I", buf[pos + 12:pos + 16])[0]
        elif cmd == 0x24:  # LC_VERSION_MIN_MACOSX: version
            version = struct.unpack("<I", buf[pos + 8:pos + 12])[0]
        pos += max(size, 8)
    return None if version is None else (version >> 16, (version >> 8) & 0xFF)


def minimum_macos(root: Path) -> tuple[tuple[int, int], list[str]]:
    """The highest minimum macOS version of the Mach-O files under root, and those files."""
    best, files = (11, 0), []
    for dirpath, _, names in os.walk(root):
        for name in names:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            v = _macho_minos(p)
            if v is None:
                continue
            if v > best:
                best, files = v, [str(p.relative_to(root))]
            elif v == best and v > (11, 0):
                files.append(str(p.relative_to(root)))
    return best, files


# ---------------------------------------------------------------- steps


def pack_env(conda_pack: str, name: str, prefix: Path, dest: Path, tmp: Path) -> None:
    """conda-pack the environment, unpack it at dest, conda-unpack it there."""
    tar = tmp / f"{name}.tar"
    # --ignore-missing-files: pip upgraded conda's own pip and setuptools in
    # ja-electra310, so files conda lists are gone; nothing the pipeline uses.
    run([conda_pack, "-p", prefix, "-o", tar, "--format", "tar", "--ignore-missing-files", "--quiet", "--force"])
    dest.mkdir(parents=True)
    run(["tar", "-xf", tar, "-C", dest])
    tar.unlink()
    # Rewrites the build prefix into the files that hold it, and re-signs the
    # Mach-O files it changes (ad hoc). Not needed again after a move: shebangs
    # are `/usr/bin/env python...`, and the stages put the environment's bin first on PATH.
    run([dest / "bin" / "python", dest / "bin" / "conda-unpack"], env=clean_env())
    (dest / "bin" / "conda-unpack").unlink()


def fix_ncurses(env: Path) -> list[str]:
    """Make libncurses(w) load libtinfo(w) next to itself instead of by absolute path.

    conda-unpack leaves this one absolute load command (to the build folder) in
    every environment; on a user's Mac MFA's sqlite3 then aborts and Stage 2
    silently falls back to rough timings for every word.
    """
    changed = []
    for lib in ("libncurses.6.dylib", "libncursesw.6.dylib"):
        f = env / "lib" / lib
        if not f.exists() or f.is_symlink():
            continue
        for line in run(["otool", "-L", f]).stdout.splitlines()[1:]:
            old = line.strip().split(" (")[0]
            if old.startswith("/") and "libtinfo" in old:
                run(["install_name_tool", "-change", old, f"@loader_path/{Path(old).name}", f])
                run(["codesign", "--force", "--sign", "-", f])
                changed.append(f"{f.relative_to(env.parent)}: {old} -> @loader_path/{Path(old).name}")
    return changed


def absolute_loads(env: Path) -> list[str]:
    """Load commands in the environment's libraries that name a path inside the build folder."""
    hits = []
    for f in list(env.glob("lib/*.dylib")) + list(env.glob("bin/*")):
        if f.is_symlink() or not f.is_file() or _macho_minos(f) is None:
            continue
        for line in run(["otool", "-L", f]).stdout.splitlines()[1:]:
            if str(env.parent.parent) in line:
                hits.append(f"{f.relative_to(env.parent)}: {line.strip()}")
    return hits


def verify_env(env: Path, source: Path) -> list[str]:
    """Check an unpacked environment against its source; returns the problems.

    - conda-unpack rewrote every file that held the source prefix (none is
      left but .pyc files);
    - the same symlinks, with the same targets, as the source, except those
      trim_env removed;
    - code signatures valid on the Pythons, MFA's Kaldi programs, torch, MLX
      and the libncurses files the builder changed;
    - the Python starts without conda (-B: writes no .pyc).
    """
    problems = []
    # .pyc files keep the source path they were compiled from (shown only in tracebacks).
    left = subprocess.run(["grep", "-rlF", "--exclude=*.pyc", str(source), str(env)],
                          capture_output=True, text=True).stdout.split()
    if left:
        problems.append(f"{len(left)} files still hold {source}: {left[:3]}")

    def links(root: Path) -> dict[str, str]:
        out = {}
        for dirpath, dirnames, names in os.walk(root):
            for name in names + dirnames:
                p = Path(dirpath) / name
                if p.is_symlink():
                    out[str(p.relative_to(root))] = os.readlink(p)
        return out

    trimmed = re.compile(r"(^include/|/torch/include/|\.a$)")
    src, dst = links(source), links(env)
    missing = [k for k in src if k not in dst and not trimmed.search(k) and not trimmed.search(src[k])]
    changed = [k for k in src if k in dst and src[k] != dst[k]]
    extra = [k for k in dst if k not in src]
    for what, items in (("missing", missing), ("with another target", changed), ("extra", extra)):
        if items:
            problems.append(f"{len(items)} symlinks {what}: {items[:3]}")

    signed = [*env.glob("bin/python3.*"), *env.glob("lib/libncurses*.dylib"), *sorted(env.glob("bin/align-*"))[:3],
              *env.glob("lib/python3*/site-packages/torch/lib/libtorch_cpu.dylib"),
              *env.glob("lib/python3*/site-packages/mlx/lib/*.dylib"),
              *env.glob("lib/python3*/site-packages/mlx/core*.so")]
    for f in signed:
        if f.is_symlink() or _macho_minos(f) is None:  # e.g. python3.12-config, a script
            continue
        result = subprocess.run(["codesign", "-v", str(f)], capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"codesign -v {f.relative_to(env.parent)}: {result.stderr.strip()}")
    result = subprocess.run([str(env / "bin" / "python"), "-B", "-c", "import sys, ssl, sqlite3"],
                            capture_output=True, text=True, env=clean_env())
    if result.returncode != 0:
        problems.append(f"python does not start: {result.stderr.strip()[-300:]}")
    return problems


def trim_env(env: Path) -> int:
    """Remove what the pipeline never uses; returns bytes freed.

    Headers, static libraries and torch's headers (tested: outputs unchanged),
    and numba caches (.nbi/.nbc, written next to librosa when NUMBA_CACHE_DIR
    was not set; keyed by path, so useless elsewhere, and they hold the
    build machine's paths). Not tests/ folders: without site-packages/**/tests
    MFA fails silently.
    """
    before = du(env)
    targets = [env / "include", *env.glob("lib/python3*/site-packages/torch/include")]
    for t in targets:
        if t.is_dir() and not t.is_symlink():
            shutil.rmtree(t)
    for dirpath, dirnames, names in os.walk(env):
        for name in names:
            if name.endswith((".a", ".nbi", ".nbc")):
                (Path(dirpath) / name).unlink()
    for dirpath, dirnames, names in os.walk(env):  # links to what was removed
        for name in names + dirnames:
            p = Path(dirpath) / name
            if p.is_symlink() and not p.exists() and (os.readlink(p).endswith(".a") or "include" in os.readlink(p)):
                p.unlink()
    return before - du(env)


def copy_models(src: Path, dest: Path) -> None:
    """The pinned models only (--include-models), placed as runner/models.py places them.

    Clones (cp -c: no extra space on APFS) of the Hugging Face blobs and MFA
    files listed in the lock, the snapshot links and refs/main. Nothing else
    from the cache: not other models or revisions, not spacy/ (the spaCy models
    are installed in the environments), numba/ (keyed by path) or MFA's
    global_config.yaml (it names the build machine's folder).
    """
    for m in models.lock():
        for (_, _, stored_src), (_, used, stored) in zip(models.targets(m, src), models.targets(m, dest)):
            stored.parent.mkdir(parents=True, exist_ok=True)
            run(["cp", "-cp", stored_src, stored])
            models.place(m, used, stored)
        models.finish_repo(m, dest)


def export_code(dest: Path) -> None:
    dest.mkdir(parents=True)
    archive = subprocess.Popen(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", "--", *APP_PATHS],
                               stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xf", "-", "-C", str(dest)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        sys.exit("git archive failed")


def make_app(dest: Path, version: str, minimum: str) -> None:
    contents = dest / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()
    short = re.search(r"\d+(\.\d+){0,2}", version)  # 1.0.0 of 1.0.0.dev0 or v1.0
    short = short.group(0) if short else "0"
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump({
            "CFBundleName": "Clausa",
            "CFBundleDisplayName": "Clausa",
            "CFBundleIdentifier": "org.clausa.app",
            "CFBundleExecutable": "Clausa",
            "CFBundleIconFile": "Clausa",
            "CFBundlePackageType": "APPL",
            "CFBundleInfoDictionaryVersion": "6.0",
            "CFBundleShortVersionString": short,
            "CFBundleVersion": short,
            "LSMinimumSystemVersion": minimum,
            "LSArchitecturePriority": ["arm64"],
            "LSApplicationCategoryType": "public.app-category.education",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Clausa: code MIT, disfluency detector CC BY-NC-SA 4.0",
        }, f)
    (contents / "PkgInfo").write_text("APPL????")
    shutil.copy2(MACOS / "launcher.sh", contents / "MacOS" / "Clausa")
    (contents / "MacOS" / "Clausa").chmod(0o755)
    shutil.copy2(MACOS / "Clausa.icns", contents / "Resources" / "Clausa.icns")


def make_guide(dest: Path, version: str, minimum: str, numbers: dict[str, str], tmp: Path) -> None:
    """Read me first.html from the template; numbers fills {{ZIP_GB}}, {{FOLDER_GB}} and {{FREE_GB}}."""
    icon = tmp / "icon_256.png"
    run(["sips", "-z", "256", "256", paths.APP_ICON, "--out", icon])
    html = (MACOS / "read_me_first.html").read_text(encoding="utf-8")
    html = (html.replace("{{VERSION}}", version).replace("{{MIN_MACOS}}", minimum)
                .replace("{{ICON}}", "data:image/png;base64," + base64.b64encode(icon.read_bytes()).decode()))
    for key, value in numbers.items():
        html = html.replace("{{" + key + "}}", value)
    if "{{" in html:
        sys.exit(f"unfilled placeholder in the guide: {re.findall(r'{{[A-Z_]+}}', html)}")
    dest.write_text(html, encoding="utf-8")


def setup_numbers(zip_bytes: int, root: Path, tars: dict[str, Path], models_included: bool) -> dict[str, str]:
    """The guide's sizes: download, folder afterwards, free space to have before starting, models.

    Most disk is in use either while First start unpacks the largest
    environment (the zip, if kept, the unzipped folder and that environment
    twice) or once it has downloaded the models (the zip, the folder and the
    models; the tars are gone by then). Plus 3 GB to spare, rounded up to 5 GB.
    """
    folder = du(root)
    download = 0 if models_included else sum(f["size"] for m in models.lock() for f in m["files"])
    peak = zip_bytes + folder + max(max(t.stat().st_size for t in tars.values()), download)
    free = -(-(peak + 3 * 10**9) // (5 * 10**9)) * 5
    if models_included:
        models_note = "<li>The speech models are included. No internet connection is needed.</li>"
        first_start = "Setting up takes about two minutes."
        unzip = "a few minutes"
    else:
        models_note = (f"<li>First start downloads <b>{download / 1e9:.1f} GB</b> of speech models once, "
                       "so it needs an internet connection.</li>")
        unzip = "a minute or less"
        first_start = ("Setting up takes a few minutes, longer on a slow connection. If it stops, click "
                       "<b>Try again</b>: it continues where it stopped.")
    return {"ZIP_GB": f"{zip_bytes / 1e9:.1f}", "FOLDER_GB": f"{(folder + download) / 1e9:.1f}",
            "FREE_GB": str(free), "MODELS_NOTE": models_note, "FIRST_START_NOTE": first_start, "UNZIP_TIME": unzip}


def audio_outside_envs(root: Path) -> list[str]:
    """Audio files outside envs/ (the environments' packages ship a few test sounds)."""
    hits = []
    for dirpath, dirnames, names in os.walk(root):
        if Path(dirpath) == root:
            dirnames[:] = [d for d in dirnames if d != "envs"]
        hits += [str(Path(dirpath, n).relative_to(root)) for n in names if Path(n).suffix.lower() in AUDIO_EXTS]
    return hits


def write_zip(root: Path, zip_path: Path, tmp: Path, last: str,
              make_last: Callable[[int, dict[str, Path]], None]) -> int:
    """Zip root as <root.name>/ with each environment as one tar (envs/<name>.tar).

    The file `last` (relative to root) goes in last: make_last(bytes written
    so far, the tars) writes it first, so that it can give the zip's size.

    Archive Utility, which a double-click uses, handles a zip of the 120,000
    files of the unpacked environments very slowly: on a quarantined download
    it went on for more than 40 minutes. As three tars the zip has about 190
    entries; First start unpacks them (tar keeps times, links, modes and
    signatures). The tars hold no extended attributes (so no quarantine
    flag either): checked after making them. Written with zipfile, a standard ZIP64: ditto writes 32-bit
    offsets past 4 GB, which Archive Utility reports as a corrupt archive.
    Each file keeps its mode (and symlinks stay symlinks) in the Unix
    attributes, and its time to the second in an extended-timestamp field.
    Model weights compress to about 0.8 at any level, so they get level 1 (2.4
    times faster); everything else level 6. Returns the number of entries.
    """
    tars = {}
    for name in ENV_NAMES:
        tars[name] = tmp / f"{name}.tar"
        # No extended attributes, ACLs, file flags or AppleDouble entries.
        run(["tar", "-cf", tars[name], "--no-xattrs", "--no-acls", "--no-fflags", "--no-mac-metadata",
             "-C", root / "envs", name])
        with tarfile.open(tars[name]) as t:
            extras = [m.name for m in t if any("xattr" in k.lower() or "acl" in k.lower() for k in m.pax_headers)]
        if extras:
            sys.exit(f"{tars[name].name} holds extended attributes: {extras[:3]}")

    def info(arc: str, st: os.stat_result) -> zipfile.ZipInfo:
        zi = zipfile.ZipInfo(arc, time.localtime(max(st.st_mtime, 315532800))[:6])
        zi.create_system = 3  # Unix: external_attr holds the mode
        zi.external_attr = (st.st_mode & 0xFFFF) << 16
        zi.extra = struct.pack("<HHBl", 0x5455, 5, 1, int(st.st_mtime))
        return zi

    def add_file(z: zipfile.ZipFile, path: Path, arc: str) -> None:
        st = path.stat()
        zi = info(arc, st)
        zi.file_size = st.st_size
        zi.compress_type = zipfile.ZIP_DEFLATED
        level = 1 if "/blobs/" in arc or arc.endswith((".safetensors", ".bin", ".pt")) else 6
        for attr in ("compress_level", "_compresslevel"):  # name since Python 3.13, before
            if hasattr(zi, attr):
                setattr(zi, attr, level)
        with path.open("rb") as src, z.open(zi, "w", force_zip64=st.st_size > 2**30) as dst:
            shutil.copyfileobj(src, dst, 8 << 20)

    count = 0
    top = root.name
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as z:
        for dirpath, dirnames, names in os.walk(root):
            d = Path(dirpath)
            rel = d.relative_to(root)
            dirnames.sort()
            arc_dir = top if rel == Path(".") else f"{top}/{rel.as_posix()}"
            zi = info(arc_dir + "/", d.stat())
            zi.external_attr |= 0x10  # MS-DOS directory flag
            z.writestr(zi, b"")
            count += 1
            if rel == Path("envs"):
                dirnames[:] = []  # the environments go in as tars
                for name, tar in tars.items():
                    add_file(z, tar, f"{top}/envs/{name}.tar")
                    count += 1
            for name in sorted(names) + [n for n in dirnames if (d / n).is_symlink()]:
                path = d / name
                arc = f"{arc_dir}/{name}"
                if rel == Path(".") and name == last:
                    continue
                if path.is_symlink():
                    zi = info(arc, path.lstat())
                    z.writestr(zi, os.readlink(path), compress_type=zipfile.ZIP_STORED)
                else:
                    add_file(z, path, arc)
                count += 1
            dirnames[:] = [n for n in dirnames if not (d / n).is_symlink()]
        z.fp.flush()
        make_last(z.fp.tell(), tars)
        add_file(z, root / last, f"{top}/{last}")
        count += 1
    for tar in tars.values():
        tar.unlink()
    return count


def staged_extras(root: Path) -> list[str]:
    """What a staging folder holds in cache/ and envs/ beyond what a build puts there.

    A build leaves the three environment folders in envs/; a run in the
    staging folder would add First start's marker there, which must not reach
    the zip. (cache/ is made again on every build.)
    """
    allowed = {"envs": set(ENV_NAMES)}
    return [f"{folder}/{name}" for folder, names in allowed.items() if (root / folder).is_dir()
            for name in sorted(os.listdir(root / folder)) if name not in names]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_version() -> str:
    try:
        from runner import __version__
        return __version__
    except ImportError:
        return subprocess.run(["git", "-C", str(REPO), "describe", "--tags", "--always"],
                              capture_output=True, text=True).stdout.strip() or "dev"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="folder for the zip, its .sha256 and (while building) staging/")
    ap.add_argument("--envs-from", help="folder holding the three conda environments as <name>/ "
                                        "(default: the conda environments this checkout runs with)")
    ap.add_argument("--include-models", action="store_true",
                    help="put the pinned models into the zip (about 8 GB more), for use without internet; "
                         "by default First start downloads them")
    ap.add_argument("--cache-from", default=str(paths.CACHE_DIR),
                    help=f"with --include-models: the cache to take them from (default: {paths.CACHE_DIR})")
    ap.add_argument("--conda-pack", help="conda-pack executable (default: on PATH)")
    ap.add_argument("--version", default=default_version(), help="version in the zip name and the app (default: runner.__version__)")
    ap.add_argument("--keep-staging", action="store_true", help="keep the unzipped folder (<out>/staging/Clausa)")
    ap.add_argument("--no-zip", action="store_true", help="build the folder only (implies --keep-staging)")
    ap.add_argument("--check", action="store_true", help="only check the inputs and print their sizes")
    ap.add_argument("--from-staging", action="store_true",
                    help="reuse the environments of <out>/staging/Clausa from an earlier build; export the "
                         "code and make cache/, Clausa.app, First start and the guide again, and zip")
    args = ap.parse_args()

    prefixes, problems = check(args)
    if problems:
        print(f"\n{len(problems)} problem(s): not building.")
        return 1
    if args.check:
        print("\nAll inputs found.")
        return 0

    out = Path(args.out).expanduser().resolve()
    staging = out / "staging"
    root = staging / "Clausa"
    zip_path = out / f"Clausa-{args.version}-macos-arm64.zip"
    times = {}
    if args.from_staging:
        if not all((root / "envs" / name / "bin" / "python").exists() for name in ENV_NAMES):
            sys.exit(f"--from-staging: no complete earlier build in {root}")
        extra = staged_extras(root)
        if extra:
            sys.exit(f"--from-staging: {root} holds more than a build leaves there (a run or test "
                     f"wrote into it?); build again without --from-staging:\n  " + "\n  ".join(extra))
        say(f"Reusing the environments of {root}")
        run(["rm", "-rf", root / "app", root / "logs", root / "cache", root / "Clausa.app",
             root / "First start.command", root / "Read me first.html"])
    else:
        if staging.exists():
            say(f"Removing the previous {staging}")
            run(["rm", "-rf", staging])
        root.mkdir(parents=True)

    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)
        say("Code: git archive HEAD")
        export_code(root / "app")
        (root / "logs").mkdir(exist_ok=True)

        for name in ([] if args.from_staging else ENV_NAMES):
            t = time.monotonic()
            say(f"Environment {name}: conda-pack, unpack, conda-unpack")
            pack_env(args.conda_pack, name, prefixes[name], root / "envs" / name, tmp)
            for line in fix_ncurses(root / "envs" / name):
                say(f"  ncurses fix: {line}")
            freed = trim_env(root / "envs" / name)
            say(f"  trimmed {gb(freed)}; now {gb(du(root / 'envs' / name))}")
            left = absolute_loads(root / "envs" / name)
            if left:
                sys.exit("Libraries still load from the build folder:\n  " + "\n  ".join(left))
            times[name] = time.monotonic() - t

        (root / "cache").mkdir()
        if args.include_models:
            say("Models: cache/ (--include-models)")
            copy_models(Path(args.cache_from).expanduser().resolve(), root / "cache")

        say("Minimum macOS of the environments")
        (major, minor), newest = minimum_macos(root / "envs")
        minimum = f"{major}.{minor}"
        say(f"  macOS {minimum} (" + ", ".join(newest[:3]) + (", ..." if len(newest) > 3 else "") + ")")

        say("Clausa.app, First start, Read me first")
        make_app(root / "Clausa.app", args.version, minimum)
        (root / "First start.command").write_text(
            (MACOS / "first_start.command").read_text(encoding="utf-8").replace("{{MIN_MACOS}}", minimum)
            .replace("{{VERSION}}", args.version),
            encoding="utf-8")
        (root / "First start.command").chmod(0o755)
        if args.no_zip:  # the sizes come from the zip
            make_guide(root / "Read me first.html", args.version, minimum,
                       {"ZIP_GB": "?", "FOLDER_GB": "?", "FREE_GB": "?", "MODELS_NOTE": "", "FIRST_START_NOTE": "",
                        "UNZIP_TIME": "a minute or two"},
                       tmp)

    stray = audio_outside_envs(root)
    if stray:
        sys.exit(f"Audio files outside envs/ (must not be shipped): {stray}")
    say("Checking the environments")
    for name in ENV_NAMES:
        problems = verify_env(root / "envs" / name, prefixes[name])
        if problems:
            sys.exit(f"envs/{name}:\n  " + "\n  ".join(problems))
    for ds in root.rglob(".DS_Store"):
        ds.unlink()
    size = du(root)
    say(f"Folder: {gb(size)} in {root}")
    for name in ENV_NAMES:
        say(f"  envs/{name}: {gb(du(root / 'envs' / name))}" + (f" ({times[name]:.0f} s)" if name in times else ""))
    say(f"  cache: {gb(du(root / 'cache'))}, app: {gb(du(root / 'app'))}")

    if args.no_zip:
        say("Not zipping (--no-zip)")
        return 0
    say(f"Zipping to {zip_path.name} (the environments as tars)")
    zip_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp = Path(tmp)

        def guide(zip_bytes: int, tars: dict[str, Path]) -> None:
            numbers = setup_numbers(zip_bytes + 100_000, root, tars, args.include_models)  # + the guide, the directory
            say(f"  guide: download {numbers['ZIP_GB']} GB, folder afterwards {numbers['FOLDER_GB']} GB, "
                f"free space before starting {numbers['FREE_GB']} GB")
            make_guide(root / "Read me first.html", args.version, minimum, numbers, tmp)

        entries = write_zip(root, zip_path, tmp, "Read me first.html", guide)
    say(f"  {entries} entries")
    digest = sha256(zip_path)
    Path(f"{zip_path}.sha256").write_text(f"{digest}  {zip_path.name}\n")
    say(f"Zip: {gb(zip_path.stat().st_size)}  {zip_path}")
    say(f"SHA-256: {digest}")
    if not args.keep_staging:
        run(["rm", "-rf", staging])
    say(f"Done in {(time.monotonic() - T0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
