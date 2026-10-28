#!/usr/bin/env python3
"""Build the portable Windows app: Clausa-<version>-windows-x64.zip and its .sha256.

Run on a Windows x64 PC from a checkout of the commit to release, with any
Python 3.9+ (standard library only), git and conda-pack:

    python packaging\\build_windows.py --out <folder> [--conda-pack <path>] [--git <path>]

The zip holds one folder, Clausa\\:

    Clausa.exe            the launcher (packaging/windows/launcher.cs, compiled here with the
                          .NET Framework's csc.exe); the first start also runs First start
    Read me first.html    the one-page guide (packaging/windows/read_me_first.html)
    envs\\                 qwen3-asr.tar, mfa.tar, ja-electra310.tar: each environment packed
                          by conda-pack as <name>\\..., trimmed (TRIM); First start unpacks it
                          and runs conda-unpack there
    cache\\                empty: First start downloads the pinned models into it
                          (runner/models.py); with --include-models it holds their files
    app\\                  the code: `git archive` of HEAD (no paper/, docs/, bench/, training/,
                          container/; no Stage 4 weights, which are not distributable), and First start's
                          Python part, packaging/windows/first_start.py
    logs\\                 empty; the launcher, First start and the app write here

Unlike the Mac build, conda-unpack cannot run here: on Windows it writes absolute
paths, so it runs at First start, in the folder the user chose; the entry
points are then made relative, so the folder can still be moved. The tars come
straight from conda-pack (no unpacking here); --verify unpacks each into a
scratch folder as First start would and checks it. See docs/development.md,
"The app zips".
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.pop("CLAUSA_ENVS_DIR", None)  # find the conda environments, not a bundle's

from runner import envs, models, paths  # noqa: E402

# What the two builds share, from the Mac build (which this one leaves unchanged).
_spec = importlib.util.spec_from_file_location("build_macos", REPO / "packaging" / "build_macos.py")
mac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mac)

WINDOWS = REPO / "packaging" / "windows"
# The PyTorch wheels First start downloads instead of the zip carrying their files (split_torch).
TORCH_LOCK = WINDOWS / "torch-lock.json"
TORCH_MANIFEST = "qwen3-asr-torch.json"  # in envs\: the wheels and the files First start takes from them
ENV_NAMES = (paths.ASR_ENV, paths.MFA_ENV, paths.JA_ENV)
APP_PATHS = (*mac.APP_PATHS, "packaging/windows/first_start.py")
AUDIO_EXTS = mac.AUDIO_EXTS
# conda-pack --exclude patterns (relative to the environment): headers, static and import
# libraries (.lib: torch's dnnl.lib alone is 2.3 GB), debug symbols and numba caches; nothing
# loads them when the pipeline runs. The Mac build trims the same kinds (include/, *.a).
TRIM = ("include/*", "Library/include/*", "Lib/site-packages/torch/include/*", "*.lib", "*.pdb",
        "*.nbi", "*.nbc", "etc/conda/test-files/*")
CSC = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe"
TAR = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"
# Paths inside the folder that are not in the zip but are made later: the partial download
# <blob>.incomplete and the code's .pyc files; and a margin. (Hugging Face writes no
# .no_exist entries offline: none in the regular install's cache after the 40 + 40 runs.)
MARGIN = 4

T0 = time.monotonic()
say = mac.say  # "[  12 s] message"


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)
    if result.returncode != 0:
        sys.exit(f"Failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-3000:]}{result.stderr[-3000:]}")
    return result


def size_of(path: Path) -> int:
    """Bytes of the files under path (or of the file)."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


# ---------------------------------------------------------------- checks


def check(args) -> tuple[dict[str, Path], list[str]]:
    """Check every input; print what will be packed. Returns (env prefixes, problems)."""
    problems, notes = [], []
    print(f"Repository: {REPO}")
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        problems.append(f"needs Windows on x64 (this is {platform.system()} {platform.machine()})")
    args.git = args.git or shutil.which("git")
    if not args.git:
        problems.append("git not found: put it on PATH or pass --git")
    for tool in (CSC, TAR):
        if not tool.exists():
            problems.append(f"{tool} not found")
    conda_pack = args.conda_pack or shutil.which("conda-pack")
    if not conda_pack or not Path(conda_pack).exists():
        problems.append("conda-pack not found: install conda-pack (0.9.2) and pass --conda-pack")
    else:
        args.conda_pack = conda_pack
        print(f"conda-pack: {run([conda_pack, '--version']).stdout.strip()} ({conda_pack})")
    if problems:
        return {}, problems

    head = run([args.git, "-C", REPO, "rev-parse", "--short", "HEAD"]).stdout.strip()
    dirty = run([args.git, "-C", REPO, "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
    print(f"Commit:     {head}" + (" (the working tree has uncommitted changes; they are NOT in the bundle)"
                                   if dirty else ""))
    audio = [f for f in run([args.git, "-C", REPO, "ls-files", "--", *APP_PATHS]).stdout.splitlines()
             if Path(f).suffix.lower() in AUDIO_EXTS]
    if audio:
        problems.append(f"audio files in the code: {audio[:5]}")
    if not paths.APP_ICON_ICO.exists():
        problems.append(f"{paths.APP_ICON_ICO} is missing (packaging/windows/make_icon.py on a Mac makes it)")
    print(f"Version:    {args.version}")

    print("\nEnvironments:")
    if args.envs_from:
        prefixes = {name: Path(args.envs_from).expanduser().resolve() / name for name in ENV_NAMES}
    else:
        prefixes = {name: envs.env_prefix(name) for name in ENV_NAMES}
    found = {}
    for name in ENV_NAMES:
        prefix = prefixes.get(name)
        if prefix is None or not (prefix / "python.exe").exists():
            problems.append(f"environment {name} not found ({prefix or 'conda does not know it'})")
            continue
        if not (prefix / "conda-meta").is_dir():
            problems.append(f"{prefix} is not a conda environment (no conda-meta\\): conda-pack needs one")
        print(f"  {name:14s} {gb(size_of(prefix)):>8s}  {prefix}")
        found[name] = prefix
    if paths.MFA_ENV in found and not (found[paths.MFA_ENV] / "Scripts" / "mfa.exe").exists():
        problems.append(f"no Scripts\\mfa.exe in {found[paths.MFA_ENV]}")
    if paths.ASR_ENV in found:
        site = found[paths.ASR_ENV] / "Lib" / "site-packages"
        if not (site / "webview").is_dir() or not (site / "clr_loader").is_dir():
            problems.append(f"pywebview (with pythonnet) is not installed in {paths.ASR_ENV}: the app window "
                            "cannot open (pip install -r runner/requirements/asr-windows.txt)")

    if args.include_models:
        cache = Path(args.cache_from).expanduser().resolve()
        print(f"\nModels, in the zip (--include-models), from {cache} (checked against the lock, sha256):")
        for m in models.lock():
            differs = models.problems(m, hashes=True, cache=cache)
            differs += [f"{f['path']}: the stored file is missing" for f, _, stored in models.targets(m, cache)
                        if not stored.is_file()]
            print(f"  {m['label']:50s} {gb(sum(f['size'] for f in m['files'])):>8s}  {'ok' if not differs else 'NOT OK'}")
            problems += [f"{m['label']} in {cache}: {d}" for d in differs[:3]]
    else:
        total = sum(f["size"] for m in models.lock() for f in m["files"])
        print(f"\nModels: not in the zip; First start downloads them ({gb(total)}, pinned in "
              f"{paths.MODELS_LOCK.relative_to(REPO)})")

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    need = sum(size_of(p) for p in found.values()) * 2 + 3 * 10**9
    free = shutil.disk_usage(out).free
    print(f"\nOutput:     {out} ({gb(free)} free; about {gb(need)} needed)")
    if free < need:
        problems.append(f"not enough free space in {out}: {gb(free)}, need about {gb(need)}")

    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(f"PROBLEM: {p}")
    return found, problems


# ---------------------------------------------------------------- steps


def pack_env(conda_pack: str, name: str, prefix: Path, tar: Path) -> list[str]:
    """conda-pack the environment into tar as <name>\\..., trimmed; returns conda-pack's warnings."""
    excludes = [x for pattern in TRIM for x in ("--exclude", pattern)]
    # --ignore-missing-files: pip upgraded conda's own pip and setuptools in the
    # environments, so files conda lists are gone; nothing the pipeline uses.
    result = run([conda_pack, "-p", prefix, "-o", tar, "--format", "tar", "--arcroot", name,
                  "--ignore-missing-files", "--force", "--quiet", *excludes])
    return [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]


def filter_unpack_records(name: str, tar: Path, drop: frozenset = frozenset()) -> tuple[int, int]:
    """Keep in conda-unpack's list only the files that hold a path it replaces; leave out `drop`.

    conda-pack lists every file pip installed (it cannot tell which hold the build path), so
    conda-unpack at First start read all of them: 36,211 of qwen3-asr's 41,271 files, 172 s
    on the test PC. It changes a file only where the file holds the record's placeholder
    (text) or the placeholder or its lower case (binary: pip's launcher .exe files), so a
    record of a file that holds neither does nothing and is left out. The tar is rewritten
    (members and their order as they were), with that list, without the members named in drop
    (the files First start takes from the PyTorch wheels: split_torch). Returns (records
    before, after).
    """
    script = f"{name}/Scripts/conda-unpack-script.py"
    with tarfile.open(tar) as t:
        source = t.extractfile(script).read().decode("utf-8")
    nl = "\r\n" if "\r\n" in source else "\n"  # conda-pack writes it in text mode: CRLF on Windows
    start = source.index("_prefix_records = [")
    end = source.index(f"{nl}]{nl}", start) + len(f"{nl}]{nl}")
    records = ast.literal_eval(source[start + len("_prefix_records = "):end].strip())
    wanted = {}
    for path, placeholder, mode in records:
        wanted.setdefault(f"{name}/" + path.replace("\\", "/"), []).append((path, placeholder, mode))
    kept = []
    out = tar.with_name(tar.name + ".filtering")
    with tarfile.open(tar) as t, tarfile.open(out, "w", format=t.format) as o:
        for m in t:
            if m.name in drop:
                continue
            if m.name == script or not m.isfile():
                if m.name != script:
                    o.addfile(m)
                continue
            data = t.extractfile(m).read()
            for record in wanted.get(m.name, ()):
                placeholder = record[1].encode("utf-8")
                if placeholder in data or (record[2] == "binary" and placeholder.lower() in data):
                    kept.append(record)
            o.addfile(m, io.BytesIO(data))
        info = t.getmember(script)
        text = source[:start] + f"_prefix_records = [{nl}" + f",{nl}".join(map(repr, kept)) + f"{nl}]{nl}" + source[end:]
        data = text.encode("utf-8")
        info.size = len(data)
        o.addfile(info, io.BytesIO(data))
    os.replace(out, tar)
    return len(records), len(kept)


def split_torch(name: str, prefix: Path, wheels_dir: Path) -> tuple[frozenset, dict]:
    """The files of the environment that First start can take from the PyTorch wheels instead.

    torch 2.8.0+cu128's files are 4.2 GB of qwen3-asr's 6.1 GB (its NVIDIA libraries); First
    start downloads the pinned wheels (TORCH_LOCK) and puts them back. A file is left out of
    the tar only if a wheel holds it byte for byte at the same place (Lib\\site-packages\\<member>)
    and TRIM does not remove it; everything else of torch's stays in the tar (the .pyc files
    pip compiled, its RECORD, INSTALLER, the launchers in Scripts\\), so that the environment
    after First start is the validated one file for file. The manifest lists the wheels and,
    for each file, its wheel member, path, size, sha256 and time (Python checks a .pyc against
    its source's time). Returns (tar member names to leave out, manifest).
    """
    lock = json.loads(TORCH_LOCK.read_text(encoding="utf-8"))
    drop, files = set(), []
    for w in lock["cuda"]:
        whl = wheels_dir / w["file"]
        models.fetch_file(w["url"], whl, w["size"], w["sha256"], f"{w['file']} (for the build)")
        with zipfile.ZipFile(whl) as z:
            for info in z.infolist():
                member = info.filename
                if info.is_dir() or member.split("/", 1)[0].endswith(".data"):
                    continue  # .data\ goes elsewhere when pip installs: stays in the tar
                rel = "Lib/site-packages/" + member
                env_file = prefix / rel
                if any(fnmatch.fnmatchcase(rel.lower(), pattern.lower()) for pattern in TRIM) \
                        or not env_file.is_file() or env_file.stat().st_size != info.file_size:
                    continue
                h = hashlib.sha256()
                with z.open(info) as src:
                    for chunk in iter(lambda: src.read(8 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != mac.sha256(env_file):
                    continue
                drop.add(f"{name}/{rel}")
                files.append({"wheel": w["file"], "member": member, "path": rel, "size": info.file_size,
                              "sha256": h.hexdigest(), "mtime": env_file.stat().st_mtime})
    manifest = {"about": "PyTorch files First start takes from the wheels (packaging/build_windows.py "
                         "split_torch, packaging/windows/first_start.py install_torch)",
                "env": name, "wheels": lock["cuda"], "cpu_wheels": lock["cpu"],
                "bytes": sum(f["size"] for f in files), "files": files}
    return frozenset(drop), manifest


def check_tar(name: str, tar: Path) -> tuple[int, str, list[str]]:
    """The tar's members: all under <name>/, conda-unpack there, nothing left that TRIM removes, no
    links (Windows' tar.exe would need a right to make symbolic links). Returns (files, the longest
    path as it will be inside the folder, problems)."""
    problems, count, longest, names = [], 0, "", set()
    with tarfile.open(tar) as t:
        for m in t:
            names.add(m.name)
            parts = PurePosixPath(m.name).parts
            if not parts or parts[0] != name or ".." in parts:
                problems.append(f"{m.name}: outside {name}/")
                continue
            if m.issym() or m.islnk():
                problems.append(f"{m.name}: a link (to {m.linkname})")
            inside = "/".join(parts[1:]).lower()
            if m.isfile():
                count += 1
                if any(fnmatch.fnmatchcase(inside, pattern.lower()) for pattern in TRIM):
                    problems.append(f"{m.name}: should have been left out (TRIM)")
            longest = max(longest, "envs\\" + m.name.replace("/", "\\"), key=len)
    for needed in (f"{name}/python.exe", f"{name}/Scripts/conda-unpack-script.py"):
        if needed not in names:
            problems.append(f"{needed} is missing")
    return count, longest, problems[:10]


def verify_env(name: str, tar: Path, scratch: Path, source: Path) -> list[str]:
    """Unpack the tar into scratch as First start does (tar.exe, conda-unpack, relative entry
    points: the same code), then check: the Python starts; no file but .pyc files and conda-meta
    still holds the source prefix; for MFA, Scripts\\mfa.exe runs."""
    sys.path.insert(0, str(WINDOWS))
    import first_start  # noqa: E402
    problems = []
    scratch.mkdir(parents=True, exist_ok=True)
    run([TAR, "-xf", tar, "-C", scratch])
    prefix = scratch / name
    system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    # MFA_ROOT_DIR: else `mfa version` makes Documents\MFA (as it did in a first build).
    env = dict(os.environ, CONDA_PREFIX=str(prefix), PYTHONNOUSERSITE="1", MFA_ROOT_DIR=str(scratch / "mfa-root"),
               PATH=os.pathsep.join(map(str, [prefix, prefix / "Library" / "bin", prefix / "Scripts",
                                              system / "System32", system])))
    for v in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(v, None)

    def call(*cmd) -> subprocess.CompletedProcess:
        return subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, errors="replace")

    result = call(prefix / "python.exe", "-B", "-s", prefix / "Scripts" / "conda-unpack-script.py")
    if result.returncode != 0:
        return [f"conda-unpack failed: {(result.stdout + result.stderr)[-500:]}"]
    first_start.relative_entry_points(prefix)
    result = call(prefix / "python.exe", "-B", "-s", "-c", "import sys, ssl, sqlite3, ctypes")
    if result.returncode != 0:
        problems.append(f"python does not start: {result.stderr.strip()[-300:]}")
    if name == paths.MFA_ENV:
        result = call(prefix / "Scripts" / "mfa.exe", "version")
        if result.returncode != 0 or not result.stdout.strip():
            problems.append(f"mfa.exe version failed: {(result.stdout + result.stderr).strip()[-300:]}")
    needles = {str(source).encode(), str(source).replace("\\", "/").encode()}
    held = []
    for dirpath, _, names in os.walk(prefix):
        if "conda-meta" in Path(dirpath).parts:
            continue
        for n in names:
            p = Path(dirpath) / n
            try:
                if n.endswith(".pyc") or p.stat().st_size > 64 << 20:
                    continue
                data = p.read_bytes()
            except OSError:
                continue
            if any(x in data for x in needles):
                held.append(str(p.relative_to(scratch)))
    if held:
        problems.append(f"{len(held)} files still hold {source}: {held[:5]}")
    return problems


def export_code(git: str, dest: Path) -> None:
    dest.mkdir(parents=True)
    archive = subprocess.Popen([git, "-C", str(REPO), "archive", "--format=tar", "HEAD", "--", *APP_PATHS],
                               stdout=subprocess.PIPE)
    subprocess.run([str(TAR), "-xf", "-", "-C", str(dest)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        sys.exit("git archive failed")


def copy_models(src: Path, dest: Path) -> None:
    """The pinned models' files only (--include-models): Hugging Face blobs and MFA files, and
    refs/main. Not the snapshots' links (a zip cannot hold them; on Windows they are hard links or
    copies): First start links each checked blob into its snapshot (runner.models.place)."""
    for m in models.lock():
        for (_, _, stored_src), (_, _, stored) in zip(models.targets(m, src), models.targets(m, dest)):
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stored_src, stored)
        models.finish_repo(m, dest)


def longest_cache_path() -> str:
    """The longest path in cache\\ once the models are in place (from the lock), and
    <blob>.incomplete while one downloads."""
    longest = ""
    cache = Path("C:/x/cache")  # only the part below cache\ counts
    for m in models.lock():
        for f, used, stored in models.targets(m, cache):
            for p in (str(used), str(stored) + ".incomplete"):
                rel = "cache\\" + str(Path(p).relative_to(cache)).replace("/", "\\")
                longest = max(longest, rel, key=len)
    return longest


def compile_launcher(root: Path, version: str, longest: int) -> None:
    """Clausa.exe from launcher.cs, with the icon and the numbers filled in."""
    numbers = re.findall(r"\d+", version)[:3]
    version4 = ".".join((numbers + ["0", "0", "0"])[:3] + ["0"])
    source = (WINDOWS / "launcher.cs").read_text(encoding="utf-8")
    source = (source.replace("{{VERSION4}}", version4).replace("{{VERSION}}", version)
                    .replace("{{LONGEST_INSIDE}}", str(longest)))
    if re.search(r"{{[A-Z_0-9]+}}", source):
        sys.exit(f"unfilled placeholder in launcher.cs: {re.findall(r'{{[A-Z_0-9]+}}', source)}")
    with tempfile.TemporaryDirectory() as tmp:
        cs = Path(tmp) / "launcher.cs"
        cs.write_text(source, encoding="utf-8")
        run([CSC, "/nologo", "/target:winexe", "/platform:anycpu", "/optimize+",
             f"/win32icon:{paths.APP_ICON_ICO}", f"/out:{root / 'Clausa.exe'}",
             "/reference:System.Windows.Forms.dll", "/reference:System.Drawing.dll", "/codepage:65001", cs])


def setup_numbers(zip_bytes: int, root: Path, models_included: bool) -> dict[str, str]:
    """The guide's sizes: download, folder afterwards, free space to have before starting.

    The folder afterwards: the environments (the tars' size), PyTorch's files from its wheels,
    the models and the code. Most disk is in use, at most, with the zip kept, that folder and
    the PyTorch wheels (downloaded, then deleted once their files are in place) at once; plus
    3 GB to spare, rounded up to 5 GB.
    """
    folder = size_of(root)  # the tars, the code (and with --include-models the models and wheels)
    torch_files = json.loads((root / "envs" / TORCH_MANIFEST).read_text(encoding="utf-8"))
    wheels = sum(w["size"] for w in torch_files["wheels"])
    models_bytes = sum(f["size"] for m in models.lock() for f in m["files"])
    download = 0 if models_included else models_bytes + wheels
    after = folder + torch_files["bytes"] + (0 if models_included else models_bytes) - (wheels if models_included else 0)
    peak = zip_bytes + after + wheels
    free = -(-(peak + 3 * 10**9) // (5 * 10**9)) * 5
    if models_included:
        models_note = "<li>Everything is included. No internet connection is needed.</li>"
        first_start = "Setting up takes a few minutes."
    else:
        models_note = (f"<li>The first start downloads <b>{download / 1e9:.1f} GB</b> once, so it needs an "
                       "internet connection.</li>")
        first_start = ("Setting up takes a few minutes, longer on a slow connection. If it stops, click "
                       "<b>Try again</b>: it continues where it stopped.")
    return {"ZIP_GB": f"{zip_bytes / 1e9:.1f}", "FOLDER_GB": f"{after / 1e9:.1f}",
            "FREE_GB": str(free), "MODELS_NOTE": models_note, "FIRST_START_NOTE": first_start}


def make_guide(dest: Path, version: str, max_root: int, numbers: dict[str, str]) -> None:
    """Read me first.html from the template; the icon is embedded (no external loads)."""
    import base64
    html = (WINDOWS / "read_me_first.html").read_text(encoding="utf-8")
    png = paths.APP_ICON_ICO.read_bytes()
    # the 256 px PNG inside the .ico (make_icon.py writes it last)
    start = png.rfind(b"\x89PNG\r\n\x1a\n")
    html = (html.replace("{{VERSION}}", version).replace("{{MAX_ROOT}}", str(max_root))
                .replace("{{ICON}}", "data:image/png;base64," + base64.b64encode(png[start:]).decode()))
    for key, value in numbers.items():
        html = html.replace("{{" + key + "}}", value)
    if "{{" in html:
        sys.exit(f"unfilled placeholder in the guide: {re.findall(r'{{[A-Z_]+}}', html)}")
    dest.write_text(html, encoding="utf-8")


def write_zip(root: Path, zip_path: Path, last: str, make_last) -> int:
    """Zip root as <root.name>/, a standard ZIP64 that Explorer's "Extract all" reads.

    Entries in the order of a sorted walk; the file `last` (relative to root) goes in
    last: make_last(bytes written so far) writes it first, so that it can give the
    zip's size. Each entry keeps its time; folders carry the MS-DOS folder flag.
    Model weights compress to about 0.8 at any level, so they get level 1; everything
    else level 6. Returns the number of entries.
    """
    def info(arc: str, st: os.stat_result) -> zipfile.ZipInfo:
        zi = zipfile.ZipInfo(arc, time.localtime(max(st.st_mtime, 315532800))[:6])
        zi.create_system = 0  # MS-DOS attributes in external_attr
        return zi

    def add_file(z: zipfile.ZipFile, path: Path, arc: str) -> None:
        st = path.stat()
        zi = info(arc, st)
        zi.file_size = st.st_size
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0x20  # archive
        level = 1 if "/blobs/" in arc or arc.endswith((".safetensors", ".bin", ".pt", ".zip")) else 6
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
            zi.external_attr = 0x10  # MS-DOS folder
            z.writestr(zi, b"")
            count += 1
            for name in sorted(names):
                if rel == Path(".") and name == last:
                    continue
                add_file(z, d / name, f"{arc_dir}/{name}")
                count += 1
        z.fp.flush()
        make_last(z.fp.tell())
        add_file(z, root / last, f"{top}/{last}")
        count += 1
    return count


def staged_extras(root: Path) -> list[str]:
    """What a staging folder holds in envs\\ beyond the three tars (a test that ran First start
    in it would have unpacked them); such a folder must not be zipped again."""
    allowed = {f"{name}.tar" for name in ENV_NAMES} | {TORCH_MANIFEST}
    return [f"envs/{n}" for n in sorted(os.listdir(root / "envs")) if n not in allowed] \
        if (root / "envs").is_dir() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="folder for the zip, its .sha256 and (while building) staging\\")
    ap.add_argument("--envs-from", help="folder holding the three conda environments as <name>\\ "
                                        "(default: the conda environments this checkout runs with)")
    ap.add_argument("--include-models", action="store_true",
                    help="put the pinned models and PyTorch's wheels into the zip (about 12 GB more), for use "
                         "without internet; by default First start downloads them")
    ap.add_argument("--cache-from", default=str(paths.CACHE_DIR),
                    help=f"with --include-models: the cache to take them from (default: {paths.CACHE_DIR})")
    ap.add_argument("--conda-pack", help="conda-pack executable (default: on PATH)")
    ap.add_argument("--git", help="git executable (default: on PATH)")
    ap.add_argument("--version", default=mac.default_version(), help="version in the zip name and the app "
                                                                     "(default: runner.__version__)")
    ap.add_argument("--keep-staging", action="store_true", help="keep the folder (<out>\\staging\\Clausa)")
    ap.add_argument("--no-zip", action="store_true", help="build the folder only (implies --keep-staging)")
    ap.add_argument("--no-verify", action="store_true", help="skip unpacking each environment to check it")
    ap.add_argument("--check", action="store_true", help="only check the inputs and print their sizes")
    ap.add_argument("--from-staging", action="store_true",
                    help="reuse the environment tars of <out>\\staging\\Clausa from an earlier build; export the "
                         "code and make cache\\, Clausa.exe and the guide again, and zip")
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
    zip_path = out / f"Clausa-{args.version}-windows-x64.zip"
    times, tar_info = {}, {}
    if args.from_staging:
        if not all((root / "envs" / f"{name}.tar").is_file() for name in ENV_NAMES):
            sys.exit(f"--from-staging: no complete earlier build in {root}")
        extra = staged_extras(root)
        if extra:
            sys.exit(f"--from-staging: {root} holds more than a build leaves there (a test ran First start "
                     f"in it?); build again without --from-staging:\n  " + "\n  ".join(extra))
        say(f"Reusing the environment tars of {root}")
        for item in ("app", "logs", "cache", "Clausa.exe", "Read me first.html"):
            p = root / item
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    else:
        if staging.exists():
            say(f"Removing the previous {staging}")
            shutil.rmtree(staging)
        (root / "envs").mkdir(parents=True)

    say("Code: git archive HEAD")
    export_code(args.git, root / "app")
    (root / "logs").mkdir(exist_ok=True)
    (root / "cache").mkdir()

    for name in ENV_NAMES:
        tar = root / "envs" / f"{name}.tar"
        if not args.from_staging:
            t = time.monotonic()
            say(f"Environment {name}: conda-pack (trimmed: {', '.join(TRIM)})")
            for line in pack_env(args.conda_pack, name, prefixes[name], tar):
                say(f"  conda-pack: {line}")
            drop = frozenset()
            if name == paths.ASR_ENV:
                drop, manifest = split_torch(name, prefixes[name], out / "wheels")
                (root / "envs" / TORCH_MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
                say(f"  PyTorch: {len(drop):,} files ({gb(manifest['bytes'])}) left to First start, from "
                    + ", ".join(w["file"] for w in manifest["wheels"]))
            before, after = filter_unpack_records(name, tar, drop)
            say(f"  conda-unpack's list: {after:,} of {before:,} files hold a path it replaces")
            times[name] = time.monotonic() - t
        count, longest, problems = check_tar(name, tar)
        if problems:
            sys.exit(f"envs\\{name}.tar:\n  " + "\n  ".join(problems))
        tar_info[name] = (count, longest)
        say(f"  {name}.tar: {gb(tar.stat().st_size)}, {count:,} files, longest path {len(longest)}: {longest}")

    if not args.no_verify:
        with tempfile.TemporaryDirectory(dir=out, prefix="verify-") as tmp:
            for name in ENV_NAMES:
                t = time.monotonic()
                say(f"Checking {name}: unpacked, conda-unpack, entry points, Python, as First start does")
                problems = verify_env(name, root / "envs" / f"{name}.tar", Path(tmp), prefixes[name])
                if problems:
                    sys.exit(f"envs\\{name}.tar:\n  " + "\n  ".join(problems))
                say(f"  ok ({time.monotonic() - t:.0f} s)")
                shutil.rmtree(Path(tmp) / name, ignore_errors=True)

    if args.include_models:
        say("Models and PyTorch wheels: cache\\ (--include-models)")
        copy_models(Path(args.cache_from).expanduser().resolve(), root / "cache")
        for w in json.loads(TORCH_LOCK.read_text(encoding="utf-8"))["cuda"]:  # the offline zip: CUDA only
            whl = out / "wheels" / w["file"]
            models.fetch_file(w["url"], whl, w["size"], w["sha256"], f"{w['file']} (for the zip)")
            (root / "cache" / "wheels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(whl, root / "cache" / "wheels" / w["file"])

    # The longest path inside the folder decides how long the folder's own path may be.
    candidates = [longest for _, longest in tar_info.values()] + [longest_cache_path()]
    if not (root / "envs" / TORCH_MANIFEST).is_file():  # a staging folder from before PyTorch was split off
        sys.exit(f"--from-staging: the staged environments have no envs\\{TORCH_MANIFEST}; build again without "
                 "--from-staging")
    torch_manifest = json.loads((root / "envs" / TORCH_MANIFEST).read_text(encoding="utf-8"))
    lock = json.loads(TORCH_LOCK.read_text(encoding="utf-8"))
    if [w["sha256"] for w in torch_manifest["wheels"]] != [w["sha256"] for w in lock["cuda"]]:
        sys.exit("--from-staging: the staged environment was split against other CUDA wheels than "
                 f"{TORCH_LOCK.name} pins; build again without --from-staging")
    if torch_manifest.get("cpu_wheels") != lock["cpu"]:  # the CPU wheels are only downloaded, never split
        torch_manifest["cpu_wheels"] = lock["cpu"]
        (root / "envs" / TORCH_MANIFEST).write_text(json.dumps(torch_manifest, indent=1) + "\n", encoding="utf-8")
    candidates += [f"envs\\{paths.ASR_ENV}\\" + f["path"].replace("/", "\\") for f in torch_manifest["files"]]
    candidates += [f"cache\\wheels\\{w['file']}.incomplete" for w in torch_manifest["wheels"]]
    for f in (root / "app").rglob("*"):
        rel = "app\\" + str(f.relative_to(root / "app"))
        if f.suffix == ".py":  # its .pyc once it has run: <dir>\__pycache__\<stem>.cpython-312.pyc
            rel = str(Path(rel).parent / "__pycache__" / f"{f.stem}.cpython-312.pyc")
        candidates.append(rel)
    longest = max(candidates, key=len)
    longest_inside = len(longest) + 1 + MARGIN  # the separator after the folder's own path
    max_root = 259 - longest_inside
    say(f"Longest path inside the folder: {len(longest)} characters ({longest}); the folder's path may have "
        f"at most {max_root}")

    say("Clausa.exe (csc.exe)")
    compile_launcher(root, args.version, longest_inside)
    stray = mac.audio_outside_envs(root)
    if stray:
        sys.exit(f"Audio files outside envs\\ (must not be shipped): {stray}")
    size = size_of(root)
    say(f"Folder: {gb(size)} in {root}")
    for name in ENV_NAMES:
        tar = root / "envs" / f"{name}.tar"
        say(f"  envs\\{name}.tar: {gb(tar.stat().st_size)}" + (f" ({times[name]:.0f} s)" if name in times else ""))
    say(f"  cache: {gb(size_of(root / 'cache'))}, app: {gb(size_of(root / 'app'))}")

    if args.no_zip:
        make_guide(root / "Read me first.html", args.version, max_root,
                   {"ZIP_GB": "?", "FOLDER_GB": "?", "FREE_GB": "?", "MODELS_NOTE": "", "FIRST_START_NOTE": ""})
        say("Not zipping (--no-zip)")
        return 0
    say(f"Zipping to {zip_path.name}")
    zip_path.unlink(missing_ok=True)

    def guide(zip_bytes: int) -> None:
        numbers = setup_numbers(zip_bytes + 100_000, root, args.include_models)  # + the guide, the directory
        say(f"  guide: download {numbers['ZIP_GB']} GB, folder afterwards {numbers['FOLDER_GB']} GB, "
            f"free space before starting {numbers['FREE_GB']} GB")
        make_guide(root / "Read me first.html", args.version, max_root, numbers)

    entries = write_zip(root, zip_path, "Read me first.html", guide)
    say(f"  {entries} entries")
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
    if bad:
        sys.exit(f"The zip does not read back: {bad}")
    digest = mac.sha256(zip_path)
    Path(f"{zip_path}.sha256").write_text(f"{digest}  {zip_path.name}\n")
    say(f"Zip: {gb(zip_path.stat().st_size)}  {zip_path}")
    say(f"SHA-256: {digest}")
    (out / "build-info.json").write_text(json.dumps({
        "zip": zip_path.name, "bytes": zip_path.stat().st_size, "sha256": digest, "entries": entries,
        "version": args.version, "longest_inside": longest_inside, "longest_path": longest, "max_root": max_root,
        "tars": {n: {"bytes": (root / "envs" / f"{n}.tar").stat().st_size, "files": tar_info[n][0],
                     "longest": tar_info[n][1]} for n in ENV_NAMES},
    }, indent=2) + "\n", encoding="utf-8")
    if not args.keep_staging:
        shutil.rmtree(staging)
    say(f"Done in {(time.monotonic() - T0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
