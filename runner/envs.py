"""The environments and the process environment each pipeline stage runs in.

Bundled mode (paths.ENVS_DIR, set by CLAUSA_ENVS_DIR): the environments are
folders inside the portable app. The regular install's are pixi's, in
paths.PIXI_ENVS_DIR (pixi_prefix); the app builders pack these. Without them,
conda's (setup's conda route, and the container, which sets CLAUSA_PIXI_ENVS
empty): conda is asked once per process where each environment is
(env_prefix). Every command then runs the environment's own executable
directly, with the folders activation would put first on PATH and
CONDA_PREFIX (env_vars): `conda run` cost 1-3 s per command, which the
pipeline runs a dozen of. Setup's conda route still installs through conda
(in_env's via_conda), and with conda's environments CLAUSA_CONDA_RUN=1 makes
every command go through `conda run` again.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from runner import paths


def conda_exe() -> str:
    """Find micromamba, mamba or conda."""
    for var in ("MAMBA_EXE", "CONDA_EXE"):
        hit = os.environ.get(var)
        if hit and Path(hit).exists():
            return hit
    for name in ("micromamba", "mamba", "conda"):
        hit = shutil.which(name)
        if hit:
            return hit
    for candidate in (
        Path.home() / "miniforge3" / "bin" / "conda",
        Path.home() / "miniconda3" / "bin" / "conda",
        Path.home() / "anaconda3" / "bin" / "conda",
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No conda found. Install Miniforge (https://conda-forge.org/download/) or micromamba.")


def pixi_exe() -> str:
    """Find pixi: on PATH, else where its installer puts it."""
    hit = os.environ.get("PIXI_EXE") or shutil.which("pixi")
    if hit and Path(hit).exists():
        return hit
    exe = "pixi.exe" if os.name == "nt" else "pixi"
    for home in filter(None, (os.environ.get("PIXI_HOME"), Path.home() / ".pixi")):
        if (Path(home) / "bin" / exe).exists():
            return str(Path(home) / "bin" / exe)
    raise FileNotFoundError("No pixi found. Install it (https://pixi.sh), or use ./clausa setup --conda")


def pixi_prefix(env_name: str) -> Path | None:
    """The pixi environment of that name, else its -cpu variant (qwen3-asr-cpu); None if neither is installed
    (or CLAUSA_PIXI_ENVS is empty)."""
    if paths.PIXI_ENVS_DIR is None:
        return None
    for name in (env_name, f"{env_name}-cpu"):
        prefix = paths.PIXI_ENVS_DIR / name
        if (prefix / "conda-meta").is_dir():
            return prefix
    return None


def pixi_lock_mark(prefix: Path) -> Path:
    """Where setup notes the sha256 of the pixi.lock it installed the environment from (doctor compares)."""
    return prefix / "conda-meta" / "clausa-pixi-lock.sha256"


def pixi_lock_sha256() -> str:
    """pixi.lock's sha256 with LF line endings, so that a checkout with CRLF (git's autocrlf on
    Windows) gives the same as any other."""
    return hashlib.sha256(paths.PIXI_LOCK.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _bin_dirs(prefix: Path) -> list[Path]:
    """The folders of an environment that conda activation puts first on PATH."""
    if os.name == "nt":
        return [prefix, prefix / "Library" / "mingw-w64" / "bin", prefix / "Library" / "usr" / "bin",
                prefix / "Library" / "bin", prefix / "Scripts", prefix / "bin"]
    return [prefix / "bin"]


def _conda_run(env_name: str, cmd: tuple, live: bool = False) -> list[str]:
    """`conda run`, which holds the command's output until it ends unless live (micromamba streams it anyway)."""
    try:
        exe = conda_exe()
    except FileNotFoundError:
        where = f" in {paths.PIXI_ENVS_DIR}" if paths.PIXI_ENVS_DIR else ""
        raise FileNotFoundError(f"environment {env_name} not found{where}, and no conda: "
                                "run ./clausa setup") from None
    stream = ["--no-capture-output"] if live and Path(exe).stem.lower() == "conda" else []
    return [exe, "run", *stream, "-n", env_name, *map(str, cmd)]


def _direct_prefix(env_name: str) -> Path | None:
    """The environment's folder when its executables are run directly (module docstring), else None."""
    if paths.ENVS_DIR is not None:
        return paths.ENVS_DIR / env_name
    if os.environ.get("CLAUSA_CONDA_RUN") == "1":
        return pixi_prefix(env_name)
    return env_prefix(env_name)


def in_env(env_name: str, *cmd: str | Path, live: bool = False, via_conda: bool = False) -> list[str]:
    """Command line that runs cmd in an environment: the environment's own
    executable; run it with env_vars(env_name).

    `conda run` instead when via_conda (setup's conda route) or when the
    environment is not found (the command then fails as conda reports, or with
    FileNotFoundError without conda).
    """
    prefix = None if via_conda and paths.ENVS_DIR is None else _direct_prefix(env_name)
    if prefix is None:
        return _conda_run(env_name, cmd, live)
    program, *args = map(str, cmd)
    if not os.path.dirname(program):  # a name such as python or mfa: the environment's own
        dirs = _bin_dirs(prefix)
        program = shutil.which(program, path=os.pathsep.join(map(str, dirs))) or str(dirs[0] / program)
    return [program, *args]


def python_in_env(env_name: str, script: Path, *args: str | Path) -> list[str]:
    """A stage script's command line; its output streams as it is printed."""
    return in_env(env_name, "python", script, *args, live=True)


def env_vars(env_name: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """Process environment for a command from in_env(env_name).

    base (default: this process's environment) with what activation sets that
    the stages use: the environment's folders first on PATH, and CONDA_PREFIX.
    The environments' activation scripts (activate.d) set nothing a stage
    reads (libxml2's catalogs and GLib's schemas in the MFA environment,
    OpenSSL's certificates on Windows; the stages work offline). For a `conda
    run` command (in_env's fallback), base as is: conda activates.
    """
    env = dict(os.environ if base is None else base)
    prefix = _direct_prefix(env_name)
    if prefix is not None:
        env["PATH"] = os.pathsep.join([*map(str, _bin_dirs(prefix)), *filter(None, [env.get("PATH")])])
        env["CONDA_PREFIX"] = str(prefix)
    return env


def env_exists(env_name: str, via_conda: bool = False) -> bool:
    """Whether the environment runs Python (via_conda: conda's environment of that name)."""
    cmd = in_env(env_name, "python", "-c", "print('ok')", via_conda=via_conda)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=None if via_conda else env_vars(env_name))
    except OSError:  # bundled mode: no such environment folder; no conda
        return False
    return result.returncode == 0


_PREFIXES: dict[str, Path] = {}


def env_prefix(env_name: str) -> Path | None:
    """The environment's folder: bundled, pixi's, else conda's (conda is asked once per
    process; None, not remembered, if it is missing)."""
    if paths.ENVS_DIR is not None:
        prefix = paths.ENVS_DIR / env_name
        return prefix if prefix.is_dir() else None
    prefix = pixi_prefix(env_name)
    if prefix is not None:
        return prefix
    if env_name not in _PREFIXES:
        try:
            cmd = _conda_run(env_name, ("python", "-c", "import sys; print(sys.prefix)"))
        except FileNotFoundError:  # no conda either
            return None
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
        if not lines:
            return None
        _PREFIXES[env_name] = Path(lines[-1].strip())
    return _PREFIXES[env_name]


def _path_without_spaces(prefix: Path) -> Path:
    """The MFA environment under a path without spaces: a symlink in the temp folder.

    MFA 3.3.9 cannot run its Kaldi programs from a folder whose path contains a
    space: utils.thirdparty_binary puts such a path in quotes and then runs it
    as a single argument, so every alignment fails and Stage 2 silently falls
    back to the rough ASR timings. The portable app may be unzipped anywhere
    (e.g. "~/Downloads/Clausa 1.0/"), so Stage 2 reaches the environment through
    <temp>/clausa-mfa-<hash of its path> instead, which is the same folder
    (outputs unchanged). Spaces elsewhere (cache, code, results) are fine.
    On Windows, where MFA fails the same way, see _windows_path_without_spaces.
    """
    link = Path(tempfile.gettempdir()) / f"clausa-mfa-{hashlib.sha1(str(prefix).encode()).hexdigest()[:10]}"
    if os.name == "nt":
        return _windows_path_without_spaces(prefix, link)
    if " " in str(link):
        return prefix
    try:
        if not (link.is_symlink() and os.readlink(link) == str(prefix)):
            if link.is_symlink():
                link.unlink()
            link.symlink_to(prefix, target_is_directory=True)
    except FileExistsError:  # made at the same moment by another stage or app
        pass
    except OSError:
        return prefix
    return link


def _short_path(path: Path | str) -> str | None:
    """Windows: the path's 8.3 short form (C:\\Users\\JOHNSM~1\\...), or None."""
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    return buf.value if 0 < n < len(buf) else None


def _windows_path_without_spaces(prefix: Path, link: Path) -> Path:
    """_path_without_spaces on Windows: the folder's 8.3 short form, else a directory junction.

    The short form (C:\\Users\\JOHNSM~1\\CLAUSA~1\\envs\\mfa) is the same folder under another name
    and needs no file of its own. Where the volume keeps no short names, a junction in the
    temp folder (or in the temp folder's short form, if its path has a space): it needs no
    special right, unlike a symbolic link. A junction is used only if this process can go
    through it: Windows refuses some tokens a junction their own user made ("untrusted mount
    point", seen with a basic user's token on the test PC), and it is removed again then.
    Failing both, the path as it is (Stage 2 then reports the recordings as not aligned).
    """
    short = _short_path(prefix)
    if short and " " not in short:
        return Path(short)
    import _winapi
    if " " in str(link):
        temp = _short_path(link.parent)
        link = Path(temp) / link.name if temp and " " not in temp else None
    if link is None:
        return prefix
    try:
        if os.path.lexists(link) and os.path.realpath(link) != os.path.realpath(prefix):
            os.rmdir(link)  # a junction to a folder that has moved: removes the junction only
        if not os.path.lexists(link):
            _winapi.CreateJunction(str(prefix), str(link))
    except FileExistsError:  # made at the same moment by another stage or app
        pass
    except OSError:
        return prefix
    try:
        os.stat(link / "python.exe")
        return link
    except OSError:
        try:
            os.rmdir(link)
        except OSError:
            pass
        return prefix


def stage_env(env_name: str | None = None, *, pipeline_dir: Path = paths.PIPELINE_DIR,
              aliases: tuple[tuple[str, str], ...] = ()) -> dict[str, str]:
    """Environment for every pipeline stage: caches, MFA location, import path.

    With env_name, for a command of that environment (env_vars: in bundled
    mode, its folders first on PATH).

    pipeline_dir: the folder the stage scripts run from, put first on
    PYTHONPATH (paper mode: paper/pipeline/; runner/run.py MODES).

    aliases: (setting, other name) pairs; each setting that is set is also
    given under the other name (Mode.env_aliases).
    """
    for d in (paths.HF_CACHE, paths.MFA_ROOT, paths.SPACY_CACHE):
        d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(paths.HF_CACHE)
    env["HF_HUB_CACHE"] = str(paths.HF_CACHE / "hub")
    # The Japanese environment's transformers 4.25 reads TRANSFORMERS_CACHE;
    # point it at the same hub cache so every environment shares one copy.
    env["TRANSFORMERS_CACHE"] = str(paths.HF_CACHE / "hub")
    env["MFA_ROOT_DIR"] = str(paths.MFA_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    if os.name == "nt":
        # Text files opened without an encoding (by the stages, MFA, spaCy) are
        # UTF-8, as on macOS and Linux, not the Windows code page.
        env["PYTHONUTF8"] = "1"
    # Unbuffered, a stage's per-file lines reach the runner as they are printed
    # (progress events, runner/run.py).
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(pipeline_dir), env.get("PYTHONPATH")]))

    # The pipeline's MFA stage calls the mfa executable of the MFA environment.
    prefix = env_prefix(paths.MFA_ENV)
    if prefix and " " in str(prefix):
        prefix = _path_without_spaces(prefix)
    if prefix and (prefix / "bin" / "mfa").exists():
        env["CLAUSA_MFA_ENV_NAME"] = paths.MFA_ENV
        env["CLAUSA_MFA_PREFIX"] = str(prefix)
        env["CLAUSA_MFA_EXE"] = str(prefix / "bin" / "mfa")
        env["CLAUSA_MFA_LIB_BIN"] = str(prefix / "bin")
    elif prefix and (os.name == "nt" or paths.ENVS_DIR is not None):
        # Windows layout (Scripts/mfa.exe, Library/bin), conda or bundled: the stage scripts look there.
        env["CLAUSA_MFA_PREFIX"] = str(prefix)
    for name, other in aliases:
        if name in env:
            env[other] = env[name]

    # Startup caches that persist across runs (none changes any output):
    # Numba-compiled librosa code (~10 s in each stage that uses librosa) and
    # CUDA kernels JIT-compiled for the GPU (~5 s on an RTX 5090).
    env.setdefault("NUMBA_CACHE_DIR", str(paths.NUMBA_CACHE))
    env.setdefault("CUDA_CACHE_PATH", str(paths.CUDA_CACHE))
    env.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))
    # The stages never fetch from the Hugging Face hub themselves: runner/models.py
    # has put the pinned versions in place before (run_pipeline), and offline the
    # hub resolves a model id through refs/main to the pinned snapshot. Online,
    # loading by id would follow the repository's newest commit.
    env["HF_HUB_OFFLINE"] = "1"  # also over a user's HF_HUB_OFFLINE=0
    return env if env_name is None else env_vars(env_name, env)
