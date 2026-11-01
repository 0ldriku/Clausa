"""Create the environments and download the models (`clausa setup`).

Three environments, with the pins of the paper's pipeline:
  qwen3-asr       Python 3.12: Qwen3-ASR, English clause segmentation, Stage 4
                  (on Apple Silicon also MLX, which runs Stage 1 on the Mac GPU)
  mfa             Python 3.10: Montreal Forced Aligner 3.3.9
  ja-electra310   Python 3.10: GiNZA-Electra for Japanese clause segmentation

pixi (the command line's default) installs them from pixi.lock: every package
at the version and build that lock holds. The conda route (`--conda`, the
container) creates conda environments from the pins in runner/requirements/;
the packages those pins leave open come out at their newest.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from runner import envs, models, paths

EN_SPACY_MODEL_URL = (
    "https://github.com/LCR-ADS-Lab/l2-nlp-training-spacy/raw/main/"
    "zzz_trained_models/en_ud_L1L2e_combined_trf-0.0.1.zip"
)
EN_SPACY_ARCHIVE = "en_ud_L1L2e_combined_trf-0.0.1.tar.gz"
# The paper used cu121; Blackwell GPUs (RTX 50xx) need cu128.
TORCH = {
    "cuda": (["torch==2.8.0+cu128", "torchaudio==2.8.0+cu128"], "https://download.pytorch.org/whl/cu128"),
    "cpu": (["torch==2.5.1", "torchaudio==2.5.1"], None),
}


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def requirements(name: str) -> list[str]:
    lines = (paths.REQUIREMENTS_DIR / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def create_env(name: str, python: str) -> None:
    if envs.env_exists(name, via_conda=True):
        print(f"Using existing environment: {name}")
        return
    # conda-forge only: Anaconda's default channel needs its terms accepted.
    run([envs.conda_exe(), "create", "-y", "-n", name, "--override-channels", "-c", "conda-forge",
         f"python={python}", "pip"])


def pip(env_name: str, *args: str) -> None:
    run(envs.in_env(env_name, "python", "-m", "pip", "install", *args, via_conda=True))


def upgrade_pip(env_name: str) -> None:
    pip(env_name, "--upgrade", "pip", "setuptools", "wheel")


def en_spacy_model() -> Path:
    """The English UD spaCy model ships as a project zip with the package inside."""
    paths.SPACY_CACHE.mkdir(parents=True, exist_ok=True)
    archive = paths.SPACY_CACHE / EN_SPACY_ARCHIVE
    if archive.exists():
        return archive
    zipped = paths.SPACY_CACHE / "en_ud_L1L2e_combined_trf-0.0.1.zip"
    if not zipped.exists():
        print(f"Downloading {EN_SPACY_MODEL_URL}")
        urllib.request.urlretrieve(EN_SPACY_MODEL_URL, zipped)
    with zipfile.ZipFile(zipped) as zf:
        member = next(n for n in zf.namelist() if n.replace("\\", "/").endswith(f"/dist/{EN_SPACY_ARCHIVE}"))
        with zf.open(member) as src, archive.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return archive


def setup_asr_env(torch: str) -> None:
    create_env(paths.ASR_ENV, "3.12.12")
    upgrade_pip(paths.ASR_ENV)
    specs, index = TORCH[torch]
    pip(paths.ASR_ENV, *specs, *(["--extra-index-url", index] if index else []))
    reqs = requirements("asr")
    # spacy-transformers is installed without dependency resolution, as in the
    # paper's environment, next to the newer transformers that qwen-asr needs.
    pip(paths.ASR_ENV, *[r for r in reqs if not r.lower().startswith("spacy-transformers")])
    pip(paths.ASR_ENV, "--no-deps", *[r for r in reqs if r.lower().startswith("spacy-transformers")])
    pip(paths.ASR_ENV, "--no-deps", str(en_spacy_model()))
    if is_apple_silicon():
        pip(paths.ASR_ENV, "-r", str(paths.REQUIREMENTS_DIR / "asr-macos.txt"))
    if platform.system() == "Windows":
        pip(paths.ASR_ENV, "-r", str(paths.REQUIREMENTS_DIR / "asr-windows.txt"))


# Windows: the builds the Linux and macOS environments have. With an NVIDIA driver
# present, the win-64 solver otherwise picks Kaldi's CUDA build, with MKL and the
# CUDA libraries (5.8 GB instead of 2.1 GB on the test PC); MFA aligns on the CPU.
WINDOWS_MFA = ["kaldi=*=cpu*", "libblas=*=*openblas"]


def setup_mfa_env() -> None:
    create_env(paths.MFA_ENV, "3.10.19")
    specs = [spec.replace("==", "=") for spec in requirements("mfa")]
    if platform.system() == "Windows":
        specs += WINDOWS_MFA
    run([envs.conda_exe(), "install", "-y", "-n", paths.MFA_ENV, "--override-channels", "-c", "conda-forge",
         *specs])


def setup_ja_env() -> None:
    create_env(paths.JA_ENV, "3.10.11")
    upgrade_pip(paths.JA_ENV)
    pip(paths.JA_ENV, "torch==2.10.0", "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    pip(paths.JA_ENV, "-r", str(paths.REQUIREMENTS_DIR / "ja.txt"))


def setup_easyaligner_env(torch: str) -> None:
    """Optional: only for the aligner benchmark in bench/."""
    create_env("easyaligner", "3.12.12")
    run([envs.conda_exe(), "install", "-y", "-n", "easyaligner", "--override-channels", "-c", "conda-forge",
         "ffmpeg=7.1"])
    specs, index = TORCH[torch]
    pip("easyaligner", "-r", str(paths.REQUIREMENTS_DIR / "easyaligner.txt"), *specs,
        *(["--extra-index-url", index] if index else []))


def pixi_asr_env(torch: str) -> str:
    """qwen3-asr, or qwen3-asr-cpu: the CPU build of PyTorch on Windows and Linux (pixi.toml)."""
    return paths.ASR_ENV + ("-cpu" if torch == "cpu" and platform.system() != "Darwin" else "")


class SetupError(Exception):
    """Setup cannot do what was asked; the message says what to do instead."""


def same_folder(a: Path, b: Path) -> bool:
    """Whether two paths name one folder, which need not exist yet: their nearest existing folders are
    the same (however spelt: letter case, Unicode form, \\\\?\\ prefixes, symbolic links) and the
    names below them are equal."""
    a, b = a.resolve(), b.resolve()
    below_a, below_b = [], []
    while not a.exists() and a.parent != a:
        below_a.append(os.path.normcase(a.name))
        a = a.parent
    while not b.exists() and b.parent != b:
        below_b.append(os.path.normcase(b.name))
        b = b.parent
    return below_a == below_b and a.exists() and b.exists() and os.path.samefile(a, b)


def check_pixi(pixi: str, names: tuple[str, ...]) -> None:
    """Refuse, before installing, what the runner would not find: a name that pixi.toml has no
    environment of (CLAUSA_ASR_ENV and the others name conda's), or pixi installing elsewhere
    than PIXI_ENVS_DIR (CLAUSA_PIXI_ENVS, or pixi's detached-environments)."""
    if paths.PIXI_ENVS_DIR is None:
        raise SetupError("CLAUSA_PIXI_ENVS is empty, so the runner uses no pixi environment: "
                         "unset it, or use ./clausa setup --conda")
    # pixi writes UTF-8 (paths under the user's folder), not the Windows code page.
    info = subprocess.run([pixi, "info", "--json", "--manifest-path", str(paths.PIXI_MANIFEST)],
                          stdout=subprocess.PIPE, encoding="utf-8", check=True).stdout
    prefixes = {e["name"]: Path(e["prefix"]) for e in json.loads(info)["environments_info"]}
    unknown = [name for name in names if name not in prefixes]
    if unknown:
        raise SetupError(f"pixi.toml has no environment {', '.join(unknown)}: CLAUSA_ASR_ENV, CLAUSA_MFA_ENV "
                         "and CLAUSA_JA_ENV name conda's environments (./clausa setup --conda)")
    for name in names:
        # After pixi info: with detached-environments on macOS and Linux, it links .pixi/envs there.
        if not same_folder(prefixes[name], paths.PIXI_ENVS_DIR / name):
            raise SetupError(f"pixi installs {name} in {prefixes[name]}, but the runner looks in "
                             f"{paths.PIXI_ENVS_DIR}: set CLAUSA_PIXI_ENVS={prefixes[name].parent}")


def setup_pixi(torch: str) -> None:
    """The three environments from pixi.lock, then the English spaCy model (not on PyPI)."""
    pixi = envs.pixi_exe()
    asr = pixi_asr_env(torch)
    check_pixi(pixi, (asr, paths.MFA_ENV, paths.JA_ENV))
    for name in (asr, paths.MFA_ENV, paths.JA_ENV):
        # --run-post-link-scripts: as conda does (caches of GTK and Graphviz in the MFA environment).
        run([pixi, "install", "--manifest-path", str(paths.PIXI_MANIFEST), "--locked", "--run-post-link-scripts",
             "-e", name])
    # One PyTorch build at a time: the runner takes qwen3-asr before qwen3-asr-cpu (envs.pixi_prefix).
    other = paths.ASR_ENV if asr != paths.ASR_ENV else f"{paths.ASR_ENV}-cpu"
    if (paths.PIXI_ENVS_DIR / other).exists():
        run([pixi, "clean", "--manifest-path", str(paths.PIXI_MANIFEST), "-e", other])
    # Without its dependencies, as in the conda route: they are in the lock.
    run(envs.in_env(asr, "python", "-m", "pip", "install", "--no-deps", str(en_spacy_model())),
        env=envs.env_vars(asr))
    for name in (asr, paths.MFA_ENV, paths.JA_ENV):
        envs.pixi_lock_mark(paths.PIXI_ENVS_DIR / name).write_text(envs.pixi_lock_sha256() + "\n", encoding="ascii")


def download_models() -> None:
    """Every model, at its pinned version (runner/models.py, runner/requirements/models-lock.json)."""
    models.download(models.needed())


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def default_torch() -> str:
    """CUDA where there is an NVIDIA driver (nvidia-smi) on Linux or Windows; the CPU build otherwise."""
    return "cuda" if platform.system() in ("Linux", "Windows") and shutil.which("nvidia-smi") else "cpu"


def setup(*, torch: str | None = None, models: bool = True, easyaligner: bool = False, pixi: bool = False) -> None:
    """pixi: the environments from pixi.lock; else the conda route (easyaligner only there)."""
    torch = torch or os.environ.get("CLAUSA_TORCH") or default_torch()
    if pixi:
        print(f"pixi: {envs.pixi_exe()}\nTorch build: {torch}\nCache: {paths.CACHE_DIR}")
        setup_pixi(torch)
    else:
        print(f"Conda: {envs.conda_exe()}\nTorch build: {torch}\nCache: {paths.CACHE_DIR}")
        setup_asr_env(torch)
        setup_mfa_env()
        setup_ja_env()
        if easyaligner:
            setup_easyaligner_env(torch)
        if any(envs.pixi_prefix(name) for name in (paths.ASR_ENV, paths.MFA_ENV, paths.JA_ENV)):
            print(f"\nNote: the runner uses pixi's environments in {paths.PIXI_ENVS_DIR} while they are there.")
    if models:
        download_models()
    print("\nSetup complete. Check it with: ./clausa doctor")
