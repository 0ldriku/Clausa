"""Create the conda environments and download the models (`clausa setup`).

Three environments, with the pins of the paper's pipeline:
  qwen3-asr       Python 3.12: Qwen3-ASR, English clause segmentation, Stage 4
                  (on Apple Silicon also MLX, which runs Stage 1 on the Mac GPU)
  mfa             Python 3.10: Montreal Forced Aligner 3.3.9
  ja-electra310   Python 3.10: GiNZA-Electra for Japanese clause segmentation
"""

from __future__ import annotations

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
    if envs.env_exists(name):
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


def download_models() -> None:
    """Every model, at its pinned version (runner/models.py, runner/requirements/models-lock.json)."""
    models.download(models.needed())


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def default_torch() -> str:
    """CUDA where there is an NVIDIA driver (nvidia-smi) on Linux or Windows; the CPU build otherwise."""
    return "cuda" if platform.system() in ("Linux", "Windows") and shutil.which("nvidia-smi") else "cpu"


def setup(*, torch: str | None = None, models: bool = True, easyaligner: bool = False) -> None:
    torch = torch or os.environ.get("CLAUSA_TORCH") or default_torch()
    print(f"Conda: {envs.conda_exe()}\nTorch build: {torch}\nCache: {paths.CACHE_DIR}")
    setup_asr_env(torch)
    setup_mfa_env()
    setup_ja_env()
    if easyaligner:
        setup_easyaligner_env(torch)
    if models:
        download_models()
    print("\nSetup complete. Check it with: ./clausa doctor")
