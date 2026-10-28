"""Where everything lives.

All locations are relative to the repository, so a fresh clone runs as is.
Large downloaded files go to a cache directory (default `.cache/` in the
repository) that can be moved with CLAUSA_CACHE.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
# Paper mode (--paper-compat) runs this frozen copy instead (paper/README.md).
PAPER_DIR = REPO_ROOT / "paper"
PAPER_PIPELINE_DIR = PAPER_DIR / "pipeline"
MODELS_DIR = REPO_ROOT / "models"
REQUIREMENTS_DIR = REPO_ROOT / "runner" / "requirements"

FILLER_MODEL = Path(
    os.environ.get(
        "CLAUSA_FILLER_MODEL",
        MODELS_DIR / "filler_classifier" / "model_podcastfillers_neural_v1_full" / "model.pt",
    )
)

CACHE_DIR = Path(os.environ.get("CLAUSA_CACHE", REPO_ROOT / ".cache")).resolve()
HF_CACHE = CACHE_DIR / "huggingface"
MFA_ROOT = CACHE_DIR / "mfa"
SPACY_CACHE = CACHE_DIR / "spacy"
NUMBA_CACHE = CACHE_DIR / "numba"
CUDA_CACHE = CACHE_DIR / "cuda"

JOBS_DIR = Path(os.environ.get("CLAUSA_JOBS", REPO_ROOT / "jobs")).resolve()

# The app window (./clausa app, runner/app.py). Its settings and logs stay in the
# app's own folder. Its projects (runner/projects.py) are folders in PROJECTS_DIR
# unless the user chose another folder (remembered in APP_SETTINGS). RESULTS_DIR is
# where the app put each run before projects existed: its dated folders are listed
# as "Earlier analyses", together with those of a results folder the user had chosen.
APP_UI_DIR = REPO_ROOT / "runner" / "app_ui"
APP_ICON = APP_UI_DIR / "icon.png"  # rendered from icon.html; the .icns of the Mac bundle is made from it
APP_ICON_ICO = APP_UI_DIR / "icon.ico"  # the same for Windows: the window's and the launcher's icon
APP_SETTINGS = Path(os.environ.get("CLAUSA_SETTINGS", REPO_ROOT / "settings.json")).resolve()
LOGS_DIR = Path(os.environ.get("CLAUSA_LOGS", REPO_ROOT / "logs")).resolve()


def _documents() -> Path:
    """The user's Documents folder: on Windows the one Explorer shows, which OneDrive
    may have moved (OneDrive\\Documents); elsewhere ~/Documents."""
    if os.name == "nt":
        try:
            import ctypes
            import uuid
            folder_id = ctypes.create_string_buffer(uuid.UUID("FDD39AD0-238F-46AF-ADB4-6C85480369C7").bytes_le, 16)
            found = ctypes.c_wchar_p()  # FOLDERID_Documents above
            if ctypes.windll.shell32.SHGetKnownFolderPath(folder_id, 0, None, ctypes.byref(found)) == 0:
                try:
                    return Path(found.value)
                finally:
                    ctypes.windll.ole32.CoTaskMemFree(found)
        except (OSError, AttributeError, ValueError):
            pass
    return Path.home() / "Documents"


PROJECTS_DIR = Path(os.path.abspath(os.path.expanduser(
    os.environ.get("CLAUSA_PROJECTS", _documents() / "Clausa projects"))))
RESULTS_DIR = Path(os.path.abspath(os.path.expanduser(
    os.environ.get("CLAUSA_RESULTS", _documents() / "Clausa results"))))

# Conda environments created by `clausa setup`
ASR_ENV = os.environ.get("CLAUSA_ASR_ENV", "qwen3-asr")
MFA_ENV = os.environ.get("CLAUSA_MFA_ENV", "mfa")
JA_ENV = os.environ.get("CLAUSA_JA_ENV", "ja-electra310")
# Bundled environments (the portable app): a folder holding one environment
# prefix per name above. When set, stages run those environments' own
# executables and conda is not used (runner/envs.py).
ENVS_DIR = Path(os.environ["CLAUSA_ENVS_DIR"]).resolve() if os.environ.get("CLAUSA_ENVS_DIR") else None

# The models Clausa downloads, pinned (repository, commit, every file's size and
# sha256): runner/models.py reads this lock.
MODELS_LOCK = REQUIREMENTS_DIR / "models-lock.json"

# The disfluency detector (huggingface.co/oldriku/Clausa): downloaded and pinned like
# the other models; the clause stage loads it from its snapshot folder, models.snapshot().
DISFLUENCY_MODEL_ID = "oldriku/Clausa"
