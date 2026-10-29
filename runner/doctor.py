"""Check that everything the pipeline needs is in place (`clausa doctor`)."""

from __future__ import annotations

import ast
import hashlib
import platform
import re
import subprocess

from runner import __version__, envs, models, paths

CHECKS = [
    (paths.ASR_ENV, "import torch, qwen_asr, spacy; spacy.load('en_ud_L1L2e_combined_trf'); "
                    "print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"),
    (paths.MFA_ENV, "import subprocess; print('MFA', subprocess.run(['mfa', 'version'], capture_output=True, text=True, check=True).stdout.strip())"),
    (paths.JA_ENV, "import spacy; spacy.load('ja_ginza_electra'); print('GiNZA-Electra loads')"),
]
if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Stage 1 runs through MLX on the Mac GPU (pipeline/shared/asr_mlx.py).
    CHECKS.insert(1, (paths.ASR_ENV, "import mlx.core as mx, mlx_qwen3_asr; "
                                     "print('MLX', mx.__version__, '| mlx-qwen3-asr', mlx_qwen3_asr.__version__, "
                                     "'| GPU', mx.metal.is_available())"))
    # The app window (./clausa app) runs in the ASR environment.
    CHECKS.insert(2, (paths.ASR_ENV, "import importlib.metadata, webview; "
                                     "print('pywebview', importlib.metadata.version('pywebview'), '(app window)')"))


def paper_changes() -> list[str]:
    """Files of paper/pipeline/ that differ from paper/SHA256SUMS, or are missing or not listed.

    Every file counts except bytecode caches (__pycache__/, .pyc) and Finder's
    .DS_Store. Line ends are compared as LF, so a checkout with CRLF line ends passes.
    """
    listed = {}
    for line in (paths.PAPER_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            listed[name.lstrip("*")] = digest
    found = {p.relative_to(paths.PAPER_DIR).as_posix(): p for p in paths.PAPER_PIPELINE_DIR.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store"}
    changed = [f"{name} (not listed)" for name in sorted(set(found) - set(listed))]
    for name, digest in sorted(listed.items()):
        if name not in found:
            changed.append(f"{name} (missing)")
        elif hashlib.sha256(found[name].read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
            changed.append(name)
    return changed


def paper_switches() -> list[str]:
    """Places outside the runner's mode table that depend on paper mode.

    pipeline/ must not mention it at all: paper mode is the frozen copy in
    paper/pipeline/. runner/ may pass paper_compat on (a keyword argument),
    record it (a dict value) and look up its mode (MODES[paper_compat]), but
    not branch on it: every difference belongs in runner/run.py MODES.
    """
    places = []
    for path in sorted(paths.PIPELINE_DIR.rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"paper[_-]?compat", line, re.IGNORECASE):
                places.append(f"{path.relative_to(paths.REPO_ROOT)}:{i}")
    for path in sorted((paths.REPO_ROOT / "runner").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "paper_compat":
                allowed.add(id(node.value))
            elif isinstance(node, ast.Dict):
                allowed.update(id(v) for k, v in zip(node.keys, node.values)
                               if isinstance(k, ast.Constant) and k.value == "paper_compat")
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "MODES":
                index = node.slice
                allowed.add(id(index.value if type(index).__name__ == "Index" else index))  # Index: Python 3.8
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name == "paper_compat" and isinstance(node.ctx, ast.Load) and id(node) not in allowed:
                places.append(f"{path.relative_to(paths.REPO_ROOT)}:{node.lineno}")
    return places


def doctor() -> int:
    problems = 0

    def report(ok: bool, what: str, fix: str = "", bad: str = "MISSING") -> None:
        nonlocal problems
        print(f"  {'ok' if ok else bad:8s} {what}" + ("" if ok else f"\n           -> {fix}"))
        problems += not ok

    print(f"Clausa {__version__}\nRepository: {paths.REPO_ROOT}\nCache:      {paths.CACHE_DIR}\n")
    if paths.ENVS_DIR is not None:
        print(f"Bundled environments (no conda): {paths.ENVS_DIR}\n")
    else:
        try:
            print(f"Conda: {envs.conda_exe()}\n")
        except FileNotFoundError as exc:
            print(f"MISSING  conda\n           -> {exc}")
            return 1

    print("Environments")
    for env_name, code in CHECKS:
        fix = ("./clausa setup" if paths.ENVS_DIR is None
               else f"environment folder missing or broken: {paths.ENVS_DIR / env_name}")
        try:
            # As the stages run: the cache's models, offline (HF_HUB_OFFLINE), so that
            # GiNZA-Electra's transformer comes from the pinned snapshot, not the hub.
            result = subprocess.run(envs.in_env(env_name, "python", "-c", code), capture_output=True, text=True,
                                    env=envs.stage_env(env_name))
        except OSError:  # bundled mode: no such environment folder
            report(False, env_name, fix)
            continue
        detail = result.stdout.strip().splitlines()[-1] if result.returncode == 0 and result.stdout.strip() else ""
        report(result.returncode == 0, f"{env_name}  {detail}", fix)

    print("\nModels")
    if paths.FILLER_MODEL.exists():
        print(f"  ok       filler classifier (Stage 4)  {paths.FILLER_MODEL}")
    else:
        print("  optional filler classifier (Stage 4) not found; Stage 4 will be skipped "
              "(English does not use it; see models/README.md)")

    # Downloaded models (runner/models.py): Hugging Face by refs/main and the
    # pinned snapshot's files and sizes (hashing 8 GB would take a while; that
    # is ./clausa models --check --hashes), MFA by sha256 (0.2 GB).
    for m in models.lock():
        found = models.problems(m, hashes=m["source"] == "mfa")
        pin = m.get("revision", m.get("version", ""))[:12]
        if not found:
            print(f"  ok       {m['label']}  pinned {pin}")
        elif all(p.endswith(" missing") or p.startswith("refs/main is missing") for p in found):
            print(f"  later    {m['label']}  not downloaded yet (fetched on the first run that needs it, "
                  "or now with ./clausa models)")
        else:
            report(False, f"{m['label']}  not the pinned {pin}: {'; '.join(found[:3])}",
                   "./clausa models (downloads the pinned version)", bad="OTHER")

    print("\nCode")
    if paths.PAPER_PIPELINE_DIR.is_dir():
        changed = paper_changes()
        report(not changed, f"paper/pipeline/ (paper mode) unchanged  {', '.join(changed)}".rstrip(),
               "paper/ is frozen (paper/README.md): undo the change, or update paper/SHA256SUMS only for a "
               "change the article's results share", bad="CHANGED")
    else:  # the app zips leave paper/ out
        print("  absent   paper/ (paper mode, --paper-compat): not part of the app; it is in the repository")
    places = paper_switches()
    report(not places, f"no paper-mode switches in pipeline/ or runner/  {', '.join(places)}".rstrip(),
           "paper mode is paper/pipeline/ plus runner/run.py MODES; move the difference there",
           bad="PROBLEM")

    print("\nAll good." if problems == 0 else f"\n{problems} item(s) missing or wrong.")
    return 0 if problems == 0 else 1
