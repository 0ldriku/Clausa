"""Command line: ./clausa {setup,doctor,models,run,webui,app}."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runner import envs, paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="clausa",
        description="Automatic clause-based utterance fluency measures for L2 English and Japanese speech.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="create the environments (pixi) and download the models")
    s.add_argument("--torch", choices=["cuda", "cpu"], help="torch build (default: cuda on Linux or Windows with an NVIDIA GPU, else cpu)")
    s.add_argument("--no-models", action="store_true", help="skip model downloads (they also download on first run)")
    s.add_argument("--conda", action="store_true",
                   help="conda environments from the pins in runner/requirements/ instead of pixi.lock")
    s.add_argument("--easyaligner", action="store_true",
                   help="with --conda: also create the easyaligner environment used by bench/")

    sub.add_parser("doctor", help="check environments and models")

    m = sub.add_parser("models", help="download the pinned models that are missing (or check them)")
    m.add_argument("--check", action="store_true", help="only report whether the pinned models are in place")
    m.add_argument("--hashes", action="store_true", help="with --check: also compare every file's sha256 (slow)")
    m.add_argument("--language", choices=["en", "ja"], help="only the models a run in this language uses")

    r = sub.add_parser("run", help="process one recording or a folder of recordings")
    r.add_argument("language", choices=["en", "ja"])
    r.add_argument("input", type=Path, help="audio file or folder (.wav, .mp3, .flac, .ogg, .m4a)")
    r.add_argument("-o", "--output", type=Path, help="output folder (default: output/<language>/<input name>)")
    r.add_argument("--device", choices=["auto", "cuda", "cpu", "mlx"], default="auto",
                   help="ASR device (auto: CUDA, else MLX on Apple Silicon, else CPU)")
    r.add_argument("--mfa-workers", type=int, metavar="N",
                   help="recordings aligned at once in Stage 2 (folders; default: as many as the cores and memory allow)")
    r.add_argument("--threshold", type=float, default=0.50, help="Stage 4 filler probability threshold (Japanese; English only with --paper-compat)")
    r.add_argument("--paper-compat", action="store_true",
                   help="reproduce the pipeline used for the published results (none of the later accuracy changes or general rules)")
    r.add_argument("--window", type=float, nargs=2, metavar=("START", "END"),
                   help="analyse only this span (seconds from the start) of every recording, e.g. the "
                        "learner's task speech; words crossing an edge are dropped")
    r.add_argument("--windows", type=Path, metavar="CSV",
                   help="per-recording spans: CSV with header file_id,start,end (file_id = file name without "
                        "extension); unlisted files use --window if given, else the whole recording")

    ms = sub.add_parser("measures", help="compute the measures again from a run's TextGrids, as they are now "
                                         "(after you correct them in Praat)")
    ms.add_argument("language", choices=["en", "ja"])
    ms.add_argument("run_dir", type=Path, help="the output folder of `clausa run` (its clauses/ and islands/)")
    ms.add_argument("-o", "--output", type=Path, help="the CSV to write (default: <run_dir>/measures.csv)")

    w = sub.add_parser("webui", help="browser interface for uploading recordings")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=7860)

    sub.add_parser("app", help="the app window: add recordings, start, see the measures")

    args = ap.parse_args(argv)

    if args.command == "setup":
        if paths.ENVS_DIR is not None:
            ap.error("setup creates the environments; unset CLAUSA_ENVS_DIR (bundled environments) to use it")
        if args.easyaligner and not args.conda:
            ap.error("--easyaligner needs --conda: the easyaligner environment is not in pixi.toml")
        from runner.models import DownloadError
        from runner.setup_envs import SetupError, setup
        try:
            setup(torch=args.torch, models=not args.no_models, easyaligner=args.easyaligner, pixi=not args.conda)
        except DownloadError as exc:
            print(f"\nERROR: model download failed: {exc}\nRun ./clausa models to continue it.", file=sys.stderr)
            return 1
        except (FileNotFoundError, SetupError) as exc:  # no pixi, no conda for --conda, check_pixi
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.command == "doctor":
        from runner.doctor import doctor
        return doctor()
    if args.command == "models":
        from runner import models
        return models.main([*(["--check"] if args.check else []), *(["--hashes"] if args.hashes else []),
                            *(["--language", args.language] if args.language else [])])
    if args.command == "run":
        from runner.run import MODES, run_pipeline
        if args.device == "mlx" and not MODES[args.paper_compat].mlx:
            ap.error("--paper-compat runs the official qwen-asr code: use --device auto, cuda or cpu")
        output = args.output or Path("output") / args.language / args.input.resolve().stem
        try:
            uf_csv = run_pipeline(args.language, args.input, output, device=args.device,
                                  mfa_workers=args.mfa_workers, threshold=args.threshold,
                                  paper_compat=args.paper_compat,
                                  window=tuple(args.window) if args.window else None,
                                  windows_csv=args.windows)
        except Exception as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1
        print(f"\nDone. Measures: {uf_csv}")
        return 0
    if args.command == "measures":
        from runner.measures import main as measures
        try:
            return measures(args.language, args.run_dir.resolve(), args.output and args.output.resolve())
        except Exception as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1
    if args.command == "webui":
        import subprocess
        cmd = envs.in_env(paths.ASR_ENV, "python", "-m", "runner.web_ui", "--host", args.host, "--port", str(args.port))
        return subprocess.run(cmd, cwd=str(paths.REPO_ROOT), env=envs.env_vars(paths.ASR_ENV)).returncode
    if args.command == "app":
        # In the ASR environment, which has pywebview (runner/app.py).
        import subprocess
        cmd = envs.in_env(paths.ASR_ENV, "python", "-m", "runner.app", live=True)
        proc = subprocess.Popen(cmd, cwd=str(paths.REPO_ROOT), env=envs.env_vars(paths.ASR_ENV))
        while True:
            try:
                return proc.wait()
            except KeyboardInterrupt:
                continue  # the app got the same Ctrl+C: it stops its run, then closes
    return 1
