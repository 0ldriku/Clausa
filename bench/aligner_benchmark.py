#!/usr/bin/env python3
"""
Compare Stage 2 aligners (MFA vs easyaligner) on the same Stage 1 output.

Every stage calls the same scripts and environments as `./clausa run`; with
--paper-compat, those of paper mode (runner/run.py MODES: the frozen copy in
paper/pipeline/). The easyaligner variants exist only in pipeline/ and have no
paper-mode code, so they run from there in both modes.

In the container: container/clausa is for the CLI; run this script with
  podman run ... micromamba run -n base python /repo/bench/aligner_benchmark.py ...
or on a host where `./clausa setup --easyaligner` has been run.

  asr         Stage 1 once per language          -> <work>/<lang>/asr/pre_mfa_json
  align       Stage 2 for one variant             -> <work>/<lang>/<variant>/asr/{json,textgrids_clean}
  downstream  span clipping, clauses, Stage 4, UF -> <work>/<lang>/<variant>/{auto_textgrids_clean,auto_clauses,...}

Variants: mfa (paper aligner), ctc (easyaligner with filler placeholders and
VAD edge refinement), ctc_raw (no edge refinement), ctc_noph (no placeholders). Wall-clock times are written to
<work>/<lang>/<variant>/timing.json.

The paper's automatic TextGrids were restricted to the reference speech span
before clause segmentation, which `./clausa run` does not do. Rebuilding
TextGrids from the shipped asr_json and comparing word tiers with the shipped
auto_textgrids_clean identifies the steps: English used
span_clipping/en_blank_outside_reference_span.py (40/40 identical), Japanese
used span_clipping/ja_crop_to_reference_span.py (32/40 identical; the
other 8 differ by one repeated word each, two intervals here versus one merged
interval in the shipped files). The downstream stage repeats these steps, using
--reference-dir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))

from runner import envs, models, paths  # noqa: E402
from runner.run import MODES  # noqa: E402

ENVS = {"qwen": paths.ASR_ENV, "mfa": paths.MFA_ENV, "ja": paths.JA_ENV, "ctc": "easyaligner"}
qwen_python_cmd = envs.python_in_env


def run(cmd: list, env: dict) -> float:
    cmd = [str(c) for c in cmd]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env, cwd=str(paths.REPO_ROOT))
    return round(time.perf_counter() - t0, 2)


def mark_mode(folder: Path, paper_compat: bool) -> None:
    """Record which mode wrote a stage's output (MODE.json next to it)."""
    (folder / "MODE.json").write_text(json.dumps({"paper_compat": paper_compat}) + "\n")


def check_mode(folder: Path, paper_compat: bool, allow_mix: bool) -> None:
    """Refuse to build on output written in the other mode (e.g. new-prompt ASR under --paper-compat)."""
    f = folder / "MODE.json"
    if not f.exists() or allow_mix:
        return
    made = json.loads(f.read_text()).get("paper_compat")
    if made != paper_compat:
        sys.exit(f"{folder} was written with paper_compat={made}, this run has paper_compat={paper_compat}; "
                 "rerun that stage or pass --allow-mode-mix")


def record(path: Path, key: str, seconds: float) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = seconds
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["asr", "align", "downstream"])
    ap.add_argument("--lang", choices=["en", "ja"], required=True)
    ap.add_argument("--variant", choices=["mfa", "ctc", "ctc_raw", "ctc_noph"])
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, help="Reference TextGrids (downstream span clipping)")
    ap.add_argument("--file", nargs="+")
    ap.add_argument("--mfa-workers", type=int, default=1)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu", "mlx"], default="auto",
                    help="asr: Stage 1 device (auto: CUDA, else MLX on Apple Silicon, else CPU; paper mode: no MLX)")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--paper-compat", action="store_true",
                    help="run the pipeline used for the article (paper/pipeline/), as ./clausa run --paper-compat")
    ap.add_argument("--spans", choices=["auto", "on", "off"], default="auto",
                    help="downstream: placeholder spans in the islands (auto: the runner's language default)")
    ap.add_argument("--classifier-islands", choices=["auto", "on", "off"], default="auto",
                    help="downstream: keep the Stage 4 classifier's islands (auto: the runner's language default)")
    ap.add_argument("--no-stage4", action="store_true", help="downstream: skip Stage 4, as without its weights")
    ap.add_argument("--ja-pause-split", type=float, default=0.6, help="downstream: JA --pause-split-sec")
    ap.add_argument("--allow-mode-mix", action="store_true",
                    help="build on Stage 1/2 output written in the other mode (deliberate experiments only)")
    args = ap.parse_args()

    lang = args.lang
    models.ensure(lang)  # the stages load the pinned models offline (runner/models.py)
    mode = MODES[args.paper_compat]
    pipe = mode.pipeline_dir
    scripts = pipe / lang
    # Each command runs its environment's python directly (runner/envs.py), with that environment's process environment.
    env_of = {k: envs.stage_env(v, pipeline_dir=pipe, aliases=mode.env_aliases) for k, v in ENVS.items() if k != "ctc"}
    env = env_of["qwen"]
    asr_root = args.work / lang / "asr"
    files = args.file or sorted(p.stem for p in args.audio_dir.iterdir() if p.suffix.lower() == ".wav")

    if args.stage == "asr":
        asr_root.mkdir(parents=True, exist_ok=True)
        secs = run(qwen_python_cmd(ENVS["qwen"], scripts / "asr_to_pre_mfa_json.py",
                                   "-i", args.audio_dir, "-o", asr_root, "--device", args.device, "--file", *files), env)
        record(args.work / lang / "asr_timing.json", "stage1_seconds", secs)
        mark_mode(asr_root, args.paper_compat)
        return

    if not args.variant:
        ap.error("--variant is required for align and downstream")
    vdir = args.work / lang / args.variant
    timing = vdir / "timing.json"
    out = vdir / "asr"

    if args.stage == "align":
        check_mode(asr_root, args.paper_compat, args.allow_mode_mix)
        vdir.mkdir(parents=True, exist_ok=True)
        common = ["-i", asr_root / "pre_mfa_json", "-a", args.audio_dir, "-o", out, "--file", *files]
        if args.variant == "mfa":
            cmd = qwen_python_cmd(ENVS["qwen"], scripts / "mfa_from_pre_mfa_json.py", *common,
                                  "--workers", args.mfa_workers, "--mfa-num-jobs", 1)
            key = f"stage2_mfa_workers{args.mfa_workers}_seconds"
        else:
            cmd = qwen_python_cmd(ENVS["ctc"], paths.PIPELINE_DIR / lang / "ctc_from_pre_mfa_json.py", *common)
            if args.variant == "ctc_noph":
                cmd.append("--no-placeholders")
            if args.variant == "ctc_raw":
                cmd.append("--no-refine")
            key = "stage2_seconds"
        record(timing, key, run(cmd, env if args.variant == "mfa" else envs.stage_env(ENVS["ctc"])))
        mark_mode(out, args.paper_compat)
        return

    # downstream
    if not args.reference_dir:
        ap.error("--reference-dir is required for downstream")
    check_mode(out, args.paper_compat, args.allow_mode_mix)
    clipped = vdir / "auto_textgrids_clean"
    clauses = vdir / "auto_clauses"
    gap = vdir / "stage4_candidates"
    patched = vdir / "stage4_textgrids_fillerpatched"
    islands = vdir / "islands"
    for d in (clipped, clauses, gap, patched, islands):
        shutil.rmtree(d, ignore_errors=True)
    file_list = vdir / "file_list.json"
    file_list.write_text(json.dumps({"all_selected": files}, indent=2, ensure_ascii=False), encoding="utf-8")

    if lang == "en":
        clip = qwen_python_cmd(ENVS["qwen"], BENCH / "span_clipping" / "en_blank_outside_reference_span.py",
                               "--asr-dir", out / "textgrids_clean", "--manual-dir", args.reference_dir,
                               "--out-dir", clipped)
        clause = qwen_python_cmd(ENVS["qwen"], scripts / "textgrid_caf_segmenter.py", "-i", clipped, "-o", clauses,
                                 "--disfluency-model", models.snapshot(paths.DISFLUENCY_MODEL_ID))
        caf_env, caf_script = ENVS["qwen"], scripts / "caf_calculator_vad_classifier.py"
    else:
        clip = qwen_python_cmd(ENVS["qwen"], BENCH / "span_clipping" / "ja_crop_to_reference_span.py",
                               "--input-dir", out / "textgrids_clean", "--reference-dir", args.reference_dir,
                               "--output-dir", clipped, "--report-csv", vdir / "clip_report.csv")
        clause = qwen_python_cmd(ENVS["ja"], scripts / "ja_clause_segmenter_v102.py", "-i", clipped, "-o", clauses,
                                 "--disfluency-model", models.snapshot(paths.DISFLUENCY_MODEL_ID),
                                 "--pause-split-sec", args.ja_pause_split)
        caf_env, caf_script = ENVS["ja"], scripts / "caf_calculator_ja_gap_classifier.py"

    record(timing, "clip_seconds", run(clip, env))
    record(timing, "clause_seconds", run(clause, env_of["ja" if lang == "ja" else "qwen"]))
    if args.no_stage4:
        (gap / "per_file").mkdir(parents=True, exist_ok=True)
    else:
        record(timing, "stage4_seconds", run(qwen_python_cmd(
            ENVS["qwen"], pipe / "shared" / "postprocess_vad_filler_classifier.py",
            "--file-list-json", file_list, "--file-list-key", "all_selected",
            "--audio-dir", args.audio_dir, "--asr-json-dir", out / "json", "--model-path", paths.FILLER_MODEL,
            "--out-dir", gap, "--textgrid-dir", clauses, "--out-textgrid-dir", patched,
            "--threshold", args.threshold, "--max-gap", 3.0), env))
    # auto: the runner's choice for this language and mode (runner/run.py MODES). In
    # paper mode the islands script always uses the Stage 4 islands only.
    spans = args.spans if args.spans != "auto" else ("on" if lang in mode.placeholder_spans else "off")
    classifier = (args.classifier_islands if args.classifier_islands != "auto"
                  else ("on" if lang in mode.classifier_islands else "off"))
    record(timing, "islands_seconds", run(qwen_python_cmd(
        ENVS["qwen"], pipe / "shared" / "placeholder_islands.py",
        "--file-list-json", file_list, "--file-list-key", "all_selected",
        "--asr-json-dir", out / "json", "--stage4-dir", gap / "per_file", "--out-dir", islands,
        "--placeholder-spans", spans, "--classifier-islands", classifier), env))
    record(timing, "uf_seconds", run(qwen_python_cmd(
        caf_env, caf_script, clauses, "--candidate-dir", islands / "per_file",
        "--file-list-json", file_list, "--file-list-key", "all_selected",
        "--output", vdir / "uf_measures.csv"), env_of["ja" if caf_env == ENVS["ja"] else "qwen"]))


if __name__ == "__main__":
    main()
