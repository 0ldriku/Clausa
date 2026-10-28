#!/usr/bin/env python
"""
English Stage 2-only pipeline: pre_mfa_json -> final JSON/TextGrid.

Consumes `pre_mfa_json/` from asr_to_pre_mfa_json.py and runs the
filler-augmented MFA stage (logic in asr_mfa_en.py). --workers aligns several
files in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import asr_mfa_en as pipeline


def _collect_pre_mfa_json(pre_mfa_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        paths: list[Path] = []
        for name in requested:
            candidate = pre_mfa_dir / f"{name}.json"
            if candidate.exists():
                paths.append(candidate)
            else:
                print(f"Warning: pre_mfa_json not found: {name}")
        return paths

    return sorted(p for p in pre_mfa_dir.iterdir() if p.suffix.lower() == ".json" and p.is_file())


def _resolve_audio_path(audio_dir: Path, payload: dict, file_id: str) -> Path:
    audio_file = payload.get("audio_file")
    if audio_file:
        candidate = audio_dir / str(audio_file)
        if candidate.exists():
            return candidate

    for ext in pipeline.AUDIO_EXTS:
        candidate = audio_dir / f"{file_id}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Audio file not found for {file_id} under {audio_dir}")


def _process_one(
    pre_mfa_path_str: str,
    audio_dir_str: str,
    output_dir_str: str,
    mfa_num_jobs: int,
    mfa_temp_root_str: str,
) -> dict:
    pre_mfa_path = Path(pre_mfa_path_str)
    audio_dir = Path(audio_dir_str)
    output_dir = Path(output_dir_str)
    payload = json.loads(pre_mfa_path.read_text(encoding="utf-8"))

    file_id = str(payload.get("file") or pre_mfa_path.stem)
    transcript = str(payload.get("transcript") or "")
    duration = float(payload["duration"])
    pre_mfa_words = payload.get("pre_mfa_words") or payload.get("rough_words") or []
    if not pre_mfa_words:
        raise ValueError(f"No pre_mfa_words available for {file_id}")

    audio_path = _resolve_audio_path(audio_dir, payload, file_id)

    worker_temp_root = Path(mfa_temp_root_str) / f"worker_{os.getpid()}"
    worker_temp_root.mkdir(parents=True, exist_ok=True)

    precise_words, stats = pipeline.step2_filler_augmented_mfa(
        pre_mfa_words,
        str(audio_path),
        duration,
        mfa_num_jobs=mfa_num_jobs,
        mfa_temporary_directory=str(worker_temp_root),
    )

    tg_dir = output_dir / "textgrids"
    tg_clean_dir = output_dir / "textgrids_clean"
    json_dir = output_dir / "json"
    tg_dir.mkdir(parents=True, exist_ok=True)
    tg_clean_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    pipeline.build_final_textgrid(precise_words, duration, str(tg_dir / f"{file_id}.TextGrid"), clean=False)
    pipeline.build_final_textgrid(
        precise_words,
        duration,
        str(tg_clean_dir / f"{file_id}.TextGrid"),
        clean=True,
    )
    pipeline.save_json(precise_words, duration, transcript, str(json_dir / f"{file_id}.json"), stats=stats)

    n_mfa = sum(1 for w in precise_words if w.get("source") == "mfa")
    n_rough = sum(1 for w in precise_words if w.get("source") == "rough")
    return {
        "name": file_id,
        "n_precise_words": len(precise_words),
        "n_mfa": n_mfa,
        "n_rough_fallback": n_rough,
        "filler_injected": stats.get("filler_injected", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filler-augmented MFA from pre_mfa_json (English)")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing pre_mfa_json/")
    parser.add_argument("-a", "--audio-dir", required=True, help="Audio folder used for MFA")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--file", nargs="+", help="Process specific files by name (without extension)")
    parser.add_argument("--workers", type=int, default=1, help="Number of file-level worker processes")
    parser.add_argument("--mfa-num-jobs", type=int, default=1, help="MFA --num_jobs value per worker")
    parser.add_argument(
        "--keep-mfa-temp",
        action="store_true",
        help="Keep the internal _mfa_temporary worker folder after a successful run",
    )
    args = parser.parse_args()

    pre_mfa_dir = Path(args.input)
    if pre_mfa_dir.name != "pre_mfa_json" and (pre_mfa_dir / "pre_mfa_json").exists():
        pre_mfa_dir = pre_mfa_dir / "pre_mfa_json"

    if not pre_mfa_dir.exists():
        raise FileNotFoundError(f"pre_mfa_json folder not found: {pre_mfa_dir}")

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")

    files = _collect_pre_mfa_json(pre_mfa_dir, args.file)
    if not files:
        raise FileNotFoundError(f"No pre_mfa_json files found in: {pre_mfa_dir}")

    output_dir = Path(args.output)
    mfa_temp_root = output_dir / "_mfa_temporary"
    mfa_temp_root.mkdir(parents=True, exist_ok=True)

    print("\nPipeline: pre_mfa_json -> Filler-Augmented MFA -> TextGrid")
    print(f"Processing {len(files)} files...")
    print(f"Input:  {pre_mfa_dir}")
    print(f"Audio:  {audio_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}, MFA num_jobs per worker: {args.mfa_num_jobs}")
    print("-" * 60)

    results: list[dict] = []
    failed: list[str] = []

    if args.workers <= 1:
        for i, path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {path.stem}")
            try:
                result = _process_one(
                    str(path),
                    str(audio_dir),
                    str(output_dir),
                    args.mfa_num_jobs,
                    str(mfa_temp_root),
                )
                results.append(result)
                print(
                    f"  Done: {result['n_precise_words']} words "
                    f"({result['n_mfa']} MFA, {result['n_rough_fallback']} rough, "
                    f"{result['filler_injected']} fillers injected)"
                )
            except Exception as exc:
                print(f"  FAILED: {exc}")
                failed.append(path.stem)
    else:
        future_to_name = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for path in files:
                future = ex.submit(
                    _process_one,
                    str(path),
                    str(audio_dir),
                    str(output_dir),
                    args.mfa_num_jobs,
                    str(mfa_temp_root),
                )
                future_to_name[future] = path.stem

            done_count = 0
            for future in as_completed(future_to_name):
                done_count += 1
                name = future_to_name[future]
                print(f"\n[{done_count}/{len(files)}] {name}")
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"  Done: {result['n_precise_words']} words "
                        f"({result['n_mfa']} MFA, {result['n_rough_fallback']} rough, "
                        f"{result['filler_injected']} fillers injected)"
                    )
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    failed.append(name)

    print("\n" + "=" * 60)
    print(f"Done! {len(results)}/{len(files)} files processed")
    if results:
        total_mfa = sum(r["n_mfa"] for r in results)
        total_rough = sum(r["n_rough_fallback"] for r in results)
        total_words = sum(r["n_precise_words"] for r in results)
        total_fillers = sum(r["filler_injected"] for r in results)
        print(f"Total words: {total_words} ({total_mfa} MFA, {total_rough} rough)")
        print(f"Total fillers injected: {total_fillers}")
    if failed:
        print(f"Failed: {', '.join(sorted(failed))}")
    elif not args.keep_mfa_temp and mfa_temp_root.exists():
        shutil.rmtree(mfa_temp_root, ignore_errors=True)
        print(f"Removed temporary MFA worker folder: {mfa_temp_root}")


if __name__ == "__main__":
    main()
