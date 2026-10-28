#!/usr/bin/env python
"""
English Stage 1-only pipeline: audio -> pre_mfa_json.

Runs the Qwen3-ASR stage only and writes the rough word timings that
mfa_from_pre_mfa_json.py aligns. The logic lives in asr_mfa_en.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import asr_mfa_en as pipeline


def _collect_audio_files(input_dir: str, requested: list[str] | None) -> list[str]:
    if requested:
        audio_files: list[str] = []
        for name in requested:
            found = False
            for ext in pipeline.AUDIO_EXTS:
                candidate = f"{name}{ext}"
                if os.path.exists(os.path.join(input_dir, candidate)):
                    audio_files.append(candidate)
                    found = True
                    break
            if not found:
                print(f"Warning: File not found: {name}")
        return audio_files

    return sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(pipeline.AUDIO_EXTS)
    )


def _write_pre_mfa_json(
    path: Path,
    *,
    file_id: str,
    audio_file: str,
    duration: float,
    transcript: str,
    rough_words: list[dict],
) -> None:
    payload = {
        "file": file_id,
        "audio_file": audio_file,
        "duration": duration,
        "transcript": transcript,
        "rough_words": rough_words,
        # Stage-2 input is identical to rough words for EN.
        "pre_mfa_words": rough_words,
        "pre_mfa_words_kind": "rough_words",
        "stage1": {
            "language": "English",
            "n_rough_words": len(rough_words),
        },
    }
    # Pinned so a changed model or prompt shows up in the output.
    payload["stage1"].update({
        "asr_model": pipeline.ASR_MODEL_PATH,
        "forced_aligner": pipeline.FORCED_ALIGNER_PATH,
        "asr_backend": pipeline.ASR_BACKEND,
        "filler_context_sha256": hashlib.sha256(pipeline.FILLER_CONTEXT.encode("utf-8")).hexdigest(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_file(audio_path: str, asr_model, output_dir: str, language: str = "English") -> dict | None:
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    audio_file = os.path.basename(audio_path)

    print("  Step 1: ASR + rough alignment...")
    rough_words, duration, transcript = pipeline.step1_transcribe(audio_path, asr_model, language)
    print(f"    Transcript: {len(transcript)} chars, {len(rough_words)} words")

    if not rough_words:
        print("    No words detected, skipping.")
        return None

    pre_mfa_dir = Path(output_dir) / "pre_mfa_json"
    _write_pre_mfa_json(
        pre_mfa_dir / f"{basename}.json",
        file_id=basename,
        audio_file=audio_file,
        duration=duration,
        transcript=transcript,
        rough_words=rough_words,
    )
    return {
        "name": basename,
        "duration": duration,
        "n_rough_words": len(rough_words),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR only -> pre_mfa_json (English)")
    parser.add_argument("-i", "--input", required=True, help="Input folder with audio files")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--lang", "-l", default="English", help="Language (default: English)")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mlx"],
        default="auto",
        help="ASR device: auto (CUDA, else MLX on Apple Silicon, else CPU), cuda, cpu or mlx",
    )
    parser.add_argument("--file", nargs="+", help="Process specific files by name (without extension)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        sys.exit(1)

    audio_files = _collect_audio_files(args.input, args.file)
    if not audio_files:
        print("Error: No files found")
        sys.exit(1)

    asr_model = pipeline.load_models(args.device)

    print("\nPipeline: Qwen3 ASR -> pre_mfa_json")
    print(f"Processing {len(audio_files)} files...")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("-" * 60)

    results: list[dict] = []
    failed: list[str] = []
    for i, audio_file in enumerate(audio_files, 1):
        audio_path = os.path.join(args.input, audio_file)
        basename = os.path.splitext(audio_file)[0]
        print(f"\n[{i}/{len(audio_files)}] {basename}")
        try:
            result = process_file(audio_path, asr_model, args.output, args.lang)
            if result:
                results.append(result)
                print(f"  Done: {result['n_rough_words']} rough words")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback
            traceback.print_exc()
            failed.append(basename)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"Done! {len(results)}/{len(audio_files)} files processed")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
