#!/usr/bin/env python
"""
Decode audio that libsndfile cannot read (e.g. .m4a/AAC) to WAV for the pipeline.

The stages read audio with soundfile/librosa (libsndfile), which handles wav,
mp3, flac and ogg but not AAC. This writes <out-dir>/<id>.wav (16-bit PCM,
original sample rate and channels) for every input that needs decoding, using
PyAV (bundled FFmpeg), and a relative symlink for every other input, so that
<out-dir> can serve as the audio folder of the whole run.

Usage (the runner passes these):
  python convert_audio.py --in-dir <audio folder> --out-dir <output>/audio --file <id> ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf

DECODE_EXTS = {".m4a"}
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def decode_to_wav(src: Path, dst: Path) -> float:
    import av  # only needed here

    with av.open(str(src)) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        chunks = []
        for frame in container.decode(stream):
            a = frame.to_ndarray()  # (channels, samples), planar or packed
            if frame.format.is_planar:
                chunks.append(a)
            else:
                chunks.append(a.reshape(-1, len(frame.layout.channels)).T)
    data = np.concatenate(chunks, axis=1).T
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max)
    sf.write(str(dst), data.astype(np.float32), rate, subtype="PCM_16")
    return len(data) / float(rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--file", nargs="+", required=True, help="recording ids (file stems)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stem in args.file:
        src = next((args.in_dir / f"{stem}{ext}" for ext in AUDIO_EXTS
                    if (args.in_dir / f"{stem}{ext}").exists()), None)
        if src is None:
            raise SystemExit(f"Audio not found for {stem} under {args.in_dir}")
        if src.suffix.lower() in DECODE_EXTS:
            dst = args.out_dir / f"{stem}.wav"
            secs = decode_to_wav(src, dst)
            print(f"{src.name}: decoded to {dst.name} ({secs:.1f} s)")
        else:
            dst = args.out_dir / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(os.path.relpath(src.resolve(), args.out_dir.resolve()))


if __name__ == "__main__":
    main()
