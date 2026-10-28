#!/usr/bin/env python
"""
Keep only the learner's task speech: blank every word outside an analysis window.

Recordings often hold more than the task: an interviewer's prompt, an ID
announcement, a closing exchange. Give the window of the learner's speech per
recording, and every interval of every tier that is not wholly inside it gets
an empty label. The time axis and interval boundaries stay as they are, so the
TextGrids still line up with the audio and with Stage 4. The UF calculators
measure from the first to the last clause word, so nothing outside the window
counts. Recordings without a window are copied unchanged.

Usage (the runner passes these):
  python apply_window.py --in-dir <asr>/textgrids_clean --out-dir <asr>/textgrids_target \
    --file-list-json <file_list.json> (--window START END | --windows-csv <file_id,start,end CSV>)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

from praatio import textgrid
from praatio.utilities.constants import Interval


def blank_outside(in_path: Path, out_path: Path, start: float, end: float) -> int:
    """Write in_path with labels outside [start, end] blanked; return the number of words kept."""
    tg = textgrid.openTextgrid(str(in_path), includeEmptyIntervals=True)
    out = textgrid.Textgrid()
    kept = 0
    for name in tg.tierNames:
        tier = tg.getTier(name)
        entries = []
        for e in tier.entries:
            inside = e.start >= start - 1e-9 and e.end <= end + 1e-9
            entries.append(Interval(e.start, e.end, e.label if inside else ""))
            if inside and e.label.strip() and name == "words":
                kept += 1
        out.addTier(textgrid.IntervalTier(name, entries, tier.minTimestamp, tier.maxTimestamp))
    out.save(str(out_path), format="long_textgrid", includeBlankSpaces=True)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--file-list-json", type=Path, required=True)
    ap.add_argument("--file-list-key", type=str, default="files")
    ap.add_argument("--window", type=float, nargs=2, metavar=("START", "END"), help="the same window for every file")
    ap.add_argument("--windows-csv", type=Path, help="CSV with columns file_id,start,end")
    args = ap.parse_args()

    files = [str(x) for x in json.loads(args.file_list_json.read_text(encoding="utf-8"))[args.file_list_key]]
    windows: Dict[str, Tuple[float, float]] = {}
    if args.window:
        windows = {f: (args.window[0], args.window[1]) for f in files}
    if args.windows_csv:
        with args.windows_csv.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                windows[row["file_id"].strip()] = (float(row["start"]), float(row["end"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        src, dst = args.in_dir / f"{f}.TextGrid", args.out_dir / f"{f}.TextGrid"
        if not src.exists():
            continue
        if f in windows:
            s, e = windows[f]
            if e <= s:
                raise SystemExit(f"Window end must be after start for {f}: {s} {e}")
            n = blank_outside(src, dst, s, e)
            print(f"{f}: window {s:.2f}-{e:.2f} s, {n} words kept")
        else:
            shutil.copyfile(src, dst)
            print(f"{f}: no window, whole recording")


if __name__ == "__main__":
    main()
