#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from praatio import textgrid


def nonempty_word_span(tg_path: Path) -> tuple[float, float]:
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    word_entries = [entry for entry in tg.getTier("words").entries if entry.label.strip()]
    if not word_entries:
        raise ValueError(f"No non-empty words in {tg_path}")
    return float(word_entries[0].start), float(word_entries[-1].end)


def clip_one(input_path: Path, ref_path: Path, output_path: Path) -> dict:
    ref_start, ref_end = nonempty_word_span(ref_path)
    tg = textgrid.openTextgrid(str(input_path), includeEmptyIntervals=True)
    words = tg.getTier("words").entries
    nonempty = [entry for entry in words if entry.label.strip()]
    auto_start = float(nonempty[0].start) if nonempty else float(tg.minTimestamp)
    auto_end = float(nonempty[-1].end) if nonempty else float(tg.maxTimestamp)

    clipped = tg.crop(ref_start, ref_end, mode="truncated", rebaseToZero=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.save(
        str(output_path),
        format="long_textgrid",
        includeBlankSpaces=True,
        minTimestamp=ref_start,
        maxTimestamp=ref_end,
        reportingMode="silence",
    )

    changed = abs(auto_start - ref_start) > 1e-6 or abs(auto_end - ref_end) > 1e-6
    return {
        "file": input_path.name,
        "changed": int(changed),
        "auto_nonempty_start": auto_start,
        "auto_nonempty_end": auto_end,
        "ref_nonempty_start": ref_start,
        "ref_nonempty_end": ref_end,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--report-csv", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for ref_path in sorted(args.reference_dir.glob("*.TextGrid")):
        input_path = args.input_dir / ref_path.name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input TextGrid: {input_path}")
        rows.append(clip_one(input_path, ref_path, args.output_dir / ref_path.name))

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    changed = [row["file"] for row in rows if int(row["changed"]) == 1]
    print(f"Saved clipped TextGrids: {args.output_dir}")
    print(f"Saved report: {args.report_csv}")
    print(f"Modified files: {len(changed)} / {len(rows)}")
    for name in changed:
        print(name)


if __name__ == "__main__":
    main()
