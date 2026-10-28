#!/usr/bin/env python3
"""
Blank ASR TextGrid labels outside the manual non-empty word span.

This is the English-side counterpart to the Japanese manual-span sync helper:
  - keep the original time axis unchanged
  - keep interval boundaries unchanged
  - blank ASR labels before the manual first non-blank word
  - blank ASR labels after the manual last non-blank word
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from praatio import textgrid
from praatio.utilities.constants import Interval


def _tier_names(tg: textgrid.Textgrid) -> List[str]:
    if hasattr(tg, "tierNames"):
        return list(getattr(tg, "tierNames"))
    if hasattr(tg, "tierNameList"):
        return list(getattr(tg, "tierNameList"))
    if hasattr(tg, "_tierDict"):
        return list(getattr(tg, "_tierDict").keys())
    raise RuntimeError("Could not read tier names from TextGrid.")


def _get_tier(tg: textgrid.Textgrid, name: str):
    if hasattr(tg, "getTier"):
        return tg.getTier(name)
    if hasattr(tg, "_tierDict"):
        return tg._tierDict[name]
    raise RuntimeError("Could not access tier in TextGrid.")


def _find_words_tier_name(tg: textgrid.Textgrid) -> str:
    names = _tier_names(tg)
    for name in names:
        if name.lower() == "words":
            return name
    if not names:
        raise RuntimeError("TextGrid has no tiers.")
    return names[0]


def _manual_span(manual_tg_path: Path) -> Tuple[float, float]:
    tg = textgrid.openTextgrid(str(manual_tg_path), includeEmptyIntervals=True)
    tier_name = _find_words_tier_name(tg)
    tier = _get_tier(tg, tier_name)
    nonblank = [entry for entry in tier.entries if entry.label.strip()]
    if not nonblank:
        raise RuntimeError(f"No non-blank words in manual TextGrid: {manual_tg_path}")
    return float(nonblank[0].start), float(nonblank[-1].end)


def _blank_labels(entries: Iterable[Interval], cut_start: float, cut_end: float) -> List[Interval]:
    out: List[Interval] = []
    for entry in entries:
        label = entry.label
        if entry.start < cut_start or entry.end > cut_end:
            label = ""
        out.append(Interval(entry.start, entry.end, label))
    return out


def process_one(asr_tg_path: Path, manual_tg_path: Path, out_tg_path: Path) -> Tuple[float, float]:
    start, end = _manual_span(manual_tg_path)
    tg = textgrid.openTextgrid(str(asr_tg_path), includeEmptyIntervals=True)
    out_tg = textgrid.Textgrid()

    for name in _tier_names(tg):
        tier = _get_tier(tg, name)
        new_entries = _blank_labels(tier.entries, start, end)
        min_t = getattr(tier, "minTimestamp", new_entries[0].start if new_entries else 0.0)
        max_t = getattr(tier, "maxTimestamp", new_entries[-1].end if new_entries else min_t)
        out_tier = textgrid.IntervalTier(name, new_entries, minT=min_t, maxT=max_t)
        out_tg.addTier(out_tier)

    out_tg.save(str(out_tg_path), format="long_textgrid", includeBlankSpaces=True)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blank ASR TextGrid labels outside the manual non-empty word span."
    )
    parser.add_argument("--asr-dir", required=True, type=Path, help="Input ASR TextGrid directory.")
    parser.add_argument("--manual-dir", required=True, type=Path, help="Manual gold TextGrid directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for synced ASR TextGrids.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    asr_files = sorted(args.asr_dir.glob("*.TextGrid"))
    if not asr_files:
        raise SystemExit(f"No TextGrid files found: {args.asr_dir}")

    processed = 0
    skipped = 0
    for asr_path in asr_files:
        manual_path = args.manual_dir / asr_path.name
        if not manual_path.exists():
            print(f"[skip] manual missing: {asr_path.name}")
            skipped += 1
            continue
        out_path = args.out_dir / asr_path.name
        try:
            start, end = process_one(asr_path, manual_path, out_path)
            processed += 1
            print(f"[ok] {asr_path.stem}: span=({start:.3f}, {end:.3f})")
        except Exception as exc:
            print(f"[error] {asr_path.stem}: {exc}")
            skipped += 1

    print(f"Done. processed={processed}, skipped={skipped}, out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
