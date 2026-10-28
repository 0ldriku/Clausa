#!/usr/bin/env python3
"""
Word-boundary and pause agreement of aligner outputs with the reference word tier.

The reference TextGrids carry the corpus word tier (ALLSSTAR / I-JAS), which is
independent of both aligners. For each recording, the automatic words inside
the reference speech span are matched to the reference words by label
(difflib on normalised labels), and for matched words the absolute start and
end differences are collected. Silent pauses (>= 0.25 s between consecutive
words, the pipeline's pause threshold) are counted on both sides.

  compare_word_timing.py --reference-dir REF --variant NAME=DIR [--variant ...] --out CSV
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import statistics
import unicodedata
from pathlib import Path

from praatio import textgrid

PAUSE_MIN = 0.25


def norm(label: str) -> str:
    label = unicodedata.normalize("NFKC", label).lower()
    return re.sub(r"[^\w']", "", label)


def word_entries(path: Path) -> list[tuple[float, float, str]]:
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
    tier = tg.getTier("words") if "words" in tg.tierNames else tg.getTier(tg.tierNames[0])
    return [(e.start, e.end, norm(e.label)) for e in tier.entries if norm(e.label)]


def pauses(words: list[tuple[float, float, str]]) -> list[float]:
    return [b[0] - a[1] for a, b in zip(words, words[1:]) if b[0] - a[1] >= PAUSE_MIN]


def compare(ref: list, auto: list) -> dict:
    span0, span1 = ref[0][0], ref[-1][1]
    auto = [w for w in auto if w[1] > span0 and w[0] < span1]
    sm = difflib.SequenceMatcher(a=[w[2] for w in ref], b=[w[2] for w in auto], autojunk=False)
    diffs = []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            r, a = ref[block.a + k], auto[block.b + k]
            diffs.append(abs(r[0] - a[0]))
            diffs.append(abs(r[1] - a[1]))
    rp, ap = pauses(ref), pauses(auto)
    return {
        "ref_words": len(ref),
        "matched_words": len(diffs) // 2,
        "boundary_mae_ms": 1000 * statistics.fmean(diffs) if diffs else float("nan"),
        "boundary_median_ms": 1000 * statistics.median(diffs) if diffs else float("nan"),
        "within_20ms": sum(d <= 0.02 for d in diffs) / len(diffs) if diffs else float("nan"),
        "within_50ms": sum(d <= 0.05 for d in diffs) / len(diffs) if diffs else float("nan"),
        "within_100ms": sum(d <= 0.10 for d in diffs) / len(diffs) if diffs else float("nan"),
        "ref_pauses": len(rp),
        "auto_pauses": len(ap),
        "ref_mean_pause_s": statistics.fmean(rp) if rp else 0.0,
        "auto_mean_pause_s": statistics.fmean(ap) if ap else 0.0,
        "auto_mean_word_dur_s": statistics.fmean(w[1] - w[0] for w in auto) if auto else 0.0,
        "ref_mean_word_dur_s": statistics.fmean(w[1] - w[0] for w in ref),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--variant", action="append", required=True, help="NAME=DIR with TextGrids")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.variant:
        name, d = spec.split("=", 1)
        for auto_path in sorted(Path(d).glob("*.TextGrid")):
            ref_path = args.reference_dir / auto_path.name
            if not ref_path.exists():
                continue
            rows.append({"variant": name, "file": auto_path.stem,
                         **compare(word_entries(ref_path), word_entries(auto_path))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'variant':12s} files  MAE_ms  median_ms  <=20ms  <=50ms  <=100ms  pauses(auto/ref)  meanpause(auto/ref)  worddur(auto/ref)")
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        m = lambda k: statistics.fmean(r[k] for r in rs)
        s = lambda k: sum(r[k] for r in rs)
        print(f"{name:12s} {len(rs):5d} {m('boundary_mae_ms'):7.1f} {m('boundary_median_ms'):10.1f} "
              f"{m('within_20ms'):7.3f} {m('within_50ms'):7.3f} {m('within_100ms'):8.3f} "
              f"{s('auto_pauses'):8d}/{s('ref_pauses'):<8d} {m('auto_mean_pause_s'):9.3f}/{m('ref_mean_pause_s'):.3f} "
              f"{m('auto_mean_word_dur_s'):9.3f}/{m('ref_mean_word_dur_s'):.3f}")


if __name__ == "__main__":
    main()
