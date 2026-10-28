#!/usr/bin/env python3
"""Per-file RQ1 counts with reference clause starts projected by word midpoint.

The paper's RQ1 script projects a reference clause start onto the first
automatic word that overlaps the clause in time by any amount. When the
reference clause starts a few milliseconds inside the previous automatic word,
that word is counted as the boundary, so a correct automatic boundary scores
one false positive and one false negative (verified: 172 of 483 English
errors). This script repeats RQ1 with the reference start projected onto the
first automatic word whose temporal midpoint lies inside the clause. Automatic
boundaries are unchanged (the same under both rules).

  rq1_midpoint.py --analysis <analysis repo> --lang en|ja --variant <dir> --out <csv>

<dir> holds auto_textgrids_clean/ and auto_clauses/ as written by
aligner_benchmark.py downstream (or the paper's data/<lang>/).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from praatio import textgrid


def _intervals(path: Path, tier: str) -> list[tuple[float, float]]:
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    return [(e.start, e.end) for e in tg.getTier(tier).entries if e.label.strip()]


def _starts_overlap(words, clauses) -> set[int]:
    pos = set()
    for ci, (cs, ce) in enumerate(clauses):
        for wi, (s, e) in enumerate(words):
            if min(e, ce) - max(s, cs) > 1e-9:
                if ci > 0 and wi > 0:
                    pos.add(wi)
                break
    return pos


def _starts_midpoint(words, clauses) -> set[int]:
    pos = set()
    for ci, (cs, ce) in enumerate(clauses):
        for wi, (s, e) in enumerate(words):
            if cs <= (s + e) / 2 < ce:
                if ci > 0 and wi > 0:
                    pos.add(wi)
                break
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--lang", choices=["en", "ja"], required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = args.analysis / "data" / args.lang
    files = json.loads((data / "file_list_40.json").read_text())["all_selected"]
    rows = []
    for f in files:
        words = _intervals(args.variant / "auto_textgrids_clean" / f"{f}.TextGrid", "words")
        auto = _starts_overlap(words, _intervals(args.variant / "auto_clauses" / f"{f}.TextGrid", "clauses"))
        gold = _starts_midpoint(words, _intervals(data / "reference_textgrids" / f"{f}.TextGrid", "clauses"))
        tp, fp, fn = len(auto & gold), len(auto - gold), len(gold - auto)
        rows.append({"file": f, "n_words": len(words), "tp": tp, "fp": fp, "fn": fn,
                     "tn": len(words) - tp - fp - fn})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
