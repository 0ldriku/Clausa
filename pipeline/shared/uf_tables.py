"""The words, pauses and clauses behind a recording's measures, as three tables.

The calculators (en/caf_calculator_vad_classifier.py, ja/caf_calculator_ja_gap_classifier.py)
give what they computed the measures from, with --tables-dir: uf_words.csv (one row per word),
uf_pauses.csv (one row per pause, or part of a pause, as MCPR, ECPR and PR count them) and
uf_clauses.csv (one row per clause). The file column holds <id>.TextGrid, as in uf_measures.csv.
UTF-8 with a BOM, so that Excel reads Japanese; times in seconds, to the millisecond. Labels
are written on one line. A recording's rows never change its measures: the calculators make them
after its measures row, and a failure here leaves that row as it is (and the recording out of the
tables).

- words: every interval of the words tier that is not a pause. disfluent: 1 if the middle of
  the word lies in a labelled interval of the disfluency tier, 0 if not, empty without that
  tier. clause: the clause the middle of the word lies in. syllables (English) or morae
  (Japanese): as the measures count them; their sum is the recording's total.
- pauses: pause is the pause's number. A pause that crosses a gap between clauses has a row for
  each part (a part may be shorter than 0.25 s): MCPR counts the mid-clause rows, ECPR the end-clause rows, PR the pauses; MPD is the mean
  of the pauses' durations, MCPD and ECPD of their rows'. clause: the clause a mid-clause part
  lies in, or the one an end-clause part follows; other parts lie outside every clause.
- clauses: the labelled intervals of the clauses tier, numbered in order, with their words and
  syllables (morae) as in uf_words.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

NAMES = {"words": "uf_words.csv", "pauses": "uf_pauses.csv", "clauses": "uf_clauses.csv"}
DISFLUENCY_TIER = "disfluency"

Span = Tuple[float, float]


def _t(x: float) -> float:
    return round(float(x), 3)


def _find(spans: Sequence[Span], t: float) -> Optional[int]:
    """The 1-based number of the first span that holds t, or None."""
    for i, (s, e) in enumerate(spans, 1):
        if s <= t < e:
            return i
    return None


def disfluent_spans(tg) -> Optional[List[Span]]:
    """The labelled intervals of a praatio TextGrid's disfluency tier; None without that tier (or if it
    is a point tier)."""
    from praatio import textgrid
    if DISFLUENCY_TIER not in tg.tierNames:
        return None
    tier = tg.getTier(DISFLUENCY_TIER)
    if not isinstance(tier, textgrid.IntervalTier):
        return None
    return [(float(e.start), float(e.end)) for e in tier.entries if e.label.strip()]


def _one_line(text: str) -> str:
    """A label as one line (Praat allows line breaks in labels; a CSV row should not hold one)."""
    return " ".join(str(text).split())


def headers(unit: str) -> dict:
    """The columns of each table; unit is syllables (English) or morae (Japanese)."""
    return {
        "words": ["file", "word", "start", "end", "disfluent", "clause", unit],
        "pauses": ["file", "pause", "start", "end", "duration", "location", "clause"],
        "clauses": ["file", "clause", "start", "end", "words", unit, "text"],
    }


def rows(file: str, words: Iterable[Tuple[float, float, str, int]], disfluent: Optional[Sequence[Span]],
         clauses: Sequence[Tuple[float, float, str]], pauses: Sequence[Span],
         parts: Iterable[Tuple[float, float, str]]) -> dict:
    """One recording's rows of the three tables.

    words: (start, end, text, syllables) of the words tier's intervals that are not pauses; clauses:
    (start, end, text) of the labelled clauses; pauses: the pauses the measures count; parts: their
    parts as (start, end, location), location mid-clause, end-clause or other.
    """
    spans = [(s, e) for s, e, _ in clauses]
    counts = [[0, 0] for _ in clauses]
    out = {"words": [], "pauses": [], "clauses": []}
    for s, e, text, n in words:
        mid = (s + e) / 2
        clause = _find(spans, mid)
        if clause is not None:
            counts[clause - 1][0] += 1
            counts[clause - 1][1] += n
        flag = "" if disfluent is None else int(_find(disfluent, mid) is not None)
        out["words"].append([file, _one_line(text), _t(s), _t(e), flag, "" if clause is None else clause, n])
    for s, e, location in sorted(parts):
        mid = (s + e) / 2
        clause = None
        if location == "mid-clause":
            clause = _find(spans, mid)
        elif location == "end-clause":
            before = [i for i, (_, ce) in enumerate(spans, 1) if ce <= mid]
            clause = before[-1] if before else None
        out["pauses"].append([file, _find(pauses, mid), _t(s), _t(e), _t(e - s), location,
                              "" if clause is None else clause])
    for i, ((s, e, text), (n_words, n)) in enumerate(zip(clauses, counts), 1):
        out["clauses"].append([file, i, _t(s), _t(e), n_words, n, _one_line(text)])
    return out


def write(folder: Path, unit: str, recordings: Iterable[dict]) -> None:
    """Write the three tables of these recordings' rows (rows()) into folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    recordings = list(recordings)
    for table, columns in headers(unit).items():
        with open(folder / NAMES[table], "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(columns)
            for r in recordings:
                w.writerows(r[table])
