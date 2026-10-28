#!/usr/bin/env python
"""
Japanese CAF calculator with classifier-guided pause refinement (gap-candidate mode).

This mirrors the EN `caf_calculator_vad_classifier.py` flow:
1) Keep ASR/segment outputs unchanged.
2) Read per-file candidate CSV from gap-candidate postprocess.
3) Split pause intervals when predicted filler islands overlap them.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from praatio import textgrid


@dataclass
class WordInterval:
    start: float
    end: float
    text: str
    is_pause: bool
    mora_count: int = 0


@dataclass
class PauseInterval:
    start: float
    end: float
    duration: float
    location: str = "unknown"


@dataclass
class ClauseInterval:
    start: float
    end: float
    text: str
    clause_type: str = "clause"


@dataclass
class ClfRefineStats:
    pause_candidates: int = 0
    pauses_split: int = 0
    predicted_speech_islands: int = 0
    refined_pause_count: int = 0


class TextGridReaderJA:
    PAUSE_MARKERS = {
        "",
        "sp",
        "sil",
        "spn",
        "noise",
        "<sil>",
        "<sp>",
        "<spn>",
        "<p>",
        "<pause>",
        "pause",
        "breath",
        "<breath>",
        "silb",
        "sile",
        "#",
        "...",
    }
    FILLER_MARKERS = {"<filler_speech>", "<filler>", "filler_speech"}

    SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮっッー")

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_id = Path(filepath).stem
        self.tg = textgrid.openTextgrid(filepath, includeEmptyIntervals=True)
        self.words: List[WordInterval] = []
        self.clauses: List[ClauseInterval] = []
        self.total_mora = 0
        self.analysis_start = 0.0
        self.analysis_end = 0.0
        self._extract_tiers()

    def _extract_tiers(self):
        tier_names = self.tg.tierNames
        word_tier_name = None
        for name in ["words", "word", "Word", "Words", "単語", "transcription"]:
            if name in tier_names:
                word_tier_name = name
                break
        if not word_tier_name:
            word_tier_name = tier_names[0]

        clause_tier_name = None
        for name in ["clauses", "clause", "Clauses", "Clause"]:
            if name in tier_names:
                clause_tier_name = name
                break

        word_tier = self.tg.getTier(word_tier_name)
        content_words: List[WordInterval] = []
        for entry in word_tier.entries:
            text = entry.label.strip()
            is_pause = text.lower() in self.PAUSE_MARKERS or text == ""
            is_filler_marker = text.lower() in self.FILLER_MARKERS
            interval = WordInterval(
                start=float(entry.start),
                end=float(entry.end),
                text=text,
                is_pause=is_pause,
                mora_count=0 if (is_pause or is_filler_marker) else self._count_mora(text),
            )
            self.words.append(interval)
            if (not is_pause) and (not is_filler_marker):
                self.total_mora += interval.mora_count
                content_words.append(interval)

        if content_words:
            self.analysis_start = float(content_words[0].start)
            self.analysis_end = float(content_words[-1].end)
        elif self.words:
            self.analysis_start = float(self.words[0].start)
            self.analysis_end = float(self.words[-1].end)

        if clause_tier_name:
            clause_tier = self.tg.getTier(clause_tier_name)
            for entry in clause_tier.entries:
                txt = entry.label.strip()
                if txt:
                    self.clauses.append(
                        ClauseInterval(start=float(entry.start), end=float(entry.end), text=txt)
                    )

    def _count_mora(self, text: str) -> int:
        count = 0
        for ch in text:
            if ch.isspace():
                continue
            if ch in self.SMALL_KANA:
                continue
            count += 1
        return max(count, 1) if text else 0

    def get_pauses(self, min_duration: float = 0.25) -> List[PauseInterval]:
        pauses = []
        for w in self.words:
            if w.is_pause:
                start = max(w.start, self.analysis_start)
                end = min(w.end, self.analysis_end)
                dur = end - start
                if dur >= min_duration:
                    pauses.append(PauseInterval(start=start, end=end, duration=dur))
        return pauses


class ClassifierPauseRefiner:
    def __init__(self, candidate_dir: str, min_pause: float = 0.25, merge_gap: float = 0.03):
        self.candidate_dir = Path(candidate_dir)
        self.min_pause = min_pause
        self.merge_gap = merge_gap
        self._cache: Dict[str, List[Tuple[float, float]]] = {}

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        out: List[Tuple[float, float]] = []
        s0, e0 = intervals[0]
        for s1, e1 in intervals[1:]:
            if s1 <= e0 + self.merge_gap:
                e0 = max(e0, e1)
            else:
                out.append((s0, e0))
                s0, e0 = s1, e1
        out.append((s0, e0))
        return out

    def _load_intervals(self, file_id: str) -> List[Tuple[float, float]]:
        if file_id in self._cache:
            return self._cache[file_id]

        path = self.candidate_dir / f"{file_id}_vad_classifier.csv"
        if not path.exists():
            self._cache[file_id] = []
            return []

        df = pd.read_csv(path)
        if df.empty:
            self._cache[file_id] = []
            return []

        required = {"pred_filler", "candidate_start", "candidate_end"}
        if not required.issubset(df.columns):
            self._cache[file_id] = []
            return []

        keep = df[
            (df["pred_filler"] == 1)
            & df["candidate_start"].notna()
            & df["candidate_end"].notna()
        ].copy()
        ints: List[Tuple[float, float]] = []
        for _, row in keep.iterrows():
            s = float(row["candidate_start"])
            e = float(row["candidate_end"])
            if e > s:
                ints.append((s, e))

        ints = self._merge_intervals(sorted(ints, key=lambda x: x[0]))
        self._cache[file_id] = ints
        return ints

    def refine(self, file_id: str, pauses: List[PauseInterval]) -> Tuple[List[PauseInterval], ClfRefineStats]:
        stats = ClfRefineStats(pause_candidates=len(pauses))
        preds = self._load_intervals(file_id)
        if not preds:
            stats.refined_pause_count = len(pauses)
            return pauses, stats

        refined: List[PauseInterval] = []
        for pause in pauses:
            overlaps: List[Tuple[float, float]] = []
            for s, e in preds:
                if e <= pause.start or s >= pause.end:
                    continue
                overlaps.append((max(pause.start, s), min(pause.end, e)))

            if not overlaps:
                refined.append(pause)
                continue

            overlaps = self._merge_intervals(sorted(overlaps, key=lambda x: x[0]))
            silence_parts: List[Tuple[float, float]] = []
            cursor = pause.start
            for s, e in overlaps:
                if (s - cursor) >= self.min_pause:
                    silence_parts.append((cursor, s))
                cursor = max(cursor, e)
            if (pause.end - cursor) >= self.min_pause:
                silence_parts.append((cursor, pause.end))

            stats.pauses_split += 1
            stats.predicted_speech_islands += len(overlaps)
            if not silence_parts:
                continue

            for s, e in silence_parts:
                refined.append(PauseInterval(start=s, end=e, duration=e - s))

        refined.sort(key=lambda x: (x.start, x.end))
        stats.refined_pause_count = len(refined)
        return refined, stats


class CAFCalculatorJA:
    def __init__(
        self,
        tg_reader: TextGridReaderJA,
        min_pause: float = 0.25,
        refiner: Optional[ClassifierPauseRefiner] = None,
    ):
        self.tg = tg_reader
        self.min_pause = min_pause
        self.total_mora = tg_reader.total_mora
        self.clauses = tg_reader.clauses
        self.clf_stats = ClfRefineStats()

        raw_pauses = self.tg.get_pauses(self.min_pause)
        if refiner is not None:
            self.pauses, self.clf_stats = refiner.refine(self.tg.file_id, raw_pauses)
        else:
            self.pauses = raw_pauses
            self.clf_stats.refined_pause_count = len(self.pauses)
        self._classify_pauses()

    @staticmethod
    def _merge_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not ranges:
            return []
        merged: List[Tuple[float, float]] = []
        cur_s, cur_e = sorted(ranges, key=lambda x: (x[0], x[1]))[0]
        for s, e in sorted(ranges, key=lambda x: (x[0], x[1]))[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    @staticmethod
    def _intersections(
        start: float, end: float, ranges: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        hits: List[Tuple[float, float]] = []
        for s, e in ranges:
            if e <= start or s >= end:
                continue
            hits.append((max(start, s), min(end, e)))
        return hits

    @staticmethod
    def _subtract_ranges(
        start: float, end: float, cuts: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not cuts:
            return [(start, end)]
        parts: List[Tuple[float, float]] = []
        cursor = start
        for s, e in CAFCalculatorJA._merge_ranges(cuts):
            if s > cursor:
                parts.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < end:
            parts.append((cursor, end))
        return [(s, e) for s, e in parts if e > s]

    def _clause_ranges(self) -> List[Tuple[float, float]]:
        ranges = []
        for c in self.clauses:
            s = max(c.start, self.tg.analysis_start)
            e = min(c.end, self.tg.analysis_end)
            if e > s:
                ranges.append((s, e))
        return self._merge_ranges(ranges)

    def _inter_clause_gaps(self) -> List[Tuple[float, float]]:
        clause_ranges = self._clause_ranges()
        gaps: List[Tuple[float, float]] = []
        for left, right in zip(clause_ranges, clause_ranges[1:]):
            if right[0] > left[1]:
                gaps.append((left[1], right[0]))
        return gaps

    def _classify_pauses(self):
        self.mid_clause_pauses: List[PauseInterval] = []
        self.end_clause_pauses: List[PauseInterval] = []
        self.other_pauses: List[PauseInterval] = []
        clause_ranges = self._clause_ranges()
        gap_ranges = self._inter_clause_gaps()
        for pause in self.pauses:
            covered: List[Tuple[float, float]] = []

            for s, e in self._intersections(pause.start, pause.end, gap_ranges):
                seg = PauseInterval(start=s, end=e, duration=e - s, location="end-clause")
                self.end_clause_pauses.append(seg)
                covered.append((s, e))

            for s, e in self._intersections(pause.start, pause.end, clause_ranges):
                seg = PauseInterval(start=s, end=e, duration=e - s, location="mid-clause")
                self.mid_clause_pauses.append(seg)
                covered.append((s, e))

            for s, e in self._subtract_ranges(pause.start, pause.end, covered):
                seg = PauseInterval(start=s, end=e, duration=e - s, location="other")
                self.other_pauses.append(seg)

    def _calculate_mora_runs(self) -> List[int]:
        runs = []
        cur = 0
        for w in self.tg.words:
            if w.is_pause:
                if (w.end - w.start) >= self.min_pause:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            else:
                cur += w.mora_count
        if cur > 0:
            runs.append(cur)
        return runs

    def calculate_all(self) -> Dict:
        if not self.tg.words:
            return {"error": "No words found"}

        total_duration = self.tg.analysis_end - self.tg.analysis_start
        total_pause_duration = sum(p.duration for p in self.pauses)
        phonation_time = total_duration - total_pause_duration
        mora = self.total_mora

        n_mid = len(self.mid_clause_pauses)
        n_end = len(self.end_clause_pauses)
        n_all = len(self.pauses)

        mid_durs = [p.duration for p in self.mid_clause_pauses]
        end_durs = [p.duration for p in self.end_clause_pauses]
        all_durs = [p.duration for p in self.pauses]

        ar = mora / phonation_time if phonation_time > 0 else 0
        mcpr = n_mid / mora if mora > 0 else 0
        ecpr = n_end / mora if mora > 0 else 0
        pr = n_all / mora if mora > 0 else 0
        mcpd = float(np.mean(mid_durs)) if mid_durs else 0
        ecpd = float(np.mean(end_durs)) if end_durs else 0
        mpd = float(np.mean(all_durs)) if all_durs else 0
        sr = mora / total_duration if total_duration > 0 else 0
        runs = self._calculate_mora_runs()
        mlr = float(np.mean(runs)) if runs else 0

        return {
            "AR": round(ar, 3),
            "SR": round(sr, 3),
            "MLR": round(mlr, 2),
            "MCPR": round(mcpr, 4),
            "ECPR": round(ecpr, 4),
            "PR": round(pr, 4),
            "MCPD": round(mcpd, 3),
            "ECPD": round(ecpd, 3),
            "MPD": round(mpd, 3),
            "total_mora": mora,
            "total_duration": round(total_duration, 2),
            "phonation_time": round(phonation_time, 2),
            "n_pauses": n_all,
            "n_mid_clause_pauses": n_mid,
            "n_end_clause_pauses": n_end,
            "n_other_pauses": len(self.other_pauses),
            "n_clauses": len(self.clauses),
            "clf_pause_candidates": self.clf_stats.pause_candidates,
            "clf_pauses_split": self.clf_stats.pauses_split,
            "clf_predicted_speech_islands": self.clf_stats.predicted_speech_islands,
            "clf_refined_pause_count": self.clf_stats.refined_pause_count,
        }


def _load_selected_ids(file_list_json: Optional[str], file_list_key: Optional[str]) -> Optional[Set[str]]:
    if not file_list_json:
        return None
    with open(file_list_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if not file_list_key:
            raise ValueError("--file-list-key is required when JSON top-level is an object")
        if file_list_key not in data:
            raise KeyError(f"Key '{file_list_key}' not found in {file_list_json}")
        items = data[file_list_key]
    else:
        raise ValueError("file-list JSON must be list or object")

    return {str(x).strip() for x in items if str(x).strip()}


def process_textgrid(
    filepath: str,
    min_pause: float = 0.25,
    refiner: Optional[ClassifierPauseRefiner] = None,
) -> Dict:
    tg_reader = TextGridReaderJA(filepath)
    calc = CAFCalculatorJA(tg_reader, min_pause=min_pause, refiner=refiner)
    out = calc.calculate_all()
    out["file"] = os.path.basename(filepath)
    return out


def process_directory(
    input_dir: str,
    min_pause: float = 0.25,
    refiner: Optional[ClassifierPauseRefiner] = None,
    selected_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".textgrid")]
    if selected_ids is not None:
        files = [f for f in files if Path(f).stem in selected_ids]
    rows = []
    for fname in sorted(files):
        path = os.path.join(input_dir, fname)
        try:
            row = process_textgrid(path, min_pause=min_pause, refiner=refiner)
            rows.append(row)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            rows.append({"file": fname, "error": str(e)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="TextGrid file or directory")
    ap.add_argument("--candidate-dir", required=True, help="Directory with *_vad_classifier.csv files")
    ap.add_argument("--merge-gap", type=float, default=0.03, help="Merge close predicted islands")
    ap.add_argument("--min-pause", type=float, default=0.25)
    ap.add_argument("--file-list-json")
    ap.add_argument("--file-list-key")
    ap.add_argument("--output", "-o")
    args = ap.parse_args()

    refiner = ClassifierPauseRefiner(
        candidate_dir=args.candidate_dir,
        min_pause=args.min_pause,
        merge_gap=args.merge_gap,
    )
    selected_ids = _load_selected_ids(args.file_list_json, args.file_list_key)

    if os.path.isfile(args.input):
        df = pd.DataFrame([process_textgrid(args.input, min_pause=args.min_pause, refiner=refiner)])
    else:
        df = process_directory(
            args.input,
            min_pause=args.min_pause,
            refiner=refiner,
            selected_ids=selected_ids,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")
    else:
        print(df.to_string(index=False))

    for c in ["clf_pause_candidates", "clf_pauses_split", "clf_predicted_speech_islands"]:
        if c in df.columns:
            print(f"{c}: {int(pd.to_numeric(df[c], errors='coerce').fillna(0).sum())}")


if __name__ == "__main__":
    main()
