"""
CAF calculator with classifier-guided pause refinement.

Purpose:
  Keep the existing pipeline unchanged and run a postprocess-only variant where
  pause intervals are split by predicted filler speech islands from
  `shared/postprocess_vad_filler_classifier.py`.

Input:
  - Clause TextGrids (same as baseline CAF)
  - Per-file candidate CSVs with columns:
      pred_filler, candidate_start, candidate_end

Usage:
  python caf_calculator_vad_classifier.py <clauses folder> \
    --candidate-dir <gap_candidates>/per_file \
    --file-list-json <file_list.json> --output uf_measures.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from praatio import textgrid


@dataclass
class WordInterval:
    start: float
    end: float
    text: str
    is_pause: bool
    syllable_count: int = 0


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


class TextGridReader:
    PAUSE_MARKERS = {
        "sp",
        "sil",
        "spn",
        "noise",
        "",
        "<sil>",
        "<sp>",
        "<spn>",
        "<p>",
        "<pause>",
        "breath",
        "<breath>",
    }
    FILLER_MARKERS = {"<filler_speech>", "<filler>", "filler_speech"}

    VOWELS = {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AX",
        "AXR",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IX",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
        "UX",
    }

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_id = Path(filepath).stem
        self.tg = textgrid.openTextgrid(filepath, includeEmptyIntervals=True)
        self.words: List[WordInterval] = []
        self.phones: List[Tuple[float, float, str]] = []
        self.clauses: List[ClauseInterval] = []
        self.total_syllables = 0
        self.analysis_start = 0.0
        self.analysis_end = 0.0
        self._extract_tiers()

    def _extract_tiers(self):
        tier_names = self.tg.tierNames

        word_tier_name = None
        for name in ["words", "word", "Word", "Words"]:
            if name in tier_names:
                word_tier_name = name
                break
        if not word_tier_name:
            word_tier_name = tier_names[0]

        phone_tier_name = None
        for name in ["phones", "phone", "Phone", "Phones"]:
            if name in tier_names:
                phone_tier_name = name
                break

        clause_tier_name = None
        for name in ["clauses", "clause", "Clauses", "Clause"]:
            if name in tier_names:
                clause_tier_name = name
                break

        if phone_tier_name:
            phone_tier = self.tg.getTier(phone_tier_name)
            for entry in phone_tier.entries:
                self.phones.append((entry.start, entry.end, entry.label.strip()))

        word_tier = self.tg.getTier(word_tier_name)
        content_words: List[WordInterval] = []
        for entry in word_tier.entries:
            text = entry.label.strip()
            is_pause = text.lower() in self.PAUSE_MARKERS or text == ""
            is_filler_marker = text.lower() in self.FILLER_MARKERS

            syllables = 0
            if is_filler_marker:
                syllables = 0
            elif not is_pause and self.phones:
                syllables = self._count_syllables_in_range(entry.start, entry.end)
            elif not is_pause:
                syllables = self._estimate_syllables(text)

            interval = WordInterval(
                start=entry.start,
                end=entry.end,
                text=text,
                is_pause=is_pause,
                syllable_count=syllables,
            )
            self.words.append(interval)
            if (not is_pause) and (not is_filler_marker):
                self.total_syllables += syllables
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
                text = entry.label.strip()
                if text:
                    self.clauses.append(ClauseInterval(start=entry.start, end=entry.end, text=text))

    def _count_syllables_in_range(self, start: float, end: float) -> int:
        count = 0
        for p_start, p_end, label in self.phones:
            if p_start >= start and p_end <= end:
                base_phone = label.split(",")[0].strip()
                base_phone = re.sub(r"[0-9*]", "", base_phone).upper()
                if base_phone in self.VOWELS:
                    count += 1
        return max(count, 1) if count == 0 else count

    def _estimate_syllables(self, text: str) -> int:
        text = text.lower().strip()
        if not text:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for ch in text:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if text.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    def get_pauses(self, min_duration: float = 0.25) -> List[PauseInterval]:
        pauses: List[PauseInterval] = []
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

        keep = df[(df["pred_filler"] == 1) & df["candidate_start"].notna() & df["candidate_end"].notna()].copy()
        ints: List[Tuple[float, float]] = []
        for _, r in keep.iterrows():
            s = float(r["candidate_start"])
            e = float(r["candidate_end"])
            if e > s:
                ints.append((s, e))
        ints = self._merge_intervals(sorted(ints, key=lambda x: x[0]))
        self._cache[file_id] = ints
        return ints

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

    def refine(self, file_id: str, pauses: List[PauseInterval]) -> Tuple[List[PauseInterval], ClfRefineStats]:
        stats = ClfRefineStats(pause_candidates=len(pauses))
        preds = self._load_intervals(file_id)
        if not preds:
            stats.refined_pause_count = len(pauses)
            return pauses, stats

        refined: List[PauseInterval] = []
        for p in pauses:
            overlaps: List[Tuple[float, float]] = []
            for s, e in preds:
                if e <= p.start or s >= p.end:
                    continue
                overlaps.append((max(p.start, s), min(p.end, e)))

            if not overlaps:
                refined.append(p)
                continue

            overlaps = self._merge_intervals(sorted(overlaps, key=lambda x: x[0]))
            silence_parts: List[Tuple[float, float]] = []
            cursor = p.start
            for s, e in overlaps:
                if (s - cursor) >= self.min_pause:
                    silence_parts.append((cursor, s))
                cursor = max(cursor, e)
            if (p.end - cursor) >= self.min_pause:
                silence_parts.append((cursor, p.end))

            if not silence_parts:
                # Predicted speech covers the whole pause; drop it from pause set.
                stats.pauses_split += 1
                stats.predicted_speech_islands += len(overlaps)
                continue

            stats.pauses_split += 1
            stats.predicted_speech_islands += len(overlaps)
            for s, e in silence_parts:
                refined.append(PauseInterval(start=s, end=e, duration=e - s))

        refined.sort(key=lambda x: (x.start, x.end))
        stats.refined_pause_count = len(refined)
        return refined, stats


class CAFCalculator:
    def __init__(
        self,
        tg_reader: TextGridReader,
        min_pause: float = 0.25,
        refiner: Optional[ClassifierPauseRefiner] = None,
    ):
        self.tg = tg_reader
        self.min_pause = min_pause
        self.total_syllables = tg_reader.total_syllables
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
    def _intersections(start: float, end: float, ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        hits: List[Tuple[float, float]] = []
        for s, e in ranges:
            if e <= start or s >= end:
                continue
            hits.append((max(start, s), min(end, e)))
        return hits

    @staticmethod
    def _subtract_ranges(start: float, end: float, cuts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not cuts:
            return [(start, end)]
        parts: List[Tuple[float, float]] = []
        cursor = start
        for s, e in CAFCalculator._merge_ranges(cuts):
            if s > cursor:
                parts.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < end:
            parts.append((cursor, end))
        return [(s, e) for s, e in parts if e > s]

    def _clause_ranges(self) -> List[Tuple[float, float]]:
        ranges = []
        for clause in self.clauses:
            start = max(clause.start, self.tg.analysis_start)
            end = min(clause.end, self.tg.analysis_end)
            if end > start:
                ranges.append((start, end))
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
        for p in self.pauses:
            covered: List[Tuple[float, float]] = []
            for s, e in self._intersections(p.start, p.end, gap_ranges):
                seg = PauseInterval(start=s, end=e, duration=e - s, location="end-clause")
                self.end_clause_pauses.append(seg)
                covered.append((s, e))
            for s, e in self._intersections(p.start, p.end, clause_ranges):
                seg = PauseInterval(start=s, end=e, duration=e - s, location="mid-clause")
                self.mid_clause_pauses.append(seg)
                covered.append((s, e))
            for s, e in self._subtract_ranges(p.start, p.end, covered):
                seg = PauseInterval(start=s, end=e, duration=e - s, location="other")
                self.other_pauses.append(seg)

    def _calculate_syllable_runs(self) -> List[int]:
        runs = []
        cur = 0
        for w in self.tg.words:
            if w.is_pause:
                if (w.end - w.start) >= self.min_pause:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            else:
                cur += w.syllable_count
        if cur > 0:
            runs.append(cur)
        return runs

    def calculate_all(self) -> Dict:
        if not self.tg.words:
            return {"error": "No words found"}

        total_duration = self.tg.analysis_end - self.tg.analysis_start
        total_pause_duration = sum(p.duration for p in self.pauses)
        phonation_time = total_duration - total_pause_duration
        syllables = self.total_syllables

        n_mid = len(self.mid_clause_pauses)
        n_end = len(self.end_clause_pauses)
        n_total = len(self.pauses)

        mid_d = [p.duration for p in self.mid_clause_pauses]
        end_d = [p.duration for p in self.end_clause_pauses]
        all_d = [p.duration for p in self.pauses]

        AR = syllables / phonation_time if phonation_time > 0 else 0
        SR = syllables / total_duration if total_duration > 0 else 0
        MCPR = n_mid / syllables if syllables > 0 else 0
        ECPR = n_end / syllables if syllables > 0 else 0
        PR = n_total / syllables if syllables > 0 else 0
        MCPD = np.mean(mid_d) if mid_d else 0
        ECPD = np.mean(end_d) if end_d else 0
        MPD = np.mean(all_d) if all_d else 0
        runs = self._calculate_syllable_runs()
        MLR = np.mean(runs) if runs else 0

        return {
            "AR": round(AR, 3),
            "SR": round(SR, 3),
            "MLR": round(MLR, 2),
            "MCPR": round(MCPR, 4),
            "ECPR": round(ECPR, 4),
            "PR": round(PR, 4),
            "MCPD": round(MCPD, 3),
            "ECPD": round(ECPD, 3),
            "MPD": round(MPD, 3),
            "total_syllables": syllables,
            "total_duration": round(total_duration, 2),
            "phonation_time": round(phonation_time, 2),
            "n_pauses": n_total,
            "n_mid_clause_pauses": n_mid,
            "n_end_clause_pauses": n_end,
            "n_other_pauses": len(self.other_pauses),
            "n_clauses": len(self.clauses),
            "clf_pause_candidates": self.clf_stats.pause_candidates,
            "clf_pauses_split": self.clf_stats.pauses_split,
            "clf_predicted_speech_islands": self.clf_stats.predicted_speech_islands,
            "clf_refined_pause_count": self.clf_stats.refined_pause_count,
        }


def _load_file_id_filter(json_path: Optional[str], key: str) -> Optional[set]:
    if not json_path:
        return None
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    values = data.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Key '{key}' not found or not a list in {json_path}")
    return set(values)


def process_directory(
    input_dir: str,
    candidate_dir: str,
    min_pause: float = 0.25,
    file_ids_filter: Optional[set] = None,
) -> pd.DataFrame:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".textgrid")]
    if file_ids_filter:
        files = [f for f in files if Path(f).stem in file_ids_filter]

    refiner = ClassifierPauseRefiner(candidate_dir=candidate_dir, min_pause=min_pause)
    rows: List[Dict] = []
    for fname in sorted(files):
        path = os.path.join(input_dir, fname)
        try:
            tg_reader = TextGridReader(path)
            calc = CAFCalculator(tg_reader=tg_reader, min_pause=min_pause, refiner=refiner)
            m = calc.calculate_all()
            m["file"] = fname
            rows.append(m)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            rows.append({"file": fname, "error": str(e)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="CAF calculator with classifier-guided pause splitting")
    ap.add_argument("input", help="Input TextGrid file or directory")
    ap.add_argument("--candidate-dir", required=True, help="Per-file classifier candidate CSV directory")
    ap.add_argument("--min-pause", type=float, default=0.25)
    ap.add_argument("--output", "-o", help="Output CSV path")
    ap.add_argument("--file-list-json", help="Optional JSON file containing file_id list")
    ap.add_argument("--file-list-key", default="all_selected", help="JSON key for file list")
    args = ap.parse_args()

    file_filter = _load_file_id_filter(args.file_list_json, args.file_list_key)

    if os.path.isdir(args.input):
        df = process_directory(
            input_dir=args.input,
            candidate_dir=args.candidate_dir,
            min_pause=args.min_pause,
            file_ids_filter=file_filter,
        )
    else:
        tg_reader = TextGridReader(args.input)
        refiner = ClassifierPauseRefiner(candidate_dir=args.candidate_dir, min_pause=args.min_pause)
        calc = CAFCalculator(tg_reader=tg_reader, min_pause=args.min_pause, refiner=refiner)
        out = calc.calculate_all()
        out["file"] = os.path.basename(args.input)
        df = pd.DataFrame([out])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")

    main_cols = ["file", "AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]
    cols = [c for c in main_cols if c in df.columns]
    print("\n" + "=" * 100)
    print("CAF MEASURES (classifier-guided pause refinement)")
    print("=" * 100)
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
