#!/usr/bin/env python
"""
Create coder-reviewable TextGrid views for gap/classifier filler candidates.

Why:
- CAF gap/classifier calculators read candidate CSV files.
- Human coders review TextGrids more reliably than CSV rows.
- This script bridges both directions:
  1) build   : candidate CSV -> multi-tier review TextGrid
  2) extract : edited review TextGrid tier -> candidate CSV

The output candidate CSV from `extract` is compatible with:
- pipeline/en/caf_calculator_vad_classifier.py
- pipeline/ja/caf_calculator_ja_gap_classifier.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from praatio import textgrid
from praatio.data_classes.interval_tier import IntervalTier
from praatio.utilities.constants import Interval


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

SUPPORTED_AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a"]


def _is_pause(label: str) -> bool:
    return str(label).strip().lower() in PAUSE_MARKERS


def _find_word_tier_name(tg: textgrid.Textgrid, hint: Optional[str] = None) -> str:
    if hint and hint in tg.tierNames:
        return hint
    for name in ["words", "word", "Word", "Words", "transcription"]:
        if name in tg.tierNames:
            return name
    return tg.tierNames[0]


def _merge_intervals(intervals: Sequence[Tuple[float, float]], merge_gap: float = 0.0) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    out: List[Tuple[float, float]] = []
    s0, e0 = intervals[0]
    for s1, e1 in intervals[1:]:
        if s1 <= e0 + merge_gap:
            e0 = max(e0, e1)
            continue
        out.append((s0, e0))
        s0, e0 = s1, e1
    out.append((s0, e0))
    return out


def _clean_nonoverlap(intervals: Sequence[Tuple[float, float, str]], min_t: float, max_t: float) -> List[Tuple[float, float, str]]:
    clean: List[Tuple[float, float, str]] = []
    cursor = min_t
    for s0, e0, lab in sorted(intervals, key=lambda x: (x[0], x[1])):
        s = max(float(s0), min_t)
        e = min(float(e0), max_t)
        if e <= s:
            continue
        if s < cursor:
            s = cursor
        if e <= s:
            continue
        clean.append((s, e, str(lab)))
        cursor = e
        if cursor >= max_t:
            break
    return clean


def _timeline_entries(min_t: float, max_t: float, labeled_intervals: Sequence[Tuple[float, float, str]]) -> List[Interval]:
    clean = _clean_nonoverlap(labeled_intervals, min_t=min_t, max_t=max_t)
    out: List[Interval] = []
    cursor = min_t
    for s, e, lab in clean:
        if s > cursor:
            out.append(Interval(cursor, s, ""))
        out.append(Interval(s, e, lab))
        cursor = e
    if cursor < max_t:
        out.append(Interval(cursor, max_t, ""))
    if not out:
        out = [Interval(min_t, max_t, "")]
    return out


def _patch_word_entries(word_tier: IntervalTier, filler_intervals: Sequence[Tuple[float, float]], filler_label: str) -> List[Interval]:
    ints = _merge_intervals(sorted(filler_intervals, key=lambda x: x[0]), merge_gap=0.02) if filler_intervals else []

    new_entries: List[Interval] = []
    n = len(ints)
    fidx = 0
    for entry in word_tier.entries:
        s0 = float(entry.start)
        e0 = float(entry.end)
        lab = str(entry.label)
        if e0 <= s0:
            continue

        while fidx < n and ints[fidx][1] <= s0:
            fidx += 1

        if not _is_pause(lab):
            new_entries.append(Interval(s0, e0, lab))
            continue

        overlaps: List[Tuple[float, float]] = []
        j = fidx
        while j < n:
            fs, fe = ints[j]
            if fs >= e0:
                break
            if fe > s0:
                overlaps.append((max(s0, fs), min(e0, fe)))
            j += 1

        if not overlaps:
            new_entries.append(Interval(s0, e0, lab))
            continue

        overlaps = _merge_intervals(overlaps, merge_gap=0.0)
        cursor = s0
        for fs, fe in overlaps:
            if fs - cursor > 1e-6:
                new_entries.append(Interval(cursor, fs, lab))
            if fe - fs > 1e-6:
                new_entries.append(Interval(fs, fe, filler_label))
            cursor = max(cursor, fe)
        if e0 - cursor > 1e-6:
            new_entries.append(Interval(cursor, e0, lab))

    merged: List[Interval] = []
    for iv in new_entries:
        if merged and abs(float(iv.start) - float(merged[-1].end)) < 1e-6 and iv.label == merged[-1].label:
            merged[-1] = Interval(float(merged[-1].start), float(iv.end), iv.label)
        else:
            merged.append(iv)
    return merged


def _load_file_ids_from_json(path: Path, key: str) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload[key]
    return [str(x) for x in values]


def _candidate_file_to_id(path: Path, suffix: str) -> Optional[str]:
    name = path.name
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def _safe_prob(v) -> str:
    try:
        x = float(v)
        return f"{x:.3f}"
    except Exception:
        return "NA"


def _copy_audio_if_present(fid: str, audio_dir: Path, out_audio_dir: Path) -> Optional[Path]:
    for ext in SUPPORTED_AUDIO_EXTS:
        src = audio_dir / f"{fid}{ext}"
        if src.exists():
            out_audio_dir.mkdir(parents=True, exist_ok=True)
            dst = out_audio_dir / src.name
            shutil.copy2(src, dst)
            return dst
    return None


def build_review_textgrids(args: argparse.Namespace) -> None:
    args.out_textgrid_dir.mkdir(parents=True, exist_ok=True)
    if args.out_audio_dir and args.audio_dir:
        args.out_audio_dir.mkdir(parents=True, exist_ok=True)

    requested_ids: Optional[List[str]] = None
    if args.file_list_json:
        requested_ids = _load_file_ids_from_json(args.file_list_json, args.file_list_key)

    candidate_files = sorted(args.candidate_dir.glob(f"*{args.candidate_suffix}"))
    by_id: Dict[str, Path] = {}
    for p in candidate_files:
        fid = _candidate_file_to_id(p, args.candidate_suffix)
        if fid:
            by_id[fid] = p

    file_ids = sorted(by_id.keys())
    if requested_ids is not None:
        requested = set(requested_ids)
        file_ids = [f for f in file_ids if f in requested]

    rows: List[Dict] = []
    for i, fid in enumerate(file_ids, start=1):
        row = {
            "file_id": fid,
            "source_textgrid": "",
            "candidate_csv": str(by_id[fid]),
            "review_textgrid": "",
            "n_candidates": 0,
            "n_predicted": 0,
            "audio_copied": "",
            "error": "",
        }
        try:
            src_tg = args.source_textgrid_dir / f"{fid}.TextGrid"
            row["source_textgrid"] = str(src_tg)
            if not src_tg.exists():
                raise FileNotFoundError(f"missing source textgrid: {src_tg}")

            df = pd.read_csv(by_id[fid])
            if "pred_filler" not in df.columns:
                df["pred_filler"] = 0

            valid = df[df["candidate_start"].notna() & df["candidate_end"].notna()].copy()
            cand_intervals: List[Tuple[float, float, str]] = []
            pred_intervals: List[Tuple[float, float]] = []
            pred_labeled: List[Tuple[float, float, str]] = []

            for _, r in valid.iterrows():
                s = float(r["candidate_start"])
                e = float(r["candidate_end"])
                if e <= s:
                    continue
                gap_idx = int(r["gap_index"]) if "gap_index" in r and pd.notna(r["gap_index"]) else -1
                isl_idx = int(r["island_index"]) if "island_index" in r and pd.notna(r["island_index"]) else -1
                pred = int(r["pred_filler"]) if pd.notna(r["pred_filler"]) else 0
                prob = _safe_prob(r["prob_filler"]) if "prob_filler" in r else "NA"
                dur = e - s
                lab = f"g{gap_idx}|i{isl_idx}|p={prob}|pred={pred}|d={dur:.2f}"
                cand_intervals.append((s, e, lab))
                if pred == 1:
                    pred_intervals.append((s, e))
                    pred_labeled.append((s, e, f"{args.filler_label}|p={prob}"))

            tg = textgrid.openTextgrid(str(src_tg), includeEmptyIntervals=True)
            word_tier_name = _find_word_tier_name(tg, args.textgrid_word_tier)
            word_tier = tg.getTier(word_tier_name)
            min_t = float(word_tier.minTimestamp)
            max_t = float(word_tier.maxTimestamp)

            original_entries = [
                Interval(float(e.start), float(e.end), str(e.label))
                for e in word_tier.entries
            ]
            patched_entries = _patch_word_entries(
                word_tier=word_tier,
                filler_intervals=pred_intervals,
                filler_label=args.filler_label,
            )

            out_tg = textgrid.Textgrid()
            out_tg.addTier(IntervalTier("words_original", original_entries, minT=min_t, maxT=max_t))
            out_tg.addTier(
                IntervalTier(
                    "cand_all",
                    _timeline_entries(min_t, max_t, cand_intervals),
                    minT=min_t,
                    maxT=max_t,
                )
            )
            out_tg.addTier(
                IntervalTier(
                    "cand_pred_auto",
                    _timeline_entries(min_t, max_t, pred_labeled),
                    minT=min_t,
                    maxT=max_t,
                )
            )
            out_tg.addTier(
                IntervalTier(
                    "review_edit",
                    _timeline_entries(
                        min_t,
                        max_t,
                        [(s, e, args.filler_label) for (s, e, _) in pred_labeled],
                    ),
                    minT=min_t,
                    maxT=max_t,
                )
            )
            out_tg.addTier(
                IntervalTier("words_patched_auto", patched_entries, minT=min_t, maxT=max_t)
            )

            dst = args.out_textgrid_dir / f"{fid}.TextGrid"
            out_tg.save(str(dst), format="long_textgrid", includeBlankSpaces=True, reportingMode="error")
            row["review_textgrid"] = str(dst)
            row["n_candidates"] = int(len(cand_intervals))
            row["n_predicted"] = int(len(pred_intervals))

            if args.audio_dir and args.out_audio_dir:
                copied = _copy_audio_if_present(fid, args.audio_dir, args.out_audio_dir)
                if copied is not None:
                    row["audio_copied"] = str(copied)

            print(f"[{i:03d}/{len(file_ids)}] {fid}: candidates={row['n_candidates']} pred={row['n_predicted']}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"[{i:03d}/{len(file_ids)}] {fid}: FAILED -> {exc}")
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = args.out_textgrid_dir / "REVIEW_BUILD_SUMMARY.csv"
    summary_df.to_csv(summary_csv, index=False)

    n_ok = int((summary_df["error"] == "").sum()) if not summary_df.empty else 0
    n_fail = int((summary_df["error"] != "").sum()) if not summary_df.empty else 0
    log_lines = [
        "# Candidate Review TextGrid Build Log",
        "",
        "## Inputs",
        f"- candidate_dir: `{args.candidate_dir}`",
        f"- source_textgrid_dir: `{args.source_textgrid_dir}`",
        f"- file_list_json: `{args.file_list_json}`" if args.file_list_json else "- file_list_json: (all candidate files)",
        f"- file_list_key: `{args.file_list_key}`",
        f"- candidate_suffix: `{args.candidate_suffix}`",
        f"- filler_label: `{args.filler_label}`",
        "",
        "## Outputs",
        f"- review_textgrid_dir: `{args.out_textgrid_dir}`",
        f"- review_summary_csv: `{summary_csv}`",
        f"- copied_audio_dir: `{args.out_audio_dir}`" if args.out_audio_dir else "- copied_audio_dir: (disabled)",
        "",
        "## Counts",
        f"- files_total: {len(file_ids)}",
        f"- files_success: {n_ok}",
        f"- files_failed: {n_fail}",
        f"- total_candidates: {int(summary_df['n_candidates'].fillna(0).sum()) if not summary_df.empty else 0}",
        f"- total_predicted: {int(summary_df['n_predicted'].fillna(0).sum()) if not summary_df.empty else 0}",
    ]
    (args.out_textgrid_dir / "RUN_LOG.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def _choose_review_tier(tg: textgrid.Textgrid, preferred: str) -> str:
    if preferred in tg.tierNames:
        return preferred
    for name in ["review_edit", "cand_pred_auto", "cand_pred", "candidate_pred"]:
        if name in tg.tierNames:
            return name
    raise ValueError(f"review tier not found: {preferred}")


def _list_review_file_ids(review_dir: Path, file_list_json: Optional[Path], file_list_key: str) -> List[str]:
    if file_list_json:
        return _load_file_ids_from_json(file_list_json, file_list_key)
    return sorted([p.stem for p in review_dir.glob("*.TextGrid")])


def extract_candidates_from_review(args: argparse.Namespace) -> None:
    args.out_candidate_dir.mkdir(parents=True, exist_ok=True)
    file_ids = _list_review_file_ids(args.review_textgrid_dir, args.file_list_json, args.file_list_key)

    rows: List[Dict] = []
    out_cols = [
        "file_id",
        "gap_index",
        "island_index",
        "gap_start",
        "gap_end",
        "gap_duration",
        "vad_occupancy",
        "n_voiced_islands",
        "island_duration",
        "candidate_start",
        "candidate_end",
        "prob_filler",
        "pred_filler",
        "skip_reason",
    ]

    for i, fid in enumerate(file_ids, start=1):
        meta = {
            "file_id": fid,
            "review_textgrid": "",
            "out_csv": "",
            "n_review_intervals": 0,
            "error": "",
        }
        try:
            in_tg = args.review_textgrid_dir / f"{fid}.TextGrid"
            meta["review_textgrid"] = str(in_tg)
            if not in_tg.exists():
                raise FileNotFoundError(f"missing review textgrid: {in_tg}")

            tg = textgrid.openTextgrid(str(in_tg), includeEmptyIntervals=True)
            tier_name = _choose_review_tier(tg, args.review_tier_name)
            tier = tg.getTier(tier_name)

            intervals: List[Tuple[float, float]] = []
            for e in tier.entries:
                lab = str(e.label).strip()
                s = float(e.start)
                t = float(e.end)
                if not lab:
                    continue
                if t <= s:
                    continue
                intervals.append((s, t))

            intervals = _merge_intervals(sorted(intervals, key=lambda x: x[0]), merge_gap=args.merge_gap)
            meta["n_review_intervals"] = int(len(intervals))

            out_rows: List[Dict] = []
            for j, (s, e) in enumerate(intervals):
                out_rows.append(
                    {
                        "file_id": fid,
                        "gap_index": j,
                        "island_index": 0,
                        "gap_start": round(s, 4),
                        "gap_end": round(e, 4),
                        "gap_duration": round(e - s, 4),
                        "vad_occupancy": 1.0,
                        "n_voiced_islands": 1,
                        "island_duration": round(e - s, 4),
                        "candidate_start": round(s, 4),
                        "candidate_end": round(e, 4),
                        "prob_filler": None,
                        "pred_filler": 1,
                        "skip_reason": "manual_review",
                    }
                )

            out_df = pd.DataFrame(out_rows, columns=out_cols)
            out_csv = args.out_candidate_dir / f"{fid}{args.candidate_suffix}"
            out_df.to_csv(out_csv, index=False)
            meta["out_csv"] = str(out_csv)
            print(f"[{i:03d}/{len(file_ids)}] {fid}: extracted={len(intervals)}")
        except Exception as exc:
            meta["error"] = str(exc)
            print(f"[{i:03d}/{len(file_ids)}] {fid}: FAILED -> {exc}")
        rows.append(meta)

    summary_df = pd.DataFrame(rows)
    summary_csv = args.out_candidate_dir / "REVIEW_EXTRACT_SUMMARY.csv"
    summary_df.to_csv(summary_csv, index=False)

    n_ok = int((summary_df["error"] == "").sum()) if not summary_df.empty else 0
    n_fail = int((summary_df["error"] != "").sum()) if not summary_df.empty else 0
    log_lines = [
        "# Candidate Review Extract Log",
        "",
        "## Inputs",
        f"- review_textgrid_dir: `{args.review_textgrid_dir}`",
        f"- review_tier_name: `{args.review_tier_name}`",
        f"- merge_gap: {args.merge_gap}",
        f"- file_list_json: `{args.file_list_json}`" if args.file_list_json else "- file_list_json: (all review textgrids)",
        f"- file_list_key: `{args.file_list_key}`",
        "",
        "## Outputs",
        f"- out_candidate_dir: `{args.out_candidate_dir}`",
        f"- summary_csv: `{summary_csv}`",
        "",
        "## Counts",
        f"- files_total: {len(file_ids)}",
        f"- files_success: {n_ok}",
        f"- files_failed: {n_fail}",
        f"- total_extracted_intervals: {int(summary_df['n_review_intervals'].fillna(0).sum()) if not summary_df.empty else 0}",
    ]
    (args.out_candidate_dir / "RUN_LOG.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def make_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build/extract candidate-review TextGrid artifacts.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build review TextGrids from candidate CSV + source TextGrid.")
    ap_build.add_argument("--candidate-dir", type=Path, required=True)
    ap_build.add_argument("--source-textgrid-dir", type=Path, required=True)
    ap_build.add_argument("--out-textgrid-dir", type=Path, required=True)
    ap_build.add_argument("--candidate-suffix", type=str, default="_vad_classifier.csv")
    ap_build.add_argument("--file-list-json", type=Path, default=None)
    ap_build.add_argument("--file-list-key", type=str, default="all_selected")
    ap_build.add_argument("--textgrid-word-tier", type=str, default=None)
    ap_build.add_argument("--filler-label", type=str, default="<filler_speech>")
    ap_build.add_argument("--audio-dir", type=Path, default=None)
    ap_build.add_argument("--out-audio-dir", type=Path, default=None)

    ap_extract = sub.add_parser("extract", help="Extract candidate CSVs from edited review TextGrid tier.")
    ap_extract.add_argument("--review-textgrid-dir", type=Path, required=True)
    ap_extract.add_argument("--out-candidate-dir", type=Path, required=True)
    ap_extract.add_argument("--review-tier-name", type=str, default="review_edit")
    ap_extract.add_argument("--merge-gap", type=float, default=0.0)
    ap_extract.add_argument("--candidate-suffix", type=str, default="_vad_classifier.csv")
    ap_extract.add_argument("--file-list-json", type=Path, default=None)
    ap_extract.add_argument("--file-list-key", type=str, default="all_selected")

    return ap


def main() -> None:
    ap = make_arg_parser()
    args = ap.parse_args()
    if args.cmd == "build":
        if (args.audio_dir is None) ^ (args.out_audio_dir is None):
            raise ValueError("Use both --audio-dir and --out-audio-dir together, or neither.")
        build_review_textgrids(args)
        return
    if args.cmd == "extract":
        extract_candidates_from_review(args)
        return
    raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
