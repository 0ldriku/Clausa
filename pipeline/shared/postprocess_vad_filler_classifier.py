#!/usr/bin/env python
"""
Postprocess ASR outputs with VAD-like gap detection + acoustic filler classifier.

This script is intentionally separate from ASR/MFA/segment scripts.
It reads existing ASR JSON word timings and audio, detects ASR gaps with speech,
scores candidates with a trained filler classifier, and writes analysis outputs.

Usage (the runner passes these, plus --max-gap 3.0 as in the reported run):
  python postprocess_vad_filler_classifier.py \
    --file-list-json <file_list.json> --audio-dir <audio folder> \
    --asr-json-dir <asr>/json --model-path <model.pt> --out-dir <gap_candidates> \
    --textgrid-dir <clauses> --out-textgrid-dir <patched_textgrids>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from praatio import textgrid
from praatio.data_classes.interval_tier import IntervalTier
from praatio.utilities.constants import Interval

_THIS = Path(__file__).resolve()
for _p in [_THIS.parent, *_THIS.parents]:
    if (_p / "shared").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from shared.filler_classifier.filler_model_inference import FillerProbabilityScorer


@dataclass
class FileStats:
    file_id: str
    n_words: int
    n_gap_candidates: int
    n_scored: int
    n_pred_fillers: int
    max_prob: float
    mean_prob: float


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


def load_asr_word_intervals(json_path: Path) -> List[Tuple[float, float, str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    words = data.get("words", [])
    out: List[Tuple[float, float, str]] = []
    for w in words:
        token = str(w.get("word", "")).strip()
        try:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", 0.0))
        except Exception:
            continue
        if token and end > start:
            out.append((start, end, token))
    out.sort(key=lambda x: x[0])
    return out


def merge_intervals(intervals: Sequence[Tuple[float, float]], merge_gap: float = 0.03) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + merge_gap:
            cur_e = max(cur_e, e)
            continue
        merged.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _is_pause_label(label: str) -> bool:
    return label.strip().lower() in PAUSE_MARKERS


def _find_word_tier_name(tg: textgrid.Textgrid, hint: str | None) -> str:
    if hint and hint in tg.tierNames:
        return hint
    for name in ["words", "word", "Word", "Words", "単語", "transcription"]:
        if name in tg.tierNames:
            return name
    return tg.tierNames[0]


def _patch_textgrid_with_fillers(
    in_tg_path: Path,
    out_tg_path: Path,
    filler_intervals: List[Tuple[float, float]],
    filler_label: str,
    word_tier_hint: str | None,
) -> Dict[str, int]:
    tg = textgrid.openTextgrid(str(in_tg_path), includeEmptyIntervals=True)
    word_tier_name = _find_word_tier_name(tg, word_tier_hint)
    word_tier = tg.getTier(word_tier_name)

    if filler_intervals:
        filler_intervals = merge_intervals(sorted(filler_intervals, key=lambda x: x[0]), merge_gap=0.02)

    new_entries: List[Interval] = []
    n_inserted = 0
    n_touched_pauses = 0
    fidx = 0

    for entry in word_tier.entries:
        s0 = float(entry.start)
        e0 = float(entry.end)
        lab = entry.label
        if e0 <= s0:
            continue

        while fidx < len(filler_intervals) and filler_intervals[fidx][1] <= s0:
            fidx += 1

        if not _is_pause_label(lab):
            new_entries.append(entry)
            continue

        overlaps: List[Tuple[float, float]] = []
        j = fidx
        while j < len(filler_intervals):
            fs, fe = filler_intervals[j]
            if fs >= e0:
                break
            if fe > s0:
                overlaps.append((max(s0, fs), min(e0, fe)))
            j += 1

        if not overlaps:
            new_entries.append(entry)
            continue

        n_touched_pauses += 1
        overlaps = merge_intervals(overlaps, merge_gap=0.0)
        cursor = s0
        for fs, fe in overlaps:
            if fs - cursor > 1e-4:
                new_entries.append(Interval(cursor, fs, lab))
            if fe - fs > 1e-4:
                new_entries.append(Interval(fs, fe, filler_label))
                n_inserted += 1
            cursor = max(cursor, fe)
        if e0 - cursor > 1e-4:
            new_entries.append(Interval(cursor, e0, lab))

    rebuilt: List[Interval] = []
    for iv in new_entries:
        if rebuilt and abs(iv.start - rebuilt[-1].end) < 1e-6 and iv.label == rebuilt[-1].label:
            rebuilt[-1] = Interval(rebuilt[-1].start, iv.end, iv.label)
        else:
            rebuilt.append(iv)

    new_word_tier = IntervalTier(
        word_tier_name,
        rebuilt,
        minT=word_tier.minTimestamp,
        maxT=word_tier.maxTimestamp,
    )
    tg.replaceTier(word_tier_name, new_word_tier, reportingMode="error")
    out_tg_path.parent.mkdir(parents=True, exist_ok=True)
    tg.save(str(out_tg_path), format="long_textgrid", includeBlankSpaces=True, reportingMode="error")
    return {
        "n_pred_intervals": len(filler_intervals),
        "n_inserted_intervals": n_inserted,
        "n_touched_pauses": n_touched_pauses,
    }


def build_asr_gaps(coverage: Sequence[Tuple[float, float]], duration: float, min_gap: float = 0.12) -> List[Tuple[float, float]]:
    gaps: List[Tuple[float, float]] = []
    prev = 0.0
    for s, e in coverage:
        if s - prev >= min_gap:
            gaps.append((prev, s))
        prev = max(prev, e)
    if duration - prev >= min_gap:
        gaps.append((prev, duration))
    return gaps


def merge_short_gaps(intervals: Sequence[Tuple[int, int]], gap_frames: int) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    out: List[Tuple[int, int]] = []
    s0, e0 = intervals[0]
    for s1, e1 in intervals[1:]:
        if s1 - e0 <= gap_frames:
            e0 = e1
            continue
        out.append((s0, e0))
        s0, e0 = s1, e1
    out.append((s0, e0))
    return out


def detect_voiced_islands(
    y: np.ndarray,
    sr: int,
    top_db: float,
    min_voiced: float,
    merge_gap: float,
) -> List[Tuple[float, float]]:
    raw = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    if len(raw) == 0:
        return []
    merged = merge_short_gaps([(int(s), int(e)) for s, e in raw], gap_frames=int(merge_gap * sr))
    out: List[Tuple[float, float]] = []
    for s, e in merged:
        dur = (e - s) / float(sr)
        if dur >= min_voiced:
            out.append((s / float(sr), e / float(sr)))
    return out


def extract_features(y: np.ndarray, sr: int) -> np.ndarray | None:
    if len(y) < int(0.03 * sr):
        return None
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=128)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=128)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=128)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=128)

    feats: List[float] = [
        float(len(y) / float(sr)),
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(rolloff)),
        float(np.std(rolloff)),
    ]
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())
    return np.asarray(feats, dtype=np.float32)


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")


def _find_audio(audio_dir: Path, file_id: str) -> Path:
    for ext in AUDIO_EXTS:
        candidate = audio_dir / f"{file_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Audio not found for {file_id} under {audio_dir}")


def process_one(
    file_id: str,
    audio_dir: Path,
    asr_json_dir: Path,
    scorer: FillerProbabilityScorer,
    target_sr: int,
    min_gap: float,
    max_gap: float,
    gap_only: bool,
    vad_top_db: float,
    vad_min_voiced: float,
    vad_merge_gap: float,
    min_occupancy: float,
    max_occupancy: float,
    threshold: float,
) -> Tuple[pd.DataFrame, FileStats]:
    audio_path = _find_audio(audio_dir, file_id)
    json_path = asr_json_dir / f"{file_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"ASR JSON not found: {json_path}")

    words = load_asr_word_intervals(json_path)
    if not words:
        return pd.DataFrame(), FileStats(file_id, 0, 0, 0, 0, np.nan, np.nan)

    y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    duration = len(y) / float(sr)
    coverage = merge_intervals([(s, e) for s, e, _ in words], merge_gap=0.03)
    gaps = build_asr_gaps(coverage, duration=duration, min_gap=min_gap)

    rows: List[Dict] = []
    for i, (gs, ge) in enumerate(gaps):
        gap_dur = ge - gs
        if gap_dur > max_gap:
            continue

        s_idx = max(0, int(round(gs * sr)))
        e_idx = min(len(y), int(round(ge * sr)))
        if e_idx <= s_idx:
            continue
        seg = y[s_idx:e_idx]
        if gap_only:
            islands_local = [(0.0, gap_dur)]
            occupancy = 1.0
        else:
            islands_local = detect_voiced_islands(
                seg,
                sr=sr,
                top_db=vad_top_db,
                min_voiced=vad_min_voiced,
                merge_gap=vad_merge_gap,
            )

            voiced_total = float(sum(e - s for s, e in islands_local))
            occupancy = voiced_total / gap_dur if gap_dur > 0 else 0.0
            keep_by_vad = (occupancy >= min_occupancy) and (occupancy <= max_occupancy)

            if not keep_by_vad:
                rows.append(
                    {
                        "file_id": file_id,
                        "gap_index": i,
                        "island_index": -1,
                        "gap_start": round(gs, 4),
                        "gap_end": round(ge, 4),
                        "gap_duration": round(gap_dur, 4),
                        "vad_occupancy": round(occupancy, 4),
                        "n_voiced_islands": len(islands_local),
                        "island_duration": None,
                        "candidate_start": None,
                        "candidate_end": None,
                        "prob_filler": None,
                        "pred_filler": 0,
                        "skip_reason": "vad_occupancy_out_of_range",
                    }
                )
                continue

            if not islands_local:
                rows.append(
                    {
                        "file_id": file_id,
                        "gap_index": i,
                        "island_index": -1,
                        "gap_start": round(gs, 4),
                        "gap_end": round(ge, 4),
                        "gap_duration": round(gap_dur, 4),
                        "vad_occupancy": round(occupancy, 4),
                        "n_voiced_islands": 0,
                        "island_duration": None,
                        "candidate_start": None,
                        "candidate_end": None,
                        "prob_filler": None,
                        "pred_filler": 0,
                        "skip_reason": "no_voiced_island",
                    }
                )
                continue

        for island_idx, (l0, l1) in enumerate(islands_local):
            cand_start = gs + l0
            cand_end = gs + l1
            prob = None
            pred = 0
            reason = ""

            c0 = max(0, int(round(cand_start * sr)))
            c1 = min(len(y), int(round(cand_end * sr)))
            clip = y[c0:c1]
            prob = scorer.predict_proba(clip, sr=sr)
            if prob is None:
                reason = "feature_extraction_failed"
            else:
                pred = int(prob >= threshold)

            rows.append(
                {
                    "file_id": file_id,
                    "gap_index": i,
                    "island_index": island_idx,
                    "gap_start": round(gs, 4),
                    "gap_end": round(ge, 4),
                    "gap_duration": round(gap_dur, 4),
                    "vad_occupancy": round(occupancy, 4),
                    "n_voiced_islands": len(islands_local),
                    "island_duration": round(l1 - l0, 4),
                    "candidate_start": round(cand_start, 4),
                    "candidate_end": round(cand_end, 4),
                    "prob_filler": None if prob is None else round(prob, 6),
                    "pred_filler": pred,
                    "skip_reason": reason,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, FileStats(file_id, len(words), 0, 0, 0, np.nan, np.nan)
    scored = df["prob_filler"].notna()
    probs = df.loc[scored, "prob_filler"]
    return df, FileStats(
        file_id=file_id,
        n_words=len(words),
        n_gap_candidates=int(len(df)),
        n_scored=int(scored.sum()),
        n_pred_fillers=int((df["pred_filler"] == 1).sum()),
        max_prob=float(probs.max()) if not probs.empty else np.nan,
        mean_prob=float(probs.mean()) if not probs.empty else np.nan,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-list-json", type=Path, required=True)
    ap.add_argument("--file-list-key", type=str, default="all_selected")
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--asr-json-dir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--textgrid-dir", type=Path, help="Optional source TextGrid dir to patch")
    ap.add_argument("--out-textgrid-dir", type=Path, help="Optional output dir for patched TextGrids")
    ap.add_argument("--textgrid-word-tier", type=str, default=None, help="Optional words tier name")
    ap.add_argument("--filler-label", type=str, default="<filler_speech>", help="Label inserted for predicted fillers")
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--min-gap", type=float, default=0.12)
    ap.add_argument("--max-gap", type=float, default=1.20)
    ap.add_argument(
        "--gap-only",
        action="store_true",
        help="Do not split gaps with VAD; score each ASR gap as one full candidate.",
    )
    ap.add_argument("--vad-top-db", type=float, default=30.0)
    ap.add_argument("--vad-min-voiced", type=float, default=0.08)
    ap.add_argument("--vad-merge-gap", type=float, default=0.06)
    ap.add_argument("--min-occupancy", type=float, default=0.20)
    ap.add_argument(
        "--max-occupancy",
        type=float,
        default=1.01,
        help="Allow 1.0 (+small numerical tolerance).",
    )
    ap.add_argument("--threshold", type=float, default=0.50)
    args = ap.parse_args()
    if (args.textgrid_dir is None) ^ (args.out_textgrid_dir is None):
        raise ValueError("Use both --textgrid-dir and --out-textgrid-dir together, or neither.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_file_dir = args.out_dir / "per_file"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    patched_stats_rows: List[Dict] = []
    if args.out_textgrid_dir:
        args.out_textgrid_dir.mkdir(parents=True, exist_ok=True)

    selected = json.loads(args.file_list_json.read_text(encoding="utf-8"))[args.file_list_key]
    file_ids = [str(x) for x in selected]

    scorer = FillerProbabilityScorer(args.model_path, target_sr=args.target_sr)

    all_rows: List[pd.DataFrame] = []
    stats_rows: List[Dict] = []
    failed: List[Dict] = []

    for i, fid in enumerate(file_ids, start=1):
        try:
            df, st = process_one(
                file_id=fid,
                audio_dir=args.audio_dir,
                asr_json_dir=args.asr_json_dir,
                scorer=scorer,
                target_sr=args.target_sr,
                min_gap=args.min_gap,
                max_gap=args.max_gap,
                gap_only=args.gap_only,
                vad_top_db=args.vad_top_db,
                vad_min_voiced=args.vad_min_voiced,
                vad_merge_gap=args.vad_merge_gap,
                min_occupancy=args.min_occupancy,
                max_occupancy=args.max_occupancy,
                threshold=args.threshold,
            )
            if not df.empty:
                df.to_csv(per_file_dir / f"{fid}_vad_classifier.csv", index=False)
                all_rows.append(df)
                if args.out_textgrid_dir:
                    in_tg = args.textgrid_dir / f"{fid}.TextGrid"
                    out_tg = args.out_textgrid_dir / f"{fid}.TextGrid"
                    if in_tg.exists():
                        keep = df[
                            (df["pred_filler"] == 1)
                            & df["candidate_start"].notna()
                            & df["candidate_end"].notna()
                        ]
                        pred_intervals = []
                        for _, r in keep.iterrows():
                            s = float(r["candidate_start"])
                            e = float(r["candidate_end"])
                            if e > s:
                                pred_intervals.append((s, e))
                        pst = _patch_textgrid_with_fillers(
                            in_tg_path=in_tg,
                            out_tg_path=out_tg,
                            filler_intervals=pred_intervals,
                            filler_label=args.filler_label,
                            word_tier_hint=args.textgrid_word_tier,
                        )
                        patched_stats_rows.append({"file_id": fid, **pst})
                    else:
                        patched_stats_rows.append(
                            {
                                "file_id": fid,
                                "n_pred_intervals": int((df["pred_filler"] == 1).sum()),
                                "n_inserted_intervals": 0,
                                "n_touched_pauses": 0,
                                "error": f"missing_textgrid:{in_tg}",
                            }
                        )
            stats_rows.append(st.__dict__)
            print(f"[{i:02d}/{len(file_ids)}] {fid}: gaps={st.n_gap_candidates}, scored={st.n_scored}, pred={st.n_pred_fillers}")
        except Exception as e:
            failed.append({"file_id": fid, "error": str(e)})
            print(f"[{i:02d}/{len(file_ids)}] {fid}: FAILED -> {e}")

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = args.out_dir / "SUMMARY_BY_FILE.csv"
    stats_df.to_csv(stats_csv, index=False)

    all_csv = args.out_dir / "ALL_CANDIDATES.csv"
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(all_csv, index=False)
    else:
        pd.DataFrame(columns=[
            "file_id", "gap_index", "gap_start", "gap_end", "gap_duration",
            "vad_occupancy", "n_voiced_islands", "candidate_start", "candidate_end",
            "prob_filler", "pred_filler", "skip_reason"
        ]).to_csv(all_csv, index=False)

    run = {
        "n_requested_files": len(file_ids),
        "n_success_files": len(stats_rows),
        "n_failed_files": len(failed),
        "total_gap_candidates": int(stats_df["n_gap_candidates"].sum()) if not stats_df.empty else 0,
        "total_scored": int(stats_df["n_scored"].sum()) if not stats_df.empty else 0,
        "total_pred_fillers": int(stats_df["n_pred_fillers"].sum()) if not stats_df.empty else 0,
        "mean_pred_fillers_per_file": float(stats_df["n_pred_fillers"].mean()) if not stats_df.empty else 0.0,
        "failed": failed,
        "textgrid_patch": {
            "enabled": bool(args.out_textgrid_dir),
            "source_dir": None if args.textgrid_dir is None else str(args.textgrid_dir),
            "output_dir": None if args.out_textgrid_dir is None else str(args.out_textgrid_dir),
            "filler_label": args.filler_label,
            "model_kind": scorer.model_kind,
            "files_patched": len(patched_stats_rows),
            "inserted_intervals_total": int(
                pd.to_numeric(pd.DataFrame(patched_stats_rows).get("n_inserted_intervals", pd.Series(dtype=float)), errors="coerce")
                .fillna(0)
                .sum()
            ) if patched_stats_rows else 0,
        },
        "params": {
            "gap_only": bool(args.gap_only),
            "min_gap": args.min_gap,
            "max_gap": args.max_gap,
            "vad_top_db": args.vad_top_db,
            "vad_min_voiced": args.vad_min_voiced,
            "vad_merge_gap": args.vad_merge_gap,
            "min_occupancy": args.min_occupancy,
            "max_occupancy": args.max_occupancy,
            "threshold": args.threshold,
        },
    }
    run_json = args.out_dir / "RUN_SUMMARY.json"
    run_json.write_text(json.dumps(run, indent=2), encoding="utf-8")

    if patched_stats_rows:
        pd.DataFrame(patched_stats_rows).to_csv(args.out_dir / "PATCHED_TEXTGRID_SUMMARY.csv", index=False)

    run_log = [
        "# VAD + Filler Classifier Postprocess Run Log",
        "",
        "## Inputs",
        f"- file_list_json: `{args.file_list_json}`",
        f"- file_list_key: `{args.file_list_key}`",
        f"- audio_dir: `{args.audio_dir}`",
        f"- asr_json_dir: `{args.asr_json_dir}`",
        f"- model_path: `{args.model_path}`",
        f"- model_kind: `{scorer.model_kind}`",
        "",
        "## Params",
        f"- min_gap: {args.min_gap}",
        f"- max_gap: {args.max_gap}",
        f"- gap_only: {bool(args.gap_only)}",
        f"- vad_top_db: {args.vad_top_db}",
        f"- vad_min_voiced: {args.vad_min_voiced}",
        f"- vad_merge_gap: {args.vad_merge_gap}",
        f"- min_occupancy: {args.min_occupancy}",
        f"- max_occupancy: {args.max_occupancy}",
        f"- threshold: {args.threshold}",
        "",
        "## Outputs",
        f"- per_file CSV dir: `{per_file_dir}`",
        f"- summary by file: `{stats_csv}`",
        f"- all candidates: `{all_csv}`",
        f"- run summary: `{run_json}`",
        f"- patched textgrids: `{args.out_textgrid_dir}`" if args.out_textgrid_dir else "- patched textgrids: (disabled)",
        "",
        "## Aggregates",
        f"- requested files: {run['n_requested_files']}",
        f"- success files: {run['n_success_files']}",
        f"- failed files: {run['n_failed_files']}",
        f"- total gap candidates: {run['total_gap_candidates']}",
        f"- total scored: {run['total_scored']}",
        f"- total predicted fillers: {run['total_pred_fillers']}",
    ]
    (args.out_dir / "RUN_LOG.md").write_text("\n".join(run_log) + "\n", encoding="utf-8")

    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
