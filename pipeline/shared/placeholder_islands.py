#!/usr/bin/env python
"""
Add the speech that MFA found inside pauses to the pause-refinement islands.

Stage 2 aligns the transcript with a filler placeholder in every gap between
words. Where the gap holds speech the ASR did not transcribe (mostly fillers,
also word edges and missed words), MFA stretches the placeholder over it;
over silence the placeholder stays short. Stage 2 records where each
placeholder landed (`filler_augmentation.placeholder_spans` in the Stage 2
JSON).

This script copies each Stage 4 per-file CSV and adds one row per placeholder
span of at least --min-dur seconds (pred_filler=1, source=placeholder_span).
The UF calculators then count that time as phonation, exactly as they count a
Stage 4 filler island. Files with nothing to add are copied unchanged; with
--placeholder-spans off (and classifier islands on) the output equals the
Stage 4 output. It also runs
when Stage 4 was skipped, so the spans are used even without the Stage 4
weights.

The 0.10 s default was chosen on the paper's 40 Japanese recordings
(leave-one-file-out picked it in 40 of 40 folds), so it is in-sample for
that corpus. The runner turns the spans on for both languages. For English
it also turns the classifier islands off (--classifier-islands off): spans
alone matched the reference better than spans plus the classifier, which
lowered English AR.

Usage (the runner passes these):
  python placeholder_islands.py \
    --file-list-json <file_list.json> --asr-json-dir <asr>/json \
    --stage4-dir <gap_candidates>/per_file --out-dir <islands> \
    [--min-dur 0.10] [--placeholder-spans on|off] [--classifier-islands on|off]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Column order of the Stage 4 per-file CSVs (postprocess_vad_filler_classifier.py).
STAGE4_COLUMNS = [
    "file_id", "gap_index", "island_index", "gap_start", "gap_end", "gap_duration",
    "vad_occupancy", "n_voiced_islands", "island_duration", "candidate_start",
    "candidate_end", "prob_filler", "pred_filler", "skip_reason",
]


def load_spans(json_path: Path, min_dur: float) -> Tuple[List[Tuple[float, float]], int]:
    """Return (kept spans, number of spans recorded) from one Stage 2 JSON."""
    if not json_path.exists():
        return [], 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    spans = (data.get("filler_augmentation") or {}).get("placeholder_spans") or []
    kept = [(float(s), float(e)) for _, s, e in spans if float(e) - float(s) >= min_dur - 1e-9]
    return kept, len(spans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-list-json", type=Path, required=True)
    ap.add_argument("--file-list-key", type=str, default="all_selected")
    ap.add_argument("--asr-json-dir", type=Path, required=True, help="Stage 2 JSON folder (<asr>/json)")
    ap.add_argument("--stage4-dir", type=Path, required=True, help="Stage 4 per_file folder (may be empty)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-dur", type=float, default=0.10, help="Shortest placeholder span kept (s)")
    ap.add_argument("--placeholder-spans", choices=["on", "off"], default="on")
    ap.add_argument("--classifier-islands", choices=["on", "off"], default="on",
                    help="Keep the Stage 4 classifier's islands (off: placeholder spans only)")
    args = ap.parse_args()

    use_spans = args.placeholder_spans == "on"
    use_stage4 = args.classifier_islands == "on"
    per_file_dir = args.out_dir / "per_file"
    shutil.rmtree(per_file_dir, ignore_errors=True)
    per_file_dir.mkdir(parents=True, exist_ok=True)

    file_ids = [str(x) for x in json.loads(args.file_list_json.read_text(encoding="utf-8"))[args.file_list_key]]
    rows = []
    for fid in file_ids:
        src = args.stage4_dir / f"{fid}_vad_classifier.csv"
        dst = per_file_dir / src.name
        if not use_stage4:
            src = per_file_dir / "_no_stage4" / src.name  # never exists
        kept, n_spans = load_spans(args.asr_json_dir / f"{fid}.json", args.min_dur) if use_spans else ([], 0)
        if not kept:
            if src.exists():
                shutil.copyfile(src, dst)
        else:
            # Stage 4 rows are kept as text, exactly as written.
            df = (pd.read_csv(src, dtype=str, keep_default_na=False) if src.exists()
                  else pd.DataFrame(columns=STAGE4_COLUMNS))
            df["source"] = "stage4"
            add = pd.DataFrame({
                "file_id": fid,
                "gap_index": "-1",
                "island_index": "-1",
                "island_duration": [repr(round(e - s, 4)) for s, e in kept],
                "candidate_start": [repr(s) for s, _ in kept],
                "candidate_end": [repr(e) for _, e in kept],
                "pred_filler": "1",
                "source": "placeholder_span",
            })
            pd.concat([df, add], ignore_index=True).fillna("").to_csv(dst, index=False)
        rows.append({
            "file_id": fid,
            "stage4_csv": src.exists(),
            "n_placeholder_spans": n_spans,
            "n_spans_kept": len(kept),
            "seconds_kept": round(sum(e - s for s, e in kept), 3),
        })

    summary = {
        "placeholder_spans": "on" if use_spans else "off",
        "classifier_islands": "on" if use_stage4 else "off",
        "min_dur": args.min_dur,
        "n_files": len(rows),
        "n_spans_kept": sum(r["n_spans_kept"] for r in rows),
        "seconds_kept": round(sum(r["seconds_kept"] for r in rows), 3),
        "files": rows,
    }
    (args.out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Islands: placeholder spans {summary['placeholder_spans']}, "
          f"{summary['n_spans_kept']} spans kept ({summary['seconds_kept']} s) in {len(rows)} files")


if __name__ == "__main__":
    main()
