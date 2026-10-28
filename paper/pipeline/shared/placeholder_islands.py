#!/usr/bin/env python
"""
The pause-refinement islands of paper mode: the Stage 4 classifier's islands.

The paper's UF calculators read the Stage 4 per-file CSVs. This script copies
them into <out-dir>/per_file, where the runner's UF stage reads them, and
writes SUMMARY.json; with Stage 4 skipped (no weights) it copies nothing, which
is the paper's Stage 4 ablation. Stage 2 records no placeholder spans in paper
mode, so none are added: --placeholder-spans and --classifier-islands are
accepted, as the developed pipeline's script takes them, and ignored.

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-list-json", type=Path, required=True)
    ap.add_argument("--file-list-key", type=str, default="all_selected")
    ap.add_argument("--asr-json-dir", type=Path, required=True, help="Stage 2 JSON folder (<asr>/json)")
    ap.add_argument("--stage4-dir", type=Path, required=True, help="Stage 4 per_file folder (may be empty)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-dur", type=float, default=0.10, help="Recorded in SUMMARY.json only")
    ap.add_argument("--placeholder-spans", choices=["on", "off"], default="on", help="Ignored (always off)")
    ap.add_argument("--classifier-islands", choices=["on", "off"], default="on", help="Ignored (always on)")
    args = ap.parse_args()

    per_file_dir = args.out_dir / "per_file"
    shutil.rmtree(per_file_dir, ignore_errors=True)
    per_file_dir.mkdir(parents=True, exist_ok=True)

    file_ids = [str(x) for x in json.loads(args.file_list_json.read_text(encoding="utf-8"))[args.file_list_key]]
    rows = []
    for fid in file_ids:
        src = args.stage4_dir / f"{fid}_vad_classifier.csv"
        dst = per_file_dir / src.name
        if src.exists():
            shutil.copyfile(src, dst)
        rows.append({
            "file_id": fid,
            "stage4_csv": src.exists(),
            "n_placeholder_spans": 0,
            "n_spans_kept": 0,
            "seconds_kept": 0,
        })

    summary = {
        "placeholder_spans": "off",
        "classifier_islands": "on",
        "min_dur": args.min_dur,
        "n_files": len(rows),
        "n_spans_kept": 0,
        "seconds_kept": 0,
        "files": rows,
    }
    (args.out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Islands: placeholder spans {summary['placeholder_spans']}, "
          f"{summary['n_spans_kept']} spans kept ({summary['seconds_kept']} s) in {len(rows)} files")


if __name__ == "__main__":
    main()
