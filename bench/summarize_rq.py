#!/usr/bin/env python3
"""
Side-by-side RQ1-RQ3 headline statistics for several runs of run_rq_variant.sh.

  summarize_rq.py --lang en --run paper=DIR --run mfa=DIR --run ctc=DIR [--out CSV]

DIR is an output folder of run_rq_variant.sh (or analysis/main_results/<lang>
of the analysis repository, for the published values).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def stats(run_dir: Path) -> dict[str, float]:
    out = {}
    for rq in ("rq1", "rq2"):
        for row in read(run_dir / rq / f"{rq}_kappa_cis.csv"):
            if row["subset"] == "Overall":
                out[f"{rq.upper()} kappa"] = float(row["kappa"])
    for row in read(run_dir / "rq3" / "rq3_overall_cis.csv"):
        out[f"RQ3 {row['measure']} r"] = float(row["pearson_r"])
        out[f"RQ3 {row['measure']} ICC"] = float(row["icc_2_1"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        name, d = spec.split("=", 1)
        runs[name] = stats(Path(d))
    names = list(runs)
    keys = list(dict.fromkeys(k for r in runs.values() for k in r))

    base = names[0]
    header = ["statistic", *names, *[f"{n}-{base}" for n in names[1:]]]
    rows = []
    for k in keys:
        vals = [runs[n].get(k) for n in names]
        diffs = [None if v is None or vals[0] is None else v - vals[0] for v in vals[1:]]
        rows.append([k, *vals, *diffs])

    fmt = lambda v: "" if v is None else f"{v:.3f}" if abs(v) >= 0.0005 or v == 0 else f"{v:.4f}"
    print(f"[{args.lang}]  " + "  ".join(f"{h:>14s}" for h in header))
    for r in rows:
        print(f"[{args.lang}]  {r[0]:>14s}  " + "  ".join(f"{fmt(v):>14s}" for v in r[1:]))
    if args.out:
        with args.out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


if __name__ == "__main__":
    main()
