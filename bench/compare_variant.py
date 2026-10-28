#!/usr/bin/env python3
"""Paired comparison of a pipeline variant against a baseline, on the paper's 40+40 files.

  compare_variant.py --analysis <analysis repo> --lang en|ja --base <rq dir> --variant <rq dir> [--out csv]

<rq dir> is an output folder of run_rq_variant.sh (rq1/, rq2/, rq3/ with the
paper's per-file CSVs, plus rq1/rq1_midpoint.csv).

For every statistic (RQ1 kappa under the shipped and the midpoint projection,
RQ2 kappa, Pearson r and ICC(2,1) of the nine measures) it reports the
baseline and variant values, the delta, a paired file bootstrap (10,000
draws, the same resample for both runs), a paired speaker-cluster bootstrap,
the leave-one-file-out range of the delta and the change's own minimum
detectable effect (2.8 x SE).

A change passes on a statistic when the file CI and the speaker CI both
exclude 0 and the sign of the delta survives leaving out any one file.
Decide the primary statistic before running.

Speaker identity comes from the file name (ALLSSTAR and I-JAS names end in
_ST1/_ST2 or -ST1/-ST2); --speaker-regex changes that for other corpora.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CAF = ["AR", "SR", "MLR", "MCPR", "ECPR", "PR", "MCPD", "ECPD", "MPD"]
STATS = ["RQ1 kappa", "RQ1 kappa (midpoint)", "RQ2 kappa"] + [f"{m} {s}" for m in CAF for s in ("r", "ICC")]


def _fid(v) -> str:
    return str(v).replace(".TextGrid", "")


def _rq2_counts(row) -> list[int]:
    # derive_confusion_counts() of the paper's compute_rq2_kappa_cis.py
    n, agree, auto_mcp, gold_mcp = int(row.n_pauses), int(row.n_agree), int(row.auto_mcp), int(row.gold_mcp)
    tp = (agree - n + auto_mcp + gold_mcp) / 2.0
    return [round(tp), round(auto_mcp - tp), round(gold_mcp - tp), round(agree - tp)]


def load_run(rq_dir: Path, lang: str, files: list[str]) -> dict[str, np.ndarray | dict]:
    rq1_name = "rq1_textgrid_vs_textgrid.csv" if lang == "en" else "rq1_v102_vs_manual20_textgrid.csv"
    rq1 = pd.read_csv(rq_dir / "rq1" / rq1_name, encoding="utf-8-sig").assign(file=lambda d: d.file.map(_fid))
    mid = pd.read_csv(rq_dir / "rq1" / "rq1_midpoint.csv").assign(file=lambda d: d.file.map(_fid))
    rq2 = pd.read_csv(rq_dir / "rq2" / "rq2_pause_location_textgrid_all40.csv",
                      encoding="utf-8-sig").assign(file=lambda d: d.file.map(_fid)).set_index("file")
    auto = pd.read_csv(rq_dir / "rq3" / "auto_caf_vad_classifier_all40.csv").assign(file=lambda d: d.file.map(_fid))
    man = pd.read_csv(rq_dir / "rq3" / "manual_caf_results_textgrid.csv").assign(file=lambda d: d.file.map(_fid))
    cols = ["tp", "fp", "fn", "tn"]
    auto, man = auto.set_index("file"), man.set_index("file")
    return {
        "RQ1 kappa": rq1.set_index("file").loc[files, cols].to_numpy(float),
        "RQ1 kappa (midpoint)": mid.set_index("file").loc[files, cols].to_numpy(float),
        "RQ2 kappa": np.array([_rq2_counts(rq2.loc[f]) for f in files], float),
        "auto": {m: auto.loc[files, m].to_numpy(float) for m in CAF},
        "man": {m: man.loc[files, m].to_numpy(float) for m in CAF},
    }


def _kappa(c: np.ndarray) -> np.ndarray:
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    tot = tp + fp + fn + tn
    po = (tp + tn) / tot
    pe = ((tp + fn) / tot) * ((tp + fp) / tot) + ((fp + tn) / tot) * ((fn + tn) / tot)
    return (po - pe) / (1.0 - pe)


def _pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xm, ym = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    return (xm * ym).sum(-1) / np.sqrt((xm ** 2).sum(-1) * (ym ** 2).sum(-1))


def _icc21(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # ICC(2,1), absolute agreement; same algebra as compute_rq3_overall_cis.py
    n = x.shape[-1]
    gm = (x.sum(-1) + y.sum(-1)) / (2 * n)
    ssr = 2 * (((x + y) / 2 - gm[..., None]) ** 2).sum(-1)
    ssc = n * ((x.mean(-1) - gm) ** 2 + (y.mean(-1) - gm) ** 2)
    sse = ((x - gm[..., None]) ** 2 + (y - gm[..., None]) ** 2).sum(-1) - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc, sse / (n - 1)
    den = msr + mse + (2 / n) * (msc - mse)
    return np.divide(msr - mse, den, out=np.ones_like(msr), where=np.abs(den) > 1e-15)


def stats(run: dict, idx: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Statistics for each index array in idx (resamples may differ in length)."""
    out = {k: np.empty(len(idx)) for k in STATS}
    for b, ix in enumerate(idx):
        for k in ("RQ1 kappa", "RQ1 kappa (midpoint)", "RQ2 kappa"):
            out[k][b] = _kappa(run[k][ix].sum(0))
        for m in CAF:
            x, y = run["auto"][m][ix], run["man"][m][ix]
            out[f"{m} r"][b] = _pearson(x, y)
            out[f"{m} ICC"][b] = _icc21(x, y)
    return out


def compare(analysis: Path, lang: str, base: Path, variant: Path, *, draws: int = 10_000,
            seed: int = 20261020, speaker_regex: str = r"[-_]ST\d$") -> pd.DataFrame:
    files = json.loads((analysis / "data" / lang / "file_list_40.json").read_text())["all_selected"]
    n = len(files)
    P, V = load_run(base, lang, files), load_run(variant, lang, files)
    rng = np.random.default_rng(seed)
    file_idx = list(rng.integers(0, n, size=(draws, n)))
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(files):
        groups.setdefault(re.sub(speaker_regex, "", f), []).append(i)
    g = list(groups.values())
    spk_idx = [np.concatenate([g[j] for j in row]) for row in rng.integers(0, len(g), size=(draws, len(g)))]
    loo_idx = [np.delete(np.arange(n), i) for i in range(n)]

    full = [np.arange(n)]
    point_p, point_v = stats(P, full), stats(V, full)
    fb_p, fb_v = stats(P, file_idx), stats(V, file_idx)
    sb_p, sb_v = stats(P, spk_idx), stats(V, spk_idx)
    lo_p, lo_v = stats(P, loo_idx), stats(V, loo_idx)

    rows = []
    for k in STATS:
        delta = float(point_v[k][0] - point_p[k][0])
        fd, sd, ld = fb_v[k] - fb_p[k], sb_v[k] - sb_p[k], lo_v[k] - lo_p[k]
        f_lo, f_hi = np.nanpercentile(fd, [2.5, 97.5])
        s_lo, s_hi = np.nanpercentile(sd, [2.5, 97.5])
        se = float(np.nanstd(fd, ddof=1))
        sign_ok = bool(np.all(ld > 0)) if delta > 0 else bool(np.all(ld < 0)) if delta < 0 else False
        passes = (f_lo > 0 or f_hi < 0) and (s_lo > 0 or s_hi < 0) and sign_ok
        rows.append({
            "lang": lang, "statistic": k, "base": float(point_p[k][0]), "variant": float(point_v[k][0]),
            "delta": delta, "file_CI_lo": f_lo, "file_CI_hi": f_hi, "speaker_CI_lo": s_lo, "speaker_CI_hi": s_hi,
            "LOO_min": float(ld.min()), "LOO_max": float(ld.max()), "own_MDE_2.8SE": 2.8 * se,
            "verdict": ("better" if delta > 0 else "worse") if passes else "no clear change",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--lang", required=True, choices=["en", "ja"])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--variant", required=True, type=Path)
    ap.add_argument("--speaker-regex", default=r"[-_]ST\d$")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    df = compare(args.analysis, args.lang, args.base, args.variant, speaker_regex=args.speaker_regex)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.round(4).to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
