#!/usr/bin/env bash
# Run the paper's RQ1-RQ3 analysis on one aligner variant.
#
#   run_rq_variant.sh <analysis_repo> <lang> <variant_dir> <out_dir>
#
# Same commands as reproduce_macos.sh in the analysis repository, with the
# automatic side (auto_textgrids_clean, auto_clauses, islands) taken from
# <variant_dir> as written by aligner_benchmark.py downstream. The
# reference side is the analysis repository's data/<lang>/reference_textgrids.
# Set PYTHON to the analysis environment's interpreter.
#
# Optional:
#   CANDIDATES=<dir>   pause-refinement islands for the automatic measures
#                      (default <variant_dir>/islands/per_file, or
#                      <variant_dir>/stage4_candidates/per_file for runs made
#                      before the islands stage)
#   CALC_REPO=<dir>    score the automatic side with this tool repository's
#                      calculators instead of the analysis repository's;
#                      the Japanese reference side then counts morae the same way
#                      (ja_reference_measures.py)
#   CALC_PYTHON=<python>  interpreter for the Japanese measures (default PYTHON);
#                      it needs SudachiPy, e.g. the analysis environment's or ja-electra310's
# Also writes rq1/rq1_midpoint.csv (RQ1 with midpoint projection, rq1_midpoint.py).
set -euo pipefail
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANA="$(cd "$1" && pwd)"; LANG_="$2"; V="$(cd "$3" && pwd)"; OUT="$4"
PYTHON="${PYTHON:-python3}"
CALC_PYTHON="${CALC_PYTHON:-$PYTHON}"
mkdir -p "$OUT"/{rq1,rq2,rq3}
OUT="$(cd "$OUT" && pwd)"
D="$ANA/data/$LANG_"
if [ -z "${CANDIDATES:-}" ]; then
  CANDIDATES="$V/islands/per_file"
  [ -d "$CANDIDATES" ] || CANDIDATES="$V/stage4_candidates/per_file"
fi
CANDIDATES="$(cd "$CANDIDATES" && pwd)"
if [ -n "${CALC_REPO:-}" ]; then
  CALC_EN="$(cd "$CALC_REPO" && pwd)/pipeline/en/caf_calculator_vad_classifier.py"
  CALC_JA="$(cd "$CALC_REPO" && pwd)/pipeline/ja/caf_calculator_ja_gap_classifier.py"
  MANUAL_JA=("$BENCH/ja_reference_measures.py" "$CALC_JA")
else
  CALC_EN="$ANA/code/en/caf_calculator_vad_classifier.py"
  CALC_JA="$ANA/code/ja/caf_calculator_ja_gap_classifier.py"
  MANUAL_JA=(code/ja/caf_calculator_ja.py)
fi
cd "$ANA"

if [ "$LANG_" = en ]; then
  "$PYTHON" code/en/run_rq1_textgrid_vs_textgrid.py \
    --source-textgrid-dir "$V/auto_textgrids_clean" --gold-textgrid-dir "$D/reference_textgrids" \
    --auto-textgrid-dir "$V/auto_clauses" --file-list-json "$D/file_list_40.json" --file-list-key all_selected \
    --output-csv "$OUT/rq1/rq1_textgrid_vs_textgrid.csv"
  "$PYTHON" code/en/run_rq2_pause_location_textgrid.py \
    --manual-textgrid-dir "$D/reference_textgrids" --auto-textgrid-dir "$V/auto_clauses" \
    --file-list-json "$D/file_list_40.json" --file-list-key all_selected \
    --output-csv "$OUT/rq2/rq2_pause_location_textgrid_all40.csv" \
    --summary-csv "$OUT/rq2/rq2_pause_location_textgrid_summary_all40.csv"
  "$PYTHON" code/en/caf_calculator.py "$D/reference_textgrids" \
    --file-list-json "$D/file_list_40.json" --file-list-key all_selected \
    --output "$OUT/rq3/manual_caf_results_textgrid.csv"
  "$PYTHON" "$CALC_EN" "$V/auto_clauses" \
    --candidate-dir "$CANDIDATES" \
    --file-list-json "$D/file_list_40.json" --file-list-key all_selected \
    --output "$OUT/rq3/auto_caf_vad_classifier_all40.csv"
  "$PYTHON" code/en/run_rq3_vad_classifier_correlation_en.py \
    --auto-clf-csv "$OUT/rq3/auto_caf_vad_classifier_all40.csv" \
    --manual-csv "$OUT/rq3/manual_caf_results_textgrid.csv" \
    --file-list-json "$D/file_list_40.json" --out-dir "$OUT/rq3"
else
  "$PYTHON" code/ja/run_rq1_textgrid_vs_textgrid_ja.py \
    --source-textgrid-dir "$V/auto_textgrids_clean" --gold-textgrid-dir "$D/reference_textgrids" \
    --auto-textgrid-dir "$V/auto_clauses" --output-csv "$OUT/rq1/rq1_v102_vs_manual20_textgrid.csv"
  "$PYTHON" code/ja/run_rq2_pause_location_textgrid_ja.py \
    --manual-textgrid-dir "$D/reference_textgrids" --auto-textgrid-dir "$V/auto_clauses" \
    --output-csv "$OUT/rq2/rq2_pause_location_textgrid_all40.csv" \
    --summary-csv "$OUT/rq2/rq2_pause_location_textgrid_summary_all40.csv"
  "$CALC_PYTHON" "${MANUAL_JA[@]}" "$D/reference_textgrids" \
    --output "$OUT/rq3/manual_caf_results_textgrid.csv"
  "$CALC_PYTHON" "$CALC_JA" "$V/auto_clauses" \
    --candidate-dir "$CANDIDATES" \
    --file-list-json "$D/file_list_40.json" --file-list-key all_selected \
    --output "$OUT/rq3/auto_caf_vad_classifier_all40.csv"
  "$PYTHON" code/ja/run_rq3_vad_classifier_correlation_ja.py \
    --clf-auto-csv "$OUT/rq3/auto_caf_vad_classifier_all40.csv" \
    --manual-csv "$OUT/rq3/manual_caf_results_textgrid.csv" \
    --out-summary "$OUT/rq3/rq3_vad_classifier_summary_textgrid.csv"
fi

for RQ in rq1 rq2 rq3; do
  cp "$ANA/analysis/main_results/$LANG_/$RQ/compute_${RQ}_"*.py "$OUT/$RQ/"
  ( cd "$OUT/$RQ" && "$PYTHON" compute_${RQ}_*.py )
done

"$PYTHON" "$BENCH/rq1_midpoint.py" --analysis "$ANA" --lang "$LANG_" --variant "$V" \
  --out "$OUT/rq1/rq1_midpoint.csv"
