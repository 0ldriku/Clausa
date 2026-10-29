# How the measures are computed

Clausa computes the nine measures from each recording's TextGrid in
`clauses/`: the words, the silences between them, and the clauses. The code is
`pipeline/en/caf_calculator_vad_classifier.py` and
`pipeline/ja/caf_calculator_ja_gap_classifier.py`.

## Formulas

| Measure | Formula |
|---|---|
| AR | syllables ÷ speaking time |
| SR | syllables ÷ span |
| MLR | mean number of syllables between two pauses |
| MCPR | mid-clause pauses ÷ syllables |
| ECPR | end-clause pauses ÷ syllables |
| PR | pauses ÷ syllables |
| MCPD | mean duration of mid-clause pauses |
| ECPD | mean duration of end-clause pauses |
| MPD | mean duration of all pauses |

- Japanese uses morae in place of syllables, counted from each word's kana
  reading, read in context with SudachiPy and its core dictionary (Takaoka et
  al., 2018). A small kana joins the kana before it; ッ, ン and ー count one
  each. Durations are in seconds.
- For MCPR, ECPR, MCPD and ECPD, a pause that crosses a clause boundary is
  split there. PR and MPD count it whole.
- For MLR, a silence of 0.25 s or longer ends a run even when speech was found
  inside it.
- AR and SR are rounded to 3 decimals, MLR to 2, the rates to 4, the durations
  to 3.

## Other columns

`uf_measures.csv` also gives the counts behind the measures:
`total_syllables` (`total_mora` in Japanese), `total_duration` (the span),
`phonation_time` (speaking time), `n_pauses`, `n_mid_clause_pauses`,
`n_end_clause_pauses`, `n_other_pauses` and `n_clauses`. `clf_pause_candidates`
and `clf_refined_pause_count` are the numbers of pauses before and after the
speech inside them is taken out; `clf_pauses_split` and
`clf_predicted_speech_islands` count the pauses split and the speech found.

## Reference

Takaoka, K., Hisamoto, S., Kawahara, N., Sakamoto, M., Uchida, Y., & Matsumoto,
Y. (2018). Sudachi: A Japanese tokenizer for business. In *Proceedings of the
Eleventh International Conference on Language Resources and Evaluation (LREC
2018)*. European Language Resources Association.
https://aclanthology.org/L18-1355/
