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
  each. Numbers, in digits or kanji, are read by place (25: に・じゅう・ご) and
  before counters as they are said (四時 よじ, 一人 ひとり, 三日 みっか, 二十歳
  はたち), with the rules of the Open JTalk speech synthesizer (HTS Working
  Group, 2018). Durations are in seconds.
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

## Tables

Compute, and each run in `uf/` (not in paper mode), also writes what the
measures came from, for measures of your own in R or Excel: `uf_words.csv`,
`uf_pauses.csv` and `uf_clauses.csv`. `file` is the recording's TextGrid, as in `uf_measures.csv`;
times are in seconds. The files are UTF-8 with a BOM, so that Excel shows
Japanese (base R: `read.csv(..., fileEncoding = "UTF-8-BOM")`).

- `uf_words.csv`, one row per word: `word`, `start`, `end`, `disfluent` (1 if
  the disfluency tier marks it; empty without that tier), `clause` (its number
  in `uf_clauses.csv`) and `syllables` (`morae` in Japanese) as the measures
  count them.
- `uf_pauses.csv`, one row per pause the measures count: `pause` (its number),
  `start`, `end`, `duration`, `location` (mid-clause, end-clause or other) and
  `clause` (the clause it is in; for an end-clause pause, the one before it). A
  pause across a gap between clauses has a row for each part, which may be
  shorter than 0.25 s: MCPR, ECPR, MCPD and ECPD use the rows, PR and MPD the
  pauses. Use `duration` (rounded on its own) for the durations.
- `uf_clauses.csv`, one row per clause: `clause`, `start`, `end`, `words`,
  `syllables` (`morae`) and `text`.

The tables give back every measure but MLR, whose runs end at silences of the
words tier before the speech found inside them is taken out.

## References

HTS Working Group. (2018). *Open JTalk* (Version 1.11) [Computer software].
https://open-jtalk.sourceforge.net/

Takaoka, K., Hisamoto, S., Kawahara, N., Sakamoto, M., Uchida, Y., & Matsumoto,
Y. (2018). Sudachi: A Japanese tokenizer for business. In *Proceedings of the
Eleventh International Conference on Language Resources and Evaluation (LREC
2018)*. European Language Resources Association.
https://aclanthology.org/L18-1355/
