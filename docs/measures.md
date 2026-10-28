# How the measures are computed

Clausa computes the nine measures from each recording's TextGrid in
`clauses/`: the words, the silences between them, and the clauses. The code is
`pipeline/en/caf_calculator_vad_classifier.py` and
`pipeline/ja/caf_calculator_ja_gap_classifier.py`.

## Units

- **Span:** from the start of the first word to the end of the last word.
  Silence before and after is left out.
- **Syllables, English:** counted from each word's spelling. Each group of
  vowels (a, e, i, o, u, y) is one syllable. A final e is silent when the word
  has more than one group. Every word has at least one.
- **Morae, Japanese:** counted from each word's characters. Each character is
  one, except small kana (ぁ ぃ ぅ ぇ ぉ ゃ ゅ ょ ゎ and their katakana), っ, ッ
  and ー. A kanji counts as one.
- Every word counts, fillers and repetitions included.
- **Pause:** a silence between words of 0.25 s or longer. Speech that the word
  aligner finds inside a silence, 0.10 s or longer, splits it. The silent parts
  left are pauses if they are 0.25 s or longer.
- **Speaking time:** the span minus all pauses.
- **Pause location:** the part of a pause inside a clause is a mid-clause
  pause. The part between two clauses is an end-clause pause. Pauses before the
  first clause or after the last one count only in PR and MPD.

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

- Japanese uses morae in place of syllables. Durations are in seconds.
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
