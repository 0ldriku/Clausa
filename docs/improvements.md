# Improvements over the paper

Clausa's default analysis improves on the pipeline used for the article. Paper
mode, `--paper-compat`, still runs the article's pipeline exactly. See
[Reproducing the paper](reproducing-the-paper.md).

## Agreement with human annotation

Scored with the article's RQ1-RQ3 scripts on its 40 English and 40 Japanese
recordings, against the same human annotation. "Article" is what the article
reports; "Clausa" is Clausa's default analysis (1.0.2; its results are the same
as 1.0.1's). r is Pearson's correlation and ICC the intraclass correlation
(single measure, absolute agreement) between Clausa's and the human
annotation's measures.

| Statistic | Japanese, article | Japanese, Clausa | English, article | English, Clausa |
|---|---|---|---|---|
| RQ1 κ, clause boundaries | .674 | **.704** | .745 | **.763** |
| RQ2 κ, pause location | .673 | **.718** | .848 | **.879** |

Japanese:

| Measure | r, article | r, Clausa | ICC, article | ICC, Clausa |
|---|---|---|---|---|
| AR | .946 | **.975** | .918 | **.975** |
| SR | .996 | .995 | .991 | .991 |
| MLR | .993 | .992 | .981 | .980 |
| MCPR | .968 | .974 | .958 | .963 |
| ECPR | .836 | .818 | .732 | .807 |
| PR | .986 | .989 | .978 | **.988** |
| MCPD | .951 | **.979** | .883 | **.964** |
| ECPD | .954 | .977 | .840 | **.970** |
| MPD | .969 | **.992** | .895 | **.985** |

English:

| Measure | r, article | r, Clausa | ICC, article | ICC, Clausa |
|---|---|---|---|---|
| AR | .952 | .956 | .949 | .951 |
| SR | .990 | .990 | .982 | .982 |
| MLR | .968 | .969 | .961 | .962 |
| MCPR | .968 | .972 | .964 | .953 |
| ECPR | .873 | **.943** | .862 | **.943** |
| PR | .970 | .970 | .956 | .952 |
| MCPD | .895 | **.930** | .880 | **.921** |
| ECPD | .979 | .982 | .973 | .979 |
| MPD | .976 | .982 | .959 | **.975** |

Bold means clearly better: the bootstrap confidence intervals of the
difference, by file and by speaker, both exclude 0, and the difference keeps
its sign when any one file is left out. Nothing got clearly worse. One value
is a little lower, Japanese ECPR's r (.836 to .818), well within chance; its
ICC rose (.732 to .807).

## What changed

Each change was measured on its own, against a rerun of the article's pipeline.

| Change | Language | Effect |
|---|---|---|
| The filler えっと stays one word | Japanese | RQ1 κ +.024 |
| Speech the aligner finds inside pauses counts as speaking time | both | AR ICC +.052, MCPD +.077, ECPD +.109, MPD +.083 in Japanese |
| English uses that speech instead of the filled-pause classifier | English | MCPD ICC +.036 |
| Words outside every clause join the next clause | English | RQ2 κ +.024, RQ1 κ +.014, ECPR ICC +.042 |
| The disfluency detector labels long recordings in full | both | RQ1 κ +.007 |
| A bug fix in splitting Japanese run-on clauses | Japanese | RQ1 κ +.003 |
