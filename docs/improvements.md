# Improvements over the paper

Clausa's default analysis improves on the pipeline used for the article. Paper
mode, `--paper-compat`, still runs the article's pipeline exactly. See
[Reproducing the paper](reproducing-the-paper.md).

## Agreement with human annotation

Scored with the article's RQ1-RQ3 scripts on its 40 English and 40 Japanese
recordings. "Paper mode" is the article's pipeline run again, so its values
differ slightly from the article's. "Default" is Clausa 1.0.0.

| Statistic | Japanese, paper mode | Japanese, default | English, paper mode | English, default |
|---|---|---|---|---|
| RQ1 κ, clause boundaries | .673 | **.704** | .749 | **.763** |
| RQ1 κ, midpoint projection | .677 | **.711** | .827 | **.848** |
| RQ2 κ, pause location | .689 | **.718** | .855 | **.879** |
| AR ICC | .910 | **.969** | .952 | .951 |
| SR ICC | .987 | .985 | .982 | .982 |
| MLR ICC | .977 | .972 | .962 | .962 |
| MCPR ICC | .943 | .953 | .965 | .953 |
| ECPR ICC | .753 | .806 | .902 | **.943** |
| PR ICC | .970 | .983 | .958 | .952 |
| MCPD ICC | .878 | **.964** | .879 | **.921** |
| ECPD ICC | .861 | **.970** | .972 | .979 |
| MPD ICC | .899 | **.985** | .960 | .975 |

Bold means clearly better: both bootstrap confidence intervals, by file and by
speaker, exclude 0, and the result holds when any one file is left out. Nothing
got clearly worse.

The improvements were found on these same recordings. The figures show that
real errors were fixed, not how Clausa will do on new data.

## What changed

| Change | Language | Effect |
|---|---|---|
| The filler えっと stays one word | Japanese | RQ1 κ +.024 |
| Speech the aligner finds inside pauses counts as speaking time | both | AR ICC +.051, MCPD +.077, ECPD +.109, MPD +.083 in Japanese |
| English uses that speech instead of the filled-pause classifier | English | MCPD ICC +.036 |
| Words outside every clause join the next clause | English | RQ2 κ +.024, RQ1 κ +.014, ECPR ICC +.042 |
| The disfluency detector labels long recordings in full | both | RQ1 κ +.007 |
| General rules replace rules written for the evaluation recordings | both | no measurable change |
| A bug fix in splitting Japanese run-on clauses | Japanese | RQ1 κ +.003 |

Clausa no longer removes interview phrases from Japanese recordings. Each
recording should hold only the learner's speech.
