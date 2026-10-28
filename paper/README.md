# paper/: the article's pipeline, frozen

`paper/pipeline/` is what `--paper-compat` runs. On the same machine and
software its outputs are identical to the pipeline used for the article.

It is never changed. Every change goes to `pipeline/`. `./clausa doctor`
checks these files against `SHA256SUMS`. A fix that paper mode must have too,
such as a crash on a new platform, must keep the article's outputs identical
and update `SHA256SUMS` on purpose.

## How it differs from pipeline/

It is a copy of the files of `pipeline/` that paper mode uses, with the
article's behaviour:

- The Japanese speech-recognition prompt is the article's, and えっと is not
  joined.
- Only the filled-pause classifier finds speech inside pauses, in both
  languages.
- Speech recognition never runs through MLX.
- The disfluency detector leaves words past its 512-subword limit fluent.
- The English and Japanese clause rules include the rules written for the
  evaluation recordings, and English words outside every clause stay outside.

It keeps the speed-ups and fixes that do not change outputs. Its layout is
that of `pipeline/`, so imports work the same way.

## How it is checked

- Two reference recordings, one English and one Japanese, run with
  `--paper-compat`, must stay identical to the article's own code.
- After any change to the runner or this folder, `--paper-compat` runs of all
  40 + 40 evaluation recordings must stay identical.
- `./clausa doctor` checks `SHA256SUMS`, and that `pipeline/` and `runner/`
  have no paper-mode switches outside the mode table in `runner/run.py`.

How to use it: [Reproducing the paper](../docs/reproducing-the-paper.md).
