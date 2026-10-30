# Reproducing the paper

For everyday use, the default analysis is better. See
[Improvements over the paper](improvements.md).

## The reported numbers

The article's analysis code and data are on Zenodo,
https://doi.org/10.5281/zenodo.23729881: the TextGrids of the article's run
and its RQ1-RQ3 scripts. They reproduce every reported table exactly, without
a GPU. The speakers' words are not included; the README there says how to put
them back from your own copy of ALLSSTAR and I-JAS. Then:

```bash
python3 -m venv venv && venv/bin/pip install -r requirements-analysis.txt
PYTHON=venv/bin/python bash reproduce_macos.sh
```

## Paper mode

`--paper-compat` runs the article's pipeline, kept frozen in `paper/pipeline/`.
On the same machine and software, its outputs are identical to the article's
pipeline. Paper mode is not in the app. Install the
[command line](command-line.md), then run:

```bash
./clausa run en recording.wav --paper-compat
./clausa run ja recordings/ --paper-compat -o results/ja
```

- Paper mode needs the filled-pause classifier, which the article used. Its
  weights are not shared; see [models/README.md](../models/README.md). Without
  them, paper mode skips it.
- Speech recognition gives identical results only on the same GPU and
  software. The article ran on Linux with an NVIDIA GPU and PyTorch 2.5.1;
  Clausa uses PyTorch 2.8.0 there, so a few words can differ.
- On a Mac, paper mode runs speech recognition on the CPU, which takes 2-3
  minutes per recording.

## Rerunning the evaluation

You need the 40 + 40 recordings from ALLSSTAR and I-JAS and the analysis
repository, which holds the reference TextGrids. The article cut each automatic
TextGrid to the reference speech span before finding clauses. `bench/` does
the same; `./clausa run` does not. With a conda install:

```bash
A=<analysis repository>; W=<work folder>
python3 bench/aligner_benchmark.py asr        --lang en --audio-dir <audio>/en --work $W --paper-compat
python3 bench/aligner_benchmark.py align      --lang en --audio-dir <audio>/en --work $W --paper-compat \
        --variant mfa --mfa-workers 8
python3 bench/aligner_benchmark.py downstream --lang en --audio-dir <audio>/en --work $W --paper-compat \
        --variant mfa --reference-dir $A/data/en/reference_textgrids
PYTHON=<analysis environment's python> bench/run_rq_variant.sh $A en $W/en/mfa <scores folder>
```

Repeat with `--lang ja`. `bench/summarize_rq.py` and `bench/compare_variant.py`
summarise and compare the scores.

The result is close to the article, not identical, because speech recognition
runs again. One rerun on Linux with an RTX 5090:

| | Japanese RQ1 κ | Japanese RQ2 κ | English RQ1 κ | English RQ2 κ |
|---|---|---|---|---|
| article | .674 | .673 | .745 | .848 |
| rerun in paper mode | .673 | .689 | .749 | .855 |

[Improvements over the paper](improvements.md) compares Clausa with the
article's own results; this rerun is the baseline for the effect of each change.
