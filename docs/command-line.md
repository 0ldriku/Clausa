# Command line

Clausa also runs from a terminal on macOS, Windows and Linux. On the same
computer it gives the same results as the app.

## Install

**Linux: container.** Needs [Podman](https://podman.io). For an NVIDIA GPU it
also needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
with its CDI spec.

```bash
git clone https://github.com/0ldriku/Clausa.git ~/Clausa
~/Clausa/container/clausa build
cd ~/my_recordings
~/Clausa/container/clausa run en speaker01.wav
```

Recordings must be inside the current folder. For a computer without a GPU,
build with `--build-arg TORCH=cpu`. Run `container/clausa clean` before
deleting the folder.

**Mac or Linux.** Needs [pixi](https://pixi.sh) and Python 3.9 or later.

```bash
git clone https://github.com/0ldriku/Clausa.git
cd Clausa
./clausa setup
./clausa doctor
```

**Windows.** Needs git, [pixi](https://pixi.sh) and Python 3.9 or later. Use
a short path.

```bat
git clone https://github.com/0ldriku/Clausa.git C:\Clausa-src
cd C:\Clausa-src
python clausa setup
python clausa doctor
```

On Windows, write `python clausa` instead of `./clausa`.

`setup` installs three environments, `qwen3-asr`, `mfa` and `ja-electra310`,
into `.pixi/envs/`, every package at the version in `pixi.lock`, and
downloads the models into `.cache/`, about 8 GB. On Windows and Linux,
PyTorch is the CUDA build when there is an NVIDIA GPU; `--torch cpu` installs
`qwen3-asr-cpu` instead. pixi is needed only for `setup`; run it again after
updating Clausa.

`./clausa setup --conda` makes conda environments instead, with
[Miniforge](https://conda-forge.org/download/) or, on Windows,
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
They have the article's pins; the packages those leave open come out at their
newest.

## Run

```bash
./clausa run en recording.wav                    # results in output/en/recording/
./clausa run ja recordings/ -o results/ja        # every audio file in a folder
./clausa run en recording.wav --device cpu       # without the GPU
./clausa run ja interview.wav --window 12.5 140  # only 12.5 s to 140 s
./clausa run ja recordings/ --windows spans.csv  # one span per file
./clausa webui                                   # browser interface at http://127.0.0.1:7860
./clausa app                                     # the app window
./clausa run --help                              # all options
```

On a Mac, double-clicking `Start Clausa.command` starts the browser interface.

The whole file is analysed. To analyse only part of it, cut it, or give the
span:

- `--window START END`: seconds, the same for every file.
- `--windows CSV`: columns `file_id,start,end`, where `file_id` is the file
  name without its extension.

## Measures after corrections

Correct the TextGrids in `clauses/` in Praat, save them as text files with the
same tier names, then compute the measures again from them as they are now:

```bash
./clausa measures en output/en/recording               # writes output/en/recording/measures.csv
./clausa measures en output/en/recording -o fixed.csv
```

Beside it go `words.csv`, `pauses.csv` and `clauses.csv` (with `-o fixed.csv`:
`fixed_words.csv` ...), the tables of [Measures](measures.md#tables). A
TextGrid you did not change gives the same row as the run's
`uf/uf_measures.csv`. A TextGrid without a words or a clauses tier is named and
left out.

## Output

```text
output/en/recording/
├── uf/uf_measures.csv     the nine measures, one row per recording
├── uf/uf_words.csv ...    their words, pauses and clauses (measures.md, Tables; not with --paper-compat)
├── asr/textgrids/         words and pauses, for Praat
├── clauses/               words, pauses, disfluencies and clauses, for Praat
├── islands/               speech found inside pauses
└── manifest.json          the settings of the run
```

## Speed

One recording of 2-3 minutes:

| Computer | Alone | In a batch of 10, each |
|---|---|---|
| Linux, RTX 5090 | 30 s | 10 s |
| Mac, M1 Max | 50 s | 15-20 s |
| Windows, RTX 4060 Ti 8 GB | 90 s | 30 s |
| Without a GPU | 2.5-4.5 min | |

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `CLAUSA_CACHE` | `.cache/` | where models and caches go |
| `CLAUSA_FILLER_MODEL` | `models/filler_classifier/model_podcastfillers_neural_v1_full/model.pt` | the filled-pause classifier |
| `CLAUSA_PROJECTS` | `~/Documents/Clausa projects` | the app's projects |
| `CLAUSA_UI_LANGUAGE` | `ui_language` in `settings.json`, else the system's | `en`, `ja` or `zh-CN`: the app window's language. Settings › Display language sets `ui_language`. |
| `CLAUSA_TORCH` | auto | `cuda` or `cpu` for `setup` |
| `HF_ENDPOINT` | `https://huggingface.co` | a mirror for the models |

The [developer guide](development.md) lists the rest.
