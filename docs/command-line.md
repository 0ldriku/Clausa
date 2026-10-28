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

**Mac or Linux: conda.** Needs [Miniforge](https://conda-forge.org/download/)
and Python 3.9 or later.

```bash
git clone https://github.com/0ldriku/Clausa.git
cd Clausa
./clausa setup
./clausa doctor
```

**Windows: micromamba.** Needs git,
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
and Python 3.9 or later. Use a short path.

```bat
git clone https://github.com/0ldriku/Clausa.git C:\Clausa-src
cd C:\Clausa-src
python clausa setup
python clausa doctor
```

On Windows, write `python clausa` instead of `./clausa`.

`setup` creates three conda environments, `qwen3-asr`, `mfa` and
`ja-electra310`, and downloads the models into `.cache/`, about 8 GB.

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

The whole file is analysed. Cut out everything that is not the learner's
speech, or give the learner's span:

- `--window START END`: seconds, the same for every file.
- `--windows CSV`: columns `file_id,start,end`, where `file_id` is the file
  name without its extension.

## Output

```text
output/en/recording/
├── uf/uf_measures.csv     the nine measures, one row per recording
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
| `CLAUSA_TORCH` | auto | `cuda` or `cpu` for `setup` |
| `HF_ENDPOINT` | `https://huggingface.co` | a mirror for the models |

The [developer guide](development.md) lists the rest.
