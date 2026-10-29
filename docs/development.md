# Developer guide

For people who change Clausa or build the app zips.

## Layout

```text
clausa              command line: setup | doctor | models | run | webui | app
runner/             paths.py     every location; overridable by CLAUSA_* variables
                    envs.py      the environments and each stage's process environment
                    run.py       the six stages (run_pipeline); MODES: what paper mode does differently
                    resources.py how many recordings Stage 2 aligns at once
                    models.py    pinned model downloads and checks
                    app.py, app_ui/   the app window
                    projects.py  the app's projects
                    first_start_*.py  First start's window and events
                    requirements/     the environments' pins and exact package lists
pipeline/en, ja/    the stage scripts
pipeline/shared/    filled-pause classifier, speech inside pauses, --window, audio decoding, MLX
paper/              the article's pipeline, frozen
models/             the filled-pause classifier goes here; README.md lists every model
container/          Containerfile, the Podman wrapper, exact package lists
packaging/          build_macos.py, build_windows.py and their launchers, First start and guides
bench/              the evaluation harness
training/           training code for the two trained models
```

## Rules

- Every path in code comes from `runner/paths.py`.
- Change `pipeline/` only. `paper/` is frozen.
- No code branches on paper mode outside `MODES` in `runner/run.py`.
- The pipeline scripts keep their command-line interfaces.
- Commit messages say why, with the validation result.

## How a run works

`run_pipeline` in `runner/run.py` runs six stages, each over all recordings,
so the models load once per stage.

| Stage | Script | Environment | Output |
|---|---|---|---|
| decode | `shared/convert_audio.py`, `.m4a` only | qwen3-asr | `audio/` |
| 1 ASR | `<lang>/asr_to_pre_mfa_json.py`: Qwen3-ASR transcript and rough timings | qwen3-asr | `asr/pre_mfa_json/` |
| 2 MFA | `<lang>/mfa_from_pre_mfa_json.py`: word timings, several recordings at once | qwen3-asr, calling mfa | `asr/json/`, `asr/textgrids*/` |
| 3 clauses | `en/textgrid_caf_segmenter.py`, `ja/ja_clause_segmenter_v102.py`, with the disfluency detector | qwen3-asr for en, ja-electra310 for ja | `clauses/` |
| 4 fillers | `shared/postprocess_vad_filler_classifier.py`, only with its weights | qwen3-asr | `gap_candidates/`, `patched_textgrids/` |
| 5 islands | `shared/placeholder_islands.py`: speech inside pauses | qwen3-asr | `islands/` |
| 6 measures | `<lang>/caf_calculator_*.py` | as Stage 3 | `uf/uf_measures.csv` |

With `--window` or `--windows`, `shared/apply_window.py` blanks the words
outside the span after Stage 2.

- **Environments.** Each command runs its environment's Python directly, set up
  as activation would. `CLAUSA_CONDA_RUN=1` uses `conda run` instead. With
  `CLAUSA_ENVS_DIR`, as in the app zips, no conda is needed.
- **Records.** `progress.json`, `events.jsonl` (format: `Events` in
  `runner/run.py`) and `manifest.json` hold paths and times, so byte checks
  skip them. A recording MFA could not align keeps its rough timings and is
  reported as failed.
- **Stop.** A `threading.Event` passed as `stop` ends the running stage and
  everything it started. Finished stages keep their output; a stopped run
  starts again from the beginning.
- **Parallel alignment.** Each MFA call has its own MFA root, so the outputs do
  not depend on the number of workers. Without `--mfa-workers`,
  `runner/resources.py` uses min(8, recordings, physical cores − 1,
  (available memory − 2 GB) / 2 GB), at least 1. Each worker takes about 1.7 GB.

## Paper mode

`paper/pipeline/` is a frozen copy of the files paper mode uses, with the
article's behaviour ([paper/README.md](../paper/README.md)). `MODES` in
`runner/run.py` holds everything else that differs:

| | default | paper mode |
|---|---|---|
| stage scripts | `pipeline/` | `paper/pipeline/` |
| speech found inside pauses | en, ja | none |
| the filled-pause classifier's islands | ja | en, ja |
| recordings MFA could not align reported as failed | yes | no |
| Stage 1 through MLX | yes on Apple Silicon | never |
| settings also passed under the old names `NEWCAF_MFA_*`, `CAFASR_*` | no | yes |

`./clausa doctor` checks `paper/pipeline/` against `paper/SHA256SUMS` and that
nothing else depends on the mode. The app zips leave `paper/` out and refuse
paper mode.

## Validating changes

**A change that must not change outputs**, such as speed, refactoring,
packaging or the runner, is checked byte for byte, old against new, on one
machine:

- `./clausa run` on one English and one Japanese recording, default and
  `--paper-compat`; a batch of 10 per language when Stage 2 or the runner
  changes. Compare every file except `events.jsonl`, `manifest.json`,
  `progress.json`, `file_list.json`, `RUN_LOG.md` and `RUN_SUMMARY.json`.
- ALL_001_F_GER_ENG_ST1 and CCH16-ST1 with `--paper-compat` must stay identical
  to the outputs of the article's own code. This needs the classifier's weights.
- After a change that paper mode shares: `--paper-compat` on all 40 + 40
  recordings, identical before and after.

**A change to the default's outputs** is run on all 40 + 40 recordings and
scored with the article's RQ scripts against the current default:

- Runs: `bench/aligner_benchmark.py`, which cuts each TextGrid to the reference
  span as the article did. Commands: [reproducing-the-paper.md](reproducing-the-paper.md).
- Scores: `bench/run_rq_variant.sh`. Comparison: `bench/compare_variant.py`.
  Clearly better or worse means both bootstrap intervals exclude 0 and the
  sign holds when any one file is left out.
- A change to the measure calculators is scored with `CALC_REPO` set: the
  automatic side then uses this repository's calculators, and the Japanese
  reference's morae are counted the same way. Japanese scoring needs SudachiPy,
  which the analysis environment has; or set `CALC_PYTHON` to ja-electra310's
  Python.
- Write the decision rule down before the run. Never tune settings on the
  results.
- For a new platform, compare with the Linux default. Differences under .014
  κ or .010 ICC are GPU noise.

## Pitfalls

- Stage 1 is identical only on the same GPU and software, not across GPUs,
  PyTorch versions, CPU and GPU, or CUDA and MLX. Compare on one machine.
- The Stage 1 prompt matters: an empty prompt lowers English filler recall by
  about .23. Treat a prompt edit as an output change.
- `qwen-asr` applies its 8 s attention windows only with flash-attn. Stage 1
  runs without it, as in the article. Installing flash-attn changes Stage 1.
- The English parser leaves `pos_` empty. Rules that test `pos_`, and the
  Japanese forward case-tail merge, are off on purpose: on, they lowered
  agreement.
- Parallel MFA is safe only with one MFA root per call. MFA's `align_one` and a
  reused `--no_clean` folder change the output.
- Both reference word tiers come from MFA, so word-timing comparisons favour
  MFA. Compare on the RQ measures.
- Stages run with `HF_HUB_OFFLINE=1`. Anything that loads a model, doctor too,
  must run in `envs.stage_env`.
- `conda run ... python -` does not pass stdin.

## Platforms

**macOS.** Native on Apple Silicon through conda. Stage 1 runs on the Mac GPU
through MLX (`pipeline/shared/asr_mlx.py`). `asr_mlx.py` undoes five defaults
of `mlx-qwen3-asr` that changed the transcripts, so it behaves like the official
package: pieces of up to 180 s, no stop on a repeated phrase, no 8 s attention
windows, no space between two English pieces, librosa resampling. Keep them.
`--device cpu` and paper mode run the official code on the CPU. On the 40 + 40
recordings nothing was clearly worse than Linux.

**Windows.** Installed with micromamba. Setup picks the CUDA build of PyTorch
when `nvidia-smi` exists, and a CPU build of Kaldi for MFA. Outputs have CRLF
line ends.

- MFA's working files go to the temp folder, because Windows limits paths to
  260 characters. A later run removes folders a Stop left behind.
- 8 GB graphics cards need the NVIDIA driver's default sysmem fallback.
- On the 40 + 40 recordings a few statistics were slightly lower than Linux,
  by up to .019. The cause is Stage 1's numerics on another GPU, not the code.

**Without a GPU.** Stage 1 runs on the CPU, 4-5 times slower. No statistic was
worse beyond the GPU's own noise.

Transcripts can differ between computers, so analyse one study on one machine.

## Environments

`./clausa setup` creates three conda environments from conda-forge with the
article's pins in `runner/requirements/`. CUDA installs use PyTorch 2.8.0,
because the article's 2.5.1 has no kernels for the newest GPUs. Exact package
lists: `container/locks/`, `runner/requirements/macos-lock/`,
`runner/requirements/windows-lock/`.

| Variable | Default | Purpose |
|---|---|---|
| `CLAUSA_CACHE` | `.cache/` | models, compiled-code caches, container storage |
| `CLAUSA_FILLER_MODEL` | `models/filler_classifier/model_podcastfillers_neural_v1_full/model.pt` | the filled-pause classifier |
| `CLAUSA_ASR_ENV`, `CLAUSA_MFA_ENV`, `CLAUSA_JA_ENV` | `qwen3-asr`, `mfa`, `ja-electra310` | environment names |
| `CLAUSA_ENVS_DIR` | unset | a folder holding the three environments, no conda |
| `CLAUSA_CONDA_RUN` | unset | `1`: run stages through `conda run` |
| `CLAUSA_TORCH` | auto | `cuda` or `cpu` for `setup` |
| `CLAUSA_PROJECTS` | `~/Documents/Clausa projects` | the app's projects |
| `CLAUSA_RESULTS` | `~/Documents/Clausa results` | single runs, shown as Earlier analyses |
| `CLAUSA_SETTINGS`, `CLAUSA_LOGS` | `settings.json`, `logs/` | the app's settings and logs |
| `CLAUSA_JOBS` | `jobs/` | the web interface's jobs |
| `HF_ENDPOINT` | `https://huggingface.co` | a mirror for the models |
| `CLAUSA_MFA_BASE_URL` | the mfa-models releases | a mirror for the MFA models |
| `CLAUSA_MFA_FULL_DICTIONARY` | unset | `1`: MFA loads its whole dictionary, slower, same output |
| `CLAUSA_EN_SPACY_MODEL` | `en_ud_L1L2e_combined_trf` | the English parser |
| `CLAUSA_IMAGE` | `localhost/clausa:latest` | the container image |

The runner sets `CLAUSA_MFA_PREFIX`, `CLAUSA_MFA_EXE`, `CLAUSA_MFA_LIB_BIN` and
`CLAUSA_MFA_ENV_NAME` for Stage 2. `CLAUSA_FIRST_START_*` and
`CLAUSA_LAUNCHER_TEST` are for tests.

## Models

`runner/requirements/models-lock.json` pins every downloaded model: repository,
commit, and each file's size and sha256. `runner/models.py` is the only
downloader, for setup, runs, the bench and First start.

- Hugging Face files go into the hub cache layout, with `refs/main` set to the
  pinned commit. On Windows, snapshots are hard links.
- The clause stage gets the disfluency detector's snapshot folder as
  `--disfluency-model`.
- MFA models go into `<cache>/mfa/pretrained_models/`.
- A download is resumed, checked against its size and sha256, and deleted if
  wrong. Exit codes: 2 network, 3 disk space, 4 checksum.

## The app

`runner/app.py` shows `runner/app_ui/index.html` in a native window through
pywebview. The page calls the Python class `Api`.

- A project is a folder with `project.json`, copies of the recordings, one
  folder per session and a combined CSV. A session runs `run_pipeline` with its
  defaults, one session at a time.
- Each session writes `log.txt`, copied to the app's `logs/`.
- On Windows, project names are limited to 60 characters and paths to 259.
- `runner/awake.py` keeps the computer awake during a session.
- `./clausa webui` is a separate browser interface for servers.

## The app zips

One zip per platform holds one `Clausa` folder. Nothing is installed.

| | macOS | Windows |
|---|---|---|
| launcher | `Clausa.app` | `Clausa.exe` |
| First start | `First start.command` | inside `Clausa.exe` |
| guide | `Read me first.html` | `Read me first.html` |
| `envs/` | the three environments as tars | the same, without PyTorch's files |
| `cache/` | empty; First start downloads the models | the same |
| `app/` | the code listed in `APP_PATHS` | the same, plus `first_start.py` |

The zips leave out `paper/`, `docs/`, `bench/`, `training/`, `container/` and
the classifier's weights. `--include-models` puts the models in, for use
without internet.

**First start** writes events to `logs/first-start-events.jsonl`
(`runner/first_start_events.py` describes the format). The window
`runner/first_start_window.py` shows them and tells its starter what to do by
its exit code: 0 open the app, 1 closed early, 3 closed when done, 75 try
again, 10 no window.

**Mac build.** On an Apple Silicon Mac, from a checkout of the release commit:

```bash
python3 packaging/build_macos.py --out <folder> --conda-pack <conda-pack>
```

`--from-staging` reuses the environments of the last build and takes minutes.
conda-pack packs each environment, the builder runs `conda-unpack` and fixes
`libncurses`, and each environment goes in as one tar, because Archive Utility
would quarantine 120,000 files one by one. The minimum macOS, 14.0, comes from
the environments' binaries.

**Windows build.** On a Windows PC with the regular install:

```bat
python packaging\build_windows.py --out <folder> --conda-pack <conda-pack.exe> --git <git.exe>
```

`conda-unpack` runs at First start, where the folder is. `Clausa.exe` is
compiled from `packaging/windows/launcher.cs` with Windows' own `csc.exe`; it
checks the folder's path, unpacks with `tar.exe`, and runs First start.
PyTorch's files are left out of the zip and downloaded at First start from the
pinned wheels in `packaging/windows/torch-lock.json`. MFA gets the 8.3 short
path of its environment, because Kaldi cannot run from a path with a space.

**Releasing.**

1. Set the version in `runner/__init__.py` and merge to `main`.
2. Build both zips from that commit.
3. Check each zip on its platform: a fresh First start, ALL_001 and CCH16-ST1
   identical to the regular install, `doctor`, paper mode refused, the app
   opens.
4. Tag the commit, `v1.0.0` and so on, and publish the zips with their
   `.sha256` files.

## Container

`container/Containerfile` builds the Linux image from conda-forge with the
package lists in `container/locks/`. `container/clausa` mounts the current
folder and the code at run time, so code changes need no rebuild. Rebuild after
changing the Containerfile or `runner/requirements/`. Podman's storage is in
`.cache/podman/`: do not move the clone, and run `container/clausa clean`
before deleting it.
