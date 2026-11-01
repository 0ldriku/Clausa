# Developer guide

For people who change Clausa or build the app zips.

## Layout

```text
clausa              command line: setup | doctor | models | run | measures | webui | app
runner/             paths.py     every location; overridable by CLAUSA_* variables
                    envs.py      the environments and each stage's process environment
                    run.py       the six stages (run_pipeline); MODES: what paper mode does differently
                    measures.py  the measures again from a run's TextGrids as they are now (Compute)
                    resources.py how many recordings Stage 2 aligns at once
                    models.py    pinned model downloads and checks
                    app.py, app_ui/   the app window
                    projects.py  the app's projects
                    first_start_*.py  First start's window and events
                    requirements/     the conda route's pins, and exact package lists
pixi.toml, pixi.lock  the environments: the article's pins, and every package's build
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
| 6 measures | `<lang>/caf_calculator_*.py` | as Stage 3 | `uf/uf_measures.csv`, `uf_words.csv`, `uf_pauses.csv`, `uf_clauses.csv` (`shared/uf_tables.py`; not in paper mode) |

With `--window` or `--windows`, `shared/apply_window.py` blanks the words
outside the span after Stage 2.

`runner/measures.py` runs Stage 6 again on a run's `clauses/` TextGrids as
they are now, with the run's `islands/per_file/`, after people correct them in
Praat (`./clausa measures`, the app's Compute). A TextGrid nobody changed gives
its run's row byte for byte. It first checks each TextGrid: there, a text file
(praatio reads UTF-8 and UTF-16, not Praat's binary format), with a words and a
clauses tier under the names the calculators accept; the calculators would
otherwise carry on without them. A file the calculator cannot read gives a row
with an `error` column, and that column in every row: the others are computed
again without it.

Japanese morae (`mora_counts` in `ja/caf_calculator_ja_gap_classifier.py`):
each word's reading comes from SudachiPy (SudachiDict-core), read in context.
Numbers are read by the calculator instead, because the dictionary reads
digits one at a time, some kanji numbers as names (三百 ミツオ) and a number
before a counter as if alone (四時 ヨンジ). It reads them by place, in digits or
kanji, joining the digit words the aligner splits a number into (2 5). The
counter rules come from Open JTalk's digit module
(`src/njd_set_digit/njd_set_digit_rule_utf_8.h` in its source), and only the ones
that change the number of morae are kept: 4 ヨ, 9 ク, 4 シ before some counters;
whole readings like 一人 ひとり, the days (三日 みっか), 二十歳 はたち, ひとつ to
ここのつ. Sound changes (いっぽん, さんびゃく) keep the count and are left out.
Words that only contain a numeral (一緒) keep the dictionary's reading. Paper mode
keeps the article's count.

- **Environments.** pixi's in `.pixi/envs/`, else conda's of the same name.
  Each command runs its environment's Python directly, set up as activation
  would. `CLAUSA_CONDA_RUN=1` uses `conda run` for conda's. With
  `CLAUSA_ENVS_DIR`, as in the app zips, neither is needed.
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
- **Long recordings.** Stage 1: qwen-asr cuts a recording into pieces of up to
  180 s at quiet points and transcribes each on its own, with the same prompt.
  On PyTorch two pieces are decoded at once (`ASR_BATCH_PIECES` in
  `asr_mfa_*.py`), through MLX one at a time. On an 8 GB card two full pieces
  spill into shared memory and Stage 1 slows down over 20 times (a 9.6-min
  recording: 24 min); the Mac's memory does not grow with the length. Stage 2
  aligns the whole recording as one MFA utterance. A Japanese
  text longer than SudachiPy takes (49,149 bytes, about 55 min of learner
  speech) goes to MFA with `--no_tokenization`: its words are MeCab's already.
  Aligned that way, 39 of the 40 Japanese recordings give the same alignment;
  in one, 7 words move (MFA's tokenizer splits one word in two).

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
| `uf_words.csv`, `uf_pauses.csv`, `uf_clauses.csv` | yes | no |

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

**macOS.** Native on Apple Silicon. Stage 1 runs on the Mac GPU
through MLX (`pipeline/shared/asr_mlx.py`). `asr_mlx.py` undoes five defaults
of `mlx-qwen3-asr` that changed the transcripts, so it behaves like the official
package: pieces of up to 180 s, no stop on a repeated phrase, no 8 s attention
windows, no space between two English pieces, librosa resampling. Keep them.
`--device cpu` and paper mode run the official code on the CPU. On the 40 + 40
recordings nothing was clearly worse than Linux.

**Windows.** Setup picks the CUDA build of PyTorch when `nvidia-smi` exists,
and a CPU build of Kaldi for MFA. Outputs have CRLF line ends.

- MFA's working files go to the temp folder, because Windows limits paths to
  260 characters: to a folder of Stage 2's own there (`clausa_s2_*`), which the
  runner removes when the stage ends, also after a Stop.
- 8 GB graphics cards need the NVIDIA driver's default sysmem fallback.
- On the 40 + 40 recordings a few statistics were slightly lower than Linux,
  by up to .019. The cause is Stage 1's numerics on another GPU, not the code.

**Without a GPU.** Stage 1 runs on the CPU, 4-5 times slower. No statistic was
worse beyond the GPU's own noise.

Transcripts can differ between computers, so analyse one study on one machine.

## Environments

`./clausa setup` installs three environments with pixi: Python and the `mfa`
environment from conda-forge, the rest from PyPI. `pixi.toml` has the article's pins, the same
as `runner/requirements/`. `pixi.lock` has every package's version and build
for osx-arm64, win-64 and linux-64: those of the conda environments the
results were checked with (`runner/requirements/macos-lock/`, `windows-lock/`,
`container/locks/`), apart from five packages nothing needs in the Windows
`mfa` and conda's own pip and setuptools in the Mac's `ja-electra310`. Setup
installs it with `--locked`. CUDA installs use PyTorch 2.8.0, because the
article's 2.5.1 has no kernels for the newest GPUs.

- `qwen3-asr` and `qwen3-asr-cpu` differ only in PyTorch: on Windows and Linux
  CUDA 12.8 or the CPU build; on the Mac both have 2.5.1 and setup uses
  `qwen3-asr`. Setup installs one and removes the other.
- spacy-transformers 1.3.9 asks for transformers below 4.50; a
  `dependency-overrides` entry lets it sit beside 4.57.6, which qwen-asr needs.
- osx-arm64 is declared for macOS 14, so the lock has the macOS 14 builds of
  MLX that the app needs.
- The English spaCy model is not on PyPI: setup installs it with pip after
  pixi, as the conda route does.
- Setup notes the sha256 of `pixi.lock` (with LF line endings) in each environment's
  `conda-meta/clausa-pixi-lock.sha256`; `doctor` says when the checkout's
  lock is another.
- To change a pin: Updating packages and models, below.

`./clausa setup --conda` makes conda environments from the pins in
`runner/requirements/` instead; the packages they leave open come out at their
newest. The container uses conda's environments; the app zips take pixi's.

| Variable | Default | Purpose |
|---|---|---|
| `CLAUSA_CACHE` | `.cache/` | models, compiled-code caches, container storage |
| `CLAUSA_FILLER_MODEL` | `models/filler_classifier/model_podcastfillers_neural_v1_full/model.pt` | the filled-pause classifier |
| `CLAUSA_ASR_ENV`, `CLAUSA_MFA_ENV`, `CLAUSA_JA_ENV` | `qwen3-asr`, `mfa`, `ja-electra310` | conda's environment names (pixi.toml's are fixed) |
| `CLAUSA_PIXI_ENVS` | `.pixi/envs/` | where pixi's environments are (with pixi's detached-environments, where it puts them); empty: none |
| `CLAUSA_ENVS_DIR` | unset | a folder holding the three environments, no pixi or conda |
| `CLAUSA_CONDA_RUN` | unset | `1`: run stages in conda's environments through `conda run` |
| `CLAUSA_TORCH` | auto | `cuda` or `cpu` for `setup` |
| `CLAUSA_PROJECTS` | `~/Documents/Clausa projects` | the app's projects |
| `CLAUSA_RESULTS` | `~/Documents/Clausa results` | single runs, shown as Earlier analyses |
| `CLAUSA_SETTINGS`, `CLAUSA_LOGS` | `settings.json`, `logs/` | the app's settings and logs |
| `CLAUSA_UI_LANGUAGE` | unset | `en`, `ja` or `zh-CN`: the app window's language |
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

## Updating packages and models

Three lists pin what Clausa installs; each has its own way to change.

| What | The list | Who reads it |
|---|---|---|
| Python packages, all platforms | `pixi.toml` (pins) and `pixi.lock` (every package) | `./clausa setup`; the app builders pack these environments |
| PyTorch for the Windows app | `packaging/windows/torch-lock.json` | First start on Windows (the zip leaves PyTorch out) |
| Models | `runner/requirements/models-lock.json` | `runner/models.py`: setup, runs, First start |

**A package.** Edit `pixi.toml`, run `pixi lock` on the Mac or on Windows, and
read the lock's diff. Keep `runner/requirements/*.txt` at the same pins (the
conda route and the container read those). Run `./clausa setup` on both build
machines, then validate (Validating changes) and build the zips; the builders
refuse environments that are not the committed lock's.

**PyTorch on Windows.** When PyTorch's version changes in `pixi.toml`, update
`torch-lock.json` too: `cuda` lists the wheels `pixi.lock` has (file name, URL,
size, sha256; the Windows builder stops if they differ); `cpu` lists the same
version built for the processor, from `download.pytorch.org/whl/cpu`, which
First start installs on a PC without an NVIDIA driver. The `cpu` wheels are
not in `pixi.lock` (its `qwen3-asr-cpu` keeps the article's 2.5.1): check that
route on a PC without the driver before a release.

**A model.** Change its entry in `models-lock.json`: the repository's new
commit and each file's size and sha256 (Hugging Face lists them on the
commit's file pages). `./clausa models` downloads it; `./clausa models --check
--hashes` checks every file. A model that changes outputs needs the validation
of a change to the default (Validating changes).

## The app

`runner/app.py` shows `runner/app_ui/index.html` in a native window through
pywebview. The page calls the Python class `Api`.

- A project is a folder with `project.json`, copies of the recordings, one
  folder per session and `uf_measures.csv`. The sidebar's tabs are its two
  modes (`mode` in `settings.json`). **Annotate**: a session runs
  `run_pipeline` with its defaults; the window shows progress and the
  TextGrids, no measures. **Compute**: `Api.compute` runs `runner/measures.py`
  on every annotated recording's clauses TextGrid in the session that last
  annotated it, and writes the project's `uf_measures.csv`, its three tables
  (`projects.table_path`) and `compute_log.txt` when all are done; a Stop
  leaves them as they were. Each recording keeps its last Compute in
  `project.json` (`compute`: the session and the TextGrid's modification time
  and size then); a newer session or a changed TextGrid shows it as Ready
  again. One session or Compute at a time.
- A recording's part to analyse (`part` in `project.json`; `part_used`: the
  part of its last session; once it is done, `part` is `part_used` and locked)
  goes to the session's `parts.csv`, which
  `run_pipeline` takes as `windows_csv`. The waveform and Play come from
  `runner/audio.py` (PyAV; played by `afplay` or `winsound`, since the webview
  cannot read the project's files).
- Each session writes `log.txt`, copied to the app's `logs/`.
- On Windows, project names are limited to 60 characters and paths to 259.
- `runner/awake.py` keeps the computer awake during a session.
- `./clausa webui` is a separate browser interface for servers.

The window's texts are in `runner/app_ui/strings.js`, one table per language:
`en`, `ja` and `zh-CN`. A missing text shows in English. The backend sends a
key and its values beside the English text. The words of the native dialogs
are in `NATIVE` in `runner/app.py`. The language comes from
`CLAUSA_UI_LANGUAGE`, else from `ui_language` in `settings.json`, else from the
system. Settings › Display language sets `ui_language`. The window switches at
once; the native dialogs follow at the next start. The log stays in English,
and so do the measures' names and the research terms: clause, pause, filled
pause, disfluency, fluency, speed, syllable and mora.

The window is dark or light: Settings › Appearance sets `appearance` in
`settings.json` (`system`, the default, follows the system); the button beside
Settings switches to the other theme. `theme.js` sets
`data-theme` on the page before its first paint, and `app.css` gives each theme
Radix Colors' scales of the same steps. The title bar follows too: on macOS the
application's appearance, on Windows pywebview's title-bar theme
(`native_appearance` in `runner/app.py`). Screenshots: `?mock=...&theme=light`.

README.md, README.ja.md and README.zh-CN.md change together; the English is
the source. The same holds for the tables of `strings.js` and `NATIVE`, and for
the English, Japanese and Chinese sections of each guide,
`packaging/*/read_me_first.html`, whose sentences that depend on the build are
in the builders' `guide_notes()`.

To add a language:

1. Add its table to `strings.js` and to `NATIVE`, with every English key.
2. Add its code to `UI_LANGUAGES` and `ui_language_of` in `runner/app.py`, to
   `LANG_TAGS` in `app.js` and to the Settings menu in `index.html`.
3. Give it fonts in `app.css`, as `ja` and `zh-Hans` have.
4. Check its keys and placeholders against English, and look at every state of
   the page with `?mock=...&ui=<code>`.
5. Translate `README.md` as `README.<code>.md`, add it to the language line
   of every README and to `APP_PATHS` in `packaging/build_macos.py`, and give
   each `read_me_first.html` a section and `guide_notes()` its sentences.

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

The builders pack the checkout's pixi environments with conda-pack and refuse
one that setup did not install from that checkout's committed `pixi.lock`.
conda-pack takes the conda packages' files from pixi's package cache (the
builders stop if the cache was cleaned since setup: delete the environment and
run setup again); uv's PyPI packages and the spaCy model go in as found. pixi
writes no `.pyc` files for PyPI or noarch packages, so the builders
byte-compile each environment's `site-packages`, as pip and conda did (Mac: in
staging after `conda-unpack`, because conda-pack drops `.pyc` files its records
do not list; Windows: before packing, where conda-pack 0.9.2 keeps them). pixi's
own files (`conda-meta/pixi`, `pixi_env_prefix`, `.pixi-environment-fingerprint`
and rattler's `CACHEDIR.TAG`) stay out.

**Mac build.** On an Apple Silicon Mac, from a checkout of the release commit:

```bash
./clausa setup --no-models
python3 packaging/build_macos.py --out <folder> --conda-pack <conda-pack>
```

`--from-staging` reuses the environments a `--keep-staging` build left in
`<out>/staging` and takes minutes; use it only when `packaging/` has not
changed since (the builders check that they are `pixi.lock`'s).
conda-pack packs each environment, the builder runs `conda-unpack` and fixes
`libncurses`, and each environment goes in as one tar, because Archive Utility
would quarantine 120,000 files one by one. The minimum macOS, 14.0, comes from
the environments' binaries.

**Windows build.** On a Windows PC, from a checkout of the release commit:

```bat
python clausa setup --no-models --torch cuda
python packaging\build_windows.py --out <folder> --conda-pack <conda-pack.exe> --git <git.exe>
```

`conda-unpack` runs at First start, where the folder is. `Clausa.exe` is
compiled from `packaging/windows/launcher.cs` with Windows' own `csc.exe`; it
checks the folder's path, unpacks with `tar.exe`, and runs First start.
PyTorch's files are left out of the zip and downloaded at First start from the
pinned wheels in `packaging/windows/torch-lock.json`; the builder checks that
they are `pixi.lock`'s. Without an NVIDIA driver First start installs that
file's CPU wheels instead (not in `pixi.lock`). MFA gets the 8.3 short path of
its environment, because Kaldi cannot run from a path with a space. The PyPI
packages' launchers (`Scripts\*.exe`, uv's) keep the build's path; nothing
Clausa runs uses them.

**Releasing.**

1. Set the version in `runner/__init__.py` and in the window's mock
   (`runner/app_ui/app.js`, `version`, `version_full`); name the CHANGELOG's
   "Unreleased" after it, with the date; point the three READMEs' download
   links at it; merge to `main`.
2. Build both zips from that commit, with the setup command of each build
   section above.
3. Check each zip on its platform: a fresh First start, ALL_001 and CCH16-ST1
   identical to the regular install, `doctor`, paper mode refused, the app
   opens.
4. Tag the commit, `v1.0.0` and so on, and publish the zips with their
   `.sha256` files.

## Container

`container/Containerfile` builds the Linux image from conda-forge with setup's
conda route (the pins in `runner/requirements/`); `container/locks/` records
the exact packages of the validated image. `container/clausa` mounts the current
folder and the code at run time, so code changes need no rebuild. Rebuild after
changing the Containerfile or `runner/requirements/`. Podman's storage is in
`.cache/podman/`: do not move the clone, and run `container/clausa clean`
before deleting it.
