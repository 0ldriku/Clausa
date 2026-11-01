# Changelog

Each version of Clausa, newest first. **Measures** says whether a recording
gives other numbers than in the version before.

## 1.0.4 (1 November 2026)

- **Part to analyse** is locked once a recording is annotated: its details
  show the part its TextGrids come from, and Play plays it, but it can no
  longer change (1.0.3 let it change afterwards, so the row could show another
  part than the TextGrids had). A project from 1.0.3 shows each annotated
  recording's analysed part again. To analyse another part, remove the
  recording and add it again.

**Measures:** the same.

## 1.0.3 (1 November 2026)

- Two modes in the sidebar, under **New project**. **Annotate** runs the
  analysis and shows its progress and the TextGrids, no measures. **Compute**
  computes the nine measures from the TextGrids as they are now, after you
  correct them in Praat, into the project's `uf_measures.csv`. A TextGrid
  without a words or a clauses tier is named and left out.
- `./clausa measures` does the same on the command line
  ([Command line](docs/command-line.md)).
- **Part to analyse**: in Annotate, a recording's details choose the whole
  recording or a part of it, on its waveform (drag the edges, type the times,
  play the part); one part for all new recordings, or parts from a CSV
  (`file_id,start,end`). Words outside the part are left out, as with
  `--window` on the command line.
- Compute, and each run in `uf/` (not in paper mode), also writes the words,
  pauses and clauses the measures came from: `uf_words.csv`, `uf_pauses.csv`,
  `uf_clauses.csv` ([Tables](docs/measures.md#tables)). **Open CSV** is now
  **Show CSV**, a menu that shows each file in its folder.
- Japanese numbers are counted as they are said, in digits or in kanji: by
  place (25 is にじゅうご, 4 morae, not 2) and with their counter (四時 よじ,
  一人 ひとり, 三日 みっか, 二十歳 はたち).
- On Windows, a Stop no longer leaves the alignment's temporary files (about
  0.1 GB per recording being aligned) in the temp folder.
- The window no longer asks for one speaker per file. A recording's details
  say which TextGrid is the full one (the clauses TextGrid, which the measures
  come from) and no longer hide the sidebar. The status bar shows the whole
  version number.
- Long recordings. With PyTorch (Windows, Linux, and `--device cpu` on a Mac),
  speech recognition takes two 3-minute pieces at a time instead of all of
  them, so a long recording needs less graphics memory (on an 8 GB card, 22 minutes took 35 minutes instead of not
  finishing; recordings over about 6 minutes are still slow there). Japanese
  recordings with a long transcript (more than about 55 minutes of learner
  speech) now get word timings from the aligner; before, all their words kept
  rough timings.
- README: a shorter opening and an FAQ.
- The command line installs its environments with [pixi](https://pixi.sh)
  from `pixi.lock`, every package at the version and build the results were
  checked with: `./clausa setup` needs pixi instead of Miniforge or
  micromamba. `./clausa setup --conda` still makes conda environments
  ([Command line](docs/command-line.md)).
- The app zips' environments come from `pixi.lock` too (on Windows without an
  NVIDIA card, PyTorch is still the CPU build First start downloads).

**Measures:** Japanese recordings with numbers can change: AR, SR, MLR, MCPR,
ECPR and PR. In the article's 40 Japanese recordings one changes slightly; the
figures in [Improvements over the paper](docs/improvements.md) stay the same.
Paper mode and English are the same. A TextGrid you did not change gives the
same row as before in Compute. The tables are new; `uf_measures.csv` is the
same. On the Mac and Windows, pixi's environments give the same files as the
conda ones, byte for byte (on Linux the lock has the container's packages).
With PyTorch (Windows, Linux, `--device cpu` on a Mac), recordings longer than
about 6 minutes can change, and Japanese recordings longer than about 55
minutes on every computer; the 80 evaluation recordings are shorter.

## 1.0.2 (30 October 2026)

- The window in Japanese and Simplified Chinese. It follows your computer's
  language; Settings › Display language changes it.
- A light theme. It follows your computer's appearance; Settings › Appearance,
  or the button beside Settings, changes it.
- About Clausa in Settings: the version, the licences, the article to cite with
  Copy citation, and how to reach us.
- README and Read me first in Japanese and Simplified Chinese.
- README: Clausa works with first-language speech too; the memory it needs
  (16 GB; 32 GB on Windows without an NVIDIA card); it collects and sends
  nothing; how to choose another projects folder.

**Measures:** the same.

## 1.0.1 (29 October 2026)

- Japanese morae are counted from each word's kana reading in context
  (SudachiPy), not from its characters, by default and in paper mode. The
  article's final version counts them this way.
- README: Limitations.

**Measures:** Japanese AR, SR, MLR, MCPR, ECPR and PR change. English is the
same.

## 1.0.0 (28 October 2026)

The first public version.

- The app for Mac (Apple Silicon) and Windows, one zip each. First start
  downloads the models.
- The command line for macOS, Windows and Linux, and a container for Linux.
- Projects: the recordings of one study in one language, analysed over as
  many sessions as you need.
- For English and Japanese speech: TextGrids with words, pauses, disfluencies
  and clauses, and nine clause-based measures.
- More accurate than the article's pipeline
  ([Improvements over the paper](docs/improvements.md)); paper mode,
  `--paper-compat`, runs the article's pipeline exactly.
