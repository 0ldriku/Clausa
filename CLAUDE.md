# Notes for coding agents

Clausa computes clause-based utterance-fluency measures for L2 English and
Japanese speech. Read first:

- `README.md`: what it does and how to use the app, for people who do not
  program (also `README.ja.md`, `README.zh-CN.md`); `docs/command-line.md`:
  the command line.
- `docs/development.md`: how the code is organised, how changes are
  validated, how the app zips are built.
- `docs/improvements.md`: what the default does differently from the paper's
  pipeline, with the evidence.
- `docs/reproducing-the-paper.md`: paper mode (`--paper-compat`).

## Rules

- **Change `pipeline/` only; `paper/` is frozen.** `paper/pipeline/` is the
  pipeline used for the article; `./clausa doctor` checks it against
  `paper/SHA256SUMS`. It changed once, for the Japanese mora count of the
  article's final version (`paper/README.md`). No pipeline or runner code branches on the mode: a
  difference of paper mode lives in `paper/pipeline/` or in `MODES`
  (`runner/run.py`), nowhere else.
- **A change that must not change outputs** (speed, refactoring, the runner,
  packaging, environments) is checked byte for byte, old against new, on the
  same machine; after a change to what paper mode shares, `--paper-compat`
  runs too (`docs/development.md`, "Validating changes").
- **A change to the default's outputs** is run on all 40 + 40 evaluation
  recordings, scored with the article's RQ scripts and compared with the
  current default, with a decision rule written down before the run. Never
  tune settings on the results.
- Every path in code comes from `runner/paths.py` (repository-relative,
  overridable by `CLAUSA_*` variables); no absolute paths in code.
- The pipeline scripts keep their command-line interfaces.
- **README.md, README.ja.md and README.zh-CN.md change together; the English
  is the source.** The same holds for the window's texts
  (`runner/app_ui/strings.js`, and `NATIVE` in `runner/app.py`) and for the
  three sections of each `packaging/*/read_me_first.html`.
- Commit messages explain why, with the validation result.
- Docs describe the code as it is; no development history or decision logs in
  the repository.

## Running

```bash
./clausa setup && ./clausa doctor          # conda route (macOS, Linux; Windows: python clausa ...)
./clausa run en file.wav                   # default
./clausa run en file.wav --paper-compat    # paper mode
container/clausa build && container/clausa doctor            # Linux container
cd <folder with audio> && <repo>/container/clausa run en file.wav
```

Paths given to `container/clausa` must be inside the current folder (it is
mounted at /work); copy audio in rather than linking it. The runner and
pipeline code are mounted at run time, so code changes need no rebuild.

## Things that have bitten before

- Stage 1 (ASR) is deterministic on the same GPU and software but not across
  GPUs, PyTorch versions, CPU and GPU, or CUDA and MLX. Compare old and new on
  the same machine.
- Parallel MFA is safe only because each call gets an isolated MFA root;
  without it files silently fall back to rough timestamps. MFA `align_one`
  and reusing a `--no_clean` temp folder change or corrupt the output.
- Stages always run with `HF_HUB_OFFLINE=1`; anything that loads a model
  (doctor too) must run in `envs.stage_env`.
- The English parser fills `tag_` but leaves `pos_` empty. Two rules are off on
  purpose (the English `pos_`-only rules and the Japanese forward case-tail
  merge): switched on, they lowered agreement. Do not "fix" them without a
  new test.
- `asr_mlx.py` undoes five defaults of `mlx-qwen3-asr` that changed the
  transcripts (`docs/development.md`, "Platforms"); keep them undone.
- Bench output records its mode in `MODE.json`; mixing modes is refused.
- `conda run ... python -` does not pass stdin to the script.
- Setup uses conda-forge only (`--override-channels`): Anaconda's default
  channel needs its terms accepted.
