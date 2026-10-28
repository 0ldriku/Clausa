# Models

| Model | Included |
|---|---|
| Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B, speech recognition | downloaded |
| MFA `english_us_arpa` and `japanese_mfa`, word alignment | downloaded |
| Clausa's disfluency detector, `oldriku/Clausa` | downloaded |
| GiNZA-Electra, Japanese parsing | downloaded |
| spaCy `en_ud_L1L2e_combined_trf` and GiNZA `ja_ginza_electra` | installed by `./clausa setup` |
| Filled-pause classifier | no; only paper mode uses it, see below |

## Downloaded models

Each model is pinned to one version in `runner/requirements/models-lock.json`,
about 8 GB in all. Every file is checked against its sha256 before use, and a
broken download continues where it stopped. `./clausa models --check` shows
whether the models in place are the pinned ones.

| Model | Version | Size |
|---|---|---|
| `Qwen/Qwen3-ASR-1.7B` | commit `7278e1e70fe206f11671096ffdd38061171dd6e5` | 4.7 GB |
| `Qwen/Qwen3-ForcedAligner-0.6B` | commit `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | 1.8 GB |
| `oldriku/Clausa` | commit `1f5d280b20587c70c7e3a0dbd3534872440dec16` | 1.1 GB |
| `megagonlabs/transformers-ud-japanese-electra-base-ginza-510` | commit `8498b019a97bdf693fa138efcf7e118f099e2c4e` | 0.4 GB |
| MFA `english_us_arpa` | mfa-models release v3.0.0 | 0.1 GB |
| MFA `japanese_mfa` | mfa-models release v3.0.0 | 0.1 GB |

## Disfluency detector

Marks each word as fluent or disfluent before clauses are found: fillers,
repetitions, false starts. It is a fine-tuned XLM-RoBERTa, licensed CC BY-NC-SA
4.0. Model card: https://huggingface.co/oldriku/Clausa. Training code:
`training/disfluency_detector/`.

## Filled-pause classifier

Finds filled pauses such as "uh" and "um" that the transcript missed. The
article used it. Clausa's default finds that speech with the word aligner
instead, and in our tests its results without the classifier are as good.
Only paper mode needs it, to reproduce the article exactly.

Its weights are not shared, because its training data includes PodcastFillers
clips under a no-derivatives licence. Get them from the authors or train them
with `training/filler_classifier/train_podcastfillers_neural_classifier.py`.
Put the file at

    models/filler_classifier/model_podcastfillers_neural_v1_full/model.pt

or set `CLAUSA_FILLER_MODEL` to its path. `./clausa doctor` shows whether it
was found. The article's version has the sha256
`4fb8211b3a87a324526b0247d47086f071928d2c1aefcce77301f64262854e30`.
