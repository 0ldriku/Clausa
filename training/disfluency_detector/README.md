# Disfluency detector: training

Code that trains and evaluates Clausa's disfluency detector, an
`xlm-roberta-base` model that marks each word as fluent or disfluent. The
method and results are on the model card: https://huggingface.co/oldriku/Clausa.
Training writes the model to `model_v2/final/` in this folder.

The scripts in `scripts/`, in order:

1. `download_source_data.py`: 50K fluent English sentences from NLTK and 50K
   Japanese sentences from Tatoeba.
2. `inject_english.py`, `inject_japanese.py`: add learner disfluencies by rule.
3. `download_switchboard.py`: disfluency annotations from the Switchboard
   Dialog Act corpus.
4. `prepare_labels.py`: word labels.
5. `train.py`: fine-tuning.
6. `eval_english.py`, `eval_japanese.py`, `eval_disfluencyspeech*.py`:
   evaluation.
