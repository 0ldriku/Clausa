# Filled-pause classifier: training

`train_podcastfillers_neural_classifier.py` trains the filled-pause classifier
on PodcastFillers (`ylacombe/podcast_fillers_by_license`): a small CNN on
log-mel features, with "Uh" and "Um" as fillers and words, breath, laughter and
music as non-fillers. The article's model used the CC BY, CC BY-SA and CC BY-ND
splits, reached F1 .933 on the test set, and is not shared. Where to put trained
weights: [models/README.md](../../models/README.md).

The pipeline runs it through `pipeline/shared/postprocess_vad_filler_classifier.py`
with a threshold of 0.50. `candidate_review_textgrids.py` turns its candidates
into TextGrids for checking in Praat.

Reference: Zhu et al. (2022), Filler Word Detection and Classification: A
Dataset and Benchmark, arXiv:2203.15135.
