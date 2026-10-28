"""
Evaluate our model on DisfluencySpeech dataset (HuggingFace).
This is real English disfluency data derived from Switchboard.

transcript_annotated: full transcript with disfluency annotations
transcript_c: clean transcript (false starts + filled pauses removed)

We compare transcript_a (all text including disfluencies) vs transcript_c (clean)
to derive word-level labels, then evaluate our model.
"""
import sys
import io
import os
import re
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

MODEL_DIR = os.path.join(PROJECT_DIR, "model", "final")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. Load DisfluencySpeech ---
print("Loading DisfluencySpeech from HuggingFace (text only)...")
ds = load_dataset("amaai-lab/DisfluencySpeech", split="test")
# Drop audio column to avoid FFmpeg dependency
if "audio" in ds.column_names:
    ds = ds.remove_columns(["audio"])

print(f"Test set: {len(ds)} utterances")
print(f"Columns: {ds.column_names}")

# Show a sample
print("\nSample:")
for col in ["transcript_annotated", "transcript_a", "transcript_b", "transcript_c"]:
    if col in ds.column_names:
        print(f"  {col}: {ds[0][col][:100]}...")

# --- 2. Extract disfluent/clean pairs ---
# transcript_a = all spoken words (disfluencies included, non-speech events removed)
# transcript_c = clean (filled pauses + false starts removed)
# We use these two to derive word-level labels

def word_labels(dis_words, flu_words):
    """0 for each word of the disfluent sentence that the fluent one keeps, 1 for the others;
    None if the fluent sentence is not the disfluent one with words left out. Words are matched
    from the end, so of a repeated word the last one is the fluent one."""
    labels = [1] * len(dis_words)
    left = len(flu_words)
    for pos in reversed(range(len(dis_words))):
        if left and dis_words[pos] == flu_words[left - 1]:
            labels[pos] = 0
            left -= 1
    return None if left else labels


print("\nExtracting disfluent/clean pairs and generating labels...")
all_dis_words = []
all_labels = []
skipped = 0
total = 0

for item in ds:
    # transcript_a has all spoken words; transcript_c has cleaned version
    dis_text = item.get("transcript_a", "")
    flu_text = item.get("transcript_c", "")

    if not dis_text or not flu_text:
        skipped += 1
        continue

    # Clean up: lowercase, normalize whitespace
    dis_text = dis_text.strip().lower()
    flu_text = flu_text.strip().lower()

    dis_words = dis_text.split()
    flu_words = flu_text.split()

    if len(dis_words) < 2 or len(flu_words) < 1:
        skipped += 1
        continue

    labels = word_labels(dis_words, flu_words)
    if labels is None:
        skipped += 1
        continue

    all_dis_words.append(dis_words)
    all_labels.append(labels)
    total += 1

n_disfl_tokens = sum(sum(l) for l in all_labels)
n_total_tokens = sum(len(l) for l in all_labels)
print(f"Usable: {total} utterances ({skipped} skipped)")
print(f"Tokens: {n_total_tokens} total, {n_disfl_tokens} disfluent ({100*n_disfl_tokens/max(n_total_tokens,1):.1f}%)")

# --- 3. Load model ---
print(f"\nLoading model from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, num_labels=2)

# --- 4. Tokenize and evaluate ---
data = {"disfluent": all_dis_words, "labels": all_labels}

def tokenize_and_align(examples):
    tok = tokenizer(examples["disfluent"], truncation=True, is_split_into_words=True)
    aligned_labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tok.word_ids(batch_index=i)
        prev = None
        lab = []
        for wid in word_ids:
            if wid is None:
                lab.append(-100)
            elif wid != prev:
                lab.append(label[wid])
            else:
                lab.append(label[wid])
            prev = wid
        aligned_labels.append(lab)
    tok["labels"] = aligned_labels
    return tok

eval_ds = Dataset.from_dict(data)
eval_ds = eval_ds.map(tokenize_and_align, batched=True)

collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(model, data_collator=collator, processing_class=tokenizer)

print("Running predictions...")
predictions, labels, _ = trainer.predict(eval_ds)
predictions = np.argmax(predictions, axis=2)

true_pred = [p for pred, lab in zip(predictions, labels) for p, l in zip(pred, lab) if l != -100]
true_lab = [l for pred, lab in zip(predictions, labels) for p, l in zip(pred, lab) if l != -100]

print()
print("=" * 60)
print("DISFLUENCYSPEECH EVALUATION (Real English Disfluency Data)")
print("=" * 60)
print(classification_report(true_lab, true_pred, target_names=["is_fluent", "is_disfluent"], zero_division=0))
print("Confusion Matrix (normalized):")
cm = confusion_matrix(true_lab, true_pred, normalize="all")
print(cm)

results = precision_recall_fscore_support(true_lab, true_pred, zero_division=0)
print(f"\nSummary:")
print(f"  Fluent     - P: {results[0][0]:.4f}  R: {results[1][0]:.4f}  F1: {results[2][0]:.4f}")
print(f"  Disfluent  - P: {results[0][1]:.4f}  R: {results[1][1]:.4f}  F1: {results[2][1]:.4f}")

# --- 5. Show some example predictions ---
print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)
for idx in range(min(10, total)):
    words = all_dis_words[idx]
    gold = all_labels[idx]

    # Get predictions for this sentence
    single_ds = Dataset.from_dict({"disfluent": [words], "labels": [gold]})
    single_ds = single_ds.map(tokenize_and_align, batched=True)
    pred, _, _ = trainer.predict(single_ds)
    pred = np.argmax(pred, axis=2)

    tok_input = tokenizer.tokenize(words, is_split_into_words=True)
    word_ids = tokenizer(words, is_split_into_words=True).word_ids()[1:-1]

    prev_wid = None
    fc = 0
    dc = 0
    pred_labels = []
    for i, pl in enumerate(pred[0][1:1+len(tok_input)]):
        if word_ids[i] != prev_wid:
            if prev_wid is not None:
                pred_labels.append(1 if dc > fc else 0)
            fc, dc = 0, 0
        if pl == 0:
            fc += 1
        else:
            dc += 1
        prev_wid = word_ids[i]
    if prev_wid is not None:
        pred_labels.append(1 if dc > fc else 0)

    # Trim to match
    pred_labels = pred_labels[:len(words)]
    gold = gold[:len(pred_labels)]

    print(f"\n--- Example {idx+1} ---")
    for w, g, p in zip(words, gold, pred_labels):
        g_tag = "D" if g == 1 else "."
        p_tag = "D" if p == 1 else "."
        match = "OK" if g == p else "MISS"
        if g != p:
            print(f"  {w:<20} gold={g_tag} pred={p_tag}  <-- {match}")
        elif g == 1:
            print(f"  {w:<20} gold={g_tag} pred={p_tag}")
