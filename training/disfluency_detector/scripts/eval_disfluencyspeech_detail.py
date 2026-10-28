"""
Detailed error analysis on DisfluencySpeech dataset.
Shows: missed disfluencies, false positives (mislabelled), and categorizes errors.
"""
import sys
import io
import os
import numpy as np
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification

MODEL_DIR = os.path.join(PROJECT_DIR, "model", "final")

# --- 1. Load data ---
print("Loading DisfluencySpeech...")
ds = load_dataset("amaai-lab/DisfluencySpeech", split="test")
if "audio" in ds.column_names:
    ds = ds.remove_columns(["audio"])


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


all_dis_words = []
all_labels = []
all_flu_texts = []
all_dis_texts = []

for item in ds:
    dis_text = item.get("transcript_a", "").strip().lower()
    flu_text = item.get("transcript_c", "").strip().lower()
    if not dis_text or not flu_text:
        continue
    dis_words = dis_text.split()
    flu_words = flu_text.split()
    if len(dis_words) < 2 or len(flu_words) < 1:
        continue
    labels = word_labels(dis_words, flu_words)
    if labels is None:
        continue
    all_dis_words.append(dis_words)
    all_labels.append(labels)
    all_flu_texts.append(flu_text)
    all_dis_texts.append(dis_text)

print(f"Loaded {len(all_dis_words)} utterances")

# --- 2. Load model and predict ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, num_labels=2)


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


data = {"disfluent": all_dis_words, "labels": all_labels}
eval_ds = Dataset.from_dict(data)
eval_ds = eval_ds.map(tokenize_and_align, batched=True)

collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(model, data_collator=collator, processing_class=tokenizer)

print("Running predictions...")
predictions, labels_arr, _ = trainer.predict(eval_ds)
predictions = np.argmax(predictions, axis=2)

# --- 3. Reconstruct word-level predictions ---
all_pred_labels = []
for idx in range(len(all_dis_words)):
    words = all_dis_words[idx]
    tok_input = tokenizer.tokenize(words, is_split_into_words=True)
    word_ids = tokenizer(words, is_split_into_words=True).word_ids()[1:-1]

    prev_wid = None
    fc = 0
    dc = 0
    pred_labels = []
    for i, pl in enumerate(predictions[idx][1:1 + len(tok_input)]):
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

    pred_labels = pred_labels[:len(words)]
    # Pad if short
    while len(pred_labels) < len(words):
        pred_labels.append(0)
    all_pred_labels.append(pred_labels)

# --- 4. Error analysis ---
missed_words = Counter()  # gold=1, pred=0 (false negatives)
false_pos_words = Counter()  # gold=0, pred=1 (false positives)
correct_disfl = Counter()  # gold=1, pred=1 (true positives)

missed_examples = []
false_pos_examples = []

for idx in range(len(all_dis_words)):
    words = all_dis_words[idx]
    gold = all_labels[idx]
    pred = all_pred_labels[idx]

    for w, g, p in zip(words, gold, pred):
        w_clean = w.strip(".,!?;:'\"")
        if g == 1 and p == 0:
            missed_words[w_clean] += 1
        elif g == 0 and p == 1:
            false_pos_words[w_clean] += 1
        elif g == 1 and p == 1:
            correct_disfl[w_clean] += 1

    # Collect example sentences with errors
    has_miss = any(g == 1 and p == 0 for g, p in zip(gold, pred))
    has_fp = any(g == 0 and p == 1 for g, p in zip(gold, pred))

    if has_miss and len(missed_examples) < 15:
        missed_examples.append(idx)
    if has_fp and len(false_pos_examples) < 15:
        false_pos_examples.append(idx)

# --- 5. Print results ---
total_missed = sum(missed_words.values())
total_fp = sum(false_pos_words.values())
total_correct = sum(correct_disfl.values())
total_gold_disfl = total_missed + total_correct
total_pred_disfl = total_fp + total_correct

print()
print("=" * 70)
print("ERROR ANALYSIS: DisfluencySpeech")
print("=" * 70)

print(f"\nGold disfluent tokens: {total_gold_disfl}")
print(f"Correctly caught:     {total_correct} ({100*total_correct/max(total_gold_disfl,1):.1f}%)")
print(f"Missed (FN):          {total_missed} ({100*total_missed/max(total_gold_disfl,1):.1f}%)")
print(f"False positives (FP): {total_fp}")

print("\n" + "-" * 70)
print("TOP 30 MISSED WORDS (gold=disfluent, model said fluent)")
print("-" * 70)
for word, count in missed_words.most_common(30):
    print(f"  {word:<25} missed {count} times")

print("\n" + "-" * 70)
print("TOP 20 FALSE POSITIVES (gold=fluent, model said disfluent)")
print("-" * 70)
for word, count in false_pos_words.most_common(20):
    print(f"  {word:<25} false positive {count} times")

print("\n" + "-" * 70)
print("TOP 20 CORRECTLY CAUGHT (gold=disfluent, model agreed)")
print("-" * 70)
for word, count in correct_disfl.most_common(20):
    print(f"  {word:<25} caught {count} times")

# --- 6. Show missed examples ---
print("\n" + "=" * 70)
print("EXAMPLE SENTENCES WITH MISSES")
print("=" * 70)
for idx in missed_examples[:10]:
    words = all_dis_words[idx]
    gold = all_labels[idx]
    pred = all_pred_labels[idx]
    flu = all_flu_texts[idx]

    print(f"\n--- Utterance {idx} ---")
    print(f"  Disfluent: {' '.join(words)}")
    print(f"  Clean:     {flu}")
    for w, g, p in zip(words, gold, pred):
        if g == 1 and p == 0:
            print(f"    MISSED: '{w}' (gold=disfl, pred=fluent)")
        elif g == 0 and p == 1:
            print(f"    FALSE+: '{w}' (gold=fluent, pred=disfl)")
        elif g == 1 and p == 1:
            print(f"    CAUGHT: '{w}'")

# --- 7. Show false positive examples ---
print("\n" + "=" * 70)
print("EXAMPLE SENTENCES WITH FALSE POSITIVES (mislabelled)")
print("=" * 70)
for idx in false_pos_examples[:10]:
    words = all_dis_words[idx]
    gold = all_labels[idx]
    pred = all_pred_labels[idx]
    flu = all_flu_texts[idx]

    print(f"\n--- Utterance {idx} ---")
    print(f"  Disfluent: {' '.join(words)}")
    print(f"  Clean:     {flu}")
    for w, g, p in zip(words, gold, pred):
        if g == 0 and p == 1:
            print(f"    FALSE+: '{w}' (gold=fluent, pred=disfl)")
        elif g == 1 and p == 0:
            print(f"    MISSED: '{w}'")
        elif g == 1 and p == 1:
            print(f"    CAUGHT: '{w}'")
