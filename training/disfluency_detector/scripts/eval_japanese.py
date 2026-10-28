"""
Evaluate the trained model on fresh Japanese synthetic test data (not seen during training).
Generates 2000 new JA sentences with a different random seed, then computes P/R/F1.
"""
import sys
import io
import os
import random
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
import inject_japanese as ij
import prepare_labels as pl

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

# --- 1. Generate fresh JA test set ---
random.seed(9999)  # different seed from training (which used 42)

source_path = os.path.join(PROJECT_DIR, "data", "source", "japanese_clean.txt")
with open(source_path, "r", encoding="utf-8") as f:
    flu_lines = f.readlines()

print("Generating 2000 fresh JA test sentences (seed=9999)...")
test_dis, test_flu = ij.generate_data(2000, flu_lines, "ja_eval")

eval_dir = os.path.join(PROJECT_DIR, "data", "eval_ja")
os.makedirs(eval_dir, exist_ok=True)

with open(os.path.join(eval_dir, "test.dis"), "w", encoding="utf-8") as f:
    for s in test_dis:
        f.write(s + "\n")
with open(os.path.join(eval_dir, "test.flu"), "w", encoding="utf-8") as f:
    for s in test_flu:
        f.write(s + "\n")

dis_lines, label_lines, skipped = pl.generate_labels(
    os.path.join(eval_dir, "test.dis"),
    os.path.join(eval_dir, "test.flu"),
)
with open(os.path.join(eval_dir, "test.dis"), "w", encoding="utf-8") as f:
    for d in dis_lines:
        f.write(d + "\n")
with open(os.path.join(eval_dir, "test.labels"), "w", encoding="utf-8") as f:
    for l in label_lines:
        f.write(l + "\n")

print(f"Generated {len(dis_lines)} labeled JA test sentences (skipped {skipped})")

# --- 2. Load model ---
MODEL = os.path.join(PROJECT_DIR, "model", "final")
print(f"Loading model from {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=2)

# --- 3. Build dataset ---
data = {"labels": [], "disfluent": []}
with open(os.path.join(eval_dir, "test.dis"), "r", encoding="utf-8") as f_d, \
     open(os.path.join(eval_dir, "test.labels"), "r", encoding="utf-8") as f_l:
    for d_line, l_line in zip(f_d, f_l):
        words = d_line.strip().split()
        labels = list(map(int, l_line.strip().split()))
        if len(words) == len(labels) and len(words) > 0:
            data["disfluent"].append(words)
            data["labels"].append(labels)

n_loaded = len(data["disfluent"])
print(f"Loaded {n_loaded} test sentences")


def tokenize_and_align(examples):
    tok = tokenizer(examples["disfluent"], truncation=True, is_split_into_words=True)
    all_labels = []
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
        all_labels.append(lab)
    tok["labels"] = all_labels
    return tok


ds = Dataset.from_dict(data)
ds = ds.map(tokenize_and_align, batched=True)

collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(model, data_collator=collator, processing_class=tokenizer)

# --- 4. Predict and evaluate ---
print("Running predictions...")
predictions, labels, _ = trainer.predict(ds)
predictions = np.argmax(predictions, axis=2)

true_pred = [p for pred, lab in zip(predictions, labels) for p, l in zip(pred, lab) if l != -100]
true_lab = [l for pred, lab in zip(predictions, labels) for p, l in zip(pred, lab) if l != -100]

print()
print("=" * 60)
print("JAPANESE-ONLY EVALUATION (2000 fresh synthetic sentences)")
print("=" * 60)
print(classification_report(true_lab, true_pred, target_names=["is_fluent", "is_disfluent"], zero_division=0))
print("Confusion Matrix (normalized):")
cm = confusion_matrix(true_lab, true_pred, normalize="all")
print(cm)

results = precision_recall_fscore_support(true_lab, true_pred, zero_division=0)
print(f"\nSummary:")
print(f"  Fluent     - P: {results[0][0]:.4f}  R: {results[1][0]:.4f}  F1: {results[2][0]:.4f}")
print(f"  Disfluent  - P: {results[0][1]:.4f}  R: {results[1][1]:.4f}  F1: {results[2][1]:.4f}")
