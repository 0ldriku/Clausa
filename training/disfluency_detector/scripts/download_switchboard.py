"""
Download Switchboard Dialog Act corpus (swda) and extract disfluency labels.
The swda transcriptions are freely available under CC BY-NC-SA 3.0.
Source: https://github.com/cgpotts/swda

Then combine with existing synthetic EN+JA labeled data for retraining.

Output:
  data/switchboard/  - Switchboard-only labeled data (train/valid/test)
  data/combined/     - Switchboard train + synthetic EN+JA train (for retraining)
"""

import os
import sys
import re
import csv
import zipfile
import random
import shutil
from urllib.request import urlretrieve
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SWBD_DIR = os.path.join(PROJECT_DIR, "data", "switchboard")
COMBINED_DIR = os.path.join(PROJECT_DIR, "data", "combined")
LABELED_DIR = os.path.join(PROJECT_DIR, "data", "labeled")

SWDA_ZIP_URL = "https://github.com/cgpotts/swda/archive/refs/heads/master.zip"


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------
def download_swda():
    """Download swda repo as zip and extract. Returns path to extracted dir."""
    zip_path = os.path.join(SWBD_DIR, "swda-master.zip")
    extract_dir = os.path.join(SWBD_DIR, "swda-master")

    os.makedirs(SWBD_DIR, exist_ok=True)

    # Download repo zip if needed
    if not os.path.exists(extract_dir):
        print(f"Downloading swda from GitHub...")
        urlretrieve(SWDA_ZIP_URL, zip_path)
        print("Extracting repo...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(SWBD_DIR)
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # The repo contains swda.zip with the actual CSV data — extract it
    inner_zip = os.path.join(extract_dir, "swda.zip")
    inner_data = os.path.join(extract_dir, "swda")
    if os.path.exists(inner_zip) and not os.path.exists(inner_data):
        print("Extracting inner swda.zip (CSV data)...")
        with zipfile.ZipFile(inner_zip, 'r') as z:
            z.extractall(extract_dir)
        print(f"Data extracted to {inner_data}")
    elif os.path.exists(inner_data):
        print("swda data already extracted.")
    else:
        print("WARNING: swda.zip not found in repo.")

    return extract_dir


# ---------------------------------------------------------------------------
# 2. Parse disfluency annotations
# ---------------------------------------------------------------------------
def parse_swda_text(text):
    """
    Parse swda annotated text to extract words and disfluency labels.

    Annotation markers:
      {D ...} = Discourse marker (disfluent)
      {F ...} = Filled pause (disfluent)
      {E ...} = Editing term (disfluent)
      {A ...} = Aside (disfluent)
      {C ...} = Coordinating conjunction (fluent — not a disfluency)
      [ reparandum + {interregnum} repair ] = repair structure
      word- = partial word (disfluent)

    Returns: (words, labels) where labels[i] = 1 if disfluent, 0 if fluent.
             Returns (None, None) if parsing fails or empty.
    """
    if not text or not text.strip():
        return None, None

    # Clean up
    text = text.strip()
    # Remove utterance-final /
    text = re.sub(r'\s*/\s*$', '', text)
    # Remove noise markers <<...>>, <...>
    text = re.sub(r'<<[^>]*>>', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    # Remove [[ ]] (overlapping speech markers)
    text = text.replace('[[', '').replace(']]', '')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    words = []
    labels = []

    # State tracking
    repair_depth = 0          # nesting level of [...]
    before_plus = []          # stack: True = in reparandum (before +)
    brace_stack = []          # stack of brace types (D, F, E, A, C)

    i = 0
    while i < len(text):
        c = text[i]

        # --- Repair structure ---
        if c == '[':
            repair_depth += 1
            before_plus.append(True)  # start in reparandum
            i += 1

        elif c == '+' and repair_depth > 0:
            if before_plus:
                before_plus[-1] = False  # now in repair section
            i += 1

        elif c == ']' and repair_depth > 0:
            repair_depth -= 1
            if before_plus:
                before_plus.pop()
            i += 1

        # --- Brace markers ---
        elif c == '{':
            match = re.match(r'\{([A-Za-z])\s', text[i:])
            if match:
                brace_stack.append(match.group(1).upper())
                i += len(match.group(0))
            else:
                i += 1

        elif c == '}':
            if brace_stack:
                brace_stack.pop()
            i += 1

        # --- Skip markers ---
        elif c == '#':
            i += 1  # pause

        elif c == '-' and i + 1 < len(text) and text[i + 1] == '-':
            i += 2  # restart marker --

        elif c in ' \t\n':
            i += 1

        # --- Word token ---
        else:
            word_match = re.match(r'[^\s\[\]+{}/#<>]+', text[i:])
            if word_match:
                raw_word = word_match.group(0)
                i += len(raw_word)

                # Clean: lowercase, strip punctuation
                clean = raw_word.lower()
                clean = re.sub(r'^["\'\(\[]+', '', clean)
                clean = re.sub(r'["\'\)\]\.,;:!?]+$', '', clean)

                if not clean or clean == '--' or clean == '-':
                    continue

                # Determine label
                is_disfluent = False

                # Words in {D}, {F}, {E}, {A} are disfluent
                if any(bt in ('D', 'F', 'E', 'A') for bt in brace_stack):
                    is_disfluent = True

                # Words in reparandum (before + in [...]) are disfluent
                if repair_depth > 0 and before_plus and before_plus[-1]:
                    is_disfluent = True

                # Partial words (ending with -) are disfluent
                if clean.endswith('-'):
                    is_disfluent = True

                words.append(clean)
                labels.append(1 if is_disfluent else 0)
            else:
                i += 1

    if len(words) == 0:
        return None, None

    return words, labels


# ---------------------------------------------------------------------------
# 3. Conversation ID and split
# ---------------------------------------------------------------------------
def get_conv_id(filename):
    """Extract conversation ID from swda CSV filename like sw_0001_4325.utt.csv"""
    match = re.search(r'sw_\d+_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def get_split(conv_id):
    """
    Standard Switchboard split following Wang et al. (2021) / Kundu et al. (2022):
      Test:  sw_04[0-1]*.utt  → conv IDs 4000-4199
      Dev:   sw_04[5-9]*.utt  → conv IDs 4500-4999
      Train: everything else
    """
    if 4000 <= conv_id <= 4199:
        return "test"
    elif 4500 <= conv_id <= 4999:
        return "valid"
    else:
        return "train"


# ---------------------------------------------------------------------------
# 4. Process all CSV files
# ---------------------------------------------------------------------------
def process_swda(swda_dir):
    """
    Process all swda CSV files, extract disfluency labels.
    Returns dict: {split: [(words, labels), ...]}
    """
    # Find the swda data directory (contains sw*utt/ subdirs)
    data_dir = os.path.join(swda_dir, "swda")
    if not os.path.exists(data_dir):
        # Try alternate structure
        for d in os.listdir(swda_dir):
            candidate = os.path.join(swda_dir, d, "swda")
            if os.path.exists(candidate):
                data_dir = candidate
                break

    if not os.path.exists(data_dir):
        print(f"ERROR: Could not find swda data directory in {swda_dir}")
        sys.exit(1)

    print(f"Processing swda data from {data_dir}")

    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.utt.csv'):
                csv_files.append(os.path.join(root, f))

    print(f"Found {len(csv_files)} conversation files")

    split_data = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(int))
    parse_errors = 0
    total_utts = 0
    disfluent_utts = 0

    for csv_path in sorted(csv_files):
        filename = os.path.basename(csv_path)
        conv_id = get_conv_id(filename)
        if conv_id is None:
            continue

        split = get_split(conv_id)

        # Read CSV rows
        # swda CSV: caller, act_tag, caller_no, utterance_index, subutterance_index, text, ...
        # The text column might be at different positions. Let's find it by looking for annotation markers.
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                # swda CSV columns:
                #  0: swda_filename, 1: ptb_basename, 2: conversation_no,
                #  3: transcript_index, 4: act_tag, 5: caller,
                #  6: utterance_index, 7: subutterance_index, 8: text,
                #  9: pos, 10: trees, 11: ptb_treenumbers
                utterances = defaultdict(list)
                for row_num, row in enumerate(reader):
                    # Skip header
                    if row_num == 0 and row[0] == 'swda_filename':
                        continue
                    if len(row) < 9:
                        continue
                    text = row[8]
                    try:
                        utt_idx = int(row[6])
                    except (ValueError, IndexError):
                        continue
                    caller = row[5]
                    key = (caller, utt_idx)
                    utterances[key].append(text)

                # Process each utterance
                for key in sorted(utterances.keys()):
                    # Concatenate sub-utterances
                    full_text = ' '.join(utterances[key])
                    total_utts += 1

                    words, word_labels = parse_swda_text(full_text)
                    if words is None or len(words) < 2:
                        continue

                    has_disfluency = any(l == 1 for l in word_labels)

                    if has_disfluency:
                        disfluent_utts += 1

                    # For training: only sentences WITH disfluencies (following Kundu)
                    # For valid/test: include all sentences
                    if split == "train" and not has_disfluency:
                        continue

                    split_data[split].append((words, word_labels))
                    stats[split]['total'] += 1
                    if has_disfluency:
                        stats[split]['disfluent'] += 1
                    else:
                        stats[split]['fluent'] += 1

        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:
                print(f"  Parse error in {filename}: {e}")

    print(f"\nProcessing complete:")
    print(f"  Total utterances parsed: {total_utts}")
    print(f"  Utterances with disfluencies: {disfluent_utts}")
    print(f"  Parse errors: {parse_errors}")
    for split in ['train', 'valid', 'test']:
        s = stats[split]
        print(f"  {split}: {s['total']} sentences ({s['disfluent']} disfluent, {s['fluent']} fluent)")

    return split_data


# ---------------------------------------------------------------------------
# 5. Save data
# ---------------------------------------------------------------------------
def save_split(data, output_dir, split_name):
    """Save (words, labels) pairs to .dis and .labels files."""
    os.makedirs(output_dir, exist_ok=True)
    dis_path = os.path.join(output_dir, f"{split_name}.dis")
    lab_path = os.path.join(output_dir, f"{split_name}.labels")

    with open(dis_path, 'w', encoding='utf-8') as f_dis, \
         open(lab_path, 'w', encoding='utf-8') as f_lab:
        for words, labels in data:
            f_dis.write(' '.join(words) + '\n')
            f_lab.write(' '.join(map(str, labels)) + '\n')

    print(f"  Saved {split_name}: {len(data)} sentences -> {dis_path}")


# ---------------------------------------------------------------------------
# 6. Combine Switchboard + synthetic
# ---------------------------------------------------------------------------
def combine_with_synthetic():
    """
    Merge Switchboard training data with existing synthetic EN+JA training data.
    Valid and test remain synthetic-only for fair comparison.
    """
    os.makedirs(COMBINED_DIR, exist_ok=True)

    swbd_train_dis = os.path.join(SWBD_DIR, "train.dis")
    swbd_train_lab = os.path.join(SWBD_DIR, "train.labels")
    synth_train_dis = os.path.join(LABELED_DIR, "train.dis")
    synth_train_lab = os.path.join(LABELED_DIR, "train.labels")

    # Check files exist
    for p in [swbd_train_dis, swbd_train_lab, synth_train_dis, synth_train_lab]:
        if not os.path.exists(p):
            print(f"ERROR: Required file not found: {p}")
            return False

    # Combine training data
    print("\nCombining training data...")
    combined_dis = []
    combined_lab = []

    # Read synthetic
    with open(synth_train_dis, 'r', encoding='utf-8') as f:
        synth_dis_lines = f.readlines()
    with open(synth_train_lab, 'r', encoding='utf-8') as f:
        synth_lab_lines = f.readlines()
    print(f"  Synthetic train: {len(synth_dis_lines)} sentences")
    combined_dis.extend(synth_dis_lines)
    combined_lab.extend(synth_lab_lines)

    # Read Switchboard
    with open(swbd_train_dis, 'r', encoding='utf-8') as f:
        swbd_dis_lines = f.readlines()
    with open(swbd_train_lab, 'r', encoding='utf-8') as f:
        swbd_lab_lines = f.readlines()
    print(f"  Switchboard train: {len(swbd_dis_lines)} sentences")
    combined_dis.extend(swbd_dis_lines)
    combined_lab.extend(swbd_lab_lines)

    # Shuffle combined data
    random.seed(42)
    indices = list(range(len(combined_dis)))
    random.shuffle(indices)
    combined_dis = [combined_dis[i] for i in indices]
    combined_lab = [combined_lab[i] for i in indices]

    # Write combined train
    with open(os.path.join(COMBINED_DIR, "train.dis"), 'w', encoding='utf-8') as f:
        f.writelines(combined_dis)
    with open(os.path.join(COMBINED_DIR, "train.labels"), 'w', encoding='utf-8') as f:
        f.writelines(combined_lab)
    print(f"  Combined train: {len(combined_dis)} sentences")

    # Copy synthetic valid/test as-is
    for split in ["valid", "test"]:
        for ext in [".dis", ".labels"]:
            src = os.path.join(LABELED_DIR, f"{split}{ext}")
            dst = os.path.join(COMBINED_DIR, f"{split}{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
    print(f"  Valid/test: copied from synthetic data (unchanged)")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Switchboard Disfluency Data Pipeline")
    print("=" * 60)

    # 1. Download swda
    swda_dir = download_swda()

    # 2. Process and extract disfluency labels
    split_data = process_swda(swda_dir)

    # 3. Save Switchboard-only data
    print("\nSaving Switchboard data...")
    for split in ['train', 'valid', 'test']:
        if split in split_data:
            save_split(split_data[split], SWBD_DIR, split)

    # 4. Quick sanity check — show first 3 examples
    print("\n--- Sample sentences ---")
    for split in ['train']:
        if split in split_data and len(split_data[split]) > 0:
            for idx in range(min(5, len(split_data[split]))):
                words, labs = split_data[split][idx]
                dis_words = [w for w, l in zip(words, labs) if l == 1]
                flu_words = [w for w, l in zip(words, labs) if l == 0]
                print(f"  Full:    {' '.join(words)}")
                print(f"  Labels:  {' '.join(map(str, labs))}")
                print(f"  Disfl:   {dis_words}")
                print(f"  Fluent:  {' '.join(flu_words)}")
                print()

    # 5. Combine with synthetic data
    print("=" * 60)
    print("Combining with synthetic EN+JA data")
    print("=" * 60)
    success = combine_with_synthetic()

    if success:
        print(f"\nOutput directories:")
        print(f"  Switchboard only: {SWBD_DIR}")
        print(f"  Combined (for retraining): {COMBINED_DIR}")
        print(f"\nTo retrain, run:")
        print(f"  python scripts/train.py -d data/combined -o model_v2")
    else:
        print("\nCombining failed. Check files above.")

    print("\nDone!")
