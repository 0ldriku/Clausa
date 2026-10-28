"""
Prepare labeled data for disfluency detection training.
The labels follow Kundu et al. (2022), https://github.com/RKKUNDU/zero-shot-disfluency-detection.

Takes .dis (disfluent) + .flu (fluent) pairs and generates .dis + .labels files
where labels are 0 (fluent) or 1 (disfluent) at the word level.

Combines English and Japanese data into a single training set.
"""
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SYNTHETIC_DIR = os.path.join(PROJECT_DIR, "data", "synthetic")
LABELED_DIR = os.path.join(PROJECT_DIR, "data", "labeled")
os.makedirs(LABELED_DIR, exist_ok=True)


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


def generate_labels(dis_path, flu_path):
    """Generate labels from disfluent/fluent pairs. Returns (dis_lines, label_lines)."""
    with open(dis_path, 'r', encoding='utf-8') as f:
        dis_lines = f.readlines()
    with open(flu_path, 'r', encoding='utf-8') as f:
        flu_lines = f.readlines()

    out_dis = []
    out_labels = []
    skipped = 0

    for dis_line, flu_line in zip(dis_lines, flu_lines):
        dis_line = dis_line.strip()
        flu_line = flu_line.strip()

        if flu_line == "None":
            flu_line = ""

        labels = word_labels(dis_line.split(), flu_line.split())
        if labels is None:
            skipped += 1
            continue

        out_dis.append(dis_line)
        out_labels.append(" ".join(map(str, labels)))

    return out_dis, out_labels, skipped


if __name__ == "__main__":
    print("=" * 60)
    print("Preparing labeled data")
    print("=" * 60)

    all_train_dis = []
    all_train_labels = []
    all_valid_dis = []
    all_valid_labels = []

    for lang in ["en", "ja"]:
        lang_dir = os.path.join(SYNTHETIC_DIR, lang)
        if not os.path.exists(lang_dir):
            print(f"  [{lang}] No synthetic data found, skipping.")
            continue

        for split in ["train", "valid"]:
            dis_path = os.path.join(lang_dir, f"{split}.dis")
            flu_path = os.path.join(lang_dir, f"{split}.flu")

            if not os.path.exists(dis_path) or not os.path.exists(flu_path):
                print(f"  [{lang}/{split}] Files not found, skipping.")
                continue

            dis_lines, label_lines, skipped = generate_labels(dis_path, flu_path)
            print(f"  [{lang}/{split}] {len(dis_lines)} labeled (skipped {skipped})")

            if split == "train":
                all_train_dis.extend(dis_lines)
                all_train_labels.extend(label_lines)
            else:
                all_valid_dis.extend(dis_lines)
                all_valid_labels.extend(label_lines)

    # Split valid into valid + test
    valid_count = len(all_valid_dis)
    test_split = valid_count // 3  # ~1/3 for test
    test_dis = all_valid_dis[:test_split]
    test_labels = all_valid_labels[:test_split]
    valid_dis = all_valid_dis[test_split:]
    valid_labels = all_valid_labels[test_split:]

    # Write combined files
    for split, dis, labels in [
        ("train", all_train_dis, all_train_labels),
        ("valid", valid_dis, valid_labels),
        ("test", test_dis, test_labels),
    ]:
        with open(os.path.join(LABELED_DIR, f"{split}.dis"), "w", encoding="utf-8") as f:
            for line in dis:
                f.write(line + "\n")
        with open(os.path.join(LABELED_DIR, f"{split}.labels"), "w", encoding="utf-8") as f:
            for line in labels:
                f.write(line + "\n")
        print(f"  {split}: {len(dis)} sentences")

    total = len(all_train_dis) + len(valid_dis) + len(test_dis)
    print(f"\nTotal: {total} labeled sentences")
    print(f"Output: {LABELED_DIR}")
    print("Done!")
