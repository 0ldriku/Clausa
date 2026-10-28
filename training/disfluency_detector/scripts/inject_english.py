"""
English L2 Disfluency Injection Script
Follows Kundu et al. (2022) approach: inject synthetic disfluencies into clean sentences.
Includes L2-specific patterns common for Japanese L1 speakers learning English.

Disfluency types:
  1: filled pause (um, uh, er, like, you know, well, so, I mean)
  2: word repetition (single word repeated)
  3: phrase repetition (2-4 word phrase repeated)
  4: self-correction (wrong word + editing term + correct word)
  5: false start / restart (partial phrase abandoned)
  6: partial word (word truncated, then said fully)
  7: article/preposition correction (L2-specific: wrong function word corrected)

Output: .dis (disfluent) + .flu (fluent/clean) files in Kundu format
"""

import os
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT = os.path.join(PROJECT_DIR, "data", "source", "english_clean.txt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "synthetic", "en")
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

## Config ##
MAX_NUM_OF_DISFLUENCY = 5
NUM_TRAIN = 20000
NUM_VALID = 2500
MAX_TRY = 100

# ============================================================
# Disfluency generation functions
# ============================================================

FILLERS = [
    "um", "uh", "er", "erm", "like", "you know", "well", "so",
    "I mean", "hmm", "right", "okay", "let me see", "how do you say",
    "what is it", "uh like"
]

EDITING_TERMS = [
    "no", "I mean", "no I mean", "wait", "sorry", "uh",
    "no no", "well", "actually", ""
]

# Common L2 confusions (Japanese L1 → English)
ARTICLE_CONFUSIONS = {
    "the": ["a", ""],
    "a": ["the", ""],
    "an": ["the", "a", ""],
}

PREPOSITION_CONFUSIONS = {
    "to": ["at", "for", "in"],
    "at": ["in", "to", "on"],
    "in": ["at", "on", "to"],
    "on": ["in", "at", "to"],
    "for": ["to", "of", "at"],
    "of": ["for", "from", "about"],
    "with": ["by", "from", "to"],
    "from": ["of", "by", "to"],
    "by": ["with", "from", "at"],
    "about": ["of", "for", "on"],
}

# Common word-level L2 confusions
WORD_CONFUSIONS = {
    "go": ["went", "going", "goes"],
    "went": ["go", "going", "gone"],
    "is": ["are", "was", "were"],
    "are": ["is", "was", "were"],
    "was": ["is", "were", "are"],
    "were": ["was", "are", "is"],
    "has": ["have", "had"],
    "have": ["has", "had"],
    "had": ["has", "have"],
    "do": ["does", "did"],
    "does": ["do", "did"],
    "did": ["do", "does"],
    "say": ["said", "tell", "told"],
    "said": ["say", "tell", "told"],
    "make": ["made", "do", "did"],
    "made": ["make", "do"],
    "take": ["took", "bring", "brought"],
    "took": ["take", "bring"],
    "come": ["came", "go", "went"],
    "came": ["come", "go", "went"],
    "see": ["saw", "look", "watch"],
    "saw": ["see", "looked", "watched"],
    "think": ["thought", "believe"],
    "thought": ["think", "believed"],
    "big": ["large", "great"],
    "small": ["little", "tiny"],
    "good": ["nice", "great", "well"],
    "bad": ["terrible", "poor", "worse"],
    "many": ["much", "a lot of", "lots of"],
    "much": ["many", "a lot of"],
}


def get_filler():
    return random.choice(FILLERS)


def get_word_repetition(word):
    """Repeat a single word 1-2 times."""
    repeats = random.choices([1, 2], weights=[0.8, 0.2])[0]
    return " ".join([word] * repeats)


def get_phrase_repetition(words, start_idx):
    """Repeat a phrase of 2-4 words."""
    phrase_len = random.choices([2, 3, 4], weights=[0.5, 0.3, 0.2])[0]
    if start_idx + phrase_len > len(words):
        phrase_len = len(words) - start_idx
    if phrase_len < 2:
        return None, 0
    phrase = " ".join(words[start_idx:start_idx + phrase_len])
    return phrase, phrase_len


def get_self_correction(word):
    """Generate a wrong word + editing term before the correct word."""
    # Check if we have a known confusion
    if word.lower() in WORD_CONFUSIONS:
        wrong = random.choice(WORD_CONFUSIONS[word.lower()])
    else:
        # Generic: repeat with slight modification or use a filler
        if len(word) > 3:
            wrong = word[:len(word)//2 + random.randint(0, 2)]
        else:
            wrong = word
    edit_term = random.choice(EDITING_TERMS)
    if edit_term:
        return f"{wrong} {edit_term}"
    else:
        return wrong


def get_partial_word(word):
    """Truncate a word to simulate a false start at the word level."""
    if len(word) <= 2:
        return word[0]
    cut = random.randint(1, max(1, len(word) // 2))
    return word[:cut]


def get_false_start(words, idx):
    """Generate a false start: 1-3 random words that get abandoned."""
    starter_phrases = [
        "the", "I", "so", "and then", "but", "it was", "he", "she",
        "they", "we", "the thing is", "I think", "you", "that"
    ]
    starter = random.choice(starter_phrases)
    edit = random.choice(["uh", "um", "no", "wait", ""])
    if edit:
        return f"{starter} {edit}"
    return starter


def get_article_prep_correction(word):
    """L2-specific: insert a wrong article/preposition before the correct one."""
    word_lower = word.lower()
    if word_lower in ARTICLE_CONFUSIONS:
        wrong = random.choice(ARTICLE_CONFUSIONS[word_lower])
        if wrong == "":
            return None  # skip - empty replacement doesn't work well
        edit = random.choice(["uh", "no", "I mean", "wait", ""])
        if edit:
            return f"{wrong} {edit}"
        return wrong
    elif word_lower in PREPOSITION_CONFUSIONS:
        wrong = random.choice(PREPOSITION_CONFUSIONS[word_lower])
        edit = random.choice(["uh", "no", "I mean", "um", ""])
        if edit:
            return f"{wrong} {edit}"
        return wrong
    return None


# ============================================================
# Main injection logic (follows Kundu's pattern)
# ============================================================

DISFLUENCY_TYPES = {
    1: 'filler',
    2: 'word_repetition',
    3: 'phrase_repetition',
    4: 'self_correction',
    5: 'false_start',
    6: 'partial_word',
    7: 'article_prep_correction',
}


def get_article_prep_indices(words):
    indices = []
    articles_preps = set(ARTICLE_CONFUSIONS.keys()) | set(PREPOSITION_CONFUSIONS.keys())
    for idx, w in enumerate(words):
        if w.lower() in articles_preps:
            indices.append(idx)
    return indices


def get_correctable_indices(words):
    indices = []
    for idx, w in enumerate(words):
        if w.lower() in WORD_CONFUSIONS:
            indices.append(idx)
    return indices


def inject_disfluencies_into_sentence(sentence, num_disfluencies):
    """Inject N disfluencies into a clean sentence. Returns (disfluent, fluent) or None."""
    words = sentence.strip().split()
    if len(words) < 5:
        return None

    disfluency_codes = [random.randint(1, 7) for _ in range(num_disfluencies)]
    remaining = disfluency_codes.copy()
    used_idx = set()
    insertions = {}  # idx -> text to insert BEFORE words[idx]

    art_prep_idx = get_article_prep_indices(words)
    correct_idx = get_correctable_indices(words)

    tried = 0

    # Article/prep corrections (type 7)
    while 7 in remaining:
        if not art_prep_idx:
            return None  # can't satisfy, retry with different sentence
        idx = random.choice(art_prep_idx)
        if idx not in used_idx:
            result = get_article_prep_correction(words[idx])
            if result:
                insertions[idx] = result
                used_idx.add(idx)
                remaining.remove(7)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Self-corrections (type 4)
    while 4 in remaining:
        candidates = correct_idx if correct_idx else list(range(len(words)))
        idx = random.choice(candidates)
        if idx not in used_idx:
            insertions[idx] = get_self_correction(words[idx])
            used_idx.add(idx)
            remaining.remove(4)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Partial word (type 6)
    while 6 in remaining:
        idx = random.randint(0, len(words) - 1)
        if idx not in used_idx and len(words[idx]) > 2:
            insertions[idx] = get_partial_word(words[idx])
            used_idx.add(idx)
            remaining.remove(6)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Word repetition (type 2)
    while 2 in remaining:
        idx = random.randint(0, len(words) - 1)
        if idx not in used_idx:
            insertions[idx] = get_word_repetition(words[idx])
            used_idx.add(idx)
            remaining.remove(2)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Filler (type 1)
    while 1 in remaining:
        idx = random.randint(0, len(words) - 1)
        if idx not in used_idx:
            insertions[idx] = get_filler()
            used_idx.add(idx)
            remaining.remove(1)
        tried += 1
        if tried > MAX_TRY:
            return None

    # False start (type 5)
    while 5 in remaining:
        idx = random.randint(0, len(words) - 1)
        if idx not in used_idx:
            insertions[idx] = get_false_start(words, idx)
            used_idx.add(idx)
            remaining.remove(5)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Phrase repetition (type 3)
    while 3 in remaining:
        idx = random.randint(0, max(0, len(words) - 2))
        if idx not in used_idx:
            phrase, plen = get_phrase_repetition(words, idx)
            if phrase and plen >= 2:
                # Check no overlap with existing insertions
                overlap = any(i in used_idx for i in range(idx, idx + plen))
                if not overlap:
                    insertions[idx] = phrase
                    for i in range(idx, idx + plen):
                        used_idx.add(i)
                    remaining.remove(3)
        tried += 1
        if tried > MAX_TRY:
            return None

    if remaining:
        return None

    # Build disfluent sentence
    disfluent_parts = []
    for idx, word in enumerate(words):
        if idx in insertions:
            disfluent_parts.append(insertions[idx])
        disfluent_parts.append(word)

    disfluent = " ".join(disfluent_parts)
    fluent = sentence.strip()

    return disfluent, fluent


def generate_data(num_examples, flu_lines, label):
    """Generate num_examples disfluent sentences."""
    dis_sentences = []
    flu_sentences = []
    type_counts = defaultdict(int)

    count = 0
    attempts = 0
    max_attempts = num_examples * 20

    while count < num_examples and attempts < max_attempts:
        sentence = random.choice(flu_lines).strip()
        if len(sentence.split()) < 5 or len(sentence.split()) > 30:
            attempts += 1
            continue

        num_disfl = random.choices([1, 2, 3, 4, 5], weights=[0.25, 0.30, 0.25, 0.12, 0.08])[0]
        result = inject_disfluencies_into_sentence(sentence, num_disfl)

        if result:
            dis, flu = result
            dis_sentences.append(dis)
            flu_sentences.append(flu)
            count += 1

            if count % 2000 == 0:
                print(f"  [{label}] {count}/{num_examples}")

        attempts += 1

    print(f"  [{label}] Generated {count}/{num_examples} (attempts: {attempts})")
    return dis_sentences, flu_sentences


if __name__ == "__main__":
    print("=" * 60)
    print("English L2 Disfluency Injection")
    print("=" * 60)

    if not os.path.exists(INPUT):
        print(f"ERROR: Source file not found: {INPUT}")
        print("Run download_source_data.py first.")
        exit(1)

    with open(INPUT, "r", encoding="utf-8") as f:
        flu_lines = f.readlines()
    print(f"Loaded {len(flu_lines)} clean English sentences")

    # Generate train
    print(f"\nGenerating train ({NUM_TRAIN})...")
    train_dis, train_flu = generate_data(NUM_TRAIN, flu_lines, "train")

    # Generate valid
    print(f"\nGenerating valid ({NUM_VALID})...")
    valid_dis, valid_flu = generate_data(NUM_VALID, flu_lines, "valid")

    # Write output files
    for split, dis, flu in [("train", train_dis, train_flu), ("valid", valid_dis, valid_flu)]:
        with open(os.path.join(OUTPUT_DIR, f"{split}.dis"), "w", encoding="utf-8") as f:
            for s in dis:
                f.write(s + "\n")
        with open(os.path.join(OUTPUT_DIR, f"{split}.flu"), "w", encoding="utf-8") as f:
            for s in flu:
                f.write(s + "\n")
        print(f"Saved {split}: {len(dis)} sentences")

    print(f"\nOutput: {OUTPUT_DIR}")
    print("Done!")
