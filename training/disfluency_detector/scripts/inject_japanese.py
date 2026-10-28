"""
Japanese L2 Disfluency Injection Script
Follows Kundu et al. (2022) approach: inject synthetic disfluencies into clean sentences.
Includes L2-specific patterns common for English L1 speakers learning Japanese.

Input: Space-tokenized Japanese sentences (from MeCab/fugashi)

Disfluency types:
  1: filled pause (えーと, あの, えー, まあ, うーん, なんか, そのー, あのー)
  2: word repetition (single token repeated)
  3: phrase repetition (2-3 token phrase repeated)
  4: self-correction (wrong word + editing marker + correct word)
  5: false start (partial token then full token)
  6: partial word (first character(s) of a token)
  7: particle correction (L2-specific: wrong particle → correct particle)
  8: reading false start (L2-specific: kana reading prefix before kanji word)
     8a: cross-script — kana prefix of kanji reading (e.g., ケイ 警察, さが 探し)
     8b: same-script — valid-word prefix of compound (e.g., バス バスケット)
     8c: mispronunciation — L2 phonological error prefix (e.g., けいさちゅ 警察)
     8d: repeated mora — stuttered first mora(s) before kanji (e.g., みみ 蜜柑, けけい 警察)

Output: .dis (disfluent) + .flu (fluent/clean) files in Kundu format
"""

import os
import re
import random
from collections import defaultdict

try:
    import fugashi
    _TAGGER = fugashi.Tagger()
except ImportError:
    _TAGGER = None
    print('WARNING: fugashi not installed. Type 8 (reading false start) disabled.')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT = os.path.join(PROJECT_DIR, "data", "source", "japanese_clean.txt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "synthetic", "ja")
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

## Config ##
MAX_NUM_OF_DISFLUENCY = 5
NUM_TRAIN = 25000
NUM_VALID = 3000
MAX_TRY = 100

# ============================================================
# Japanese L2 disfluency patterns
# ============================================================

# Filled pauses common in Japanese (both native and L2)
FILLERS = [
    "えーと", "あの", "えー", "まあ", "うーん", "なんか", "そのー", "あのー",
    "えっと", "ああ", "うん", "ええ", "まあまあ", "ちょっと", "なんていうか",
    "なんだっけ", "あー", "うー",
]

# Editing terms for self-corrections
EDITING_TERMS = [
    "あ", "いや", "じゃなくて", "ちがう", "えっと", "あの",
    "いやいや", "ごめん", "ではなく", ""
]

# Particle confusions common for English L1 speakers
PARTICLE_CONFUSIONS = {
    "が": ["は", "を", "の"],
    "は": ["が", "を", "の"],
    "を": ["が", "は", "に"],
    "に": ["で", "へ", "を", "の"],
    "で": ["に", "を", "が"],
    "へ": ["に", "で", "を"],
    "の": ["が", "は", "を"],
    "と": ["に", "で", "や"],
    "から": ["より", "に", "で"],
    "まで": ["に", "へ", "で"],
    "も": ["は", "が", "を"],
    "より": ["から", "で", "に"],
}

# Common verb form confusions for L2
VERB_CONFUSIONS = {
    "行き": ["行っ", "行か", "行く"],
    "行っ": ["行き", "行か", "行く"],
    "来": ["行き", "行っ"],
    "食べ": ["飲み", "食べる"],
    "飲み": ["食べ", "飲む"],
    "見": ["見え", "見る", "観"],
    "し": ["する", "やっ", "やり"],
    "でき": ["できる", "し", "やっ"],
    "思い": ["思っ", "考え"],
    "思っ": ["思い", "考え"],
    "言い": ["言っ", "話し"],
    "言っ": ["言い", "話し"],
    "書き": ["書い", "書く"],
    "読み": ["読ん", "読む"],
    "聞き": ["聞い", "聞く"],
    "買い": ["買っ", "買う"],
    "作り": ["作っ", "作る"],
    "歩き": ["歩い", "歩く"],
    "走り": ["走っ", "走る"],
}

# Common adjective/word confusions
WORD_CONFUSIONS = {
    "大きい": ["大きな", "おおきい"],
    "小さい": ["小さな", "ちいさい"],
    "多い": ["たくさん", "いっぱい"],
    "少ない": ["少し", "ちょっと"],
    "いい": ["良い", "よい"],
    "悪い": ["だめ", "わるい"],
    "きれい": ["きれいな", "美しい"],
    "面白い": ["おもしろい", "楽しい"],
    "難しい": ["むずかしい", "大変"],
    "簡単": ["簡単な", "やさしい", "かんたん"],
    "今日": ["きょう", "今"],
    "明日": ["あした", "あす"],
    "昨日": ["きのう", "さくじつ"],
}

PARTICLES = set(PARTICLE_CONFUSIONS.keys())


def get_filler():
    return random.choice(FILLERS)


def get_word_repetition(token):
    """Repeat a single token 1-2 times."""
    repeats = random.choices([1, 2], weights=[0.85, 0.15])[0]
    return " ".join([token] * repeats)


def get_phrase_repetition(tokens, start_idx):
    """Repeat a phrase of 2-3 tokens."""
    phrase_len = random.choices([2, 3], weights=[0.6, 0.4])[0]
    if start_idx + phrase_len > len(tokens):
        phrase_len = len(tokens) - start_idx
    if phrase_len < 2:
        return None, 0
    phrase = " ".join(tokens[start_idx:start_idx + phrase_len])
    return phrase, phrase_len


def get_self_correction(token):
    """Generate a wrong token + editing term."""
    # Check known confusions
    for confusion_dict in [VERB_CONFUSIONS, WORD_CONFUSIONS]:
        if token in confusion_dict:
            wrong = random.choice(confusion_dict[token])
            edit = random.choice(EDITING_TERMS)
            if edit:
                return f"{wrong} {edit}"
            return wrong

    # Generic: partial token as false start
    if len(token) >= 2:
        cut = random.randint(1, max(1, len(token) - 1))
        wrong = token[:cut]
        edit = random.choice(EDITING_TERMS[:6])  # shorter editing terms
        if edit:
            return f"{wrong} {edit}"
        return wrong

    return token


def get_partial_word(token):
    """First character(s) of a token as a false start."""
    if len(token) <= 1:
        return token
    cut = random.randint(1, max(1, len(token) // 2))
    return token[:cut]


def get_false_start(tokens, idx):
    """Generate a false start: partial token or abandoned word."""
    token = tokens[idx]
    if len(token) >= 2:
        cut = random.randint(1, max(1, len(token) - 1))
        partial = token[:cut]
    else:
        partial = token

    # Sometimes add a filler after the false start
    if random.random() < 0.5:
        edit = random.choice(["あ", "えっと", "あの", "うーん"])
        return f"{partial} {edit}"
    return partial


def get_particle_correction(token):
    """L2-specific: insert wrong particle before correct one."""
    if token in PARTICLE_CONFUSIONS:
        wrong = random.choice(PARTICLE_CONFUSIONS[token])
        edit = random.choice(["あ", "えっと", "じゃなくて", "いや", "あの", ""])
        if edit:
            return f"{wrong} {edit}"
        return wrong
    return None


def get_particle_indices(tokens):
    indices = []
    for idx, t in enumerate(tokens):
        if t in PARTICLES:
            indices.append(idx)
    return indices


def get_correctable_indices(tokens):
    indices = []
    all_confusions = {}
    all_confusions.update(VERB_CONFUSIONS)
    all_confusions.update(WORD_CONFUSIONS)
    for idx, t in enumerate(tokens):
        if t in all_confusions:
            indices.append(idx)
    return indices



# ============================================================
# Type 8: Reading false start (cross-script kana/kanji)
# ============================================================

# Module-level caches (populated by build_reading_cache)
KANJI_READINGS = {}   # kanji_token -> katakana reading
VOCAB = set()         # all tokens seen in source data

# L2 mispronunciation rules (English L1 -> Japanese)
# Maps katakana substrings to common L2 error variants
MISPRONUNCIATION = {
    "ツ": "チュ",    # tsu -> chu
    "ズ": "ジュ",    # zu -> ju
    "ヅ": "ジュ",    # du -> ju
    "リョウ": "ロウ",  # ryou -> rou
    "リャ": "ラ",    # rya -> ra
    "ッ": "",        # geminate dropped
    "ン": "ヌ",  # n -> nu (common L2 error)
}


def has_kanji(s):
    """Check if string contains kanji characters."""
    return any('\u4e00' <= c <= '\u9fff' for c in s)


def kata_to_hira(s):
    """Convert katakana to hiragana."""
    return ''.join(chr(ord(c) - 0x60) if '\u30A1' <= c <= '\u30F6' else c for c in s)


def apply_l2_mispronunciation(reading_kata):
    """Apply common L2 phonological errors to a katakana reading.
    Returns a mispronounced variant or None if no rule applies."""
    result = reading_kata
    applied = False
    for old, new in MISPRONUNCIATION.items():
        if old in result:
            result = result.replace(old, new, 1)
            applied = True
            break
    return result if applied and result != reading_kata else None


def build_reading_cache(flu_lines):
    """Build kanji->reading dictionary and vocab set from source sentences."""
    global KANJI_READINGS, VOCAB
    if _TAGGER is None:
        return

    for line in flu_lines:
        tokens = line.strip().split()
        for tok in tokens:
            VOCAB.add(tok)
            if has_kanji(tok) and len(tok) >= 2 and tok not in KANJI_READINGS:
                parsed = _TAGGER(tok)
                if parsed and parsed[0].feature.kana:
                    reading = parsed[0].feature.kana  # katakana
                    if len(reading) >= 3:
                        KANJI_READINGS[tok] = reading

    print(f"  Reading cache: {len(KANJI_READINGS)} kanji words, {len(VOCAB)} vocab tokens")


def get_reading_false_start(token):
    """Generate a kana reading prefix as a false start for a kanji token.

    Sub-patterns:
      8a: Cross-script kana prefix (kei before keisatsu)
      8b: Same-script valid-word prefix (basu before basuketto)
      8c: Mispronunciation prefix (keisachu before keisatsu)
    """
    sub_type = random.choice(['8a', '8a', '8b', '8c', '8d'])  # 8a cross-script, 8b same-script, 8c mispron, 8d repeated mora

    if sub_type == '8b':
        # Same-script: check if a prefix of the token is in vocab
        if len(token) >= 3:
            for cut in range(2, len(token)):
                prefix = token[:cut]
                if prefix in VOCAB and prefix != token:
                    # Sometimes add elongation
                    if random.random() < 0.3:
                        prefix = prefix + '\u30fc'
                    return prefix
        return None

    # 8a or 8c: need kanji reading
    if token not in KANJI_READINGS:
        return None
    reading = KANJI_READINGS[token]

    if sub_type == '8c':
        # Mispronunciation: apply L2 error to reading, then take prefix
        mispron = apply_l2_mispronunciation(reading)
        if mispron:
            # Take 2-4 chars of mispronounced reading
            cut = random.randint(2, min(4, len(mispron)))
            prefix = mispron[:cut]
            # Use hiragana (more natural for L2)
            prefix = kata_to_hira(prefix)
            return prefix
        # Fall through to 8a if no mispronunciation rule applied
        sub_type = '8a'

    if sub_type == '8d':
        # Repeated-mora false start: first 1-2 mora repeated (e.g., みみ before 蜜柑)
        mora_cut = random.randint(1, min(2, len(reading) - 1))
        mora_prefix = reading[:mora_cut]
        # Use hiragana (natural for L2 stuttering)
        mora_prefix = kata_to_hira(mora_prefix)
        repeats = random.choices([2, 3], weights=[0.8, 0.2])[0]
        repeated = mora_prefix * repeats
        # Sometimes add elongation on last repeat
        if random.random() < 0.3:
            repeated = repeated + 'ー'
        return repeated

    # 8a: Cross-script kana prefix
    cut = random.randint(2, min(3, len(reading) - 1))
    prefix = reading[:cut]
    # Randomly use hiragana or katakana
    if random.random() < 0.6:
        prefix = kata_to_hira(prefix)
    # Sometimes add elongation
    if random.random() < 0.25:
        prefix = prefix + '\u30fc'
    return prefix


def get_kanji_indices(tokens):
    """Find indices of tokens suitable for Type 8 injection."""
    indices = []
    for idx, tok in enumerate(tokens):
        if has_kanji(tok) and len(tok) >= 2:
            indices.append(idx)
        elif len(tok) >= 4 and tok not in PARTICLES:
            # Long katakana/hiragana words also eligible for 8b
            indices.append(idx)
    return indices


# ============================================================
# Main injection logic
# ============================================================

def inject_disfluencies_into_sentence(sentence, num_disfluencies):
    """Inject N disfluencies into a clean tokenized Japanese sentence."""
    tokens = sentence.strip().split()
    if len(tokens) < 3:
        return None

    # Weighted sampling: Type 8 at ~5% to avoid diluting types 1-7
    disfluency_codes = random.choices(
        [1, 2, 3, 4, 5, 6, 7, 8],
        weights=[0.15, 0.15, 0.12, 0.13, 0.13, 0.10, 0.12, 0.10],
        k=num_disfluencies
    )
    remaining = disfluency_codes.copy()
    used_idx = set()
    insertions = {}

    particle_idx = get_particle_indices(tokens)
    correct_idx = get_correctable_indices(tokens)
    tried = 0

    # Particle corrections (type 7)
    while 7 in remaining:
        if not particle_idx:
            return None
        idx = random.choice(particle_idx)
        if idx not in used_idx:
            result = get_particle_correction(tokens[idx])
            if result:
                insertions[idx] = result
                used_idx.add(idx)
                remaining.remove(7)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Self-corrections (type 4)
    while 4 in remaining:
        candidates = correct_idx if correct_idx else [i for i in range(len(tokens)) if len(tokens[i]) >= 2]
        if not candidates:
            return None
        idx = random.choice(candidates)
        if idx not in used_idx:
            insertions[idx] = get_self_correction(tokens[idx])
            used_idx.add(idx)
            remaining.remove(4)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Partial word (type 6)
    while 6 in remaining:
        idx = random.randint(0, len(tokens) - 1)
        if idx not in used_idx and len(tokens[idx]) >= 2:
            insertions[idx] = get_partial_word(tokens[idx])
            used_idx.add(idx)
            remaining.remove(6)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Word repetition (type 2)
    while 2 in remaining:
        idx = random.randint(0, len(tokens) - 1)
        if idx not in used_idx:
            insertions[idx] = get_word_repetition(tokens[idx])
            used_idx.add(idx)
            remaining.remove(2)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Filler (type 1)
    while 1 in remaining:
        idx = random.randint(0, len(tokens) - 1)
        if idx not in used_idx:
            insertions[idx] = get_filler()
            used_idx.add(idx)
            remaining.remove(1)
        tried += 1
        if tried > MAX_TRY:
            return None

    # False start (type 5)
    while 5 in remaining:
        idx = random.randint(0, len(tokens) - 1)
        if idx not in used_idx and len(tokens[idx]) >= 2:
            insertions[idx] = get_false_start(tokens, idx)
            used_idx.add(idx)
            remaining.remove(5)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Reading false start (type 8)
    kanji_idx = get_kanji_indices(tokens) if 8 in remaining else []
    while 8 in remaining:
        if not kanji_idx:
            # Fallback: convert to type 5 (regular false start)
            remaining[remaining.index(8)] = 5
            continue
        idx = random.choice(kanji_idx)
        if idx not in used_idx:
            result = get_reading_false_start(tokens[idx])
            if result:
                insertions[idx] = result
                used_idx.add(idx)
                remaining.remove(8)
            else:
                kanji_idx.remove(idx)
        tried += 1
        if tried > MAX_TRY:
            return None

    # Phrase repetition (type 3)
    while 3 in remaining:
        idx = random.randint(0, max(0, len(tokens) - 2))
        if idx not in used_idx:
            phrase, plen = get_phrase_repetition(tokens, idx)
            if phrase and plen >= 2:
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
    for idx, token in enumerate(tokens):
        if idx in insertions:
            disfluent_parts.append(insertions[idx])
        disfluent_parts.append(token)

    disfluent = " ".join(disfluent_parts)
    fluent = sentence.strip()

    return disfluent, fluent


def generate_data(num_examples, flu_lines, label):
    """Generate num_examples disfluent sentences."""
    dis_sentences = []
    flu_sentences = []

    count = 0
    attempts = 0
    max_attempts = num_examples * 20

    while count < num_examples and attempts < max_attempts:
        sentence = random.choice(flu_lines).strip()
        tokens = sentence.split()
        if len(tokens) < 3 or len(tokens) > 25:
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
    print("Japanese L2 Disfluency Injection")
    print("=" * 60)

    if not os.path.exists(INPUT):
        print(f"ERROR: Source file not found: {INPUT}")
        print("Run download_source_data.py first.")
        exit(1)

    with open(INPUT, "r", encoding="utf-8") as f:
        flu_lines = f.readlines()
    print(f"Loaded {len(flu_lines)} clean Japanese sentences")

    # Build reading cache for Type 8
    print("Building kanji reading cache...")
    build_reading_cache(flu_lines)

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
