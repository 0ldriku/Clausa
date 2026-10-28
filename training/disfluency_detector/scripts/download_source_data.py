"""
Download clean source sentences for English and Japanese.
English: NLTK brown + gutenberg corpora (diverse, pre-tokenized)
Japanese: Tatoeba sentences, tokenized with fugashi/MeCab
"""
import os
import sys
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "source")
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_COUNT = 50000  # download more than needed, filter later

# ============================================================
# English: NLTK corpora
# ============================================================
def download_english():
    import nltk
    nltk.download('brown', quiet=True)
    nltk.download('gutenberg', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    from nltk.corpus import brown, gutenberg

    sentences = []

    # Brown corpus - diverse genres
    for sent in brown.sents():
        text = " ".join(sent)
        word_count = len(sent)
        # Filter: 5-30 words, no weird characters
        if 5 <= word_count <= 30 and all(c.isalpha() or c in " '-.,!?" for c in text):
            # Lowercase and strip punctuation for clean input
            clean = " ".join(w.lower() for w in sent if w.isalpha())
            if len(clean.split()) >= 5:
                sentences.append(clean)

    # Gutenberg corpus
    for fileid in gutenberg.fileids():
        for sent in gutenberg.sents(fileid):
            text = " ".join(sent)
            word_count = len(sent)
            if 5 <= word_count <= 30 and all(c.isalpha() or c in " '-.,!?" for c in text):
                clean = " ".join(w.lower() for w in sent if w.isalpha())
                if len(clean.split()) >= 5:
                    sentences.append(clean)

    # Deduplicate
    sentences = list(set(sentences))
    random.shuffle(sentences)
    sentences = sentences[:TARGET_COUNT]

    out_path = os.path.join(DATA_DIR, "english_clean.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"English: {len(sentences)} clean sentences saved to {out_path}")
    return len(sentences)


# ============================================================
# Japanese: Tatoeba sentences, tokenized with fugashi
# ============================================================
def download_japanese():
    try:
        import fugashi
    except ImportError:
        print("Installing fugashi + unidic-lite for Japanese tokenization...")
        os.system(f"{sys.executable} -m pip install fugashi unidic-lite")
        import fugashi

    tagger = fugashi.Tagger()

    # Try to download Tatoeba Japanese sentences
    tatoeba_path = os.path.join(DATA_DIR, "tatoeba_jpn.tsv")

    if not os.path.exists(tatoeba_path):
        print("Downloading Tatoeba Japanese sentences...")
        import urllib.request
        url = "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences_detailed.tsv.bz2"
        bz2_path = tatoeba_path + ".bz2"
        try:
            urllib.request.urlretrieve(url, bz2_path)
            import bz2
            with bz2.open(bz2_path, 'rt', encoding='utf-8') as f_in:
                with open(tatoeba_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(f_in.read())
            os.remove(bz2_path)
            print("Downloaded and extracted Tatoeba Japanese sentences.")
        except Exception as e:
            print(f"Could not download Tatoeba: {e}")
            print("Falling back to built-in Japanese sentences...")
            return generate_fallback_japanese(tagger)

    # Parse Tatoeba TSV: id\tlang\tsentence\t...
    sentences = []
    with open(tatoeba_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                sent = parts[2].strip()
                # Filter: reasonable length, no English mixed in
                if 3 <= len(sent) <= 80 and not any(c.isascii() and c.isalpha() for c in sent):
                    # Tokenize with MeCab
                    words = [word.surface for word in tagger(sent)]
                    if 3 <= len(words) <= 25:
                        sentences.append(" ".join(words))

    sentences = list(set(sentences))
    random.shuffle(sentences)
    sentences = sentences[:TARGET_COUNT]

    out_path = os.path.join(DATA_DIR, "japanese_clean.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"Japanese: {len(sentences)} clean tokenized sentences saved to {out_path}")
    return len(sentences)


def generate_fallback_japanese(tagger):
    """Fallback: generate Japanese sentences from basic patterns if Tatoeba unavailable."""
    # Basic sentence templates
    subjects = ["私", "彼", "彼女", "友達", "先生", "学生", "母", "父", "子供", "猫", "犬",
                 "田中さん", "山田さん", "鈴木さん", "佐藤さん", "みんな", "あの人"]
    places = ["学校", "家", "公園", "図書館", "駅", "病院", "スーパー", "レストラン", "映画館",
              "大学", "会社", "コンビニ", "空港", "ホテル", "カフェ", "銀行", "郵便局"]
    actions = ["行きました", "来ました", "帰りました", "食べました", "飲みました", "見ました",
               "読みました", "書きました", "話しました", "聞きました", "買いました", "作りました",
               "歩きました", "走りました", "泳ぎました", "遊びました", "勉強しました", "働きました"]
    particles_place = ["に", "で", "へ", "から", "まで"]
    objects = ["本", "映画", "音楽", "料理", "写真", "手紙", "新聞", "雑誌", "ゲーム",
               "ケーキ", "コーヒー", "お茶", "水", "ビール", "パン", "魚", "肉", "野菜"]
    adjectives = ["大きい", "小さい", "新しい", "古い", "美しい", "きれいな", "面白い",
                  "楽しい", "難しい", "簡単な", "高い", "安い", "長い", "短い", "おいしい"]
    adverbs = ["とても", "すごく", "ちょっと", "少し", "たくさん", "いつも", "よく", "時々"]

    sentences = set()
    for _ in range(TARGET_COUNT * 3):
        pattern = random.choice(range(5))
        if pattern == 0:
            # Subject が place に action
            s = f"{random.choice(subjects)} が {random.choice(places)} {random.choice(particles_place)} {random.choice(actions)}"
        elif pattern == 1:
            # Subject は object を action
            s = f"{random.choice(subjects)} は {random.choice(objects)} を {random.choice(actions)}"
        elif pattern == 2:
            # Subject は adverb adjective です
            s = f"{random.choice(subjects)} は {random.choice(adverbs)} {random.choice(adjectives)} です"
        elif pattern == 3:
            # place で object を action
            s = f"{random.choice(places)} で {random.choice(objects)} を {random.choice(actions)}"
        else:
            # Subject は adverb place に action
            s = f"{random.choice(subjects)} は {random.choice(adverbs)} {random.choice(places)} {random.choice(particles_place)} {random.choice(actions)}"

        # Tokenize
        words = [word.surface for word in tagger(s)]
        sentences.add(" ".join(words))

    sentences = list(sentences)[:TARGET_COUNT]

    out_path = os.path.join(DATA_DIR, "japanese_clean.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"Japanese (fallback): {len(sentences)} clean tokenized sentences saved to {out_path}")
    return len(sentences)


if __name__ == "__main__":
    random.seed(42)
    print("=" * 60)
    print("Downloading clean source sentences")
    print("=" * 60)

    en_count = download_english()
    ja_count = download_japanese()

    print(f"\nDone! English: {en_count}, Japanese: {ja_count}")
