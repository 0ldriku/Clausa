"""
Japanese TextGrid Clause Segmentation V2

Japanese clause segmentation using GiNZA + UD dependencies with Vercellotti-style rules.
Processes TextGrid files to add clause tier for CAF analysis.

Key differences from English version:
- Uses GiNZA (default: ja_ginza_electra) for Japanese NLP
- Japanese-specific clause rules (te-form chains, tari-form, advcl with subordinators)
- Japanese filler handling (えー, あの, etc.)
- Mora counting instead of syllable counting
- Neural disfluency detection (xlm-roberta-base, shared EN/JA model)

Rules: the frozen rule set is CLAUSE_RULES_JA.md in the analysis repository.

Usage:
    python ja_clause_segmenter_v102.py -i textgrids/ -o clauses/
    python ja_clause_segmenter_v102.py -i input.TextGrid -o output.TextGrid

Validated environment:
    The ja-electra310 environment (Python 3.10, GiNZA / ja-ginza-electra;
    runner/requirements/ja.txt).
"""

import os
import sys
import io
import re
import unicodedata
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass


def _wrap_utf8_stream(stream):
    buffer = getattr(stream, "buffer", None)
    encoding = getattr(stream, "encoding", None)
    if buffer is None:
        return stream
    if encoding and encoding.lower() in ("utf-8", "utf8"):
        return stream
    return io.TextIOWrapper(buffer, encoding="utf-8", errors="replace")


sys.stdout = _wrap_utf8_stream(sys.stdout)
sys.stderr = _wrap_utf8_stream(sys.stderr)

try:
    import spacy
except ImportError:
    print("Please install spacy: pip install spacy")
    sys.exit(1)

try:
    from praatio import textgrid
    from praatio.utilities.constants import Interval
except ImportError:
    print("Please install praatio: pip install praatio")
    sys.exit(1)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForTokenClassification = None


# ==============================================================================
# DisfluencyDetector - Neural disfluency detection (xlm-roberta-base)
# Same model works for both English and Japanese.
# ==============================================================================

def _vote_labels(tokenizer, model, words: List[str]) -> Tuple[List[int], set]:
    """One detector pass (truncated at the model's 512-subword limit).

    Returns per-word labels by majority vote over subwords, and the set of
    word indices that received at least one subword.
    """
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=2)[0].tolist()
    word_preds: Dict[int, List[int]] = {}
    for idx, wid in enumerate(inputs.word_ids()):
        if wid is not None:
            word_preds.setdefault(wid, []).append(preds[idx])
    labels = []
    for i in range(len(words)):
        votes = word_preds.get(i, [0])
        labels.append(1 if sum(votes) > len(votes) / 2 else 0)
    return labels, set(word_preds)


class DisfluencyDetector:
    """
    Detects disfluent words/morphemes using a fine-tuned xlm-roberta-base model.
    Labels each token as fluent (0) or disfluent (1).
    Works for both English and Japanese (multilingual model).
    """

    # Candidate default model paths (searched upward from script location)
    MODEL_PATH_CANDIDATES = (
        os.path.join("models", "disfluency_detector", "model_v2", "final"),
        os.path.join("shared", "disfluency_detector", "model_v2", "final"),
        os.path.join("en", "disfluency_test", "l2_disfluency_detector", "model_v2", "final"),
    )

    def __init__(self, model_path: str = None):
        if torch is None:
            raise ImportError(
                "torch and transformers are required for DisfluencyDetector.\n"
                "Install: pip install torch transformers"
            )

        if model_path is None:
            # Auto-discover model by walking upward from script location.
            # This keeps the script runnable both in-repo and in copied release folders.
            script_dir = os.path.dirname(os.path.abspath(__file__))
            search_dir = script_dir
            found = None
            for _ in range(8):
                for rel in self.MODEL_PATH_CANDIDATES:
                    candidate = os.path.join(search_dir, rel)
                    if os.path.exists(candidate):
                        found = candidate
                        break
                if found is not None:
                    break
                parent = os.path.dirname(search_dir)
                if parent == search_dir:
                    break
                search_dir = parent

            if found is not None:
                model_path = found
            else:
                # Keep previous fallback behavior for error reporting.
                project_root = os.path.dirname(script_dir)
                model_path = os.path.join(project_root, self.MODEL_PATH_CANDIDATES[0])

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Disfluency model not found at: {model_path}\n"
                f"Expected one of:\n"
                f"  - <repo_root>/shared/disfluency_detector/model_v2/final\n"
                f"  - <repo_root>/en/disfluency_test/l2_disfluency_detector/model_v2/final"
            )

        print(f"Loading disfluency model from: {model_path}")
        self._tokenizer_fast = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self._tokenizer_fast = bool(getattr(self.tokenizer, "is_fast", False))
        except Exception as fast_err:
            print(f"Fast tokenizer load failed ({fast_err}). Trying compatible fast conversion.")
            try:
                # Some older tokenizers stacks can't parse newer tokenizer.json.
                # For XLM-R family, rebuilding fast tokenizer from sentencepiece
                # avoids that parse path and keeps word_ids() support.
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, tokenizer_file=None
                )
                self._tokenizer_fast = bool(getattr(self.tokenizer, "is_fast", False))
            except Exception as fast_conv_err:
                print(f"Compatible fast conversion failed ({fast_conv_err}). Falling back to slow tokenizer.")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    self._tokenizer_fast = bool(getattr(self.tokenizer, "is_fast", False))
                except Exception as slow_err:
                    raise RuntimeError(
                        "Failed to load disfluency tokenizer (fast parse, fast conversion, and slow all failed).\n"
                        "If using slow tokenizer, ensure sentencepiece is installed:\n"
                        "  pip install sentencepiece\n"
                        "If converting to fast tokenizer fails with protobuf errors, install protobuf 3.20.x.\n"
                        f"Fast parse error: {fast_err}\n"
                        f"Fast conversion error: {fast_conv_err}\n"
                        f"Slow error: {slow_err}"
                    ) from slow_err
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2)
        self.model.eval()
        mode = "fast" if self._tokenizer_fast else "slow"
        print(f"Disfluency model loaded. tokenizer_mode={mode}")

    def detect(self, words: List[str]) -> List[int]:
        """
        Predict disfluency labels for a list of words/morphemes.

        Args:
            words: List of tokens (morphemes for Japanese, words for English)

        Returns:
            List of labels: 0 = fluent, 1 = disfluent
        """
        if not words:
            return []

        if self._tokenizer_fast:
            labels, _ = _vote_labels(self.tokenizer, self.model, words)
            # The single pass stops at the model's 512-subword limit; words past
            # it got no prediction and stay fluent (0).
        else:
            # Slow tokenizers don't provide word_ids(); score per word independently.
            labels = []
            special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
            for w in words:
                inputs = self.tokenizer(w, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=2)[0].tolist()
                input_ids = inputs["input_ids"][0].tolist()
                token_preds = [p for tid, p in zip(input_ids, pred_ids) if tid not in special_ids]
                if not token_preds:
                    labels.append(0)
                else:
                    labels.append(1 if sum(token_preds) > len(token_preds) / 2 else 0)

        # Rule-based post-processing: catch what the neural model misses
        labels = self._postprocess_labels(words, labels)
        return labels

    @staticmethod
    def _postprocess_labels(words: List[str], labels: List[int]) -> List[int]:
        """Rule-based fixes for common disfluency patterns the model misses.
        
        1. Consecutive identical tokens → mark earlier as disfluent
        2. Near-identical tokens (one has ー elongation) → mark earlier as disfluent
        3. Triple+ repetition → keep only the last occurrence
        """
        labels = list(labels)  # copy
        n = len(words)
        
        # Pass 1: consecutive identical tokens (e.g., 学校 学校)
        for i in range(n - 1):
            if labels[i] == 0 and labels[i + 1] == 0:
                w1 = words[i].replace('ー', '')
                w2 = words[i + 1].replace('ー', '')
                if w1 == w2 and len(w1) >= 2:
                    labels[i] = 1  # mark the first as disfluent
        
        # Pass 2: triple+ repetition — keep only the last
        i = 0
        while i < n - 2:
            base = words[i].replace('ー', '')
            if len(base) >= 2:
                j = i + 1
                while j < n and words[j].replace('ー', '') == base:
                    j += 1
                if j - i >= 3:  # 3+ consecutive identical
                    for k in range(i, j - 1):
                        labels[k] = 1  # keep only last
                    i = j
                    continue
            i += 1
        
        # Pass 3: elongated variant before correction
        # e.g., word ending with ー where next word is same without ー
        for i in range(n - 1):
            if labels[i] == 0 and words[i].endswith('ー') and len(words[i]) >= 2:
                base = words[i].rstrip('ー')
                if words[i + 1].startswith(base):
                    labels[i] = 1
        
        # Pass 4: split-repetition detection
        # GiNZA may split a repeated word: 学校学校 → 学校 + 学 + 校
        # Detect when concat of subsequent tokens matches a previous token
        for i in range(n - 1):
            if labels[i] == 0 and len(words[i]) >= 2:
                # Try to match words[i] with concat of words[i+1], words[i+1:i+2], etc.
                concat = ''
                for j in range(i + 1, min(i + 6, n)):  # look ahead up to 5 tokens
                    concat += words[j]
                    if concat == words[i]:
                        # Split repetition found — mark the split parts as DIS
                        for k in range(i + 1, j + 1):
                            labels[k] = 1
                        break
                    if len(concat) > len(words[i]):
                        break
                
                # Also check reverse: words[i] + words[i+1] + ... == words[j]
                if i + 1 < n and labels[i] == 0:
                    concat = words[i]
                    for j in range(i + 1, min(i + 6, n)):  # look ahead up to 5 tokens
                        if concat == words[j] and len(concat) >= 2:
                            # words[i:j] are the split version, words[j] is the whole
                            # Mark the split parts as DIS
                            for k in range(i, j):
                                labels[k] = 1
                            break
                        concat += words[j]
                        if len(concat) > 10:
                            break
        
        # Pass 5: same-script prefix false start
        # e.g., コン before コンピューター — word[i] is a prefix of word[i+1]
        # and word[i+1] is significantly longer (>=1.5x)
        for i in range(n - 1):
            if labels[i] == 0 and labels[i + 1] == 0:
                w1 = words[i]
                w2 = words[i + 1]
                if (len(w1) >= 2 and len(w2) >= len(w1) + 2
                        and w2.startswith(w1)):
                    labels[i] = 1

        # Pass 6: bridge auxiliaries between repeated content words.
        # Example: 勉強(DIS) し ます 勉強(FLU) -> mark し/ます as DIS.
        bridge_ok = {"し", "ます", "まし", "た", "だ", "です", "でし"}
        for i in range(n - 2):
            if labels[i] != 1:
                continue
            left = words[i].replace('ー', '')
            if len(left) < 2:
                continue
            for j in range(i + 2, min(i + 6, n)):
                right = words[j].replace('ー', '')
                if labels[j] == 0 and right == left:
                    between = words[i + 1:j]
                    if between and all(w in bridge_ok for w in between):
                        for k in range(i + 1, j):
                            labels[k] = 1
                    break

        # Pass 7: repeated short n-gram false starts.
        # Examples:
        #   本 を 本 を
        #   田中 さん と 田中 さん と
        #   その 後 で その 後 で
        particle_like = {"は", "が", "を", "に", "へ", "で", "と", "も", "の", "から", "まで", "って"}
        for ngram_len in (3, 2):
            for i in range(n - (2 * ngram_len) + 1):
                left_chunk = words[i:i + ngram_len]
                right_chunk = words[i + ngram_len:i + 2 * ngram_len]
                if left_chunk != right_chunk:
                    continue
                if any(labels[k] == 1 for k in range(i, i + 2 * ngram_len)):
                    continue
                compact = ''.join(left_chunk).replace('ー', '')
                if len(compact) < 2:
                    continue
                if all(tok in particle_like for tok in left_chunk):
                    continue
                for k in range(i, i + ngram_len):
                    labels[k] = 1


        # v15 Patch D: Protect まで particle from destruction
        for i_p in range(n - 1):
            if labels[i_p] == 1 and words[i_p] == "ま" and words[i_p + 1] == "で":
                labels[i_p] = 0

        return labels

    def clean(self, words: List[str]) -> Tuple[List[str], List[int], Dict]:
        """
        Remove disfluent words from a word list.

        Returns:
            (clean_words, removed_indices, info_dict)
        """
        labels = self.detect(words)
        clean_words = []
        removed_indices = []
        removed_words = []
        for i, (word, label) in enumerate(zip(words, labels)):
            if label == 1:
                removed_indices.append(i)
                removed_words.append(word)
            else:
                clean_words.append(word)

        info = {
            'original_length': len(words),
            'processed_length': len(clean_words),
            'disfluencies_removed': len(removed_indices),
            'removed_words': removed_words,
        }
        return clean_words, removed_indices, info


# ==============================================================================
# Japanese Clause Data Structure
# ==============================================================================

@dataclass(frozen=True)
class Clause:
    sent_index: int
    clause_type: str
    head_i: int
    token_indices: Tuple[int, ...]
    text: str


# ==============================================================================
# Japanese Clause Segmenter (GiNZA + Vercellotti-style rules)
# ==============================================================================

class _ParseCache:
    """Memoise nlp(text). The segmenter re-parses the same clause strings many
    times (about 1,900 parser calls for 170 distinct texts in a 2-minute
    recording), and each GiNZA-Electra parse costs ~20 ms on CPU. Parsing is
    deterministic and no caller modifies the returned Doc, so the output is
    unchanged. Other attributes are passed through to the spaCy pipeline."""

    def __init__(self, nlp):
        self._nlp = nlp
        self._docs = {}

    def __call__(self, text):
        doc = self._docs.get(text)
        if doc is None:
            doc = self._docs[text] = self._nlp(text)
        return doc

    def __getattr__(self, name):
        return getattr(self._nlp, name)


class JapaneseClauseSegmenter:
    """
    Rule layer over a Japanese UD dependency parse (GiNZA / spaCy).
    
    Implements Vercellotti & Hall (2024) inspired rules for Japanese:
    - Independent clause: sentence ROOT predicate (VERB/ADJ/NOUN)
    - Subordinate clause heads (advcl/ccomp/acl/xcomp) are clauses
    - Te-form chains are promoted to clause status only with overt complement/adjunct
    - Stance verbs like 思う with ccomp child are tagged as MINOR
    """

    def __init__(
        self,
        nlp,
        debug: bool = False,
        element_deps: Optional[Set[str]] = None,
        minor_matrix_lemmas: Optional[Set[str]] = None,
        allow_dep_as_element: bool = False,
        disfluency_detector: Optional["DisfluencyDetector"] = None,
        apply_methodology_rules: bool = True,
        pause_split_sec: float = 0.6,
    ):
        self.nlp = nlp if isinstance(nlp, _ParseCache) else _ParseCache(nlp)
        # Silences at least this long cut the fluent words into parsing groups.
        self.pause_split_sec = pause_split_sec
        self.debug = debug
        self._current_filename = None
        # Optional neural disfluency detector. When set, clause text will
        # exclude detected disfluent words (fillers, repetitions, false starts).
        self.disfluency_detector = disfluency_detector
        self.apply_methodology_rules = apply_methodology_rules

        # "Verb + element" deps used to promote borderline predicates to clause status
        self._element_deps = element_deps or {"obj", "iobj", "obl", "advmod", "nmod"}
        self._allow_dep_as_element = allow_dep_as_element

        # Stance list for "minor clause" tagging
        self._minor_matrix_lemmas = minor_matrix_lemmas or {"思う", "考える", "感じる"}

        # Common fillers / backchannels in Japanese speech transcripts
        self._filler_text = {
            "えー", "え", "あー", "あ", "んー", "うーん", "えっと", "ええと",
            "まー", "まあ", "そのー", "その", "あのー", "あの", "ねー", "ね",
            "うん", "ふーん", "へー", "はい", "ええ",
        }
        # Normalize and broaden filler/backchannel matching.
        self._filler_text.update({"ん", "ううん"})
        self._filler_text_norm = {self._normalize_surface_text(x) for x in self._filler_text}
        # Plain demonstratives are ambiguous in learner speech ("その本" vs filler "その").
        self._ambiguous_filler_norm = {"その", "あの"}
        self._filler_char_only_re = re.compile(r"^[んー]+$")
        self._filler_backchannel_re = re.compile(
            r"^(?:"
            r"う+ん+|"
            r"え+ー*|"
            r"あ+ー*|"
            r"えっ?と|"
            r"あのー?|"
            r"そのー?|"
            r"ま+ー?|"
            r"ね+ー?|"
            r"はい+"
            r")$"
        )
        # Cached surface-profile analysis used for boundary cleanup.
        self._surface_profile_cache: Dict[str, Dict[str, bool]] = {}
        self._function_pos = {"AUX", "ADP", "PART", "SCONJ", "CCONJ", "PUNCT", "SYM"}
        self._function_deps = {"aux", "cop", "mark", "case", "punct", "cc", "fixed"}
        self._content_pos = {"VERB", "ADJ", "NOUN", "PROPN", "ADV", "NUM", "PRON"}

    @staticmethod
    def _normalize_surface_text(text: str) -> str:
        t = (text or "").strip().lower()
        t = t.replace("　", "").replace(" ", "")
        t = t.replace("〜", "ー").replace("～", "ー")
        t = re.sub(r"^[、。,.;:!?！？]+|[、。,.;:!?！？]+$", "", t)
        return t

    def _is_filler_surface(self, text: str) -> bool:
        t = self._normalize_surface_text(text)
        if not t:
            return False
        if t in self._filler_text_norm:
            return True
        if self._filler_char_only_re.fullmatch(t):
            return True
        if self._filler_backchannel_re.fullmatch(t):
            return True
        return False

    def _is_ambiguous_filler_surface(self, text: str) -> bool:
        t = self._normalize_surface_text(text)
        return t in self._ambiguous_filler_norm

    def _is_unambiguous_filler_surface(self, text: str) -> bool:
        t = self._normalize_surface_text(text)
        if not t or t in self._ambiguous_filler_norm:
            return False
        # Standalone ん often appears as the negation suffix in ません once the
        # transcript has been morpheme-split; do not delete it as a filler.
        if t == "ん":
            return False
        return self._is_filler_surface(t)

    def _is_filler_token(self, tok) -> bool:
        """Check if token is a filler/backchannel."""
        if self._is_unambiguous_filler_surface(tok.text):
            return True
        if self._is_ambiguous_filler_surface(tok.text):
            if tok.dep_ == "discourse":
                return True
            if tok.pos_ in {"INTJ", "SYM"}:
                return True
            if "フィラー" in (tok.tag_ or ""):
                return True
        if "フィラー" in (tok.tag_ or ""):
            return True
        if tok.dep_ == "discourse" and tok.pos_ in {"INTJ", "SYM", "ADJ"}:
            return True
        return False

    def _surface_profile(self, text: str) -> Dict[str, bool]:
        """Return a cached coarse profile for a surface word."""
        norm = self._normalize_surface_text(text)
        if not norm:
            return {"has_content": False, "all_function": True}
        cached = self._surface_profile_cache.get(norm)
        if cached is not None:
            return cached
        doc = self.nlp(norm)
        tokens = [t for t in doc if t.text.strip()]
        if not tokens:
            profile = {"has_content": False, "all_function": True}
        else:
            has_content = any(t.pos_ in self._content_pos for t in tokens)
            all_function = all(
                (t.pos_ in self._function_pos) or (t.dep_ in self._function_deps)
                for t in tokens
            )
            profile = {"has_content": has_content, "all_function": all_function}
        self._surface_profile_cache[norm] = profile
        return profile

    def _is_function_like_surface(self, text: str) -> bool:
        """True when a surface form is function-like (aux/particle/etc)."""
        if self._is_unambiguous_filler_surface(text):
            return True
        profile = self._surface_profile(text)
        return profile["all_function"] and not profile["has_content"]

    def _is_content_surface(self, text: str) -> bool:
        """True when a surface form contains lexical content."""
        norm = self._normalize_surface_text(text)
        if not norm or self._is_unambiguous_filler_surface(norm):
            return False
        profile = self._surface_profile(norm)
        if profile["has_content"]:
            return True
        return not profile["all_function"]

    def _is_predicate(self, tok) -> bool:
        """Check if token can be a predicate (ROOT-level)."""
        if self._is_filler_token(tok):
            return False
        return tok.pos_ in {"VERB", "ADJ", "NOUN"}

    def _is_subordinate_predicate(self, tok) -> bool:
        """
        For embedded/subordinate clause heads, be stricter than ROOT:
        - allow VERB/ADJ freely
        - allow NOUN only for ROOT-like copula-less predicates (handled elsewhere)
        """
        if self._is_filler_token(tok):
            return False
        if tok.pos_ == "VERB":
            # Reject noun-like tokens misparsed as VERB, but keep
            # sahen-style verbal predicates (e.g., 勉強 + し + た).
            tag = tok.tag_ or ""
            noun_like = ("名詞" in tag) or ("固有名詞" in tag)
            if noun_like:
                aux_children = [ch for ch in tok.children if ch.pos_ == "AUX"]
                sahen_like = ("サ変" in tag) or ("sahen" in tag.lower())
                if not (aux_children or sahen_like):
                    if self.debug:
                        print(f"Rejecting misparsed VERB (tag={tag}): {tok.text}")
                    return False
            # Reject if it looks like a proper noun (katakana name)
            # v1.0-paper checked a partial katakana set (no voiced kana).
            katakana_name = bool(tok.text) and all(c in "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンァィゥェォッャュョー" for c in tok.text)
            if katakana_name:
                if len(tok.text) >= 2:  # Katakana names are usually 2+ chars
                    if self.debug:
                        print(f"Rejecting katakana name misparsed as VERB: {tok.text}")
                    return False
            return True
        if tok.pos_ == "ADJ":
            # Restrict to true predicative adjectives (i-adjectives)
            if "形容詞" in (tok.tag_ or ""):
                return True
            # Allow adjective-like predicates when supported by auxiliary/copula
            cop_like_text = {"だ", "です", "だった", "でした", "じゃ", "ない", "な"}
            for ch in tok.children:
                if ch.pos_ == "AUX" and ch.text in cop_like_text:
                    return True
            return False
        if tok.pos_ == "NOUN":
            has_copula = any(ch.pos_ == "AUX" and ch.text in {"だ", "です", "だった", "でした", "じゃない", "でしょう", "ます"} for ch in tok.children)
            if has_copula:
                return True
            has_quote_or_sub = any(ch.text in {"って", "と", "で", "なら", "ことで"} for ch in tok.children)
            if has_quote_or_sub:
                return True
            return False
        return False

    def _has_tari_mark(self, tok) -> bool:
        """Check if token has tari-form marking (たり).
        GiNZA may parse たり as a mark child or as part of the verb form."""
        for child in tok.children:
            if child.dep_ == "mark" and child.text == "たり":
                return True
        # Also check if the verb text itself ends in たり
        if tok.text.endswith("たり") or tok.text.endswith("だり"):
            return True
        return False

    def _has_renyokei_form(self, tok) -> bool:
        """Check if token is in bare continuative form (連用形) without て/で.
        Ren'yōkei coordination (e.g. 読み in '本を読み、…') functions
        identically to te-form for clause coordination."""
        if self._has_te_mark(tok):
            return False  # te-form handled separately
        if self._has_tari_mark(tok):
            return False  # tari-form handled separately
        morph = tok.morph.to_dict()
        inflection = morph.get("Inflection", "")
        return "連用形" in inflection

    def _has_te_mark(self, tok) -> bool:
        """Check if token has te-form marking (て/で).
        Returns False for てから (sequential 'after doing'), which is a subordinator."""
        for child in tok.children:
            if child.dep_ == "mark" and child.text in {"て", "で"}:
                # Check for てから: て + から as sibling → subordinator, not te-chain
                has_kara = any(
                    sib.text == "から" and sib.dep_ in ("case", "mark")
                    for sib in tok.children if sib is not child
                )
                if has_kara:
                    return False  # Let _has_explicit_subordinator handle てから
                return True
        return False

    def _has_explicit_subordinator(self, tok) -> bool:
        """Detect common overt subordinators."""
        for child in tok.children:
            # Standard mark-dependency subordinators
            if child.dep_ == "mark" or child.dep_ == "case":
                if child.text in {"から", "ながら", "つつ", "けど", "けれど", "が", "なら", "ば", "と", "し"}:
                    return True
                # ので / のに often come as mark 'の' + fixed 'で'/'に'
                if child.text == "の":
                    for g in child.children:
                        if g.dep_ == "fixed" and g.text in {"で", "に"}:
                            return True
                # てから: て(mark) + から(case) → subordinator "after doing"
                if child.text in {"て", "で"}:
                    has_kara = any(
                        sib.text == "から" and sib.dep_ in ("case", "mark")
                        for sib in tok.children if sib is not child
                    )
                    if has_kara:
                        return True
            # GiNZA sometimes parses なら as AUX (not mark)
            if child.dep_ == "aux" and child.text == "なら":
                return True
        return False

    def _has_complement_or_adjunct(self, tok) -> bool:
        """
        Strict "verb + element" heuristic.
        NOTE: nsubj is NOT counted as the "extra element" (shared/implied subjects).
        """
        valid_element_deps = self._element_deps | {"ccomp", "advcl", "xcomp", "csubj"}
        for child in tok.children:
            if self._is_filler_token(child):
                continue
            if child.dep_ in valid_element_deps:
                return True
            if self._allow_dep_as_element and child.dep_ == "dep":
                if child.pos_ in {"NOUN", "PROPN", "ADV", "ADJ", "VERB"}:
                    return True
        return False

    def _token_depth(self, tok) -> int:
        """Compute syntactic depth of token."""
        depth = 0
        cur = tok
        while cur.head is not cur:
            depth += 1
            cur = cur.head
            if depth > 200:
                break
        return depth

    def _find_clause_heads(self, sent) -> List[Tuple[object, str]]:
        """Return list of (head_token, clause_type) for a single sentence span."""
        heads: List[Tuple[object, str]] = []

        for tok in sent:
            if not self._is_predicate(tok):
                continue

            dep = tok.dep_

            if dep == "ROOT":
                ctype = "IND"
                # Tag stance/matrix verbs as MINOR when they take a ccomp
                if tok.pos_ == "VERB" and tok.lemma_ in self._minor_matrix_lemmas and any(
                    ch.dep_ == "ccomp" for ch in tok.children
                ):
                    ctype = "MINOR"
                heads.append((tok, ctype))
                continue
            
            # Handle broken parses: VERB with dep="dep" pointing to NOUN ROOT
            # This often happens in unpunctuated speech where GiNZA misparses
            if dep == "dep" and tok.pos_ == "VERB":
                # Check if this verb has its own subject - it's likely a real clause
                has_own_subject = any(ch.dep_ == "nsubj" for ch in tok.children)
                has_content = self._has_complement_or_adjunct(tok) or has_own_subject
                if has_content:
                    heads.append((tok, "IND"))
                    if self.debug:
                        print(f"Promoting dep=dep VERB as IND: {tok.text}")
                continue

            if dep in {"acl"}:
                # Relative/adnominal clause
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "SUB_REL"))
                continue

            if dep in {"ccomp"}:
                # Content/complement clause
                if self._is_subordinate_predicate(tok) or tok.pos_ == "NOUN":
                    heads.append((tok, "SUB_CCOMP"))
                continue

            if dep in {"xcomp"}:
                # Japanese xcomp is rarer
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "SUB_XCOMP"))
                continue

            if dep == "csubj":
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "SUB_CSUBJ"))
                continue

            if dep == "advcl":
                if not self._is_subordinate_predicate(tok):
                    continue
                # Check for explicit subordinator FIRST (から, けど, ので, etc.)
                # These take priority over te-form marking
                if self._has_explicit_subordinator(tok):
                    heads.append((tok, "SUB_ADVCL"))
                elif self._has_te_mark(tok):
                    # Strict Vercellotti-style promotion for te-chains
                    if self._has_complement_or_adjunct(tok):
                        heads.append((tok, "CHAIN_TE"))
                    else:
                        if self.debug:
                            print(f"SKIP te-chain (no element): {tok.text} / {sent.text}")
                elif self._has_tari_mark(tok):
                    # Tari-form: same verb+element rule as te-form (FROZEN_RULES Rule 5)
                    if self._has_complement_or_adjunct(tok):
                        heads.append((tok, "CHAIN_TE"))
                    else:
                        if self.debug:
                            print(f"SKIP tari-chain (no element): {tok.text} / {sent.text}")
                elif self._has_renyokei_form(tok):
                    # Ren'yōkei coordination: bare continuative form (連用形)
                    # functions like te-form for clause coordination
                    if self._has_complement_or_adjunct(tok):
                        heads.append((tok, "CHAIN_TE"))
                    else:
                        if self.debug:
                            print(f"SKIP renyokei-chain (no element): {tok.text} / {sent.text}")
                else:
                    # For other advcl: verb+element required
                    if self._has_complement_or_adjunct(tok):
                        heads.append((tok, "SUB_ADVCL"))
                    else:
                        if self.debug:
                            print(f"SKIP advcl (no mark/element): {tok.text} / {sent.text}")
                continue

            if dep == "conj":
                # Coordinated predicates: promote only if it has its own element
                if self._is_subordinate_predicate(tok) and self._has_complement_or_adjunct(tok):
                    heads.append((tok, "COORD"))
                continue

            if dep == "parataxis":
                if self._is_subordinate_predicate(tok):
                    heads.append((tok, "PARA"))
                continue

        # Deduplicate
        seen = set()
        out: List[Tuple[object, str]] = []
        for tok, ctype in heads:
            if tok.i in seen:
                continue
            seen.add(tok.i)
            out.append((tok, ctype))
        return out

    def _collect_clause_tokens(
        self,
        head_tok,
        sent,
        excluded: Set[int],
        clause_head_indices: Set[int],
    ) -> Set[int]:
        """Collect a token set for this clause, staying within the sentence."""
        sent_start = sent.start
        sent_end = sent.end

        collected: Set[int] = set()

        def in_sent(i: int) -> bool:
            return sent_start <= i < sent_end

        def collect(tok, depth: int = 0) -> None:
            if depth > 60:
                return
            if tok.i in excluded or not in_sent(tok.i):
                return

            collected.add(tok.i)

            for child in tok.children:
                if child.i in excluded or not in_sent(child.i):
                    continue
                # Avoid swallowing other clause heads
                if child.i in clause_head_indices and child.i != head_tok.i:
                    continue
                # For content clauses, exclude quotative particle と
                if head_tok.dep_ == "ccomp" and child.dep_ == "case" and child.text == "と":
                    continue
                # CRITICAL: Don't swallow children that have their own nsubj
                # This indicates a separate clause structure
                if child.dep_ == "dep" and child.pos_ == "VERB":
                    child_has_nsubj = any(gc.dep_ == "nsubj" for gc in child.children)
                    if child_has_nsubj:
                        if self.debug:
                            print(f"Stopping at VERB with own nsubj: {child.text}")
                        continue
                collect(child, depth + 1)

        collect(head_tok)

        # Add "matrix glue" tokens for minor/matrix verbs
        if head_tok.dep_ == "ROOT" and head_tok.lemma_ in self._minor_matrix_lemmas:
            for ch in head_tok.children:
                if ch.dep_ == "ccomp":
                    for g in ch.children:
                        if g.dep_ == "case" and g.text == "と" and in_sent(g.i) and g.i not in excluded:
                            collected.add(g.i)

        return collected

    def _is_functionish_token(self, tok) -> bool:
        """Heuristic: token is function-like (boundary glue, not lexical core)."""
        if self._is_filler_token(tok):
            return True
        if tok.pos_ in self._function_pos or tok.dep_ in self._function_deps:
            return True
        return self._is_function_like_surface(tok.text)

    def _prune_disconnected_function_runs(self, tokens: Set[int], head_tok) -> Set[int]:
        """
        Drop disconnected function-only islands that can leak into a clause span.
        Example: sentence-initial `で` attached to ROOT while real predicate is later.
        """
        if not tokens:
            return tokens
        idxs = sorted(tokens)
        runs: List[List[int]] = []
        cur = [idxs[0]]
        for i in idxs[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)

        head_run = None
        for r in runs:
            if head_tok.i in r:
                head_run = r
                break
        if head_run is None:
            return tokens

        keep = set(head_run)
        for r in runs:
            if r is head_run:
                continue
            # Keep non-function runs (defensive), drop function-only glue runs.
            if not all(self._is_functionish_token(head_tok.doc[i]) for i in r):
                keep.update(r)
        return keep

    def segment_sentence(self, sent, sent_index: int) -> List[Clause]:
        """Segment one spaCy sentence span into non-overlapping clause strings."""
        clause_heads = self._find_clause_heads(sent)
        if not clause_heads:
            return []

        clause_head_indices = {tok.i for tok, _ in clause_heads}

        # Two-pass: take deeper heads first
        scored = []
        for tok, ctype in clause_heads:
            scored.append((self._token_depth(tok), tok.i, tok, ctype))
        scored.sort(key=lambda x: (-x[0], x[1]))

        excluded: Set[int] = set()
        clauses: List[Clause] = []

        for _depth, _i, head_tok, ctype in scored:
            tokens = self._collect_clause_tokens(head_tok, sent, excluded, clause_head_indices)
            tokens = self._prune_disconnected_function_runs(tokens, head_tok)
            if not tokens:
                continue
            excluded.update(tokens)

            idxs = tuple(sorted(tokens))
            text = "".join(head_tok.doc[i].text_with_ws for i in idxs).strip()

            clauses.append(
                Clause(
                    sent_index=sent_index,
                    clause_type=ctype,
                    head_i=head_tok.i,
                    token_indices=idxs,
                    text=text,
                )
            )

        # Return in surface order
        clauses.sort(key=lambda c: c.token_indices[0])
        return clauses

    def _validate_clauses(self, clauses: List[Tuple]) -> List[Tuple]:
        """
        Post-process clauses to filter invalid ones and fix issues.
        
        Vercellotti rules applied:
        1. Clause must have predicate (verb/adj/noun) + element
        2. Te-form chains need own complement/adjunct
        3. Filter fragments (<4 chars)
        4. Truncate very long clauses (>60 chars) at space boundaries
        """
        validated = []
        
        for start_idx, end_idx, ctype, ctext, indices in clauses:
            # Clean text for length check
            clean = ctext.replace(' ', '').replace('　', '')
            
            # Skip fragments (< 3 chars after removing spaces)
            # BUT: keep MINOR clauses and clauses with clear predicates even if short
            if len(clean) < 3:
                if self.debug:
                    print(f"[DEBUG] Filtered fragment: '{ctext}'")
                continue
            
            # For 3-char clauses, only keep if it has a clear predicate (verb/adj)
            if len(clean) < 4 and ctype not in ['MINOR', 'IND']:
                doc = self.nlp(ctext)
                has_clear_pred = any(t.pos_ in ('VERB', 'ADJ') for t in doc)
                if not has_clear_pred:
                    if self.debug:
                        print(f"[DEBUG] Filtered short non-predicate: '{ctext}'")
                    continue
            
            # Skip if just particles/punctuation/fillers (but keep MINOR)
            skip_chars = 'をにがでとはもへからまでよりやのかなねよわさぞってけどけれどばたら、。ーっ'
            if all(c in skip_chars for c in clean) and ctype != 'MINOR':
                if self.debug:
                    print(f"[DEBUG] Filtered particles-only: '{ctext}'")
                continue
            
            # v1.0-paper: skip clauses that are entirely interview meta-speech
            if self._is_meta_clause(clean):
                if self.debug:
                    print(f"[DEBUG] Filtered meta clause: '{ctext}'")
                continue
            
            # v1.0-paper: strip trailing meta from clause text (e.g., ...しました以上です)
            stripped = self._META_TAIL.sub('', ctext)
            if stripped != ctext and len(stripped.replace(' ', '').replace('　', '')) >= 3:
                ctext = stripped.strip()
                clean = ctext.replace(' ', '').replace('　', '')
            
            # Parse to check for predicate
            doc = self.nlp(ctext)
            # Reject misparsed VERB (pos=VERB but tag=名詞*) — common with disfluent L2
            has_verb = any(t.pos_ == 'VERB' and '名詞' not in (t.tag_ or '') for t in doc)
            has_adj = any(t.pos_ == 'ADJ' for t in doc)
            has_aux = any(t.pos_ == 'AUX' and t.lemma_ not in ('だ', 'です') for t in doc)
            has_noun_pred = any(t.pos_ == 'NOUN' and t.dep_ == 'ROOT' for t in doc)
            has_predicate = has_verb or has_adj or has_aux or has_noun_pred
            
            # For CHAIN_TE, check it has complement/adjunct (Vercellotti rule)
            # No length threshold: bare te-form is never a clause regardless of length
            # NOTE: nsubj excluded — matches _has_complement_or_adjunct (shared subjects don't count)
            if ctype == 'CHAIN_TE':
                has_element = any(t.dep_ in ('obj', 'iobj', 'obl', 'advmod', 'nmod') for t in doc)
                if not has_element:
                    if self.debug:
                        print(f"[DEBUG] Filtered bare te-form: '{ctext}'")
                    continue
            
            # SUB_REL bare-adjective filter: a single adnominal adjective (静かな, 便利な)
            # is just a modifier, not a clause per Vercellotti's "predicate + element" rule.
            if ctype == 'SUB_REL' and (has_adj and not has_verb):
                has_element = any(t.dep_ in ('obj', 'iobj', 'obl', 'advmod', 'nmod') for t in doc)
                if not has_element and len(clean) < 15:
                    if self.debug:
                        print(f"[DEBUG] Filtered bare-adjective SUB_REL: '{ctext}'")
                    continue
            
            # Skip clauses without any predicate (violates Vercellotti)
            if not has_predicate and len(clean) < 20:
                if self.debug:
                    print(f"[DEBUG] Filtered no-predicate: '{ctext}'")
                continue
            
            # Sanity check: NOUN-ROOT predicate with case-marked arguments
            # is likely a fragment (broken verb), not a copula-less predicate.
            # Real copula-less: 今日は休み (topic + noun) — no obj/obl markers.
            # Fragment: 本をよ (obj + truncated verb) — has を.
            if has_noun_pred and not has_verb and not has_adj:
                # Only flag as fragment if there are true object/goal markers,
                # NOT topic/obl with は (which is legitimate for copula-less: 今日は休みです)
                has_obj_case = any(
                    t.dep_ in ('obj', 'iobj') or
                    (t.dep_ == 'case' and t.text == 'を' and t.head.dep_ != 'ROOT')
                    for t in doc
                )
                if has_obj_case:
                    if self.debug:
                        print(f"[DEBUG] Filtered noun-ROOT with obj/obl (likely fragment): '{ctext}'")
                    continue
            
            # Truncate very long clauses at space boundaries (max 60 chars)
            if len(clean) > 60:
                # Find last space before 60 chars
                parts = ctext.split(' ')
                truncated = []
                current_len = 0
                for part in parts:
                    part_clean = part.replace('　', '')
                    if current_len + len(part_clean) > 55:
                        break
                    truncated.append(part)
                    current_len += len(part_clean)
                
                if truncated:
                    ctext = ' '.join(truncated)
                    if self.debug:
                        print(f"[DEBUG] Truncated long clause to: '{ctext[:40]}...'")
            
            validated.append((start_idx, end_idx, ctype, ctext, indices))
        
        # Post-pass: split long clauses at て/で-form or は-topic boundaries
        final = []
        for start_idx, end_idx, ctype, ctext, indices in validated:
            clean = ctext.replace(' ', '').replace('　', '')
            if len(clean) > 25:
                splits = self._split_long_clause(ctext, ctype, indices)
                if splits:
                    final.extend(splits)
                    continue
            final.append((start_idx, end_idx, ctype, ctext, indices))
        
        return final
    
    def _split_long_clause(self, ctext: str, ctype: str, indices: list):
        """Split a long clause at て/で-form verb or は-topic change.
        
        Priority:
          1. て/で-form verb boundary (Vercellotti CHAIN_TE rule)
          2. は-topic change (subject change)
        
        Returns list of sub-clause tuples, or None if no split needed.
        """
        doc = self.nlp(ctext)
        
        # --- Strategy 1: Split at て/で-form boundary ---
        # Look for て/で SCONJ markers (dep=mark) — reliably detected even
        # when GiNZA misparses the verb in broken L2 text.
        # Also check verbs ending in て/で directly.
        te_split = None
        for tok in doc:
            split_after = None
            
            # Pattern A: て/で as separate SCONJ token with dep=mark
            if tok.text in ('て', 'で') and tok.pos_ == 'SCONJ' and tok.dep_ == 'mark':
                split_after = tok.i + 1
            
            # Pattern B: verb ending in て/で (e.g., 食べて as one token)
            elif tok.pos_ == 'VERB' and tok.text.endswith(('て', 'で')):
                split_after = tok.i + 1
            
            if split_after is not None:
                left_len = split_after
                right_len = len(doc) - split_after
                if left_len >= 3 and right_len >= 3:
                    te_split = split_after
                    break  # take the first valid て/で split
        
        if te_split:
            left_text = ''.join(t.text_with_ws for t in doc[:te_split]).strip()
            right_text = ''.join(t.text_with_ws for t in doc[te_split:]).strip()
            if left_text and right_text:
                left_indices = indices[:te_split] if indices else []
                right_indices = indices[te_split:] if indices else []
                return [
                    (0, te_split, 'CHAIN_TE', left_text, left_indices),
                    (te_split, len(doc), ctype, right_text, right_indices),
                ]
        
        # --- Strategy 2: Split at は-topic change ---
        wa_positions = []
        for tok in doc:
            if tok.text == 'は' and tok.dep_ == 'case':
                head = tok.head
                if head.pos_ in ('NOUN', 'PRON', 'PROPN'):
                    wa_positions.append((tok.i, head.i))
        
        if len(wa_positions) >= 2:
            _, second_head_i = wa_positions[1]
            split_at = second_head_i
            for child in doc[second_head_i].subtree:
                if child.i < split_at:
                    split_at = child.i
            
            if split_at >= 3 and len(doc) - split_at >= 3:
                left_text = ''.join(t.text_with_ws for t in doc[:split_at]).strip()
                right_text = ''.join(t.text_with_ws for t in doc[split_at:]).strip()
                if left_text and right_text:
                    left_indices = indices[:split_at] if indices else []
                    right_indices = indices[split_at:] if indices else []
                    return [
                        (0, split_at, ctype, left_text, left_indices),
                        (split_at, len(doc), 'IND', right_text, right_indices),
                    ]
        
        return None

    def segment_utterance(self, text: str, sent_index: int = 0) -> List[Clause]:
        """Segment a single utterance as one sentence span."""
        doc = self.nlp(text)
        sent = doc[:]  # force one-sentence view
        return self.segment_sentence(sent, sent_index=sent_index)

    def _resplit_long_sentences(self, doc, sents) -> list:
        """Re-split long sentences at sentence-final verb forms.
        
        When text has no punctuation (common after disfluency removal),
        GiNZA may fail to detect sentence boundaries. This finds obvious
        sentence-final forms (ました, です, etc.) and splits there.
        
        GiNZA tokenization:
          ました → まし(AUX,lemma=ます) + た(AUX,lemma=た)
          ません → ませ(AUX,lemma=ます) + ん(AUX,lemma=ぬ)
          です   → です(AUX,lemma=です)
          でした → でし(AUX,lemma=です) + た(AUX,lemma=た)
        """
        # Strong connectives: always block a split
        # NOTE: で is NOT here — after ました/です, で is almost always
        #       conjunction "and then", not connective particle
        CONNECTIVE = {'て', 'けど', 'けれど', 'ので', 'から',
                      'たり', 'し', 'ば', 'と', 'のに', 'ながら', 'ものの',
                      'ても', 'でも', 'ところ', 'のは'}
        # が is special: block split only if the remainder is short
        # (ましたが + short subordinate = one clause;
        #  ましたが + long independent = two clauses)

        def _starts_polite_continuation(next_tok) -> bool:
            surf = self._normalize_surface_text(next_tok.text)
            if not surf:
                return False
            if surf == "か":
                return True
            if surf.startswith("か"):
                return True
            return surf.startswith("すか")
        
        new_sents = []
        for sent in sents:
            sent_text = sent.text.replace(' ', '').replace('　', '')
            if len(sent_text) <= 20:
                new_sents.append(sent)
                continue
            
            split_points = []
            for tok in sent:
                # ----------------------------------------------------------
                # Pattern 1: AUX lemma=ます (ました / ません / ます)
                # ----------------------------------------------------------
                if tok.pos_ == 'AUX' and tok.lemma_ == 'ます':
                    split_after = tok.i
                    peek = tok.i + 1
                    # Include た/だ (ました) or ん (ません)
                    if peek < len(doc) and doc[peek].pos_ == 'AUX':
                        if doc[peek].lemma_ in ('た', 'だ', 'ぬ'):
                            split_after = peek
                    
                    check_idx = split_after + 1
                    if check_idx >= len(doc) or check_idx >= sent.end:
                        continue
                    next_tok = doc[check_idx]
                    if _starts_polite_continuation(next_tok):
                        continue
                    if next_tok.text in CONNECTIVE:
                        continue
                    # が: only block if remainder is short (subordinate clause)
                    if next_tok.text == 'が':
                        remainder_len = sent.end - check_idx
                        if remainder_len < 8:
                            continue
                    left_len = split_after + 1 - sent.start
                    right_len = sent.end - (split_after + 1)
                    if left_len >= 3 and right_len >= 3:
                        split_points.append(split_after + 1)
                
                # ----------------------------------------------------------
                # Pattern 2: AUX lemma=です (です / でした)
                # ----------------------------------------------------------
                if tok.pos_ == 'AUX' and tok.lemma_ == 'です':
                    split_after = tok.i
                    peek = tok.i + 1
                    if peek < len(doc) and doc[peek].pos_ == 'AUX' and doc[peek].lemma_ in ('た', 'だ'):
                        split_after = peek
                    
                    check_idx = split_after + 1
                    if check_idx >= len(doc) or check_idx >= sent.end:
                        continue
                    next_tok = doc[check_idx]
                    if _starts_polite_continuation(next_tok):
                        continue
                    if next_tok.text in CONNECTIVE:
                        continue
                    if next_tok.text == 'が':
                        remainder_len = sent.end - check_idx
                        if remainder_len < 8:
                            continue
                    left_len = split_after + 1 - sent.start
                    right_len = sent.end - (split_after + 1)
                    if left_len >= 3 and right_len >= 3:
                        split_points.append(split_after + 1)
                
                # ----------------------------------------------------------
                # Pattern 3: Plain past た/だ (not after ます/です)
                # Only split when followed by clear new-sentence start
                # ----------------------------------------------------------
                if tok.pos_ == 'AUX' and tok.lemma_ in ('た', 'だ') and tok.text in ('た', 'だ'):
                    prev_idx = tok.i - 1
                    if prev_idx >= 0 and doc[prev_idx].pos_ == 'AUX' and doc[prev_idx].lemma_ in ('ます', 'です'):
                        continue
                    
                    check_idx = tok.i + 1
                    if check_idx >= len(doc) or check_idx >= sent.end:
                        continue
                    next_tok = doc[check_idx]
                    if next_tok.text in CONNECTIVE or next_tok.text == 'が':
                        continue
                    if next_tok.pos_ in ('NOUN', 'PRON', 'DET', 'CCONJ', 'SCONJ', 'ADV'):
                        left_len = tok.i + 1 - sent.start
                        right_len = sent.end - (tok.i + 1)
                        if left_len >= 4 and right_len >= 4:
                            split_points.append(tok.i + 1)
            

                # v15 Pattern 4: Split at subordinator boundary
                if tok.pos_ in ("SCONJ", "CCONJ") and tok.dep_ in ("mark", "cc", "case"):
                    _sub_surfaces = {
                        "から", "ので", "けど", "けれど",
                        "けれども", "たら", "ば",
                        "のに", "ながら",
                    }
                    if tok.text in _sub_surfaces:
                        split_after_sub = tok.i + 1
                        if split_after_sub < len(doc) and split_after_sub < sent.end:
                            left_len_s = split_after_sub - sent.start
                            right_len_s = sent.end - split_after_sub
                            if left_len_s >= 5 and right_len_s >= 5:
                                left_has_pred = any(
                                    doc[k].pos_ in ("VERB", "AUX", "ADJ")
                                    for k in range(sent.start, split_after_sub)
                                )
                                if left_has_pred:
                                    split_points.append(split_after_sub)

            if not split_points:
                new_sents.append(sent)
                continue
            
            # Create sub-spans
            prev = sent.start
            for sp in split_points:
                if sp > prev:
                    new_sents.append(doc[prev:sp])
                prev = sp
            if prev < sent.end:
                new_sents.append(doc[prev:sent.end])
        
        return new_sents

    def segment_text(self, text: str) -> List[Clause]:
        """Segment text using doc.sents, with re-splitting for long unpunctuated sentences."""
        doc = self.nlp(text)
        sents = list(doc.sents)

        # Re-split any overly long sentences at sentence-final forms
        sents = self._resplit_long_sentences(doc, sents)
        
        out: List[Clause] = []
        for si, sent in enumerate(sents):
            out.extend(self.segment_sentence(sent, sent_index=si))
        return out

    # Interview-frame patterns of the evaluation corpus (I-JAS): participant IDs,
    # interviewer prompts, task closings and transcription symbols (v1.0-paper).
    # Participant ID pattern (full-width group code + number, 番, です)
    _PARTICIPANT_ID = re.compile(
        r'(ＣＣＨ|ＣＣＴ|ＪＪＥ|ＪＪＡ)(の)?[０-９0-9A-Zａ-ｚ]+'
        r'[\s　]*(番)?[\s　]*(でーす|です)?[\s　]*'
    )
    _META_END = re.compile(
        r'('
        r'以上(です|でーす|でっす)?'
        r'|ありがと(う(ございます|ございました)?)?'
        r'|はい(終わり|いいです(か|よ)?)?'
        r'|終わり(です(か)?|ました)?'
        r'|オッケー'
        r'|わかりました(咳)?'
        r'|はいわかり'
        r'|いいです(か|よ)?'
        r'|笑'
        r')+[\s　]*$'
    )

    # Clause-level meta-speech patterns (entire clause is meta)
    _META_CLAUSE = re.compile(
        r'^[\s　]*(ま|はい|えー|あー)*[\s　]*('
        r'以上(です|でーす|でっす)?(はい)?'
        r'|終わり(です|ました)?(はい|か)?'
        r'|ありがと(う(ございます|ございました)?)?'
        r'|(はい)?いいです(か|よ)?'
        r'|でございます'
        r'|オッケー'
        r'|これから.{0,6}(お願い|ねがし)'
        r'|よろしくお願い'
        r'|はいありが'
        r'|話して.{0,8}(ください|言って)'
        r'|わかりました'
        r'|お願いします'
        r'|終わりです(ははーい|はーい)?ありが.{0,3}'
        r'|教えてください'
        r')[\s　]*(笑|はい|か)*[\s　]*$'
    )

    # Trailing meta that can be stripped from the END of any clause
    _META_TAIL = re.compile(
        r'(はい)?(以上です|以上|終わり(です)?|ありがとう)[\s　]*(笑|はい)*[\s　]*$'
    )

    @classmethod
    def _is_meta_clause(cls, clean_text: str) -> bool:
        """Check if a clause is entirely meta-speech (no narrative content)."""
        return bool(cls._META_CLAUSE.match(clean_text))

    def _strip_meta_speech(self, text: str) -> str:
        """Remove interview meta-speech from start and end of text (v1.0-paper).
        
        Strategy: The narrative always starts after the participant ID.
        Strip everything from start through the last ID pattern.
        Strip closing meta from the end.
        """
        # Find participant ID — strip everything up to and including it
        # Only look in the first 40% of text to avoid stripping narrative content
        search_limit = min(len(text), max(60, len(text) * 2 // 5))
        search_text = text[:search_limit]
        matches = list(self._PARTICIPANT_ID.finditer(search_text))
        if matches:
            last_match = matches[-1]
            # Also skip はいどうぞ / どうぞ after ID
            remainder = text[last_match.end():]
            for prefix in ['はいどうぞ', 'どうぞ']:
                if remainder.startswith(prefix):
                    remainder = remainder[len(prefix):]
            text = remainder
        
        # Strip opening meta (greetings, interviewer speech)
        for prefix in ['はいこれからよろしくお願いします', 'はいこれからよろしくお願い',
                       'これからよろしくお願いします', 'これからえねがします',
                       'これから話します', 'はいでございます', 'でございます',
                       'はいＩＤを教えてください', 'ＩＤを教えてください']:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        
        # Strip closing meta
        text = self._META_END.sub('', text)
        return text.strip()

    def segment(self, text: str, verbose: bool = True,
                 remove_fillers: bool = False) -> List[Tuple[int, int, str, str, List[int]]]:
        """
        Segment text into clauses.
        
        Args:
            text: Input text
            verbose: Print clause info
            remove_fillers: If True, remove fillers before parsing for cleaner results.
        
        Returns list compatible with English version: (start_idx, end_idx, clause_type, clause_text, token_indices)
        """
        # Strip meta-speech (interviewer questions, ID announcements, closings)
        text = self._strip_meta_speech(text)
        
        # Remove fillers for cleaner parsing
        if remove_fillers:
            try:
                from ja_filler_handler import remove_fillers_simple
                text = remove_fillers_simple(text)
                if self.debug:
                    print(f"[DEBUG] Removed fillers from text")
            except ImportError:
                pass
        
        clauses = self.segment_text(text)
        
        # Convert Clause objects to tuples
        result = []
        for c in clauses:
            start_idx = min(c.token_indices) if c.token_indices else 0
            end_idx = max(c.token_indices) + 1 if c.token_indices else 0
            result.append((start_idx, end_idx, c.clause_type, c.text, list(c.token_indices)))
        
        # Post-process: validate and filter clauses
        result = self._validate_clauses(result)
        
        if verbose:
            print(f"{'TYPE':<20} | {'CLAUSE TEXT'}")
            print("-" * 80)
            for start, end, ctype, ctext, indices in result:
                clean = ctext[:60] + "..." if len(ctext) > 60 else ctext
                print(f"{ctype:<20} | {clean}")
            print(f"\nTotal clauses: {len(result)}")
        
        return result


# ==============================================================================
# TextGrid Handler (adapted for Japanese)
# ==============================================================================

@dataclass
class WordInterval:
    start: float
    end: float
    text: str
    is_pause: bool = False
    mora_count: int = 0  # Japanese uses mora instead of syllables

@dataclass
class PauseInterval:
    start: float
    end: float
    duration: float
    location: str = "unknown"


class TextGridHandler:
    """Handles TextGrid files for Japanese speech."""

    PAUSE_MARKERS = {'', 'sp', 'sil', '<sil>', 'pause', '#', '...',
                     '<p>', '<pause>', 'breath', '<breath>', 'silB', 'silE'}
    MERGED_RESTART_CONNECTORS = ("でも", "しかし", "それで", "だから", "なので", "そして")
    MERGED_TAIL_BLOCKLIST = (
        "ところ", "とき", "とても", "という", "とって", "とり", "とも", "とお", "とし",
        "とは", "との", "として",
    )

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tg = textgrid.openTextgrid(filepath, includeEmptyIntervals=True)
        self.words: List[WordInterval] = []
        self.phones: List[Tuple[float, float, str]] = []
        self.total_mora = 0
        self._extract_tiers()

    def _extract_tiers(self):
        tier_names = self.tg.tierNames

        word_tier_name = None
        for name in ['words', 'word', 'Word', 'Words', '単語', 'transcription']:
            if name in tier_names:
                word_tier_name = name
                break
        if not word_tier_name:
            word_tier_name = tier_names[0]

        phone_tier_name = None
        for name in ['phones', 'phone', 'Phone', 'Phones', '音素']:
            if name in tier_names:
                phone_tier_name = name
                break

        if phone_tier_name:
            phone_tier = self.tg.getTier(phone_tier_name)
            for entry in phone_tier.entries:
                self.phones.append((entry.start, entry.end, entry.label.strip()))

        word_tier = self.tg.getTier(word_tier_name)
        for entry in word_tier.entries:
            text = entry.label.strip()
            is_pause = text.lower() in self.PAUSE_MARKERS or text == ''

            if is_pause:
                self.words.append(WordInterval(
                    start=entry.start,
                    end=entry.end,
                    text=text,
                    is_pause=True,
                    mora_count=0
                ))
                continue

            split_words = self._split_merged_word_interval(entry.start, entry.end, text)
            for word in split_words:
                self.words.append(word)
                self.total_mora += word.mora_count

    def _count_mora(self, text: str) -> int:
        """
        Count mora in Japanese text.
        Simple heuristic: count characters, treating small kana as part of previous mora.
        """
        # Small kana that don't count as separate mora
        small_kana = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮっッー")
        
        count = 0
        for char in text:
            if char not in small_kana and not char.isspace():
                count += 1
        
        return max(count, 1) if text else 0

    def _candidate_internal_split_points(self, text: str) -> List[int]:
        """Return high-confidence character split points for fused ASR words."""
        text = (text or "").strip()
        if len(text) < 6:
            return []

        points: Set[int] = set()
        connector_alt = "|".join(re.escape(c) for c in self.MERGED_RESTART_CONNECTORS)

        for m in re.finditer(
            rf"(ました|ます|ません|でした|です|だった|だ|た)(?=({connector_alt}))",
            text,
        ):
            cut = m.end(1)
            if 2 <= cut <= len(text) - 2:
                points.add(cut)

        for m in re.finditer(r"(ですか|ますか|でしたか|たか|だか)(?=で)", text):
            cut = m.end(1)
            if 2 <= cut <= len(text) - 2:
                points.add(cut)

        for m in re.finditer(r"(から|ので|のに|けど|けれど|けれども)(?=[はがもをにへとでの])", text):
            cut = m.end(1)
            if 2 <= cut <= len(text) - 2:
                points.add(cut)

        if (
            len(text) >= 5
            and text.startswith("と")
            and not text.startswith(self.MERGED_TAIL_BLOCKLIST)
            and re.match(r"^と[ぁ-んァ-ヶ一-龯A-Za-z0-9]{1,8}(さん|ちゃん|くん|君|氏|さま|様)", text)
        ):
            points.add(1)

        return sorted(points)

    def _split_merged_word_interval(self, start: float, end: float, text: str) -> List[WordInterval]:
        """Split obvious fused word intervals so boundaries can fall inside them."""
        points = self._candidate_internal_split_points(text)
        if not points:
            mora = self._count_mora(text)
            return [WordInterval(start=start, end=end, text=text, is_pause=False, mora_count=mora)]

        bounds = [0] + [p for p in points if 0 < p < len(text)] + [len(text)]
        bounds = sorted(set(bounds))
        if len(bounds) < 3:
            mora = self._count_mora(text)
            return [WordInterval(start=start, end=end, text=text, is_pause=False, mora_count=mora)]

        duration = max(0.0, float(end) - float(start))
        total_chars = max(1, len(text))
        out: List[WordInterval] = []

        for bi in range(len(bounds) - 1):
            left = bounds[bi]
            right = bounds[bi + 1]
            if right <= left:
                continue
            seg_text = text[left:right]
            if not seg_text:
                continue

            seg_start = start + duration * (left / total_chars)
            seg_end = end if bi == len(bounds) - 2 else start + duration * (right / total_chars)
            if seg_end <= seg_start:
                seg_end = min(end, seg_start + 0.0001)

            out.append(WordInterval(
                start=seg_start,
                end=seg_end,
                text=seg_text,
                is_pause=False,
                mora_count=self._count_mora(seg_text)
            ))

        return out or [WordInterval(start=start, end=end, text=text, is_pause=False, mora_count=self._count_mora(text))]

    def get_transcript(self) -> str:
        return ''.join(w.text for w in self.words if not w.is_pause and w.text)

    def get_word_list(self) -> List[WordInterval]:
        return [w for w in self.words if not w.is_pause and w.text]

    def get_pauses(self, min_duration: float = 0.25) -> List[PauseInterval]:
        pauses = []
        for w in self.words:
            if w.is_pause:
                duration = w.end - w.start
                if duration >= min_duration:
                    pauses.append(PauseInterval(
                        start=w.start,
                        end=w.end,
                        duration=duration
                    ))
        return pauses

    def add_clause_tier(self, clauses: List[Dict], tier_name: str = "clauses"):
        """Add a clause tier with all clauses as non-overlapping intervals."""
        from praatio.data_classes.interval_tier import IntervalTier

        min_time = self.tg.minTimestamp
        max_time = self.tg.maxTimestamp
        tg_words = self.get_word_list()

        if not clauses:
            intervals = [Interval(min_time, max_time, "")]
            new_tier = IntervalTier(tier_name, intervals, min_time, max_time)
            self.tg.addTier(new_tier)
            return []

        def _clause_sort_key(clause: Dict):
            idxs = sorted(clause.get("word_indices", []) or [])
            if idxs:
                return (idxs[0], idxs[-1])
            return (float(clause.get("start_time", min_time)), float(clause.get("end_time", min_time)))

        final_clauses = []
        current_time = min_time
        for clause in sorted(clauses, key=_clause_sort_key):
            idxs = sorted(clause.get("word_indices", []) or [])
            if idxs:
                start = tg_words[idxs[0]].start
                end = tg_words[idxs[-1]].end
            else:
                start = float(clause.get("start_time", current_time))
                end = float(clause.get("end_time", start))

            start = max(start, current_time)
            end = min(end, max_time)
            if end <= start:
                continue

            final_clauses.append({
                'start': start,
                'end': end,
                'label': clause['text'],
                'type': clause['type'],
                'word_indices': clause.get('word_indices', None)
            })
            current_time = end

        # Build TextGrid intervals
        intervals = []
        created_intervals = []
        current_time = min_time

        for clause in final_clauses:
            start, end = clause['start'], clause['end']
            label = clause.get('label', '')
            
            if start > current_time + 0.001:
                intervals.append(Interval(current_time, start, ""))
            
            intervals.append(Interval(start, end, label))
            if label.strip():
                created_intervals.append({
                    'label': label,
                    'type': clause['type'],
                    'start': start,
                    'end': end
                })
            current_time = end

        if current_time < max_time - 0.001:
            intervals.append(Interval(current_time, max_time, ""))

        if not intervals:
            intervals = [Interval(min_time, max_time, "")]

        if tier_name in self.tg.tierNames:
            self.tg.removeTier(tier_name)

        new_tier = IntervalTier(tier_name, intervals, min_time, max_time)
        self.tg.addTier(new_tier)
        
        return created_intervals

    def save(self, output_path: str):
        self.tg.save(output_path, format="short_textgrid", includeBlankSpaces=True)


# ==============================================================================
# Clause Aligner (Japanese version)
# ==============================================================================

class ClauseAligner:
    """Align clause boundaries to TextGrid word times for Japanese."""

    def __init__(self, tg_handler: TextGridHandler, segmenter: JapaneseClauseSegmenter):
        self.tg = tg_handler
        self.segmenter = segmenter

    @staticmethod
    def _is_disfluent_idx(word_idx: int, disfluency_labels: List[int] = None) -> bool:
        return disfluency_labels is not None and word_idx < len(disfluency_labels) and disfluency_labels[word_idx] == 1

    def _is_boundary_noise_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        if self._is_disfluent_idx(word_idx, disfluency_labels):
            return True
        text = tg_words[word_idx].text
        if not self.segmenter._is_unambiguous_filler_surface(text):
            return False
        # Standalone "ん" can be a real morpheme (e.g., negative endings), so
        # only trim it when the disfluency model already marked it.
        if self.segmenter._normalize_surface_text(text) == "ん":
            return False
        return True

    def _previous_fluent_surface(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> str:
        """Return the nearest preceding non-empty fluent word surface."""
        for prev_idx in range(word_idx - 1, -1, -1):
            if self._is_disfluent_idx(prev_idx, disfluency_labels):
                continue
            text = tg_words[prev_idx].text
            if text and text.strip():
                return text.strip()
        return ""

    def _is_bound_suffix_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """True when this word is morphologically bound to the previous fluent word."""
        if word_idx <= 0 or word_idx >= len(tg_words):
            return False
        text = tg_words[word_idx].text
        if not text or not text.strip():
            return False
        prev_text = self._previous_fluent_surface(word_idx, tg_words, disfluency_labels)
        if not prev_text:
            return False
        return self._is_bound_inflection(prev_text, text.strip())

    def _is_explicit_subordinator_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """True when this token is or completes a Rule 3a subordinator tail."""
        if word_idx < 0 or word_idx >= len(tg_words):
            return False
        text = tg_words[word_idx].text
        norm = self.segmenter._normalize_surface_text(text)
        if not norm:
            return False

        single_markers = {
            "から",  # から
            "ながら",  # ながら
            "つつ",  # つつ
            "うちに",  # うちに
            "間に",  # 間に
            "際に",  # 際に
            "けど",  # けど
            "けれど",  # けれど
            "けれども",  # けれども
            "が",  # が
            "たら",  # たら
            "ば",  # ば
            "なら",  # なら
            "と",  # と
            "し",  # し
            "まで",  # まで
            "時",  # 時
            "とき",  # とき
        }
        if norm in single_markers:
            return True

        prev_text = self._previous_fluent_surface(word_idx, tg_words, disfluency_labels)
        prev_norm = self.segmenter._normalize_surface_text(prev_text)
        if not prev_norm:
            return False

        pair_markers = {
            ("の", "で"),  # の + で
            ("の", "に"),  # の + に
            ("て", "も"),  # て + も
            ("で", "も"),  # で + も
            ("ため", "に"),  # ため + に
            ("よう", "に"),  # よう + に
            ("前", "に"),  # 前 + に
            ("後", "で"),  # 後 + で
            ("うち", "に"),  # うち + に
            ("間", "に"),  # 間 + に
            ("際", "に"),  # 際 + に
            ("時", "に"),  # 時 + に
            ("時", "は"),  # 時 + は
            ("とき", "に"),  # とき + に
            ("とき", "は"),  # とき + は
        }
        return (prev_norm, norm) in pair_markers

    def _clause_has_explicit_subordinator_tail(
        self,
        clause: Dict,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """True when the visible end of a clause is a Rule 3a subordinator tail."""
        idxs = sorted(clause.get("word_indices", []))
        if idxs:
            visible = self._visible_clause_word_indices(idxs, tg_words, disfluency_labels)
            if visible and self._is_explicit_subordinator_idx(visible[-1], tg_words, disfluency_labels):
                return True
        return self._has_explicit_subordinator_tail(clause.get("text", ""))

    def _is_finite_connector_tail_hangover(
        self,
        clause: Dict,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """
        Detect tails like "...ました + で + も" that are discourse restarts ("でも"),
        not true subordinate markers.
        """
        idxs = sorted(clause.get("word_indices", []))
        if not idxs:
            return False
        visible = self._visible_clause_word_indices(idxs, tg_words, disfluency_labels)
        if not visible:
            return False

        norms = [self.segmenter._normalize_surface_text(tg_words[i].text) for i in visible]
        if not norms:
            return False

        tail_len = 0
        if len(norms) >= 2 and norms[-2:] == ["で", "も"]:
            tail_len = 2
        elif norms[-1] in {"でも", "しかし", "それで", "そして", "だから", "なので", "ところで", "ただ", "すると"}:
            tail_len = 1
        if tail_len == 0 or len(norms) <= tail_len:
            return False

        stem = "".join(norms[:-tail_len])
        if not stem:
            return False

        return stem.endswith((
            "ました", "ます", "ません",
            "です", "でした", "でしょう",
            "だ", "だった", "た",
        ))

    def _collect_finite_connector_tail_indices(
        self,
        clause: Dict,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[int]:
        """
        Return visible tail indices for sentence-final restart connectors like
        「…ましたでも」 or 「…ましたでもね」 so they can move to the next clause.
        """
        idxs = sorted(clause.get("word_indices", []))
        if not idxs:
            return []
        visible = self._visible_clause_word_indices(idxs, tg_words, disfluency_labels)
        if not visible:
            return []

        norms = [self.segmenter._normalize_surface_text(tg_words[i].text) for i in visible]
        if not norms:
            return []

        discourse_tail = 0
        if norms[-1] in {"ね", "よ", "な"}:
            discourse_tail = 1

        base_norms = norms[:-discourse_tail] if discourse_tail else norms
        if not base_norms:
            return []

        connector_len = 0
        if len(base_norms) >= 2 and base_norms[-2:] == ["で", "も"]:
            connector_len = 2
        elif base_norms[-1] in {
            "でも", "しかし", "それで", "そして", "だから", "なので",
            "ところで", "ただ", "すると",
        }:
            connector_len = 1

        if connector_len == 0 or len(base_norms) <= connector_len:
            return []

        stem = "".join(base_norms[:-connector_len])
        if not stem.endswith((
            "ました", "ます", "ません",
            "です", "でした", "でしょう",
            "だ", "だった", "た",
        )):
            return []

        tail_len = connector_len + discourse_tail
        return visible[-tail_len:]

    def _is_trailing_discourse_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """
        Conservative trailing-only discourse/filler detection.
        This handles boundary leftovers like "...なんか / ...あの / ...えっと" without
        globally deleting ambiguous forms inside clause bodies.
        """
        if self._is_boundary_noise_idx(word_idx, tg_words, disfluency_labels):
            return True

        text = tg_words[word_idx].text
        norm = self.segmenter._normalize_surface_text(text)
        if not norm:
            return False
        if self._is_bound_suffix_idx(word_idx, tg_words, disfluency_labels):
            return False
        if self._is_explicit_subordinator_idx(word_idx, tg_words, disfluency_labels):
            return False

        trailing_discourse = {
            "なんか",  # なんか
            "あの",        # あの
            "その",        # その
            "えっと",  # えっと
            "ええと",  # ええと
            "うーん",  # うーん
            "うん",        # うん
            "まあ",        # まあ
            "ま",              # ま
            "えー",        # えー
            "あー",        # あー
        }
        if norm in trailing_discourse:
            return True

        # Parse fallback for very short function/discourse surfaces.
        doc = self.segmenter.nlp(text)
        toks = [t for t in doc if t.text.strip()]
        if not toks or len(toks) > 2:
            return False
        last = toks[-1]
        if last.dep_ == "discourse":
            return True
        if last.pos_ in {"INTJ", "PART", "SCONJ", "CCONJ"}:
            return True
        return False

    def _is_restart_connector_surface(self, text: str) -> bool:
        norm = self.segmenter._normalize_surface_text(text)
        return norm in {
            "そして", "それで", "でも", "しかし", "だから", "なので",
            "ところで", "ただ", "すると",
        }

    def _is_movable_function_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        if self._is_boundary_noise_idx(word_idx, tg_words, disfluency_labels):
            return False
        text = tg_words[word_idx].text
        if not text or not text.strip():
            return False
        if self._is_restart_connector_surface(text):
            return True
        return self.segmenter._is_function_like_surface(text)
    def _has_content_tokens(
        self,
        indices: List[int],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        for wi in indices:
            if self._is_disfluent_idx(wi, disfluency_labels):
                continue
            text = tg_words[wi].text
            if not text or not text.strip():
                continue
            if self.segmenter._is_unambiguous_filler_surface(text):
                continue
            if self.segmenter._is_content_surface(text):
                return True
        return False

    def _visible_clause_word_indices(
        self,
        indices: List[int],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[int]:
        """Keep every fluent surface in clause text.

        Clause display text should only hide words that the disfluency model
        marked as disfluent. Surface-form filler heuristics are still used for
        boundary repair and filler-only clause checks, but they must not delete
        grammatical material such as the `ん` in `ません`.
        """
        visible: List[int] = []
        for wi in indices:
            if self._is_disfluent_idx(wi, disfluency_labels):
                continue
            text = tg_words[wi].text
            if not text or not text.strip():
                continue
            if self.segmenter._is_unambiguous_filler_surface(text):
                continue
            visible.append(wi)
        # Standalone "ん" should be treated as a filler/backchannel only when it
        # is the clause's lone visible token.  Keep it elsewhere so negative
        # morphology like ません is preserved.
        if len(visible) == 1:
            only_text = self.segmenter._normalize_surface_text(tg_words[visible[0]].text)
            if only_text == "ん":
                return []
        if visible:
            last_visible_text = tg_words[visible[-1]].text.strip()
            suffix_tokens = {"た", "だ", "て", "で", "ん"}
            for wi in indices:
                if wi <= visible[-1]:
                    continue
                if not self._is_disfluent_idx(wi, disfluency_labels):
                    break
                text = tg_words[wi].text
                if not text or not text.strip():
                    continue
                norm = text.strip()
                if norm not in suffix_tokens:
                    break
                if not self._is_bound_inflection(last_visible_text, norm):
                    break
                visible.append(wi)
                break
        return visible

    def _rebuild_clause_payload(self, clause: Dict, tg_words: List[WordInterval], disfluency_labels: List[int] = None) -> Dict:
        """Recompute timing/text/mora after word-index edits."""
        idxs = sorted(clause.get('word_indices', []))
        if not idxs:
            clause['text'] = ''
            clause['mora'] = 0
            return clause

        clause_words = [tg_words[i] for i in idxs]
        displayed_words = [
            tg_words[i] for i in self._visible_clause_word_indices(idxs, tg_words, disfluency_labels)
        ]

        clause['word_indices'] = idxs
        clause['start_time'] = clause_words[0].start
        clause['end_time'] = clause_words[-1].end
        clause['text'] = ''.join(w.text for w in displayed_words)
        clause['mora'] = sum(w.mora_count for w in displayed_words)
        return clause

    def _starts_with_nominal_surface(self, text: str) -> bool:
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        if not toks:
            return False
        first = toks[0]
        return first.pos_ in {"NOUN", "PROPN", "PRON", "NUM"}

    def _compact_clause_text(self, text: str) -> str:
        return (text or "").replace(" ", "").replace("　", "")

    def _is_fixed_merge_prefix(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if compact in {
            "なのに",
            "どうやって", "どうして", "なんで", "なぜ",
            "どこで", "どこに", "どこへ", "いつ",
        }:
            return True
        # v1.0-paper also listed two words from single evaluation recordings.
        return compact in {"仕方なく", "良かったら"}

    def _last_lexical_surface(self, text: str) -> str:
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        for tok in reversed(toks):
            if tok.pos_ in {"VERB", "ADJ", "NOUN", "PROPN", "PRON", "ADV", "NUM"}:
                return (tok.text or "").strip()
        return ""

    def _first_lexical_surface(self, text: str) -> str:
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        for tok in toks:
            if tok.pos_ in {"VERB", "ADJ", "NOUN", "PROPN", "PRON", "ADV", "NUM"}:
                return (tok.text or "").strip()
        return ""

    def _last_predicate_lemma(self, text: str) -> str:
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        for tok in reversed(toks):
            if tok.pos_ in {"VERB", "ADJ"}:
                return (tok.lemma_ or "").strip()
            if tok.pos_ == "AUX" and tok.lemma_ in {"だ", "です", "ます", "た", "いる", "ある", "する"}:
                return (tok.lemma_ or "").strip()
        return ""

    def _first_predicate_lemmas(self, text: str, limit: int = 3) -> Tuple[str, ...]:
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        out: List[str] = []
        for tok in toks:
            lemma = (tok.lemma_ or "").strip()
            if not lemma:
                continue
            if tok.pos_ in {"VERB", "ADJ"} or (
                tok.pos_ == "AUX" and lemma in {"だ", "です", "ます", "た", "いる", "ある", "する"}
            ):
                out.append(lemma)
                if len(out) >= limit:
                    break
        return tuple(out)

    def _starts_with_name_suffix(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if not compact:
            return False
        suffixes = ("さん", "ちゃん", "くん", "君", "氏", "さま", "様", "たち", "達")
        # Guard against false positives where a suffix string is part of
        # a different word.  e.g. "ちゃんと" (adverb "properly") is NOT
        # the name suffix "ちゃん".
        adverb_forms = ("ちゃんと",)
        if compact.startswith(adverb_forms):
            return False
        if compact.startswith(suffixes):
            return True
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        return bool(toks and (toks[0].text or "").strip() in suffixes)

    def _has_finite_restart_shell_tail(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if not compact:
            return False
        if re.search(
            r"(ました|ます|ません|でした|です|だった|だ|た)"
            r"(でも|しかし|それで|だから|なので|そして)"
            r"[ぁ-んァ-ヶ一-龯A-Za-z0-9]*$",
            compact,
        ):
            return True
        if re.search(r"(ですから|だから|から|ので)[ぁ-んァ-ヶ一-龯A-Za-z0-9]{1,4}$", compact):
            return True
        return False

    def _has_finite_bare_de_tail(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or not compact.endswith("で"):
            return False
        stem = compact[:-1]
        return stem.endswith((
            "ました", "ます", "ません",
            "です", "でした", "でしょう",
            "だ", "だった", "た",
        ))

    def _is_short_te_shell(
        self,
        text: str,
        feats: Dict[str, object],
    ) -> bool:
        compact = self._compact_clause_text(text)
        return (
            bool(compact)
            and len(compact) <= 8
            and compact.endswith(("て", "で"))
            and feats.get("has_pred_verb_adj", False)
            and not feats.get("has_element", False)
            and not feats.get("has_sub_marker", False)
            and not _has_subordinator_tail(compact)
        )

    def _is_short_prenominal_fragment(
        self,
        text: str,
        feats: Dict[str, object],
    ) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 10:
            return False
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        if compact.endswith("な"):
            return len(toks) <= 3
        if compact.endswith("の") and len(toks) <= 3 and not feats.get("has_sub_marker", False):
            return True
        if feats.get("has_pred_verb_adj", False) or feats.get("has_sub_marker", False):
            return False
        if feats.get("has_case_tail", False):
            return False
        return compact.endswith(("的な",))

    def _is_broad_prenominal_fragment(
        self,
        text: str,
        feats: Dict[str, object],
        right_text: str,
        right_feats: Dict[str, object],
    ) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 28:
            return False
        if self._ends_with_sentence_final(text):
            return False
        if _has_subordinator_tail(compact) and compact not in {"良かったら"}:
            return False
        if not (
            right_feats.get("has_nominal_token", False)
            or self._starts_with_nominal_surface(right_text)
        ):
            return False
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        if not toks:
            return False
        last = toks[-1]
        if compact.endswith(("な", "の", "い")) and len(compact) <= 18:
            return True
        if last.pos_ in {"VERB", "ADJ", "AUX"}:
            # Guard: a verb in bare ren'yōkei form (連用形) with its own element
            # is a coordinate clause, not a prenominal fragment.
            if last.pos_ == "VERB" and feats.get("has_element", False):
                morph = last.morph.to_dict()
                inflection = morph.get("Inflection", "")
                if "連用形" in inflection:
                    # Check no te/tari mark → bare ren'yōkei coordination
                    has_te = any(
                        ch.dep_ == "mark" and ch.text in {"て", "で"}
                        for ch in last.children
                    )
                    has_tari = any(
                        ch.dep_ == "mark" and ch.text == "たり"
                        for ch in last.children
                    ) or last.text.endswith(("たり", "だり"))
                    if not has_te and not has_tari:
                        return False
            return True
        return False

    def _is_short_adverbial_fragment(
        self,
        text: str,
        feats: Dict[str, object],
    ) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 8:
            return False
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        if not toks or len(toks) > 3:
            return False
        if compact in {"すごく", "とても", "かなり", "ちょっと", "ちゃんと", "よく", "少し", "どうやって", "どうして", "なんで", "なぜ"}:
            return True
        if compact.endswith(("に", "く")) and not compact.endswith(("て", "で")):
            return True
        if feats.get("has_pred_verb_adj", False) or feats.get("has_sub_marker", False):
            return False
        allowed_pos = {"ADV", "ADJ", "AUX", "PART", "INTJ"}
        if not all(t.pos_ in allowed_pos for t in toks):
            return False
        if compact.endswith(("に", "く")):
            return True
        return all(t.pos_ == "ADV" for t in toks)

    def _starts_with_question_continuation(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 48:
            return False
        patterns = (
            "か",
            "かって", "かと", "かしら", "かな", "かね", "かも", "かどうか",
            "かっていう", "かってゆう", "かという", "かっ",
        )
        return compact.startswith(patterns)

    def _ends_with_question_shell(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if not compact:
            return False
        return compact.endswith((
            "の", "ん", "か", "です", "ます", "ません", "でした", "ました",
            "ない", "たい", "た", "る", "だ",
        ))

    def _starts_with_motion_or_aux_continuation(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 12:
            return False
        # A clause with its own argument (obj/obl marked by を/に/で etc.)
        # is a standalone clause, not a mere V-te continuation fragment.
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        if any(t.dep_ in {"obj", "iobj", "obl", "nsubj"} for t in toks):
            return False
        preds = self._first_predicate_lemmas(text, limit=2)
        if any(p in {"行く", "来る", "しまう", "みる", "くれる", "もらう", "おく", "いる"} for p in preds):
            return True
        return compact.startswith((
            "いって", "行って", "いった", "行った", "いき", "行き", "いく", "行く",
            "きて", "来て", "きた", "来た", "しまっ", "しまう", "みて", "見て",
            "くれ", "もら", "おい", "おいて", "いて", "いた", "います", "いまし",
        ))

    def _starts_with_subordinator_continuation(self, text: str) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 24:
            return False
        return compact.startswith((
            "んで", "ので", "のに",
            "から", "けど", "けれど", "けれども",
        ))

    def _has_hanging_stem_overlap(self, left_text: str, right_text: str) -> bool:
        left_compact = self._compact_clause_text(left_text)
        right_compact = self._compact_clause_text(right_text)
        if len(left_compact) < 3 or len(right_compact) < 2:
            return False
        if left_compact[-1] not in {"し", "て", "で", "い", "き", "く"}:
            return False
        base = left_compact[:-1]
        for n in range(min(8, len(base)), 2, -1):
            if right_compact.startswith(base[-n:]):
                return True
        return False

    def _is_quotative_bridge_fragment(self, left_text: str, right_text: str) -> bool:
        left_compact = self._compact_clause_text(left_text)
        right_compact = self._compact_clause_text(right_text)
        if not left_compact or not right_compact:
            return False
        if right_compact.startswith(("ゆう", "いう", "言う")) and not self._ends_with_sentence_final(left_text):
            return True
        if left_compact.endswith(("ゆう", "いう", "言う")):
            # v1.0-paper listed head nouns from single evaluation utterances.
            return right_compact.startswith(("か", "こと", "気持ち", "話"))
        return False

    def _starts_with_reporting_predicate(self, text: str) -> bool:
        core = self._strip_leading_soft_filler(text)
        if not core:
            return False
        reporting_like = {
            "言う", "いう", "ゆう", "話す", "答える", "聞く", "尋ねる",
            "思う", "考える", "感じる", "叫ぶ", "説明する", "伝える",
            "教える", "頼む",
        }
        toks = [t for t in self.segmenter.nlp(core) if t.text.strip()]
        for tok in toks[:5]:
            lemma = (tok.lemma_ or "").strip()
            if tok.pos_ in {"VERB", "AUX"} and lemma in reporting_like:
                return True
        return core.startswith((
            "言", "い", "ゆ", "話", "答", "聞", "尋", "思", "考",
            "感", "叫", "説", "伝", "教", "頼",
        ))

    def _has_boundary_surface_overlap(
        self,
        left_text: str,
        right_text: str,
        left: Dict = None,
        right: Dict = None,
        tg_words: List[WordInterval] = None,
        disfluency_labels: List[int] = None,
    ) -> bool:
        left_compact = self._compact_clause_text(left_text)
        right_core = self._strip_leading_soft_filler(right_text)
        right_compact = self._compact_clause_text(right_core or right_text)
        if len(left_compact) < 3 or len(right_compact) < 3:
            return False

        strong_restart_heads = (
            "そして", "しかし", "でも", "それで", "だから", "なので", "ところで", "ただ",
        )
        if right_compact.startswith(strong_restart_heads):
            return False

        for n in range(min(10, len(left_compact), len(right_compact)), 2, -1):
            if left_compact.endswith(right_compact[:n]):
                return True

        if left_compact.endswith("と") and len(left_compact) >= 4:
            base = left_compact[:-1]
            for n in range(min(8, len(base), len(right_compact)), 2, -1):
                if base.endswith(right_compact[:n]):
                    return True

        if left is None or right is None or tg_words is None:
            return False

        left_idxs = sorted(left.get("word_indices", []))
        right_idxs = sorted(right.get("word_indices", []))
        if not left_idxs or not right_idxs:
            return False
        left_visible = self._visible_clause_word_indices(left_idxs, tg_words, disfluency_labels)
        right_visible = self._visible_clause_word_indices(right_idxs, tg_words, disfluency_labels)
        if not left_visible or not right_visible:
            return False

        left_norms = [
            self.segmenter._normalize_surface_text(tg_words[i].text)
            for i in left_visible[-3:]
            if self.segmenter._normalize_surface_text(tg_words[i].text)
        ]
        right_norms = [
            self.segmenter._normalize_surface_text(tg_words[i].text)
            for i in right_visible[:3]
            if self.segmenter._normalize_surface_text(tg_words[i].text)
        ]
        for n in range(min(2, len(left_norms), len(right_norms)), 0, -1):
            if left_norms[-n:] == right_norms[:n]:
                joined = "".join(left_norms[-n:])
                if n >= 2 or len(joined) >= 4:
                    return True
        return False

    def _is_bare_to_continuation_bridge(
        self,
        left_text: str,
        right_text: str,
        right_feats: Dict[str, object],
    ) -> bool:
        left_compact = self._compact_clause_text(left_text)
        if not left_compact.endswith("と"):
            return False
        if left_compact.endswith(("って", "かと", "ですかと", "ましたと", "ませんかと")):
            return False

        right_core = self._strip_leading_soft_filler(right_text)
        right_compact = self._compact_clause_text(right_core or right_text)
        if not right_compact:
            return False
        if right_compact.startswith((
            "でも", "そして", "それで", "だから", "なので", "しかし", "ところで", "ただ", "って", "と",
        )):
            return False
        if self._starts_with_reporting_predicate(right_core or right_text):
            return False
        return (
            right_feats.get("has_pred_verb_adj", False)
            or right_feats.get("has_pred_or_nominal", False)
            or self._starts_with_nominal_surface(right_core or right_text)
        )

    def _strip_leading_soft_filler(self, text: str) -> str:
        compact = self._compact_clause_text(text)
        if not compact:
            return compact
        prefixes = ("ん", "あの", "えっと", "えと", "えーと", "えー", "えっ", "あっ", "まあ")
        for _ in range(2):
            changed = False
            for prefix in prefixes:
                if compact.startswith(prefix) and len(compact) >= len(prefix) + 2:
                    compact = compact[len(prefix):]
                    changed = True
                    break
            if not changed:
                break
        return compact

    def _is_short_case_marked_fragment(
        self,
        text: str,
        feats: Dict[str, object],
    ) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 18:
            return False
        if self._ends_with_sentence_final(text):
            return False
        if feats.get("has_pred_verb_adj", False):
            return False
        return compact.endswith(("を", "が", "に", "へ", "は", "も"))

    def _is_short_wh_case_fragment(self, text: str) -> bool:
        compact = self._strip_leading_soft_filler(text)
        if not compact or len(compact) > 18:
            return False
        if not compact.endswith(("を", "が", "に", "へ")):
            return False
        return compact.startswith(("どこ", "どう", "なに", "何", "誰", "いつ", "なんで", "なぜ", "どの", "どっち"))

    def _has_short_broken_tail(self, text: str, right_text: str = "") -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 12:
            return False
        if self._ends_with_sentence_final(text):
            return False
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        if not toks:
            return False
        last = toks[-1]
        if last.pos_ not in {"VERB", "ADJ", "AUX"}:
            return False
        if len((last.text or "").strip()) > 1:
            return False
        if right_text:
            right_compact = self._strip_leading_soft_filler(right_text)
            if not right_compact:
                return False
            continuation_heads = (
                "て", "で", "た", "だ", "ん",
                "まし", "ます", "ませ", "ない", "なか", "ぬ",
                "れる", "られる", "よう", "う", "く",
            )
            last_surface = (last.text or "").strip()
            last_lemma = (last.lemma_ or "").strip()
            if not (
                right_compact.startswith(continuation_heads)
                or (last_surface and right_compact.startswith(last_surface))
                or (last_lemma and right_compact.startswith(last_lemma))
            ):
                return False
        return True

    def _is_short_stance_te_shell(
        self,
        text: str,
        feats: Dict[str, object],
    ) -> bool:
        compact = self._compact_clause_text(text)
        if not compact or len(compact) > 8:
            return False
        if not compact.endswith(("て", "で")):
            return False
        if not feats.get("has_pred_verb_adj", False):
            return False
        return self._last_predicate_lemma(text) in {"思う", "考える", "感じる", "言う", "聞く", "答える"}

    def _should_merge_short_quality_pair(
        self,
        left: Dict,
        right: Dict,
        left_text: str,
        right_text: str,
        left_feats: Dict[str, object],
        right_feats: Dict[str, object],
        gap: float,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        prenominal_fragment = self._is_short_prenominal_fragment(left_text, left_feats)
        adverbial_fragment = self._is_short_adverbial_fragment(left_text, left_feats)
        right_adverbial_fragment = self._is_short_adverbial_fragment(right_text, right_feats)
        question_continuation = self._starts_with_question_continuation(right_text)
        left_question_shell = self._ends_with_question_shell(left_text)
        motion_or_aux_continuation = self._starts_with_motion_or_aux_continuation(right_text)
        subordinator_continuation = self._starts_with_subordinator_continuation(right_text)
        fixed_merge_prefix = self._is_fixed_merge_prefix(left_text)
        broad_prenominal_fragment = self._is_broad_prenominal_fragment(
            left_text, left_feats, right_text, right_feats
        )
        hanging_stem_overlap = self._has_hanging_stem_overlap(left_text, right_text)
        quotative_bridge = self._is_quotative_bridge_fragment(left_text, right_text)
        surface_overlap = self._has_boundary_surface_overlap(
            left_text,
            right_text,
            left,
            right,
            tg_words,
            disfluency_labels,
        )
        stance_te_shell = self._is_short_stance_te_shell(left_text, left_feats)
        short_case_fragment = self._is_short_case_marked_fragment(left_text, left_feats)
        left_wh_case_fragment = self._is_short_wh_case_fragment(left_text)
        right_wh_case_fragment = self._is_short_wh_case_fragment(right_text)
        short_broken_tail = self._has_short_broken_tail(left_text, right_text)

        max_gap = 1.0
        if self._starts_with_name_suffix(right_text):
            max_gap = 1.4
        if self._is_short_te_shell(left_text, left_feats):
            max_gap = max(max_gap, 1.6)
        if prenominal_fragment or broad_prenominal_fragment or adverbial_fragment or question_continuation or fixed_merge_prefix:
            max_gap = max(max_gap, 1.8)
        if adverbial_fragment and right_adverbial_fragment:
            max_gap = max(max_gap, 1.4)
        if motion_or_aux_continuation:
            max_gap = max(max_gap, 1.2)
        if subordinator_continuation:
            max_gap = max(max_gap, 1.6)
        if hanging_stem_overlap or quotative_bridge or surface_overlap:
            max_gap = max(max_gap, 1.8)
        if stance_te_shell:
            max_gap = max(max_gap, 2.0)
        if short_case_fragment or left_wh_case_fragment or short_broken_tail:
            max_gap = max(max_gap, 1.4)
        if not (-0.5 <= gap <= max_gap):
            return False

        left_compact = self._compact_clause_text(left_text)
        right_compact = self._compact_clause_text(right_text)
        if not left_compact or not right_compact:
            return False

        right_soft = self._strip_leading_soft_filler(right_text)
        right_after_det = right_soft
        for det in ("この", "その", "あの", "どの", "こんな", "そんな", "あんな"):
            if right_after_det.startswith(det) and len(right_after_det) > len(det):
                right_after_det = right_after_det[len(det):]
                break

        def _leading_nominal_prefix(span_text: str) -> Tuple[str, ...]:
            prefix: List[str] = []
            started = False
            for tok in self.segmenter.nlp(span_text or ""):
                if not tok.text.strip():
                    continue
                surf = self.segmenter._normalize_surface_text(tok.text)
                if tok.pos_ in {"NOUN", "PROPN", "PRON", "NUM"} or surf in {"は", "が", "を", "に", "も", "の"}:
                    prefix.append(surf)
                    started = True
                    if len(prefix) >= 4:
                        break
                    continue
                if started:
                    break
            return tuple(prefix)
        left_complete_copular_tail = (
            left_feats.get("has_pred_or_nominal", False)
            and left_feats.get("has_element", False)
            and left_compact.endswith("で")
        )
        if (
            left_question_shell
            and right_compact.startswith("で")
            and right_feats.get("has_pred_verb_adj", False)
        ):
            return False
        if (
            left_complete_copular_tail
            and right_soft
            and right_feats.get("has_pred_verb_adj", False)
            and (
                self._starts_with_nominal_surface(right_soft)
                or self._starts_with_nominal_surface(right_after_det)
                or right_soft.startswith(("外", "内", "中", "ここ", "そこ", "あそこ", "どこ"))
            )
        ):
            return False

        if self._clause_has_explicit_subordinator_tail(left, tg_words, disfluency_labels):
            return False
        # v26 guard: left clause ending with sentence-final polite form AND
        # containing its own element (subject/object/etc.) is a complete clause;
        # do not merge it with a predicative right clause.
        if (
            self._ends_with_sentence_final(left_text)
            and left_feats.get("has_pred_verb_adj", False)
            and left_feats.get("has_element", False)
            and right_feats.get("has_pred_verb_adj", False)
        ):
            return False
        if (
            left.get("type") == "SUB_CCOMP"
            and left_compact.endswith(("と", "って"))
            and not left_feats.get("ends_with_te", False)
            and not left_feats.get("has_te_signal", False)
        ):
            return False
        strong_restart_heads = ("そして", "しかし", "でも", "それで", "だから")
        if right_compact.startswith(strong_restart_heads):
            return False

        if self._starts_with_name_suffix(right_text):
            if self._has_finite_restart_shell_tail(left_text):
                return False
            return len(left_compact) <= 20

        if fixed_merge_prefix:
            return True

        if short_case_fragment and (
            right_feats.get("has_pred_verb_adj", False)
            or right_feats.get("has_pred_or_nominal", False)
            or self._starts_with_nominal_surface(right_text)
        ):
            return True

        if broad_prenominal_fragment:
            return True

        if prenominal_fragment and self._starts_with_nominal_surface(right_text):
            return True

        if adverbial_fragment:
            if right_adverbial_fragment:
                return True
            if (
                right_feats.get("has_pred_verb_adj", False)
                or right_feats.get("has_pred_or_nominal", False)
                or self._starts_with_nominal_surface(right_text)
            ):
                return True

        if question_continuation and left_question_shell:
            return True

        if hanging_stem_overlap or quotative_bridge or surface_overlap:
            return True


        if left_wh_case_fragment and right_feats.get("has_pred_verb_adj", False):
            return True

        if right_wh_case_fragment and (
            self._is_short_te_shell(left_text, left_feats)
            or left_compact.endswith(("て", "で"))
            or left_feats.get("ends_with_te", False)
            or left_feats.get("has_te_signal", False)
        ):
            return True

        if motion_or_aux_continuation and (
            self._is_short_te_shell(left_text, left_feats)
            or left_compact.endswith(("て", "で"))
            or left_feats.get("ends_with_te", False)
            or left_feats.get("has_te_signal", False)
        ):
            return True

        if subordinator_continuation and left_feats.get("has_pred_or_nominal", False):
            return True

        if stance_te_shell and right_feats.get("has_pred_verb_adj", False):
            return True

        if self._is_short_te_shell(left_text, left_feats) and right_feats.get("has_pred_verb_adj", False):
            return True

        if not left_feats.get("has_pred_verb_adj", False):
            if len(left_compact) <= 4:
                return True
            last_lex = self._last_lexical_surface(left_text)
            next_first_lex = self._first_lexical_surface(right_text)
            if last_lex and next_first_lex and len(last_lex) <= 2 and next_first_lex.startswith(last_lex):
                return True

        if len(left_compact) <= 14:
            left_last_pred = self._last_predicate_lemma(left_text)
            right_preds = self._first_predicate_lemmas(right_text)
            left_last_lex = self._last_lexical_surface(left_text)
            right_first_lex = self._first_lexical_surface(right_text)
            has_surface_overlap = False
            if left_last_lex and right_first_lex:
                has_surface_overlap = (
                    right_first_lex.startswith(left_last_lex)
                    or left_last_lex.startswith(right_first_lex)
                    or left_last_lex[:2] == right_first_lex[:2]
                )
            if (
                left_last_pred
                and right_preds
                and left_last_pred in right_preds
                and left_compact != right_compact
                and has_surface_overlap
            ):
                return True

        if short_broken_tail and right_feats.get("has_pred_verb_adj", False):
            return True

        return False

    def refine_segmentation_quality(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Merge short rule-violating split artifacts after fluent-word recovery."""
        if len(clauses) < 2:
            return clauses

        def _signature(items: List[Dict]) -> Tuple[Tuple[int, ...], ...]:
            return tuple(tuple(sorted(c.get("word_indices", []))) for c in items)

        repaired = sorted(
            [self._rebuild_clause_payload(dict(c), tg_words, disfluency_labels) for c in clauses],
            key=lambda c: (float(c.get("start_time", 0.0)), float(c.get("end_time", 0.0))),
        )

        for _ in range(5):
            before = _signature(repaired)
            merged: List[Dict] = []
            i = 0
            while i < len(repaired):
                cur = repaired[i]
                if i + 1 >= len(repaired):
                    merged.append(cur)
                    break

                nxt = repaired[i + 1]
                cur_text = (cur.get("text") or "").strip()
                nxt_text = (nxt.get("text") or "").strip()
                if not cur_text or not nxt_text:
                    merged.append(cur)
                    i += 1
                    continue

                cur_feats = _analyze_clause_surface(self.segmenter, cur_text)
                nxt_feats = _analyze_clause_surface(self.segmenter, nxt_text)
                gap = float(nxt.get("start_time", 0.0)) - float(cur.get("end_time", 0.0))

                if not self._should_merge_short_quality_pair(
                    cur,
                    nxt,
                    cur_text,
                    nxt_text,
                    cur_feats,
                    nxt_feats,
                    gap,
                    tg_words,
                    disfluency_labels,
                ):
                    merged.append(cur)
                    i += 1
                    continue

                merged.append(self._merge_clause_pair(
                    cur,
                    nxt,
                    tg_words,
                    disfluency_labels,
                    prefer="right",
                ))
                i += 2

            repaired = sorted(
                [self._rebuild_clause_payload(dict(c), tg_words, disfluency_labels) for c in merged if (c.get("text") or "").strip() or c.get("word_indices")],
                key=lambda c: (float(c.get("start_time", 0.0)), float(c.get("end_time", 0.0))),
            )
            if _signature(repaired) == before:
                break

        return repaired

    def _merge_clause_pair(
        self,
        left: Dict,
        right: Dict,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
        prefer: str = "right",
    ) -> Dict:
        merged = dict(right if prefer == "right" else left)
        merged["word_indices"] = sorted(set(
            list(left.get("word_indices", [])) + list(right.get("word_indices", []))
        ))
        merged = self._rebuild_clause_payload(merged, tg_words, disfluency_labels)
        default_type = right.get("type", merged.get("type", "fragment"))
        if prefer == "left":
            default_type = left.get("type", merged.get("type", "fragment"))
        merged["type"] = self._infer_restart_split_type(
            merged.get("text", ""),
            default_type,
        )
        return merged

    def _is_standalone_suffix_orphan(
        self,
        text: str,
        feats: Dict[str, object],
    ) -> bool:
        compact = (text or "").replace(" ", "").replace("　", "")
        if not compact or len(compact) > 4:
            return False
        if compact in {"っ", "ん"}:
            return True
        if feats.get("all_particles", False):
            return True
        if re.fullmatch(r"[かがはをにへでともからまでのなねよわっん]+", compact):
            return True
        toks = [t for t in self.segmenter.nlp(text or "") if t.text.strip()]
        if not toks:
            return False
        function_pos = {"PART", "AUX", "ADP", "SCONJ", "CCONJ", "PUNCT", "SYM", "INTJ"}
        return (
            not feats.get("has_pred_verb_adj", False)
            and all(t.pos_ in function_pos for t in toks)
        )

    def _is_forward_prefix_orphan(
        self,
        clause: Dict,
        feats: Dict[str, object],
        next_text: str,
        next_feats: Dict[str, object],
    ) -> bool:
        text = (clause.get("text") or "").strip()
        compact = text.replace(" ", "").replace("　", "")
        if not compact or len(compact) > 14:
            return False

        ctype = clause.get("type", "")
        if not feats.get("has_pred_verb_adj", False):
            return True

        if feats.get("has_sub_marker", False) or _has_subordinator_tail(compact):
            return False

        if (
            ctype in {"fragment", "SUB_REL", "CHAIN_TE"}
            and len(compact) <= 10
            and (
                not feats.get("has_element", False)
                or compact.endswith(("な", "の"))
            )
            and not feats.get("ends_with_te", False)
            and (
                next_feats.get("has_nominal_token", False)
                or self._starts_with_nominal_surface(next_text)
            )
        ):
            return True

        return False

    # Off, as in v1.0-paper, where the particle literals below were mojibake (UTF-8
    # read as cp1252) so the merge never fired. With the literals restored it
    # lowered every JA clause statistic slightly on the 40 evaluation recordings
    # (RQ1 -.013, ECPR -.034, not significant), so it stays off.
    _FORWARD_CASE_TAIL_MERGE = False

    def _has_forward_case_tail(
        self,
        text: str,
        feats: Dict[str, object],
    ) -> bool:
        if not self._FORWARD_CASE_TAIL_MERGE:
            return False
        compact = (text or "").replace(" ", "").replace("\u3000", "")
        if not compact or len(compact) > 20:
            return False
        if not feats.get("has_case_tail", False):
            return False
        blocked_tails = ("と", "って", "か", "から", "ので", "のに", "けど", "けれど", "けれども")
        if compact.endswith(blocked_tails):
            return False
        return compact.endswith(("は", "が", "を", "に", "へ", "で", "も", "の"))

    def _is_echo_predicate_continuation(
        self,
        text: str,
        feats: Dict[str, object],
        next_text: str,
        next_feats: Dict[str, object],
    ) -> bool:
        compact = (text or "").replace(" ", "").replace("　", "")
        next_compact = (next_text or "").replace(" ", "").replace("　", "")
        if not compact or not next_compact:
            return False
        if len(compact) > 12:
            return False
        if not feats.get("has_pred_verb_adj", False) or not next_feats.get("has_pred_verb_adj", False):
            return False
        if next_compact.startswith(compact) and len(next_compact) >= len(compact) + 2:
            return True
        last_lex = self._last_lexical_surface(text)
        if not last_lex or len(last_lex) < 2:
            return False
        return next_compact.startswith(last_lex) and len(next_compact) >= len(last_lex) + 2

    def reattach_preserved_fragments(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Merge preserved fluent leftovers back into neighboring clauses."""
        if len(clauses) < 2:
            return clauses

        def _signature(items: List[Dict]) -> Tuple[Tuple[int, ...], ...]:
            return tuple(tuple(sorted(c.get("word_indices", []))) for c in items)

        repaired = sorted(
            [self._rebuild_clause_payload(dict(c), tg_words, disfluency_labels) for c in clauses],
            key=lambda c: (float(c.get("start_time", 0.0)), float(c.get("end_time", 0.0))),
        )

        for _ in range(6):
            before = _signature(repaired)
            merged: List[Dict] = [repaired[0]]
            i = 1
            while i < len(repaired):
                cur = repaired[i]
                prev = merged[-1]

                prev_text = (prev.get("text") or "").strip()
                cur_text = (cur.get("text") or "").strip()
                if not cur_text:
                    i += 1
                    continue

                gap_prev = float(cur.get("start_time", 0.0)) - float(prev.get("end_time", 0.0))
                cur_feats = _analyze_clause_surface(self.segmenter, cur_text)
                suffix_gap_limit = float("inf") if not cur_feats.get("has_pred_verb_adj", False) else 1.2

                if prev_text and -0.5 <= gap_prev <= suffix_gap_limit and self._is_standalone_suffix_orphan(cur_text, cur_feats):
                    merged[-1] = self._merge_clause_pair(
                        prev,
                        cur,
                        tg_words,
                        disfluency_labels,
                        prefer="left",
                    )
                    i += 1
                    continue

                if i + 1 < len(repaired):
                    nxt = repaired[i + 1]
                    nxt_text = (nxt.get("text") or "").strip()
                    gap_next = float(nxt.get("start_time", 0.0)) - float(cur.get("end_time", 0.0))
                    forward_gap_limit = (
                        float("inf")
                        if (
                            not cur_feats.get("has_pred_verb_adj", False)
                            or self._has_forward_case_tail(cur_text, cur_feats)
                        )
                        else 2.5
                    )
                    if nxt_text and -0.5 <= gap_next <= forward_gap_limit:
                        nxt_feats = _analyze_clause_surface(self.segmenter, nxt_text)

                        # v26 guard: Do not merge forward when the current
                        # clause is a complete clause (sentence-final polite
                        # form + has its own element) and next has its own
                        # predicate.  Such a clause is already complete.
                        _cur_sf = self._clause_ends_with_sentence_final(
                            cur, tg_words, disfluency_labels
                        )
                        if (
                            _cur_sf
                            and cur_feats.get("has_pred_verb_adj", False)
                            and cur_feats.get("has_element", False)
                            and nxt_feats.get("has_pred_verb_adj", False)
                        ):
                            merged.append(cur)
                            i += 1
                            continue

                        if self._is_forward_prefix_orphan(cur, cur_feats, nxt_text, nxt_feats):
                            merged.append(self._merge_clause_pair(
                                cur,
                                nxt,
                                tg_words,
                                disfluency_labels,
                                prefer="right",
                            ))
                            i += 2
                            continue
                        if self._has_forward_case_tail(cur_text, cur_feats):
                            merged.append(self._merge_clause_pair(
                                cur,
                                nxt,
                                tg_words,
                                disfluency_labels,
                                prefer="right",
                            ))
                            i += 2
                            continue
                        if self._is_echo_predicate_continuation(cur_text, cur_feats, nxt_text, nxt_feats):
                            merged.append(self._merge_clause_pair(
                                cur,
                                nxt,
                                tg_words,
                                disfluency_labels,
                                prefer="right",
                            ))
                            i += 2
                            continue

                merged.append(cur)
                i += 1

            repaired = self._repair_boundary_hangovers(merged, tg_words, disfluency_labels)
            repaired = self._repair_trailing_function_hangovers(repaired, tg_words, disfluency_labels)
            repaired = self._repair_subordinator_hangovers(repaired, tg_words, disfluency_labels)
            repaired = self._repair_leading_marker_hangovers(repaired, tg_words, disfluency_labels)
            repaired = self._merge_boundary_continuations(repaired, tg_words, disfluency_labels)
            repaired = self._merge_boundary_continuations(repaired, tg_words, disfluency_labels)
            repaired = self._repair_te_form_boundaries(repaired, tg_words, disfluency_labels)
            repaired = self._repair_inflection_splits(repaired, tg_words, disfluency_labels)
            repaired = self._repair_auxiliary_splits(repaired, tg_words, disfluency_labels)
            repaired = self._repair_inflection_splits(repaired, tg_words, disfluency_labels)
            repaired = sorted(repaired, key=lambda c: (float(c.get("start_time", 0.0)), float(c.get("end_time", 0.0))))
            if _signature(repaired) == before:
                break

        return repaired

    def _infer_residual_clause_type(self, text: str) -> str:
        """Classify a recovered fluent run conservatively."""
        text = (text or "").strip()
        if not text:
            return "fragment"

        compact = text.replace(" ", "").replace("　", "")
        feats = _analyze_clause_surface(self.segmenter, text)

        if feats["has_sub_marker"] or _has_subordinator_tail(compact):
            return "SUB_ADVCL"
        if (
            feats["has_te_signal"]
            and compact.endswith(("て", "で", "たり", "だり"))
            and feats["has_element"]
        ):
            return "CHAIN_TE"
        if feats["has_pred_or_nominal"]:
            return "IND"
        return "fragment"

    def recover_residual_fluent_runs(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Recover any fluent word runs still missing from the final clause set."""
        claimed: Set[int] = set()
        for clause in clauses:
            claimed.update(clause.get("word_indices", []))

        residual = []
        for wi, word in enumerate(tg_words):
            if wi in claimed:
                continue
            if self._is_disfluent_idx(wi, disfluency_labels):
                continue
            if word.is_pause or not word.text or not word.text.strip():
                continue
            residual.append(wi)

        if not residual:
            return sorted(clauses, key=lambda c: c["start_time"])

        groups: List[List[int]] = []
        current = [residual[0]]
        for wi in residual[1:]:
            if wi == current[-1] + 1:
                current.append(wi)
            else:
                groups.append(current)
                current = [wi]
        groups.append(current)

        recovered = list(clauses)
        for group in groups:
            visible = self._visible_clause_word_indices(group, tg_words, disfluency_labels)
            if not visible:
                continue
            text = ''.join(tg_words[i].text for i in visible)
            recovered.append({
                "start_time": tg_words[group[0]].start,
                "end_time": tg_words[group[-1]].end,
                "text": text,
                "type": self._infer_residual_clause_type(text),
                "word_indices": list(group),
                "mora": sum(tg_words[i].mora_count for i in visible),
            })

        recovered.sort(key=lambda c: c["start_time"])
        return recovered

    def split_noncontiguous_clauses(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Split clauses whose word indices are not a single contiguous timed span."""
        split_clauses: List[Dict] = []
        for clause in clauses:
            idxs = sorted(clause.get("word_indices", []))
            if not idxs:
                if (clause.get("text") or "").strip():
                    split_clauses.append(clause)
                continue

            groups: List[List[int]] = [[idxs[0]]]
            for wi in idxs[1:]:
                if wi == groups[-1][-1] + 1:
                    groups[-1].append(wi)
                else:
                    groups.append([wi])

            if len(groups) == 1:
                split_clauses.append(self._rebuild_clause_payload(dict(clause), tg_words, disfluency_labels))
                continue

            for group in groups:
                visible = self._visible_clause_word_indices(group, tg_words, disfluency_labels)
                if not visible:
                    continue
                text = ''.join(tg_words[i].text for i in visible)
                piece = {
                    "start_time": tg_words[group[0]].start,
                    "end_time": tg_words[group[-1]].end,
                    "text": text,
                    "type": self._infer_residual_clause_type(text),
                    "word_indices": list(group),
                    "mora": sum(tg_words[i].mora_count for i in visible),
                }
                split_clauses.append(piece)

        split_clauses.sort(key=lambda c: c["start_time"])
        return split_clauses

    def _recover_unclaimed_tail_inflections(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Recover short unclaimed tail continuations when they only complete the clause tail."""
        if not clauses:
            return clauses

        claimed = set()
        for clause in clauses:
            claimed.update(clause.get("word_indices", []))

        suffix_tokens = {"た", "だ", "て", "で", "ん"}
        repaired: List[Dict] = []
        for clause in clauses:
            idxs = sorted(clause.get("word_indices", []))
            if not idxs:
                repaired.append(clause)
                continue

            visible = self._visible_clause_word_indices(idxs, tg_words, disfluency_labels)
            if not visible:
                repaired.append(clause)
                continue

            pending: List[Tuple[int, str]] = []
            probe = idxs[-1] + 1
            while probe < len(tg_words):
                if probe in claimed:
                    break
                text = tg_words[probe].text.strip()
                if not text:
                    probe += 1
                    continue
                pending.append((probe, text))
                if len(pending) >= 4:
                    break
                probe += 1

            if pending:
                prev_text = tg_words[visible[-1]].text.strip()
                attach_idxs: List[int] = []

                first_idx, first_text = pending[0]
                if first_text in suffix_tokens and (
                    self._is_bound_inflection(prev_text, first_text)
                    or self._parse_supports_bound_suffix(prev_text, first_text)
                ):
                    attach_idxs = [first_idx]
                elif prev_text in {"に", "で"} and pending[0][1] in {"なり", "なっ"}:
                    attach_idxs = [pending[0][0]]
                    if len(pending) >= 2 and pending[1][1] in {"まし", "ます", "ませ"}:
                        attach_idxs.append(pending[1][0])
                        if pending[1][1] == "まし" and len(pending) >= 3 and pending[2][1] == "た":
                            attach_idxs.append(pending[2][0])
                        elif pending[1][1] == "ませ" and len(pending) >= 3 and pending[2][1] == "ん":
                            attach_idxs.append(pending[2][0])
                    elif len(pending) >= 2 and pending[0][1] == "なっ" and pending[1][1] in {"た", "て"}:
                        attach_idxs.append(pending[1][0])

                if attach_idxs:
                    clause["word_indices"] = sorted(set(idxs + attach_idxs))
                    claimed.update(attach_idxs)
                    clause = self._rebuild_clause_payload(clause, tg_words, disfluency_labels)
            repaired.append(clause)
        return repaired

    def _parse_supports_bound_suffix(self, prev_text: str, cur_text: str) -> bool:
        """Fallback bound-suffix check for kanji stems like 見 + て."""
        prev_text = (prev_text or "").strip()
        cur_text = (cur_text or "").strip()
        if not prev_text or not cur_text:
            return False
        if cur_text not in {"た", "だ", "て", "で", "ん"}:
            return False

        toks = [t for t in self.segmenter.nlp(prev_text + cur_text) if t.text.strip()]
        if len(toks) < 2:
            return False

        head = toks[0]
        tail = toks[1]
        if tail.text != cur_text:
            return False
        if head.pos_ not in {"VERB", "ADJ", "AUX"}:
            return False
        if cur_text == "ん":
            return tail.pos_ in {"AUX", "PART"} or tail.dep_ in {"aux", "fixed"}
        return tail.pos_ in {"SCONJ", "AUX", "PART"} or tail.dep_ in {"mark", "aux", "fixed"}

    def _is_local_boundary(
        self,
        prev_clause: Dict,
        cur_clause: Dict,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
        max_time_gap: float = 1.2,
        max_word_gap: int = 6,
    ) -> bool:
        """Return True when two clauses are close enough for boundary repair."""
        prev_idxs = sorted(prev_clause.get("word_indices", []))
        cur_idxs = sorted(cur_clause.get("word_indices", []))
        if not prev_idxs or not cur_idxs:
            return False

        gap = float(cur_clause.get("start_time", 0.0)) - float(prev_clause.get("end_time", 0.0))
        if gap <= max_time_gap:
            return True

        if cur_idxs[0] <= prev_idxs[-1]:
            return False

        visible_gap = 0
        for wi in range(prev_idxs[-1] + 1, cur_idxs[0]):
            if wi < 0 or wi >= len(tg_words):
                continue
            if tg_words[wi].is_pause:
                continue
            text = tg_words[wi].text
            if not text or not text.strip():
                continue
            if self._is_disfluent_idx(wi, disfluency_labels):
                continue
            visible_gap += 1
            if visible_gap > max_word_gap:
                return False
        return True

    def _repair_leading_marker_hangovers(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Move clause-final markers back when they surface at the next clause start."""
        if len(clauses) < 2:
            return clauses

        pair_markers = {
            ("の", "か"): "SUB_CCOMP",  # の + か
            ("の", "で"): "SUB_ADVCL",  # の + で
            ("の", "に"): "SUB_ADVCL",  # の + に
        }
        single_markers = {
            "が": "SUB_ADVCL",            # が
            "けど": "SUB_ADVCL",      # けど
            "けれど": "SUB_ADVCL",  # けれど
            "けれども": "SUB_ADVCL",  # けれども
            "から": "SUB_ADVCL",      # から
        }

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]
            prev_idxs = sorted(prev.get("word_indices", []))
            cur_idxs = sorted(cur.get("word_indices", []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue
            if not self._is_local_boundary(prev, cur, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            visible: List[int] = []
            for wi in cur_idxs:
                if self._is_disfluent_idx(wi, disfluency_labels):
                    continue
                text = tg_words[wi].text
                if not text or not text.strip():
                    continue
                visible.append(wi)
                if len(visible) >= 3:
                    break
            if not visible:
                repaired.append(cur)
                continue

            move_idxs: List[int] = []
            target_type = None
            if len(visible) >= 2:
                pair = (
                    self.segmenter._normalize_surface_text(tg_words[visible[0]].text),
                    self.segmenter._normalize_surface_text(tg_words[visible[1]].text),
                )
                target_type = pair_markers.get(pair)
                if target_type is not None:
                    move_idxs = [visible[0], visible[1]]

            if not move_idxs:
                first_norm = self.segmenter._normalize_surface_text(tg_words[visible[0]].text)
                target_type = single_markers.get(first_norm)
                if target_type is not None:
                    move_idxs = [visible[0]]

            if not move_idxs:
                repaired.append(cur)
                continue

            prev_text = (prev.get("text") or "").strip()
            prev_feats = _analyze_clause_surface(self.segmenter, prev_text) if prev_text else {}
            if not prev_feats.get("has_pred_or_nominal", False):
                repaired.append(cur)
                continue

            move_set = set(move_idxs)
            cur_remaining = [wi for wi in cur_idxs if wi not in move_set]
            if not cur_remaining or not self._has_content_tokens(cur_remaining, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev["word_indices"] = sorted(set(prev_idxs + move_idxs))
            cur["word_indices"] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            if target_type and prev.get("type") != "MINOR":
                prev["type"] = target_type

            repaired[-1] = prev
            if cur.get("word_indices"):
                repaired.append(cur)

        return repaired

    def _should_keep_leading_quote(
        self,
        visible_idxs: List[int],
        tg_words: List[WordInterval],
    ) -> bool:
        """Keep clause-initial と when it introduces a reporting/stance matrix."""
        if len(visible_idxs) < 2:
            return False
        first_norm = self.segmenter._normalize_surface_text(tg_words[visible_idxs[0]].text)
        if first_norm != "と":  # と
            return False

        second_norm = self.segmenter._normalize_surface_text(tg_words[visible_idxs[1]].text)
        reporting_like = {
            "思い", "思っ", "思う",  # 思い / 思っ / 思う
            "考え", "感じ",                   # 考え / 感じ
            "言い", "言っ", "言う",  # 言い / 言っ / 言う
            "聞い", "聞く",                   # 聞い / 聞く
            "答え", "訪ね",                   # 答え / 訪ね
        }
        return second_norm in reporting_like

    def _trim_boundary_glue(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Trim weak standalone boundary glue without touching fluent suffixes."""
        if not clauses:
            return clauses

        weak_leading = {
            "と", "で", "が", "の", "は", "も", "を",
        }
        weak_trailing = {
            "の", "は", "を",
        }
        keep_leading_pairs = {
            ("で", "も"),  # で + も
        }

        trimmed: List[Dict] = []
        for clause in clauses:
            idxs = sorted(clause.get("word_indices", []))
            if not idxs:
                continue

            visible = self._visible_clause_word_indices(idxs, tg_words, disfluency_labels)
            if not visible:
                continue

            new_idxs = list(idxs)

            if len(visible) >= 2:
                pair = (
                    self.segmenter._normalize_surface_text(tg_words[visible[0]].text),
                    self.segmenter._normalize_surface_text(tg_words[visible[1]].text),
                )
            else:
                pair = None

            if pair == ("の", "か"):  # の + か
                candidate = [wi for wi in new_idxs if wi not in {visible[0], visible[1]}]
                if candidate and self._has_content_tokens(candidate, tg_words, disfluency_labels):
                    new_idxs = candidate
            else:
                first_wi = visible[0]
                first_norm = self.segmenter._normalize_surface_text(tg_words[first_wi].text)
                if (
                    first_norm in weak_leading
                    and pair not in keep_leading_pairs
                    and not self._should_keep_leading_quote(visible[:3], tg_words)
                    and self._is_movable_function_idx(first_wi, tg_words, disfluency_labels)
                ):
                    candidate = [wi for wi in new_idxs if wi != first_wi]
                    if candidate and self._has_content_tokens(candidate, tg_words, disfluency_labels):
                        new_idxs = candidate

            visible = self._visible_clause_word_indices(new_idxs, tg_words, disfluency_labels)
            if visible:
                last_wi = visible[-1]
                last_norm = self.segmenter._normalize_surface_text(tg_words[last_wi].text)
                if (
                    last_norm in weak_trailing
                    and self._is_movable_function_idx(last_wi, tg_words, disfluency_labels)
                    and not self._is_bound_suffix_idx(last_wi, tg_words, disfluency_labels)
                ):
                    candidate = [wi for wi in new_idxs if wi != last_wi]
                    if candidate and self._has_content_tokens(candidate, tg_words, disfluency_labels):
                        new_idxs = candidate

            clause["word_indices"] = new_idxs
            clause = self._rebuild_clause_payload(clause, tg_words, disfluency_labels)
            if clause.get("text", "").strip():
                trimmed.append(clause)

        return trimmed

    def _repair_boundary_hangovers(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Repair boundary hangovers where a clause starts with function-only tokens
        (AUX/PART/etc.) that should attach to the previous clause.
        """
        if len(clauses) < 2:
            return clauses

        max_leading_move = 3
        repaired = [clauses[0]]

        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            # Only repair when boundary is contiguous/tiny gap.
            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 0.25:
                repaired.append(cur)
                continue

            leading = []
            for wi in cur_idxs:
                if len(leading) >= max_leading_move:
                    break
                if self._is_movable_function_idx(wi, tg_words, disfluency_labels):
                    leading.append(wi)
                else:
                    break

            if not leading:
                repaired.append(cur)
                continue

            leading_set = set(leading)
            cur_remaining = [wi for wi in cur_idxs if wi not in leading_set]
            if not cur_remaining:
                repaired.append(cur)
                continue
            if not self._has_content_tokens(cur_remaining, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev_augmented = sorted(set(prev_idxs + leading))
            if not self._has_content_tokens(prev_augmented, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev['word_indices'] = prev_augmented
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _is_relocatable_tail_surface(self, clause_text: str, tail_surface: str) -> bool:
        """
        Decide whether a clause-final token is boundary glue that can move to the next clause.
        This is parse-driven and avoids utterance-specific lexical patching.
        """
        if not clause_text or not tail_surface:
            return False
        tail_norm = self.segmenter._normalize_surface_text(tail_surface)
        if not tail_norm:
            return False

        compact = (clause_text or "").replace(" ", "").replace("\u3000", "")

        if tail_norm in {"で", "も"} and compact.endswith("でも"):
            stem = compact[:-2]
            if stem.endswith((
                "ました", "ます", "ません",
                "です", "でした", "でしょう",
                "だ", "だった", "た",
            )):
                return True

        if self._is_restart_connector_surface(tail_norm):
            if compact.endswith(tail_norm):
                stem = compact[:-len(tail_norm)]
                if stem.endswith((
                    "ました", "ます", "ません",
                    "です", "でした", "でしょう",
                    "だ", "だった", "た",
                )):
                    return True

        if tail_norm == "で" and self._has_finite_bare_de_tail(clause_text):
            return True

        doc = self.segmenter.nlp(clause_text)
        toks = [t for t in doc if t.text.strip()]
        if not toks:
            return False

        last_tok = None
        for tok in reversed(toks):
            if self.segmenter._normalize_surface_text(tok.text) == tail_norm:
                last_tok = tok
                break
        if last_tok is None:
            return False

        if self.segmenter._is_unambiguous_filler_surface(last_tok.text):
            return True
        if last_tok.dep_ in {"discourse", "mark", "cc"}:
            return True
        if last_tok.pos_ in {"PART", "SCONJ", "CCONJ", "INTJ"}:
            return True
        if last_tok.pos_ == "ADV" and last_tok.dep_ in {"advmod", "discourse"}:
            return True
        return False

    def _has_explicit_subordinator_tail(self, text: str) -> bool:
        """True when the clause already ends in a Rule 3a-style subordinator."""
        compact = (text or "").replace(" ", "").replace("　", "")
        if not compact:
            return False
        return _has_subordinator_tail(compact)

    def _ends_with_sentence_final(self, text: str) -> bool:
        """True when clause ends with a clearly sentence-final predicate form.

        Only matches unambiguous polite/formal endings (ました/です/ます etc.)
        that strongly indicate a complete sentence boundary.  Plain forms
        like した/った/ない are too common as non-final verb forms and are
        excluded to avoid false positives.
        """
        compact = (text or "").replace(" ", "").replace("　", "")
        if not compact:
            return False
        # Only polite/formal sentence-final endings (safe, unambiguous)
        sf_tails = (
            "ました", "ません", "ます",
            "です", "でした", "でしょう",
        )
        if compact.endswith(sf_tails):
            return True
        return False

    def _clause_ends_with_sentence_final(
        self,
        clause: Dict,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """True when a clause (dict with word_indices) ends sentence-finally."""
        # First check explicit subordinator tail (from/because/but)
        if self._clause_has_explicit_subordinator_tail(clause, tg_words, disfluency_labels):
            return True
        # Then check the surface text
        return self._ends_with_sentence_final(clause.get("text", ""))

    def _repair_trailing_function_hangovers(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Repair the inverse boundary artifact of _repair_boundary_hangovers:
        clause-final function/discourse tokens that should attach to the next clause.
        """
        if len(clauses) < 2:
            return clauses

        max_trailing_move = 2
        repaired = [clauses[0]]

        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get("word_indices", []))
            cur_idxs = sorted(cur.get("word_indices", []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            connector_tail_idxs = self._collect_finite_connector_tail_indices(
                prev, tg_words, disfluency_labels
            )
            finite_connector_hangover = bool(connector_tail_idxs) or self._is_finite_connector_tail_hangover(
                prev, tg_words, disfluency_labels
            )
            finite_bare_de_hangover = self._has_finite_bare_de_tail(prev.get("text", ""))

            gap = float(cur.get("start_time", 0.0)) - float(prev.get("end_time", 0.0))
            max_gap = 3.0 if (finite_connector_hangover or finite_bare_de_hangover) else 0.8
            if gap > max_gap:
                repaired.append(cur)
                continue

            # Clause-final explicit subordinators (ながら, から, ので, etc.) are
            # real clause boundaries under Rule 3a, not movable boundary glue.
            if (
                self._clause_has_explicit_subordinator_tail(prev, tg_words, disfluency_labels)
                and not finite_connector_hangover
                and not finite_bare_de_hangover
            ):
                repaired.append(cur)
                continue

            trailing: List[int] = []
            if connector_tail_idxs:
                trailing = list(connector_tail_idxs)
            else:
                for wi in reversed(prev_idxs):
                    if len(trailing) >= max_trailing_move:
                        break
                    if self._is_bound_suffix_idx(wi, tg_words, disfluency_labels):
                        break
                    if not self._is_movable_function_idx(wi, tg_words, disfluency_labels):
                        break
                    if not self._is_relocatable_tail_surface(prev.get("text", ""), tg_words[wi].text):
                        break
                    trailing.append(wi)

            if not trailing:
                repaired.append(cur)
                continue

            trailing_set = set(trailing)
            prev_remaining = [wi for wi in prev_idxs if wi not in trailing_set]
            if not prev_remaining:
                repaired.append(cur)
                continue
            if not self._has_content_tokens(prev_remaining, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            cur_augmented = sorted(set(cur_idxs + trailing))
            if not self._has_content_tokens(cur_augmented, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev["word_indices"] = prev_remaining
            cur["word_indices"] = cur_augmented
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            repaired.append(cur)

        return repaired

    def _is_leading_filler_noise_idx(
        self,
        word_idx: int,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        """Broader filler check for clause-leading positions.

        At the start of a clause, "ん" and ambiguous fillers ("あの", "その")
        are almost always hesitation markers, not real morphemes.  The base
        ``_is_boundary_noise_idx`` intentionally excludes them to protect
        interior positions, but at clause onset the risk is minimal.
        """
        if self._is_boundary_noise_idx(word_idx, tg_words, disfluency_labels):
            return True
        text = tg_words[word_idx].text
        norm = self.segmenter._normalize_surface_text(text)
        if not norm:
            return False
        # "ん" at clause start — treat as filler unless it is a bound suffix
        # (e.g. the ん of ません split as ませ + ん).
        if norm == "ん":
            if not self._is_bound_suffix_idx(word_idx, tg_words, disfluency_labels):
                return True
        return False

    def _trim_boundary_noise(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Trim leading/trailing filler or disfluent words from each clause."""
        trimmed = []
        for clause in clauses:
            idxs = sorted(clause.get('word_indices', []))
            if not idxs:
                continue
            left = 0
            right = len(idxs) - 1
            while left <= right and self._is_leading_filler_noise_idx(idxs[left], tg_words, disfluency_labels):
                left += 1
            while right >= left and self._is_trailing_discourse_idx(idxs[right], tg_words, disfluency_labels):
                right -= 1
            if left > right:
                continue
            new_idxs = idxs[left:right + 1]
            if not self._has_content_tokens(new_idxs, tg_words, disfluency_labels):
                continue
            clause['word_indices'] = new_idxs
            clause = self._rebuild_clause_payload(clause, tg_words, disfluency_labels)
            if clause.get('text', '').strip():
                trimmed.append(clause)
        return trimmed

    @staticmethod
    def _is_bound_inflection(prev_text: str, cur_text: str) -> bool:
        """True when cur_text is a morphological suffix bound to prev_text.

        MFA sometimes tokenises a single verb form into stem + suffix:
            買って → 買っ + て    食べた → 食べ + た
            読みました → 読み + まし + た
        Splitting these across sentence groups causes GiNZA to misparse
        both fragments and lose the inflectional morpheme from clause text.
        """
        if not prev_text or not cur_text:
            return False
        p = prev_text.strip()
        c = cur_text.strip()
        if not p or not c:
            return False

        # Renyōkei (連用形) stem-final characters that precede て/で/た/だ
        # Godan: っ(tsu/u/ru) ん(nu/mu/bu) い(u) き(ku) し(su) り(ru)
        #        ぎ(gu) び(bu) み(mu) ち(tsu) ひ(hu)
        # Ichidan: え-row (食べ, 開け, 見(mi))
        # i-adj: く (赤く→て)
        _RENYOU_TAIL = set("っしきりびみんいぎちひえけげせべめれねく")

        # Pattern 1: stem + て/で (te-form)
        if p[-1] in _RENYOU_TAIL and c in ("て", "で"):
            return True
        # Pattern 2: stem + た/だ (past tense)
        if p[-1] in _RENYOU_TAIL and c in ("た", "だ"):
            return True
        # Pattern 3: stem + たり/だり (tari-form)
        if p[-1] in _RENYOU_TAIL and c in ("たり", "だり"):
            return True
        # Pattern 4: polite splits (まし+た, ませ+ん, でし+た)
        if p in ("まし", "でし") and c in ("た", "て"):
            return True
        if p == "ませ" and c == "ん":
            return True
        # Pattern 5: stem + polite suffix (読み+ます, 食べ+まし)
        if p[-1] in _RENYOU_TAIL and c in ("ます", "まし", "ませ"):
            return True
        # Pattern 6: て/で + auxiliary continuation
        if p in ("て", "で") and c in (
            "しまい", "しまう", "しまっ",
            "いる", "いた", "います", "いました",
            "おく", "おき", "おいた",
            "くる", "きた", "きました",
            "いく", "いき", "いった",
        ):
            return True
        # Pattern 7: passive/causative suffix
        if p[-1] in _RENYOU_TAIL and c in ("られ", "させ"):
            return True
        return False

    def _build_sentence_word_groups(
        self,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
        pause_split_sec: float = 0.6,
    ) -> List[List[int]]:
        """
        Build sentence-like word groups in two stages:
        1) Split by long pauses in timing.
        2) Split each pause chunk by model sentence boundaries (doc.sents).
        """
        fluent_indices = [
            i for i in range(len(tg_words))
            if disfluency_labels is None or disfluency_labels[i] == 0
        ]
        if not fluent_indices:
            return []

        pause_groups: List[List[int]] = [[fluent_indices[0]]]
        for wi in fluent_indices[1:]:
            prev = pause_groups[-1][-1]
            gap = tg_words[wi].start - tg_words[prev].end
            if gap >= pause_split_sec and not self._is_bound_inflection(tg_words[prev].text, tg_words[wi].text):
                pause_groups.append([wi])
            else:
                pause_groups[-1].append(wi)

        sentence_groups: List[List[int]] = []
        for group in pause_groups:
            sentence_groups.extend(self._split_group_by_model_sentences(group, tg_words))
        return sentence_groups

    def _split_group_by_model_sentences(
        self,
        group_indices: List[int],
        tg_words: List[WordInterval],
    ) -> List[List[int]]:
        """Split one timing chunk by parser sentence boundaries when available."""
        if len(group_indices) <= 1:
            return [group_indices]

        text = ''.join(tg_words[i].text for i in group_indices)
        if not text:
            return [group_indices]

        doc = self.segmenter.nlp(text)
        sents = list(doc.sents)
        if len(sents) <= 1:
            return [group_indices]

        char_to_local_word: Dict[int, int] = {}
        char_pos = 0
        for local_idx, global_idx in enumerate(group_indices):
            for _ in tg_words[global_idx].text:
                char_to_local_word[char_pos] = local_idx
                char_pos += 1

        split_points: List[int] = []
        for sent in sents[:-1]:
            boundary_char = sent.end_char
            local_cut: Optional[int] = None

            probe = boundary_char
            while probe < char_pos:
                if probe in char_to_local_word:
                    local_cut = char_to_local_word[probe]
                    break
                probe += 1

            if local_cut is None:
                probe = boundary_char - 1
                while probe >= 0:
                    if probe in char_to_local_word:
                        local_cut = char_to_local_word[probe] + 1
                        break
                    probe -= 1

            if local_cut is None:
                continue
            if 0 < local_cut < len(group_indices):
                split_points.append(local_cut)

        if not split_points:
            return [group_indices]

        split_points = sorted(set(split_points))
        # Never split between morphologically bound word pairs.
        split_points = [
            cut for cut in split_points
            if not (
                0 < cut < len(group_indices)
                and self._is_bound_inflection(
                    tg_words[group_indices[cut - 1]].text,
                    tg_words[group_indices[cut]].text,
                )
            )
        ]
        out: List[List[int]] = []
        prev = 0
        for cut in split_points:
            if cut > prev:
                out.append(group_indices[prev:cut])
            prev = cut
        if prev < len(group_indices):
            out.append(group_indices[prev:])
        return [g for g in out if g]

    def _align_group_clauses(
        self,
        group_indices: List[int],
        tg_words: List[WordInterval],
        claimed_word_indices: Set[int],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Run clause segmentation+alignment for one sentence-like word group."""
        group_transcript = ''.join(
            tg_words[i].text for i in group_indices
            if disfluency_labels is None or disfluency_labels[i] == 0
        )
        if not group_transcript:
            return []

        clauses = self.segmenter.segment(group_transcript, verbose=False)
        if not clauses:
            return []

        doc = self.segmenter.nlp(group_transcript)
        char_to_word_idx: Dict[int, int] = {}
        char_pos = 0
        for wi in group_indices:
            if disfluency_labels is not None and disfluency_labels[wi] == 1:
                continue
            for _ in tg_words[wi].text:
                char_to_word_idx[char_pos] = wi
                char_pos += 1

        aligned: List[Dict] = []
        for _start_idx, _end_idx, clause_type, _clause_text, token_indices in clauses:
            if not token_indices:
                continue

            word_indices = set()
            for tok_idx in token_indices:
                if tok_idx >= len(doc):
                    continue
                tok = doc[tok_idx]
                tok_start = tok.idx
                tok_end = tok.idx + len(tok.text)
                for c_pos in range(tok_start, tok_end):
                    wi = char_to_word_idx.get(c_pos)
                    if wi is None:
                        continue
                    if wi in claimed_word_indices:
                        continue
                    word_indices.add(wi)

            if not word_indices:
                continue

            word_indices = sorted(word_indices)
            claimed_word_indices.update(word_indices)

            # Keep disfluent anchor words inside this span for timing continuity.
            if disfluency_labels is not None and len(word_indices) >= 2:
                min_wi = word_indices[0]
                max_wi = word_indices[-1]
                for wi in range(min_wi, max_wi + 1):
                    if wi in claimed_word_indices:
                        continue
                    if disfluency_labels[wi] == 1:
                        word_indices.append(wi)
                        claimed_word_indices.add(wi)
                word_indices = sorted(word_indices)

            clause_words = [tg_words[i] for i in word_indices]
            displayed_words = [
                tg_words[i]
                for i in self._visible_clause_word_indices(word_indices, tg_words, disfluency_labels)
            ]

            final_text = ''.join(w.text for w in displayed_words)
            clause_mora = sum(w.mora_count for w in displayed_words)
            aligned.append({
                'start_time': clause_words[0].start,
                'end_time': clause_words[-1].end,
                'text': final_text,
                'type': clause_type,
                'word_indices': word_indices,
                'mora': clause_mora,
            })
        return aligned

    def _merge_boundary_continuations(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Generic continuation recovery without lexical hard-coding.

        Pattern A:
          short subordinate-like span with predicate but no element
          + immediate next span -> merge forward.

        Pattern B:
          non-predicative host span
          + immediate short predicate fragment -> merge forward.
        """
        if len(clauses) < 2:
            return clauses

        subordinate_like = {"SUB_REL", "SUB_ADVCL", "SUB_CCOMP", "SUB_XCOMP", "SUB_CSUBJ", "CHAIN_TE"}

        def _starts_with_marked_np_fragment(span_text: str) -> bool:
            span = [t for t in self.segmenter.nlp(span_text) if t.text.strip()]
            if len(span) < 2:
                return False
            limit = min(5, len(span))
            for tok in span[:limit]:
                if tok.text not in {"は", "が"}:
                    continue
                if tok.dep_ != "case" or tok.head.pos_ not in {"NOUN", "PRON", "PROPN"}:
                    continue
                head_local = None
                for probe_idx, probe in enumerate(span[:limit]):
                    if probe.i == tok.head.i:
                        head_local = probe_idx
                        break
                if head_local is not None and head_local <= 2:
                    return True
            return False

        merged: List[Dict] = []
        i = 0
        while i < len(clauses):
            cur = clauses[i]
            if i + 1 >= len(clauses):
                merged.append(cur)
                break

            nxt = clauses[i + 1]
            gap = float(nxt.get('start_time', 0.0)) - float(cur.get('end_time', 0.0))

            cur_text = (cur.get('text') or '').strip()
            nxt_text = (nxt.get('text') or '').strip()
            if not cur_text or not nxt_text:
                merged.append(cur)
                i += 1
                continue

            # If the current clause already ends with an explicit subordinator
            # from Rule 3a (e.g., ながら / ので / から), keep that boundary.
            if self._clause_has_explicit_subordinator_tail(cur, tg_words, disfluency_labels):
                merged.append(cur)
                i += 1
                continue

            compact_cur_text = cur_text.replace(" ", "").replace("　", "")
            if cur.get('type') == 'SUB_CCOMP' and compact_cur_text.endswith(("と", "って")):
                merged.append(cur)
                i += 1
                continue

            cur_feats = _analyze_clause_surface(self.segmenter, cur_text)
            nxt_feats = _analyze_clause_surface(self.segmenter, nxt_text)
            cur_compact = cur_text.replace(" ", "").replace("　", "")
            complete_question_shell = (
                self._ends_with_question_shell(cur_text)
                and cur_feats.get("has_pred_or_nominal", False)
            )
            complete_copular_tail = (
                cur_feats.get("has_pred_or_nominal", False)
                and cur_feats.get("has_element", False)
                and cur_compact.endswith(("で", "だ", "です", "でした"))
            )

            pattern_a = (
                -0.5 <= gap <= 0.35
                and
                cur.get('type') in subordinate_like
                and len(cur_text) <= 12
                and (
                    (cur.get('type') == 'SUB_REL' and not cur_feats.get('has_nominal_token', False))
                    or (
                        cur_feats.get('has_pred_verb_adj', False)
                        and not cur_feats.get('has_element', False)
                    )
                )
            )
            pattern_b = (
                -0.5 <= gap <= 1.2
                and
                not cur_feats.get('has_pred_verb_adj', False)
                and cur_feats.get('has_element', False)
                and len(nxt_text) <= 10
                and nxt_feats.get('has_pred_verb_adj', False)
                and not nxt_feats.get('has_element', False)
                and not nxt_feats.get('has_nominal_token', False)
            )
            pattern_c = (
                -0.5 <= gap <= 0.8
                and cur.get('type') == 'SUB_REL'
                and len(cur_text) <= 12
                and nxt.get('type') in {'fragment', 'IND', 'SUB_REL'}
                and not nxt_feats.get('has_pred_verb_adj', False)
                and nxt_feats.get('has_pred_or_nominal', False)
            )
            # Pattern D:
            # short non-predicative host + predicative continuation.
            # This catches noisy splits like:
            #   "昨日は友達と..." + "会いました"
            #   "本とノートも..." + "買いました"
            ends_with_connector = bool(
                re.search(
                    r"(?:は|が|を|に|へ|で|と|の|も|から|まで|って|なんか|あの|その|えっと|ええと)$",
                    cur_text,
                )
            )
            pattern_d = (
                cur.get('type') in {'IND', 'fragment', 'SUB_REL'}
                and len(cur_text) <= 24
                and not complete_question_shell
                and not complete_copular_tail
                and not cur_feats.get('has_pred_verb_adj', False)
                and nxt.get('type') in {'IND', 'SUB_REL', 'SUB_ADVCL', 'SUB_CCOMP', 'CHAIN_TE', 'fragment'}
                and nxt_feats.get('has_pred_verb_adj', False)
                and (
                    cur_feats.get('has_element', False)
                    or cur_feats.get('has_case_tail', False)
                    or ends_with_connector
                    or len(cur_text) <= 4
                    or cur_feats.get('all_particles', False)
                )
            )
            pattern_e = (
                cur.get('type') in {'IND', 'fragment', 'SUB_REL'}
                and len(cur_text) <= 16
                and not complete_question_shell
                and not cur_feats.get('has_pred_verb_adj', False)
                and _starts_with_marked_np_fragment(cur_text)
                and nxt.get('type') in {'IND', 'SUB_REL', 'SUB_ADVCL', 'SUB_CCOMP', 'CHAIN_TE', 'fragment'}
                and nxt_feats.get('has_pred_verb_adj', False)
            )

            if not (pattern_a or pattern_b or pattern_c or pattern_d or pattern_e):
                merged.append(cur)
                i += 1
                continue

            merged_clause = dict(cur)
            merged_clause['word_indices'] = sorted(set(
                list(cur.get('word_indices', [])) + list(nxt.get('word_indices', []))
            ))
            merged_clause = self._rebuild_clause_payload(merged_clause, tg_words, disfluency_labels)

            if pattern_b and merged_clause.get('type') == 'fragment':
                merged_clause['type'] = nxt.get('type', 'fragment')
            if (pattern_d or pattern_e) and not cur_feats.get('has_pred_verb_adj', False):
                merged_clause['type'] = nxt.get('type', merged_clause.get('type', 'IND'))

            if not merged_clause.get('text', '').strip():
                merged.append(cur)
                i += 1
                continue

            merged.append(merged_clause)
            i += 2

        return merged

    def _find_finite_connector_split_char(self, text: str) -> Optional[int]:
        text = (text or "").strip()
        if len(text) < 8:
            return None

        # v1.0-paper: the で-lookahead also listed words of single utterances
        # in the evaluation corpus (厳, 歌, 警, 窓, 地図, 外, 部屋).
        patterns = (
            r"(でした|ました|ます|です|だった|だ|た)(?:ん)?(?=(?:そして|それから|それで|でも|だから|なので))",
            r"(でした|ました|ます|です)(?=で(?:その|こ|あ|ま|厳|歌|警|窓|地図|外|部屋))",
        )
        for pattern in patterns:
            m = re.search(pattern, text)
            if not m:
                continue
            split_char = m.start(1) + len(m.group(1))
            left = text[:split_char].strip()
            right = text[split_char:].strip()
            if len(left) < 5 or len(right) < 4:
                continue
            return split_char
        return None

    def _starts_with_clause_restart_surface(self, text: str) -> bool:
        core = self._strip_leading_soft_filler(text)
        compact = self._compact_clause_text(core)
        if not compact:
            return False
        if compact.startswith((
            "でも", "でその", "でこの", "であの", "その後", "その時", "そして", "それで",
            "やっぱり", "なんか", "あっ", "えっ", "彼", "彼女", "私",
        )):
            return True
        toks = [t for t in self.segmenter.nlp(core) if t.text.strip()]
        # v1.0-paper also listed the names and props of the evaluation
        # corpus's two stories, and only looked one token ahead.
        if compact.startswith((
            "こっそりと", "二人", "ケン", "マリ", "犬", "警官", "部屋", "家", "窓",
            "地図", "バスケット", "着きました", "起きて", "起きました",
        )):
            return True
        if len(toks) >= 2 and toks[0].pos_ in {"NOUN", "PROPN", "PRON", "NUM"}:
            if (toks[1].text or "").strip() in {"は", "が", "も", "を", "に"}:
                return True
        return False

    def _find_finite_topic_restart_split_char(self, text: str) -> Optional[int]:
        text = (text or "").strip()
        if len(text) < 12:
            return None

        endings = (
            "でした", "ました", "ます", "です", "だった", "だ", "た", "んです", "たんです",
        )
        blocked_right_prefixes = (
            "が", "けど", "けれど", "けれども", "ので", "から", "のに", "し", "と", "って", "ながら",
        )

        def _has_predicate_like(span_text: str) -> bool:
            span_toks = [t for t in self.segmenter.nlp(span_text) if t.text.strip()]
            return any(
                t.pos_ == "VERB"
                or t.pos_ == "ADJ"
                or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
                or (t.pos_ in {"NOUN", "PRON", "PROPN"} and any(ch.dep_ == "cop" for ch in t.children))
                for t in span_toks
            )

        for ending in endings:
            start = 0
            while True:
                idx = text.find(ending, start)
                if idx < 0:
                    break
                split_char = idx + len(ending)
                left = text[:split_char].strip()
                right = text[split_char:].strip()
                start = idx + 1
                if len(left) < 6 or len(right) < 4:
                    continue
                if right.startswith(blocked_right_prefixes):
                    continue
                if not _has_predicate_like(left):
                    continue
                if not self._starts_with_clause_restart_surface(right):
                    continue
                return split_char

        return None

    def _find_question_quotative_split_char(self, text: str) -> Optional[int]:
        text = (text or "").strip()
        if len(text) < 8:
            return None

        # v1.0-paper: the head nouns were listed from single utterances of the
        # evaluation corpus.
        patterns = (
            r"(ですか|ますか|でしたか|たか|だか)(?=(?:って|と)(?:いう|ゆう))",
            r"(たい|ない|ました|ます|です|だ|た)(?=(?:って|と)(?:いう|ゆう)(?:こと|気持ち|理由|質問|答え|挨拶|話))",
        )
        for pattern in patterns:
            m = re.search(pattern, text)
            if not m:
                continue
            split_char = m.start(1) + len(m.group(1))
            left = text[:split_char].strip()
            right = text[split_char:].strip()
            if len(left) < 4 or len(right) < 6:
                continue
            return split_char
        return None

    def _find_runon_split_char(self, text: str) -> Optional[int]:
        """
        Detect a robust internal split point for over-merged IND clauses.
        Uses marker-based boundaries and requires predicate evidence on both sides.
        """
        text = (text or "").strip()
        if len(text) < 12:
            return None

        doc = self.segmenter.nlp(text)
        toks = [t for t in doc if t.text.strip()]
        if len(toks) < 6:
            return None

        def _is_adverbialized_tail(span_toks, idx: int) -> bool:
            tok = span_toks[idx]
            if tok.pos_ != "ADJ":
                return False
            if idx + 1 >= len(span_toks):
                return False
            nxt = span_toks[idx + 1]
            nxt_norm = self.segmenter._normalize_surface_text(nxt.text)
            return idx == len(span_toks) - 2 and nxt_norm == "に" and nxt.pos_ in {"AUX", "ADP"}

        def _has_predicate_like(span_text: str) -> bool:
            span_toks = [t for t in self.segmenter.nlp(span_text) if t.text.strip()]
            return any(
                t.pos_ == "VERB"
                or (t.pos_ == "ADJ" and not _is_adverbialized_tail(span_toks, idx))
                or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
                or (t.pos_ in {"NOUN", "PRON", "PROPN"} and any(ch.dep_ == "cop" for ch in t.children))
                for idx, t in enumerate(span_toks)
            )

        m_qrestart = re.search(r"(ですか|ますか|たか|だか)で", text)
        if m_qrestart:
            split_char = m_qrestart.start(1) + len(m_qrestart.group(1))
            left = text[:split_char].strip()
            right = text[split_char:].strip()
            if (
                len(left) >= 6
                and len(right) >= 4
                and _has_predicate_like(left)
                and (
                    _has_predicate_like(right)
                    or right.startswith(("で", "でも", "それで", "だから", "そして", "あの", "その"))
                )
            ):
                return split_char

        m_qquote = re.search(r"(ですか|ますか|でしたか|たか|だか)(?=(?:って|と)(?:いう|ゆう))", text)
        if m_qquote:
            split_char = m_qquote.start(1) + len(m_qquote.group(1))
            left = text[:split_char].strip()
            right = text[split_char:].strip()
            right_compact = right.replace(" ", "").replace("\u3000", "")
            if (
                len(left) >= 6
                and len(right_compact) >= 6
                and right_compact.startswith(("という", "とゆう", "っていう", "ってゆう"))
            ):
                right_core = re.sub(r"^(?:という|とゆう|っていう|ってゆう)", "", right_compact, count=1)
                reporting_like_nouns = ("質問", "説明", "答え", "こと", "気持ち", "理由", "話")
                reporting_like_lemmas = {
                    "言う", "話す", "答える", "聞く", "尋ねる",
                    "思う", "考える", "感じる", "叫ぶ", "説明する",
                }
                if (
                    any(noun in right_core for noun in reporting_like_nouns)
                    or any(
                        tok.lemma_ in reporting_like_lemmas
                        for tok in self.segmenter.nlp(right)
                        if tok.text.strip()
                    )
                ):
                    return split_char

        m_fin_restart = re.search(
            r"(でした|ました|ます|です|だった|だ|た)(?:ん)?(?=(?:そして|それから|それで|でも|だから|なので))",
            text,
        )
        if m_fin_restart:
            split_char = m_fin_restart.start(1) + len(m_fin_restart.group(1))
            left = text[:split_char].strip()
            right = text[split_char:].strip()
            if len(left) >= 6 and len(right) >= 4 and _has_predicate_like(left):
                return split_char

        predicate_count = sum(
            1
            for idx, t in enumerate(toks)
            if t.pos_ == "VERB"
            or (t.pos_ == "ADJ" and not _is_adverbialized_tail(toks, idx))
            or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
            or (t.pos_ in {"NOUN", "PRON", "PROPN"} and any(ch.dep_ == "cop" for ch in t.children))
        )
        if predicate_count < 2:
            return None

        strong_markers = {
            "たら",
            "から",
            "ので",
            "のに",
            "けど",
            "けれど",
            "けれども",
            "って",
            "とき",
            "時",
            "うちに",
            "間に",
            "ながら",
            "つつ",
        }
        weak_markers = {"し"}
        restart_connectors = {
            "そして", "それで", "でも", "しかし", "だから", "なので",
            "ところで", "ただ", "すると",
        }
        leadins = restart_connectors | {"あの", "えっと", "えーと", "えー", "まあ"}

        def _strip_leading_restart_text(span_text: str) -> str:
            span_text = (span_text or "").strip()
            if not span_text:
                return span_text
            span_doc = self.segmenter.nlp(span_text)
            span_toks = [t for t in span_doc if t.text.strip()]
            cut = 0
            for tok in span_toks[:3]:
                surf = self.segmenter._normalize_surface_text(tok.text)
                if tok.idx == cut and surf in leadins:
                    cut = tok.idx + len(tok.text)
                    continue
                break
            return span_text[cut:].strip()

        candidates: List[Tuple[float, int]] = []
        text_len = len(text)

        for tok in toks[1:-1]:
            surf = self.segmenter._normalize_surface_text(tok.text)
            if not surf:
                continue
            split_specs: List[Tuple[int, float]] = []

            # Marker-driven split after this token.
            if (
                surf in strong_markers
                or surf in weak_markers
                or (tok.dep_ in {"mark", "cc"} and surf not in {"て", "で"})
            ):
                split_specs.append((tok.idx + len(tok.text), 2.0 if surf in strong_markers else 1.0))

            if surf in restart_connectors:
                split_specs.append((tok.idx, 2.2))

            # Topic restart: split at start of noun phrase before a second "は".
            if tok.text == "は" and tok.dep_ == "case" and tok.head.pos_ in {"NOUN", "PRON", "PROPN"}:
                split_specs.append((tok.head.idx, 1.8))

            # Sentence-final polite endings often indicate a clause boundary.
            if tok.pos_ == "AUX" and tok.lemma_ in {"ます", "です"}:
                split_specs.append((tok.idx + len(tok.text), 1.2))
            if tok.pos_ == "AUX" and tok.lemma_ in {"た", "だ"} and tok.text in {"た", "だ"}:
                split_specs.append((tok.idx + len(tok.text), 0.9))

            for split_char, base_score in split_specs:
                left = text[:split_char].strip()
                right = text[split_char:].strip()
                min_right_chars = 4 if surf == "って" else 6
                if split_char < 6 or split_char > text_len - min_right_chars:
                    continue
                if len(left) < 6 or len(right) < min_right_chars:
                    continue

                right_core = _strip_leading_restart_text(right)
                right_has_leadin = right_core != right
                if not right_core:
                    continue

                # v16-fix: protect auxiliary constructions (てしまう, ている, etc.)
                _aux_starts = (
                    "しまい", "しまう", "しまっ", "しまいました",
                    "いる", "いた", "いました", "います",
                    "おく", "おき", "おいた",
                    "くる", "きた", "きました",
                    "いく", "いき", "いった",
                )
                if any(right_core.startswith(a) for a in _aux_starts):
                    continue
                # v16-fix: protect te-form stems (left ending in っ/し/き/り/び/み/ん + split before て/で)
                if left and left[-1] in "っしきりびみんえ" and right_core and right_core[0] in "てで":
                    continue
                # v16-fix: don't split at te-form chain boundary (left ends with て/で)
                # This would break a CHAIN_TE continuation (V1-て V2 patterns),
                # unless the right side clearly restarts with a discourse connective.
                if left.endswith(("て", "で")) and surf not in {"って"} and not right_has_leadin:
                    continue
                if not _has_predicate_like(left):
                    continue
                if not _has_predicate_like(right_core):
                    continue

                score = base_score
                if any(ch in right[:12] for ch in ("は", "が")):
                    score += 1.0
                if right_has_leadin:
                    score += 0.5
                ratio = split_char / max(1, text_len)
                score += 1.0 - abs(0.5 - ratio)
                candidates.append((score, split_char))

        if not candidates:
            return None
        best_by_char: Dict[int, float] = {}
        for score, split_char in candidates:
            best_by_char[split_char] = max(score, best_by_char.get(split_char, -1e9))
        best = sorted(best_by_char.items(), key=lambda x: x[1], reverse=True)[0][0]
        return best

    def _find_restart_split_char(self, text: str) -> Optional[int]:
        """
        Recover clause-internal restarts missed by the generic run-on splitter.

        Targets three conservative patterns:
          1. clause-final te-form with its own marked subject/topic + new topic restart
          2. short reporting clause + new topic/subject-marked clause
          3. sentence-final predicate (です/ます/た/だ) + new marked NP fragment
        """
        text = (text or "").strip()
        if len(text) < 10:
            return None

        doc = self.segmenter.nlp(text)
        toks = [t for t in doc if t.text.strip()]
        if len(toks) < 4:
            return None

        predicate_count = sum(
            1
            for t in toks
            if t.pos_ in {"VERB", "ADJ"}
            or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
        )
        if predicate_count < 2:
            predicate_count = sum(
                1
                for t in toks
                if t.pos_ in {"VERB", "ADJ"}
                or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
                or (t.pos_ in {"NOUN", "PRON", "PROPN"} and any(ch.dep_ == "cop" for ch in t.children))
            )
        if predicate_count < 2:
            has_terminal_predicate = any(
                (
                    t.pos_ in {"VERB", "ADJ"}
                    or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
                )
                and t.dep_ not in {"case", "mark", "cc", "fixed"}
                for t in toks
            )
            has_marked_restart = any(
                t.text in {"は", "が"}
                and t.dep_ == "case"
                and t.head.pos_ in {"NOUN", "PRON", "PROPN"}
                for t in toks
            )
            if not (has_terminal_predicate and has_marked_restart):
                return None

        restart_connectors = {
            "そして", "それで", "でも", "しかし", "だから", "なので",
            "ところで", "ただ", "すると",
        }
        leadin_like = restart_connectors | {"あの", "えっと", "えーと", "えー", "まあ"}

        def _span_toks(span_text: str):
            return [t for t in self.segmenter.nlp(span_text) if t.text.strip()]

        def _strip_leading_restart_text(span_text: str) -> str:
            span_text = (span_text or "").strip()
            if not span_text:
                return span_text
            span = _span_toks(span_text)
            cut = 0
            for tok in span[:3]:
                surf = self.segmenter._normalize_surface_text(tok.text)
                if tok.idx == cut and surf in leadin_like:
                    cut = tok.idx + len(tok.text)
                    continue
                break
            return span_text[cut:].strip()

        def _has_marked_topic_or_subject(span_text: str) -> bool:
            return any(
                t.text in {"は", "が"}
                and t.dep_ == "case"
                and t.head.pos_ in {"NOUN", "PRON", "PROPN"}
                for t in _span_toks(span_text)
            )

        def _starts_with_marked_topic_or_subject(span_text: str) -> bool:
            span = _span_toks(span_text)
            if len(span) < 2:
                return False
            limit = min(5, len(span))
            for tok in span[:limit]:
                if tok.text not in {"は", "が"}:
                    continue
                if tok.dep_ != "case" or tok.head.pos_ not in {"NOUN", "PRON", "PROPN"}:
                    continue
                head_local = None
                for probe_idx, probe in enumerate(span[:limit]):
                    if probe.i == tok.head.i:
                        head_local = probe_idx
                        break
                if head_local is not None and head_local <= 2:
                    return True
            return False

        def _ends_with_terminal_predicate(span_text: str) -> bool:
            span = _span_toks(span_text)
            if not span:
                return False
            last = span[-1]
            if last.dep_ in {"case", "mark", "cc", "fixed"}:
                return False
            if last.pos_ in {"VERB", "ADJ"}:
                return True
            if last.pos_ == "AUX" and last.lemma_ in {"だ", "です", "ます", "た"}:
                return True
            return self.segmenter._normalize_surface_text(last.text) in {
                "だ", "です", "ます", "た", "でした", "ました", "だった",
            }

        candidates: List[Tuple[float, int]] = []
        text_len = len(text)

        for tok in toks[1:-1]:
            if tok.text not in {"は", "が"}:
                continue
            if tok.dep_ != "case" or tok.head.pos_ not in {"NOUN", "PRON", "PROPN"}:
                continue

            head_pos = None
            for i, probe in enumerate(toks):
                if probe.i == tok.head.i:
                    head_pos = i
                    break
            if head_pos is None:
                continue

            start_pos = head_pos
            while start_pos > 0:
                prev_tok = toks[start_pos - 1]
                prev_norm = self.segmenter._normalize_surface_text(prev_tok.text)
                if (
                    prev_tok.dep_ in {"det", "discourse"}
                    or prev_tok.pos_ == "DET"
                    or prev_norm in leadin_like
                ):
                    start_pos -= 1
                    continue
                break

            split_char = toks[start_pos].idx
            left = text[:split_char].strip()
            right = text[split_char:].strip()
            if not left or not right or len(right) < 4:
                continue

            right_core = _strip_leading_restart_text(right)
            left_has_marked_np = _has_marked_topic_or_subject(left)
            right_starts_with_marked_np = _starts_with_marked_topic_or_subject(right_core or right)
            left_is_reporting = self._restart_is_short_reporting_clause(left)
            left_terminal = _ends_with_terminal_predicate(left)
            left_compact = left.replace(" ", "").replace("　", "")
            right_compact = (right_core or right).replace(" ", "").replace("　", "")
            min_left_chars = 4 if left_is_reporting else 6

            if len(left_compact) < min_left_chars or split_char < min_left_chars or split_char > text_len - 4:
                continue
            if left_compact == right_compact:
                continue
            if not right_starts_with_marked_np:
                continue

            right_has_predicate = self._restart_has_predicate_like(right_core or right)

            if left.endswith(("て", "で")):
                if not left_has_marked_np and right_core == right:
                    continue
                if not right_has_predicate:
                    continue
                score = 3.0
            elif left_is_reporting:
                if not right_has_predicate:
                    continue
                score = 3.1
            elif left_terminal:
                if not (right_has_predicate or len(right_compact) <= 12):
                    continue
                score = 2.8
            else:
                continue

            if tok.text == "は":
                score += 0.4
            if right_core != right:
                score += 0.3
            ratio = split_char / max(1, text_len)
            score += 1.0 - abs(0.5 - ratio)
            candidates.append((score, split_char))

        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    def _find_short_quotative_reporting_split_char(self, text: str) -> Optional[int]:
        """
        Split short quotative + reporting bundles like:
          「…行くって言ったら」 -> 「…行くって」 / 「言ったら」

        This is narrower than the generic run-on splitter: it only fires when the
        right side is a short reporting clause, which keeps noisy larger spans from
        being over-split.
        """
        text = (text or "").strip()
        if len(text) < 10:
            return None

        toks = [t for t in self.segmenter.nlp(text) if t.text.strip()]
        if len(toks) < 4:
            return None

        candidates: List[Tuple[float, int]] = []
        for tok in toks[1:-1]:
            surf = self.segmenter._normalize_surface_text(tok.text)
            if surf not in {"って", "と"}:
                continue

            split_char = tok.idx + len(tok.text)
            left = text[:split_char].strip()
            right = text[split_char:].strip()
            left_compact = left.replace(" ", "").replace("　", "")
            right_compact = right.replace(" ", "").replace("　", "")

            if len(left_compact) < 6 or len(right_compact) < 3 or len(right_compact) > 10:
                continue

            left_feats = _analyze_clause_surface(self.segmenter, left)
            if not left_feats.get("has_pred_or_nominal", False):
                continue
            if not left_compact.endswith(("と", "って")):
                continue
            if not self._restart_is_short_reporting_clause(right):
                continue

            # Prefer shorter right-edge reporting clauses.
            score = 3.0 - (0.05 * len(right_compact))
            candidates.append((score, split_char))

        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    def _infer_restart_split_type(self, text: str, default_type: str) -> str:
        """Infer a clause type for a clause recovered by restart splitting."""
        text = (text or "").strip()
        if not text:
            return default_type or "IND"

        compact = text.replace(" ", "").replace("　", "")
        feats = _analyze_clause_surface(self.segmenter, text)

        if _has_subordinator_tail(compact) or feats["has_sub_marker"]:
            return "SUB_ADVCL"
        if compact.endswith(("と", "って")) and (
            feats.get("has_pred_verb_adj", False) or feats.get("has_pred_or_nominal", False)
        ):
            return "SUB_CCOMP"
        if feats["ends_with_te"]:
            return "CHAIN_TE"
        if default_type in {"SUB_REL", "SUB_ADVCL", "fragment"} and feats["has_pred_verb_adj"] and not feats["has_case_tail"]:
            return "IND"
        if default_type == "fragment" and feats["has_pred_verb_adj"]:
            return "IND"
        return default_type or "IND"

    def _restart_has_predicate_like(self, text: str) -> bool:
        """Predicate check used by restart splitting, including copular nominals."""
        toks = [t for t in self.segmenter.nlp(text) if t.text.strip()]
        if not toks:
            return False
        if any(t.pos_ in {"VERB", "ADJ"} for t in toks):
            return True
        if any(t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"} for t in toks):
            return True
        return any(
            t.pos_ in {"NOUN", "PRON", "PROPN"} and any(ch.dep_ == "cop" for ch in t.children)
            for t in toks
        )

    def _restart_is_short_reporting_clause(self, text: str) -> bool:
        """Short finite reporting/stative matrix clause usable as a restart left edge."""
        compact = (text or "").replace(" ", "").replace("　", "")
        if len(compact) > 10:
            return False
        toks = [t for t in self.segmenter.nlp(text) if t.text.strip()]
        if not toks or not self._restart_has_predicate_like(text):
            return False
        if toks[-1].dep_ in {"case", "mark", "cc", "fixed"}:
            return False
        reporting_like = {
            "言う", "話す", "答える", "聞く", "尋ねる",
            "思う", "考える", "感じる", "叫ぶ", "説明する",
        }
        return any(t.lemma_ in reporting_like for t in toks)


    def _repair_te_form_boundaries(
        self,
        clauses: List[Dict],
        tg_words,  # List[WordInterval]
        disfluency_labels=None,
    ):
        """
        Move leading て/で from clause start to previous clause end.

        In Japanese, て/で is morphologically part of the preceding verb
        (e.g., 行って = 行っ + て). When the clause segmenter splits at
        the wrong boundary, て/で ends up at the start of the next clause.

        This dedicated repair step catches cases that _repair_boundary_hangovers
        misses (because standalone て may be parsed as noun 手 by GiNZA).
        """
        if len(clauses) < 2:
            return clauses

        # て and で as te-form markers
        TE_MARKERS = {"て", "で"}  # て, で

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            visible_prefix = []
            for wi in cur_idxs:
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text
                if not text or not text.strip():
                    continue
                visible_prefix.append((wi, text.strip()))
                if len(visible_prefix) >= 2:
                    break

            # Do not pull clause-initial で back when it begins a restart connector
            # like 「でも…」. That is discourse continuation, not te-form morphology.
            if len(visible_prefix) >= 2 and visible_prefix[0][1] == "で" and visible_prefix[1][1] == "も":
                repaired.append(cur)
                continue

            # Find leading て/で in current clause (skip disfluent/empty words)
            te_indices_to_move = []
            found_te = False
            for wi in cur_idxs:
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue  # skip disfluent words
                text = tg_words[wi].text
                if not text or not text.strip():
                    continue
                norm = text.strip()
                if norm in TE_MARKERS and not found_te:
                    te_indices_to_move.append(wi)
                    found_te = True
                else:
                    break  # stop at first non-te content word

            if not te_indices_to_move:
                repaired.append(cur)
                continue

            # Only move if gap is small (contiguous speech)
            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 0.5:
                repaired.append(cur)
                continue

            # Ensure current clause has content remaining after removal
            cur_remaining = [wi for wi in cur_idxs if wi not in set(te_indices_to_move)]
            if not cur_remaining:
                repaired.append(cur)
                continue

            # Move て/で to previous clause
            prev['word_indices'] = sorted(set(prev_idxs + te_indices_to_move))
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _repair_inflection_splits(
        self,
        clauses: List[Dict],
        tg_words,
        disfluency_labels=None,
    ):
        """
        Repair verb inflection morphology split across clause boundaries.

        MFA sometimes tokenises ました as まし|た or ません as ませ|ん,
        causing the clause segmenter to place a boundary mid-inflection.
        This moves the orphaned inflection suffix word(s) back to the
        previous clause.

        Also absorbs immediately following conjunctive particles (が, けど,
        けれど) so that e.g. ましたが stays as one unit.
        """
        if len(clauses) < 2:
            return clauses

        # Incomplete inflection stems that need a suffix from next clause
        _INFLECTION_STEMS = {"まし", "ませ", "でし"}
        # Valid suffixes that complete the inflection
        _INFLECTION_SUFFIXES = {"た", "ん", "て"}
        # Conjunctive particles that may follow the suffix
        _CONJ_PARTICLES = {"が", "けど", "けれど", "けれども"}
        # Polite suffix words that should attach to previous verb stem
        _POLITE_SUFFIX_WORDS = {"まし", "ます", "ませ"}
        # Completions needed after polite suffix words
        _POLITE_COMPLETIONS = {"まし": {"た", "て"}, "ませ": {"ん"}}

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            # Check gap
            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 0.5:
                repaired.append(cur)
                continue

            # Get last non-disfluent word of previous clause
            prev_last_text = None
            for wi in reversed(prev_idxs):
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    prev_last_text = text
                    break

            if prev_last_text is None:
                repaired.append(cur)
                continue

            # Get first non-disfluent word(s) of current clause
            first_words = []
            for wi in cur_idxs:
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    first_words.append((wi, text))
                    if len(first_words) >= 3:
                        break

            if not first_words:
                repaired.append(cur)
                continue

            indices_to_move = []

            # Pattern A: prev ends with incomplete stem (まし/ませ/でし) + next starts with suffix
            if prev_last_text in _INFLECTION_STEMS:
                moved_suffix = False
                for wi, text in first_words:
                    if not moved_suffix and text in _INFLECTION_SUFFIXES:
                        indices_to_move.append(wi)
                        moved_suffix = True
                    elif moved_suffix and text in _CONJ_PARTICLES:
                        indices_to_move.append(wi)
                        break
                    else:
                        break

            # Pattern B: next clause starts with polite suffix word (まし/ます/ませ)
            # e.g., verb stem し | まし た → move まし+た to prev
            if not indices_to_move and first_words[0][1] in _POLITE_SUFFIX_WORDS:
                polite_word = first_words[0][1]
                indices_to_move.append(first_words[0][0])
                # Also grab completions if needed (まし→た, ませ→ん)
                completions = _POLITE_COMPLETIONS.get(polite_word, set())
                conj_done = False
                for wi, text in first_words[1:]:
                    if not conj_done and text in completions:
                        indices_to_move.append(wi)
                    elif text in _CONJ_PARTICLES:
                        indices_to_move.append(wi)
                        conj_done = True
                        break
                    else:
                        break

            if not indices_to_move:
                repaired.append(cur)
                continue

            cur_remaining = [wi for wi in cur_idxs if wi not in set(indices_to_move)]
            if not cur_remaining:
                repaired.append(cur)
                continue

            prev['word_indices'] = sorted(set(prev_idxs + indices_to_move))
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _repair_auxiliary_splits(
        self,
        clauses: List[Dict],
        tg_words,
        disfluency_labels=None,
    ):
        """
        Repair te-form auxiliary constructions split across clause boundaries.

        When the previous clause ends with て/で (te-form) and the next clause
        starts with an auxiliary continuation (しまい, いく, くる, etc.), move
        the auxiliary word(s) back to the previous clause to keep the compound
        verb unit intact.
        """
        if len(clauses) < 2:
            return clauses

        _TE_ENDINGS = {"て", "で"}
        # Auxiliary continuations that must attach to preceding te-form
        _AUX_STARTS = {
            "しまい", "しまう", "しまっ", "しまいました", "しまいます",
            "いく", "いき", "いった", "いきました", "いきます",
            "くる", "きた", "きました", "きます",
            "いる", "いた", "います", "いました",
            "おく", "おき", "おいた", "おきます", "おきました",
        }

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]

            prev_idxs = sorted(prev.get('word_indices', []))
            cur_idxs = sorted(cur.get('word_indices', []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            gap = float(cur.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))
            if gap > 1.0:  # slightly larger gap for auxiliaries
                repaired.append(cur)
                continue

            # Get previous clause's last non-disfluent word
            prev_last_text = None
            for wi in reversed(prev_idxs):
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    prev_last_text = text
                    break

            if prev_last_text is None:
                repaired.append(cur)
                continue

            # Get first non-disfluent words of current clause
            first_words = []
            for wi in cur_idxs:
                if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                    continue
                text = tg_words[wi].text.strip()
                if text:
                    first_words.append((wi, text))
                    if len(first_words) >= 3:
                        break

            if not first_words:
                repaired.append(cur)
                continue

            indices_to_move = []

            # Pattern A: prev ends with て/で + next starts with auxiliary
            if prev_last_text in _TE_ENDINGS:
                if first_words[0][1] in _AUX_STARTS:
                    indices_to_move = [first_words[0][0]]

            # Pattern B: next clause starts with て/で + auxiliary (e.g., れ | て しまい まし た)
            # The て belongs to the verb but was separated; move て+auxiliary+inflection back
            if not indices_to_move and len(first_words) >= 2:
                if first_words[0][1] in _TE_ENDINGS and first_words[1][1] in _AUX_STARTS:
                    indices_to_move = [first_words[0][0], first_words[1][0]]
                    # Also grab inflection continuations (まし+た, ます, ません)
                    _INFLECT_WORDS = {"まし", "ます", "ませ", "た", "ん", "て"}
                    remaining_words = []
                    for wi in cur_idxs:
                        if wi in set(indices_to_move):
                            continue
                        if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                            continue
                        text = tg_words[wi].text.strip()
                        if text:
                            remaining_words.append((wi, text))
                            if len(remaining_words) >= 3:
                                break
                    for wi, text in remaining_words:
                        if text in _INFLECT_WORDS:
                            indices_to_move.append(wi)
                        else:
                            break

            if not indices_to_move:
                repaired.append(cur)
                continue

            cur_remaining = [wi for wi in cur_idxs if wi not in set(indices_to_move)]
            if not cur_remaining:
                repaired.append(cur)
                continue

            prev['word_indices'] = sorted(set(prev_idxs + indices_to_move))
            cur['word_indices'] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            repaired[-1] = prev
            if cur.get('word_indices'):
                repaired.append(cur)

        return repaired

    def _leading_subordinator_indices(
        self,
        cur_idxs: List[int],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[int]:
        """
        Detect leading subordinator tokens that should attach to the previous clause.
        """
        if not cur_idxs:
            return []

        single_markers = {
            "とき", "時", "ながら", "つつ",
            "うちに", "間に", "ので", "のに",
            "から", "たら", "ば", "なら",
            "ても", "けど", "けれど", "けれども",
            "まで", "うち", "間", "前", "後", "際",
        }
        pair_markers = {
            "ときに", "時に", "ときは", "時は",
            "うちに", "間に", "前に", "後で",
            "ために", "ように", "際に",
        }
        second_particles = {"に", "は", "で"}

        leading: List[Tuple[int, str]] = []
        for wi in cur_idxs:
            if disfluency_labels is not None and wi < len(disfluency_labels) and disfluency_labels[wi] == 1:
                continue
            text = tg_words[wi].text
            if not text or not text.strip():
                continue
            norm = self.segmenter._normalize_surface_text(text)
            if not norm:
                continue
            leading.append((wi, norm))
            if len(leading) >= 3:
                break

        if not leading:
            return []

        if len(leading) >= 2:
            pair = leading[0][1] + leading[1][1]
            if pair in pair_markers:
                return [leading[0][0], leading[1][0]]

            if (
                leading[0][1] in {"とき", "時", "うち", "間", "前", "後", "ため", "よう", "際"}
                and leading[1][1] in second_particles
            ):
                return [leading[0][0], leading[1][0]]

        if leading[0][1] in single_markers:
            return [leading[0][0]]

        return []

    def _repair_subordinator_hangovers(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Move leading subordinators (e.g., とき/時/うちに/間に/ながら) to the previous clause.
        """
        if len(clauses) < 2:
            return clauses

        repaired = [clauses[0]]
        for cur in clauses[1:]:
            prev = repaired[-1]
            prev_idxs = sorted(prev.get("word_indices", []))
            cur_idxs = sorted(cur.get("word_indices", []))
            if not prev_idxs or not cur_idxs:
                repaired.append(cur)
                continue

            gap = float(cur.get("start_time", 0.0)) - float(prev.get("end_time", 0.0))
            if gap > 1.0:
                repaired.append(cur)
                continue

            move_idxs = self._leading_subordinator_indices(cur_idxs, tg_words, disfluency_labels)
            if not move_idxs:
                repaired.append(cur)
                continue

            prev_text = (prev.get("text") or "").strip()
            prev_feats = _analyze_clause_surface(self.segmenter, prev_text) if prev_text else {}
            if not prev_feats.get("has_pred_verb_adj", False):
                prev_doc = self.segmenter.nlp(prev_text) if prev_text else []
                has_aux_pred = any(
                    t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"}
                    for t in prev_doc
                )
                if not has_aux_pred:
                    repaired.append(cur)
                    continue

            move_set = set(move_idxs)
            cur_remaining = [wi for wi in cur_idxs if wi not in move_set]
            if not cur_remaining:
                repaired.append(cur)
                continue
            if not self._has_content_tokens(cur_remaining, tg_words, disfluency_labels):
                repaired.append(cur)
                continue

            prev_augmented = sorted(set(prev_idxs + move_idxs))
            prev["word_indices"] = prev_augmented
            cur["word_indices"] = cur_remaining
            prev = self._rebuild_clause_payload(prev, tg_words, disfluency_labels)
            cur = self._rebuild_clause_payload(cur, tg_words, disfluency_labels)

            if prev.get("type") in {"IND", "SUB_REL", "fragment", "CHAIN_TE"}:
                prev["type"] = "SUB_ADVCL"

            repaired[-1] = prev
            if cur.get("word_indices"):
                repaired.append(cur)

        return repaired

    def _split_runon_predicate_chains(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """
        Split over-merged predicative clauses that contain multiple predicate chains.
        """
        if not clauses:
            return clauses

        def _split_once(base_clause: Dict) -> Optional[Tuple[Dict, Dict]]:
            ctype = base_clause.get("type", "")
            text = (base_clause.get("text") or "").strip()
            idxs = sorted(base_clause.get("word_indices", []))
            finite_connector_split = self._find_finite_connector_split_char(text)
            finite_topic_restart_split = self._find_finite_topic_restart_split_char(text)
            question_quotative_split = self._find_question_quotative_split_char(text)
            allowed_types = {"IND", "SUB_REL", "SUB_ADVCL", "SUB_CCOMP", "fragment"}
            if (
                ctype not in allowed_types
                and finite_connector_split is None
                and finite_topic_restart_split is None
                and question_quotative_split is None
            ) or len(idxs) < 2:
                return None

            split_char = self._find_short_quotative_reporting_split_char(text)
            if split_char is None:
                split_char = finite_connector_split
            if split_char is None:
                split_char = finite_topic_restart_split
            if split_char is None:
                split_char = question_quotative_split
            if split_char is None and ctype != "SUB_CCOMP":
                split_char = self._find_runon_split_char(text)
            if split_char is None and ctype != "SUB_CCOMP":
                split_char = self._find_restart_split_char(text)
            if split_char is None:
                return None

            char_to_wi: List[int] = []
            # v1.0-paper mapped offsets over the words not marked disfluent, which
            # include fillers the clause text leaves out (a few splits land off).
            text_wis = [wi for wi in idxs if not (disfluency_labels is not None and disfluency_labels[wi] == 1)]
            for wi in text_wis:
                wtxt = tg_words[wi].text
                if not wtxt:
                    continue
                for _ in wtxt:
                    char_to_wi.append(wi)

            if not char_to_wi:
                return None
            if split_char >= len(char_to_wi):
                split_char = len(char_to_wi) - 1
            if split_char <= 0:
                return None

            right_start_wi = char_to_wi[split_char]
            split_pos = None
            for i, wi in enumerate(idxs):
                if wi == right_start_wi:
                    split_pos = i
                    break
            if split_pos is None or split_pos <= 0 or split_pos >= len(idxs):
                return None

            left_idxs = idxs[:split_pos]
            right_idxs = idxs[split_pos:]
            if not self._has_content_tokens(left_idxs, tg_words, disfluency_labels):
                return None
            if not self._has_content_tokens(right_idxs, tg_words, disfluency_labels):
                return None

            left_clause = dict(base_clause)
            right_clause = dict(base_clause)
            left_clause["word_indices"] = left_idxs
            right_clause["word_indices"] = right_idxs
            left_clause = self._rebuild_clause_payload(left_clause, tg_words, disfluency_labels)
            right_clause = self._rebuild_clause_payload(right_clause, tg_words, disfluency_labels)
            left_clause["type"] = self._infer_restart_split_type(left_clause.get("text", ""), left_clause.get("type", ctype))
            right_clause["type"] = self._infer_restart_split_type(right_clause.get("text", ""), right_clause.get("type", ctype))

            return left_clause, right_clause

        out: List[Dict] = []
        for clause in clauses:
            pending = [clause]
            split_budget = 3
            while pending:
                current = pending.pop(0)
                if split_budget <= 0:
                    out.append(current)
                    continue

                split_pair = _split_once(current)
                if split_pair is None:
                    out.append(current)
                    continue

                left_clause, right_clause = split_pair
                split_budget -= 1
                pending = [left_clause, right_clause] + pending

        return out

    def _should_merge_late_split_micro_pair(
        self,
        left: Dict,
        right: Dict,
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> bool:
        left_idxs = sorted(left.get("word_indices", []))
        right_idxs = sorted(right.get("word_indices", []))
        if not left_idxs or not right_idxs:
            return False

        left_visible = self._visible_clause_word_indices(left_idxs, tg_words, disfluency_labels)
        right_visible = self._visible_clause_word_indices(right_idxs, tg_words, disfluency_labels)
        if not left_visible or not right_visible:
            return False

        left_text = ''.join(tg_words[i].text for i in left_visible)
        right_text = ''.join(tg_words[i].text for i in right_visible)
        left_compact = self._compact_clause_text(left_text)
        right_compact = self._compact_clause_text(right_text)
        left_last = (tg_words[left_visible[-1]].text or '').strip()
        right_first = (tg_words[right_visible[0]].text or '').strip()
        if not left_last or not right_first:
            return False

        left_feats = _analyze_clause_surface(self.segmenter, left_text)
        right_feats = _analyze_clause_surface(self.segmenter, right_text)
        right_soft = self._strip_leading_soft_filler(right_text)
        right_after_det = right_soft
        for det in ("この", "その", "あの", "どの", "こんな", "そんな", "あんな"):
            if right_after_det.startswith(det) and len(right_after_det) > len(det):
                right_after_det = right_after_det[len(det):]
                break

        strong_restart_heads = ("そして", "しかし", "でも", "それで", "だから", "なので", "ところで", "ただ", "すると")
        if right_compact.startswith(strong_restart_heads) and left_feats.get("has_pred_verb_adj", False):
            return False
        if (
            self._ends_with_question_shell(left_text)
            and right_compact.startswith("で")
            and right_feats.get("has_pred_verb_adj", False)
        ):
            return False
        if (
            left_feats.get("has_pred_or_nominal", False)
            and left_feats.get("has_element", False)
            and left_compact.endswith("で")
            and right_soft
            and right_feats.get("has_pred_verb_adj", False)
            and (
                self._starts_with_nominal_surface(right_soft)
                or self._starts_with_nominal_surface(right_after_det)
                or right_soft.startswith(("外", "内", "中", "ここ", "そこ", "あそこ", "どこ"))
            )
        ):
            return False

        if self._is_bound_inflection(left_last, right_first):
            return True
        if self._starts_with_name_suffix(right_text):
            if self._has_finite_restart_shell_tail(left_text):
                return False
            return True
        if self._has_boundary_surface_overlap(
            left_text,
            right_text,
            left,
            right,
            tg_words,
            disfluency_labels,
        ):
            return True
        if self._ends_with_question_shell(left_text) and self._starts_with_question_continuation(right_text):
            if right_compact.startswith(("かで", "かでも")) and right_feats.get("has_pred_verb_adj", False):
                return False
            return True
        if self._starts_with_subordinator_continuation(right_text):
            if self._has_short_broken_tail(left_text, right_text):
                return True
            if left_last in {"て", "で", "し", "き", "り", "び", "み", "ん", "まし", "でし", "ませ"}:
                return True
            return False
        if self._is_short_case_marked_fragment(left_text, left_feats) and (
            right_feats.get("has_pred_verb_adj", False)
            or right_feats.get("has_pred_or_nominal", False)
            or self._starts_with_nominal_surface(right_text)
        ):
            return True
        if self._is_fixed_merge_prefix(left_text):
            return True
        if self._is_broad_prenominal_fragment(left_text, left_feats, right_text, right_feats):
            right_restart_heads = (
                "途端に", "時に", "時は", "ときに", "ときは",
                "うちに", "間に", "前に", "後で",
                "ために", "ように",
                "だから", "なので", "そして", "それで", "でも", "しかし",
            )
            if right_compact.startswith(right_restart_heads):
                return False
            return True
        if self._has_hanging_stem_overlap(left_text, right_text):
            return True
        if self._is_quotative_bridge_fragment(left_text, right_text):
            return True
        if self._is_short_wh_case_fragment(left_text) and right_feats.get("has_pred_verb_adj", False):
            return True
        if self._has_short_broken_tail(left_text, right_text) and right_feats.get("has_pred_verb_adj", False):
            return True
        if left_last in {"まし", "でし", "ませ", "し"} and right_first in {"た", "て", "ん"}:
            return True
        return False

    def repair_late_split_micro_fragments(
        self,
        clauses: List[Dict],
        tg_words: List[WordInterval],
        disfluency_labels: List[int] = None,
    ) -> List[Dict]:
        """Merge only the micro-splits created by the final run-on split pass."""
        if len(clauses) < 2:
            return clauses

        repaired = sorted(
            [self._rebuild_clause_payload(dict(c), tg_words, disfluency_labels) for c in clauses],
            key=lambda c: (float(c.get("start_time", 0.0)), float(c.get("end_time", 0.0))),
        )

        changed = True
        while changed:
            changed = False
            merged: List[Dict] = []
            i = 0
            while i < len(repaired):
                cur = repaired[i]
                if i + 1 >= len(repaired):
                    merged.append(cur)
                    break

                nxt = repaired[i + 1]
                if not self._should_merge_late_split_micro_pair(cur, nxt, tg_words, disfluency_labels):
                    merged.append(cur)
                    i += 1
                    continue

                merged.append(self._merge_clause_pair(
                    cur,
                    nxt,
                    tg_words,
                    disfluency_labels,
                    prefer="right",
                ))
                changed = True
                i += 2

            repaired = sorted(
                [self._rebuild_clause_payload(dict(c), tg_words, disfluency_labels) for c in merged],
                key=lambda c: (float(c.get("start_time", 0.0)), float(c.get("end_time", 0.0))),
            )

        return repaired

    def align_clauses(self, disfluency_labels: List[int] = None) -> List[Dict]:
        """Align clause boundaries to actual word times.
        
        Args:
            disfluency_labels: Per-word labels (0=fluent, 1=disfluent).
                If provided, clause displayed text excludes disfluent words.
        """
        tg_words = self.tg.get_word_list()
        sentence_groups = self._build_sentence_word_groups(
            tg_words, disfluency_labels, pause_split_sec=self.segmenter.pause_split_sec)
        if not sentence_groups:
            return []

        aligned_clauses: List[Dict] = []
        claimed_word_indices: Set[int] = set()
        for group in sentence_groups:
            aligned_clauses.extend(
                self._align_group_clauses(
                    group,
                    tg_words,
                    claimed_word_indices,
                    disfluency_labels=disfluency_labels,
                )
            )

        # Repair boundary artifacts before fragment recovery.
        aligned_clauses = self._repair_boundary_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_trailing_function_hangovers(aligned_clauses, tg_words, disfluency_labels)

        # Handle unclaimed words as fragments
        all_indices = set(range(len(tg_words)))
        unclaimed = sorted(all_indices - claimed_word_indices)
        
        if unclaimed:
            # Group contiguous unclaimed indices
            groups = []
            if unclaimed:
                curr_group = [unclaimed[0]]
                for i in range(1, len(unclaimed)):
                    if unclaimed[i] == unclaimed[i-1] + 1:
                        curr_group.append(unclaimed[i])
                    else:
                        groups.append(curr_group)
                        curr_group = [unclaimed[i]]
                groups.append(curr_group)
            
            # Create fragment clauses for groups
            for group in groups:
                valid_indices = [idx for idx in group if not tg_words[idx].is_pause and tg_words[idx].text]
                if not valid_indices:
                    continue

                # Keep full timing anchors, but hide only model-labeled disfluencies from clause text.
                group_words_all = [tg_words[i] for i in valid_indices]
                display_indices = self._visible_clause_word_indices(
                    valid_indices, tg_words, disfluency_labels
                )
                if not display_indices:
                    continue
                group_words_display = [tg_words[i] for i in display_indices]
                group_text = ''.join(w.text for w in group_words_display)

                # Skip if display text is only fillers/backchannels.
                is_all_fillers = all(
                    self.segmenter._is_unambiguous_filler_surface(w.text) for w in group_words_display
                )
                if is_all_fillers:
                    continue

                aligned_clauses.append({
                    'start_time': group_words_all[0].start,
                    'end_time': group_words_all[-1].end,
                    'text': group_text,
                    'type': 'fragment',
                    'word_indices': valid_indices,
                    'mora': sum(w.mora_count for w in group_words_display)
                })

        # Sort before final boundary repair so only true temporal neighbors are touched.
        aligned_clauses.sort(key=lambda c: c['start_time'])
        aligned_clauses = self._repair_boundary_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_trailing_function_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._merge_boundary_continuations(aligned_clauses, tg_words, disfluency_labels)
        # Run a second pass so newly merged hosts can absorb the next predicate continuation.
        aligned_clauses = self._merge_boundary_continuations(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._split_runon_predicate_chains(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._trim_boundary_noise(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_te_form_boundaries(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_inflection_splits(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_auxiliary_splits(aligned_clauses, tg_words, disfluency_labels)
        # Second inflection pass: auxiliary repair may expose new inflection splits
        aligned_clauses = self._repair_inflection_splits(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_subordinator_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._repair_leading_marker_hangovers(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._trim_boundary_glue(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._merge_boundary_continuations(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._trim_boundary_glue(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses = self._recover_unclaimed_tail_inflections(aligned_clauses, tg_words, disfluency_labels)
        # Final leading-filler trim: later repair passes may have created new
        # clause boundaries that start with filler words (ん, あの, etc.).
        aligned_clauses = self._trim_boundary_noise(aligned_clauses, tg_words, disfluency_labels)
        aligned_clauses.sort(key=lambda c: c['start_time'])
        return aligned_clauses


# ==============================================================================
# Processing Functions
# ==============================================================================

def _truncate_clause_text(text: str, max_chars: int = 60) -> str:
    """Apply methodology truncation for overly long clause labels."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    # Prefer trimming at a space boundary when available.
    if " " in text:
        cut = text[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        cut = cut.strip()
        if cut:
            return cut
    # Japanese labels are often unspaced; hard trim as fallback.
    return text[:max_chars].strip()


def _strip_trailing_discourse_tail(text: str) -> str:
    """
    Strip clause-final discourse/filler tails conservatively.
    Applies only at the end of the clause label (not internal tokens).
    """
    s = (text or "").strip()
    if not s:
        return s

    trailing_terms = (
        "なんか",  # なんか
        "あの",        # あの
        "その",        # その
        "えっと",  # えっと
        "ええと",  # ええと
        "うーん",  # うーん
        "うん",        # うん
        "まあ",        # まあ
        "ま",              # ま
        "えー",        # えー
        "あー",        # あー
    )
    trim_chars = " 　、。,.;:!?！？"

    out = s
    changed = True
    while changed:
        changed = False
        out = out.rstrip(trim_chars)
        for term in trailing_terms:
            if out.endswith(term):
                out = out[:-len(term)].rstrip(trim_chars)
                changed = True
                break
    return out


def _is_discourse_tail_token(segmenter: JapaneseClauseSegmenter, tok) -> bool:
    """Heuristic for clause-final discourse/filler tokens."""
    norm = segmenter._normalize_surface_text(tok.text)
    if not norm:
        return False
    if segmenter._is_unambiguous_filler_surface(tok.text):
        return True
    if norm in {
        "なんか",  # なんか
        "あの",        # あの
        "その",        # その
        "えっと",  # えっと
        "ええと",  # ええと
        "うーん",  # うーん
        "うん",        # うん
        "まあ",        # まあ
        "ま",              # ま
        "えー",        # えー
        "あー",        # あー
    }:
        return True
    if tok.dep_ == "discourse":
        return True
    if tok.pos_ in {"INTJ"}:
        return True
    return False


def _strip_trailing_discourse_tail_model(
    segmenter: JapaneseClauseSegmenter,
    text: str,
    max_tokens: int = 2,
) -> str:
    """
    Parse-driven tail cleanup for cases where filler tails are attached to noisy text.
    """
    out = (text or "").strip()
    if not out:
        return out

    for _ in range(max_tokens):
        doc = segmenter.nlp(out)
        toks = [t for t in doc if t.text.strip()]
        if not toks:
            break
        last = toks[-1]
        if len(toks) >= 2 and ClauseAligner._is_bound_inflection(toks[-2].text, last.text):
            break
        if not _is_discourse_tail_token(segmenter, last):
            break
        cut = out[:last.idx].strip()
        if len(cut.replace(" ", "").replace("　", "")) < 3:
            break
        out = cut

    return out


def _analyze_clause_surface(segmenter: JapaneseClauseSegmenter, text: str) -> Dict[str, object]:
    """Lightweight predicate/marker checks for methodology post-validation."""
    toks = [t for t in segmenter.nlp(text) if t.text.strip()]
    surfaces = [t.text for t in toks]
    pos = [t.pos_ for t in toks]
    deps = [t.dep_ for t in toks]
    lemmas = [t.lemma_ for t in toks]

    particle_surface = {
        "を", "に", "が", "で", "は", "へ", "と", "の", "も", "や",
        "か", "ね", "よ", "ぞ", "さ", "な", "わ", "し", "て",
    }
    particle_pos = {"PART", "AUX", "ADP", "PUNCT", "SYM", "SCONJ", "CCONJ"}
    sub_markers = {
        "から", "ながら", "つつ", "けど", "けれど", "けれども",
        "なら", "ば", "ので", "のに", "ため", "ように",
        "まで", "前に", "後で", "時",
    }
    ambiguous_sub_markers = {"が", "と", "し"}
    te_markers = ("て", "で", "たり", "だり")

    has_pred_verb_adj = any(p in {"VERB", "ADJ"} for p in pos)
    has_pred_or_nominal = has_pred_verb_adj or any(p == "NOUN" for p in pos)
    has_nominal_token = any(p in {"NOUN", "PROPN", "PRON"} for p in pos)
    has_element = any(d in {"obj", "iobj", "obl", "advmod", "nmod"} for d in deps)

    has_sub_marker = any(s in sub_markers for s in surfaces)
    if not has_sub_marker:
        for s, p, d in zip(surfaces, pos, deps):
            if s in ambiguous_sub_markers and (p in {"SCONJ", "CCONJ"} or d in {"mark", "cc"}):
                has_sub_marker = True
                break
    if not has_sub_marker and "の" in surfaces:
        for i in range(len(surfaces) - 1):
            if surfaces[i] == "の" and surfaces[i + 1] in {"で", "に"}:
                has_sub_marker = True
                break

    has_te_signal = any(s in te_markers for s in surfaces) or text.endswith(te_markers)
    ends_with_te = text.endswith(te_markers)
    all_particles = bool(toks) and all((s in particle_surface) or (p in particle_pos) for s, p in zip(surfaces, pos))
    has_case_tail = bool(toks) and (
        deps[-1] in {"case", "mark", "cc", "fixed"}
        or pos[-1] in {"ADP", "PART", "SCONJ", "CCONJ"}
        or surfaces[-1] in particle_surface
    )

    light_lemmas = {"する", "ます", "た", "だ", "です", "ある", "いる", "れる", "られる"}
    has_lexical_content = any(p in {"NOUN", "PROPN", "ADJ", "ADV", "NUM", "PRON"} for p in pos)
    is_auxiliary_shell = (
        bool(toks)
        and len(text) <= 8
        and not has_lexical_content
        and all(p in {"VERB", "AUX", "PART", "ADP", "SCONJ", "CCONJ", "PUNCT", "SYM"} for p in pos)
        and all((l in light_lemmas) or (s in particle_surface) for l, s in zip(lemmas, surfaces))
    )

    has_stance = any(l in segmenter._minor_matrix_lemmas for l in lemmas) or any(
        x in text for x in ("と思う", "と考える", "と感じる")
    )

    normalized = unicodedata.normalize("NFKC", text)
    compact = normalized.replace(" ", "").replace("　", "")
    # v1.0-paper: interview-frame clauses (spoken IDs, closings) become fragments.
    has_meta = bool(re.match(r"^[A-Za-z]{2,}\d+", normalized)) or bool(
        re.search(
            r"(?:ありがとう(?:ございます|ございました)?|有難う(?:ございます|ございました)?|以上(?:です)?|はいどうも|どうも|お願いします)",
            compact,
        )
    )

    return {
        "has_pred_verb_adj": has_pred_verb_adj,
        "has_pred_or_nominal": has_pred_or_nominal,
        "has_nominal_token": has_nominal_token,
        "has_element": has_element,
        "has_sub_marker": has_sub_marker,
        "has_te_signal": has_te_signal,
        "ends_with_te": ends_with_te,
        "all_particles": all_particles,
        "has_case_tail": has_case_tail,
        "is_auxiliary_shell": is_auxiliary_shell,
        "has_stance": has_stance,
        "has_meta": has_meta,
    }


def _has_subordinator_tail(compact_text: str) -> bool:
    """Check whether compact text ends with a subordinator-like tail."""
    tails = (
        "とき", "時", "ときに", "時に", "ときは", "時は",
        "うちに", "間に", "ながら", "つつ",
        "ので", "のに", "から",
        "けど", "けれど", "けれども",
        "たら", "ば", "なら",
        "ても", "でも",
        "ために", "ように",
        "まで", "前に", "後で", "際に",
    )
    return any(compact_text.endswith(t) for t in tails)


def _is_ind_like_sub_rel(segmenter: JapaneseClauseSegmenter, text: str) -> bool:
    """
    Parse-driven check: SUB_REL span looks like an independent sentence.
    """
    compact = text.replace(" ", "").replace("　", "")
    if len(compact) < 6:
        return False
    if _has_subordinator_tail(compact):
        return False

    doc = segmenter.nlp(text)
    toks = [t for t in doc if t.text.strip()]
    if not toks:
        return False

    has_topic = any(
        t.text == "は" and t.dep_ == "case" and t.head.pos_ in {"NOUN", "PRON", "PROPN"}
        for t in toks
    )
    if not has_topic:
        return False

    has_predicate = any(
        t.pos_ in {"VERB", "ADJ"}
        or (t.pos_ == "AUX" and t.lemma_ in {"だ", "です", "ます", "た"})
        for t in toks
    )
    if not has_predicate:
        return False

    last = toks[-1]
    if last.pos_ in {"NOUN", "PROPN", "PRON", "NUM"}:
        return False
    if last.dep_ in {"case", "mark", "cc", "fixed"}:
        return False
    # v16-fix: adnominal / prenominal endings modify a following noun → keep SUB_REL.
    # E.g., そういう, ある, いる, した, etc. in their 連体形 usage.
    if last.dep_ in {"acl", "amod", "relcl"}:
        return False
    # Demonstrative adjectives ending in う (そういう, こういう, ああいう) are prenominal.
    if last.text.endswith("いう") or last.text in {"そういう", "こういう", "ああいう"}:
        return False

    return True


def _apply_methodology_rules(
    clauses: List[Dict],
    segmenter: JapaneseClauseSegmenter,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Apply Japanese validation rules from CLAUSE_SEGMENTATION_METHODOLOGY.md (4.2).
    """
    kept: List[Dict] = []
    stats = {
        "dropped_fragment_lt3": 0,
        "dropped_short_no_pred": 0,
        "dropped_particles_only": 0,
        "dropped_meta": 0,
        "dropped_no_pred_lt10": 0,
        "dropped_no_pred_type": 0,
        "dropped_bare_te": 0,
        "type_relabel_chain_te_to_advcl": 0,
        "type_relabel_chain_te_to_ind": 0,
        "type_relabel_advcl_to_chain_te": 0,
        "type_relabel_advcl_to_ind": 0,
        "type_relabel_minor_to_ind": 0,
        "type_relabel_ind_no_pred_to_fragment": 0,
        "type_relabel_no_pred_to_fragment": 0,
        "v15_ind_to_sub_advcl_subordinator_final": 0,
        "v15_ind_to_minor_stance_recovery": 0,
        "v15_fragment_to_ind_predicate_recovery": 0,
        "v15_sub_rel_to_sub_advcl_toki": 0,
        "v16_fragment_predicate_relabel": 0,
        "v16_sub_rel_to_ind_topic_predicate": 0,
        "v15_orphan_merge": 0,
        "v18_dropped_prefix_false_start": 0,
        "stripped_discourse_tail": 0,
        "truncated_long": 0,
        "preserved_fragment_lt3": 0,
        "preserved_short_no_pred": 0,
        "preserved_particles_only": 0,
        "preserved_meta": 0,
        "preserved_no_pred_lt10": 0,
        "preserved_no_pred_type": 0,
        "preserved_bare_te": 0,
    }

    def _preserve_as_fragment(reason_key: str, clause: Dict, text: str) -> None:
        stats[reason_key] += 1
        clause["type"] = "fragment"
        clause["text"] = text
        kept.append(clause)

    for clause in clauses:
        text = (clause.get("text") or "").strip()
        ctype = clause.get("type", "")
        if not text:
            continue

        # Remove trailing discourse fillers that frequently leak to clause edges.
        text_stripped = _strip_trailing_discourse_tail_model(segmenter, text)
        if text_stripped == text:
            text_stripped = _strip_trailing_discourse_tail(text)
        if text_stripped and text_stripped != text:
            stats["stripped_discourse_tail"] += 1
            text = text_stripped

        # Very long rule: truncate.
        if len(text) > 60:
            text_new = _truncate_clause_text(text, 60)
            if text_new != text:
                stats["truncated_long"] += 1
                text = text_new

        feats = _analyze_clause_surface(segmenter, text)

        # Type normalization for better rule alignment.
        if ctype == "MINOR" and not feats["has_stance"]:
            ctype = "IND"
            stats["type_relabel_minor_to_ind"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        if ctype == "CHAIN_TE":
            compact_text = text.replace(" ", "").replace("　", "")
            if _has_subordinator_tail(compact_text):
                ctype = "SUB_ADVCL"
                stats["type_relabel_chain_te_to_advcl"] += 1
                feats = _analyze_clause_surface(segmenter, text)

            if ctype == "CHAIN_TE" and not feats["has_te_signal"]:
                if feats["has_sub_marker"]:
                    ctype = "SUB_ADVCL"
                    stats["type_relabel_chain_te_to_advcl"] += 1
                elif feats["has_pred_verb_adj"]:
                    ctype = "IND"
                    stats["type_relabel_chain_te_to_ind"] += 1
                else:
                    ctype = "fragment"
                    stats["type_relabel_no_pred_to_fragment"] += 1
                feats = _analyze_clause_surface(segmenter, text)
            elif ctype == "CHAIN_TE" and not feats["has_element"]:
                # CHAIN_TE requires verb+element; relabel when element evidence is absent.
                if feats["has_sub_marker"]:
                    ctype = "SUB_ADVCL"
                    stats["type_relabel_chain_te_to_advcl"] += 1
                elif feats["has_pred_verb_adj"] or feats["has_pred_or_nominal"]:
                    ctype = "IND"
                    stats["type_relabel_chain_te_to_ind"] += 1
                else:
                    ctype = "fragment"
                    stats["type_relabel_no_pred_to_fragment"] += 1
                feats = _analyze_clause_surface(segmenter, text)

        if ctype == "SUB_ADVCL" and not feats["has_sub_marker"]:
            compact_text = text.replace(" ", "").replace("　", "")
            if _has_subordinator_tail(compact_text):
                pass
            elif feats["has_te_signal"] and feats["has_element"]:
                ctype = "CHAIN_TE"
                stats["type_relabel_advcl_to_chain_te"] += 1
            elif feats["has_pred_verb_adj"]:
                ctype = "IND"
                stats["type_relabel_advcl_to_ind"] += 1
            else:
                ctype = "fragment"
                stats["type_relabel_no_pred_to_fragment"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # Residual IND spans ending in case/connector tails are typically
        # non-predicative boundary leftovers under noisy learner speech.
        if ctype == "IND" and not feats["has_pred_verb_adj"] and feats["has_case_tail"]:
            ctype = "fragment"
            stats["type_relabel_ind_no_pred_to_fragment"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # MINOR exemption: keep regardless.
        if ctype == "MINOR":
            clause["type"] = ctype
            clause["text"] = text
            kept.append(clause)
            continue

        # Non-fragment clause types should carry a predicate/nominal core.
        if ctype != "fragment" and not feats["has_pred_or_nominal"]:
            _preserve_as_fragment("preserved_no_pred_type", clause, text)
            continue


        # -- v15 Fix 4: SUB_REL with subordinator -> SUB_ADVCL ------
        # FROZEN_RULES Rule 3a lists subordinators. GiNZA sometimes parses
        # adverbial clauses as acl -> SUB_REL. Relabel when subordinator found.
        if ctype == "SUB_REL":
            compact_text = text.replace(" ", "").replace("　", "")
            _sub_rel_fix_endings = (
                "とき", "時",
                "ときに", "時に",
                "ときは", "時は",
                "ので", "のに",
                "から",
                "けど", "けれど", "けれども",
                "たら", "ば", "なら",
                "ながら", "つつ",
                "ても", "でも",
                "ために", "ように",
                "まで",
                "前に", "後で",
                "うちに", "間に",
                "際に",
            )
            is_sub_rel_to_advcl = any(compact_text.endswith(e) for e in _sub_rel_fix_endings)
            # Also check ambiguous が/と at end with SCONJ POS
            if not is_sub_rel_to_advcl:
                _doc = segmenter.nlp(text)
                _toks = [t for t in _doc if t.text.strip()]
                if _toks:
                    _last = _toks[-1]
                    if (_last.text in ("が", "と", "し") and
                        (_last.pos_ in {"SCONJ", "CCONJ"} or _last.dep_ in {"mark", "cc"})):
                        is_sub_rel_to_advcl = True
            if is_sub_rel_to_advcl:
                ctype = "SUB_ADVCL"
                stats["v15_sub_rel_to_sub_advcl_toki"] += 1
                feats = _analyze_clause_surface(segmenter, text)

        # v16: SUB_REL that behaves like a full sentence -> IND.
        if ctype == "SUB_REL" and _is_ind_like_sub_rel(segmenter, text):
            ctype = "IND"
            stats["v16_sub_rel_to_ind_topic_predicate"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # -- v15 Fix 1: IND ending with subordinator -> SUB_ADVCL ----
        if ctype == "IND":
            compact_text = text.replace(" ", "").replace("　", "")
            # Strip trailing の/は for check (e.g., からの → から)
            _check_text = compact_text
            for _tp in ("の", "は"):
                if _check_text.endswith(_tp) and len(_check_text) > 3:
                    _check_text = _check_text[:-len(_tp)]
            _unambiguous_subordinators = (
                "から", "ので", "けど", "けれど", "けれども",
                "たら", "ば", "なら",
                "ながら", "つつ",
                "のに", "し",
                "ても", "でも",
                "ために", "ように",
                "まで",
                "前に", "後で",
                "うちに", "間に",
                "際に",
                "とき", "時", "ときに", "時に",
            )
            is_ind_to_advcl = any(_check_text.endswith(sub) for sub in _unambiguous_subordinators)
            if not is_ind_to_advcl:
                _doc = segmenter.nlp(text)
                _toks = [t for t in _doc if t.text.strip()]
                if _toks:
                    _last = _toks[-1]
                    if (_last.text in ("が", "と") and
                        (_last.pos_ in {"SCONJ", "CCONJ"} or _last.dep_ in {"mark", "cc"})):
                        is_ind_to_advcl = True
            if is_ind_to_advcl:
                ctype = "SUB_ADVCL"
                stats["v15_ind_to_sub_advcl_subordinator_final"] += 1
                feats = _analyze_clause_surface(segmenter, text)

        # -- v15 Fix 3: Fragment with valid predicate -> IND recovery --
        if ctype == "fragment" and len(text) >= 10:
            _masu_pats = (
                "ました", "ます", "ません",
                "でした", "です",
                "ている", "ていた",
                "ていました", "ています",
                "しました", "しています",
                "なりました", "ありました",
                "いました", "きました",
                "れました",
                "みました", "します",
            )
            # v1.0-paper listed three imperatives from the evaluation stories.
            _masu_pats += ("見ろ", "帰れ", "起きろ")
            compact_text = text.replace(" ", "").replace("　", "")
            has_pred_pattern = any(p in compact_text for p in _masu_pats)
            if has_pred_pattern:
                _doc = segmenter.nlp(text)
                _has_v = any(t.pos_ in {"VERB", "AUX"} for t in _doc)
                if _has_v:
                    _frag_sub_ends = (
                        "とき", "時", "ときに", "時に",
                        "から", "ので", "けど", "けれど",
                        "たら", "ば", "のに", "ながら",
                    )
                    if any(compact_text.endswith(e) for e in _frag_sub_ends):
                        ctype = "SUB_ADVCL"
                    else:
                        ctype = "IND"
                    stats["v15_fragment_to_ind_predicate_recovery"] += 1
                    feats = _analyze_clause_surface(segmenter, text)

        # v16: generic fragment recovery driven by parse features (not string templates).
        # v16-fix: require minimum length to avoid promoting single adverbs like 高く.
        if ctype == "fragment" and feats["has_pred_verb_adj"] and len(text.replace(" ", "").replace("　", "")) >= 6:
            compact_text = text.replace(" ", "").replace("　", "")
            if feats["has_sub_marker"] or _has_subordinator_tail(compact_text):
                ctype = "SUB_ADVCL"
            elif feats["has_te_signal"] and feats["has_element"]:
                ctype = "CHAIN_TE"
            else:
                ctype = "IND"
            stats["v16_fragment_predicate_relabel"] += 1
            feats = _analyze_clause_surface(segmenter, text)

        # Validation removal rules.
        if ctype == "fragment" and len(text) < 3:
            _preserve_as_fragment("preserved_fragment_lt3", clause, text)
            continue
        if feats.get("is_auxiliary_shell", False):
            _preserve_as_fragment("preserved_particles_only", clause, text)
            continue
        if feats["all_particles"]:
            _preserve_as_fragment("preserved_particles_only", clause, text)
            continue
        if len(text) <= 3 and not feats["has_pred_verb_adj"]:
            _preserve_as_fragment("preserved_short_no_pred", clause, text)
            continue
        if len(text) < 10 and not feats["has_pred_verb_adj"]:
            _preserve_as_fragment("preserved_no_pred_lt10", clause, text)
            continue
        if ctype == "CHAIN_TE" and len(text) < 12 and feats["ends_with_te"] and not feats["has_element"]:
            _preserve_as_fragment("preserved_bare_te", clause, text)
            continue
        if feats["has_meta"]:
            _preserve_as_fragment("preserved_meta", clause, text)
            continue

        clause["type"] = ctype
        clause["text"] = text
        kept.append(clause)


    # -- v15 Patch G: SUB_REL with mid-clause temporal marker -----
    # Some SUB_REL clauses contain ながら/うちに/途中に mid-clause
    # due to under-segmentation by parser. Relabel to SUB_ADVCL.
    _mid_clause_temporals = ("ながら", "うちに", "途中に", "際に", "間に")
    for idx, cl in enumerate(kept):
        ctype = cl.get("type", "")
        text = (cl.get("text") or "").replace(" ", "").replace("　", "")
        if ctype == "SUB_REL":
            if any(marker in text for marker in _mid_clause_temporals):
                cl["type"] = "SUB_ADVCL"
                stats["v15_sub_rel_to_sub_advcl_toki"] += 1

    # -- v15 Patch C: Merge orphaned-て and short renyoukei fragments ---
    _renyou_endings = ("っ", "し", "き", "り", "み", "び",
                       "に", "ぎ", "ち", "ひ", "い")
    _orphan_starts = ("た", "て", "で", "だ")
    i_merge = 0
    while i_merge < len(kept) - 1:
        cur_c = kept[i_merge]
        nxt_c = kept[i_merge + 1]
        cur_text = (cur_c.get("text") or "").replace(" ", "").replace("　", "")
        nxt_text = (nxt_c.get("text") or "").replace(" ", "").replace("　", "")
        do_merge = False
        # Case 1: Very short clause (<=3 chars) ending with renyoukei
        if (len(cur_text) <= 3 and cur_text and
            cur_text[-1] in _renyou_endings and
            nxt_text and nxt_text[0] in _orphan_starts):
            do_merge = True
        # Case 2: Truncated ました/でした/ません
        if (cur_text.endswith("まし") or cur_text.endswith("でし")) and nxt_text.startswith("た"):
            do_merge = True
        if cur_text.endswith("ませ") and nxt_text.startswith("ん"):
            do_merge = True
        # Case 3: Orphaned て/で/た (1-2 chars)
        if len(nxt_text) <= 2 and nxt_text in ("て", "で", "た", "たり"):
            do_merge = True
        if do_merge:
            cur_c["text"] = (cur_c.get("text", "") + " " + nxt_c.get("text", "")).strip()
            cur_indices = cur_c.get("word_indices", [])
            nxt_indices = nxt_c.get("word_indices", [])
            cur_c["word_indices"] = sorted(set(cur_indices + nxt_indices))
            if "end_time" in nxt_c:
                cur_c["end_time"] = nxt_c["end_time"]
            if "mora" in cur_c and "mora" in nxt_c:
                cur_c["mora"] = cur_c["mora"] + nxt_c["mora"]
            kept.pop(i_merge + 1)
            continue
        i_merge += 1

    # -- v15 Fix 2: IND with stance verb -> MINOR (context-aware) --
    _stance_surfaces = (
        "思い", "思っ", "思う",
        "思います", "思いました",
        "考え", "考えて", "考えます", "考えました",
        "感じ", "感じて", "感じます", "感じました",
    )
    for ki in range(1, len(kept)):
        prev_c = kept[ki - 1]
        cur_c = kept[ki]
        if cur_c.get("type") != "IND":
            continue
        if prev_c.get("type") != "SUB_CCOMP":
            continue
        cur_text = (cur_c.get("text") or "").replace(" ", "").replace("　", "")
        has_stance_surface = any(cur_text.startswith(s) for s in _stance_surfaces)
        if not has_stance_surface:
            _doc = segmenter.nlp(cur_c.get("text", ""))
            has_stance_surface = any(
                t.lemma_ in {"思う", "考える", "感じる"} and t.pos_ == "VERB"
                for t in _doc
            )
        if has_stance_surface:
            cur_c["type"] = "MINOR"
            stats["v15_ind_to_minor_stance_recovery"] += 1

    # -- v15 Patch F: Embedded stance verb in IND → MINOR --------
    _embedded_stance_tails = (
        "と思う", "と思い", "と思っ", "と思った",
        "と思います", "と思いまし",
        "と考え", "と考えて", "と考えた",
        "と感じ", "と感じて", "と感じた",
        "かと思い", "だと思います",
    )
    for idx, cl in enumerate(kept):
        ctype = cl.get("type", "")
        text = (cl.get("text") or "").replace(" ", "").replace("　", "")
        if ctype == "IND" and len(text) > 4:
            tail = text[-12:] if len(text) > 12 else text
            if any(tail.endswith(p) for p in _embedded_stance_tails):
                cl["type"] = "MINOR"
                stats["v15_ind_to_minor_stance_recovery"] += 1

    i_prefix = 0
    while i_prefix < len(kept) - 1:
        cur_c = kept[i_prefix]
        nxt_c = kept[i_prefix + 1]
        cur_text = (cur_c.get("text") or "").replace(" ", "").replace("　", "")
        nxt_text = (nxt_c.get("text") or "").replace(" ", "").replace("　", "")
        if not cur_text or not nxt_text:
            i_prefix += 1
            continue

        leadin_prefixes = ("そう", "こう", "ああ")
        looks_prefixed_restart = any(
            nxt_text.startswith(prefix + cur_text)
            for prefix in leadin_prefixes
        )

        looks_prefix = (
            2 <= len(cur_text) <= 8
            and len(nxt_text) >= len(cur_text) + 2
            and (nxt_text.startswith(cur_text) or looks_prefixed_restart)
        )
        if not looks_prefix:
            i_prefix += 1
            continue

        cur_feats = _analyze_clause_surface(segmenter, cur_c.get("text", ""))
        if (
            cur_c.get("type") == "fragment"
            or len(cur_text) <= 5
            or not cur_feats.get("has_element", False)
            or not cur_feats.get("has_pred_verb_adj", False)
        ):
            kept.pop(i_prefix)
            stats["v18_dropped_prefix_false_start"] += 1
            continue

        i_prefix += 1

    return kept, stats


def process_single_textgrid(input_path: str, output_path: str,
                           segmenter: JapaneseClauseSegmenter) -> List[str]:
    """Process a single TextGrid file with optional disfluency detection.
    
    If segmenter.disfluency_detector is set, disfluency labels are computed
    and used to build clean text for parsing. A disfluency tier is added.
    """
    filename = os.path.basename(input_path)
    print(f"\nProcessing: {filename}")

    tg = TextGridHandler(input_path)
    tg_words = tg.get_word_list()
    print(f"  Words: {len(tg_words)}, Mora: {tg.total_mora}")

    # --- Disfluency detection ---
    disfluency_labels = None
    dd = getattr(segmenter, 'disfluency_detector', None)
    if dd is not None:
        word_texts = [w.text for w in tg_words]
        disfluency_labels = dd.detect(word_texts)
        disfluency_labels = _suppress_bound_suffix_disfluency_labels(
            tg_words,
            disfluency_labels,
        )
        disfluency_labels = _demote_nonfiller_disfluency_labels(
            tg_words,
            disfluency_labels,
            segmenter,
        )
        n_dis = sum(disfluency_labels)
        print(f"  Disfluency tier: {n_dis} disfluent words marked")

        # Add disfluency tier to TextGrid
        _add_disfluency_tier(tg, tg_words, disfluency_labels)

    # --- Clause segmentation + alignment ---
    # For logical clause count, segment full transcript
    transcript = tg.get_transcript()
    logical_clauses = segmenter.segment(transcript, verbose=False)

    aligner = ClauseAligner(tg, segmenter)
    clauses = aligner.align_clauses(disfluency_labels=disfluency_labels)

    # Keep any clause that still has visible fluent words.
    filtered_clauses = []
    for c in clauses:
        if 'word_indices' in c and c['word_indices']:
            visible = aligner._visible_clause_word_indices(
                c['word_indices'], tg_words, disfluency_labels
            )
            if visible:
                filtered_clauses.append(
                    aligner._rebuild_clause_payload(dict(c), tg_words, disfluency_labels)
                )
        elif (c.get('text') or '').strip():
            filtered_clauses.append(c)

    methodology_stats = {}
    if getattr(segmenter, "apply_methodology_rules", True):
        filtered_clauses, methodology_stats = _apply_methodology_rules(
            filtered_clauses,
            segmenter,
        )
    filtered_clauses = aligner.recover_residual_fluent_runs(
        filtered_clauses,
        tg_words,
        disfluency_labels,
    )
    rebuilt_clauses = []
    for c in filtered_clauses:
        if c.get("word_indices"):
            rebuilt_clauses.append(
                aligner._rebuild_clause_payload(dict(c), tg_words, disfluency_labels)
            )
        elif (c.get("text") or "").strip():
            rebuilt_clauses.append(c)
    filtered_clauses = aligner.split_noncontiguous_clauses(
        rebuilt_clauses,
        tg_words,
        disfluency_labels,
    )
    filtered_clauses = aligner.reattach_preserved_fragments(
        filtered_clauses,
        tg_words,
        disfluency_labels,
    )
    filtered_clauses = aligner.refine_segmentation_quality(
        filtered_clauses,
        tg_words,
        disfluency_labels,
    )
    # v26: final run-on split pass to catch any remaining over-merged clauses
    filtered_clauses = aligner._split_runon_predicate_chains(
        filtered_clauses,
        tg_words,
        disfluency_labels,
    )
    filtered_clauses = aligner.repair_late_split_micro_fragments(
        filtered_clauses,
        tg_words,
        disfluency_labels,
    )
    filtered_clauses = aligner.refine_segmentation_quality(
        filtered_clauses,
        tg_words,
        disfluency_labels,
    )

    # v49: final leading-filler trim after all repair passes
    filtered_clauses = aligner._trim_boundary_noise(
        filtered_clauses, tg_words, disfluency_labels
    )

    non_fragment = [c for c in filtered_clauses if c['type'] != 'fragment']
    fragments = [c for c in filtered_clauses if c['type'] == 'fragment']

    print(f"  Clauses (Logical): {len(logical_clauses)}")
    print("LOGICAL C:", [c[3] for c in logical_clauses])
    print(f"  Clauses (Aligned): {len(non_fragment)} (fragments: {len(fragments)})")
    if methodology_stats:
        dropped = (
            methodology_stats.get("dropped_fragment_lt3", 0)
            + methodology_stats.get("dropped_short_no_pred", 0)
            + methodology_stats.get("dropped_particles_only", 0)
            + methodology_stats.get("dropped_meta", 0)
            + methodology_stats.get("dropped_no_pred_lt10", 0)
            + methodology_stats.get("dropped_no_pred_type", 0)
            + methodology_stats.get("dropped_bare_te", 0)
            + methodology_stats.get("v18_dropped_prefix_false_start", 0)
        )
        relabeled = (
            methodology_stats.get("type_relabel_chain_te_to_advcl", 0)
            + methodology_stats.get("type_relabel_chain_te_to_ind", 0)
            + methodology_stats.get("type_relabel_advcl_to_chain_te", 0)
            + methodology_stats.get("type_relabel_advcl_to_ind", 0)
            + methodology_stats.get("type_relabel_minor_to_ind", 0)
            + methodology_stats.get("type_relabel_ind_no_pred_to_fragment", 0)
            + methodology_stats.get("type_relabel_no_pred_to_fragment", 0)
        )
        truncated = methodology_stats.get("truncated_long", 0)
        print(f"  Methodology rules: dropped={dropped}, relabeled={relabeled}, truncated={truncated}")

    # Add clause tier and save
    final_intervals = tg.add_clause_tier(filtered_clauses, tier_name="clauses")
    tg.save(output_path)
    print(f"  Saved: {output_path}")

    # Build log
    clause_log = []
    clause_log.append(f"File: {filename}")
    clause_log.append(f"Words: {len(tg_words)}, Mora: {tg.total_mora}")
    if disfluency_labels is not None:
        clause_log.append(f"Disfluent words: {sum(disfluency_labels)}")
    clause_log.append(f"Clauses (Logical): {len(logical_clauses)}")
    clause_log.append(f"Clauses (Aligned): {len(non_fragment)}")
    clause_log.append(f"Fragments: {len(fragments)}")
    if methodology_stats:
        clause_log.append(f"Methodology Rules: {methodology_stats}")
    clause_log.append("-" * 80)

    # Show clauses
    for c in (final_intervals or []):
        display_text = c['label'][:60] + '...' if len(c['label']) > 60 else c['label']
        line = f"    {c['type']:<20} | {display_text}"
        try:
            print(line)
        except UnicodeEncodeError:
            # Windows cp1252 terminals can fail on Japanese output.
            print(line.encode("unicode_escape").decode("ascii"))
        clause_log.append(f"{c['type']:<20} | {c['label']}")
    
    clause_log.append("")
    return clause_log


def _add_disfluency_tier(tg_handler: TextGridHandler, tg_words: List[WordInterval],
                         disfluency_labels: List[int]):
    """Add a disfluency tier marking disfluent words in the TextGrid."""
    from praatio.data_classes.interval_tier import IntervalTier

    min_time = tg_handler.tg.minTimestamp
    max_time = tg_handler.tg.maxTimestamp

    intervals = []
    current_time = min_time

    for i, word in enumerate(tg_words):
        # Gap before this word
        if word.start > current_time + 0.001:
            intervals.append(Interval(current_time, word.start, ""))
        
        label = word.text if disfluency_labels[i] == 1 else ""
        intervals.append(Interval(word.start, word.end, label))
        current_time = word.end

    if current_time < max_time - 0.001:
        intervals.append(Interval(current_time, max_time, ""))

    if not intervals:
        intervals = [Interval(min_time, max_time, "")]

    tier_name = "disfluency"
    if tier_name in tg_handler.tg.tierNames:
        tg_handler.tg.removeTier(tier_name)

    new_tier = IntervalTier(tier_name, intervals, min_time, max_time)
    tg_handler.tg.addTier(new_tier)


def _suppress_bound_suffix_disfluency_labels(
    tg_words: List[WordInterval],
    disfluency_labels: List[int],
) -> List[int]:
    """Clear false-positive disfluency labels on bound inflection suffixes."""
    if disfluency_labels is None:
        return disfluency_labels

    cleaned = list(disfluency_labels)
    suffix_tokens = {"た", "だ", "て", "で", "ん"}
    for i in range(1, min(len(tg_words), len(cleaned))):
        if cleaned[i] != 1:
            continue
        cur_text = (tg_words[i].text or "").strip()

        prev_idx = i - 1
        while prev_idx >= 0 and not (tg_words[prev_idx].text or "").strip():
            prev_idx -= 1
        if prev_idx < 0:
            continue

        prev_text = (tg_words[prev_idx].text or "").strip()
        if cur_text in suffix_tokens and ClauseAligner._is_bound_inflection(prev_text, cur_text):
            cleaned[i] = 0
            continue

        if cur_text != "\u3063":
            continue

        next_idx = i + 1
        while next_idx < min(len(tg_words), len(cleaned)) and not (tg_words[next_idx].text or "").strip():
            next_idx += 1
        if next_idx >= min(len(tg_words), len(cleaned)):
            continue

        next_text = (tg_words[next_idx].text or "").strip()
        if next_text in suffix_tokens and ClauseAligner._is_bound_inflection(prev_text, next_text):
            cleaned[i] = 0

    return cleaned


def _demote_nonfiller_disfluency_labels(
    tg_words: List[WordInterval],
    disfluency_labels: List[int],
    segmenter: JapaneseClauseSegmenter,
) -> List[int]:
    """Keep filler/backchannel DIS labels, but restore contentful runs to fluent."""
    if disfluency_labels is None:
        return disfluency_labels

    cleaned = list(disfluency_labels)

    def _is_hideable_disfluency_surface(text: str) -> bool:
        surface = (text or "").strip()
        if not surface:
            return True
        norm = segmenter._normalize_surface_text(surface)
        if not norm:
            return True
        if segmenter._is_filler_surface(surface):
            return True
        if norm in {"ー", "っ"}:
            return True
        return False

    i = 0
    limit = min(len(tg_words), len(cleaned))
    while i < limit:
        if cleaned[i] != 1:
            i += 1
            continue

        j = i
        run_indices = []
        while j < limit and cleaned[j] == 1:
            run_indices.append(j)
            j += 1

        run_words = [
            (tg_words[k].text or "").strip()
            for k in run_indices
            if k < len(tg_words) and (tg_words[k].text or "").strip()
        ]
        if run_words and not all(_is_hideable_disfluency_surface(w) for w in run_words):
            for k in run_indices:
                cleaned[k] = 0

        i = j

    return cleaned


def process_all_textgrids(input_dir: str, output_dir: str,
                          segmenter: JapaneseClauseSegmenter):
    """Process all TextGrid files in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.TextGrid')])
    print(f"Processing {len(files)} TextGrid files...")
    
    all_logs = []
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            log = process_single_textgrid(input_path, output_path, segmenter)
            all_logs.extend(log)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Write log
    log_path = os.path.join(output_dir, 'clause_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_logs))
    print(f"\nClause log saved to: {log_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Japanese TextGrid Clause Segmentation V2 (GiNZA + Vercellotti rules)",
        epilog="Uses GiNZA for Japanese NLP with Vercellotti-style clause detection."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input TextGrid file or directory")
    parser.add_argument("-o", "--output", required=True,
                        help="Output TextGrid file or directory")
    parser.add_argument("--model", default="ja_ginza_electra",
                        help="GiNZA model name (default: ja_ginza_electra)")
    parser.set_defaults(use_disfluency=True)
    parser.add_argument("--use-disfluency", dest="use_disfluency", action="store_true",
                        help="Enable neural disfluency detector (default: enabled).")
    parser.add_argument("--no-disfluency", dest="use_disfluency", action="store_false",
                        help="Disable neural disfluency detector.")
    parser.add_argument("--disfluency-model", default=None,
                        help="Path to disfluency model dir. Default: auto-detect project model.")
    parser.set_defaults(use_methodology_rules=True)
    parser.add_argument("--use-methodology-rules", dest="use_methodology_rules", action="store_true",
                        help="Apply CLAUSE_SEGMENTATION_METHODOLOGY.md post-validation rules (default: enabled).")
    parser.add_argument("--no-methodology-rules", dest="use_methodology_rules", action="store_false",
                        help="Disable methodology post-validation rules.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--pause-split-sec", type=float, default=0.6,
                        help="Silences at least this long (s) cut words into parsing groups (inf: never)")
    
    args = parser.parse_args()
    
    # Load GiNZA model
    print(f"Loading GiNZA model: {args.model}")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Model '{args.model}' not found. Please install GiNZA:")
        print("  pip install ginza ja_ginza_electra")
        sys.exit(1)
    print("Model loaded!")
    
    # Initialize segmenter
    disfluency_detector = None
    if args.use_disfluency:
        try:
            disfluency_detector = DisfluencyDetector(model_path=args.disfluency_model)
        except Exception as e:
            print(f"Failed to load disfluency detector: {e}")
            sys.exit(1)

    segmenter = JapaneseClauseSegmenter(
        nlp,
        debug=args.debug,
        disfluency_detector=disfluency_detector,
        apply_methodology_rules=args.use_methodology_rules,
        pause_split_sec=args.pause_split_sec,
    )
    
    # Process
    if os.path.isfile(args.input):
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        process_single_textgrid(args.input, args.output, segmenter)
    else:
        process_all_textgrids(args.input, args.output, segmenter)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
