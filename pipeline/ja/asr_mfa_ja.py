#!/usr/bin/env python
"""
Qwen3-ASR + Filler-Augmented MFA Pipeline (Japanese)

Pipeline:
  1. Qwen3 ASR + ForcedAligner -> full transcript + rough word timestamps
     (with disfluency prompt to capture fillers)
  2. Filler-augmented MFA with high beam (--beam 100 --retry_beam 400):
     inject filler placeholders into gaps, run MFA on full file.
     High beam prevents alignment drift on long files (>30s).
     Filler injection prevents stretching over pause regions.
  3. Build final TextGrid + JSON

The runner calls the two halves separately (asr_to_pre_mfa_json.py, then
mfa_from_pre_mfa_json.py). This combined script runs both in one go:
    python asr_mfa_ja.py -i <audio folder> -o <output folder> [--file NAME ...]

Requires:
  - qwen3-asr conda env (for ASR + rough alignment)
  - mfa conda env (for precise alignment, called as subprocess)
"""

import os
import sys
import io
import re
import json
import math
import argparse
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import soundfile as sf
import librosa
from praatio import textgrid
from praatio.utilities.constants import Interval


# Line-buffered, so each line reaches the runner when printed (per-file
# progress events); a plain TextIOWrapper held output back in 8 KB blocks.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# ── Config ──────────────────────────────────────────────────────────────────
ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"
SAMPLE_RATE = 16000
AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg')
PAD_SEC = 0.15
FILLER_CONTEXT = (
    "Transcribe Japanese speech verbatim and keep ALL disfluencies.\n"
    "Rules:\n"
    "1) Keep fillers exactly as spoken.\n"
    "2) Keep repetitions and false starts exactly as spoken.\n"
    "3) Do not normalize or rewrite wording.\n"
    "4) Keep names, numbers and alphanumeric forms exactly as spoken.\n"
    "5) If uncertain, preserve hesitation tokens instead of deleting.\n"
    "Examples:\n"
    "- Input speech: あの えー きのう こうえんに いきました\n"
    "  Output text: あの えー きのう こうえんに いきました\n"
    "- Input speech: えっと ぼ ぼくは ともだちと\n"
    "  Output text: えっと ぼ ぼくは ともだちと\n"
    "- Input speech: そのー それで うーん いえに かえりました\n"
    "  Output text: そのー それで うーん いえに かえりました\n"
    "- Input speech: うん うん それで はじめました\n"
    "  Output text: うん うん それで はじめました\n"
    "- Input speech: あ あの ひとが きました\n"
    "  Output text: あ あの ひとが きました\n"
    "- Input speech: いっ いったん やめて つづけます\n"
    "  Output text: いっ いったん やめて つづけます\n"
    "- Input speech: その その ばしょに いきました\n"
    "  Output text: その その ばしょに いきました\n"
    "- Input speech: ま あ ちがう まあ そうです\n"
    "  Output text: ま あ ちがう まあ そうです\n"
    "- Input speech: うーん でも そのー むずかしいです\n"
    "  Output text: うーん でも そのー むずかしいです\n"
    "- Input speech: えーと えっと それで つぎに\n"
    "  Output text: えーと えっと それで つぎに\n"
    "- Input speech: にじゅう さん にん でした\n"
    "  Output text: にじゅう さん にん でした"
)

# MFA config
MFA_ACOUSTIC = "japanese_mfa"
MFA_DICT = "japanese_mfa"

# Filler injection params
# Keep placeholders less lexical to reduce accidental token consumption.
FILLER_TOKENS_SHORT = [
    "\u3048\u30fc",        # えー
    "\u3042\u30fc",        # あー
    "\u3093\u30fc",        # んー
]
FILLER_TOKENS_MEDIUM = [
    "\u3048\u3063\u3068",  # えっと
    "\u3048\u30fc\u3068",  # えーと
    "\u3046\u30fc\u3093",  # うーん
    "\u3048\u30fc",        # えー
    "\u3042\u30fc",        # あー
    "\u3093\u30fc",        # んー
]
FILLER_TOKENS_LONG = [
    "\u3048\u3063\u3068",  # えっと
    "\u3048\u30fc\u3068",  # えーと
    "\u3046\u30fc\u3093",  # うーん
    "\u3048\u3063\u3068",  # えっと
    "\u3048\u30fc\u3068",  # えーと
    "\u3048\u30fc",        # えー
    "\u3042\u30fc",        # あー
    "\u3093\u30fc",        # んー
]


def _resolve_mfa_paths() -> Tuple[str, str]:
    """Find MFA in a portable way across Windows/macOS/other Conda layouts."""
    env_root = os.environ.get("CLAUSA_MFA_PREFIX")
    env_name = os.environ.get("CLAUSA_MFA_ENV_NAME", "mfa")
    explicit_exe = os.environ.get("CLAUSA_MFA_EXE")
    explicit_lib_bin = os.environ.get("CLAUSA_MFA_LIB_BIN", "")

    if explicit_exe:
        return explicit_exe, explicit_lib_bin

    candidate_roots: List[Path] = []
    if env_root:
        candidate_roots.append(Path(env_root))

    home = Path.home()
    for base_name in ("miniconda3", "mambaforge", "anaconda3", "micromamba"):
        candidate_roots.append(home / base_name / "envs" / env_name)

    for root in candidate_roots:
        if sys.platform.startswith("win"):
            exe = root / "Scripts" / "mfa.exe"
            lib_bin = root / "Library" / "bin"
        else:
            exe = root / "bin" / "mfa"
            lib_bin = root / "lib"
        if exe.exists():
            return str(exe), (str(lib_bin) if lib_bin.exists() else "")

    path_hit = shutil.which("mfa")
    if path_hit:
        lib_bin = ""
        if sys.platform.startswith("win"):
            exe_path = Path(path_hit)
            candidate = exe_path.parents[1] / "Library" / "bin"
            if candidate.exists():
                lib_bin = str(candidate)
        return path_hit, lib_bin

    if sys.platform.startswith("win"):
        fallback_root = home / "miniconda3" / "envs" / env_name
        return str(fallback_root / "Scripts" / "mfa.exe"), str(fallback_root / "Library" / "bin")

    fallback_root = home / "miniconda3" / "envs" / env_name
    return str(fallback_root / "bin" / "mfa"), str(fallback_root / "lib")


MFA_EXE, MFA_LIB_BIN = _resolve_mfa_paths()
GAP_MIN = 0.4         # minimum gap (s) to inject a filler
GAP_OFFSET = 0.35     # offset for filler count formula
GAP_STEP = 0.55       # step for additional fillers
FILLER_MAX = 3        # max fillers per gap


# ── Step 1: Qwen3 ASR ──────────────────────────────────────────────────────

def resolve_asr_device(device: str) -> Tuple[str, torch.dtype]:
    """Resolve CLI device choice to a torch-compatible device/dtype pair."""
    device = str(device).strip().lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        return "cpu", torch.float32

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA device is available. "
                "Use --device auto or --device cpu instead."
            )
        return "cuda:0", torch.bfloat16

    if device == "cpu":
        return "cpu", torch.float32

    raise ValueError(f"Unsupported device option: {device}")


ASR_BACKEND = "qwen-asr (PyTorch)"  # set by load_models; recorded in the Stage 1 JSON


def _asr_mlx():
    """pipeline/shared/asr_mlx.py, also when this script runs without pipeline/ on PYTHONPATH."""
    pipeline_dir = str(Path(__file__).resolve().parents[1])
    if pipeline_dir not in sys.path:
        sys.path.append(pipeline_dir)
    from shared import asr_mlx
    return asr_mlx


def _use_mlx(device: str) -> bool:
    """MLX on the Mac GPU: asked for, or by default on Apple Silicon."""
    device = str(device).strip().lower()
    if device not in ("auto", "mlx"):
        return False
    asr_mlx = _asr_mlx()
    if device == "mlx":
        if not asr_mlx.available():
            raise RuntimeError("--device mlx needs an Apple Silicon Mac with mlx-qwen3-asr installed.")
        return True
    return asr_mlx.available()


def load_models(device: str = "auto"):
    """Load ASR model with integrated ForcedAligner for chunked alignment."""
    global ASR_BACKEND
    if _use_mlx(device):
        print("Loading Qwen3-ASR model (with integrated aligner) through MLX on the Mac GPU...")
        asr = _asr_mlx().MLXQwen3ASR(ASR_MODEL_PATH, FORCED_ALIGNER_PATH)
        ASR_BACKEND = asr.backend
        print("Models loaded!")
        return asr
    resolved_device, dtype = resolve_asr_device(device)
    ASR_BACKEND = f"qwen-asr (PyTorch), {resolved_device}, {str(dtype).replace('torch.', '')}"
    print(f"Loading Qwen3-ASR model (with integrated aligner) on {resolved_device}...")
    from qwen_asr import Qwen3ASRModel  # deferred so Stage 2 can import this module without the ASR stack

    asr = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        forced_aligner=FORCED_ALIGNER_PATH,
        forced_aligner_kwargs={"dtype": dtype, "device_map": resolved_device},
        dtype=dtype,
        device_map=resolved_device,
        max_inference_batch_size=32,
        max_new_tokens=4096,
    )
    print("Models loaded!")
    return asr


def step1_transcribe(audio_path: str, asr_model,
                     language: str = "Japanese") -> Tuple[List[Dict], float, str]:
    """Step 1: ASR + chunked forced alignment via return_time_stamps.
    Uses FILLER_CONTEXT to encourage disfluency transcription."""
    info = sf.info(audio_path)
    duration = info.duration

    results = asr_model.transcribe(
        audio=audio_path, language=language, context=FILLER_CONTEXT,
        return_time_stamps=True)
    r = results[0]
    transcript = r.text or ""

    if not transcript.strip():
        return [], duration, ""

    words = []
    if r.time_stamps:
        for ts in r.time_stamps:
            text = ts.text.strip()
            if text:
                words.append({
                    'word': text,
                    'start': ts.start_time,
                    'end': ts.end_time,
                })

    return words, duration, transcript


# ── Step 1.5: MeCab re-tokenization ─────────────────────────────────────────

def retokenize_with_mecab(words: List[Dict], tagger) -> List[Dict]:
    """Re-tokenize ASR words using MeCab morphological analysis.
    Splits merged tokens like '選びえあ' -> ['選び', 'え', 'あ'].
    Distributes rough timestamps proportionally by character count."""
    new_words = []
    for w in words:
        text = w['word']
        start = float(w['start'])
        end = float(w['end'])
        dur = end - start

        morphemes = tagger(text)
        surfaces = [m.surface for m in morphemes if m.surface.strip()]

        if len(surfaces) <= 1:
            new_words.append(w.copy())
            continue

        # Verify concatenation matches original
        if ''.join(surfaces) != text:
            new_words.append(w.copy())
            continue

        # Distribute timestamps proportionally by character count
        total_chars = sum(len(s) for s in surfaces)
        t = start
        for s in surfaces:
            frac = len(s) / total_chars
            s_end = t + dur * frac
            new_words.append({
                'word': s,
                'start': round(t, 3),
                'end': round(s_end, 3),
            })
            t = s_end

    return new_words


# ── Step 2: Filler-Augmented MFA (full-file, high beam) ─────────────────────

def _norm_token(s: str) -> str:
    """Normalize token for MFA: strip whitespace and common punctuation."""
    s = str(s).strip()
    # Remove Japanese and Western punctuation from edges
    s = re.sub(
        r'^[\s\u3000\u3001\u3002\uff0c\uff0e\uff01\uff1f\u300c\u300d\u300e\u300f'
        r'\u3008\u3009\uff08\uff09.,!?\-\"\'\(\)]+', '', s)
    s = re.sub(
        r'[\s\u3000\u3001\u3002\uff0c\uff0e\uff01\uff1f\u300c\u300d\u300e\u300f'
        r'\u3008\u3009\uff08\uff09.,!?\-\"\'\(\)]+$', '', s)
    return s


def _filler_count_for_gap(gap: float) -> int:
    """How many filler tokens to inject for a given gap duration."""
    if gap < GAP_MIN:
        return 0
    k = math.floor((gap - GAP_OFFSET) / GAP_STEP) + 1
    if k < 0:
        return 0
    return min(FILLER_MAX, k)


def _choose_filler_token(gap: float, left_token: str, right_token: str, idx: int) -> str:
    """Choose a filler token deterministically from a gap-length palette."""
    if gap < 0.8:
        pool = FILLER_TOKENS_SHORT
    elif gap < 1.6:
        pool = FILLER_TOKENS_MEDIUM
    else:
        pool = FILLER_TOKENS_LONG

    # Keep deterministic behavior across runs.
    sig = f"{left_token}|{right_token}|{int(round(gap * 100))}|{idx}"
    h = sum(ord(ch) for ch in sig)
    return pool[h % len(pool)]


# Each MFA call gets its own copy of the MFA root, so parallel workers do not
# write the shared global_config.yaml at the same time (without it,
# --mfa-workers > 1 silently loses files to the rough-timestamp fallback).
def _prepare_isolated_mfa_root(env: Dict[str, str]) -> Tuple[Dict[str, str], Optional[Path]]:
    source_root_str = env.get("MFA_ROOT_DIR")
    if not source_root_str:
        return env, None
    source_root = Path(source_root_str)
    if not source_root.exists():
        return env, None

    runtime_base = os.environ.get("CLAUSA_MFA_RUNTIME_BASE") or tempfile.gettempdir()
    runtime_root = Path(tempfile.mkdtemp(prefix="clausa_mfa_root_", dir=runtime_base))

    for relative_path in (
        "global_config.yaml",
        "pretrained_models/cache.json",
        f"pretrained_models/acoustic/{MFA_ACOUSTIC}.zip",
        f"pretrained_models/dictionary/{MFA_DICT}.dict",
    ):
        source = source_root / relative_path
        if not source.exists():
            continue
        target = runtime_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    isolated = env.copy()
    isolated["MFA_ROOT_DIR"] = str(runtime_root)
    return isolated, runtime_root


def _write_pruned_dictionary(corpus_dir: str, env: Dict[str, str]) -> Optional[str]:
    """Write a copy of the MFA dictionary restricted to words that can occur in
    this utterance, and return its path (None to use the full dictionary).

    MFA loads and compiles the whole dictionary on every call (most of its run
    time for a single file), but forced alignment only uses the words of the
    transcript. A dictionary line is kept if its word, raw or NFKC-normalised
    and lowercased, is any substring of the transcript (raw, NFKC, lowercase,
    with and without spaces), which covers MFA's own tokenisation, clitic
    splitting and normalisation. Kept lines are copied verbatim. Set
    CLAUSA_MFA_FULL_DICTIONARY=1 to use the full dictionary.
    """
    if os.environ.get("CLAUSA_MFA_FULL_DICTIONARY", "").strip() == "1":
        return None
    root = env.get("MFA_ROOT_DIR")
    full = Path(root) / "pretrained_models" / "dictionary" / f"{MFA_DICT}.dict" if root else None
    lab = Path(corpus_dir) / "utt.lab"
    if full is None or not full.exists() or not lab.exists():
        return None

    text = lab.read_text(encoding="utf-8").strip()
    nfkc = unicodedata.normalize("NFKC", text)
    substrings = set()
    for variant in {text, nfkc, text.lower(), nfkc.lower()}:
        for t in (variant, variant.replace(" ", "")):
            for i in range(len(t)):
                for j in range(i + 1, min(len(t), i + 40) + 1):
                    substrings.add(t[i:j])

    out = Path(corpus_dir).parent / f"{MFA_DICT}_pruned.dict"
    with full.open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            word = line.split("\t", 1)[0]
            norm = unicodedata.normalize("NFKC", word)
            if (word in substrings or word.lower() in substrings
                    or norm in substrings or norm.lower() in substrings):
                dst.write(line)
    return str(out)


def _run_mfa(
    corpus_dir: str,
    output_dir: str,
    output_format: str = "json",
    num_jobs: int = 1,
    temporary_directory: str | None = None,
) -> bool:
    """Run MFA align as subprocess."""
    env = os.environ.copy()
    if MFA_LIB_BIN and os.path.isdir(MFA_LIB_BIN):
        env["PATH"] = MFA_LIB_BIN + os.pathsep + env.get("PATH", "")
    env, runtime_root = _prepare_isolated_mfa_root(env)
    dictionary = _write_pruned_dictionary(corpus_dir, env) or MFA_DICT
    cmd = [
        MFA_EXE, "align",
        corpus_dir, dictionary, MFA_ACOUSTIC, output_dir,
        "--clean", "--single_speaker",
        "--overwrite", "--output_format", output_format,
        "--beam", "100", "--retry_beam", "400",
        "--num_jobs", str(num_jobs),
    ]
    if temporary_directory:
        cmd.extend(["--temporary_directory", temporary_directory])
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            ext = ".json" if output_format == "json" else ".TextGrid"
            out_files = list(Path(output_dir).glob(f"*{ext}"))
            if out_files:
                return True
            print(f"    MFA error: {result.stderr[-500:] if result.stderr else 'no stderr'}")
            return False
        return True
    finally:
        if runtime_root is not None:
            shutil.rmtree(runtime_root, ignore_errors=True)


def _parse_mfa_json(jpath: str) -> List[Dict]:
    """Parse word entries from MFA JSON output."""
    data = json.load(open(jpath, encoding='utf-8'))
    entries = data.get('tiers', {}).get('words', {}).get('entries', [])
    out = []
    for s, e, w in entries:
        w = str(w)
        if w == '' or w == '<eps>':
            continue
        out.append({'word': w, 'start': float(s), 'end': float(e),
                     'token': _norm_token(w)})
    return out


def _build_augmented_sequence(words: List[Dict], duration: float) -> Tuple[List[Dict], int, int, int]:
    """Build token sequence with filler placeholders injected into gaps.
    Returns (sequence, n_injected, n_leading, n_trailing).
    Each item: {'token': str, 'raw_index': int|None, 'kind': 'raw'|'filler'}"""
    seq = []
    injected = 0
    leading_injected = 0
    trailing_injected = 0

    # Leading gap fillers
    first_tok = _norm_token(words[0].get('word', ''))
    lead_gap = max(0.0, float(words[0]['start']))
    k_lead = _filler_count_for_gap(lead_gap)
    for j in range(k_lead):
        filler_tok = _choose_filler_token(lead_gap, "<BOS>", first_tok, j)
        seq.append({'token': filler_tok, 'raw_index': None, 'kind': 'filler'})
        injected += 1
        leading_injected += 1

    for i, w in enumerate(words):
        tok = _norm_token(w.get('word', ''))
        if tok:
            seq.append({'token': tok, 'raw_index': i, 'kind': 'raw'})
        if i + 1 < len(words):
            gap = float(words[i + 1]['start']) - float(words[i]['end'])
            k = _filler_count_for_gap(gap)
            right_tok = _norm_token(words[i + 1].get('word', ''))
            for j in range(k):
                filler_tok = _choose_filler_token(gap, tok, right_tok, j)
                seq.append({'token': filler_tok, 'raw_index': None, 'kind': 'filler'})
                injected += 1

    # Trailing gap fillers
    last_tok = _norm_token(words[-1].get('word', ''))
    tail_gap = max(0.0, float(duration) - float(words[-1]['end']))
    k_tail = _filler_count_for_gap(tail_gap)
    for j in range(k_tail):
        filler_tok = _choose_filler_token(tail_gap, last_tok, "<EOS>", j)
        seq.append({'token': filler_tok, 'raw_index': None, 'kind': 'filler'})
        injected += 1
        trailing_injected += 1

    return seq, injected, leading_injected, trailing_injected


def _map_back_to_raw(raw_words: List[Dict], seq: List[Dict],
                     mfa_words: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Map MFA-aligned words back to original raw words.
    Filler tokens are consumed (discarded), real words get MFA timestamps.
    Any unmatched raw words fall back to rough timestamps.

    Japanese-aware: handles MFA lowercasing and token re-splitting.
    E.g. ASR token 'え学校' may become MFA tokens 'え'+'学校'.
    Uses greedy concatenation: try consuming 1..N MFA tokens to match."""
    aligned = [None] * len(raw_words)
    spans = []  # [token, start, end] of each placeholder as MFA aligned it
    j = 0
    consumed = 0
    skipped = 0
    MAX_CONCAT = 5  # max MFA tokens to concatenate for one ASR token

    for item in seq:
        expected = item['token'].lower()
        if j >= len(mfa_words):
            continue

        # Try exact match first
        if mfa_words[j]['token'].lower() == expected:
            hit_start = float(mfa_words[j]['start'])
            hit_end = float(mfa_words[j]['end'])
            j += 1
        else:
            # Try concatenating consecutive MFA tokens
            matched = False
            for n in range(2, MAX_CONCAT + 1):
                if j + n > len(mfa_words):
                    break
                concat = ''.join(mfa_words[j + k]['token'].lower() for k in range(n))
                if concat == expected:
                    hit_start = float(mfa_words[j]['start'])
                    hit_end = float(mfa_words[j + n - 1]['end'])
                    j += n
                    matched = True
                    break
            if not matched:
                # Skip MFA tokens until we find a match
                orig_j = j
                while j < len(mfa_words) and mfa_words[j]['token'].lower() != expected:
                    j += 1
                if j < len(mfa_words):
                    skipped += (j - orig_j)
                    hit_start = float(mfa_words[j]['start'])
                    hit_end = float(mfa_words[j]['end'])
                    j += 1
                else:
                    # No match found at all, reset j and skip this seq item
                    j = orig_j
                    skipped += 1
                    continue

        if item['raw_index'] is None:
            consumed += 1
            spans.append([item['token'], hit_start, hit_end])
            continue
        idx = item['raw_index']
        aligned[idx] = {
            'word': raw_words[idx]['word'],
            'start': round(hit_start, 3),
            'end': round(hit_end, 3),
            'source': 'mfa',
        }

    fallback = 0
    for i, w in enumerate(aligned):
        if w is not None:
            continue
        rs = float(raw_words[i]['start'])
        re_ = float(raw_words[i]['end'])
        if re_ <= rs:
            re_ = rs + 0.001
        aligned[i] = {
            'word': raw_words[i]['word'],
            'start': round(rs, 3),
            'end': round(re_, 3),
            'source': 'rough',
        }
        fallback += 1

    stats = {
        'mfa_word_count': len(mfa_words),
        'filler_injected': sum(1 for x in seq if x['kind'] == 'filler'),
        'filler_consumed': consumed,
        'fallback_count': fallback,
        'skipped_mfa_tokens': skipped,
    }
    # Read by shared/placeholder_islands.py; the MFA-failure fallbacks have none.
    stats['placeholder_spans'] = spans
    return aligned, stats


def merge_split_etto(words: List[Dict], max_gap: float = 0.25) -> Tuple[List[Dict], int]:
    """Join an adjacent えっ + と back into the filler えっと.

    The Qwen3 forced aligner splits the filler えっと into two tokens, えっ and
    と (70 of its 92 occurrences on the evaluation corpus). The stray と then
    looks like a particle to the clause segmenter. A pair is joined only when
    the gap between the two words is under max_gap, the pause threshold, so no
    pause is removed, and not when it is the exclamation えっ + quotative と
    (えっ、と思った: punctuation after えっ, or a reporting verb after と).
    Returns the new word list and the number of joins.
    """
    reporting = ("思", "言", "云", "いう", "聞", "考", "感", "叫", "答", "尋")
    out: List[Dict] = []
    merged = 0
    i = 0
    while i < len(words):
        a = words[i]
        b = words[i + 1] if i + 1 < len(words) else None
        nxt = clean_word_text(str(words[i + 2].get("word", ""))) if i + 2 < len(words) else ""
        if (b is not None
                and str(a.get("word", "")).strip() == "えっ"
                and clean_word_text(str(b.get("word", ""))) == "と"
                and not nxt.startswith(reporting)
                and float(b["start"]) - float(a["end"]) < max_gap):
            out.append({
                "word": "えっ" + str(b["word"]),
                "start": a["start"],
                "end": b["end"],
                "source": a.get("source") if a.get("source") == b.get("source") else "rough",
            })
            merged += 1
            i += 2
            continue
        out.append(a)
        i += 1
    return out, merged


def step2_filler_augmented_mfa(
    rough_words: List[Dict],
    audio_path: str,
    duration: float,
    mfa_num_jobs: int = 1,
    mfa_temporary_directory: str | None = None,
) -> Tuple[List[Dict], Dict]:
    """Step 2: Filler-augmented MFA alignment (full-file, high beam).
    1. Inject filler placeholders into gaps between words
    2. Run MFA on the full file with --beam 100 to prevent drift
    3. Map MFA results back to original words, discarding fillers
    """
    if not rough_words:
        return [], {}

    # Build augmented sequence
    seq, injected, injected_leading, injected_trailing = _build_augmented_sequence(
        rough_words, duration
    )
    if not seq:
        return [], {}

    tokens = [x['token'] for x in seq]
    print(f"    {len(rough_words)} words + {injected} filler placeholders = {len(tokens)} tokens")

    # Prepare temp corpus
    tmp = tempfile.mkdtemp(prefix="mfa_filler_", dir=mfa_temporary_directory)
    corpus = os.path.join(tmp, "corpus")
    out = os.path.join(tmp, "output")
    os.makedirs(corpus)
    os.makedirs(out)

    # MFA needs WAV; convert if input is mp3/other format
    corpus_wav = os.path.join(corpus, "utt.wav")
    if audio_path.lower().endswith('.wav'):
        shutil.copy2(audio_path, corpus_wav)
    else:
        audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        sf.write(corpus_wav, audio_data, sr)
    with open(os.path.join(corpus, "utt.lab"), 'w', encoding='utf-8') as f:
        f.write(' '.join(tokens) + '\n')

    # Run MFA with high beam to prevent drift on long files
    print(f"    Running MFA (beam=100)...")
    if not _run_mfa(
        corpus,
        out,
        output_format="json",
        num_jobs=mfa_num_jobs,
        temporary_directory=mfa_temporary_directory,
    ):
        print(f"    MFA failed, falling back to rough timestamps")
        shutil.rmtree(tmp, ignore_errors=True)
        fallback = [{
            'word': w['word'],
            'start': round(float(w['start']), 3),
            'end': round(float(w['end']), 3),
            'source': 'rough',
        } for w in rough_words]
        return fallback, {'fallback_count': len(rough_words)}

    # Parse MFA output and map back
    jpath = os.path.join(out, "utt.json")
    if not os.path.exists(jpath):
        print(f"    MFA output not found, falling back to rough timestamps")
        shutil.rmtree(tmp, ignore_errors=True)
        fallback = [{
            'word': w['word'],
            'start': round(float(w['start']), 3),
            'end': round(float(w['end']), 3),
            'source': 'rough',
        } for w in rough_words]
        return fallback, {'fallback_count': len(rough_words)}

    mfa_words = _parse_mfa_json(jpath)
    aligned, stats = _map_back_to_raw(rough_words, seq, mfa_words)

    shutil.rmtree(tmp, ignore_errors=True)

    n_mfa = sum(1 for w in aligned if w['source'] == 'mfa')
    n_rough = sum(1 for w in aligned if w['source'] == 'rough')
    print(f"    {len(aligned)} words ({n_mfa} MFA, {n_rough} rough fallback)")
    stats['filler_injected_leading'] = injected_leading
    stats['filler_injected_trailing'] = injected_trailing
    print(
        f"    Fillers: {stats['filler_injected']} injected "
        f"(lead={injected_leading}, tail={injected_trailing}), "
        f"{stats['filler_consumed']} consumed"
    )

    return aligned, stats


# ── Step 3: Build outputs ─────────────────────────────────────────────────

def clean_word_text(word: str) -> str:
    """Remove punctuation from word edges (Japanese-aware)."""
    cleaned = re.sub(
        r'^[\s\u3000\u3001\u3002\uff0c\uff0e\uff01\uff1f\u300c\u300d\u300e\u300f'
        r'\u3008\u3009\uff08\uff09.,!?\-\"\'\(\)]+', '', word)
    cleaned = re.sub(
        r'[\s\u3000\u3001\u3002\uff0c\uff0e\uff01\uff1f\u300c\u300d\u300e\u300f'
        r'\u3008\u3009\uff08\uff09.,!?\-\"\'\(\)]+$', '', cleaned)
    return cleaned


def build_final_textgrid(words: List[Dict], duration: float, path: str,
                         clean: bool = False):
    """Build TextGrid with word and pause tiers."""
    entries = []
    last_end = 0.0
    MIN_DUR = 0.001

    for w in words:
        text = clean_word_text(w['word']) if clean else w['word']
        if not text:
            continue

        ws, we = w['start'], w['end']
        if we <= ws:
            we = ws + MIN_DUR
        if ws < last_end:
            ws = last_end
        if we <= ws:
            we = ws + MIN_DUR

        if ws > last_end + 0.001:
            entries.append(Interval(last_end, ws, ""))
        entries.append(Interval(ws, we, text))
        last_end = we

    if last_end < duration:
        entries.append(Interval(last_end, duration, ""))

    tg = textgrid.Textgrid()
    words_tier = textgrid.IntervalTier("words", entries, minT=0, maxT=duration)
    tg.addTier(words_tier)

    pause_entries = []
    for e in entries:
        if e.label.strip() == "" and (e.end - e.start) >= 0.25:
            pause_entries.append(Interval(e.start, e.end, "sp"))
        else:
            pause_entries.append(Interval(e.start, e.end, ""))
    pause_tier = textgrid.IntervalTier("pauses", pause_entries, minT=0, maxT=duration)
    tg.addTier(pause_tier)

    tg.save(path, format="long_textgrid", includeBlankSpaces=True)


def save_json(words: List[Dict], duration: float, transcript: str, path: str,
              stats: Dict = None):
    """Save results to JSON."""
    data = {
        'duration': duration,
        'transcript': transcript,
        'n_words': len(words),
        'words': [{k: v for k, v in w.items()} for w in words],
    }
    if stats:
        data['filler_augmentation'] = stats
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main ────────────────────────────────────────────────────────────────────

def process_file(audio_path: str, asr_model,
                 output_dir: str, language: str = "Japanese"):
    """Full pipeline for one file."""
    basename = os.path.splitext(os.path.basename(audio_path))[0]

    # Step 1: ASR + rough alignment (with filler prompt)
    print(f"  Step 1: ASR + rough alignment...")
    rough_words, duration, transcript = step1_transcribe(
        audio_path, asr_model, language)
    print(f"    Transcript: {len(transcript)} chars, {len(rough_words)} words")

    if not rough_words:
        print(f"    No words detected, skipping.")
        return None

    # Step 1.5: MeCab re-tokenization (split merged filler+word tokens)
    import fugashi

    tagger = fugashi.Tagger()
    retok_words = retokenize_with_mecab(rough_words, tagger)
    n_split = len(retok_words) - len(rough_words)
    if n_split > 0:
        print(f"    MeCab re-tokenization: {len(rough_words)} -> {len(retok_words)} words ({n_split} splits)")

    # Step 2: Filler-augmented MFA alignment (full-file, beam=100)
    print(f"  Step 2: Filler-augmented MFA alignment...")
    precise_words, stats = step2_filler_augmented_mfa(
        retok_words, audio_path, duration)
    precise_words, stats["etto_merged"] = merge_split_etto(precise_words)

    # Step 3: Build outputs
    print(f"  Step 3: Building outputs...")
    tg_dir = os.path.join(output_dir, "textgrids")
    tg_clean_dir = os.path.join(output_dir, "textgrids_clean")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(tg_dir, exist_ok=True)
    os.makedirs(tg_clean_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    build_final_textgrid(precise_words, duration,
                         os.path.join(tg_dir, f"{basename}.TextGrid"), clean=False)
    build_final_textgrid(precise_words, duration,
                         os.path.join(tg_clean_dir, f"{basename}.TextGrid"), clean=True)
    save_json(precise_words, duration, transcript,
              os.path.join(json_dir, f"{basename}.json"), stats=stats)

    n_mfa = sum(1 for w in precise_words if w.get('source') == 'mfa')
    n_rough = sum(1 for w in precise_words if w.get('source') == 'rough')
    return {
        'name': basename,
        'duration': duration,
        'n_rough_words': len(rough_words),
        'n_precise_words': len(precise_words),
        'n_mfa': n_mfa,
        'n_rough_fallback': n_rough,
        'filler_injected': stats.get('filler_injected', 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR + Filler-Augmented MFA Pipeline (Japanese)")
    parser.add_argument("-i", "--input", required=True,
                        help="Input folder with audio files")
    parser.add_argument("-o", "--output", required=True,
                        help="Output folder")
    parser.add_argument("--lang", "-l", default="Japanese",
                        help="Language (default: Japanese)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mlx"], default="auto",
                        help="ASR device: auto (CUDA, else MLX on Apple Silicon, else CPU), cuda, cpu or mlx")
    parser.add_argument("--file", nargs="+", help="Process specific files by name (without extension)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input folder not found: {args.input}")
        sys.exit(1)

    asr_model = load_models(args.device)

    # Get files
    if args.file:
        audio_files = []
        for name in args.file:
            found = False
            for ext in AUDIO_EXTS:
                candidate = f"{name}{ext}"
                if os.path.exists(os.path.join(args.input, candidate)):
                    audio_files.append(candidate)
                    found = True
                    break
            if not found:
                print(f"Warning: File not found: {name}")
        if not audio_files:
            print(f"Error: No files found")
            sys.exit(1)
    else:
        audio_files = sorted([
            f for f in os.listdir(args.input)
            if f.lower().endswith(AUDIO_EXTS)
        ])

    print(f"\nPipeline: Qwen3 ASR -> Filler-Augmented MFA -> TextGrid")
    print(f"Processing {len(audio_files)} files...")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("-" * 60)

    results = []
    failed = []
    for i, audio_file in enumerate(audio_files, 1):
        audio_path = os.path.join(args.input, audio_file)
        basename = os.path.splitext(audio_file)[0]
        print(f"\n[{i}/{len(audio_files)}] {basename}")

        try:
            result = process_file(audio_path, asr_model, args.output, args.lang)
            if result:
                results.append(result)
                print(f"  Done: {result['n_precise_words']} words "
                      f"({result['n_mfa']} MFA, {result['n_rough_fallback']} rough, "
                      f"{result['filler_injected']} fillers injected)")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(basename)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print(f"Done! {len(results)}/{len(audio_files)} files processed")
    if results:
        total_mfa = sum(r['n_mfa'] for r in results)
        total_rough = sum(r['n_rough_fallback'] for r in results)
        total_words = sum(r['n_precise_words'] for r in results)
        total_fillers = sum(r['filler_injected'] for r in results)
        print(f"Total words: {total_words} ({total_mfa} MFA, {total_rough} rough)")
        print(f"Total fillers injected: {total_fillers}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
