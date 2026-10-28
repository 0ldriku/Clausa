#!/usr/bin/env python
"""
Qwen3-ASR + Filler-Augmented MFA Pipeline (English)

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
    python asr_mfa_en.py -i <audio folder> -o <output folder> [--file NAME ...]

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
FILLER_CONTEXT = "IMPORTANT: Transcribe ALL disfluencies exactly as spoken. Include every filled pause, hesitation, repetition, false start, and self-correction. Never clean up or omit any disfluency."

# MFA config
MFA_ACOUSTIC = "english_us_arpa"
MFA_DICT = "english_us_arpa"

# Filler injection params
FILLER_TOKEN = "uh"
GAP_MIN = 0.4        # minimum gap (s) to inject a filler
GAP_OFFSET = 0.35    # offset for filler count formula
GAP_STEP = 0.55      # step for additional fillers
FILLER_MAX = 3        # max fillers per gap



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
                     language: str = "English") -> Tuple[List[Dict], float, str]:
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


# ── Step 2: Filler-Augmented MFA (full-file, high beam) ─────────────────────

def _norm_token(s: str) -> str:
    """Normalize token for MFA: lowercase, strip punctuation edges."""
    s = str(s).strip().lower()
    s = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", s)
    s = re.sub(r"'+", "'", s)
    return s


def _filler_count_for_gap(gap: float) -> int:
    """How many filler tokens to inject for a given gap duration."""
    if gap < GAP_MIN:
        return 0
    k = math.floor((gap - GAP_OFFSET) / GAP_STEP) + 1
    if k < 0:
        return 0
    return min(FILLER_MAX, k)


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


def _build_augmented_sequence(words: List[Dict]) -> Tuple[List[Dict], int]:
    """Build token sequence with filler placeholders injected into gaps.
    Returns (sequence, n_injected).
    Each item: {'token': str, 'raw_index': int|None, 'kind': 'raw'|'filler'}"""
    seq = []
    injected = 0
    for i, w in enumerate(words):
        tok = _norm_token(w.get('word', ''))
        if tok:
            seq.append({'token': tok, 'raw_index': i, 'kind': 'raw'})
        if i + 1 < len(words):
            gap = float(words[i + 1]['start']) - float(words[i]['end'])
            k = _filler_count_for_gap(gap)
            for _ in range(k):
                seq.append({'token': FILLER_TOKEN, 'raw_index': None, 'kind': 'filler'})
                injected += 1
    return seq, injected


def _map_back_to_raw(raw_words: List[Dict], seq: List[Dict],
                     mfa_words: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Map MFA-aligned words back to original raw words.
    Filler tokens are consumed (discarded), real words get MFA timestamps.
    Any unmatched raw words fall back to rough timestamps."""
    aligned = [None] * len(raw_words)
    spans = []  # [token, start, end] of each placeholder as MFA aligned it
    j = 0
    consumed = 0
    skipped = 0

    for item in seq:
        expected = item['token']
        while j < len(mfa_words) and mfa_words[j]['token'] != expected:
            skipped += 1
            j += 1
        if j >= len(mfa_words):
            continue
        hit = mfa_words[j]
        j += 1
        if item['raw_index'] is None:
            consumed += 1
            spans.append([item['token'], float(hit['start']), float(hit['end'])])
            continue
        idx = item['raw_index']
        aligned[idx] = {
            'word': raw_words[idx]['word'],
            'start': round(float(hit['start']), 3),
            'end': round(float(hit['end']), 3),
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
    seq, injected = _build_augmented_sequence(rough_words)
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
        import librosa  # only needed here; slow to import
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
    print(f"    Fillers: {stats['filler_injected']} injected, {stats['filler_consumed']} consumed")

    return aligned, stats


# ── Step 3: Build outputs ─────────────────────────────────────────────────

def clean_word_text(word: str) -> str:
    """Remove punctuation from word edges."""
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', word)
    cleaned = re.sub(r"[^\w']", '', cleaned)
    return cleaned.lower()


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
                 output_dir: str, language: str = "English"):
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

    # Step 2: Segmented filler-augmented MFA alignment
    print(f"  Step 2: Segmented filler-augmented MFA alignment...")
    precise_words, stats = step2_filler_augmented_mfa(
        rough_words, audio_path, duration)

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
        description="Qwen3-ASR + Filler-Augmented MFA Pipeline (English)")
    parser.add_argument("-i", "--input", required=True,
                        help="Input folder with audio files")
    parser.add_argument("-o", "--output", required=True,
                        help="Output folder")
    parser.add_argument("--lang", "-l", default="English",
                        help="Language (default: English)")
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
            result = process_file(audio_path, asr_model,
                                  args.output, args.lang)
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
