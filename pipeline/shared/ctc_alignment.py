"""
Stage 2 with easyaligner (wav2vec2 CTC forced alignment) in place of MFA.

Consumes the same pre_mfa_json as mfa_from_pre_mfa_json.py and writes the same
outputs (textgrids/, textgrids_clean/, json/), so the later stages run
unchanged. What stays the same as the MFA stage:
  - filler placeholders are injected into gaps by the language module's own
    _build_augmented_sequence (can be switched off with --no-placeholders)
  - the full recording is aligned in one pass
  - words that cannot be aligned keep their rough Qwen3 timestamps
What changes: the aligner. Each token is written in the alignment model's
character set (English: lowercase letters; Japanese: the UniDic reading, in
romaji for a Latin-alphabet model or in kana for a Japanese model)
and aligned with easyaligner on the GPU. Since we control the tokenisation,
every aligned token maps back to exactly one sequence item, so no re-splitting
is needed.

CTC alignments are peaky: a word spans only the frames where its characters
fire, so raw word edges are too short and the gaps between words too long,
which would inflate pause counts. By default the edges are refined with the
Silero VAD speech regions (see _refine_edges).
"""

from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Callable

DEFAULT_MODELS = {
    "en": "facebook/wav2vec2-large-960h-lv60-self",
    # Romaji readings aligned with the English model: on 8 tuning recordings the
    # median boundary error was 35 ms, against 133 ms for
    # jonatasgrosman/wav2vec2-large-xlsr-53-japanese with kana readings.
    "ja": "facebook/wav2vec2-large-960h-lv60-self",
}

# Kana missing from the Japanese model's vocabulary in both scripts
KANA_SUBSTITUTES = {"ヂ": "ジ", "ヅ": "ズ", "ヲ": "オ", "ヮ": "ワ", "ヰ": "イ", "ヱ": "エ", "ヵ": "カ"}


def _speller(language: str, vocab: set[str]) -> Callable[[str], str]:
    """Return a function that writes a token in the model's character set."""
    if language != "ja":
        return lambda tok: unicodedata.normalize("NFKC", tok).lower()

    import fugashi
    import jaconv

    tagger = fugashi.Tagger()

    def kana_char(k: str) -> str:
        k = KANA_SUBSTITUTES.get(k, k)
        h = jaconv.kata2hira(k)
        return h if h in vocab else k

    # A model without kana in its vocabulary (e.g. an English character model)
    # gets the reading in romaji instead.
    romaji = "あ" not in vocab and "ア" not in vocab

    def spell(tok: str) -> str:
        tok = unicodedata.normalize("NFKC", tok)
        reading = []
        for word in tagger(tok):
            kana = getattr(word.feature, "kana", None)
            if not kana or kana == "*":
                kana = jaconv.hira2kata(word.surface)
            reading.append(kana)
        reading = "".join(reading)
        if romaji:
            return jaconv.kana2alphabet(jaconv.kata2hira(reading)).replace("-", "").lower()
        return "".join(kana_char(ch) for ch in reading)

    return spell


def _vocab_chars(processor) -> set[str]:
    special = set(processor.tokenizer.all_special_tokens)
    return {
        ch.lower()
        for ch in processor.tokenizer.get_vocab()
        if len(ch) == 1 and ch not in special and ch not in {"|", " "}
    }


def _to_alignment_token(tok: str, spell: Callable[[str], str], vocab: set[str]) -> str:
    return "".join(ch for ch in spell(tok) if ch in vocab)


def _blank_id(processor) -> int:
    vocab = processor.tokenizer.get_vocab()
    return vocab["<blank>"] if "<blank>" in vocab else processor.tokenizer.pad_token_id


def _load_models(language: str, model_id: str | None, device: str):
    import torch
    from easyaligner.vad.silero import load_vad_model
    from transformers import AutoModelForCTC, Wav2Vec2Processor

    model_id = model_id or DEFAULT_MODELS[language]
    model = AutoModelForCTC.from_pretrained(model_id).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    return load_vad_model(), model, processor, model_id


def _span_tokenizer(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def _speech_regions(vad_model, audio_path: Path, min_silence_ms: int) -> list[tuple[float, float]]:
    import librosa
    import torch
    from silero_vad import get_speech_timestamps

    wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    stamps = get_speech_timestamps(
        torch.from_numpy(wav), vad_model, sampling_rate=16000,
        min_silence_duration_ms=min_silence_ms, speech_pad_ms=0, return_seconds=True,
    )
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def _refine_edges(words: list[dict], regions: list[tuple[float, float]], alpha: float) -> int:
    """Refine peaky CTC word edges with VAD speech regions.

    Each aligned word is assigned the speech region that contains its centre.
    Between two aligned words in the same region, the shared boundary is placed
    at the fraction alpha of the CTC gap between them (raw CTC starts come out
    late and ends early, so the true boundary lies inside that gap). A word
    that opens or closes a region is extended to the region edge, never past a
    neighbouring word. Gaps therefore survive only where VAD found silence.
    Rough-fallback words and words outside any region are left alone.
    Returns the number of words changed.
    """
    import bisect

    starts = [r[0] for r in regions]

    def region_of(t: float):
        k = bisect.bisect_right(starts, t) - 1
        return k if k >= 0 and regions[k][0] <= t <= regions[k][1] else None

    orig = [(w["start"], w["end"]) for w in words]
    reg = [region_of((s + e) / 2) if w["source"] == "ctc" else None for w, (s, e) in zip(words, orig)]

    changed = 0
    for i, w in enumerate(words):
        r = reg[i]
        if r is None:
            continue
        s, e = orig[i]
        if i > 0 and reg[i - 1] == r:
            s = orig[i - 1][1] + alpha * (s - orig[i - 1][1])
        elif regions[r][0] < s:
            s = max(regions[r][0], orig[i - 1][1] if i > 0 else 0.0)
        if i + 1 < len(words) and reg[i + 1] == r:
            e = e + alpha * (orig[i + 1][0] - e)
        elif regions[r][1] > e:
            e = min(regions[r][1], orig[i + 1][0] if i + 1 < len(words) else regions[r][1])
        if e <= s:
            continue
        if (round(s, 3), round(e, 3)) != (w["start"], w["end"]):
            w["start"], w["end"] = round(s, 3), round(e, 3)
            changed += 1
    return changed


def align_payloads(
    payloads: list[dict],
    audio_paths: list[Path],
    build_sequence: Callable[[list[dict], float], list[dict]],
    language: str,
    work_dir: Path,
    *,
    model_id: str | None = None,
    device: str = "cuda",
    placeholders: bool = True,
    refine_edges: bool = True,
    min_silence_ms: int = 100,
    boundary_alpha: float = 0.2,
    batch_size: int = 8,
    keep_temp: bool = False,
) -> tuple[list[tuple[list[dict], dict]], dict]:
    """Align every payload; return [(precise_words, stats)] and run timings."""
    from easyaligner.data.datamodel import SpeechSegment
    from easyaligner.pipelines import pipeline
    from easyaligner.text import text_normalizer

    t0 = time.perf_counter()
    vad_model, model, processor, model_id = _load_models(language, model_id, device)
    vocab = _vocab_chars(processor)
    spell = _speller(language, vocab)
    t_load = time.perf_counter() - t0

    plans = []
    speeches = []
    for payload in payloads:
        rough = payload.get("pre_mfa_words") or payload.get("rough_words") or []
        duration = float(payload["duration"])
        if placeholders:
            seq = build_sequence(rough, duration)
        else:
            seq = [
                {"token": w.get("word", ""), "raw_index": i, "kind": "raw"}
                for i, w in enumerate(rough)
            ]
        aligned_items = []
        parts = []
        for item in seq:
            atok = _to_alignment_token(item["token"], spell, vocab)
            if atok:
                aligned_items.append(item)
                parts.append(atok)
        text = " ".join(parts)
        plans.append({"rough": rough, "seq": seq, "items": aligned_items, "duration": duration})
        speeches.append([
            SpeechSegment(speech_id=0, text=text, text_spans=_span_tokenizer(text), start=None, end=None)
        ])

    audio_dir = audio_paths[0].parent
    if any(p.parent != audio_dir for p in audio_paths):
        raise ValueError("All audio files must be in one folder")

    t1 = time.perf_counter()
    results = pipeline(
        vad_model=vad_model,
        emissions_model=model,
        processor=processor,
        audio_paths=[p.name for p in audio_paths],
        audio_dir=str(audio_dir),
        speeches=speeches,
        alignment_strategy="speech",
        text_normalizer_fn=text_normalizer,
        tokenizer=_span_tokenizer,
        start_wildcard=True,
        end_wildcard=True,
        blank_id=_blank_id(processor),
        word_boundary="|",
        batch_size_features=batch_size,
        return_alignments=True,
        output_vad_dir=str(work_dir / "vad"),
        output_emissions_dir=str(work_dir / "emissions"),
        output_alignments_dir=str(work_dir / "alignments"),
        device=device,
    )
    t_align = time.perf_counter() - t1

    # One list of AlignmentSegment per file, in input order.
    out = [_map_back(plan, segments) for plan, segments in zip(plans, results)]

    t2 = time.perf_counter()
    if refine_edges:
        for (words, stats), audio in zip(out, audio_paths):
            regions = _speech_regions(vad_model, audio, min_silence_ms)
            stats["refined_words"] = _refine_edges(words, regions, boundary_alpha)
    t_refine = time.perf_counter() - t2

    if not keep_temp:
        shutil.rmtree(work_dir, ignore_errors=True)
    timings = {
        "model": model_id,
        "device": device,
        "placeholders": placeholders,
        "refine_edges": refine_edges,
        "min_silence_ms": min_silence_ms,
        "boundary_alpha": boundary_alpha,
        "load_seconds": round(t_load, 2),
        "align_seconds": round(t_align, 2),
        "refine_seconds": round(t_refine, 2),
        "n_files": len(payloads),
    }
    return out, timings


def _map_back(plan: dict, segments: list) -> tuple[list[dict], dict]:
    rough = plan["rough"]
    items = plan["items"]
    aligned: list[dict | None] = [None] * len(rough)
    consumed = 0
    mismatch = len(segments) != len(items)
    if not mismatch:
        for item, seg in zip(items, segments):
            if item["raw_index"] is None:
                consumed += 1
                continue
            start, end = float(seg.start), float(seg.end)
            if end <= start:
                end = start + 0.001
            aligned[item["raw_index"]] = {
                "word": rough[item["raw_index"]]["word"],
                "start": round(start, 3),
                "end": round(end, 3),
                "source": "ctc",
            }

    fallback = 0
    for i, w in enumerate(aligned):
        if w is not None:
            continue
        rs = float(rough[i]["start"])
        re_ = float(rough[i]["end"])
        if re_ <= rs:
            re_ = rs + 0.001
        aligned[i] = {"word": rough[i]["word"], "start": round(rs, 3), "end": round(re_, 3), "source": "rough"}
        fallback += 1

    stats = {
        "ctc_token_count": len(segments),
        "filler_injected": sum(1 for x in plan["seq"] if x["kind"] == "filler"),
        "filler_consumed": consumed,
        "fallback_count": fallback,
        "segment_count_mismatch": mismatch,
    }
    return aligned, stats


def run_cli(pipeline_module, language: str, build_sequence: Callable[[list[dict], float], list[dict]]) -> None:
    """Command-line entry shared by en/ and ja/ ctc_from_pre_mfa_json.py."""
    import argparse

    parser = argparse.ArgumentParser(description=f"easyaligner Stage 2 from pre_mfa_json ({language})")
    parser.add_argument("-i", "--input", required=True, help="Folder containing pre_mfa_json/")
    parser.add_argument("-a", "--audio-dir", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--file", nargs="+", help="Process specific files by name (without extension)")
    parser.add_argument("--model", default=None, help=f"CTC model (default {DEFAULT_MODELS[language]})")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-placeholders", action="store_true", help="Do not inject filler placeholders")
    parser.add_argument("--no-refine", action="store_true", help="Keep raw CTC word edges (no VAD edge refinement)")
    parser.add_argument("--min-silence-ms", type=int, default=100, help="VAD minimum silence for edge refinement")
    parser.add_argument("--boundary-alpha", type=float, default=0.2,
                        help="Within a speech region, place word boundaries this fraction into the CTC gap "
                             "(0.2 minimised English boundary error on 8 tuning recordings)")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    pre_dir = Path(args.input)
    if pre_dir.name != "pre_mfa_json" and (pre_dir / "pre_mfa_json").exists():
        pre_dir = pre_dir / "pre_mfa_json"
    paths = sorted(p for p in pre_dir.glob("*.json"))
    if args.file:
        wanted = set(args.file)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"No pre_mfa_json files in {pre_dir}")

    audio_dir = Path(args.audio_dir)
    payloads, audio_paths, ids = [], [], []
    for p in paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        file_id = str(payload.get("file") or p.stem)
        audio = None
        if payload.get("audio_file") and (audio_dir / payload["audio_file"]).exists():
            audio = audio_dir / payload["audio_file"]
        else:
            for ext in pipeline_module.AUDIO_EXTS:
                if (audio_dir / f"{file_id}{ext}").exists():
                    audio = audio_dir / f"{file_id}{ext}"
                    break
        if audio is None:
            raise FileNotFoundError(f"Audio file not found for {file_id} under {audio_dir}")
        payloads.append(payload)
        audio_paths.append(audio)
        ids.append(file_id)

    output_dir = Path(args.output)
    print(f"\nPipeline: pre_mfa_json -> easyaligner ({language}) -> TextGrid, {len(paths)} files")
    results, timings = align_payloads(
        payloads,
        audio_paths,
        build_sequence,
        language,
        output_dir / "_ctc_temporary",
        model_id=args.model,
        device=args.device,
        placeholders=not args.no_placeholders,
        refine_edges=not args.no_refine,
        min_silence_ms=args.min_silence_ms,
        boundary_alpha=args.boundary_alpha,
        batch_size=args.batch_size,
        keep_temp=args.keep_temp,
    )

    tg_dir, tg_clean_dir, json_dir = (output_dir / d for d in ("textgrids", "textgrids_clean", "json"))
    for d in (tg_dir, tg_clean_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)
    total = rough = 0
    for file_id, payload, (words, stats) in zip(ids, payloads, results):
        duration = float(payload["duration"])
        pipeline_module.build_final_textgrid(words, duration, str(tg_dir / f"{file_id}.TextGrid"), clean=False)
        pipeline_module.build_final_textgrid(words, duration, str(tg_clean_dir / f"{file_id}.TextGrid"), clean=True)
        pipeline_module.save_json(words, duration, str(payload.get("transcript") or ""),
                                  str(json_dir / f"{file_id}.json"), stats=stats)
        n_rough = sum(1 for w in words if w["source"] == "rough")
        total += len(words)
        rough += n_rough
        flag = "  SEGMENT MISMATCH" if stats["segment_count_mismatch"] else ""
        print(f"  {file_id}: {len(words)} words, {n_rough} rough fallback{flag}")

    timings.update({"total_words": total, "rough_fallback_words": rough})
    (output_dir / "ctc_run.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"\nDone: {total} words, {rough} rough fallback. Alignment took {timings['align_seconds']} s "
          f"(+{timings['load_seconds']} s model load).")
