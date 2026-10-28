"""Stage 1 on Apple Silicon: Qwen3-ASR and its forced aligner through MLX.

mlx-qwen3-asr reimplements the official PyTorch models in MLX, Apple's array
framework, so they run on the Mac GPU; the official qwen-asr package needs
CUDA for the GPU and runs on the CPU on a Mac. This module wraps it in the part
of the qwen_asr.Qwen3ASRModel interface that Stage 1 uses:

    transcribe(audio=..., language=..., context=..., return_time_stamps=True)

returns a list with one result whose .text is the transcript and whose
.time_stamps hold one item per word with .text, .start_time and .end_time.
"""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from typing import List


def available() -> bool:
    """True on an Apple Silicon Mac with mlx-qwen3-asr installed."""
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_qwen3_asr") is not None
    )


def _loops_only(tokens: List[int], threshold: int = 20) -> bool:
    """Stop decoding only on a runaway loop.

    One token, or a pattern of 2-10 tokens, repeated `threshold` (20, the
    official limit) times in a row. The port's default stops after two
    repeats of a 10-token phrase, which cuts off learners who repeat a
    sentence; the official package never stops on repetition.
    """
    if len(tokens) < threshold:
        return False
    count = 1
    for i in range(len(tokens) - 1, 0, -1):
        if tokens[i] != tokens[i - 1]:
            break
        count += 1
        if count >= threshold:
            return True
    for pattern_len in range(2, min(11, len(tokens) // 2 + 1)):
        pattern = tokens[-pattern_len:]
        repeats, pos = 1, len(tokens) - pattern_len * 2
        while pos >= 0 and tokens[pos:pos + pattern_len] == pattern:
            repeats += 1
            pos -= pattern_len
        if repeats >= threshold:
            return True
    return False


@dataclass(frozen=True)
class WordStamp:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class Transcription:
    text: str
    time_stamps: List[WordStamp]


class MLXQwen3ASR:
    """Qwen3-ASR plus forced aligner on the Mac GPU, loaded once for many files."""

    def __init__(self, asr_model: str, aligner_model: str, dtype: str = "bfloat16"):
        import mlx.core as mx
        import mlx_qwen3_asr
        from mlx_qwen3_asr import ForcedAligner, load_model

        # bfloat16, as Stage 1 uses on CUDA GPUs.
        self._dtype = getattr(mx, dtype)
        self._model, _ = load_model(asr_model, dtype=self._dtype)
        self._aligner = ForcedAligner(aligner_model, dtype=self._dtype)
        self.backend = f"mlx-qwen3-asr {mlx_qwen3_asr.__version__}, {dtype}"

        # One attention window per audio piece in both audio encoders, as the
        # official qwen-asr computes them on Linux: it applies the 8 s windows
        # (n_window_infer) only under flash-attn, which Stage 1 does not use.
        # The port applies the 8 s windows; with them, Japanese transcripts
        # differed clearly from the official ones.
        self._aligner._ensure_loaded()
        for encoder in (self._model.audio_tower, self._aligner._backend.model.audio_tower):
            encoder.config.n_window_infer = encoder.config.n_window * 2 * 10**6  # longer than any piece

    def transcribe(self, audio: str, language: str | None = None, context: str = "",
                   return_time_stamps: bool = False) -> List[Transcription]:
        # The module, not the package's transcribe() function of the same name.
        mlx_transcribe = importlib.import_module("mlx_qwen3_asr.transcribe")
        from qwen_asr.inference.utils import (
            MAX_ASR_INPUT_SECONDS, MAX_FORCE_ALIGN_INPUT_SECONDS, normalize_audios, split_audio_into_chunks)

        # Split the audio as the official package does: at most 180 s per piece
        # with timestamps (1200 s without), at low-energy points. The port's own
        # 30 s pieces let the model copy the prompt into quiet pieces.
        max_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS
        mlx_transcribe.split_audio_into_chunks = (
            lambda audio_np, sr=16000, **_: split_audio_into_chunks(wav=audio_np, sr=sr, max_chunk_sec=max_sec))
        importlib.import_module("mlx_qwen3_asr.generate")._detect_repetition = _loops_only
        # Join the pieces' texts as the official package does: with nothing in
        # between, in every language (the port puts a space between English pieces).
        mlx_transcribe.join_text_parts = lambda parts, language=None: "".join(p for p in parts if p is not None)

        # Read and resample the audio as the official package does (librosa,
        # mono, 16 kHz), so MLX gets the samples Stage 1 gets on Linux. Given a
        # path, the port resamples through an ffmpeg subprocess instead.
        wav = normalize_audios(audio)[0]
        result = mlx_transcribe.transcribe(
            wav, model=self._model, context=context, language=language,
            return_timestamps=return_time_stamps, forced_aligner=self._aligner,
            dtype=self._dtype, max_new_tokens=4096,  # as in the official configuration (asr_mfa_*.load_models)
        )
        stamps = [WordStamp(str(s["text"]), float(s["start"]), float(s["end"]))
                  for s in (result.segments or [])]
        return [Transcription(result.text or "", stamps)]
