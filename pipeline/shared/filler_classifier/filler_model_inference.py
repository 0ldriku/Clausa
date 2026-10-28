#!/usr/bin/env python
"""
Unified filler probability inference for legacy joblib models and neural .pt models.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import librosa
import numpy as np
import pandas as pd


def _extract_handcrafted_features(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    if len(y) < int(0.03 * sr):
        return None
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=128)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=128)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=128)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=128)

    feats: List[float] = [
        float(len(y) / float(sr)),
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(rolloff)),
        float(np.std(rolloff)),
    ]
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())
    return np.asarray(feats, dtype=np.float32)


class _ResBlock1D:
    def __init__(self):
        raise RuntimeError("Torch model classes are instantiated inside _load_neural()")


class FillerProbabilityScorer:
    def __init__(self, model_path: str | Path, target_sr: int = 16000, device: str | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.target_sr = int(target_sr)
        self.model_kind = "joblib" if self.model_path.suffix.lower() == ".joblib" else "neural"

        self.pipe = None
        self.feature_cols: List[str] = []
        self.torch = None
        self.model = None
        self.device = None
        self.n_mels = 64
        self.n_fft = 400
        self.win_length = 400
        self.hop_length = 160
        self.max_frames = 101

        if self.model_kind == "joblib":
            self._load_joblib()
        else:
            self._load_neural(device=device)

    def _load_joblib(self) -> None:
        payload = joblib.load(self.model_path)
        self.pipe = payload["pipeline"]
        self.feature_cols = list(payload["feature_cols"])

    def _load_neural(self, device: str | None = None) -> None:
        import torch
        import torch.nn as nn

        class ResBlock1D(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9, stride: int = 1):
                super().__init__()
                pad = kernel_size // 2
                self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
                self.bn1 = nn.BatchNorm1d(out_ch)
                self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
                self.bn2 = nn.BatchNorm1d(out_ch)
                self.relu = nn.ReLU(inplace=True)
                if stride != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(out_ch),
                    )
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + self.shortcut(x)
                out = self.relu(out)
                return out

        class TCResNet8(nn.Module):
            def __init__(self, n_mels: int = 64, dropout: float = 0.5, num_classes: int = 2):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(n_mels, 16, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm1d(16),
                    nn.ReLU(inplace=True),
                )
                self.block1 = ResBlock1D(16, 24, kernel_size=9, stride=2)
                self.block2 = ResBlock1D(24, 32, kernel_size=9, stride=2)
                self.block3 = ResBlock1D(32, 48, kernel_size=9, stride=2)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.drop = nn.Dropout(dropout)
                self.fc = nn.Linear(48, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.dim() == 4:
                    x = x.squeeze(1)
                x = self.stem(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.pool(x).squeeze(-1)
                x = self.drop(x)
                return self.fc(x)

        ckpt = torch.load(self.model_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        spec = ckpt.get("feature_spec", {})
        self.n_mels = int(spec.get("n_mels", cfg.get("n_mels", 64)))
        self.n_fft = int(spec.get("n_fft", 400))
        self.win_length = int(spec.get("win_length", self.n_fft))
        self.hop_length = int(spec.get("hop_length", 160))
        self.max_frames = int(spec.get("max_frames", cfg.get("max_frames", 101)))
        self.target_sr = int(spec.get("target_sr", self.target_sr))
        dropout = float(cfg.get("dropout", 0.5))
        num_classes = int(cfg.get("num_classes", 2))

        model = TCResNet8(n_mels=self.n_mels, dropout=dropout, num_classes=num_classes)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.torch = torch

    def _extract_neural_input(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        if len(y) < int(0.02 * sr):
            return None
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=2.0,
            center=True,
        )
        if mel.size == 0:
            return None

        logmel = librosa.power_to_db(mel, ref=np.max)
        mean = float(np.mean(logmel))
        std = float(np.std(logmel))
        if std < 1e-6:
            std = 1.0
        logmel = (logmel - mean) / std

        t = logmel.shape[1]
        if t < self.max_frames:
            pad = np.zeros((self.n_mels, self.max_frames - t), dtype=np.float32)
            logmel = np.concatenate([logmel.astype(np.float32), pad], axis=1)
        elif t > self.max_frames:
            logmel = logmel[:, : self.max_frames].astype(np.float32)
        else:
            logmel = logmel.astype(np.float32)

        return logmel[np.newaxis, np.newaxis, :, :]

    def predict_proba(self, clip: np.ndarray, sr: int) -> Optional[float]:
        if clip is None or len(clip) == 0:
            return None

        if self.model_kind == "joblib":
            feat = _extract_handcrafted_features(clip, sr=sr)
            if feat is None:
                return None
            x = pd.DataFrame([feat], columns=[f"f{k:02d}" for k in range(len(feat))])
            return float(self.pipe.predict_proba(x[self.feature_cols])[:, 1][0])

        x_np = self._extract_neural_input(clip, sr=sr)
        if x_np is None:
            return None
        with self.torch.no_grad():
            x = self.torch.from_numpy(x_np).to(self.device, dtype=self.torch.float32)
            logits = self.model(x)
            prob = self.torch.softmax(logits, dim=1)[:, 1].item()
        return float(prob)

