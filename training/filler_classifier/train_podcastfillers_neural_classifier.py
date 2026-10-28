#!/usr/bin/env python
"""
Train a neural acoustic filler classifier on PodcastFillers official labels.

A TC-ResNet8-style neural model on log-mel inputs.

Labels:
- Positive: Uh, Um
- Negative: Words, Breath, Laughter, Music
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset

POS_LABELS_DEFAULT = {"Uh", "Um"}
NEG_LABELS_DEFAULT = {"Words", "Breath", "Laughter", "Music"}
LICENSE_SPLITS_DEFAULT = ["CC_BY_3.0", "CC_BY_SA_3.0", "CC_BY_ND_3.0"]


def normalize_split(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    v = name.strip().lower()
    if v in {"train", "training"}:
        return "train"
    if v in {"validation", "valid", "val", "dev"}:
        return "validation"
    if v in {"test", "testing"}:
        return "test"
    return None


def parse_label_set(v: str) -> set[str]:
    return {x.strip() for x in v.split(",") if x.strip()}


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(p),
        "recall_pos": float(r),
        "f1_pos": float(f),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }


def load_metadata(
    csv_path: Path,
    pos_labels: set[str],
    neg_labels: set[str],
    rng: np.random.Generator,
    max_train: Optional[int],
    max_validation: Optional[int],
    max_test: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    req_cols = {
        "podcast_filename",
        "episode_split_subset",
        "clip_split_subset",
        "clip_start_inepisode",
        "clip_end_inepisode",
        "label_consolidated_vocab",
        "clip_name",
        "pfID",
    }
    missing = sorted(req_cols - set(df.columns))
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    df = df[df["label_consolidated_vocab"].isin(pos_labels | neg_labels)].copy()
    df["label"] = df["label_consolidated_vocab"].isin(pos_labels).astype(int)
    df["split"] = df["clip_split_subset"].map(normalize_split)
    df = df[df["split"].isin(["train", "validation", "test"])].copy()

    limits = {
        "train": max_train,
        "validation": max_validation,
        "test": max_test,
    }
    parts: List[pd.DataFrame] = []
    for split in ["train", "validation", "test"]:
        part = df[df["split"] == split]
        lim = limits[split]
        if lim is not None and len(part) > lim:
            idx = rng.choice(len(part), size=lim, replace=False)
            part = part.iloc[np.sort(idx)]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_episode_groups(meta_df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (podcast_filename, episode_split_subset), g in meta_df.groupby(["podcast_filename", "episode_split_subset"]):
        split_norm = normalize_split(str(episode_split_subset))
        if split_norm is None:
            continue
        groups[(str(podcast_filename), split_norm)] = g.reset_index(drop=True)
    return groups


def extract_logmel_segment(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    max_frames: int,
) -> Optional[np.ndarray]:
    if y is None or len(y) < int(0.02 * sr):
        return None

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
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
    if t < max_frames:
        pad = np.zeros((n_mels, max_frames - t), dtype=np.float32)
        logmel = np.concatenate([logmel.astype(np.float32), pad], axis=1)
    elif t > max_frames:
        logmel = logmel[:, :max_frames].astype(np.float32)
    else:
        logmel = logmel.astype(np.float32)

    return logmel[np.newaxis, :, :].astype(np.float16)


@dataclass
class SplitTensors:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def extract_dataset(
    meta_df: pd.DataFrame,
    groups: Dict[Tuple[str, str], pd.DataFrame],
    hf_dataset: str,
    license_splits: Sequence[str],
    target_sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    max_frames: int,
) -> Tuple[Dict[str, SplitTensors], Dict]:
    split_feats: Dict[str, List[np.ndarray]] = {"train": [], "validation": [], "test": []}
    split_meta: Dict[str, List[Dict]] = {"train": [], "validation": [], "test": []}

    needed_keys = set(groups.keys())
    matched_episode_keys: set[Tuple[str, str]] = set()
    episode_decode_fail = 0
    event_bounds_fail = 0
    feature_fail = 0

    for lic in license_splits:
        ds = load_dataset(hf_dataset, split=lic)
        ds = ds.cast_column("audio", Audio(decode=False))

        for i in range(len(ds)):
            ex = ds[i]
            key = (str(ex.get("episode_name", "")), normalize_split(str(ex.get("original_split", ""))))
            if key[1] is None:
                continue
            if key not in needed_keys:
                continue

            audio_dict = ex.get("audio") or {}
            audio_bytes = audio_dict.get("bytes")
            if not audio_bytes:
                episode_decode_fail += 1
                continue

            try:
                wav, _sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
            except Exception:
                episode_decode_fail += 1
                continue

            matched_episode_keys.add(key)
            episode_events = groups[key]
            wav_len = len(wav)

            for _, ev in episode_events.iterrows():
                st = float(ev["clip_start_inepisode"])
                et = float(ev["clip_end_inepisode"])
                s = int(round(st * target_sr))
                e = int(round(et * target_sr))

                if e <= s:
                    event_bounds_fail += 1
                    continue

                s = max(0, s)
                e = min(wav_len, e)
                if e <= s:
                    event_bounds_fail += 1
                    continue

                seg = wav[s:e]
                feat = extract_logmel_segment(
                    y=seg,
                    sr=target_sr,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    max_frames=max_frames,
                )
                if feat is None:
                    feature_fail += 1
                    continue

                split = str(ev["split"])
                split_feats[split].append(feat)
                split_meta[split].append(
                    {
                        "pfID": int(ev["pfID"]),
                        "clip_name": str(ev["clip_name"]),
                        "episode_name": key[0],
                        "episode_split": key[1],
                        "split": split,
                        "label_consolidated_vocab": str(ev["label_consolidated_vocab"]),
                        "label": int(ev["label"]),
                        "clip_start_inepisode": float(ev["clip_start_inepisode"]),
                        "clip_end_inepisode": float(ev["clip_end_inepisode"]),
                    }
                )

    out: Dict[str, SplitTensors] = {}
    for split in ["train", "validation", "test"]:
        meta_list = split_meta[split]
        if meta_list:
            X = np.stack(split_feats[split], axis=0).astype(np.float16)
            y = np.array([int(r["label"]) for r in meta_list], dtype=np.int64)
            m = pd.DataFrame(meta_list)
        else:
            X = np.empty((0, 1, n_mels, max_frames), dtype=np.float16)
            y = np.empty((0,), dtype=np.int64)
            m = pd.DataFrame(columns=["label"])
        out[split] = SplitTensors(X=X, y=y, meta=m)

    stats = {
        "metadata_events_requested": int(len(meta_df)),
        "episodes_requested": int(len(needed_keys)),
        "episodes_matched": int(len(matched_episode_keys)),
        "episode_decode_fail": int(episode_decode_fail),
        "event_bounds_fail": int(event_bounds_fail),
        "feature_fail": int(feature_fail),
        "events_extracted": int(sum(len(v.meta) for v in out.values())),
    }
    return out, stats


class ClipDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).to(torch.float32)
        y = int(self.y[idx])
        return x, y, idx


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
        # Expected input: (B, 1, n_mels, T)
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    loss_sum = 0.0
    n_obs = 0
    y_true: List[int] = []
    y_prob: List[float] = []

    for xb, yb, _idx in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        if train_mode:
            loss.backward()
            optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_np = yb.detach().cpu().numpy()

        bs = len(y_np)
        loss_sum += float(loss.item()) * bs
        n_obs += bs
        y_true.extend(y_np.tolist())
        y_prob.extend(probs.tolist())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.int64)

    out = metric_bundle(y_true_arr, y_pred_arr, y_prob_arr)
    out["loss"] = float(loss_sum / max(1, n_obs))
    return out


def predict_with_metadata(
    model: nn.Module,
    loader: DataLoader,
    meta_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict]:
    model.eval()
    n = len(meta_df)
    probs = np.zeros(n, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    with torch.no_grad():
        for xb, yb, idx in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            idx_np = idx.numpy()
            probs[idx_np] = p
            labels[idx_np] = yb.numpy()

    preds = (probs >= 0.5).astype(np.int64)
    out_df = meta_df.copy().reset_index(drop=True)
    out_df["prob_filler"] = probs
    out_df["pred_label"] = preds
    out_df["true_label"] = labels
    metrics = metric_bundle(labels, preds, probs)
    return out_df, metrics


def write_run_log(out_dir: Path, args: argparse.Namespace, extract_stats: Dict, metrics: Dict, best_epoch: int) -> None:
    lines = [
        "# PodcastFillers Neural Classifier Run Log",
        "",
        "## Command",
        "```powershell",
        "python training/filler_classifier/train_podcastfillers_neural_classifier.py \\",
        f"  --metadata-csv {str(args.metadata_csv).replace('\\\\', '/')} \\",
        f"  --out-dir {str(args.out_dir).replace('\\\\', '/')} \\",
        f"  --hf-dataset {args.hf_dataset} \\",
        f"  --license-splits {args.license_splits}",
        "```",
        "",
        "## Label Setup",
        f"- Positive: {args.pos_labels}",
        f"- Negative: {args.neg_labels}",
        "",
        "## Feature Setup",
        f"- target_sr: {args.target_sr}",
        f"- n_mels: {args.n_mels}",
        f"- win_ms: {args.win_ms}",
        f"- hop_ms: {args.hop_ms}",
        f"- max_frames: {args.max_frames}",
        "",
        "## Model/Train Setup",
        "- Architecture: TC-ResNet8-style 1D temporal Conv-ResNet",
        f"- batch_size: {args.batch_size}",
        f"- epochs: {args.epochs}",
        f"- lr: {args.lr}",
        f"- early_stop_patience: {args.early_stop_patience}",
        f"- lr_patience: {args.lr_patience}",
        f"- lr_factor: {args.lr_factor}",
        f"- best_epoch: {best_epoch}",
        "",
        "## Extraction Stats",
        f"- metadata_events_requested: {extract_stats['metadata_events_requested']}",
        f"- episodes_requested: {extract_stats['episodes_requested']}",
        f"- episodes_matched: {extract_stats['episodes_matched']}",
        f"- episode_decode_fail: {extract_stats['episode_decode_fail']}",
        f"- event_bounds_fail: {extract_stats['event_bounds_fail']}",
        f"- feature_fail: {extract_stats['feature_fail']}",
        f"- events_extracted: {extract_stats['events_extracted']}",
        "",
        "## Metrics",
        f"- validation F1: {metrics['validation']['f1_pos']:.4f}, P: {metrics['validation']['precision_pos']:.4f}, R: {metrics['validation']['recall_pos']:.4f}, AUC: {metrics['validation']['auc'] if metrics['validation']['auc'] is not None else 'n/a'}",
        f"- test F1: {metrics['test']['f1_pos']:.4f}, P: {metrics['test']['precision_pos']:.4f}, R: {metrics['test']['recall_pos']:.4f}, AUC: {metrics['test']['auc'] if metrics['test']['auc'] is not None else 'n/a'}",
    ]
    (out_dir / "RUN_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("training/filler_classifier/podcastfillers_data/PodcastFillers.csv"),
    )
    ap.add_argument("--hf-dataset", type=str, default="ylacombe/podcast_fillers_by_license")
    ap.add_argument(
        "--license-splits",
        type=str,
        default="CC_BY_3.0,CC_BY_SA_3.0,CC_BY_ND_3.0",
    )
    ap.add_argument("--pos-labels", type=str, default="Uh,Um")
    ap.add_argument("--neg-labels", type=str, default="Words,Breath,Laughter,Music")
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--win-ms", type=float, default=25.0)
    ap.add_argument("--hop-ms", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=101)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--lr-patience", type=int, default=3)
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-validation", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/filler_classifier/model_podcastfillers_neural_v1_full"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.random_seed)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_labels = parse_label_set(args.pos_labels)
    neg_labels = parse_label_set(args.neg_labels)
    license_splits = [x.strip() for x in args.license_splits.split(",") if x.strip()]

    win_length = int(round(args.target_sr * (args.win_ms / 1000.0)))
    hop_length = int(round(args.target_sr * (args.hop_ms / 1000.0)))
    n_fft = win_length

    rng = np.random.default_rng(args.random_seed)
    meta_df = load_metadata(
        csv_path=args.metadata_csv,
        pos_labels=pos_labels,
        neg_labels=neg_labels,
        rng=rng,
        max_train=args.max_train,
        max_validation=args.max_validation,
        max_test=args.max_test,
    )
    groups = build_episode_groups(meta_df)

    split_data, extract_stats = extract_dataset(
        meta_df=meta_df,
        groups=groups,
        hf_dataset=args.hf_dataset,
        license_splits=license_splits,
        target_sr=args.target_sr,
        n_mels=args.n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        max_frames=args.max_frames,
    )

    train = split_data["train"]
    val = split_data["validation"]
    test = split_data["test"]

    if min(len(train.y), len(val.y), len(test.y)) == 0:
        raise RuntimeError(
            f"Missing split after extraction: train={len(train.y)} validation={len(val.y)} test={len(test.y)}"
        )

    train_ds = ClipDataset(train.X, train.y)
    val_ds = ClipDataset(val.X, val.y)
    test_ds = ClipDataset(test.X, test.y)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TCResNet8(n_mels=args.n_mels, dropout=args.dropout, num_classes=2).to(device)

    cls_counts = np.bincount(train.y.astype(np.int64), minlength=2)
    cls_weights = (len(train.y) / (2.0 * np.maximum(cls_counts, 1))).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=False,
    )

    best_state = None
    best_val_auc = -1.0
    best_epoch = 0
    stale = 0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        va = run_epoch(model, val_loader, criterion, device, optimizer=None)
        target_metric = va["auc"] if va["auc"] is not None else va["f1_pos"]
        scheduler.step(target_metric)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_f1": tr["f1_pos"],
                "train_auc": tr["auc"],
                "val_loss": va["loss"],
                "val_f1": va["f1_pos"],
                "val_auc": va["auc"],
            }
        )

        improved = target_metric > (best_val_auc + 1e-6)
        if improved:
            best_val_auc = float(target_metric)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        print(
            f"epoch {epoch:02d} "
            f"train_loss={tr['loss']:.4f} train_f1={tr['f1_pos']:.4f} "
            f"val_loss={va['loss']:.4f} val_f1={va['f1_pos']:.4f} val_auc={va['auc']}"
        )

        if stale >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val AUC improvement for {stale} epochs)")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint found")

    model.load_state_dict(best_state)

    val_pred_df, val_metrics = predict_with_metadata(model, val_loader, val.meta, device)
    test_pred_df, test_metrics = predict_with_metadata(model, test_loader, test.meta, device)

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "model_name": "TCResNet8",
            "n_mels": args.n_mels,
            "max_frames": args.max_frames,
            "dropout": args.dropout,
            "num_classes": 2,
        },
        "feature_spec": {
            "target_sr": args.target_sr,
            "n_mels": args.n_mels,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "max_frames": args.max_frames,
            "normalization": "per_clip_zscore",
        },
        "train_config": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "early_stop_patience": args.early_stop_patience,
            "lr_patience": args.lr_patience,
            "lr_factor": args.lr_factor,
            "class_weights": cls_weights.tolist(),
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
        },
        "source": {
            "metadata_csv": str(args.metadata_csv),
            "hf_dataset": args.hf_dataset,
            "license_splits": license_splits,
            "pos_labels": sorted(pos_labels),
            "neg_labels": sorted(neg_labels),
        },
    }
    torch.save(ckpt, out_dir / "model.pt")

    val_pred_df.to_csv(out_dir / "validation_predictions.csv", index=False)
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    train.meta.to_csv(out_dir / "train_manifest.csv", index=False)

    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "extract_stats": extract_stats,
        "split_counts": {
            "train": int(len(train.y)),
            "validation": int(len(val.y)),
            "test": int(len(test.y)),
        },
        "class_balance": {
            "train_pos_rate": float(np.mean(train.y)),
            "validation_pos_rate": float(np.mean(val.y)),
            "test_pos_rate": float(np.mean(test.y)),
        },
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "device": str(device),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    readme = [
        "# PodcastFillers Neural Model (v1)",
        "",
        "Model: TC-ResNet8-style temporal CNN",
        "Input: log-mel spectrogram (1 x 64 x 101), per-clip z-score normalization",
        "",
        "Data source:",
        f"- Metadata CSV: `{args.metadata_csv}`",
        f"- Audio dataset: `{args.hf_dataset}`",
        f"- License splits: `{','.join(license_splits)}`",
        "",
        "Labels:",
        f"- Positive: `{','.join(sorted(pos_labels))}`",
        f"- Negative: `{','.join(sorted(neg_labels))}`",
        "",
        "Artifacts:",
        "- `model.pt`",
        "- `metrics.json`",
        "- `RUN_LOG.md`",
        "- `train_history.csv`",
        "- `train_manifest.csv`",
        "- `validation_predictions.csv`",
        "- `test_predictions.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    write_run_log(out_dir, args, extract_stats, {"validation": val_metrics, "test": test_metrics}, best_epoch)

    print("PodcastFillers neural classifier trained")
    print("extracted:", extract_stats)
    print(
        f"validation: F1={val_metrics['f1_pos']:.4f} "
        f"P={val_metrics['precision_pos']:.4f} R={val_metrics['recall_pos']:.4f} "
        f"AUC={val_metrics['auc']}"
    )
    print(
        f"test: F1={test_metrics['f1_pos']:.4f} "
        f"P={test_metrics['precision_pos']:.4f} R={test_metrics['recall_pos']:.4f} "
        f"AUC={test_metrics['auc']}"
    )


if __name__ == "__main__":
    main()
