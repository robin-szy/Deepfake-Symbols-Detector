"""
Train and evaluate a compact PyTorch sequence classifier for fake-symbol detection.

Usage for training:
    python symbol_classifier_solution.py --data_dir /path/to/symbols --model_file model.pth

Expected training layout:
    /path/to/symbols/real/*.csv   -> label 0
    /path/to/symbols/fake/*.csv   -> label 1

The assignment evaluator can import load_and_predict(directory, model_file) from this file.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# CSV reading and preprocessing
# -----------------------------

def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def read_symbol_csv(path: str) -> pd.DataFrame:
    """Read one symbol CSV robustly. The provided example is whitespace/tab separated."""
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True)

    df = _normalise_column_names(df)

    required = {"x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    # Make optional fields available, even if hidden test files omit them.
    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=np.float32) * 10.0
    if "stroke_id" not in df.columns:
        df["stroke_id"] = 1.0
    if "is_writing" not in df.columns:
        df["is_writing"] = 1.0

    use_cols = ["x", "y", "time", "stroke_id", "is_writing"]
    df = df[use_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return df


def sequence_features(path: str, max_len: int | None = 256) -> np.ndarray:
    """
    Convert a CSV into a T x F float32 sequence.

    Features are deliberately generic and computed within each file:
    - centered/scaled x and y
    - first differences dx, dy
    - time delta dt, log-compressed
    - stroke transition indicator
    - is_writing
    """
    df = read_symbol_csv(path)
    x = df["x"].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    t = df["time"].to_numpy(np.float32)
    stroke = df["stroke_id"].to_numpy(np.float32)
    writing = df["is_writing"].to_numpy(np.float32)

    # Per-symbol coordinate normalization preserves shape while removing absolute canvas scale.
    xy = np.stack([x, y], axis=1)
    center = xy.mean(axis=0, keepdims=True)
    scale = xy.std(axis=0, keepdims=True).mean()
    scale = float(scale) if float(scale) > 1e-6 else 1.0
    xy_norm = (xy - center) / scale

    dxy = np.zeros_like(xy_norm)
    dxy[1:] = xy_norm[1:] - xy_norm[:-1]

    dt = np.zeros((len(df), 1), dtype=np.float32)
    if len(df) > 1:
        raw_dt = np.diff(t, prepend=t[0]).astype(np.float32)
        raw_dt = np.clip(raw_dt, 0.0, None)
        dt[:, 0] = np.log1p(raw_dt) / np.log1p(1000.0)

    stroke_change = np.zeros((len(df), 1), dtype=np.float32)
    if len(df) > 1:
        stroke_change[1:, 0] = (stroke[1:] != stroke[:-1]).astype(np.float32)

    writing = writing.reshape(-1, 1).astype(np.float32)
    feats = np.concatenate([xy_norm, dxy, dt, stroke_change, writing], axis=1).astype(np.float32)

    if max_len is not None and len(feats) > max_len:
        # Uniform downsampling keeps the full trajectory instead of chopping off the tail.
        idx = np.linspace(0, len(feats) - 1, max_len).round().astype(int)
        feats = feats[idx]

    return feats


def fit_standardizer(paths: Sequence[str], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    chunks = [sequence_features(p, max_len=max_len) for p in paths]
    all_feats = np.concatenate(chunks, axis=0)
    mean = all_feats.mean(axis=0).astype(np.float32)
    std = all_feats.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


# -----------------------------
# Dataset and batching
# -----------------------------

class SymbolDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[str, int]], mean: np.ndarray, std: np.ndarray, max_len: int):
        self.items = list(items)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        x = sequence_features(path, max_len=self.max_len)
        x = (x - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), path


def collate_batch(batch):
    seqs, labels, paths = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    feat_dim = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    for i, seq in enumerate(seqs):
        padded[i, : seq.shape[0]] = seq
    labels = torch.stack(labels)
    return padded, lengths, labels, list(paths)


# -----------------------------
# Model: small GRU sequence classifier
# -----------------------------

class TinyGRUClassifier(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 12, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_last = h_n[-1]
        return self.head(h_last).squeeze(1)


# -----------------------------
# Training helpers
# -----------------------------

def collect_training_items(data_dir: str) -> List[Tuple[str, int]]:
    real = sorted(glob.glob(os.path.join(data_dir, "real", "*.csv")))
    fake = sorted(glob.glob(os.path.join(data_dir, "fake", "*.csv")))
    items = [(p, 0) for p in real] + [(p, 1) for p in fake]
    if not items:
        raise ValueError(f"No CSV files found in {data_dir}/real and {data_dir}/fake")
    return items


def stratified_split(items: Sequence[Tuple[str, int]], val_fraction: float, seed: int):
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for item in items:
        by_label[item[1]].append(item)
    train, val = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val




def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, total_correct, total = 0.0, 0, 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, lengths, y, _ in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            logits = model(x, lengths)

            # Loss
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)

            # Predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_correct += (preds == y).sum().item()
            total += len(y)

            # For AUC
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = float("nan")

    return avg_loss, acc, auc

def train_model(args) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    items = collect_training_items(args.data_dir)
    train_items, val_items = stratified_split(items, args.val_fraction, args.seed)
    mean, std = fit_standardizer([p for p, _ in train_items], max_len=args.max_len)

    train_ds = SymbolDataset(train_items, mean, std, args.max_len)
    val_ds = SymbolDataset(val_items, mean, std, args.max_len) if val_items else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        if val_ds is not None
        else None
    )

    model = TinyGRUClassifier(input_dim=7, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_metric = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss, total_train = 0.0, 0
        for x, lengths, y, _ in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_train_loss += float(loss.item()) * len(y)
            total_train += len(y)

        train_loss = total_train_loss / max(total_train, 1)
        if val_loader is not None:
            val_loss, val_acc, val_auc = evaluate(model, val_loader, device)

            print(
                f"epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.3f} "
                f"val_auc={val_auc:.3f}"
            )

            metric = val_acc
        else:
            metric = -train_loss
            print(f"epoch {epoch:03d} train_loss={train_loss:.4f}")

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 7,
        "hidden_dim": args.hidden_dim,
        "dropout": 0.0,
        "max_len": args.max_len,
        "mean": mean,
        "std": std,
        "threshold": 0.5,
    }
    torch.save(checkpoint, args.model_file)
    print(f"saved checkpoint to {os.path.abspath(args.model_file)}")


# -----------------------------
# Required evaluator function
# -----------------------------

def load_and_predict(directory, model_file) -> Dict[str, int]:
    """
    Required by the assignment evaluator.

    Returns:
        dict mapping absolute CSV path -> predicted sparse label
        0 = real, 1 = fake
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    max_len = int(checkpoint.get("max_len", 256))
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    threshold = float(checkpoint.get("threshold", 0.5))

    model = TinyGRUClassifier(
        input_dim=int(checkpoint.get("input_dim", len(mean))),
        hidden_dim=int(checkpoint.get("hidden_dim", 12)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    pred_dict: Dict[str, int] = {}

    with torch.no_grad():
        for path in paths:
            feats = sequence_features(path, max_len=max_len)
            feats = (feats - mean) / std
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([feats.shape[0]], dtype=torch.long).to(device)
            logit = model(x, lengths)
            prob_fake = torch.sigmoid(logit).item()
            pred_dict[os.path.abspath(path)] = int(prob_fake >= threshold)

    return pred_dict


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing real/ and fake/ subfolders")
    parser.add_argument("--model_file", default="model.pth", help="Output .pth checkpoint path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
