# Homework in Deep Learning: Deepfake written symbols detection
# Author: Robin Szymanski

# I hereby declare that I have used an LLM (ChatGPT, model GPT-5.3) to assist me with
# some small passages of the code, debugging, improve some readability, to get new ideas,
# and to assist me with training on the HPC server (by generating configs for
# hyperparameter sweeps).
# This is in accordance with the "Guidelines on the Use of Generative AI for
# Teaching and Learning", Version: 1.0, Date: 2026-02-16

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import roc_auc_score, accuracy_score
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./symbols")
    parser.add_argument("--model-file", default="model.pth")
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seq-max-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument("--final-train", action="store_true", default=True)
    return parser.parse_args()


# -----------------
# Utilities
# -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_sequence(path, seq_max_len):

    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalize x/y per symbol
    xy = df[["x", "y"]].to_numpy(dtype=np.float32)

    x = xy[:, 0]
    y = xy[:, 1]

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    time = df["time"].to_numpy(dtype=np.float32) #// 10
    dt = np.diff(time, prepend=time[0])

    stroke_id = df["stroke_id"].to_numpy()
    stroke_change = np.zeros(len(df), dtype=np.float32)
    stroke_change[1:] = (stroke_id[1:] != stroke_id[:-1]).astype(np.float32)
    dt_stroke_transition = dt * stroke_change

    vx = np.gradient(x, time)
    vy = np.gradient(y, time)

    ax = np.gradient(vx, time)
    ay = np.gradient(vy, time)

    speed = np.sqrt(vx ** 2 + vy ** 2)
    acceleration = np.sqrt(ax ** 2 + ay ** 2)
    curvature = np.abs(vx * ay - vy * ax) / (speed ** 3 + 1e-6)
    acceleration[stroke_change == 1] = 0
    curvature[stroke_change == 1] = 0
    speed = np.clip(speed, 0, 4)   # 1.0
    acceleration = np.clip(acceleration, 0, 0.03)  # 0.03, 0.047
    curvature = np.clip(curvature, 0, 0.1)   # 0.46; 0.192



    # Correction: Whenever stroke_ID changes, we need to reset r and dtheta to 0,
    # as the pen is lifted and moved to a new position.
    mask = stroke_change == 1
    dx[mask] = 0.0
    dy[mask] = 0.0
    # r[mask] = 0.0
    # dtheta[mask] = 0.0
    # theta_sin[mask] = 0.0
    # theta_cos[mask] = 0.0
    # theta[mask] = 0.0

    # Polar coordinates and angular velocity
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    theta_sin = np.sin(theta)
    theta_cos = np.cos(theta)
    theta_unwrapped = np.unwrap(theta)
    dtheta = np.diff(theta_unwrapped, prepend=theta_unwrapped[0])
    dtheta[mask] = 0.0
    theta_cos[mask] = 0.0   # Otherwise it's 1
    abs_dtheta = np.abs(dtheta) # Different signal than just dtheta. It's the magnitude, no directional information


    seq = np.stack(
        [
            #x,
            #y,
            dx,
            dy,
            np.log1p(r),
            #theta_unwrapped,
            theta_sin,
            theta_cos,
            dtheta,
            #abs_dtheta,
            dt,
            #stroke_id,
            stroke_change,
            np.log1p(curvature),
            np.log1p(speed),
            np.log1p(acceleration),
            #dt_stroke_transition,
        ],
        axis=1,
    ).astype(np.float32)

    global_features = np.array(
        [
            float((dt > 50).any()),
            #dt_stroke_transition.max(),
            np.log1p(len(df)),
            np.log1p(time.max() - time.min()),
            np.log1p(r.mean()),
            np.log1p(r.std()),
            dtheta.std(),
            #abs_dtheta.mean()
        ],
        dtype=np.float32,
    )

    if len(seq) > seq_max_len:
        #print(
        #    f"Warning: For the sequence {path}, the maximum length was adapted.")
        idx = np.linspace(0, len(seq) - 1, seq_max_len).astype(int)
        seq = seq[idx]

    return seq, global_features


def compute_global_norm(items, seq_max_len):
    """
    Helper function in order to normalize the global features
    """
    features = []
    for path, _ in items:
        _, g = read_sequence(path, seq_max_len)
        features.append(g)

    features = np.stack(features)

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6

    # Do not normalize binary has_large_dt feature
    mean[0] = 0.0
    std[0] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def compute_seq_norm(items, seq_max_len):
    """
    Compute mean/std for sequence features using only the training set.
    """
    features = []

    for path, _ in items:
        seq, _ = read_sequence(path, seq_max_len)
        features.append(seq)

    features = np.concatenate(features, axis=0)

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6

    return mean.astype(np.float32), std.astype(np.float32)


class SymbolDataset(Dataset):
    def __init__(self, items, seq_max_len, global_mean=None, global_std=None, seq_mean=None, seq_std=None):
        self.items = items
        self.seq_max_len = seq_max_len
        self.global_mean = global_mean
        self.global_std = global_std
        self.seq_mean = seq_mean
        self.seq_std = seq_std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq, global_features = read_sequence(path, self.seq_max_len)

        if self.seq_mean is not None:
            seq = (seq - self.seq_mean) / self.seq_std

        if self.global_mean is not None:    # Normalize global features
            global_features = (global_features - self.global_mean) / self.global_std

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(global_features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


def collate_batch(batch):
    """Pad variable-length sequences in a batch."""
    seqs, global_attr, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    feat_dim = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_len, feat_dim)
    # x = torch.zeros(len(seqs), max_len, 2)    # Only for x, y
    for i, seq in enumerate(seqs):
        x[i, : seq.shape[0]] = seq

    g = torch.stack(global_attr)
    y = torch.stack(labels)
    return x, lengths, g, y


# -----------------
# Model
# -----------------
class SmallGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, global_dropout=0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=11,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.global_proj = nn.Sequential(
            nn.Linear(6, 3),
            nn.ReLU()
        )

        self.global_dropout = nn.Dropout(global_dropout)

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size + 3),
            nn.Linear(hidden_size + 3, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x, lengths, global_features):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, h = self.gru(packed)
        h_last = h[-1]

        global_alpha = 1.0

        global_features = self.global_dropout(global_features)
        g = self.global_proj(global_features)
        g = global_alpha * g
        combined = torch.cat([h_last, g], dim=1)
        #combined = torch.cat([h_last, global_features], dim=1)

        return self.fc(combined).squeeze(1)


# -----------------
# Training
# -----------------
def train(args):



    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_files = glob.glob(os.path.join(args.data_dir, "real", "*.csv"))
    fake_files = glob.glob(os.path.join(args.data_dir, "fake", "*.csv"))

    # Stratified split: Same amount of items in fake and real
    real_items = [(p, 0) for p in real_files]
    fake_items = [(p, 1) for p in fake_files]

    random.shuffle(real_items)
    random.shuffle(fake_items)

    if args.final_train:
        print("FINAL TRAINING MODE: Model is trained on validation and test set. No validation metrics or early stopping are used during training.")
        train_items = real_items + fake_items
        val_items = []
    else:
        split_real = int(0.8 * len(real_items))
        split_fake = int(0.8 * len(fake_items))

        train_items = real_items[:split_real] + fake_items[:split_fake]
        val_items = real_items[split_real:] + fake_items[split_fake:]

        if not train_items or not val_items:
            raise ValueError(f"No CSV files found in {args.data_dir}/real and {args.data_dir}/fake")

    random.shuffle(train_items)
    random.shuffle(val_items)

    # Global means for normalization
    global_mean, global_std = compute_global_norm(train_items, args.seq_max_len)
    seq_mean, seq_std = compute_seq_norm(train_items, args.seq_max_len)   # Also the sequence norm you calculate globally for the training set

    train_dataset = SymbolDataset(
        train_items,
        seq_max_len=args.seq_max_len,
        global_mean=global_mean,
        global_std=global_std,
        seq_mean=seq_mean,
        seq_std=seq_std,
    )

    # Load datasets
    train_loader = DataLoader(
        train_dataset,
        #SymbolDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    if not args.final_train:

        val_dataset = SymbolDataset(
            val_items,
            seq_max_len=args.seq_max_len,
            global_mean=global_mean,
            global_std=global_std,
            seq_mean=seq_mean,
            seq_std=seq_std,
        )

        val_loader = DataLoader(
            val_dataset,
            #SymbolDataset(val_items),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
    else:
        val_loader = None

    model = SmallGRU(hidden_size=args.hidden_size, dropout=args.dropout).to(device)

    # How many model parameters do I have?
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print("Parameters per layer:")
    for name, p in model.named_parameters():
        print(name, p.numel())

    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer
    if args.weight_decay == 0.0:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )


    # Variables to save the best model (we want to save the one from an overfitted model)
    best_auc = -float("inf")
    best_state = None
    best_threshold = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        total_train_loss, total_train = 0.0, 0
        for x, lengths, g, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            g = g.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths, g)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * len(y)
            total_train += len(y)

        train_loss = total_train_loss / max(total_train, 1)

        if not args.final_train:    # No evaluation and early stopping for final training

            val_loss, val_acc, val_auc, val_threshold, val_acc_opt = evaluate(model, val_loader, device)

            # Save best model (to be able to regress to earlier epoch)
            improved = not np.isnan(val_auc) and val_auc > best_auc + args.min_delta # Min-delta for early stopping
            if improved:
                best_auc = val_auc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                best_threshold = val_threshold
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"epoch {epoch + 1:02d} | "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.3f} "
                f"val_auc={val_auc:.3f} "
                f"val_acc_opt={val_acc_opt:.3f} "
                f"best_val_thresh={val_threshold:.2f} "
            )

            # Early stopping
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best val AUC={best_auc:.3f}"
                )
                break

        else:
            print(
                f"epoch {epoch + 1:02d} | "
                f"FINAL TRAINING MODE | "
                f"train_loss={train_loss:.4f}"
            )

    if args.final_train:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        best_threshold = 0.5

    if best_state is not None:
        model.load_state_dict(best_state)

    # Better to use a checkpoint, so it is clear what the architecture etc was like
    # Had to change it a bit like below due to unpickling error. Dtypes are important for that.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "seq_max_len": args.seq_max_len,
        "global_mean": torch.tensor(global_mean, dtype=torch.float32),
        "global_std": torch.tensor(global_std, dtype=torch.float32),
        "seq_mean": torch.tensor(seq_mean, dtype=torch.float32),
        "seq_std": torch.tensor(seq_std, dtype=torch.float32),
        "threshold": float(best_threshold),
    }

    torch.save(checkpoint, args.model_file)
    if args.final_train:
        print(
            f"Saved final model to {args.model_file} with threshold={best_threshold:.2f}")
    else:
        print(f"Saved best model to {args.model_file} with AUC={best_auc:.3f}")

    # Summary for HPC testing (to not open every log file every time)
    results_file = "runs/results.csv"
    row = {
        "model_file": args.model_file,
        "final_train": args.final_train,
        "best_threshold": best_threshold,
        "total_params": total_params,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seq_max_len": args.seq_max_len,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_auc": best_auc if not args.final_train else None,
    }

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode="a", header=False, index=False)
    else:
        df.to_csv(results_file, index=False)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, total_correct, total = 0.0, 0, 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, lengths, g, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            g = g.to(device)
            y = y.to(device)

            logits = model(x, lengths, g)

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
    except ValueError:
        auc = float("nan")

    # Find the best threshold
    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)

    thresholds = np.linspace(0.05, 0.95, 91)

    best_threshold = max(
        thresholds,
        key=lambda t: ((all_probs_np >= t) == all_labels_np).mean()
    )

    best_acc = ((all_probs_np >= best_threshold) == all_labels_np).mean()

    return avg_loss, acc, auc, best_threshold, best_acc


def test_on_labeled_set(test_dir, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_file, map_location=device, weights_only=True)

    hidden_size = int(checkpoint["hidden_size"])
    dropout = float(checkpoint.get("dropout", 0.0))

    model = SmallGRU(hidden_size=hidden_size, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    seq_max_len = int(checkpoint["seq_max_len"])
    global_mean = checkpoint["global_mean"].cpu().numpy()
    global_std = checkpoint["global_std"].cpu().numpy()
    seq_mean = checkpoint["seq_mean"].cpu().numpy()
    seq_std = checkpoint["seq_std"].cpu().numpy()

    threshold = 0.5#checkpoint.get("threshold", 0.5)
    if threshold is None:
        threshold = 0.5
    threshold = float(threshold)

    items = []
    for path in glob.glob(os.path.join(test_dir, "real", "*.csv")):
        items.append((path, 0))
    for path in glob.glob(os.path.join(test_dir, "fake", "*.csv")):
        items.append((path, 1))

    y_true = []
    y_prob = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for path, label in sorted(items):
            seq, global_features = read_sequence(path, seq_max_len)

            seq = (seq - seq_mean) / seq_std
            global_features = (global_features - global_mean) / global_std

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)
            g = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0).to(device)

            logit = model(x, lengths, g)
            prob_fake = torch.sigmoid(logit).item()
            pred = int(prob_fake >= threshold)

            y_true.append(label)
            y_prob.append(prob_fake)
            y_pred.append(pred)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    print(f"Test samples: {len(y_true)}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test accuracy @ threshold {threshold:.3f}: {acc:.4f}")

    return {
        "auc": auc,
        "accuracy": acc,
        "threshold": threshold,
        "n": len(y_true),
    }


if __name__ == "__main__":
    train(parse_args())
    #test_on_labeled_set("symbols_split/test", "models/gru_final_model_1.pth")

