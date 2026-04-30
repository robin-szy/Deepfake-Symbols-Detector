import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import roc_auc_score
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./symbols")
    parser.add_argument("--model-file", default="model.pth")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seq-max-len", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=523)
    return parser.parse_args()

args = parse_args()


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
            r,
            #theta_unwrapped,
            theta_sin,
            theta_cos,
            dtheta,
            #abs_dtheta,
            dt,
            #stroke_id,
            stroke_change,
            #dt_stroke_transition,
        ],
        axis=1,
    ).astype(np.float32)

    global_features = np.array(
        [
            float((dt > 50).any()),
            #dt_stroke_transition.max(),
            len(df),
            time.max() - time.min(),
            r.mean(),
            r.std(),
            dtheta.std(),
            #abs_dtheta.mean()  # Todo: Put in?
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
    def __init__(self, hidden_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        #self.fc = nn.Sequential(
        #    nn.LayerNorm(hidden_size + 6),
        #    nn.Linear(hidden_size + 6, 1),
        #)

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size + 6),
            nn.Linear(hidden_size + 6, 16),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x, lengths, global_features):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, h = self.gru(packed)
        h_last = h[-1]

        combined = torch.cat([h_last, global_features], dim=1)

        return self.fc(combined).squeeze(1)


# -----------------
# Training
# -----------------
def train():

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_files = glob.glob(os.path.join(args.data_dir, "real", "*.csv"))
    fake_files = glob.glob(os.path.join(args.data_dir, "fake", "*.csv"))

    # Stratified split: Same amount of items in fake and real
    real_items = [(p, 0) for p in real_files]
    fake_items = [(p, 1) for p in fake_files]

    random.shuffle(real_items)
    random.shuffle(fake_items)

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

    val_dataset = SymbolDataset(
        val_items,
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
    val_loader = DataLoader(
        val_dataset,
        #SymbolDataset(val_items),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = SmallGRU(hidden_size=args.hidden_size).to(device)

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

    if best_state is not None:
        model.load_state_dict(best_state)

    # Better to use a checkpoint, so it is clear what the architecture etc was like
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_size": args.hidden_size,
        "seq_max_len": args.seq_max_len,
        "global_mean": global_mean,
        "global_std": global_std,
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "threshold": best_threshold,
    }
    torch.save(checkpoint, args.model_file)
    print(f"Saved best model to {args.model_file} with AUC={best_auc:.3f}")

    # Summary for HPC testing (to not open every log file every time)
    results_file = "runs/results.csv"
    row = {
        "model_file": args.model_file,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seq_max_len": args.seq_max_len,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_auc": best_auc,
    }

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


# -----------------
# Required eval function
# -----------------
def load_and_predict(directory, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(model_file, map_location=device)

    model = SmallGRU(hidden_size=checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    global_mean = checkpoint["global_mean"]
    global_std = checkpoint["global_std"]
    seq_mean = checkpoint["seq_mean"]
    seq_std = checkpoint["seq_std"]
    seq_max_len = checkpoint["seq_max_len"]

    pred_dict = {}
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))

    with torch.no_grad():
        for path in paths:
            # The following function read_sequence performs the two first steps required in eval.py:
            # (1) Read the data from the provided directory
            # (2) Prepare the data according to preprocessing pipeline of model training
            seq, global_features = read_sequence(path, seq_max_len)

            seq = (seq - seq_mean) / seq_std
            global_features = (global_features - global_mean) / global_std

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            g = torch.tensor(global_features, dtype=torch.float32).unsqueeze(
                0).to(device)
            lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)
            threshold = checkpoint.get("threshold", 0.5)

            # Query the model with the data in order to get the predicted class probabilities for each instance.
            prob_fake = torch.sigmoid(model(x, lengths, g)).item()

            # Convert probabilities to labels (integer numbers): 0 for "real" and 1 for "fake".
            pred = int(prob_fake >= threshold)
            pred_dict[os.path.abspath(path)] = pred

    # Return a dictionary where keys are absolute file paths and values are the predicted labels for each file
    return pred_dict


if __name__ == "__main__":
    train()





"""
def load_csv(file_path):
    # So, I don't forget to e.g. include the separator
    df = pd.read_csv(file_path, sep="\t")
    df.columns = df.columns.str.strip()
    return df

real_files = glob.glob("symbols/real/*.csv")
fake_files = glob.glob("symbols/fake/*.csv")

X = real_files + fake_files
y = [0]*len(real_files) + [1]*len(fake_files)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

sample_file = real_files[0]
df = load_csv(sample_file)

# Todo: Normalize x and y? Bounding box size: Maybe evaluate again with normalized x and y?

df['dx'] = df['x'].diff().fillna(0)
df['dy'] = df['y'].diff().fillna(0)

dx = df['dx'].to_numpy()
dy = df['dy'].to_numpy()

r = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(dy, dx)
dtheta = np.diff(theta, prepend=0)

df["r"] = r
df["sin_theta"] = np.sin(theta)
df["cos_theta"] = np.cos(theta)
df["dtheta"] = dtheta

# Normalize time
# Todo: Replace 10 by something smarter -> Most prominent number
df["time"] = df["time"] // 10
dt = df["time"].diff().fillna(0)
df["dt"] = dt

# Time when stroke ID changes
stroke_change = df["stroke_id"].diff().fillna(0) != 0
df["dt_stroke_transition"] = df["dt"] * stroke_change.astype(int)

# Global features (per file)
features = {
    "seq_len": len(df),
    "total_time": df["time"].max() - df["time"].min(),
    "mean_r": df["r"].mean(),
    "std_r": df["r"].std(),
    "std_dtheta": df["dtheta"].std(),
    "has_large_dt": (df["dt"] > 50).any(),
}

pass
# Todo: Pad sequence for RNN
# from torch.nn.utils.rnn import pad_sequence

"""