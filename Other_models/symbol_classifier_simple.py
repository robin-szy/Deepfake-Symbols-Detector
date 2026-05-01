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

# Todo: Normalize? If not, I'll also need to think about the padding during batching and how this affects angular velocities etc.
# Todo: change from LSTM to GRU

# -----------------
# Simple config
# -----------------
DATA_DIR = "./symbols"          # must contain real/ and fake/
MODEL_FILE = "model.pth"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
MAX_LEN = 512 #256
SEED = 42
HIDDEN_SIZE = 16


# -----------------
# Utilities
# -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_sequence(path):
    """Read one CSV and return a normalized T x 2 array with x/y coordinates."""
    df = pd.read_csv(path, sep="\t")
    #df.columns = df.columns.str.strip()
    df.columns = [c.lower().strip() for c in df.columns]

    xy = df[["x", "y"]].to_numpy(dtype=np.float32)

    # Normalize each symbol independently.
    xy = xy - xy.mean(axis=0, keepdims=True)
    scale = xy.std() + 1e-6
    xy = xy / scale

    # Keep long sequences manageable.
    if len(xy) > MAX_LEN:
        # Todo: Later when finished, for delivery: Remove the print
        print(f"Warning: For the sequence {path}, the maximum length was adapted.")
        idx = np.linspace(0, len(xy) - 1, MAX_LEN).astype(int)
        xy = xy[idx]

    return xy


class SymbolDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = read_sequence(path)
        return torch.tensor(x), torch.tensor(label, dtype=torch.float32)


def collate_batch(batch):
    """Pad variable-length sequences in a batch."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    feat_dim = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_len, feat_dim)
    # x = torch.zeros(len(seqs), max_len, 2)    # Only for x, y
    for i, seq in enumerate(seqs):
        x[i, : seq.shape[0]] = seq

    y = torch.stack(labels)
    return x, lengths, y


# -----------------
# Model
# -----------------
class SmallGRU(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(h[-1]).squeeze(1)


# -----------------
# Training
# -----------------
def train():

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_files = glob.glob(os.path.join(DATA_DIR, "real", "*.csv"))
    fake_files = glob.glob(os.path.join(DATA_DIR, "fake", "*.csv"))

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
        raise ValueError(f"No CSV files found in {DATA_DIR}/real and {DATA_DIR}/fake")

    random.shuffle(train_items)
    random.shuffle(val_items)

    # Load datasets
    train_loader = DataLoader(
        SymbolDataset(train_items),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        SymbolDataset(val_items),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = SmallGRU().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Variables to save the best model (we want to save the one from an overfitted model)
    best_auc = -float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, total_train = 0.0, 0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * len(y)
            total_train += len(y)

        train_loss = total_train_loss / max(total_train, 1)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, device)

        # Save best model (to be able to regress to earlier epoch)
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} "
            f"val_auc={val_auc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Better to use a checkpoint, so it is clear what the architecture etc was like
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_size": HIDDEN_SIZE,
        "max_len": MAX_LEN
    }
    torch.save(checkpoint, MODEL_FILE)
    print(f"Saved best model to {MODEL_FILE} with AUC={best_auc:.3f}")

    # Todo: Add checkpoint


def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, total_correct, total = 0.0, 0, 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, lengths, y in loader:
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
    except ValueError:
        auc = float("nan")

    return avg_loss, acc, auc


# -----------------
# Required eval function
# -----------------
def load_and_predict(directory, model_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallGRU().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    pred_dict = {}
    paths = sorted(glob.glob(os.path.join(directory, "*.csv")))

    with torch.no_grad():
        for path in paths:
            seq = read_sequence(path)
            x = torch.tensor(seq).unsqueeze(0).to(device)
            lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)

            prob_fake = torch.sigmoid(model(x, lengths)).item()
            pred = int(prob_fake >= 0.5)
            pred_dict[os.path.abspath(path)] = pred

    return pred_dict


if __name__ == "__main__":
    train()
