# Assignment: Classification of fake symbols.
# We will import the `load_and_predict()` function below to assess your assignment.

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


# -----------------
# Preprocessing
# -----------------


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


# -----------------
# Model architecture
# -----------------
class SmallGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, global_dropout=0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=11,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.global_proj = nn.Sequential(  # Todo: Bottleneck
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
# Required eval function
# -----------------
def load_and_predict(directory, model_file):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
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

    threshold = checkpoint.get("threshold", 0.5)
    if threshold is None:
        threshold = 0.5
    threshold = float(threshold)

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
            lengths = torch.tensor([len(seq)], dtype=torch.long).to(device)
            g = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0).to(device)

            # Query the model with the data in order to get the predicted class probabilities for each instance.
            logit = model(x, lengths, g)
            prob_fake = torch.sigmoid(logit).item()

            # Convert probabilities to labels (integer numbers): 0 for "real" and 1 for "fake".
            pred = int(prob_fake >= threshold)
            pred_dict[os.path.abspath(path)] = pred

    # Return a dictionary where keys are absolute file paths and values are the predicted labels for each file
    return pred_dict


# Local smoke test. To run:
# python eval.py /symbols_split/test_flat model.pth
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("model_file")
    args = parser.parse_args()

    predictions = load_and_predict(args.directory, args.model_file)
    for path, pred in predictions.items():
        print(f"{path}\t{pred}")
