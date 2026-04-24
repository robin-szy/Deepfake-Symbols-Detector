from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np

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