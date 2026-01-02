import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# -------------------------
# Config
# -------------------------
DATA_PATH = Path("data/processed/multi_asset_sequences.npz")

SEQ_LEN = 30
N_FEATURES = 3
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Dataset
# -------------------------
class MultiAssetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    data = np.load(DATA_PATH)
    X, y = data["X"], data["y"]

    split = int(len(X) * 0.8)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = MultiAssetDataset(X_train, y_train)
    val_ds = MultiAssetDataset(X_val, y_val)

    return train_ds, val_ds


# -------------------------
# Model
# -------------------------
class LSTMVolModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0
        )

        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze()


# -------------------------
# Training
# -------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total += loss.item() * len(y)

    return (total / len(loader.dataset)) ** 0.5


def eval_epoch(model, loader, criterion):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            loss = criterion(preds, y)
            total += loss.item() * len(y)

    return (total / len(loader.dataset)) ** 0.5


# -------------------------
# Main
# -------------------------
def main():
    train_ds, val_ds = load_data()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMVolModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Training multi-asset LSTM on {DEVICE}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for epoch in range(1, EPOCHS + 1):
        train_rmse = train_epoch(model, train_loader, optimizer, criterion)
        val_rmse = eval_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d} | "
            f"Train RMSE: {train_rmse:.6f} | "
            f"Val RMSE: {val_rmse:.6f}"
        )


if __name__ == "__main__":
    main()
