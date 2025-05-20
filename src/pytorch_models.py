import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm


# ------------------------------
# Data Processing Functions
# ------------------------------

def extract_values(df: pd.DataFrame, column: str = 'data') -> np.ndarray:
    """
    Extracts and casts the column to float numpy array.
    """
    return df[column].values.astype(float)


def split_indices(length: int, lag: int, train_ratio=0.75, val_ratio=0.125):
    N = length - lag
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val
    return n_train, n_val, n_test


def fit_scaler(values: np.ndarray, lag: int, n_train: int) -> StandardScaler:
    scaler = StandardScaler()
    to_fit = values[: n_train + lag].reshape(-1,1)
    scaler.fit(to_fit)
    return scaler


def transform_values(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(values.reshape(-1,1)).flatten()


def create_sequences(values: np.ndarray, lag: int):
    X, y = [], []
    for i in range(len(values) - lag):
        X.append(values[i:i+lag])
        y.append(values[i+lag])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y


class TS_Dataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X, y, n_train, n_val, batch_size=32):
    # split
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    # loaders
    train_loader = DataLoader(TS_Dataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TS_Dataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TS_Dataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ------------------------------
# Model Definitions
# ------------------------------


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else None)
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, : -self.conv1.padding[0]]
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = out[:, :, : -self.conv2.padding[0]]
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            layers.append(TemporalBlock(in_ch, out_ch,
                                        kernel_size=kernel_size,
                                        dilation=2**i,
                                        dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.network(x)
        out = y[:, :, -1]
        return self.linear(out)

class FeedforwardNN(nn.Module):
    """
    Simple fully-connected network: input -> hidden -> output.
    """
    def __init__(self, input_size: int, hidden_size: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)
    
    
class LSTMModel(nn.Module):
    """
    LSTM-based model: sequences -> LSTM -> output.
    """
    def __init__(self, seq_len: int, input_dim: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        # reshape to (batch, seq_len, input_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.input_dim)
        out, _ = self.lstm(x)            # (batch, seq_len, hidden_size)
        last = out[:, -1, :]             # (batch, hidden_size)
        return self.fc(last)


# ------------------------------
# Training, Evaluation, Metrics
# ------------------------------


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=1000, patience=10, device=None):
    if device:
        model.to(device)
    best_val_loss = np.inf
    trigger = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, num_epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
            best_state = model.state_dict()
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_state)
                break
    return model, history


def evaluate_model(model, loader, scaler=None, device=None):
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.extend(out.cpu().squeeze().tolist())
            actuals.extend(yb.cpu().squeeze().tolist())
    preds_arr = np.array(preds).reshape(-1,1)
    acts_arr = np.array(actuals).reshape(-1,1)
    if scaler:
        preds_arr = scaler.inverse_transform(preds_arr).flatten()
        acts_arr = scaler.inverse_transform(acts_arr).flatten()
    return preds_arr, acts_arr


def compute_metrics(preds: np.ndarray, actuals: np.ndarray):
    errors = preds - actuals
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / actuals)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def plot_loss(history: dict):
    plt.figure(figsize=(8,4))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(actuals: np.ndarray, preds: np.ndarray):
    plt.figure(figsize=(8,4))
    plt.plot(actuals, label='Actual')
    plt.plot(preds, label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Prediction vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.show()