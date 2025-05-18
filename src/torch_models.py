
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rmse_loss(y_pred, y_true):
    return torch.sqrt(F.mse_loss(y_pred, y_true))

# ----- Model Definitions -----
class LSTMModel(nn.Module):
    def __init__(self, look_back, n_features, hidden_size=50):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # take last time step
        out = self.fc(x[:, -1, :])
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self, look_back, n_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=128, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64*( ( (look_back-1)//2 -1)//2 ), 1)

    def forward(self, x):
        # x: (batch, time, features)
        x = x.permute(0, 2, 1)  # to (batch, features, time)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)  # back to (batch, time, features)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)

# For TCN, assume pytorch-tcn package installed: from tcn import TCN
try:
    from pytorch_tcn import TCN
    has_tcn = True
    print("have tcn")
except ImportError:
    has_tcn = False
    print("no tcn")




class SimpleNN(nn.Module):
    def __init__(self, look_back: int, n_features: int, hidden_dims: list = [128, 64, 32]):
        """
        一个简单的 MLP，用于时序回归。
        
        参数：
          look_back   -- 时间步长度
          n_features  -- 每个时间步特征数
          hidden_dims -- 隐藏层维度列表
        """
        super().__init__()
        input_dim = look_back * n_features
        
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.05))
            last_dim = h
        
        # 最后一层输出1个值
        layers.append(nn.Linear(last_dim, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算
        输入 x 形状: (batch, time_steps, features)
        输出 y 形状: (batch, 1)
        """
        # 扁平化
        x = x.reshape(x.size(0), -1)
        return self.model(x)
    
class TCNModel(nn.Module):
    def __init__(self, look_back, n_features):
        super().__init__()
        # PyTorch-TCN 参数与 Keras-TCN 对齐：
        # - num_inputs: 特征维度
        # - num_channels: 每层通道数列表，共6层，每层64
        # - kernel_size: 核大小=2
        # - dilations: 扩张列表，与 Keras 一致
        # - dropout: 0.05
        # - causal: True
        self.tcn = TCN(
            num_inputs=n_features,
            num_channels=[64] * 6,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32],
            dropout=0.05,
            causal=True
        )
        # 全连接层
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.05)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, time_steps, features)
        x = x.permute(0, 2, 1)  # → (batch, features, time_steps)
        y = self.tcn(x)        # → (batch, 64, time_steps)
        # 取最后一个时间步
        y = y[:, :, -1]         # → (batch, 64)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.dropout(y)
        return self.out(y)


class DoubleTCNModel(nn.Module):
    def __init__(self, look_back, n_features):
        super().__init__()
        self.tcn1 = TCN(input_size=n_features, num_channels=[64]*6, kernel_size=2, dropout=0.0)
        self.tcn2 = TCN(input_size=64, num_channels=[64]*6, kernel_size=2, dropout=0.0)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.05)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)

# ----- Training & Early Stopping -----
def train_with_early_stopping(model, train_data, val_data, epochs=1000, batch_size=128, patience=10, lr=1e-3):

    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,           # 在每个 epoch 开始时打乱所有滑窗样本
        drop_last=True,         # 丢弃最后一个不满 batch 的小批次，保证每个 batch 都是同样大小
        num_workers=4,          # 根据你的 CPU 核心数做调整，一般 2–8 之间
        pin_memory=True,        # 对 GPU 加速非常有帮助
        persistent_workers=True,# PyTorch ≥1.7：worker 进程在整个训练过程中保持活跃，减少重启开销
        prefetch_factor=2       # 每个 worker 预取 2 个 batch（默认就是 2，可根据显存/带宽微调）
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,          # 验证时不需要打乱
        drop_last=False,        # 保留最后一个可能更小的 batch，用于完整评估
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = nn.MSELoss()
    criterion = rmse_loss

    best_loss = np.inf
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                val_losses.append(criterion(y_pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # load best weights
    model.load_state_dict(torch.load('best_model.pth'))
    return history





# ----- Evaluation -----
def evaluate_regression(model, test_data):
    model.eval()
    loader = DataLoader(test_data, batch_size=len(test_data))
    x_test, y_test = next(iter(loader))
    x_test = x_test.to(device)
    with torch.no_grad():
        y_pred = model(x_test).cpu().numpy().reshape(-1)
    y_true = y_test.numpy().reshape(-1)

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\nMAPE: {mape:.4%}")
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# ----- Plotting -----
def plot_loss(history, figsize=(10,4)):
    plt.figure(figsize=figsize)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def plot_real_vs_predicted(
    model,
    X_test,
    y_test,
    start: int = 0,
    end: int = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    figsize: tuple = (10, 4),
    title: str = "Real vs Predicted Value",
    ylabel: str = "Value",
    xlabel: str = "Time Point",
    alpha: float = 0.7
):
    
    """
      model   -- 已训练好的 PyTorch 模型
      X_test  -- 测试集特征，numpy array 或 tensor，shape=(N, time, features)
      y_test  -- 测试集标签，numpy array 或 tensor，shape=(N,) 或 (N,1)
      start   -- 起始索引（默认 0）
      end     -- 截止索引（不含；None 表示到末尾）
      device  -- 推断设备
      figsize -- 图像大小
      title   -- 图标题
      ylabel  -- y 轴标签
      xlabel  -- x 轴标签
      alpha   -- 预测曲线透明度
    """
    # 1. 准备输入 tensor 并搬到 device
    if not torch.is_tensor(X_test):
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
    else:
        X_tensor = X_test.float()
    X_tensor = X_tensor.to(device)

    # 2. 推断
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_tensor)
    y_pred = y_pred_tensor.cpu().numpy().reshape(-1)

    # 3. 准备真实值数组
    if torch.is_tensor(y_test):
        y_true = y_test.cpu().numpy().reshape(-1)
    else:
        y_true = np.array(y_test).reshape(-1)

    # 4. 确定 end
    if end is None or end > len(y_true):
        end = len(y_true)

    print(f"Testing Length: {len(y_true)}")

    # 5. 绘图
    plt.figure(figsize=figsize)
    plt.plot(y_true[start:end], label='Real Value')
    plt.plot(y_pred[start:end], label='Predicted Value', alpha=alpha)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()