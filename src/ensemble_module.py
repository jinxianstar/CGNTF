
def _softmax(z, axis=1):
    z = np.asarray(z)
    z = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

class EnsembleModel:
    """簡化版均權 (1/n) ensemble，專用於 regression"""
    def __init__(self, models):
        self.models = models
        self.n = len(models)

    def predict(self, X):
        preds = [np.asarray(m.predict(X)) for m in self.models]
        return np.mean(preds, axis=0)


import json


import numpy as np
from typing import List, Tuple, Optional

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))

def _mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

class EnsembleDMA:
    """
    Dynamic Model Averaging for regression.
    - 多模型並行預測
    - 每個 batch 依據最近 window 的 RMSE 反比加權
    - 提供溫度 (temperature) 與動量 (momentum) 穩定權重
    """
    def __init__(
        self,
        models: List[object],
        window: int = 50,
        temperature: float = 1.0,   # <1 強化差異，>1 平滑差異
        momentum: float = 0.0,      # 0~0.9，與前一批權重做 EMA
        min_weight: float = 0.0,    # 權重下限（避免完全歸零）
        seed_weights: Optional[np.ndarray] = None  # 初始權重
    ):
        self.models = models
        self.n = len(models)
        self.window = int(window)
        self.temperature = max(1e-6, float(temperature))
        self.momentum = float(momentum)
        self.min_weight = float(min_weight)
        if seed_weights is None:
            self.weights = np.ones(self.n) / self.n
        else:
            w = np.asarray(seed_weights, dtype=float)
            self.weights = w / w.sum()
        # 儲存每個模型最近窗口的誤差
        self._err_buffers = [[] for _ in range(self.n)]
        self.weights_history = []

    def reset(self):
        self._err_buffers = [[] for _ in range(self.n)]
        self.weights_history = []
        # 保留 self.weights 作為起點，不清空

    def _update_weights(self, batch_preds: np.ndarray, y_true: np.ndarray):
        # 計算每個模型這個 batch 的 RMSE，放進各自滑動窗
        for i in range(self.n):
            e = _rmse(y_true, batch_preds[i])
            buf = self._err_buffers[i]
            buf.append(e)
            if len(buf) > self.window:
                buf.pop(0)

        # 用窗口平均 RMSE 反比加權
        mean_errs = np.array([np.mean(b) if len(b) else np.inf for b in self._err_buffers])
        inv = 1.0 / (mean_errs + 1e-8)
        # 溫度縮放（temperature<1 放大差異）
        inv = inv ** (1.0 / self.temperature)

        # 施加下限，避免某模型被「永久淘汰」
        inv = np.maximum(inv, self.min_weight * inv.sum())

        new_w = inv / inv.sum()

        # 動量平滑（與上一批權重做 EMA）
        if self.momentum > 0:
            new_w = self.momentum * self.weights + (1 - self.momentum) * new_w
            new_w = np.maximum(new_w, 1e-12)
            new_w = new_w / new_w.sum()

        self.weights = new_w
        self.weights_history.append(self.weights.copy())

    def predict_stream(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 128,
        update_every: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        逐批預測；若提供 y，則每批更新權重。
        回傳: (y_pred_all, weights_history 2D [t_steps, n_models])
        """
        N = len(X)
        y_pred_all = []

        # 步進
        for t in range(0, N, batch_size):
            Xb = X[t:t + batch_size]
            # 每個模型對 batch 的預測
            preds = [np.asarray(m.predict(Xb)) for m in self.models]  # list of (B,) or (B,1)
            preds = [p.ravel() for p in preds]
            P = np.stack(preds, axis=0)  # (n_models, B)

            # 加權
            yb_pred = np.tensordot(self.weights, P, axes=(0, 0))
            y_pred_all.append(yb_pred)

            # 是否更新權重
            if y is not None and ((t // batch_size) % max(1, update_every) == 0):
                yb_true = y[t:t + batch_size].ravel()
                self._update_weights(P, yb_true)

        y_pred_all = np.concatenate(y_pred_all, axis=0)
        wh = np.array(self.weights_history) if self.weights_history else np.zeros((0, self.n))
        return y_pred_all, wh



import matplotlib.pyplot as plt

def dma_plot_predictions(
    ensemble_dma: EnsembleDMA,
    X: np.ndarray,
    y: np.ndarray,
    start: Optional[int] = None,
    end: Optional[int] = None,
    title: str = "",
    batch_size: int = 128,
    update_every: int = 1
):
    # 逐批動態預測
    y_pred, w_hist = ensemble_dma.predict_stream(X, y, batch_size=batch_size, update_every=update_every)

    s = 0 if start is None else int(start)
    e = len(y) if end is None else int(end)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(s, e), y[s:e], label="Ground Truth")
    ax.plot(range(s, e), y_pred[s:e], label="DMA Prediction")
    ax.set_title(title)
    ax.set_xlabel("Index / Time")
    ax.set_ylabel("Target")
    ax.legend()
    plt.show()

    # 如果你想看權重隨時間的變化（可選）
    if w_hist.shape[0] > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        for k in range(w_hist.shape[1]):
            ax2.plot(w_hist[:, k], label=f"w_model{k}")
        ax2.set_title("DMA Weights over Batches")
        ax2.set_xlabel("Batch index (updates)")
        ax2.set_ylabel("Weight")
        ax2.legend()
        plt.show()

    return y_pred, w_hist

def dma_report_results(
    ensemble_dma: EnsembleDMA,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    update_every: int = 1,
    verbose: bool = True
):
    y_pred, _ = ensemble_dma.predict_stream(X, y, batch_size=batch_size, update_every=update_every)
    rmse = _rmse(y, y_pred)
    mae  = _mae(y, y_pred)
    mape = _mape(y, y_pred)
    if verbose:
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "y_pred": y_pred}

