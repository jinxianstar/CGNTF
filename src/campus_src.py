import numpy as np

def create_dataset(data, n_steps, target_index):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, target_index])
    return np.array(X), np.array(y)


def split_dataset(X, y, train_ratio=0.75, validation_ratio=0.15):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    - X: Features (NumPy array or similar)
    - y: Labels (NumPy array or similar)
    - train_ratio: Proportion of data to be used for training
    - validation_ratio: Proportion of data to be used for validation

    Returns:
    - X_train, X_validation, X_test
    - y_train, y_validation, y_test
    """
    total_samples = X.shape[0]

    num_train = int(total_samples * train_ratio)
    num_validation = int(total_samples * validation_ratio)

    index_train = num_train
    index_validation = index_train + num_validation

    X_train = X[:index_train]
    X_validation = X[index_train:index_validation]
    X_test = X[index_validation:]

    y_train = y[:index_train]
    y_validation = y[index_train:index_validation]
    y_test = y[index_validation:]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


from sklearn.preprocessing import StandardScaler
def scaling(X_train, X_validation, X_test, max_num_of_features):
    # 初始化标准化器

    # 初始化用于存储缩放后的数据的变量
    X_train_scaled = np.copy(X_train)
    X_validation_scaled = np.copy(X_validation)
    X_test_scaled = np.copy(X_test)

    # 对每个特征进行标准化处理
    for i in range(max_num_of_features):
        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        # 对训练数据进行标准化
        X_train_feature = X_train[:, :, i]  # 提取对应特征
        X_train_feature_scaled = scaler.fit_transform(X_train_feature.reshape(-1, 1)).reshape(X_train_feature.shape)
        X_train_scaled[:, :, i] = X_train_feature_scaled  # 将标准化后的数据放回

        #print(scaler.mean_)
        #print(scaler.scale_)
        # 对验证集和测试集使用相同的转换
        X_validation_feature = X_validation[:, :, i]
        X_validation_feature_scaled = scaler.transform(X_validation_feature.reshape(-1, 1)).reshape(X_validation_feature.shape)
        X_validation_scaled[:, :, i] = X_validation_feature_scaled

        X_test_feature = X_test[:, :, i]
        X_test_feature_scaled = scaler.transform(X_test_feature.reshape(-1, 1)).reshape(X_test_feature.shape)
        X_test_scaled[:, :, i] = X_test_feature_scaled
    return X_train_scaled, X_validation_scaled, X_test_scaled

# 定义噪声添加的函数
def add_noise(data, noise_level=0.01):
    # 生成与数据形状相同的随机噪声
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    # 仅对非0和非1的元素添加噪声
    noise_mask = (data != 0) & (data != 1)
    data_noisy = data.copy()
    data_noisy[noise_mask] += noise[noise_mask]
    return data_noisy




"""
MODELS
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN  # Make sure you have installed the tcn package
from keras.layers import Conv1D, LSTM, MaxPooling1D


def build_model_cnn_lstm(look_back, n_features):
    model = Sequential()
    model.add(Conv1D(filters=2, kernel_size=2, activation='relu', input_shape=(look_back, n_features)))
    model.add(
        MaxPooling1D(pool_size=2, strides=1, padding="valid")
    )
    #model.add(LSTM(100, activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')

    # Summary of the model
    model.summary()
    return model
from tensorflow.keras.layers import Dense, Dropout

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))  # RMSE 計算方式

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense

def build_model_GRU_with_Conv1D(look_back, n_features):
    model = Sequential()

    # Conv1D layer
    model.add(Conv1D(filters=2, kernel_size=2, activation='relu', input_shape=(look_back, n_features)))

    # First GRU layer
    model.add(GRU(100, activation='relu', return_sequences=True))

    # Second GRU layer
    model.add(GRU(100, activation='relu'))

    # Output layer
    model.add(Dense(1))

    # Compile with MAE
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae')

    model.summary()
    return model


def build_model_TCN(look_back, n_features):
    model = Sequential()
    model.add(TCN(input_shape=(look_back, n_features),
                  return_sequences=False,
                  kernel_size=2,
                  nb_filters=64,
                  dilations=[1, 2, 4, 8, 16, 32],
                  padding='causal',
                  use_skip_connections=True,
                  activation='relu'))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=rmse)  # 設定 RMSE 作為損失函數

    model.summary()
    return model




def build_model_LSTM(look_back, n_features):
    model = Sequential()

    # 第一層 LSTM（回傳序列以便第二層 LSTM 接收）
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(look_back, n_features)))

    # 第二層 LSTM（不回傳序列）
    model.add(LSTM(50, activation='relu'))

    # 輸出層
    model.add(Dense(1))

    # 編譯模型，使用 MSE
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')

    model.summary()
    return model

def build_model_CNN_LSTM(look_back, n_features):
    model = Sequential()

    # 第一層 Conv1D + ReLU
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(look_back, n_features)))
    model.add(MaxPooling1D(pool_size=2))

    # 第二層 Conv1D + ReLU
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM 層
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu'))

    # 展平並輸出
    model.add(Flatten())
    model.add(Dense(1))  # 輸出層

    # 編譯模型，使用 MAE
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mae')

    model.summary()
    return model


def build_model_double_TCN(look_back, n_features):
    model = Sequential()
    model.add(TCN(input_shape=(look_back, n_features),
                  return_sequences=True,
                  kernel_size=2,
                  nb_filters=64,
                  dilations=[1, 2, 4, 8, 16, 32],
                  padding='causal',
                  use_skip_connections=True,
                  activation='relu'))
    model.add(TCN(input_shape=(look_back, n_features),
              return_sequences=False,
              kernel_size=2,
              nb_filters=64,
              dilations=[1, 2, 4, 8, 16, 32],
              padding='causal',
              use_skip_connections=True,
              activation='relu'))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=rmse)

    # Summary of the model
    model.summary()
    return model


"""
    EVALUATIONS
"""
import matplotlib.pyplot as plt

def plot_loss(history):
    """
    Plot training and validation loss from a Keras history object.

    Parameters:
    - history: A Keras History object returned by model.fit()
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(model, X_test, y_test, start=0, end=400, label="Predicted", title=""):
    """
    Plot real vs. predicted values for a model on test data.

    Parameters:
    - model: Trained model with a predict method
    - X_test: Test features
    - y_test: True values for the test set
    - start: Starting index for plotting
    - end: Ending index for plotting
    """
    print(f"Testing Length: {len(y_test)}")

    predicted = model.predict(X_test).reshape(-1, 1)

    plt.figure(figsize=(10, 4))
    plt.plot(y_test[start:end], label='Real Traffic')
    plt.plot(predicted[start:end], label=label, alpha=0.7)
    plt.title(f"{title}")
    plt.ylabel('Value')
    plt.xlabel('Time Point')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt

def evaluate_regression(y_true, y_pred, print_result=True):
    """
    輸入真實值和預測值，回傳各種回歸評估指標
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
    if print_result:
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R^2 Score: {r2}")
        print(f"Mean Absolute Percentage Error: {mape}")
    return results


def evaluate_regression_above_half_range(y_true, y_pred, print_result=True):
    """
    只对 y_true > (max(y_true)-min(y_true)) / 2 的子集计算回归指标
    """
    # 1. 将输入转为 NumPy 数组（如果不是的话）
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 2. 计算 y_true 的最大值和最小值
    y_max = y_true.max()
    y_min = y_true.min()
    # 3. 计算阈值：range 的一半
    half_range = (y_max - y_min) / 2.0

    # 4. 过滤出 y_true > half_range 的索引
    mask = y_true > half_range
    # 如果想要取“位于上半区间”而不是直接 vs. 0，也可以改为：
    # midpoint = y_min + (y_max - y_min) / 2.0
    # mask = y_true > midpoint

    # 5. 取出子集
    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]

    # 如果没有样本满足条件，直接返回空或提示
    if y_true_sub.size == 0:
        if print_result:
            print("没有样本的 y_true 大于  (max-min)/2，无法计算指标。")
        return {}

    # 6. 计算各指标
    mse = mean_squared_error(y_true_sub, y_pred_sub)
    mae = mean_absolute_error(y_true_sub, y_pred_sub)
    rmse = sqrt(mse)
    r2 = r2_score(y_true_sub, y_pred_sub)
    mape = mean_absolute_percentage_error(y_true_sub, y_pred_sub)

    results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

    # 7. 可选：打印结果
    if print_result:
        print(f"筛选后样本数：{y_true_sub.size}（原始样本数：{y_true.size}）")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R^2 Score: {r2}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}")

    return results

# ------------------------------------------------------------------
# 1. 定義「CAP cost‐sensitive loss」，適用於單一輸出 (scalar)
# ------------------------------------------------------------------
def cap_loss(C_SLA, C_over):
    """
    回傳一個自訂 loss 函數，計算單一輸出容量與真實流量之間的
    過度配置成本 + SLA 違約成本。
    """
    def loss_fn(y_true, y_pred):
        # y_true, y_pred shape = (batch_size, 1)
        diff = y_pred - y_true  # shape=(batch_size, 1)

        # 過度配置 = max(y_pred - y_true, 0)
        over = tf.maximum(diff, 0.0)

        # SLA 違約 = max(y_true - y_pred, 0) = max(-diff, 0)
        violation = tf.maximum(-diff, 0.0)

        # 加權
        loss_over = C_over * over
        loss_sla  = C_SLA  * violation

        # batch 方向做平均
        return tf.reduce_mean(loss_over + loss_sla)
    return loss_fn


# ------------------------------------------------------------------
# 2. 建立「單輸出」的 CAP TCN 模型
# ------------------------------------------------------------------
def build_model_CAP(look_back, n_features, C_SLA=10.0, C_over=1.0):
    """
    對應單一輸出 (Dense(1)) 的 Capacity Forecasting 模型。
    look_back  : 歷史時間步長 (e.g. 24、48)。 
    n_features : 特徵數 (通常=1, 代表該時間序列的流量值)。
    C_SLA      : 當預測容量 < 真實流量時 (違約) 的成本權重。
    C_over     : 當預測容量 > 真實流量時 (過度配置) 的成本權重。
    """
    model = Sequential()

    # (1) TCN 隱藏層，output shape = (None, 64) 例如
    model.add(
        TCN(
            input_shape=(look_back, n_features),
            return_sequences=False,
            kernel_size=2,
            nb_filters=64,
            dilations=[1, 2, 4, 8, 16, 32],
            padding='causal',
            use_skip_connections=True,
            activation='relu',
        )
    )

    # (2) 幾層 Dense + Dropout
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.05))

    # (3) 最終輸出一個容量值
    model.add(Dense(1, activation='linear'))

    # (4) 用自訂的 CAP loss 編譯
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=cap_loss(C_SLA=C_SLA, C_over=C_over),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='MAE')]
    )

    model.summary()
    return model


def fgsm_inject_one_pos(
    model, X_np, y_np, epsilon,
    targeted=False, step_idx=None, feat_idx=None, max_num_of_features = 3
):
    """
    基于 FGSM 的单点/多点注入扰动函数（修改后：只做“增加”扰动，不做“减少”）。
    - model: tf.keras.Model，输出形状需与 y_np 一致
    - X_np: numpy.ndarray，shape=(batch_size, time_steps, num_features) 或 (batch_size, num_features)
    - y_np: numpy.ndarray，shape 与 model(X).numpy() 匹配
    - epsilon: 扰动幅度（float）
    - targeted: 是否做定向攻击；False 表示 untargeted（最大化 loss），True 表示定向（最小化 loss）
    - step_idx: int or None，当 None 时做“全局”扰动；否则做“定点”扰动（只针对 timestep=step_idx）
    - feat_idx: int or list/ndarray or None，当 None 时做“全局”扰动
    """

    # -------- 1. 转成 float32、声明为 tf.Variable --------
    X_adv = X_np.copy().astype(np.float32)
    X_var = tf.Variable(X_adv)

    # -------- 2. 强转标签为 tf.Tensor，确保 dtype 一致 --------
    y_tf = tf.convert_to_tensor(y_np, dtype=tf.float32)

    # -------- 3. 计算梯度 --------
    loss_fn = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        tape.watch(X_var)
        preds = model(X_var, training=False)
        loss = loss_fn(y_tf, preds)
        if targeted:
            loss = -loss  # 定向攻击：让预测更“贴近” y_np

    grad = tape.gradient(loss, X_var).numpy()
    grad_sign = np.sign(grad)  # 取符号：-1, 0, +1

    # -------- 3.1 只保留正方向扰动，将负号和 0 都置为 0 --------
    # 这样一来，grad_pos 只有 0 或 +1
    grad_pos = np.where(grad_sign > 0, 1.0, 0.0)

    # -------- 4. 应用扰动（仅往上增加） --------
    if (step_idx is not None) and (feat_idx is not None):
        # “定点”扰动：只针对特定 time step 和特征列
        mask = np.zeros_like(grad_pos)

        if isinstance(feat_idx, (list, tuple, np.ndarray)):
            # 三维示例：(batch_size, time_steps, num_features)
            mask[:, step_idx, feat_idx] = 1.0
        else:
            mask[:, step_idx, feat_idx] = 1.0

        # 只会在 grad_pos 为 1 的位置，叠加 +epsilon
        X_adv = X_adv + epsilon * (grad_pos * mask)

    else:
        # “全局”扰动：针对所有样本，所有 time step / 所有位置的前 3 列
        mask = np.zeros_like(grad_pos)
        if mask.ndim == max_num_of_features:
            mask[..., :max_num_of_features] = 1.0   # (batch_size, time_steps, num_features)
        elif mask.ndim == 2:
            mask[:, :max_num_of_features] = 1.0     # (batch_size, num_features)
        else:
            raise ValueError("X_np 的维度既不是 2 也不是 3，无法按前 3 列做扰动。")

        # 同样，只会在 grad_pos 为 1 的位置，叠加 +epsilon
        X_adv = X_adv + epsilon * (grad_pos * mask)

    return X_adv



def compute_violation_rate(model, X_test, y_test):
    """
    計算回歸型模型在測試資料上的 SLA 違約率。
    
    參數:
      - model: 已經訓練好的回歸模型，需有 predict 方法。
      - X_test: 測試集特徵，形狀 (num_samples, look_back, n_features)。
      - y_test: 測試集真實容量或流量標籤，形狀 (num_samples, 1) 或 (num_samples,)。
      
    回傳:
      - violation_rate: SLA 違約率 (float)，即 y_pred < y_true 的比例。
      - num_violations: 違約 (y_pred < y_true) 的樣本數。
      - total: 總樣本數。
    """
    # 1. 取得模型預測值
    y_pred = model.predict(X_test)
    
    # 2. 確保 y_test 與 y_pred 形狀匹配為 (num_samples,)
    y_true = y_test.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # 3. 計算違約次數 (預測 < 真實)
    violations = (y_pred_flat < y_true)
    num_violations = np.sum(violations)
    total = len(y_true)
    
    # 4. 違約率
    violation_rate = num_violations / total
    
    return violation_rate, int(num_violations), total



"""

MIXUP

"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class WrapperTCNWithFGSMMixup(Model):
    """
    TCN + FGSM + 標準 Mixup (per-example λ，使用 Beta 分布)
    
    - look_back, n_features : 交給 backbone 建模
    - max_num_of_features   : FGSM 注入上限
    - epsilon               : FGSM 擾動幅度
    - alpha                 : Mixup 的 Beta(α, α) 參數
    - step_idx / feat_idx   : 指定 FGSM 注入的時間步 / 特徵 (選填)
    """
    def __init__(self,
                 look_back,
                 n_features,
                 max_num_of_features,
                 epsilon=0.1,
                 alpha=0.3,
                 step_idx=None,
                 feat_idx=None,
                 model_name=None):
        super().__init__()
        self.max_num_of_features = max_num_of_features
        self.epsilon  = epsilon
        self.alpha    = alpha
        self.step_idx = step_idx
        self.feat_idx = feat_idx

        # 1) 建好 TCN backbone
        if model_name == 'TCN':
            self.backbone = build_model_TCN(look_back, n_features)
        elif model_name == "D-TCN":
            self.backbone = build_model_double_TCN(look_back, n_features)
        else:
            raise ValueError("model_name must be 'TCN' or 'D-TCN'.")

        # 2) 預設 optimizer / loss_fn
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss_fn   = lambda y_t, y_p: tf.sqrt(
            tf.reduce_mean(tf.square(y_t - y_p))
        )

    # ------------------------ compile -------------------------
    def compile(self, optimizer=None, loss=None, **kwargs):
        super().compile(run_eagerly=False, **kwargs)
        if optimizer is not None:
            self.optimizer = optimizer
        if loss is not None:
            self.loss_fn   = loss

        self.train_metric = tf.keras.metrics.Mean(name="train_rmse")
        self.val_metric   = tf.keras.metrics.Mean(name="val_rmse")

    # ------------------------ FGSM (呼叫 numpy 版) -------------
    def fgsm_generate(self, x_clean, y_true):
        def _fgsm_np(x_tf, y_tf):
            x_np = x_tf.numpy()
            y_np = y_tf.numpy()

            adv_np = fgsm_inject_one_pos(
                model   = self.backbone,
                X_np    = x_np,
                y_np    = y_np,
                epsilon = self.epsilon,
                targeted=False,
                step_idx=self.step_idx,
                feat_idx=self.feat_idx,
                max_num_of_features=self.max_num_of_features
            )
            return adv_np.astype(np.float32)

        x_adv = tf.py_function(
            func=_fgsm_np,
            inp=[x_clean, y_true],
            Tout=tf.float32
        )
        x_adv.set_shape(x_clean.shape)
        return x_adv

    # ------------------------ 標準 Mix-up (使用 Beta 分布) -------
    def mixup(self, x1, y1, x2, y2):
        """
        標準 Mixup 實現 (使用 Beta 分布):
        - 每個樣本有自己的 λ ~ Beta(α, α)
        - λ 同時應用於輸入和標籤
        """
        batch_size = tf.shape(x1)[0]
        
        # 1) 產生 per-example λ ~ Beta(α, α) (shape = [batch_size])
        # 使用 tf.random 的 beta 函數
        lam = tf.random.uniform(
            shape=[batch_size], 
            minval=0, 
            maxval=1,
            dtype=x1.dtype
        )
        
        # 使用逆變換採樣生成 Beta 分布
        # 這比使用 tensorflow_probability 更輕量
        def sample_beta(u, alpha):
            # 使用逆變換採樣 Beta(α, α)
            # 對於對稱 Beta 分布，CDF 的反函數可以簡化
            return tf.where(
                u < 0.5,
                0.5 * tf.pow(2 * u, 1 / alpha),
                1 - 0.5 * tf.pow(2 * (1 - u), 1 / alpha)
            )
        
        lam = sample_beta(lam, self.alpha)
        
        # 2) 擴展維度以匹配輸入和標籤形狀
        lam_x = tf.reshape(lam, [-1, 1, 1])  # (B, 1, 1) → 用於 (B, T, F)
        lam_y = tf.reshape(lam, [-1, 1])     # (B, 1)   → 用於 (B, 1)
        
        # 3) 執行混合
        x_mix = lam_x * x1 + (1 - lam_x) * x2
        y_mix = lam_y * y1 + (1 - lam_y) * y2
        
        return x_mix, y_mix

    # ------------------------ train_step (標準 Mixup) -----------
    def train_step(self, data):
        """
        標準 Mixup 訓練步驟:
        1. 生成對抗樣本
        2. 將原始樣本和對抗樣本合併
        3. 打亂後拆分為兩個集合
        4. 對兩個集合進行 Mixup
        5. 使用混合數據訓練
        """
        x_clean, y_true = data
        y_true = tf.reshape(y_true, [-1, 1])  # (B,) → (B,1)
        batch_size = tf.shape(x_clean)[0]

        # 1) 生成對抗樣本 (保持原始標籤)
        x_adv = self.fgsm_generate(x_clean, y_true)

        # 2) 合併原始樣本和對抗樣本
        x_all = tf.concat([x_clean, x_adv], axis=0)  # (2B, T, F)
        y_all = tf.concat([y_true, y_true], axis=0)  # (2B, 1)

        # 3) 打亂整個數據集
        total_size = 2 * batch_size
        indices = tf.random.shuffle(tf.range(total_size))
        x_shuffled = tf.gather(x_all, indices)
        y_shuffled = tf.gather(y_all, indices)

        # 4) 拆分為兩個集合 (各B個樣本)
        x1 = x_shuffled[:batch_size]
        y1 = y_shuffled[:batch_size]
        x2 = x_shuffled[batch_size:batch_size*2]
        y2 = y_shuffled[batch_size:batch_size*2]

        # 5) 執行標準 Mixup
        x_mix, y_mix = self.mixup(x1, y1, x2, y2)

        # 6) 前向傳播 + 反向傳播
        with tf.GradientTape() as tape:
            preds = self.backbone(x_mix, training=True)
            loss = self.loss_fn(y_mix, preds)

        grads = tape.gradient(loss, self.backbone.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_weights))

        self.train_metric.update_state(loss)
        return {"loss": self.train_metric.result()}

    # ------------------------ test_step -----------------------
    def test_step(self, data):
        x, y = data
        y = tf.reshape(y, [-1, 1])  # (B,) → (B,1)

        preds = self.backbone(x, training=False)
        val_loss = self.loss_fn(y, preds)

        self.val_metric.update_state(val_loss)
        return {"loss": self.val_metric.result()}

    # ------------------------ call ----------------------------
    def call(self, inputs, training=None):
        return self.backbone(inputs, training=training)

class WrapperTCNWithAT(Model):
    """
    TCN + FGSM 对抗训练 (AT-TDNS)
    
    - look_back, n_features : 交給 backbone 建模
    - max_num_of_features   : FGSM 注入上限
    - epsilon               : FGSM 擾動幅度
    - step_idx / feat_idx   : 指定 FGSM 注入的時間步 / 特徵 (選填)
    """
    def __init__(self,
                 look_back,
                 n_features,
                 max_num_of_features,
                 epsilon=0.1,
                 step_idx=None,
                 feat_idx=None,
                 model_name=None,
                 alpha=0.5,
                 mixed = False
                 ):  # 新增：对抗样本与原始样本的混合权重
        super().__init__()
        self.max_num_of_features = max_num_of_features
        self.epsilon  = epsilon
        self.step_idx = step_idx
        self.feat_idx = feat_idx
        self.alpha = alpha  # 控制对抗样本的混合比例 (0.0~1.0)
        self.mixed = mixed

        # 1) 建好 TCN backbone
        if model_name == 'TCN':
            self.backbone = build_model_TCN(look_back, n_features)
        elif model_name == "D-TCN":
            self.backbone = build_model_double_TCN(look_back, n_features)
        else:
            raise ValueError("model_name must be 'TCN' or 'D-TCN'.")

        # 2) 預設 optimizer / loss_fn
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss_fn   = lambda y_t, y_p: tf.sqrt(
            tf.reduce_mean(tf.square(y_t - y_p))
        )

    # ------------------------ compile -------------------------
    def compile(self, optimizer=None, loss=None, **kwargs):
        super().compile(run_eagerly=False, **kwargs)
        if optimizer is not None:
            self.optimizer = optimizer
        if loss is not None:
            self.loss_fn   = loss

        self.train_metric = tf.keras.metrics.Mean(name="train_rmse")
        self.val_metric   = tf.keras.metrics.Mean(name="val_rmse")

    # ------------------------ FGSM 生成对抗样本 -------------
    def fgsm_generate(self, x_clean, y_true):
        def _fgsm_np(x_tf, y_tf):
            x_np = x_tf.numpy()
            y_np = y_tf.numpy()

            adv_np = fgsm_inject_one_pos(
                model   = self.backbone,
                X_np    = x_np,
                y_np    = y_np,
                epsilon = self.epsilon,
                targeted=False,
                step_idx=self.step_idx,
                feat_idx=self.feat_idx,
                max_num_of_features=self.max_num_of_features
            )
            return adv_np.astype(np.float32)

        x_adv = tf.py_function(
            func=_fgsm_np,
            inp=[x_clean, y_true],
            Tout=tf.float32
        )
        x_adv.set_shape(x_clean.shape)
        return x_adv

    # ------------------------ AT 訓練步驟 -----------------------
    def train_step(self, data):
        """
        對抗訓練步驟:
        1. 生成對抗樣本
        2. 混合原始樣本與對抗樣本 (按權重 alpha)
        3. 前向傳播 + 反向傳播
        """
        x_clean, y_true = data
        y_true = tf.reshape(y_true, [-1, 1])  # (B,) → (B,1)
        batch_size = tf.shape(x_clean)[0]

        # 1. 生成對抗樣本 (保持原始標籤)
        x_adv = self.fgsm_generate(x_clean, y_true)

        #2. 混合原始樣本與對抗樣本 (按權重 alpha)
        #注意：这里采用论文 AT-TDNS 的加权混合策略
        if self.mixed == True:
            x_mixed = self.alpha * x_adv + (1 - self.alpha) * x_clean
            y_mixed = y_true  # 标签保持不变（回归任务无标签翻转）

            with tf.GradientTape() as tape:
                preds = self.backbone(x_mixed, training=True)
                loss = self.loss_fn(y_mixed, preds)
        else:
            # 3. 前向傳播 + 反向傳播
            with tf.GradientTape() as tape:
                preds = self.backbone(x_adv, training=True)
                loss = self.loss_fn(y_true, preds)

        grads = tape.gradient(loss, self.backbone.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_weights))

        self.train_metric.update_state(loss)
        return {"loss": self.train_metric.result()}

    # ------------------------ test_step -----------------------
    def test_step(self, data):
        x, y = data
        y = tf.reshape(y, [-1, 1])  # (B,) → (B,1)

        preds = self.backbone(x, training=False)
        val_loss = self.loss_fn(y, preds)

        self.val_metric.update_state(val_loss)
        return {"loss": self.val_metric.result()}

    # ------------------------ call ----------------------------
    def call(self, inputs, training=None):
        return self.backbone(inputs, training=training)
