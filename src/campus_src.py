import numpy as np

def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, 1])
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
def scaling(X_train, X_validation, X_test, num_features=4):
    # 初始化标准化器

    # 初始化用于存储缩放后的数据的变量
    X_train_scaled = np.copy(X_train)
    X_validation_scaled = np.copy(X_validation)
    X_test_scaled = np.copy(X_test)

    # 对每个特征进行标准化处理
    for i in range(num_features):
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


def plot_predictions(model, X_test, y_test, start=0, end=400, label="Predicted"):
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
    plt.title(f"Real vs {label} Value")
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

def fgsm_inject_one_pos(model, X_np, y_np, epsilon, targeted=False, step_idx=None, feat_idx=None):
    X_adv = X_np.copy().astype(np.float32)
    y_tf = tf.convert_to_tensor(y_np, dtype=tf.float32)
    X_var = tf.Variable(X_adv)
    
    # 创建损失函数实例
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    with tf.GradientTape() as tape:
        tape.watch(X_var)
        preds = model(X_var, training=False)
        preds = tf.reshape(preds, y_tf.shape)
        
        # 使用损失函数实例计算损失
        loss = loss_fn(y_tf, preds)
        
        if targeted:  # 添加定向攻击支持
            loss = -loss

    grad = tape.gradient(loss, X_var).numpy()
    grad_sign = np.sign(grad)
    grad_sign[grad_sign < 0] = 0  # 负梯度归零（与原始逻辑一致）
    
    # 新增一行，查看被选中位置的 gs 值
    print("grad_sign 在 (step,feat)=", step_idx, feat_idx, "上的值：", 
        np.unique(grad_sign[:, step_idx, feat_idx]))

    # === 新增：创建特征掩码（只允许修改前3个特征） ===
    # 假设输入形状为 (batch, time_steps, features)
    feature_mask = np.zeros(X_adv.shape[-1])  # 特征维度的掩码
    feature_mask[:3] = 1.0  # 只允许修改前3个特征
    
    # 应用扰动
    if step_idx is not None and feat_idx is not None:
        # 定点扰动模式：只修改特定位置
        mask = np.zeros_like(grad_sign)
        mask[:, step_idx, feat_idx] = 1.0
        X_adv += epsilon * (grad_sign * mask)
    else:
        # 全局扰动模式：只修改前3个特征
        # 将特征掩码广播到整个张量
        broadcast_mask = np.zeros_like(X_adv)
        broadcast_mask[..., :3] = 1.0  # 所有批次和时间步的前3个特征
        
        # 应用扰动时只修改允许的特征
        X_adv += epsilon * (grad_sign * broadcast_mask)

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
