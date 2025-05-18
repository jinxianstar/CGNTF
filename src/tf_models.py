from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))  # RMSE 計算方式

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense
from tensorflow.keras.layers import LSTM, MaxPooling1D, Flatten
from tcn import TCN  # Make sure you have installed the tcn package

# LSTM
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


#CNN-LSTM
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

# DOUBLE TCN
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


def plot_model_loss(history, figsize=(10, 4)):
    """
    繪製訓練過程中的損失曲線。

    參數:
    - history: keras History 物件 (history.history 要包含 'loss' 和 'val_loss')
    - figsize: 圖片大小，tuple (寬, 高)
    """
    plt.figure(figsize=figsize)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt

def evaluate_regression(model, X_test, y_test, verbose=True):
    """
    使用模型对 X_test 做预测，并计算回归评估指标：
      - MAE  (Mean Absolute Error)
      - RMSE (Root Mean Squared Error)
      - R2   (Coefficient of Determination)
      - MAPE (Mean Absolute Percentage Error)
    
    参数：
      model     -- 已训练好的回归模型，需实现 .predict()
      X_test    -- 测试集特征，array-like
      y_test    -- 测试集标签，array-like
      verbose   -- 是否打印各项指标（默认 True）
    
    返回：
      dict 包含各项指标值
    """
    # 1. 预测
    y_pred = model.predict(X_test).reshape(-1, 1)
    
    # 2. 计算指标
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    metrics = {
        'MAE' : mae,
        'MSE' : mse,
        'RMSE': rmse,
        'R2'  : r2,
        'MAPE': mape
    }
    
    if verbose:
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE):  {mse:.4f}")
        print(f"Root MSE (RMSE):           {rmse:.4f}")
        print(f"R² Score:                  {r2:.4f}")
        print(f"Mean Absolute % Error (MAPE): {mape:.4%}")
    
    return metrics


from tensorflow.keras.callbacks import EarlyStopping

def train_with_earlystopping(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=1000,
    batch_size=128,
    patience=10,
    monitor='val_loss',
    restore_best_weights=True,
    verbose=1
):
    """
    使用 EarlyStopping 回调训练模型的通用函数。
    
    参数：
      model               -- 已构建（并可选地已编译）的 keras Model
      X_train, y_train    -- 训练集
      X_val, y_val        -- 验证集
      epochs              -- 最大训练轮数
      batch_size          -- 批大小
      patience            -- EarlyStopping 的耐心值
      monitor             -- EarlyStopping 监控的指标
      restore_best_weights-- 是否在训练结束时恢复最佳权重
      verbose             -- fit 的 verbosity
      
    返回：
      history 对象（keras.callbacks.History）
    """
    # 1. 构造 EarlyStopping 回调
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights
    )
    
    # 2. 训练
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    return history

