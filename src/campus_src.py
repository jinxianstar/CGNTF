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


def plot_predictions(model, X_test, y_test, start=0, end=400):
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
    plt.plot(y_test[start:end], label='Real Value')
    plt.plot(predicted[start:end], label='Predicted Value', alpha=0.7)
    plt.title('Real vs Predicted Value')
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