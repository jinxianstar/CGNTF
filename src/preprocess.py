import numpy as np

from sklearn.preprocessing import StandardScaler


def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


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
