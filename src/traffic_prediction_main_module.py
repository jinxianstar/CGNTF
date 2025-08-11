import campus_src as cs

def load_data(n_steps, target_index, dataset_name, only_one_feature):
    '''读取并预处理数据。

    假設 df_encoded 是你的 DataFrame，且已經按時間順序排序
    '''
    # 获取当前脚本文件的位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed', f"{dataset_name}.csv")

    df = pd.read_csv(data_path)
    if "campus" in dataset_name:
        df = df.set_index('DateTime')
        if only_one_feature:
            df = df[["value_avg"]]
    else:
        df = df.set_index('date')

    X, y = cs.create_dataset(df.to_numpy(), n_steps, target_index)
    return X, y


def prepare_datasets(X, y, max_num_of_features, train_ratio, validation_ratio):
    '''拆分並縮放資料集。'''
    X_train, X_validation, X_test, y_train, y_validation, y_test = cs.split_dataset(
        X, y, train_ratio=train_ratio, validation_ratio=validation_ratio
    )

    # 數值縮放
    X_train, X_validation, X_test = cs.scaling(X_train, X_validation, X_test, max_num_of_features)
    # 轉成 float32（TensorFlow 更友好）
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_validation = np.array(X_validation, dtype=np.float32)
    y_validation = np.array(y_validation, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


# 忘記 mixup feature 2 了！！ (6/4)

def build_mixup_model(look_back, n_features, model_name, max_num_of_features, attack_method, epsilon=0.5, alpha=0.3, step_idx=None, feat_idx=None):
    '''
    构建带 FGSM + Mixup 的 TCN 模型。

    Parameters
    ----------
    look_back : int
        回看步數。
    n_features : int
        特徵維度。
    epsilon : float, default 0.5
        FGSM 擾動幅度。
    alpha : float, default 0.3
        Beta(α, α) 的形狀參數。
    step_idx : int | None
        若需定點打擾，可指定。
    feat_idx : list[int] | None
        若需定點打擾，可指定。
    '''

    # 2) compile：只要给一个 optimizer + loss_fn，与 build_model_TCN 中的保持一致即可
    model = cs.WrapperTCNWithFGSMMixup(
        look_back=look_back,
        n_features=n_features,
        epsilon=epsilon,   # FGSM 扰动幅度
        alpha=alpha,       # Beta(α,α)，Mixup 的形状参数
        step_idx=step_idx, # 如果要“定点打扰”，可以设置一个整数
        feat_idx=feat_idx,  # 如果要“定点打扰”，可以设置一个整数
        model_name=model_name,
        max_num_of_features=max_num_of_features,
        attack_method = attack_method 
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=lambda y_true, y_pred: tf.sqrt(
            tf.reduce_mean(tf.square(y_true - y_pred))
        )  # RMSE
    )
    return model


def build_normal_model(look_back, n_features, model_name):
    '''original models.'''
    if model_name == "TCN":
        return cs.build_model_TCN(look_back, n_features)
    elif model_name == "CNN-LSTM":
        return cs.build_model_CNN_LSTM(look_back, n_features)
    elif model_name == "CNN-GRU":
        return cs.build_model_GRU_with_Conv1D(look_back, n_features)
    elif model_name == "LSTM":
        return cs.build_model_LSTM(look_back, n_features)
    elif model_name == "D-TCN":
        return cs.build_model_double_TCN(look_back, n_features)



# 举几个常见的 α 值特点：
#
# α=1（Beta(1,1)）：均匀分布，λ 在 [0,1] 上等概率——可能落到 0.1、0.9、0.5，都是一样的概率。
#
# α=2 或 3（Beta(2,2) / Beta(3,3)）：比均匀分布更偏向中间，抽到 0.5 左右的几率会更高，但也还会有比较大的概率落到 0.2 或 0.8。
#
# α=5 或 10（Beta(5,5) / Beta(10,10)）：分布就非常集中在 0.5 附近，几乎不会抽出 0.1 或 0.9，大部分 λ 都在 0.4–0.6（甚至 0.45–0.55）之间。


from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=128):
    '''训练模型并返回 history。'''
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_validation, y_validation),
        callbacks=[early_stopping]
    )
    return history

def train_attack_model(model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=128):
    '''训练模型并返回 history。'''
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_validation, y_validation),
        callbacks=[early_stopping]
    )
    return history

import matplotlib.pyplot as plt

def plot_perturbation(X_orig, X_adv):
    plt.figure(figsize=(12,6))
    
    # 绘制原始信号
    plt.subplot(2,1,1)
    plt.plot(X_orig[0,:,0], label='Original')
    plt.title('Original Signal')
    
    # 绘制扰动差异
    plt.subplot(2,1,2)
    plt.plot(X_adv[0,:,0] - X_orig[0,:,0], 'r', label='Perturbation')
    plt.title('Adversarial Perturbation')
    plt.ylim(-epsilon*1.1, epsilon*1.1)
    
    plt.tight_layout()
    plt.show()





def evaluate_and_attack(model_mixup, model_normal, X_test, y_test, epsilon, step_idx, max_num_of_features, attack_method="FGSM", feat_idx=[0, 1, 2]):
    '''生成對抗樣本並返回它們。'''

    if attack_method == "FGSM":
        X_test_adv_mixup = cs.fgsm_inject_one_pos(
            model_mixup,
            X_test,
            y_test,
            epsilon,
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features = max_num_of_features
        )

        # 被普通模型干扰
        X_test_adv_normal = cs.fgsm_inject_one_pos(
            model_normal,
            X_test,
            y_test,
            epsilon,
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features=max_num_of_features
        )
    elif attack_method == "PGD":
        # 被 mixup 模型干扰
        X_test_adv_mixup = cs.pgd_inject_one_pos(
            model_mixup,
            X_test,
            y_test,
            epsilon=epsilon,                      # ✅ 改這裡
            num_iter=40,                   # ✅ 一定要補上這個，因為沒有預設值
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features=max_num_of_features
        )

        # 被普通模型干扰
        X_test_adv_normal = cs.pgd_inject_one_pos(
            model_normal,
            X_test,
            y_test,
            epsilon=epsilon,                      # ✅ 改這裡
            num_iter=40,                   # ✅ 同上，PGD 要知道疊幾步
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features=max_num_of_features
        )

    diff = X_test_adv_mixup - X_test

    # To show the element-wise difference, you can print the first few rows (or handle it as necessary)
    print(diff[:5])  # Show the first 5 examples' differences

    # If you want to see statistics for the differences, you could do:
    mean_diff = np.mean(diff, axis=0)
    std_diff = np.std(diff, axis=0)

    print("Mean of differences per feature:", mean_diff)
    print("Standard deviation of differences per feature:", std_diff)
    
    return X_test_adv_mixup, X_test_adv_normal

def report_results(model, X_test, y_test, above_half_range):
    predicted = model.predict(X_test).reshape(-1, 1)
    if above_half_range:
        return cs.evaluate_regression_above_half_range(y_test, predicted)
    else:
        return cs.evaluate_regression(y_test, predicted)


def attack_all_add_delta(X, step_idx=7, feat_idx=[], delta=0.2):
    """
    对 X 中所有样本，在指定的 step_idx 和 feat_idx 位置，
    将原值增加 pct * 原值（即乘以 1+pct）。

    参数：
        X (np.ndarray): 待“攻击”的三维数组，shape=(n_samples, n_steps, n_features)
        step_idx (int): 时间步的索引，比如第8步就传 7
        feat_idx (int): 特征的索引，比如第2个特征就传 1
        pct (float): 增加比例，默认 0.05（5%）

    返回：
        np.ndarray: 攻击后的数组副本
    """
    X_attacked = X.copy()
    for i in feat_idx:
        X_attacked[:, step_idx, i] += delta
    return X_attacked



import os
import numpy as np
import pandas as pd

def save_predictions(model, X, y_true, filename, start=None, end=None):
    """
    对给定 model 和数据 X 进行预测，并把 y_true 和 y_pred 按照 slice(start,end) 保存到 filename。
    
    参数：
      - model: 已训练好的模型，需支持 model.predict(X)
      - X:      输入特征，形状 (n_samples, n_features)
      - y_true: 真实标签，形状 (n_samples,) 或 (n_samples,1)
      - filename: 保存用的文件名（带 .csv 后缀）
      - start:  可选，起始索引（inclusive）
      - end:    可选，结束索引（exclusive）
    """
    # 预测
    y_pred = model.predict(X)
    
    # 切片
    if start is not None or end is not None:
        y_true_slice = y_true[start:end]
        y_pred_slice = y_pred[start:end]
    else:
        y_true_slice = y_true
        y_pred_slice = y_pred
    
    # 组织 DataFrame
    df = pd.DataFrame({
        'y_true': np.ravel(y_true_slice),
        'y_pred': np.ravel(y_pred_slice),
    })
    
    # 创建目录（如果需要）
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # 写入 CSV
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")



import os, json

def save_models_h5(results_dir,
                   adversarial_model,
                   normal_model,
                   rebuild_kwargs,              # 用來重建 adversarial_model 的參數（看下一段怎麼組）
                   look_back,
                   n_features):
    os.makedirs(results_dir, exist_ok=True)

    # 1) normal_model：整個模型存成 .h5（Functional/Sequential OK）
    normal_h5 = os.path.join(results_dir, "normal_model.h5")
    normal_model.save(normal_h5)
    print(f"[SAVE] normal_model → {normal_h5}")

    # 2) adversarial_model：只存權重（.h5）＋一份 JSON 參數（用來重建）
    adv_weights = os.path.join(results_dir, "adversarial_model.weights.h5")
    adv_config  = os.path.join(results_dir, "adversarial_model_config.json")

    # 補上必要的 input meta（讀回時好 build）
    rebuild_payload = dict(rebuild_kwargs)
    rebuild_payload.setdefault("look_back", look_back)
    rebuild_payload.setdefault("n_features", n_features)

    with open(adv_config, "w") as f:
        json.dump(rebuild_payload, f, indent=2)

    adversarial_model.save_weights(adv_weights)
    print(f"[SAVE] adversarial_model weights → {adv_weights}")
    print(f"[SAVE] adversarial_model config  → {adv_config}")


def load_models_h5(results_dir, compile_models=False):
    # 1) normal_model：直接從 .h5 讀整個模型
    normal_h5 = os.path.join(results_dir, "normal_model.h5")
    normal_model = KM.load_model(normal_h5, compile=compile_models)
    print(f"[LOAD] normal_model ← {normal_h5}")

    # 2) adversarial_model：讀 JSON→重建→load_weights
    adv_weights = os.path.join(results_dir, "adversarial_model.weights.h5")
    adv_config  = os.path.join(results_dir, "adversarial_model_config.json")

    with open(adv_config, "r") as f:
        cfg = json.load(f)

    look_back  = int(cfg.pop("look_back"))
    n_features = int(cfg.pop("n_features"))

    adversarial_model = cs.WrapperTCNWithAT(
        look_back=look_back,
        n_features=n_features,
        **cfg  # 其餘參數：max_num_of_features/epsilon/model_name/... 都在這
    )

    # 重要：先 build 再載權重（input_shape: (None, look_back, n_features)）
    adversarial_model.build(input_shape=(None, look_back, n_features))
    adversarial_model.load_weights(adv_weights)
    print(f"[LOAD] adversarial_model weights ← {adv_weights}")

    # 如需再 compile（做評估/訓練）
    if compile_models:
        adversarial_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=lambda y_true, y_pred: tf.sqrt(
                tf.reduce_mean(tf.square(y_true - y_pred))
            )
        )
        # normal_model 在上面 KM.load_model(..., compile=True) 就會一起編譯

    return adversarial_model, normal_model