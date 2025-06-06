# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from contextlib import redirect_stdout
from datetime import datetime

# 添加 src 文件夹到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# 然后导入模块
import campus_src as cs

# ==================== 函数定义 ====================

def load_data(n_steps):
    '''读取并预处理数据。

    假設 df_encoded 是你的 DataFrame，且已經按時間順序排序
    '''
    # 获取当前脚本文件的位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'data', 'processed', 'campus_processed.csv')

    df = pd.read_csv(data_path)
    df = df.set_index('DateTime')

    X, y = cs.create_dataset(df.to_numpy(), n_steps)
    return X, y


def prepare_datasets(X, y, n_steps):
    '''拆分並縮放資料集。'''
    X_train, X_validation, X_test, y_train, y_validation, y_test = cs.split_dataset(
        X, y, train_ratio=0.75, validation_ratio=0.15
    )

    # 數值縮放
    X_train, X_validation, X_test = cs.scaling(X_train, X_validation, X_test, n_steps)

    # 轉成 float32（TensorFlow 更友好）
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_validation = np.array(X_validation, dtype=np.float32)
    y_validation = np.array(y_validation, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


# 忘記 mixup feature 2 了！！ (6/4)

def build_mixup_model(look_back, n_features, epsilon=0.5, alpha=0.3, step_idx=None, feat_idx=None):
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
        feat_idx=feat_idx  # 如果要“定点打扰”，可以设置一个整数
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=lambda y_true, y_pred: tf.sqrt(
            tf.reduce_mean(tf.square(y_true - y_pred))
        )  # RMSE
    )
    return model


def build_normal_model(look_back, n_features):
    '''original models.'''
    return cs.build_model_TCN(look_back, n_features)


# 举几个常见的 α 值特点：
#
# α=1（Beta(1,1)）：均匀分布，λ 在 [0,1] 上等概率——可能落到 0.1、0.9、0.5，都是一样的概率。
#
# α=2 或 3（Beta(2,2) / Beta(3,3)）：比均匀分布更偏向中间，抽到 0.5 左右的几率会更高，但也还会有比较大的概率落到 0.2 或 0.8。
#
# α=5 或 10（Beta(5,5) / Beta(10,10)）：分布就非常集中在 0.5 附近，几乎不会抽出 0.1 或 0.9，大部分 λ 都在 0.4–0.6（甚至 0.45–0.55）之间。


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


def evaluate_and_attack(model_mixup, model_normal, X_test, y_test, epsilon, n_steps):
    '''生成對抗樣本並返回它們。'''
    # 被mixup模型干擾
    X_test_adv_mixup = cs.fgsm_inject_one_pos(model_mixup, X_test, y_test, epsilon, step_idx=n_steps-1, feat_idx=[1, 2])
    # 被普通模型干擾
    X_test_adv_normal = cs.fgsm_inject_one_pos(model_normal, X_test, y_test, epsilon, step_idx=n_steps-1, feat_idx=[1, 2])
    return X_test_adv_mixup, X_test_adv_normal

def report_results(model, X_test, y_test):
    predicted = model.predict(X_test).reshape(-1, 1)
    return cs.evaluate_regression(y_test, predicted)
    

# def report_results(model_mixup, model_normal, X_test, y_test, X_test_adv_mixup, X_test_adv_normal):
#     '''列印評估結果。'''
#     print('Mixup Method')
#     print('未擾動')
#     predicted = model_mixup.predict(X_test).reshape(-1, 1)
#     cs.evaluate_regression(y_test, predicted)

#     print('擾動')
#     predicted = model_mixup.predict(X_test_adv_mixup).reshape(-1, 1)
#     cs.evaluate_regression(y_test, predicted)

#     print('Normal Method')
#     print('未擾動')
#     predicted = model_normal.predict(X_test).reshape(-1, 1)
#     cs.evaluate_regression(y_test, predicted)

#     print('擾動')
#     predicted = model_normal.predict(X_test_adv_normal).reshape(-1, 1)
#     cs.evaluate_regression(y_test, predicted)


# ==================== 主程式入口 ====================

def main():
    # n_steps = 8  # 例如，使用t以及前面5步作為特徵
    # 論文設定成3
    n_steps = 8

    X, y = load_data(n_steps)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(X, y, n_steps)

    look_back = n_steps
    n_features = X_train.shape[2]
    batch_size = 128

    # ===== 构建並训练 Mixup 模型 =====
    mixup_model = build_mixup_model(
        look_back=look_back,
        n_features=n_features,
        epsilon=0.5,
        alpha=0.3,
        step_idx=look_back-1,
        feat_idx=[1, 2]
    )

    mixup_history = train_model(mixup_model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=batch_size)
    #cs.plot_loss(mixup_history)

    # ===== 构建並训练普通模型 =====
    normal_model = build_normal_model(look_back, n_features)
    normal_history = train_model(normal_model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=batch_size)
    #cs.plot_loss(normal_history)

    # ===== 生成對抗樣本並評估 =====
    epsilon = 0.3
    X_test_adv_mixup, X_test_adv_normal = evaluate_and_attack(mixup_model, normal_model, X_test, y_test, epsilon, n_steps)

    #cs.plot_predictions(mixup_model, X_test_adv_mixup, y_test, start=0, end=400, title="Preidcted by Mixup model, FGSM by Mixup model")
    #cs.plot_predictions(mixup_model, X_test, y_test, start=0, end=400, title="Predicted Mixup model, Non-attack Input")
    #cs.plot_predictions(normal_model, X_test_adv_normal, y_test, start=0, end=400, title="Preidcted by Normal model, FGSM by Normal model")
    #cs.plot_predictions(normal_model, X_test, y_test, start=0, end=400, title="Preidcted by Normal model, Non-attack Input")




    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'outputs', 'logs', 'log.txt')

    with open(data_path, "a") as f:
        with redirect_stdout(f):
            print("\n" + "="*30)
            print(f"新紀錄：{datetime.now()}")
            print("="*30)

            print("=============================================")
            print('Mixup Method')
            print('以下：未擾動')
            report_results(mixup_model, X_test, y_test)
            print('以下：擾動')
            report_results(mixup_model, X_test_adv_mixup, y_test)
            print("---------------------------------------------")
            print('Normal Method')
            print('以下：未擾動')
            report_results(normal_model, X_test, y_test)
            print('以下：擾動')
            report_results(normal_model, X_test_adv_mixup, y_test)
            print("=============================================")
    #report_results(mixup_model, normal_model, X_test, y_test, X_test_adv_mixup, X_test_adv_normal)


if __name__ == '__main__':
    for i in range(10):
        main()
