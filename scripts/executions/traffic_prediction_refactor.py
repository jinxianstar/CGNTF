# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from contextlib import redirect_stdout
from datetime import datetime
import argparse
import shap
import matplotlib.pyplot as plt

# 添加 src 文件夹到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
# 然后导入模块
import campus_src as cs

# === 可在程式內修改的設定（不想打參數時就改這裡） ===
INPROG_CFG = dict(
    # 原本最下方的網格迭代組合
    datasets=['campus_processed'],
    epsilons=[0.05, 0.2],
    attack_methods=['FGSM'],           # 'FGSM' / 'PGD' / 'Normal'
    adversarial_model_names=['AT'],    # 'AT' / 'mixup'
    at_mixed=False,
    times=10,

    # main 內部的超參數（原本 None，這裡給出預設）
    look_back=24,
    mixup_alpha=0.3,
    batch_size=64,
    step_idx=None,        # None 代表全局擾動；若要定點擾動就給整數索引
    feat_idx=[0],         # 單特徵情境預設攻擊第 0 個特徵；多特徵時可改如 [0,2]
    model_name='TCN',     # 'TCN' / 'CNN-LSTM' / 'CNN-GRU' / 'LSTM' / 'D-TCN'
    train_ratio=0.7,
    validation_ratio=0.15,
    max_features=1,       # 只用一個特徵時為 1
    target_index=0,
    only_one_feature=True,
    evalute_above_50percent=False,
)

# ==================== 函数定义 ====================

def load_data(n_steps, target_index, dataset_name, only_one_feature):
    '''读取并预处理数据。'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', 'data', 'processed', f"{dataset_name}.csv")

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


def build_mixup_model(look_back, n_features, model_name, max_num_of_features, attack_method, epsilon=0.5, alpha=0.3, step_idx=None, feat_idx=None):
    '''
    构建带 FGSM + Mixup 的 TCN 模型。
    '''
    model = cs.WrapperTCNWithFGSMMixup(
        look_back=look_back,
        n_features=n_features,
        epsilon=epsilon,
        alpha=alpha,
        step_idx=step_idx,
        feat_idx=feat_idx,
        model_name=model_name,
        max_num_of_features=max_num_of_features,
        attack_method=attack_method
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=lambda y_true, y_pred: tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))  # RMSE
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


def plot_perturbation(X_orig, X_adv):
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(X_orig[0,:,0], label='Original')
    plt.title('Original Signal')
    plt.subplot(2,1,2)
    plt.plot(X_adv[0,:,0] - X_orig[0,:,0], 'r', label='Perturbation')
    plt.title('Adversarial Perturbation')
    # 注意：此函数未传入 epsilon，仅示例用途
    plt.tight_layout()
    plt.show()


def evaluate_and_attack(
    model_mixup,
    model_normal,
    X_test,
    y_test,
    epsilon,
    step_idx,
    max_num_of_features,
    attack_method="FGSM",
    feat_idx=None
):
    '''生成對抗樣本並返回它們。'''
    if feat_idx is None:
        feat_idx = [0, 1, 2]

    if attack_method == "FGSM":
        X_test_adv_mixup = cs.fgsm_inject_one_pos(
            model_mixup, X_test, y_test, epsilon,
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features=max_num_of_features
        )
        X_test_adv_normal = cs.fgsm_inject_one_pos(
            model_normal, X_test, y_test, epsilon,
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features=max_num_of_features
        )
    elif attack_method == "PGD":
        X_test_adv_mixup = cs.pgd_inject_one_pos(
            model_mixup, X_test, y_test,
            epsilon=epsilon, num_iter=40,
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features=max_num_of_features
        )
        X_test_adv_normal = cs.pgd_inject_one_pos(
            model_normal, X_test, y_test,
            epsilon=epsilon, num_iter=40,
            step_idx=step_idx,
            feat_idx=(feat_idx if feat_idx is not None else None),
            max_num_of_features=max_num_of_features
        )

    diff = X_test_adv_mixup - X_test
    print(diff[:5])
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
    對所有樣本在指定步與特徵加上常數增量 delta。
    """
    X_attacked = X.copy()
    for i in feat_idx:
        X_attacked[:, step_idx, i] += delta
    return X_attacked


def save_predictions(model, X, y_true, filename, start=None, end=None):
    """
    將 y_true 與 y_pred 存成 CSV。
    """
    y_pred = model.predict(X)
    if start is not None or end is not None:
        y_true_slice = y_true[start:end]
        y_pred_slice = y_pred[start:end]
    else:
        y_true_slice = y_true
        y_pred_slice = y_pred
    df = pd.DataFrame({
        'y_true': np.ravel(y_true_slice),
        'y_pred': np.ravel(y_pred_slice),
    })
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")


# ==================== 主程式入口 ====================

def main(
    dt_now,
    epsilon,
    adversarial_model_name,
    attack_method,
    dataset_name,
    at_mixed,
    subfolder,
    # === 可選參數（None 時使用預設） ===
    look_back=None,
    mixup_alpha=None,
    batch_size=None,
    step_idx=None,
    feat_idx=None,
    model_name=None,
    train_ratio=None,
    validation_ratio=None,
    max_features=None,
    target_index=None,
    evalute_above_50percent=None,
    only_one_feature=None
):
    """
        START: PARAMETERS
    """
    adversarial_model_name = adversarial_model_name if adversarial_model_name is not None else "AT"
    dataset_name = dataset_name if dataset_name is not None else "campus_processed"
    attack_method = attack_method if attack_method is not None else "FGSM"
    at_mixed = at_mixed if at_mixed is not None else False

    # 這些原先寫死，現在允許外部傳入（保持原預設）
    look_back = 24 if look_back is None else look_back
    test_epsilon = epsilon if epsilon is not None else 0.2
    mixup_epsilon = epsilon if epsilon is not None else 0.2
    mixup_alpha = 0.3 if mixup_alpha is None else mixup_alpha
    batch_size = 64 if batch_size is None else batch_size

    step_idx = None if step_idx is None else step_idx
    feat_idx = [0] if feat_idx is None else feat_idx
    model_name = "TCN" if model_name is None else model_name
    train_ratio = 0.7 if train_ratio is None else train_ratio
    validation_ratio = 0.15 if validation_ratio is None else validation_ratio
    max_features = 1 if max_features is None else max_features
    target_index = 0 if target_index is None else target_index
    evalute_above_50percent = False if evalute_above_50percent is None else evalute_above_50percent
    only_one_feature = True if only_one_feature is None else only_one_feature

    if only_one_feature:
        max_features = 1
        target_index = 0
        feat_idx = [0]

    """
        END: PARAMETERS
    """

    # ===== 資料處理 =====
    X, y = load_data(look_back, target_index=target_index, dataset_name=dataset_name, only_one_feature=only_one_feature)
    print(X)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(X, y, max_features, train_ratio=train_ratio, validation_ratio=validation_ratio)
    n_features = X_train.shape[2]
    print("current number of features:", n_features)

    # ===== 建立對抗模型 =====
    adversarial_model = None
    if adversarial_model_name == "mixup":
        adversarial_model = build_mixup_model(
            look_back=look_back,
            n_features=n_features,
            epsilon=mixup_epsilon,
            alpha=mixup_alpha,
            step_idx=step_idx,
            feat_idx=feat_idx,
            model_name=model_name,
            attack_method=attack_method,
            max_num_of_features=max_features
        )
    elif adversarial_model_name == "AT":
        steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
        adversarial_model = cs.WrapperTCNWithAT(
            look_back=look_back,
            n_features=n_features,
            max_num_of_features=max_features,
            epsilon=mixup_epsilon,
            model_name=model_name,
            step_idx=step_idx,
            feat_idx=feat_idx,
            alpha=mixup_alpha,
            attack_method=attack_method,
            mixed=at_mixed
        )
        adversarial_model.steps_per_epoch = steps_per_epoch
        adversarial_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=lambda y_true, y_pred: tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        )

    # ===== 訓練 =====
    mixup_history = train_model(adversarial_model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=batch_size)
    normal_model = build_normal_model(look_back, n_features, model_name)
    normal_history = train_model(normal_model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=batch_size)

    # ===== 產生擾動 =====
    if attack_method != "Normal":
        if attack_method == "FGSM":
            # 注意：和你原本一樣的賦值順序（不更動行為）
            _, X_test_adv_mixup = evaluate_and_attack(
                adversarial_model, normal_model, X_test, y_test,
                test_epsilon, step_idx, max_num_of_features=max_features,
                attack_method="FGSM", feat_idx=feat_idx
            )
        if attack_method == "PGD":
            _, X_test_adv_mixup = evaluate_and_attack(
                adversarial_model, normal_model, X_test, y_test,
                test_epsilon, step_idx, max_num_of_features=max_features,
                attack_method="PGD", feat_idx=feat_idx
            )
    elif attack_method == "Normal":  # 固定黑盒式擾動
        X_test_attacked = attack_all_add_delta(X_test, step_idx=step_idx, feat_idx=feat_idx, delta=test_epsilon)
        X_test_adv_mixup = X_test_attacked.copy()

    # ===== 存推論結果（Ensemble 用）=====
    start = 0
    end = len(X_test) - 1
    configs = [
        ("defense_adv", adversarial_model,      X_test_adv_mixup),
        ("defense_clean", adversarial_model,    X_test),
        ("normal_adv",   normal_model,          X_test_adv_mixup),
        ("normal_clean", normal_model,          X_test),
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "..", "data", "results", subfolder)
    os.makedirs(results_dir, exist_ok=True)

    for prefix, model, X_infer in configs:
        fname = os.path.join(results_dir, f"{prefix}.csv")
        save_predictions(model, X_infer, y_test, fname, start=start, end=end)



    """
        繪畫
    """
    cs.plot_predictions(adversarial_model, X_test_adv_mixup, y_test, start=start, end=end, title="Preidcted by Defense model, Input: FGSM inject.")
    cs.plot_predictions(adversarial_model, X_test, y_test, start=start, end=end, title="Predicted Defense model, Non-attack Input")
    cs.plot_predictions(normal_model, X_test_adv_mixup, y_test, start=start, end=end, title="Preidcted by Normal model, Input: FGSM inject.")
    cs.plot_predictions(normal_model, X_test, y_test, start=start, end=end, title="Preidcted by Normal model, Non-attack Input")
    

    # ===== 紀錄評估 =====
    data_path = os.path.join(script_dir, '..', '..', 'outputs', 'logs', f"log{dt_now}.txt")
    with open(data_path, "a") as f:
        with redirect_stdout(f):
            print("\n" + "="*30)
            print(f"新紀錄：{datetime.now()}")
            print(f"dataset_name: {dataset_name}, epsilon: {epsilon}, attack_method: {attack_method}, adversarial_model_name: {adversarial_model_name}")
            print("="*30)
            print("=============================================")
            print('Defense Method')
            print('以下：未擾動')
            report_results(adversarial_model, X_test, y_test, evalute_above_50percent)
            print('以下：擾動')
            report_results(adversarial_model, X_test_adv_mixup, y_test, evalute_above_50percent)
            print("---------------------------------------------")
            print('Normal Method')
            print('以下：未擾動')
            report_results(normal_model, X_test, y_test, evalute_above_50percent)
            print('以下：擾動')
            report_results(normal_model, X_test_adv_mixup, y_test, evalute_above_50percent)
            print("=============================================")


# ========== 命令列參數 ==========
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if v.lower() in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if v.lower() in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adversarial training/attack runner（參數可在程式內或 CLI 設定）")

    # 將所有參數設為「可省略」，省略時以 INPROG_CFG 為準
    parser.add_argument('--datasets', nargs='+', default=None, help='資料集清單')
    parser.add_argument('--epsilons', nargs='+', type=float, default=None, help='epsilon 清單')
    parser.add_argument('--attack_methods', nargs='+', default=None, help='攻擊方法清單（FGSM/PGD/Normal）')
    parser.add_argument('--adversarial_model_names', nargs='+', default=None, help='對抗模型清單（AT/mixup）')
    parser.add_argument('--at_mixed', type=str2bool, default=None, help='AT 是否混合 (mixed)')
    parser.add_argument('--times', type=int, default=None, help='每組參數重複次數')

    # 可選：把 main 裡的硬编码參數也開放出來（None 就用預設或 INPROG_CFG）
    parser.add_argument('--look_back', type=int, default=None)
    parser.add_argument('--mixup_alpha', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--step_idx', type=int, default=None, help='若不設為 None，將定點打擾該步')
    parser.add_argument('--feat_idx', nargs='+', type=int, default=None, help='定點打擾的特徵索引列表')
    parser.add_argument('--model_name', type=str, default=None, help='TCN / CNN-LSTM / CNN-GRU / LSTM / D-TCN')
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--validation_ratio', type=float, default=None)
    parser.add_argument('--max_features', type=int, default=None)
    parser.add_argument('--target_index', type=int, default=None)
    parser.add_argument('--only_one_feature', type=str2bool, default=None)
    parser.add_argument('--evalute_above_50percent', type=str2bool, default=None)

    args = parser.parse_args()

    # --- 合併邏輯：CLI > 程式內設定 ---
    def pick(cli_val, key):
        return cli_val if cli_val is not None else INPROG_CFG[key]

    datasets  = pick(args.datasets, 'datasets')
    epsilons  = pick(args.epsilons, 'epsilons')
    attacks   = pick(args.attack_methods, 'attack_methods')
    adv_names = pick(args.adversarial_model_names, 'adversarial_model_names')
    at_mixed  = pick(args.at_mixed, 'at_mixed')
    times     = pick(args.times, 'times')

    # 可選超參數（None 代表交由 main 使用內建預設）
    kw = dict(
        look_back=pick(args.look_back, 'look_back'),
        mixup_alpha=pick(args.mixup_alpha, 'mixup_alpha'),
        batch_size=pick(args.batch_size, 'batch_size'),
        step_idx=pick(args.step_idx, 'step_idx'),
        feat_idx=pick(args.feat_idx, 'feat_idx'),
        model_name=pick(args.model_name, 'model_name'),
        train_ratio=pick(args.train_ratio, 'train_ratio'),
        validation_ratio=pick(args.validation_ratio, 'validation_ratio'),
        max_features=pick(args.max_features, 'max_features'),
        target_index=pick(args.target_index, 'target_index'),
        only_one_feature=pick(args.only_one_feature, 'only_one_feature'),
        evalute_above_50percent=pick(args.evalute_above_50percent, 'evalute_above_50percent'),
    )

    for dataset in datasets:
        for epsilon in epsilons:
            for attack_method in attacks:
                for adversarial_model_name in adv_names:
                    dt = datetime.now()
                    for i in range(times):
                        saved_folder_name = f"dataset_{dataset}_ep_{epsilon}_attack_{attack_method}_defence_{adversarial_model_name}_{i+1}"
                        print(dataset)
                        main(
                            dt_now=dt,
                            epsilon=epsilon,
                            adversarial_model_name=adversarial_model_name,
                            attack_method=attack_method,
                            dataset_name=dataset,
                            at_mixed=at_mixed,
                            subfolder=saved_folder_name,
                            **kw
                        )
