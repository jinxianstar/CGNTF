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
import shap
import campus_src as cs
import traffic_prediction_main_module as tpmm

# ==================== 主程式入口 ====================

def main(dt_now, epsilon, adversarial_model_name, attack_method, dataset_name, at_mixed, subfolder):
    """
        START: PARAMETERS
    """
    adversarial_model_name = adversarial_model_name if adversarial_model_name is not None else "AT"
    dataset_name = dataset_name if dataset_name is not None else "campus_processed" # campus_processed, Abilene, CERNET
    attack_method = attack_method if attack_method is not None else "FGSM" # FGSM or Normal
    at_mixed = at_mixed if at_mixed is not None else False
    look_back = 24
    test_epsilon = epsilon if epsilon is not None else 0.2
    mixup_epsilon = epsilon if epsilon is not None else 0.2
    mixup_alpha = 0.3 # B
    batch_size = 64

    #step_idx = look_back - 1
    step_idx = None # 全局擾動 #### 注意哦！ TRANS ON POWER SYSTEM 這樣做！

    feat_idx = [0]
    model_name = "TCN"
    train_ratio = 0.7
    validation_ratio = 0.15 #目前修改 暫時修改 之後要改回來喔！！
    max_features = 1
    target_index = 0
    evalute_above_50percent=False

    only_one_feature = True
    """"""

    if only_one_feature:
        max_features = 1
        target_index = 0
        feat_idx = [0]

    """
        END: PARAMETERS
    """
    """
        START:建立模型、與訓練
    """

    X, y = tpmm.load_data(look_back, target_index=target_index, dataset_name=dataset_name, only_one_feature=only_one_feature)
    
    print(X)
    X_train, X_validation, X_test, y_train, y_validation, y_test = tpmm.prepare_datasets(X, y, max_features, train_ratio=train_ratio, validation_ratio=validation_ratio)


    n_features = X_train.shape[2]
    print("current number of features:", n_features)

    adversarial_model = None
    if adversarial_model_name == "mixup":
        adversarial_model = tpmm.build_mixup_model(
            look_back=look_back,
            n_features=n_features,
            epsilon=mixup_epsilon,
            alpha=mixup_alpha,
            step_idx=step_idx,
            feat_idx=feat_idx,
            model_name=model_name,
            attack_method = attack_method,
            max_num_of_features=max_features
        )
    elif adversarial_model_name == "AT":
        steps_per_epoch = int(np.ceil(len(X_train) / batch_size))   # ★ 1. 計算

        adversarial_model = cs.WrapperTCNWithAT(
            look_back=look_back,
            n_features=n_features,
            max_num_of_features=max_features,
            epsilon=mixup_epsilon,
            model_name=model_name,
            step_idx=step_idx,
            feat_idx=feat_idx,
            alpha= mixup_alpha,
            attack_method = attack_method,
            mixed = at_mixed
        )
        adversarial_model.steps_per_epoch = steps_per_epoch                 # ★ 2. 手動塞
        adversarial_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=lambda y_true, y_pred: tf.sqrt(
                tf.reduce_mean(tf.square(y_true - y_pred))
            )
        )
    
    mixup_history = tpmm.train_model(adversarial_model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=batch_size)
    #cs.plot_loss(mixup_history)
    
    # ===== 构建並训练普通模型 =====
    normal_model = tpmm.build_normal_model(look_back, n_features, model_name) #需要改回 model_name
    normal_history = tpmm.train_model(normal_model, X_train, y_train, X_validation, y_validation, epochs=100, batch_size=batch_size)
    #cs.plot_loss(normal_history)

    """
        END: 建立模型、與訓練
    """




    """
        START: 擾動
    """

    # cs.explain_with_kernel(normal_model, X_train, X_test[:30])

    # ===== 生成對抗樣本並評估 =====
    #epsilon = 0.2
    # FGSM + Mixup 擾動
    if attack_method != "Normal":
        if attack_method == "FGSM":
            _, X_test_adv_mixup = tpmm.evaluate_and_attack(adversarial_model, normal_model, X_test, y_test, test_epsilon, attack_method="FGSM", step_idx=step_idx, feat_idx=feat_idx, max_num_of_features=max_features)
        if attack_method == "PGD":
            _, X_test_adv_mixup = tpmm.evaluate_and_attack(adversarial_model, normal_model, X_test, y_test, test_epsilon, attack_method="PGD", step_idx=step_idx, feat_idx=feat_idx, max_num_of_features=max_features)
    elif attack_method == "Normal": # 固定黑盒式擾動
        X_test_attacked = tpmm.attack_all_add_delta(X_test, step_idx=step_idx, feat_idx=feat_idx, delta=test_epsilon)
        X_test_adv_mixup = X_test_attacked.copy()

    # plot_perturbation_at_single_step(X_test, X_test_adv_mixup, target_step=target_index, sample_idx=step_idx)


    """
        END: 擾動
    """

    
    """
        存預測值、真實值，用於 Ensemble.
    """
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
    os.makedirs(results_dir, exist_ok=True)    # 不存在就自動建立

    for prefix, model, X in configs:
        fname = os.path.join(
            results_dir,                       # 改用新的資料夾
            f"{prefix}.csv"
        )
        tpmm.save_predictions(model, X, y_test, fname, start=start, end=end)
    

    """
        繪畫
    """
    # cs.plot_predictions(adversarial_model, X_test_adv_mixup, y_test, start=start, end=end, title="Preidcted by Defense model, Input: FGSM inject.")
    # cs.plot_predictions(adversarial_model, X_test, y_test, start=start, end=end, title="Predicted Defense model, Non-attack Input")
    # cs.plot_predictions(normal_model, X_test_adv_mixup, y_test, start=start, end=end, title="Preidcted by Normal model, Input: FGSM inject.")
    # cs.plot_predictions(normal_model, X_test, y_test, start=start, end=end, title="Preidcted by Normal model, Non-attack Input")
    
    """
        SAVE MODELS (Defense, Normal)
    """
    results_dir = os.path.join(script_dir, "..", "..", "models", subfolder)
    os.makedirs(results_dir, exist_ok=True)    # 不存在就自動建立
    # normal_model：整個模型用 .keras
    normal_model.save(os.path.join(results_dir, "normal_model.keras"), include_optimizer=False)

    # adversarial_model：只存權重 + 一份重建用 config
    adversarial_model.save_weights(os.path.join(results_dir, "adversarial_model.weights.h5"))

    import json
    rebuild_kwargs = dict(
        max_num_of_features=max_features,
        epsilon=mixup_epsilon,
        model_name=model_name,
        step_idx=step_idx,
        feat_idx=feat_idx,
        alpha=mixup_alpha,
        attack_method=attack_method,
        mixed=at_mixed,
        look_back=look_back,      # 供重建
        n_features=n_features,    # 供重建
    )
    with open(os.path.join(results_dir, "adversarial_model_config.json"), "w") as f:
        json.dump(rebuild_kwargs, f, indent=2)



    adversarial_model.save(f"{results_dir}/adversarial_model.keras")
    normal_model.save(f"{results_dir}/normal_model.keras")
    """
        LOG 準確率存檔
    """
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
            tpmm.report_results(adversarial_model, X_test, y_test, evalute_above_50percent)
            print('以下：擾動')
            tpmm.report_results(adversarial_model, X_test_adv_mixup, y_test, evalute_above_50percent)
            print("---------------------------------------------")
            print('Normal Method')
            print('以下：未擾動')
            tpmm.report_results(normal_model, X_test, y_test, evalute_above_50percent)
            print('以下：擾動')
            # 注意這邊，如果你想要測試集一樣
            tpmm.report_results(normal_model, X_test_adv_mixup, y_test, evalute_above_50percent)
            # 注意這邊，測試集不一樣
            #report_results(normal_model, X_test_adv_normal, y_test)
            print("=============================================")


if __name__ == '__main__':
    # ensemble(
    #     dataset_name="campus_processed",
    #     attack_method="FGSM",
    #     test_epsilon=0.10,
    #     step_idx=None
    # )

    
    attack_methods = ["FGSM"] #, "FGSM", "Normal"]
    epsilons = [0.05, 0.1, 0.2] #[0.06, 0.09, 0.15, 0.2]

    datasets = ["campus_processed"] #["campus_processed"]
    adversarial_model_names = ["AT"] #, "mixup"]
    #adversarial_model_names = ["AT"]
    at_mixed = False

    times = 1

    for dataset in datasets:
        for epsilon in epsilons:
            for attack_method in attack_methods:
                for adversarial_model_name in adversarial_model_names:
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
                            subfolder=saved_folder_name
                        )