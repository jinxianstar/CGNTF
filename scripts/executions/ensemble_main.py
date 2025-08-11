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
import ensemble_module as em

from tensorflow.keras.models import load_model

import json
from keras import models as KM


# load capacity model
def ensemble(dataset_name, attack_method, test_epsilon, step_idx):
    """
        START: PARAMETERS
    """
    dataset_name = dataset_name if dataset_name is not None else "campus_processed" # campus_processed, Abilene, CERNET
    attack_method = attack_method if attack_method is not None else "FGSM" # FGSM or Normal
    look_back = 24
    test_epsilon = test_epsilon if test_epsilon is not None else 0.2

    #step_idx = look_back - 1
    step_idx = step_idx if step_idx is not None else None # 全局擾動 #### 注意哦！ TRANS ON POWER SYSTEM 這樣做！

    feat_idx = [0]
    model_name = "TCN"
    train_ratio = 0.7
    validation_ratio = 0.15 #目前修改 暫時修改 之後要改回來喔！！
    max_features = 1
    target_index = 0
    only_one_feature = True
    """"""

    if only_one_feature:
        max_features = 1
        target_index = 0
        feat_idx = [0]
    

    """
        END: PARAMETERS
    """
    # Load Saved Model
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 你要載入的資料夾清單
    model_dirs = [
        "dataset_campus_processed_ep_0.05_attack_FGSM_defence_AT_1",
        "dataset_campus_processed_ep_0.1_attack_FGSM_defence_AT_1",
        "dataset_campus_processed_ep_0.2_attack_FGSM_defence_AT_1",
    ]

    # 存放載入的模型
    adversarial_models = {}
    normal_models = {}

    for model_dir in model_dirs:
        results_dir = os.path.join(script_dir, "..", "..", "models", model_dir)
        
        # normal_model (如果每個資料夾都有的話才載)
        normal_model = KM.load_model(os.path.join(results_dir, "normal_model.keras"), compile=False)
        
        # adversarial_model
        with open(os.path.join(results_dir, "adversarial_model_config.json")) as f:
            cfg = json.load(f)
        look_back  = int(cfg.pop("look_back"))
        n_features = int(cfg.pop("n_features"))

        adversarial_model = cs.WrapperTCNWithAT(look_back=look_back, n_features=n_features, **cfg)
        adversarial_model.build(input_shape=(None, look_back, n_features))
        adversarial_model.load_weights(os.path.join(results_dir, "adversarial_model.weights.h5"))

        adversarial_models[model_dir] = adversarial_model  # 存起來方便之後用
        normal_models[model_dir] = normal_model

    # Create noise data.
    train_ratio = 0.7
    validation_ratio = 0.15 #目前修改 暫時修改 之後要改回來喔！！
    
    X, y = tpmm.load_data(look_back, target_index=target_index, dataset_name=dataset_name, only_one_feature=only_one_feature)
    
    X_train, X_validation, X_test, y_train, y_validation, y_test = tpmm.prepare_datasets(X, y, max_features, train_ratio=train_ratio, validation_ratio=validation_ratio)

    if attack_method != "Normal":
        _, X_test_adv_mixup = tpmm.evaluate_and_attack(adversarial_model, normal_model, X_test, y_test, test_epsilon, attack_method=attack_method, step_idx=step_idx, feat_idx=feat_idx, max_num_of_features=max_features)
    elif attack_method == "Normal": # 固定黑盒式擾動
        X_test_attacked = tpmm.attack_all_add_delta(X_test, step_idx=step_idx, feat_idx=feat_idx, delta=test_epsilon)
        X_test_adv_mixup = X_test_attacked.copy()

    # Predict
    predicted = adversarial_models[model_dirs[0]].predict(X_test).reshape(-1, 1)
    """
    if above_half_range:
        return cs.evaluate_regression_above_half_range(y_test, predicted)
    else:
        return cs.evaluate_regression(y_test, predicted)
    """
    start = 0 
    end = 645

    model_index = 2
    adversarial_model = adversarial_models[model_dirs[model_index]]

    cs.plot_predictions(adversarial_model, X_test_adv_mixup, y_test, start=start, end=end, title="Preidcted by Defense model, Input: FGSM inject.")
    cs.plot_predictions(adversarial_model, X_test, y_test, start=start, end=end, title="Predicted Defense model, Non-attack Input")
    cs.plot_predictions(normal_model, X_test_adv_mixup, y_test, start=start, end=end, title="Preidcted by Normal model, Input: FGSM inject.")
    cs.plot_predictions(normal_model, X_test, y_test, start=start, end=end, title="Preidcted by Normal model, Non-attack Input")
    


    print("\n" + "="*30)
    print(f"新紀錄：{datetime.now()}")
    print(f"dataset_name: {dataset_name}, test epsilon: {test_epsilon}, attack_method: {attack_method}, defense: {model_dirs[model_index]}")#adversarial_model_name: {adversarial_model_name}")
    print("="*30)
    print("=============================================")
    print('Defense Method')
    print('以下：未擾動')
    tpmm.report_results(adversarial_model, X_test, y_test, False)
    print('以下：擾動')
    tpmm.report_results(adversarial_model, X_test_adv_mixup, y_test, False)
    print("---------------------------------------------")
    print('Normal Method')
    print('以下：未擾動')
    tpmm.report_results(normal_model, X_test, y_test, False)
    print('以下：擾動')
    # 注意這邊，如果你想要測試集一樣
    tpmm.report_results(normal_model, X_test_adv_mixup, y_test, False)
    # 注意這邊，測試集不一樣
    #report_results(normal_model, X_test_adv_normal, y_test)
    print("=============================================")
    
    # 想要 ensemble 哪些 model_index（例如 0,1,2）
    # ensemble_indices = [0, 1, 2]
    # ensemble_defense = EnsembleModel([adversarial_models[model_dirs[i]] for i in ensemble_indices])
    # ensemble_normal  = EnsembleModel([normal_models[model_dirs[i]] for i in ensemble_indices])


    # # 之後一律把 ensemble 當成一個模型使用即可：
    # cs.plot_predictions(ensemble_defense, X_test_adv_mixup, y_test, start=start, end=end,
    #                     title="Predicted by Defense ENSEMBLE, Input: FGSM inject.")
    # cs.plot_predictions(ensemble_defense, X_test, y_test, start=start, end=end,
    #                     title="Predicted Defense ENSEMBLE, Non-attack Input")

    # cs.plot_predictions(ensemble_normal, X_test_adv_mixup, y_test, start=start, end=end,
    #                     title="Predicted by Normal ENSEMBLE, Input: FGSM inject.")
    # cs.plot_predictions(ensemble_normal, X_test, y_test, start=start, end=end,
    #                     title="Predicted by Normal ENSEMBLE, Non-attack Input")

    # print("\n" + "="*30)
    # print(f"新紀錄：{datetime.now()}")
    # print(f"dataset_name: {dataset_name}, test epsilon: {test_epsilon}, attack_method: {attack_method}, defense: ENSEMBLE({ensemble_indices})")
    # print("="*30)
    # print("=============================================")
    # print('Defense Method (ENSEMBLE)')
    # print('以下：未擾動')
    # report_results(ensemble_defense, X_test, y_test, False)
    # print('以下：擾動')
    # report_results(ensemble_defense, X_test_adv_mixup, y_test, False)
    # print("---------------------------------------------")
    # print('Normal Method (ENSEMBLE)')
    # print('以下：未擾動')
    # report_results(ensemble_normal, X_test, y_test, False)
    # print('以下：擾動')
    # report_results(ensemble_normal, X_test_adv_mixup, y_test, False)
    # print("=============================================")


    # 先把要 ensemble 的模型抓出來
    ensemble_indices = [0, 1, 2]
    defense_models = [adversarial_models[model_dirs[i]] for i in ensemble_indices]
    normal_models  = [normal_models[model_dirs[i]]      for i in ensemble_indices]

    window=3
    batch_size=128
    # 建立 DMA（建議 window 小一點、加點動量會更穩）
    dma_defense = em.EnsembleDMA(defense_models, window=window, temperature=0.8, momentum=0.3, min_weight=0.02)
    dma_normal  = em.EnsembleDMA(normal_models,  window=window, temperature=0.8, momentum=0.3, min_weight=0.02)

    #====== A) 畫圖 ======
    #攻擊資料
    em.dma_plot_predictions(
        dma_defense,
        X_test_adv_mixup, y_test,
        start=start, end=end,
        title="DMA Defense Model, Input: FGSM inject.",
        batch_size=batch_size, update_every=1
    )

    # 乾淨資料
    em.dma_plot_predictions(
        dma_defense,
        X_test, y_test,
        start=start, end=end,
        title="DMA Defense Model, Non-attack Input",
        batch_size=batch_size, update_every=1
    )

    em.dma_plot_predictions(
        dma_normal,
        X_test_adv_mixup, y_test,
        start=start, end=end,
        title="DMA Normal Model, Input: FGSM inject.",
        batch_size=batch_size, update_every=1
    )

    em.dma_plot_predictions(
        dma_normal,
        X_test, y_test,
        start=start, end=end,
        title="DMA Normal Model, Non-attack Input",
        batch_size=batch_size, update_every=1
    )

    #====== B) 報告 ======
    print("\n" + "="*30)
    print(f"新紀錄：{datetime.now()}")
    print(f"dataset_name: {dataset_name}, test epsilon: {test_epsilon}, attack_method: {attack_method}, defense: DMA({ensemble_indices})")
    print("="*30)
    print("=============================================")
    print('Defense Method (DMA)')
    print('以下：未擾動')
    em.dma_report_results(dma_defense, X_test, y_test, batch_size=batch_size, update_every=1)
    print('以下：擾動')
    em.dma_report_results(dma_defense, X_test_adv_mixup, y_test, batch_size=batch_size, update_every=1)
    print("---------------------------------------------")
    print('Normal Method (DMA)')
    print('以下：未擾動')
    em.dma_report_results(dma_normal, X_test, y_test, batch_size=batch_size, update_every=1)
    print('以下：擾動')
    em.dma_report_results(dma_normal, X_test_adv_mixup, y_test, batch_size=batch_size, update_every=1)
    print("=============================================")

    
    # Ensemble
    return 0;




if __name__ == '__main__':
    ensemble(
        dataset_name="campus_processed",
        attack_method="FGSM",
        test_epsilon=0.10,
        step_idx=None
    )