# -*- coding: utf-8 -*-
"""
Drop-in runner that adds caching + DWA (static & dynamic) to your pipeline.

It reuses your existing functions from this file's directory structure:
- load_data, prepare_datasets
- build_normal_model, build_mixup_model
- cs.WrapperTCNWithAT / cs.WrapperTCNWithFGSMMixup
- train_model
- evaluate_and_attack, attack_all_add_delta
- save_predictions, report_results

Folders used (relative to this script):
  ../../models/                         (cached models)
  ../../data/attacks/                   (cached attacked test sets)
  ../../data/results/<subfolder>/       (CSV & NumPy outputs)
  ../../outputs/logs/                   (logs)

Run: python main_dwa_cached.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# -------------------- Bring your src/ into path --------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (SCRIPT_DIR / '..' / '..').resolve()
SRC_DIR = (PROJECT_ROOT / 'src').resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Your project modules
import campus_src as cs
from campus_src import WrapperTCNWithAT  # for custom_objects
try:
    from campus_src import WrapperTCNWithFGSMMixup  # if available
except Exception:
    WrapperTCNWithFGSMMixup = None  # optional

# ---- Register / collect custom TCN layers to make .keras round-trip safe ----
TCN_CUSTOM_OBJECTS = {}
try:
    try:
        from keras.saving import register_keras_serializable  # Keras 3
    except Exception:
        from keras.utils import register_keras_serializable   # Keras 2 fallback
    from tcn.tcn import ResidualBlock
    register_keras_serializable(package="tcn")(ResidualBlock)
    TCN_CUSTOM_OBJECTS['ResidualBlock'] = ResidualBlock
except Exception:
    pass

# If you keep the helpers in the same file you pasted earlier, import them.
# Otherwise, we assume they are defined in the same runtime (this file can also import them).
from traffic_prediction import (
    load_data,
    prepare_datasets,
    build_normal_model,
    build_mixup_model,
    train_model,
    evaluate_and_attack,
    attack_all_add_delta,
    save_predictions,
    report_results,
)


# =================== Config & Paths ===================
@dataclass
class ExpCfg:
    dataset_name: str = 'campus_processed'   # campus_processed, Abilene, CERNET
    adversarial_model_name: str = 'AT'       # 'AT' or 'mixup'
    model_name: str = 'TCN'
    attack_method: str = 'FGSM'              # 'FGSM' | 'PGD' | 'Normal'
    epsilon: float = 0.2
    at_mixed: bool = False

    look_back: int = 24
    batch_size: int = 64
    mixup_alpha: float = 0.3

    step_idx: int | None = None              # None => global perturbation
    feat_idx: List[int] = None               # default [0]

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    max_features: int = 1
    target_index: int = 0
    only_one_feature: bool = True

    # experiments / caching
    seed: int = 0

    def __post_init__(self):
        if self.feat_idx is None:
            self.feat_idx = [0]


class Paths:
    def __init__(self, project_root: Path):
        self.root = project_root
        self.models = (self.root / 'models').resolve()
        self.attacks = (self.root / 'data' / 'attacks').resolve()
        self.results = (self.root / 'data' / 'results').resolve()
        self.logs = (self.root / 'outputs' / 'logs').resolve()
        for p in [self.models, self.attacks, self.results, self.logs]:
            p.mkdir(parents=True, exist_ok=True)

    def model_file(self, cfg: ExpCfg, kind: str) -> Path:
        # kind: 'adversarial' | 'normal'
        tag = (
            f"{cfg.dataset_name}_{kind}_{cfg.adversarial_model_name}_"
            f"{cfg.model_name}_eps{cfg.epsilon}_alpha{cfg.mixup_alpha}_"
            f"mixed{int(cfg.at_mixed)}_seed{cfg.seed}"
        )
        return self.models / f"{tag}.keras"

    def val_pred_file(self, model_path: Path) -> Path:
        return model_path.with_suffix('.val_pred.npy')

    def val_rmse_file(self, model_path: Path) -> Path:
        return model_path.with_suffix('.rmse.json')

    def adv_test_file(self, cfg: ExpCfg, attacked_by: str = 'adv') -> Path:
        # attacked_by: 'adv' (AT/mixup model) or 'normal'
        tag = f"{cfg.dataset_name}_{cfg.attack_method}_eps{cfg.epsilon}_lb{cfg.look_back}_{attacked_by}.npy"
        return self.attacks / tag


# =================== Utils ===================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mape(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((a - b) / (b + eps))) * 100)


def _ensure_model_built(model: tf.keras.Model, look_back: int, n_features: int):
    """Ensure model is built before calling `load_weights`.
    Some subclassed models don't implement `build`; in that case run a dummy forward pass.
    """
    try:
        model.build((None, look_back, n_features))
    except Exception:
        dummy = np.zeros((1, look_back, n_features), dtype=np.float32)
        # predict() also works and is backend-agnostic
        try:
            model(dummy, training=False)
        except Exception:
            model.predict(dummy, batch_size=1)


# =================== Stage 1: Train/Load models ===================

def _build_and_compile_adv_model(cfg: ExpCfg, n_features: int) -> tf.keras.Model:
    if cfg.adversarial_model_name == 'mixup':
        # use your build_mixup_model helper
        model = build_mixup_model(
            look_back=cfg.look_back,
            n_features=n_features,
            model_name=cfg.model_name,
            max_num_of_features=cfg.max_features,
            attack_method=cfg.attack_method,
            epsilon=cfg.epsilon,
            alpha=cfg.mixup_alpha,
            step_idx=cfg.step_idx,
            feat_idx=cfg.feat_idx,
        )
        return model

    # 'AT' branch (Wrapper with adversarial training)
    model = cs.WrapperTCNWithAT(
        look_back=cfg.look_back,
        n_features=n_features,
        max_num_of_features=cfg.max_features,
        epsilon=cfg.epsilon,
        model_name=cfg.model_name,
        step_idx=cfg.step_idx,
        feat_idx=cfg.feat_idx,
        alpha=cfg.mixup_alpha,
        attack_method=cfg.attack_method,
        mixed=cfg.at_mixed,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=lambda y_true, y_pred: tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred))),
    )
    return model


def ensure_adversarial_model(paths: Paths, cfg: ExpCfg, X_train, y_train, X_val, y_val) -> Tuple[tf.keras.Model, Path, float]:
    model_path = paths.model_file(cfg, kind='adversarial')
    n_features = X_train.shape[2]
    steps_per_epoch = int(np.ceil(len(X_train) / cfg.batch_size))

    # Rebuild architecture locally each time; load cached **weights** if available
    model = _build_and_compile_adv_model(cfg, n_features)
    if hasattr(model, 'steps_per_epoch'):
        setattr(model, 'steps_per_epoch', steps_per_epoch)

    # IMPORTANT: build before load_weights
    _ensure_model_built(model, cfg.look_back, n_features)

    if model_path.exists():
        model.load_weights(model_path)
    else:
        train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=cfg.batch_size)
        model.save_weights(model_path)

    # cache validation predictions & rmse
    val_pred_path = paths.val_pred_file(model_path)
    rmse_path = paths.val_rmse_file(model_path)
    if val_pred_path.exists() and rmse_path.exists():
        with open(rmse_path, 'r') as f:
            data = json.load(f)
            val_rmse = float(data['rmse'])
    else:
        val_pred = model.predict(X_val, batch_size=cfg.batch_size)
        np.save(val_pred_path, val_pred)
        val_rmse = _rmse(val_pred.squeeze(), y_val.squeeze())
        with open(rmse_path, 'w') as f:
            json.dump({'rmse': val_rmse}, f)
    return model, model_path, val_rmse


def ensure_normal_model(paths: Paths, cfg: ExpCfg, X_train, y_train, X_val, y_val) -> Tuple[tf.keras.Model, Path, float]:
    model_path = paths.model_file(cfg, kind='normal')
    n_features = X_train.shape[2]

    # Rebuild + load cached **weights** if available
    model = build_normal_model(cfg.look_back, n_features, cfg.model_name)

    # IMPORTANT: build before load_weights
    _ensure_model_built(model, cfg.look_back, n_features)

    if model_path.exists():
        model.load_weights(model_path)
    else:
        train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=cfg.batch_size)
        model.save_weights(model_path)

    val_pred_path = paths.val_pred_file(model_path)
    rmse_path = paths.val_rmse_file(model_path)
    if val_pred_path.exists() and rmse_path.exists():
        with open(rmse_path, 'r') as f:
            data = json.load(f)
            val_rmse = float(data['rmse'])
    else:
        val_pred = model.predict(X_val, batch_size=cfg.batch_size)
        np.save(val_pred_path, val_pred)
        val_rmse = _rmse(val_pred.squeeze(), y_val.squeeze())
        with open(rmse_path, 'w') as f:
            json.dump({'rmse': val_rmse}, f)
    return model, model_path, val_rmse


# =================== Stage 2: Cache attacked test set ===================

def ensure_attacked_testset(paths: Paths, cfg: ExpCfg, adv_model, normal_model, X_test, y_test, attacked_by: str = 'adv') -> np.ndarray:
    adv_path = paths.adv_test_file(cfg, attacked_by=attacked_by)
    if adv_path.exists():
        return np.load(adv_path)

    if cfg.attack_method == 'Normal':
        X_test_adv = attack_all_add_delta(X_test, step_idx=cfg.step_idx, feat_idx=cfg.feat_idx, delta=cfg.epsilon)
    else:
        # your helper returns (X_adv_by_mixup, X_adv_by_normal)
        X_adv_by_adv, X_adv_by_normal = evaluate_and_attack(
            adv_model, normal_model, X_test, y_test, cfg.epsilon,
            step_idx=cfg.step_idx, max_num_of_features=cfg.max_features,
            attack_method=cfg.attack_method, feat_idx=cfg.feat_idx
        )
        X_test_adv = X_adv_by_adv if attacked_by == 'adv' else X_adv_by_normal

    np.save(adv_path, X_test_adv)
    return X_test_adv


# =================== Stage 3: DWA (static + dynamic) ===================

def dwa_from_val_rmse(val_rmses: List[float], eps: float = 1e-8) -> np.ndarray:
    inv = np.array([1.0 / (v + eps) for v in val_rmses], dtype=float)
    s = inv.sum()
    return inv / s if s > 0 else np.ones_like(inv) / len(inv)


def dwa_dynamic_sliding(y_true: np.ndarray, preds: List[np.ndarray], window: int = 24, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Return (E_pred, weights_seq). Weights at time t use errors on (t-window, t]."""
    K = len(preds)
    T = preds[0].shape[0]
    pred_mat = np.vstack([p.squeeze() for p in preds])  # (K, T)
    y = y_true.squeeze()

    E_pred = np.zeros(T, dtype=float)
    Wseq = np.zeros((T, K), dtype=float)

    # first window: uniform weights
    Wseq[:window, :] = 1.0 / K
    E_pred[:window] = (Wseq[0] @ pred_mat[:, :window]).ravel()

    for t in range(window, T):
        y_win = y[t - window:t]
        errs = []
        for k in range(K):
            pk = pred_mat[k, t - window:t]
            errs.append(np.sqrt(np.mean((y_win - pk) ** 2)))
        w = dwa_from_val_rmse(errs, eps=eps)
        Wseq[t, :] = w
        E_pred[t] = float(w @ pred_mat[:, t])

    return E_pred.reshape(-1, 1), Wseq


def save_array_as_csv(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    import pandas as pd
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'y_true': y_true.squeeze(), 'y_pred': y_pred.squeeze()})
    df.to_csv(out_path, index=False)


# =================== Orchestration ===================

def run_once(cfg: ExpCfg, subfolder: str | None = None, attacked_by: str = 'adv'):
    set_seed(cfg.seed)
    paths = Paths(PROJECT_ROOT)

    # ----- Load & split -----
    X, y = load_data(cfg.look_back, target_index=cfg.target_index, dataset_name=cfg.dataset_name, only_one_feature=cfg.only_one_feature)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(
        X, y, cfg.max_features, train_ratio=cfg.train_ratio, validation_ratio=cfg.validation_ratio
    )

    # ----- Models (train-or-load) -----
    adv_model, adv_path, adv_val_rmse = ensure_adversarial_model(paths, cfg, X_train, y_train, X_val, y_val)
    normal_model, norm_path, norm_val_rmse = ensure_normal_model(paths, cfg, X_train, y_train, X_val, y_val)

    # ----- Attacked test set (build-or-load) -----
    X_test_adv = ensure_attacked_testset(paths, cfg, adv_model, normal_model, X_test, y_test, attacked_by=attacked_by)

    # ----- Save base predictions like your original script -----
    results_dir = (paths.results / (subfolder or 'default_run')).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ('defense_adv', adv_model, X_test_adv),
        ('defense_clean', adv_model, X_test),
        ('normal_adv', normal_model, X_test_adv),
        ('normal_clean', normal_model, X_test),
    ]
    for prefix, model, Xsrc in configs:
        fname = results_dir / f"{prefix}.csv"
        save_predictions(model, Xsrc, y_test, str(fname), start=0, end=len(X_test) - 1)

    # ===== DWA over the two models (extendable to more models) =====
    models = [adv_model, normal_model]
    val_rmses = [adv_val_rmse, norm_val_rmse]

    # Static DWA (weights from validation RMSE)
    w_static = dwa_from_val_rmse(val_rmses)
    preds_clean = [m.predict(X_test, batch_size=cfg.batch_size) for m in models]
    preds_adv = [m.predict(X_test_adv, batch_size=cfg.batch_size) for m in models]

    E_clean_static = np.tensordot(w_static, np.stack(preds_clean), axes=1)
    E_adv_static = np.tensordot(w_static, np.stack(preds_adv), axes=1)

    save_array_as_csv(y_test, E_clean_static, results_dir / 'E_clean_static.csv')
    save_array_as_csv(y_test, E_adv_static, results_dir / 'E_adv_static.csv')

    # Dynamic DWA (sliding window)
    W = cfg.look_back  # choose your window
    E_clean_dyn, Wseq_clean = dwa_dynamic_sliding(y_test, preds_clean, window=W)
    E_adv_dyn, Wseq_adv = dwa_dynamic_sliding(y_test, preds_adv, window=W)

    save_array_as_csv(y_test, E_clean_dyn, results_dir / 'E_clean_dynamic.csv')
    save_array_as_csv(y_test, E_adv_dyn, results_dir / 'E_adv_dynamic.csv')
    np.save(results_dir / 'weights_clean_dynamic.npy', Wseq_clean)
    np.save(results_dir / 'weights_adv_dynamic.npy', Wseq_adv)

    # ----- Print/Log metrics -----
    rmse_clean_static = _rmse(E_clean_static.squeeze(), y_test.squeeze())
    mape_clean_static = _mape(E_clean_static.squeeze(), y_test.squeeze())
    rmse_adv_static = _rmse(E_adv_static.squeeze(), y_test.squeeze())
    mape_adv_static = _mape(E_adv_static.squeeze(), y_test.squeeze())

    rmse_clean_dyn = _rmse(E_clean_dyn.squeeze(), y_test.squeeze())
    mape_clean_dyn = _mape(E_clean_dyn.squeeze(), y_test.squeeze())
    rmse_adv_dyn = _rmse(E_adv_dyn.squeeze(), y_test.squeeze())
    mape_adv_dyn = _mape(E_adv_dyn.squeeze(), y_test.squeeze())

    log_path = paths.logs / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_path, 'a', encoding='utf-8') as f:
        print('\n' + '=' * 30, file=f)
        print(f"dataset: {cfg.dataset_name}, eps: {cfg.epsilon}, attack: {cfg.attack_method}", file=f)
        print(f"adv_val_rmse: {adv_val_rmse:.6f}, normal_val_rmse: {norm_val_rmse:.6f}", file=f)
        print('- Static DWA -', file=f)
        print(f"clean  RMSE={rmse_clean_static:.4f}, MAPE={mape_clean_static:.2f}%", file=f)
        print(f"attack RMSE={rmse_adv_static:.4f}, MAPE={mape_adv_static:.2f}%", file=f)
        print('- Dynamic DWA -', file=f)
        print(f"clean  RMSE={rmse_clean_dyn:.4f}, MAPE={mape_clean_dyn:.2f}%", file=f)
        print(f"attack RMSE={rmse_adv_dyn:.4f}, MAPE={mape_adv_dyn:.2f}%", file=f)
        print('=' * 30, file=f)

    print('Done. Results saved to:', results_dir)
    print('Log:', log_path)


# =================== Batch runner (grid) ===================

def run_grid():
    attack_methods = ['FGSM']
    epsilons = [0.05, 0.2]
    datasets = ['campus_processed']
    adversarial_model_names = ['AT']  # or ['AT','mixup'] if you train both
    at_mixed = False
    times = 10

    for dataset in datasets:
        for epsilon in epsilons:
            for attack_method in attack_methods:
                for adv_name in adversarial_model_names:
                    for i in range(times):
                        cfg = ExpCfg(
                            dataset_name=dataset,
                            adversarial_model_name=adv_name,
                            model_name='TCN',
                            attack_method=attack_method,
                            epsilon=epsilon,
                            at_mixed=at_mixed,
                            seed=i,  # vary seeds if you want true repeats
                        )
                        saved_folder_name = f"dataset_{dataset}_ep_{epsilon}_attack_{attack_method}_defence_{adv_name}_{i+1}"
                        run_once(cfg, subfolder=saved_folder_name, attacked_by='adv')


if __name__ == '__main__':
    # Choose one
    # run_grid()

    # or a single quick run
    cfg = ExpCfg(
        dataset_name='campus_processed',
        adversarial_model_name='AT',
        model_name='TCN',
        attack_method='FGSM',
        epsilon=0.2,
        at_mixed=False,
        seed=0,
    )
    subfolder = f"dataset_{cfg.dataset_name}_ep_{cfg.epsilon}_attack_{cfg.attack_method}_defence_{cfg.adversarial_model_name}"
    run_once(cfg, subfolder=subfolder, attacked_by='adv')
