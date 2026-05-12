"""Quick diagnostic: train GPC on train, sweep ALL thresholds on test, see best score.
Intentionally leaky (threshold cherry-picked on test) — only to verify the mechanism has signal.
Mirrors xgboost_threshold_sweep.py but uses Gaussian Process Classifier with ARD-RBF kernel
(per-feature length scales, optimized internally by GPC).
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels
from utils.cross_solver_eval import shared_test_borda, map_to_global

# Canonical group-by-problem split
X_canon, Y_canon, meta_canon = get_cpsat8_k1_ek1_data()
y_labels_canon, Y_borda_canon = prepare_labels(Y_canon)

groups_canon = meta_canon["problem"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X_canon, y_labels_canon, groups=groups_canon))
Y_test_borda_3solver = Y_borda_canon[test_idx]

baselines = np.sum(Y_test_borda_3solver, axis=0)
oracle = np.sum(np.max(Y_test_borda_3solver, axis=1))
print(f"shared test baselines [cpsat8, k1, ek1]: {baselines}")
print(f"shared test oracle: {oracle:.2f}")

datasets = [
    ('cpsat8_k1', get_cpsat8_k1_data),
]

thresholds = np.arange(0.0, 1.001, 0.01)

for name, getter in datasets:
    print(f"\n========== dataset: {name} ==========")
    X, Y, _ = getter()
    y_labels, _ = prepare_labels(Y)
    X_train = X[train_idx]
    y_train = y_labels[train_idx]
    X_test = X[test_idx]

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
    gpc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=2, random_state=42, n_jobs=28))
    ])
    gpc.fit(X_train, y_train)
    proba = gpc.predict_proba(X_test)
    pred_argmax = np.argmax(proba, axis=1)
    pred_max_proba = proba.max(axis=1)
    rows = np.arange(len(Y_test_borda_3solver))

    best_t, best_borda = None, -np.inf
    for t in thresholds:
        pred = pred_argmax.copy()
        pred[pred_max_proba < t] = 0  # cpsat8 fallback
        pred_global = map_to_global(pred, name)
        borda = np.sum(Y_test_borda_3solver[rows, pred_global])
        if borda > best_borda:
            best_borda = borda
            best_t = t

    borda_no_threshold = np.sum(Y_test_borda_3solver[rows, map_to_global(pred_argmax, name)])
    borda_always_cpsat = baselines[0]

    print(f"  no threshold (raw argmax):    test_borda={borda_no_threshold:.2f}")
    print(f"  always cpsat8 baseline:       test_borda={borda_always_cpsat:.2f}")
    print(f"  best threshold (cherry-picked on test): t={best_t:.2f}  test_borda={best_borda:.2f}")
    print(f"  -> headroom captured: {(best_borda - borda_always_cpsat) / (oracle - borda_always_cpsat) * 100:.1f}% of (oracle - cpsat baseline)")
