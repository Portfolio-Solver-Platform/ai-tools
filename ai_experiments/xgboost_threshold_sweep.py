"""XGBoost threshold pipeline: fit on train, sweep threshold on val proba,
refit on train+val with that threshold locked, evaluate on test. Local Borda."""
from pathlib import Path
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels
from utils.cross_solver_eval import make_train_val_test_indices

_, _, meta_canon = get_cpsat8_k1_ek1_data()
groups_canon = meta_canon["problem"]
train_idx, val_idx, test_idx = make_train_val_test_indices(groups_canon)
trainval_idx = np.concatenate([train_idx, val_idx])

print(f"split sizes: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
print(f"problems:    train={len(np.unique(groups_canon[train_idx]))}  val={len(np.unique(groups_canon[val_idx]))}  test={len(np.unique(groups_canon[test_idx]))}")

datasets = [
    ('cpsat8_k1_ek1', get_cpsat8_k1_ek1_data),
    ('cpsat8_k1',     get_cpsat8_k1_data),
    ('cpsat8_ek1',    get_cpsat8_ek1_data),
]

thresholds = np.arange(0.0, 1.001, 0.01)

for name, getter in datasets:
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    y_labels, Y_borda = prepare_labels(Y)
    Y_val_borda = Y_borda[val_idx]
    Y_test_borda_local = Y_borda[test_idx]
    val_rows = np.arange(len(val_idx))
    test_rows = np.arange(len(test_idx))

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        tree_method='hist',
        n_jobs=28,
        random_state=42,
    )

    model.fit(X[train_idx], y_labels[train_idx])
    val_proba = model.predict_proba(X[val_idx])
    val_argmax = np.argmax(val_proba, axis=1)
    val_max_proba = val_proba.max(axis=1)

    best_t, best_val_borda = None, -np.inf
    print(f"  threshold sweep on validation set:")
    for t in thresholds:
        pred = val_argmax.copy()
        pred[val_max_proba < t] = 0
        borda = np.sum(Y_val_borda[val_rows, pred])
        print(f"    t={t:.2f}  val_borda={borda:.2f}")
        if borda > best_val_borda:
            best_val_borda = borda
            best_t = t

    val_cpsat = np.sum(Y_val_borda[:, 0])
    print(f"  best val threshold: t={best_t:.2f}")
    print(f"  best val Borda at that threshold:  {best_val_borda:.2f}")
    print(f"  val always-cpsat baseline:         {val_cpsat:.2f}")

    model.fit(X[trainval_idx], y_labels[trainval_idx])
    test_proba = model.predict_proba(X[test_idx])
    test_pred = np.argmax(test_proba, axis=1)
    test_pred[test_proba.max(axis=1) < best_t] = 0
    test_borda = np.sum(Y_test_borda_local[test_rows, test_pred])

    test_baselines_local = np.sum(Y_test_borda_local, axis=0)
    test_oracle_local = np.sum(np.max(Y_test_borda_local, axis=1))
    borda_always_cpsat = test_baselines_local[0]
    y_test_local = y_labels[test_idx]
    test_accuracy = (test_pred == y_test_local).mean()
    test_majority = max((y_test_local == c).mean() for c in range(Y.shape[1]))

    print(f"  test baselines (local always solver i): {test_baselines_local}")
    print(f"  test oracle (local best per row):       {test_oracle_local:.2f}")
    print(f"  test Borda with locked threshold:       {test_borda:.2f}")
    print(f"  test accuracy: {test_accuracy*100:.1f}%  (test majority baseline: {test_majority*100:.1f}%)")
    print(f"  -> headroom captured on test: {(test_borda - borda_always_cpsat) / (test_oracle_local - borda_always_cpsat) * 100:.1f}% of (oracle - cpsat baseline)")
