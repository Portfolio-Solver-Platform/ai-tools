"""
Leave-one-year-out evaluation for GPC portfolio selection (cpsat8_k1 only).

GPC tunes its kernel hyperparameters internally during .fit(), so there is no
inner-CV HPO loop — we just fix n_restarts_optimizer and let GPC do its thing.
Threshold-aware decision rules live in gpc_threshold_sweep.py.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.shared_data import get_cpsat8_k1_data, prepare_labels
from utils.cross_solver_eval import year_kfold_folds

N_RESTARTS = 3
N_JOBS = 10

DATASETS = [
    ("cpsat8_k1", get_cpsat8_k1_data),
]


def make_pipeline():
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=N_RESTARTS,
            random_state=42,
        )),
    ])


def _evaluate_fold(X, y_labels, Y_borda, fold_label, train_idx, test_idx):
    t0 = time.time()
    pipe = make_pipeline()
    pipe.fit(X[train_idx], y_labels[train_idx])
    pred = pipe.predict(X[test_idx])

    Y_te = Y_borda[test_idx]
    test_borda     = Y_te[np.arange(len(test_idx)), pred].sum()
    oracle         = Y_te.max(axis=1).sum()
    cpsat_baseline = Y_te[:, 0].sum()
    accuracy       = (pred == y_labels[test_idx]).mean()

    return {
        "fold_label":     fold_label,
        "n_test":         len(test_idx),
        "test_borda":     float(test_borda),
        "oracle":         float(oracle),
        "cpsat_baseline": float(cpsat_baseline),
        "accuracy":       float(accuracy),
        "fit_seconds":    time.time() - t0,
    }


def evaluate_dataset(name, getter):
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    folds = year_kfold_folds(years, n_splits=5)
    print(f"  outer folds: {len(folds)} (5-fold GroupKFold by year, parallel n_jobs={N_JOBS})")

    fold_records = Parallel(n_jobs=N_JOBS)(
        delayed(_evaluate_fold)(X, y_labels, Y_borda, fold_label, train_idx, test_idx)
        for fold_label, train_idx, test_idx in folds
    )
    fold_records.sort(key=lambda r: r["fold_label"])
    for r in fold_records:
        print(f"    {r['fold_label']}: borda={r['test_borda']:>6.2f}  "
              f"oracle={r['oracle']:>6.2f}  cpsat={r['cpsat_baseline']:>6.2f}  "
              f"ratio={r['test_borda'] / r['oracle'] if r['oracle'] else float('nan'):.3f}  "
              f"acc={r['accuracy'] * 100:>5.1f}%  ({r['fit_seconds']:.0f}s)")

    sum_borda  = sum(r["test_borda"]     for r in fold_records)
    sum_oracle = sum(r["oracle"]         for r in fold_records)
    sum_cpsat  = sum(r["cpsat_baseline"] for r in fold_records)
    n_total    = sum(r["n_test"]         for r in fold_records)
    acc_weighted = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    print(f"\n  totals: borda={sum_borda:.2f}  oracle={sum_oracle:.2f}  "
          f"cpsat={sum_cpsat:.2f}  oracle_ratio={sum_borda / sum_oracle:.3f}  "
          f"acc={acc_weighted * 100:.1f}%  ({n_total} test instances)")

    print("  fitting deliverable model on all 15 years...")
    final_pipe = make_pipeline()
    final_pipe.fit(X, y_labels)

    Path("models").mkdir(exist_ok=True)
    out_path = Path(f"models/gpc_model_{name}.joblib")
    joblib.dump(final_pipe, out_path)
    print(f"  saved {out_path}")
    return fold_records


def main():
    for name, getter in DATASETS:
        evaluate_dataset(name, getter)


if __name__ == "__main__":
    main()
