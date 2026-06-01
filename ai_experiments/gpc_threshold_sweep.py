"""
GPC + threshold fallback (predict cpsat8 when max(proba) < threshold),
threshold picked from inner-OOF Borda. LOYO outer eval, cpsat8_k1 only.
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
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.shared_data import get_cpsat8_k1_data, prepare_labels
from utils.cross_solver_eval import year_kfold_folds

INNER_K = 5
N_RESTARTS = 3
N_JOBS = 10
THRESHOLDS = np.arange(0.0, 1.001, 0.01)

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


def apply_threshold(proba, threshold):
    pred = np.argmax(proba, axis=1)
    pred[proba.max(axis=1) < threshold] = 0
    return pred


def pick_threshold(proba, Y_borda):
    best_t, best_borda = 0.0, -np.inf
    rows = np.arange(len(proba))
    for t in THRESHOLDS:
        pred = apply_threshold(proba, t)
        borda = Y_borda[rows, pred].sum()
        if borda > best_borda:
            best_t, best_borda = float(t), float(borda)
    return best_t, best_borda


def _evaluate_fold(X, y_labels, Y_borda, years, fold_label, train_idx, test_idx):
    t0 = time.time()
    train_years = years[train_idx]

    gkf = GroupKFold(n_splits=INNER_K)
    oof_proba = cross_val_predict(
        make_pipeline(), X[train_idx], y_labels[train_idx],
        method="predict_proba", cv=gkf, groups=train_years,
    )

    best_t, best_oof_borda = pick_threshold(oof_proba, Y_borda[train_idx])

    pipe = make_pipeline()
    pipe.fit(X[train_idx], y_labels[train_idx])
    test_proba = pipe.predict_proba(X[test_idx])
    pred = apply_threshold(test_proba, best_t)

    Y_te = Y_borda[test_idx]
    test_borda     = Y_te[np.arange(len(test_idx)), pred].sum()
    oracle         = Y_te.max(axis=1).sum()
    cpsat_baseline = Y_te[:, 0].sum()
    gap = oracle - cpsat_baseline
    headroom = (test_borda - cpsat_baseline) / gap if gap else float("nan")
    accuracy = (pred == y_labels[test_idx]).mean()

    return {
        "fold_label":       fold_label,
        "n_test":           len(test_idx),
        "best_threshold":   best_t,
        "oof_borda":        best_oof_borda,
        "test_borda":       float(test_borda),
        "oracle":           float(oracle),
        "cpsat_baseline":   float(cpsat_baseline),
        "headroom_capture": float(headroom),
        "accuracy":         float(accuracy),
        "fit_seconds":      time.time() - t0,
    }


def evaluate_dataset(name, getter):
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    folds = year_kfold_folds(years, n_splits=5)
    print(f"  outer folds: {len(folds)} (5-fold GroupKFold by year, parallel n_jobs={N_JOBS})")

    fold_records = Parallel(n_jobs=N_JOBS)(
        delayed(_evaluate_fold)(X, y_labels, Y_borda, years,
                                fold_label, train_idx, test_idx)
        for fold_label, train_idx, test_idx in folds
    )
    fold_records.sort(key=lambda r: r["fold_label"])
    for r in fold_records:
        print(f"    {r['fold_label']}: borda={r['test_borda']:>6.2f}  "
              f"oracle={r['oracle']:>6.2f}  cpsat={r['cpsat_baseline']:>6.2f}  "
              f"t={r['best_threshold']:.2f}  headroom={r['headroom_capture']*100:>5.1f}%  "
              f"acc={r['accuracy']*100:>5.1f}%  ({r['fit_seconds']:.0f}s)")

    sum_borda  = sum(r["test_borda"]     for r in fold_records)
    sum_oracle = sum(r["oracle"]         for r in fold_records)
    sum_cpsat  = sum(r["cpsat_baseline"] for r in fold_records)
    gap_total  = sum_oracle - sum_cpsat
    headroom_total = (sum_borda - sum_cpsat) / gap_total if gap_total else float("nan")
    n_total    = sum(r["n_test"] for r in fold_records)
    acc_weighted = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    print(f"\n  totals: borda={sum_borda:.2f}  oracle={sum_oracle:.2f}  "
          f"cpsat={sum_cpsat:.2f}  oracle_ratio={sum_borda/sum_oracle:.3f}  "
          f"headroom={headroom_total*100:.1f}%  acc={acc_weighted*100:.1f}%  "
          f"({n_total} test instances)")
    thr_mean = np.mean([r["best_threshold"] for r in fold_records])
    thr_min  = min(r["best_threshold"] for r in fold_records)
    thr_max  = max(r["best_threshold"] for r in fold_records)
    print(f"  thresholds per fold: mean={thr_mean:.2f}  range=[{thr_min:.2f}, {thr_max:.2f}]")

    print("  fitting deliverable model + locking threshold on all 15 years...")
    gkf = GroupKFold(n_splits=INNER_K)
    full_oof_proba = cross_val_predict(
        make_pipeline(), X, y_labels,
        method="predict_proba", cv=gkf, groups=years,
    )
    final_t, _ = pick_threshold(full_oof_proba, Y_borda)
    final_pipe = make_pipeline()
    final_pipe.fit(X, y_labels)

    Path("models").mkdir(exist_ok=True)
    out_path = Path(f"models/gpc_threshold_{name}.joblib")
    joblib.dump({"model": final_pipe, "threshold": final_t}, out_path)
    print(f"  saved {out_path}  (locked threshold = {final_t:.2f})")
    return fold_records


def main():
    for name, getter in DATASETS:
        evaluate_dataset(name, getter)


if __name__ == "__main__":
    main()
