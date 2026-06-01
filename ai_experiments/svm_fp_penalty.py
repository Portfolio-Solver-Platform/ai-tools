"""
SVM portfolio selection on cpsat8_k1 with FP-asymmetric class weights.
Same nested CV as svm.py, but SVC gets a class_weight that biases it toward
predicting cpsat8 (class 0); cpsat_weight is itself Optuna-tuned.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

from pathlib import Path
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn import svm
import optuna
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_cpsat8_k1_data, prepare_labels
from utils.cross_solver_eval import leave_one_year_out_folds

INNER_K    = 5
N_TRIALS   = 100
OUTER_JOBS = 15
INNER_JOBS = 1

DATASETS = [
    ("cpsat8_k1", get_cpsat8_k1_data),
]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_pipeline(C, gamma, cpsat_weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  svm.SVC(
            kernel="rbf", C=C, gamma=gamma,
            class_weight={0: cpsat_weight, 1: 1.0},
        )),
    ])


def _one_inner_fold(X, y_labels, Y_borda, params, tr, te):
    pipe = make_pipeline(**params)
    pipe.fit(X[tr], y_labels[tr])
    pred = pipe.predict(X[te])
    bordas = Y_borda[te][np.arange(len(te)), pred]
    return bordas.mean()


def cv_score(X, y_labels, Y_borda, params, splits):
    fold_means = Parallel(n_jobs=INNER_JOBS, prefer="threads")(
        delayed(_one_inner_fold)(X, y_labels, Y_borda, params, tr, te)
        for tr, te in splits
    )
    return float(np.mean(fold_means))


def run_hpo(X, y_labels, Y_borda, groups, n_splits, n_trials):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y_labels, groups=groups))

    def objective(trial):
        params = {
            "C":            trial.suggest_float("C",            0.1, 100, log=True),
            "gamma":        trial.suggest_float("gamma",        1e-3, 1e1, log=True),
            "cpsat_weight": trial.suggest_float("cpsat_weight", 1.0, 50.0, log=True),
        }
        return cv_score(X, y_labels, Y_borda, params, splits)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _evaluate_one_fold(X, y_labels, Y_borda, years, fold_label, train_idx, test_idx):
    t0 = time.time()
    train_years = years[train_idx]
    best_params, best_inner = run_hpo(
        X[train_idx], y_labels[train_idx], Y_borda[train_idx], train_years,
        n_splits=INNER_K, n_trials=N_TRIALS,
    )

    final = make_pipeline(**best_params)
    final.fit(X[train_idx], y_labels[train_idx])
    pred = final.predict(X[test_idx])

    Y_te = Y_borda[test_idx]
    y_te = y_labels[test_idx]
    n_true_cpsat = int((y_te == 0).sum())
    n_fp_k1 = int(((pred == 1) & (y_te == 0)).sum())
    fp_rate = n_fp_k1 / n_true_cpsat if n_true_cpsat else float("nan")

    return {
        "fold_label":     fold_label,
        "n_test":         len(test_idx),
        "test_borda":     float(Y_te[np.arange(len(test_idx)), pred].sum()),
        "oracle":         float(Y_te.max(axis=1).sum()),
        "cpsat_baseline": float(Y_te[:, 0].sum()),
        "accuracy":       float((pred == y_te).mean()),
        "fp_rate":        float(fp_rate),
        "best_params":    best_params,
        "inner_cv_score": float(best_inner),
        "fit_seconds":    time.time() - t0,
    }


def evaluate_dataset(name, getter):
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    folds = leave_one_year_out_folds(years)
    print(f"  outer folds: {len(folds)} (leave-one-year-out, "
          f"{OUTER_JOBS} processes × {INNER_JOBS} threads = "
          f"{OUTER_JOBS * INNER_JOBS} cores)")

    fold_records = Parallel(n_jobs=OUTER_JOBS)(
        delayed(_evaluate_one_fold)(X, y_labels, Y_borda, years,
                                    fold_label, train_idx, test_idx)
        for fold_label, train_idx, test_idx in folds
    )
    fold_records.sort(key=lambda r: r["fold_label"])
    for r in fold_records:
        bp = r["best_params"]
        print(f"    {r['fold_label']}: borda={r['test_borda']:>6.2f}  "
              f"oracle={r['oracle']:>6.2f}  cpsat={r['cpsat_baseline']:>6.2f}  "
              f"ratio={r['test_borda'] / r['oracle']:.3f}  "
              f"acc={r['accuracy'] * 100:>5.1f}%  "
              f"fp={r['fp_rate'] * 100:>4.1f}%  "
              f"(C={bp['C']:.3g}, gamma={bp['gamma']:.3g}, "
              f"w_cpsat={bp['cpsat_weight']:.2f}, "
              f"{r['fit_seconds']:.0f}s)")

    sum_borda    = sum(r["test_borda"]     for r in fold_records)
    sum_oracle   = sum(r["oracle"]         for r in fold_records)
    sum_cpsat    = sum(r["cpsat_baseline"] for r in fold_records)
    n_total      = sum(r["n_test"]         for r in fold_records)
    acc_weighted = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    fp_mean      = float(np.mean([r["fp_rate"] for r in fold_records]))
    print(f"\n  totals: borda={sum_borda:.2f}  oracle={sum_oracle:.2f}  "
          f"cpsat={sum_cpsat:.2f}  oracle_ratio={sum_borda / sum_oracle:.3f}  "
          f"acc={acc_weighted * 100:.1f}%  fp_rate={fp_mean * 100:.1f}%  "
          f"({n_total} test instances)")

    print("  fitting deliverable model (HPO on full data)...")
    final_params, final_score = run_hpo(
        X, y_labels, Y_borda, years, n_splits=INNER_K, n_trials=N_TRIALS,
    )
    final_pipe = make_pipeline(**final_params)
    final_pipe.fit(X, y_labels)

    Path("models").mkdir(exist_ok=True)
    out_path = Path(f"models/svm_fp_model_{name}.joblib")
    joblib.dump(final_pipe, out_path)
    print(f"  saved {out_path}  (C={final_params['C']:.3g}, "
          f"gamma={final_params['gamma']:.3g}, "
          f"w_cpsat={final_params['cpsat_weight']:.2f}, "
          f"cv_score={final_score:.4f})")

    return fold_records, final_params, final_score


def main():
    for name, getter in DATASETS:
        evaluate_dataset(name, getter)


if __name__ == "__main__":
    main()
