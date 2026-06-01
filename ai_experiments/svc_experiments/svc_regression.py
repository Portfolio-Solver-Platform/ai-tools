"""
SVR regression variant of svm.py: regress the Borda margin (k1 - cpsat8) and
take the class from sign(predicted margin). Inner-CV scoring still uses the
Borda matrix to stay comparable with the classification variant.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import get_cpsat8_k1_data
from utils.cross_solver_eval import leave_one_year_out_folds

INNER_K    = 5
N_TRIALS   = 100
OUTER_JOBS = 15
INNER_JOBS = 1

DATASETS = [
    ("cpsat8_k1", get_cpsat8_k1_data),
]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_pipeline(C, gamma, epsilon):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  svm.SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)),
    ])


def _one_inner_fold(X, margin, Y_borda, params, tr, te):
    pipe = make_pipeline(**params)
    pipe.fit(X[tr], margin[tr])
    pred_margin = pipe.predict(X[te])
    pred_class  = (pred_margin > 0).astype(int)
    bordas = Y_borda[te][np.arange(len(te)), pred_class]
    return bordas.mean()


def cv_score(X, margin, Y_borda, params, splits):
    fold_means = Parallel(n_jobs=INNER_JOBS, prefer="threads")(
        delayed(_one_inner_fold)(X, margin, Y_borda, params, tr, te)
        for tr, te in splits
    )
    return float(np.mean(fold_means))


def run_hpo(X, margin, Y_borda, groups, n_splits, n_trials):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, margin, groups=groups))

    def objective(trial):
        params = {
            "C":       trial.suggest_float("C",       0.1, 100, log=True),
            "gamma":   trial.suggest_float("gamma",   1e-3, 1e1, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 0.5, log=True),
        }
        return cv_score(X, margin, Y_borda, params, splits)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _evaluate_one_fold(X, margin, Y_borda, years, fold_label, train_idx, test_idx):
    t0 = time.time()
    train_years = years[train_idx]
    best_params, best_inner = run_hpo(
        X[train_idx], margin[train_idx], Y_borda[train_idx], train_years,
        n_splits=INNER_K, n_trials=N_TRIALS,
    )

    final = make_pipeline(**best_params)
    final.fit(X[train_idx], margin[train_idx])
    pred_margin = final.predict(X[test_idx])
    pred_class  = (pred_margin > 0).astype(int)

    Y_te = Y_borda[test_idx]
    y_labels_te = (margin[test_idx] > 0).astype(int)
    return {
        "fold_label":     fold_label,
        "n_test":         len(test_idx),
        "test_borda":     float(Y_te[np.arange(len(test_idx)), pred_class].sum()),
        "oracle":         float(Y_te.max(axis=1).sum()),
        "cpsat_baseline": float(Y_te[:, 0].sum()),
        "accuracy":       float((pred_class == y_labels_te).mean()),
        "best_params":    best_params,
        "inner_cv_score": float(best_inner),
        "fit_seconds":    time.time() - t0,
    }


def evaluate_dataset(name, getter):
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    Y_borda = Y
    margin  = Y_borda[:, 1] - Y_borda[:, 0]
    years   = meta["year"]

    folds = leave_one_year_out_folds(years)
    print(f"  outer folds: {len(folds)} (LOYO, "
          f"{OUTER_JOBS} processes × {INNER_JOBS} threads = "
          f"{OUTER_JOBS * INNER_JOBS} cores)")

    fold_records = Parallel(n_jobs=OUTER_JOBS)(
        delayed(_evaluate_one_fold)(X, margin, Y_borda, years,
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
              f"(C={bp['C']:.3g}, gamma={bp['gamma']:.3g}, "
              f"epsilon={bp['epsilon']:.3g}, {r['fit_seconds']:.0f}s)")

    sum_borda  = sum(r["test_borda"]     for r in fold_records)
    sum_oracle = sum(r["oracle"]         for r in fold_records)
    sum_cpsat  = sum(r["cpsat_baseline"] for r in fold_records)
    n_total    = sum(r["n_test"]         for r in fold_records)
    acc_weighted = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    print(f"\n  totals: borda={sum_borda:.2f}  oracle={sum_oracle:.2f}  "
          f"cpsat={sum_cpsat:.2f}  oracle_ratio={sum_borda / sum_oracle:.3f}  "
          f"acc={acc_weighted * 100:.1f}%  ({n_total} test instances)")

    print("  fitting deliverable model (HPO on full data)...")
    final_params, final_score = run_hpo(
        X, margin, Y_borda, years, n_splits=INNER_K, n_trials=N_TRIALS,
    )
    final_pipe = make_pipeline(**final_params)
    final_pipe.fit(X, margin)

    Path("models").mkdir(exist_ok=True)
    out_path = Path(f"models/svc_regression_{name}.joblib")
    joblib.dump(final_pipe, out_path)
    print(f"  saved {out_path}  (params: C={final_params['C']:.3g}, "
          f"gamma={final_params['gamma']:.3g}, epsilon={final_params['epsilon']:.3g},  "
          f"cv_score={final_score:.4f})")

    return fold_records, final_params, final_score


def main():
    for name, getter in DATASETS:
        evaluate_dataset(name, getter)


if __name__ == "__main__":
    main()
