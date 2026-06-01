"""Recency-weighted bagged SVM-MW LOYO experiment.

Standalone (doesn't use the main harness) because the experiment needs
year info in fit/predict and the harness doesn't pass years to
fit_predict callbacks.

Sample weight per training point: margin^wpow * exp(alpha * (year - year_min)).
HPO jointly tunes (C, gamma, wpow, alpha, max_samples).
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
import time
from pathlib import Path

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cross_solver_eval import leave_one_year_out_folds  # noqa: E402
from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402
from ai_experiments.best_search import experiments as E  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

INNER_K = 5
N_ESTIMATORS = 10
N_TRIALS = 60


def _recency_bag_predict(X_tr, y_tr, Yb_tr, years_tr, X_te, params,
                          n_estimators=N_ESTIMATORS, pre="log_std"):
    pre_pipe = Pipeline(E._pre(pre))
    pre_pipe.fit(X_tr)
    Xs_tr = pre_pipe.transform(X_tr)
    Xs_te = pre_pipe.transform(X_te)

    margin = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0])
    rel_year = (years_tr - years_tr.min()).astype(np.float64)
    w_full = (margin ** params["wpow"]) * np.exp(params["alpha"] * rel_year) + 1e-12

    n = len(X_tr); sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos

    probs = np.zeros((len(X_te), 2))
    for seed in range(n_estimators):
        rng = np.random.default_rng(seed)
        sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                              rng.choice(idx_neg, size=n_neg, replace=True)])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w_full[sel])
        probs += base.predict_proba(Xs_te)
    return probs / n_estimators


def _cv_score(X, y, Yb, years, params, splits, pre):
    fold_means = []
    for tr, te in splits:
        probs = _recency_bag_predict(X[tr], y[tr], Yb[tr], years[tr], X[te],
                                      params, pre=pre)
        pred = np.argmax(probs, axis=1)
        fold_means.append(Yb[te][np.arange(len(te)), pred].mean())
    return float(np.mean(fold_means))


def _run_hpo(X, y, Yb, years, n_splits, pre, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=years))

    def objective(trial):
        p = {
            "C":           trial.suggest_float("C",     0.5, 30, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
            "alpha":       trial.suggest_float("alpha", 0.0, 0.5),
        }
        return _cv_score(X, y, Yb, years, p, splits, pre)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def _evaluate_fold(X, y, Yb, years, fold_label, train_idx, test_idx, pre):
    t0 = time.time()
    best_params, best_inner = _run_hpo(
        X[train_idx], y[train_idx], Yb[train_idx], years[train_idx],
        n_splits=INNER_K, pre=pre,
    )
    probs = _recency_bag_predict(X[train_idx], y[train_idx], Yb[train_idx],
                                  years[train_idx], X[test_idx], best_params,
                                  pre=pre)
    pred = np.argmax(probs, axis=1)
    Y_te = Yb[test_idx]
    return {
        "fold_label":     fold_label,
        "n_test":         len(test_idx),
        "test_borda":     float(Y_te[np.arange(len(test_idx)), pred].sum()),
        "oracle":         float(Y_te.max(axis=1).sum()),
        "cpsat":          float(Y_te[:, 0].sum()),
        "accuracy":       float((pred == y[test_idx]).mean()),
        "best_params":    best_params,
        "inner_cv_score": best_inner,
        "fit_seconds":    time.time() - t0,
    }


def main(pre: str = "log_std"):
    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    folds = leave_one_year_out_folds(years)
    print(f"========== RecencyBag-MW/{pre}  (n_trials={N_TRIALS}, "
          f"n_estimators={N_ESTIMATORS}) ==========", flush=True)

    records = Parallel(n_jobs=15)(
        delayed(_evaluate_fold)(X, y_labels, Y_borda, years,
                                 fold_label, tr, te, pre)
        for fold_label, tr, te in folds
    )
    records.sort(key=lambda r: r["fold_label"])

    for r in records:
        bp = r["best_params"]
        print(f"    {r['fold_label']}: borda={r['test_borda']:>6.2f}  "
              f"oracle={r['oracle']:>6.2f}  acc={r['accuracy']*100:>5.1f}%  "
              f"(C={bp['C']:.3g}, g={bp['gamma']:.3g}, alpha={bp['alpha']:.3f}, "
              f"{r['fit_seconds']:.0f}s)")

    sb = sum(r["test_borda"] for r in records)
    so = sum(r["oracle"] for r in records)
    sc = sum(r["cpsat"] for r in records)
    n_total = sum(r["n_test"] for r in records)
    acc = sum(r["accuracy"] * r["n_test"] for r in records) / n_total
    print(f"  totals: borda={sb:.2f}  oracle={so:.2f}  cpsat={sc:.2f}  "
          f"ratio={sb/so:.3f}  acc={acc*100:.1f}%  ({n_total} test instances)")


if __name__ == "__main__":
    pre = sys.argv[1] if len(sys.argv) > 1 else "log_std"
    main(pre)
