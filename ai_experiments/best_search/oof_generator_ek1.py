#!/usr/bin/env python3
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env, "1")

HERE = Path(__file__).resolve().parent
ROOT_AI = HERE.parents[1]
sys.path.insert(0, str(ROOT_AI))

from utils.cross_solver_eval import leave_one_year_out_folds
from utils.shared_data import get_cpsat8_ek1_data, prepare_labels
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from joblib import Parallel, delayed
from ai_experiments.best_search import experiments as E

FOLDS_CSV = HERE / "results" / "folds_ek1.csv"
OUT_NPZ = HERE / "oof_bagsvm_logstd_ek1.npz"
TARGET = "BagSVM-MW/log_std"


def load_fold_params(path, exp_name):
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["experiment"] == exp_name:
                out[r["fold_label"]].append(json.loads(r["best_params"]))
    return out


def median_params(rows_per_fold):
    out = {}
    for fold, rows in rows_per_fold.items():
        keys = set().union(*[set(r.keys()) for r in rows])
        d = {}
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                d[k] = float(np.median(vals))
            else:
                d[k] = vals[0]
        out[fold] = d
    return out


def predict_fold(X, y, Yb, train_idx, test_idx, params, n_estimators=10):
    pre = Pipeline(E._pre("log_std"))
    pre.fit(X[train_idx])
    Xs_tr = pre.transform(X[train_idx])
    Xs_te = pre.transform(X[test_idx])
    w = np.abs(Yb[train_idx, 1] - Yb[train_idx, 0]) ** params["wpow"] + 1e-12
    n = len(train_idx); sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y[train_idx] == 1)[0]
    idx_neg = np.where(y[train_idx] == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos
    probs = np.zeros((len(test_idx), 2))
    for seed in range(n_estimators):
        rng = np.random.default_rng(seed)
        sel = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=True),
            rng.choice(idx_neg, size=n_neg, replace=True),
        ])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs_tr[sel], y[train_idx][sel], sample_weight=w[sel])
        probs += base.predict_proba(Xs_te)
    return test_idx, probs / n_estimators


def main():
    X, Y, meta = get_cpsat8_ek1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    rows_per_fold = load_fold_params(FOLDS_CSV, TARGET)
    params_per_fold = median_params(rows_per_fold)
    print(f"loaded median params for {len(params_per_fold)} folds")

    folds = leave_one_year_out_folds(years)
    t0 = time.time()
    results = Parallel(n_jobs=15)(
        delayed(predict_fold)(X, y_labels, Y_borda, tr, te,
                              params_per_fold[str(fold_label)])
        for fold_label, tr, te in folds
    )
    print(f"OOF done in {time.time()-t0:.0f}s")

    mean_p = np.zeros((len(X), 2))
    for test_idx, probs in results:
        mean_p[test_idx] = probs
    pred = np.argmax(mean_p, axis=1)
    np.savez(OUT_NPZ, mean_p=mean_p, pred=pred,
             y_labels=y_labels, Y_borda=Y_borda, years=years)
    print(f"saved {OUT_NPZ}")
    print(f"OOF Borda={float(Y_borda[np.arange(len(X)), pred].sum()):.2f} "
          f"oracle={float(Y_borda.max(axis=1).sum()):.2f} "
          f"acc={(pred==y_labels).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
