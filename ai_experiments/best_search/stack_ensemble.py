"""Stacking ensemble: averages predict_proba of base models across LOYO folds.

For each base experiment, reuses the per-fold best_params from folds.csv,
refits on the outer-train, and averages predict_proba on the outer-test.
LOYO-safe because each fold's params were chosen by inner CV on its train.

Usage:
    python -m ai_experiments.best_search.stack_ensemble \
        "BagSVM-MW/log_std" "BagSVM-MW/quantile" "XGB-MW/quantile"
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cross_solver_eval import leave_one_year_out_folds  # noqa: E402
from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402

from ai_experiments.best_search import experiments as E  # noqa: E402

FOLDS_CSV = Path(__file__).resolve().parent / "results" / "folds.csv"


def load_folds_csv():
    table = defaultdict(dict)
    with open(FOLDS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            table[row["experiment"]][row["fold_label"]] = json.loads(row["best_params"])
    return table


def proba_svm_mw(pre, params, X_tr, y_tr, Yb_tr, X_te):
    pipe = Pipeline(E._pre(pre) + [
        ("model", SVC(kernel="rbf", probability=True, random_state=42,
                      C=params["C"], gamma=params["gamma"])),
    ])
    w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12
    pipe.fit(X_tr, y_tr, model__sample_weight=w)
    return pipe.predict_proba(X_te)


def proba_bag_svm_mw(pre, params, X_tr, y_tr, Yb_tr, X_te, n_estimators=10):
    pre_pipe = Pipeline(E._pre(pre))
    pre_pipe.fit(X_tr)
    Xs_tr = pre_pipe.transform(X_tr)
    Xs_te = pre_pipe.transform(X_te)
    w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12

    n = len(X_tr)
    sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y_tr == 1)[0]
    idx_neg = np.where(y_tr == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos

    probs = np.zeros((len(X_te), 2))
    for seed in range(n_estimators):
        rng = np.random.default_rng(seed)
        sel_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sel_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w[sel])
        probs += base.predict_proba(Xs_te)
    return probs / n_estimators


def proba_xgb_mw(pre, params, X_tr, y_tr, Yb_tr, X_te):
    import xgboost as xgb
    pipe = Pipeline(E._pre(pre) + [
        ("model", xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"],
            tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
        )),
    ])
    wpow = params["wpow"]
    if wpow == 0:
        pipe.fit(X_tr, y_tr)
    else:
        w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** wpow + 1e-12
        pipe.fit(X_tr, y_tr, model__sample_weight=w)
    return pipe.predict_proba(X_te)


def proba_mo_svr(pre, params, X_tr, y_tr, Yb_tr, X_te):
    """Multi-output SVR returning normalized predicted bordas as pseudo-probs."""
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR
    base = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"])
    pipe = Pipeline(E._pre(pre) + [("model", MultiOutputRegressor(base, n_jobs=1))])
    pipe.fit(X_tr, Yb_tr)
    Y_pred = pipe.predict(X_te)
    Y_pred = np.clip(Y_pred, 0, None)
    s = Y_pred.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return Y_pred / s


DISPATCH = {
    "SVM-RBF-MW":  proba_svm_mw,
    "BagSVM-MW":   proba_bag_svm_mw,
    "XGB-MW":      proba_xgb_mw,
    "MO-SVR":      proba_mo_svr,
}


def _parse_exp(name):
    prefix, pre = name.split("/", 1)
    if prefix not in DISPATCH:
        raise ValueError(f"unsupported base for stacking: {name}")
    return prefix, pre


def predict_one_fold(base_exps, params_per_exp, X_tr, y_tr, Yb_tr, X_te):
    probs_sum = np.zeros((len(X_te), 2))
    for exp_name in base_exps:
        prefix, pre = _parse_exp(exp_name)
        fn = DISPATCH[prefix]
        probs_sum += fn(pre, params_per_exp[exp_name], X_tr, y_tr, Yb_tr, X_te)
    return np.argmax(probs_sum, axis=1)


def _evaluate_fold(base_exps, params_for_each_fold, X, y, Yb, years,
                   fold_label, train_idx, test_idx):
    t0 = time.time()
    params_per_exp = {e: params_for_each_fold[e][str(fold_label)] for e in base_exps}
    pred = predict_one_fold(base_exps, params_per_exp,
                            X[train_idx], y[train_idx], Yb[train_idx], X[test_idx])
    Y_te = Yb[test_idx]
    return {
        "fold_label":     fold_label,
        "n_test":         len(test_idx),
        "test_borda":     float(Y_te[np.arange(len(test_idx)), pred].sum()),
        "oracle":         float(Y_te.max(axis=1).sum()),
        "cpsat_baseline": float(Y_te[:, 0].sum()),
        "accuracy":       float((pred == y[test_idx]).mean()),
        "fit_seconds":    time.time() - t0,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    base_exps = sys.argv[1:]

    table = load_folds_csv()
    missing = [e for e in base_exps if e not in table]
    if missing:
        print(f"missing experiment(s) in folds.csv: {missing}")
        print(f"available: {sorted(table)}")
        sys.exit(2)

    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    folds = leave_one_year_out_folds(years)
    fold_records = Parallel(n_jobs=15)(
        delayed(_evaluate_fold)(
            base_exps, table, X, y_labels, Y_borda, years,
            fold_label, train_idx, test_idx,
        )
        for fold_label, train_idx, test_idx in folds
    )
    fold_records.sort(key=lambda r: r["fold_label"])

    name = "Stack(" + "+".join(b.replace("/", ":") for b in base_exps) + ")"
    print(f"========== {name} ==========")
    for r in fold_records:
        print(f"    {r['fold_label']}: borda={r['test_borda']:>6.2f}  "
              f"oracle={r['oracle']:>6.2f}  acc={r['accuracy']*100:>5.1f}%  "
              f"({r['fit_seconds']:.0f}s)")
    sb = sum(r["test_borda"]     for r in fold_records)
    so = sum(r["oracle"]         for r in fold_records)
    sc = sum(r["cpsat_baseline"] for r in fold_records)
    n_total = sum(r["n_test"]    for r in fold_records)
    acc = sum(r["accuracy"] * r["n_test"] for r in fold_records) / n_total
    print(f"  totals: borda={sb:.2f}  oracle={so:.2f}  cpsat={sc:.2f}  "
          f"ratio={sb/so:.3f}  acc={acc*100:.1f}%  ({n_total} test instances)")


if __name__ == "__main__":
    main()
