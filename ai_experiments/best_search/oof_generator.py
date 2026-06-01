"""Generate out-of-fold (OOF) predictions for the base models we want to stack.

For each LOYO outer fold, refit the base model using its saved best_params
from folds.csv on the outer-train, and predict on the outer-test. Saves
oof_<experiment_slug>.npz with mean_p, pred, y_labels, Y_borda, years.
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cross_solver_eval import leave_one_year_out_folds  # noqa: E402
from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402
from ai_experiments.best_search.stack_ensemble import (  # noqa: E402
    DISPATCH, _parse_exp, load_folds_csv,
)

OOF_DIR = Path(__file__).resolve().parent / "oof"
OOF_DIR.mkdir(exist_ok=True)


def _slug(name):
    return name.replace("/", "__").replace(":", "_")


def _predict_fold(exp_name, params, X, y, Yb, train_idx, test_idx):
    prefix, pre = _parse_exp(exp_name)
    fn = DISPATCH[prefix]
    probs = fn(pre, params, X[train_idx], y[train_idx], Yb[train_idx], X[test_idx])
    return probs


def make_oof(exp_name, X, y, Yb, years, table):
    folds = leave_one_year_out_folds(years)
    params_per_fold = table[exp_name]
    if len(params_per_fold) < len(folds):
        raise ValueError(f"missing fold params for {exp_name}: have "
                         f"{len(params_per_fold)}/{len(folds)}")

    def _one(fold_label, train_idx, test_idx):
        params = params_per_fold[str(fold_label)]
        probs = _predict_fold(exp_name, params, X, y, Yb, train_idx, test_idx)
        return fold_label, test_idx, probs

    t0 = time.time()
    results = Parallel(n_jobs=15)(
        delayed(_one)(fl, tr, te) for fl, tr, te in folds
    )
    print(f"  {exp_name}: parallel fit/predict done in {time.time()-t0:.0f}s",
          flush=True)

    mean_p = np.zeros((len(X), 2))
    for fl, te, probs in results:
        mean_p[te] = probs
    pred = np.argmax(mean_p, axis=1)
    return mean_p, pred


def main(targets: list[str]):
    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    table = load_folds_csv()
    missing = [t for t in targets if t not in table]
    if missing:
        print(f"missing from folds.csv: {missing}")
        sys.exit(2)

    for exp_name in targets:
        out = OOF_DIR / f"oof_{_slug(exp_name)}.npz"
        if out.exists():
            print(f"  {exp_name}: already exists at {out}, skipping")
            continue
        mean_p, pred = make_oof(exp_name, X, y_labels, Y_borda, years, table)
        np.savez(out, mean_p=mean_p, pred=pred, y_labels=y_labels,
                 Y_borda=Y_borda, years=years)
        train_borda = float(Y_borda[np.arange(len(X)), pred].sum())
        oracle = float(Y_borda.max(axis=1).sum())
        cpsat = float(Y_borda[:, 0].sum())
        print(f"  {exp_name}: OOF borda={train_borda:.2f} oracle={oracle:.2f} "
              f"ratio={train_borda/oracle:.3f}  saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
