"""Save a deliverable model using median/mode best_params from folds.csv (no fresh HPO).

Usage:
    python -m ai_experiments.best_search.save_deliverable_fast "BagSVM-MW/log_std"
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402

from ai_experiments.best_search import experiments as E  # noqa: E402
from ai_experiments.best_search.save_deliverable import BagSVMPredictor  # noqa: E402

FOLDS_CSV = Path(__file__).resolve().parent / "results" / "folds.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_fold_params(experiment_name: str) -> list[dict]:
    rows = []
    with open(FOLDS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == experiment_name:
                rows.append(json.loads(row["best_params"]))
    if not rows:
        raise ValueError(f"no rows found for {experiment_name}")
    return rows


def summarize_params(params_list: list[dict]) -> dict:
    """Median for numeric params, mode for categorical."""
    out: dict = {}
    keys = set()
    for p in params_list:
        keys.update(p.keys())
    for k in keys:
        vals = [p[k] for p in params_list if k in p]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.median(vals))
            if all(isinstance(v, int) for v in vals):
                out[k] = int(round(out[k]))
        else:
            out[k] = Counter(vals).most_common(1)[0][0]
    return out


def fit_bag_svm_with_params(pre_name, params, n_estimators, X, y_labels, Y_borda):
    pre_pipe = Pipeline(E._pre(pre_name))
    pre_pipe.fit(X)
    Xs = pre_pipe.transform(X)
    w = np.abs(Y_borda[:, 1] - Y_borda[:, 0]) ** params["wpow"] + 1e-12

    n = len(X)
    sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y_labels == 1)[0]
    idx_neg = np.where(y_labels == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos

    svm_list = []
    for k in range(n_estimators):
        seed = k
        rng = np.random.default_rng(seed)
        sel_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sel_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs[sel], y_labels[sel], sample_weight=w[sel])
        svm_list.append(base)

    return BagSVMPredictor(
        params=params, pres=[pre_name], n_each=n_estimators,
        fitted_pre_pipes=[pre_pipe], fitted_svms=[svm_list],
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    exp_name = sys.argv[1]
    n_estimators = int(sys.argv[2]) if len(sys.argv) >= 3 else 20

    if not exp_name.startswith("BagSVM-MW/"):
        print(f"only BagSVM-MW/* experiments are supported by this script; "
              f"got {exp_name}")
        sys.exit(2)
    pre_name = exp_name.split("/", 1)[1]

    params_per_fold = load_fold_params(exp_name)
    params = summarize_params(params_per_fold)
    print(f"loaded {len(params_per_fold)} per-fold best_params for {exp_name}")
    print(f"median/mode params: {params}", flush=True)

    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)

    t0 = time.time()
    print(f"fitting bagged ensemble: n_estimators={n_estimators}, pre={pre_name}",
          flush=True)
    predictor = fit_bag_svm_with_params(pre_name, params, n_estimators,
                                        X, y_labels, Y_borda)
    dt = time.time() - t0

    out_path = MODELS_DIR / f"deliverable_BagSVM_MW_{pre_name}.joblib"
    joblib.dump(predictor, out_path)
    print(f"saved {out_path}  (wall={dt:.0f}s)", flush=True)

    pred = predictor.predict(X)
    train_borda = float(Y_borda[np.arange(len(X)), pred].sum())
    train_acc = float((pred == y_labels).mean())
    oracle = float(Y_borda.max(axis=1).sum())
    print(f"training-set sanity: borda={train_borda:.2f} / oracle={oracle:.2f}  "
          f"acc={train_acc*100:.1f}%")


if __name__ == "__main__":
    main()
