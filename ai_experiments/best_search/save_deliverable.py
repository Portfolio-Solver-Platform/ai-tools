"""Fit the best experiment on the full dataset and save it as a deliverable.

Usage:
    python -m ai_experiments.best_search.save_deliverable bag_svm_mw_log_std

Runs one final HPO on all years, refits the bag ensemble on the full
dataset, and joblib-dumps a BagSVMPredictor.
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402

from ai_experiments.best_search import experiments as E  # noqa: E402
from ai_experiments.best_search.harness import run_hpo  # noqa: E402
from ai_experiments.best_search.run import EXPERIMENTS  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class BagSVMPredictor:
    """Pickleable wrapper for a bagged SVM-MW ensemble."""
    def __init__(self, params, pres, n_each, fitted_pre_pipes, fitted_svms):
        self.params = params
        self.pres = pres
        self.n_each = n_each
        self.pre_pipes = fitted_pre_pipes
        self.svms = fitted_svms

    def predict(self, X):
        probs_sum = np.zeros((len(X), 2))
        for pre_pipe, svm_list in zip(self.pre_pipes, self.svms):
            Xs = pre_pipe.transform(X)
            for svm in svm_list:
                probs_sum += svm.predict_proba(Xs)
        return np.argmax(probs_sum, axis=1)

    def predict_proba(self, X):
        probs_sum = np.zeros((len(X), 2))
        n_models = 0
        for pre_pipe, svm_list in zip(self.pre_pipes, self.svms):
            Xs = pre_pipe.transform(X)
            for svm in svm_list:
                probs_sum += svm.predict_proba(Xs)
                n_models += 1
        return probs_sum / n_models


def fit_bag_svm_deliverable(name, exp, X, y_labels, Y_borda, years):
    print(f"  running final HPO on all 15 years (n_trials={exp.n_trials})...",
          flush=True)
    best_params, best_score = run_hpo(X, y_labels, Y_borda, years, exp, inner_k=5)
    print(f"  best params: {best_params}  cv_borda={best_score:.4f}", flush=True)

    from sklearn.pipeline import Pipeline as P
    from sklearn.svm import SVC

    pipe = exp.build(best_params)
    is_diverse = hasattr(pipe, "_pres")
    if is_diverse:
        pres_list = pipe._pres
        n_each = pipe._n_each
    else:
        pres_list = [getattr(pipe, "_pre_name", None) or _infer_pre_from_steps(pipe)]
        n_each = pipe._n_estimators

    wpow = pipe._wpow
    max_samples = pipe._max_samples
    svm_params = pipe._svm_params
    w_full = np.abs(Y_borda[:, 1] - Y_borda[:, 0]) ** wpow + 1e-12

    n = len(X)
    sample_n = int(max_samples * n)
    idx_pos = np.where(y_labels == 1)[0]
    idx_neg = np.where(y_labels == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos

    fitted_pre_pipes = []
    fitted_svms = []
    seed_base = 0
    for pre_name in pres_list:
        pre_pipe = P(E._pre(pre_name))
        pre_pipe.fit(X)
        Xs = pre_pipe.transform(X)
        svm_list = []
        for k in range(n_each):
            seed = seed_base + k
            rng = np.random.default_rng(seed)
            sel_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            sel_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            sel = np.concatenate([sel_pos, sel_neg])
            base = SVC(kernel="rbf", probability=True, random_state=seed, **svm_params)
            base.fit(Xs[sel], y_labels[sel], sample_weight=w_full[sel])
            svm_list.append(base)
        fitted_pre_pipes.append(pre_pipe)
        fitted_svms.append(svm_list)
        seed_base += n_each

    return BagSVMPredictor(
        params=best_params, pres=pres_list, n_each=n_each,
        fitted_pre_pipes=fitted_pre_pipes, fitted_svms=fitted_svms,
    )


def _infer_pre_from_steps(pipe):
    return getattr(pipe, "_pre_name", "log_std")


def main():
    if len(sys.argv) < 2:
        print(f"usage: python -m ai_experiments.best_search.save_deliverable <key>\n"
              f"available: {sorted(EXPERIMENTS)}")
        sys.exit(1)
    key = sys.argv[1]
    if key not in EXPERIMENTS:
        print(f"unknown experiment: {key}")
        sys.exit(2)

    exp = EXPERIMENTS[key]()
    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    print(f"========== fitting deliverable: {exp.name} ==========", flush=True)
    t0 = time.time()
    predictor = fit_bag_svm_deliverable(key, exp, X, y_labels, Y_borda, years)
    dt = time.time() - t0
    out_path = MODELS_DIR / f"deliverable_{key}.joblib"
    joblib.dump(predictor, out_path)
    print(f"  saved {out_path}  (wall={dt:.0f}s)", flush=True)

    # Sanity check: train-set accuracy and borda — NOT generalization.
    pred = predictor.predict(X)
    train_borda = float(Y_borda[np.arange(len(X)), pred].sum())
    train_acc = float((pred == y_labels).mean())
    print(f"  training-set: borda={train_borda:.2f}  acc={train_acc*100:.1f}%  "
          f"(see LOYO summary.csv for generalization)",
          flush=True)


if __name__ == "__main__":
    main()
