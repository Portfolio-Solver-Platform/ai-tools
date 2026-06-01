"""Train the best ensemble on all 15 years and save it as a deliverable.

Reads results/summary.csv, picks the highest-Borda experiment, refits on
the full dataset using the median LOYO best_params, and joblib-dumps it.
Use load_and_predict(path, X) to predict.
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402

from ai_experiments.best_search import experiments as E  # noqa: E402
from ai_experiments.best_search.harness import run_hpo  # noqa: E402
from ai_experiments.best_search.run import EXPERIMENTS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def best_experiment_key() -> tuple[str, float]:
    rows = list(csv.DictReader(open(RESULTS_DIR / "summary.csv")))
    rows.sort(key=lambda r: -float(r["total_borda"]))
    top = rows[0]
    extras = json.loads(top["extras"])
    return extras["key"], float(top["total_borda"])


def median_loyo_params(experiment_name: str) -> dict | None:
    """Median (numeric) / mode (categorical) of best_params across LOYO folds."""
    rows = [r for r in csv.DictReader(open(RESULTS_DIR / "folds.csv"))
            if r["experiment"] == experiment_name]
    if not rows:
        return None
    all_params = [json.loads(r["best_params"]) for r in rows]
    keys = set().union(*(p.keys() for p in all_params))
    out: dict = {}
    for k in keys:
        vals = [p[k] for p in all_params if k in p]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.median(vals))
            if all(isinstance(v, int) for v in vals):
                out[k] = int(round(out[k]))
        else:
            counts: dict = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            out[k] = max(counts.items(), key=lambda kv: kv[1])[0]
    return out


def fit_bagged_svm_mw_final(X, y, Yb, params, pre_name: str, n_estimators: int):
    from ai_experiments.best_search.experiments import _pre

    pre_pipe = Pipeline(_pre(pre_name))
    pre_pipe.fit(X)
    Xs = pre_pipe.transform(X)

    w_full = np.abs(Yb[:, 1] - Yb[:, 0]) ** params["wpow"] + 1e-12
    n = len(X)
    sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos

    models = []
    for seed in range(n_estimators):
        rng = np.random.default_rng(seed)
        sel_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        sel_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        sel = np.concatenate([sel_pos, sel_neg])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs[sel], y[sel], sample_weight=w_full[sel])
        models.append(base)

    return {
        "preprocessor": pre_pipe,
        "models":       models,
        "wpow":         params["wpow"],
        "max_samples":  params["max_samples"],
        "n_estimators": n_estimators,
        "params":       params,
        "pre_name":     pre_name,
        "kind":         "bagged_svm_mw",
    }


def load_and_predict(path: Path, X: np.ndarray) -> np.ndarray:
    payload = joblib.load(path)
    Xs = payload["preprocessor"].transform(X)
    probs_sum = np.zeros((len(X), 2))
    for m in payload["models"]:
        probs_sum += m.predict_proba(Xs)
    return np.argmax(probs_sum, axis=1)


def main():
    key, borda = best_experiment_key()
    print(f"Best experiment key: {key}  (total LOYO Borda = {borda:.2f})")
    exp = EXPERIMENTS[key]()
    print(f"Experiment: {exp.name}  n_trials={exp.n_trials}")

    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    # Use median per-fold HPO winners as deliverable params — avoids a
    # redundant full-data HPO since LOYO already explored the search space.
    final_params = median_loyo_params(exp.name)
    if final_params is None:
        print(f"No LOYO folds found for {exp.name}; running fresh HPO...")
        final_params, _ = run_hpo(
            X, y_labels, Y_borda, years, exp, inner_k=5, seed=42,
        )
    print(f"final params (median of LOYO folds) = {final_params}")

    if exp.name.startswith("BagSVM-MW"):
        pre_name = exp.name.split("/", 1)[1]
        sample_pipe = exp.build(final_params)
        n_est = sample_pipe._n_estimators
        payload = fit_bagged_svm_mw_final(
            X, y_labels, Y_borda, final_params, pre_name, n_est,
        )
        payload["experiment_name"] = exp.name
        out_path = MODELS_DIR / f"best_{key}.joblib"
        joblib.dump(payload, out_path)
        print(f"Saved {out_path}")
        pred = load_and_predict(out_path, X)
        train_borda = Y_borda[np.arange(len(y_labels)), pred].sum()
        train_acc = (pred == y_labels).mean()
        print(f"Self-check on training data: borda={train_borda:.2f} "
              f"acc={train_acc*100:.1f}%")
    else:
        print(f"WARNING: don't know how to save model family '{exp.name}'.")
        pipe = exp.build(final_params)
        fp = exp.fit_predict
        if fp is not None:
            _ = fp(pipe, X, y_labels, Y_borda, X[:1])
        else:
            pipe.fit(X, y_labels)
        out_path = MODELS_DIR / f"best_{key}.joblib"
        joblib.dump({"pipe": pipe, "experiment_name": exp.name,
                     "params": final_params, "kind": "raw_pipeline"}, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
