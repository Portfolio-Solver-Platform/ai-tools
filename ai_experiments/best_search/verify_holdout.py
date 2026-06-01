"""Sanity check: 80/20 random split with FIXED hyperparameters.

Verifies the bag is actually learning (not gaming LOYO / overfitting HPO).
Uses median LOYO params from cpsat8_k1 across BOTH datasets so neither
benefits from per-dataset tuning. Repeats over multiple seeds.
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.shared_data import (  # noqa: E402
    get_cpsat8_ek1_data, get_cpsat8_k1_data, prepare_labels,
)
from ai_experiments.best_search.preprocessing import SignedLog1p  # noqa: E402


# Median LOYO params from the cpsat8_k1 winning run.
FIXED_PARAMS = {
    "C": 8.30,
    "gamma": 0.0167,
    "wpow": 0.78,
    "max_samples": 0.86,
}
N_ESTIMATORS = 30


def bag_svm_mw_predict(X_tr, y_tr, Yb_tr, X_te, params, n_est=N_ESTIMATORS,
                       pre_name="log_std"):
    steps = [SignedLog1p(), StandardScaler()] if pre_name == "log_std" else [StandardScaler()]
    Xs_tr = X_tr.copy()
    for s in steps:
        s.fit(Xs_tr); Xs_tr = s.transform(Xs_tr)
    Xs_te = X_te.copy()
    for s in steps:
        Xs_te = s.transform(Xs_te)

    w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12
    n = len(X_tr); sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y_tr == 1)[0]
    idx_neg = np.where(y_tr == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos)/n))
    n_neg = sample_n - n_pos
    probs = np.zeros((len(X_te), 2))
    for seed in range(n_est):
        rng = np.random.default_rng(seed)
        sel = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=True),
            rng.choice(idx_neg, size=n_neg, replace=True),
        ])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w[sel])
        probs += base.predict_proba(Xs_te)
    return np.argmax(probs, axis=1)


def score(pred, Yb_te, y_te):
    return {
        "borda":  float(Yb_te[np.arange(len(Yb_te)), pred].sum()),
        "oracle": float(Yb_te.max(axis=1).sum()),
        "cpsat":  float(Yb_te[:, 0].sum()),
        "k1":     float(Yb_te[:, 1].sum()),
        "acc":    float((pred == y_te).mean()),
        "n":      len(Yb_te),
    }


def eval_split(X, y, Yb, seed):
    Xtr, Xte, ytr, yte, Ybtr, Ybte = train_test_split(
        X, y, Yb, test_size=0.20, random_state=seed, stratify=y,
    )

    rng = np.random.default_rng(seed)
    out = {}
    out["always_cpsat"] = score(np.zeros(len(yte), dtype=int), Ybte, yte)
    out["always_k1"]    = score(np.ones(len(yte), dtype=int),  Ybte, yte)
    out["random"]       = score(rng.integers(0, 2, len(yte)), Ybte, yte)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000))])
    pipe.fit(Xtr, ytr)
    out["logreg_default"] = score(pipe.predict(Xte), Ybte, yte)

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))])
    pipe.fit(Xtr, ytr)
    out["svm_rbf_default"] = score(pipe.predict(Xte), Ybte, yte)

    pred = bag_svm_mw_predict(Xtr, ytr, Ybtr, Xte, FIXED_PARAMS)
    out["bag_svm_mw_fixed"] = score(pred, Ybte, yte)
    return out


def main():
    datasets = [
        ("cpsat8_k1",  get_cpsat8_k1_data),
        ("cpsat8_ek1", get_cpsat8_ek1_data),
    ]
    seeds = [0, 1, 2, 3, 4]

    for name, getter in datasets:
        X, Y, _ = getter()
        y, Yb = prepare_labels(Y)
        print(f"\n========== {name}  ({len(y)} instances, "
              f"{(y==1).mean()*100:.1f}% positives) ==========")
        print(f"  FIXED params (no HPO): {FIXED_PARAMS}")
        rows = {}
        for s in seeds:
            r = eval_split(X, y, Yb, seed=s)
            for k, v in r.items():
                rows.setdefault(k, []).append(v)
            print(f"  seed={s}:")
            for k, v in r.items():
                print(f"    {k:18s} borda={v['borda']:6.2f}  "
                      f"oracle={v['oracle']:6.2f}  ratio={v['borda']/v['oracle']:.3f}  "
                      f"acc={v['acc']*100:5.1f}%")
        print(f"  ----- averaged over {len(seeds)} seeds -----")
        for k, vs in rows.items():
            bordas = [v["borda"] for v in vs]
            ratios = [v["borda"]/v["oracle"] for v in vs]
            accs   = [v["acc"] for v in vs]
            n = vs[0]["n"]
            print(f"    {k:18s} borda={np.mean(bordas):6.2f}±{np.std(bordas):.2f}  "
                  f"ratio={np.mean(ratios):.3f}±{np.std(ratios):.3f}  "
                  f"acc={np.mean(accs)*100:5.1f}±{np.std(accs)*100:.1f}%  (n_test={n})")


if __name__ == "__main__":
    main()
