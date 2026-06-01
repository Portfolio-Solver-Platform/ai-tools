#!/usr/bin/env python3
"""Train svc_{k1,ek1} on 2011-2024 only; 2025 is held out.

Outputs:
    models/svc_k1_2011_2024.joblib
    models/svc_ek1_2011_2024.joblib

The BagSVCPredictor / SignedLog1p classes are imported from
parasol/command-line-ai/svc_common.py so the resulting .joblib loads
cleanly in parasol with no path fiddling.

Hyperparameters: median of the LOYO fold rows where 2025 was the
held-out test year (3 reps), so HPO never saw 2025 either. After
fitting, the script also predicts on the held-out 2025 rows and prints
their borda/oracle/cpsat-only numbers as a sanity check.

Usage:
    python build_pre2025_model.py            # build both
    python build_pre2025_model.py k1         # build only k1
    python build_pre2025_model.py ek1        # build only ek1
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
PARASOL_CLI = Path("/home/sofus/speciale/ai/parasol/command-line-ai")
sys.path.insert(0, str(PARASOL_CLI))
from svc_common import BagSVCPredictor, SignedLog1p  # noqa: E402

AI_TOOLS_ROOT = Path("/home/sofus/speciale/ai/ai-tools")
N_ESTIMATORS = 30
HOLDOUT_YEAR = 2025
TRAIN_YEARS = set(range(2011, 2025))

DATASETS = {
    "k1": {
        "npz":      AI_TOOLS_ROOT / "data/portfolios_cpsat8_k1_training_data.npz",
        "folds":    AI_TOOLS_ROOT / "ai_experiments/best_search/results/folds.csv",
        "exp_name": "BagSVM-MW/log_std",
        "out":      SCRIPT_DIR / "models/svc_k1_2011_2024.joblib",
    },
    "ek1": {
        "npz":      AI_TOOLS_ROOT / "data/portfolios_cpsat8_ek1_training_data.npz",
        "folds":    AI_TOOLS_ROOT / "ai_experiments/best_search/results/folds_ek1.csv",
        "exp_name": "BagSVM-MW/log_std",
        "out":      SCRIPT_DIR / "models/svc_ek1_2011_2024.joblib",
    },
}


def load_holdout_params(folds_csv: Path, exp_name: str, fold_label: int):
    rows = []
    with open(folds_csv) as f:
        for r in csv.DictReader(f):
            if r["experiment"] == exp_name and int(r["fold_label"]) == fold_label:
                rows.append(json.loads(r["best_params"]))
    if not rows:
        raise RuntimeError(
            f"no rows for {exp_name} fold={fold_label} in {folds_csv}")
    keys = set().union(*[set(r.keys()) for r in rows])
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.median(vals))
        else:
            out[k] = Counter(vals).most_common(1)[0][0]
    return out, len(rows)


def fit_bag(X, y_labels, Y_borda, params, n_estimators=N_ESTIMATORS):
    pre_pipe = Pipeline([("log", SignedLog1p()), ("scaler", StandardScaler())])
    pre_pipe.fit(X)
    Xs = pre_pipe.transform(X)

    w = np.abs(Y_borda[:, 1] - Y_borda[:, 0]) ** params["wpow"] + 1e-12
    n = len(X); sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y_labels == 1)[0]
    idx_neg = np.where(y_labels == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos

    svms = []
    for seed in range(n_estimators):
        rng = np.random.default_rng(seed)
        sel = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=True),
            rng.choice(idx_neg, size=n_neg, replace=True),
        ])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs[sel], y_labels[sel], sample_weight=w[sel])
        svms.append(base)
    return pre_pipe, svms


def build_one(name: str, cfg: dict):
    print(f"\n=== building {name} on years 2011-2024 ===")
    if not cfg["npz"].exists():
        print(f"  missing training npz: {cfg['npz']}"); return False
    if not cfg["folds"].exists():
        print(f"  missing folds csv: {cfg['folds']}"); return False

    d = np.load(cfg["npz"], allow_pickle=True)
    X_all = d["X"]; Y_all = d["Y"]; meta_all = d["meta"]
    years_all = np.array([int(m["year"]) for m in meta_all])
    print(f"  full dataset:  X={X_all.shape}  "
          f"years={sorted(Counter(years_all.tolist()).items())}")

    train_mask = years_all != HOLDOUT_YEAR
    test_mask  = years_all == HOLDOUT_YEAR
    X = X_all[train_mask]; Y = Y_all[train_mask]
    y_labels = np.argmax(Y, axis=1)

    print(f"  train rows: {train_mask.sum()}  "
          f"({sorted(Counter(years_all[train_mask].tolist()).items())})")
    print(f"  held-out {HOLDOUT_YEAR} rows: {test_mask.sum()} (NOT used in training)")
    assert train_mask.sum() + test_mask.sum() == len(X_all)
    assert HOLDOUT_YEAR not in set(years_all[train_mask].tolist())
    assert set(years_all[train_mask].tolist()) == TRAIN_YEARS

    params, n_fold_rows = load_holdout_params(
        cfg["folds"], cfg["exp_name"], HOLDOUT_YEAR)
    print(f"  HP source: median of {n_fold_rows} LOYO fold-{HOLDOUT_YEAR} rows "
          f"for {cfg['exp_name']!r}  (HP tuning never saw {HOLDOUT_YEAR})")
    print(f"  params: {params}")

    t0 = time.time()
    pre_pipe, svms = fit_bag(X, y_labels, Y, params)
    predictor = BagSVCPredictor(pre_pipe=pre_pipe, svms=svms, params=params)
    cfg["out"].parent.mkdir(exist_ok=True)
    joblib.dump(predictor, cfg["out"])
    print(f"  saved {cfg['out']}  ({time.time()-t0:.0f}s, "
          f"{cfg['out'].stat().st_size/1e6:.1f} MB)")

    pred_tr = predictor.predict(X)
    tb_tr = float(Y[np.arange(len(X)), pred_tr].sum())
    oracle_tr = float(Y.max(axis=1).sum())
    cpsat_tr  = float(Y[:, 0].sum())
    print(f"  training-set (2011-2024) sanity:")
    print(f"    borda={tb_tr:.2f}  oracle={oracle_tr:.2f}  cpsat-only={cpsat_tr:.2f}")
    print(f"    acc vs argmax label = {(pred_tr == y_labels).mean()*100:.1f}%")

    X_te = X_all[test_mask]; Y_te = Y_all[test_mask]
    pred_te = predictor.predict(X_te)
    tb_te = float(Y_te[np.arange(len(X_te)), pred_te].sum())
    oracle_te = float(Y_te.max(axis=1).sum())
    cpsat_te = float(Y_te[:, 0].sum())
    picks_alt = int((pred_te == 1).sum())
    y_labels_te = np.argmax(Y_te, axis=1)
    print(f"  >>> held-out {HOLDOUT_YEAR} eval (informational) <<<")
    print(f"    n={len(X_te)}  borda={tb_te:.2f}  oracle={oracle_te:.2f}  "
          f"cpsat-only={cpsat_te:.2f}")
    print(f"    ratio borda/oracle = {tb_te/oracle_te:.3f}  "
          f"({tb_te - cpsat_te:+.2f} vs cpsat-only)")
    print(f"    acc vs argmax = {(pred_te == y_labels_te).mean()*100:.1f}%")
    print(f"    picked alt portfolio on {picks_alt}/{len(X_te)} 2025 instances "
          f"({picks_alt/len(X_te)*100:.1f}%)")
    return True


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which == "both":
        targets = list(DATASETS)
    elif which in DATASETS:
        targets = [which]
    else:
        print(f"unknown target {which!r}; expected one of: both, k1, ek1")
        sys.exit(1)

    ok = True
    for name in targets:
        ok = build_one(name, DATASETS[name]) and ok
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
