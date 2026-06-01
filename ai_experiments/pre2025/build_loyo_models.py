#!/usr/bin/env python3
"""Train svc_{k1,ek1} for each held-out year in {2020, 2021, 2022}.

Outputs (6 files):
    models/svc_k1_no2020.joblib   models/svc_ek1_no2020.joblib
    models/svc_k1_no2021.joblib   models/svc_ek1_no2021.joblib
    models/svc_k1_no2022.joblib   models/svc_ek1_no2022.joblib

For each held-out year Y, the model is fit on all rows where year != Y,
with HPO from the median of the 3 LOYO fold rows where Y was held out.

Usage:
    python build_loyo_models.py            # build all 6
    python build_loyo_models.py 2021       # build only the two 2021 models
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
HOLDOUT_YEARS = (2020, 2021, 2022)
ALL_YEARS = set(range(2011, 2026))

DATASETS = {
    "k1": {
        "npz":      AI_TOOLS_ROOT / "data/portfolios_cpsat8_k1_training_data.npz",
        "folds":    AI_TOOLS_ROOT / "ai_experiments/best_search/results/folds.csv",
        "exp_name": "BagSVM-MW/log_std",
    },
    "ek1": {
        "npz":      AI_TOOLS_ROOT / "data/portfolios_cpsat8_ek1_training_data.npz",
        "folds":    AI_TOOLS_ROOT / "ai_experiments/best_search/results/folds_ek1.csv",
        "exp_name": "BagSVM-MW/log_std",
    },
}


def out_path(family: str, holdout_year: int) -> Path:
    return SCRIPT_DIR / "models" / f"svc_{family}_no{holdout_year}.joblib"


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


def build_one(family: str, cfg: dict, holdout_year: int) -> bool:
    print(f"\n=== building svc_{family}_no{holdout_year} ===")
    if not cfg["npz"].exists():
        print(f"  missing training npz: {cfg['npz']}"); return False
    if not cfg["folds"].exists():
        print(f"  missing folds csv: {cfg['folds']}"); return False

    d = np.load(cfg["npz"], allow_pickle=True)
    X_all = d["X"]; Y_all = d["Y"]; meta_all = d["meta"]
    years_all = np.array([int(m["year"]) for m in meta_all])

    train_mask = years_all != holdout_year
    test_mask  = years_all == holdout_year
    X = X_all[train_mask]; Y = Y_all[train_mask]
    y_labels = np.argmax(Y, axis=1)

    train_year_set = set(years_all[train_mask].tolist())
    print(f"  train rows: {train_mask.sum()}  years={sorted(train_year_set)}")
    print(f"  held-out {holdout_year} rows: {test_mask.sum()} (NOT used in training)")
    assert train_mask.sum() + test_mask.sum() == len(X_all)
    assert holdout_year not in train_year_set
    assert train_year_set == ALL_YEARS - {holdout_year}

    params, n_fold_rows = load_holdout_params(
        cfg["folds"], cfg["exp_name"], holdout_year)
    print(f"  HP source: median of {n_fold_rows} LOYO fold-{holdout_year} rows "
          f"(HP tuning never saw {holdout_year})")
    print(f"  params: {params}")

    t0 = time.time()
    pre_pipe, svms = fit_bag(X, y_labels, Y, params)
    predictor = BagSVCPredictor(pre_pipe=pre_pipe, svms=svms, params=params)
    out = out_path(family, holdout_year)
    out.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(predictor, out)
    print(f"  saved {out}  ({time.time()-t0:.0f}s, {out.stat().st_size/1e6:.1f} MB)")

    pred_tr = predictor.predict(X)
    tb_tr = float(Y[np.arange(len(X)), pred_tr].sum())
    oracle_tr = float(Y.max(axis=1).sum())
    cpsat_tr  = float(Y[:, 0].sum())
    print(f"  training-set sanity:")
    print(f"    borda={tb_tr:.2f}  oracle={oracle_tr:.2f}  cpsat-only={cpsat_tr:.2f}")
    print(f"    acc vs argmax = {(pred_tr == y_labels).mean()*100:.1f}%")

    X_te = X_all[test_mask]; Y_te = Y_all[test_mask]
    pred_te = predictor.predict(X_te)
    tb_te = float(Y_te[np.arange(len(X_te)), pred_te].sum())
    oracle_te = float(Y_te.max(axis=1).sum())
    cpsat_te = float(Y_te[:, 0].sum())
    picks_alt = int((pred_te == 1).sum())
    y_labels_te = np.argmax(Y_te, axis=1)
    print(f"  >>> held-out {holdout_year} eval (informational) <<<")
    print(f"    n={len(X_te)}  borda={tb_te:.2f}  oracle={oracle_te:.2f}  "
          f"cpsat-only={cpsat_te:.2f}")
    print(f"    ratio borda/oracle = {tb_te/oracle_te:.3f}  "
          f"({tb_te - cpsat_te:+.2f} vs cpsat-only)")
    print(f"    acc vs argmax = {(pred_te == y_labels_te).mean()*100:.1f}%")
    print(f"    picked alt on {picks_alt}/{len(X_te)} {holdout_year} instances "
          f"({picks_alt/len(X_te)*100:.1f}%)")
    return True


def main():
    if len(sys.argv) > 1:
        years = (int(sys.argv[1]),)
        if years[0] not in HOLDOUT_YEARS:
            print(f"year {years[0]} not in {HOLDOUT_YEARS}"); sys.exit(1)
    else:
        years = HOLDOUT_YEARS

    ok = True
    for y in years:
        for family in DATASETS:
            ok = build_one(family, DATASETS[family], y) and ok
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
