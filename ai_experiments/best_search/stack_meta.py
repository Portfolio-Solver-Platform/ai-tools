"""Train a meta-classifier on stacked OOF predictions from multiple base models.

Loads OOF P(k1) from oof/oof_*.npz, expands to [p, p**2, log(p+eps)] per
base, and runs LOYO with the chosen meta-model.

Usage:
    python -m ai_experiments.best_search.stack_meta \
        oof_BagSVM-MW__log_std.npz oof_XGB-MW__quantile.npz oof_MO-SVR__log_std.npz
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cross_solver_eval import leave_one_year_out_folds  # noqa: E402

OOF_DIR = Path(__file__).resolve().parent / "oof"


def build_meta_features(npz_paths: list[Path]):
    feats = []
    labels = years = Yb = None
    for path in npz_paths:
        d = np.load(path)
        p = d["mean_p"][:, 1]
        feats.append(p)
        feats.append(p ** 2)
        feats.append(np.log(p + 1e-3))
        if labels is None:
            labels = d["y_labels"]; years = d["years"]; Yb = d["Y_borda"]
    X_meta = np.stack(feats, axis=1).astype(np.float64)
    return X_meta, labels, years, Yb


def loyo_meta(X_meta, y, Yb, years, meta_kind="logreg"):
    folds = leave_one_year_out_folds(years)
    fold_records = []
    for fold_label, train_idx, test_idx in folds:
        if meta_kind == "logreg":
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(C=1.0, max_iter=2000))])
        elif meta_kind == "logreg_strong":
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(C=10.0, max_iter=2000,
                                                        class_weight="balanced"))])
        elif meta_kind == "xgb":
            import xgboost as xgb
            pipe = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, n_jobs=1, random_state=42, verbosity=0,
            )
        elif meta_kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            pipe = RandomForestClassifier(n_estimators=200, max_depth=8,
                                           n_jobs=1, random_state=42)
        else:
            raise ValueError(f"unknown meta_kind: {meta_kind}")
        # Borda-margin sample weight so the meta is Borda-aware.
        margin = np.abs(Yb[train_idx, 1] - Yb[train_idx, 0]) + 1e-12
        try:
            if hasattr(pipe, "fit"):
                if "logreg" in meta_kind:
                    pipe.fit(X_meta[train_idx], y[train_idx],
                             clf__sample_weight=margin)
                else:
                    pipe.fit(X_meta[train_idx], y[train_idx], sample_weight=margin)
        except TypeError:
            pipe.fit(X_meta[train_idx], y[train_idx])
        pred = pipe.predict(X_meta[test_idx])
        Y_te = Yb[test_idx]
        fold_records.append({
            "fold_label":     fold_label,
            "n_test":         len(test_idx),
            "test_borda":     float(Y_te[np.arange(len(test_idx)), pred].sum()),
            "oracle":         float(Y_te.max(axis=1).sum()),
            "cpsat_baseline": float(Y_te[:, 0].sum()),
            "accuracy":       float((pred == y[test_idx]).mean()),
        })
    return fold_records


def main(npz_names: list[str], meta_kinds: list[str] = ("logreg", "logreg_strong", "xgb", "rf")):
    paths = [OOF_DIR / n for n in npz_names]
    for p in paths:
        if not p.exists():
            print(f"missing: {p}"); sys.exit(2)

    X_meta, y, years, Yb = build_meta_features(paths)
    print(f"meta feature shape: {X_meta.shape}")
    print(f"bases: {[p.stem for p in paths]}")

    for meta_kind in meta_kinds:
        records = loyo_meta(X_meta, y, Yb, years, meta_kind=meta_kind)
        sb = sum(r["test_borda"] for r in records)
        so = sum(r["oracle"] for r in records)
        sc = sum(r["cpsat_baseline"] for r in records)
        n_total = sum(r["n_test"] for r in records)
        acc = sum(r["accuracy"] * r["n_test"] for r in records) / n_total
        print(f"  meta={meta_kind:16s}  borda={sb:7.2f}  ratio={sb/so:.3f}  "
              f"acc={acc*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
