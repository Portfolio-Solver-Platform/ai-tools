"""LOYO stacking with original features + base OOF predictions as joint meta input.

OOF predictions from oof/*.npz are LOYO-safe (each instance was held out
of the model that produced its prediction), so they can be used as
features for a meta-classifier trained per outer fold.
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cross_solver_eval import leave_one_year_out_folds  # noqa: E402
from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402
from ai_experiments.best_search.preprocessing import SignedLog1p  # noqa: E402

OOF_DIR = Path(__file__).resolve().parent / "oof"


def load_base_probs(names):
    probs = []
    for name in names:
        d = np.load(OOF_DIR / name)
        probs.append(d["mean_p"][:, 1])
    return np.stack(probs, axis=1)


def _evaluate_fold(meta_kind, X, X_meta, y, Yb, fold_label, train_idx, test_idx,
                   margin_w):
    if meta_kind.startswith("xgb"):
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=1, random_state=42, verbosity=0,
        )
        clf.fit(X_meta[train_idx], y[train_idx], sample_weight=margin_w[train_idx])
        pred = clf.predict(X_meta[test_idx])
    elif meta_kind.startswith("svm"):
        clf = Pipeline([("scaler", StandardScaler()),
                        ("clf", SVC(kernel="rbf", C=3.0, gamma="scale",
                                    probability=True, random_state=42))])
        clf.fit(X_meta[train_idx], y[train_idx], clf__sample_weight=margin_w[train_idx])
        pred = clf.predict(X_meta[test_idx])
    elif meta_kind.startswith("logreg"):
        clf = Pipeline([("scaler", StandardScaler()),
                        ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                                    class_weight=None))])
        clf.fit(X_meta[train_idx], y[train_idx], clf__sample_weight=margin_w[train_idx])
        pred = clf.predict(X_meta[test_idx])
    else:
        raise ValueError(meta_kind)
    Y_te = Yb[test_idx]
    return {
        "fold_label":     fold_label,
        "n_test":         len(test_idx),
        "test_borda":     float(Y_te[np.arange(len(test_idx)), pred].sum()),
        "oracle":         float(Y_te.max(axis=1).sum()),
        "cpsat_baseline": float(Y_te[:, 0].sum()),
        "accuracy":       float((pred == y[test_idx]).mean()),
    }


def main(meta_kind, base_names, use_features=True, log_features=True):
    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    P = load_base_probs(base_names)
    if use_features:
        if log_features:
            X_pre = SignedLog1p().fit_transform(X)
            X_pre = StandardScaler().fit_transform(X_pre)
        else:
            X_pre = StandardScaler().fit_transform(X.astype(float))
        X_meta = np.hstack([X_pre, P, P ** 2, np.log(P + 1e-3)])
    else:
        X_meta = np.hstack([P, P ** 2, np.log(P + 1e-3)])
    print(f"meta feature shape: {X_meta.shape}  meta_kind={meta_kind}")

    margin_w = np.abs(Y_borda[:, 1] - Y_borda[:, 0]) + 1e-12
    folds = leave_one_year_out_folds(years)
    records = Parallel(n_jobs=15)(
        delayed(_evaluate_fold)(meta_kind, X, X_meta, y_labels, Y_borda,
                                fold_label, tr, te, margin_w)
        for fold_label, tr, te in folds
    )
    records.sort(key=lambda r: r["fold_label"])
    for r in records:
        print(f"    {r['fold_label']}: borda={r['test_borda']:>6.2f}  "
              f"oracle={r['oracle']:>6.2f}  acc={r['accuracy']*100:5.1f}%")
    sb = sum(r["test_borda"] for r in records)
    so = sum(r["oracle"] for r in records)
    n_total = sum(r["n_test"] for r in records)
    acc = sum(r["accuracy"] * r["n_test"] for r in records) / n_total
    print(f"  totals: borda={sb:.2f}  ratio={sb/so:.3f}  acc={acc*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: python -m ai_experiments.best_search.stack_with_features "
              "<meta_kind> <oof_name1.npz> [oof_name2.npz...]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2:])
