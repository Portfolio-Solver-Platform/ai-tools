"""BagSVM-MW/log_std on the rep-enriched data.

Variants compared via 15-fold LOYO + Optuna HPO:
  baseline       - original X, no rep info
  +features      - X plus 8 per-portfolio rep-variance features
  +conf          - margin sample weight multiplied by label_confidence
  +features+conf - both

Pass the dataset stem ('cpsat8_k1' or 'cpsat8_ek1') as argv[1].
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
import time
from pathlib import Path

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cross_solver_eval import leave_one_year_out_folds  # noqa: E402
from ai_experiments.best_search import experiments as E  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = ROOT / "data"
N_ESTIMATORS = 10
N_TRIALS = 60
INNER_K = 5


def load(stem: str):
    d = np.load(DATA_DIR / f"portfolios_{stem}_training_data_reps.npz",
                allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    X_rep = d["X_rep_features"].reshape(len(X), -1)
    conf = d["label_confidence"]
    years = d["meta"]["year"]
    y = np.argmax(Y, axis=1)
    return X, X_rep, conf, Y, y, years


def _bag_predict(X_tr, y_tr, Yb_tr, w_extra_tr, X_te, params, pre="log_std",
                 n_est=N_ESTIMATORS):
    pre_pipe = Pipeline(E._pre(pre)); pre_pipe.fit(X_tr)
    Xs_tr = pre_pipe.transform(X_tr); Xs_te = pre_pipe.transform(X_te)
    w_margin = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12
    w_full = w_margin * w_extra_tr
    n = len(X_tr); sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y_tr == 1)[0]; idx_neg = np.where(y_tr == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos)/n))
    n_neg = sample_n - n_pos
    probs = np.zeros((len(X_te), 2))
    for seed in range(n_est):
        rng = np.random.default_rng(seed)
        sel = np.concatenate([rng.choice(idx_pos, size=n_pos, replace=True),
                              rng.choice(idx_neg, size=n_neg, replace=True)])
        b = SVC(kernel="rbf", probability=True, random_state=seed,
                C=params["C"], gamma=params["gamma"])
        b.fit(Xs_tr[sel], y_tr[sel], sample_weight=w_full[sel])
        probs += b.predict_proba(Xs_te)
    return probs / n_est


def _cv_score(X, y, Yb, w_extra, years, params, splits, pre):
    fold_means = []
    for tr, te in splits:
        probs = _bag_predict(X[tr], y[tr], Yb[tr], w_extra[tr], X[te], params, pre)
        pred = np.argmax(probs, axis=1)
        fold_means.append(Yb[te][np.arange(len(te)), pred].mean())
    return float(np.mean(fold_means))


def _run_hpo(X, y, Yb, w_extra, years, pre):
    gkf = GroupKFold(n_splits=INNER_K)
    splits = list(gkf.split(X, y, groups=years))

    def objective(trial):
        p = {
            "C":           trial.suggest_float("C",     0.5, 50, log=True),
            "gamma":       trial.suggest_float("gamma", 1e-3, 1, log=True),
            "wpow":        trial.suggest_float("wpow", 0.5, 2.5),
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
        }
        return _cv_score(X, y, Yb, w_extra, years, p, splits, pre)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def _eval_fold(X, y, Yb, w_extra, years, fold_label, tr, te, pre):
    t0 = time.time()
    best_params, _ = _run_hpo(X[tr], y[tr], Yb[tr], w_extra[tr], years[tr], pre)
    probs = _bag_predict(X[tr], y[tr], Yb[tr], w_extra[tr], X[te], best_params, pre)
    pred = np.argmax(probs, axis=1)
    Y_te = Yb[te]
    return {
        "fold": fold_label,
        "borda": float(Y_te[np.arange(len(te)), pred].sum()),
        "oracle": float(Y_te.max(axis=1).sum()),
        "cpsat": float(Y_te[:, 0].sum()),
        "acc": float((pred == y[te]).mean()),
        "best_params": best_params,
        "wall": time.time() - t0,
    }


def evaluate(name, X, X_rep, conf, Y, y, years, use_rep_feats, use_conf):
    if use_rep_feats:
        X_full = np.hstack([X, X_rep])
    else:
        X_full = X
    w_extra = conf if use_conf else np.ones(len(X))

    folds = leave_one_year_out_folds(years)
    print(f"\n========== {name}  (X shape={X_full.shape}, use_conf={use_conf}) ==========",
          flush=True)
    records = Parallel(n_jobs=15)(
        delayed(_eval_fold)(X_full, y, Y, w_extra, years,
                            fold_label, tr, te, "log_std")
        for fold_label, tr, te in folds
    )
    records.sort(key=lambda r: r["fold"])
    for r in records:
        bp = r["best_params"]
        print(f"    {r['fold']}: borda={r['borda']:>6.2f}  oracle={r['oracle']:>6.2f}  "
              f"acc={r['acc']*100:>5.1f}%  (C={bp['C']:.3g}, g={bp['gamma']:.3g}, "
              f"wpow={bp['wpow']:.2f}, {r['wall']:.0f}s)")
    sb = sum(r["borda"] for r in records); so = sum(r["oracle"] for r in records)
    sc = sum(r["cpsat"] for r in records); n = sum(int(r['fold'] != '') for r in records)
    print(f"  totals: borda={sb:.2f}  oracle={so:.2f}  cpsat={sc:.2f}  ratio={sb/so:.3f}",
          flush=True)


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "cpsat8_k1"
    X, X_rep, conf, Y, y, years = load(stem)
    print(f"Dataset: {stem}  X={X.shape}  X_rep={X_rep.shape}  conf range [{conf.min():.3f}, {conf.max():.3f}]")

    evaluate(f"{stem} baseline (orig X, no conf)",
             X, X_rep, conf, Y, y, years,
             use_rep_feats=False, use_conf=False)

    evaluate(f"{stem} +rep_features (no conf)",
             X, X_rep, conf, Y, y, years,
             use_rep_feats=True, use_conf=False)

    evaluate(f"{stem} +conf_weight only",
             X, X_rep, conf, Y, y, years,
             use_rep_feats=False, use_conf=True)

    evaluate(f"{stem} +rep_features +conf_weight",
             X, X_rep, conf, Y, y, years,
             use_rep_feats=True, use_conf=True)


if __name__ == "__main__":
    main()
