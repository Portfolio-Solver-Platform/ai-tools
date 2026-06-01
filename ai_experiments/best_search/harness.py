"""LOYO nested-CV harness for portfolio-selection model comparison."""
from __future__ import annotations

import os
# Pin BLAS threads BEFORE numpy is imported in any worker.
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.cross_solver_eval import leave_one_year_out_folds  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class Experiment:
    name: str
    build: Callable[[dict], BaseEstimator]
    suggest: Callable[[optuna.Trial], dict]
    n_trials: int = 100
    # Optional custom fit/predict (for regression-with-argmax style models).
    fit_predict: Callable | None = None


@dataclass
class FoldResult:
    fold_label: str
    n_test: int
    test_borda: float
    oracle: float
    cpsat_baseline: float
    accuracy: float
    inner_cv_score: float
    fit_seconds: float
    best_params: dict


def default_fit_predict(pipe, X_tr, y_tr_labels, Y_borda_tr, X_te):
    pipe.fit(X_tr, y_tr_labels)
    return pipe.predict(X_te)


def cv_score(X, y_labels, Y_borda, params, exp: Experiment, splits):
    fold_means = []
    fp = exp.fit_predict or default_fit_predict
    for tr, te in splits:
        pipe = exp.build(params)
        pred = fp(pipe, X[tr], y_labels[tr], Y_borda[tr], X[te])
        bordas = Y_borda[te][np.arange(len(te)), pred]
        fold_means.append(bordas.mean())
    return float(np.mean(fold_means))


def run_hpo(X, y_labels, Y_borda, groups, exp: Experiment, inner_k: int, seed: int = 42):
    gkf = GroupKFold(n_splits=inner_k)
    splits = list(gkf.split(X, y_labels, groups=groups))

    def objective(trial):
        params = exp.suggest(trial)
        return cv_score(X, y_labels, Y_borda, params, exp, splits)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=exp.n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def _evaluate_outer_fold(exp: Experiment, X, y_labels, Y_borda, years,
                         fold_label, train_idx, test_idx, inner_k):
    t0 = time.time()
    best_params, best_inner = run_hpo(
        X[train_idx], y_labels[train_idx], Y_borda[train_idx],
        years[train_idx], exp, inner_k=inner_k,
    )

    pipe = exp.build(best_params)
    fp = exp.fit_predict or default_fit_predict
    pred = fp(pipe, X[train_idx], y_labels[train_idx], Y_borda[train_idx], X[test_idx])

    Y_te = Y_borda[test_idx]
    return FoldResult(
        fold_label=str(fold_label),
        n_test=len(test_idx),
        test_borda=float(Y_te[np.arange(len(test_idx)), pred].sum()),
        oracle=float(Y_te.max(axis=1).sum()),
        cpsat_baseline=float(Y_te[:, 0].sum()),
        accuracy=float((pred == y_labels[test_idx]).mean()),
        inner_cv_score=float(best_inner),
        fit_seconds=time.time() - t0,
        best_params=dict(best_params),
    )


def run_experiment(exp: Experiment, X, y_labels, Y_borda, years, *,
                   inner_k: int = 5, n_jobs: int = 15) -> list[FoldResult]:
    folds = leave_one_year_out_folds(years)
    fold_records: list[FoldResult] = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_outer_fold)(
            exp, X, y_labels, Y_borda, years,
            fold_label, train_idx, test_idx, inner_k,
        )
        for fold_label, train_idx, test_idx in folds
    )
    fold_records.sort(key=lambda r: r.fold_label)
    return fold_records


@dataclass
class Summary:
    name: str
    total_borda: float
    total_oracle: float
    total_cpsat: float
    oracle_ratio: float
    weighted_accuracy: float
    n_total: int
    folds: list[FoldResult] = field(default_factory=list)

    @classmethod
    def from_folds(cls, name: str, folds: list[FoldResult]) -> "Summary":
        n = sum(f.n_test for f in folds)
        sb = sum(f.test_borda for f in folds)
        so = sum(f.oracle for f in folds)
        sc = sum(f.cpsat_baseline for f in folds)
        acc = sum(f.accuracy * f.n_test for f in folds) / n
        return cls(
            name=name, total_borda=sb, total_oracle=so, total_cpsat=sc,
            oracle_ratio=sb / so if so else float("nan"),
            weighted_accuracy=acc, n_total=n, folds=folds,
        )

    def line(self) -> str:
        return (f"  totals: borda={self.total_borda:.2f}  "
                f"oracle={self.total_oracle:.2f}  "
                f"cpsat={self.total_cpsat:.2f}  "
                f"ratio={self.oracle_ratio:.3f}  "
                f"acc={self.weighted_accuracy*100:.1f}%  "
                f"({self.n_total} test instances)")


def append_fold_rows(path: Path, experiment_name: str, folds: list[FoldResult]):
    fields = [
        "experiment", "fold_label", "n_test",
        "test_borda", "oracle", "cpsat_baseline",
        "accuracy", "inner_cv_score", "fit_seconds", "best_params",
    ]
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for fold in folds:
            w.writerow({
                "experiment":     experiment_name,
                "fold_label":     fold.fold_label,
                "n_test":         fold.n_test,
                "test_borda":     fold.test_borda,
                "oracle":         fold.oracle,
                "cpsat_baseline": fold.cpsat_baseline,
                "accuracy":       fold.accuracy,
                "inner_cv_score": fold.inner_cv_score,
                "fit_seconds":    fold.fit_seconds,
                "best_params":    json.dumps(fold.best_params, default=str),
            })


def append_summary_row(path: Path, summary: Summary, extras: dict | None = None):
    fields = ["experiment", "total_borda", "total_oracle", "total_cpsat",
              "oracle_ratio", "accuracy", "n_total", "extras"]
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({
            "experiment":   summary.name,
            "total_borda":  summary.total_borda,
            "total_oracle": summary.total_oracle,
            "total_cpsat":  summary.total_cpsat,
            "oracle_ratio": summary.oracle_ratio,
            "accuracy":     summary.weighted_accuracy,
            "n_total":      summary.n_total,
            "extras":       json.dumps(extras or {}, default=str),
        })
