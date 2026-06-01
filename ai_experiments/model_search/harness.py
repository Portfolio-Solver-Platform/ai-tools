"""Nested-CV harness for portfolio-selection model comparison."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold


@dataclass
class ModelSpec:
    name: str
    build: Callable[[dict], BaseEstimator]
    suggest: Callable[[optuna.Trial], dict]
    n_trials: int = 50
    expensive: bool = False


@dataclass
class FoldResult:
    fold_label: str
    n_test: int
    test_borda: float
    oracle: float
    cpsat_baseline: float
    accuracy: float
    best_params: dict
    inner_cv_score: float
    fit_seconds: float


@dataclass
class ModelResult:
    name: str
    dataset: str
    folds: list[FoldResult]
    final_params: dict = field(default_factory=dict)
    final_cv_score: float = 0.0

    @property
    def total_borda(self) -> float:
        return sum(f.test_borda for f in self.folds)

    @property
    def total_oracle(self) -> float:
        return sum(f.oracle for f in self.folds)

    @property
    def total_cpsat(self) -> float:
        return sum(f.cpsat_baseline for f in self.folds)

    @property
    def oracle_ratio(self) -> float:
        return self.total_borda / self.total_oracle if self.total_oracle else float("nan")

    @property
    def weighted_accuracy(self) -> float:
        n = sum(f.n_test for f in self.folds)
        return sum(f.accuracy * f.n_test for f in self.folds) / n if n else float("nan")


def _cv_score(X, y, Y_borda, pipeline_factory, splits) -> float:
    fold_means = []
    for tr, te in splits:
        pipe = pipeline_factory()
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        bordas = Y_borda[te][np.arange(len(te)), pred]
        fold_means.append(bordas.mean())
    return float(np.mean(fold_means))


def _run_hpo(X, y, Y_borda, groups, spec: ModelSpec, n_splits: int) -> tuple[dict, float]:
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))

    def objective(trial: optuna.Trial):
        params = spec.suggest(trial)
        return _cv_score(X, y, Y_borda, lambda p=params: spec.build(p), splits)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=spec.n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _evaluate_one_fold(spec, dataset_name, X, y, Y_borda, years,
                       fold_label, train_idx, test_idx, inner_k, log):
    t0 = time.time()
    train_years = years[train_idx]
    best_params, best_inner = _run_hpo(
        X[train_idx], y[train_idx], Y_borda[train_idx], train_years,
        spec, n_splits=inner_k,
    )

    pipe = spec.build(best_params)
    pipe.fit(X[train_idx], y[train_idx])
    pred = pipe.predict(X[test_idx])
    Y_te = Y_borda[test_idx]

    record = FoldResult(
        fold_label=str(fold_label),
        n_test=len(test_idx),
        test_borda=float(Y_te[np.arange(len(test_idx)), pred].sum()),
        oracle=float(Y_te.max(axis=1).sum()),
        cpsat_baseline=float(Y_te[:, 0].sum()),
        accuracy=float((pred == y[test_idx]).mean()),
        best_params=dict(best_params),
        inner_cv_score=float(best_inner),
        fit_seconds=time.time() - t0,
    )
    if log:
        print(f"    {spec.name:14s} {dataset_name:14s} fold {fold_label}: "
              f"borda={record.test_borda:>6.2f} oracle={record.oracle:>6.2f} "
              f"cpsat={record.cpsat_baseline:>6.2f} acc={record.accuracy*100:>5.1f}% "
              f"({record.fit_seconds:.0f}s)", flush=True)
    return record


def run_nested_cv(X, y, Y_borda, years, spec: ModelSpec,
                  dataset_name: str, inner_k: int = 5, log: bool = True,
                  n_jobs: int = 1, outer_k: int = 5) -> ModelResult:
    # Outer folds use joblib threads to avoid pickling closures; BLAS threads
    # should be pinned to 1 by the caller to avoid oversubscription.
    from utils.cross_solver_eval import year_kfold_folds

    folds = year_kfold_folds(years, n_splits=outer_k)

    fold_records: list[FoldResult] = Parallel(
        n_jobs=n_jobs, prefer="threads"
    )(
        delayed(_evaluate_one_fold)(
            spec, dataset_name, X, y, Y_borda, years,
            label, train_idx, test_idx, inner_k, log,
        )
        for label, train_idx, test_idx in folds
    )
    fold_records.sort(key=lambda r: r.fold_label)

    final_params, final_score = _run_hpo(
        X, y, Y_borda, years, spec, n_splits=inner_k,
    )
    return ModelResult(
        name=spec.name, dataset=dataset_name, folds=fold_records,
        final_params=final_params, final_cv_score=final_score,
    )
