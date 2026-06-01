"""
Search across ML methods for the portfolio-selection task. Runs nested CV
on each (model, dataset) pair, saves per-fold metrics, prints a comparison
table, and saves the winning model per dataset.

Usage:
    python -m ai_experiments.model_search.run_search           # all models
    python -m ai_experiments.model_search.run_search SVM-RBF   # one model
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import csv
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import (
    get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels,
)
from ai_experiments.model_search.harness import run_nested_cv
from ai_experiments.model_search.models import ALL_SPECS

DATASETS = [
    # ("cpsat8_k1_ek1", get_cpsat8_k1_ek1_data),
    ("cpsat8_k1",     get_cpsat8_k1_data),
    # ("cpsat8_ek1",    get_cpsat8_ek1_data),
]

RESULTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_JOBS = 10


def write_results_csv(rows, path: Path):
    fields = [
        "model", "dataset", "fold_label", "n_test",
        "test_borda", "oracle", "cpsat_baseline",
        "accuracy", "inner_cv_score", "fit_seconds", "best_params",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    selected = set(sys.argv[1:]) or None
    specs = [s for s in ALL_SPECS if selected is None or s.name in selected]
    if not specs:
        print(f"no specs matched {sys.argv[1:]}; available: "
              f"{[s.name for s in ALL_SPECS]}")
        sys.exit(1)

    all_rows = []
    summary = []

    for dataset_name, getter in DATASETS:
        print(f"\n========== dataset: {dataset_name} ==========")
        X, Y, meta = getter()
        y_labels, Y_borda = prepare_labels(Y)
        years = meta["year"]

        for spec in specs:
            print(f"\n--- {spec.name} on {dataset_name} (n_trials={spec.n_trials}) ---")
            t0 = time.time()
            result = run_nested_cv(
                X, y_labels, Y_borda, years, spec, dataset_name,
                inner_k=5, n_jobs=N_JOBS,
            )
            dt = time.time() - t0
            print(f"  totals  borda={result.total_borda:8.2f}  "
                  f"oracle={result.total_oracle:8.2f}  "
                  f"cpsat={result.total_cpsat:8.2f}  "
                  f"ratio={result.oracle_ratio:.3f}  "
                  f"acc={result.weighted_accuracy*100:.1f}%  "
                  f"({dt:.0f}s wall)")

            for f in result.folds:
                all_rows.append({
                    "model":          spec.name,
                    "dataset":        dataset_name,
                    "fold_label":     f.fold_label,
                    "n_test":         f.n_test,
                    "test_borda":     f.test_borda,
                    "oracle":         f.oracle,
                    "cpsat_baseline": f.cpsat_baseline,
                    "accuracy":       f.accuracy,
                    "inner_cv_score": f.inner_cv_score,
                    "fit_seconds":    f.fit_seconds,
                    "best_params":    str(f.best_params),
                })
            summary.append((
                spec.name, dataset_name,
                result.total_borda, result.oracle_ratio,
                result.weighted_accuracy, result.final_params,
            ))

    write_results_csv(all_rows, RESULTS_DIR / "results.csv")
    print(f"\nwrote {RESULTS_DIR / 'results.csv'} ({len(all_rows)} rows)")

    print("\n========== Per-dataset ranking (by total test Borda) ==========")
    winners: dict[str, tuple] = {}
    by_dataset: dict[str, list] = {}
    for entry in summary:
        by_dataset.setdefault(entry[1], []).append(entry)
    for dataset_name, entries in by_dataset.items():
        entries.sort(key=lambda r: -r[2])
        print(f"\n  {dataset_name}")
        print(f"    {'model':<14}  {'borda':>8}  {'ratio':>6}  {'acc':>6}")
        for name, _, borda, ratio, acc, _ in entries:
            print(f"    {name:<14}  {borda:>8.2f}  {ratio:>6.3f}  {acc*100:>5.1f}%")
        winners[dataset_name] = entries[0]

    print("\n========== Saving winning model per dataset ==========")
    for dataset_name, (name, _, _, _, _, params) in winners.items():
        getter = dict(DATASETS)[dataset_name]
        X, Y, _ = getter()
        y_labels, _ = prepare_labels(Y)
        spec = next(s for s in ALL_SPECS if s.name == name)
        pipe = spec.build(params)
        pipe.fit(X, y_labels)
        out = MODELS_DIR / f"best_{dataset_name}_{name}.joblib"
        joblib.dump(pipe, out)
        print(f"  {dataset_name:18s}  winner={name:<14}  saved -> {out}")


if __name__ == "__main__":
    main()
