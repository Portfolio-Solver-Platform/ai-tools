"""Round 2 on cpsat8_ek1: rerun the close competitors from k1 to compare."""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.shared_data import get_cpsat8_ek1_data, prepare_labels  # noqa: E402

from ai_experiments.best_search import experiments as E  # noqa: E402
from ai_experiments.best_search.harness import (  # noqa: E402
    Summary, append_fold_rows, append_summary_row, run_experiment,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FOLDS_CSV = RESULTS_DIR / "folds_ek1.csv"
SUMMARY_CSV = RESULTS_DIR / "summary_ek1.csv"


def main():
    X, Y, meta = get_cpsat8_ek1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    runs = [
        ("bag_svm_mw_log_quantile", E.bagged_svm_mw("log_quantile", n_estimators=10, n_trials=60)),
        ("bag_svm_mw_log_power",    E.bagged_svm_mw("log_power",    n_estimators=10, n_trials=60)),
        ("bag_svm_mw_std",          E.bagged_svm_mw("std",          n_estimators=10, n_trials=60)),
        ("bag_svm_mw_log_std_x20",  E.bagged_svm_mw("log_std",      n_estimators=20, n_trials=100)),
        ("dec_theor_log_std",       E.decision_theoretic_exp("log_std", "log_std", n_trials=60, n_estimators=10)),
        ("calib_bag_log_std",       E.calibrated_bag_svm_mw("log_std", n_estimators=10, n_trials=60)),
        ("xgb_quantile",            E.xgb_regret("quantile",   n_trials=80)),
        ("mo_svr_log_std",          E.mo_svr_borda("log_std",  n_trials=60)),
    ]

    for key, exp in runs:
        print(f"\n========== {exp.name}  (n_trials={exp.n_trials}) ==========",
              flush=True)
        t0 = time.time()
        folds = run_experiment(exp, X, y_labels, Y_borda, years, inner_k=5, n_jobs=15)
        dt = time.time() - t0
        for f in folds:
            print(f"    {f.fold_label}: borda={f.test_borda:>6.2f}  "
                  f"oracle={f.oracle:>6.2f}  acc={f.accuracy*100:>5.1f}%  "
                  f"({f.fit_seconds:.0f}s)", flush=True)
        summary = Summary.from_folds(exp.name, folds)
        print(summary.line() + f"   wall={dt:.0f}s", flush=True)
        append_fold_rows(FOLDS_CSV, exp.name, folds)
        append_summary_row(SUMMARY_CSV, summary, {"key": key, "n_trials": exp.n_trials})


if __name__ == "__main__":
    main()
