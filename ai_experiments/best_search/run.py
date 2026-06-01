"""Run an experiment by name. Appends per-fold rows and a summary row.

Usage:
    python -m ai_experiments.best_search.run <experiment_key>...
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.shared_data import get_cpsat8_k1_data, prepare_labels  # noqa: E402

from ai_experiments.best_search import experiments as E  # noqa: E402
from ai_experiments.best_search.harness import (  # noqa: E402
    Summary, append_fold_rows, append_summary_row, run_experiment,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FOLDS_CSV = RESULTS_DIR / "folds.csv"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


N_TRIALS_DEFAULT = 100
N_TRIALS_FAST = 60

EXPERIMENTS = {
    # SVM-RBF preprocessing sweep
    "svm_std":          lambda: E.svm_rbf("std",          n_trials=N_TRIALS_DEFAULT),
    "svm_robust":       lambda: E.svm_rbf("robust",       n_trials=N_TRIALS_DEFAULT),
    "svm_quantile":     lambda: E.svm_rbf("quantile",     n_trials=N_TRIALS_DEFAULT),
    "svm_power":        lambda: E.svm_rbf("power",        n_trials=N_TRIALS_DEFAULT),
    "svm_log_std":      lambda: E.svm_rbf("log_std",      n_trials=N_TRIALS_DEFAULT),
    "svm_log_quantile": lambda: E.svm_rbf("log_quantile", n_trials=N_TRIALS_DEFAULT),
    "svm_log_robust":   lambda: E.svm_rbf("log_robust",   n_trials=N_TRIALS_DEFAULT),
    "svm_log_power":    lambda: E.svm_rbf("log_power",    n_trials=N_TRIALS_DEFAULT),

    # SVM-RBF with regret-margin sample weights
    "svm_mw_log_std":      lambda: E.svm_rbf_sample_weighted("log_std",      n_trials=N_TRIALS_DEFAULT),
    "svm_mw_log_quantile": lambda: E.svm_rbf_sample_weighted("log_quantile", n_trials=N_TRIALS_DEFAULT),
    "svm_mw_quantile":     lambda: E.svm_rbf_sample_weighted("quantile",     n_trials=N_TRIALS_DEFAULT),

    # SVM-RBF with threshold fallback to cpsat
    "svm_t_log_std":      lambda: E.svm_rbf_threshold("log_std",      n_trials=N_TRIALS_FAST),
    "svm_t_log_quantile": lambda: E.svm_rbf_threshold("log_quantile", n_trials=N_TRIALS_FAST),

    # multi-output Borda regression
    "mo_svr_log_std":      lambda: E.mo_svr_borda("log_std",      n_trials=N_TRIALS_FAST),
    "mo_svr_log_quantile": lambda: E.mo_svr_borda("log_quantile", n_trials=N_TRIALS_FAST),
    "mo_svr_quantile":     lambda: E.mo_svr_borda("quantile",     n_trials=N_TRIALS_FAST),
    "mo_xgb_std":          lambda: E.mo_xgb_borda("std",          n_trials=N_TRIALS_DEFAULT),
    "mo_xgb_quantile":     lambda: E.mo_xgb_borda("quantile",     n_trials=N_TRIALS_DEFAULT),

    # regret-weighted gradient boosting
    "xgb_std":          lambda: E.xgb_regret("std",          n_trials=N_TRIALS_DEFAULT),
    "xgb_quantile":     lambda: E.xgb_regret("quantile",     n_trials=N_TRIALS_DEFAULT),
    "xgb_log_quantile": lambda: E.xgb_regret("log_quantile", n_trials=N_TRIALS_DEFAULT),
    "lgb_std":          lambda: E.lgb_regret("std",          n_trials=N_TRIALS_DEFAULT),
    "lgb_quantile":     lambda: E.lgb_regret("quantile",     n_trials=N_TRIALS_DEFAULT),

    # regret-weighted ExtraTrees
    "et_std":          lambda: E.et_regret("std",          n_trials=N_TRIALS_FAST),
    "et_quantile":     lambda: E.et_regret("quantile",     n_trials=N_TRIALS_FAST),

    # bagged regret-weighted SVM ensemble
    "bag_svm_mw_std":          lambda: E.bagged_svm_mw("std",          n_estimators=10, n_trials=60),
    "bag_svm_mw_quantile":     lambda: E.bagged_svm_mw("quantile",     n_estimators=10, n_trials=60),
    "bag_svm_mw_log_quantile": lambda: E.bagged_svm_mw("log_quantile", n_estimators=10, n_trials=60),
    "bag_svm_mw_log_std":      lambda: E.bagged_svm_mw("log_std",      n_estimators=10, n_trials=60),
    "bag_svm_mw_log_std_x20":  lambda: E.bagged_svm_mw("log_std",      n_estimators=20, n_trials=100),
    "bag_svm_mw_log_std_x40":  lambda: E.bagged_svm_mw("log_std",      n_estimators=40, n_trials=100),
    # n_est=10 for HPO speed; n_est=30 used at deploy
    "bag_svm_mw_log_std_t300": lambda: E.bagged_svm_mw("log_std",      n_estimators=10, n_trials=300),
    "bag_svm_mw_log_power":    lambda: E.bagged_svm_mw("log_power",    n_estimators=10, n_trials=60),
    "bag_svm_mw_log_robust":   lambda: E.bagged_svm_mw("log_robust",   n_estimators=10, n_trials=60),

    # diverse-preprocessing bagged SVM
    "div_bag_3pre":            lambda: E.diverse_bagged_svm_mw(("log_std", "quantile", "log_quantile"), n_each=5, n_trials=80),
    "div_bag_logstd_quantile": lambda: E.diverse_bagged_svm_mw(("log_std", "quantile"),                 n_each=8, n_trials=80),
    "div_bag_logstd_logpower": lambda: E.diverse_bagged_svm_mw(("log_std", "log_power"),                n_each=8, n_trials=60),

    "catboost_std":            lambda: E.catboost_exp("std",       n_trials=80),
    "catboost_quantile":       lambda: E.catboost_exp("quantile",  n_trials=80),
    "catboost_log_std":        lambda: E.catboost_exp("log_std",   n_trials=80),
    "tabpfn_std":              lambda: E.tabpfn_exp("std",         n_trials=8),
    "tabpfn_quantile":         lambda: E.tabpfn_exp("quantile",    n_trials=8),
    "tabpfn_log_std":          lambda: E.tabpfn_exp("log_std",     n_trials=8),
    "gpc_log_std":             lambda: E.gpc_exp("log_std",        n_trials=30),
    "gpc_quantile":            lambda: E.gpc_exp("quantile",       n_trials=30),

    "calib_bag_log_std":       lambda: E.calibrated_bag_svm_mw("log_std",  n_estimators=10, n_trials=60),
    "calib_bag_quantile":      lambda: E.calibrated_bag_svm_mw("quantile", n_estimators=10, n_trials=60),
    "dec_theor_log_std":       lambda: E.decision_theoretic_exp("log_std", "log_std", n_trials=60, n_estimators=10),
    "dec_theor_mixed":         lambda: E.decision_theoretic_exp("log_std", "quantile", n_trials=60, n_estimators=10),

    "regret_mlp_log_std":      lambda: E.regret_mlp_exp("log_std",  n_trials=40, n_estimators=5),
    "regret_mlp_quantile":     lambda: E.regret_mlp_exp("quantile", n_trials=40, n_estimators=5),
    "regret_mlp_log_quantile": lambda: E.regret_mlp_exp("log_quantile", n_trials=40, n_estimators=5),

    "outlier_combo_log_std":   lambda: E.outlier_combo_exp("log_std",  n_trials=60, n_estimators=10),
    "outlier_combo_quantile":  lambda: E.outlier_combo_exp("quantile", n_trials=60, n_estimators=10),
    "pairwise_rank_log_std":   lambda: E.pairwise_rank_exp("log_std",  n_trials=60, n_estimators=10),
    "pairwise_rank_quantile":  lambda: E.pairwise_rank_exp("quantile", n_trials=60, n_estimators=10),

    "fe_bag_log_std":          lambda: E.fe_bag_svm_mw("log_std",  n_estimators=10, n_trials=60, k_top=15),
    "fe_bag_quantile":         lambda: E.fe_bag_svm_mw("quantile", n_estimators=10, n_trials=60, k_top=15),

    "moe_log_std_k3":          lambda: E.mixture_experts_exp("log_std", n_trials=50, n_estimators=8, n_clusters=3),
    "moe_log_std_k5":          lambda: E.mixture_experts_exp("log_std", n_trials=50, n_estimators=8, n_clusters=5),

    "rand_sub_log_std":        lambda: E.random_subspace_svm_mw("log_std",  n_estimators=30, n_trials=60),
    "rand_sub_quantile":       lambda: E.random_subspace_svm_mw("quantile", n_estimators=30, n_trials=60),
    "rand_sub_log_std_n50":    lambda: E.random_subspace_svm_mw("log_std",  n_estimators=50, n_trials=60),

    "hp_div_bag_log_std":      lambda: E.hp_diverse_bag_svm_mw("log_std",  n_estimators=50, n_trials=60),
    "hp_div_bag_quantile":     lambda: E.hp_diverse_bag_svm_mw("quantile", n_estimators=50, n_trials=60),

    "bag_svm_mw_asinh_std":      lambda: E.bagged_svm_mw("asinh_std",      n_estimators=10, n_trials=60),
    "bag_svm_mw_asinh_quantile": lambda: E.bagged_svm_mw("asinh_quantile", n_estimators=10, n_trials=60),
    "bag_svm_mw_rank_normal":    lambda: E.bagged_svm_mw("rank_normal",    n_estimators=10, n_trials=60),
    "bag_svm_mw_rank_std":       lambda: E.bagged_svm_mw("rank_std",       n_estimators=10, n_trials=60),

    "knn_log_quantile": lambda: E.knn_exp("log_quantile", n_trials=50),
    "knn_quantile":     lambda: E.knn_exp("quantile",     n_trials=50),
    "logreg_log_std":   lambda: E.logreg_exp("log_std",   n_trials=40),
}


def main(keys: list[str]):
    X, Y, meta = get_cpsat8_k1_data()
    y_labels, Y_borda = prepare_labels(Y)
    years = meta["year"]

    for key in keys:
        if key not in EXPERIMENTS:
            print(f"unknown experiment: {key}; available: {sorted(EXPERIMENTS)}")
            sys.exit(2)
        exp = EXPERIMENTS[key]()
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
    if len(sys.argv) < 2:
        print(f"usage: python -m ai_experiments.best_search.run <key>...\n"
              f"available: {sorted(EXPERIMENTS)}")
        sys.exit(1)
    main(sys.argv[1:])
