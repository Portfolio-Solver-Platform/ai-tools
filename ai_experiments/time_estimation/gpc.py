"""Multi-seed GPC evaluation on raw runtime (with timeout objective tiebreak).
For each seed:
  - outer GroupShuffleSplit train/test
  - inner GroupKFold(5) CV proba on train
  - threshold grid sweep on CV proba — minimize total CV time
  - refit GPC on full train, predict_proba on test, apply locked threshold
  - score test on time AND Borda
Reports per-seed numbers and mean ± std. No model saving.
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import get_cpsat8_k1_ek1_data, get_cpsat8_k1_data, prepare_labels
from ai_experiments.time_estimation.load_times import build_Y_times_with_tiebreak

SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5
TIMEOUT_MS = 1_200_000
THRESHOLDS = np.arange(0.0, 1.001, 0.01)

X, _, meta = get_cpsat8_k1_ek1_data()
groups_all = meta["problem"]
all_idx = np.arange(len(X))

PORTFOLIOS = ('cpsat8', 'k1-8c-8s-v1')
Y_times, n_missing, n_tiebroken, n_genuine_tie = build_Y_times_with_tiebreak(meta, portfolios=PORTFOLIOS, timeout_ms=TIMEOUT_MS)
y_labels = np.argmin(Y_times, axis=1)
print(f"timeout-tiebreak applied to {n_tiebroken} rows; {n_genuine_tie} rows left as genuine ties")
print(f"label balance: {np.bincount(y_labels)}  (cpsat8 fastest, k1 fastest)")

_, Y_local_raw, _ = get_cpsat8_k1_data()
_, Y_borda = prepare_labels(Y_local_raw)

print(f"\n========== dataset: cpsat8_k1 (time-based, GPC) ==========")
seed_results = []

for seed in SEEDS:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(all_idx, groups=groups_all))
    train_groups = groups_all[train_idx]
    Y_train_times = Y_times[train_idx]
    Y_test_times = Y_times[test_idx]
    Y_test_borda = Y_borda[test_idx]
    train_rows = np.arange(len(train_idx))
    test_rows = np.arange(len(test_idx))

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
    gpc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=1, random_state=42, n_jobs=1))
    ])

    cv = GroupKFold(n_splits=N_FOLDS)
    cv_proba = cross_val_predict(gpc, X[train_idx], y_labels[train_idx],
                                  method="predict_proba", cv=cv, groups=train_groups, n_jobs=28)
    cv_argmax = np.argmax(cv_proba, axis=1)
    cv_max_proba = cv_proba.max(axis=1)

    best_t, best_cv_time = None, np.inf
    for t in THRESHOLDS:
        pred = cv_argmax.copy()
        pred[cv_max_proba < t] = 0
        total = np.sum(Y_train_times[train_rows, pred])
        if total < best_cv_time:
            best_cv_time = total
            best_t = t

    gpc.fit(X[train_idx], y_labels[train_idx])
    test_proba = gpc.predict_proba(X[test_idx])
    test_pred = np.argmax(test_proba, axis=1)
    test_pred[test_proba.max(axis=1) < best_t] = 0

    test_time = np.sum(Y_test_times[test_rows, test_pred])
    test_oracle_time = np.sum(np.min(Y_test_times, axis=1))
    test_cpsat_time = np.sum(Y_test_times[:, 0])
    headroom = (test_cpsat_time - test_time) / (test_cpsat_time - test_oracle_time) if test_cpsat_time > test_oracle_time else 0.0

    test_borda = np.sum(Y_test_borda[test_rows, test_pred])
    test_borda_oracle = np.sum(np.max(Y_test_borda, axis=1))
    test_borda_cpsat = np.sum(Y_test_borda[:, 0])
    borda_headroom = (test_borda - test_borda_cpsat) / (test_borda_oracle - test_borda_cpsat) if test_borda_oracle > test_borda_cpsat else 0.0

    y_test = y_labels[test_idx]
    test_acc = (test_pred == y_test).mean()
    test_majority = max((y_test == c).mean() for c in range(Y_times.shape[1]))

    print(f"  seed={seed}  t={best_t:.2f}  test_time={test_time:.0f}  cpsat_time={test_cpsat_time:.0f}  time_headroom={headroom*100:.1f}%  ||  test_borda={test_borda:.2f}  cpsat_borda={test_borda_cpsat:.2f}  borda_headroom={borda_headroom*100:.1f}%  acc={test_acc*100:.1f}%")

    seed_results.append({
        'seed': seed,
        'threshold': best_t,
        'test_time': test_time,
        'test_oracle_time': test_oracle_time,
        'test_cpsat_time': test_cpsat_time,
        'time_headroom': headroom,
        'test_borda': test_borda,
        'test_borda_oracle': test_borda_oracle,
        'test_borda_cpsat': test_borda_cpsat,
        'borda_headroom': borda_headroom,
        'test_acc': test_acc,
        'test_majority': test_majority,
    })

print(f"\n  === aggregated over {len(SEEDS)} seeds ===")
times = np.array([r['test_time'] for r in seed_results])
cpsat_times = np.array([r['test_cpsat_time'] for r in seed_results])
oracle_times = np.array([r['test_oracle_time'] for r in seed_results])
time_headrooms = np.array([r['time_headroom'] for r in seed_results])
bordas = np.array([r['test_borda'] for r in seed_results])
cpsat_bordas = np.array([r['test_borda_cpsat'] for r in seed_results])
oracle_bordas = np.array([r['test_borda_oracle'] for r in seed_results])
borda_headrooms = np.array([r['borda_headroom'] for r in seed_results])
accs = np.array([r['test_acc'] for r in seed_results])
thresholds_used = np.array([r['threshold'] for r in seed_results])
print(f"  TIME-BASED metrics:")
print(f"    test time:           mean={times.mean():.0f}  std={times.std():.0f}  values={times.round().astype(int).tolist()}")
print(f"    cpsat baseline time: mean={cpsat_times.mean():.0f}  std={cpsat_times.std():.0f}")
print(f"    oracle time:         mean={oracle_times.mean():.0f}  std={oracle_times.std():.0f}")
print(f"    time headroom:       mean={time_headrooms.mean()*100:.1f}%  std={time_headrooms.std()*100:.1f}%  values={[f'{h*100:.1f}%' for h in time_headrooms]}")
print(f"  BORDA metrics (same predictions, scored on local cpsat8_k1 Borda):")
print(f"    test borda:          mean={bordas.mean():.2f}  std={bordas.std():.2f}  values={bordas.round(2).tolist()}")
print(f"    cpsat baseline borda:mean={cpsat_bordas.mean():.2f}  std={cpsat_bordas.std():.2f}")
print(f"    oracle borda:        mean={oracle_bordas.mean():.2f}  std={oracle_bordas.std():.2f}")
print(f"    borda headroom:      mean={borda_headrooms.mean()*100:.1f}%  std={borda_headrooms.std()*100:.1f}%  values={[f'{h*100:.1f}%' for h in borda_headrooms]}")
print(f"  test accuracy:         mean={accs.mean()*100:.1f}%  std={accs.std()*100:.1f}%  values={[f'{a*100:.1f}%' for a in accs]}")
print(f"  chosen thresholds:     mean={thresholds_used.mean():.2f}  values={thresholds_used.round(2).tolist()}")
