"""Multi-seed XGBoost evaluation on raw runtime (with timeout objective tiebreak).
For each seed: outer GroupShuffleSplit train/test → inner GroupKFold(5) Optuna selection
(minimize CV total time) → refit on full train → evaluate on test on time AND Borda.
Reports per-seed numbers and mean ± std. No model saving. cpsat8 vs k1 (2-way).
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
import xgboost as xgb
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import get_cpsat8_k1_ek1_data, get_cpsat8_k1_data, prepare_labels
from ai_experiments.time_estimation.load_times import build_Y_times_with_tiebreak

SEEDS = [42, 43, 44, 45, 46]
N_TRIALS = 100
N_FOLDS = 5
TIMEOUT_MS = 1_200_000

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

print(f"\n========== dataset: cpsat8_k1 (time-based, XGBoost) ==========")
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

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 300),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'tree_method':      'hist',
            'n_jobs':           1,
            'random_state':     42,
        }
        model = xgb.XGBClassifier(**params)
        cv = GroupKFold(n_splits=N_FOLDS)
        preds = cross_val_predict(model, X[train_idx], y_labels[train_idx],
                                   method="predict", cv=cv, groups=train_groups, n_jobs=28)
        return np.sum(Y_train_times[train_rows, preds])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = dict(study.best_params)
    model = xgb.XGBClassifier(**best_params, tree_method='hist', n_jobs=28, random_state=42)
    model.fit(X[train_idx], y_labels[train_idx])
    preds = model.predict(X[test_idx])

    test_time = np.sum(Y_test_times[test_rows, preds])
    test_oracle_time = np.sum(np.min(Y_test_times, axis=1))
    test_cpsat_time = np.sum(Y_test_times[:, 0])
    headroom = (test_cpsat_time - test_time) / (test_cpsat_time - test_oracle_time) if test_cpsat_time > test_oracle_time else 0.0

    test_borda = np.sum(Y_test_borda[test_rows, preds])
    test_borda_oracle = np.sum(np.max(Y_test_borda, axis=1))
    test_borda_cpsat = np.sum(Y_test_borda[:, 0])
    borda_headroom = (test_borda - test_borda_cpsat) / (test_borda_oracle - test_borda_cpsat) if test_borda_oracle > test_borda_cpsat else 0.0

    y_test = y_labels[test_idx]
    test_acc = (preds == y_test).mean()
    test_majority = max((y_test == c).mean() for c in range(Y_times.shape[1]))

    print(f"  seed={seed}  test_time={test_time:.0f}  cpsat_time={test_cpsat_time:.0f}  time_headroom={headroom*100:.1f}%  ||  test_borda={test_borda:.2f}  cpsat_borda={test_borda_cpsat:.2f}  borda_headroom={borda_headroom*100:.1f}%  acc={test_acc*100:.1f}%  params={best_params}")

    seed_results.append({
        'seed': seed,
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
