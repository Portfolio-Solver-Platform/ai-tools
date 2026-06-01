"""Multi-seed GPC + threshold-fallback eval. Per seed: GroupShuffleSplit
train/test, inner GroupKFold(5) CV proba on train, grid threshold sweep,
refit on full train, eval on test with locked threshold. Local Borda."""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels

SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5
THRESHOLDS = np.arange(0.0, 1.001, 0.01)

_, _, meta_canon = get_cpsat8_k1_ek1_data()
groups_all = meta_canon["problem"]
all_idx = np.arange(len(groups_all))

datasets = [
    ('cpsat8_k1', get_cpsat8_k1_data),
]

for name, getter in datasets:
    print(f"\n========== dataset: {name} ==========")
    X, Y, _ = getter()
    y_labels, Y_borda = prepare_labels(Y)

    seed_results = []

    for seed in SEEDS:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(all_idx, groups=groups_all))
        train_groups = groups_all[train_idx]
        Y_train_borda = Y_borda[train_idx]
        Y_test_borda_local = Y_borda[test_idx]
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

        best_t, best_cv_borda = None, -np.inf
        for t in THRESHOLDS:
            pred = cv_argmax.copy()
            pred[cv_max_proba < t] = 0
            borda = np.sum(Y_train_borda[train_rows, pred])
            if borda > best_cv_borda:
                best_cv_borda = borda
                best_t = t

        gpc.fit(X[train_idx], y_labels[train_idx])
        test_proba = gpc.predict_proba(X[test_idx])
        test_pred = np.argmax(test_proba, axis=1)
        test_pred[test_proba.max(axis=1) < best_t] = 0
        test_borda = np.sum(Y_test_borda_local[test_rows, test_pred])

        test_oracle = np.sum(np.max(Y_test_borda_local, axis=1))
        test_cpsat = np.sum(Y_test_borda_local[:, 0])
        y_test_local = y_labels[test_idx]
        test_acc = (test_pred == y_test_local).mean()
        test_majority = max((y_test_local == c).mean() for c in range(Y.shape[1]))
        headroom = (test_borda - test_cpsat) / (test_oracle - test_cpsat) if test_oracle > test_cpsat else 0.0

        print(f"  seed={seed}  cv_borda={best_cv_borda:.2f}  t={best_t:.2f}  test_borda={test_borda:.2f}  cpsat={test_cpsat:.2f}  oracle={test_oracle:.2f}  acc={test_acc*100:.1f}%  majority={test_majority*100:.1f}%  headroom={headroom*100:.1f}%")

        seed_results.append({
            'seed': seed,
            'cv_borda': best_cv_borda,
            'threshold': best_t,
            'test_borda': test_borda,
            'test_oracle': test_oracle,
            'test_cpsat': test_cpsat,
            'test_acc': test_acc,
            'test_majority': test_majority,
            'headroom': headroom,
        })

    print(f"\n  === aggregated over {len(SEEDS)} seeds ===")
    bordas = np.array([r['test_borda'] for r in seed_results])
    accs = np.array([r['test_acc'] for r in seed_results])
    headrooms = np.array([r['headroom'] for r in seed_results])
    cpsats = np.array([r['test_cpsat'] for r in seed_results])
    thresholds_used = np.array([r['threshold'] for r in seed_results])
    print(f"  test borda:        mean={bordas.mean():.2f}  std={bordas.std():.2f}  values={bordas.round(2).tolist()}")
    print(f"  cpsat baselines:   mean={cpsats.mean():.2f}  std={cpsats.std():.2f}  values={cpsats.round(2).tolist()}")
    print(f"  test accuracy:     mean={accs.mean()*100:.1f}%  std={accs.std()*100:.1f}%  values={[f'{a*100:.1f}%' for a in accs]}")
    print(f"  headroom captured: mean={headrooms.mean()*100:.1f}%  std={headrooms.std()*100:.1f}%  values={[f'{h*100:.1f}%' for h in headrooms]}")
    print(f"  chosen thresholds: mean={thresholds_used.mean():.2f}  values={thresholds_used.round(2).tolist()}")
