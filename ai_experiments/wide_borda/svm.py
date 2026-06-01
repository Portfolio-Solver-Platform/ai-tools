"""SVC trained on per-instance Borda computed against open-category solvers (years 2023-2025),
tested on years 2011-2022 from the existing npz training data (3-solver pairwise Borda).

Train Y is wider (each portfolio's Borda vs ~16 opponents), so the labels (argmax) reflect
"which portfolio is most globally competitive" rather than "which beats the other 1 or 2".
The model still produces a 3-way classifier (cpsat / k1 / ek1).
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.shared_data import get_cpsat8_k1_ek1_data, prepare_labels
from ai_experiments.wide_borda.load_wide_borda import load_wide_borda

N_TRIALS = 100
N_FOLDS = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)

X_train, Y_train_wide, meta_train = load_wide_borda()
y_train_labels = np.argmax(Y_train_wide, axis=1)
groups_train = np.array([m[1] for m in meta_train])  # problem field
print(f"\nTrain X shape: {X_train.shape}  Y_train_wide shape: {Y_train_wide.shape}")
print(f"Train label balance [cpsat, k1, ek1]: {np.bincount(y_train_labels, minlength=3)}")
print(f"Train wide-Borda mean per portfolio: {Y_train_wide.mean(axis=0).round(3)}")

X, Y, meta_npz = get_cpsat8_k1_ek1_data()
y_labels_npz, Y_borda_npz = prepare_labels(Y)
test_mask = (meta_npz['year'] == 2023)
test_idx = np.where(test_mask)[0]
X_test = X[test_idx]
Y_test_borda = Y_borda_npz[test_idx]
y_test_labels = y_labels_npz[test_idx]
print(f"\nTest set (2023): {len(test_idx)} instances")
print(f"Test cpsat baseline: {np.sum(Y_test_borda[:, 0]):.2f}")
print(f"Test k1    baseline: {np.sum(Y_test_borda[:, 1]):.2f}")
print(f"Test ek1   baseline: {np.sum(Y_test_borda[:, 2]):.2f}")
print(f"Test oracle:         {np.sum(np.max(Y_test_borda, axis=1)):.2f}")

train_rows = np.arange(len(Y_train_wide))

def objective(trial):
    gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
    c_value = trial.suggest_float("C", 0.1, 100, log=True)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVC(kernel='rbf', C=c_value, gamma=gamma))
    ])
    cv = GroupKFold(n_splits=N_FOLDS)
    preds = cross_val_predict(pipe, X_train, y_train_labels, method="predict",
                               cv=cv, groups=groups_train, n_jobs=28)
    # Score by wide Borda earned: regret-aware (continuous reward, not 0/1)
    return np.sum(Y_train_wide[train_rows, preds])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
print(f"\nBest CV wide-Borda: {study.best_value:.2f}  params: {study.best_params}")

best_params = dict(study.best_params)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', svm.SVC(kernel='rbf', **best_params))
])
pipe.fit(X_train, y_train_labels)
preds = pipe.predict(X_test)

test_rows = np.arange(len(test_idx))
test_borda = np.sum(Y_test_borda[test_rows, preds])
test_oracle = np.sum(np.max(Y_test_borda, axis=1))
test_baselines = np.sum(Y_test_borda, axis=0)
test_acc = (preds == y_test_labels).mean()
test_majority = max((y_test_labels == c).mean() for c in range(3))
borda_headroom_vs_cpsat = (test_borda - test_baselines[0]) / (test_oracle - test_baselines[0]) if test_oracle > test_baselines[0] else 0.0
borda_headroom_vs_best_baseline = (test_borda - max(test_baselines)) / (test_oracle - max(test_baselines)) if test_oracle > max(test_baselines) else 0.0

print(f"\n========== Test Results (2011-2022, 3-solver pairwise Borda) ==========")
print(f"  test_borda:          {test_borda:.2f}")
print(f"  cpsat baseline:      {test_baselines[0]:.2f}")
print(f"  k1 baseline:         {test_baselines[1]:.2f}")
print(f"  ek1 baseline:        {test_baselines[2]:.2f}")
print(f"  oracle:              {test_oracle:.2f}")
print(f"  test accuracy:       {test_acc*100:.1f}%  (majority: {test_majority*100:.1f}%)")
print(f"  prediction balance [cpsat, k1, ek1]: {np.bincount(preds, minlength=3)}")
print(f"  borda headroom vs cpsat baseline:        {borda_headroom_vs_cpsat*100:.1f}%")
print(f"  borda headroom vs best single baseline:  {borda_headroom_vs_best_baseline*100:.1f}%")
