"""SVC trained on the 3-solver pairwise Borda from the existing npz, year-split:
train = 2024+2025, test = 2023. No wide Borda, no multi-seed — replicates the original
year-split experiment with current data.
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

N_TRIALS = 100
N_FOLDS = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, Y, meta = get_cpsat8_k1_ek1_data()
y_labels, Y_borda = prepare_labels(Y)

train_mask = (meta['year'] == 2024) | (meta['year'] == 2025)
test_mask  = (meta['year'] == 2023)
train_idx = np.where(train_mask)[0]
test_idx  = np.where(test_mask)[0]

groups_train = meta['problem'][train_idx]
Y_train_borda = Y_borda[train_idx]
Y_test_borda  = Y_borda[test_idx]
y_train = y_labels[train_idx]
y_test  = y_labels[test_idx]
train_rows = np.arange(len(train_idx))
test_rows  = np.arange(len(test_idx))

print(f"Train (2024+2025): {len(train_idx)} instances  label balance: {np.bincount(y_train, minlength=3)}")
print(f"Test  (2023):      {len(test_idx)} instances")
print(f"Train cpsat baseline: {np.sum(Y_train_borda[:, 0]):.2f}  k1: {np.sum(Y_train_borda[:, 1]):.2f}  ek1: {np.sum(Y_train_borda[:, 2]):.2f}")
print(f"Test  cpsat baseline: {np.sum(Y_test_borda[:, 0]):.2f}  k1: {np.sum(Y_test_borda[:, 1]):.2f}  ek1: {np.sum(Y_test_borda[:, 2]):.2f}")
print(f"Test  oracle:         {np.sum(np.max(Y_test_borda, axis=1)):.2f}")

def objective(trial):
    gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
    c_value = trial.suggest_float("C", 0.1, 100, log=True)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVC(kernel='rbf', C=c_value, gamma=gamma))
    ])
    cv = GroupKFold(n_splits=N_FOLDS)
    preds = cross_val_predict(pipe, X[train_idx], y_train, method="predict",
                               cv=cv, groups=groups_train, n_jobs=28)
    return np.sum(Y_train_borda[train_rows, preds])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)
print(f"\nBest CV train Borda: {study.best_value:.2f}  params: {study.best_params}")

best_params = dict(study.best_params)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', svm.SVC(kernel='rbf', **best_params))
])
pipe.fit(X[train_idx], y_train)
preds = pipe.predict(X[test_idx])

test_borda = np.sum(Y_test_borda[test_rows, preds])
test_oracle = np.sum(np.max(Y_test_borda, axis=1))
test_baselines = np.sum(Y_test_borda, axis=0)
test_acc = (preds == y_test).mean()
test_majority = max((y_test == c).mean() for c in range(3))
borda_headroom_vs_cpsat = (test_borda - test_baselines[0]) / (test_oracle - test_baselines[0]) if test_oracle > test_baselines[0] else 0.0
borda_headroom_vs_best  = (test_borda - max(test_baselines)) / (test_oracle - max(test_baselines)) if test_oracle > max(test_baselines) else 0.0

print(f"\n========== Test Results (2023, 3-solver pairwise Borda) ==========")
print(f"  test_borda:                     {test_borda:.2f}")
print(f"  cpsat baseline:                 {test_baselines[0]:.2f}")
print(f"  k1 baseline:                    {test_baselines[1]:.2f}")
print(f"  ek1 baseline:                   {test_baselines[2]:.2f}")
print(f"  oracle:                         {test_oracle:.2f}")
print(f"  test accuracy:                  {test_acc*100:.1f}%  (majority: {test_majority*100:.1f}%)")
print(f"  prediction balance [cpsat,k1,ek1]: {np.bincount(preds, minlength=3)}")
print(f"  borda headroom vs cpsat:        {borda_headroom_vs_cpsat*100:.1f}%")
print(f"  borda headroom vs best baseline:{borda_headroom_vs_best*100:.1f}%")
