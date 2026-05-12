from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import cross_val_predict, GroupShuffleSplit, GroupKFold
import xgboost as xgb
import optuna
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels
from utils.cross_solver_eval import shared_test_borda

# Canonical 3-solver split — defines the shared test rows reused for every dataset.
# Group-by-problem so no problem family appears in both train and test.
X_canon, Y_canon, meta_canon = get_cpsat8_k1_ek1_data()
y_labels_canon, Y_borda_canon = prepare_labels(Y_canon)

groups_canon = meta_canon["problem"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X_canon, y_labels_canon, groups=groups_canon))
Y_test_borda_3solver = Y_borda_canon[test_idx]

print(f"shared test baselines (always solver i): {np.sum(Y_test_borda_3solver, axis=0)}")
print(f"shared test oracle (best solver per row): {np.sum(np.max(Y_test_borda_3solver, axis=1))}")
print(f"train problems: {len(np.unique(groups_canon[train_idx]))}, test problems: {len(np.unique(groups_canon[test_idx]))}")

datasets = [
    ('cpsat8_k1_ek1', get_cpsat8_k1_ek1_data),
    ('cpsat8_k1',     get_cpsat8_k1_data),
    ('cpsat8_ek1',    get_cpsat8_ek1_data),
]

results = {}
Path('models').mkdir(exist_ok=True)

for name, getter in datasets:
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    assert np.array_equal(meta["problem"], groups_canon), f"meta['problem'] misaligned for {name}"
    y_labels, Y_borda = prepare_labels(Y)
    X_train = X[train_idx]
    y_train = y_labels[train_idx]
    X_test = X[test_idx]
    Y_train_borda = Y_borda[train_idx]
    train_groups = meta["problem"][train_idx]

    print(f"train baselines (always local solver i): {np.sum(Y_train_borda, axis=0)}")
    print(f"train oracle (best solver per row): {np.sum(np.max(Y_train_borda, axis=1))}")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'tree_method': 'hist',
            'n_jobs': 1,
            'random_state': 42,
        }
        threshold = trial.suggest_float('threshold', 0.34, 0.99)
        model = xgb.XGBClassifier(**params)

        cv = GroupKFold(n_splits=5)
        proba = cross_val_predict(model, X_train, y_train, method="predict_proba", cv=cv, groups=train_groups, n_jobs=28, verbose=1)

        predicted_solvers = np.argmax(proba, axis=1)
        predicted_solvers[proba.max(axis=1) < threshold] = 0  # cpsat8 fallback

        row_indices = np.arange(len(Y_train_borda))
        bordas = Y_train_borda[row_indices, predicted_solvers]
        return np.sum(bordas)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print(f"best train total borda: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_params = dict(study.best_params)
    best_threshold = best_params.pop('threshold')
    best_model = xgb.XGBClassifier(
        **best_params,
        tree_method='hist',
        n_jobs=28,
        random_state=42,
    )
    best_model.fit(X_train, y_train)
    test_proba = best_model.predict_proba(X_test)
    predictions_local = np.argmax(test_proba, axis=1)
    predictions_local[test_proba.max(axis=1) < best_threshold] = 0  # cpsat8 fallback
    test_borda = shared_test_borda(predictions_local, name, Y_test_borda_3solver)
    print(f"chosen threshold: {best_threshold:.3f}")
    print(f"test total borda on shared 3-solver test set: {test_borda}")

    results[name] = {
        'train_borda': study.best_value,
        'test_borda_shared': test_borda,
        'threshold': best_threshold,
        'best_params': best_params,
    }

    final_model = xgb.XGBClassifier(
        **best_params,
        tree_method='hist',
        n_jobs=28,
        random_state=42,
    )
    final_model.fit(X, y_labels)
    joblib.dump({'model': final_model, 'threshold': best_threshold}, f'models/xgb_model_{name}.joblib')

print("\n========== Summary (test Borda on shared 3-solver test set) ==========")
oracle = np.sum(np.max(Y_test_borda_3solver, axis=1))
baselines = np.sum(Y_test_borda_3solver, axis=0)
print(f"oracle: {oracle:.2f}")
print(f"baselines [cpsat8, k1, ek1]: {baselines}")
for name, r in results.items():
    ratio = r['test_borda_shared'] / oracle if oracle else float('nan')
    print(f"  {name:8s}  test_borda={r['test_borda_shared']:.2f}  oracle_ratio={ratio:.3f}  threshold={r['threshold']:.3f}")
