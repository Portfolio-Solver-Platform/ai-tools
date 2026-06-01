from pathlib import Path
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import optuna
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_cpsat8_ek1_data, get_cpsat8_k1_data, get_cpsat8_k1_ek1_data, prepare_labels
from utils.cross_solver_eval import make_train_val_test_indices

_, _, meta_canon = get_cpsat8_k1_ek1_data()
groups_canon = meta_canon["problem"]
train_idx, val_idx, test_idx = make_train_val_test_indices(groups_canon)
trainval_idx = np.concatenate([train_idx, val_idx])

print(f"split sizes: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
print(f"problems:    train={len(np.unique(groups_canon[train_idx]))}  val={len(np.unique(groups_canon[val_idx]))}  test={len(np.unique(groups_canon[test_idx]))}")

datasets = [
    # ('cpsat8_k1_ek1', get_cpsat8_k1_ek1_data),
    ('cpsat8_k1',     get_cpsat8_k1_data),
    # ('cpsat8_ek1',    get_cpsat8_ek1_data),
]

results = {}
Path('models').mkdir(exist_ok=True)

for name, getter in datasets:
    print(f"\n========== dataset: {name} ==========")
    X, Y, meta = getter()
    assert np.array_equal(meta["problem"], groups_canon), f"meta['problem'] misaligned for {name}"
    y_labels, Y_borda = prepare_labels(Y)
    Y_val_borda = Y_borda[val_idx]
    Y_test_borda_local = Y_borda[test_idx]
    val_rows = np.arange(len(val_idx))
    test_rows = np.arange(len(test_idx))

    print(f"val baselines (always local solver i): {np.sum(Y_val_borda, axis=0)}")
    print(f"val oracle (best solver per row): {np.sum(np.max(Y_val_borda, axis=1))}")

    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights     = trial.suggest_categorical("weights", ["uniform", "distance"])
        p           = trial.suggest_categorical("p", [1, 2])
        # threshold = trial.suggest_float("threshold", 0.34, 0.99)

        knn_args = {'n_neighbors': n_neighbors, 'weights': weights, 'p': p}

        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(**knn_args))
        ])

        knn_pipeline.fit(X[train_idx], y_labels[train_idx])
        predicted_solvers = knn_pipeline.predict(X[val_idx])
        # val_proba = knn_pipeline.predict_proba(X[val_idx])
        # predicted_solvers = np.argmax(val_proba, axis=1)
        # predicted_solvers[val_proba.max(axis=1) < threshold] = 0  # cpsat8 fallback

        bordas = Y_val_borda[val_rows, predicted_solvers]
        return np.sum(bordas)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print(f"best val total borda: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_params = dict(study.best_params)
    # best_threshold = best_params.pop('threshold')
    best_knn_args = {**best_params}

    test_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(**best_knn_args))
    ])
    test_pipeline.fit(X[trainval_idx], y_labels[trainval_idx])
    predictions_local = test_pipeline.predict(X[test_idx])
    # test_proba = test_pipeline.predict_proba(X[test_idx])
    # predictions_local = np.argmax(test_proba, axis=1)
    # predictions_local[test_proba.max(axis=1) < best_threshold] = 0  # cpsat8 fallback
    test_borda = np.sum(Y_test_borda_local[test_rows, predictions_local])

    test_baselines_local = np.sum(Y_test_borda_local, axis=0)
    test_oracle_local = np.sum(np.max(Y_test_borda_local, axis=1))
    y_test_local = y_labels[test_idx]
    test_accuracy = (predictions_local == y_test_local).mean()
    test_majority = max((y_test_local == c).mean() for c in range(Y.shape[1]))
    # print(f"chosen threshold: {best_threshold:.3f}")
    print(f"test total borda (local {name}): {test_borda:.2f}")
    print(f"test baselines (local always solver i): {test_baselines_local}")
    print(f"test oracle (local best per row):  {test_oracle_local:.2f}")
    print(f"test accuracy: {test_accuracy*100:.1f}%  (test majority baseline: {test_majority*100:.1f}%)")

    results[name] = {
        'val_borda': study.best_value,
        'test_borda_local': test_borda,
        'test_oracle_local': test_oracle_local,
        'test_cpsat_baseline_local': test_baselines_local[0],
        # 'threshold': best_threshold,
        'best_params': best_params,
    }

    final_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(**best_knn_args))
    ])
    final_pipe.fit(X, y_labels)
    joblib.dump(final_pipe, f'models/knn_model_{name}.joblib')
    # joblib.dump({'model': final_pipe, 'threshold': best_threshold}, f'models/knn_model_{name}.joblib')

print("\n========== Summary (test Borda using each dataset's local Borda) ==========")
for name, r in results.items():
    ratio = r['test_borda_local'] / r['test_oracle_local'] if r['test_oracle_local'] else float('nan')
    print(f"  {name:14s}  test_borda={r['test_borda_local']:.2f}  oracle={r['test_oracle_local']:.2f}  cpsat={r['test_cpsat_baseline_local']:.2f}  oracle_ratio={ratio:.3f}")
    # print(f"  {name:8s}  test_borda={r['test_borda_shared']:.2f}  oracle_ratio={ratio:.3f}  threshold={r['threshold']:.3f}")
