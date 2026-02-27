from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from sklearn import svm
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_24_25_data, prepare_labels

X, Y = get_24_25_data()
y_labels, Y_eval = prepare_labels(Y)

def objective(trial):
    gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)  
    c_value = trial.suggest_float("C", 0.1, 100, log=True)
    
    svc_args = {'kernel': 'rbf', 'C': c_value, 'gamma': gamma}

    gpc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', svm.SVC(**svc_args))
    ])

    loo = LeaveOneOut()
    predicted_solvers = cross_val_predict(gpc, X, y_labels, method="predict", cv=loo, n_jobs=28, verbose=1)

    # solvers, counts = np.unique(predicted_solvers, return_counts=True)
    # total_samples = len(predicted_solvers)

    # print("\n--- Solver Distribution ---")
    # for solver_idx, count in zip(solvers, counts):
    #     percentage = (count / total_samples) * 100
    #     print(f"Solver {solver_idx}: chosen {count} times ({percentage:.1f}%)")
    
    row_indices = np.arange(len(Y_eval))
    times = Y_eval[row_indices, predicted_solvers]
    total_time = np.sum(times)
    return total_time    
    
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("-------------------")
print(f"best total time: {study.best_value}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
    
gpc = Pipeline([
    ('scaler', StandardScaler()),
    ('model', svm.SVC(**study.best_params))
])

gpc.fit(X, y_labels)

import joblib
joblib.dump(gpc, 'models/svm_model.joblib')

