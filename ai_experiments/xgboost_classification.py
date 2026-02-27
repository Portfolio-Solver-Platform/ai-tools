from pathlib import Path

import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process.kernels import RBF
import xgboost as xgb
import optuna
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_24_25_data, prepare_labels

bounds = [i*0.01 for i in range(0,99)]

X, Y = get_24_25_data()
y_labels, Y_eval = prepare_labels(Y)



def objective(trial):
    params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'objective': 'binary:logistic',
            'n_jobs': 1,
            'random_state': 42 
        }
    model = xgb.XGBClassifier(
        **params
    )

    loo = LeaveOneOut()
    predicted_proba = cross_val_predict(model, X, y_labels, method="predict_proba", cv=loo, n_jobs=-1, verbose=1)

    best = np.inf
    for bound in bounds: 
        final_solvers = np.argmax(predicted_proba, axis=1)
        uncertainty_mask = np.max(predicted_proba,axis=1) < bound
        final_solvers[uncertainty_mask] = 0


        row_indices = np.arange(len(Y_eval))
        times = Y_eval[row_indices, final_solvers]
        total_time = np.sum(times)
        if best > total_time:
            best = total_time
        
                
    return best



study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)


print("-------------------")
print(f"best total time: {study.best_value}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")