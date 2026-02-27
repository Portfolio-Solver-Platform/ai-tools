from pathlib import Path

import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_24_25_data, prepare_labels

bounds = [i*0.01 for i in range(0,99)]
neighbours = [1,2,3,4,5,7,9,11,13,15,17]

X, Y = get_24_25_data()
y_labels, Y_eval = prepare_labels(Y)

optimal_time = np.sum(np.min(Y_eval, axis=1))
print(f"optimal time: {optimal_time}")

for neighbour in neighbours:
    print(f"\n\n========== Neigbours {neighbour} ==========")
    gpc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=neighbour))
    ])

    loo = LeaveOneOut()
    predicted_proba = cross_val_predict(gpc, X, y_labels, method="predict_proba", cv=loo, n_jobs=-1, verbose=1)

    best = np.inf
    best_bound = 0
    for bound in bounds: 
        final_solvers = np.argmax(predicted_proba, axis=1)
        uncertainty_mask = np.max(predicted_proba,axis=1) < bound
        final_solvers[uncertainty_mask] = 0


        row_indices = np.arange(len(Y_eval))
        times = Y_eval[row_indices, final_solvers]
        total_time = np.sum(times)
        if total_time < best:
            best = total_time
            best_bound = bound
    print(f"best bound ({best_bound}): {best}")