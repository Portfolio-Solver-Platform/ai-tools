from pathlib import Path

import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shared_data import get_24_25_data, prepare_labels

bounds = [i*0.01 for i in range(0,99)]

X, Y = get_24_25_data()
y_labels, Y_eval = prepare_labels(Y)

n_features = X.shape[1]

kernel = 1.0 * RBF(
    length_scale=np.ones(n_features), 
    length_scale_bounds=(1e-5, 1e5) 
)
gpc = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GaussianProcessClassifier(kernel))
])

loo = LeaveOneOut()
predicted_proba = cross_val_predict(gpc, X, y_labels, method="predict_proba", cv=26, n_jobs=-1, verbose=1)


for bound in bounds: 
    final_solvers = np.argmax(predicted_proba, axis=1)
    uncertainty_mask = np.max(predicted_proba,axis=1) < bound
    final_solvers[uncertainty_mask] = 0


    row_indices = np.arange(len(Y_eval))
    times = Y_eval[row_indices, final_solvers]
    total_time = np.sum(times)
    print(f"bound ({bound}): {total_time}")