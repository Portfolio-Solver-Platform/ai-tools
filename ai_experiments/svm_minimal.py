"""
Minimal SVC baseline: train on 2011-2020, test on 2021-2025. No HPO, no CV.
Sanity check reference against the nested-CV results.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.shared_data import get_cpsat8_k1_ek1_data, prepare_labels

TRAIN_THROUGH = 2020

X, Y, meta = get_cpsat8_k1_ek1_data()
y_labels, Y_borda = prepare_labels(Y)
years = meta["year"]

train_mask = years <= TRAIN_THROUGH
test_mask  = years > TRAIN_THROUGH

pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])
pipe.fit(X[train_mask], y_labels[train_mask])
pred = pipe.predict(X[test_mask])

Y_te = Y_borda[test_mask]
n = len(pred)
test_borda = Y_te[np.arange(n), pred].sum()
oracle     = Y_te.max(axis=1).sum()
cpsat      = Y_te[:, 0].sum()
accuracy   = (pred == y_labels[test_mask]).mean()

train_years = sorted(np.unique(years[train_mask]).tolist())
test_years  = sorted(np.unique(years[test_mask]).tolist())
print(f"train: {train_mask.sum()} instances, years {train_years[0]}-{train_years[-1]}")
print(f"test:  {test_mask.sum()} instances, years {test_years[0]}-{test_years[-1]}")
print(f"test borda     = {test_borda:.2f}")
print(f"oracle         = {oracle:.2f}  ({test_borda / oracle * 100:.1f}% of oracle)")
print(f"cpsat baseline = {cpsat:.2f}")
print(f"accuracy       = {accuracy * 100:.1f}%")
