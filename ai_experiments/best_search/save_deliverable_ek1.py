"""
Save BagSVM-MW/log_std fitted on the full cpsat8_ek1 dataset, using the
median LOYO best_params from results/folds_ek1.csv.
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.shared_data import get_cpsat8_ek1_data, prepare_labels  # noqa: E402
from ai_experiments.best_search.save_deliverable import BagSVMPredictor  # noqa: E402
from ai_experiments.best_search.save_deliverable_fast import (  # noqa: E402
    fit_bag_svm_with_params, summarize_params,
)

FOLDS_CSV = Path(__file__).resolve().parent / "results" / "folds_ek1.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    target = "BagSVM-MW/log_std"
    n_estimators = 30

    params_per_fold = []
    with open(FOLDS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == target:
                params_per_fold.append(json.loads(row["best_params"]))
    params = summarize_params(params_per_fold)
    print(f"using median params: {params}")

    X, Y, meta = get_cpsat8_ek1_data()
    y_labels, Y_borda = prepare_labels(Y)

    t0 = time.time()
    predictor = fit_bag_svm_with_params("log_std", params, n_estimators,
                                         X, y_labels, Y_borda)
    out_path = MODELS_DIR / "deliverable_BagSVM_MW_log_std_ek1.joblib"
    joblib.dump(predictor, out_path)
    print(f"saved {out_path}  (wall={time.time()-t0:.0f}s)")

    pred = predictor.predict(X)
    print(f"training-set sanity: borda={float(Y_borda[np.arange(len(X)), pred].sum()):.2f} "
          f"/ oracle={float(Y_borda.max(axis=1).sum()):.2f}  "
          f"acc={(pred == y_labels).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
