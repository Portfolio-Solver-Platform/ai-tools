from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from utils.shared_data import get_23_data

X, Y = get_23_data()


model = joblib.load("/home/sofus/speciale/ai/ai-tools/ai_experiments/models/svm_model.joblib")           
  

predicted_portfolio = model.predict(X)

model_total_time = np.sum(Y[np.arange(len(Y)), predicted_portfolio])
print(f"model_total_time: {model_total_time}")
cpsat8_total_time = np.sum(Y[np.arange(len(Y)), 0])
print(f"cpsat8_total_time: {cpsat8_total_time}")
optimal = np.sum(np.min(Y, axis=1))
print(f"optimal: {optimal}")
