import joblib
import numpy as np
import xgboost2 as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- 1. LOAD & PREPARE DATA (Same as before) ---
data = np.load('training_data.npz')
X = data["X"]  # Raw X (do not scale yet!)
Y = data["Y"]

# Handle timeouts
max_real_time = np.nanmax(Y)
timeout_penalty = max_real_time * 1.1 
Y_filled = np.nan_to_num(Y, nan=timeout_penalty)

# Create Log Targets
Y_log = np.log(Y_filled + 1e-6)

# --- 2. DEFINE THE PIPELINE ---
# We use a Pipeline so that StandardScaler is re-fit 
# on the N-1 training samples every single time.
xgb_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=6, 
    n_jobs=1  # Set to 1 here; we parallelize the LOO loop instead
)

# Wrap in Pipeline: Scaler -> Model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MultiOutputRegressor(xgb_estimator))
])

# --- 3. RUN LEAVE-ONE-OUT ---
print(f"Starting LOOCV on {len(X)} samples...")
loo = LeaveOneOut()

# cross_val_predict splits the data N times. 
# For each split, it trains on N-1 and predicts on the remaining 1.
# n_jobs=-1 uses all CPU cores to speed this up.
loo_predictions_log = cross_val_predict(
    pipeline, 
    X, 
    Y_log, 
    cv=loo, 
    n_jobs=-1,
    verbose=1 # Prints progress
)

# --- 4. EVALUATE PERFORMANCE ---
# Now we compare the LOO predictions against the ground truth
pred_choices = np.argmin(loo_predictions_log, axis=1)
true_best = np.argmin(Y_filled, axis=1)

# Accuracy
acc = np.mean(pred_choices == true_best)
print(f"\nLOOCV Accuracy: {acc * 100:.2f}%")

# Time Loss (SBS - VBS)
chosen_times = Y_filled[np.arange(len(Y_filled)), pred_choices]
ideal_times = np.min(Y_filled, axis=1)
avg_loss = np.mean(chosen_times - ideal_times)
print(f"Avg Time Lost: {avg_loss:.4f}s")

joblib.dump(xgb_estimator, 'xgboost_multioutput_model.pkl')
print("xgboost and Scaler saved successfully.")
